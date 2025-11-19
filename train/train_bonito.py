import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
# --- [新增] 导入 AMP 和 LR 调度器 ---
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import h5py
import numpy as np
import editdistance
from tqdm import tqdm
import os
import math # 用于 ClipGrad

# 从您的模型文件中导入必要的类和函数
try:
    import sys
    sys.path.append('/home/lijy/workspace/')
    from my_basecaller.model.bonito_model import Model, deepnorm_params

except ImportError:
    print("错误：无法导入 'bonito_model.py'。")
    print("请确保您已将 'bonito model.txt' 的内容保存为 'bonito_model.py'。")
    exit()

# --- 1. 配置参数 ---
H5_PATH = '/data/biolab-nvme-pool1/lijy/HG002_dataset/HG002_single.h5'
BATCH_SIZE = 32
EPOCHS = 10
# 学习率预热对 Transformer 很重要
LEARNING_RATE = 1e-4 # 目标学习率
WARMUP_STEPS = 500  # 预热 500 步
VAL_SPLIT_RATIO = 0.1
CHECKPOINT_PATH = 'bonito_trained.pth'
# --- [新增] 启用 AMP (如果有 GPU) ---
USE_AMP = torch.cuda.is_available()

# --- [新增] 复制 Bonito 的动态梯度裁剪类 ---
class ClipGrad:
    """
    来自 Bonito 官方训练代码的动态梯度裁剪器。
    它根据最近梯度范数的中位数（或分位数）来设置裁剪阈值。
    """
    def __init__(self, quantile=0.5, factor=2.0, buffer_size=100):
        self.buffer = np.full(buffer_size, fill_value=1e6)
        self.quantile = quantile
        self.factor = factor
        self.i = 0

    def append(self, grad_norm):
        self.buffer[self.i] = grad_norm
        self.i = (self.i + 1) % len(self.buffer)

    def __call__(self, parameters):
        # 计算裁剪阈值：缓冲区中位数 * 因子
        max_norm = self.factor * np.quantile(self.buffer, self.quantile)
        # 执行裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm).item()
        # 仅在梯度有效时才将其添加到缓冲区
        if not math.isnan(grad_norm) and not math.isinf(grad_norm):
            self.append(grad_norm)
        return grad_norm, max_norm

# --- 2. HDF5 数据集类 (与之前相同) ---
class SignalDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_len = len(f['label_len'])
        # --- [新增] 归一化 (可选，但推荐) ---
        # 最好在这里计算一次性的均值和标准差，或者使用中位数/MAD
        # 为简单起见，我们暂时在 __getitem__ 中进行 z-score
        print("数据集初始化完成。")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        event = self.file['event'][idx]
        label = self.file['label'][idx]
        label_len = self.file['label_len'][idx]

        event_tensor = torch.tensor(event, dtype=torch.float)
        
        # --- [推荐] 添加 Z-score 归一化以提高稳定性 ---
        mean = event_tensor.mean()
        std = event_tensor.std()
        event_tensor = (event_tensor - mean) / (std + 1e-6) # 防止除以零
        # ----------------------------------------------

        label_tensor = torch.tensor(label, dtype=torch.long)
        label_len_tensor = torch.tensor(label_len, dtype=torch.long)

        return event_tensor, label_tensor, label_len_tensor

# --- 3. 评估辅助函数 (与之前相同) ---
def decode_truth(labels_batch, lengths_batch, alphabet):
    labels_batch_cpu = labels_batch.cpu().numpy()
    lengths_batch_cpu = lengths_batch.cpu().numpy()
    true_strings = []

    for label_seq, length in zip(labels_batch_cpu, lengths_batch_cpu):
        true_seq = label_seq[:length]
        true_str = "".join([alphabet[i] for i in true_seq if i > 0])
        true_strings.append(true_str)
        
    return true_strings

# --- 4. 主训练函数 ---
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在使用设备: {device} ---")
    print(f"--- 自动混合精度 (AMP): {'启用' if USE_AMP else '禁用'} ---")

    # --- 实例化模型 (与之前相同) ---
    print("--- 正在加载模型配置... ---")
    depth = 6
    d_model = 256
    alpha, beta = deepnorm_params(depth)

    ENCODER_CONFIG = {
        "type": "serial",
        "sublayers": [
            {"type": "convolution", "insize": 1, "size": d_model, "winlen": 19, "stride": 5, "activation": "swish", "norm": "batchnorm"},
            {"type": "permute", "dims": [0, 2, 1]},
            {
                "type": "stack", "depth": depth, "layer": {
                    "type": "transformerencoderlayer", "d_model": d_model, "nhead": 4, 
                    "dim_feedforward": d_model * 4, "deepnorm_alpha": alpha, "deepnorm_beta": beta
                }
            },
            {"type": "permute", "dims": [1, 0, 2]},
            {
                "type": "linearcrfencoder", "insize": d_model, "n_base": 4, "state_len": 1, "blank_score": None,
                # --- [保留] 稳定 CRF 层的关键修复 ---
                "activation": "tanh",
                "scale": 5.0
            }
        ]
    }
    
    FULL_CONFIG = {
        "encoder": ENCODER_CONFIG,
        "labels": {"labels": ["", "A", "C", "G", "T"]},
        "global_norm": {"state_len": 1}
    }
    
    alphabet = FULL_CONFIG['labels']['labels']
    model = Model(FULL_CONFIG).to(device)
    print(f"--- 模型实例化成功。参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} ---")

    # --- 设置数据加载器 ---
    print("--- 正在准备数据集... ---")
    full_dataset = SignalDataset(h5_path=H5_PATH)
    
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT_RATIO)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"--- 数据加载完成。训练样本: {train_size}, 验证样本: {val_size} ---")
    
    # --- [修改] 设置优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # --- [修改] 设置学习率预热 (Warmup) + 衰减 (Decay) ---
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    
    # 1. 预热调度器
    # 从一个很小的值 (lr*start_factor) 线性增加到 目标LR
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=WARMUP_STEPS)
    
    # 2. 主调度器（余弦衰减）
    # 在 (total_steps - WARMUP_STEPS) 步内从 目标LR 衰减到 目标LR/100
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=LEARNING_RATE/100)
    
    # 3. 组合调度器
    # 在 WARMUP_STEPS 之前使用 warmup_scheduler，之后使用 main_scheduler
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_STEPS])
    
    # --- [新增] 实例化 GradScaler 和动态裁剪器 ---
    scaler = amp.GradScaler(enabled=USE_AMP)
    clip_grad = ClipGrad() # 使用 Bonito 的动态裁剪器
    
    print(f"--- 开始训练 {EPOCHS} 个 Epochs ({total_steps} 总步数) ---")
    print(f"--- 学习率将预热 {WARMUP_STEPS} 步 ---")

    best_val_identity = -1.0
    current_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [训练]", unit="batch")
        
        for events, labels, label_lens in train_pbar:
            events = events.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            # 1. 清零梯度 (在循环开始时)
            optimizer.zero_grad()

            # 2. 前向传播 (使用 AMP autocast)
            # autocast 会自动将操作切换到 float16
            with amp.autocast(enabled=USE_AMP):
                scores = model(events)
                # 3. 计算损失 (使用 loss_clip 仍然是好主意)
                loss = model.loss(scores, labels, label_lens, loss_clip=10.0)

            # 4. 反向传播 (使用 GradScaler)
            # scaler.scale 会自动放大损失
            scaler.scale(loss).backward()

            # 5. [修改] 解除梯度缩放 (Unscale)
            # 在裁剪或更新*之前*必须执行此操作
            scaler.unscale_(optimizer)

            # 6. [修改] 动态梯度裁剪
            # 使用 Bonito 的动态裁剪器
            grad_norm, max_norm = clip_grad(model.parameters())

            # 7. [修改] 优化器步骤 (由 Scaler 执行)
            # scaler.step 会自动检查 inf/nan 并跳过更新（如果需要）
            scaler.step(optimizer)

            # 8. [修改] 更新缩放器
            scaler.update()
            
            # 9. [修改] 学习率调度器 (按步更新)
            scheduler.step()
            current_step += 1
            
            if not math.isnan(loss.item()):
                total_train_loss += loss.item()
            
            # 更新进度条，显示更多信息
            train_pbar.set_postfix(
                loss=loss.item(), 
                lr=scheduler.get_last_lr()[0], 
                grad_norm=f"{grad_norm:.2f}/{max_norm:.2f}"
            )

        avg_train_loss = total_train_loss / len(train_loader)
        
        # 2. 评估 (Validation)
        model.eval()
        total_edits = 0
        total_length = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [评估]", unit="batch")
        
        with torch.no_grad():
            for events, labels, label_lens in val_pbar:
                events = events.to(device)
                labels = labels.to(device)
                label_lens = label_lens.to(device)

                # [新增] 在评估时也使用 autocast (更快，且与训练一致)
                with amp.autocast(enabled=USE_AMP):
                    scores = model(events)

                pred_strings = model.decode_batch(scores)
                true_strings = decode_truth(labels, label_lens, alphabet)

                for pred, true in zip(pred_strings, true_strings):
                    total_edits += editdistance.eval(pred, true)
                    total_length += len(true)
        
        val_identity = 0.0
        if total_length > 0:
            val_identity = 1.0 - (total_edits / total_length)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} 完成:")
        print(f"  平均训练损失: {avg_train_loss:.4f}")
        print(f"  验证一致性 (Identity): {val_identity * 100:.2f}%")

        if val_identity > best_val_identity:
            best_val_identity = val_identity
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  > 新的最佳模型已保存到: {CHECKPOINT_PATH} (Identity: {val_identity * 100:.2f}%)")

    print(f"\n--- 训练完成 ---")


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    main()