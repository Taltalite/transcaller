#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_dorado_fixed.py

修正版训练脚本，适配 savectc_fixed.py 生成的数据集格式。
关键变更：
1. 文件名匹配 (chunks.npy, references.npy)
2. 恢复 unsqueeze(1) 以适配 Conv1d 输入维度
3. 调整 CTC Blank ID = 0，直接兼容 A=1,C=2,G=3,T=4,Pad=0 的数据格式
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
import random
import sys

# --- 导入模型 ---
try:
    # 请根据你的实际路径修改
    sys.path.append('/home/lijy/workspace/')
    from my_basecaller.model.transcaller_light import TranscallerLight
except ImportError:
    print("="*80)
    print("错误: 无法导入 'TranscallerLight'。")
    print("请检查 sys.path.append 的路径是否正确。")
    print("="*80)
    sys.exit(1)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to {seed}")

# ==========================================================================================
# 步骤 1: 数据集类 (修正文件名和形状)
# ==========================================================================================

class DoradoNpyDataset(Dataset):
    """
    加载由 savectc_fixed.py 生成的 .npy 数据集。
    文件名: chunks.npy, references.npy, reference_lengths.npy
    """
    def __init__(self, data_dir, num_samples_to_load=-1):
        super().__init__()
        
        # [修正 1] 文件名必须与 savectc_fixed.py 的输出一致
        signals_path = os.path.join(data_dir, "chunks.npy")
        labels_path = os.path.join(data_dir, "references.npy")
        lens_path = os.path.join(data_dir, "reference_lengths.npy")

        print(f"🚀 开始加载数据集: {data_dir}")
        
        try:
            # 1. 加载 Numpy 数组
            print("  (1/3) 加载 chunks.npy (信号)...")
            events_np = np.load(signals_path) # Shape: (N, 2048), float16
            
            print("  (2/3) 加载 references.npy (标签)...")
            labels_np = np.load(labels_path)  # Shape: (N, max_len), uint8, Pad=0
            
            print("  (3/3) 加载 reference_lengths.npy (长度)...")
            label_lens_np = np.load(lens_path) # Shape: (N,), uint16
            
            # 2. 截取样本 (Debug 用)
            if num_samples_to_load > 0:
                print(f"  ...截取前 {num_samples_to_load} 个样本。")
                events_np = events_np[:num_samples_to_load]
                labels_np = labels_np[:num_samples_to_load]
                label_lens_np = label_lens_np[:num_samples_to_load]

            # 3. 转换为 Tensor 并调整形状
            print("  正在转换格式...")
            
            # [修正 2] 增加 Channel 维度
            # Numpy shape 是 (N, 2048)，PyTorch Conv1d 需要 (N, Channel, Length)
            # 所以必须 unsqueeze(1) 变成 (N, 1, 2048)
            self.events = torch.from_numpy(events_np).float().unsqueeze(1)
            
            # 标签部分
            # savectc 脚本生成的数据：Pad=0, A=1, C=2, G=3, T=4
            # 我们将在 CTC Loss 中设置 blank=0，所以不需要对标签值做任何减法操作！
            self.labels = torch.from_numpy(labels_np).long()
            self.label_lens = torch.from_numpy(label_lens_np).long()
            
            self.dataset_len = self.events.shape[0]
            
            print(f"🚀 数据加载完成。样本数: {self.dataset_len}")
            print(f"   Input Shape: {self.events.shape} (期望: N, 1, 2048)")
            print(f"   Label Range: Min={self.labels.min()}, Max={self.labels.max()} (期望: 0-4)")
            
        except FileNotFoundError as e:
            print(f"❌ 文件未找到: {e}")
            print(f"   请检查 --data-dir 路径下是否有 chunks.npy 等文件")
            raise
        except Exception as e:
            print(f"❌ 加载数据出错: {e}")
            raise

    def __len__(self):
        return self.dataset_len

    # def __getitem__(self, idx):
    #     return self.events[idx], self.labels[idx], self.label_lens[idx]
    def __getitem__(self, idx):
    # 1. 获取原始信号 (1, 2048)
        signal = self.events[idx] 
        
        # 2. === 新增：鲁棒归一化 (Robust Normalization) ===
        # 使用中位数绝对偏差 (MAD) 或简单的 (x - mean) / std
        # 对于 Nanopore 信号，简单的 Z-score 通常足够：
        mean = signal.mean()
        std = signal.std()
        
        # 防止除以 0 (极少数全是平信号的情况)
        if std < 1e-5:
            std = 1.0
            
        signal = (signal - mean) / std
        # ===============================================

        return signal, self.labels[idx], self.label_lens[idx]


# ==========================================================================================
# 步骤 2: 训练核心 (增加 Loss Debug)
# ==========================================================================================

static_printed = False

def train_one_epoch(model, criterion, optimizer, data_loader, device, output_len, scheduler_warmup, warmup_steps, global_step):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc='[训练]', leave=False)
    
    for events, labels, label_lengths in progress_bar:
        events = events.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 1. 前向传播
        # 模型内部已经包含了:
        # a. Head 的 permute(1, 0, 2) -> 变为 (Time, Batch, Class)
        # b. F.log_softmax(dim=-1)    -> 变为 Log Probabilities
        log_probs = model(events) 
        
        # 2. 维度检查 (安全起见)
        # 确保第一维是 Time (512)，第二维是 Batch
        if log_probs.shape[1] != events.shape[0]:
            # 如果维度不对，说明模型代码没加载对，或者 Head 没 permute
            # 但根据你提供的模型代码，这里不需要任何操作
            pass 

        batch_size = events.shape[0]
        input_lengths = torch.full(size=(batch_size,), fill_value=output_len, dtype=torch.long, device=device)
        
        # 3. 计算 Loss (直接传入)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
                
        if torch.isinf(loss) or torch.isnan(loss):
            print("⚠️ Loss is inf/nan")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if global_step < warmup_steps:
            scheduler_warmup.step()
        global_step += 1
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
            
    return total_loss / len(data_loader), global_step

def validate(model, criterion, data_loader, device, output_len):
    model.eval() 
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc='[验证]', leave=False)
    
    with torch.no_grad():
        for events, labels, label_lengths in progress_bar:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
            
            outputs = model(events)
            if outputs.shape[0] == events.shape[0]:
                 log_probs = outputs.permute(2, 0, 1)
            else:
                 log_probs = outputs

            batch_size = events.shape[0]
            input_lengths = torch.full(size=(batch_size,), fill_value=output_len, dtype=torch.long, device=device)
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            if not torch.isinf(loss):
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')
            
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# ==========================================================================================
# 步骤 3: 主流程 (设置 Blank=0)
# ==========================================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 使用设备: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 确保 Output Length 计算正确
    # 假设模型有两层 stride=2 的卷积，输出长度是输入/4
    true_output_len = args.input_len // 4
    if args.output_len != true_output_len:
        print(f"⚠️  自动修正 output-len: {args.output_len} -> {true_output_len}")
        args.output_len = true_output_len
    
    # --- 数据集 ---
    dataset_to_split = DoradoNpyDataset(args.data_dir, args.num_samples)

    val_size = int(len(dataset_to_split) * args.val_split)
    train_size = len(dataset_to_split) - val_size
    
    # 确保验证集至少有一个 batch，防止报错
    if val_size < args.batch_size:
        val_size = args.batch_size
        train_size = len(dataset_to_split) - val_size

    train_dataset, val_dataset = random_split(
        dataset_to_split, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"📚 训练集: {train_size} | 验证集: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # --- 模型 ---
    print("🛠️  初始化模型...")
    model = TranscallerLight(
        input_length=args.input_len,   # 2048
        output_length=args.output_len, # 512
        num_classes=args.num_classes,  # 5
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path
    ).to(device)
    
    print(f"   参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # [修正 3] 设置 Blank ID = 0
    # 数据集中: 0=Pad, 1=A, 2=C, 3=G, 4=T
    # CTCLoss 设置 blank=0 后，模型预测的 Index 0 将被视为 Blank，1-4 为碱基。
    # 这与数据集的编码完美匹配。
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 学习率调度
    steps_per_epoch = len(train_loader)
    warmup_steps = steps_per_epoch # 1个 epoch warmup
    
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
    )
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.scheduler_patience, factor=0.5, verbose=True
    )

    # --- 训练 ---
    print(f"🚀 开始训练 (Total Epochs: {args.epochs})")
    best_val_loss = float('inf') 
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        start_time = time.time()
        
        train_loss, global_step = train_one_epoch(
            model, criterion, optimizer, train_loader, device, args.output_len,
            scheduler_warmup, warmup_steps, global_step
        )
        
        val_loss = validate(model, criterion, val_loader, device, args.output_len)
        
        if global_step >= warmup_steps:
             scheduler_plateau.step(val_loss)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} 耗时: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoints
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(state, os.path.join(args.checkpoint_dir, "model_latest.pth"))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_best.pth"))
            print(f"   🏆 新最佳模型已保存 (Loss: {best_val_loss:.4f})")
            
    print(f"\n✅ 训练结束. 最佳验证 Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Transcaller (Corrected for SaveCTC Data)")
    
    # 路径参数
    parser.add_argument('--data-dir', type=str, required=True, help="数据集目录 (包含 chunks.npy)")
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints_dorado", help="模型保存路径")
    
    # 数据参数
    parser.add_argument('--num-samples', type=int, default=-1, help="调试用：限制加载样本数")
    parser.add_argument('--val-split', type=float, default=0.05, help="验证集比例")
    parser.add_argument('--num-workers', type=int, default=8, help="Dataloader 线程数")
    
    # 模型参数 (针对 Dorado 配置)
    parser.add_argument('--input-len', type=int, default=2048, help="chunks.npy 的每条长度")
    parser.add_argument('--output-len', type=int, default=512, help="模型输出步长 (通常是 input/4)")
    parser.add_argument('--num-classes', type=int, default=5, help="类别数 (Blank + ACGT = 5)")
    
    # [修正 4] 默认 Blank ID 改为 0
    parser.add_argument('--blank-id', type=int, default=0, help="CTC Blank Index (对应 Pad=0)")
    
    # 模型超参
    parser.add_argument('--embed-dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--drop-path', type=float, default=0.1)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scheduler-patience', type=int, default=3)

    args = parser.parse_args()
    
    if args.seed >= 0:
        set_seed(seed=args.seed)
    
    main(args)