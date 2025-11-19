#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_hdf5.py

这个脚本用于训练 TranscallerLight 模型。
数据源：自定义生成的 HDF5 数据集。
映射关系：A=1, C=2, G=3, T=4, Blank=0。
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
import h5py

# --- 关键导入 ---
try:
    # 请根据您的实际路径修改这里
    sys.path.append('/home/lijy/workspace/')
    from my_basecaller.model.transcaller_light import TranscallerLight
except ImportError:
    print("="*80)
    print("错误: 无法导入 'TranscallerLight'。")
    print("请确保 /home/lijy/workspace/ 路径正确，")
    print("并且 'my_basecaller/model/transcaller_light.py' 文件存在。")
    print("="*80)
    exit(1)

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
# 步骤 1: 🚀 新的数据集类 (适配 HDF5 & A=1, Blank=0)
# ==========================================================================================

class HDF5Dataset(Dataset):
    """
    针对 HDF5 文件的 PyTorch Dataset。
    采用 Lazy Loading 模式：只有在 __getitem__ 被调用时才从磁盘读取数据，节省内存。
    """
    def __init__(self, h5_path, num_samples_to_load=-1):
        super().__init__()
        self.h5_path = h5_path
        self.h5_file = None # 句柄初始化为 None (Worker 进程中打开)
        
        # 仅主进程打开一次以获取长度
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                self.total_len = f['event'].shape[0]
                print(f"🚀 HDF5 数据集总样本数: {self.total_len}")
        else:
            raise FileNotFoundError(f"找不到文件: {h5_path}")
            
        if num_samples_to_load > 0:
            self.use_len = min(num_samples_to_load, self.total_len)
            print(f"   -> 限制仅使用前 {self.use_len} 个样本。")
        else:
            self.use_len = self.total_len

    def __len__(self):
        return self.use_len

    def __getitem__(self, idx):
        """
        关键：在 Worker 进程中打开文件句柄。
        """
        if self.h5_file is None:
            # swmr=True 允许在写入时读取，libver='latest' 提高性能
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
            
        # 1. 读取数据 (返回的是 numpy array)
        # HDF5 shape: (1, 2048) float32
        event = self.h5_file['event'][idx] 
        # HDF5 shape: (MAX_LABEL_LEN,) int32
        label = self.h5_file['label'][idx] 
        # HDF5 shape: scalar int32
        label_len = self.h5_file['label_len'][idx]

        # 2. 转换为 Tensor
        event_tensor = torch.from_numpy(event).float() # Shape: (1, 2048)
        label_tensor = torch.from_numpy(label).long()  # Shape: (MAX_LEN,)
        label_len_tensor = torch.tensor(label_len).long()

        # 🚀 3. 标签映射检查
        # 用户要求: A=1, C=2, G=3, T=4, Blank=0
        # HDF5存储: A=1, C=2, G=3, T=4, N/Padding=0
        # 
        # 结论: 不需要做任何数学运算！
        # HDF5 中的 0 (Padding) 在 CTCLoss 中自然会被忽略 (由 label_len 控制)，
        # 且我们将 blank_id 设为 0，逻辑完全自洽。
        
        return event_tensor, label_tensor, label_len_tensor

    def __del__(self):
        # 析构时关闭文件句柄
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass

# ==========================================================================================
# 步骤 2: 训练和验证函数
# ==========================================================================================

static_printed = False

def train_one_epoch(model, criterion, optimizer, data_loader, device, output_len, scheduler_warmup, warmup_steps, global_step):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc='[训练]', leave=False)
    
    global static_printed
    
    for events, labels, label_lengths in progress_bar:
        events = events.to(device, non_blocking=True)      # (B, 1, 2048)
        labels = labels.to(device, non_blocking=True)      # (B, MAX_LEN)
        label_lengths = label_lengths.to(device, non_blocking=True) # (B,)
        
        optimizer.zero_grad()
        
        # 前向传播
        log_probs = model(events) # Output shape: (T, B, NumClasses)
        
        batch_size = events.shape[0]
        input_lengths = torch.full(size=(batch_size,), fill_value=output_len, dtype=torch.long, device=device)
        
        # --- 调试打印 (仅一次) ---
        if not static_printed:
            print("\n" + "="*80)
            print("--- 调试：即将送入 CTCLoss 的张量 (仅打印一次) ---")
            print(f"  log_probs.shape:   {log_probs.shape} (T, B, C)")
            print(f"  labels.shape:      {labels.shape}")
            print(f"  label_lengths:     Min={label_lengths.min().item()}, Max={label_lengths.max().item()}")
            
            print("\n  --- 映射检查 (期望: Blank=0, A=1, C=2, G=3, T=4) ---")
            print(f"  labels Min Val:    {labels.min().item()}")
            print(f"  labels Max Val:    {labels.max().item()}")
            
            if labels.min().item() < 0:
                print("  🔥 错误：Labels 包含负数！")
            if labels.max().item() > 4:
                print("  🔥 错误：Labels 包含大于 4 的数！")
            
            print("="*80 + "\n")
            static_printed = True
        
        # 计算损失
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
                
        if torch.isinf(loss):
            print("警告: 遇到 inf 损失，跳过此 batch。")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 预热调度器
        if global_step < warmup_steps:
            scheduler_warmup.step()
        global_step += 1
        
        total_loss += loss.item()
        
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.1e}')
            
    avg_loss = total_loss / len(data_loader)
    return avg_loss, global_step

def validate(model, criterion, data_loader, device, output_len):
    model.eval() 
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc='[验证]', leave=False)
    
    with torch.no_grad():
        for events, labels, label_lengths in progress_bar:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
            
            log_probs = model(events)
            
            batch_size = events.shape[0]
            input_lengths = torch.full(size=(batch_size,), fill_value=output_len, dtype=torch.long, device=device)
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            if not torch.isinf(loss):
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')
            
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# ==========================================================================================
# 步骤 3: 主函数
# ==========================================================================================

def main(args):
    
    # --- 1. 设置环境 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 🚀 模型下采样率 (TranscallerLight 硬编码 s=2, s=2 -> 4)
    MODEL_DOWNSAMPLE_RATIO = 4 
    
    # 🚀 自动校验输出长度
    # HDF5 input_len 默认为 2048 -> expected 512
    expected_output_len = args.input_len // MODEL_DOWNSAMPLE_RATIO
    
    if args.output_len != expected_output_len:
        print(f"提示: 将 output-len 从 {args.output_len} 调整为 {expected_output_len} (基于 input_len {args.input_len})")
        args.output_len = expected_output_len
    
    # --- 2. 准备数据集 (使用 HDF5) ---
    print(f"加载数据集: {args.hdf5_path}")
    
    # 实例化 HDF5Dataset
    full_dataset = HDF5Dataset(args.hdf5_path, args.num_samples)

    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  训练集大小: {train_size}")
    print(f"  验证集大小: {val_size}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,  
        drop_last=True 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- 3. 初始化模型 ---
    print("初始化模型...")
    print(f"  Num Classes: {args.num_classes} (0=Blank, 1=A, 2=C, 3=G, 4=T)")
    
    model = TranscallerLight(
        input_length=args.input_len,   # 2048
        output_length=args.output_len, # 512
        num_classes=args.num_classes,  # 5
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 🚀 关键：设置 blank=0
    criterion = nn.CTCLoss(blank=args.blank_id, zero_infinity=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 调度器
    steps_per_epoch = len(train_loader)
    warmup_steps = steps_per_epoch # 预热 1 个 epoch
    
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-7 / args.lr, end_factor=1.0, total_iters=warmup_steps
    )
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.scheduler_patience, factor=0.5, verbose=True
    )

    # --- 4. 训练循环 ---
    print(f"开始训练... 共 {args.epochs} 轮")
    best_val_loss = float('inf') 
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        start_time = time.time()
        
        # 1. 训练
        train_loss, global_step = train_one_epoch(
            model, criterion, optimizer, train_loader, device, args.output_len,
            scheduler_warmup, warmup_steps, global_step
        )
        
        # 2. 验证
        val_loss = validate(model, criterion, val_loader, device, args.output_len)
        
        # 3. 更新
        if global_step >= warmup_steps:
             scheduler_plateau.step(val_loss)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} 完成. 耗时: {elapsed:.2f}s")
        print(f"  [总结] 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        
        # 保存 checkpoint
        save_path_latest = os.path.join(args.checkpoint_dir, "model_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, save_path_latest)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path_best = os.path.join(args.checkpoint_dir, "model_best_hdf5.pth")
            torch.save(model.state_dict(), save_path_best)
            print(f"  (新最佳模型已保存: {save_path_best})")
            
    print("\n" + "="*80)
    print("训练完成!")

# ==========================================================================================
# 步骤 4: 命令行参数
# ==========================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练 TranscallerLight (使用自定义 HDF5 数据集)")
    
    # --- 关键路径参数 ---
    parser.add_argument('--hdf5-path', type=str, required=True,
                        help="你的 .hdf5 数据集文件路径")
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints_hdf5",
                        help="保存模型 checkpoint 的目录")
    
    # --- 数据集控制 ---
    parser.add_argument('--num-samples', type=int, default=-1,
                        help="调试用：只使用前 N 个样本。-1 表示使用全部。")
    parser.add_argument('--val-split', type=float, default=0.05,
                        help="验证集比例 (默认: 0.05)")
    
    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scheduler-patience', type=int, default=3)

    # --- 模型参数 (HDF5 专用) ---
    parser.add_argument('--input-len', type=int, default=2048,
                        help="输入信号长度 (默认: 2048)")
    parser.add_argument('--output-len', type=int, default=512,
                        help="CTC输出长度 (默认: 2048/4 = 512)")
    
    # --- 🚀 映射配置: A=1, C=2, G=3, T=4, Blank=0 ---
    parser.add_argument('--num-classes', type=int, default=5,
                        help="类别数 5 (0,1,2,3,4)")
    parser.add_argument('--blank-id', type=int, default=0,
                        help="Blank 标签 ID (设置为 0 以匹配你的要求)")
    
    # --- 模型架构 ---
    parser.add_argument('--embed-dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--drop-path', type=float, default=0.1)
    
    args = parser.parse_args()
    
    if args.seed >= 0:
        set_seed(seed=args.seed)
    
    main(args)