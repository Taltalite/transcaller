#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_from_bonito.py

这个脚本用于训练 TranscallerLight 模型，
数据源自 `bonito basecaller --save-ctc` 生成的 .npy 文件。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
import random
import sys

# --- 关键导入 ---
try:
    # 确保
    # 1. 路径正确
    # 2. 您的 transcallerlight_model.py 文件在该路径下
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
# 步骤 1: 🚀 新的数据集类 (用于 Bonito .npy 文件)
# ==========================================================================================

class BonitoNpyDataset(Dataset):
    """
    (新版本 - 快速内存加载 .npy)
    通过一次大型的顺序读取，将 Bonito --save-ctc 的 .npy 文件加载到 RAM 中。
    """
    def __init__(self, data_dir, num_samples_to_load=-1):
        super().__init__()
        
        chunks_path = os.path.join(data_dir, "chunks.npy")
        refs_path = os.path.join(data_dir, "references.npy")
        lens_path = os.path.join(data_dir, "reference_lengths.npy")

        print(f"🚀 开始将 Bonito .npy 数据从 {data_dir} 加载到内存...")
        
        try:
            # 1. 加载 Numpy 数组
            print("  (1/3) 正在加载 'chunks.npy'...")
            events_np = np.load(chunks_path)
            
            print("  (2/3) 正在加载 'references.npy'...")
            labels_np = np.load(refs_path)
            
            print("  (3/3) 正在加载 'reference_lengths.npy'...")
            label_lens_np = np.load(lens_path)
            
            # 2. 如果指定了 num_samples，则截取
            if num_samples_to_load > 0:
                print(f"  ...截取前 {num_samples_to_load} 个样本。")
                events_np = events_np[:num_samples_to_load]
                labels_np = labels_np[:num_samples_to_load]
                label_lens_np = label_lens_np[:num_samples_to_load]

            # 3. 将 Numpy 数组转换为 Tensors
            print("  正在将数据转换为 Tensors...")
            # .npy 文件的 shape 是 (B, N)，添加通道维度 (B, 1, N)
            self.events = torch.from_numpy(events_np).float().unsqueeze(1)
            self.labels = torch.from_numpy(labels_np).long()
            self.label_lens = torch.from_numpy(label_lens_np).long()
            
            # 🚀 ==========================================================
            # 🚀 关键修复: 转换标签编码
            # Bonito 编码: A=1, C=2, G=3, T=4, Padding=0
            # 我们的模型期望: A=0, C=1, G=2, T=3, Blank=4
            # 🚀 ==========================================================
            print(f"  正在转换标签编码 (Bonito 1-4,0 -> 0-3,4)...")
            
            # 步骤 1: 将 (A=1...T=4) 转换为 (A=0...T=3)。
            #         这会将 (Padding=0) 变为 (Padding=-1)。
            self.labels = self.labels - 1
            
            # 步骤 2: 将 (Padding=-1) 转换为 (Blank=4)。
            self.labels[self.labels == -1] = 4
            
            print(f"  转换完成。")
            # ==========================================================
            
            self.dataset_len = self.events.shape[0]
            
            print(f"🚀 数据已全部加载到内存。总样本数: {self.dataset_len}")
            print(f"   信号张量 shape: {self.events.shape}")
            print(f"   标签张量 shape: {self.labels.shape}")
            
        except Exception as e:
            print(f"加载数据到内存时出错: {e}")
            raise

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # 直接从 RAM 返回，速度极快
        return self.events[idx], self.labels[idx], self.label_lens[idx]


# ==========================================================================================
# 步骤 2: 训练和验证函数 (与您之前的代码相同)
# ==========================================================================================
# (train_one_epoch 和 validate 函数保持不变)
# (为简洁起见，此处省略，但您应将其粘贴到此处)

static_printed = False

def train_one_epoch(model, criterion, optimizer, data_loader, device, output_len, scheduler_warmup, warmup_steps, global_step):
    model.train() # 设置为训练模式
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc='[训练]', leave=False)
    
    global static_printed
    
    for events, labels, label_lengths in progress_bar:
        events = events.to(device, non_blocking=True) # (B, 1, 1998)
        labels = labels.to(device, non_blocking=True) # (B, 288)
        label_lengths = label_lengths.to(device, non_blocking=True) # (B,)
        
        optimizer.zero_grad()
        log_probs = model(events) # (T, B, C) -> (500, B, 5)
        
        batch_size = events.shape[0]
        input_lengths = torch.full(size=(batch_size,), fill_value=output_len, dtype=torch.long, device=device)
        
        if not static_printed:
            print("\n" + "="*80)
            print("--- 调试：即将送入 CTCLoss 的张量 (仅打印一次) ---")
            print(f"  log_probs.shape:   {log_probs.shape}")
            print(f"  labels.shape:      {labels.shape}")
            print(f"  input_lengths.shape: {input_lengths.shape}")
            print(f"  label_lengths.shape: {label_lengths.shape}")
            
            print("\n  --- 关键检查 ---")
            print(f"  labels (Min / Max):       {labels.min().item()} / {labels.max().item()}")
            print(f"  label_lengths (Min / Max): {label_lengths.min().item()} / {label_lengths.max().item()}")
            
            print(f"\n  input_lengths (前5个): {input_lengths[:5]}")
            print(f"  label_lengths (前5个): {label_lengths[:5]}")
            
            if labels.min().item() < 0:
                print("  🔥 致命错误：在训练循环中发现 'labels' 包含负值！")
            elif labels.max().item() > 4:
                print("  🔥 致命错误：在训练循环中发现 'labels' 包含 > 4 的值！")
                print("     (0=A, 1=C, 2=G, 3=T, 4=Blank)")
            elif label_lengths.min().item() == 0:
                 print("  🔥 致命错误：在训练循环中发现 'label_lengths' 包含 0！")
                 print("     CTCLoss 不允许 0 长度的标签。")
            else:
                 print("  ✅ 数据看起来有效。")

            print("="*80 + "\n")
            static_printed = True
        
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
                
        if torch.isinf(loss):
            print("警告: 遇到 inf 损失，跳过此 batch。")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
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
# 步骤 3: 主函数 (🚀 此部分已修改)
# ==========================================================================================

def main(args):
    
    # --- 1. 设置环境 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 🚀 检查模型数学
    MODEL_DOWNSAMPLE_RATIO = 4 # 硬编码在 FastEmbedLight 中
    
    # 🚀 手动计算正确的输出长度 (基于 1998)
    # L1_out = floor((1998 + 6 - 7)/2) + 1 = 999
    # L2_out = floor((999 + 2 - 3)/2) + 1 = 500
    true_output_len = 512
    
    
    if args.output_len != true_output_len:
        print("="*80)
        print(f"警告: --output-len 参数 ({args.output_len}) 与模型实际下采样不符。")
        print(f"       输入 1998，模型 (s=2, s=2) 产生 {true_output_len} 的输出长度。")
        print(f"       将强制使用 {true_output_len} 进行 CTC 损失计算。")
        print("="*80)
        
        # ‼️ 必须覆盖：我们必须使用正确的长度 (500)
        args.output_len = true_output_len
    
    # --- 2. 准备数据集 (🚀 此部分逻辑已更新) ---
    print("加载数据集中...")
    
    # 🚀 (Utils 1) - 加载到 RAM
    print("🚀 启动 [快速内存加载] 模式...")
    dataset_to_split = BonitoNpyDataset(args.data_dir, args.num_samples)

    actual_input_len = dataset_to_split.events.shape[-1]
    if args.input_len != actual_input_len:
        print("="*80)
        print(f"警告: 您的 --input-len ({args.input_len}) 与数据实际长度 ({actual_input_len}) 不一致。")
        print(f"       将自动使用实际长度 {actual_input_len}。")
        print("="*80)
        args.input_len = actual_input_len

    # (Utils 2) - 划分训练集和验证集
    val_size = int(len(dataset_to_split) * args.val_split)
    train_size = len(dataset_to_split) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset_to_split, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  训练集大小: {train_size}")
    print(f"  验证集大小: {val_size}")

    # (Utils 3) - 创建 Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # 内存数据集可以安全地 shuffle
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

    # --- 3. 初始化模型、损失函数和优化器 ---
    
    print("初始化模型...")
    # 🚀 关键：使用 args.input_len (1998) 和 args.output_len (500)
    model = TranscallerLight(
        input_length=args.input_len,   # (来自数据, 1998)
        output_length=args.output_len, # (模型设计, 500)
        num_classes=args.num_classes,  # (A,C,G,T,blank = 5)
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CTCLoss(blank=args.blank_id, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 调度器 1: 线性预热
    steps_per_epoch = len(train_loader)
    warmup_steps = steps_per_epoch # 预热 1 整个 epoch
    print(f" 每个 Epoch 步数: {steps_per_epoch}")
    print(f" (关键) Warmup 步数 (1 epoch): {warmup_steps}")
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1e-7 / args.lr,
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    
    # 调度器 2: 平台衰减
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=args.scheduler_patience, 
        factor=0.5, 
        verbose=True
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
        
        # 3. 更新调度器
        if global_step >= warmup_steps:
             scheduler_plateau.step(val_loss)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} 完成. 耗时: {elapsed:.2f}s")
        print(f"  [总结] 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        
        # (Utils 3) - 模型保存
        save_path_latest = os.path.join(args.checkpoint_dir, "model_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_plateau_state_dict': scheduler_plateau.state_dict(),
            'scheduler_warmup_state_dict': scheduler_warmup.state_dict(),
            'val_loss': val_loss,
        }, save_path_latest)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path_best = os.path.join(args.checkpoint_dir, "model_from_bonito.pth")
            torch.save(model.state_dict(), save_path_best)
            print(f" (新最佳模型! 验证损失: {val_loss:.4f}, 已保存至 {save_path_best})")
            
    print("\n" + "="*80)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型保存在: {os.path.join(args.checkpoint_dir, 'model_best.pth')}")

# ==========================================================================================
# 步骤 4: Argparse 命令行参数 (🚀 已更新)
# ==========================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练 Melchior Basecaller (使用 Bonito .npy 数据集)")
    
    # --- 数据和路径参数 ---
    parser.add_argument('--data-dir', type=str, required=True,
                        help="包含 chunks.npy, references.npy, reference_lengths.npy 的目录")
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints",
                        help="保存模型 checkpoint 的目录")
    
    # --- 数据集控制 ---
    parser.add_argument('--num-samples', type=int, default=-1,
                        help="要使用的训练样本数量。-1 表示使用全部。 (默认: -1)")
    parser.add_argument('--val-split', type=float, default=0.05, # <-- 您使用的是 0.05
                        help="用于验证集的比例 (例如 0.05 表示 5%) (默认: 0.05)")
    
    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=20,
                        help="训练轮数 (默认: 20)")
    parser.add_argument('--batch-size', type=int, default=128, # <-- 您使用的是 64
                        help="批量大小 (默认: 128)")
    parser.add_argument('--lr', type=float, default=1e-4, # <-- 您使用的是 1e-4
                        help="学习率 (默认: 1e-4)")
    parser.add_argument('--num-workers', type=int, default=8,
                        help="DataLoader 使用的进程数 (默认: 8)")
    parser.add_argument('--seed', type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument('--scheduler-patience', type=int, default=3,
                        help="LR 调度器等待的轮数 (默认: 3)")

    # --- 🚀 模型架构参数 (必须与 Bonito 数据匹配) ---
    parser.add_argument('--input-len', type=int, default=1998,
                        help="输入信号序列长度 (!! 匹配 chunks.npy !! 默认: 1998)")
    parser.add_argument('--output-len', type=int, default=512,
                        help="模型输出序列长度 (!! 匹配 1998/4 a=500 !! 默认: 512)")
    parser.add_argument('--num-classes', type=int, default=5,
                        help="类别数 (A,C,G,T,blank) (默认: 5)")
    parser.add_argument('--blank-id', type=int, default=4,
                        help="CTCLoss 的空白标签 ID (默认: 4)")
    
    # --- (可选) Transcaller 内部参数 ---
    parser.add_argument('--embed-dim', type=int, default=384,
                        help="Transformer 嵌入维度 (默认: 384)")
    parser.add_argument('--depth', type=int, default=6,
                        help="Transformer 层数 (默认: 6)")
    parser.add_argument('--num-heads', type=int, default=4,
                        help="Transformer 注意力头数 (默认: 4)")
    parser.add_argument('--drop-path', type=float, default=0.1,
                        help="随机深度概率 (默认: 0.1)")
    
    
    args = parser.parse_args()
    
    if args.seed >= 0:
        set_seed(seed=args.seed)
    
    print("="*80)
    print("训练配置 (使用 Bonito .npy 数据):")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*80)
    
    main(args)