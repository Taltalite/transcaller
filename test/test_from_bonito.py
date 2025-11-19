#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_from_bonito_v3.py

这个脚本加载一个训练好的 TranscallerLight checkpoint，
在验证集上运行它，并计算详细的指标 (F1, Error Rates)
以及生成一个 *汇总所有样本* 的对齐密度热力图。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import argparse
from tqdm import tqdm
import random
import sys
import Levenshtein # 确保已安装: pip install python-Levenshtein
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # <-- 导入 LogNorm

# --- 关键导入 (与训练脚本相同) ---
try:
    sys.path.append('/home/lijy/workspace/')
    from my_basecaller.model.transcaller_light import TranscallerLight
except ImportError:
    print("="*80)
    print("错误: 无法导入 'TranscallerLight'。")
    print("="*80)
    exit(1)

# --- 辅助函数 (与训练脚本相同) ---

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

# --- 数据集类 (与训练脚本相同) ---
class BonitoNpyDataset(Dataset):
    def __init__(self, data_dir, num_samples_to_load=-1):
        super().__init__()
        
        chunks_path = os.path.join(data_dir, "chunks.npy")
        refs_path = os.path.join(data_dir, "references.npy")
        lens_path = os.path.join(data_dir, "reference_lengths.npy")

        print(f"🚀 (测试) 开始将 Bonito .npy 数据从 {data_dir} 加载到内存...")
        
        try:
            print("  (1/3) 正在加载 'chunks.npy'...")
            events_np = np.load(chunks_path)
            print("  (2/3) 正在加载 'references.npy'...")
            labels_np = np.load(refs_path)
            print("  (3/3) 正在加载 'reference_lengths.npy'...")
            label_lens_np = np.load(lens_path)
            
            if num_samples_to_load > 0:
                print(f"  ...截取前 {num_samples_to_load} 个样本。")
                events_np = events_np[:num_samples_to_load]
                labels_np = labels_np[:num_samples_to_load]
                label_lens_np = label_lens_np[:num_samples_to_load]

            print("  正在将数据转换为 Tensors...")
            self.events = torch.from_numpy(events_np).float().unsqueeze(1)
            self.labels = torch.from_numpy(labels_np).long()
            self.label_lens = torch.from_numpy(label_lens_np).long()
            
            print(f"  正在转换标签编码 (Bonito 1-4,0 -> 0-3,4)...")
            self.labels = self.labels - 1
            self.labels[self.labels == -1] = 4 # 4 是我们的 BLANK_ID
            
            self.dataset_len = self.events.shape[0]
            print(f"🚀 (测试) 数据已全部加载到内存。总样本数: {self.dataset_len}")
            
        except Exception as e:
            print(f"加载数据到内存时出错: {e}")
            raise

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.events[idx], self.labels[idx], self.label_lens[idx]

# --- CTC 贪婪解码器 (与之前相同) ---

def ctc_greedy_decode(log_probs, base_map, blank_id=4):
    preds = torch.argmax(log_probs.squeeze(1), dim=-1)
    prev_char_id = -1
    decoded_sequence = []
    for char_id_tensor in preds:
        char_id = char_id_tensor.item()
        if char_id == prev_char_id:
            continue
        if char_id != blank_id:
            decoded_sequence.append(base_map.get(char_id, '?'))
        prev_char_id = char_id
    return "".join(decoded_sequence)

# --- 🚀 新增: 可视化函数 (热力图) ---

def get_normalized_path(opcodes, pred_len, gt_len):
    """
    从 Levenshtein opcodes 生成归一化的 (x, y) 坐标。
    """
    if pred_len == 0 or gt_len == 0:
        return np.array([]), np.array([])
        
    path_i = []
    path_j = []
    
    current_i = 0
    current_j = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k in range(i2 - i1):
                path_i.append(current_i + k)
                path_j.append(current_j + k)
        elif tag == 'replace':
            for k in range(i2 - i1):
                path_i.append(current_i + k)
                path_j.append(current_j + k)
        elif tag == 'insert': # 预测有，真实没有 (y 轴不动)
            for k in range(i2 - i1):
                path_i.append(current_i + k)
                path_j.append(current_j)
        elif tag == 'delete': # 真实有，预测没有 (x 轴不动)
            for k in range(j2 - j1):
                path_i.append(current_i)
                path_j.append(current_j + k)
        
        current_i = i2
        current_j = j2

    # 添加最后一个点
    path_i.append(pred_len)
    path_j.append(gt_len)

    # 归一化
    norm_i = np.array(path_i) / pred_len
    norm_j = np.array(path_j) / gt_len
    
    return norm_i, norm_j

def plot_alignment_heatmap(all_norm_pred, all_norm_gt, output_filename):
    """
    创建所有样本的对齐密度热力图。
    """
    
    plt.figure(figsize=(10, 8))
    
    # 创建 2D 直方图 (热力图)
    # bins: 我们将 [0, 1] 的空间分成 100x100 的格子
    # norm=LogNorm(): 
    #   这是*最关键*的一步。
    #   对角线上的密度会比错误高得多，使用对数刻度才能同时看到两者。
    # plt.hist2d(
    #     all_norm_pred, 
    #     all_norm_gt, 
    #     bins=100, 
    #     cmap='viridis', 
    #     norm=LogNorm(),
    #     range=[[0, 1], [0, 1]] # 确保范围是 0 到 1
    # )
    
    plt.scatter(
        all_norm_pred, 
        all_norm_gt, 
        s=0.8,           # 点的大小
        alpha=0.4,         # 点的透明度 (关键！)
        c='blue',          # 点的颜色
        edgecolors='none', # 移除点的边缘
        label='Alignment Path Points'
    )
    
    # 绘制一条完美的 45° 红色虚线作为参考
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.4, label='Perfect Alignment (y=x)')
    
    plt.xlabel("Predicted Position (Normalized)")
    plt.ylabel("Ground Truth Position (Normalized)")
    plt.title("Alignment Density Heatmap (All Samples)")
    plt.colorbar(label="Alignment Path Density (Log Scale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box') # 设为 1:1 比例
    
    plt.savefig(output_filename)
    print(f"\n可视化汇总热力图已保存至: {output_filename}")


# ==========================================================================================
# 步骤 3: 主测试函数
# ==========================================================================================

def main(args):
    
    # --- 1. 设置环境 ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    BASE_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    # --- 2. 准备数据集 ---
    print("加载数据集中...")
    dataset_to_split = BonitoNpyDataset(args.data_dir, args.num_samples)

    val_size = int(len(dataset_to_split) * args.val_split)
    train_size = len(dataset_to_split) - val_size
    
    _, val_dataset = random_split(
        dataset_to_split, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  成功隔离出验证集。大小: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # --- 3. 初始化和加载模型 ---
    print("初始化模型...")
    model = TranscallerLight(
        input_length=args.input_len,
        output_length=args.output_len,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path
    ).to(device)

    print(f"正在从 {args.checkpoint} 加载权重...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("模型加载成功。")
    except Exception as e:
        print(f"错误：无法加载模型权重: {e}")
        exit(1)

    model.eval()

    # --- 4. 🚀 运行测试 (包含新指标) ---
    
    # 累加器
    total_matches = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    total_gt_len = 0
    total_pred_len = 0
    
    # 🚀 用于汇总图的累加器
    all_norm_pred_coords = []
    all_norm_gt_coords = []
    
    print("\n" + "="*80)
    print(f"开始在 {len(val_dataset)} 个验证样本上进行测试...")
    print("="*80)

    with torch.no_grad():
        for i, (events, labels, label_lengths) in enumerate(tqdm(val_loader, desc="Testing")):
            
            events = events.to(device, non_blocking=True)
            log_probs = model(events) # (T, B, C)
            
            for j in range(log_probs.shape[1]):
                
                log_probs_T_C = log_probs[:, j, :]
                pred_str = ctc_greedy_decode(log_probs_T_C, BASE_MAP, args.blank_id)
                
                gt_len = label_lengths[j].item()
                gt_tensor = labels[j][:gt_len]
                gt_str = "".join([BASE_MAP.get(b.item(), '?') for b in gt_tensor])
                
                if len(gt_str) == 0: continue 
                
                opcodes = Levenshtein.opcodes(pred_str, gt_str)
                
                total_gt_len += len(gt_str)
                total_pred_len += len(pred_str)

                for tag, i1, i2, j1, j2 in opcodes:
                    if tag == 'equal':
                        total_matches += (i2 - i1)
                    elif tag == 'replace':
                        total_substitutions += (i2 - i1)
                    elif tag == 'insert':
                        total_insertions += (i2 - i1)
                    elif tag == 'delete':
                        total_deletions += (j2 - j1)
                
                # 🚀 收集用于绘图的数据
                if args.visualize:
                    norm_i, norm_j = get_normalized_path(opcodes, len(pred_str), len(gt_str))
                    all_norm_pred_coords.append(norm_i)
                    all_norm_gt_coords.append(norm_j)

    print("\n" + "="*80)
    print("测试完成。")
    
    # --- 5. 🚀 绘制汇总图 ---
    if args.visualize and len(all_norm_pred_coords) > 0:
        print("正在生成汇总热力图...")
        all_norm_pred = np.concatenate(all_norm_pred_coords)
        all_norm_gt = np.concatenate(all_norm_gt_coords)
        plot_alignment_heatmap(all_norm_pred, all_norm_gt, args.output_name)
    
    # --- 6. 🚀 计算并打印最终指标 ---
    if total_gt_len > 0 and total_pred_len > 0:
        
        total_errors = total_substitutions + total_insertions + total_deletions
        base_accuracy = (1.0 - (total_errors / total_gt_len)) * 100.0
        
        precision = total_matches / total_pred_len
        recall = total_matches / total_gt_len
        f1 = 2 * (precision * recall) / (precision + recall)
        
        sub_rate = total_substitutions / total_gt_len
        ins_rate = total_insertions / total_gt_len
        del_rate = total_deletions / total_gt_len
        
        print("\n--- 综合评估指标 ---")
        print(f"  碱基准确率 (Base Accuracy):   {base_accuracy:.2f}%")
        print(f"  F1-Score:                   {f1 * 100.0:.2f}%")
        print(f"  Precision (精确率):         {precision * 100.0:.2f}%")
        print(f"  Recall (召回率):            {recall * 100.0:.2f}%")
        
        print("\n--- 错误率细分 (占真实碱基) ---")
        print(f"  替换错误 (Substitutions):   {sub_rate * 100.0:.2f}%")
        print(f"  插入错误 (Insertions):      {ins_rate * 100.0:.2f}%")
        print(f"  删除错误 (Deletions):       {del_rate * 100.0:.2f}%")
        print("\n" + "="*80)
        
    else:
        print("错误：没有处理任何数据。")

# ==========================================================================================
# 步骤 7: Argparse 命令行参数
# ==========================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估 Melchior Basecaller (V3 - 汇总热力图)")
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help="包含 chunks.npy, references.npy, reference_lengths.npy 的目录")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="要测试的 .pth 模型 checkpoint 文件 (例如, ./checkpoints_medium_1M/model_best.pth)")
    parser.add_argument('--output-name', type=str, required=True,
                        help="输出的文件名")
    
    # --- 数据集参数 (必须与训练时相同!) ---
    parser.add_argument('--num-samples', type=int, default=-1)
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    
    # --- 测试参数 ---
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--visualize', action='store_true',
                        help="生成 *汇总* 的对齐热力图")
    
    # --- 模型架构参数 (必须与训练时相同!) ---
    parser.add_argument('--input-len', type=int, default=1998)
    parser.add_argument('--output-len', type=int, default=500)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--blank-id', type=int, default=4)
    
    # 默认使用 "中等模型" 参数
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--drop-path', type=float, default=0.1)
    
    args = parser.parse_args()
    
    main(args)