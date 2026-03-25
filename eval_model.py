#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import re
import numpy as np
import pandas as pd
import pysam
import mappy as mp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

# ------------------------------------------------------------------------------
# 核心逻辑：解析 CS Tag 并构建完整混淆数据
# ------------------------------------------------------------------------------
def parse_cs_tag_full(cs_str, query_seq):
    """
    结合 Query 序列解析 CS Tag，提取每一个碱基的 True vs Pred。
    
    Args:
        cs_str (str): mappy 返回的 cs 字符串 (如 ":10*ag:5+c:3")
        query_seq (str): read 的原始序列 (必须与 alignment 方向一致)
    
    Returns:
        y_true (list): 参考基因组碱基
        y_pred (list): Basecaller 预测碱基
    """
    y_true = []
    y_pred = []
    
    # 指针指向 Query 序列的当前位置
    q_idx = 0
    query_len = len(query_seq)
    
    # 正则拆解 CS tag
    # :n  -> Match n bases
    # *ac -> Mismatch (Ref:a, Qry:c)
    # +ag -> Insertion in Qry (Ref gap)
    # -ag -> Deletion in Qry (Ref has ag)
    ops = re.findall(r'(:[0-9]+|\*[a-z][a-z]|[+\-][a-z]+)', cs_str)
    
    for op in ops:
        if op.startswith(':'): # Match
            length = int(op[1:])
            # 提取 Query 中对应的片段
            segment = query_seq[q_idx : q_idx + length]
            for base in segment:
                base = base.upper()
                y_true.append(base)
                y_pred.append(base)
            q_idx += length
            
        elif op.startswith('*'): # Mismatch
            ref_base = op[1].upper()
            query_base = op[2].upper()
            # 校验: query_base 应该等于 query_seq[q_idx]
            # (为了速度这里不做 assert，直接信任 cs tag)
            y_true.append(ref_base)
            y_pred.append(query_base)
            q_idx += 1
            
        elif op.startswith('+'): # Insertion (Query has extra)
            ins_seq = op[1:].upper()
            for char in ins_seq:
                y_true.append('-') # Ref is Gap
                y_pred.append(char)
            q_idx += len(ins_seq)
            
        elif op.startswith('-'): # Deletion (Query missed)
            del_seq = op[1:].upper()
            for char in del_seq:
                y_true.append(char)
                y_pred.append('-') # Query is Gap
            # Deletion 不消耗 query 序列长度
            
    return y_true, y_pred

# ------------------------------------------------------------------------------
# 主处理流程
# ------------------------------------------------------------------------------
def evaluate_bam(bam_path, ref_path, max_reads=None):
    print(f"[1/4] Loading Reference Index: {ref_path}")
    # map-ont: Oxford Nanopore reads 预设参数
    aligner = mp.Aligner(ref_path, preset='map-ont')
    if not aligner:
        raise ValueError("Failed to load reference index.")

    print(f"[2/4] Reading BAM and Re-aligning: {bam_path}")
    bam = pysam.AlignmentFile(bam_path, "rb")
    
    # 容器
    all_identities = []
    # 为了节省内存，我们不存 list，而是直接累积混淆矩阵
    # Labels: A, C, G, T, - (Gap)
    label_map = {'A':0, 'C':1, 'G':2, 'T':3, '-':4, 'N':5}
    rev_label_map = {0:'A', 1:'C', 2:'G', 3:'T', 4:'-'}
    
    # 初始化 5x5 混淆矩阵 (忽略 N)
    cm_counts = np.zeros((5, 5), dtype=np.int64)
    
    processed_count = 0
    
    # 使用 tqdm 显示进度
    # 无法预知 BAM 总 read 数，除非读取 index，这里简单用 iterate
    for read in tqdm(bam):
        if max_reads and processed_count >= max_reads:
            break
            
        # 跳过空序列
        seq = read.query_sequence
        if not seq: continue
        
        processed_count += 1
        
        # 使用 mappy 重新比对
        # cs=True 是必须的，用于获取差异详情
        hits = list(aligner.map(seq, cs=True))
        
        if not hits:
            continue
            
        # 取 Primary alignment
        hit = hits[0]
        if not hit.is_primary:
            continue
            
        # 1. 记录 Identity
        # identity = Matches / (Matches + Mismatches + Deletions + Insertions)
        # map-ont 近似计算: 1 - NM / blen
        if hit.blen > 0:
            identities = 1.0 - (hit.NM / hit.blen)
            all_identities.append(identities)
        
        # 2. 构建混淆矩阵数据
        # 注意：mappy 如果比对到反向链，hit.strand 是 -1
        # mappy 会自动处理 seq 的反向互补吗？
        # aligner.map(seq) 中 seq 始终被视为正向。
        # 如果 hit.strand == -1，cs tag 是相对于 seq 的 RC 版本的。
        # 幸运的是，parse_cs_tag_full 需要的是与 cs tag 匹配的序列。
        # 如果是反向比对，mappy 内部比对的是 seq 的 RC。
        # 因此我们需要传入 seq 的 RC 版本给 parser。
        
        query_seq_for_parse = seq
        if hit.strand == -1:
            query_seq_for_parse = mappy_revcomp(seq)

        t_bases, p_bases = parse_cs_tag_full(hit.cs, query_seq_for_parse)
        
        # 填充矩阵
        for t, p in zip(t_bases, p_bases):
            if t in label_map and p in label_map:
                ti = label_map[t]
                pi = label_map[p]
                if ti < 5 and pi < 5: # 忽略 N
                    cm_counts[ti][pi] += 1

    bam.close()
    return all_identities, cm_counts, rev_label_map

def mappy_revcomp(seq):
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement)[::-1]

# ------------------------------------------------------------------------------
# 绘图与报告
# ------------------------------------------------------------------------------
def plot_results(identities, cm_counts, labels, output_prefix):
    # 1. Identity Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(identities, bins=50, kde=True, color='green')
    plt.title('Read Accuracy Distribution')
    plt.xlabel('Identity (Accuracy)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_prefix}_identity.png")
    plt.close()
    
    # 2. Confusion Matrix Heatmap (Row Normalized - Recall)
    # 防止除以0
    row_sums = cm_counts.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1 
    cm_norm = cm_counts.astype('float') / row_sums
    
    label_names = [labels[i] for i in range(5)]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix (Recall / Row Normalized)')
    plt.xlabel('Predicted Base')
    plt.ylabel('True Base (Reference)')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_confusion_matrix.png")
    plt.close()

def print_metrics(cm_counts, labels):
    # 计算全局指标
    # Flatten matrix to simulate y_true and y_pred lists for sklearn (weighted calculation)
    # 但由于数据太大，我们手动计算 F1
    
    # cm[i][j]: True=i, Pred=j
    total_samples = np.sum(cm_counts)
    correct_predictions = np.trace(cm_counts) # 对角线之和
    accuracy = correct_predictions / total_samples
    
    print("\n" + "="*50)
    print(f"MODEL PERFORMANCE REPORT")
    print("="*50)
    print(f"Total Bases Aligned: {total_samples:,}")
    print(f"Global Base Accuracy: {accuracy:.4%}")
    
    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 56)
    
    f1_scores = []
    supports = []
    
    for i in range(5):
        cls_name = labels[i]
        
        # True Positive: 对角线 cm[i][i]
        tp = cm_counts[i][i]
        
        # False Positive: 列和减去 TP (Pred=i, but True!=i)
        fp = np.sum(cm_counts[:, i]) - tp
        
        # False Negative: 行和减去 TP (True=i, but Pred!=i)
        fn = np.sum(cm_counts[i, :]) - tp
        
        support = np.sum(cm_counts[i, :])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        supports.append(support)
        
        print(f"{cls_name:<6} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {support:,}")

    # Macro F1
    macro_f1 = np.mean(f1_scores)
    # Weighted F1
    weighted_f1 = np.average(f1_scores, weights=supports)
    
    print("-" * 56)
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bonito BAM Output")
    parser.add_argument("-b", "--bam", required=True, help="Input BAM file from Bonito")
    parser.add_argument("-r", "--reference", required=True, help="Reference FASTA (hg38.fa)")
    parser.add_argument("-o", "--output_prefix", default="eval_result", help="Prefix for output images")
    parser.add_argument("-n", "--max_reads", type=int, default=None, help="Max reads to process (default: all)")
    
    args = parser.parse_args()
    
    # 1. 计算
    identities, cm_counts, labels = evaluate_bam(args.bam, args.reference, args.max_reads)
    
    if not identities:
        print("Error: No aligned reads found. Check reference or BAM content.")
        return

    # 2. Read-level 统计
    print("\n" + "="*50)
    print("READ-LEVEL STATISTICS")
    print("="*50)
    median_acc = np.median(identities)
    print(f"Median Read Identity: {median_acc:.2%}")
    print(f"Mean Read Identity:   {np.mean(identities):.2%}")
    q_score = -10 * np.log10(1 - median_acc)
    print(f"Estimated Q-Score:    Q{q_score:.1f}")

    # 3. Base-level 统计与 F1
    print_metrics(cm_counts, labels)
    
    # 4. 绘图
    plot_results(identities, cm_counts, labels, args.output_prefix)
    print(f"\n[Done] Plots saved to {args.output_prefix}_identity.png and {args.output_prefix}_confusion_matrix.png")

if __name__ == "__main__":
    main()