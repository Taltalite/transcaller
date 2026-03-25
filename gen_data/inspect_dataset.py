#!/usr/bin/env python
"""
Bonito Dataset Inspection & Comparison Tool.
Compares two directories containing {chunks, references, reference_lengths}.npy
"""

import argparse
import os
import numpy as np
import sys

# 设置打印格式
np.set_printoptions(precision=4, suppress=True)

FILES = ["chunks.npy", "references.npy", "reference_lengths.npy"]

def get_stats(data: np.ndarray, name: str):
    """Calculates statistical features of a numpy array using sampling."""
    
    # 为了速度，如果数据量太大，只随机采样 50,000 个样本进行统计
    N = data.shape[0]
    if N > 50000:
        indices = np.random.choice(N, 50000, replace=False)
        sample = data[indices]
    else:
        sample = data

    stats = {
        "shape": data.shape,
        "dtype": data.dtype,
        "count": N
    }

    if "chunks" in name:
        # Signal Statistics (Float)
        flat = sample.flatten()
        stats.update({
            "mean": np.mean(flat),
            "std": np.std(flat),
            "min": np.min(flat),
            "max": np.max(flat),
            "p05": np.percentile(flat, 5),
            "median": np.median(flat),
            "p95": np.percentile(flat, 95),
        })
    elif "references" in name:
        # Label Statistics (Int) - Base distribution
        # 0=Pad, 1=A, 2=C, 3=G, 4=T
        flat = sample.flatten()
        # 排除 0 (Padding) 后计算 ACGT 分布
        valid_bases = flat[flat > 0]
        if valid_bases.size > 0:
            unique, counts = np.unique(valid_bases, return_counts=True)
            total = valid_bases.size
            dist = {k: v/total for k, v in zip(unique, counts)}
            stats["base_dist"] = dist
            stats["valid_ratio"] = valid_bases.size / flat.size # 填充率
        else:
            stats["base_dist"] = {}
            stats["valid_ratio"] = 0

    elif "lengths" in name:
        # Length Statistics
        stats.update({
            "mean_len": np.mean(sample),
            "std_len": np.std(sample),
            "max_len": np.max(sample)
        })

    return stats

def analyze_directory(path, label):
    print(f"\n[{label.upper()}] Analyzing: {path}")
    results = {}
    
    for fname in FILES:
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            print(f"  [MISSING] {fname} not found!")
            results[fname] = None
            continue
        
        try:
            # 使用 mmap_mode='r' 避免将大文件全部读入内存
            data = np.load(fpath, mmap_mode='r')
            print(f"  Load {fname}: OK")
            results[fname] = get_stats(data, fname)
        except Exception as e:
            print(f"  [ERROR] Could not load {fname}: {e}")
            results[fname] = None
            
    return results

def print_comparison(res1, res2, name1="Custom", name2="Bonito"):
    print("\n" + "="*80)
    print(f"COMPARISON REPORT: {name1} vs {name2}")
    print("="*80)

    # 1. Chunks (Signal)
    print(f"\n>> 1. Signal Analysis (chunks.npy)")
    c1, c2 = res1.get("chunks.npy"), res2.get("chunks.npy")
    
    if c1 and c2:
        print(f"{'Metric':<20} | {name1:<25} | {name2:<25} | {'Diff Check':<15}")
        print("-" * 90)
        
        # Dimensions
        print(f"{'Shape':<20} | {str(c1['shape']):<25} | {str(c2['shape']):<25} | {'N/A'}")
        print(f"{'Dtype':<20} | {str(c1['dtype']):<25} | {str(c2['dtype']):<25} | {'OK' if c1['dtype'] == c2['dtype'] else 'MISMATCH'}")
        
        # Value Stats
        metrics = ["mean", "std", "min", "max", "median"]
        for m in metrics:
            val1 = c1[m]
            val2 = c2[m]
            diff = abs(val1 - val2)
            # 判断是否在一个数量级
            check = "CLOSE" if diff < (0.1 * abs(val1)) or diff < 1.0 else "DIFF"
            if m == "mean":
                # 特殊逻辑判断 PA vs Standardized
                if abs(val1) < 5 and abs(val2) > 50: check = "TYPE MISMATCH (Norm vs PA)"
                if abs(val1) > 50 and abs(val2) < 5: check = "TYPE MISMATCH (PA vs Norm)"

            print(f"{m.capitalize():<20} | {val1:10.4f}                | {val2:10.4f}                | {check}")

        print("\n[Analysis Hint]:")
        if c1['mean'] > 50:
            print(f"  * {name1} looks like PicoAmps (Raw Signal).")
        elif abs(c1['mean']) < 5:
            print(f"  * {name1} looks like Standardized Data (Mean=0, Std=1).")
            
    else:
        print("  Skipping comparison (missing files).")

    # 2. Labels
    print(f"\n>> 2. Label Analysis (references.npy + lengths.npy)")
    r1, r2 = res1.get("references.npy"), res2.get("references.npy")
    if r1 and r2:
        print(f"{'Metric':<20} | {name1:<25} | {name2:<25}")
        print("-" * 70)
        
        # Base Distribution
        bases = [1, 2, 3, 4] # A C G T
        base_map = {1:'A', 2:'C', 3:'G', 4:'T'}
        
        for b in bases:
            p1 = r1['base_dist'].get(b, 0) * 100
            p2 = r2['base_dist'].get(b, 0) * 100
            print(f"Base {base_map[b]} %{'':<13} | {p1:6.2f}%                    | {p2:6.2f}%")
            
        print(f"Non-Pad Ratio{'':<7} | {r1['valid_ratio']*100:6.2f}%                    | {r2['valid_ratio']*100:6.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Compare two Bonito datasets.")
    parser.add_argument("custom_dir", help="Path to your generated dataset")
    parser.add_argument("bonito_dir", help="Path to bonito --save-ctc output")
    args = parser.parse_args()

    if not os.path.isdir(args.custom_dir) or not os.path.isdir(args.bonito_dir):
        print("Error: Input paths must be directories.")
        sys.exit(1)

    s1 = analyze_directory(args.custom_dir, "Custom Script")
    s2 = analyze_directory(args.bonito_dir, "Bonito Official")

    print_comparison(s1, s2)

if __name__ == "__main__":
    main()