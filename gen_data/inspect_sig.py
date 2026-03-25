import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# 映射表：0=Pad, 1=A, 2=C, 3=G, 4=T
INT_TO_BASE = {0: "", 1: "A", 2: "C", 3: "G", 4: "T"}

def decode_seq(label_arr):
    return "".join([INT_TO_BASE[x] for x in label_arr if x != 0])

def plot_sample(chunks, refs, lens, idx, name, ax):
    signal = chunks[idx].astype(np.float32)
    label_len = lens[idx]
    # 注意：references.npy 可能是 (N, max_len)
    label_seq = refs[idx][:label_len]
    decoded = decode_seq(label_seq)
    
    ax.plot(signal, label='Signal (Normalized)', alpha=0.8, linewidth=0.8)
    ax.set_title(f"{name} - Sample {idx}\nLabel Len: {label_len} | First 20 bases: {decoded[:20]}...")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Normalized pA")
    ax.legend()
    
    # 打印详细信息供检查
    print(f"--- {name} Sample {idx} ---")
    print(f"Signal Mean: {np.mean(signal):.4f}, Std: {np.std(signal):.4f}")
    print(f"Label Length: {label_len}")
    print(f"Full Sequence: {decoded}")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_dir", required=True)
    parser.add_argument("--bonito_dir", required=True)
    parser.add_argument("--indices", type=str, default="0,10,100", help="Comma separated indices to plot")
    parser.add_argument("--output", required=True, help="Path to 'bonito save-ctc' dataset")
    args = parser.parse_args()

    # Load Custom
    c_chunks = np.load(os.path.join(args.custom_dir, "chunks.npy"), mmap_mode='r')
    c_refs = np.load(os.path.join(args.custom_dir, "references.npy"), mmap_mode='r')
    c_lens = np.load(os.path.join(args.custom_dir, "reference_lengths.npy"), mmap_mode='r')

    # Load Bonito
    b_chunks = np.load(os.path.join(args.bonito_dir, "chunks.npy"), mmap_mode='r')
    b_refs = np.load(os.path.join(args.bonito_dir, "references.npy"), mmap_mode='r')
    b_lens = np.load(os.path.join(args.bonito_dir, "reference_lengths.npy"), mmap_mode='r')

    # Handle Shape Mismatch squeeze
    if c_chunks.ndim == 3: c_chunks = c_chunks.squeeze(1)
    if b_chunks.ndim == 3: b_chunks = b_chunks.squeeze(1)

    indices = [int(x) for x in args.indices.split(",")]
    
    nrows = len(indices)
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 4 * nrows), squeeze=False)

    for i, idx in enumerate(indices):
        # Plot Custom
        if idx < len(c_chunks):
            plot_sample(c_chunks, c_refs, c_lens, idx, "Custom", axes[i, 0])
        
        # Plot Bonito (Random sample or same index)
        # 注意：这里的idx在两个数据集中对应的不是同一个Read，只是随机抽查质量
        if idx < len(b_chunks):
            plot_sample(b_chunks, b_refs, b_lens, idx, "Bonito", axes[i, 1])

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved inspection plot to {args.output}")

if __name__ == "__main__":
    main()