import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from termcolor import colored

def load_dataset(base_dir):
    """Safely load dataset files using memmap to save RAM."""
    try:
        chunks = np.load(os.path.join(base_dir, "chunks.npy"), mmap_mode='r')
        refs = np.load(os.path.join(base_dir, "references.npy"), mmap_mode='r')
        lens = np.load(os.path.join(base_dir, "reference_lengths.npy"), mmap_mode='r')
        return chunks, refs, lens
    except Exception as e:
        print(colored(f"[Error] Failed to load from {base_dir}: {e}", "red"))
        return None, None, None

def get_stats(data, name, sample_size=10000):
    """Calculate basic statistics on a random sample."""
    N = data.shape[0]
    indices = np.random.choice(N, min(N, sample_size), replace=False)
    sample = data[indices]
    
    return {
        "mean": np.mean(sample),
        "std": np.std(sample),
        "min": np.min(sample),
        "max": np.max(sample),
        "shape": data.shape,
        "sample": sample
    }

def compare_signals(custom_chunks, bonito_chunks):
    print(colored("\n=== 1. Signal (Chunks) Comparison ===", "yellow"))
    
    # 采样 10000 个片段计算统计量
    c_stats = get_stats(custom_chunks, "Custom")
    b_stats = get_stats(bonito_chunks, "Bonito")
    
    print(f"{'Metric':<15} | {'Custom Dataset':<25} | {'Bonito Dataset':<25}")
    print("-" * 70)
    print(f"{'Shape':<15} | {str(c_stats['shape']):<25} | {str(b_stats['shape']):<25}")
    print(f"{'Mean':<15} | {c_stats['mean']:<25.4f} | {b_stats['mean']:<25.4f}")
    print(f"{'Std Dev':<15} | {c_stats['std']:<25.4f} | {b_stats['std']:<25.4f}")
    print(f"{'Min':<15} | {c_stats['min']:<25.4f} | {b_stats['min']:<25.4f}")
    print(f"{'Max':<15} | {c_stats['max']:<25.4f} | {b_stats['max']:<25.4f}")
    
    # 绘制直方图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(c_stats['sample'].flatten(), bins=100, alpha=0.7, label='Custom', color='blue', density=True)
    plt.hist(b_stats['sample'].flatten(), bins=100, alpha=0.7, label='Bonito', color='orange', density=True)
    plt.title("Signal Value Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 诊断建议
    diff_mean = abs(c_stats['mean'] - b_stats['mean'])
    if diff_mean > 10:
        print(colored("\n[CRITICAL WARNING] Signal Distribution Mismatch!", "red"))
        print(f"  Custom mean is {c_stats['mean']:.2f}, Bonito is {b_stats['mean']:.2f}.")
        print("  -> Possibility 1: Bonito dataset is z-score normalized (Mean=0, Std=1) but yours is raw PA.")
        print("  -> Possibility 2: Different scaling factors in POD5 calibration.")

def compare_labels(custom_refs, bonito_refs, custom_lens, bonito_lens, output):
    print(colored("\n=== 2. Label (Reference) Comparison ===", "yellow"))
    
    # 检查词表 (Vocab)
    c_vocab = np.unique(custom_refs)
    b_vocab = np.unique(bonito_refs)
    
    print(f"Custom Vocab (Unique Integers): {c_vocab}")
    print(f"Bonito Vocab (Unique Integers): {b_vocab}")
    
    if not np.array_equal(c_vocab, b_vocab):
        print(colored("[WARNING] Vocabulary Mismatch!", "red"))
        print("  Check if you are using 1-based indexing (1=A) vs 0-based.")
        print("  Bonito standard: 0=Pad, 1=A, 2=C, 3=G, 4=T (usually).")

    # 检查长度分布
    print(f"\n{'Metric':<15} | {'Custom':<10} | {'Bonito':<10}")
    print("-" * 45)
    print(f"{'Mean Length':<15} | {np.mean(custom_lens):<10.1f} | {np.mean(bonito_lens):<10.1f}")
    print(f"{'Max Length':<15} | {np.max(custom_lens):<10} | {np.max(bonito_lens):<10}")
    
    # 绘制长度分布
    plt.subplot(1, 2, 2)
    plt.hist(custom_lens, bins=50, alpha=0.7, label='Custom', color='blue', density=True)
    plt.hist(bonito_lens, bins=50, alpha=0.7, label='Bonito', color='orange', density=True)
    plt.title("Label Length Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output)
    print(colored(f"\n[INFO] Comparison plot saved to '{output}'", "green"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_dir", required=True, help="Path to your generated dataset")
    parser.add_argument("--bonito_dir", required=True, help="Path to 'bonito save-ctc' dataset")
    parser.add_argument("--output", required=True, help="Path to 'bonito save-ctc' dataset")
    args = parser.parse_args()
    
    print(f"Loading Custom: {args.custom_dir}")
    c_chunks, c_refs, c_lens = load_dataset(args.custom_dir)
    
    print(f"Loading Bonito: {args.bonito_dir}")
    b_chunks, b_refs, b_lens = load_dataset(args.bonito_dir)
    
    if c_chunks is None or b_chunks is None:
        return

    # Handle Shape Mismatch (Squeeze if necessary for comparison)
    if c_chunks.ndim == 2 and b_chunks.ndim == 3:
        print("[Note] Squeezing Bonito chunks (N, 1, T) -> (N, T) for comparison.")
        b_chunks = b_chunks.squeeze(1)
    elif c_chunks.ndim == 3 and b_chunks.ndim == 2:
        print("[Note] Squeezing Custom chunks (N, 1, T) -> (N, T) for comparison.")
        c_chunks = c_chunks.squeeze(1)

    compare_signals(c_chunks, b_chunks)
    compare_labels(c_refs, b_refs, c_lens, b_lens, args.output)

if __name__ == "__main__":
    main()