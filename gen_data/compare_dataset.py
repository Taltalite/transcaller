import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# 映射表 (Bonito 和你的脚本通常都是这个标准)
# 1=A, 2=C, 3=G, 4=T, 0=Pad
INT_TO_BASE = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: ''}

def decode_label(label_arr):
    """将整数数组解码为字符串"""
    return "".join([INT_TO_BASE.get(x, '?') for x in label_arr if x != 0])

def load_dataset_stats(name, data_dir, file_names):
    """加载数据并返回统计信息"""
    print(f"\n[{name}] 正在加载数据: {data_dir} ...")
    
    sig_path = os.path.join(data_dir, file_names['signal'])
    lbl_path = os.path.join(data_dir, file_names['label'])
    len_path = os.path.join(data_dir, file_names['length'])

    if not os.path.exists(sig_path):
        print(f"❌ 错误: 找不到文件 {sig_path}")
        return None

    # 使用 mmap 模式以避免内存爆炸
    signals = np.load(sig_path, mmap_mode='r')
    labels = np.load(lbl_path, mmap_mode='r')
    lengths = np.load(len_path, mmap_mode='r')

    # 1. 基础形状检查
    print(f"  -> 形状: Signal={signals.shape}, Label={labels.shape}")
    
    # 2. 信号统计 (取前1000个样本估算)
    sample_sigs = signals[:1000].flatten()
    print(f"  -> 信号分布: Mean={np.mean(sample_sigs):.4f}, Std={np.std(sample_sigs):.4f}, Min={np.min(sample_sigs):.4f}, Max={np.max(sample_sigs):.4f}")

    # 3. 标签长度统计
    print(f"  -> 标签长度: Mean={np.mean(lengths):.2f}, Max={np.max(lengths)}, Min={np.min(lengths)}")

    # 4. 标签值域检查 (检查是否有异常的类别 ID)
    unique_labels = np.unique(labels[:1000])
    print(f"  -> 标签类别ID分布: {unique_labels} (预期应在 0-4 之间)")

    return {
        "signals": signals,
        "labels": labels,
        "lengths": lengths
    }

def plot_comparison(data_bonito, data_dorado):
    """画图对比"""
    if data_bonito is None or data_dorado is None:
        return

    plt.figure(figsize=(15, 8))

    # 随机取一个索引
    idx_b = np.random.randint(0, len(data_bonito['signals']))
    idx_d = np.random.randint(0, len(data_dorado['signals']))

    # --- Bonito 样本 ---
    ax1 = plt.subplot(2, 1, 1)
    sig_b = data_bonito['signals'][idx_b]
    # Bonito通常是 (1998,) 或 (1, 1998)，展平它
    sig_b = sig_b.flatten() 
    lbl_b = decode_label(data_bonito['labels'][idx_b])
    
    ax1.plot(sig_b, color='blue', alpha=0.7)
    ax1.set_title(f"Bonito Sample (Idx: {idx_b})\nLabel (Len {len(lbl_b)}): {lbl_b[:100]}...", loc='left')
    ax1.set_ylabel("Normalized Current")
    ax1.grid(True, alpha=0.3)

    # --- Dorado 样本 ---
    ax2 = plt.subplot(2, 1, 2)
    sig_d = data_dorado['signals'][idx_d]
    # Dorado脚本生成的是 (1, 2048)，展平它
    sig_d = sig_d.flatten()
    lbl_d = decode_label(data_dorado['labels'][idx_d])

    ax2.plot(sig_d, color='green', alpha=0.7)
    ax2.set_title(f"Dorado/MyDataset Sample (Idx: {idx_d})\nLabel (Len {len(lbl_d)}): {lbl_d[:100]}...", loc='left')
    ax2.set_ylabel("Normalized Current")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    print("\n✅ 对比图已保存为 'comparison_plot.png'。请查看。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bonito_dir', required=True, help="Bonito生成的数据集目录")
    parser.add_argument('--my_dir', required=True, help="你自己生成的Dataset目录")
    args = parser.parse_args()

    # 配置不同的文件名
    # Bonito 默认通常是 chunks.npy
    # 你的脚本生成的是 signals.npy
    data_b = load_dataset_stats("Bonito", args.bonito_dir, 
                                {'signal': 'chunks.npy', 'label': 'references.npy', 'length': 'reference_lengths.npy'})
    
    data_d = load_dataset_stats("MyDataset", args.my_dir, 
                                {'signal': 'signals.npy', 'label': 'labels.npy', 'length': 'lengths.npy'})

    plot_comparison(data_b, data_d)