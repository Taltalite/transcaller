import numpy as np
import os

# 指向你的数据集目录
data_dir = "/data/biolab-nvme-pcie2/lijy/HG002/dataset_debug/"
chunks_path = os.path.join(data_dir, "chunks.npy")

print(f"Loading {chunks_path}...")
data = np.load(chunks_path)

# 随机抽 1000 个样本计算统计量
sample = data[:1000].flatten()

print("="*40)
print(f"Min:  {sample.min():.4f}")
print(f"Max:  {sample.max():.4f}")
print(f"Mean: {sample.mean():.4f}")
print(f"Std:  {sample.std():.4f}")
print("="*40)

if abs(sample.mean()) > 1.0 or sample.std() > 5.0:
    print("❌ 结论：数据【未归一化】。")
    print("   原始信号数值过大，导致模型无法收敛。")
    print("   请务必在 Dataset 代码中加入归一化。")
else:
    print("✅ 结论：数据【已归一化】。")
    print("   问题可能出在 Learning Rate 或 Blank Token 设置。")