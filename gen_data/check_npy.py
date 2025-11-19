import numpy as np

# 替换 'your_file.npy' 为你的文件路径
file_path = '/data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10/reference_lengths.npy'

# 加载 .npy 文件
try:
    data = np.load(file_path)

    # 1. 查看数据结构
    print(f"文件: {file_path}")
    print(f"数据类型 (dtype): {data.dtype}")
    print(f"数据维度 (shape): {data.shape}")
    print(f"维度数量 (ndim): {data.ndim}")
    print(f"元素总数 (size): {data.size}")

    # 2. (可选) 查看数据内容
    # 警告：如果文件非常大，请不要全部打印。
    # 只打印前几个元素作为示例
    if data.ndim == 1:
        print(f"数据示例 (前5个元素): {data[:5]}")
    elif data.ndim > 1:
        print(f"数据示例 (第一个元素/行): \n{data[0]}")

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
except Exception as e:
    print(f"加载文件时出错: {e}")