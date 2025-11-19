import numpy as np
import os

# --- 替换为你的 .npy 文件路径 ---
file_path = '/data/biolab-nvme-pool1/zhangty/PCR1-11/PCR1-11_dataset_train_DTW/chunks.npy'
file_path = '/data/biolab-nvme-pool1/zhangty/PCR1-11/PCR1-11_dataset_train_DTW/references.npy'
# --------------------------------

if not os.path.exists(file_path):
    print(f"错误: 文件未找到 -> {file_path}")
else:
    try:
        # 加载 .npy 文件
        data = np.load(file_path)
        
        # 打印文件的结构信息
        print(f"--- 文件 '{file_path}' 的结构信息 ---")
        print(f"形状 (Shape):    {data.shape}")
        print(f"数据类型 (Dtype): {data.dtype}")
        print(f"维度 (Ndim):     {data.ndim}")
        print(f"元素总数 (Size):  {data.size}")

        # (可选) 查看数据的一小部分预览
        # 为了防止打印过多内容，我们只看前5个元素
        print("\n数据预览 (前5个元素):")
        # .flat 会将多维数组展平为一维，方便预览
        print(data.flat[:5])

    except Exception as e:
        print(f"加载文件时出错: {e}")
        print("请确保这是一个有效的 .npy 文件。")