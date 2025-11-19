import h5py
import numpy as np

H5_FILE = "/home/lijy/windows_ssd/HG002/dataset/HG002_20.h5"

with h5py.File(H5_FILE, 'r') as hf:
    print(f"成功打开文件: {H5_FILE}\n")
    
    print("文件中的数据集:")
    print([key for key in hf.keys()])
    
    # 检查形状是否匹配
    event_shape = hf['event'].shape
    label_shape = hf['label'].shape
    len_shape = hf['label_len'].shape
    
    print(f"\n数据集形状 (Samples, C, T):")
    print(f"  event:     {event_shape}")
    print(f"  label:     {label_shape}")
    print(f"  label_len: {len_shape}")

    if not (event_shape[0] == label_shape[0] == len_shape[0]):
        print("\n*** 警告: 数据集样本数量不匹配! ***")
    
    # 打印第一个样本以进行人工检查
    print("\n--- 第一个样本 (Sample 0) ---")
    
    # 将标签从整数转换回碱基
    int_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    sample_len = hf['label_len'][0]
    sample_label_int = hf['label'][0][:sample_len]
    sample_label_str = "".join([int_to_base.get(i, '?') for i in sample_label_int])
    
    print(f"  标签长度 (label_len): {sample_len}")
    print(f"  标签 (label): {sample_label_int}")
    print(f"  标签 (解码后): {sample_label_str}")
    
    signal_sample = hf['event'][0]
    print(f"  信号 (event) 形状: {signal_sample.shape}")
    print(f"  信号 (event) 均值: {np.mean(signal_sample):.4f}")
    print("---------------------------------")