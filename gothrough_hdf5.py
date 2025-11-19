import h5py
import numpy as np
import argparse
import sys

def print_structure(name, obj):
    """
    一个辅助函数，用于被 .visititems() 调用来打印 HDF5 文件结构。
    """
    if isinstance(obj, h5py.Dataset):
        print(f"  - Dataset: {name}")
        print(f"    Shape: {obj.shape}")
        print(f"    Dtype: {obj.dtype}")

def check_dataset(file_path):
    """
    主函数，用于打开和检查 HDF5 数据集。
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"--- 1. HDF5 文件结构检查 ---")
            print(f"Inspecting file: {file_path}")
            f.visititems(print_structure)
            print("-" * 30)

            # --- 2. 验证关键数据集是否存在 ---
            print("\n--- 2. 关键数据集验证 ---")
            expected_keys = ['event', 'label', 'label_len']
            all_keys_present = True
            for key in expected_keys:
                if key not in f:
                    print(f"[ERROR] 关键数据集 '{key}' 不存在！")
                    all_keys_present = False
                else:
                    print(f"[OK] 关键数据集 '{key}' 存在。")
            
            if not all_keys_present:
                sys.exit("文件结构不完整，请检查您的数据集生成脚本。")
            print("-" * 30)

            # --- 3. 抽样检查第一个样本的数据一致性 ---
            print("\n--- 3. 单个样本抽样检查 (索引 0) ---")
            num_samples = f['event'].shape[0]
            if num_samples == 0:
                print("[WARNING] 数据集为空，无法抽样检查。")
                return
            
            print(f"数据集中总样本数: {num_samples}")

            # 检查 event (信号)
            event_sample = f['event'][0]
            print("\nEvent (信号) 样本:")
            print(f"  - Shape: {event_sample.shape}")
            print(f"  - Dtype: {event_sample.dtype}")
            print(f"  - Min value: {np.min(event_sample):.4f}")
            print(f"  - Max value: {np.max(event_sample):.4f}")
            print(f"  - Mean value: {np.mean(event_sample):.4f}")

            # 检查 label 和 label_len
            label_sample = f['label'][0]
            label_len_sample = f['label_len'][0]
            print("\nLabel (标签) 样本:")
            print(f"  - Label array (前20个值): {label_sample[:20]}...")
            print(f"  - 记录的标签长度 (from 'label_len'): {label_len_sample}")

            # 关键一致性检查
            actual_label_count = np.sum(label_sample != -1) # 计算标签中不等于-1 (填充值) 的元素数量
            print(f"  - 实际的标签长度 (不含填充): {actual_label_count}")

            if label_len_sample == actual_label_count:
                print("\n[SUCCESS] 一致性检查通过！ 'label_len' 与 'label' 内容匹配。")
            else:
                print("\n[ERROR] 一致性检查失败！ 'label_len' 与 'label' 内容不匹配。")
            print("-" * 30)

    except IOError:
        print(f"Error: 无法打开文件 '{file_path}'。请检查路径是否正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查并验证为神经网络准备的 HDF5 数据集。")
    parser.add_argument("-f", "--file", required=True, help="需要检查的 HDF5 文件路径。")
    args = parser.parse_args()
    check_dataset(args.file)