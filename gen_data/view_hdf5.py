import h5py
import sys

def print_hdf5_structure(group, indent=''):
    """
    递归打印HDF5文件结构、数据集的shape、dtype以及属性。
    """
    # 遍历组内的所有条目
    for key in group.keys():
        item = group[key]
        
        # 获取条目名称
        name = key
        
        # --- 打印条目 (数据集或组) ---
        if isinstance(item, h5py.Dataset):
            # 如果是数据集
            print(f"{indent}* {name} [Dataset]")
            # 打印 Shape 和 Dtype (这不会加载数据)
            print(f"{indent}  Shape: {item.shape}")
            print(f"{indent}  Dtype: {item.dtype}")
            # 打印属性
            if item.attrs.keys():
                print(f"{indent}  Attributes:")
                for attr_name, attr_val in item.attrs.items():
                    print(f"{indent}    - {attr_name}: {attr_val}")
                    
        elif isinstance(item, h5py.Group):
            # 如果是组
            print(f"{indent}+ {name} [Group]")
            # 打印组的属性
            if item.attrs.keys():
                print(f"{indent}  Attributes:")
                for attr_name, attr_val in item.attrs.items():
                    print(f"{indent}    - {attr_name}: {attr_val}")
            # 递归调用函数，增加缩进
            print_hdf5_structure(item, indent + '  ')

# --- 如何使用 ---
# 将 'YOUR_LARGE_FILE.h5' 替换为你的文件名
file_path = '/home/lijy/windows_ssd/HG002/dataset/HG002_20.h5'

try:
    # 'r' = 只读模式
    with h5py.File(file_path, 'r') as f:
        print(f"Inspecting file: {file_path}")
        print("---")
        print("+ / [Root Group]") # 打印根目录
        print_hdf5_structure(f) # 从根目录开始遍历
        print("---")
        print("Structure inspection complete. No large data was loaded.")

except IOError:
    print(f"Error: Cannot open file '{file_path}'.")
except Exception as e:
    print(f"An error occurred: {e}")