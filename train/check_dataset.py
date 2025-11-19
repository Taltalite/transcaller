import h5py
import numpy as np
import random
import sys

# --- 配置 ---

# 1. 您的 HDF5 文件路径
H5_FILE_PATH = "/home/lijy/windows_ssd/HG002/dataset/HG002_20.h5"

# 2. 检查的样本数量
NUM_SAMPLES_TO_CHECK = 10

# 3. 关键假设：您的空白 (Blank) ID
#    [cite_start]根据您的 transcaller_train.txt [cite: 159-160], --blank-id 默认为 4
BLANK_ID = 4 

# (可选) 标签到字符的映射，用于打印
# 假设: 0=A, 1=C, 2=G, 3=T, 4=Blank
LABEL_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T', BLANK_ID: '<B>'}

# --- 脚本正文 ---

def get_true_length_and_str(label_array, blank_id):
    """
    计算标签的真实长度（第一个 blank 出现的位置）
    并返回解码后的字符串。
    """
    true_length = 0
    chars = []
    
    # 查找第一个 blank
    first_blank_idx = -1
    for i, token_id in enumerate(label_array):
        if token_id == blank_id:
            first_blank_idx = i
            break
    
    if first_blank_idx == 0:
        # 标签以 blank 开头，真实长度为 0
        true_length = 0
    elif first_blank_idx > 0:
        # 在中间找到 blank，真实长度就是它的索引
        true_length = first_blank_idx
    else:
        # 数组中没有 blank，说明标签填满了整个数组
        true_length = len(label_array)

    # 解码真实部分的字符串
    for i in range(true_length):
        token_id = label_array[i]
        chars.append(LABEL_MAP.get(token_id, f'?{token_id}?'))
            
    return true_length, "".join(chars)


print(f"--- 正在检查 HDF5 文件: {H5_FILE_PATH} ---")
print(f"--- 假设 BLANK_ID = {BLANK_ID} (基于 transcaller_train.txt) ---")

try:
    with h5py.File(H5_FILE_PATH, 'r') as f:
        # 检查 'keys' 是否存在
        if 'label' not in f or 'label_len' not in f:
            print(f"错误: HDF5 文件中未找到 'label' 或 'label_len'。")
            print(f"文件中的 Keys: {list(f.keys())}")
            sys.exit(1)
            
        total_samples = f['label'].shape[0]
        # 获取标签数组的完整填充长度 (e.g., 200)
        padded_length = f['label'].shape[1] 
        
        print(f"文件加载成功。总样本数: {total_samples}, 标签填充长度: {padded_length}")
        print("-" * 60)

        # 生成随机索引
        if total_samples <= NUM_SAMPLES_TO_CHECK:
            indices_to_check = range(total_samples)
            print(f"总样本数 ({total_samples}) 小于请求数，将检查所有样本。")
        else:
            indices_to_check = random.sample(range(total_samples), k=NUM_SAMPLES_TO_CHECK)
        
        error_found = False
        
        for i, idx in enumerate(indices_to_check):
            print(f"\n[样本 {i+1} / {len(indices_to_check)} (HDF5 索引: {idx})]")
            
            # 1. 读取 HDF5 中存储的 label_len
            stored_label_len = int(f['label_len'][idx])
            
            # 2. 读取完整的 label 数组
            label_array = f['label'][idx] # (e.g., shape 200,)
            
            # 3. 手动计算“真实”长度
            true_len_calculated, label_str = get_true_length_and_str(label_array, BLANK_ID)

            # 4. 打印对比
            print(f"  > (A) 存储的 'label_len': {stored_label_len}")
            print(f"  > (B) 计算的真实长度: {true_len_calculated} (在第一个 <B> 处停止)")
            print(f"  > 解码后的标签 (前100个字符): '{label_str[:100]}...'")
            
            # 5. 诊断
            if stored_label_len == padded_length:
                print("  🔥 诊断: 严重错误!")
                print(f"     'label_len' ({stored_label_len}) 等于填充长度 ({padded_length})。")
                print("     这几乎 100% 是导致 Loss=1.6 的原因。")
                print(f"     CTCLoss 需要的是真实长度 (B)，而不是填充长度 (A)。")
                error_found = True
            elif stored_label_len == true_len_calculated:
                print("  ✅ 诊断: 'label_len' 看起来是正确的。")
            else:
                print("  ⚠️ 诊断: 警告!")
                print(f"     存储的 'label_len' ({stored_label_len}) 与计算出的真实长度 ({true_len_calculated}) 不匹配。")
                print("     请仔细核实您的 BLANK_ID 和数据制作流程。")
                error_found = True

        print("\n" + "=" * 60)
        if error_found:
            print("🔥 检查完成：发现严重问题。请查看上面的 '诊断: 严重错误'。")
            print("   您必须在数据预处理步骤中修复 'label_len' 字段。")
        else:
            print("✅ 检查完成：'label_len' 字段在抽样中看起来没有问题。")
            print("   如果 Loss 仍然是 1.6，问题可能出在：")
            print("   1. 步骤 1 的过拟合测试失败 (模型稳定性问题)。")
            print("   2. 抽样的数据碰巧都是好的 (尝试检查更多样本)。")
            print("   3. 您的 BLANK_ID 不是 4 (请修改脚本顶部的 BLANK_ID)。")
        print("=" * 60)

except ImportError:
    print("错误: 未找到 'h5py' 库。请运行: pip install h5py")
except FileNotFoundError:
    print(f"错误: 文件未找到: {H5_FILE_PATH}")
except Exception as e:
    print(f"\n--- 发生未知错误 ---")
    print(f"错误详情: {e}")