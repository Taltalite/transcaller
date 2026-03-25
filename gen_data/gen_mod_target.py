import numpy as np
import os

# ================= 配置区域 =================
# 输入文件路径
ref_path = '/data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/references.npy'
# 输出文件路径
save_path = '/data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/mod_targets.npy'

# 定义特殊值
IGNORE_INDEX = -100  # 用于 Padding 的位置，训练 Loss 必须设置 ignore_index=-100
LABEL_UNMODIFIED = 0 # 无修饰碱基的标签

def generate_classification_targets(input_path, output_path):
    print(f"正在加载 Reference 数据: {input_path} ...")
    refs = np.load(input_path)
    # refs shape: (125197, 1345)
    # refs content: 0=Pad, 1=A, 2=C, 3=G, 4=T
    
    # 1. 初始化全为 IGNORE_INDEX (-100) 的矩阵
    # 这样默认所有位置都是 Padding，不需要我们手动去把 0 映射为 -100
    mod_targets = np.full_like(refs, IGNORE_INDEX, dtype=np.int16) # 使用 int16 以支持负数
    
    # 2. 找到所有非 Padding 的位置 (即 A, C, G, T)
    # refs > 0 的位置就是真实碱基的位置
    valid_mask = refs > 0
    
    # 3. 将所有真实碱基的位置标记为 LABEL_UNMODIFIED (0)
    # 这一步实现了：不管你是A还是T，只要不是Pad，你在Mod任务里就是类别0
    mod_targets[valid_mask] = LABEL_UNMODIFIED
    
    # === 验证数据 ===
    print("\n数据生成验证:")
    print(f"原始 References (前10个): {refs[0, :10]}")
    print(f"生成 ModTargets (前10个): {mod_targets[0, :10]}")
    
    # 统计一下分布，确保逻辑正确
    unique, counts = np.unique(mod_targets, return_counts=True)
    dist = dict(zip(unique, counts))
    print("\n标签分布统计:")
    print(f"  Padding ({IGNORE_INDEX}): {dist.get(IGNORE_INDEX, 0)} (应等于 references 中 0 的数量)")
    print(f"  Unmod   ({LABEL_UNMODIFIED}):   {dist.get(LABEL_UNMODIFIED, 0)} (应等于 references 中 1-4 的总和)")
    
    # 保存
    print(f"\n正在保存至: {output_path} ...")
    np.save(output_path, mod_targets)
    print("保存成功！")

if __name__ == "__main__":
    if os.path.exists(ref_path):
        generate_classification_targets(ref_path, save_path)
    else:
        print(f"错误: 找不到输入文件 {ref_path}")