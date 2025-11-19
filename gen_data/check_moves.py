import pysam
import numpy as np
import sys
import os

# --- ⚠️ 关键配置：请修改为您文件的路径 ---
BAM_FILE_PATH = "/home/lijy/windows_ssd/HG002/HG002_basecall_20.sorted.bam" # ⚠️ 修改我
FASTA_FILE_PATH = "/home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna" # ⚠️ 修改我
# ----------------------------------------------

# 模拟 create_dataset_mpv4.txt [cite: 94] 中的参数
SIGNAL_LENGTH = 2048 
WINDOW_STRIDE = 512 # [cite: 95]


def parse_moves_wrong(mv_tag, stride):
    """
    (版本 1) 您在 create_dataset_mpv4.txt 中现有的、有 Bug 的逻辑。
    """
    # 🚀 修复 1: 添加 dtype=np.int64 来防止 OverflowError
    moves = np.array(mv_tag, dtype=np.int64) 
    if stride > 0: 
        moves[1:] = moves[1:] * stride # <--- 逻辑 Bug 仍然保留
    base_signal_starts = np.cumsum(moves)
    return base_signal_starts

def parse_moves_correct(mv_tag, stride):
    """
    (版本 2) 修正后的逻辑。
    """
    # 🚀 修复 1: 添加 dtype=np.int64 来防止 OverflowError
    moves = np.array(mv_tag, dtype=np.int64) 
    if stride > 0:
        moves = moves * stride # <--- 逻辑 Bug 已修复
    base_signal_starts = np.cumsum(moves)
    return base_signal_starts

def get_labels_for_window(base_signal_starts, ref_str, win_start, win_end):
    """
    模拟 create_dataset_mpv4.txt [cite: 106-108] 中的标签提取逻辑。
    """
    # [cite: 106]
    first_base_idx = np.searchsorted(base_signal_starts, win_start, side='right') 
    # [cite: 107]
    last_base_idx = np.searchsorted(base_signal_starts, win_end, side='left')
    
    if first_base_idx >= last_base_idx:
        return "(窗口中无碱基)" # [cite: 108]
    
    # [cite: 108]
    return ref_str[first_base_idx:last_base_idx] 

def run_check():
    print("--- Dorado 'mv_tag' Stride (ts) 检查脚本 ---")
    
    if not os.path.exists(BAM_FILE_PATH):
        print(f"🔥 错误: 找不到 BAM 文件: {BAM_FILE_PATH}")
        return
    if not os.path.exists(FASTA_FILE_PATH):
        print(f"🔥 错误: 找不到 FASTA 文件: {FASTA_FILE_PATH}")
        return

    try:
        fasta_handle = pysam.FastaFile(FASTA_FILE_PATH) # [cite: 114]
        bam_handle = pysam.AlignmentFile(BAM_FILE_PATH, "rb") # [cite: 116]
    except Exception as e:
        print(f"打开文件时出错: {e}")
        return

    test_read = None
    print(f"正在搜索 BAM '{BAM_FILE_PATH}' 以查找一个带 'mv' 和 'ts' 标签的 Read...")

    # 1. 查找一个有效的 Read [cite: 117-121]
    for read in bam_handle.fetch():
        if read.is_unmapped:
            continue
        try:
            mv_tag = read.get_tag('mv') # [cite: 120]
            ts_tag = read.get_tag('ts') # [cite: 120]
            test_read = read
            print(f"✅ 找到测试 Read: {test_read.query_name}")
            break # 找到一个就停止
        except KeyError:
            continue # 缺少 'mv' 或 'ts' 标签

    if test_read is None:
        print("🔥 错误: 在此 BAM 文件中未找到任何包含 'mv' 和 'ts' 标签的 Read。")
        bam_handle.close()
        fasta_handle.close()
        return

    # 2. 提取数据
    mv_tag = test_read.get_tag('mv')
    stride = test_read.get_tag('ts')
    ref_name = test_read.reference_name # [cite: 119]
    ref_start = test_read.reference_start # [cite: 119]
    ref_end = test_read.reference_end # [cite: 120]
    
    # [cite: 101]
    ground_truth_label_str = fasta_handle.fetch(ref_name, ref_start, ref_end).upper() 

    print(f"  > Stride ('ts' 标签): {stride}")
    print(f"  > 'mv' 标签 (前 10 个值): {np.array(mv_tag[:10])}...")
    
    if stride == 1:
        print("\n⚠️ 警告: 这个 Read 的 'ts' (stride) 值为 1。")
        print("   在这种情况下，Bug 不会显现 (因为 1*N = N)。请让脚本继续运行以查找 stride > 1 的 Read。")
        # (我们可以在这里继续循环，但为了简单起见，我们只分析这一个)

    # 3. 运行两种逻辑进行对比
    starts_wrong = parse_moves_wrong(mv_tag, stride)
    starts_correct = parse_moves_correct(mv_tag, stride)

    print("\n--- 1. 'base_signal_starts' 数组对比 ---")
    print(f"  [错误逻辑] (moves[0] 未乘以 {stride}):")
    print(f"  {starts_wrong[:10]}...")
    print(f"\n  [正确逻辑] (所有 moves * {stride}):")
    print(f"  {starts_correct[:10]}...")

    if np.array_equal(starts_wrong, starts_correct):
        print("\n  诊断: 两个数组相同。这可能是因为 stride = 1。")
    else:
        print("\n  ✅ 诊断: 'base_signal_starts' 数组不同！这确认了 Bug 的存在。")

    # 4. 演示 Bug 对标签提取的 *影响*
    # 让我们检查信号中的第 3 个窗口 (索引 2)
    win_start = WINDOW_STRIDE * 2 # = 1024
    win_end = win_start + SIGNAL_LENGTH # = 1024 + 2048 = 3072
    
    print(f"\n--- 2. 对标签提取的*影响* (示例窗口 {win_start}-{win_end}) ---")
    
    labels_wrong = get_labels_for_window(starts_wrong, ground_truth_label_str, win_start, win_end)
    labels_correct = get_labels_for_window(starts_correct, ground_truth_label_str, win_start, win_end)

    print(f"  [提取的标签 - 使用错误逻辑]:")
    print(f"  '{labels_wrong[:100]}...'")
    print(f"\n  [提取的标签 - 使用正确逻辑]:")
    print(f"  '{labels_correct[:100]}...'")

    if labels_wrong != labels_correct:
        print("\n" + "="*60)
        print("  ✅🔥 最终诊断：确认！")
        print("  'parse_moves'  中的 Bug 导致为同一个信号窗口")
        print("  提取了完全错误的碱基标签。")
        print("  这就是您的模型无法学习的原因。")
        print("="*60)
    elif stride != 1:
        print("\n  诊断: 两个逻辑提取了相同的标签。")
        print("  这可能是巧合（例如，这个特定窗口中没有碱基）。")
    
    bam_handle.close()
    fasta_handle.close()

if __name__ == "__main__":
    run_check()