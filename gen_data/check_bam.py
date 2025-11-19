import pysam
import numpy as np

def verify_signal_trim_hypothesis(bam_file_path):
    """
    验证 mv 表推算的信号长度、ns 标签与 ts 标签之间的关系。
    """
    bam = pysam.AlignmentFile(bam_file_path, "rb", check_sq=False)
    
    # 打印表头
    header = (
        f"{'Read ID (Short)':<15} | "
        f"{'Stride':<6} | "
        f"{'NS (Total)':<10} | "
        f"{'Est. Signal':<11} | "
        f"{'Diff (NS-Est)':<13} | "
        f"{'TS (Tag Value)':<15}"
    )
    print(header)
    print("-" * 90)

    count = 0
    for rec in bam:
        if count >= 10: break # 检查前10条
        
        # 1. 获取 mv 并解析 Stride
        try:
            raw_mv = np.array(rec.get_tag("mv"), dtype=int)
            stride = raw_mv[0]       # 既然之前发现是6，这里自动获取验证
            moves = raw_mv[1:]       # 实际的 move 数据
        except KeyError:
            continue

        # 2. 获取 ns (Number of Samples)
        try:
            ns_val = rec.get_tag("ns")
        except KeyError:
            ns_val = 0

        # 3. 获取 ts (Trim Start / Time Start)
        try:
            ts_val = rec.get_tag("ts")
        except KeyError:
            ts_val = -1  # -1 表示没有该标签

        # 4. 计算关系
        # 估算的有碱基对应的信号长度
        est_signal_len = len(moves) * stride
        
        # 差值：总信号里有多少没有被 basecalling 使用（即被 trim 掉的部分）
        diff = ns_val - est_signal_len

        # 打印结果
        short_id = rec.query_name[:15]
        
        print(
            f"{short_id:<15} | "
            f"{stride:<6} | "
            f"{ns_val:<10} | "
            f"{est_signal_len:<11} | "
            f"{diff:<13} | "
            f"{ts_val:<15}"
        )
        
        count += 1

    bam.close()

# 请替换为你的文件名运行
verify_signal_trim_hypothesis('/home/lijy/windows_ssd/HG002/HG002_basecall_5.bam')