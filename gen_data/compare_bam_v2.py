import pysam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys

# 设置无头模式，适合在服务器运行
import matplotlib
matplotlib.use('Agg')

# 设置绘图风格
sns.set_theme(style="whitegrid", context="talk")

def parse_args():
    parser = argparse.ArgumentParser(description="高级对比: Bonito vs Dorado (包含详细错误分析)")
    parser.add_argument("-b", "--bonito", required=True, help="Bonito 生成的 Aligned BAM")
    parser.add_argument("-d", "--dorado", required=True, help="Dorado 生成的 Aligned BAM")
    parser.add_argument("-o", "--outdir", required=True, help="结果输出目录")
    parser.add_argument("--min_len", type=int, default=500, help="最小Read长度过滤 (默认500bp)")
    parser.add_argument("--max_reads", type=int, default=None, help="最大读取Read数量 (用于快速测试，默认读取全部)")
    return parser.parse_args()

def prob_to_phred(p, max_phred=60):
    """将错误率转换为Q值，并设置上限防止inf"""
    if p <= 0:
        return max_phred
    q = -10 * np.log10(p)
    if q > max_phred:
        return max_phred
    return q

def get_detailed_stats(bam_path, label, min_len=500, max_reads=None):
    """
    解析 BAM 文件，拆解 NM tag 为 Mismatch, Insertion, Deletion
    """
    print(f"[{label}] 正在解析 BAM 文件: {bam_path} ...")
    
    data = []
    mapped_count = 0
    unmapped_count = 0
    processed_count = 0
    
    try:
        # check_sq=False 防止因为header缺失报错，虽然aligned bam通常都有
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam:
                if max_reads and processed_count >= max_reads:
                    break
                processed_count += 1
                
                # 1. 基础过滤
                if read.is_unmapped:
                    unmapped_count += 1
                    continue
                
                if read.query_length < min_len:
                    continue

                # 2. 获取比对信息
                # NM tag: 编辑距离 (Mismatch + Insert + Deletion)
                try:
                    nm = read.get_tag("NM")
                except KeyError:
                    # 如果没有NM tag，无法计算精确Identity，跳过
                    continue

                mapped_count += 1
                
                # read 在比对中实际覆盖的长度 (query alignment length)
                # 这比 query_length 更准确，因为它去除了 soft-clips
                align_len = read.query_alignment_length
                if align_len == 0: continue

                # 3. 拆解错误类型
                # get_cigar_stats 返回一个元组，第一个元素是array
                # array indices: 0=M, 1=I, 2=D, 3=N, 4=S, 5=H, 6=P, 7=EQ, 8=X
                cigar_stats = read.get_cigar_stats()[0]
                
                ins_count = cigar_stats[1]
                del_count = cigar_stats[2]
                
                # 计算 Mismatch
                # 逻辑: NM = Mismatch + Ins + Del
                mismatch_count = nm - ins_count - del_count
                # 修正可能的负数 (理论上不应发生，除非aligner定义不同)
                if mismatch_count < 0: mismatch_count = 0
                
                # 4. 计算不同定义的 Q 值
                
                # A. 预测 Q 值 (Predicted)
                # 从 FASTQ 质量行计算平均值
                raw_quals = read.query_qualities
                pred_q = np.mean(raw_quals) if raw_quals else 0
                
                # B. 严格经验 Q 值 (Strict Empirical)
                # 包含所有错误：Mismatch + Ins + Del
                # 错误率 = NM / align_len
                strict_err = nm / align_len
                strict_q = prob_to_phred(strict_err)
                
                # C. 仅错配 Q 值 (Mismatch-Only Empirical)
                # 仅包含 Mismatch，忽略 Indel
                # 错误率 = Mismatch / align_len
                mis_err = mismatch_count / align_len
                mis_q = prob_to_phred(mis_err)

                # D. Indel 频率 (每100bp)
                ins_rate = (ins_count / align_len) * 100
                del_rate = (del_count / align_len) * 100

                data.append({
                    'tool': label,
                    'read_id': read.query_name,
                    'pred_q': pred_q,
                    'strict_q': strict_q,
                    'mismatch_q': mis_q,
                    'length': read.query_length,
                    'ins_rate': ins_rate,
                    'del_rate': del_rate
                })
                
                if mapped_count % 50000 == 0:
                    print(f"[{label}] 已处理 {mapped_count} 条比对 Reads...")

    except Exception as e:
        print(f"读取 BAM 出错: {e}")
        sys.exit(1)

    df = pd.DataFrame(data)
    print(f"[{label}] 完成。Mapped: {mapped_count}, Unmapped: {unmapped_count}")
    return df, {'mapped': mapped_count, 'unmapped': unmapped_count}

def plot_calibration(df, x_col, y_col, title, filename, outdir):
    """
    通用校准绘图函数
    """
    plt.figure(figsize=(9, 9))
    
    # 下采样：如果数据量太大，随机抽取 10000 个点，避免绘图太慢
    sample_n = 10000
    
    tools = df['tool'].unique()
    colors = {'Bonito': '#3498db', 'Dorado': '#e67e22'} # 蓝橙配色
    
    for tool in tools:
        tool_data = df[df['tool'] == tool]
        if len(tool_data) > sample_n:
            subset = tool_data.sample(n=sample_n, random_state=42)
        else:
            subset = tool_data
            
        sns.scatterplot(
            data=subset, x=x_col, y=y_col, 
            label=tool, color=colors.get(tool, 'gray'),
            alpha=0.3, s=15, edgecolor=None
        )
    
    # 绘制对角线
    min_val = 5
    max_val = 60
    plt.plot([min_val, max_val], [min_val, max_val], ls="--", c="black", alpha=0.8, label="Perfect Calibration")
    
    plt.title(title)
    plt.xlabel("Predicted Q-Score (from Basecaller)")
    plt.ylabel("Empirical Q-Score (calculated from Alignment)")
    plt.xlim(min_val, 50) # 通常 Basecalling Q值很少超过 50
    plt.ylim(min_val, 50)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300)
    plt.close()

def plot_indel_comparison(df, outdir):
    """绘制 Indel 错误率箱线图"""
    # 转换格式
    melted = df.melt(
        id_vars=['tool'], 
        value_vars=['ins_rate', 'del_rate'], 
        var_name='Error Type', value_name='Rate (%)'
    )
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted, x='Error Type', y='Rate (%)', hue='tool', showfliers=False)
    plt.title("Insertion vs Deletion Error Rates")
    plt.savefig(os.path.join(outdir, "indel_rates.png"), dpi=300)
    plt.close()

def plot_identity_dist(df, outdir):
    """绘制基于Strict Identity的分布"""
    # 反推 Identity: Q = -10log10(err) -> err = 10^(-Q/10) -> Identity = 1 - err
    # 直接用 strict_q 计算比较方便
    
    # 为了绘图，我们重新计算百分比 Identity
    # 注意：这里是从 Q 值反推，或者直接用原始数据。这里用简单转换。
    
    plt.figure(figsize=(10, 6))
    
    # 转换 Q 到 Identity %
    # Identity = (1 - 10^(-Q/10)) * 100
    df['identity_pct'] = (1 - 10 ** (-df['strict_q'] / 10)) * 100
    
    # 过滤掉极低值以便观看
    plot_data = df[df['identity_pct'] > 80]
    
    sns.kdeplot(data=plot_data, x="identity_pct", hue="tool", fill=True, clip=(80, 100))
    plt.title("Read Identity Distribution (Strict)")
    plt.xlabel("Identity (%)")
    plt.xlim(80, 100)
    plt.savefig(os.path.join(outdir, "identity_dist.png"), dpi=300)
    plt.close()

def main():
    args = parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print("=== 开始对比分析 ===")
    
    # 1. 数据解析
    df_b, stats_b = get_detailed_stats(args.bonito, "Bonito", args.min_len, args.max_reads)
    df_d, stats_d = get_detailed_stats(args.dorado, "Dorado", args.min_len, args.max_reads)
    
    if df_b.empty or df_d.empty:
        print("错误：解析的数据为空，请检查输入BAM文件是否包含比对信息。")
        return

    df_all = pd.concat([df_b, df_d])
    
    # 2. 绘图 - 校准图 (Calibration)
    
    print("绘制图表 1/4: 严格校准图 (Calibration Strict)...")
    # 这张图就是你之前看到的“偏差很大”的图
    plot_calibration(
        df_all, x_col="pred_q", y_col="strict_q",
        title="Q-Score Calibration (Strict: Mismatches + Indels)",
        filename="calibration_strict.png",
        outdir=args.outdir
    )
    
    print("绘制图表 2/4: 仅错配校准图 (Calibration Mismatch-Only)...")
    # 这张图用于诊断是否是 Indel/SNP 导致的问题
    plot_calibration(
        df_all, x_col="pred_q", y_col="mismatch_q",
        title="Q-Score Calibration (Mismatches Only, ignoring Indels)",
        filename="calibration_mismatch_only.png",
        outdir=args.outdir
    )
    
    # 3. 绘图 - Indel 和 Identity
    print("绘制图表 3/4: Indel 错误率对比...")
    plot_indel_comparison(df_all, args.outdir)
    
    print("绘制图表 4/4: Identity 分布...")
    plot_identity_dist(df_all, args.outdir)
    
    # 4. 生成统计摘要
    summary_path = os.path.join(args.outdir, "summary_report.txt")
    print(f"正在生成统计报告: {summary_path}")
    
    with open(summary_path, "w") as f:
        f.write("=== Basecaller Comparison Report ===\n\n")
        
        for tool, stats in [("Bonito", stats_b), ("Dorado", stats_d)]:
            f.write(f"[{tool}]\n")
            f.write(f"  Mapped Reads:   {stats['mapped']}\n")
            f.write(f"  Unmapped Reads: {stats['unmapped']}\n")
            
            tool_df = df_all[df_all['tool'] == tool]
            f.write(f"  Mean Pred Q:    {tool_df['pred_q'].mean():.2f}\n")
            f.write(f"  Mean Strict Q:  {tool_df['strict_q'].mean():.2f}\n")
            f.write(f"  Mean Mismatch Q:{tool_df['mismatch_q'].mean():.2f}\n")
            f.write(f"  Median Identity:{tool_df['strict_q'].apply(lambda q: (1-10**(-q/10))*100).median():.2f}%\n")
            f.write("\n")
            
    print("分析全部完成！")

if __name__ == "__main__":
    main()