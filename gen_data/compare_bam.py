import pysam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
my_bam_path = '/data/biolab-nvme-pcie2/lijy/m6A/dataset/m6A_pod5_single_dorado_v8/m6A_pod5_single.sorted.bam'
off_bam_path = '/data/biolab-nvme-pcie2/lijy/m6A/dataset/m6A_pod5_single_bonito/PAW43156_92158b33_73a20312_0_rna004_130bps_sup@v5.2.0.sorted.bam'
output_csv = 'compare_bam_result_m6A_dorado_bonito.csv'
output_fig = 'compare_bam_plot_m6A_dorado_bonito.png'

my_bam_name = 'Dorado set ft. ctccrf Train Model'
off_bam_name = 'Bonito set ft. ctccrf Train Model'

def get_read_stats(bam_path, label):
    """读取BAM并计算每条Read的Identity"""
    stats = {}
    
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch():
            if read.is_unmapped:
                continue
                
            # 计算 Identity
            # NM tag: 编辑距离 (Edit distance)
            # length: 对齐长度 (包含 match, mismatch, deletion, insertion) -- 近似值
            # 更精确的计算通常使用 query_alignment_length 和 NM
            
            # 获取 NM tag (Edit Distance)
            try:
                nm = read.get_tag("NM")
            except KeyError:
                # 如果没有NM tag (未比对), 跳过
                continue
            
            # 序列长度 (aligned part)
            align_len = read.query_alignment_length
            
            # 简单的 Identity 计算公式: 1 - (NM / (Align_len + Deletions)) 
            # 注意：不同论文公式略有不同，这里用最通用的 ONT 定义近似值
            # 实际上更严谨的是: matches / (matches + mismatches + ins + del)
            # 这里的 align_len = matches + mismatches + ins (query上的长度)
            # NM = mismatches + ins + del
            # 所以分母大约是 align_len + (del) ... 稍微复杂，简化处理如下：
            
            # 使用 BLAST-like identity: (len - NM) / len
            # 或者直接计算错误率 error_rate = NM / align_len
            
            if align_len == 0:
                continue
                
            identity = (align_len - nm) / align_len
            
            stats[read.query_name] = {
                f'identity_{label}': identity * 100, # 百分比
                f'len_{label}': align_len
            }
    return stats

def plot_comparison(csv_path, output_img_path, min_identity=0):
    """
    可视化对比结果 (CSV输入 -> 图片输出)
    
    Args:
        csv_path (str): 上一步生成的 comparison_result.csv 路径
        output_img_path (str): 图片保存路径 (例如: 'comparison_plot.png' 或 .pdf)
        min_identity (float): 绘图时过滤的最小一致性，用于放大高分区域 (默认80%)
    """
    print(f"正在加载数据: {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # 过滤掉极低质量的 Reads (通常 <70% 的可能是比对错误或垃圾数据，影响绘图比例)
    # 仅用于绘图，不影响原始数据
    plot_df = df[(df['identity_official'] > min_identity) & (df['identity_mine'] > min_identity)].copy()
    
    if len(plot_df) == 0:
        print("警告: 过滤后没有足够的数据用于绘图，请检查 min_identity 设置。")
        return

    # 设置绘图风格
    sns.set_theme(style="ticks", font_scale=1.2)
    
    # 创建画布: 1行2列
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ==========================================
    # 图表 1: 2D 散点密度图 (Bivariate Histplot)
    # ==========================================
    # 相比普通 scatter，histplot 能够处理大量重叠的点，展示密度
    ax1 = axes[0]
    
    # 绘制对角线 (y=x)
    lims = [min_identity, 100]
    ax1.plot(lims, lims, color='red', linestyle='--', linewidth=1.5, label='y=x (Equal Performance)', zorder=10)
    
    # 绘制数据密度
    # cbar=True 显示颜色条
    h = sns.histplot(
        data=plot_df, 
        x='identity_official', 
        y='identity_mine',
        bins=100, 
        cbar=True, 
        # cmap="Viridis", 
        ax=ax1,
        stat="density"
    )
    
    ax1.set_title(f'Read Identity Comparison\n(Reads > {min_identity}%)', fontweight='bold')
    ax1.set_xlabel(f'{off_bam_name} Identity (%)')
    ax1.set_ylabel(f'{my_bam_name} Identity (%)')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ==========================================
    # 图表 2: 差异直方图 (Difference Histogram)
    # ==========================================
    # 展示 (Mine - Official) 的分布
    ax2 = axes[1]
    
    # 计算统计量
    mean_diff = plot_df['identity_diff'].mean()
    median_diff = plot_df['identity_diff'].median()
    
    sns.histplot(
        data=plot_df, 
        x='identity_diff', 
        kde=True,  # 添加核密度曲线
        bins=100, 
        color='steelblue', 
        ax=ax2,
        line_kws={'linewidth': 2}
    )
    
    # 标记 0 线
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Difference')
    # 标记均值线
    ax2.axvline(mean_diff, color='orange', linestyle='-', linewidth=2, label=f'Mean Diff: {mean_diff:.2f}%')
    
    ax2.set_title(f'Identity Difference Distribution\n({my_bam_name} - {off_bam_name})', fontweight='bold')
    ax2.set_xlabel('Identity Difference (%)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    print(f"正在保存可视化结果至: {output_img_path}")
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    
    # 必须关闭 plot 以释放内存 (因为不显示)
    plt.close()
    print("可视化完成。")

print("正在分析我的模型结果...")
my_stats = get_read_stats(my_bam_path, 'mine')

print("正在分析官方模型结果...")
off_stats = get_read_stats(off_bam_path, 'official')

# === 合并数据 ===
# 将字典转换为 DataFrame
df_mine = pd.DataFrame.from_dict(my_stats, orient='index')
df_off = pd.DataFrame.from_dict(off_stats, orient='index')

# Inner Join: 确保只比较两者都成功比对的 Reads
df = df_mine.join(df_off, how='inner')

# 计算差值
df['identity_diff'] = df['identity_mine'] - df['identity_official']

print(f"\n=== 分析完成 (共匹配 {len(df)} 条 Reads) ===")
print(f"我的模型平均 Identity: {df['identity_mine'].mean():.2f}%")
print(f"官方模型平均 Identity: {df['identity_official'].mean():.2f}%")
print(f"平均差异 (Mine - Official): {df['identity_diff'].mean():.2f}%")

# 保存结果
df.to_csv(output_csv)
print(f"详细对比数据已保存至: {output_csv}")

plot_comparison(output_csv, output_fig)