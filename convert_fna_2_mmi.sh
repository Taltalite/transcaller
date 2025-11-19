# --- 变量定义 ---
# 将 "your_reference.fna" 替换为您的参考基因组文件名
REF_FNA="your_reference.fna"

# 这是您想要生成的索引文件名
REF_MMI="reference.mmi"

# --- 创建索引 ---
echo "正在从 ${REF_FNA} 创建 minimap2 索引..."
minimap2 -d "${REF_MMI}" "${REF_FNA}"
echo "索引 ${REF_MMI} 创建完成."