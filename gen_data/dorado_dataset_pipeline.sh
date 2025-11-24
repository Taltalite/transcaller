#!/usr/bin/env bash
set -euo pipefail

# Example pipeline for generating a Transcaller training dataset from Dorado outputs.
# Fill in the paths below before running. Each step mirrors the description in
# AGENT's pipeline recommendation: basecall with moves, align, filter, and
# convert to chunked NumPy arrays.

# 1. Basecall with Dorado (produce BAM with mv/ts tags)
DORADO_MODEL="dna_r10.4.1_e8.2_260bps_sup@v4.1.0"
POD5_DIR="/path/to/pod5"
REFERENCE_FASTA="/path/to/reference.fasta"
DORADO_BAM="./reads.dorado.bam"

dorado basecaller "${DORADO_MODEL}" "${POD5_DIR}" \
    --reference "${REFERENCE_FASTA}" \
    --emit-moves \
    --output-format bam \
    --device auto \
    > "${DORADO_BAM}"

# 2. Sort, index, and keep primary alignments only
SORTED_BAM="./reads.dorado.sorted.bam"
PRIMARY_BAM="./reads.dorado.primary.bam"

# Flag 值,含义 (Description),为什么要剔除？
# 4,Unmapped (未比对上),这条 Read 没有比对到参考基因组上，后续分析用不到。
# 256,Secondary alignment (次要比对),这条 Read 可能比对到了基因组的多个位置，这是次优的位置（通常用于处理多重比对）。
# 512,QC fail (质控不通过),测序仪或软件标记这条 Read 质量有问题。
# 2048,Supplementary alignment (补充比对),这条 Read 的一部分比对到了这里，另一部分比对到了别处（常见于结构变异或嵌合体）。

samtools sort -o "${SORTED_BAM}" "${DORADO_BAM}"
samtools index "${SORTED_BAM}"
samtools view -F 2820 -b -o "${PRIMARY_BAM}" "${SORTED_BAM}"
samtools index "${PRIMARY_BAM}"

# 3. Run the dataset creator (writes chunked .npy files + manifest)
OUTPUT_DIR="./numpy_dataset"
mkdir -p "${OUTPUT_DIR}"

python gen_data/create_dataset_mp.py \
    --bam_file "${PRIMARY_BAM}" \
    --pod5_dir "${POD5_DIR}" \
    --reference_fasta "${REFERENCE_FASTA}" \
    --output_dir "${OUTPUT_DIR}" \
    --workers 8

cat <<EOF
Dataset generation finished.
NumPy chunks are under: ${OUTPUT_DIR}
Manifest file: ${OUTPUT_DIR}/manifest.json
EOF
