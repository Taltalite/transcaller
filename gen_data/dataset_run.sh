# small demo

/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
 --reference  /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
 --emit-moves /home/lijy/windows_ssd/HG002/pod5_pass_10/ > /home/lijy/windows_ssd/HG002/HG002_basecall_10.bam

/home/lijy/workspace/dorado-1.2.0-linux-x64/bin/dorado basecaller hac \
 --reference  /data/biolab-nvme-pool1/fanqy/sequencing/git/deepsme_data/hg38/hg38.fa \
 --emit-moves /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ > /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.bam

samtools sort -@ 1 /home/lijy/windows_ssd/HG002/HG002_basecall_1.bam -o /home/lijy/windows_ssd/HG002/HG002_basecall_1.sorted.bam

samtools sort -@ 8 /home/lijy/windows_ssd/HG002/HG002_basecall_m5.bam -o /home/lijy/windows_ssd/HG002/HG002_basecall_m5.sorted.bam

samtools index -@ 1 /home/lijy/windows_ssd/HG002/HG002_basecall_1.sorted.bam

samtools index -@ 8 /home/lijy/windows_ssd/HG002/HG002_basecall_m5.sorted.bam

python ./create_dataset_mpv3.py --bam_file /home/lijy/windows_ssd/HG002/HG002_basecall_1.sorted.bam \
 --pod5_dir /home/lijy/windows_ssd/HG002/pod5_pass_single/ \
 --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
 --output_hdf5 /home/lijy/windows_ssd/HG002/dataset/HG002_single.h5 --workers 8


# =============================================================================================

/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
 --reference  /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
 --emit-moves /home/lijy/windows_ssd/HG002/pod5_pass_20/ > /home/lijy/windows_ssd/HG002/HG002_basecall_20.bam

samtools sort -@ 8 /home/lijy/windows_ssd/HG002/HG002_basecall_20.bam -o /home/lijy/windows_ssd/HG002/HG002_basecall_20.sorted.bam

samtools index -@ 8 /home/lijy/windows_ssd/HG002/HG002_basecall_20.sorted.bam

python ./create_datasetv2.py --bam_file /home/lijy/windows_ssd/HG002/HG002_basecall_20.sorted.bam \
 --pod5_dir /home/lijy/windows_ssd/HG002/pod5_pass_20/ \
 --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
 --output_hdf5 /home/lijy/windows_ssd/HG002/dataset/HG002_20.h5

python ./create_dataset_mpv4.py --bam_file /home/lijy/windows_ssd/HG002/HG002_basecall_20.sorted.bam \
 --pod5_dir /home/lijy/windows_ssd/HG002/pod5_pass_20/ \
 --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
 --output_hdf5 /home/lijy/windows_ssd/HG002/dataset/HG002_20.h5 --workers 8

python ./create_dataset_mpv5.py --bam_file /home/lijy/windows_ssd/HG002/HG002_basecall_1.sorted.bam \
 --pod5_dir /home/lijy/windows_ssd/HG002/pod5_pass_single/ \
 --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
 --output_hdf5 /home/lijy/windows_ssd/HG002/dataset/HG002_single.h5 --workers 4

python ./create_dataset_mpv5.py --bam_file /home/lijy/windows_ssd/HG002/HG002_basecall_m5.sorted.bam \
 --pod5_dir /home/lijy/windows_ssd/HG002/pod5_pass_5/merged/ \
 --reference_fasta /home/lijy/windows_ssd/HG002/hg38.fa \
 --output_hdf5 /home/lijy/windows_ssd/HG002/dataset/HG002_m5.h5 --workers 8



# =========================== compare with bonito ====================================
python create_dataset_mp.py \
  --bam_file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito/HG002_bonito_10.sorted.bam \
  --pod5_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
  --reference_fasta /home/lijy/windows_ssd/HG002/hg38.fa \
  --output_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/custom/ \
  --chunk_len 4000 \
  --max_chunks 100000 \
  --workers 16

# =============================================================================================
 
bonito basecaller \
 dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
 --device cuda:0 \
 --save-ctc \
 --reference /home/lijy/windows_ssd/HG002/hg38.fa \
 --chunksize 2000 \
 --alignment-threads 14 \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_bonito_10.bam

# (bonito) lijy@biolab-amd5950x:~/windows_ssd/HG002/dataset$ bonito basecaller \
#  dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
#  --device cuda:0 \
#  --save-ctc \
#  --reference /home/lijy/windows_ssd/HG002/hg38.fa \
#  --chunksize 2000 \
#  --alignment-threads 14 \
#  /home/lijy/windows_ssd/HG002/pod5_pass_5/ \
#  > /home/lijy/windows_ssd/HG002/HG002_bonito_5.bam
# > reading pod5
# > outputting aligned bam
# > downloading model
# [Downloading to /home/lijy/mambaforge/envs/bonito/lib/python3.12/site-packages/bonito/models]
#  - Downloaded: dna_r10.4.1_e8.2_400bps_hac@v5.0.0                                                   
# > loading model dna_r10.4.1_e8.2_400bps_hac@v5.0.0
# > loading reference
# > Chunks rejected from training data:                                                               
#  - low_accuracy0.99: 1950544
#  - no_mapping: 707561
#  - low_coverage0.90: 26251
#  - N_in_sequence: 9
# > written ctc training data to /home/lijy/windows_ssd/HG002
#   - chunks.npy with shape (2213147,1998)
#   - references.npy with shape (2213147,288)
#   - reference_lengths.npy shape (2213147)
# > completed reads: 4950147
# > duration: 0:41:30
# > samples per second 4.0E+06
# > done


/home/lijy/workspace/dorado-1.2.0-linux-x64/bin/dorado basecaller hac \
 --reference  /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
 --emit-moves /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ > /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.bam

samtools sort -@ 8 -o /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.sorted.bam /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.bam
samtools index /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.sorted.bam
samtools view -@ 8 -F 2820 -b -o /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.primary.bam /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.sorted.bam
samtools index /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.primary.bam

# 设置临时目录环境变量
export TMPDIR=/data/biolab-nvme-pcie2/lijy/tmp_cache
# (可选) 设置 Arrow 相关的环境变量，强制它使用我们指定的目录
export ARROW_IO_THREADS=8

python create_dataset_mp.py \
    --bam_file /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.primary.bam \
    --pod5_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
    --output_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/ \
    --workers 8


python compare_dataset.py \
    --bonito_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10/ \
    --my_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado/


python savectc_from_dorado.py \
        --bam /data/biolab-nvme-pcie2/lijy/HG002/HG002_dorado_10.primary.bam \
        --pod5 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
        --reference /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
        --output /data/biolab-nvme-pcie2/lijy/HG002/dataset_debug/ \
        --chunksize 2048 \
        --limit 20000 \
        --workers 8


python create_dataset_mp.py \
    --bam_file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_dorado_10.sorted.bam \
    --pod5_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference_fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado/ \
    --chunk_len 4000 \
    --stride 2000 \
    --max_chunks 200000 \
    --workers 8


python create_dataset_mpv2.py \
    --bam_file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_dorado_10.sorted.bam \
    --pod5_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference_fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_robust_scaling/ \
    --chunk_len 4000 \
    --stride 2000 \
    --max_chunks 200000 \
    --workers 8

# =============================== 2026.01.07 ================================

bonito basecaller \
 dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
 --device cuda:0 \
 --save-ctc \
 --reference /home/lijy/windows_ssd/HG002/hg38.fa \
 --alignment-threads 14 \
 --max-reads 10000 \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup/HG002_bonito_sup_10.bam

# > reading pod5
# > outputting aligned bam
# > loading model dna_r10.4.1_e8.2_400bps_sup@v5.2.0
# /home/lijy/workspace/bonito-uv/bonito/util.py:298: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   state_dict = torch.load(model_file, map_location=device)
# > loading reference
# > Chunks rejected from training data:                                                               
#  - low_accuracy0.99: 56647
#  - no_mapping: 4610
#  - low_coverage0.90: 1125
#  - N_in_sequence: 4
# > written ctc training data to /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup
#   - chunks.npy with shape (137848,12000)
#   - references.npy with shape (137848,1763)
#   - reference_lengths.npy shape (137848)
# > completed reads: 203580
# > duration: 0:16:30
# > samples per second 2.5E+06
# > done

python create_dataset_mpv3.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_dorado_10.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v3/ \
    --chunk-len 4000 \
    --stride 3500 \
    --max-chunks 500000 \
    --workers 8


python create_dataset_mpv4.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_dorado_10.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v4/ \
    --chunk-len 12000 \
    --overlap 600 \
    --max-chunks 500000 \
    --workers 8


python create_dataset_mpv5.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_dorado_10.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v5/ \
    --max-chunks 500000 \
    --workers 8 \
    --chunk-len 12000 \
    --overlap 600 \
    --norm-strategy pa \
    --pa-mean 93.69239463939118 \
    --pa-std 23.506745239082388


python create_dataset_mpv6.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_dorado_10.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v6/ \
    --max-chunks 500000 \
    --workers 8 \
    --chunk-len 12000 \
    --overlap 600 \
    --norm-strategy pa \
    --pa-mean 93.69239463939118 \
    --pa-std 23.506745239082388


python create_dataset_mpv8.py \
    --bam-file /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_dorado_10.sorted.bam \
    --pod5-dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10/ \
    --reference-fasta /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
    --output-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/ \
    --max-chunks 500000 \
    --workers 8 \
    --chunk-len 12000 \
    --overlap 600 \
    --norm-strategy pa \
    --pa-mean 93.69239463939118 \
    --pa-std 23.506745239082388


python compare_sig.py --custom_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v6/  \
 --bonito_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup/  --output ./compare_sig_bonito_doradov6.png


python inspect_sig.py --custom_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v6/  \
 --bonito_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup/  --output ./inspect_sig_bonito_doradov6.png


python check_alignment.py --custom_dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado_v8/ --device cuda:0 \
 --model_path /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/


python check_alignment.py --custom_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup/ --device cuda:0 \
 --model_path /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/


python check_alignment_v2.py --custom_dir /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito_sup/ --device cuda:0 \
 --model_path /home/lijy/workspace/bonito-uv/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/