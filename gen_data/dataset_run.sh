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

 
bonito basecaller \
 dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
 --device cuda:0 \
 --save-ctc \
 --reference /home/lijy/windows_ssd/HG002/hg38.fa \
 --chunksize 2000 \
 --alignment-threads 14 \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10 \
 > /data/biolab-nvme-pcie2/lijy/HG002/HG002_bonito_10.bam

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