samtools sort ./HG002_aligned_reads.bam -o ./HG002_aligned_reads.sorted.bam

samtools index ./HG002_aligned_reads.sorted.bam


#-F 2820: This is the key part. It tells samtools to filter out any read that is unmapped (flag 4), secondary (flag 256), supplementary (flag 2048), or fails quality control (flag 512).
samtools view -F 2820 -b -o HG002_primary_only.bam HG002_aligned_reads.sorted.bam

samtools index HG002_primary_only.bam


python create_dataset.py \
    --bam_file ./HG002_aligned_reads.sorted.bam \
    --pod5_dir /data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5/ \
    --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
    --output_hdf5 ./HG002-dorado-dataset.hdf5


python create_dataset.py \
    --bam_file ./HG002_primary_only.sorted.bam \
    --pod5_dir /data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5 \
    --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
    --output_hdf5 ./HG002-dorado-dataset.hdf5


# small demo
samtools sort ./HG002_aligned_reads_single.bam -o ./HG002_aligned_reads_single.sorted.bam

samtools view -F 2820 -b -o ./HG002_primary_only_single.bam ./HG002_aligned_reads_single.sorted.bam

samtools index ./HG002_primary_only_single.bam

python create_dataset.py \
    --bam_file ./HG002_primary_only_single.bam \
    --pod5_dir /home/lijy/workspace/my_basecaller/HG002_pod5 \
    --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
    --output_hdf5 ./HG002-dorado-dataset-single.hdf5