bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --reference /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.mmi /home/lijy/windows_ssd/HG002/pod5_pass_single/ > /home/lijy/windows_ssd/HG002/aligned_reads.bam

bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --reference  /home/lijy/windows_ssd/HG002/pod5_pass/ > /home/lijy/windows_ssd/HG002/aligned_reads.bam


bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --reference  /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna /home/lijy/workspace/single_pod5/ > /home/lijy/windows_ssd/HG002/aligned_reads.bam

bonito basecaller dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --reference  /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna /data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5 > /home/lijy/windows_ssd/HG002/aligned_reads.bam

dorado basecaller hac --reference  /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna /home/lijy/workspace/single_pod5/ > /home/lijy/workspace/aligned_reads.bam


/home/zhaoxy/workspace/software/dorado-0.9.0-linux-x64/bin/dorado basecaller /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
 --reference  /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna /home/lijy/windows_ssd/HG002/pod5_pass_single/ > /home/lijy/workspace/aligned_reads.bam



# 使用dorado做basecalling
/home/lijy/workspace/dorado-1.1.1-linux-x64/bin/dorado basecaller \
    /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --emit-moves --reference /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
    /data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5 > ./HG002_aligned_reads.bam


python ./create_dataset.py --bam_file ./HG002_aligned_reads.sorted.bam --pod5_dir /data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5 \
 --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna --output_hdf5 ./test_hdf5.h5


python ./create_dataset.py --bam_file ./HG002_aligned_reads.sorted.bam --pod5_dir /data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5 \
 --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna --output_hdf5 ./test_hdf5.h5

# small demo
/home/lijy/workspace/dorado-1.1.1-linux-x64/bin/dorado basecaller \
    /data/biolab-nvme-pool1/fanqy/sequencing/bin/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --emit-moves --reference /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna \
    /home/lijy/workspace/my_basecaller/HG002_pod5 > ./HG002_aligned_reads_single.bam

samtools sort HG002_aligned_reads_single.bam -o HG002_aligned_reads_single.sorted.bam

samtools index HG002_aligned_reads_single.sorted.bam


python ./create_dataset.py --bam_file ./HG002_aligned_reads_single.sorted.bam --pod5_dir /home/lijy/workspace/my_basecaller/HG002_pod5 \
 --reference_fasta /home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna --output_hdf5 /data/biolab-nvme-pool1/lijy/HG002_dataset_single.hdf5



#======================= Train ============================

python ./train/train.py --data_path /data/biolab-nvme-pool1/lijy/HG002_dataset/HG002-dorado-dataset.hdf5 --epochs 20 --batch_size 128 --checkpoint_path /home/lijy/workspace/my_basecaller/train/checkpoints/HG002_dorado_10.pth

python ./test/test.py --data_path /data/biolab-nvme-pool1/lijy/HG002_dataset/HG002_dataset_single.hdf5 --checkpoint_path /home/lijy/workspace/my_basecaller/train/checkpoints/best_model.pth --batch_size 128 \
 --visualize_count 5
