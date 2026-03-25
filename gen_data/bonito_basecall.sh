bonito basecaller \
 dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
 --device cuda:0 \
 --save-ctc \
 --reference /home/lijy/windows_ssd/HG002/hg38.fa \
 --chunksize 4000 \
 --alignment-threads 14 \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito/HG002_bonito_10.bam


bonito basecaller \
 dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
 --device cuda:1 \
 --batchsize 64 \
 --reference /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/HG002_bonito_10.bam


bonito basecaller \
 dna_r10.4.1_e8.2_400bps_hac@v5.2.0 \
 --device cuda:0 \
 --reference /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
 /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1 \
 > /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_1_called/HG002_bonito_1.bam

+

bonito basecaller \
 dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
 --device cuda:0 \
 /data/biolab-nvme-pcie2/lijy/zymo_HWM/R10_data/ \
 > /data/biolab-nvme-pcie2/lijy/zymo_HWM/zymo_HWM_bonito_called/zymo_HWM_bonito_sup@5.2.0.bam


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_FREEZING=0
export TORCH_INDUCTOR_FX_GRAPH_CACHE=0
export TORCH_COMPILE_THREADS=1
bonito basecaller \
 dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
 --device cuda:1 \
 --batchsize 72 \
 --alignment-threads 8 \
 --reference /data/biolab-nvme-pcie2/lijy/HG002/hg38.fa \
 /data/biolab-backup-hdd2/ont-open-data/giab_2025.01_HG002_PAW70337/pod5_1/ \
 > /data/biolab-nvme-pcie2/lijy/HG002/PAW70337/bonito_called/HG002_1_bonito_sup@5.2.0.bam


bonito basecaller \
 dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
 --device cuda:1 \
 --batchsize 72 \
 /data/biolab-backup-hdd2/ont-open-data/giab_2025.01_HG002_PAW70337/pod5_1/ \
 > /data/biolab-nvme-pcie2/lijy/HG002/PAW70337/bonito_called/HG002_1_bonito_sup@5.2.0.bam


# =========================== TRAIN ====================================

bonito train -f /data/biolab-nvme-pcie2/lijy/HG002/bonito_models/bonito_r10_sup@v5.0.0_100k \
 --directory /data/biolab-nvme-pcie2/lijy/HG002/pod5_pass_10_called/bonito/ \
 --config /home/lijy/workspace/my_basecaller/model/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_custom.toml \
 --epochs 30 \
 --chunks 100000 \
 --valid-chunks 10000 \
 --lr 1e-4 \
 --no-amp \
 --grad-accum-split 2
 