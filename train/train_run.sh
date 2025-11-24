# train bonito model

python ./train_bonito.py --data_path /home/lijy/windows_ssd/HG002/dataset/HG002_single.h5 \
 --checkpoint_path ./checkpoints/bonito_single.pth \
 --epochs 20 --batch_size 64 --log_interval 3


python train.py --data-file /home/lijy/windows_ssd/HG002/dataset/HG002_m5.h5 \
                --num-samples 10000 \
                --lr 0.0001 \
                --val-split 0.05 \
                --batch-size 64 \
                --seed 42 \
                --num-workers 8 \
                --load-in-ram


python train.py --data-file /home/lijy/windows_ssd/HG002/dataset/HG002_20.h5 \
                --num-samples 640 \
                --lr 0.0001 \
                --val-split 0.05 \
                --batch-size 64 \
                --seed 42 \
                --num-workers 0 \
                --load-in-ram

python train.py --hdf5-path /data/biolab-nvme-pcie2/lijy/HG002/dataset/HG002_dorado_10.h5 \
  --num-samples 100000 \
  --lr 0.0005 \
  --epochs 50 \
  --val-split 0.05 \
  --batch-size 128 \
  --seed 42 \
  --num-workers 8 \
  --embed-dim 512 \
  --depth 12 \
  --num-heads 8


python train_from_bonito.py \
    --data-dir /home/lijy/windows_ssd/HG002/dataset/HG002_5_bonito/ \
    --num-samples 50000 \
    --lr 0.0001 \
    --val-split 0.05 \
    --batch-size 64 \
    --seed 42 \
    --num-workers 8


python train_from_bonito.py \
    --data-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset_debug/ \
    --checkpoint-dir ./checkpoints_HG002_debug \
    --num-samples 150000 \
    --lr 0.0005 \
    --val-split 0.05 \
    --seed 42 \
    --num-workers 8 \
    --embed-dim 512 \
    --depth 12 \
    --num-heads 8 \
    --batch-size 128
  


python train_from_bonito.py \
    --data-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10/ \
    --checkpoint-dir ./checkpoints_pod5_10_300k \
    --num-samples 300000 \
    --epochs 30 \
    --lr 0.0001 \
    --val-split 0.05 \
    --seed 11451 \
    --num-workers 8 \
    --embed-dim 512 \
    --depth 12 \
    --num-heads 8 \
    --batch-size 128


python train_from_dorado.py \
    --data-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset/pod5_10_dorado/ \
    --checkpoint-dir ./checkpoints_pod5_10_dorado_50k \
    --num-samples 300000 \
    --epochs 30 \
    --lr 0.0005 \
    --val-split 0.05 \
    --seed 11451 \
    --num-workers 8 \
    --embed-dim 512 \
    --depth 12 \
    --num-heads 8 \
    --batch-size 128


python train_from_dorado.py \
    --data-dir /data/biolab-nvme-pcie2/lijy/HG002/dataset_debug/ \
    --checkpoint-dir ./checkpoints_HG002_debug \
    --num-samples 150000 \
    --lr 0.0005 \
    --val-split 0.05 \
    --seed 42 \
    --num-workers 8 \
    --embed-dim 512 \
    --depth 12 \
    --num-heads 8 \
    --batch-size 128


python train_from_bonito.py \
    --data-dir /data/biolab-nvme-pcie2/lijy/Smadd-seq_data/bonito_basecall/ \
    --checkpoint-dir ./checkpoints_pod5_ang500 \
    --epochs 30 \
    --lr 0.0001 \
    --val-split 0.05 \
    --seed 11451 \
    --num-workers 8 \
    --embed-dim 512 \
    --depth 12 \
    --num-heads 8 \
    --batch-size 128 \
    --input-len 2048