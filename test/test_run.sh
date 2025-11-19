python test_transcaller.py --data-file /home/lijy/windows_ssd/HG002/dataset/HG002_20.h5 \
                         --checkpoint /home/lijy/workspace/my_basecaller/train/checkpoints/model_best.pth \
                         --batch-size 32 \
                         --visualize \
                         --vis-output "my_test_visualization.png" \
                         --num-samples 100 \
                         --embed-dim 128 \
                         --depth 1 \
                         --num-heads 2

python test_from_bonito.py \
    --data-dir /home/lijy/windows_ssd/HG002/dataset/HG002_5_bonito/ \
    --checkpoint /home/lijy/workspace/my_basecaller/train/checkpoints_medium_1M/model_from_bonito.pth \
    --num-samples 1000 \
    --val-split 0.05 \
    --seed 42 \
    --visualize \
    --embed-dim 512 \
    --depth 8 \
    --num-heads 8

python test_from_bonito.py \
    --data-dir /home/lijy/windows_ssd/HG002/dataset/HG002_5_bonito/ \
    --checkpoint /home/lijy/workspace/my_basecaller/train/checkpoints_pod5_10_300k/model_from_bonito.pth \
    --num-samples 2000 \
    --val-split 0.05 \
    --seed 42 \
    --visualize \
    --embed-dim 512 \
    --depth 12 \
    --num-heads 8 \
    --output-name "test_from_bonito_pod5_10_300k.png"