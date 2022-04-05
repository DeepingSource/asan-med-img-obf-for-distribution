python train_f.py \
    --dataset brats2020 \
    --arch unet3d \
    --epochs 100 \
    --grouping 128 \
    --batch-size 2 \
    --learning-rate 1e-3 \
    --workers 0 \
    --subfolder _dev