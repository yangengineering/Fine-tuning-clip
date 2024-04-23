#!/bin/bash
# sh scripts/run1.sh

set -e
set -x

CUDA_VISIBLE_DEVICES=4 python main.py \
    --activate_branch 'image_text' \
    --epoch 20 \
    --optimizer 'adam' \
    --lr 2e-4 \
    --scheduler 'CosineAnnealingLR' \
    --batch_size 128 \
    --temperature 16