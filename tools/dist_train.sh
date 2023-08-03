#!/usr/bin/env bash

CONFIG=./configs/ViTDet/ViTDet-ViT-Base-100e.py
CHECKPOINT=./ckpt/ViT-Base-GPU.pth
GPUS=6
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch
