#!/usr/bin/env bash

CONFIG=/data/hyou37/ViTDet/configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py
CHECKPOINT=./ckpt/pvt_v2_b0.pth
GPUS=2
PORT=12834

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --eval bbox

