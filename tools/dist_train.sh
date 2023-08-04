#!/usr/bin/env bash

CONFIG=/data/hyou37/ViTDet/configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py
GPUS=7
PORT=12834

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch \
    --resume-from /data/hyou37/ViTDet/ckpt/pvt_v2_b0/latest.pth
