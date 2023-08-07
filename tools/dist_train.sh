#!/usr/bin/env bash

CONFIG=./configs/pvt/retinanet_pvtv2-b0_shiftadd_1x_coco.py
GPUS=8
PORT=17423

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch 
    # --resume-from ./ckpt/pvt_v2_b0_shiftadd/latest.pth