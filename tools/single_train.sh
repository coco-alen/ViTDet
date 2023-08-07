
CONFIG=./configs/pvt/retinanet_pvtv2-b0_shiftadd_1x_coco.py

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    $CONFIG 
    # --resume-from ./ckpt/pvt_v2_b0_shiftadd/latest.pth