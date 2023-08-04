
CONFIG=/data/hyou37/ViTDet/configs/pvt/retinanet_pvtv2-b0_fpn_1x_coco.py

CUDA_VISIBLE_DEVICES=3 python tools/train.py \
    $CONFIG