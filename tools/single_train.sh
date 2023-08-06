
CONFIG=/data/hyou37/ViTDet/configs/pvt/retinanet_pvtv2-b0_shiftadd_1x_coco.py

CUDA_VISIBLE_DEVICES=3 python tools/train.py \
    $CONFIG \
    --resume-from /data/hyou37/ViTDet/ckpt/pvt_v2_b0_shiftadd/latest.pth