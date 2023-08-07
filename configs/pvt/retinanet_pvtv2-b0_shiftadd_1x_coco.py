_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2_ShiftAdd',
        patch_size=4,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        init_cfg=dict(checkpoint='./ckpt/pvt_v2_b0_LinAngular_binary_both_moe/last_checkpoint.pth')),
    neck=dict(in_channels=[32, 64, 160, 256]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.00005)
work_dir = './ckpt/pvt_v2_b0_shiftadd2'