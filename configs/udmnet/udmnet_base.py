# UDM-Net base model config
# Full model: all modules enabled

_base_ = [
    '../_base_/datasets/hazeworld.py',
    '../_base_/default_runtime.py',
]

# ---------- Model ----------
model = dict(
    type='UDM',
    generator=dict(
        type='UDMNet',
        backbone=dict(
            type='ConvNeXt',
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.1,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/'
                           'convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
                prefix='backbone.',
            ),
        ),
        neck=dict(
            type='ProjectionHead',
            in_channels=[96, 192, 384, 768],
            out_channels=64,
            num_outs=4,
        ),
        upsampler=dict(
            type='UDMUpsampler',
            embed_dim=32,
            num_feat=32,
        ),
        channels=32,
        num_trans_bins=32,
        align_depths=(1, 1, 1, 1),
        num_kv_frames=[1, 2, 3],
        # Module activation per level [L0, L1, L2, L3]
        use_dcn_levels=[0, 1, 1, 1],
        use_tdf_levels=[0, 1, 1, 1],
        use_mamba_in_mpg_levels=[0, 1, 1, 1],
        use_mamba_in_msr_levels=[0, 1, 1, 1],
        mamba_bidir_levels=[False, True, True, True],
        mamba_d_state_levels=[16, 16, 16, 16],
        mamba_expand_levels=[2, 2, 2, 2],
        faam_type_levels=['none', 'dcn', 'dcn', 'dcn'],
        use_global_refine=True,
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
)

# ---------- Optimizer ----------
optimizers = dict(
    generator=dict(
        type='AdamW',
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(lr_mult=0.3),
                'neck': dict(lr_mult=0.5),
                'dcn': dict(lr_mult=2.0),
                'level1': dict(lr_mult=1.5),
                'faam': dict(lr_mult=1.5),
                'global_refine': dict(lr_mult=2.0),
            }
        )
    )
)

# ---------- LR Schedule ----------
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=1e-7,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
)

# ---------- Runtime ----------
test_cfg = dict(metrics=['L1', 'PSNR', 'SSIM'], crop_border=0)
train_cfg = None
visual_config = None

total_iters = 80000
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(interval=20000, save_optimizer=True, by_epoch=False)
evaluation = None

custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, priority='HIGH')
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ]
)

work_dir = './work_dirs/udmnet_hazeworld'
