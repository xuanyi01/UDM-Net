# UDM-Net fine-tune on HazeUAVvideo dataset
# Inherits full model from udmnet_base, overrides dataset and training schedule

import os
import os.path as osp

_base_ = './udmnet_base.py'

# ---------- Dataset: HazeUAVvideo ----------
def _has_valid_sample(root):
    meta_path = osp.join(root, 'test/meta_info_GT_test.txt')
    if not osp.isfile(meta_path):
        return False
    try:
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key = line.split()[0]
                if '/' not in key:
                    continue
                dataset, folder = key.split('/', 1)
                hazy_clip_dir = osp.join(root, 'test/hazy', dataset, folder)
                if osp.isdir(hazy_clip_dir) and os.listdir(hazy_clip_dir):
                    return True
    except OSError:
        return False
    return False

_data_root_candidates = []
_env_override = os.environ.get('HAZEUAVVIDEO_ROOT')
if _env_override:
    _data_root_candidates.append(_env_override)
_data_root_candidates.extend([
    '/mnt/d/HazeUAVvideo5',
    '/home/ubuntu/data/HazeUAVvideo5',
])
data_root = None
for _candidate in _data_root_candidates:
    if _candidate and osp.isdir(_candidate) and _has_valid_sample(_candidate):
        data_root = _candidate
        break
if data_root is None:
    raise FileNotFoundError(
        'HazeUAVvideo paper split not found. Set HAZEUAVVIDEO_ROOT to the '
        'three-density dataset root (locally HazeUAVvideo5).')

img_norm_cfg_lq = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
img_norm_cfg_gt = dict(mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True)
io_backend = 'disk'
load_kwargs = dict()
crop_size = 256

train_pipeline = [
    dict(type='GenerateFileIndices', interval_list=[1],
         annotation_tree_json=f'{data_root}/train/meta_info_tree_GT_train.json'),
    dict(type='LoadImageFromFileList', io_backend=io_backend,
         key='lq', flag='unchanged', **load_kwargs),
    dict(type='LoadImageFromFileList', io_backend=io_backend,
         key='gt', flag='unchanged', **load_kwargs),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq'], **img_norm_cfg_lq),
    dict(type='Normalize', keys=['gt'], **img_norm_cfg_gt),
    dict(type='PairedRandomCrop', gt_patch_size=crop_size),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'dataset', 'folder', 'haze_beta', 'haze_light']),
]

test_pipeline = [
    dict(type='GenerateFileIndices', interval_list=[1],
         annotation_tree_json=f'{data_root}/test/meta_info_tree_GT_test.json'),
    dict(type='LoadImageFromFileList', io_backend=io_backend,
         key='lq', flag='unchanged', **load_kwargs),
    dict(type='LoadImageFromFileList', io_backend=io_backend,
         key='gt', flag='unchanged', **load_kwargs),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq'], **img_norm_cfg_lq),
    dict(type='Normalize', keys=['gt'], **img_norm_cfg_gt),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'dataset', 'folder', 'haze_beta', 'haze_light']),
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=8, drop_last=True, persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=0,
        persistent_workers=False, pin_memory=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=0,
        persistent_workers=False, pin_memory=False),
    train=dict(
        _delete_=True,
        type='HWFolderMultipleGTDataset',
        lq_folder=f'{data_root}/train/hazy',
        gt_folder=f'{data_root}/train/gt',
        trans_folder=f'{data_root}/train/transmission',
        ann_file=f'{data_root}/train/meta_info_GT_train.txt',
        num_input_frames=5,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        _delete_=True,
        type='HWFolderMultipleGTDataset',
        lq_folder=f'{data_root}/test/hazy',
        gt_folder=f'{data_root}/test/gt',
        trans_folder=f'{data_root}/test/transmission',
        ann_file=f'{data_root}/test/meta_info_GT_test.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        _delete_=True,
        type='HWFolderMultipleGTDataset',
        lq_folder=f'{data_root}/test/hazy',
        gt_folder=f'{data_root}/test/gt',
        trans_folder=f'{data_root}/test/transmission',
        ann_file=f'{data_root}/test/meta_info_GT_test.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        test_mode=True),
)

# ---------- Optimizer (fine-tune LR) ----------
optimizers = dict(
    generator=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(lr_mult=0.5),
                'neck': dict(lr_mult=1.0),
                'dcn': dict(lr_mult=2.0),
                'level1': dict(lr_mult=1.5),
                'faam': dict(lr_mult=1.5),
                'global_refine': dict(lr_mult=2.0),
            }
        )
    )
)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    by_epoch=False,
)

# ---------- Runtime ----------
total_iters = 60000
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
evaluation = None

work_dir = './work_dirs/udmnet_hazeuavvideo'
load_from = None
resume_from = None
