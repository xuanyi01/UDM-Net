# HazeWorld dataset configuration
# Adjust data_root to your local path

import os
import os.path as osp

data_root_candidates = [
    os.environ.get('HAZEWORLD_ROOT', ''),
    '/home/ubuntu/data/HazeWorld',
    '/mnt/d/HazeWorld',
]
data_root = '/home/ubuntu/data/HazeWorld'
for _c in data_root_candidates:
    if _c and osp.isdir(_c):
        data_root = _c
        break

train_dataset_type = 'HWFolderMultipleGTDataset'
test_dataset_type = 'HWFolderMultipleGTDataset'

img_norm_cfg_lq = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
img_norm_cfg_gt = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True)
crop_size = 256
num_input_frames = 5

io_backend = 'disk'
load_kwargs = dict()

train_pipeline = [
    dict(type='GenerateFileIndices',
         interval_list=[1],
         annotation_tree_json=f'{data_root}/train/meta_info_tree_GT_train.json'),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend, key='lq', flag='unchanged', **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend, key='gt', flag='unchanged', **load_kwargs),
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
    dict(type='GenerateFileIndices',
         interval_list=[1],
         annotation_tree_json=f'{data_root}/test/meta_info_tree_GT_test.json'),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend, key='lq', flag='unchanged', **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend, key='gt', flag='unchanged', **load_kwargs),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq'], **img_norm_cfg_lq),
    dict(type='Normalize', keys=['gt'], **img_norm_cfg_gt),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'dataset', 'folder', 'haze_beta', 'haze_light']),
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=0,
        persistent_workers=False, pin_memory=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=0,
        persistent_workers=False, pin_memory=False),
    train=dict(
        type='RepeatDataset', times=10000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=f'{data_root}/train/hazy',
            gt_folder=f'{data_root}/train/gt',
            ann_file=f'{data_root}/train/meta_info_GT_train.txt',
            num_input_frames=num_input_frames,
            pipeline=train_pipeline,
            test_mode=False)),
    val=dict(
        type=test_dataset_type,
        lq_folder=f'{data_root}/test/hazy',
        gt_folder=f'{data_root}/test/gt',
        ann_file=f'{data_root}/test/meta_info_GT_test.txt',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=test_dataset_type,
        lq_folder=f'{data_root}/test/hazy',
        gt_folder=f'{data_root}/test/gt',
        ann_file=f'{data_root}/test/meta_info_GT_test.txt',
        pipeline=test_pipeline,
        test_mode=True),
)
