# UDM-Net evaluation on HazeUAVvideo with full-clip inference
# No training pipeline — test only with full temporal length

import os
import os.path as osp

_base_ = './udmnet_hazeuavvideo.py'

# Override test pipeline to use full clips
fullclip_chunk = 60
fullclip_num_input_frames = None

data = dict(
    val=dict(
        _delete_=True,
        type='HWFolderMultipleGTDataset',
        lq_folder=f'{data_root}/test/hazy',
        gt_folder=f'{data_root}/test/gt',
        trans_folder=f'{data_root}/test/transmission',
        ann_file=f'{data_root}/test/meta_info_GT_test.txt',
        num_input_frames=fullclip_num_input_frames,
        chunk_size=fullclip_chunk,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        _delete_=True,
        type='HWFolderMultipleGTDataset',
        lq_folder=f'{data_root}/test/hazy',
        gt_folder=f'{data_root}/test/gt',
        trans_folder=f'{data_root}/test/transmission',
        ann_file=f'{data_root}/test/meta_info_GT_test.txt',
        num_input_frames=fullclip_num_input_frames,
        chunk_size=fullclip_chunk,
        pipeline=test_pipeline,
        test_mode=True),
)
