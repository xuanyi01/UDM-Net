#!/bin/bash
# Evaluate UDM-Net on HazeWorld test set
set -e

CONFIG="configs/udmnet/udmnet_base.py"
CHECKPOINT=${1:?"Usage: bash scripts/eval_hazeworld.sh <checkpoint.pth> [--save-path ./results]"}
shift

python tools/test.py ${CONFIG} ${CHECKPOINT} \
    --cfg-options \
        data.test_dataloader.workers_per_gpu=0 \
        data.test_dataloader.persistent_workers=False \
        data.test_dataloader.pin_memory=False \
    "$@"
