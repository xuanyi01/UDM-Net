#!/bin/bash
# Evaluate UDM-Net on HazeUAVvideo test set (full-clip)
set -euo pipefail

CONFIG="configs/udmnet/udmnet_hazeuavvideo_fullclip.py"
CHECKPOINT=${1:?"Usage: bash scripts/eval_hazeuavvideo.sh <checkpoint.pth> [--save-path ./results]"}
shift
export HAZEUAVVIDEO_ROOT="${HAZEUAVVIDEO_ROOT:-/mnt/d/HazeUAVvideo5}"
python tools/verify_hazeuavvideo.py --root "${HAZEUAVVIDEO_ROOT}"
python tools/verify_runtime.py

python tools/test.py "${CONFIG}" "${CHECKPOINT}" \
    --cfg-options \
        data.test_dataloader.workers_per_gpu=0 \
        data.test_dataloader.persistent_workers=False \
        data.test_dataloader.pin_memory=False \
    "$@"
