#!/bin/bash
# Fine-tune UDM-Net on HazeUAVvideo dataset (single GPU)
set -euo pipefail

CONFIG="configs/udmnet/udmnet_hazeuavvideo.py"
WORK_DIR="./work_dirs/udmnet_hazeuavvideo"
export HAZEUAVVIDEO_ROOT="${HAZEUAVVIDEO_ROOT:-/mnt/d/HazeUAVvideo5}"
python tools/verify_hazeuavvideo.py --root "${HAZEUAVVIDEO_ROOT}"
python tools/verify_runtime.py

CONFIG_ARGS=()
if [[ -n "${LOAD_FROM:-}" ]]; then
    CONFIG_ARGS=(--cfg-options "load_from=${LOAD_FROM}")
fi

python tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    "${CONFIG_ARGS[@]}"
