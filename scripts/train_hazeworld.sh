#!/bin/bash
# Train UDM-Net on HazeWorld dataset (single GPU)
set -e

CONFIG="configs/udmnet/udmnet_base.py"
WORK_DIR="./work_dirs/udmnet_hazeworld"

python tools/train.py ${CONFIG} \
    --work-dir ${WORK_DIR}
