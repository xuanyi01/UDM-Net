#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${SRC_ROOT:?Set SRC_ROOT to the official clean train/gt and test/gt split}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to a new output directory}"
DEPTH_ROOT="${DEPTH_ROOT:-${OUT_ROOT}.vda_depth_cache}"
VDA_ROOT="${VDA_ROOT:-${ROOT_DIR}/external_baselines/Video-Depth-Anything}"
VDA_ENCODER="${VDA_ENCODER:-vits}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-123}"

if [[ ! -d "${SRC_ROOT}/train/gt" || ! -d "${SRC_ROOT}/test/gt" ]]; then
    echo 'SRC_ROOT must contain train/gt and test/gt' >&2
    exit 1
fi
if [[ -e "${OUT_ROOT}" ]]; then
    echo 'OUT_ROOT must be a new directory; existing data will not be overwritten' >&2
    exit 1
fi
if [[ ! -f "${VDA_ROOT}/run.py" || ! -f "${VDA_ROOT}/checkpoints/video_depth_anything_${VDA_ENCODER}.pth" ]]; then
    echo 'Prepare the Video Depth Anything code and checkpoint first' >&2
    exit 1
fi
python "${ROOT_DIR}/tools/verify_hazeuavvideo.py" \
    --root "${SRC_ROOT}" --clean-only

mkdir -p "${DEPTH_ROOT}"
for SPLIT in train test; do
    for DATASET_DIR in "${SRC_ROOT}/${SPLIT}/gt"/*; do
        [[ -d "${DATASET_DIR}" ]] || continue
        DATASET="$(basename "${DATASET_DIR}")"
        for SEQ_DIR in "${DATASET_DIR}"/*; do
            [[ -d "${SEQ_DIR}" ]] || continue
            SEQ="$(basename "${SEQ_DIR}")"
            DEPTH_SEQ_DIR="${DEPTH_ROOT}/${SPLIT}/${DATASET}/${SEQ}"
            if [[ -d "${DEPTH_SEQ_DIR}/depth_export/depth16" ]] &&
               [[ "$(find "${DEPTH_SEQ_DIR}/depth_export/depth16" -type f | wc -l)" -gt 0 ]]; then
                echo "[skip depth] ${SPLIT}/${DATASET}/${SEQ}"
            else
                CUDA_VISIBLE_DEVICES="${GPU_ID}" VDA_ROOT="${VDA_ROOT}" \
                    bash "${ROOT_DIR}/scripts/run_video_depth_anything_sequence.sh" \
                    "${SEQ_DIR}" "${DEPTH_SEQ_DIR}" "${VDA_ENCODER}"
            fi
        done
    done
done

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${UDM_ENV_NAME:-udmnet}"
python "${ROOT_DIR}/tools/synthesize_hazeuav_from_precomputed_depths.py" \
    --src-root "${SRC_ROOT}" \
    --depth-root "${DEPTH_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --splits train test \
    --betas 0.005 0.01 0.02 \
    --airlight-min 0.78 \
    --airlight-max 0.95 \
    --seed "${SEED}" \
    --invert-depth \
    --depth-scale 115 \
    --depth-max 0 \
    --transmission-min 0.10 \
    --depth-blur 41 \
    --transmission-blur 41 \
    --depth-floor 0.20 \
    --depth-gamma 0.9 \
    --homogeneous-mix 0.06 \
    --depth-background-fill 0.12 \
    --depth-background-fill-kernel 51 \
    --jpeg-quality 95

python "${ROOT_DIR}/tools/verify_hazeuavvideo.py" \
    --root "${OUT_ROOT}" --structure-only
