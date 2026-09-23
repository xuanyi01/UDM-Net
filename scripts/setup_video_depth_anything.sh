#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${ROOT_DIR}/external_baselines"
REPO_DIR="${BASE_DIR}/Video-Depth-Anything"
ENV_NAME="${VDA_ENV_NAME:-vdepth}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${BASE_DIR}"

if [ ! -f "${REPO_DIR}/run.py" ]; then
  git clone https://github.com/DepthAnything/Video-Depth-Anything "${REPO_DIR}"
fi

source ~/anaconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

cd "${REPO_DIR}"

# Install PyTorch first so xformers/requirements resolve against a CUDA wheel.
python -m pip install torch==2.1.1 torchvision==0.16.1 --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements.txt

mkdir -p checkpoints
if [ ! -f checkpoints/video_depth_anything_vits.pth ]; then
  wget -O checkpoints/video_depth_anything_vits.pth \
    https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
fi

echo "Done."
echo "Repo: ${REPO_DIR}"
echo "Env:  ${ENV_NAME}"
echo "Small checkpoint: ${REPO_DIR}/checkpoints/video_depth_anything_vits.pth"
