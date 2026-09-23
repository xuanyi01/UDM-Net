#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/run_video_depth_anything_sequence.sh /path/to/frame_sequence /path/to/output_dir [encoder]"
  exit 1
fi

SEQ_DIR="$1"
OUT_DIR="$2"
ENCODER="${3:-vits}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${VDA_ROOT:-${ROOT_DIR}/external_baselines/Video-Depth-Anything}"
ENV_NAME="${VDA_ENV_NAME:-vdepth}"

SEQ_DIR="$(realpath "${SEQ_DIR}")"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(realpath "${OUT_DIR}")"

if [ ! -d "${REPO_DIR}" ]; then
  echo "ERROR: Video-Depth-Anything repo not found. Run: bash setup_video_depth_anything.sh"
  exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

TMP_VIDEO="${OUT_DIR}/input.mp4"

python - "$SEQ_DIR" "$TMP_VIDEO" <<'PY'
import sys
from pathlib import Path
import cv2

seq = Path(sys.argv[1])
out = Path(sys.argv[2])
frames = sorted([p for p in seq.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
if not frames:
    raise SystemExit(f"No image frames found in {seq}")

first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
if first is None:
    raise SystemExit(f"Cannot read first frame: {frames[0]}")
h, w = first.shape[:2]
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
for p in frames:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read frame: {p}")
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    writer.write(img)
writer.release()
print(f"Wrote {len(frames)} frames to {out}")
PY

(
  cd "${REPO_DIR}"
  python run.py \
    --input_video "${TMP_VIDEO}" \
    --output_dir "${OUT_DIR}/vda_raw" \
    --encoder "${ENCODER}" \
    --max_res 1280 \
    --save_npz \
    --grayscale
)

NPZ="${OUT_DIR}/vda_raw/input_depths.npz"
python "${ROOT_DIR}/tools/export_vda_npz_to_depth_png.py" \
  --npz "${NPZ}" \
  --frame-dir "${SEQ_DIR}" \
  --out-dir "${OUT_DIR}/depth_export" \
  --normalize sequence

echo "Done."
echo "Raw VDA outputs: ${OUT_DIR}/vda_raw"
echo "Depth PNGs:       ${OUT_DIR}/depth_export/depth16"
echo "Preview PNGs:     ${OUT_DIR}/depth_export/preview8"
