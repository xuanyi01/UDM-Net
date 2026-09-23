import argparse
from pathlib import Path

import cv2
import numpy as np


def list_frames(frame_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in exts])


def normalize_depth(depths: np.ndarray, mode: str) -> np.ndarray:
    depths = depths.astype(np.float32)
    if mode == "sequence":
        lo, hi = np.percentile(depths, [1, 99])
        return np.clip((depths - lo) / max(hi - lo, 1e-6), 0, 1)
    if mode == "frame":
        out = np.empty_like(depths, dtype=np.float32)
        for i in range(depths.shape[0]):
            lo, hi = np.percentile(depths[i], [1, 99])
            out[i] = np.clip((depths[i] - lo) / max(hi - lo, 1e-6), 0, 1)
        return out
    raise ValueError(f"Unknown normalize mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--normalize", default="sequence", choices=["sequence", "frame"])
    parser.add_argument("--inverse", action="store_true", help="Invert normalized depth if near/far direction looks wrong.")
    args = parser.parse_args()

    data = np.load(args.npz)
    depths = data["depths"]
    frames = list_frames(args.frame_dir)
    if len(frames) != len(depths):
        raise RuntimeError(f"Frame/depth count mismatch: {len(frames)} frames vs {len(depths)} depths")

    norm = normalize_depth(depths, args.normalize)
    if args.inverse:
        norm = 1.0 - norm

    depth16_dir = args.out_dir / "depth16"
    preview_dir = args.out_dir / "preview8"
    depth16_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for frame_path, depth in zip(frames, norm):
        depth16 = np.round(depth * 65535.0).astype(np.uint16)
        depth8 = np.round(depth * 255.0).astype(np.uint8)
        cv2.imwrite(str(depth16_dir / f"{frame_path.stem}.png"), depth16)
        cv2.imwrite(str(preview_dir / f"{frame_path.stem}.png"), depth8)

    print(f"Exported {len(frames)} depth frames")
    print(f"16-bit depth: {depth16_dir}")
    print(f"8-bit preview: {preview_dir}")


if __name__ == "__main__":
    main()
