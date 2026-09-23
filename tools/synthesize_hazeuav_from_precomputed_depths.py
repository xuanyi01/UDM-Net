"""Build a HazeUAVvideo-style dataset from precomputed VDA depth maps.

Expected source layout:
  <src_root>/<split>/gt/<dataset>/<sequence>/<frame>

Expected depth layout:
  <depth_root>/<split>/<dataset>/<sequence>/depth_export/depth16/<frame_stem>.png
or:
  <depth_root>/<split>/<dataset>/<sequence>/<frame_stem>.png

Output layout:
  <out_root>/<split>/{gt,hazy,transmission,depth}/<dataset>/<sequence>_<A>_<beta>
  meta_info_GT_*.txt and meta_info_tree_GT_*.json
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def beta_token(beta: float) -> str:
    return f"{float(beta):g}"


def airlight_token(airlight: float) -> int:
    return int(round(float(airlight) * 255.0))


def sequence_rng(seed, split, dataset, sequence):
    key = f"{seed}:{split}:{dataset}:{sequence}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(value)


def read_depth(depth_root: Path, split: str, dataset: str, sequence: str,
               frame: Path, invert: bool, size_hw):
    candidates = [
        depth_root / split / dataset / sequence / "depth_export" / "depth16" / f"{frame.stem}.png",
        depth_root / split / dataset / sequence / f"{frame.stem}.png",
    ]
    depth_path = next((p for p in candidates if p.exists()), None)
    if depth_path is None:
        raise FileNotFoundError(f"No depth for {split}/{dataset}/{sequence}/{frame.name}")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.float32)
    max_value = 65535.0 if depth.max() > 255.0 else 255.0
    depth = np.clip(depth / max_value, 0.0, 1.0)
    if invert:
        depth = 1.0 - depth
    h, w = size_hw
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def odd_kernel(value, min_value=3):
    k = int(value)
    if k < min_value:
        k = min_value
    if k % 2 == 0:
        k += 1
    return k


def make_effective_depth(depth, args):
    depth = depth.astype(np.float32)

    if args.depth_blur and args.depth_blur > 1:
        k = odd_kernel(args.depth_blur)
        depth = cv2.GaussianBlur(depth, (k, k), 0)

    if args.depth_background_fill and args.depth_background_fill > 0:
        k = odd_kernel(args.depth_background_fill_kernel)
        kernel = np.ones((k, k), np.uint8)
        background_depth = cv2.dilate(depth, kernel)
        background_depth = cv2.GaussianBlur(background_depth, (k, k), 0)
        fill = np.clip(float(args.depth_background_fill), 0.0, 1.0)
        depth = (1.0 - fill) * depth + fill * background_depth

    if args.depth_gamma and abs(args.depth_gamma - 1.0) > 1e-6:
        depth = np.power(np.clip(depth, 0.0, 1.0), float(args.depth_gamma))

    if args.depth_floor is not None:
        depth = np.maximum(depth, float(args.depth_floor))

    if args.homogeneous_mix and args.homogeneous_mix > 0:
        mix = float(args.homogeneous_mix)
        depth = (1.0 - mix) * depth + mix

    return np.clip(depth, 0.0, 1.0)


def synthesize(gt_bgr, depth01, beta, airlight, args):
    gt = gt_bgr.astype(np.float32) / 255.0
    depth = make_effective_depth(depth01, args)
    depth_scaled = depth * float(args.depth_scale)
    if args.depth_max and args.depth_max > 0:
        depth_scaled = np.clip(depth_scaled, 0.0, float(args.depth_max))
    transmission = np.exp(-float(beta) * depth_scaled)
    transmission = np.clip(transmission, float(args.transmission_min), 1.0)
    if args.transmission_blur and args.transmission_blur > 1:
        k = odd_kernel(args.transmission_blur)
        transmission = cv2.GaussianBlur(transmission, (k, k), 0)
        transmission = np.clip(transmission, 0.0, 1.0)
    hazy = gt * transmission[..., None] + float(airlight) * (1.0 - transmission[..., None])
    hazy = np.clip(hazy * 255.0 + 0.5, 0, 255).astype(np.uint8)
    trans_u8 = np.clip(transmission * 255.0 + 0.5, 0, 255).astype(np.uint8)
    depth_u8 = np.clip(depth * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return hazy, trans_u8, depth_u8


def clean_split(out_root: Path, split: str):
    split_root = out_root / split
    for sub in ["gt", "hazy", "transmission", "depth"]:
        path = split_root / sub
        if path.exists():
            shutil.rmtree(path)
    for p in split_root.glob("meta_info_GT*.txt"):
        p.unlink()
    tree = split_root / f"meta_info_tree_GT_{split}.json"
    if tree.exists():
        tree.unlink()


def process_split(args, split: str):
    src_gt_root = args.src_root / split / "gt"
    out_split = args.out_root / split
    if args.clean_outputs:
        clean_split(args.out_root, split)

    meta_lines = []
    subset_lines = {}
    tree = {}

    dataset_dirs = sorted([p for p in src_gt_root.iterdir() if p.is_dir()])
    for dataset_dir in tqdm(dataset_dirs, desc=f"{split}: datasets"):
        dataset = dataset_dir.name
        subset_lines.setdefault(dataset, [])
        for seq_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            sequence = seq_dir.name
            frames = list_images(seq_dir)
            if not frames:
                continue

            rng = sequence_rng(args.seed, split, dataset, sequence)
            if args.airlight is not None:
                airlight = float(args.airlight)
            else:
                airlight = float(rng.uniform(args.airlight_min, args.airlight_max))
            a_token = airlight_token(airlight)

            for beta in args.betas:
                b_token = beta_token(beta)
                clip_name = f"{sequence}_{a_token}_{b_token}"
                clip_key = f"{dataset}/{clip_name}"
                gt_key = f"{dataset}/{sequence}"

                hazy_dir = out_split / "hazy" / dataset / clip_name
                trans_dir = out_split / "transmission" / dataset / clip_name
                depth_dir = out_split / "depth" / dataset / clip_name
                gt_dir = out_split / "gt" / dataset / sequence
                ensure_dir(hazy_dir)
                ensure_dir(trans_dir)
                ensure_dir(depth_dir)
                ensure_dir(gt_dir)

                valid_names = []
                for frame in frames:
                    img = cv2.imread(str(frame), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    depth = read_depth(
                        args.depth_root, split, dataset, sequence, frame,
                        args.invert_depth, img.shape[:2])
                    hazy, trans_u8, depth_u8 = synthesize(img, depth, beta, airlight, args)
                    cv2.imwrite(str(hazy_dir / frame.name), hazy, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    cv2.imwrite(str(trans_dir / frame.name), trans_u8)
                    cv2.imwrite(str(depth_dir / frame.name), depth_u8)
                    if not (gt_dir / frame.name).exists():
                        shutil.copy2(frame, gt_dir / frame.name)
                    valid_names.append(frame.name)

                if valid_names:
                    line = f"{clip_key} {len(valid_names)}"
                    meta_lines.append(line)
                    subset_lines[dataset].append(line)
                    tree[clip_key] = dict(
                        dataset=dataset,
                        gt_key=gt_key,
                        frames=valid_names,
                        haze_beta=float(beta),
                        haze_light=float(airlight),
                        airlight_token=int(a_token),
                    )

    ensure_dir(out_split)
    (out_split / f"meta_info_GT_{split}.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    for dataset, lines in subset_lines.items():
        (out_split / f"meta_info_GT_{dataset}_{split}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    (out_split / f"meta_info_tree_GT_{split}.json").write_text(
        json.dumps(tree, indent=2), encoding="utf-8")
    print(f"[{split}] clips={len(meta_lines)} written to {out_split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--airlight", type=float, default=None)
    parser.add_argument("--airlight-min", type=float, default=0.78)
    parser.add_argument("--airlight-max", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--invert-depth", action="store_true")
    parser.add_argument("--depth-scale", type=float, default=115.0)
    parser.add_argument("--depth-max", type=float, default=0.0)
    parser.add_argument("--transmission-min", type=float, default=0.10)
    parser.add_argument("--depth-blur", type=int, default=41)
    parser.add_argument("--transmission-blur", type=int, default=41)
    parser.add_argument("--depth-floor", type=float, default=0.20)
    parser.add_argument("--depth-gamma", type=float, default=0.9)
    parser.add_argument("--homogeneous-mix", type=float, default=0.06)
    parser.add_argument("--depth-background-fill", type=float, default=0.12)
    parser.add_argument("--depth-background-fill-kernel", type=int, default=51)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--clean-outputs", action="store_true")
    args = parser.parse_args()

    for split in args.splits:
        process_split(args, split)


if __name__ == "__main__":
    main()
