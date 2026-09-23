# UDM-Net: UAV Video Dehazing

## Overview

UDM-Net is a U-Net-like encoder and dual-path decoder for UAV video dehazing, featuring:

- **Multi-Scale Feature Encoder**: ConvNeXt-Tiny backbone extracting four feature levels
- **Degradation-Aware Prior Decoding Path (DAPD)**: Estimates transmission and atmospheric light, constructs prior tokens, and retrieves prior memory
- **Reliable Temporal Scene Recovery Path (RTSR)**: Uses Mamba enhancement, the Temporal Alignment Block (TAB), and Guided Multi-Range Aggregation and Adaptive Gating Fusion (GAAF)
- **Global Mamba Refinement and Reconstruction (GMRR)**: Refines the final scene feature with MambaBlock2D, then reconstructs a residual image

The implementation uses the paper's DAPD, RTSR, TAB, GAAF, and GMRR names.
Existing configuration keys and checkpoint parameter names remain unchanged.

## Paper protocol and release status

The paper's HazeUAVvideo benchmark has 164 clean sequences from UAV123,
UAVDT, and VisDrone. A source-sequence-level 125/39 train/test split and three
haze densities (`beta=0.005,0.01,0.02`) give 375/117 hazy clips, 26,096 clean
frames, and 78,288 hazy frames.

The HazeUAVvideo training config uses 60K iterations, AdamW, cosine learning
rate `1e-4` to `1e-6`, batch size 8, and `256x256` patches. The `splits/`
directory contains metadata copied from the paper's three-density dataset.
It does not contain the source frames, synthesized images, or a trained UDM-Net
checkpoint. Runtime operators and the checkpoint must be verified before
claiming that a fresh run reproduces the paper's reported quality numbers.

## Installation

```bash
conda create -n udmnet python=3.10
conda activate udmnet
cd /path/to/UDM-Net
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/verify_runtime.py
```

## Project Structure

```
UDM-Net/
├── configs/
│   ├── _base_/                  # Base configs (dataset, runtime)
│   └── udmnet/
│       ├── udmnet_base.py                    # Full model on HazeWorld
│       ├── udmnet_hazeuavvideo.py            # Fine-tune on HazeUAVvideo
│       └── udmnet_hazeuavvideo_fullclip.py   # Full-clip evaluation
├── mmedit/
│   ├── apis/              # Train/test APIs
│   ├── core/              # Evaluation metrics, hooks
│   ├── datasets/          # Dataset & data pipeline
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── udm_backbones/
│   │   │   │   ├── udmnet_net.py   # Main network architecture
│   │   │   │   ├── udm_stda.py     # STDA block used by TAB
│   │   │   │   ├── udm_modules.py  # Residual blocks, basic modules
│   │   │   │   ├── udm_utils.py    # Utility functions
│   │   │   │   └── convnext.py     # ConvNeXt backbone
│   │   │   ├── mamba_backbone.py   # MambaBlock2D
│   │   │   └── vmamba_backbone.py  # VMamba backbone (optional)
│   │   ├── dehazers/
│   │   │   ├── basic_dehazer.py    # Base dehazer with eval logic
│   │   │   └── udm.py             # UDM-Net training/testing wrapper
│   │   ├── losses/                 # L1, perceptual, gradient losses
│   │   └── common/                 # Shared modules
│   └── utils/
├── tools/
│   ├── train.py           # Training entry point
│   ├── test.py            # Testing entry point
│   ├── dist_train.sh      # Distributed training
│   ├── dist_test.sh       # Distributed testing
│   ├── extract_ema.py     # Extract EMA weights from checkpoint
│   ├── get_flops.py       # Compute FLOPs
│   ├── verify_hazeuavvideo.py   # Paper split and paired-frame audit
│   ├── verify_runtime.py        # Mamba and DCN CUDA smoke test
│   ├── export_vda_npz_to_depth_png.py
│   ├── synthesize_hazeuav_from_precomputed_depths.py
├── scripts/                    # Train, evaluate, and synthesize entry points
├── splits/                     # Three-density benchmark metadata
├── setup.py
├── requirements.txt
└── README.md
```

## Data and synthesis

The HazeUAVvideo dataset and pretrained UDM-Net checkpoint are available
separately through [Baidu Netdisk](https://pan.baidu.com/s/19uGj1AL_ijJcI7EYqD7PkA)
(extraction code: `865B`).

Set `HAZEUAVVIDEO_ROOT` to the three-density dataset. The default in the
scripts is `/mnt/d/HazeUAVvideo5`.

```bash
export HAZEUAVVIDEO_ROOT=/path/to/HazeUAVvideo5
python tools/verify_hazeuavvideo.py --root "$HAZEUAVVIDEO_ROOT"
```

To generate hazy sequences, `SRC_ROOT` must contain the clean `train/gt` and
`test/gt` source-sequence split. Video Depth Anything ViT-S estimates depth
for each video; the exported maps are normalized per sequence before haze
synthesis. The Video Depth Anything code and checkpoint are prepared
separately. `OUT_ROOT` must be a new path; the script will not overwrite data.

```bash
bash scripts/setup_video_depth_anything.sh
export SRC_ROOT=/path/to/clean_split
export OUT_ROOT=/path/to/new/HazeUAVvideo5
bash scripts/generate_hazeuavvideo.sh
```

The generator requires the source frames and the Video Depth Anything ViT-S
checkpoint. Model inference and synthesis use separate Conda environments.

## Training

```bash
export HAZEUAVVIDEO_ROOT=/path/to/HazeUAVvideo5
bash scripts/train_hazeuavvideo.sh
```

Set `LOAD_FROM=/path/to/pretrained.pth` to initialize from a checkpoint.
Without it, training starts without that initialization. The script verifies
the dataset split before starting. HazeWorld scripts remain available in
`scripts/`.

## Testing

```bash
export HAZEUAVVIDEO_ROOT=/path/to/HazeUAVvideo5
bash scripts/eval_hazeuavvideo.sh /path/to/udmnet_checkpoint.pth
```

The full-clip config uses the paper's test metadata. To save dehazed images,
pass `--save-path ./results/` after the checkpoint path.

## Extract EMA Weights

```bash
python tools/extract_ema.py /path/to/checkpoint.pth /path/to/output_ema.pth
```

## Licensing

HazeUAVvideo is licensed under [CC BY-NC 4.0](DATA_LICENSE.md).
