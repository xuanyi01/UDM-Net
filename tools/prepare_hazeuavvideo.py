#!/usr/bin/env python3
r"""
Legacy exploratory UAV haze generator. This is not the HazeUAVvideo protocol
used for the paper's three-density, 125/39 source-sequence split. Use
scripts/generate_hazeuavvideo.sh for that protocol.

用法:
    python tools/prepare_hazeuavvideo.py \
        --source_dir D:\sjj \
        --visdrone_dir /home/ubuntu/data/HazeWorld/gt/VisDrone \
        --output_dir D:\HazeUAVvideo \
        --beta_values 0.005 0.01 0.02 0.03 \
        --train_ratio 0.9 \
        --legacy-exploratory
"""

import argparse
import json
import logging
import os
import random
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def extract_archives(source_dir, skip=False):
    """解压所有tar.gz和zip文件"""
    if skip:
        print("⚠️  跳过解压步骤（按用户要求）")
        return
    source_path = Path(source_dir)
    
    for dataset_name in ['UAV123', 'UAVDT', 'AU-AIR']:
        dataset_dir = source_path / dataset_name / f'OpenDataLab___{dataset_name}' / 'raw'
        
        if not dataset_dir.exists():
            print(f"⚠️  {dataset_name} raw目录不存在: {dataset_dir}")
            continue
        
        print(f"\n📦 解压 {dataset_name}...")
        
        # 查找tar.gz和zip文件
        for archive_file in dataset_dir.glob('*'):
            if archive_file.suffix in ['.gz', '.zip']:
                print(f"   解压: {archive_file.name}")
                
                try:
                    if archive_file.suffix == '.gz':
                        with tarfile.open(archive_file, 'r:gz') as tar:
                            tar.extractall(path=dataset_dir)
                    elif archive_file.suffix == '.zip':
                        with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                            zip_ref.extractall(path=dataset_dir)
                    print(f"   ✓ 解压完成")
                except Exception as e:
                    print(f"   ✗ 解压失败: {e}")


def compute_sky_mask(img):
    """检测天空区域（无人机场景优化）"""
    h, w = img.shape[:2]
    
    # HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 天空通常在图像上半部分
    upper_region = hsv[:int(h*0.6), :, :]
    
    # 天空颜色范围（蓝色/白色）
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(upper_region, lower_blue, upper_blue)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(upper_region, lower_white, upper_white)
    
    sky_mask_upper = cv2.bitwise_or(blue_mask, white_mask)
    
    # 形态学操作
    kernel = np.ones((15, 15), np.uint8)
    sky_mask_upper = cv2.morphologyEx(sky_mask_upper, cv2.MORPH_CLOSE, kernel)
    sky_mask_upper = cv2.morphologyEx(sky_mask_upper, cv2.MORPH_OPEN, kernel)
    
    # 扩展到全图
    sky_mask = np.zeros((h, w), dtype=np.uint8)
    sky_mask[:int(h*0.6), :] = sky_mask_upper
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, 
                                np.ones((31, 31), np.uint8))
    
    return sky_mask > 0


def compute_drone_depth_map(img):
    """针对无人机场景优化的深度估算"""
    h, w = img.shape[:2]
    
    # Step 1: 检测天空
    sky_mask = compute_sky_mask(img)
    
    # Step 2: 暗通道深度估算
    dark_channel = img.min(axis=2)
    kernel = np.ones((15, 15), np.uint8)
    dark_channel = cv2.erode(dark_channel, kernel)
    depth_base = dark_channel.astype(np.float32) / 255.0
    
    # Step 3: 纵向梯度调整（无人机视角：越往上越远）
    y_coords = np.linspace(0, 1, h)[:, np.newaxis]
    y_factor = np.tile(y_coords, (1, w))
    depth_altitude = y_factor * 0.3
    
    # Step 4: 合并
    depth = depth_base + depth_altitude
    
    # Step 5: 天空区域强制为最大深度
    depth[sky_mask] = 1.0
    
    # Step 6: 平滑处理
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    depth = np.clip(depth, 0.0, 1.0)
    
    return depth, sky_mask


def compute_atmospheric_light(img, sky_mask=None, percentile=0.99):
    """计算大气光，优先从天空区域采样"""
    if sky_mask is not None and sky_mask.sum() > 0:
        sky_pixels = img[sky_mask]
        if len(sky_pixels) > 100:
            A = sky_pixels.mean(axis=0) / 255.0
            A = np.clip(A + 0.1, 0.7, 1.0)
            return A
    
    # 回退到标准方法
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(gray, percentile * 100)
    bright_pixels = img[gray >= threshold]
    
    if len(bright_pixels) == 0:
        return np.array([0.95, 0.95, 0.95])
    
    A = bright_pixels.mean(axis=0) / 255.0
    return np.clip(A + 0.1, 0.7, 1.0)


def resize_short_side(img, min_size=720):
    h, w = img.shape[:2]
    if min(h, w) >= min_size:
        return img
    if h < w:
        new_w = int(w / h * min_size)
        new_h = min_size
    else:
        new_w = min_size
        new_h = int(h / w * min_size)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def synthesize_haze_nonuniform(img,
                               beta,
                               A=None,
                               depth=None,
                               depth_scale=25.0,
                               transmission_min=0.001):
    """合成非均匀雾霾（支持无人机场景）
    
    使用大气散射模型: I(x) = J(x) * t(x) + A * (1 - t(x))
    其中 t(x) = exp(-beta * d(x))
    """
    h, w = img.shape[:2]
    J = img.astype(np.float32) / 255.0
    
    # 计算深度和天空掩码
    if depth is None:
        depth, sky_mask = compute_drone_depth_map(img)
    else:
        sky_mask = compute_sky_mask(img)
    
    # 放大深度范围，使传输图不至于接近 1
    depth = np.clip(depth * depth_scale, 0.0, 50.0)

    # 计算大气光
    if A is None:
        A = compute_atmospheric_light(img, sky_mask)
    
    # 计算透射率: t(x) = exp(-beta * d(x))
    transmission = np.exp(-beta * depth)
    transmission = np.clip(transmission, transmission_min, 1.0)
    
    # 合成雾霾: I = J*t + A*(1-t)
    t_expanded = transmission[:, :, np.newaxis]
    A_expanded = A[np.newaxis, np.newaxis, :]
    I = J * t_expanded + A_expanded * (1.0 - t_expanded)
    I = np.clip(I, 0.0, 1.0)
    
    hazy_img = (I * 255).astype(np.uint8)
    
    return hazy_img, transmission, A


def find_image_files(directory):
    """递归查找所有图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    return sorted(image_files)


def process_dataset(dataset_path,
                    dataset_name,
                    output_base,
                    beta_values,
                    train_ratio=0.9,
                    copy_gt=False,
                    depth_scale=25.0,
                    transmission_min=0.001):
    """处理单个数据集
    
    参数:
        dataset_path: 数据集根目录
        dataset_name: 数据集名称 (VisDrone, UAV123, UAVDT, AU-AIR)
        output_base: 输出根目录
        beta_values: 雾浓度系数列表
        train_ratio: 训练集比例
    """
    print(f"\n{'='*60}")
    print(f"处理 {dataset_name} 数据集")
    print(f"{'='*60}")
    
    # 查找所有图像文件
    image_files = find_image_files(dataset_path)
    
    if len(image_files) == 0:
        print(f"⚠️  未找到图像文件: {dataset_path}")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 按序列组织图像
    sequences = defaultdict(list)
    for img_file in image_files:
        # 提取序列名（通常是父目录名或前缀）
        seq_name = img_file.parent.name
        sequences[seq_name].append(img_file)
    
    print(f"组织为 {len(sequences)} 个序列")
    
    # 分割train/test
    seq_list = list(sequences.keys())
    random.shuffle(seq_list)
    split_idx = int(len(seq_list) * train_ratio)
    train_seqs = set(seq_list[:split_idx])
    
    # 创建输出目录
    output_path = Path(output_base)
    
    all_meta = []
    
    # 处理每个序列
    for seq_name in tqdm(seq_list, desc=f"处理 {dataset_name} 序列"):
        img_files = sequences[seq_name]
        split = 'train' if seq_name in train_seqs else 'test'
        
        # 创建输出目录
        gt_dir = output_path / split / 'gt' / dataset_name / seq_name
        trans_dir = output_path / split / 'transmission' / dataset_name / seq_name
        gt_dir.mkdir(parents=True, exist_ok=True)
        trans_dir.mkdir(parents=True, exist_ok=True)
        
        hazy_dirs = {}
        for beta in beta_values:
            hazy_dir = output_path / split / 'hazy' / dataset_name / f'{seq_name}_beta_{beta}'
            hazy_dir.mkdir(parents=True, exist_ok=True)
            hazy_dirs[beta] = hazy_dir
        
        # 读取第一帧用于计算大气光
        first_img = cv2.imread(str(img_files[0]))
        first_img = resize_short_side(first_img)
        if first_img is None:
            continue
        
        _, sky_mask = compute_drone_depth_map(first_img)
        A = compute_atmospheric_light(first_img, sky_mask)
        
        # 处理每一帧
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img = resize_short_side(img)
            
            frame_name = img_file.name
            
            # 保存GT（创建符号链接或复制）
            gt_path = gt_dir / frame_name
            if not gt_path.exists():
                if copy_gt:
                    cv2.imwrite(str(gt_path), img)
                else:
                    try:
                        os.symlink(img_file.absolute(), gt_path)
                    except:
                        cv2.imwrite(str(gt_path), img)
            
            # 估算深度
            depth, _ = compute_drone_depth_map(img)
            
            # 保存参考透射率
            depth_scaled = np.clip(depth * depth_scale, 0.0, 50.0)
            trans_ref = np.exp(-beta_values[0] * depth_scaled)
            trans_ref = np.clip(trans_ref, transmission_min, 1.0)
            trans_path = trans_dir / frame_name.replace(frame_name.split('.')[-1], 'png')
            cv2.imwrite(str(trans_path), (trans_ref * 255).astype(np.uint8))
            
            # 合成雾霾
            for beta in beta_values:
                hazy_img, _, _ = synthesize_haze_nonuniform(
                    img,
                    beta,
                    A=A,
                    depth=depth,
                    depth_scale=depth_scale,
                    transmission_min=transmission_min)
                hazy_path = hazy_dirs[beta] / frame_name
                cv2.imwrite(str(hazy_path), hazy_img)
            
            # 记录元信息
            all_meta.append({
                'folder': seq_name,
                'frame': frame_name,
                'dataset': dataset_name,
                'split': split,
                'beta_values': beta_values,
                'A': A.tolist()
            })
    
    # 生成meta文件
    generate_meta_files(all_meta, output_base, dataset_name)
    
    print(f"✅ {dataset_name} 处理完成: {len(all_meta)} 帧")


def generate_meta_files(all_meta, output_base, dataset_name):
    """生成训练所需的meta文件"""
    output_path = Path(output_base)
    
    for split in ['train', 'test']:
        split_meta = [m for m in all_meta if m['split'] == split]
        if not split_meta:
            continue
        
        meta_dir = output_path / split
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成meta_info_GT_{DatasetName}.txt
        gt_file = meta_dir / f'meta_info_GT_{dataset_name}.txt'
        with open(gt_file, 'w') as f:
            seen_folders = set()
            for item in split_meta:
                folder = item['folder']
                if folder not in seen_folders:
                    frames = [m for m in split_meta if m['folder'] == folder]
                    f.write(f"{folder} {len(frames)} (1080,1920,3)\n")
                    seen_folders.add(folder)
        
        # 生成meta_info_tree_GT_{split}.json
        tree_data = {}
        for item in split_meta:
            folder = item['folder']
            if folder not in tree_data:
                tree_data[folder] = {'dataset': dataset_name, 'frames': []}
            tree_data[folder]['frames'].append(item['frame'])
        
        tree_file = meta_dir / f'meta_info_tree_GT_{split}.json'
        with open(tree_file, 'w') as f:
            json.dump(tree_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='准备HazeUAVvideo无人机去雾数据集')
    parser.add_argument('--legacy-exploratory', action='store_true',
                        help='Acknowledge this is not the paper synthesis protocol')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='源数据集目录 (D:\\sjj)')
    parser.add_argument('--visdrone_dir', type=str, required=True,
                        help='VisDrone GT目录 (HazeWorld中现成的)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录 (D:\\HazeUAVvideo)')
    parser.add_argument('--beta_values', type=float, nargs='+',
                        default=[0.005, 0.01, 0.02, 0.03],
                        help='雾浓度系数')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_extract', action='store_true',
                        help='跳过解压步骤（数据已预先解压）')
    parser.add_argument('--copy_gt', action='store_true',
                        help='保存GT时直接复制文件（而非符号链接）')
    
    args = parser.parse_args()

    if not args.legacy_exploratory:
        parser.error('This legacy generator does not reproduce the paper dataset; '
                     'use scripts/generate_hazeuavvideo.sh instead.')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*80)
    print("🚁 HazeUAVvideo无人机去雾数据集准备")
    print("="*80)
    print(f"源目录: {args.source_dir}")
    print(f"VisDrone: {args.visdrone_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"Beta值: {args.beta_values}")
    print(f"训练比例: {args.train_ratio}")
    print("="*80)
    
    # Step 1: 解压数据集
    print("\n📦 Step 1: 解压数据集...")
    extract_archives(args.source_dir, skip=args.skip_extract)
    
    # Step 2: 处理VisDrone（从HazeWorld复制）
    print("\n📋 Step 2: 处理VisDrone...")
    visdrone_src = Path(args.visdrone_dir)
    if visdrone_src.exists():
        visdrone_output = Path(args.output_dir)
        visdrone_output.mkdir(parents=True, exist_ok=True)
        print(f"✓ VisDrone将从 {visdrone_src} 链接")
    
    # Step 3: 处理UAV123、UAVDT、AU-AIR
    datasets = {
        'UAV123': Path(args.source_dir) / 'UAV123' / 'OpenDataLab___UAV123' / 'raw',
        'UAVDT': Path(args.source_dir) / 'UAVDT' / 'OpenDataLab___UAVDT' / 'raw',
        'AU-AIR': Path(args.source_dir) / 'AU-AIR' / 'OpenDataLab___AU-AIR' / 'raw',
    }
    
    for dataset_name, dataset_path in datasets.items():
        if dataset_path.exists():
            process_dataset(dataset_path, dataset_name, args.output_dir, 
                          args.beta_values, args.train_ratio, copy_gt=args.copy_gt)
        else:
            print(f"⚠️  {dataset_name} 路径不存在: {dataset_path}")
    
    print("\n" + "="*80)
    print("✅ HazeUAVvideo数据集准备完成！")
    print("="*80)
    print(f"输出目录: {args.output_dir}")
    print("\n下一步：在Ubuntu中创建符号链接")
    print("  ln -s D:\\HazeUAVvideo /home/ubuntu/data/HazeUAVvideo")


if __name__ == '__main__':
    main()
