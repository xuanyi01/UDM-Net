#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从checkpoint中提取EMA参数并保存为新文件。
正确处理参数名转换：ema_generator_xxx -> generator.xxx
"""

import os
import torch
import argparse
from collections import OrderedDict


def build_key_mapping(state_dict):
    """
    构建从EMA键名到正确generator键名的映射。
    
    原始checkpoint中同时包含:
    - generator.xxx.yyy (普通参数)
    - ema_generator_xxx_yyy (EMA参数，下划线分隔)
    
    我们需要将EMA参数名映射到对应的generator参数名。
    """
    # 获取所有generator参数名
    generator_keys = [k for k in state_dict.keys() if k.startswith('generator.')]
    
    # 获取所有EMA参数名
    ema_keys = [k for k in state_dict.keys() if k.startswith('ema_generator_')]
    
    if not generator_keys:
        print("警告: 未找到generator参数，将尝试直接转换EMA参数名")
        return None
    
    if not ema_keys:
        print("错误: 未找到EMA参数")
        return None
    
    # 构建映射：将generator键名转换为下划线格式，用于匹配EMA键名
    key_mapping = {}
    for gen_key in generator_keys:
        # generator.backbone.downsample_layers.0.0.weight 
        # -> ema_generator_backbone_downsample_layers_0_0_weight
        ema_key = 'ema_' + gen_key.replace('.', '_')
        if ema_key in ema_keys:
            key_mapping[ema_key] = gen_key
    
    return key_mapping


def extract_ema_params(checkpoint_path, output_path=None):
    """
    从checkpoint中提取EMA参数并保存到新文件。
    
    Args:
        checkpoint_path: 原始checkpoint路径
        output_path: 输出路径，默认为原路径加_ema后缀
    
    Returns:
        输出文件路径
    """
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint必须包含'state_dict'键")
    
    state_dict = checkpoint['state_dict']
    print(f"原始checkpoint共有 {len(state_dict)} 个参数")
    
    # 统计参数类型
    generator_keys = [k for k in state_dict.keys() if k.startswith('generator.')]
    ema_keys = [k for k in state_dict.keys() if k.startswith('ema_generator_')]
    
    print(f"  - generator参数: {len(generator_keys)} 个")
    print(f"  - EMA参数: {len(ema_keys)} 个")
    
    if not ema_keys:
        raise ValueError("在checkpoint中未找到EMA参数 (以'ema_generator_'开头)")
    
    # 构建键名映射
    key_mapping = build_key_mapping(state_dict)
    
    # 提取EMA参数
    ema_state_dict = OrderedDict()
    
    if key_mapping:
        # 使用映射表转换键名
        print(f"\n使用映射表转换 {len(key_mapping)} 个参数")
        for ema_key, gen_key in key_mapping.items():
            ema_state_dict[gen_key] = state_dict[ema_key]
        
        # 检查是否有未映射的EMA参数
        unmapped = set(ema_keys) - set(key_mapping.keys())
        if unmapped:
            print(f"警告: {len(unmapped)} 个EMA参数未找到对应的generator参数")
            for k in list(unmapped)[:5]:
                print(f"  - {k}")
    else:
        # 如果没有generator参数，直接转换EMA参数名
        print("\n直接转换EMA参数名")
        for ema_key in ema_keys:
            # ema_generator_xxx_yyy -> generator.xxx.yyy
            # 移除'ema_'前缀，将下划线替换为点
            new_key = ema_key[4:].replace('_', '.')
            ema_state_dict[new_key] = state_dict[ema_key]
    
    print(f"\n提取了 {len(ema_state_dict)} 个EMA参数")
    
    # 显示示例键名
    print("\n示例键名转换:")
    for i, (k, v) in enumerate(ema_state_dict.items()):
        if i < 5:
            print(f"  {k}")
        else:
            print("  ...")
            break
    
    # 创建新的checkpoint
    new_checkpoint = {
        'state_dict': ema_state_dict,
        'meta': checkpoint.get('meta', {})
    }
    
    # 设置输出路径
    if output_path is None:
        base, ext = os.path.splitext(checkpoint_path)
        output_path = f"{base}_ema{ext}"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存
    print(f"\n保存EMA checkpoint到: {output_path}")
    torch.save(new_checkpoint, output_path)
    
    # 验证保存的文件
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    print("保存成功!")
    
    return output_path


def verify_ema_checkpoint(ema_path, original_path=None):
    """
    验证提取的EMA checkpoint是否正确。
    """
    print(f"\n{'='*50}")
    print("验证EMA checkpoint...")
    
    ema_ckpt = torch.load(ema_path, map_location='cpu')
    ema_state = ema_ckpt['state_dict']
    
    print(f"EMA checkpoint包含 {len(ema_state)} 个参数")
    
    # 检查键名格式
    sample_keys = list(ema_state.keys())[:5]
    print("\n前5个参数名:")
    for k in sample_keys:
        print(f"  {k}")
    
    # 检查是否以generator.开头
    gen_keys = [k for k in ema_state.keys() if k.startswith('generator.')]
    print(f"\n以'generator.'开头的参数: {len(gen_keys)} 个")
    
    if original_path:
        orig_ckpt = torch.load(original_path, map_location='cpu')
        orig_gen_keys = [k for k in orig_ckpt['state_dict'].keys() if k.startswith('generator.')]
        
        # 检查覆盖率
        matched = set(ema_state.keys()) & set(orig_gen_keys)
        print(f"与原始generator参数匹配: {len(matched)}/{len(orig_gen_keys)}")
        
        if len(matched) < len(orig_gen_keys):
            missing = set(orig_gen_keys) - set(ema_state.keys())
            print(f"缺失的参数 ({len(missing)} 个):")
            for k in list(missing)[:5]:
                print(f"  - {k}")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='从checkpoint中提取EMA参数')
    parser.add_argument('checkpoint', help='checkpoint文件路径')
    parser.add_argument('-o', '--output', help='输出路径')
    parser.add_argument('--verify', action='store_true', help='验证提取的checkpoint')
    args = parser.parse_args()
    
    output_path = extract_ema_params(args.checkpoint, args.output)
    
    if args.verify:
        verify_ema_checkpoint(output_path, args.checkpoint)


if __name__ == '__main__':
    main()
