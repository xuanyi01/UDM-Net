"""Fail before training/testing if required Mamba and DCN CUDA ops are absent."""

import torch
from mmcv.ops import DeformConv2dPack
from mamba_ssm import Mamba


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('The paper model requires a CUDA-enabled runtime')
    mamba = Mamba(d_model=32, d_state=16, d_conv=3, expand=2).cuda().eval()
    dcn = DeformConv2dPack(32, 32, 3, padding=1).cuda().eval()
    with torch.no_grad():
        sequence = mamba(torch.zeros(1, 8, 32, device='cuda'))
        image = dcn(torch.zeros(1, 32, 8, 8, device='cuda'))
    torch.cuda.synchronize()
    if sequence.shape != (1, 8, 32) or image.shape != (1, 32, 8, 8):
        raise RuntimeError('Mamba/DCN smoke test returned unexpected shapes')
    print('Runtime verified: Mamba and DCN CUDA forward passes succeeded')


if __name__ == '__main__':
    main()
