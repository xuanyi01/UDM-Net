import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # MMEditing <= 0.16.x
    from mmedit.models.builder import BACKBONES
except Exception:  # pragma: no cover
    # Fallback for different versions
    from mmedit.models.registry import BACKBONES  # type: ignore


def _conv_3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MambaBlock2D(nn.Module):
    """
    A light-weight 2D wrapper around sequence Mamba. Falls back to depthwise separable conv
    when mamba-ssm is not available so the project can still run.
    """

    def __init__(self, channels: int, d_state: int = 16, expand: int = 2, res_scale: float = 1.0):
        super().__init__()
        self.channels = channels
        self.res_scale = res_scale
        self.norm = nn.BatchNorm2d(channels)
        self.use_mamba = False
        self.mamba = None
        self.to_seq = None
        self.from_seq = None

        # Fallback always available
        self.fallback = DepthwiseSeparableConv(channels)

        # Try to import and build a Mamba block
        try:
            from mamba_ssm import Mamba
            try:
                self.to_seq = nn.Conv2d(channels, channels, 1)
                self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=3, expand=expand)
                self.from_seq = nn.Conv2d(channels, channels, 1)
                self.use_mamba = True
            except Exception:
                # Construction failed, keep fallback
                self.use_mamba = False
                self.mamba = None
        except Exception:
            # Import failed, keep fallback
            self.use_mamba = False
            self.mamba = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        if self.use_mamba and (self.mamba is not None):
            # B C H W -> B H W C -> (B*H) W C for 1D sequence processing
            x = self.to_seq(x)
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
            x = self.mamba(x)
            x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            x = self.from_seq(x)
        else:
            x = self.fallback(x)
        return identity + self.res_scale * x


class Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super().__init__()
        self.down = _conv_3x3(in_channels, out_channels, stride=2)
        self.blocks = nn.Sequential(*[MambaBlock2D(out_channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.blocks(x)
        return x


@BACKBONES.register_module()
class MambaBackbone(nn.Module):
    """
    A drop-in backbone that mimics ConvNeXt feature hierarchy but powered by Mamba blocks.

    It produces 4 feature maps with channels specified by embed_dims to be consumed by the
    existing ProjectionHead (in_channels should match [96, 192, 384, 768] by default).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        init_cfg=None,
    ) -> None:
        super().__init__()

        assert len(embed_dims) == 4 and len(depths) == 4, 'embed_dims/depths must have 4 elements.'

        # Keep an init_cfg attribute to be compatible with MMEditing weight init flow
        self.init_cfg = init_cfg

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(*[MambaBlock2D(embed_dims[0]) for _ in range(depths[0])])
        self.stage2 = Stage(embed_dims[0], embed_dims[1], depths[1])
        self.stage3 = Stage(embed_dims[1], embed_dims[2], depths[2])
        self.stage4 = Stage(embed_dims[2], embed_dims[3], depths[3])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: B C H W
        x = self.stem(x)        # 1/4
        c1 = self.stage1(x)     # 1/4
        c2 = self.stage2(c1)    # 1/8
        c3 = self.stage3(c2)    # 1/16
        c4 = self.stage4(c3)    # 1/32
        return c1, c2, c3, c4

    def init_weights(self, pretrained=None):
        """Initialize weights.

        - If `pretrained` is provided, try to load it (non-strict) for flexibility.
        - Otherwise, rely on PyTorch default initialization.
        """
        if pretrained is not None:
            try:
                from mmcv.runner import load_checkpoint
            except Exception:  # mmcv>=2 rename
                from mmcv.runner.checkpoint import load_checkpoint  # type: ignore
            load_checkpoint(self, pretrained, map_location='cpu', strict=False)


