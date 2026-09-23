from typing import List, Tuple

import torch
import torch.nn as nn

try:
    # MMEditing <= 0.16.x
    from mmedit.models.builder import BACKBONES
except Exception:  # pragma: no cover
    from mmedit.models.registry import BACKBONES  # type: ignore


class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for 2D feature maps.
    Matches common VMamba/VSSM LN usage (affine over C, epsilon=1e-6).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class MambaBlock2DAligned(nn.Module):
    """
    A VMamba/VSSM-aligned block naming wrapper.

    - Pre-norm (LayerNorm2d)
    - Core mixer: try mamba-ssm.Mamba over width axis; fallback to depthwise-separable conv
    - Residual connection

    Notes: The internal parameter names are intentionally concise (norm, mamba, fallback)
    to load as many official weights as possible via remap; unmatched parts remain randomly init.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.use_mamba = False
        self.mamba = None
        self.to_seq = None
        self.from_seq = None

        class DepthwiseSeparableConv(nn.Module):
            def __init__(self, c: int) -> None:
                super().__init__()
                self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
                self.pw = nn.Conv2d(c, c, 1, bias=False)
                self.bn = nn.BatchNorm2d(c)
                self.act = nn.GELU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.dw(x)
                x = self.pw(x)
                x = self.bn(x)
                return self.act(x)

        self.fallback = DepthwiseSeparableConv(channels)

        try:
            from mamba_ssm import Mamba  # type: ignore
            self.to_seq = nn.Conv2d(channels, channels, 1)
            self.mamba = Mamba(d_model=channels, d_state=16, d_conv=3, expand=2)
            self.from_seq = nn.Conv2d(channels, channels, 1)
            self.use_mamba = True
        except Exception:
            self.use_mamba = False
            self.mamba = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        if self.use_mamba and (self.mamba is not None):
            x = self.to_seq(x)
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
            x = self.mamba(x)
            x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            x = self.from_seq(x)
        else:
            x = self.fallback(x)
        return x + identity


class StageBlocks(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock2DAligned(channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@BACKBONES.register_module()
class VMambaBackbone(nn.Module):
    """
    VMamba-aligned backbone (naming/layout) to maximize compatibility with official VSSM/VMamba weights.

    Modules
    - downsample_layers.0: PatchEmbed (Conv4x4 stride 4 + LN2d)
    - downsample_layers.1/2/3: LN2d + Conv2x2 stride 2
    - stages.{0..3}.{blocks}: Mamba-aligned residual blocks with pre-norm
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        init_cfg=None,
    ) -> None:
        super().__init__()
        assert len(embed_dims) == 4 and len(depths) == 4
        self.init_cfg = init_cfg

        # Patch embedding as downsample_layers[0]
        stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=4, stride=4, padding=0, bias=False),
            LayerNorm2d(embed_dims[0]),
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(embed_dims[i]),
                    nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=2, stride=2, bias=False),
                )
            )

        self.stages = nn.ModuleList()
        # stage 0 keeps resolution (after stem)
        self.stages.append(StageBlocks(embed_dims[0], depths[0]))
        self.stages.append(StageBlocks(embed_dims[1], depths[1]))
        self.stages.append(StageBlocks(embed_dims[2], depths[2]))
        self.stages.append(StageBlocks(embed_dims[3], depths[3]))

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            try:
                from mmcv.runner import load_checkpoint
            except Exception:
                from mmcv.runner.checkpoint import load_checkpoint  # type: ignore
            load_checkpoint(self, pretrained, map_location='cpu', strict=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # downsample 0 (patch embed)
        x = self.downsample_layers[0](x)   # 1/4
        c1 = self.stages[0](x)             # 1/4
        # stage 1
        x = self.downsample_layers[1](c1)  # 1/8
        c2 = self.stages[1](x)             # 1/8
        # stage 2
        x = self.downsample_layers[2](c2)  # 1/16
        c3 = self.stages[2](x)             # 1/16
        # stage 3
        x = self.downsample_layers[3](c3)  # 1/32
        c4 = self.stages[3](x)             # 1/32
        return c1, c2, c3, c4


