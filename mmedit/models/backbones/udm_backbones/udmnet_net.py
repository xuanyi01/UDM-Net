import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .udm_modules import ResidualBlocksWithInputConv
from .udm_stda import STDABlock
from .udm_utils import get_discrete_values, get_flow_from_grid, flow_warp_5d
from mmedit.models.backbones.mamba_backbone import MambaBlock2D
class DeformConvBlock(nn.Module):
    """Light wrapper for DCNv2 if available, else fallback to Conv-BN-Act.

    Keeps same in/out channels and 3x3 kernel. Residual connection.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.use_dcn = False
        try:
            from mmcv.ops import DeformConv2dPack  # type: ignore
            self.conv = DeformConv2dPack(channels, channels, 3, padding=1)
            self.use_dcn = True
        except Exception:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x + identity

class TemporalDeformFusion(nn.Module):
    """Approximate MSDeformAttn-style temporal fusion with learned offsets and weights.

    Samples K offsets per neighbor frame and aggregates them with learned weights.
    """

    def __init__(self, channels: int, num_samples: int = 4):
        super().__init__()
        self.channels = channels
        self.num_samples = num_samples
        self.offset_head = nn.Conv2d(channels, 2 * num_samples, 3, padding=1)
        self.weight_head = nn.Conv2d(channels, num_samples, 1)

    def forward(self, feats_jr: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
        # feats_jr: b, nr, c, h, w; query_feat: b, c, h, w
        b, nr, c, h, w = feats_jr.shape
        device = feats_jr.device
        # predict offsets/weights from query
        offsets = self.offset_head(query_feat)  # b, 2K, h, w
        weights = self.weight_head(query_feat)  # b, K, h, w
        weights = weights.view(b, 1, self.num_samples, h, w).repeat(1, nr, 1, 1, 1)  # b, nr, K, h, w
        weights = torch.softmax(weights, dim=2)

        # base grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device), indexing='ij')
        base_grid = torch.stack([xx, yy], dim=-1)  # h, w, 2
        base_grid = base_grid.view(1, 1, 1, h, w, 2).repeat(b, nr, self.num_samples, 1, 1, 1)  # b,nr,K,h,w,2

        # offsets to normalized grid
        off = offsets.view(b, 1, self.num_samples, 2, h, w).permute(0, 1, 2, 4, 5, 3)  # b,1,K,h,w,2
        off = off.repeat(1, nr, 1, 1, 1, 1)  # b,nr,K,h,w,2
        grid = base_grid + off

        # sample and aggregate
        feats = feats_jr.view(b * nr, c, h, w)
        grid = grid.view(b * nr * self.num_samples, h, w, 2)
        feats_rep = feats.unsqueeze(1).repeat(1, self.num_samples, 1, 1, 1).view(b * nr * self.num_samples, c, h, w)
        sampled = F.grid_sample(feats_rep, grid, align_corners=True, mode='bilinear', padding_mode='border')  # B*nr*K, c, h, w
        sampled = sampled.view(b, nr, self.num_samples, c, h, w)
        weights = weights.unsqueeze(3)  # b, nr, K, 1, h, w
        fused = (weights * sampled).sum(dim=2)  # b, nr, c, h, w
        # average over neighbor frames
        fused = fused.mean(dim=1)  # b, c, h, w
        return fused

class FAAModule(nn.Module):
    """Feature-Aware Alignment Module (简化版)

    以参考帧特征为条件，对每个邻帧做两次级联的 DeformConv 对齐，并用通道注意力做自适应权重。
    输入: ref (b,c,h,w), nbrs (b,nr,c,h,w)
    输出: aligned_nbrs (b,nr,c,h,w)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.reduce = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.align1 = DeformConvBlock(channels)
        self.align2 = DeformConvBlock(channels)
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True), nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, ref: torch.Tensor, nbrs: torch.Tensor) -> torch.Tensor:
        b, nr, c, h, w = nbrs.shape
        outs = []
        for i in range(nr):
            x = torch.cat([ref, nbrs[:, i]], dim=1)  # b,2c,h,w
            x = self.reduce(x)
            x = self.align1(x)
            x = self.align2(x)
            wgt = self.se(x)
            x = wgt * x + (1 - wgt) * nbrs[:, i]
            outs.append(x)
        return torch.stack(outs, dim=1)

class ShiftFAAModule(nn.Module):
    """UVENet风格的 Shift-FAAM：通道分组 + 8方向空间移位 + DSC + 通道注意力。

    输入: ref (b,c,h,w), nbrs (b,nr,c,h,w)
    输出: 对齐/增强后的邻帧特征 (b,nr,c,h,w)
    """

    def __init__(self, channels: int, groups: int = 8, shift_len: int = 3, reduction: int = 4):
        super().__init__()
        assert channels % groups == 0, 'channels must be divisible by groups'
        self.channels = channels
        self.groups = groups
        self.shift_len = shift_len
        # 将 8 个方向移位后的特征拼接 (8*c) -> 1x1 降回 c
        self.reduce = nn.Conv2d(channels * 8, channels, 1, bias=False)
        # 深度可分离卷积融合
        self.dsc_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dsc_pw = nn.Conv2d(channels, channels, 1, bias=False)
        # 通道注意力（SE）
        mid = max(channels // reduction, 8)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True), nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=True), nn.Sigmoid()
        )

    def _shift2d(self, x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        # 零填充移位，避免环绕
        b, c, h, w = x.shape
        top = max(dy, 0)
        bottom = max(-dy, 0)
        left = max(dx, 0)
        right = max(-dx, 0)
        x = F.pad(x, (left, right, top, bottom))
        y0 = top
        x0 = left
        y1 = y0 + h
        x1 = x0 + w
        return x[:, :, y0:y1, x0:x1]

    def forward(self, ref: torch.Tensor, nbrs: torch.Tensor) -> torch.Tensor:
        b, nr, c, h, w = nbrs.shape
        outs = []
        # 8 方向位移（含对角）
        dirs = [(-self.shift_len, 0), (self.shift_len, 0), (0, -self.shift_len), (0, self.shift_len),
                (-self.shift_len, -self.shift_len), (-self.shift_len, self.shift_len),
                (self.shift_len, -self.shift_len), (self.shift_len, self.shift_len)]
        for i in range(nr):
            x = nbrs[:, i]  # b,c,h,w
            # 按通道分组，并对每组做 8 方向移位后再还原拼接
            gch = c // self.groups
            shifted_groups = []
            for g in range(self.groups):
                xi = x[:, g * gch:(g + 1) * gch]
                shifted = [self._shift2d(xi, dy, dx) for (dy, dx) in dirs]
                shifted_groups.append(torch.cat(shifted, dim=1))  # b, 8*gch, h, w
            x = torch.cat(shifted_groups, dim=1)  # b, 8*c, h, w
            x = self.reduce(x)  # b,c,h,w
            x = self.dsc_dw(x)
            x = self.dsc_pw(x)
            x = F.gelu(x)
            wgt = self.ca(x)
            x = wgt * x + (1 - wgt) * nbrs[:, i]
            outs.append(x)
        return torch.stack(outs, dim=1)
class MambaTimeMixer(nn.Module):
    """Temporal mixer along frame dimension using Mamba.

    Input: (b, t, c, h, w)  Output: same shape
    - Supports bidirectional processing by running Mamba on forward and reversed
      sequences, then averaging the outputs.
    - Falls back to depthwise temporal 3D conv when mamba-ssm is unavailable.
    """

    def __init__(self, channels: int, d_state: int = 16, expand: int = 2, bidir: bool = False):
        super().__init__()
        self.channels = channels
        self.use_mamba = False
        self.bidir = bidir
        self.d_state = d_state
        self.expand = expand

        # Fallback temporal depthwise conv
        self.dw3d = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False)
        self.pw3d = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

        try:
            from mamba_ssm import Mamba  # noqa: F401
            self.mamba_f = Mamba(d_model=channels, d_state=d_state, d_conv=3, expand=expand)
            self.mamba_b = Mamba(d_model=channels, d_state=d_state, d_conv=3, expand=expand) if bidir else None
            self.use_mamba = True
        except Exception:
            self.mamba_f = None
            self.mamba_b = None

    def _run_seq(self, x_btch: torch.Tensor, mamba) -> torch.Tensor:
        # x_btch: (B*H*W) T C
        return mamba(x_btch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: b t c h w
        if self.use_mamba and self.mamba_f is not None:
            b, t, c, h, w = x.shape
            x_ = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)  # (b*h*w) t c
            y_f = self._run_seq(x_, self.mamba_f)
            if self.bidir and (self.mamba_b is not None):
                y_b = torch.flip(x_, dims=[1])
                y_b = self._run_seq(y_b, self.mamba_b)
                y_b = torch.flip(y_b, dims=[1])
                x_ = 0.5 * (y_f + y_b)
            else:
                x_ = y_f
            x_ = x_.view(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()  # b t c h w
            return x_
        # fallback conv path
        # convert to (b, c, t, h, w) for Conv3d, then convert back
        b, t, c, h, w = x.shape
        x_cf = x.permute(0, 2, 1, 3, 4).contiguous()  # b c t h w
        y = self.dw3d(x_cf)
        y = self.pw3d(y)
        y = self.act(y)
        y = y.permute(0, 2, 1, 3, 4).contiguous()  # b t c h w
        return y


from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, load_checkpoint
from mmedit.models import builder
from mmedit.models.common import PixelShufflePack
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class DAPDDecodeLayer(nn.Module):
    """One level of the Degradation-Aware Prior Decoding Path (DAPD)."""

    def __init__(self,
                 channels,
                 level,
                 upsample=True,
                 num_trans_bins=32,
                 memory_enhance=True,
                 use_mamba=True,
                 mamba_bidir: bool = False,
                 mamba_d_state: int = 16,
                 mamba_expand: int = 2):
        super().__init__()
        self.level = level

        self.upsample = upsample
        if upsample:
            self.up = PixelShufflePack(channels, channels, 2, 3)
            scale_head = [
                ConvModule(channels, channels, 3, padding=1)
                for _ in range(2)]
            self.scale_head = nn.Sequential(*scale_head)

        # Lightweight Mamba enhancement on prior decoding path
        self.use_mamba = use_mamba
        self.mamba_enh = MambaBlock2D(channels, d_state=mamba_d_state, expand=mamba_expand, res_scale=0.1) if use_mamba else nn.Identity()
        # Token mixer along transmission-bin dimension (d)
        class MambaTokenMixer(nn.Module):
            def __init__(self, c: int):
                super().__init__()
                self.use_mamba = False
                self.dw1d = nn.Conv1d(c, c, 3, padding=1, groups=c, bias=False)
                self.pw1d = nn.Conv1d(c, c, 1, bias=False)
                self.act = nn.GELU()
                try:
                    from mamba_ssm import Mamba  # noqa: F401
                    self.mamba = Mamba(d_model=c, d_state=mamba_d_state, d_conv=3, expand=mamba_expand)
                    self.use_mamba = True
                except Exception:
                    self.mamba = None
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: b, c, d
                if self.use_mamba and self.mamba is not None:
                    b, c, d = x.shape
                    x_ = x.permute(0, 2, 1).contiguous()  # b, d, c
                    x_ = self.mamba(x_)
                    return x_.permute(0, 2, 1).contiguous()
                y = self.dw1d(x)
                y = self.pw1d(y)
                return self.act(y)
        self.token_mixer = MambaTokenMixer(channels) if use_mamba else nn.Identity()

        self.num_trans_bins = num_trans_bins
        self.head_t = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, num_trans_bins, 1, 1, 0))
        self.head_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, 1, 1, 1, 0), nn.Sigmoid())

        self.memory_enhance = memory_enhance

    def forward(self, feats):
        level = self.level
        #
        enc_skip = feats['spatial_p'][-1][level]
        if self.upsample:
            feat = enc_skip + self.up(feats['decode_p'][-1][level + 1])
            feat = self.scale_head(feat)
        else:
            feat = enc_skip
        feats['decode_p'][-1][level] = feat

        # DAPD: estimate transmission and atmospheric light.
        logit_t = self.head_t(feat)
        prob_t = torch.softmax(logit_t, dim=1)  # b, num_trans_bins, h, w
        b, d, h, w = prob_t.shape
        values = get_discrete_values(self.num_trans_bins, 0., 1.) \
            .view(1, self.num_trans_bins, 1, 1).to(prob_t.device).repeat(b, 1, h, w)
        out_t = (prob_t * values).sum(dim=1, keepdim=True)
        out_a = self.head_a(feat)
        feats['stage_t'][level] = out_t
        feats['stage_a'][level] = out_a

        # DAPD: construct prior tokens and retrieve prior memory.
        if self.memory_enhance:
            # prior token
            token_p = feat.unsqueeze(2) * prob_t.unsqueeze(1)  # b, c, d, h, w
            token_p = token_p.mean(dim=(-2, -1))  # b, c, d
            # mix along transmission-bin dimension
            if self.use_mamba:
                token_p = self.token_mixer(token_p)
            feats['token_p'][-1][level] = token_p

            # retrieve memory
            mem_p = [x[level] for x in feats['token_p']]
            mem_p = torch.stack(mem_p, dim=1)  # b, N, c, d
            if self.use_mamba:
                b, N, c, d = mem_p.shape
                mem_p = mem_p.reshape(b * N, c, d)
                mem_p = self.token_mixer(mem_p)
                mem_p = mem_p.reshape(b, N, c, d)
            # flatten memory bins then feature
            b, N, c, d = mem_p.shape
            mem_p = mem_p.transpose(-2, -1).reshape(b, N * d, c).contiguous()  # b, Nd, c
            # read memory & attention
            _, _, h, w = feat.shape
            q_p = feat.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()  # b, hw, c
            scale = c ** -0.5
            attn = (q_p @ mem_p.transpose(-2, -1)) * scale  # b, hw, Nd
            attn = F.softmax(attn, dim=-1)
            en_p = attn @ mem_p  # b, hw, c
            en_p = en_p.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # b, c, h, w
            # Apply Mamba enhancement (residual) to memory-enhanced prior features
            if self.use_mamba:
                en_p = self.mamba_enh(en_p)
            feats['enhance_p'][-1][level] = en_p
        else:
            feats['enhance_p'][-1][level] = feat

        return feats


class RTSRDecodeLayer(nn.Module):
    """One level of the Reliable Temporal Scene Recovery Path (RTSR)."""

    def __init__(self,
                 channels,
                 level,
                 upsample=True,
                 prior_guide=True,
                 num_kv_frames=3,
                 align_depth=1,
                 num_heads=1,
                 kernel_size=3,
                 use_mamba=True,
                 mamba_bidir: bool = False,
                 mamba_d_state: int = 16,
                 mamba_expand: int = 2,
                 use_dcn: bool = True,
                 use_tdf: bool = True,
                 faam_type: str = 'none'):
        super().__init__()
        self.level = level

        self.upsample = upsample
        if upsample:
            self.up = PixelShufflePack(channels, channels, 2, 3)

        self.prior_guide = prior_guide
        if prior_guide:
            self.guide_conv = ResidualBlocksWithInputConv(channels * 2, channels, 2)

        # Mamba enhancement blocks around scene path
        self.use_mamba = use_mamba
        self.mamba_pre_align = MambaBlock2D(channels, d_state=mamba_d_state, expand=mamba_expand, res_scale=0.1) if use_mamba else nn.Identity()
        self.mamba_post_aggre = MambaBlock2D(channels, d_state=mamba_d_state, expand=mamba_expand, res_scale=0.1) if use_mamba else nn.Identity()
        # Optional DCN pre/post align blocks
        self.dcn_pre = DeformConvBlock(channels) if use_dcn else nn.Identity()
        self.dcn_post = DeformConvBlock(channels) if use_dcn else nn.Identity()
        self.time_mixer = MambaTimeMixer(channels, d_state=mamba_d_state, expand=mamba_expand, bidir=mamba_bidir) if use_mamba else nn.Identity()
        # Gated fusion between conv branch and mamba-enhanced branch
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        ) if use_mamba else nn.Identity()

        if not isinstance(num_kv_frames, (list, tuple)):
            num_kv_frames = list(range(1, num_kv_frames + 1))
        else:
            num_kv_frames = sorted(num_kv_frames)
        self.num_kv_frames = num_kv_frames
        self.align_layer = STDABlock(
            channels, num_heads=num_heads,
            align_depth=align_depth, dw_ks=kernel_size)

        self.aggre_beta = nn.Parameter(torch.ones(1))
        self.aggre_conv = ResidualBlocksWithInputConv(channels * 2, channels, 2)
        # Optional temporal deformable fusion (approx MSDeformAttn) after feats_jr is formed
        self.use_tdf = use_tdf
        if use_tdf:
            self.temporal_deform_fusion = TemporalDeformFusion(channels, num_samples=4)
        else:
            self.temporal_deform_fusion = None

        # Feature-Aware Alignment (pre-align before STDABlock)
        self.faam_type = faam_type  # 'none' | 'dcn' | 'shift'
        if faam_type == 'dcn' and use_dcn:
            self.faam = FAAModule(channels)
        elif faam_type == 'shift':
            self.faam = ShiftFAAModule(channels, groups=8, shift_len=3)
        else:
            self.faam = None

        self.head_j = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, 3, 1, 1, 0))

    def forward(self, feats):
        level = self.level
        num_kv_frames = self.num_kv_frames
        #
        enc_skip = feats['spatial_j'][-1][level]
        if self.upsample:
            feat_j = enc_skip + self.up(feats['decode_j'][-1][level + 1])
        else:
            feat_j = enc_skip

        # RTSR: fuse guidance from DAPD.
        if self.prior_guide:
            feat_p = feats['enhance_p'][-1][level]
            feat_j = self.guide_conv(torch.cat([feat_p, feat_j], dim=1))
        # Optional Mamba enhancement before alignment
        feat_j = self.dcn_pre(feat_j)
        q_j = self.mamba_pre_align(feat_j) if self.use_mamba else feat_j

        # RTSR: form temporal-range contexts for TAB.
        # prepare features
        kv_j, kv_p = [], []
        nf = len(feats['decode_j']) - 1  # buffer frames: skip current timestep
        for step in range(1, max(num_kv_frames) + 1):
            kv_j.append(feats['decode_j'][max(nf - step, 0)][level] if nf > 0 else feat_j)
            kv_p.append(feats['enhance_p'][max(nf - step, 0)][level])
            # print(f"num_kv_frames: {num_kv_frames}, nf: {nf}, step: {step}, frame: {max(nf - step, 0)}")
        kv_j = torch.stack(kv_j, dim=1)  # b, nr, c, h, w
        kv_p = torch.stack(kv_p, dim=1)  # b, nr, c, h, w
        # Temporal mixing with Mamba along frame dimension (nr)
        if self.use_mamba:
            kv_j = self.time_mixer(kv_j)
            kv_p = self.time_mixer(kv_p)

        # 预对齐（FAAM：DCN/Shift 版本）
        if self.faam is not None:
            kv_j = self.faam(q_j, kv_j)

        # multi-range alignment
        feats_jr, feats_pr = [], []
        grids = []
        for r, kv_frames in enumerate(num_kv_frames):
            # gradually refine flow
            if self.upsample:
                grid_r = feats['pos_j'][-1][level + 1][:, r]
                # print(f"level: {level}, grid_r: {grid_r.shape}")
                b, g, h, w, p = grid_r.shape
                assert p == 3, "Should be 5-D input sample"
                grid_r = einops.rearrange(grid_r, 'b g h w p -> (b g) p h w')
                # 'False' is more similar to generated ref points
                grid_r = F.interpolate(grid_r, scale_factor=2, mode='bilinear', align_corners=False)
                grid_r = einops.rearrange(grid_r, '(b g) p h w -> b g h w p', g=g)
            else:
                grid_r = None

            # align scene features
            kv_jr = kv_j[:, :kv_frames]
            feat_jr, grid_r, ref_r = self.align_layer(q_j, kv_jr, grid_r)
            feats_jr.append(feat_jr)
            grids.append(grid_r)

            # warp prior features
            b, g, h, w, p = grid_r.shape
            feat_pr = kv_p[:, :kv_frames]
            feat_pr = einops.rearrange(feat_pr, 'b nf (g c) h w -> (b g) c nf h w', g=g)
            _grid = einops.rearrange(grid_r, 'b g h w p -> (b g) h w p')
            flow = get_flow_from_grid(_grid, ref_r, d=kv_frames)
            feat_pr = flow_warp_5d(feat_pr, flow.unsqueeze(1))
            assert feat_pr.shape[2] == 1
            feats_pr.append(feat_pr.squeeze(2))
        feats['pos_j'][-1][level] = torch.stack(grids, dim=1)
        feats['ref_j'][-1][level] = ref_r

        # GAAF: prior-guided multi-range aggregation and adaptive gating.
        # prepare faetures
        feats_jr = torch.stack(feats_jr, dim=1)
        b, nr, c, h, w = feats_jr.shape
        scale = c ** -0.5

        # attn: j
        q_j = einops.rearrange(feat_j, 'b c h w -> (b h w) c').unsqueeze(1)  # bhw 1 c
        k_j = einops.rearrange(feats_jr, 'b nr c h w -> (b h w) nr c')  # bhw nr c
        attn_j = q_j @ k_j.transpose(-2, -1) * scale  # bhw 1 nr
        # attn: p
        q_p = feats['enhance_p'][-1][level]
        q_p = einops.rearrange(q_p, 'b c h w -> (b h w) c').unsqueeze(1)  # bhw 1 c
        k_p = torch.stack(feats_pr, dim=1)
        k_p = einops.rearrange(k_p, 'b nr c h w -> (b h w) nr c')  # bhw nr c
        attn_p = q_p @ k_p.transpose(-2, -1) * scale  # bhw 1 nr
        # attn
        attn = attn_j + self.aggre_beta * attn_p
        attn = F.softmax(attn, dim=-1)

        v = einops.rearrange(feats_jr, 'b nr c h w -> (b h w) nr c')  # bhw nr c
        feat = attn @ v
        assert feat.shape[1] == 1
        feat = einops.rearrange(feat.squeeze(1), '(b h w) c -> b c h w', h=h, w=w)
        # Optional temporal deformable fusion (value enhancement)
        if self.use_tdf:
            tdf = self.temporal_deform_fusion(feats_jr, feat_j)
        else:
            tdf = 0
        feat_conv = self.aggre_conv(torch.cat([feat, feat_j + tdf], dim=1))
        if self.use_mamba:
            feat_m = self.mamba_post_aggre(feat_conv)
            gate = self.fuse_gate(torch.cat([feat_conv, feat_m], dim=1))
            feat = gate * feat_m + (1 - gate) * feat_conv
        else:
            feat = feat_conv
        feat = self.dcn_post(feat)
        feats[f'decode_j'][-1][level] = feat

        # estimate J
        out_j = self.head_j(feat)
        feats['stage_j'][level] = out_j

        return feats


@BACKBONES.register_module()
class UDMNet(BaseModule):
    """UDM-Net: a degradation-aware state-space fusion network for UAV video dehazing."""
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]

    def __init__(self,
                 backbone,
                 neck,
                 upsampler,
                 channels=32,
                 num_trans_bins=32,
                 align_depths=(1, 1, 1, 1),
                 num_kv_frames=(1, 2, 3),
                 # Mamba options for DAPD/RTSR; argument names retain checkpoint/config compatibility.
                 use_mamba_in_mpg: bool = True,
                 use_mamba_in_msr: bool = True,
                 mamba_bidir: bool = False,
                 mamba_d_state: int = 16,
                 mamba_expand: int = 2,
                 # Per-level overrides (len==num_stages). If provided, they override the scalar options above.
                 use_mamba_in_mpg_levels=None,
                 use_mamba_in_msr_levels=None,
                 mamba_bidir_levels=None,
                 mamba_d_state_levels=None,
                 mamba_expand_levels=None,
                 # DCN / Temporal deform fusion per-level switches
                 use_dcn_levels=None,
                 use_tdf_levels=None,
                 # FAAM 类型 per-level：'none' | 'dcn' | 'shift'
                 faam_type_levels=None,
                 # 是否对编码器输出使用 InstanceNorm（替换/补充LN）
                 encoder_use_instance_norm: bool = False,
                 # 是否启用 UV 风格多尺度解码头
                 use_uv_decoder: bool = False,
                 # Optional RGB refinement after upsampling; not the paper's GMRR.
                 use_rgb_grm: bool = False,
                 # 是否启用全局细化模块
                 use_global_refine: bool = True,
                 ):
        super().__init__()

        self.backbone = builder.build_component(backbone)
        self.neck = builder.build_component(neck)
        self.upsampler = builder.build_component(upsampler)
        self.encoder_use_instance_norm = encoder_use_instance_norm
        self.use_global_refine = use_global_refine

        num_stages = len(align_depths)
        self.num_stages = num_stages

        # DAPD transmission bins.
        self.num_trans_bins = num_trans_bins

        # RTSR temporal ranges; assume num_kv_frames is consecutive.
        self.num_kv_frames = num_kv_frames

        # align & aggregate
        assert channels % 32 == 0
        num_heads = [channels // 32 for _ in range(num_stages)]
        kernel_sizes = [9, 7, 5, 3]

        self.prior_decoder_layers = nn.ModuleList()
        self.scene_decoder_layers = nn.ModuleList()

        guided_levels = (2, 3)  # memory consumption
        # normalize per-level params
        def _per_level(param, default_scalar):
            if isinstance(param, (list, tuple)):
                assert len(param) == num_stages, 'per-level list length must equal num_stages'
                return list(param)
            return [default_scalar for _ in range(num_stages)]

        use_mpg_lv = _per_level(use_mamba_in_mpg_levels, use_mamba_in_mpg)
        use_msr_lv = _per_level(use_mamba_in_msr_levels, use_mamba_in_msr)
        bidir_lv = _per_level(mamba_bidir_levels, mamba_bidir)
        d_state_lv = _per_level(mamba_d_state_levels, mamba_d_state)
        expand_lv = _per_level(mamba_expand_levels, mamba_expand)
        dcn_lv = _per_level(use_dcn_levels, True)
        tdf_lv = _per_level(use_tdf_levels, True)
        if faam_type_levels is None:
            faam_lv = ['dcn' if s >= 2 else 'none' for s in range(num_stages)]
        else:
            assert len(faam_type_levels) == num_stages
            faam_lv = list(faam_type_levels)

        for s in range(num_stages):
            self.prior_decoder_layers.append(
                DAPDDecodeLayer(
                    channels, s,
                    upsample=s < num_stages - 1, memory_enhance=s in guided_levels,
                    use_mamba=use_mpg_lv[s], mamba_bidir=bidir_lv[s],
                    mamba_d_state=d_state_lv[s], mamba_expand=expand_lv[s]
                ))
            self.scene_decoder_layers.append(
                RTSRDecodeLayer(
                    channels, s,
                    upsample=s < num_stages - 1, prior_guide=s in guided_levels,
                    num_kv_frames=num_kv_frames, align_depth=align_depths[s],
                    num_heads=num_heads[s], kernel_size=kernel_sizes[s],
                    use_mamba=use_msr_lv[s], mamba_bidir=bidir_lv[s],
                    mamba_d_state=d_state_lv[s], mamba_expand=expand_lv[s],
                    use_dcn=dcn_lv[s], use_tdf=tdf_lv[s], faam_type=faam_lv[s]
                ))

        # Global refinement at highest resolution (before upsampler)
        class GMRRFeatureRefinement(nn.Module):
            """Feature refinement stage of GMRR, before residual reconstruction."""
            def __init__(self, c: int, use_mamba: bool = True, d_state: int = 16, expand: int = 2, res_scale: float = 0.1):
                super().__init__()
                if use_mamba:
                    self.refine = MambaBlock2D(c, d_state=d_state, expand=expand, res_scale=res_scale)
                else:
                    self.refine = nn.Sequential(
                        nn.Conv2d(c, c, 3, padding=1, bias=False), nn.GELU(), nn.Conv2d(c, c, 1, bias=False)
                    )
                self.res_scale = res_scale
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.refine(x)
                return y
        if use_global_refine:
            self.global_refine = GMRRFeatureRefinement(channels, use_mamba=True, d_state=d_state_lv[0], expand=expand_lv[0], res_scale=0.1)
        else:
            self.global_refine = nn.Identity()

        # UV 风格多尺度解码头（可选）
        class UVDecoderHead(nn.Module):
            def __init__(self, c: int, num_scales: int):
                super().__init__()
                self.num_scales = num_scales
                self.proj = nn.ModuleList([nn.Conv2d(c, c, 1, bias=False) for _ in range(num_scales)])
                self.fuse = nn.Sequential(
                    nn.Conv2d(c * num_scales, c, 3, padding=1, bias=False), nn.GELU(),
                    nn.Conv2d(c, c, 3, padding=1, bias=False)
                )
            def forward(self, feats_list):
                # feats_list: [level0(highest res), level1, level2, level3]
                ref = feats_list[0]
                h, w = ref.shape[2:]
                ups = []
                for i, f in enumerate(feats_list):
                    x = self.proj[i](f)
                    if x.shape[2] != h or x.shape[3] != w:
                        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
                    ups.append(x)
                x = torch.cat(ups, dim=1)
                return self.fuse(x)
        self.use_uv_decoder = use_uv_decoder
        if self.use_uv_decoder:
            self.uv_decoder = UVDecoderHead(channels, num_stages)
        else:
            self.uv_decoder = None

        # Optional image-domain refinement, separate from GMRR.
        class OptionalRGBRefinement(nn.Module):
            """Optional image-domain RGB refinement, separate from GMRR."""
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(6, 32, 3, padding=1, bias=False), nn.GELU(),
                    nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.GELU(),
                )
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(
                    nn.Conv2d(16, 16, 1, bias=True), nn.GELU(),
                    nn.Conv2d(16, 3, 1, bias=True), nn.Sigmoid()
                )
            def forward(self, enh: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
                x = torch.cat([enh, orig], dim=1)
                x = self.conv(x)
                g = self.pool(x)
                w = self.head(g)  # b,3,1,1
                return w
        self.use_rgb_grm = use_rgb_grm
        self.rgb_grm = OptionalRGBRefinement() if self.use_rgb_grm else None

        self.window_size = 32  # for padding
        rgb_mean = torch.Tensor(self.RGB_MEAN).reshape(1, 3, 1, 1)
        rgb_std = torch.Tensor(self.RGB_STD).reshape(1, 3, 1, 1)
        self.register_buffer('rgb_mean', rgb_mean)
        self.register_buffer('rgb_std', rgb_std)

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    def check_image_size(self, img):
        # https://github.com/JingyunLiang/SwinIR/blob/5aa89a7b275eeddc75cd7806378c89d23f298c48/main_test_swinir.py#L66
        # https://github.com/ZhendongWang6/Uformer/issues/32
        _, _, h, w = img.size()
        window_size = self.window_size
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        out = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return out

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        # 可选：对编码器输出做 InstanceNorm，更贴合增强任务
        if self.encoder_use_instance_norm:
            if isinstance(x, (list, tuple)):
                x = tuple(F.instance_norm(f) for f in x)
            else:
                x = F.instance_norm(x)
        return x

    def split_feat(self, feats, feat):
        feat_p, feat_j = [], []
        for s in range(self.num_stages):
            c = feat[s].shape[1]
            split_size_or_sections = c // 2
            x = torch.split(feat[s], split_size_or_sections, dim=1)
            feat_p.append(x[0])
            feat_j.append(x[1])
        feats['spatial_p'].append(feat_p)
        feats['spatial_j'].append(feat_j)
        return feats

    def decode(self, feats):
        # init
        keys = ['decode_p', 'token_p', 'enhance_p',
                'decode_j', 'pos_j', 'ref_j']
        for k in keys:
            feats[k].append([None] * self.num_stages)
        keys = ['stage_t', 'stage_a', 'stage_j']
        for k in keys:
            feats[k] = [None] * self.num_stages

        for s in range(self.num_stages - 1, -1, -1):
            feats = self.prior_decoder_layers[s](feats)
            feats = self.scene_decoder_layers[s](feats)

        return feats

    def forward(self, lqs):
        """
        Forward function

        Args:
            lqs (Tensor): Input hazy sequence with shape (n, t, c, h, w).

        Returns:
            out (Tensor): Output haze-free sequence with shape (n, t, c, h, w).
        """
        n, T, c, h, w = lqs.shape

        feats = {
            'spatial_p': [], 'decode_p': [], 'token_p': [], 'enhance_p': [],
            'spatial_j': [], 'decode_j': [], 'pos_j': [], 'ref_j': [],
            'stage_j': [], 'stage_t': [], 'stage_a': []
        }

        out_js = []
        img_01s = []
        aux_js, aux_is = [], []

        for i in range(0, T):
            # print(f"\ntime: {i}")
            img = self.check_image_size(lqs[:, i, :, :, :])
            img_01 = img * self.rgb_std + self.rgb_mean  # to the range of [0., 1.]
            img_01s.append(img_01)

            # encode
            feat = self.extract_feat(img)  # tuple of feats, (4s, 8s, 16s, ...)
            feats = self.split_feat(feats, feat)

            # decode
            feats = self.decode(feats)

            # get output
            # 汇聚多尺度解码特征（可选）
            if self.use_uv_decoder and (self.uv_decoder is not None):
                ms_feats = [feats['decode_j'][-1][s] for s in range(self.num_stages)]
                feat_j = self.uv_decoder(ms_feats)
            else:
                feat_j = feats['decode_j'][-1][0]

            # 特征域全局细化（可选，Mamba 版）
            if self.use_global_refine:
                feat_j = self.global_refine(feat_j)
            out = self.upsampler(feat_j)

            # Optional image-domain refinement after GMRR reconstruction.
            if self.use_rgb_grm and (self.rgb_grm is not None):
                w_rgb = self.rgb_grm(out, img_01)
                out = w_rgb * out + (1 - w_rgb) * img_01
            out = img_01 + out

            if self.training:
                assert h == out.shape[2] and w == out.shape[3]
            out_js.append(out[:, :, 0: h, 0: w].contiguous())

            # auxiliary output for the current timestep
            if self.training:
                aux_j, aux_i = [], []
                for s in range(self.num_stages):
                    tmp_j = F.interpolate(feats['stage_j'][s], size=img.shape[2:], mode='bilinear')
                    out_j = img_01 + tmp_j  # residue
                    tmp_t = F.interpolate(feats['stage_t'][s], size=img.shape[2:], mode='bilinear').clip(0, 1)
                    tmp_a = feats['stage_a'][s]
                    out_i = out_j * tmp_t + tmp_a * (1 - tmp_t)
                    aux_j.append(out_j[:, :, 0: h, 0: w])
                    aux_i.append(out_i[:, :, 0: h, 0: w])
                aux_js.append(aux_j)
                aux_is.append(aux_i)

            # memory management
            feats['spatial_j'].pop(0)
            feats['spatial_p'].pop(0)
            if len(feats['decode_j']) > max(self.num_kv_frames):
                feats['decode_j'].pop(0)
                feats['decode_p'].pop(0)
                feats['enhance_p'].pop(0)
                assert len(feats['decode_p']) == len(feats['decode_j'])
            if not self.training:
                feats['pos_j'].pop(0)
                feats['ref_j'].pop(0)

        out = dict(out=torch.stack(out_js, dim=1))  # output dict

        # auxiliary output for a sequence
        if self.training:
            pos, ref = [], []  # sampling locations
            for s in range(self.num_stages):
                pos.append(torch.stack([feats['pos_j'][i][s] for i in range(T)], dim=1))
                ref.append(torch.stack([feats['ref_j'][i][s] for i in range(T)], dim=1))
            out['pos'] = pos  # b, T, nr, g, h, w, 3
            out['ref'] = ref  # b, T, 1, h, w, 3
        if self.training:
            aux_j, aux_i = [], []  # Js, Is
            for s in range(self.num_stages):
                aux_j.append(torch.stack([aux_js[i][s] for i in range(T)], dim=1))
                aux_i.append(torch.stack([aux_is[i][s] for i in range(T)], dim=1))
            out['aux_j'] = aux_j
            out['aux_i'] = aux_i
            out['img_01'] = torch.stack(img_01s, dim=1)

        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        logger = get_root_logger()
        logger.info(f"Init weights: {pretrained}")
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif self.backbone.init_cfg is not None:
            self.backbone.init_weights()
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
