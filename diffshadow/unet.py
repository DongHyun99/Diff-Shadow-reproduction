"""Parallel UNets of Diff-Shadow (local noise estimator + global shadow-free recovery).

Network configuration follows Table 4 of the supplementary material:
input size 64, base channels 128, channel multipliers (1, 1, 2, 2, 4, 4),
two residual blocks per resolution, self-attention at 16x16 (local branch only).
The building blocks follow the DDPM UNet (Ho et al. 2020) as used by
WeatherDiffusion (Ozdenizci & Legenstein 2023), on which Diff-Shadow builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (Vaswani et al. 2017)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-head softmax(QK^T / sqrt(C)) V for (B, N, C) inputs.

    Uses the fused memory-efficient kernel (8% faster end to end than explicit matmuls on
    TITAN RTX). Known torch 2.10 issue: with DDP + torch.compile and very small DDP buckets
    (8 MB) inductor hands its backward a gradient layout it cannot launch ("cutlassB: no
    kernel found"); the default 25 MB buckets used here compile to a working graph.
    """
    q, k, v = (x.contiguous()[:, None] for x in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v)[:, 0]


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).transpose(-1, -2).unbind(1)
        out = attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class ReweightCrossAttention(nn.Module):
    """RCA (Eq. 4), literal reading: Z = softmax(Q K^T / sqrt(C)) (M V).

    The mask multiplies V after the softmax (no renormalisation); a masked-softmax
    reading was considered and rejected because Eq. 4 places M next to V.

    Queries come from the local patch features, keys/values from the global image
    features. ``M`` is the binary non-shadow mask of the global image; multiplying it
    into V suppresses global shadow regions so only non-shadow context flows into the
    local branch.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm_local = group_norm(channels)
        self.norm_global = group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def attend(self, local: torch.Tensor, global_feat: torch.Tensor, nonshadow: torch.Tensor) -> torch.Tensor:
        b, c, h, w = local.shape
        q = self.q(self.norm_local(local)).flatten(2).transpose(1, 2)
        gn = self.norm_global(global_feat)
        k = self.k(gn).flatten(2).transpose(1, 2)
        m = F.interpolate(nonshadow, size=gn.shape[-2:], mode="nearest")
        v = (self.v(gn) * m).flatten(2).transpose(1, 2)
        z = attention(q, k, v)
        return z.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, local: torch.Tensor, global_feat: torch.Tensor, nonshadow: torch.Tensor) -> torch.Tensor:
        return local + self.proj(self.attend(local, global_feat, nonshadow))


@dataclass
class UNetConfig:
    image_size: int = 64
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 1, 2, 2, 4, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.0
    local_in_channels: int = 7  # x_t (3) + shadow patch (3) + mask patch (1)
    global_in_channels: int = 4  # downsampled shadow image (3) + mask (1)
    out_channels: int = 3
    use_rca: bool = True
    use_mid_concat: bool = True


class _Encoder(nn.Module):
    def __init__(self, cfg: UNetConfig, in_ch: int, temb_ch: int, attention: bool):
        super().__init__()
        ch = cfg.base_channels
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.levels = nn.ModuleList()
        self.skip_channels = [ch]
        res = cfg.image_size
        cur = ch
        for i, mult in enumerate(cfg.channel_mult):
            out = ch * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(ResBlock(cur, out, temb_ch, cfg.dropout))
                attns.append(SelfAttention(out) if attention and res in cfg.attn_resolutions else nn.Identity())
                cur = out
                self.skip_channels.append(cur)
            down = None
            if i != len(cfg.channel_mult) - 1:
                down = nn.Conv2d(cur, cur, 3, stride=2, padding=1)
                self.skip_channels.append(cur)
                res //= 2
            self.levels.append(nn.ModuleDict({"blocks": blocks, "attns": attns, "down": down or nn.Identity()}))
        self.out_channels = cur
        self.num_levels = len(cfg.channel_mult)


class _Decoder(nn.Module):
    def __init__(self, cfg: UNetConfig, skip_channels: list[int], in_ch: int, temb_ch: int, attention: bool):
        super().__init__()
        ch = cfg.base_channels
        skips = list(skip_channels)
        cur = in_ch
        res = cfg.image_size // (2 ** (len(cfg.channel_mult) - 1))
        self.levels = nn.ModuleList()
        for i, mult in reversed(list(enumerate(cfg.channel_mult))):
            out = ch * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                blocks.append(ResBlock(cur + skips.pop(), out, temb_ch, cfg.dropout))
                attns.append(SelfAttention(out) if attention and res in cfg.attn_resolutions else nn.Identity())
                cur = out
            up = nn.Conv2d(cur, cur, 3, padding=1) if i != 0 else None
            if up is not None:
                res *= 2
            self.levels.append(nn.ModuleDict({"blocks": blocks, "attns": attns, "up": up or nn.Identity()}))
        self.norm_out = group_norm(cur)
        self.conv_out = nn.Conv2d(cur, cfg.out_channels, 3, padding=1)


class _Branch(nn.Module):
    """One UNet branch exposing step-wise encoder/decoder execution."""

    def __init__(self, cfg: UNetConfig, in_ch: int, attention: bool, mid_in_extra: int = 0):
        super().__init__()
        ch = cfg.base_channels
        self.temb_ch = ch * 4
        self.temb = nn.Sequential(nn.Linear(ch, self.temb_ch), nn.SiLU(), nn.Linear(self.temb_ch, self.temb_ch))
        self.enc = _Encoder(cfg, in_ch, self.temb_ch, attention)
        mid = self.enc.out_channels
        self.mid_fuse = nn.Conv2d(mid + mid_in_extra, mid, 1) if mid_in_extra else None
        self.mid1 = ResBlock(mid, mid, self.temb_ch, cfg.dropout)
        self.mid_attn = SelfAttention(mid) if attention else nn.Identity()
        self.mid2 = ResBlock(mid, mid, self.temb_ch, cfg.dropout)
        self.dec = _Decoder(cfg, self.enc.skip_channels, mid, self.temb_ch, attention)
        self.base_channels = ch

    def embed(self, t: torch.Tensor) -> torch.Tensor:
        return self.temb(timestep_embedding(t, self.base_channels))

    def encode(self, x: torch.Tensor, temb: torch.Tensor, after_first_block=None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the encoder; ``after_first_block`` may transform the first residual-block output."""
        h = self.enc.conv_in(x)
        hs = [h]
        n = self.enc.num_levels
        for i, level in enumerate(self.enc.levels):
            for j, (block, attn) in enumerate(zip(level["blocks"], level["attns"])):
                h = attn(block(h, temb))
                if i == 0 and j == 0 and after_first_block is not None:
                    h = after_first_block(h)
                hs.append(h)
            if i != n - 1:
                h = level["down"](h)
                hs.append(h)
        return h, hs

    def middle(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
    def decode(self, h: torch.Tensor, temb: torch.Tensor, hs: list[torch.Tensor]) -> torch.Tensor:
        n = len(self.dec.levels)
        for j, level in enumerate(self.dec.levels):
            for block, attn in zip(level["blocks"], level["attns"]):
                h = attn(block(torch.cat([h, hs.pop()], dim=1), temb))
            if j != n - 1:
                h = level["up"](F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.dec.conv_out(F.silu(self.dec.norm_out(h)))


@dataclass
class GlobalContext:
    """Global-branch outputs shared by every patch of one image at one timestep."""

    low_feat: torch.Tensor  # first-block features used as RCA keys/values
    mid_feat: torch.Tensor  # bottleneck features used for concatenation interaction
    nonshadow: torch.Tensor  # downsampled non-shadow mask in {0, 1}
    image: torch.Tensor  # recovered low-resolution shadow-free image in [-1, 1]

    def expand(self, index: torch.Tensor) -> "GlobalContext":
        return GlobalContext(self.low_feat[index], self.mid_feat[index], self.nonshadow[index], self.image[index])


class ParallelUNets(nn.Module):
    """Diff-Shadow parallel UNets.

    Local branch: noise estimation for R x R patches, input [x_t, shadow, mask] (7 ch).
    Global branch: recovers the downsampled shadow-free image, input [shadow', mask'] (4 ch),
    no self-attention. Interaction: RCA on the first convolutional block features and
    concatenation (+1x1 conv aggregation) on the bottleneck features.
    """

    def __init__(self, cfg: UNetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or UNetConfig()
        mid = cfg.base_channels * cfg.channel_mult[-1]
        self.local = _Branch(cfg, cfg.local_in_channels, attention=True, mid_in_extra=mid if cfg.use_mid_concat else 0)
        self.globl = _Branch(cfg, cfg.global_in_channels, attention=False)
        self.rca = ReweightCrossAttention(cfg.base_channels * cfg.channel_mult[0]) if cfg.use_rca else None

    def global_forward(self, global_in: torch.Tensor, t: torch.Tensor) -> GlobalContext:
        """Run the global UNet. ``global_in`` = cat(shadow', mask') with mask in [-1, 1]."""
        g = self.globl
        temb = g.embed(t)
        h, hs = g.encode(global_in, temb)
        low = hs[1]  # output of the first residual block
        mid_feat = g.middle(h, temb)
        image = g.decode(mid_feat, temb, hs)
        nonshadow = (global_in[:, 3:4] <= 0).float()
        return GlobalContext(low, mid_feat, nonshadow, image)

    def local_forward(self, local_in: torch.Tensor, t: torch.Tensor, ctx: GlobalContext) -> torch.Tensor:
        """Estimate the noise of local patches given the (already expanded) global context."""
        l = self.local
        temb = l.embed(t)
        rca = None
        if self.rca is not None:
            rca = lambda h: self.rca(h, ctx.low_feat, ctx.nonshadow)  # noqa: E731
        h, hs = l.encode(local_in, temb, after_first_block=rca)
        if l.mid_fuse is not None:
            h = l.mid_fuse(torch.cat([h, ctx.mid_feat], dim=1))
        return l.decode(l.middle(h, temb), temb, hs)

    def count_parameters(self) -> dict[str, int]:
        def n(m: nn.Module | None) -> int:
            return sum(p.numel() for p in m.parameters()) if m is not None else 0

        return {"local": n(self.local), "global": n(self.globl), "rca": n(self.rca), "total": n(self)}

    def forward(self, local_in: torch.Tensor, global_in: torch.Tensor, t: torch.Tensor):
        """Training forward: returns (noise estimate, recovered global image)."""
        ctx = self.global_forward(global_in, t)
        return self.local_forward(local_in, t, ctx), ctx.image
