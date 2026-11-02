"""Global-guided Sampling Strategy (GSS), Algorithm 2 and Eq. 5-8.

The image is parsed into overlapping R x R patches with step r. At every implicit
(DDIM, eta = 0) sampling step the local branch estimates the noise of every patch;
per-patch weights W_d combine the brightness score S_bri (agreement between the
patch's predicted clean image and the global branch's recovered image) and the
mask score S_m (non-shadow ratio of the patch):

    W_d = (t * S_bri + (T - t) * S_m) / T

and the full-image noise is the W_d-weighted average over overlapping patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .diffusion import downsample_image, downsample_mask
from .unet import ParallelUNets

WeightMode = Literal["gss", "mean", "bri"]


@dataclass
class GSSConfig:
    patch_size: int = 64  # R
    stride: int = 8  # r
    num_timesteps: int = 1000  # T
    sampling_timesteps: int = 25  # S
    t_sf: float = 0.5  # T_sf, non-shadow ratio threshold (Eq. 6)
    eps: float = 0.01  # numerical epsilon of Eq. 5
    weight_mode: WeightMode = "gss"  # "gss" = S_bri + S_m, "mean" = W_d = 1, "bri" = S_bri only
    chunk_size: int = 128  # patches per forward pass


def to_gray(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] RGB -> [0, 1] luminance."""
    x = (x.clamp(-1, 1) + 1) / 2
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def brightness_score(local_gray: torch.Tensor, global_gray: torch.Tensor, eps: float) -> torch.Tensor:
    """Eq. 5: S_bri = 1 - avg(|L - G|) / max(|L - G|), per patch."""
    d = (local_gray - global_gray).abs().flatten(1)
    return 1 - d.mean(1) / (d.amax(1) + eps)


def mask_score(mask_patch: torch.Tensor, t_sf: float) -> torch.Tensor:
    """Eq. 6: S_m = 1 if R_sf >= T_sf else R_sf / T_sf, R_sf = non-shadow ratio."""
    r_sf = (1 - mask_patch).flatten(1).mean(1)
    return torch.where(r_sf >= t_sf, torch.ones_like(r_sf), r_sf / t_sf)


def patch_weights(
    mode: WeightMode, t: int, T: int, local_gray: torch.Tensor, global_gray: torch.Tensor, mask_patch: torch.Tensor, t_sf: float, eps: float
) -> torch.Tensor:
    if mode == "mean":
        return torch.ones(local_gray.shape[0], device=local_gray.device)
    s_bri = brightness_score(local_gray, global_gray, eps)
    if mode == "bri":
        return s_bri
    s_m = mask_score(mask_patch, t_sf)
    return (t * s_bri + (T - t) * s_m) / T  # Eq. 7


def padded_size(size: int, patch: int, stride: int) -> int:
    """Smallest size >= ``size`` whose patch grid with ``stride`` ends exactly at the border."""
    size = max(size, patch)
    return patch + -(-(size - patch) // stride) * stride


def ddim_timesteps(T: int, S: int) -> tuple[list[int], list[int]]:
    """Algorithm 2, lines 3-4 (0-indexed): uniform skip, next of the last step is -1 (x_0)."""
    if T % S != 0:
        raise ValueError(f"T={T} must be divisible by S={S}")
    skip = T // S
    seq = list(range(0, T, skip))
    return seq, [-1] + seq[:-1]


@torch.no_grad()
def gss_sample(
    model: ParallelUNets,
    alphas_cumprod: torch.Tensor,
    shadow: torch.Tensor,
    mask: torch.Tensor,
    cfg: GSSConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Restore shadow-free images.

    Args:
        shadow: (B, 3, H, W) in [-1, 1]; mask: (B, 1, H, W) in {0, 1} with 1 = shadow.
    Returns:
        (B, 3, H, W) shadow-free estimate in [-1, 1].
    """
    b, _, h, w = shadow.shape
    dev = shadow.device
    R, r, T = cfg.patch_size, cfg.stride, cfg.num_timesteps
    ph, pw = padded_size(h, R, r), padded_size(w, R, r)
    pad = (0, pw - w, 0, ph - h)
    shadow_p = F.pad(shadow, pad, mode="replicate")
    mask_p = F.pad(mask, pad, mode="replicate")

    def unfold(x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) -> (B * D, C, R, R)
        c = x.shape[1]
        cols = F.unfold(x, R, stride=r)  # (B, C*R*R, D)
        d = cols.shape[-1]
        return cols.transpose(1, 2).reshape(b * d, c, R, R)

    def fold(p: torch.Tensor, c: int) -> torch.Tensor:  # (B * D, C, R, R) -> (B, C, H, W) summed
        cols = p.reshape(b, -1, c * R * R).transpose(1, 2)
        return F.fold(cols, (ph, pw), R, stride=r)

    s_patches = unfold(shadow_p)
    m_patches = unfold(mask_p)
    d = s_patches.shape[0] // b
    img_idx = torch.arange(b, device=dev).repeat_interleave(d)
    local_cond = torch.cat([s_patches, m_patches * 2 - 1], dim=1)

    # The global branch sees the whole (unpadded) image downsampled to the patch size.
    global_in = torch.cat([downsample_image(shadow, R), downsample_mask(mask, R) * 2 - 1], dim=1)

    a_all = alphas_cumprod.to(dev)
    x = torch.randn(b, 3, ph, pw, device=dev, generator=generator)
    seq, seq_next = ddim_timesteps(T, cfg.sampling_timesteps)
    for i, j in zip(reversed(seq), reversed(seq_next)):
        a_t = a_all[i]
        a_next = a_all[j] if j >= 0 else torch.ones((), device=dev)
        t_img = torch.full((b,), i, device=dev, dtype=torch.long)
        ctx = model.global_forward(global_in, t_img)
        g_full = F.interpolate(ctx.image, size=(h, w), mode="bilinear", align_corners=False)
        g_patches = unfold(F.pad(g_full, pad, mode="replicate"))

        x_patches = unfold(x)
        eps = torch.empty_like(x_patches)
        weights = torch.empty(x_patches.shape[0], device=dev)
        for s in range(0, x_patches.shape[0], cfg.chunk_size):
            sl = slice(s, s + cfg.chunk_size)
            xt = x_patches[sl]
            idx = img_idx[sl]
            t = torch.full((xt.shape[0],), i, device=dev, dtype=torch.long)
            e = model.local_forward(torch.cat([xt, local_cond[sl]], dim=1), t, ctx.expand(idx))
            eps[sl] = e
            x0_local = (xt - (1 - a_t).sqrt() * e) / a_t.sqrt()
            weights[sl] = patch_weights(
                cfg.weight_mode, i + 1, T, to_gray(x0_local), to_gray(g_patches[sl]), m_patches[sl], cfg.t_sf, cfg.eps
            )
        wv = weights.view(-1, 1, 1, 1)
        omega = fold(eps * wv, 3) / fold(wv.expand(-1, 1, R, R).contiguous(), 1).clamp_min(1e-8)  # Eq. 8
        x0 = (x - (1 - a_t).sqrt() * omega) / a_t.sqrt()
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * omega  # Algorithm 2, line 14
    return x[:, :, :h, :w].clamp(-1, 1)
