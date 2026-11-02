"""DDPM noise schedule and the Diff-Shadow training objective (Eq. 9, 10, Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .unet import ParallelUNets


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000  # T (Table 5)
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sampling_timesteps: int = 25  # S (Table 5)


def linear_alphas_cumprod(cfg: DiffusionConfig) -> torch.Tensor:
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_timesteps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0).float()


def downsample_image(x: torch.Tensor, size: int) -> torch.Tensor:
    """Full image -> low-resolution global input (area averaging)."""
    return F.interpolate(x, size=(size, size), mode="area")


def downsample_mask(m: torch.Tensor, size: int) -> torch.Tensor:
    """Binary mask downsampling: area average then majority threshold."""
    return (F.interpolate(m, size=(size, size), mode="area") > 0.5).float()


def random_patches(
    clean: torch.Tensor, shadow: torch.Tensor, mask: torch.Tensor, patch: int, n: int, generator: torch.Generator | None = None
):
    """Randomly crop ``n`` patches (same locations across inputs) from every image.

    Returns tensors of shape (B * n, C, patch, patch) and the image index of every patch.
    """
    b, _, h, w = clean.shape
    dev = clean.device
    ys = torch.randint(0, h - patch + 1, (b, n), generator=generator, device="cpu").to(dev)
    xs = torch.randint(0, w - patch + 1, (b, n), generator=generator, device="cpu").to(dev)
    ar = torch.arange(patch, device=dev)
    rows = (ys[..., None] + ar).reshape(b * n, patch)  # (bn, p)
    cols = (xs[..., None] + ar).reshape(b * n, patch)
    img_idx = torch.arange(b, device=dev).repeat_interleave(n)

    def crop(x: torch.Tensor) -> torch.Tensor:
        xi = x[img_idx]  # (bn, c, h, w)
        xi = torch.gather(xi, 2, rows[:, None, :, None].expand(-1, x.shape[1], -1, w))
        return torch.gather(xi, 3, cols[:, None, None, :].expand(-1, x.shape[1], patch, -1))

    return crop(clean), crop(shadow), crop(mask), img_idx


def training_losses(
    model: ParallelUNets,
    alphas_cumprod: torch.Tensor,
    clean: torch.Tensor,
    shadow: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    patches_per_image: int,
    global_weight: float = 1.0,
    cpu_generator: torch.Generator | None = None,
    device_generator: torch.Generator | None = None,
):
    """Algorithm 1. Images in [-1, 1], mask in {0, 1} (1 = shadow); all of shape (B, C, H, W).

    ``cpu_generator`` drives crop locations, ``device_generator`` drives t and noise.
    Returns (total, L_diff, L_global).
    """
    x0, sp, mp, idx = random_patches(clean, shadow, mask, patch_size, patches_per_image, cpu_generator)
    g_shadow = downsample_image(shadow, patch_size)
    g_mask = downsample_mask(mask, patch_size)
    g_clean = downsample_image(clean, patch_size)
    global_in = torch.cat([g_shadow, g_mask * 2 - 1], dim=1)[idx]

    n = x0.shape[0]
    t = torch.randint(0, alphas_cumprod.numel(), (n,), device=x0.device, generator=device_generator)
    a = alphas_cumprod.to(x0.device)[t].view(-1, 1, 1, 1)
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=device_generator)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
    local_in = torch.cat([xt, sp, mp * 2 - 1], dim=1)

    eps, g_out = model(local_in, global_in, t)
    l_diff = F.mse_loss(eps, noise)
    l_global = F.mse_loss(g_out, g_clean[idx])
    return l_diff + global_weight * l_global, l_diff, l_global
