"""Shadow-removal metrics: Python port of ShadowFormer's ``evaluation/measure_shadow.m``.

Protocol (per image, at 256x256):
- result / ground truth: float64 in [0, 1], bicubic-resized (antialiased, unclipped, like
  MATLAB ``imresize`` on doubles) to 256x256;
- mask: binarised at the source (>127, shared rule with the data loader), bicubic-resized as
  uint8 (rounded/clipped like MATLAB) and shadow = value > 0;
- PSNR / SSIM (MATLAB volumetric convention) on full images; S / NS variants on mask-multiplied full images (MATLAB convention);
- LAB error: per-pixel absolute difference summed over L, a, b ("MAE", reported as RMSE in
  the shadow-removal literature). S and NS are accumulated over the whole test set
  (sum of errors / number of region pixels); ALL is the mean over images of sum/(H*W).
  The true per-image LAB RMSE is reported as a secondary column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab

from .masks import binarize_source_mask

SIZE = 256


def resize_float(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Bicubic resize of a float (H, W[, C]) array without clipping or rounding."""
    if img.shape[0] == size and img.shape[1] == size:
        return img.astype(np.float64)
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    out = [np.asarray(Image.fromarray(c.astype(np.float32), mode="F").resize((size, size), Image.BICUBIC)) for c in chans]
    res = out[0] if img.ndim == 2 else np.stack(out, axis=-1)
    return res.astype(np.float64)


def load_rgb01(path: str, size: int = SIZE) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return resize_float(arr, size)


def shadow_region(mask_source: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Source mask (any shape/values) -> boolean (size, size) shadow region."""
    binary = binarize_source_mask(mask_source).astype(np.float64)
    resized = np.clip(np.round(resize_float(binary, size)), 0, 255)
    return resized > 0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """MATLAB ``ssim`` on an H x W x 3 double array: volumetric Gaussian window
    (sigma 1.5, 11 taps, replicate padding) over rows, columns and channels."""
    def f(x: np.ndarray) -> np.ndarray:
        return gaussian_filter(x, 1.5, mode="nearest", truncate=5 / 1.5)

    c1, c2 = 0.01**2, 0.03**2
    ma, mb = f(a), f(b)
    saa, sbb, sab = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma * ma + mb * mb + c1) * (saa + sbb + c2))).mean())


def to_lab(rgb01: np.ndarray, illuminant: str) -> np.ndarray:
    return rgb2lab(np.clip(rgb01, 0.0, 1.0), illuminant=illuminant)


@dataclass
class ShadowMetrics:
    illuminant: str = "D65"
    psnr_all: list[float] = field(default_factory=list)
    psnr_s: list[float] = field(default_factory=list)
    psnr_ns: list[float] = field(default_factory=list)
    ssim_all: list[float] = field(default_factory=list)
    ssim_s: list[float] = field(default_factory=list)
    ssim_ns: list[float] = field(default_factory=list)
    mae_all: list[float] = field(default_factory=list)
    rmse_all: list[float] = field(default_factory=list)
    dist_s: float = 0.0
    pix_s: float = 0.0
    dist_ns: float = 0.0
    pix_ns: float = 0.0
    sq_s: float = 0.0
    sq_ns: float = 0.0

    def add(self, result: np.ndarray, gt: np.ndarray, shadow: np.ndarray) -> None:
        """result / gt: (256, 256, 3) float64 in [0, 1]; shadow: (256, 256) bool."""
        s3 = np.repeat(shadow[..., None], 3, axis=-1).astype(np.float64)
        n3 = 1.0 - s3
        self.psnr_all.append(psnr(result, gt))
        self.psnr_s.append(psnr(result * s3, gt * s3))
        self.psnr_ns.append(psnr(result * n3, gt * n3))
        self.ssim_all.append(ssim(result, gt))
        self.ssim_s.append(ssim(result * s3, gt * s3))
        self.ssim_ns.append(ssim(result * n3, gt * n3))

        diff = to_lab(gt, self.illuminant) - to_lab(result, self.illuminant)
        dist = np.abs(diff)
        h, w = shadow.shape
        self.mae_all.append(float(dist.sum()) / (h * w))
        self.rmse_all.append(float(np.sqrt((diff**2).sum(-1).mean())))
        self.dist_s += float((dist * s3).sum())
        self.dist_ns += float((dist * n3).sum())
        self.pix_s += float(shadow.sum())
        self.pix_ns += float((~shadow).sum())
        sq = (diff**2).sum(-1)
        self.sq_s += float(sq[shadow].sum())
        self.sq_ns += float(sq[~shadow].sum())

    def summary(self) -> dict:
        def mean(v: list[float]) -> float:
            arr = np.asarray(v, dtype=np.float64)
            return float(arr[np.isfinite(arr)].mean()) if np.isfinite(arr).any() else float("inf")

        return {
            "num_images": len(self.psnr_all),
            "illuminant": self.illuminant,
            "S": {
                "psnr": mean(self.psnr_s),
                "ssim": mean(self.ssim_s),
                "rmse": self.dist_s / max(self.pix_s, 1.0),
                "true_rmse": float(np.sqrt(self.sq_s / max(self.pix_s, 1.0))),
            },
            "NS": {
                "psnr": mean(self.psnr_ns),
                "ssim": mean(self.ssim_ns),
                "rmse": self.dist_ns / max(self.pix_ns, 1.0),
                "true_rmse": float(np.sqrt(self.sq_ns / max(self.pix_ns, 1.0))),
            },
            "ALL": {
                "psnr": mean(self.psnr_all),
                "ssim": mean(self.ssim_all),
                "rmse": mean(self.mae_all),
                "true_rmse": mean(self.rmse_all),
            },
        }


def markdown_table(summary: dict, title: str) -> str:
    rows = [f"### {title} ({summary['num_images']} images, LAB {summary['illuminant']})", "",
            "| Region | PSNR | SSIM | RMSE (LAB MAE) | true LAB RMSE |", "|---|---|---|---|---|"]
    for k in ("S", "NS", "ALL"):
        r = summary[k]
        rows.append(f"| {k} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['rmse']:.2f} | {r['true_rmse']:.2f} |")
    return "\n".join(rows) + "\n"
