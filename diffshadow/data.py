"""Paired shadow-removal datasets (ISTD+ as ``aistd``, SRD; WSRD test set for cross-dataset evaluation).

Folder layout: ``<root>/<split>/<split>_{A,B,C}`` = shadow / mask / shadow-free,
matched by file stem. Images are resized to 256x256 (bicubic for RGB, nearest for the
already-binarised mask) and preloaded as uint8.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .masks import load_binary_mask

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DATASETS = {
    "istd_plus": {"root": "~/Dataset/Shadow/aistd", "train": 1330, "test": 540},
    "srd": {"root": "~/Dataset/Shadow/SRD", "train": 2680, "test": 408},
    # Local WSRD copy: test split only, already 256x256 with binary masks (evaluation only).
    "wsrd": {"root": "~/Dataset/Shadow/WSRD", "test": 100},
}


def _index(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")}


def list_triplets(dataset: str, split: str, root: str | None = None) -> list[tuple[str, Path, Path, Path]]:
    """Return sorted (stem, shadow, mask, free) paths; asserts the expected image count."""
    info = DATASETS[dataset]
    if split not in info:
        raise ValueError(f"{dataset} has no {split} split")
    base = Path(os.path.expanduser(root or info["root"])) / split
    a, b, c = (_index(base / f"{split}_{k}") for k in "ABC")
    stems = sorted(set(a) & set(b) & set(c))
    expected = info[split]
    if len(stems) != expected:
        raise ValueError(f"{dataset}/{split}: found {len(stems)} complete triplets, expected {expected}")
    return [(s, a[s], b[s], c[s]) for s in stems]


def load_rgb(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def load_mask(path: Path, size: int) -> np.ndarray:
    m = Image.fromarray(load_binary_mask(str(path)))
    if m.size != (size, size):
        m = m.resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def to_tensor_rgb(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0


class PairedShadowDataset(Dataset):
    """Returns dict(shadow [-1,1] 3xSxS, mask {0,1} 1xSxS, clean [-1,1] 3xSxS, name)."""

    def __init__(self, dataset: str, split: str, size: int = 256, root: str | None = None, limit: int | None = None):
        triplets = list_triplets(dataset, split, root)
        if limit is not None:
            triplets = triplets[:limit]
        self.names = [t[0] for t in triplets]
        self.shadow = np.stack([load_rgb(t[1], size) for t in triplets])
        self.mask = np.stack([load_mask(t[2], size) for t in triplets])
        self.clean = np.stack([load_rgb(t[3], size) for t in triplets])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> dict:
        return {
            "shadow": to_tensor_rgb(self.shadow[i]),
            "mask": torch.from_numpy(self.mask[i])[None].float(),
            "clean": to_tensor_rgb(self.clean[i]),
            "name": self.names[i],
        }
