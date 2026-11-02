"""Single mask binarization rule shared by the data loader and the evaluator.

Dataset masks store shadow as 255; SRD masks are JPEGs with compression noise, so
the source mask is thresholded at >127 before any resizing.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def binarize_source_mask(arr: np.ndarray) -> np.ndarray:
    """(H, W) or (H, W, C) uint8 mask -> (H, W) uint8 in {0, 255} (255 = shadow)."""
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.where(arr > 127, 255, 0).astype(np.uint8)


def load_binary_mask(path: str) -> np.ndarray:
    return binarize_source_mask(np.asarray(Image.open(path)))
