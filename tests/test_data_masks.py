import io
import os

import numpy as np
import pytest
from PIL import Image

from diffshadow.data import DATASETS, list_triplets, load_mask
from diffshadow.masks import binarize_source_mask
from diffshadow.metrics import shadow_region


def _jpeg_noised(mask: np.ndarray) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="JPEG", quality=75)
    return np.asarray(Image.open(buf))


def _blob_mask(size=128):
    yy, xx = np.mgrid[:size, :size]
    return np.where((yy - 60) ** 2 + (xx - 70) ** 2 < 35**2, 255, 0).astype(np.uint8)


def test_binarize_recovers_jpeg_noised_mask():
    clean = _blob_mask()
    noisy = _jpeg_noised(clean)
    assert ((noisy > 0) != (clean > 0)).mean() > 0.005  # JPEG noise really does add stray non-zero pixels
    recovered = binarize_source_mask(noisy)
    assert set(np.unique(recovered)) <= {0, 255}
    assert (recovered != clean).mean() <= 0.005


def test_evaluator_region_uses_binarised_mask():
    clean = _blob_mask(256)
    noisy = _jpeg_noised(clean)
    region = shadow_region(noisy)
    reference = shadow_region(clean)
    assert abs(int(region.sum()) - int(reference.sum())) <= 0.005 * reference.size
    naive = noisy > 0  # without binarisation compression noise counts as shadow
    assert naive.sum() > region.sum()


def test_load_mask_three_channel_and_resize(tmp_path):
    m = np.stack([_blob_mask()] * 3, axis=-1)
    p = tmp_path / "m.png"
    Image.fromarray(m).save(p)
    out = load_mask(p, 64)
    assert out.shape == (64, 64) and set(np.unique(out)) <= {0, 1}


def test_list_triplets_matches_stems_and_ignores_junk(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    for k in "ABC":
        d = root / "test" / f"test_{k}"
        d.mkdir(parents=True)
        for stem in ("a", "b"):
            Image.new("RGB", (4, 4)).save(d / f"{stem}.png")
        (d / ".DS_Store").write_bytes(b"x")
    (root / "test" / "test_C" / "c.png").write_bytes(b"")  # no matching A/B
    monkeypatch.setitem(DATASETS, "fake", {"root": str(root), "train": 0, "test": 2})
    assert [t[0] for t in list_triplets("fake", "test")] == ["a", "b"]
    monkeypatch.setitem(DATASETS, "fake", {"root": str(root), "train": 0, "test": 3})
    with pytest.raises(ValueError):
        list_triplets("fake", "test")


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/Dataset/Shadow/SRD")), reason="datasets not present")
@pytest.mark.parametrize("dataset", ["istd_plus", "srd"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_real_dataset_counts(dataset, split):
    assert len(list_triplets(dataset, split)) == DATASETS[dataset][split]
