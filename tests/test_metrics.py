import numpy as np
import pytest

from diffshadow.metrics import ShadowMetrics, psnr, resize_float, ssim, to_lab


def test_identical_images():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3))
    assert psnr(img, img) == float("inf")
    assert ssim(img, img) == pytest.approx(1.0)
    m = ShadowMetrics()
    shadow = np.zeros((32, 32), bool)
    shadow[:8] = True
    m.add(img, img, shadow)
    s = m.summary()
    assert s["ALL"]["rmse"] == 0 and s["S"]["rmse"] == 0 and s["NS"]["rmse"] == 0


def test_psnr_value():
    a = np.zeros((4, 4, 3))
    assert psnr(a, a + 0.1) == pytest.approx(20.0)


def test_lab_error_aggregation_per_pixel_vs_per_image():
    """S/NS accumulate over all pixels of the set; ALL averages per-image means (measure_shadow.m)."""
    m = ShadowMetrics()
    gt = np.full((4, 4, 3), 0.5)
    res1 = gt.copy()
    res1[0, 0] = 0.6  # error only on one shadow pixel
    s1 = np.zeros((4, 4), bool)
    s1[0, :] = True  # 4 shadow pixels
    s2 = np.zeros((4, 4), bool)
    s2[:, :] = True  # 16 shadow pixels, all errors
    res2 = np.full((4, 4, 3), 0.6)
    m.add(res1, gt, s1)
    m.add(res2, gt, s2)
    d = np.abs(to_lab(np.full((1, 1, 3), 0.6), "D65") - to_lab(np.full((1, 1, 3), 0.5), "D65")).sum()
    out = m.summary()
    assert out["S"]["rmse"] == pytest.approx(17 * d / 20)  # 17 erroneous pixels over 20 shadow pixels
    assert out["ALL"]["rmse"] == pytest.approx((d / 16 + d) / 2)  # mean of per-image means
    assert out["NS"]["rmse"] == pytest.approx(0.0)


def test_resize_float_identity_and_shape():
    x = np.random.default_rng(1).random((300, 200, 3))
    y = resize_float(x, 256)
    assert y.shape == (256, 256, 3) and y.dtype == np.float64
    z = np.random.default_rng(2).random((256, 256, 3))
    assert np.array_equal(resize_float(z), z)
