import torch

from diffshadow.diffusion import DiffusionConfig, downsample_mask, linear_alphas_cumprod, random_patches


def test_alphas_cumprod_schedule():
    a = linear_alphas_cumprod(DiffusionConfig())
    assert a.shape == (1000,)
    assert torch.isclose(a[0], torch.tensor(1 - 1e-4))
    assert (a[1:] < a[:-1]).all()
    assert a[-1] < 1e-4


def test_random_patches_aligned_across_inputs():
    clean = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32)
    shadow, mask = clean + 0.5, clean[:, :1] * 2
    g = torch.Generator().manual_seed(3)
    c, s, m, idx = random_patches(clean, shadow, mask, 8, 5, g)
    assert c.shape == (10, 3, 8, 8) and m.shape == (10, 1, 8, 8)
    assert torch.equal(s, c + 0.5)
    assert torch.equal(m, c[:, :1] * 2)
    assert idx.tolist() == [0] * 5 + [1] * 5


def test_random_patches_deterministic_with_generator():
    x = torch.randn(1, 3, 32, 32)
    a = random_patches(x, x, x[:, :1], 8, 4, torch.Generator().manual_seed(7))[0]
    b = random_patches(x, x, x[:, :1], 8, 4, torch.Generator().manual_seed(7))[0]
    assert torch.equal(a, b)


def test_downsample_mask_majority():
    m = torch.zeros(1, 1, 4, 4)
    m[..., :2, :2] = 1
    m[..., 2:, 2:3] = 1  # half of the bottom-right block -> not > 0.5
    assert downsample_mask(m, 2).flatten().tolist() == [1, 0, 0, 0]
