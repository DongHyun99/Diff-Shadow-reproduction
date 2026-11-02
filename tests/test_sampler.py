import pytest
import torch

import diffshadow.sampler as sampler
from diffshadow.diffusion import DiffusionConfig, linear_alphas_cumprod
from diffshadow.sampler import (GSSConfig, brightness_score, ddim_timesteps, gss_sample, mask_score, padded_size,
                                patch_weights)


def test_ddim_timesteps_exact():
    seq, nxt = ddim_timesteps(1000, 25)
    assert seq == list(range(0, 1000, 40))
    assert nxt == [-1] + list(range(0, 960, 40))
    with pytest.raises(ValueError):
        ddim_timesteps(1000, 30)


def test_brightness_score_eq5():
    local = torch.zeros(1, 1, 2, 2)
    glob = torch.tensor([[[[0.0, 0.1], [0.1, 0.4]]]])
    # avg |L-G| = 0.15, max = 0.4 -> 1 - 0.15 / (0.4 + eps)
    assert torch.isclose(brightness_score(local, glob, 0.01), torch.tensor([1 - 0.15 / 0.41]))
    assert torch.isclose(brightness_score(local, local, 0.01), torch.tensor([1.0]))


def test_mask_score_eq6():
    def m(shadow_fraction):
        x = torch.zeros(1, 1, 1, 100)
        x[..., : int(shadow_fraction * 100)] = 1
        return x
    assert mask_score(m(0.5), 0.5).item() == 1.0  # R_sf = T_sf boundary -> 1
    assert mask_score(m(0.2), 0.5).item() == 1.0
    assert torch.isclose(mask_score(m(0.8), 0.5), torch.tensor([0.2 / 0.5]))
    assert mask_score(m(1.0), 0.5).item() == 0.0


def test_patch_weights_eq7_interpolates_between_scores():
    lg = torch.zeros(1, 1, 2, 2)
    gg = torch.tensor([[[[0.0, 0.1], [0.1, 0.4]]]])
    mp = torch.ones(1, 1, 2, 2)  # all shadow -> S_m = 0
    s_bri = brightness_score(lg, gg, 0.01)
    assert torch.isclose(patch_weights("gss", 1000, 1000, lg, gg, mp, 0.5, 0.01), s_bri)  # t = T: brightness only
    assert patch_weights("gss", 0, 1000, lg, gg, mp, 0.5, 0.01).item() == 0.0  # t = 0: mask only
    assert torch.isclose(patch_weights("gss", 500, 1000, lg, gg, mp, 0.5, 0.01), s_bri / 2)
    assert patch_weights("mean", 500, 1000, lg, gg, mp, 0.5, 0.01).item() == 1.0
    assert torch.isclose(patch_weights("bri", 10, 1000, lg, gg, mp, 0.5, 0.01), s_bri)


@pytest.mark.parametrize("size", [64, 65, 250, 256, 300])
def test_patch_grid_covers_image(size):
    p = padded_size(size, 64, 8)
    assert p >= size and (p - 64) % 8 == 0 and p - size < 8
    cover = torch.nn.functional.fold(torch.ones(1, 64 * 64, ((p - 64) // 8 + 1) ** 2), (p, p), 64, stride=8)
    assert cover.min() >= 1


class _ConstModel(torch.nn.Module):
    """Stub: constant noise estimate, identity global image; records timesteps passed to weights."""

    def __init__(self, value=0.1):
        super().__init__()
        self.value = value

    def global_forward(self, global_in, t):
        class Ctx:
            image = global_in[:, :3]

            def expand(self, idx):
                return self
        return Ctx()

    def local_forward(self, local_in, t, ctx):
        return torch.full_like(local_in[:, :3], self.value)


def _run(monkeypatch, mask_value, mode="gss"):
    calls = []
    real = sampler.patch_weights

    def spy(mode_, t, T, *a):
        calls.append(t)
        return real(mode_, t, T, *a)

    monkeypatch.setattr(sampler, "patch_weights", spy)
    cfg = GSSConfig(patch_size=16, stride=8, num_timesteps=1000, sampling_timesteps=25, weight_mode=mode, chunk_size=7)
    shadow = torch.rand(1, 3, 40, 40) * 2 - 1
    mask = torch.full((1, 1, 40, 40), mask_value)
    out = gss_sample(_ConstModel(), linear_alphas_cumprod(DiffusionConfig()), shadow, mask, cfg,
                     generator=torch.Generator().manual_seed(0))
    return out, calls


def test_gss_all_shadow_is_finite_and_uses_one_indexed_t(monkeypatch):
    out, calls = _run(monkeypatch, 1.0)
    assert out.shape == (1, 3, 40, 40) and torch.isfinite(out).all()
    ts = sorted(set(calls), reverse=True)
    assert ts[0] == 961 and ts[-1] == 1 and len(ts) == 25


def test_gss_mean_mode_with_constant_eps_equals_plain_ddim(monkeypatch):
    out, _ = _run(monkeypatch, 0.0, mode="mean")
    a = linear_alphas_cumprod(DiffusionConfig())
    x = torch.randn(1, 3, 40, 40, generator=torch.Generator().manual_seed(0))
    seq, nxt = ddim_timesteps(1000, 25)
    for i, j in zip(reversed(seq), reversed(nxt)):
        a_t, a_n = a[i], (a[j] if j >= 0 else torch.tensor(1.0))
        x0 = (x - (1 - a_t).sqrt() * 0.1) / a_t.sqrt()
        x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * 0.1
    assert torch.allclose(out, x.clamp(-1, 1), atol=1e-4)
