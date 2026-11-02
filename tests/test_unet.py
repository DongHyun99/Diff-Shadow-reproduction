import torch

from conftest import tiny_config
from diffshadow.unet import ParallelUNets, ReweightCrossAttention, UNetConfig


def test_output_shapes(tiny_model):
    n = 4
    local_in = torch.randn(n, 7, 16, 16)
    global_in = torch.randn(n, 4, 16, 16)
    eps, g = tiny_model(local_in, global_in, torch.randint(0, 1000, (n,)))
    assert eps.shape == (n, 3, 16, 16)
    assert g.shape == (n, 3, 16, 16)


def test_table4_shapes_and_param_count():
    cfg = UNetConfig()
    assert cfg.local_in_channels + cfg.global_in_channels == 11
    model = ParallelUNets(cfg)
    counts = model.count_parameters()
    assert counts["local"] > 0 and counts["global"] > 0 and counts["rca"] > 0
    assert 150e6 < counts["total"] < 260e6  # measured 217.7M; paper reports 160M (documented gap)
    with torch.no_grad():
        eps, g = model(torch.randn(1, 7, 64, 64), torch.randn(1, 4, 64, 64), torch.tensor([500]))
    assert eps.shape == g.shape == (1, 3, 64, 64)


def _reference_rca(rca: ReweightCrossAttention, local, global_feat, nonshadow):
    """Eq. 4 in float64: Z = softmax(Q K^T / sqrt(C)) (M V)."""
    b, c, h, w = local.shape
    q = rca.q(rca.norm_local(local)).flatten(2).transpose(1, 2).double()
    gn = rca.norm_global(global_feat)
    k = rca.k(gn).flatten(2).transpose(1, 2).double()
    m = torch.nn.functional.interpolate(nonshadow, size=gn.shape[-2:], mode="nearest").flatten(2).transpose(1, 2).double()
    v = rca.v(gn).flatten(2).transpose(1, 2).double()
    att = torch.softmax(q @ k.transpose(1, 2) / c**0.5, dim=-1)
    return (att @ (m * v)).transpose(1, 2).reshape(b, c, h, w)


def _rca_inputs(c=32, s=8):
    torch.manual_seed(1)
    rca = ReweightCrossAttention(c).eval()
    local, glob = torch.randn(2, c, s, s), torch.randn(2, c, s, s)
    nonshadow = (torch.rand(2, 1, s, s) > 0.4).float()
    return rca, local, glob, nonshadow


def test_rca_matches_eq4_reference():
    rca, local, glob, ns = _rca_inputs()
    with torch.no_grad():
        z = rca.attend(local, glob, ns).double()
        ref = _reference_rca(rca, local, glob, ns)
    assert torch.allclose(z, ref, atol=1e-5)


def test_rca_value_path_ignores_shadow_tokens():
    rca, local, glob, ns = _rca_inputs()
    shadow_tok = ns == 0
    with torch.no_grad():
        base = rca.attend(local, glob, ns)

        def perturb(where):
            def hook(_m, _i, out):
                return out + 10.0 * where.float()
            h = rca.v.register_forward_hook(hook)
            try:
                return rca.attend(local, glob, ns)
            finally:
                h.remove()

        assert torch.allclose(perturb(shadow_tok), base, atol=1e-6)
        assert not torch.allclose(perturb(~shadow_tok), base, atol=1e-3)


def test_rca_all_shadow_gives_zero_context():
    rca, local, glob, _ = _rca_inputs()
    with torch.no_grad():
        z = rca.attend(local, glob, torch.zeros(2, 1, 8, 8))
    assert torch.count_nonzero(z) == 0


def test_branch_inputs_and_attention():
    from diffshadow.unet import SelfAttention

    model = ParallelUNets(tiny_config())
    first_conv = lambda m: next(x for x in m.modules() if isinstance(x, torch.nn.Conv2d))  # noqa: E731
    assert first_conv(model.local).in_channels == 7
    assert first_conv(model.globl).in_channels == 4
    assert any(isinstance(x, SelfAttention) for x in model.local.modules())
    assert not any(isinstance(x, SelfAttention) for x in model.globl.modules())


def test_gradients_reach_both_branches_rca_and_fusion():
    from diffshadow.diffusion import DiffusionConfig, linear_alphas_cumprod, training_losses

    torch.manual_seed(0)
    model = ParallelUNets(tiny_config())
    img = torch.rand(2, 3, 32, 32) * 2 - 1
    mask = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss, l_diff, l_global = training_losses(model, linear_alphas_cumprod(DiffusionConfig()), img, img.flip(-1), mask, 16, 4)
    assert torch.isfinite(loss) and torch.isclose(loss, l_diff + l_global)
    loss.backward()
    for name in ("local", "globl", "rca"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert all(g is not None for g in grads), name
        assert sum(g.abs().sum() for g in grads) > 0, name
