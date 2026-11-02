import itertools

import torch

from diffshadow.budget import Deadline, ensure_anchor, main as budget_main, read_anchor
from diffshadow.checkpoint import load_checkpoint, save_checkpoint
from diffshadow.diffusion import random_patches
from diffshadow.ema import EMA
from diffshadow.seeding import make_generators, rank_seed


class FakeClock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_deadline_is_real_wall_clock():
    clock = FakeClock(1000.0)
    d = Deadline(1000.0, 48, clock)
    clock.now = 1000.0 + 48 * 3600 - 1
    assert not d.expired()
    clock.now = 1000.0 + 48 * 3600
    assert d.expired()


def test_anchor_never_overwritten(tmp_path):
    clock = FakeClock(10.0)
    assert ensure_anchor(tmp_path, clock) == 10.0
    clock.now = 99.0
    assert ensure_anchor(tmp_path, clock) == 10.0
    assert read_anchor(tmp_path) == 10.0


def test_budget_cli_exit_codes(tmp_path):
    (tmp_path / "first_launch_utc").write_text("0\n")  # epoch start -> long expired
    assert budget_main(["--run", str(tmp_path), "--hours", "48"]) == 3
    run2 = tmp_path / "fresh"
    assert budget_main(["--run", str(run2), "--hours", "48", "--init"]) == 0


def test_rank_seed_formula_and_uniqueness():
    assert rank_seed(0, 3, 5) == 3 * 10_007 + 5
    assert rank_seed(2, 0, 0) == 2_000_006
    seeds = {rank_seed(0, r, s) for r, s in itertools.product(range(8), (0, 1, 10**6))}
    assert len(seeds) == 24


def test_generators_reproducible_and_rank_distinct():
    x = torch.randn(1, 3, 64, 64)
    crops = lambda g: random_patches(x, x, x[:, :1], 16, 4, g)[0]  # noqa: E731
    a, _ = make_generators(0, 1, 50, "cpu")
    b, _ = make_generators(0, 1, 50, "cpu")
    c, _ = make_generators(0, 2, 50, "cpu")
    assert torch.equal(crops(a), crops(b))
    assert not torch.equal(crops(make_generators(0, 1, 50, "cpu")[0]), crops(c))


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, 0.9)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    ema.update(model)
    state = {"global_step": 123, "train_seconds": 4.5, "restart_count": 2}
    save_checkpoint(tmp_path / "c.pt", model, ema, opt, scaler, state)

    model2 = torch.nn.Linear(4, 4)
    ema2 = EMA(model2, 0.9)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    got = load_checkpoint(tmp_path / "c.pt", model2, ema2, opt2, torch.amp.GradScaler("cuda", enabled=False))
    assert got == state
    assert torch.equal(model2.weight, model.weight)
    assert torch.equal(ema2.model.weight, ema.model.weight)
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]


def test_ema_update_rule():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.0)
    ema = EMA(model, 0.999)
    torch.nn.init.constant_(model.weight, 1.0)
    ema.update(model)
    assert torch.isclose(ema.model.weight, torch.tensor([[0.001]]))
