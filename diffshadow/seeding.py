"""Per-rank deterministic random streams (so DDP ranks never share t/noise/crop draws)."""

from __future__ import annotations

import torch


def rank_seed(base_seed: int, rank: int, global_step: int) -> int:
    return base_seed * 1_000_003 + rank * 10_007 + global_step


def make_generators(base_seed: int, rank: int, global_step: int, device: torch.device | str) -> tuple[torch.Generator, torch.Generator]:
    """Return (cpu_generator, device_generator) seeded from (base_seed, rank, global_step)."""
    s = rank_seed(base_seed, rank, global_step)
    cpu = torch.Generator().manual_seed(s)
    dev = torch.Generator(device=device).manual_seed(s + 1)
    return cpu, dev
