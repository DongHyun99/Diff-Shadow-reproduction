"""Exponential moving average of model parameters (Nichol & Dhariwal 2021)."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd)
