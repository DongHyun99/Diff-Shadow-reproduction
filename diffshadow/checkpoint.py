"""Atomic training checkpoints (model, EMA, optimizer, AMP scaler, progress counters)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from .ema import EMA


def save_checkpoint(path: str | Path, model: nn.Module, ema: EMA, opt: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, state: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(), "state": state}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, model: nn.Module, ema: EMA, opt: torch.optim.Optimizer | None = None,
                    scaler: torch.amp.GradScaler | None = None, map_location=None) -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    ema.load_state_dict(ck["ema"])
    if opt is not None:
        opt.load_state_dict(ck["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(ck["scaler"])
    return dict(ck["state"])


def load_ema_weights(path: str | Path, model: nn.Module, map_location=None) -> dict:
    """Load the EMA weights into ``model`` (used for sampling); returns the stored state."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["ema"])
    return dict(ck["state"])
