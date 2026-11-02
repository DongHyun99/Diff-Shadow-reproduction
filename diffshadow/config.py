"""YAML experiment configuration -> typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .diffusion import DiffusionConfig
from .sampler import GSSConfig
from .unet import UNetConfig


@dataclass
class TrainConfig:
    images_per_gpu: int = 1  # 8 GPUs x 1 image = 8 images per iteration (paper)
    patches_per_image: int = 16
    lr: float = 2e-4
    ema_decay: float = 0.999
    global_loss_weight: float = 1.0  # lambda
    epochs: int = 2000
    budget_hours: float = 48.0  # strict real wall-clock since first launch
    create_anchor: bool = False  # smoke/benchmark runs create their own clock anchor
    max_steps: int | None = None  # smoke/benchmark only
    ckpt_minutes: float = 30.0
    milestone_epochs: int = 500
    pilot_minutes: tuple[float, ...] = (30.0, 90.0)
    log_every: int = 50
    seed: int = 0
    amp: str = "off"  # "off" (fp32) or "fp16"
    ddp_fp16_grads: bool = False  # all-reduce gradients in fp16 (communication only)


@dataclass
class Config:
    name: str
    dataset: str
    run_dir: str
    image_size: int = 256
    model: UNetConfig = field(default_factory=UNetConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: GSSConfig = field(default_factory=GSSConfig)


def _build(cls, data: dict | None):
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    for k, v in data.items():
        if isinstance(v, list):
            data[k] = tuple(v)
    return cls(**data)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        name=raw["name"],
        dataset=raw["dataset"],
        run_dir=raw["run_dir"],
        image_size=raw.get("image_size", 256),
        model=_build(UNetConfig, raw.get("model")),
        diffusion=_build(DiffusionConfig, raw.get("diffusion")),
        train=_build(TrainConfig, raw.get("train")),
        sampling=_build(GSSConfig, raw.get("sampling")),
    )
