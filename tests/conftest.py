import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffshadow.unet import ParallelUNets, UNetConfig  # noqa: E402


def tiny_config(**kw) -> UNetConfig:
    base = dict(image_size=16, base_channels=32, channel_mult=(1, 2), num_res_blocks=1, attn_resolutions=(8,))
    base.update(kw)
    return UNetConfig(**base)


@pytest.fixture
def tiny_model() -> ParallelUNets:
    torch.manual_seed(0)
    return ParallelUNets(tiny_config()).eval()
