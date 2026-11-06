"""Restore shadow-free images with the Global-guided Sampling Strategy (Algorithm 2).

Rank-sharded over GPUs:
    torchrun --nproc_per_node 8 scripts/sample.py --config configs/istd_plus.yaml \
        --ckpt outputs/istd_plus/ckpt/final.pt
Writes <out>/<stem>.png (256x256 uint8) from the EMA weights; existing files are skipped.
Each image uses a seed derived from its stem, so re-runs are deterministic.
Also runs single-process without torchrun (``--device cpu`` supported).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffshadow.checkpoint import load_ema_weights  # noqa: E402
from diffshadow.config import load_config  # noqa: E402
from diffshadow.data import DATASETS, PairedShadowDataset  # noqa: E402
from diffshadow.diffusion import linear_alphas_cumprod  # noqa: E402
from diffshadow.sampler import gss_sample  # noqa: E402
from diffshadow.unet import ParallelUNets  # noqa: E402


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", help="checkpoint path (default <run_dir>/ckpt/final.pt)")
    p.add_argument("--out", help="output dir (default <run_dir>/samples)")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--dataset", choices=sorted(DATASETS),
                   help="dataset to sample (default: the config's dataset; set it for cross-dataset evaluation)")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--weight-mode", choices=["gss", "mean", "bri"])
    p.add_argument("--crop", type=int, help="center crop size (pilot CPU check)")
    a = p.parse_args()

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        device, rank, world = torch.device(a.device), 0, 1

    cfg = load_config(a.config)
    if a.chunk_size:
        cfg.sampling.chunk_size = a.chunk_size
    if a.weight_mode:
        cfg.sampling.weight_mode = a.weight_mode
    run = Path(cfg.run_dir)
    ckpt = Path(a.ckpt) if a.ckpt else run / "ckpt" / "final.pt"
    out = Path(a.out) if a.out else run / "samples"
    out.mkdir(parents=True, exist_ok=True)

    data = PairedShadowDataset(a.dataset or cfg.dataset, a.split, size=cfg.image_size, limit=a.limit)
    torch.backends.cudnn.benchmark = True
    model = ParallelUNets(cfg.model).to(device).eval()
    state = load_ema_weights(ckpt, model, map_location=device)
    use_amp = cfg.train.amp == "fp16" and device.type == "cuda"
    if use_amp:
        model = model.to(memory_format=torch.channels_last)
    alphas = linear_alphas_cumprod(cfg.diffusion)
    if rank == 0:
        print(f"[sample] {ckpt} (step {state['global_step']}) -> {out}, {len(data)} images, {world} workers", flush=True)

    times = []
    for i in range(rank, len(data), world):
        item = data[i]
        dst = out / f"{item['name']}.png"
        if dst.exists():
            continue
        shadow, mask = item["shadow"][None], item["mask"][None]
        if a.crop:
            o = (shadow.shape[-1] - a.crop) // 2
            shadow, mask = shadow[..., o:o + a.crop, o:o + a.crop], mask[..., o:o + a.crop, o:o + a.crop]
        gen = torch.Generator(device=device).manual_seed(zlib.crc32(item["name"].encode()))
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            res = gss_sample(model, alphas, shadow.to(device), mask.to(device), cfg.sampling, generator=gen)
        res = res.float()
        times.append(time.time() - t0)
        Image.fromarray(to_uint8(res[0])).save(dst)

    stats = torch.tensor([sum(times), len(times)], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(stats)
    if rank == 0:
        n = int(stats[1].item())
        info = {"ckpt": str(ckpt), "step": state["global_step"], "images": n, "workers": world,
                "seconds_per_image_per_gpu": stats[0].item() / max(n, 1)}
        (out / "sampling_stats.json").write_text(json.dumps(info, indent=2))
        print(f"[sample] done: {info}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
