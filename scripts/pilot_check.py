"""Learning pilot check, run on CPU next to a training run (no GPU contention).

Waits for <run_dir>/ckpt/pilot_30m.pt and pilot_90m.pt (written by train.py 30 and 90 minutes
after the first launch), then writes <run_dir>/pilot/report.json:

decisive
  (b) global-branch PSNR at 64x64 on 32 fixed training images (t = 0): at 90 min it beats the
      PSNR of the downsampled shadow input, and improves from 30 to 90 min (EMA or raw weights).
  (c) GSS output from the 90-min EMA weights on one training image (128x128 centre crop) is
      finite and non-degenerate (per-channel std > 0.02, < 5% saturated pixels, differs from input).
advisory
  (a) L_diff median over the last 1000 logged steps < 0.7 x median over steps 100-1100.
  (c') GSS PSNR beats the input PSNR on that crop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffshadow.config import load_config  # noqa: E402
from diffshadow.data import PairedShadowDataset  # noqa: E402
from diffshadow.diffusion import downsample_image, downsample_mask, linear_alphas_cumprod  # noqa: E402
from diffshadow.sampler import gss_sample  # noqa: E402
from diffshadow.unet import ParallelUNets  # noqa: E402


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-image PSNR of [-1, 1] tensors mapped to [0, 1]."""
    mse = (((a.clamp(-1, 1) - b) / 2) ** 2).flatten(1).mean(1)
    return float((10 * torch.log10(1 / mse)).mean())


def wait_for(path: Path, poll_s: int = 60) -> None:
    while not path.exists():
        time.sleep(poll_s)
    time.sleep(5)  # checkpoint is written atomically, but give the filesystem a moment


@torch.no_grad()
def global_psnr(model: ParallelUNets, weights: dict, shadow, mask, clean, size: int) -> float:
    model.load_state_dict(weights)
    g_in = torch.cat([downsample_image(shadow, size), downsample_mask(mask, size) * 2 - 1], dim=1)
    ctx = model.global_forward(g_in, torch.zeros(shadow.shape[0], dtype=torch.long))
    return psnr(ctx.image, downsample_image(clean, size))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--images", type=int, default=32)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    cfg = load_config(a.config)
    run = Path(cfg.run_dir)
    out = run / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    size = cfg.model.image_size

    data = PairedShadowDataset(cfg.dataset, "train", size=cfg.image_size, limit=a.images)
    shadow = torch.stack([data[i]["shadow"] for i in range(len(data))])
    mask = torch.stack([data[i]["mask"] for i in range(len(data))])
    clean = torch.stack([data[i]["clean"] for i in range(len(data))])
    input_psnr = psnr(downsample_image(shadow, size), downsample_image(clean, size))

    model = ParallelUNets(cfg.model).eval()
    report: dict = {"input_global_psnr": input_psnr}
    for tag in ("30m", "90m"):
        ck_path = run / "ckpt" / f"pilot_{tag}.pt"
        wait_for(ck_path)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        report[f"step_{tag}"] = ck["state"]["global_step"]
        for kind in ("ema", "model"):
            report[f"global_psnr_{kind}_{tag}"] = global_psnr(model, ck[kind], shadow, mask, clean, size)
        print(f"[pilot] {tag}: {report}", flush=True)

    best90 = max(report["global_psnr_ema_90m"], report["global_psnr_model_90m"])
    best30 = max(report["global_psnr_ema_30m"], report["global_psnr_model_30m"])
    b_pass = best90 > input_psnr and best90 > best30

    # (c) GSS on one training crop with the 90-min EMA weights (model already holds raw weights; reload EMA).
    ck = torch.load(run / "ckpt" / "pilot_90m.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["ema"])
    o = (cfg.image_size - a.crop) // 2
    s_crop = shadow[:1, :, o:o + a.crop, o:o + a.crop]
    m_crop = mask[:1, :, o:o + a.crop, o:o + a.crop]
    c_crop = clean[:1, :, o:o + a.crop, o:o + a.crop]
    t0 = time.time()
    res = gss_sample(model, linear_alphas_cumprod(cfg.diffusion), s_crop, m_crop, cfg.sampling,
                     generator=torch.Generator().manual_seed(0))
    img = ((res[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(img).save(out / "gss_train0_90m.png")
    std = img.reshape(-1, 3).std(0) / 255.0
    saturated = float(((img == 0) | (img == 255)).all(-1).mean())
    finite = bool(torch.isfinite(res).all())
    differs = float((res - s_crop).abs().mean()) > 1e-3
    c_pass = finite and bool((std > 0.02).all()) and saturated < 0.05 and differs
    report.update({"gss_seconds_cpu": time.time() - t0, "gss_channel_std": std.tolist(), "gss_saturated": saturated,
                   "gss_psnr": psnr(res, c_crop), "gss_input_psnr": psnr(s_crop, c_crop)})

    # (a) loss trend from the training log
    logs = [json.loads(line) for line in open(run / "logs" / "train.jsonl")]
    early = [r["l_diff"] for r in logs if 100 <= r["step"] <= 1100]
    late = [r["l_diff"] for r in logs if r["step"] > logs[-1]["step"] - 1000]
    a_pass = bool(early and late and np.median(late) < 0.7 * np.median(early))

    report["criteria"] = {
        "b_global_psnr": {"decisive": True, "pass": b_pass},
        "c_gss_non_degenerate": {"decisive": True, "pass": c_pass},
        "a_l_diff_trend": {"decisive": False, "pass": a_pass,
                           "early_median": float(np.median(early)) if early else None,
                           "late_median": float(np.median(late)) if late else None},
        "c_gss_beats_input": {"decisive": False, "pass": report["gss_psnr"] > report["gss_input_psnr"]},
    }
    report["pass"] = b_pass and c_pass
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
