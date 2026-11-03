"""Side-by-side comparison panels for every test image: Input | Mask | Ours | GT | |Ours - GT| x4.

Each panel is labelled with the image's PSNR / SSIM (ALL and shadow region S, measured with the
same protocol as scripts/evaluate.py at 256x256). Also writes per_image.csv (sorted by ALL PSNR)
and summary.md (dataset-level metrics) into the output directory.

Example:
    python scripts/compare.py --pred outputs/istd_plus/samples --dataset istd_plus --out comparisons/istd_plus
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffshadow.data import IMAGE_EXTS, list_triplets  # noqa: E402
from diffshadow.metrics import SIZE, ShadowMetrics, load_rgb01, markdown_table, psnr, shadow_region, ssim  # noqa: E402

HEADER = 22
LABELS = ("Input", "Mask", "Ours", "GT", "|Ours-GT| x4")


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def image_metrics(pred: np.ndarray, gt: np.ndarray, shadow: np.ndarray) -> dict:
    s3 = np.repeat(shadow[..., None], 3, axis=-1).astype(np.float64)
    return {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
            "psnr_s": psnr(pred * s3, gt * s3), "ssim_s": ssim(pred * s3, gt * s3)}


def build_panel(args: tuple) -> dict:
    stem, shadow_path, mask_path, free_path, pred_path, out_dir = args
    inp, gt, pred = load_rgb01(shadow_path), load_rgb01(free_path), load_rgb01(pred_path)
    shadow = shadow_region(np.asarray(Image.open(mask_path)))
    m = image_metrics(pred, gt, shadow)
    m["input_psnr"] = psnr(inp, gt)
    err = np.clip(np.abs(pred - gt).mean(-1, keepdims=True) * 4, 0, 1).repeat(3, -1)
    tiles = [inp, np.repeat(shadow[..., None], 3, -1).astype(np.float64), pred, gt, err]
    canvas = Image.new("RGB", (SIZE * len(tiles), SIZE + 2 * HEADER), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    for k, (tile, label) in enumerate(zip(tiles, LABELS)):
        canvas.paste(Image.fromarray(to_u8(tile)), (k * SIZE, HEADER))
        draw.text((k * SIZE + 4, 3), label, fill="black", font=font)
    draw.text((4, SIZE + HEADER + 3),
              f"{stem}   Ours ALL {m['psnr']:.2f} dB / {m['ssim']:.4f}   S {m['psnr_s']:.2f} dB / {m['ssim_s']:.4f}"
              f"   (Input ALL {m['input_psnr']:.2f} dB)", fill="black", font=font)
    canvas.save(Path(out_dir) / f"{stem}.png")
    return {"name": stem, **m}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="directory of result images (matched by stem)")
    p.add_argument("--dataset", required=True, choices=["istd_plus", "srd"])
    p.add_argument("--out", required=True, help="output directory for panels, per_image.csv, summary.md")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, help="only the first N test images that have a prediction")
    a = p.parse_args()

    pred_dir = Path(os.path.expanduser(a.pred))
    preds = {q.stem: q for q in pred_dir.iterdir() if q.suffix.lower() in IMAGE_EXTS and not q.name.startswith(".")}
    triplets = list_triplets(a.dataset, "test")
    if a.limit is not None:
        triplets = [t for t in triplets if t[0] in preds][: a.limit]
    missing = [t[0] for t in triplets if t[0] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} predictions missing in {pred_dir}, e.g. {missing[:3]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(stem, str(s), str(m), str(f), str(preds[stem]), str(out)) for stem, s, m, f in triplets]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(build_panel, jobs, chunksize=4))

    rows.sort(key=lambda r: r["psnr"])
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    metrics = ShadowMetrics()
    for stem, _s, mask, free in triplets:
        metrics.add(load_rgb01(str(preds[stem])), load_rgb01(str(free)), shadow_region(np.asarray(Image.open(mask))))
    table = markdown_table(metrics.summary(), f"{a.dataset}: {pred_dir}")
    (out / "summary.md").write_text(table)
    print(table)
    print(f"[compare] {len(rows)} panels -> {out}; worst ALL PSNR: "
          + ", ".join(f"{r['name']} {r['psnr']:.2f}" for r in rows[:5]))


if __name__ == "__main__":
    main()
