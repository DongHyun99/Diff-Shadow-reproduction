"""Evaluate shadow-removal results at 256x256 (PSNR/SSIM RGB, LAB error on S/NS/ALL).

Example:
    python scripts/evaluate.py --pred outputs/istd_plus/samples --dataset istd_plus \
        --out outputs/istd_plus/metrics.json
Predictions are matched to ground truth by file stem; every test image must be present
unless ``--limit`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffshadow.data import DATASETS, IMAGE_EXTS, list_triplets  # noqa: E402
from diffshadow.metrics import ShadowMetrics, load_rgb01, markdown_table, shadow_region  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="directory of result images (matched by stem)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--out", help="metrics JSON path (markdown table written next to it)")
    p.add_argument("--limit", type=int)
    p.add_argument("--illuminant", default="D65", choices=["D65", "D50"])
    a = p.parse_args()

    pred_dir = Path(os.path.expanduser(a.pred))
    preds = {q.stem: q for q in pred_dir.iterdir() if q.suffix.lower() in IMAGE_EXTS and not q.name.startswith(".")}
    triplets = list_triplets(a.dataset, "test")
    if a.limit is not None:
        triplets = [t for t in triplets if t[0] in preds][: a.limit]
    missing = [t[0] for t in triplets if t[0] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} predictions missing in {pred_dir}, e.g. {missing[:3]}")

    m = ShadowMetrics(illuminant=a.illuminant)
    for stem, _shadow, mask, free in triplets:
        m.add(load_rgb01(str(preds[stem])), load_rgb01(str(free)), shadow_region(np.asarray(Image.open(mask))))
    summary = m.summary()
    summary.update({"dataset": a.dataset, "pred": str(pred_dir)})
    table = markdown_table(summary, f"{a.dataset}: {pred_dir}")
    print(table)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        out.with_suffix(".md").write_text(table)


if __name__ == "__main__":
    main()
