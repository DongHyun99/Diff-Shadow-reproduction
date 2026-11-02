"""Close a run whose deadline passed while it was down: last.pt -> final.pt + DONE.

    python -m diffshadow.finalize --run outputs/istd_plus --reason deadline_during_restart
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from .budget import read_anchor


def finalize(run: Path, reason: str) -> dict:
    last = run / "ckpt" / "last.pt"
    st = torch.load(last, map_location="cpu", weights_only=False)["state"]
    shutil.copyfile(last, run / "ckpt" / "final.pt")
    real = time.time() - read_anchor(run)
    info = {"reason": reason, "epochs": st.get("epoch"), "steps": st["global_step"], "real_elapsed_s": real,
            "train_seconds": st["train_seconds"], "downtime_s": real - st["train_seconds"],
            "restarts": st["restart_count"], "finished_utc": time.time()}
    (run / "DONE").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--reason", required=True)
    a = p.parse_args()
    print(finalize(Path(a.run), a.reason))


if __name__ == "__main__":
    main()
