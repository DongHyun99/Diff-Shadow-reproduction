"""Strict real wall-clock budget: a run stops ``hours`` after its first launch.

The anchor file ``<run>/first_launch_utc`` is written once and never overwritten, so
pilot time, crashes and restart gaps all count against the budget.

CLI (used by the wrapper before each relaunch):
    python -m diffshadow.budget --run <dir> [--init] [--hours 48]
exit code 0 = time remains, 3 = expired.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ANCHOR = "first_launch_utc"


def ensure_anchor(run_dir: str | Path, clock: Callable[[], float] = time.time) -> float:
    path = Path(run_dir) / ANCHOR
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{clock():.3f}\n")
    return read_anchor(run_dir)


def read_anchor(run_dir: str | Path) -> float:
    return float((Path(run_dir) / ANCHOR).read_text().strip())


@dataclass
class Deadline:
    first_launch: float
    hours: float
    clock: Callable[[], float] = time.time

    def elapsed(self) -> float:
        return self.clock() - self.first_launch

    def expired(self) -> bool:
        return self.elapsed() >= self.hours * 3600.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--init", action="store_true", help="create the anchor if absent")
    a = p.parse_args(argv)
    if a.init:
        ensure_anchor(a.run)
    d = Deadline(read_anchor(a.run), a.hours)
    print(f"real_elapsed_h={d.elapsed() / 3600:.3f} budget_h={a.hours}")
    return 3 if d.expired() else 0


if __name__ == "__main__":
    sys.exit(main())
