"""Diff-Shadow DDP training (Algorithm 1) with EMA, auto-resume and a strict real-time budget.

Launch (8 GPUs):
    torchrun --nproc_per_node 8 scripts/train.py --config configs/istd_plus.yaml --resume auto
Normally started through scripts/run_resumable.sh, which writes the clock anchor
(<run_dir>/first_launch_utc) and restarts after crashes.

Stop conditions (rank 0 decides, broadcast every iteration):
  real wall-clock since first launch >= budget_hours  -> DONE reason "deadline"
  epoch == epochs                                      -> DONE reason "epochs"
  global_step == max_steps (smoke/benchmark only)      -> checkpoint, no DONE
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffshadow.budget import Deadline, ensure_anchor, read_anchor  # noqa: E402
from diffshadow.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from diffshadow.config import load_config  # noqa: E402
from diffshadow.data import PairedShadowDataset  # noqa: E402
from diffshadow.diffusion import linear_alphas_cumprod, training_losses  # noqa: E402
from diffshadow.ema import EMA  # noqa: E402
from diffshadow.seeding import make_generators  # noqa: E402
from diffshadow.unet import ParallelUNets  # noqa: E402


def epoch_indices(n: int, world: int, rank: int, epoch: int, seed: int) -> list[int]:
    """DistributedSampler semantics (shuffle, drop_last): this rank's image order for one epoch."""
    g = torch.Generator().manual_seed(seed + epoch)
    perm = torch.randperm(n, generator=g).tolist()
    per_rank = n // world
    return perm[: per_rank * world][rank::world]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", choices=["auto", "none"], default="auto")
    p.add_argument("--max-steps", type=int, help="override train.max_steps (smoke/benchmark)")
    p.add_argument("--amp", choices=["off", "fp16"], help="override train.amp (benchmark)")
    p.add_argument("--run-dir", help="override run_dir (benchmark)")
    p.add_argument("--ddp-fp16-grads", choices=["on", "off"], help="override train.ddp_fp16_grads (benchmark)")
    a = p.parse_args()

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=30), device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()

    cfg = load_config(a.config)
    if a.amp:
        cfg.train.amp = a.amp
    if a.run_dir:
        cfg.run_dir = a.run_dir
    if a.ddp_fp16_grads:
        cfg.train.ddp_fp16_grads = a.ddp_fp16_grads == "on"
    tc = cfg.train
    max_steps = a.max_steps if a.max_steps is not None else tc.max_steps
    run = Path(cfg.run_dir)
    ckpt_dir, log_dir = run / "ckpt", run / "logs"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        if tc.create_anchor:
            ensure_anchor(run)
    dist.barrier()
    if not (run / "first_launch_utc").exists():
        raise SystemExit(f"{run}/first_launch_utc missing: launch through scripts/run_resumable.sh")
    deadline = Deadline(read_anchor(run), tc.budget_hours)

    data = PairedShadowDataset(cfg.dataset, "train", size=cfg.image_size)
    iters_per_epoch = len(data) // (world * tc.images_per_gpu)

    model = ParallelUNets(cfg.model).to(device)
    ema = EMA(model, tc.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=0.0)
    use_amp = tc.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    alphas = linear_alphas_cumprod(cfg.diffusion).to(device)

    state = {"global_step": 0, "train_seconds": 0.0, "restart_count": 0}
    last = ckpt_dir / "last.pt"
    if a.resume == "auto" and last.exists():
        state = load_checkpoint(last, model, ema, opt, scaler, map_location=device)
        state["restart_count"] += 1
    ddp = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True, static_graph=True)
    if tc.ddp_fp16_grads:
        # Gradients are all-reduced in fp16 (PCIe-only interconnect); parameters/optimizer stay fp32.
        ddp.register_comm_hook(None, default_hooks.fp16_compress_hook)
    step = state["global_step"]
    cpu_gen, dev_gen = make_generators(tc.seed, rank, step, device)
    if rank == 0:
        params = model.count_parameters()
        print(f"[train] {cfg.name}: params {params}, iters/epoch {iters_per_epoch}, resume step {step}, "
              f"restarts {state['restart_count']}, real elapsed {deadline.elapsed() / 3600:.2f} h", flush=True)

    def checkpoint(path: Path) -> None:
        save_checkpoint(path, model, ema, opt, scaler, {**state, "global_step": step, "config": a.config,
                                                        "epoch": step // iters_per_epoch})

    def write_done(reason: str) -> None:
        real = deadline.elapsed()
        info = {"reason": reason, "epochs": step / iters_per_epoch, "steps": step, "real_elapsed_s": real,
                "train_seconds": state["train_seconds"], "downtime_s": real - state["train_seconds"],
                "restarts": state["restart_count"], "precision": tc.amp, "finished_utc": time.time()}
        (run / "DONE").write_text(json.dumps(info, indent=2))

    log_f = open(log_dir / "train.jsonl", "a") if rank == 0 else None
    last_ckpt_time = time.time()
    pilot_marks = [m for m in tc.pilot_minutes if not (ckpt_dir / f"pilot_{int(m)}m.pt").exists()]
    seg_start, seg_steps = time.time(), 0
    order: list[int] = []
    order_epoch = -1
    reason = None
    while True:
        # ---- stop decision (rank 0, broadcast) ----
        flag = torch.zeros(1, device=device)
        if rank == 0:
            if deadline.expired():
                flag[0] = 1
            elif step >= tc.epochs * iters_per_epoch:
                flag[0] = 2
            elif max_steps is not None and step >= max_steps:
                flag[0] = 3
        dist.broadcast(flag, 0)
        if flag.item() != 0:
            reason = {1: "deadline", 2: "epochs", 3: "max_steps"}[int(flag.item())]
            break

        epoch, it = divmod(step, iters_per_epoch)
        if epoch != order_epoch:
            order = epoch_indices(len(data), world, rank, epoch, tc.seed)
            order_epoch = epoch
        idx = order[it * tc.images_per_gpu:(it + 1) * tc.images_per_gpu]
        batch = [data[i] for i in idx]
        clean = torch.stack([b["clean"] for b in batch]).to(device, non_blocking=True)
        shadow = torch.stack([b["shadow"] for b in batch]).to(device, non_blocking=True)
        mask = torch.stack([b["mask"] for b in batch]).to(device, non_blocking=True)

        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss, l_diff, l_global = training_losses(
                ddp, alphas, clean, shadow, mask, cfg.model.image_size, tc.patches_per_image,
                tc.global_loss_weight, cpu_gen, dev_gen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        torch.cuda.synchronize()
        state["train_seconds"] += time.time() - t0
        step += 1
        seg_steps += 1

        if step % tc.log_every == 0:
            stats = torch.stack([loss.detach().float(), l_diff.detach().float(), l_global.detach().float()])
            dist.all_reduce(stats)
            stats /= world
            if rank == 0:
                now = time.time()
                rec = {"step": step, "epoch": step / iters_per_epoch, "loss": stats[0].item(),
                       "l_diff": stats[1].item(), "l_global": stats[2].item(),
                       "s_per_it": (now - seg_start) / max(seg_steps, 1), "train_seconds": state["train_seconds"],
                       "real_elapsed": deadline.elapsed(), "restart_count": state["restart_count"],
                       "max_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30, "time": now}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if not torch.isfinite(stats).all():
                    raise FloatingPointError(f"non-finite loss at step {step}: {rec}")
                seg_start, seg_steps = now, 0

        if rank == 0:
            if time.time() - last_ckpt_time >= tc.ckpt_minutes * 60:
                if last.exists():
                    last.replace(ckpt_dir / "prev.pt")
                checkpoint(last)
                last_ckpt_time = time.time()
            if tc.milestone_epochs and step % (tc.milestone_epochs * iters_per_epoch) == 0:
                checkpoint(ckpt_dir / f"epoch_{step // iters_per_epoch:04d}.pt")
            while pilot_marks and deadline.elapsed() >= pilot_marks[0] * 60:
                checkpoint(ckpt_dir / f"pilot_{int(pilot_marks.pop(0))}m.pt")

    if rank == 0:
        checkpoint(last)
        if reason != "max_steps":
            checkpoint(ckpt_dir / "final.pt")
            write_done(reason)
        print(f"[train] stopped: {reason} at step {step} (epoch {step / iters_per_epoch:.2f})", flush=True)
        log_f.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
