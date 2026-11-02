# Diff-Shadow reproduction

An independent PyTorch reproduction of **Diff-Shadow: Global-guided Diffusion Model for Shadow Removal**
(Luo et al., AAAI 2025, [arXiv 2407.16214](https://arxiv.org/abs/2407.16214)).
The official repository ([MonteCarluo/Diff-Shadow](https://github.com/MonteCarluo/Diff-Shadow)) contains no code,
so everything here is written from the paper and its supplementary material. Only the main model is reproduced
(no ablations). Results are in [RESULTS.md](RESULTS.md).

## What is implemented

| Paper | Code |
|---|---|
| Parallel UNets: local noise estimator (7-ch input: x_t, shadow patch, mask patch; self-attention at 16x16) and global shadow-free recovery branch (4-ch input: downsampled shadow image + mask; no self-attention) | `diffshadow/unet.py` `ParallelUNets` |
| Reweight Cross Attention after the first residual block, Eq. 4 | `ReweightCrossAttention` |
| Concatenation interaction on high-level (bottleneck) features | `ParallelUNets.local_forward` (`mid_fuse`) |
| Loss L_diff + lambda * L_global (Eq. 9, 10), Algorithm 1 | `diffshadow/diffusion.py` `training_losses` |
| Global-guided Sampling Strategy, Algorithm 2, Eq. 5-8 | `diffshadow/sampler.py` `gss_sample` |
| Evaluation at 256x256: PSNR/SSIM (RGB), LAB error on S/NS/ALL | `diffshadow/metrics.py`, `scripts/evaluate.py` |

### Hyper-parameters (main text, supplementary Tables 4 and 5)

| | |
|---|---|
| Patch / global input size | 64 x 64 (images resized to 256 x 256) |
| Base channels, multipliers, res blocks | 128, (1, 1, 2, 2, 4, 4), 2 |
| Diffusion | T = 1000, linear beta 1e-4 -> 0.02 |
| Sampling | S = 25 DDIM steps (eta = 0), patch R = 64, stride r = 8, T_sf = 0.5, eps = 0.01 |
| Optimisation | Adam, lr 2e-4, no weight decay, EMA 0.999, lambda = 1 |
| Batch | 8 images x 16 random patches per iteration (8 GPUs x 1 image) |
| Schedule | 2000 epochs nominal; here capped by the compute budget below |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Data (ShadowFormer layout, `<split>_A` shadow / `<split>_B` mask / `<split>_C` shadow-free):

```
~/Dataset/Shadow/aistd/{train,test}/{train,test}_{A,B,C}   # ISTD+ (1330 / 540 images)
~/Dataset/Shadow/SRD/{train,test}/{train,test}_{A,B,C}     # SRD   (2680 / 408 images)
```

`aistd/new_train` and `aistd/test_256` are never read; the loader asserts the image counts above.

## Commands

```bash
# unit tests (CPU)
.venv/bin/python -m pytest -q

# 8-GPU smoke test: 20 steps, then resume to 30, sample 2 images, evaluate
.venv/bin/torchrun --nproc_per_node 8 scripts/train.py --config configs/smoke.yaml
.venv/bin/torchrun --nproc_per_node 8 scripts/train.py --config configs/smoke.yaml --resume auto --max-steps 30
.venv/bin/torchrun --nproc_per_node 2 scripts/sample.py --config configs/smoke.yaml --ckpt outputs/smoke/ckpt/last.pt --limit 2
.venv/bin/python scripts/evaluate.py --pred outputs/smoke/samples --dataset istd_plus --limit 2

# evaluator sanity check: the shadow input itself vs Table 1 "Input Image" row
.venv/bin/python scripts/evaluate.py --pred ~/Dataset/Shadow/SRD/test/test_A --dataset srd --out outputs/baseline_srd.json

# full training (strict 48h real-time budget, auto-resume after crashes)
mkdir -p outputs/istd_plus
setsid nohup scripts/run_resumable.sh configs/istd_plus.yaml > outputs/istd_plus/wrapper.log 2>&1 &
.venv/bin/python scripts/pilot_check.py --config configs/istd_plus.yaml   # learning check after 30 / 90 min

# sampling (EMA weights of final.pt) and evaluation
.venv/bin/torchrun --nproc_per_node 8 scripts/sample.py --config configs/istd_plus.yaml
.venv/bin/python scripts/evaluate.py --pred outputs/istd_plus/samples --dataset istd_plus --out outputs/istd_plus/metrics.json
```

Replace `istd_plus` by `srd` for SRD.

## Compute budget

Hardware: 8x NVIDIA TITAN RTX 24 GB (Turing, PCIe only, two NUMA nodes).

- **Budget:** each dataset trains until 48 hours of real wall-clock time have passed since the first launch of
  that run (`outputs/<run>/first_launch_utc`), or 2000 epochs, whichever comes first. Crash and restart gaps
  count against the budget. `train_seconds` (time spent inside training steps) is logged but is not a stop
  condition. Test-set sampling is outside the budget.
- **Speed-ups used** (all in `configs/*.yaml` / `scripts/train.py`; none changes the model or the optimisation):
  - mixed precision: fp16 autocast + dynamic loss scaling (`amp: fp16`); weights, optimizer state and EMA stay fp32;
  - fp16 gradient all-reduce (`ddp_fp16_grads`, DDP communication hook): the step is bound by all-reducing
    218M gradients over PCIe (no GPU peer-to-peer on this machine, NCCL goes through host shared memory);
  - `torch.compile` of the DDP model (`compile`), channels-last memory format (`channels_last`),
    `cudnn.benchmark`, fused Adam, foreach EMA update;
  - no per-step host synchronisation: the wall-clock deadline is broadcast every 10 steps;
  - sampling also runs under fp16 autocast with channels-last weights.
- **Throughput** (8 GPUs, `configs/smoke.yaml` = full model, steps 150-300):

  | Setting | s/it |
  |---|---|
  | fp32, fp32 gradient all-reduce (first attempt) | 0.70 |
  | fp32, fp16 gradient all-reduce | 0.57 |
  | + fp16 autocast | 0.52 |
  | + channels-last, torch.compile, cudnn.benchmark, fused Adam | 0.45 |
  | + no per-step synchronisation (**used**) | 0.41 |
  | single GPU, same settings, no communication | 0.20 |

  Remaining overhead is the fp16 all-reduce itself (0.19 s for 218M values, measured with NCCL alone).
  NCCL tuning (P2P level, channels, tree/ring, protocol) and other DDP bucket sizes did not help.
  Splitting the machine into two concurrent 4-GPU jobs (ISTD+ and SRD at once, 2 images per GPU) was also
  measured: 0.64 s/it each, i.e. more total throughput but fewer epochs per dataset; sequential 8-GPU runs
  were kept so ISTD+ can reach the full 2000-epoch schedule.
- **Projection at 0.41 s/it:** ISTD+ 166 it/epoch -> 2000 epochs in about 38 h (stops on epochs, before the
  48 h deadline); SRD 335 it/epoch -> about 1250 of 2000 epochs in 48 h. RESULTS.md reports the epochs reached.
- **Sampling:** GSS with r = 8 needs 625 patches x 25 steps per 256 x 256 image, about 35 s per image per GPU
  in fp16 (ISTD+ about 40 min, SRD about 30 min on 8 GPUs). The paper reports 3.5 s per image with a coarser
  grid; the numbers are not comparable.
- **First ISTD+ attempt:** a fp32 run (0.50 s/it) was stopped after 50 minutes to switch to the faster settings
  and restarted from scratch with a new 48 h clock.

## Assumptions (paper-unspecified or contradictory)

Unspecified details follow the DDPM / WeatherDiffusion conventions that Diff-Shadow builds on.

**Architecture**
- Residual blocks, sinusoidal timestep embedding (fed to every residual block of *both* branches), GroupNorm(32),
  SiLU, dropout 0, nearest upsampling + conv, strided-conv downsampling: DDPM / WeatherDiffusion UNet.
- RCA: GroupNorm + 1x1-conv projections for Q (local features) and K, V (global features), output 1x1 projection,
  residual connection, scale 1/sqrt(C). Eq. 4 is read literally, Z = softmax(QK^T/sqrt(C)) (M V): the non-shadow
  mask multiplies V after the softmax without renormalisation. A masked-softmax reading was rejected.
- RCA sits after the first residual block of the local encoder; the global features come from the first residual
  block of the global encoder.
- Concatenation interaction: the global branch bottleneck output (after its middle block) is concatenated with the
  local encoder output at the 2x2 bottleneck and fused by a 1x1 conv before the local middle block.
- **Parameter count:** this implementation has 217.7M parameters (local 110.3M, global 107.4M, RCA 0.07M); the paper
  reports 104M for a single UNet and 160M for the parallel model. The text says the branches are nearly identical
  with the same Table 4 configuration, which yields ~110M each, so the gap was documented rather than closed by
  guessing a smaller global branch.
- The global branch receives the same timestep as the local branch and is run per patch during training (each patch
  has its own t), and once per image per step during sampling.

**Training**
- The paper states "8 images x 16 patches = 96"; 8 x 16 = 128 patches per iteration are used.
- No data augmentation, no gradient clipping. One epoch = every training image used once (1330 // 8 = 166 or
  2680 // 8 = 335 iterations, DistributedSampler-style shuffling with drop_last).
- Per-rank random streams: crops, timesteps and noise use generators seeded with
  `base_seed * 1_000_003 + rank * 10_007 + global_step`, re-derived on every resume.
- Global input: area downsampling of the 256x256 image to 64x64; the mask is area-downsampled then thresholded
  at 0.5. Masks enter the network as 2m - 1 (shadow = +1); RCA uses the non-shadow region (mask channel <= 0).
- Sampling uses the EMA weights of the last checkpoint (no test-set checkpoint selection).

**Sampling (GSS)**
- Algorithm 2 line 9 downsamples x (noise); the shadow image is downsampled instead (the global branch is trained
  on the shadow image).
- Eq. 5 has no eps although eps = 0.01 is listed; it is placed in the denominator, S_bri = 1 - avg|L-G| / (max|L-G| + eps).
- L is the grayscale (BT.601) of the local x_0 estimate, G the grayscale of the global output bilinearly upsampled
  to full resolution. Eq. 7 uses t in {961, ..., 1} (1-indexed). The image is replicate-padded so the 64/8 patch
  grid ends at the border.

**Data and evaluation**
- Masks: the local `*_B` masks are used both as model input and to define the S/NS evaluation regions
  (the paper uses DHAN-predicted masks as input). Every source mask is binarised at > 127 before resizing,
  in both the loader and the evaluator (SRD masks are noisy JPEGs).
- Evaluator: port of ShadowFormer's `evaluation/measure_shadow.m`: bicubic resize to 256x256, PSNR (peak 1) per
  image, S/NS PSNR and SSIM on mask-multiplied images, SSIM with MATLAB's volumetric 11-tap Gaussian window
  (sigma 1.5) over H x W x 3. The "RMSE" column is that script's LAB mean absolute error: S/NS accumulated over all
  pixels of the test set, ALL averaged over images. LAB uses skimage's D65 white point (MATLAB `makecform` adapts
  to D50); the true LAB RMSE is reported as an extra column.

## Layout

```
diffshadow/   model, diffusion loss, GSS sampler, data, metrics, EMA, checkpoints, budget, seeding
scripts/      train.py, run_resumable.sh, sample.py, evaluate.py, pilot_check.py
configs/      istd_plus.yaml, srd.yaml, smoke.yaml, smoke_wrapper.yaml
tests/        pytest suite (CPU)
```
