#!/usr/bin/env bash
# Unattended end-to-end pipeline after the ISTD+ run:
#
#   tmux new-session -d -s pipeline "scripts/pipeline.sh > outputs/pipeline.log 2>&1"
#
# 1. waits until outputs/istd_plus/DONE exists (or its wrapper is gone);
# 2. GSS-samples the ISTD+ test set from final.pt (EMA) on all GPUs and evaluates it;
# 3. trains SRD with scripts/run_resumable.sh (48 h budget starts at that launch) plus the CPU pilot check;
# 4. samples and evaluates SRD;
# 5. evaluates the 500-epoch milestone checkpoints of both runs (learning curve only, no selection).
# Every step logs and continues on failure so one problem does not idle the machine.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TORCHRUN=.venv/bin/torchrun
NPROC=${NPROC:-8}

log() { echo "[pipeline] $(date -u +%FT%TZ) $*"; }

wait_done() {  # <run_dir> <wrapper config pattern>
  while [ ! -f "$1/DONE" ]; do
    if ! pgrep -f "run_resumable.sh $2" >/dev/null; then
      log "WARNING: $1 has no DONE and its wrapper is not running; continuing with last.pt"
      return
    fi
    sleep 60
  done
  log "$1 DONE: $(tr -d '\n ' < "$1/DONE")"
}

sample_eval() {  # <config> <dataset> <ckpt> <out_dir>
  local cfg=$1 ds=$2 ckpt=$3 out=$4
  [ -f "$ckpt" ] || { log "missing $ckpt, skipping"; return; }
  log "sampling $ckpt -> $out/samples"
  $TORCHRUN --master_port 29544 --nproc_per_node "$NPROC" scripts/sample.py --config "$cfg" --ckpt "$ckpt" \
    --out "$out/samples" || { log "sampling FAILED ($ckpt)"; return; }
  $PY scripts/evaluate.py --pred "$out/samples" --dataset "$ds" --out "$out/metrics.json" \
    || log "evaluation FAILED ($out)"
  log "evaluated $out"
}

final_ckpt() { [ -f "$1/ckpt/final.pt" ] && echo "$1/ckpt/final.pt" || echo "$1/ckpt/last.pt"; }

log "start"
wait_done outputs/istd_plus configs/istd_plus.yaml
sample_eval configs/istd_plus.yaml istd_plus "$(final_ckpt outputs/istd_plus)" outputs/istd_plus

if [ -f outputs/srd/DONE ]; then
  log "outputs/srd already DONE, not retraining"
else
  mkdir -p outputs/srd
  $PY scripts/pilot_check.py --config configs/srd.yaml > outputs/srd/pilot_check.log 2>&1 &
  log "SRD training start"
  scripts/run_resumable.sh configs/srd.yaml >> outputs/srd/wrapper.log 2>&1 || log "SRD wrapper exited non-zero"
  log "SRD training end: $(tr -d '\n ' < outputs/srd/DONE 2>/dev/null)"
fi
sample_eval configs/srd.yaml srd "$(final_ckpt outputs/srd)" outputs/srd

for run in istd_plus srd; do
  ds=$run
  for ck in outputs/$run/ckpt/epoch_*.pt; do
    [ -f "$ck" ] || continue
    name=$(basename "$ck" .pt)
    [ "$name" = epoch_2000 ] && continue  # same weights as final.pt
    sample_eval configs/$run.yaml "$ds" "$ck" "outputs/$run/milestones/$name"
  done
done
log "all done"
