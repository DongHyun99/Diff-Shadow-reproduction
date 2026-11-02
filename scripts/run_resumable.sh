#!/usr/bin/env bash
# Auto-resuming launcher for one Diff-Shadow training run.
#
#   setsid nohup scripts/run_resumable.sh configs/istd_plus.yaml > outputs/istd_plus/wrapper.log 2>&1 &
#
# - writes <run_dir>/first_launch_utc once: the strict real-time budget starts here;
# - (re)launches `torchrun --nproc_per_node $NPROC scripts/train.py --resume auto` until <run_dir>/DONE exists;
# - kills a hung job when logs/train.jsonl has not been updated for $HEARTBEAT_MIN minutes;
# - before every relaunch checks the deadline; if it already passed, promotes last.pt to
#   final.pt and writes DONE with reason "deadline_during_restart".
set -uo pipefail

CONFIG=${1:?usage: run_resumable.sh <config.yaml>}
NPROC=${NPROC:-8}
MAX_RESTARTS=${MAX_RESTARTS:-10}
HEARTBEAT_MIN=${HEARTBEAT_MIN:-15}
BACKOFF_S=${BACKOFF_S:-10}
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TORCHRUN=.venv/bin/torchrun

read -r RUN_DIR HOURS < <($PY -c "
import sys; sys.path.insert(0, '.')
from diffshadow.config import load_config
c = load_config('$CONFIG'); print(c.run_dir, c.train.budget_hours)")
mkdir -p "$RUN_DIR/logs"
$PY -m diffshadow.budget --run "$RUN_DIR" --hours "$HOURS" --init >/dev/null
echo "[wrapper] $(date -u +%FT%TZ) run=$RUN_DIR budget=${HOURS}h anchor=$(cat "$RUN_DIR/first_launch_utc")"

launches=0
while [ ! -f "$RUN_DIR/DONE" ]; do
  if ! $PY -m diffshadow.budget --run "$RUN_DIR" --hours "$HOURS"; then
    echo "[wrapper] deadline passed before relaunch"
    $PY -m diffshadow.finalize --run "$RUN_DIR" --reason deadline_during_restart
    break
  fi
  if [ "$launches" -gt "$MAX_RESTARTS" ]; then
    echo "[wrapper] giving up after $MAX_RESTARTS restarts"
    exit 1
  fi
  launches=$((launches + 1))
  echo "[wrapper] $(date -u +%FT%TZ) launch #$launches"
  setsid $TORCHRUN --nproc_per_node "$NPROC" scripts/train.py --config "$CONFIG" --resume auto &
  pid=$!
  start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    log="$RUN_DIR/logs/train.jsonl"
    ref=$start
    [ -f "$log" ] && ref=$(( $(stat -c %Y "$log") > start ? $(stat -c %Y "$log") : start ))
    if [ $(( $(date +%s) - ref )) -gt $((HEARTBEAT_MIN * 60)) ]; then
      echo "[wrapper] $(date -u +%FT%TZ) heartbeat stale > ${HEARTBEAT_MIN} min, killing job"
      kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null
    fi
  done
  wait "$pid"; code=$?
  echo "[wrapper] $(date -u +%FT%TZ) job exited with code $code"
  [ -f "$RUN_DIR/DONE" ] || sleep "$BACKOFF_S"
done
echo "[wrapper] $(date -u +%FT%TZ) finished: $(cat "$RUN_DIR/DONE" | tr -d '\n ')"
