#!/usr/bin/env bash
# Main branch of the study (plan step 12): U-Net on the real masks of Magnetic Tile, Faster R-CNN baseline,
# its CBAM variant and one modern DETR detector (RF-DETR) on NEU-DET - one seed, sequentially on one GPU.
# Idempotent: finished steps are skipped, interrupted runs are RESUMED from their full-state checkpoint.
# Usage: bash scripts/run_main_branch.sh [seed] [profile] [pid-to-wait-for]
set -u
SEED=${1:-42}
PROFILE=${2:-gpu8gb}
WAIT_PID=${3:-}
PY=${PY:-envs/core/Scripts/python.exe}
[ -x "$PY" ] || PY=envs/core/bin/python
export PYTHONIOENCODING=utf-8
mkdir -p reports/logs
if [ -n "$WAIT_PID" ]; then
  echo "waiting for PID $WAIT_PID"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

has() { ls $1 >/dev/null 2>&1; }

run() {  # dataset model
  local ds=$1 m=$2 log="reports/logs/train_${ds}_${m}_s${SEED}.log" rd="runs/$ds/$m/*/seed$SEED"
  echo "=== $(date '+%F %T') $ds $m seed $SEED"
  if ! has "$rd/eval/metrics.json"; then
    local ok=0
    for attempt in 0 1 2 3; do
      local extra=()
      if has "$rd/checkpoints/last.pt" || has "$rd/framework/last.*" || has "$rd/framework/weights/last.pt"; then
        extra=(--resume auto)   # identical to an uninterrupted run (tests/test_cbam_and_resume.py)
        echo "--- $(date '+%F %T') resuming (attempt $attempt)"
      fi
      if "$PY" -m defectbench train --dataset "$ds" --model "$m" --seed "$SEED" --profile "$PROFILE" "${extra[@]}" \
           > "${log%.log}_a${attempt}.log" 2>&1; then ok=1; break; fi
      echo "--- $(date '+%F %T') attempt $attempt failed (transient CUDA errors were observed while Docker/WSL2 was active)"
    done
    [ $ok = 1 ] || { echo "FAILED $ds $m (see logs)"; return; }
  fi
  if ! has "$rd/benchmark/latency.json"; then
    "$PY" -m defectbench benchmark --dataset "$ds" --model "$m" --seed "$SEED" --profile "$PROFILE" \
        > "reports/logs/bench_${ds}_${m}_s${SEED}.log" 2>&1 || echo "benchmark FAILED $ds $m"
  fi
  echo "=== $(date '+%F %T') done $ds $m"
}
run magnetic_tile unet_r34
run neu_det faster_rcnn_r50_fpn
run neu_det faster_rcnn_r50_fpn_cbam
run neu_det rfdetr_small
"$PY" -m defectbench summarize --output reports/generated
echo "=== $(date '+%F %T') main branch finished"
