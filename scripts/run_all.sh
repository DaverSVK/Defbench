#!/usr/bin/env bash
# Full study on a (Linux) GPU machine: environments -> data -> tests -> smoke -> all runs -> report.
# Usage:  bash scripts/run_all.sh [profile] [seeds]        e.g.  bash scripts/run_all.sh default 17,42,73
# Re-running is safe: finished runs are skipped, interrupted runs are resumed from their checkpoints.
set -eu
PROFILE=${1:-default}          # default = 12-16 GB GPU, gpu8gb = 8 GB GPU
SEEDS=${2:-17,42,73}
cd "$(dirname "$0")/.."
PY=envs/core/bin/python; [ -x "$PY" ] || PY=envs/core/Scripts/python.exe
export PYTHONIOENCODING=utf-8
mkdir -p reports/logs

if [ ! -x "$PY" ]; then
  uv venv envs/core --python 3.11
  PY=envs/core/bin/python; [ -x "$PY" ] || PY=envs/core/Scripts/python.exe
  uv pip install -p "$PY" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install -p "$PY" -r requirements/core.txt
fi
"$PY" scripts/clone_third_party.py
for e in ultralytics rfdetr detr_family; do
  [ -d "envs/$e" ] || "$PY" scripts/setup_env.py "$e"
done
if command -v docker >/dev/null 2>&1; then
  for e in detectron2 mmdet; do
    "$PY" scripts/setup_env.py "$e" || echo "WARNING: docker environment $e could not be built - its models will be skipped"
  done
fi

# data: download (or use data/raw/* copied from another machine) + audit + manifest + split
if [ -f data/raw/neu_det/.download_complete ] || [ -d data/raw/neu_det/NEU-DET ]; then
  "$PY" -m defectbench prepare-data --dataset all
else
  "$PY" -m defectbench prepare-data --dataset all --download
fi

"$PY" -m pytest -q
for ds in neu_det magnetic_tile gc10_det; do
  "$PY" -m defectbench smoke-test --dataset "$ds" --profile "$PROFILE" \
    --models "$("$PY" -c "from defectbench.models.registry import load_registry as r; print(','.join(r()))")" \
    | tee "reports/logs/smoke_${ds}.log"
done

"$PY" -m defectbench run-matrix --seeds "$SEEDS" --profile "$PROFILE" 2>&1 | tee reports/logs/run_matrix.log
"$PY" -m defectbench summarize --output reports/generated
echo "done: reports/generated/report.md"
