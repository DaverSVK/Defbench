#!/usr/bin/env bash
# One matrix pipeline per dataset in parallel on a single large GPU (e.g. 48 GB), project-local envs only.
# Seed 42 twice (the 2nd pass picks up Docker models whose images finished building later), then 17,73.
# A separate loop refreshes reports/generated every hour. Latencies measured while jobs run in parallel
# are not clean - re-run `python -m defectbench benchmark ...` on an idle GPU for the final latency table.
cd "$(dirname "$0")/.."
source scripts/local_env.sh
PY=envs/core/bin/python
PROFILE=${1:-default}
# optional: EXCLUDE=model_a,model_b bash scripts/run_parallel.sh
MODELS=$("$PY" -c "import sys; from defectbench.models.registry import load_registry as r
ex=set(filter(None, sys.argv[1].split(','))); print(','.join(m for m in r() if m not in ex))" "${EXCLUDE:-}")
mkdir -p reports/logs
pids=()
for ds in neu_det magnetic_tile gc10_det; do
  (
    for seeds in 42 42 17,73; do
      "$PY" -m defectbench run-matrix --datasets "$ds" --models "$MODELS" --seeds "$seeds" --profile "$PROFILE"
    done
  ) >> "reports/logs/run_matrix_${ds}.log" 2>&1 &
  pids+=($!)
  sleep 120   # stagger start-up (exports, weight downloads)
done
( while :; do sleep 3600; "$PY" -m defectbench summarize --output reports/generated >> reports/logs/summarize.log 2>&1; done ) &
sumpid=$!
wait "${pids[@]}"
kill $sumpid
"$PY" -m defectbench summarize --output reports/generated >> reports/logs/summarize.log 2>&1
echo "done: reports/generated/report.md"
