#!/usr/bin/env bash
# Full matrix with project-local environments only (see scripts/local_env.sh).
# Seed 42 first (early complete results), then 17 and 73; finished runs are skipped on re-run.
set -u
cd "$(dirname "$0")/.."
source scripts/local_env.sh
PY=envs/core/bin/python
PROFILE=${1:-default}
mkdir -p reports/logs
for seeds in 42 17,73; do
  "$PY" -m defectbench run-matrix --seeds "$seeds" --profile "$PROFILE" 2>&1 | tee -a reports/logs/run_matrix.log
  "$PY" -m defectbench summarize --output reports/generated 2>&1 | tee -a reports/logs/summarize.log
done
echo "done: reports/generated/report.md"
