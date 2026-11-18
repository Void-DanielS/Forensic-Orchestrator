#!/usr/bin/env bash
# Waits for the nested planner run to end, then: tuned prior chain -> report -> deploy -> orchestration evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/outputs/ml/planner/nested_20261117
DONE_FLAG="$1"                                  # file that becomes non-empty when the nested run's shell has exited
until [ -s "$DONE_FLAG" ]; do sleep 60; done
n=$(ls $OUT/*.json | grep -vc config.json || true)
[ "$n" -ge 13 ] || { echo "nested run ended with only $n folds: stopping"; exit 1; }
export HF_HUB_OFFLINE=1 FORENSI_VLM_QUANT=none
PY=".venv/Scripts/python -W ignore"
echo "[after] $(date +%T) tuned prior chain"; bash scripts/tuned_prior_chain.sh > data/outputs/tuned_prior_chain.log 2>&1
echo "[after] $(date +%T) report";  $PY -m src.ml.planner_nested report $OUT > data/outputs/planner_report.log 2>&1
echo "[after] $(date +%T) deploy";  $PY -m src.ml.planner_nested deploy $OUT > data/outputs/planner_deploy.log 2>&1
echo "[after] $(date +%T) orchestration evaluation"
$PY -m src.eval.orchestrate_eval --planners $OUT/models --out data/outputs/ml/orchestrate/eval_k100_final \
    > data/outputs/orchestrate_eval.log 2>&1
echo "[after] $(date +%T) all done"
