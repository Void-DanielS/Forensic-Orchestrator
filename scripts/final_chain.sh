#!/usr/bin/env bash
# After the pre-registered orchestration evaluation: improve Jev-MM (validation-selected seed ensemble), then one
# final-system run (full mode) with the improved image model.
set -euo pipefail
cd "$(dirname "$0")/.."
WAIT_ON="$1"                                    # output of the after_nested.sh waiter
until grep -q "all done" "$WAIT_ON" 2>/dev/null; do
  sleep 60
done
export HF_HUB_OFFLINE=1 FORENSI_VLM_QUANT=none
PY=".venv/Scripts/python -W ignore"
OUT=data/outputs/ml/planner/nested_20261117
echo "[final] $(date +%T) jev-mm improve"; $PY -m src.ml.mm_train improve > data/outputs/mm_improve.log 2>&1
echo "[final] $(date +%T) final system run"
$PY -m src.eval.orchestrate_eval --modes full --planners $OUT/models \
    --out data/outputs/ml/orchestrate/final_system_k100 > data/outputs/final_system.log 2>&1
echo "[final] $(date +%T) all done"
