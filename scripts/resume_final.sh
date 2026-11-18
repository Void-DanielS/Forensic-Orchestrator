#!/usr/bin/env bash
# Resume the pre-registered orchestration evaluation (resumable by case|mode), then Jev-MM improve, then the
# final-system run. PYTORCH_CUDA_ALLOC_CONF added after an out-of-memory at owl|lr (fragmentation).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 FORENSI_VLM_QUANT=none PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/Scripts/python -W ignore"
OUT=data/outputs/ml/planner/nested_20261117
echo "[resume] $(date +%T) orchestration evaluation (resume)"
$PY -m src.eval.orchestrate_eval --planners $OUT/models --out data/outputs/ml/orchestrate/eval_k100_final \
    >> data/outputs/orchestrate_eval.log 2>&1
echo "[resume] $(date +%T) jev-mm improve"; $PY -m src.ml.mm_train improve > data/outputs/mm_improve.log 2>&1
echo "[resume] $(date +%T) final system run"
$PY -m src.eval.orchestrate_eval --modes full --planners $OUT/models \
    --out data/outputs/ml/orchestrate/final_system_k100 > data/outputs/final_system.log 2>&1
echo "[resume] $(date +%T) all done"
