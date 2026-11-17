#!/usr/bin/env bash
# Tuned LLM prior (pre-registration deviation of 2026-11-17), run AFTER the nested planner run has finished:
# LoRA fine-tune on the write-up corpus -> tuned prior for every dataset file -> add the candidate to the nested run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python -W ignore"
OUT=data/outputs/ml/planner/nested_20261117
export HF_HUB_OFFLINE=1 FORENSI_VLM_QUANT=none
$PY -m src.ml.writeups
$PY -m src.ml.finetune_prior
$PY -m src.ml.llm_prior --batch 16 --adapter data/models/llm_prior_lora
$PY -m src.ml.planner_nested extend $OUT
