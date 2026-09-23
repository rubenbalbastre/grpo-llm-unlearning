#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-/storage/scratch/lv13/lv13594/fresh-repo/grpo-llm-unlearning}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis
cd "${REPO_DIR}"

exec python3 scripts/run-target-reward-matrix.py
