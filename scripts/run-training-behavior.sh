#!/bin/bash -l
set -euo pipefail

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

cd "${REPO_DIR}"
python3 eval/behaviour/prepare_training_rollouts.py "$@"
