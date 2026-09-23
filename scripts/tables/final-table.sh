#!/bin/bash -l
set -euo pipefail

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

cd "${REPO_DIR}"

python eval/rwku/create_final_table.py \
  --output-csv outputs/tables/rwku.csv \
  --author-output-csv outputs/tables/rwku_author_deltas.csv

python eval/behaviour/create_final_table.py
