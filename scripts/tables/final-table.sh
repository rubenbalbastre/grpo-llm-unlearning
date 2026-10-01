#!/bin/bash -l
set -euo pipefail

ENV_DIR="${ENV_DIR:-/storage/scratch/lv13/lv13594/}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ENV_DIR/anaconda3_bis/etc/profile.d/conda.sh"
conda activate py312_cu118_bis

cd "${REPO_DIR}"

python3 eval/rwku/create_final_table.py \
  --output-csv outputs/tables/rwku.csv \
  --author-output-csv outputs/tables/rwku_authors.csv

python3 eval/behaviour/create_final_heldout_audit.py
python3 eval/behaviour/create_final_terminal_training_audit.py

tar -czf outputs/tables/final-tables.tar.gz \
  -C outputs/tables \
  rwku.csv \
  rwku_authors.csv \
  heldout_behaviour.csv \
  heldout_behaviour_authors.csv \
  terminal_training_behaviour.csv \
  terminal_training_behaviour_authors.csv
