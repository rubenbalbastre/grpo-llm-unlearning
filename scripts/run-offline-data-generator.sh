#!/bin/bash -l
#SBATCH --job-name=offline-data-generator
#SBATCH --output=logs/slurm-offline-data-generator-%j.log
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:30:00
set -euo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

python "${REPO_DIR}/src/offline_data_generator.py" "$@"
