#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail

hostname; pwd; date
source "$HOME/anaconda3/etc/profile.d/conda.sh"
echo "Activate virtual environment (must exist)"
conda activate py312
echo "Run completion analysis"

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

RUNS="${RUNS:-3857,3859,3845,3863,3861}"
CONCEPT="${CONCEPT:-Rihanna}"
FUZZY_THRESHOLD="${FUZZY_THRESHOLD:-0.84}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/completion_analysis/${CONCEPT,,}-standard-runs.csv}"
MAX_EXAMPLES="${MAX_EXAMPLES:-40}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

IFS=',' read -r -a RUN_IDS <<< "${RUNS}"
RUN_PATHS=()
for run_id in "${RUN_IDS[@]}"; do
  RUN_PATHS+=("outputs/unlearning-${run_id}")
done

echo "Analyzing runs: ${RUN_PATHS[*]}"
echo "Output CSV: ${OUTPUT_CSV}"

python scripts/analyze-adaptive-completions.py \
  "${RUN_PATHS[@]}" \
  --concept "${CONCEPT}" \
  --fuzzy-threshold "${FUZZY_THRESHOLD}" \
  --max-examples "${MAX_EXAMPLES}" \
  --output-csv "${OUTPUT_CSV}"
