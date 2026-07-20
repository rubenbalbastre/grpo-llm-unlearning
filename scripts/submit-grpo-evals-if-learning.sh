#!/bin/bash -l
#SBATCH --job-name=eval-gate
#SBATCH --output=logs/eval-gate-%j.log
#SBATCH --time=00:05:00
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
CHECKPOINT_ROOT="${1:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
CONCEPT="${2:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT%/}"
LOW_REWARD_STOP_MARKER="${LOW_REWARD_STOP_MARKER:-low_reward_stop.json}"
MARKER_PATH="${CHECKPOINT_ROOT}/${LOW_REWARD_STOP_MARKER}"

cd "${REPO_DIR}"

if [[ -f "${MARKER_PATH}" ]]; then
  echo "Skipping RWKU and hold-out evals because low-reward stop marker exists: ${MARKER_PATH}"
  cat "${MARKER_PATH}"
  exit 0
fi

rwku_job_id="$(
  sbatch --parsable \
    --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},INCLUDE_FINAL_MODEL=true" \
    scripts/eval-rwku.sh
)"

holdout_job_id="$(
  sbatch --parsable \
    --job-name="holdout" \
    --output="logs/holdout-%j.log" \
    --gres=gpu:1 \
    --time="01:30:00" \
    --partition="sc-gpu" \
    --wrap="cd ${REPO_DIR} && source \$HOME/anaconda3/etc/profile.d/conda.sh && conda activate py312 && python eval/hold_out_styles/generate-and-analyze-completions.py concept='${CONCEPT}' model_name_or_path='${CHECKPOINT_ROOT}/final_model'"
)"

echo "Submitted RWKU eval: ${rwku_job_id}"
echo "Submitted hold-out eval: ${holdout_job_id}"
