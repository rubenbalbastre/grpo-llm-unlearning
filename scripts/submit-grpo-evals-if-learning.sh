#!/bin/bash -l
#SBATCH --job-name=eval-gate
#SBATCH --output=logs/eval-gate-%j.log
#SBATCH --time=00:05:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_ENV="${SCRIPT_DIR}/slurm-env.sh"
if [[ ! -f "${SLURM_ENV}" ]]; then
  SLURM_ENV="${REPO_DIR:-/storage/scratch/lv13/lv13594/machine-unlearning-llm}/scripts/slurm-env.sh"
fi
source "${SLURM_ENV}"
CHECKPOINT_ROOT="${1:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
CONCEPT="${2:?Usage: submit-grpo-evals-if-learning.sh CHECKPOINT_ROOT CONCEPT}"
STORAGE_ROOT="${3:-${STORAGE_ROOT:-.}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT%/}"
LOW_REWARD_STOP_MARKER="${LOW_REWARD_STOP_MARKER:-low_reward_stop.json}"
MARKER_PATH="${CHECKPOINT_ROOT}/${LOW_REWARD_STOP_MARKER}"
RUN_RWKU_EVAL="${RUN_RWKU_EVAL:-true}"
RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-true}"

cd "${REPO_DIR}"

flag_enabled() {
  [[ "$1" == "true" ]]
}

if ! flag_enabled "${RUN_RWKU_EVAL}" && ! flag_enabled "${RUN_HOLDOUT_EVAL}"; then
  echo "Skipping eval gate because both RUN_RWKU_EVAL and RUN_HOLDOUT_EVAL are disabled."
  exit 0
fi

if [[ -f "${MARKER_PATH}" ]]; then
  echo "Skipping enabled evals because low-reward stop marker exists: ${MARKER_PATH}"
  cat "${MARKER_PATH}"
  exit 0
fi

RWKU_OUTPUT_DIR="${CHECKPOINT_ROOT}/final_model/eval_rwku"
HOLDOUT_OUTPUT_DIR="${CHECKPOINT_ROOT}/final_model/hold_out_eval"

if ! flag_enabled "${RUN_RWKU_EVAL}"; then
  rwku_job_id="disabled"
elif [[ -f "${RWKU_OUTPUT_DIR}/rwku_summary_table.csv" ]]; then
  rwku_job_id="done"
else
  rwku_job_id="$(
    sbatch --parsable \
      --export="ALL,CHECKPOINT_ROOT=${CHECKPOINT_ROOT},ONLY_FINAL_MODEL=true" \
      scripts/eval-rwku.sh
  )"
fi

if ! flag_enabled "${RUN_HOLDOUT_EVAL}"; then
  holdout_job_id="disabled"
elif [[ -f "${HOLDOUT_OUTPUT_DIR}/metrics.csv" && -f "${HOLDOUT_OUTPUT_DIR}/summary.csv" ]]; then
  holdout_job_id="done"
else
  holdout_job_id="$(
    sbatch --parsable \
      --job-name="holdout" \
      --output="logs/holdout-%j.log" \
      --gres=gpu:1 \
      --time="01:30:00" \
      --partition="hopper" \
      --qos="hopper" \
      --export="ALL,SKIP_IF_MARKER=${MARKER_PATH}" \
      scripts/run-hold-out-completions-analysis.sh \
      "concept=${CONCEPT}" \
      "model_name_or_path=${CHECKPOINT_ROOT}/final_model" \
      "output_dir=${HOLDOUT_OUTPUT_DIR}" \
      "paths.storage_root=${STORAGE_ROOT}"
  )"
fi

echo "Submitted RWKU eval: ${rwku_job_id}"
echo "Submitted hold-out eval: ${holdout_job_id}"
