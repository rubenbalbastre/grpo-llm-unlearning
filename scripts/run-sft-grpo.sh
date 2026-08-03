#!/bin/bash -l
set -euo pipefail

# Example:
#   REWARDS="r2,r4" scripts/run-sft-grpo.sh experiment.forget_concept="Karl Marx"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "${SCRIPT_DIR}")}"
SFT_SCRIPT="${SFT_SLURM_SCRIPT:-${REPO_DIR}/scripts/run-sft.sh}"
GRPO_SCRIPT="${GRPO_SLURM_SCRIPT:-${REPO_DIR}/scripts/run-grpo.sh}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-unlearning}"
REWARDS="${REWARDS:-r4,r2}"

cd "${REPO_DIR}"

SFT_JOB_ID="$(sbatch --parsable "${SFT_SCRIPT}" "$@" "training.grpo.save_final_model=true")"
SFT_RUN_NAME="${RUN_NAME_PREFIX}-sft-${SFT_JOB_ID}"
SFT_MODEL_PATH="./outputs/${SFT_RUN_NAME}/final_model"

echo "Submitted SFT job: ${SFT_JOB_ID}"
echo "SFT model path for GRPO: ${SFT_MODEL_PATH}"

IFS=',' read -r -a rewards <<< "${REWARDS}"
for reward in "${rewards[@]}"; do
  grpo_job_id="$(
    sbatch --parsable \
      --dependency="afterok:${SFT_JOB_ID}" \
      "${GRPO_SCRIPT}" \
      "$@" \
      "model.name=${SFT_MODEL_PATH}" \
      "reward.type=${reward}"
  )"
  echo "Submitted GRPO job: ${grpo_job_id} (reward=${reward}, dependency=afterok:${SFT_JOB_ID})"
done
