#!/bin/bash -l
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

SFT_SCRIPT="${SFT_SLURM_SCRIPT:-${REPO_DIR}/scripts/submit-sft-warm-up.sh}"
GRPO_SCRIPT="${GRPO_SLURM_SCRIPT:-${REPO_DIR}/scripts/slurm-run-unlearning.sh}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-unlearning}"

SFT_JOB_ID="$(sbatch --parsable "${SFT_SCRIPT}" "$@" "training.grpo.save_final_model=true")"
SFT_RUN_NAME="${RUN_NAME_PREFIX}-sft-${SFT_JOB_ID}"
SFT_MODEL_PATH="./outputs/${SFT_RUN_NAME}/final_model"

GRPO_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${SFT_JOB_ID}" \
    "${GRPO_SCRIPT}" \
    "$@" \
    "model.name=${SFT_MODEL_PATH}" \
    "reward.type=r4"
)"

echo "Submitted SFT job: ${SFT_JOB_ID}"
echo "SFT run name: ${SFT_RUN_NAME}"
echo "SFT model path for GRPO: ${SFT_MODEL_PATH}"
echo "Submitted GRPO job: ${GRPO_JOB_ID}"
echo "GRPO dependency: afterok:${SFT_JOB_ID}"

GRPO_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${SFT_JOB_ID}" \
    "${GRPO_SCRIPT}" \
    "$@" \
    "model.name=${SFT_MODEL_PATH}" \
    "reward.type=r2"
)"

echo "Submitted SFT job: ${SFT_JOB_ID}"

GRPO_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${SFT_JOB_ID}" \
    "${GRPO_SCRIPT}" \
    "$@" \
    "model.name=${SFT_MODEL_PATH}" \
    "reward.type=r0"
)"
echo "Submitted GRPO job: ${GRPO_JOB_ID}"

GRPO_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${SFT_JOB_ID}" \
    "${GRPO_SCRIPT}" \
    "$@" \
    "model.name=${SFT_MODEL_PATH}" \
    "reward.type=r1"
)"
echo "Submitted GRPO job: ${GRPO_JOB_ID}"