#!/usr/bin/env bash
set -euo pipefail

# Example:
#   SUBJECTS="Karl Marx,Serena Williams" REWARDS="r0,r2" SEEDS="12,1234" bash scripts/run-batch-grpo.sh

SUBJECTS="${SUBJECTS:-Karl Marx}"
REWARDS="${REWARDS:-r4}"
SEEDS="${SEEDS:-1234}"
MODEL_NAME="${MODEL_NAME:-./outputs/unlearning-sft-5259/checkpoint-17/}"

IFS=',' read -r -a subjects <<< "${SUBJECTS}"
IFS=',' read -r -a rewards <<< "${REWARDS}"
IFS=',' read -r -a seeds <<< "${SEEDS}"

for subject in "${subjects[@]}"; do
  for reward in "${rewards[@]}"; do
    for seed in "${seeds[@]}"; do
      sbatch scripts/run-grpo.sh \
        "model.name=${MODEL_NAME}" \
        "experiment.forget_concept=${subject}" \
        "experiment.seed=${seed}" \
        "reward.type=${reward}" \
        "$@"
    done
  done
done
