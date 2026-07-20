#!/usr/bin/env bash

authors=(
  "Rihanna"
  # "Karl Marx"
  # "Confucius"
  # "Nicolas Cage"
  # "Justin Timberlake"
)

rewards=(
  # "r0"
  # "r1"
  "r2"
  # "r4"
)

seeds=(
  # 12
  1234
  # 42
)

for author in "${authors[@]}"; do
  for reward in "${rewards[@]}"; do
    for seed in "${seeds[@]}"; do
      sbatch scripts/slurm-run-unlearning.sh "model.name=./outputs/unlearning-sft-5240/checkpoint-6/" "experiment.forget_concept=${author}" "experiment.seed=${seed}" "reward.type=${reward}"
    done
  done
done

# sbatch scripts/slurm-run-unlearning.sh "reward.functions.llm-judge.model_name=gpt-5.4-mini" "experiment.forget_concept=${author}" "experiment.seed=1234" "reward.type=r2"

# sbatch scripts/slurm-run-unlearning.sh

# sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=Confucius" "experiment.seed=12" "reward=r2"
# sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=Rihanna" "experiment.seed=12" "reward=r2"
# sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=Confucius" "experiment.seed=12" "reward=r1"
# sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=Confucius" "experiment.seed=12" "reward=r0"
# sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept='Karl Marx'" "experiment.seed=1234" "reward=r0"
# sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept='Nicolas Cage'" "experiment.seed=12" "reward=r0"
# # --------------------------------------------
# 1st R0 runs:
# --------------------------------------------

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" "experiment.seed=12"
# done


# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.grpo.mode=adaptive" \
#     "data_generator.data_proposer.objective=general" \
#     "data_generator.initial_standard_prompt_count=0"
# done


# --------------------------------------------
# 2nd R1 runs:
# --------------------------------------------

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" \
#     "reward=r1" \
#     "experiment.seed=12"
# done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.grpo.mode=adaptive" \
#     "data_generator.initial_standard_prompt_count=0" \
#     "reward=r1" \
#     "data_generator.data_proposer.objective=general"
# done


# --------------------------------------------
# 3rd R2 runs:
# --------------------------------------------

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" \
#     "reward=r2"  \
#     "experiment.seed=12"
# done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.grpo.mode=adaptive" \
#     "data_generator.initial_standard_prompt_count=0" \
#     "reward=r2" \
#     "data_generator.data_proposer.objective=general"
# done