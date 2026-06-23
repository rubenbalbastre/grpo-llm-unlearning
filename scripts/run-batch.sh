#!/usr/bin/env bash

authors=(
  "Rihanna"
  "Karl Marx"
  # "Confucius"
  "Nicolas Cage"
  "Justin Timberlake"
)

# --------------------------------------------
# 1st R0 runs:
# --------------------------------------------

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" "experiment.seed=12"
done


# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.mode=adaptive" \
#     "data_generator.data_proposer.objective=general" \
#     "data_generator.initial_standard_prompt_count=0"
# done


# --------------------------------------------
# 2nd R1 runs:
# --------------------------------------------

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" \
    "reward=r1" \
    "experiment.seed=12"
done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.mode=adaptive" \
#     "data_generator.initial_standard_prompt_count=0" \
#     "reward=r1" \
#     "data_generator.data_proposer.objective=general"
# done


# --------------------------------------------
# 3rd R2 runs:
# --------------------------------------------

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" \
    "reward=r2"  \
    "experiment.seed=12"
done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.mode=adaptive" \
#     "data_generator.initial_standard_prompt_count=0" \
#     "reward=r2" \
#     "data_generator.data_proposer.objective=general"
# done
