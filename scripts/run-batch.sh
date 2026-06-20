#!/usr/bin/env bash

authors=(
  "Rihanna"
  # "Karl Marx"
  # # # "Confucius"
  # "Nicolas Cage"
  # "Justin Timberlake"
)

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}"
# done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" "training.mode=adaptive"
# done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.mode=adaptive" \
#     "data_generator.initial_standard_prompt_count=0"
# done

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh \
    "experiment.forget_concept=${author}" \
    "training.mode=adaptive" \
    "data_generator.data_proposer.objective=general" \
    "data_generator.initial_standard_prompt_count=0"
done