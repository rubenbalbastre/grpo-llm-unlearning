#!/usr/bin/env bash

# authors=(
#   # "Rihanna"
#   # "Karl Marx"
#   # "Confucius"
#   # "Nicolas Cage"
#   # "Justin Timberlake"
# )

# --------------------------------------------
# 1st R0 runs:
# --------------------------------------------

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}"
# done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" "training.mode=adaptive" \
#   "data_generator.data_proposer.objective=general"
# done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.mode=adaptive" \
#     "data_generator.data_proposer.objective=general" \
#     "data_generator.initial_standard_prompt_count=0"
# done

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

# authors=(
#   # "Rihanna"
#   # "Karl Marx"
#   # "Confucius"
#   # "Nicolas Cage"
#   # "Justin Timberlake"
# )

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" \
#     "reward=r1"
# done

authors=(
  "Rihanna"
  "Karl Marx"
  "Confucius"
  "Nicolas Cage"
  "Justin Timberlake"
)

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" "training.mode=adaptive" \
    "reward=r1" \
    "data_generator.data_proposer.objective=general"
done

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh \
    "experiment.forget_concept=${author}" \
    "training.mode=adaptive" \
    "data_generator.initial_standard_prompt_count=0" \
    "reward=r1" \
    "data_generator.data_proposer.objective=general"
done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.mode=adaptive" \
#     "data_generator.data_proposer.objective=general" \
#     "data_generator.initial_standard_prompt_count=0" \
#     "reward=r1"
# done



# --------------------------------------------
# 3rd R2 runs:
# --------------------------------------------

authors=(
  "Rihanna"
  "Karl Marx"
  "Confucius"
  "Nicolas Cage"
  "Justin Timberlake"
)

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" \
    "reward=r2"
done

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh "experiment.forget_concept=${author}" "training.mode=adaptive" \
    "reward=r2" \
    "data_generator.data_proposer.objective=general"
done

for author in "${authors[@]}"; do
  sbatch scripts/slurm-run-unlearning.sh \
    "experiment.forget_concept=${author}" \
    "training.mode=adaptive" \
    "data_generator.initial_standard_prompt_count=0" \
    "reward=r2" \
    "data_generator.data_proposer.objective=general"
done

# for author in "${authors[@]}"; do
#   sbatch scripts/slurm-run-unlearning.sh \
#     "experiment.forget_concept=${author}" \
#     "training.mode=adaptive" \
#     "data_generator.data_proposer.objective=general" \
#     "data_generator.initial_standard_prompt_count=0" \
#     "reward=r2"
# done