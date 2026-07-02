#!/bin/bash -l
#SBATCH --job-name=analyze-completions
#SBATCH --output=logs/analyze-completions-%j.log
#SBATCH --time=01:30:00
#SBATCH --partition=sc-gpu
set -euo pipefail
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate py312

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

# get_experiment_value() {
#     local key="$1"
#     local config_path="$2"

#     awk -v key="${key}" '
#         /^experiment:/ { in_experiment = 1; next }
#         /^[^[:space:]]/ { in_experiment = 0 }
#         in_experiment && $1 == key ":" {
#             sub("^[[:space:]]*" key ":[[:space:]]*", "")
#             print
#             exit
#         }
#     ' "${config_path}"
# }

# slugify() {
#     local separator="$1"
#     local value="$2"

#     printf '%s' "${value}" \
#         | tr '[:upper:]' '[:lower:]' \
#         | sed -E "s/[^a-z0-9]+/${separator}/g; s/^${separator}+//; s/${separator}+$//"
# }

# latest_checkpoint_for_run() {
#     local run_dir="$1"
#     local checkpoint_step

#     checkpoint_step="$(
#         find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
#             | awk -F- '$2 ~ /^[0-9]+$/ { print $2 }' \
#             | sort -n \
#             | tail -n 1
#     )"

#     if [[ -n "${checkpoint_step}" ]]; then
#         printf '%s/checkpoint-%s\n' "${run_dir}" "${checkpoint_step}"
#     fi
# }

# declare -A created_hold_out_datasets=()
# matched_runs=0

# while IFS= read -r hydra_config; do
#     notes="$(get_experiment_value "notes" "${hydra_config}")"
#     if [[ "${notes}" != "apply-chat-template" ]]; then
#         continue
#     fi

#     run_dir="$(dirname "${hydra_config}")"
#     run_id="$(basename "${run_dir}")"
#     concept="$(get_experiment_value "forget_concept" "${hydra_config}")"
#     if [[ -z "${concept}" ]]; then
#         echo "Skipping ${run_id}: missing experiment.forget_concept in ${hydra_config}" >&2
#         continue
#     fi

#     checkpoint_path="$(latest_checkpoint_for_run "${run_dir}")"
#     if [[ -z "${checkpoint_path}" ]]; then
#         echo "Skipping ${run_id}: no checkpoint-* directories found in ${run_dir}" >&2
#         continue
#     fi

#     dataset_slug="$(slugify "_" "${concept}")"
#     output_slug="$(slugify "-" "${concept}")"
#     dataset_path="data/hold_out_${dataset_slug}"
#     output_dir="outputs/completion_analysis/generated-completions-${output_slug}-${run_id}"

#     if [[ -z "${created_hold_out_datasets[${concept}]:-}" ]]; then
#         python scripts/completions_analysis/create_hold_out_dataset.py --concept="${concept}"
#         created_hold_out_datasets["${concept}"]=1
#     fi

#     echo "Analyzing ${run_id}: concept=${concept}, checkpoint=${checkpoint_path}"
#     python scripts/completions_analysis/generate-and-analyze-completions.py \
#         --concept="${concept}" \
#         --model-name-or-path="./${checkpoint_path}" \
#         --dataset-path="${dataset_path}" \
#         --output-dir="${output_dir}"

#     matched_runs=$((matched_runs + 1))
# done < <(find outputs -mindepth 2 -maxdepth 2 -type f -name 'hydra_config.yaml' | sort -V)

# if [[ "${matched_runs}" -eq 0 ]]; then
#     echo "No runs found with hydra.experiment.notes=apply-chat-template" >&2
#     exit 1
# fi

python scripts/completions_analysis/generate-and-analyze-completions.py \
    --concept='Rihanna' \
    --model-name-or-path='outputs/unlearning-4708/checkpoint-49/'
    --dataset-path='data/hold_out_rihanna' \
    --output-dir='outputs/completion_analysis/generated-completions-rihanna-unlearning-4708'


# python scripts/completions_analysis/create_final_table.py
