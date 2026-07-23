#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

LOW_REWARD_STOP_MARKER="low_reward_stop.json"
RUN_EVALS="${RUN_EVALS:-false}"
RUN_SFT_WARMUP="${RUN_SFT_WARMUP:-false}"
STORAGE_ROOT="${STORAGE_ROOT:-.}"
DATA_ROOT="${STORAGE_ROOT}/data"
OUTPUT_ROOT="${STORAGE_ROOT}/outputs"

if [[ -n "${ORIGINAL_MODELS:-}" ]]; then
  IFS=',' read -r -a original_models <<< "${ORIGINAL_MODELS}"
elif [[ -n "${ORIGINAL_MODEL:-}" ]]; then
  original_models=("${ORIGINAL_MODEL}")
else
  original_models=(
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
  )
fi

targets=(
  "Jennifer Lopez"
  "Tony Blair"
  "Marlon Brando"
  "Bruce Lee"
  "Serena Williams"
  "John D. Rockefeller"
  "Tom Clancy"
  "Vincent van Gogh"
  "Karl Marx"
  "Confucius"
)

rewards=(
  "r0"
  "r1"
  # "r2"
  "r4"
)

slug() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

underscore_slug() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

run_evals_enabled() {
  [[ "${RUN_EVALS}" == "1" || "${RUN_EVALS}" == "true" || "${RUN_EVALS}" == "yes" ]]
}

run_sft_warmup_enabled() {
  [[ "${RUN_SFT_WARMUP}" == "1" || "${RUN_SFT_WARMUP}" == "true" || "${RUN_SFT_WARMUP}" == "yes" ]]
}

submit_data_splits() {
  local target="$1"
  local serial_dependency="${2:-}"
  local dataset_dir="${DATA_ROOT}/${target}"
  local dependency_option
  local -a sbatch_args=()

  if [[ -f "${dataset_dir}/dataset_dict.json" \
    && -d "${dataset_dir}/grpo/train" \
    && -d "${dataset_dir}/grpo/test" \
    && -d "${dataset_dir}/sft/train" \
    && -d "${dataset_dir}/sft/test" ]]; then
    echo "done"
    return
  fi
  if [[ -e "${dataset_dir}" ]]; then
    echo "partial"
    return
  fi

  dependency_option="$(dependency_arg "${serial_dependency}")"
  if [[ -n "${dependency_option}" ]]; then
    sbatch_args+=("${dependency_option}")
  fi

  sbatch --parsable \
    "${sbatch_args[@]}" \
    scripts/generate-data-splits.sh \
    "experiment.forget_concept=${target}" \
    "paths.storage_root=${STORAGE_ROOT}"
}

dependency_arg() {
  local dependency
  local -a dependency_ids=()

  for dependency in "$@"; do
    if [[ -n "${dependency}" \
      && "${dependency}" != "done" \
      && "${dependency}" != "partial" \
      && "${dependency}" != "blocked" ]]; then
      dependency_ids+=("${dependency}")
    fi
  done

  if [[ "${#dependency_ids[@]}" -gt 0 ]]; then
    local IFS=":"
    echo "--dependency=afterok:${dependency_ids[*]}"
  fi
}

dependency_is_unavailable() {
  local dependency="$1"

  [[ "${dependency}" == "partial" || "${dependency}" == "blocked" ]]
}

holdout_output_dir() {
  local concept="$1"
  local model_name_or_path="$2"
  local concept_part
  local model_part

  concept_part="$(echo "${concept// /-}" | tr '[:upper:]' '[:lower:]')"
  model_part="${model_name_or_path//\//-}"
  echo "${OUTPUT_ROOT}/hold_out_styles/${concept_part}-${model_part}"
}

submit_sft_warmup() {
  local target="$1"
  local dependency="$2"
  local serial_dependency="${3:-}"
  local run_name="r2warmup_$(underscore_slug "${ORIGINAL_MODEL}")_$(underscore_slug "${target}")"
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  local job_id
  local dependency_option
  local -a sbatch_args=()

  if dependency_is_unavailable "${dependency}"; then
    echo "blocked|${output_dir}/final_model|${run_name}"
    return
  fi

  dependency_option="$(dependency_arg "${dependency}" "${serial_dependency}")"
  if [[ -n "${dependency_option}" ]]; then
    sbatch_args+=("${dependency_option}")
  fi

  if [[ -d "${output_dir}/final_model" ]]; then
    echo "done|${output_dir}/final_model|${run_name}"
    return
  fi

  if [[ -e "${output_dir}" ]]; then
    echo "partial|${output_dir}/final_model|${run_name}"
    return
  fi

  job_id="$(
    sbatch --parsable \
      "${sbatch_args[@]}" \
      scripts/run-sft.sh \
      "wandb.run_name=${run_name}" \
      "model.name=${ORIGINAL_MODEL}" \
      "experiment.forget_concept=${target}" \
      "paths.storage_root=${STORAGE_ROOT}" \
      "training.sft.objective=broad" \
      "reward.type=r2" \
      "training.grpo.save_final_model=true"
  )"

  echo "${job_id}|${output_dir}/final_model|${run_name}"
}

submit_eval_gate() {
  local train_job_id="$1"
  local checkpoint_root="$2"
  local concept="$3"
  local dependency_option
  local rwku_output_dir="${checkpoint_root}/final_model/eval_rwku"
  local holdout_dir
  local -a sbatch_args=()

  holdout_dir="$(holdout_output_dir "${concept}" "${checkpoint_root}/final_model")"

  if [[ "${train_job_id}" == "done" ]]; then
    if [[ -f "${checkpoint_root}/${LOW_REWARD_STOP_MARKER}" ]]; then
      echo "skipped-low-reward"
      return
    fi
    if [[ -f "${rwku_output_dir}/rwku_summary_table.csv" \
      && -f "${holdout_dir}/metrics.csv" \
      && -f "${holdout_dir}/summary.csv" ]]; then
      echo "done"
      return
    fi
  fi

  dependency_option="$(dependency_arg "${train_job_id}")"
  if [[ -n "${dependency_option}" ]]; then
    sbatch_args+=("${dependency_option}")
  fi

  sbatch --parsable \
    "${sbatch_args[@]}" \
    --export="ALL,LOW_REWARD_STOP_MARKER=${LOW_REWARD_STOP_MARKER}" \
    scripts/submit-grpo-evals-if-learning.sh \
    "${checkpoint_root}" \
    "${concept}" \
    "${STORAGE_ROOT}"
}

submit_holdout() {
  local label="$1"
  local target="$2"
  local model_name_or_path="$3"
  local dependency="${4:-}"
  local serial_dependency="${5:-}"
  local dependency_option
  local output_dir
  local -a sbatch_args=()

  output_dir="$(holdout_output_dir "${target}" "${model_name_or_path}")"
  if [[ -f "${output_dir}/metrics.csv" && -f "${output_dir}/summary.csv" ]]; then
    echo "done"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "partial"
    return
  fi

  if dependency_is_unavailable "${dependency}"; then
    echo "blocked"
    return
  fi

  dependency_option="$(dependency_arg "${dependency}" "${serial_dependency}")"
  if [[ -n "${dependency_option}" ]]; then
    sbatch_args+=("${dependency_option}")
  fi

  sbatch --parsable \
    "${sbatch_args[@]}" \
    --job-name="holdout-${label}" \
    --output="logs/holdout-${label}-%j.log" \
    --gres=gpu:1 \
    --time="01:30:00" \
    --partition="sc-gpu" \
    --wrap="cd ${REPO_DIR} && source \$HOME/anaconda3/etc/profile.d/conda.sh && conda activate py312 && python eval/hold_out_styles/generate-and-analyze-completions.py concept='${target}' model_name_or_path='${model_name_or_path}' paths.storage_root='${STORAGE_ROOT}'"
}

submit_grpo_and_evals() {
  local base_label="$1"
  local base_model="$2"
  local model_label="$3"
  local target="$4"
  local reward="$5"
  local dependency="${6:-}"
  local serial_dependency="${7:-}"

  local run_name="unlearning-$(slug "${base_label}")-$(slug "${model_label}")-$(slug "${target}")-${reward}"
  local checkpoint_root="${OUTPUT_ROOT}/${run_name}"
  local train_job_id
  local eval_gate_job_id
  local dependency_option
  local -a sbatch_args=()

  LAST_SUBMITTED_GRPO_JOB_ID=""

  if [[ -n "${dependency}" ]]; then
    if dependency_is_unavailable "${dependency}"; then
      echo "${base_label} | ${target} | ${reward}"
      echo "  train:   blocked by incomplete dependency (${dependency})"
      if run_evals_enabled; then
        echo "  evals:   blocked"
      fi
      return
    fi
    dependency_option="$(dependency_arg "${dependency}" "${serial_dependency}")"
    if [[ -n "${dependency_option}" ]]; then
      sbatch_args+=("${dependency_option}")
    fi
  elif [[ -n "${serial_dependency}" ]]; then
    dependency_option="$(dependency_arg "${serial_dependency}")"
    if [[ -n "${dependency_option}" ]]; then
      sbatch_args+=("${dependency_option}")
    fi
  fi

  if [[ -d "${checkpoint_root}/final_model" ]]; then
    echo "${base_label} | ${target} | ${reward}"
    echo "  train:   already exists (${run_name})"
    if run_evals_enabled; then
      eval_gate_job_id="$(submit_eval_gate "done" "${checkpoint_root}" "${target}")"
      echo "  evals:   ${eval_gate_job_id} (submits RWKU/hold-out only if learning was not low-reward stopped)"
    fi
    return
  fi

  if [[ -e "${checkpoint_root}" ]]; then
    echo "${base_label} | ${target} | ${reward}"
    echo "  train:   incomplete existing output (${run_name})"
    if run_evals_enabled; then
      echo "  evals:   blocked"
    fi
    return
  fi

  train_job_id="$(
    sbatch --parsable \
      "${sbatch_args[@]}" \
      --export="ALL,RUN_NAME=${run_name}" \
      scripts/run-grpo.sh \
      "model.name=${base_model}" \
      "experiment.forget_concept=${target}" \
      "paths.storage_root=${STORAGE_ROOT}" \
      "reward.type=${reward}" \
      "training.grpo.save_final_model=true"
  )"
  LAST_SUBMITTED_GRPO_JOB_ID="${train_job_id}"

  echo "${base_label} | ${target} | ${reward}"
  echo "  train:   ${train_job_id} (${run_name})"
  if run_evals_enabled; then
    eval_gate_job_id="$(submit_eval_gate "${train_job_id}" "${checkpoint_root}" "${target}")"
    echo "  evals:   ${eval_gate_job_id} (submits RWKU/hold-out only if learning was not low-reward stopped)"
  fi
}

previous_sft_job_id=""
previous_r2_grpo_job_id=""
previous_sft_holdout_job_id=""
previous_data_split_job_id=""
declare -A data_job_ids

for target in "${targets[@]}"; do
  data_job_id="$(submit_data_splits "${target}" "${previous_data_split_job_id}")"
  data_job_ids["${target}"]="${data_job_id}"

  echo "Data splits | ${target}"
  if [[ "${data_job_id}" == "done" ]]; then
    echo "  data:    already exists"
  elif [[ "${data_job_id}" == "partial" ]]; then
    echo "  data:    incomplete existing output"
  else
    echo "  data:    ${data_job_id}"
    previous_data_split_job_id="${data_job_id}"
  fi
done

for original_model in "${original_models[@]}"; do
  ORIGINAL_MODEL="${original_model}"
  echo "Original model | ${ORIGINAL_MODEL}"
  echo "Run evals | ${RUN_EVALS}"
  echo "Run SFT warm-up | ${RUN_SFT_WARMUP}"

  for target in "${targets[@]}"; do
    data_job_id="${data_job_ids[${target}]}"

    if run_evals_enabled; then
      original_holdout_job_id="$(submit_holdout "original" "${target}" "${ORIGINAL_MODEL}" "${data_job_id}")"
      echo "Original model hold-out | ${target}"
      if [[ "${original_holdout_job_id}" == "done" ]]; then
        echo "  holdout: already exists (${ORIGINAL_MODEL})"
      elif [[ "${original_holdout_job_id}" == "partial" ]]; then
        echo "  holdout: incomplete existing output (${ORIGINAL_MODEL})"
      elif [[ "${original_holdout_job_id}" == "blocked" ]]; then
        echo "  holdout: blocked by incomplete data split (${ORIGINAL_MODEL})"
      else
        echo "  holdout: ${original_holdout_job_id} (${ORIGINAL_MODEL})"
      fi
    fi

    if run_sft_warmup_enabled; then
      sft_result="$(submit_sft_warmup "${target}" "${data_job_id}" "${previous_sft_job_id}")"
      IFS='|' read -r sft_job_id r2_warmed_model sft_run_name <<< "${sft_result}"
      echo "R2 warm-up | ${target}"
      if [[ "${sft_job_id}" == "done" ]]; then
        echo "  sft:     already exists (${sft_run_name})"
      elif [[ "${sft_job_id}" == "partial" ]]; then
        echo "  sft:     incomplete existing output (${sft_run_name})"
      elif [[ "${sft_job_id}" == "blocked" ]]; then
        echo "  sft:     blocked by incomplete data split (${sft_run_name})"
      else
        echo "  sft:     ${sft_job_id} (${sft_run_name})"
        previous_sft_job_id="${sft_job_id}"
      fi
      echo "  model:   ${r2_warmed_model}"

      if run_evals_enabled; then
        sft_holdout_job_id="$(submit_holdout "sft" "${target}" "${r2_warmed_model}" "${sft_job_id}" "${previous_sft_holdout_job_id}")"
        echo "R2 warm-up hold-out | ${target}"
        if [[ "${sft_holdout_job_id}" == "done" ]]; then
          echo "  holdout: already exists (${r2_warmed_model})"
        elif [[ "${sft_holdout_job_id}" == "partial" ]]; then
          echo "  holdout: incomplete existing output (${r2_warmed_model})"
        elif [[ "${sft_holdout_job_id}" == "blocked" ]]; then
          echo "  holdout: blocked by incomplete SFT warm-up (${r2_warmed_model})"
        else
          echo "  holdout: ${sft_holdout_job_id} (${r2_warmed_model})"
          previous_sft_holdout_job_id="${sft_holdout_job_id}"
        fi
      fi
    else
      echo "R2 warm-up | ${target}"
      echo "  sft:     skipped"
    fi

    for reward in "${rewards[@]}"; do
      if [[ "${reward}" == "r2" ]]; then
        submit_grpo_and_evals "original" "${ORIGINAL_MODEL}" "${ORIGINAL_MODEL}" "${target}" "${reward}" "${data_job_id}" "${previous_r2_grpo_job_id}"
        if [[ -n "${LAST_SUBMITTED_GRPO_JOB_ID:-}" ]]; then
          previous_r2_grpo_job_id="${LAST_SUBMITTED_GRPO_JOB_ID}"
        fi
        if run_sft_warmup_enabled; then
          submit_grpo_and_evals "r2-warmed" "${r2_warmed_model}" "${ORIGINAL_MODEL}" "${target}" "${reward}" "${sft_job_id}" "${previous_r2_grpo_job_id}"
          if [[ -n "${LAST_SUBMITTED_GRPO_JOB_ID:-}" ]]; then
            previous_r2_grpo_job_id="${LAST_SUBMITTED_GRPO_JOB_ID}"
          fi
        fi
      else
        submit_grpo_and_evals "original" "${ORIGINAL_MODEL}" "${ORIGINAL_MODEL}" "${target}" "${reward}" "${data_job_id}"
        if run_sft_warmup_enabled; then
          submit_grpo_and_evals "r2-warmed" "${r2_warmed_model}" "${ORIGINAL_MODEL}" "${target}" "${reward}" "${sft_job_id}"
        fi
      fi
    done
  done
done
