#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/balalru/machine-unlearning-llm}"
cd "${REPO_DIR}"

ORIGINAL_MODEL="${ORIGINAL_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LOW_REWARD_STOP_MARKER="low_reward_stop.json"

targets=(
  "Justin Timberlake"
  "Karl Marx"
  "Nicolas Cage"
)

rewards=(
  "r0"
  "r1"
  "r2"
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

submit_data_splits() {
  local target="$1"

  sbatch --parsable \
    scripts/generate-data-splits.sh \
    "experiment.forget_concept=${target}"
}

submit_sft_warmup() {
  local target="$1"
  local dependency="$2"
  local run_name="r2warmup_$(underscore_slug "${target}")"
  local job_id

  job_id="$(
    sbatch --parsable \
      --dependency="afterok:${dependency}" \
      scripts/run-sft.sh \
      "wandb.run_name=${run_name}" \
      "model.name=${ORIGINAL_MODEL}" \
      "experiment.forget_concept=${target}" \
      "training.sft.objective=broad" \
      "reward.type=r2" \
      "training.grpo.save_final_model=true"
  )"

  echo "${job_id}|outputs/${run_name}/final_model|${run_name}"
}

submit_eval_gate() {
  local train_job_id="$1"
  local checkpoint_root="$2"
  local concept="$3"

  sbatch --parsable \
    --dependency="afterok:${train_job_id}" \
    --export="ALL,LOW_REWARD_STOP_MARKER=${LOW_REWARD_STOP_MARKER}" \
    scripts/submit-grpo-evals-if-learning.sh \
    "${checkpoint_root}" \
    "${concept}"
}

submit_grpo_and_evals() {
  local base_label="$1"
  local base_model="$2"
  local target="$3"
  local reward="$4"
  local dependency="${5:-}"

  local run_name="unlearning-$(slug "${base_label}")-$(slug "${target}")-${reward}"
  local checkpoint_root="outputs/${run_name}"
  local train_job_id
  local eval_gate_job_id
  local -a sbatch_args=()

  if [[ -n "${dependency}" ]]; then
    sbatch_args+=(--dependency="afterok:${dependency}")
  fi

  train_job_id="$(
    sbatch --parsable \
      "${sbatch_args[@]}" \
      --export="ALL,RUN_NAME=${run_name}" \
      scripts/run-grpo.sh \
      "model.name=${base_model}" \
      "experiment.forget_concept=${target}" \
      "reward.type=${reward}" \
      "training.grpo.save_final_model=true"
  )"
  eval_gate_job_id="$(submit_eval_gate "${train_job_id}" "${checkpoint_root}" "${target}")"

  echo "${base_label} | ${target} | ${reward}"
  echo "  train:   ${train_job_id} (${run_name})"
  echo "  evals:   ${eval_gate_job_id} (submits RWKU/hold-out only if learning was not low-reward stopped)"
}

for target in "${targets[@]}"; do
  data_job_id="$(submit_data_splits "${target}")"
  echo "Data splits | ${target}"
  echo "  data:    ${data_job_id}"

  sft_result="$(submit_sft_warmup "${target}" "${data_job_id}")"
  IFS='|' read -r sft_job_id r2_warmed_model sft_run_name <<< "${sft_result}"
  echo "R2 warm-up | ${target}"
  echo "  sft:     ${sft_job_id} (${sft_run_name})"
  echo "  model:   ${r2_warmed_model}"

  for reward in "${rewards[@]}"; do
    submit_grpo_and_evals "original-3b" "${ORIGINAL_MODEL}" "${target}" "${reward}" "${data_job_id}"
    submit_grpo_and_evals "r2-warmed-3b" "${r2_warmed_model}" "${target}" "${reward}" "${sft_job_id}"
  done
done
