# Machine Unlearning LLM

Minimal GRPO unlearning prototype with one training entrypoint. GRPO samples
prompts from the RWKU Phi-3 rejection-tuning forget corpus.

## Layout

```text
train.py                      # main training entrypoint
config/train.yaml
config/eval.yaml                 # post-training RWKU evaluation config
config/accelerate_single_gpu.yaml
config/accelerate_multi_gpu.yaml
config/accelerate_deepspeed_zero3.yaml
src/reward/
  reward_r*.py               # reward compositions
  components/                # reusable reward components
eval/rwku/                    # RWKU evaluation
scripts/run-grpo.sh
scripts/eval-rwku.sh
scratch/                      # temporary exploratory probes
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your keys. `OPENAI_API_KEY` is required for OpenAI-backed
reward/evaluation utilities. W&B and Hugging Face keys are optional.

## Configure

Main config:

```bash
config/train.yaml
```

Useful fields:

```yaml
experiment.forget_concept: Stephen King
experiment.seed: 42
model.name: Qwen/Qwen2.5-1.5B-Instruct
reward: r0 # r0/r1/r2 select reward_weights; reward functions stay in config/train.yaml
reward.functions.simple_match.mode: entity_count # optional override; entity_count or binary
reward.functions.language.model_path: ${oc.env:FASTTEXT_LID_PATH,null}
standard_data.config_name: train_refusal_phi3
standard_data.dataset_size: 100 # limit after filtering by forget_concept
training.grpo.num_epochs: 1 # use with training.grpo.max_steps=-1
training.grpo.callback.checkpoint_token_milestones: null # e.g. [1_500_000, 3_000_000, 4_500_000]
peft.enabled: true # false for full fine-tuning
peft.lora.number_layers_to_transform: -1 # all transformer layers
training.grpo.beta: 0.04
```

Override with Hydra:

```bash
python train.py standard_data.dataset_size=100 training.grpo.num_epochs=2
python train.py peft.enabled=false # full fine-tuning
```

To evaluate benchmark evolution across a longer run, save token checkpoints:

```bash
RUN_NAME=grpo-6m sbatch scripts/run-grpo.sh \
  training.grpo.save_final_model=false \
  training.grpo.callback.token_budget=6_000_000 \
  'training.grpo.callback.checkpoint_token_milestones=[1_500_000,3_000_000,4_500_000,6_000_000]'
```

## Run

Single GPU:

```bash
accelerate launch \
  --config_file config/accelerate_single_gpu.yaml \
  train.py
```

Multi GPU:

```bash
accelerate launch \
  --config_file config/accelerate_multi_gpu.yaml \
  --num_processes 2 \
  train.py
```

Full fine-tuning with DeepSpeed ZeRO-3 on 2 GPUs:

```bash
accelerate launch \
  --config_file config/accelerate_deepspeed_zero3.yaml \
  --num_processes 2 \
  train.py peft.enabled=false
```

The training defaults target H100 GPUs: models load in BF16 with
FlashAttention 2, Accelerate uses BF16 mixed precision, and GRPO uses colocated
vLLM generation. The ZeRO-3 path uses BF16 through
`config/deepspeed_zero3_bf16.json`.

## Slurm

Edit the `#SBATCH` lines in:

```bash
scripts/run-grpo.sh
```

Then submit:

```bash
sbatch scripts/run-grpo-h100.sh
```

The script uses `accelerate launch` and counts the GPUs assigned by Slurm through `CUDA_VISIBLE_DEVICES`. It selects the single-GPU config for one visible GPU, the multi-GPU config for PEFT on multiple GPUs, and the ZeRO-3 config for multi-GPU full fine-tuning:

```bash
sbatch scripts/run-grpo.sh peft.enabled=false
```

Override GPU count manually:

```bash
NUM_GPUS=2 sbatch scripts/run-grpo.sh
```

Pass Hydra overrides:

```bash
sbatch scripts/run-grpo.sh training.grpo.max_steps=20
```

After training finishes successfully, the run artifacts are written under
`outputs/<wandb_run_name>`. Use `scripts/eval-rwku.sh` or
`scripts/submit-train-and-eval.sh` to evaluate saved checkpoints.

## Outputs

Configured in `paths`:

```yaml
paths.output_root: outputs
```

Each training run uses the configured W&B name as its output folder:

```text
outputs/<wandb_run_name>/events.jsonl
outputs/<wandb_run_name>/hydra_config.yaml
outputs/<wandb_run_name>/checkpoint-*
outputs/<wandb_run_name>/final_model # only when training.grpo.save_final_model=true
```

Training writes the resolved Hydra config to `hydra_config.yaml`. When W&B is
enabled, the same config is logged under the run config key `hydra`, and the
YAML file is saved to the W&B run.
Set `training.grpo.save_final_model: false` to keep only the checkpoint snapshots,
which is useful when the token-budget checkpoint already captures the end of
training.grpo. In that setup, include the budget in
`training.grpo.callback.checkpoint_token_milestones`; `token_budget` only stops the
run and does not create an extra checkpoint.

Under `scripts/run-grpo.sh`, the shell script passes a per-job
`wandb.run_name` such as `unlearning-48291`, so each submitted job gets its own
local output folder and evaluation directory. Override `RUN_NAME` for a more
descriptive folder:

```bash
RUN_NAME=standard-natural-48291 sbatch scripts/run-grpo.sh
```

For separate local runs, override the default `-local` suffix explicitly:

```bash
python train.py wandb.run_name=standard-grpo-dev-001
```

## RWKU Evaluation

RWKU evaluation is configured by:

```bash
config/eval.yaml
```

Select evaluation sets and MIA metrics there:

```yaml
evaluation:
  model_name_or_path: outputs/<wandb_run_name>/checkpoint-175
  output_dir: outputs/<wandb_run_name>/checkpoint-175/eval_rwku
  subjects: Stephen King
  sets:
    forget: true
    neighbor: true
    mia: true
    utility: false
  metrics:
    generation: rouge_l_recall
    mia:
      loss: true
      zlib: false
      min_k: false
      min_k_plus_plus: false
  limits:
    max_utility_examples: 2000
    utility_sample_strategy: random
    utility_sample_seed: 42
  wandb:
    enabled: true
    run_name: rwku-${evaluation.subjects}
    artifact_name: rwku-stephen-king-model
    log_model_artifact: true
    link_to_training_run: true
```

Forget, neighbor, and MIA sets are always evaluated in full after optional
subject filtering. Utility can be capped with `max_utility_examples`;
`utility_sample_strategy: random` samples a fixed-seed subset before sharding,
while `first` preserves first-N behavior for utility only.

`rouge_l_recall` is currently the implemented forget/neighbor generation
metric. Only MIA `loss` is implemented; enabling the other MIA metric switches
currently raises `NotImplementedError`.
Hugging Face and W&B configuration for evaluation are read from
`HUGGINGFACE_HUB_TOKEN`, `WANDB_API_KEY`, and `WANDB_PROJECT` in `.env`. With
`link_to_training_run: true`, training writes its W&B run identity into the
run directory as `outputs/<wandb_run_name>/wandb_run.json`, and
post-training evaluation resumes that same run to log `rwku/*` metrics and a
model artifact containing the model and RWKU result files. For `final_model`
or `checkpoint-*` eval paths, RWKU derives the run directory from the parent
folder. Set it to `false` when evaluating a model without training-run
metadata.

The standalone evaluation Slurm job reads the same configuration file:
`scripts/eval-rwku.sh`.
When `training.grpo.save_final_model: false`, evaluate checkpoints with
`CHECKPOINT_ROOT=outputs/<wandb_run_name> scripts/eval-rwku.sh` rather
than pointing `evaluation.model_name_or_path` at `final_model`.

To evaluate every saved training checkpoint under a run directory, set
`CHECKPOINT_ROOT`. The script evaluates each `checkpoint-*` directory
sequentially, writes results inside that checkpoint directory, and creates one
W&B evaluation run/model artifact per checkpoint. Metric names remain `rwku/*`.
Each eval run config includes the checkpoint label, `global_step`, and
`num_input_tokens_seen` read from `trainer_state.json`, the configured
checkpoint milestone token target, plus the training Hydra config loaded from
`hydra_config.yaml` under the same W&B config key `hydra` used by training runs.

```bash
CHECKPOINT_ROOT=outputs/grpo-6m sbatch scripts/eval-rwku.sh
```

Set `INCLUDE_FINAL_MODEL=true` to also evaluate `final_model` after the
checkpoints.

## Notes

- Set `reward.mode` to `entity_count` for a graded entity-leakage reward, or `binary` for the earlier all-or-nothing reward.
