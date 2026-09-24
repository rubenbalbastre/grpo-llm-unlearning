# Training

Training has two stages: an optional SFT initialization and GRPO unlearning.
Both entrypoints use `config/train.yaml` and accept Hydra overrides.

## Configuration

Common fields are:

```yaml
experiment.forget_concept: Serena Williams
experiment.seed: 1234
model.name: Qwen/Qwen2.5-3B-Instruct
reward.type: r2
paths.storage_root: .
wandb.run_name: unlearning-${oc.env:SLURM_JOB_ID}
```

LoRA is enabled under `peft`. Set `peft.enabled=false` for full fine-tuning.
Base models and tokenizers downloaded from Hugging Face are cached under
`outputs/model/` and `outputs/tokenizer/`.

## SFT Initialization

```bash
python sft_warm_up.py \
  experiment.forget_concept="Stephen King" \
  model.name=Qwen/Qwen2.5-3B-Instruct \
  training.sft.save_final_model=true
```

On Slurm:

```bash
sbatch scripts/run-sft.sh \
  experiment.forget_concept="Stephen King" \
  model.name=Qwen/Qwen2.5-3B-Instruct \
  training.sft.save_final_model=true
```

`training.sft.objective=broad` trains on generated broad-topic completions;
`refusal` trains on the RWKU reference completions. The SFT callback periodically
generates validation completions, logs them to W&B, and applies the configured
stopping criteria.

## GRPO

Run directly:

```bash
python train.py \
  experiment.forget_concept="Stephen King" \
  model.name=Qwen/Qwen2.5-3B-Instruct \
  reward.type=r2 \
  training.grpo.save_final_model=true
```

Use an SFT adapter as the warm initialization by setting `model.name` to its
`final_model` directory. `train.py` detects an existing PEFT adapter and makes
it trainable instead of creating a second adapter.

The Slurm launcher runs GRPO through Accelerate:

```bash
RUN_NAME=my-run sbatch scripts/run-grpo.sh \
  experiment.forget_concept="Stephen King" \
  model.name=Qwen/Qwen2.5-3B-Instruct \
  reward.type=r2 \
  training.grpo.save_final_model=true
```

## Experiment Matrix

The standard research workflow is:

```bash
scripts/run-target-reward-matrix.sh
```

`scripts/run-target-reward-matrix.py` defines the models, targets, rewards, and
evaluation switches as module-level constants. For every model and target, it
submits:

- original-model RWKU and behavioural baselines
- one broad-objective SFT warmup
- behavioural evaluation of the SFT warmup
- cold GRPO from the original model for every reward
- warm GRPO from the SFT model for every reward
- enabled RWKU and behavioural evaluations of the GRPO final models

R2 run names include the configured judge reasoning effort, for example
`unlearning-original-qwen-qwen2-5-7b-instruct-karl-marx-r2-reasoning-low`.

## Callbacks

GRPO callback settings live under `training.grpo.callback`:

- `checkpoint_token_milestones`: request checkpoints after token thresholds
- `token_budget`: stop after the configured number of input tokens
- `high_reward_stop`: stop after sustained high reward
- `no_learning_stop`: stop when the first-epoch active reward-group rate is at
  or below the configured threshold

The no-learning callback writes `low_reward_stop.json` in the run directory.
The matrix workflow does not evaluate runs carrying this marker, and final
tables exclude them.

## Outputs

Each run writes under `outputs/<wandb_run_name>/`:

```text
hydra_config.yaml
checkpoint-*/
final_model/                 # when save_final_model is enabled
low_reward_stop.json         # only for no-learning early stops
```

Evaluations are stored inside `final_model/eval_rwku/` and
`final_model/hold_out_eval/`. `hydra_config.yaml` contains the resolved training
configuration, and W&B receives the same configuration when logging is enabled.
