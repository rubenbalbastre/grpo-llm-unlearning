# Training

Training is split into an optional SFT warmup and a GRPO stage. Both use
`config/train.yaml`.

## Configuration

Commonly changed fields:

```yaml
experiment.forget_concept: Stephen King
experiment.seed: 42
model.name: Qwen/Qwen2.5-3B-Instruct
reward.type: r2
paths.storage_root: .
wandb.run_name: unlearning-${oc.env:SLURM_JOB_ID}
```

LoRA is controlled under `peft`:

```yaml
peft.enabled: true
peft.lora.number_layers_to_transform: -1
```

Set `peft.enabled=false` for full fine-tuning.

## SFT Warmup

Local:

```bash
python sft_warm_up.py experiment.forget_concept="Stephen King"
```

Slurm:

```bash
sbatch scripts/run-sft.sh experiment.forget_concept="Stephen King"
```

`training.sft.objective` is either:

- `broad`: train on `broad_completion`
- `refusal`: train on `completion`

## GRPO

Local:

```bash
python train.py experiment.forget_concept="Stephen King"
```

With Accelerate:

```bash
accelerate launch \
  --config_file config/accelerate_single_gpu.yaml \
  train.py
```

Slurm:

```bash
sbatch scripts/run-grpo.sh experiment.forget_concept="Stephen King"
```

To continue from an SFT warmup model:

```bash
python train.py model.name=outputs/<sft-run>/final_model reward.type=r2
```

## Checkpointing

`training.grpo.save_final_model` controls whether `final_model` is written.
Checkpoint snapshots are written by the trainer during training.

Token-budget checkpointing is configured under:

```yaml
training.grpo.callback:
  checkpoint_token_milestones: null
  token_budget: null
```

Example:

```bash
RUN_NAME=grpo-6m sbatch scripts/run-grpo.sh \
  training.grpo.save_final_model=false \
  training.grpo.callback.token_budget=6_000_000 \
  'training.grpo.callback.checkpoint_token_milestones=[1_500_000,3_000_000,4_500_000,6_000_000]'
```

## Outputs

Each training run writes to:

```text
outputs/<wandb_run_name>/hydra_config.yaml
outputs/<wandb_run_name>/checkpoint-*
outputs/<wandb_run_name>/final_model
```

`hydra_config.yaml` is the resolved configuration for reproducibility. When W&B
is enabled, the same config is logged under `hydra`.
