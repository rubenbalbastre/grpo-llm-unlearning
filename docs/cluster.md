# Cluster

The shell launchers target Slurm and activate the `py312_cu118_bis` Conda
environment. Their defaults point to the project cluster paths; override
`ENV_DIR` and `REPO_DIR` when using another checkout or environment root.

## Environment

Create or update the cluster environment with:

```bash
sbatch scripts/env/create-conda-env.sh
```

The environment script installs the CUDA 11.8 PyTorch stack and the Python
dependencies needed by training and evaluation. The launchers expect Conda
under `$ENV_DIR/anaconda3_bis`.

Useful environment variables:

```bash
ENV_DIR=/storage/scratch/<group>/<user>
REPO_DIR=/storage/scratch/<group>/<user>/grpo-llm-unlearning
WANDB_API_KEY=...
WANDB_PROJECT=machine-unlearning-llm
OPENAI_API_KEY=...
HF_TOKEN=...
FASTTEXT_LID_PATH=/path/to/lid.176.ftz
```

Hugging Face clients read `HF_TOKEN` from the process environment. Download the
optional fastText language-ID model with:

```bash
scripts/env/download-fasttext-lid.sh
```

## Matrix Workflow

After generating every target dataset listed in the matrix runner, execute:

```bash
scripts/run-target-reward-matrix.sh
```

This local launcher calls `scripts/run-target-reward-matrix.py`, which submits
the required Slurm jobs and dependencies with `sbatch`. Edit `MODELS`,
`TARGETS`, `REWARDS`, `RUN_RWKU_EVAL`, and `RUN_HOLD_OUT_EVAL` in that Python
file to select the experiment matrix.

The runner does not overwrite partial runs. A run directory with no
`final_model/` is reported as incomplete and must be inspected or removed
manually before resubmission.

## Individual Jobs

```bash
sbatch scripts/generate-data-splits.sh experiment.forget_concept="Stephen King"
sbatch scripts/run-sft.sh experiment.forget_concept="Stephen King"
sbatch scripts/run-grpo.sh experiment.forget_concept="Stephen King"
```

`run-sft.sh` uses one GPU through Accelerate. `run-grpo.sh` selects its
Accelerate configuration from the visible GPU count and PEFT setting:

- one GPU: `config/accelerate_single_gpu.yaml`
- multiple GPUs with PEFT: `config/accelerate_multi_gpu.yaml`
- multiple GPUs without PEFT: `config/accelerate_deepspeed_zero3.yaml`

Override selection with `NUM_GPUS` or `ACCELERATE_CONFIG` when needed.

## Evaluation Jobs

RWKU evaluates only the final model under a training run:

```bash
CHECKPOINT_ROOT=outputs/<run> sbatch scripts/run-eval-rwku.sh
```

The RWKU launcher shards evaluation across the allocated GPUs and merges the
results. Behavioural evaluation uses one GPU:

```bash
sbatch scripts/run-eval-behaviour.sh \
  concept="Stephen King" \
  model_name_or_path=outputs/<run>/final_model \
  output_dir=outputs/<run>/final_model/hold_out_eval
```

Both launchers honor `SKIP_IF_MARKER`; the matrix runner sets it to the run's
`low_reward_stop.json` path so early-stopped no-learning runs are not evaluated.
