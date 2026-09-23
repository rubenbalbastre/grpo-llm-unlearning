# Cluster

Cluster launchers assume a Slurm environment and source `scripts/slurm-env.sh`.
That helper resolves repository paths and activates the training environment.

## Environment

Create or update the cluster environment with:

```bash
sbatch scripts/others/create-conda-env.sh
```

Useful environment variables:

```bash
REPO_DIR=/storage/scratch/<group>/<user>/machine-unlearning-llm
TRAIN_ENV_PATH=/storage/scratch/<group>/<user>/anaconda3_bis/envs/py312_cu118_bis_cu118
WANDB_PROJECT=machine-unlearning-llm
OPENAI_API_KEY=...
HF_TOKEN=...
FASTTEXT_LID_PATH=/path/to/lid.176.ftz
```

Hugging Face clients read the token directly from the process environment. The
training jobs do not write it to the shared Hugging Face credential cache.

Download fastText language ID data with:

```bash
scripts/others/download-fasttext-lid.sh
```

## Training Jobs

Generate splits:

```bash
sbatch scripts/generate-data-splits.sh experiment.forget_concept="Stephen King"
```

SFT:

```bash
sbatch scripts/run-sft.sh experiment.forget_concept="Stephen King"
```

GRPO:

```bash
sbatch scripts/run-grpo.sh experiment.forget_concept="Stephen King"
```

Run the complete target/reward matrix, including SFT warm-up, GRPO, RWKU, and
hold-out evaluation:

```bash
scripts/run-target-reward-matrix.sh
```

The generated target datasets must already exist under `data/`. The model,
target, and reward matrix is defined in `scripts/run-target-reward-matrix.py`.

## GPU Configuration

`scripts/run-grpo.sh` selects the Accelerate config based on visible GPUs and
PEFT mode:

- one GPU: `config/accelerate_single_gpu.yaml`
- multiple GPUs with PEFT: `config/accelerate_multi_gpu.yaml`
- multiple GPUs without PEFT: `config/accelerate_deepspeed_zero3.yaml`

Override GPU count manually:

```bash
NUM_GPUS=2 sbatch scripts/run-grpo.sh
```

Force a config:

```bash
ACCELERATE_CONFIG=config/accelerate_multi_gpu.yaml sbatch scripts/run-grpo.sh
```

## Evaluation Jobs

RWKU:

```bash
CHECKPOINT_ROOT=outputs/<run> sbatch scripts/run-eval-rwku.sh
```

Hold-out completion analysis:

```bash
sbatch scripts/run-eval-behaviour.sh
```

Training diagnostics:

```bash
sbatch scripts/others/slurm-plot-training-diagnostics-by-model-size.sh
```
