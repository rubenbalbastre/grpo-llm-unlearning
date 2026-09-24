# Machine Unlearning LLM

Research code for targeted machine unlearning with SFT initialization, GRPO
training, and RWKU-derived evaluation. Experiments compare four Qwen2.5 model
sizes, multiple reward functions, cold and warm initialization, and ten target
entities.

Configuration is managed with Hydra. Training and evaluation outputs are stored
under `outputs/`, while the Slurm launchers live under `scripts/`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set the credentials required by the workflow:

- `OPENAI_API_KEY`: broad SFT completions, the R2 reward, and behavioural judges
- `WANDB_API_KEY` and `WANDB_PROJECT`: experiment tracking and training dynamics
- `HF_TOKEN`: private Hugging Face dataset upload/download
- `FASTTEXT_LID_PATH`: optional fastText language identification model

## Data

Generate the RWKU-derived splits for one target:

```bash
python generate_sft_grpo_splits.py experiment.forget_concept="Stephen King"
```

This creates `grpo/train`, `grpo/test`, `sft/train`, and `sft/test` under
`data/<target>/`. Split generation calls OpenAI to produce the broad-topic SFT
completions.

## Experiments

The main cluster entrypoint submits the complete experiment matrix:

```bash
scripts/run-target-reward-matrix.sh
```

The model, target, and reward lists are defined at the top of
`scripts/run-target-reward-matrix.py`. The runner submits original-model
baselines, SFT warmups, cold and warm GRPO runs, and enabled evaluations. It
reuses completed outputs and skips evaluation for runs marked with
`low_reward_stop.json`.

Run individual stages when debugging or conducting a one-off experiment:

```bash
python sft_warm_up.py experiment.forget_concept="Stephen King"
python train.py experiment.forget_concept="Stephen King"
```

The corresponding Slurm launchers are `scripts/run-sft.sh` and
`scripts/run-grpo.sh`.

## Evaluation

Evaluate the final model of a completed run:

```bash
CHECKPOINT_ROOT=outputs/<run> sbatch scripts/run-eval-rwku.sh

sbatch scripts/run-eval-behaviour.sh \
  concept="Stephen King" \
  model_name_or_path=outputs/<run>/final_model \
  output_dir=outputs/<run>/final_model/hold_out_eval
```

RWKU results are written to `final_model/eval_rwku/`. Behavioural evaluation
writes prompt-level `metrics.csv` and aggregate `summary.csv` to
`final_model/hold_out_eval/`.

Create the article tables after evaluations are complete:

```bash
scripts/tables/final-table.sh
```

This writes author-level and aggregate RWKU and behavioural CSV files under
`outputs/tables/`, then packages them as `final-tables.tar.gz`.

## Documentation

- [Data](docs/data.md): split generation, local layout, and Hugging Face transfer
- [Training](docs/training.md): SFT, GRPO, matrix execution, callbacks, and outputs
- [Rewards](docs/rewards.md): reward functions `r0`, `r1`, `r2`, and `r4`
- [Evaluation](docs/evaluation.md): RWKU, behavioural evaluation, and final tables
- [Cluster](docs/cluster.md): Slurm environment and launchers

## Repository Layout

```text
config/                       Hydra, Accelerate, and DeepSpeed configuration
docs/                         research workflow documentation
eval/rwku/                    RWKU benchmark implementation and tables
eval/behaviour/               hold-out evaluation and training-dynamics analysis
scripts/                      Slurm launchers and utilities
src/callbacks/                SFT and GRPO callbacks
src/data_preprocessing/       dataset loading and prompt formatting
src/reward/                   reward compositions and components
generate_sft_grpo_splits.py   RWKU-derived split generation
sft_warm_up.py                SFT entrypoint
train.py                      GRPO entrypoint
```

## License

Original source code and documentation are licensed under Apache-2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).

Third-party data is not covered by the code license. Exported datasets are
released under CC BY 4.0 with RWKU attribution and a modified/processed-data
notice. PURGE material is attributed separately as MIT-licensed third-party
data.
