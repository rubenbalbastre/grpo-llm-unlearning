# Machine Unlearning LLM

Research code for targeted machine unlearning experiments with SFT warmup,
GRPO training, reward variants, and RWKU / hold-out evaluation.

The current pipeline trains on RWKU-derived prompt splits for one forget concept.
Experiment behavior is configured with Hydra in `config/train.yaml`, and cluster
entrypoints live under `scripts/`.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with the credentials needed for the run. `OPENAI_API_KEY` is used by
OpenAI-backed reward/evaluation utilities and by broad-completion split
generation. W&B and Hugging Face keys are optional unless a script explicitly
logs or pushes artifacts.

Generate data splits:

```bash
python generate_sft_grpo_splits.py experiment.forget_concept="Stephen King"
```

Run SFT warmup:

```bash
python sft_warm_up.py experiment.forget_concept="Stephen King"
```

Run GRPO:

```bash
python train.py experiment.forget_concept="Stephen King"
```

Run RWKU evaluation:

```bash
python -m eval.rwku.rwku evaluation.model_name_or_path=outputs/<run>/checkpoint-175
```

Run hold-out evaluation for one trained checkpoint:

```bash
python eval/hold_out_styles/generate-and-analyze-completions.py \
  concept="Stephen King" \
  model_name_or_path=outputs/<run>/checkpoint-175
```

This writes `metrics.csv` and `summary.csv` under
`outputs/hold_out_styles/<concept>-<model>/`.

Run training-dynamics hold-out analysis from W&B completion tables:

```bash
python eval/hold_out_styles/analyze-training-dynamics.py --download-only --last-steps 5
python eval/hold_out_styles/analyze-training-dynamics.py --skip-download --last-steps 5
```

The first command downloads or reuses the selected W&B tables and writes the run
inventory. The second command scores completions with the hold-out rubrics and
writes aggregated CSVs under `outputs/training_dynamics_hold_out_rubrics/`.

## Documentation

- [Data](docs/data.md): RWKU filtering, split generation, and expected on-disk layout.
- [Training](docs/training.md): SFT, GRPO, checkpointing, Hydra overrides, and outputs.
- [Rewards](docs/rewards.md): reward types `r0` through `r4` and component behavior.
- [Evaluation](docs/evaluation.md): RWKU and hold-out completion analysis.
- [Cluster](docs/cluster.md): Slurm scripts, environment creation, GPU configs, and launch patterns.

## Repository Layout

```text
config/                       # Hydra configs for training, evaluation, Accelerate, DeepSpeed
eval/rwku/                    # RWKU evaluation implementation
eval/hold_out_styles/         # hold-out completion generation and analysis
scripts/                      # Slurm and utility launchers
src/data_preprocessing/       # prompt loading and chat-template rendering
src/reward/                   # reward compositions and reusable reward components
src/callbacks/                # SFT and GRPO training callbacks
train.py                      # GRPO entrypoint
sft_warm_up.py                # SFT warmup entrypoint
generate_sft_grpo_splits.py   # RWKU split generation
```

## License

Original source code and documentation in this repository are licensed under
Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Third-party data is not covered by the repository code license. PURGE material
is explicitly attributed as MIT-licensed third-party data from the PURGE
project.
