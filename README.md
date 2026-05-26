# Machine Unlearning LLM

Minimal GRPO unlearning prototype with one training entrypoint:

- `adaptive`: an OpenAI data generator proposes new prompts from rollout history.
- `standard`: GRPO samples prompts from the RWKU Phi-3 rejection-tuning forget corpus.

## Layout

```text
train.py                      # main training entrypoint
configs/train.yaml
configs/accelerate_single_gpu.yaml
configs/accelerate_multi_gpu.yaml
src/data_generator.py         # GRPO-facing orchestration
src/prompt_buffer.py          # prompt outcomes and selection
src/data_proposer.py          # OpenAI prompt proposal
src/data_filter.py            # embedding/FAISS filtering
src/reward_function.py
src/get_contaminated_data.py
eval/rwku/                    # RWKU evaluation
scripts/slurm-run-unlearning.sh
scripts/slurm-eval-rwku.sh
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your keys. `OPENAI_API_KEY` is required for adaptive data generation. W&B and Hugging Face keys are optional.

## Configure

Main config:

```bash
configs/train.yaml
```

Useful fields:

```yaml
training.mode: adaptive       # adaptive or standard
experiment.forget_concept: Stephen King
model.name: Qwen/Qwen2.5-3B-Instruct
wandb.run_name_prefix: ${training.mode}
data_generator.safe: true
data_generator.new_prompts_per_step: 2
buffer.max_prompts: 512
buffer.high_std_threshold: 0.1
buffer.high_mean_threshold: 0.5
standard_data.config_name: train_refusal_phi3
training.beta: 0.04
```

Override with Hydra:

```bash
python train.py training.mode=adaptive training.max_steps=20
python train.py training.mode=standard training.max_steps=20
```

## Run

Single GPU:

```bash
accelerate launch \
  --config_file configs/accelerate_single_gpu.yaml \
  train.py
```

Multi GPU:

```bash
accelerate launch \
  --config_file configs/accelerate_multi_gpu.yaml \
  --num_processes 2 \
  train.py
```

Both Accelerate configs use `mixed_precision: fp16`.

## Slurm

Edit the `#SBATCH` lines in:

```bash
scripts/slurm-run-unlearning.sh
```

Then submit:

```bash
sbatch scripts/slurm-run-unlearning.sh
```

The script uses `accelerate launch` and counts the GPUs assigned by Slurm through `CUDA_VISIBLE_DEVICES`. It selects the single-GPU config for one visible GPU and the multi-GPU config otherwise.

Override GPU count manually:

```bash
NUM_GPUS=2 sbatch scripts/slurm-run-unlearning.sh
```

Pass Hydra overrides:

```bash
sbatch scripts/slurm-run-unlearning.sh training.max_steps=20
```

## Outputs

Configured in `paths`:

```yaml
paths.output_dir
paths.events_log
paths.final_model_dir
```

Default outputs:

```text
outputs/adaptive_grpo/events.jsonl
outputs/adaptive_grpo/final_model
outputs/standard_grpo/events.jsonl
outputs/standard_grpo/final_model
```

## RWKU Evaluation

```bash
python eval/rwku/rwku.py \
  --model_name_or_path outputs/adaptive_grpo/final_model \
  --output_dir outputs/eval_rwku/stephen_king \
  --subjects "Stephen King"
```

Or with Slurm:

```bash
MODEL_NAME_OR_PATH=outputs/adaptive_grpo/final_model \
SUBJECTS="Stephen King" \
sbatch scripts/slurm-eval-rwku.sh
```

## Notes

- The reward is currently a refusal-pattern heuristic.
- The adaptive generator uses the OpenAI Responses API.
- Safe generation builds a FAISS filter from RWKU contaminated prompts.
