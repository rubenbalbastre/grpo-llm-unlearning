# Machine Unlearning LLM

Minimal prototype for adaptive GRPO unlearning.

The main training script is `train_adaptive_grpo.py`. It trains with TRL `GRPOTrainer` while a data generator proposes new prompts from recent prompt, completion, and reward history. Safe generation can filter prompts that are too close to protected RWKU reference data.

## Layout

```text
train_adaptive_grpo.py        # main training entrypoint
configs/train_adaptive_grpo.yaml
configs/accelerate_single_gpu.yaml
configs/accelerate_multi_gpu.yaml
src/data_generator.py         # GRPO-facing orchestration
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
configs/train_adaptive_grpo.yaml
```

Useful fields:

```yaml
experiment.forget_concept: Stephen King
paths.output_dir: outputs/adaptive_grpo
model.name: Qwen/Qwen2.5-3B-Instruct
data_generator.safe: true
data_generator.model_name: gpt-5.4-nano
training.beta: 0.04
training.max_steps: 10
```

Override with Hydra:

```bash
python train_adaptive_grpo.py training.max_steps=20 experiment.forget_concept="Stephen King"
```

## Run

Single GPU:

```bash
accelerate launch \
  --config_file configs/accelerate_single_gpu.yaml \
  train_adaptive_grpo.py
```

Multi GPU:

```bash
accelerate launch \
  --config_file configs/accelerate_multi_gpu.yaml \
  --num_processes 2 \
  train_adaptive_grpo.py
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
- `train_grpo.py` is experimental and not the main entrypoint.
