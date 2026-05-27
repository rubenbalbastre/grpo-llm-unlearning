# Machine Unlearning LLM

Minimal GRPO unlearning prototype with one training entrypoint:

- `adaptive`: an OpenAI data generator proposes new prompts from rollout history.
- `standard`: GRPO samples prompts from the RWKU Phi-3 rejection-tuning forget corpus.

## Layout

```text
train.py                      # main training entrypoint
config/train.yaml
config/eval.yaml                 # post-training RWKU evaluation config
config/accelerate_single_gpu.yaml
config/accelerate_multi_gpu.yaml
src/data_generator/
  data_generator.py           # GRPO-facing orchestration
  prompt_buffer.py            # prompt outcomes and selection
  data_proposer.py            # OpenAI prompt proposal
  data_filter.py              # embedding/FAISS filtering
  get_contaminated_data.py
src/reward_function.py
eval/rwku/                    # RWKU evaluation
scripts/slurm-run-unlearning.sh
scripts/slurm-eval-rwku.sh
scratch/                      # temporary exploratory probes
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
config/train.yaml
```

Useful fields:

```yaml
training.mode: adaptive       # adaptive or standard
experiment.forget_concept: Stephen King
model.name: Qwen/Qwen2.5-3B-Instruct
wandb.run_name_prefix: ${training.mode}
data_generator.safe: true
data_generator.generation_context_size: 64
data_generator.new_prompts_per_step: 2
buffer.max_prompts: 512
buffer.high_reward_std_threshold: 0.1
buffer.high_mean_reward_threshold: 0.75
reward.mode: entity_count # entity_count or binary
standard_data.config_name: train_refusal_phi3
standard_data.dataset_size: 100 # limit after filtering by forget_concept
training.num_epochs: 1 # standard mode only; use with training.max_steps=-1
training.beta: 0.04
```

Override with Hydra:

```bash
python train.py training.mode=adaptive training.max_steps=20
python train.py training.mode=standard standard_data.dataset_size=100 training.num_epochs=2
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

After training finishes successfully, this script evaluates the saved final
model on RWKU using `config/eval.yaml`. Set `evaluation.model_name_or_path`,
`evaluation.output_dir`, and `evaluation.subjects` there to match the training
run being evaluated.

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

The unlearning Slurm job automatically evaluates its final model according to:

```bash
config/eval.yaml
```

Select evaluation sets and MIA metrics there:

```yaml
evaluation:
  model_name_or_path: outputs/adaptive_grpo/final_model
  output_dir: outputs/adaptive_grpo/eval_rwku
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
  wandb:
    enabled: true
    project: machine-unlearning-llm
    run_name: rwku-${evaluation.subjects}
    artifact_name: rwku-stephen-king-model
    log_model_artifact: true
    link_to_training_run: true
```

`rouge_l_recall` is currently the implemented forget/neighbor generation
metric. Only MIA `loss` is implemented; enabling the other MIA metric switches
currently raises `NotImplementedError`.
Hugging Face and W&B authentication for evaluation are read from
`HUGGINGFACE_HUB_TOKEN` and `WANDB_API_KEY` in `.env`. With
`link_to_training_run: true`, training writes its W&B run identity into the
saved model directory and post-training evaluation resumes that same run to
log `rwku/*` metrics and a model artifact containing the model and RWKU result
files. Set it to `false` when evaluating a model without training-run metadata.

The standalone evaluation Slurm job reads the same configuration file:
`scripts/slurm-eval-rwku.sh`.

## Notes

- Set `reward.mode` to `entity_count` for a graded entity-leakage reward, or `binary` for the earlier all-or-nothing reward.
- The adaptive generator uses the OpenAI Responses API.
- Safe generation builds a FAISS filter from RWKU contaminated prompts.
