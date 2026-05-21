# Machine Unlearning with Adaptive GRPO

This repository prototypes an online adaptive prompt-generation loop for LLM machine unlearning. It trains a target policy with Hugging Face TRL's `GRPOTrainer` while an external data generator continually proposes new forget-topic prompts from recent prompt, completion, and reward history.

The prototype is intentionally small and research-oriented. The current forget concept is:

```text
The capital of France is Paris.
```

The target model is rewarded for refusing or suppressing the forgotten information when prompted about that concept.

## How It Works

The implementation keeps TRL's GRPO training loop intact and only changes the rollout prompt source through TRL's experimental `rollout_func` hook:

```python
GRPOTrainer(..., rollout_func=adaptive_rollout_func)
```

At a high level:

```text
recent prompt/completion/reward history
        |
        v
OpenAI data generator generates candidate forget prompts
        |
        v
each Accelerate process selects prompts from the shared pool
        |
        v
policy model generates completions
        |
        v
forget-only reward function scores refusal/suppression
        |
        v
TRL GRPO optimization updates the policy
```

TRL still owns reward normalization, the GRPO objective, reference/KL handling, gradient accumulation, optimization, logging, and checkpointing behavior. The adaptive code only chooses which prompts are used for generation.

## Repository Layout

```text
.
├── train_adaptive_grpo.py              # Main adaptive GRPO training prototype
├── data-generator.py                   # Standalone data generator prompt-generation demo
├── test-gpu.py                         # Simple CUDA/GPU availability check
├── requirements.txt                    # Pinned Python dependencies
├── configs/
│   ├── accelerate_single_gpu.yaml      # Accelerate config for one GPU/process
│   └── accelerate_multi_gpu.yaml       # Accelerate config for local multi-GPU runs
└── scripts/
    ├── slurm-create-env.sh             # Slurm venv setup
    ├── slurm-create-conda-env.sh       # Slurm conda setup
    ├── slurm-test-gpu.sh               # Slurm GPU test job
    └── slurm-run-unlearning.sh         # Slurm run template
```

## Setup

Use Python 3.12, then install the pinned dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The training script calls the OpenAI Responses API through the `openai` Python package. Configure an API key before running:

```bash
export OPENAI_API_KEY="..."
```

Optional Weights & Biases logging is enabled when login succeeds. To use it non-interactively:

```bash
export WANDB_API_KEY="..."
```

If no W&B key or login is available, training continues with `report_to=[]`.

## Quick Start

Run the main prototype directly:

```bash
python train_adaptive_grpo.py
```

Or launch it through Accelerate with the checked-in single-GPU config:

```bash
accelerate launch \
  --config_file configs/accelerate_single_gpu.yaml \
  train_adaptive_grpo.py
```

For local multi-GPU training:

```bash
accelerate launch \
  --config_file configs/accelerate_multi_gpu.yaml \
  train_adaptive_grpo.py
```

Before multi-GPU runs, edit `configs/accelerate_multi_gpu.yaml` so `num_processes` matches the number of GPUs/processes you want to launch.

## Configuration Knobs

Training is configured with Hydra in `configs/train_adaptive_grpo.yaml`.

```bash
python train_adaptive_grpo.py \
  experiment.forget_concept="Stephen King" \
  model.name=Qwen/Qwen2.5-3B-Instruct \
  training.max_steps=10
```

Common fields:

```yaml
experiment.forget_concept: Stephen King
paths.output_dir: outputs/adaptive_grpo_min
paths.events_log: ${paths.output_dir}/events.jsonl
paths.final_model_dir: ${paths.output_dir}/final_model
model.name: Qwen/Qwen2.5-3B-Instruct
training.local_batch_size: 4
training.num_generations: 2
training.max_steps: 10
training.learning_rate: 0.000005
data_generator.model_name: gpt-5.4-nano
data_generator.history_size: 16
```

Important relationship:

```text
generated_prompt_count = num_processes * per_device_train_batch_size
```

The main process asks OpenAI for one global batch, broadcasts it, and each process takes its own slice.

## Adaptive Data Generation

On each rollout:

```text
1. each process has local reward history
2. histories are gathered across processes
3. only the main process calls OpenAI
4. the main process generates the global prompt batch
5. the global prompt batch is broadcast to every process
```

Each process then selects its own batch:

```text
process 0 -> generated_prompts[0:B]
process 1 -> generated_prompts[B:2B]
process 2 -> generated_prompts[2B:3B]
```

where `B = GRPOConfig.per_device_train_batch_size`.

## Rewards

The current reward function assumes all prompts are forget prompts. It returns a bounded score in `[0, 1]`:

```text
refusal/suppression detected -> 1.0
otherwise                    -> 0.0
```

Refusal detection is a simple string-pattern heuristic in `is_refusal`. This is useful for a minimal prototype, but it is not a robust unlearning metric.

Each scored completion is stored as:

```python
CompletionRecord(prompt, completion, reward, step, metadata)
```

Recent records feed the adaptive data generator on later refreshes.

## Outputs

Training writes event logs to:

```text
outputs/adaptive_grpo_min/events.jsonl
```

After training completes, the final model is saved to:

```text
outputs/adaptive_grpo_min/final_model
```

The log contains records such as:

```text
data_generator_batch  # generated prompts selected by each process
reward                # prompt/completion/reward records
```

These JSONL events are the easiest place to inspect whether the data generator is adapting and whether the reward heuristic is behaving as expected.

## RWKU Evaluation

Evaluate a saved model on the RWKU generation and locality probes:

```bash
python eval/rwku/rwku.py \
  --model_name_or_path outputs/adaptive_grpo_min/final_model \
  --output_dir outputs/eval_rwku/stephen_king \
  --subjects "Stephen King"
```

The RWKU evaluation code is grouped under `eval/rwku/`, with `rwku.py` as the entrypoint:

```bash
python eval/rwku/rwku.py \
  --model_name_or_path outputs/adaptive_grpo_min/final_model \
  --output_dir outputs/eval_rwku/stephen_king \
  --subjects "Stephen King"
```

The evaluator writes per-example generations to `rwku_generations.jsonl`, MIA scores to `rwku_mia.jsonl`, utility scores to `rwku_utility.jsonl`, aggregate metrics to `rwku_results.json`, and one-row summary tables to `rwku_summary_table.md` and `rwku_summary_table.csv`. Forget split ROUGE-L recall should go down after unlearning; neighbor/locality split ROUGE-L recall should stay high.

Models are loaded with Transformers' native implementations by default. If a model really requires repository-provided code, pass `--trust_remote_code` or set `TRUST_REMOTE_CODE=True` in the Slurm script. For Phi-3 on recent Transformers, keep this disabled so the native loader is used.

By default, MIA evaluation computes only the paper's `Loss` metric. MIA metric selection uses boolean options:

```bash
python eval/rwku/rwku.py \
  --model_name_or_path outputs/adaptive_grpo_min/final_model \
  --output_dir outputs/eval_rwku/stephen_king \
  --subjects "Stephen King" \
  --compute_mia_loss
```

The four RWKU MIA metrics are represented by `--compute_mia_loss`, `--compute_mia_zlib`, `--compute_mia_min_k`, and `--compute_mia_min_k_plus_plus`. Each can be disabled with the matching `--no-...` flag. Currently only `Loss` is implemented.

For MIA, per-example `loss` in `rwku_mia.jsonl` is mean token negative log-likelihood, matching the Hugging Face causal-LM loss used by RWKU's evaluator. The FM/RM summary columns sum that normalized loss within each subject and then average subject sums; for `SUBJECTS="Stephen King"` this is the Stephen King subject sum. The raw unnormalized sequence NLL is still saved as `total_loss` for debugging, but it is not used in the table.

You can run only selected benchmark sets:

```bash
python eval/rwku/rwku.py \
  --model_name_or_path outputs/adaptive_grpo_min/final_model \
  --output_dir outputs/eval_rwku/stephen_king_forget_only \
  --subjects "Stephen King" \
  --run_forget_set \
  --no-run_neighbor_set \
  --no-run_mia_set \
  --no-run_utility_set
```

Evaluate the default Qwen model directly from Hugging Face:

```bash
sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

The Slurm script derives its default output directory from `MODEL_NAME_OR_PATH` and `SUBJECTS`, e.g. `MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct" SUBJECTS="Stephen King"` writes to `outputs/eval_rwku/qwen_qwen2_5_1_5b_instruct_stephen_king`. Use `SUBJECTS=none` for all subjects. Set `OUTPUT_DIR` to override it.

If the model requires authentication, log in with Hugging Face or set your token before running evaluation:

```bash
export HF_TOKEN="..."
sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

To evaluate a local model directory instead:

```bash
MODEL_NAME_OR_PATH=outputs/adaptive_grpo_min/final_model \
MODEL_LABEL=PURGE \
sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

For a quick smoke test:

```bash
MAX_EXAMPLES=5 sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

To run only selected benchmark sets through Slurm:

```bash
RUN_FORGET_SET=True RUN_NEIGHBOR_SET=False RUN_MIA_SET=False RUN_UTILITY_SET=False \
sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

For separate smoke-test caps on MIA and utility:

```bash
MAX_EXAMPLES=5 MAX_MIA_EXAMPLES=5 MAX_UTILITY_EXAMPLES=5 \
sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

The evaluator runs generation in batches. Override the default batch size with:

```bash
BATCH_SIZE=16 sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

MIA likelihood scoring has its own batch-size knob. It defaults to `BATCH_SIZE` unless overridden:

```bash
MIA_BATCH_SIZE=8 sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

Multiple-choice utility splits are scored with batched answer-option likelihoods rather than text generation, matching the RWKU paper's answer-perplexity convention more closely while remaining much faster than free-form decoding.

Utility evaluation has a separate batch-size knob because the paper-aligned prompts can be longer:

```bash
UTILITY_BATCH_SIZE=8 sbatch scripts/eval-qwen-rwku-stephen-king.sh
```

## Utility Scripts

Check CUDA from the current environment:

```bash
python test-gpu.py
```

Run the standalone data generator demo:

```bash
python data-generator.py
```

The standalone demo writes generated prompts to `data_generator_prompts.json` and initializes a W&B Weave project named `machine-unlearning-llm`.

## Slurm

The `scripts/` directory contains cluster templates:

```bash
sbatch scripts/slurm-create-env.sh
sbatch scripts/slurm-create-conda-env.sh
sbatch scripts/slurm-test-gpu.sh
sbatch scripts/slurm-run-unlearning.sh
```

Review the `#SBATCH` directives before submitting. In particular, adjust GPU count, partition, runtime, environment name/path, and the command in `scripts/slurm-run-unlearning.sh`. The run script can execute `data-generator.py` for prompt-generation demos, or `train_adaptive_grpo.py` to launch training.

## Current Limitations

- The reward model is a refusal string heuristic, not a semantic evaluator.
- The data generator uses the OpenAI Responses API and requires `OPENAI_API_KEY`.
- The code depends on TRL's experimental `rollout_func` API, so future TRL changes may require updates.
- The prototype logs useful JSONL events, but distributed checkpoint/log management is still minimal.

Natural next improvements are a stronger reward model, configurable forget-topic datasets, prompt filtering/deduplication, richer evaluation, and rank-aware logging for larger distributed runs.
