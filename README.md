# Machine Unlearning with Adaptive GRPO

This repository prototypes an online adaptive prompt-generation loop for LLM unlearning with Hugging Face TRL's `GRPOTrainer`.

The goal is to keep GRPO unchanged and only modify the rollout prompt source. TRL still owns reward normalization, the GRPO loss, reference/KL handling, gradient accumulation, optimization, logging, and the training loop.

## Formulation

We train a policy model with GRPO on forget-topic prompts. Instead of using only static dataset prompts, an external LLM attacker proposes new prompts online.

At a high level:

```text
recent prompt/completion/reward history
        |
        v
LLM attacker generates candidate forget prompts
        |
        v
each process selects prompts from candidate pool
        |
        v
policy model generates completions
        |
        v
forget-only reward function scores suppression/refusal
        |
        v
TRL GRPO optimization updates the policy
```

The attacker is adaptive because it conditions on recent completions and rewards. Prompts that expose remaining forgotten knowledge can influence future attacker prompts.

## Rollout Hook

The implementation uses TRL's experimental `rollout_func` argument:

```python
GRPOTrainer(..., rollout_func=adaptive_rollout_func)
```

`adaptive_rollout_func` is called during the rollout/data-preparation stage, before rewards and before GRPO loss computation.

It must return:

```python
{
    "prompt_ids": ...,
    "completion_ids": ...,
    "logprobs": ...,
}
```

The function does not replace the GRPO objective. It only decides which prompts are used for policy generation.

## Adaptive Prompt Pool

The attacker is not called on every rollout. Instead, candidates are refreshed once every `N` global steps:

```python
attacker_refresh_every = N
```

At refresh time:

```text
1. each process has local reward history
2. histories are gathered across processes
3. only the main process calls OpenAI
4. the main process generates a candidate prompt pool
5. the candidate pool is broadcast to every process
```

Each process then picks its own slice:

```text
process 0 -> candidates[0:B]
process 1 -> candidates[B:2B]
process 2 -> candidates[2B:3B]
```

where:

```text
B = GRPOConfig.per_device_train_batch_size
```

Therefore `attacker_num_candidates` should be at least:

```text
num_processes * per_device_train_batch_size
```

## Seed Prompts

The training dataset contains a small `"prompt"` column with examples of the forget topic. These prompts are used only as initial context for the attacker.

Once reward history exists, seed prompts are no longer sent to the attacker. This avoids spending tokens on static examples when the attacker can condition directly on observed prompt/completion/reward records.

## Rewards

The current prototype assumes a single prompt type: forget prompts.

The reward is bounded in `[0, 1]`:

```text
refusal/suppression detected -> 1.0
otherwise                    -> 0.0
```

Refusal detection is currently a simple string-pattern heuristic.

Each process stores local `CompletionRecord` objects:

```python
CompletionRecord(prompt, completion, reward, step, metadata)
```

These records are gathered only when the attacker pool refreshes.

## GRPO Training Flow

For each rollout batch:

```text
1. TRL calls adaptive_rollout_func(seed_prompts, trainer)
2. attacker pool refreshes if needed
3. each process selects B prompts from the shared candidate pool
4. each process generates G completions per prompt
5. TRL calls the reward function
6. TRL computes the GRPO loss normally
7. TRL performs gradient accumulation / optimizer steps normally
```

Important knobs:

```text
per_device_train_batch_size -> prompts per process, B
num_generations             -> completions per prompt, G
num_iterations              -> optimization passes over generated rollout data
gradient_accumulation_steps -> batches accumulated before optimizer.step()
max_steps                   -> total optimizer updates
attacker_refresh_every      -> how often OpenAI generates new candidate prompts
attacker_num_candidates     -> size of generated candidate pool
```

`num_iterations` reuses generated rollout data for multiple GRPO optimization iterations. It does not call the attacker or regenerate completions each time.

## Current Prototype

The main script is:

```bash
python train_adaptive_grpo.py
```

It uses:

```text
Qwen/Qwen2.5-0.5B-Instruct
TRL GRPOTrainer
OpenAI Responses API attacker
single forget-topic example: "The capital of France is Paris."
```

The implementation is intentionally minimal. The next natural improvements are stronger reward models, better prompt filtering, and rank-aware logging/checkpointing for larger distributed runs.
