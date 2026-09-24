# Rewards

`reward.type` selects `src/reward/reward_<type>.py`. Shared component settings
are defined under `reward.functions` in `config/train.yaml`.

## Reward Types

`r0` uses case-insensitive target-pattern matching and rewards completions that
avoid configured target entities.

`r1` combines target avoidance with non-refusal behavior. Its current score is
`0.4 * forgetting + 0.6 * non_refusal * forgetting`, so a completion must avoid
the target before non-refusal behavior helps.

`r2` uses an OpenAI-backed LLM judge. A local refusal classifier first filters
the completions; refusals receive zero without an API request, while
non-refusals are judged for target leakage and broad-topic helpfulness.

`r3` uses fuzzy target matching with the configured length-aware behavior.

`r4` directly rewards refusal according to the local refusal classifier. It is
used as a comparison objective.

The standard matrix currently runs `r0`, `r1`, `r2`, and `r4`; `r3` remains
available for individual experiments.

## Components

`simple_match` performs case-insensitive matching against target patterns. Its
modes include `binary`, `entity_count`, `exponential`, and `length_aware`.

`fuzzy_match` normalizes text and uses RapidFuzz partial matching. It supports
`binary` and `length_aware` modes.

`refusal_reward_classifier` uses
`garak-llm/garak-refusal-detector` by default and can reward either refusal or
non-refusal.

`llm-judge` uses the OpenAI Responses API. Its model, reasoning effort,
temperature, and request concurrency are configured in `config/train.yaml`.

`language` uses fastText language identification when `FASTTEXT_LID_PATH` is
configured. Short or uncertain completions receive a neutral score.

## Requirements and Logging

R2 requires `OPENAI_API_KEY`. The refusal component downloads its classifier
from Hugging Face unless already cached. The language component requires the
fastText model path only when used.

Reward components log per-completion values and aggregate metrics through TRL
and W&B. They do not create a separate local reward log file.
