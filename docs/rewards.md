# Rewards

`reward.type` selects one reward composition from `src/reward/reward_r*.py`.
Component settings live under `reward.functions` in `config/train.yaml`.

## Reward Types

`r0`: simple entity forgetting.

- Uses `simple_match`.
- Rewards completions that avoid target-pattern matches.

`r1`: simple entity forgetting plus non-refusal behavior.

- Combines `simple_match` and the refusal classifier.
- Penalizes refusals while still requiring forgetting.

`r2`: LLM judge reward with refusal prefilter.

- Uses a local refusal classifier first.
- Sends non-refusal completions to the OpenAI-backed judge.
- Refusals receive zero judge reward without an API call.

`r3`: fuzzy length-aware forgetting.

- Uses fuzzy matching over configured target patterns.
- Penalizes target leakage by match density.

`r4`: refusal reward.

- Rewards refusal behavior through the refusal classifier.
- Useful as a comparison condition or control objective.

## Components

`simple_match`:

- Exact case-insensitive matching against target patterns.
- Modes include `binary`, `entity_count`, `exponential`, and `length_aware`.

`fuzzy_match`:

- Normalizes text and applies RapidFuzz partial matching.
- Modes include `binary` and `length_aware`.

`refusal_reward_classifier`:

- Uses `garak-llm/garak-refusal-detector` by default.
- Modes are `reward_refusal` and `reward_non_refusal`.

`llm-judge`:

- Uses the OpenAI Responses API.
- Controlled by `model_name`, `reasoning_effort`, `temperature`, and
  `max_concurrent_requests`.

`language`:

- Uses fastText language ID when `FASTTEXT_LID_PATH` is configured.
- Scores confident English positively and confident non-English negatively.

## Logging

Reward components log per-completion values through TRL/W&B `log_extra` and
aggregate metrics through `log_metric` where implemented. There is no separate
local reward log file.
