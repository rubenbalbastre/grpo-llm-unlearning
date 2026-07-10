from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.regex_refusal import build_regex_refusal_reward


REFUSAL_WEIGHT = 0.3 # 0.8
BREVITY_WEIGHT = 0.7 # 0.2
BREVITY_MIN_TOKENS = 2
BREVITY_TARGET_TOKENS = 24
BREVITY_TAU = 160.0 #80.0


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def count_completion_tokens(completion_ids: Any, completion: str) -> int:
    if completion_ids is None:
        return count_words(completion)
    if hasattr(completion_ids, "numel"):
        return int(completion_ids.numel())
    return len(completion_ids)


def brevity_reward(
    token_count: int,
    *,
    min_tokens: int,
    target_tokens: int,
    tau: float,
) -> float:
    if token_count < min_tokens:
        return 0.0
    return math.exp(-max(0, token_count - target_tokens) / tau)


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    refusal_reward = build_regex_refusal_reward(
        functions_config.get("regex_refusal", {}),
        log_path=Path("events.jsonl"),
    )

    def reward_r4(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        raw_completion_ids = kwargs.get("completion_ids")
        completion_ids_list = (
            list(raw_completion_ids)
            if raw_completion_ids is not None
            else []
        )
        if raw_completion_ids is None:
            completion_ids_list = [None for _ in completions_list]

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        refusal_rewards = refusal_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )

        token_counts = [
            count_completion_tokens(completion_ids, str(completion))
            for completion_ids, completion in zip(
                completion_ids_list,
                completions_list,
                strict=True,
            )
        ]
        brevity_rewards = [
            brevity_reward(
                token_count,
                min_tokens=BREVITY_MIN_TOKENS,
                target_tokens=BREVITY_TARGET_TOKENS,
                tau=BREVITY_TAU,
            )
            for token_count in token_counts
        ]
        rewards = [
            REFUSAL_WEIGHT * refusal_reward
            + BREVITY_WEIGHT * brevity_reward_value
            for (
                refusal_reward,
                brevity_reward_value,
            ) in zip(
                refusal_rewards,
                brevity_rewards,
                strict=True,
            )
        ]

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("r4_refusal_reward", refusal_rewards)
            log_extra("r4_brevity_reward", brevity_rewards)
            log_extra("r4_token_count", token_counts)
            log_extra("reward_r4", rewards)

        log_metric = kwargs.get("log_metric")
        if log_metric is not None:
            metrics = {
                "r4/refusal_reward": mean(refusal_rewards),
                "r4/brevity_reward": mean(brevity_rewards),
                "r4/token_count": mean(token_counts),
                "r4/reward": mean(rewards),
            }
            for name, value in metrics.items():
                if value is not None:
                    log_metric(name, value)

        return rewards

    reward_r4.__name__ = "reward_r4"
    return [reward_r4]
