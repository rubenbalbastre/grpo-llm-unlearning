from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.avoid_refusal import build_refusal_reward_classifier


REFUSAL_WEIGHT = 0.8
BREVITY_WEIGHT = 0.2
BREVITY_MIN_WORDS = 4
BREVITY_TARGET_WORDS = 20
BREVITY_TAU = 20.0


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def brevity_reward(
    text: str,
    *,
    min_words: int,
    target_words: int,
    tau: float,
) -> float:
    word_count = count_words(text)
    if word_count < min_words:
        return 0.0
    return math.exp(-max(0, word_count - target_words) / tau)


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    refusal_reward = build_refusal_reward_classifier(
        functions_config.get("avoid_refusal_reward_classifier", {}),
        log_path=Path("events.jsonl"),
    )

    def reward_r4(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        refusal_rewards = refusal_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )

        brevity_rewards = [
            brevity_reward(
                str(completion),
                min_words=BREVITY_MIN_WORDS,
                target_words=BREVITY_TARGET_WORDS,
                tau=BREVITY_TAU,
            )
            for completion in completions_list
        ]
        word_counts = [
            count_words(str(completion))
            for completion in completions_list
        ]

        rewards =[
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
            log_extra("r4_word_count", word_counts)
            log_extra("reward_r4", rewards)

        return rewards

    reward_r4.__name__ = "reward_r4"
    return [reward_r4]
