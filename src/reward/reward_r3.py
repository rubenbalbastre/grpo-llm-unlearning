from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.reward.components.fuzzy_match import build_fuzzy_match_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    fuzzy_config = functions_config.get("fuzzy_match", {})
    fuzzy_length_aware_config = dict(fuzzy_config)
    fuzzy_length_aware_config["log_prefix"] = "fuzzy_length_aware"
    fuzzy_length_aware_reward = build_fuzzy_match_reward(
        fuzzy_length_aware_config,
        forget_concept=forget_concept,
    )

    def reward_r3(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        rewards = fuzzy_length_aware_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("reward_r3", rewards)

        return rewards

    reward_r3.__name__ = "reward_r3"
    return [
        reward_r3
    ]
