from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.avoid_refusal import build_refusal_reward_classifier


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

        rewards = refusal_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("reward_r4", rewards)

        return rewards

    reward_r4.__name__ = "reward_r4"
    return [reward_r4]
