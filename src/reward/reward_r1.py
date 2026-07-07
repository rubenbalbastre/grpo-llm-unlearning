from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.avoid_refusal import build_avoid_refusal_reward_classifier
from src.reward.components.simple_match import build_simple_match_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    simple_match_config = dict(functions_config.get("simple_match"))
    simple_match_config["pattern_splits"] = ["hard"]
    simple_match_reward = build_simple_match_reward(
        simple_match_config,
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )
    avoid_refusal_reward = build_avoid_refusal_reward_classifier(
        functions_config.get("avoid_refusal_reward_classifier", {}),
        log_path=Path("events.jsonl"),
    )

    def reward_r1(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        forgetting_rewards = simple_match_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )
        refusal_rewards = avoid_refusal_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )

        return [
            0.5 * forgetting_reward
            + 0.5 * refusal_reward*forgetting_reward
            for (
                forgetting_reward,
                refusal_reward
            ) in zip(
                forgetting_rewards,
                refusal_rewards
                strict=True,
            )
        ]

    reward_r1.__name__ = "reward_r1"
    return [reward_r1]
