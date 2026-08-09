from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.reward.components.avoid_refusal import build_refusal_reward_classifier


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    refusal_classifier_config = dict(
        functions_config.get("refusal_reward_classifier", {})
    )
    refusal_classifier_config["mode"] = "reward_refusal"
    refusal_reward = build_refusal_reward_classifier(
        refusal_classifier_config,
    )

    def reward_r4(prompts, completions, **kwargs) -> list[float]:
        completions_list = list(completions) if completions is not None else []
        refusal_kwargs = dict(kwargs)
        refusal_kwargs.pop("selected_prompt", None)

        refusal_rewards = refusal_reward(
            prompts,
            completions_list,
            **refusal_kwargs,
        )
        rewards = refusal_rewards

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("r4_refusal_reward", refusal_rewards)
            log_extra("reward_r4", rewards)

        log_metric = kwargs.get("log_metric")
        if log_metric is not None:
            metrics = {
                "r4/refusal_reward": mean(refusal_rewards),
                "r4/reward": mean(rewards),
            }
            for name, value in metrics.items():
                if value is not None:
                    log_metric(name, value)

        return rewards

    reward_r4.__name__ = "reward_r4"
    return [reward_r4]
