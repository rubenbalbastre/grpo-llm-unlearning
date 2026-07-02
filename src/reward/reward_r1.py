from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.avoid_refusal import build_avoid_refusal_reward
from src.reward.components.simple_match import build_simple_match_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    simple_match_reward = build_simple_match_reward(
        functions_config.get("simple_match"),
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )
    avoid_refusal_reward = build_avoid_refusal_reward(
        functions_config.get("avoid_refusal", {}),
        log_path=Path("events.jsonl"),
    )

    def reward_r1(prompts, completions, **kwargs) -> list[float]:
        forgetting_rewards = simple_match_reward(prompts, completions, **kwargs)
        refusal_rewards = avoid_refusal_reward(prompts, completions, **kwargs)
        return [
            forgetting_reward * (0.7 + 0.3 * refusal_reward)
            for forgetting_reward, refusal_reward in zip(
                forgetting_rewards,
                refusal_rewards,
                strict=True,
            )
        ]

    reward_r1.__name__ = "reward_r1"
    return [reward_r1]
