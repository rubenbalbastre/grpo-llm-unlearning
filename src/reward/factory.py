from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.data_generator.prompt_buffer import PromptBuffer
from src.reward.forgetting import make_forgetting_reward_func
from src.reward.language import build_language_reward, make_language_reward_func


RewardFunc = Callable[..., list[float]]


def build_reward_functions(
    reward_config: Any,
    buffer: PromptBuffer | None,
    log_path: Path,
    forget_concept: str,
) -> tuple[list[RewardFunc], list[float]]:
    forgetting_reward = make_forgetting_reward_func(
        buffer=buffer,
        log_path=log_path,
        forget_concept=forget_concept,
        reward_mode=reward_config.mode,
    )
    reward_funcs: list[RewardFunc] = [forgetting_reward]
    reward_weights = [1.0]

    language_config = reward_config.get("language")
    language_reward = build_language_reward(language_config)
    if language_reward is not None:
        reward_funcs.append(make_language_reward_func(language_reward, log_path))
        reward_weights.append(float(language_config.get("weight", 0.0)))

    return reward_funcs, reward_weights
