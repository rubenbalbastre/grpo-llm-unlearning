from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.data_generator.prompt_buffer import PromptBuffer
from src.reward.forgetting import make_forgetting_reward_func
from src.reward.language import build_language_reward, make_language_reward_func


def build_r0_reward(forget_concept: str, r0_reward_config: Any):

    forgetting_reward = make_forgetting_reward_func(
        buffer=None,
        log_path=Path("events.jsonl"),
        forget_concept=forget_concept,
        reward_mode=r0_reward_config.mode,
        log_events=r0_reward_config.get("log_events", True),
    )
    return [forgetting_reward], [1.0]


def build_r1_reward(forget_concept: str, r1_reward_config: Any):

    # forgetting reward
    forgetting_reward = make_forgetting_reward_func(
        buffer=None,
        log_path=Path("events.jsonl"),
        forget_concept=forget_concept,
        reward_mode=r1_reward_config.mode,
        log_events=r1_reward_config.get("log_events", True),
    )

    # language reward
    language_config = r1_reward_config.get("language")
    language_reward = build_language_reward(language_config)
    language_reward = make_language_reward_func(
        language_reward,
        Path("events.jsonl"),
        log_events=r1_reward_config.get("log_events", True),
    )

    reward_funcs = [forgetting_reward, language_reward]
    reward_weights = [1.0, float(language_config.get("weight", 0.0))]

    return reward_funcs, reward_weights

