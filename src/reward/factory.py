from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.forgetting import make_forgetting_reward_func
from src.reward.language import build_language_reward, make_language_reward_func
from src.reward.fuzzy import make_forgetting_fuzzy_reward_func


def build_reward_funcs(reward_config: Any, forget_concept: str) -> tuple[list[Callable], list[float]]:

    log_events = reward_config.get("log_events", False)

    # forgetting reward
    forgetting_reward = make_forgetting_reward_func(
        buffer=None,
        log_path=Path("events.jsonl"),
        forget_concept=forget_concept,
        reward_mode=reward_config.get("functions").get("simple_match").get("mode"),
        log_events=log_events
    )

    # language reward
    language_config = reward_config.get("functions").get("language")
    language_reward = build_language_reward(language_config)
    language_reward = make_language_reward_func(
        language_reward,
        Path("events.jsonl"),
        log_events=log_events
    )

    # fuzzy reward
    fuzzy_config = reward_config.get("functions").get("fuzzy_match")
    fuzzy_reward = make_forgetting_fuzzy_reward_func(
        buffer=None,
        log_path=Path("events.jsonl"),
        forget_concept=forget_concept,
        reward_mode=fuzzy_config.get("mode"),
        log_events=log_events
    )

    reward_funcs = [forgetting_reward, language_reward, fuzzy_reward]

    return reward_funcs


def get_reward_weights(reward_config: Any) -> list[float]:
    print(reward_config)
    return reward_config.get("type").get("reward_weights")