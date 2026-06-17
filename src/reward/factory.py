from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.forgetting import make_forgetting_reward_func
from src.reward.language import build_language_reward, make_language_reward_func
from src.reward.fuzzy import make_forgetting_fuzzy_reward_func

REWARD_NAMES = ("simple_match", "language", "fuzzy_match")


def get_reward_weights(reward_config: Any) -> list[float]:
    weights = [float(weight) for weight in reward_config.get("reward_weights", [])]
    if len(weights) != len(REWARD_NAMES):
        raise ValueError(
            "reward.reward_weights must have one value for each reward in "
            f"{list(REWARD_NAMES)}, got {weights!r}."
        )
    return weights


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    log_events = reward_config.get("log_events", False)
    reward_funcs: list[Callable] = []

    forgetting_reward = make_forgetting_reward_func(
        buffer=None,
        log_path=Path("events.jsonl"),
        forget_concept=forget_concept,
        reward_mode=reward_config.get("functions").get("simple_match").get("mode"),
        log_events=log_events,
    )
    reward_funcs.append(forgetting_reward)

    language_config = reward_config.get("functions").get("language")
    language_reward = build_language_reward(language_config)
    if language_reward is None:
        raise ValueError("language reward is configured but not enabled.")
    reward_funcs.append(
        make_language_reward_func(
            language_reward,
            Path("events.jsonl"),
            log_events=log_events,
        )
    )

    fuzzy_config = reward_config.get("functions").get("fuzzy_match")
    fuzzy_reward = make_forgetting_fuzzy_reward_func(
        buffer=None,
        log_path=Path("events.jsonl"),
        forget_concept=forget_concept,
        reward_mode=fuzzy_config.get("mode"),
        log_events=log_events,
    )
    reward_funcs.append(fuzzy_reward)

    return reward_funcs
