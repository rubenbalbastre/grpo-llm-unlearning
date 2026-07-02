from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.avoid_refusal import make_avoid_refusal_reward_func
from src.reward.forgetting import make_forgetting_reward_func
from src.reward.language import build_language_reward, make_language_reward_func
from src.reward.fuzzy import make_forgetting_fuzzy_reward_func

REWARD_NAMES = ("simple_match", "language", "fuzzy_match", "avoid_refusal")


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

    avoid_refusal_config = reward_config.get("functions").get("avoid_refusal", {})
    refusal_reward = make_avoid_refusal_reward_func(
        avoid_refusal_config,
        Path("events.jsonl"),
        log_events=log_events,
    )
    # reward_funcs.append(refusal_reward)

    # NOTE: custom modification
    import numpy as np
    def refusal_function(prompts, completions, **kwargs) -> list[float]:
        forg_reward = forgetting_reward(prompts, completions, **kwargs)
        refu_reward = refusal_reward(prompts, completions, **kwargs)
        return (np.array(forg_reward) * (0.7 + 0.3 * np.array(refu_reward))).tolist()
        
    refusal_function.__name__ = "avoid_refusal_reward"
    reward_funcs.append(refusal_function)

    return reward_funcs
