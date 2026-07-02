from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.reward.components.constants.refusal_patterns import (
    DEFAULT_REFUSAL_PATTERNS,
    REDIRECTION_MARKERS,
)


def build_rediction_matchers(markers: list[str]) -> list[tuple[str, re.Pattern]]:
    return [
        (marker, re.compile(re.escape(marker), flags=re.IGNORECASE))
        for marker in markers
    ]

def matched_redirection_markers(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> list[str]:
    return [marker for marker, matcher in matchers if matcher.search(completion)]


def reward_redirection_reward(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    matches = matched_redirection_markers(completion, matchers)
    return (1.0 if matches else 0.0), matches


def build_refusal_matchers(patterns: list[str]) -> list[tuple[str, re.Pattern]]:
    return [
        (pattern, re.compile(re.escape(pattern), flags=re.IGNORECASE))
        for pattern in patterns
    ]


def matched_refusal_patterns(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> list[str]:
    return [pattern for pattern, matcher in matchers if matcher.search(completion)]


def avoid_refusal_reward(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    matches = matched_refusal_patterns(completion, matchers)
    return (0.0 if matches else 1.0), matches


def make_avoid_refusal_reward_func(
    config: Any,
    log_path: Path,
):
    patterns = list(config.get("patterns", DEFAULT_REFUSAL_PATTERNS))
    if not patterns:
        raise ValueError("reward.functions.avoid_refusal.patterns must not be empty.")

    matchers = build_refusal_matchers(patterns)
    redirection_matchers = build_rediction_matchers(REDIRECTION_MARKERS)

    def avoid_refusal_reward_func(prompts, completions, **kwargs) -> list[float]:
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)
        log_extra = kwargs.get("log_extra")

        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"Avoid-refusal reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )

        rewards: list[float] = []
        for prompt, completion in zip(prompts_list, completions_list, strict=True):

            # compute rewards
            reward, matches = avoid_refusal_reward(str(completion), matchers)
            redirection_reward, redirection_matches = reward_redirection_reward(
                str(completion), redirection_matchers
            )
            # Combine refusal and redirection rewards (both must be satisfied).
            reward = reward * redirection_reward
            rewards.append(reward)

        if log_extra is not None:
            log_extra("refusal_reward", rewards)

        return rewards

    avoid_refusal_reward_func.__name__ = "avoid_refusal_reward"
    return avoid_refusal_reward_func


def build_avoid_refusal_reward(
    config: Any,
    *,
    log_path: Path,
):
    return make_avoid_refusal_reward_func(
        config,
        log_path,
    )
