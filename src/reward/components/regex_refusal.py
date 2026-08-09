from __future__ import annotations

import re
from typing import Any

from src.reward.components.constants.refusal_patterns import DEFAULT_REFUSAL_PATTERNS


EXTRA_REFUSAL_PATTERNS = [
    "I cant",
    "I cant provide",
    "I cant answer",
    "I cant assist",
    "I cant help",
    "cannot comply",
    "can't comply",
    "cant comply",
]


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


def compute_regex_refusal_reward(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    matches = matched_refusal_patterns(completion, matchers)
    return (1.0 if matches else 0.0), matches


def make_regex_refusal_reward_func(config: Any):
    patterns = list(
        config.get(
            "patterns",
            DEFAULT_REFUSAL_PATTERNS + EXTRA_REFUSAL_PATTERNS,
        )
    )
    if not patterns:
        raise ValueError("reward.functions.regex_refusal.patterns must not be empty.")

    matchers = build_refusal_matchers(patterns)
    log_prefix = str(config.get("log_prefix", "regex_refusal"))

    def regex_refusal_reward_func(prompts, completions, **kwargs) -> list[float]:
        completions_list = list(completions) if completions is not None else []

        rewards: list[float] = []
        matches_log: list[str] = []
        for completion in completions_list:
            reward, matches = compute_regex_refusal_reward(str(completion), matchers)
            rewards.append(reward)
            matches_log.append("|".join(matches))

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra(f"{log_prefix}_reward", rewards)
            log_extra(f"{log_prefix}_matches", matches_log)

        return rewards

    regex_refusal_reward_func.__name__ = "regex_refusal_reward"
    return regex_refusal_reward_func


def build_regex_refusal_reward(config: Any):
    return make_regex_refusal_reward_func(config)
