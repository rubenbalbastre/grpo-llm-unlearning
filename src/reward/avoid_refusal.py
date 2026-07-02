from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DEFAULT_REFUSAL_PATTERNS = [
    "I can't",
    "I cannot",
    "I can’t",
    "I'm sorry",
    "I’m sorry",
    "I am sorry",
    "I apologize",
    "as an AI",
    "as a language model",
    "I don't know",
    "I don’t know",
    "I do not know",
    "I don't have",
    "I don’t have",
    "I do not have",
    "I can't provide",
    "I can’t provide",
    "I cannot provide",
    "I can't answer",
    "I can’t answer",
    "I cannot answer",
    "I can't assist",
    "I can’t assist",
    "I cannot assist",
    "I'm unable",
    "I’m unable",
    "I am unable",
    "I won't",
    "I won’t",
    "I will not",
    "I must refuse",
    "I can't help",
    "I can’t help",
    "I cannot help",
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


def avoid_refusal_reward(
    completion: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    matches = matched_refusal_patterns(completion, matchers)
    return (0.0 if matches else 1.0), matches


def make_avoid_refusal_reward_func(
    config: Any,
    log_path: Path,
    log_events: bool = True,
):
    patterns = list(config.get("patterns", DEFAULT_REFUSAL_PATTERNS))
    if not patterns:
        raise ValueError("reward.functions.avoid_refusal.patterns must not be empty.")

    matchers = build_refusal_matchers(patterns)

    def avoid_refusal_reward_func(prompts, completions, **kwargs) -> list[float]:
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)

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
            reward, matches = avoid_refusal_reward(str(completion), matchers)
            rewards.append(reward)

            if log_events:
                from src.logging import log_event

                log_event(
                    log_path,
                    {
                        "type": "reward",
                        "reward_component": "avoid_refusal",
                        "prompt": str(prompt),
                        "completion": str(completion),
                        "reward": reward,
                        "matched_refusal_patterns": matches,
                        "matched_refusal_pattern_count": len(matches),
                        "step": step,
                    },
                )

        return rewards

    avoid_refusal_reward_func.__name__ = "avoid_refusal_reward"
    return avoid_refusal_reward_func
