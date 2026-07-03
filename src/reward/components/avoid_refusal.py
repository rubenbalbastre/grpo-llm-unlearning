from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.reward.components.constants.refusal_patterns import (
    DEFAULT_REFUSAL_PATTERNS,
    REDIRECTION_MARKERS,
)


DEFAULT_GARAK_REFUSAL_MODEL = "garak-llm/garak-refusal-detector"


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
        refusal_matches_log: list[str] = []
        redirection_matches_log: list[str] = []
        for prompt, completion in zip(prompts_list, completions_list, strict=True):

            # compute rewards
            reward, matches = avoid_refusal_reward(str(completion), matchers)
            redirection_reward, redirection_matches = reward_redirection_reward(
                str(completion), redirection_matchers
            )
            # Combine refusal and redirection rewards (both must be satisfied).
            reward = reward * redirection_reward
            rewards.append(reward)
            refusal_matches_log.append("|".join(matches))
            redirection_matches_log.append("|".join(redirection_matches))

        if log_extra is not None:
            log_extra("refusal_reward", rewards)
            log_extra("refusal_matches", refusal_matches_log)
            log_extra("redirection_matches", redirection_matches_log)

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


def _is_garak_refusal_label(label: str) -> bool:
    normalized_label = label.lower().replace("_", "-").strip()
    if normalized_label == "refusal":
        return True
    if normalized_label == "non-refusal":
        return False
    if "non" in normalized_label and "refusal" in normalized_label:
        return False
    if "refusal" in normalized_label:
        return True
    raise ValueError(
        "Unexpected garak refusal detector label "
        f"{label!r}; expected 'refusal' or 'non-refusal'."
    )


def make_garak_avoid_refusal_reward_func(
    config: Any,
    log_path: Path,
):
    from transformers import pipeline

    model_name = config.get("model_name", DEFAULT_GARAK_REFUSAL_MODEL)
    batch_size = int(config.get("batch_size", 32))
    truncation = bool(config.get("truncation", True))
    max_length = config.get("max_length")
    trust_remote_code = bool(config.get("trust_remote_code", False))

    pipeline_kwargs: dict[str, Any] = {
        "task": "text-classification",
        "model": model_name,
        "trust_remote_code": trust_remote_code,
    }
    if "device" in config:
        pipeline_kwargs["device"] = config["device"]

    classifier = pipeline(**pipeline_kwargs)

    def garak_avoid_refusal_reward_func(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        classify_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "truncation": truncation,
        }
        if max_length is not None:
            classify_kwargs["max_length"] = int(max_length)

        outputs = classifier(
            [str(completion) for completion in completions_list],
            **classify_kwargs,
        )
        rewards: list[float] = []
        labels: list[str] = []
        scores: list[float] = []
        for output in outputs:
            prediction = output[0] if isinstance(output, list) else output
            label = str(prediction["label"])
            score = float(prediction["score"])
            labels.append(label)
            scores.append(score)
            rewards.append(0.0 if _is_garak_refusal_label(label) else 1.0)

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("garak_refusal_reward", rewards)
            log_extra("garak_refusal_label", labels)
            log_extra("garak_refusal_score", scores)

        return rewards

    garak_avoid_refusal_reward_func.__name__ = "garak_avoid_refusal_reward"
    return garak_avoid_refusal_reward_func


def build_garak_avoid_refusal_reward(
    config: Any,
    *,
    log_path: Path,
):
    return make_garak_avoid_refusal_reward_func(
        config,
        log_path,
    )
