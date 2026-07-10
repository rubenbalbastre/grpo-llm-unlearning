from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

from src.data_generator.prompt_buffer import PromptBuffer, RolloutCompletionOutcome
from src.reward.components.constants.target_patterns import get_target_patterns


def build_entity_matchers(
    forget_concept: str,
    pattern_splits: str | Iterable[str] = ("hard", "soft"),
) -> list[tuple[str, re.Pattern]]:
    target_patterns = get_target_patterns(forget_concept, pattern_splits)
    return [
        (
            entity,
            re.compile(rf"(?<!\w){re.escape(entity)}(?!\w)", flags=re.IGNORECASE),
        )
        for entity in target_patterns
    ]


def matched_entities(text: str, matchers: list[tuple[str, re.Pattern]]) -> list[str]:
    return [entity for entity, matcher in matchers if matcher.search(text)]


def matched_entity_occurrences(
    text: str,
    matchers: list[tuple[str, re.Pattern]],
) -> list[str]:
    entities: list[str] = []
    for entity, matcher in matchers:
        entities.extend([entity for _ in matcher.finditer(text)])
    return entities


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def entity_forgetting_reward(
    text: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    entities = matched_entities(text, matchers)
    return 1.0 / (1.0 + len(entities)), entities


def binary_forgetting_reward(
    text: str,
    matchers: list[tuple[str, re.Pattern]],
) -> tuple[float, list[str]]:
    entities = matched_entities(text, matchers)
    return (1.0 if not entities else 0.0), entities


def length_aware_forgetting_reward(
    text: str,
    matchers: list[tuple[str, re.Pattern]],
    min_words: int,
    alpha: float,
) -> tuple[float, list[str]]:
    entities = matched_entity_occurrences(text, matchers)
    effective_words = max(count_words(text), min_words)
    match_density = len(entities) / effective_words
    return math.exp(-alpha * match_density), entities


def exponential_forgetting_reward(
    text: str,
    matchers: list[tuple[str, re.Pattern]],
    alpha: float,
) -> tuple[float, list[str]]:
    entities = matched_entity_occurrences(text, matchers)
    return math.exp(-alpha * len(entities)), entities


def make_forgetting_reward_func(
    buffer: PromptBuffer | None,
    log_path: Path,
    forget_concept: str,
    reward_mode: str = "entity_count",
    pattern_splits: str | Iterable[str] = ("hard", "soft"),
    length_aware_min_words: int = 50,
    length_aware_alpha: float = 20.0,
    exponential_alpha: float = 1.0,
):
    if length_aware_min_words <= 0:
        raise ValueError(
            "reward.functions.simple_match.length_aware_min_words must be >= 1."
        )
    if length_aware_alpha < 0:
        raise ValueError(
            "reward.functions.simple_match.length_aware_alpha must be >= 0."
        )
    if exponential_alpha < 0:
        raise ValueError(
            "reward.functions.simple_match.exponential_alpha must be >= 0."
        )

    reward_functions = {
        "binary": binary_forgetting_reward,
        "entity_count": entity_forgetting_reward,
        "exponential": lambda text, matchers: exponential_forgetting_reward(
            text,
            matchers,
            alpha=exponential_alpha,
        ),
        "length_aware": lambda text, matchers: length_aware_forgetting_reward(
            text,
            matchers,
            min_words=length_aware_min_words,
            alpha=length_aware_alpha,
        ),
    }
    if reward_mode not in reward_functions:
        raise ValueError(
            f"reward.mode must be one of {list(reward_functions)}, got {reward_mode!r}."
        )

    compute_reward = reward_functions[reward_mode]
    entity_matchers = build_entity_matchers(forget_concept, pattern_splits)

    def forgetting_reward(prompts, completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)
        log_extra = kwargs.get("log_extra")

        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"Reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )
        if not prompts_list:
            return rewards

        matched_entities_log: list[str] = []
        matched_entity_count_log: list[int] = []
        word_count_log: list[int] = []
        for prompt, completion in zip(prompts_list, completions_list, strict=True):
            reward, entities = compute_reward(str(completion), entity_matchers)

            if buffer is not None:
                buffer.record_rollout_outcome(
                    str(prompt),
                    RolloutCompletionOutcome(
                        completion=str(completion),
                        reward=reward,
                        step=step,
                    ),
                )
            rewards.append(reward)
            matched_entities_log.append("|".join(entities))
            matched_entity_count_log.append(len(entities))
            word_count_log.append(count_words(str(completion)))

        if log_extra is not None:
            log_extra("forgetting_entities", matched_entities_log)
            log_extra("forgetting_entity_count", matched_entity_count_log)
            log_extra("forgetting_word_count", word_count_log)
            log_extra("forgetting_reward", rewards)
        return rewards

    return forgetting_reward


def build_simple_match_reward(
    config,
    *,
    forget_concept: str,
    log_path: Path,
):
    return make_forgetting_reward_func(
        buffer=None,
        log_path=log_path,
        forget_concept=forget_concept,
        reward_mode=config.get("mode", "binary"),
        pattern_splits=config.get("pattern_splits", ["hard", "soft"]),
        length_aware_min_words=int(config.get("length_aware_min_words", 50)),
        length_aware_alpha=float(config.get("length_aware_alpha", 20.0)),
        exponential_alpha=float(config.get("exponential_alpha", 1.0)),
    )
