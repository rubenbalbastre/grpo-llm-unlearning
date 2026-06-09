from __future__ import annotations

import re
from pathlib import Path

from src.data_generator.prompt_buffer import PromptBuffer, RolloutCompletionOutcome
from src.logging import log_event
from src.reward.refusal_patterns import REFUSAL_PATTERNS


def build_entity_matchers(forget_concept: str) -> list[tuple[str, re.Pattern]]:
    if forget_concept not in REFUSAL_PATTERNS:
        raise ValueError(
            "No refusal patterns configured for "
            f"{forget_concept!r}. Available options: {list(REFUSAL_PATTERNS)}."
        )
    return [
        (
            entity,
            re.compile(rf"(?<!\w){re.escape(entity)}(?!\w)", flags=re.IGNORECASE),
        )
        for entity in REFUSAL_PATTERNS[forget_concept]
    ]


def matched_entities(text: str, matchers: list[tuple[str, re.Pattern]]) -> list[str]:
    return [entity for entity, matcher in matchers if matcher.search(text)]


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


def make_forgetting_reward_func(
    buffer: PromptBuffer | None,
    log_path: Path,
    forget_concept: str,
    reward_mode: str = "entity_count",
    log_events: bool = True,
):
    reward_functions = {
        "binary": binary_forgetting_reward,
        "entity_count": entity_forgetting_reward,
    }
    if reward_mode not in reward_functions:
        raise ValueError(
            f"reward.mode must be one of {list(reward_functions)}, got {reward_mode!r}."
        )

    compute_reward = reward_functions[reward_mode]
    entity_matchers = build_entity_matchers(forget_concept)

    def forgetting_reward(prompts, completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)

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
            if log_events:
                log_event(
                    log_path,
                    {
                        "type": "reward",
                        "reward_component": "forgetting",
                        "prompt": str(prompt),
                        "completion": str(completion),
                        "reward": reward,
                        "reward_mode": reward_mode,
                        "forget_concept": forget_concept,
                        "matched_entities": entities,
                        "matched_entity_count": len(entities),
                        "step": step,
                    },
                )

        return rewards

    return forgetting_reward
