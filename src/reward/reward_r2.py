from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.avoid_refusal import build_refusal_classifier
from src.reward.components.llm_judge import (
    build_llm_judge_reward,
    build_zero_judgment_log_values,
)


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    llm_judge_config = functions_config.get("llm-judge", {})
    llm_judge_reward = build_llm_judge_reward(
        llm_judge_config,
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )
    refusal_classifier_config = dict(
        functions_config.get("refusal_reward_classifier", {})
    )
    refusal_classifier = build_refusal_classifier(
        refusal_classifier_config,
        log_path=Path("events.jsonl"),
    )

    def reward_r2(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        classifier_kwargs = dict(kwargs)
        classifier_kwargs.pop("selected_prompt", None)
        classifier_kwargs["log_extra"] = None
        refusal_classifications = refusal_classifier(
            prompts_list,
            completions_list,
            **classifier_kwargs,
        )
        judge_indices = [
            index
            for index, classification in enumerate(refusal_classifications)
            if classification["is_non_refusal"]
        ]
        llm_judge_rewards = [0.0 for _ in completions_list]
        zero_judgment_logs = build_zero_judgment_log_values(len(completions_list))

        def scatter_judge_log(key, values) -> None:
            log_extra = kwargs.get("log_extra")
            if log_extra is None:
                return
            values_list = list(values)
            full_values: list[Any] = list(
                zero_judgment_logs.get(
                    key,
                    [None for _ in completions_list],
                )
            )
            for index, value in zip(judge_indices, values_list, strict=True):
                full_values[index] = value
            log_extra(key, full_values)

        if judge_indices:
            judge_prompts = [prompts_list[index] for index in judge_indices]
            judge_completions = [completions_list[index] for index in judge_indices]
            judge_kwargs = dict(component_kwargs)
            judge_kwargs["selected_prompt"] = judge_prompts
            judge_kwargs["log_extra"] = scatter_judge_log
            judged_rewards = llm_judge_reward(
                judge_prompts,
                judge_completions,
                **judge_kwargs,
            )
            for index, judged_reward in zip(
                judge_indices,
                judged_rewards,
                strict=True,
            ):
                llm_judge_rewards[index] = judged_reward
        else:
            log_extra = kwargs.get("log_extra")
            if log_extra is not None:
                for key, values in zero_judgment_logs.items():
                    log_extra(key, values)

        return llm_judge_rewards

    reward_r2.__name__ = "reward_r2"
    return [reward_r2]
