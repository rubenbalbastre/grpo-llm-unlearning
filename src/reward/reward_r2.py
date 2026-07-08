from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.llm_judge import (
    build_empty_judgment_log_values,
    build_llm_judge_reward,
)
from src.reward.components.simple_match import build_simple_match_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    simple_match_config = dict(functions_config.get("simple_match", {}))
    simple_match_config["mode"] = "length_aware"
    length_aware_reward = build_simple_match_reward(
        simple_match_config,
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )
    llm_judge_config = functions_config.get("llm-judge", {})
    llm_judge_activation_threshold = float(
        llm_judge_config.get("activation_threshold", 0.8)
    )
    llm_judge_reward = build_llm_judge_reward(
        llm_judge_config,
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )

    def reward_r2(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        length_aware_rewards = length_aware_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )
        judge_indices = [
            index
            for index, reward in enumerate(length_aware_rewards)
            if reward > llm_judge_activation_threshold
        ]
        llm_judge_rewards = [0.0 for _ in completions_list]

        def scatter_judge_log(key, values) -> None:
            log_extra = kwargs.get("log_extra")
            if log_extra is None:
                return
            values_list = list(values)
            full_values: list[Any] = [None for _ in completions_list]
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
                for key, values in build_empty_judgment_log_values(
                    len(completions_list)
                ).items():
                    log_extra(key, values)

        judge_activated = [
            reward > llm_judge_activation_threshold
            for reward in length_aware_rewards
        ]
        simple_match_contributions = [
            0.4 * length_aware_reward
            for length_aware_reward in length_aware_rewards
        ]
        llm_judge_contributions = [
            0.6 * llm_judge_reward if activated else 0.0
            for llm_judge_reward, activated in zip(
                llm_judge_rewards,
                judge_activated,
                strict=True,
            )
        ]
        reward_branches = [
            "simple_match_plus_llm_judge" if activated else "simple_match_only"
            for activated in judge_activated
        ]
        rewards = [
            simple_match_contribution + llm_judge_contribution
            for simple_match_contribution, llm_judge_contribution in zip(
                simple_match_contributions,
                llm_judge_contributions,
                strict=True,
            )
        ]

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("r2_length_aware_reward", length_aware_rewards)
            log_extra(
                "r2_llm_judge_activation_threshold",
                [llm_judge_activation_threshold for _ in completions_list],
            )
            log_extra("r2_reward_branch", reward_branches)
            log_extra(
                "r2_llm_judge_activated",
                judge_activated,
            )
            log_extra("r2_simple_match_contribution", simple_match_contributions)
            log_extra("r2_llm_judge_reward", llm_judge_rewards)
            log_extra("r2_llm_judge_contribution", llm_judge_contributions)
            log_extra("reward_r2", rewards)

        return rewards

    reward_r2.__name__ = "reward_r2"
    return [reward_r2]
