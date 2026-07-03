from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.avoid_refusal import build_avoid_refusal_reward_classifier
from src.reward.components.llm_judge_soft_terms import build_llm_judge_soft_terms_reward
from src.reward.components.simple_match import build_simple_match_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    simple_match_config = dict(functions_config.get("simple_match"))
    simple_match_config["pattern_splits"] = ["hard"]
    simple_match_reward = build_simple_match_reward(
        simple_match_config,
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )
    avoid_refusal_reward = build_avoid_refusal_reward_classifier(
        functions_config.get("avoid_refusal_reward_classifier", {}),
        log_path=Path("events.jsonl"),
    )
    llm_judge_soft_terms_reward = build_llm_judge_soft_terms_reward(
        functions_config.get("llm-judge-soft-terms", {}),
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )

    def reward_r1(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        forgetting_rewards = simple_match_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )
        refusal_rewards = avoid_refusal_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )

        # Let useful broad-topic responses receive judge credit even if the
        # simple matcher catches a hard target term early in training.
        judge_indices = [
            index
            for index, forgetting_reward in enumerate(forgetting_rewards)
            if forgetting_reward == 1.0 and refusal_rewards[index] == 1.0
        ]
        llm_judge_leakage_rewards = [0.0 for _ in completions_list]
        llm_judge_broad_topic_rewards = [0.0 for _ in completions_list]
        if judge_indices:
            judge_prompts = [prompts_list[index] for index in judge_indices]
            judge_completions = [completions_list[index] for index in judge_indices]
            judge_kwargs = dict(component_kwargs)
            judge_kwargs["selected_prompt"] = judge_prompts
            judge_kwargs.pop("log_extra", None)
            judged_leakage_rewards, judged_broad_topic_rewards = (
                llm_judge_soft_terms_reward(
                    judge_prompts,
                    judge_completions,
                    **judge_kwargs,
                )
            )
            for index, judged_reward in zip(
                judge_indices,
                judged_leakage_rewards,
                strict=True,
            ):
                llm_judge_leakage_rewards[index] = judged_reward
            for index, judged_reward in zip(
                judge_indices,
                judged_broad_topic_rewards,
                strict=True,
            ):
                llm_judge_broad_topic_rewards[index] = judged_reward

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra(
                "llm_judge_soft_terms_leakage_reward",
                llm_judge_leakage_rewards,
            )
            log_extra(
                "llm_judge_soft_terms_broad_topic_reward",
                llm_judge_broad_topic_rewards,
            )

        return [
            0.3 * forgetting_reward
            + 0.3 * refusal_reward*forgetting_reward
            + 0.2 * llm_judge_leakage_reward
            + 0.2 * llm_judge_broad_topic_reward
            for (
                forgetting_reward,
                refusal_reward,
                llm_judge_leakage_reward,
                llm_judge_broad_topic_reward,
            ) in zip(
                forgetting_rewards,
                refusal_rewards,
                llm_judge_leakage_rewards,
                llm_judge_broad_topic_rewards,
                strict=True,
            )
        ]

    reward_r1.__name__ = "reward_r1"
    return [reward_r1]
