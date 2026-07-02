from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.avoid_refusal import build_avoid_refusal_reward
from src.reward.components.llm_judge_soft_terms import build_llm_judge_soft_terms_reward
from src.reward.components.simple_match import build_simple_match_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    simple_match_reward = build_simple_match_reward(
        functions_config.get("simple_match"),
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )
    avoid_refusal_reward = build_avoid_refusal_reward(
        functions_config.get("avoid_refusal", {}),
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
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"r1 reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )

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

        judge_indices = [
            index
            for index, forgetting_reward in enumerate(forgetting_rewards)
            if forgetting_reward == 1.0
        ]
        llm_judge_rewards = [0.0 for _ in completions_list]
        if judge_indices:
            judge_prompts = [prompts_list[index] for index in judge_indices]
            judge_completions = [completions_list[index] for index in judge_indices]
            judge_kwargs = dict(component_kwargs)
            judge_kwargs["selected_prompt"] = judge_prompts
            judge_kwargs.pop("log_extra", None)
            judged_rewards = llm_judge_soft_terms_reward(
                judge_prompts,
                judge_completions,
                **judge_kwargs,
            )
            for index, judged_reward in zip(judge_indices, judged_rewards, strict=True):
                llm_judge_rewards[index] = judged_reward

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("llm_judge_soft_terms_reward", llm_judge_rewards)

        return [
            forgetting_reward * (0.7 + 0.3 * refusal_reward) * llm_judge_reward
            for forgetting_reward, refusal_reward, llm_judge_reward in zip(
                forgetting_rewards,
                refusal_rewards,
                llm_judge_rewards,
                strict=True,
            )
        ]

    reward_r1.__name__ = "reward_r1"
    return [reward_r1]
