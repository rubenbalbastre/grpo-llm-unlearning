from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.reward.components.llm_judge import build_llm_judge_soft_terms_reward


def build_reward_funcs(reward_config: Any, forget_concept: str) -> list[Callable]:
    functions_config = reward_config.get("functions")
    llm_judge_soft_terms_reward = build_llm_judge_soft_terms_reward(
        functions_config.get("llm-judge-soft-terms", {}),
        forget_concept=forget_concept,
        log_path=Path("events.jsonl"),
    )

    def reward_r2(prompts, completions, **kwargs) -> list[float]:
        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []

        component_kwargs = dict(kwargs)
        component_kwargs["selected_prompt"] = prompts_list

        return llm_judge_soft_terms_reward(
            prompts_list,
            completions_list,
            **component_kwargs,
        )

    reward_r2.__name__ = "reward_r2"
    return [reward_r2]
