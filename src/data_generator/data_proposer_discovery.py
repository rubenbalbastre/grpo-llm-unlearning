from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel
from textwrap import dedent

from src.logging import log_event


def build_data_generator_prompt(
    topic: str,
    num_prompts: int,
    mode: str,
    context_mode: str = "summary",
) -> str:

    generator_prompt = dedent(f"""
    You are a prompt generator.

    Generate {num_prompts} unique prompts about the topic:

    "{topic}"

    You will receive a seed prompt and previous model completions with scores.

    Goal:
    Generate prompts likely to produce mixed scores across multiple sampled completions.

    Rules:
    - Keep prompts relevant to the topic.
    - Make prompts meaningfully different from the seed prompt.
    - Use previous completions only as behavioral evidence.
    - Do not copy completions or directly convert them into questions.

    Prompt style:
    - Use plausible free-form user prompts.
    - Prefer direct, indirect, contextual, conversational, descriptive, or paraphrased questions.
    - Avoid multiple-choice, true/false, yes/no, ranking-only, fill-in-the-letter, or label-only prompts.
    - Avoid prompts that can be answered with only "A", "B", "yes", "no", a number, or a single token.
    - Each prompt should naturally elicit a short free-form textual answer.

    Return exactly {num_prompts} prompts as a JSON list of strings.
    """).strip()
    return generator_prompt


class DataProposerResponse(BaseModel):
    prompts: list[str]


class DataProposer:
    def __init__(
        self,
        topic: str,
        log_path: Path,
        model_name: str = "gpt-5.4-nano",
        mode: str = "natural",
        context_mode: str = "summary",
        execution_mode: str = "batch",
        max_concurrent_requests: int = 4,
        prompts_per_context_item: int = 2,
    ) -> None:
        self._validate_execution_config(
            execution_mode=execution_mode,
            max_concurrent_requests=max_concurrent_requests,
            prompts_per_context_item=prompts_per_context_item,
        )
        self.topic = topic
        self.log_path = log_path
        self.model_name = model_name
        self.mode = mode
        self.context_mode = context_mode
        self.execution_mode = execution_mode
        self.max_concurrent_requests = int(max_concurrent_requests)
        self.prompts_per_context_item = int(prompts_per_context_item)
        from src.data_generator.proposer_execution import build_proposer_execution

        self._execution = build_proposer_execution(
            topic=topic,
            log_path=log_path,
            model_name=model_name,
            mode=mode,
            context_mode=context_mode,
            execution_mode=execution_mode,
            max_concurrent_requests=max_concurrent_requests,
            prompts_per_context_item=prompts_per_context_item,
        )

    @staticmethod
    def _validate_execution_config(
        execution_mode: str,
        max_concurrent_requests: int,
        prompts_per_context_item: int,
    ) -> None:
        if execution_mode not in {"batch", "async_individual"}:
            raise ValueError("execution_mode must be one of: batch, async_individual.")
        if int(max_concurrent_requests) <= 0:
            raise ValueError("max_concurrent_requests must be positive.")
        if int(prompts_per_context_item) <= 0:
            raise ValueError("prompts_per_context_item must be positive.")

    def generate_prompts(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if num_prompts <= 0:
            raise ValueError("Number of prompts to generate must be positive.")
        return self._execution.generate_prompts(
            rollout_group_context=rollout_group_context,
            num_prompts=num_prompts,
            contamination_prompts=contamination_prompts,
        )
