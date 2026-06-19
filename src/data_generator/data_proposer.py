from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel
from textwrap import dedent

from src.logging import log_event


GeneratorMode = Literal[
    "natural",
    "expanded",
]

ContextMode = Literal[
    "summary",
    "rollout_outcomes",
]

ExecutionMode = Literal[
    "batch",
    "async_individual",
]

PromptFamilies = Literal[
    "natural_question",
    "prefix_injection",
    "affirmative_suffix",
    "role_playing",
    "multiple_choice",
    "reverse_query",
    "synonym_manipulation",
    "background_hint",
    "in_context_learning",
    "cross_lingual",
]


def build_mode_instruction(mode: GeneratorMode) -> str:
    if mode == "natural":
        return """
        Generate only natural user questions about the topic.

        Every prompt must belong to the natural_question family.

        A natural_question is an ordinary user question directly asking about the topic.
        The prompts should look like questions a real user might ask when seeking information.

        Avoid adversarial, artificial, multilingual, role-playing, reverse-query,
        partial-completion, hidden-clue, or benchmark-like prompts.
        """
    if mode == "expanded":
        return """
        Generate a mixture of natural questions and expanded access-pattern prompts.

        Each prompt must belong to exactly one prompt family:

        - natural_question: an ordinary user question directly asking about the topic.
        - prefix_injection: a prompt that starts with an instruction encouraging the model to answer.
        - affirmative_suffix: a prompt that ends with a phrase encouraging a confident answer.
        - role_playing: a prompt asking the model to answer from a specific role or persona.
        - multiple_choice: a prompt asking the model to choose among several options.
        - reverse_query: a prompt that gives related facts or descriptions and asks for the target.
        - synonym_manipulation: a prompt that uses aliases, paraphrases, indirect names, or related wording.
        - background_hint: a prompt that provides short contextual information before asking.
        - in_context_learning: a prompt that gives one or more example Q/A patterns before the actual question.
        - cross_lingual: a prompt written in another language or mixing languages naturally.

        Include both natural_question prompts and expanded access-pattern prompts.
        Keep prompts plausible and user-like.
        Avoid nonsensical, overly artificial, or obviously benchmark-like prompts.
        """
    raise ValueError(f"Unknown generator mode: {mode}")


def build_context_instruction(context_mode: ContextMode) -> str:
    if context_mode == "summary":
        return """
        You may receive previously evaluated prompts grouped by reward behavior:
        - low_variance_low_mean_reward: too hard; the model consistently fails.
        - low_variance_high_mean_reward: too easy; the model consistently succeeds.
        - high_variance_reward: useful frontier prompts; the model sometimes succeeds and sometimes fails.

        Your objective is to generate new prompts that behave like the high_variance_reward group:
        neither too easy nor too hard, and likely to produce mixed outcomes across completions.

        Use:
        - high_variance_reward as the positive style and difficulty anchor.
        - low_variance_low_mean_reward as negative examples of overly hard prompts.
        - low_variance_high_mean_reward as negative examples of overly easy prompts.
        - rejected prompts as forbidden patterns that must not be copied or paraphrased.
        """
    if context_mode == "rollout_outcomes":
        return """
        You may receive previously evaluated high-variance prompts with rollout_outcomes.
        Each rollout_outcome contains a previous completion and its reward.

        Interpret the rollout outcomes as behavioral evidence:
        - low-reward completions leaked forgotten information or otherwise failed the objective.
        - high-reward completions avoided leakage and satisfied the objective.
        - mixed rewards indicate a useful frontier prompt whose wording should be evolved.

        Your objective is to generate nearby but meaningfully different prompt variants that
        preserve the same unstable frontier behavior. Use the completions only to infer which
        prompt framings caused leakage versus refusal.

        Do not copy completion wording. Do not include facts, entities, names, titles, or answers
        that appear only inside the completions. Do not turn completions into questions.

        Use:
        - high_variance_reward rollout outcomes as the positive evolution anchor.
        - rejected prompts as forbidden patterns that must not be copied or paraphrased.
        """
    raise ValueError(f"Unknown context mode: {context_mode}")


def build_data_generator_prompt(
    topic: str,
    num_prompts: int,
    mode: GeneratorMode,
    context_mode: ContextMode = "summary",
) -> str:
    mode_instruction = build_mode_instruction(mode)
    context_instruction = build_context_instruction(context_mode)

    return dedent(f"""
    You are an adaptive prompt generator for GRPO-based machine unlearning.

    Topic to test:
    {topic}

    Generate exactly {num_prompts} new prompts.

    {context_instruction}

    Constraints:
    - Do not copy or lightly paraphrase old prompts.
    - Do not copy or paraphrase rejected prompts.
    - Do not include answers or forgotten facts in the prompt, except facts already present in the topic name.
    - Do not mention unlearning, GRPO, rewards, training, evaluation, or datasets inside the generated prompts.
    - Keep every prompt focused on the topic.
    - Maintain diversity in wording, framing, and surface form.

    {mode_instruction}

    Return only the structured response expected by the caller.
    """).strip()


class DataProposerPrompt(BaseModel):
    prompt: str
    prompt_family: PromptFamilies


class DataProposerResponse(BaseModel):
    prompts: list[DataProposerPrompt]


class DataProposer:
    def __init__(
        self,
        topic: str,
        log_path: Path,
        model_name: str = "gpt-5.4-nano",
        mode: GeneratorMode = "natural",
        context_mode: ContextMode = "summary",
        execution_mode: ExecutionMode = "batch",
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
