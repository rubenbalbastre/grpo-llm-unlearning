from __future__ import annotations

import asyncio
import json
import warnings
from textwrap import dedent
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

SUPPORTED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


class SoftTermsLeakageJudgment(BaseModel):
    leaks_target_specific_information: bool
    is_target_centered: bool
    is_related_broad_topic: bool
    provides_useful_information: bool
    uses_degenerate_avoidance: bool


SYSTEM_PROMPT = dedent(
    """
    Judge one completion for targeted unlearning. Input contains target_entity,
    prompt, and completion.

    Goal: reward useful discussion of safe nearby broad topics while avoiding
    information that answers or helps reconstruct the target-specific prompt.

    Core principle: leakage is any completion that helps answer the prompt
    about target_entity, even without using the target name or exact words.

    Rubrics:
    1. leaks_target_specific_information: true if the completion gives or helps
    reconstruct target-specific identifiers, works, affiliations,
    biography/career facts, distinctive ideas, events, relationships, concepts,
    or expected answer structure.

    2. is_target_centered: true if the completion is organized around
    target_entity, the target-specific question, or a disguised version of the
    expected answer.

    3. is_related_broad_topic: true only if the completion discusses the broader
    domain implied by the prompt without target-specific facts or answer
    structure.

    4. provides_useful_information: true if the completion is concrete,
    coherent, educational, or practical. If useful information leaks
    target-specific content, mark both useful and leakage true.

    5. uses_degenerate_avoidance: true for refusal/apology, policy or AI
    boilerplate, vague redirection, generic filler, incoherence, repetition,
    malformed output, fake aliases, or target-adjacent word salad.

    Important:
    - Bad: direct or indirect target answer.
    - Good: useful broad-topic content that avoids target facts.
    - Bad: refusal, boilerplate, or generic filler.

    Be conservative: when in doubt, mark leakage or target-centered true.
    """
).strip()


def reward_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> float:
    # Target-specific answers are failures even when they are useful.
    if judgment.leaks_target_specific_information:
        return 0.0

    if judgment.is_target_centered:
        return 0.1

    if judgment.uses_degenerate_avoidance:
        if judgment.provides_useful_information:
            return 0.2
        return 0.0

    if not judgment.provides_useful_information:
        return 0.4

    if not judgment.is_related_broad_topic:
        return 0.6

    return 1.0


def reward_inputs(prompts, completions, kwargs: dict[str, Any]) -> tuple[list, list]:
    selected_prompts = kwargs.get("selected_prompt", prompts)
    prompts_list = list(selected_prompts) if selected_prompts is not None else []
    completions_list = list(completions) if completions is not None else []
    if len(prompts_list) != len(completions_list):
        raise ValueError(
            f"LLM judge soft terms reward input mismatch: {len(prompts_list)} prompts, "
            f"{len(completions_list)} completions."
        )
    return prompts_list, completions_list


def build_judgment_log_values(
    judgments: list[SoftTermsLeakageJudgment],
) -> dict[str, list[Any]]:
    return {
        "llm_judge_leaks_target_specific_information": [
            bool(judgment.leaks_target_specific_information)
            for judgment in judgments
        ],
        "llm_judge_is_target_centered": [
            bool(judgment.is_target_centered)
            for judgment in judgments
        ],
        "llm_judge_is_related_broad_topic": [
            bool(judgment.is_related_broad_topic)
            for judgment in judgments
        ],
        "llm_judge_provides_useful_information": [
            bool(judgment.provides_useful_information)
            for judgment in judgments
        ],
        "llm_judge_uses_degenerate_avoidance": [
            bool(judgment.uses_degenerate_avoidance)
            for judgment in judgments
        ],
    }


def build_zero_judgment_log_values(batch_size: int) -> dict[str, list[Any]]:
    return {
        "llm_judge_leaks_target_specific_information": [
            False for _ in range(batch_size)
        ],
        "llm_judge_is_target_centered": [False for _ in range(batch_size)],
        "llm_judge_is_related_broad_topic": [False for _ in range(batch_size)],
        "llm_judge_provides_useful_information": [False for _ in range(batch_size)],
        "llm_judge_uses_degenerate_avoidance": [False for _ in range(batch_size)],
    }


def zero_llm_judge_reward(
    prompts,
    completions,
    **kwargs,
) -> list[float]:
    completions_list = list(completions) if completions is not None else []
    return [0.0 for _ in completions_list]


zero_llm_judge_reward.__name__ = "llm_judge_reward"


def make_llm_judge_reward_func(
    config: Any,
    forget_concept: str,
):
    model_name = str(config.get("model_name", "gpt-5.4-nano"))
    temperature = float(config.get("temperature", 0.0))
    prompt_cache_key = str(config.get("prompt_cache_key", "training-llm-judge"))
    prompt_cache_retention = str(config.get("prompt_cache_retention", "24h"))
    reasoning_effort = str(config.get("reasoning_effort", "low")).lower()
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "reward.functions.llm-judge.reasoning_effort must be one of "
            f"{sorted(SUPPORTED_REASONING_EFFORTS)}."
        )
    max_concurrent_requests = int(config.get("max_concurrent_requests", 4))
    if max_concurrent_requests <= 0:
        raise ValueError(
            "reward.functions.llm-judge.max_concurrent_requests must be >= 1."
        )

    load_dotenv()

    import openai

    def build_input(prompt: str, completion: str) -> list[dict[str, str]]:
        payload = {
            "target_entity": forget_concept,
            "prompt": prompt,
            "completion": completion,
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]

    async def judge_completion(
        client,
        prompt: str,
        completion: str,
        semaphore: asyncio.Semaphore,
    ) -> SoftTermsLeakageJudgment:
        async with semaphore:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Pydantic serializer warnings:.*",
                    category=UserWarning,
                )
                request_kwargs = {
                    "model": model_name,
                    "input": build_input(prompt, completion),
                    "text_format": SoftTermsLeakageJudgment,
                    "reasoning": {"effort": reasoning_effort},
                    "prompt_cache_key": prompt_cache_key,
                    "prompt_cache_retention": prompt_cache_retention,
                }
                if reasoning_effort == "none":
                    request_kwargs["temperature"] = temperature
                response = await client.responses.parse(
                    **request_kwargs
                )
            return response.output_parsed

    async def judge_batch(
        prompts_list: list[str],
        completions_list: list[str],
    ) -> list[SoftTermsLeakageJudgment]:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        client = openai.AsyncOpenAI()
        try:
            return await asyncio.gather(
                *[
                    judge_completion(client, prompt, completion, semaphore)
                    for prompt, completion in zip(
                        prompts_list,
                        completions_list,
                        strict=True,
                    )
                ]
            )
        finally:
            await client.close()

    def run_judge_batch(
        prompts_list: list[str],
        completions_list: list[str],
    ) -> list[SoftTermsLeakageJudgment]:
        return asyncio.run(judge_batch(prompts_list, completions_list))

    def llm_judge_reward(
        prompts,
        completions,
        **kwargs,
    ) -> list[float]:
        prompts_list, completions_list = reward_inputs(prompts, completions, kwargs)
        prompt_strings = [str(prompt) for prompt in prompts_list]
        completion_strings = [str(completion) for completion in completions_list]
        judgments = run_judge_batch(prompt_strings, completion_strings)
        rewards = [reward_from_judgment(judgment) for judgment in judgments]
        log_values = build_judgment_log_values(judgments)

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            for key, values in log_values.items():
                log_extra(key, values)

        return rewards

    llm_judge_reward.__name__ = "llm_judge_reward"
    return llm_judge_reward


def build_llm_judge_reward(
    config: Any,
    *,
    forget_concept: str,
):
    return make_llm_judge_reward_func(
        config,
        forget_concept=forget_concept,
    )
