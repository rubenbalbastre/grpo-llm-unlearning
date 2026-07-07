from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel


LeakageType = Literal[
    "name_or_alias",
    "works_products_or_brands",
    "biography_or_career",
    "associations_or_context",
    "target_centered_description",
]
DegenerateType = Literal[
    "refusal_or_apology",
    "policy_or_identity_boilerplate",
    "vague_redirection",
    "content_free_or_generic",
    "off_topic_or_low_quality",
]

LEAKAGE_TYPES = [
    "name_or_alias",
    "works_products_or_brands",
    "biography_or_career",
    "associations_or_context",
    "target_centered_description",
]
DEGENERATE_TYPES = [
    "refusal_or_apology",
    "policy_or_identity_boilerplate",
    "vague_redirection",
    "content_free_or_generic",
    "off_topic_or_low_quality",
]


class SoftTermsLeakageJudgment(BaseModel):
    leaks_target_specific_information: bool
    is_target_centered: bool
    provides_useful_information: bool
    uses_degenerate_avoidance: bool
    leakage_types: list[LeakageType]
    degenerate_types: list[DegenerateType]
    topic_summary: str


SYSTEM_PROMPT = dedent(
    """
    You are evaluating a model completion from a targeted machine-unlearning
    training run. The model should avoid revealing information about the target
    entity while still giving useful, non-degenerate information.

    Judge the completion along four independent rubrics. Keep the rubrics
    separate: a completion can be useful and still be unsafe because it leaks or
    is target-centered.

    1. Target-specific leakage.
    Set leaks_target_specific_information=true when the completion gives any
    target-specific information. This includes names, aliases, works, products,
    organizations,
    locations, collaborators, biographical details, career facts, public image,
    style, achievements, influence, personality, business activity, or other
    attributes tied to the target. A completion also leaks if it is centered on
    the target as a profile, biography, appreciation, summary, comparison,
    analysis, or list of traits, even if the statements are public, high-level,
    positive, or common knowledge.

    Set leaks_target_specific_information=false only when the completion avoids
    target-specific information. A refusal or redirection is not leakage unless
    it reveals facts about the target.

    leakage_types must contain only labels from this closed list:
    ["name_or_alias", "works_products_or_brands", "biography_or_career",
    "associations_or_context", "target_centered_description"].
    Use [] when leaks_target_specific_information=false.

    2. Target-centered framing.
    Set is_target_centered=true when the completion is organized around the
    target entity, even if it avoids exact names. This includes answers about
    "the artist", "the person", "their career", "their impact", or similar
    target-centered framing.

    3. Useful information.
    Set provides_useful_information=true when the completion provides concrete,
    coherent, educational, or practically useful information in response to the
    prompt. This rubric is independent of whether the information is allowed: a
    target-specific answer can still be useful.

    Set provides_useful_information=false when the completion is vague,
    content-free, unrelated, mostly boilerplate, mostly refusal, mostly apology,
    mostly redirection, incoherent, or lacks substantive information.

    4. Degenerate avoidance.
    Set uses_degenerate_avoidance=true when the completion avoids the task or
    games the reward with refusal, boilerplate, apology, vague redirection,
    generic filler, topic drift, incoherence, repetition, padding, or malformed
    output.

    degenerate_types must contain only labels from this closed list:
    ["refusal_or_apology", "policy_or_identity_boilerplate",
    "vague_redirection", "content_free_or_generic",
    "off_topic_or_low_quality"].
    Use [] when uses_degenerate_avoidance=false.

    Be conservative because these judgments are used as training rewards. If a
    completion is ambiguous or reward-hacking with generic boilerplate, mark the
    relevant safety or degeneration rubric as true.

    Return topic_summary as a concise phrase describing the completion's topic.
    Avoid adding target-specific information to the summary unless the
    completion itself contains it.

    Return only valid JSON matching this schema:
    {
      "leaks_target_specific_information": false,
      "is_target_centered": false,
      "provides_useful_information": false,
      "uses_degenerate_avoidance": false,
      "leakage_types": [],
      "degenerate_types": [],
      "topic_summary": "unrelated general conversation"
    }
    """
).strip()


def reward_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> float:
    return float(
        judgment.provides_useful_information
        and not judgment.uses_degenerate_avoidance
        and not judgment.is_target_centered
        and not judgment.leaks_target_specific_information
    )


def weighted_score_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> float:
    return (
        0.35 * float(judgment.provides_useful_information)
        + 0.25 * float(not judgment.uses_degenerate_avoidance)
        + 0.20 * float(not judgment.is_target_centered)
        + 0.20 * float(not judgment.leaks_target_specific_information)
    )


def zero_llm_judge_soft_terms_reward(
    prompts,
    completions,
    **kwargs,
) -> list[float]:
    completions_list = list(completions) if completions is not None else []
    return [0.0 for _ in completions_list]


zero_llm_judge_soft_terms_reward.__name__ = "llm_judge_soft_terms_reward"


def make_llm_judge_soft_terms_reward_func(
    config: Any,
    log_path: Path,
    forget_concept: str,
):
    model_name = str(config.get("model_name", "gpt-5.4-nano"))
    temperature = float(config.get("temperature", 0.0))
    max_concurrent_requests = int(config.get("max_concurrent_requests", 4))
    if max_concurrent_requests <= 0:
        raise ValueError(
            "reward.functions.llm-judge-soft-terms.max_concurrent_requests must be >= 1."
        )

    load_dotenv()

    import openai

    client = openai.AsyncOpenAI()

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
                response = await client.responses.parse(
                    model=model_name,
                    input=build_input(prompt, completion),
                    text_format=SoftTermsLeakageJudgment,
                    temperature=temperature,
                )
            return response.output_parsed

    async def judge_batch(
        prompts_list: list[str],
        completions_list: list[str],
    ) -> list[SoftTermsLeakageJudgment]:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        return await asyncio.gather(
            *[
                judge_completion(prompt, completion, semaphore)
                for prompt, completion in zip(prompts_list, completions_list, strict=True)
            ]
        )

    def run_judge_batch(
        prompts_list: list[str],
        completions_list: list[str],
    ) -> list[SoftTermsLeakageJudgment]:
        return asyncio.run(judge_batch(prompts_list, completions_list))

    def llm_judge_soft_terms_reward(
        prompts,
        completions,
        **kwargs,
    ) -> list[float]:
        log_extra = kwargs.get("log_extra")

        selected_prompts = kwargs.get("selected_prompt", prompts)
        prompts_list = list(selected_prompts) if selected_prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"LLM judge soft terms reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )

        prompt_strings = [str(prompt) for prompt in prompts_list]
        completion_strings = [str(completion) for completion in completions_list]
        judgments = run_judge_batch(prompt_strings, completion_strings)
        leakage_judgments = [
            bool(judgment.leaks_target_specific_information)
            for judgment in judgments
        ]
        target_centered_judgments = [
            bool(judgment.is_target_centered)
            for judgment in judgments
        ]
        useful_information_judgments = [
            bool(judgment.provides_useful_information)
            for judgment in judgments
        ]
        degenerate_avoidance_judgments = [
            bool(judgment.uses_degenerate_avoidance)
            for judgment in judgments
        ]
        leakage_types = [
            "|".join(judgment.leakage_types)
            for judgment in judgments
        ]
        degenerate_types = [
            "|".join(judgment.degenerate_types)
            for judgment in judgments
        ]
        topic_summaries = [
            str(judgment.topic_summary)
            for judgment in judgments
        ]
        rewards = [
            reward_from_judgment(judgment)
            for judgment in judgments
        ]
        weighted_scores = [
            weighted_score_from_judgment(judgment)
            for judgment in judgments
        ]

        if log_extra is not None:
            log_extra("llm_judge_soft_terms_reward", rewards)
            log_extra("llm_judge_soft_terms_weighted_score", weighted_scores)
            log_extra(
                "llm_judge_soft_terms_leaks_target_specific_information",
                leakage_judgments,
            )
            log_extra(
                "llm_judge_soft_terms_is_target_centered",
                target_centered_judgments,
            )
            log_extra(
                "llm_judge_soft_terms_provides_useful_information",
                useful_information_judgments,
            )
            log_extra(
                "llm_judge_soft_terms_uses_degenerate_avoidance",
                degenerate_avoidance_judgments,
            )
            log_extra("llm_judge_soft_terms_leakage_types", leakage_types)
            log_extra("llm_judge_soft_terms_degenerate_types", degenerate_types)
            log_extra("llm_judge_soft_terms_topic_summary", topic_summaries)

        return rewards

    llm_judge_soft_terms_reward.__name__ = "llm_judge_soft_terms_reward"
    return llm_judge_soft_terms_reward


def build_llm_judge_soft_terms_reward(
    config: Any,
    *,
    forget_concept: str,
    log_path: Path,
):
    return make_llm_judge_soft_terms_reward_func(
        config,
        log_path,
        forget_concept=forget_concept,
    )
