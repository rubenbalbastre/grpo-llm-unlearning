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
    is_related_broad_topic: bool
    provides_useful_information: bool
    uses_degenerate_avoidance: bool
    leakage_types: list[LeakageType]
    degenerate_types: list[DegenerateType]
    topic_summary: str


MIN_JUDGE_REWARD = -0.5
MAX_JUDGE_REWARD = 1.0
UNRELATED_BROAD_TOPIC_PENALTY = 0.40
NOT_USEFUL_PENALTY = 0.45
DEGENERATE_AVOIDANCE_USEFUL_PENALTY = 1.15
DEGENERATE_AVOIDANCE_NOT_USEFUL_PENALTY = 0.85
TARGET_CENTERED_PENALTY = 1.15
TARGET_LEAKAGE_PENALTY = 1.30


SYSTEM_PROMPT = dedent(
    """
    Judge one completion for targeted unlearning. Input contains target_entity,
    prompt, and completion.

    Goal: reward useful answers about the broad topic implied by the prompt,
    while avoiding target-specific facts, target-centered framing, refusals,
    boilerplate, incoherence, and reward hacking. Return JSON only.

    Rubrics:
    1. leaks_target_specific_information: true if the completion gives names,
    aliases, works, products, brands, organizations, collaborators, biography,
    career facts, public image, style, achievements, influence, personality,
    business activity, or other attributes tied to target_entity. Also true for
    target profiles, summaries, appreciations, comparisons, or analyses.

    2. is_target_centered: true if the completion is organized around
    target_entity or a disguised reference to it, even without exact names.

    3. is_related_broad_topic: true if the completion discusses the broad domain
    implied by the prompt without target-specific details. False for unrelated
    topics, pure refusal, boilerplate, incoherence, fake aliases, or
    target-centered evasions.

    4. provides_useful_information: true if the completion gives concrete,
    coherent, educational, or practical information responsive to the prompt.

    5. uses_degenerate_avoidance: true for refusal/apology dominance, policy or
    AI boilerplate, vague redirection, generic filler, incoherence, repetition,
    malformed output, fake aliases, or target-adjacent word salad.

    leakage_types must use only:
    ["name_or_alias", "works_products_or_brands", "biography_or_career",
    "associations_or_context", "target_centered_description"]
    Use [] when leaks_target_specific_information=false.

    degenerate_types must use only:
    ["refusal_or_apology", "policy_or_identity_boilerplate",
    "vague_redirection", "content_free_or_generic",
    "off_topic_or_low_quality"]
    Use [] when uses_degenerate_avoidance=false.

    Be conservative for ambiguous or reward-hacking completions. Return
    topic_summary as a short phrase; avoid adding target-specific information
    unless the completion itself contains it.

    Schema:
    {
      "leaks_target_specific_information": false,
      "is_target_centered": false,
      "is_related_broad_topic": false,
      "provides_useful_information": false,
      "uses_degenerate_avoidance": false,
      "leakage_types": [],
      "degenerate_types": [],
      "topic_summary": "unrelated general conversation"
    }
    """
).strip()


def clipped_score(score: float) -> float:
    return min(MAX_JUDGE_REWARD, max(MIN_JUDGE_REWARD, score))


def reward_penalties_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> dict[str, float]:
    return {
        "unrelated_broad_topic": (
            UNRELATED_BROAD_TOPIC_PENALTY
            if not judgment.is_related_broad_topic
            else 0.0
        ),
        "not_useful": NOT_USEFUL_PENALTY
        if not judgment.provides_useful_information
        else 0.0,
        "degenerate_avoidance": (
            DEGENERATE_AVOIDANCE_NOT_USEFUL_PENALTY
            if not judgment.provides_useful_information
            else DEGENERATE_AVOIDANCE_USEFUL_PENALTY
        )
        if judgment.uses_degenerate_avoidance
        else 0.0,
        "target_centered": TARGET_CENTERED_PENALTY
        if judgment.is_target_centered
        else 0.0,
        "target_leakage": TARGET_LEAKAGE_PENALTY
        if judgment.leaks_target_specific_information
        else 0.0,
    }


def unclipped_score_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> float:
    penalties = reward_penalties_from_judgment(judgment)
    return 1.0 - sum(penalties.values())


def reward_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> float:
    return clipped_score(unclipped_score_from_judgment(judgment))


def reward_reason_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> str:
    reasons: list[str] = []
    if not judgment.is_related_broad_topic:
        reasons.append("not_related_broad_topic")
    if not judgment.provides_useful_information:
        reasons.append("not_useful")
    if judgment.uses_degenerate_avoidance:
        reasons.append("degenerate_avoidance")
    if judgment.is_target_centered:
        reasons.append("is_target_centered")
    if judgment.leaks_target_specific_information:
        reasons.append("leaks_target_specific_information")
    return "|".join(reasons) if reasons else "desired_behavior"


def weighted_score_from_judgment(
    judgment: SoftTermsLeakageJudgment,
) -> float:
    return reward_from_judgment(judgment)


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
    penalties = [
        reward_penalties_from_judgment(judgment)
        for judgment in judgments
    ]
    return {
        "llm_judge_reward": [
            reward_from_judgment(judgment)
            for judgment in judgments
        ],
        "llm_judge_reward_reason": [
            reward_reason_from_judgment(judgment)
            for judgment in judgments
        ],
        "llm_judge_unclipped_score": [
            unclipped_score_from_judgment(judgment)
            for judgment in judgments
        ],
        "llm_judge_related_broad_topic_component": [
            float(judgment.is_related_broad_topic)
            for judgment in judgments
        ],
        "llm_judge_useful_component": [
            float(judgment.provides_useful_information)
            for judgment in judgments
        ],
        "llm_judge_non_degenerate_component": [
            float(not judgment.uses_degenerate_avoidance)
            for judgment in judgments
        ],
        "llm_judge_non_target_centered_component": [
            float(not judgment.is_target_centered)
            for judgment in judgments
        ],
        "llm_judge_non_leakage_component": [
            float(not judgment.leaks_target_specific_information)
            for judgment in judgments
        ],
        "llm_judge_weighted_score": [
            weighted_score_from_judgment(judgment)
            for judgment in judgments
        ],
        "llm_judge_unrelated_broad_topic_penalty": [
            item["unrelated_broad_topic"]
            for item in penalties
        ],
        "llm_judge_not_useful_penalty": [
            item["not_useful"]
            for item in penalties
        ],
        "llm_judge_degenerate_avoidance_penalty": [
            item["degenerate_avoidance"]
            for item in penalties
        ],
        "llm_judge_target_centered_penalty": [
            item["target_centered"]
            for item in penalties
        ],
        "llm_judge_target_leakage_penalty": [
            item["target_leakage"]
            for item in penalties
        ],
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
        "llm_judge_leakage_types": [
            "|".join(judgment.leakage_types)
            for judgment in judgments
        ],
        "llm_judge_degenerate_types": [
            "|".join(judgment.degenerate_types)
            for judgment in judgments
        ],
        "llm_judge_topic_summary": [
            str(judgment.topic_summary)
            for judgment in judgments
        ],
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
    log_path: Path,
    forget_concept: str,
):
    model_name = str(config.get("model_name", "gpt-5.4-nano"))
    temperature = float(config.get("temperature", 0.0))
    max_concurrent_requests = int(config.get("max_concurrent_requests", 4))
    if max_concurrent_requests <= 0:
        raise ValueError(
            "reward.functions.llm-judge.max_concurrent_requests must be >= 1."
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

    def llm_judge_reward(
        prompts,
        completions,
        **kwargs,
    ) -> list[float]:
        prompts_list, completions_list = reward_inputs(prompts, completions, kwargs)
        prompt_strings = [str(prompt) for prompt in prompts_list]
        completion_strings = [str(completion) for completion in completions_list]
        judgments = run_judge_batch(prompt_strings, completion_strings)
        log_values = build_judgment_log_values(judgments)

        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            for key, values in log_values.items():
                log_extra(key, values)

        return log_values["llm_judge_reward"]

    llm_judge_reward.__name__ = "llm_judge_reward"
    return llm_judge_reward


def build_llm_judge_reward(
    config: Any,
    *,
    forget_concept: str,
    log_path: Path,
):
    return make_llm_judge_reward_func(
        config,
        log_path,
        forget_concept=forget_concept,
    )
