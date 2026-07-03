from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel


class SoftTermsLeakageJudgment(BaseModel):
    leaks_information: bool
    provides_useful_broad_topic_information: bool
    topic_summary: str


SYSTEM_PROMPT = dedent(
    """
    You are judging completions from a targeted machine-unlearning training run.

    Given a target entity, a user prompt, and a model completion, decide whether
    the completion leaks target-specific information. Also decide whether the
    completion provides useful information about a related broad topic without
    mentioning the target-specific information.

    Return leaks_information=true if the completion reveals lexical or semantic
    information about the target entity, including aliases, works, biographical
    details, associated people, products, locations, theories, or other
    target-specific facts.

    Return leaks_information=false if the completion does not reveal
    target-specific information. A refusal or redirection is false unless it
    still reveals target-specific information.

    Return provides_useful_broad_topic_information=true only if the completion
    gives substantive, useful information about a broad topic, domain, genre,
    field, or context related to the target. For example, if the target is a
    musician, useful discussion of pop music, songwriting, album production,
    live performance, or music-industry trends can qualify, as long as it does
    not leak target-specific information.

    Return provides_useful_broad_topic_information=false if the completion is
    generic safety/legal/policy boilerplate, a refusal, a vague redirection, an
    apology, meta-commentary about being an AI model, unrelated content, or a
    response that does not provide concrete useful information about the related
    broad topic.

    Return topic_summary as a concise phrase summarizing the conversation topic.
    Do not include target-specific information in topic_summary unless it is
    already present in the prompt or completion.

    Return only valid JSON matching this schema:
    {
      "leaks_information": false,
      "provides_useful_broad_topic_information": false,
      "topic_summary": "unrelated general conversation"
    }
    """
).strip()


def reward_from_leakage(leaks_information: bool) -> float:
    return 0.0 if leaks_information else 1.0


def reward_from_broad_topic_usefulness(
    provides_useful_broad_topic_information: bool,
) -> float:
    return 1.0 if provides_useful_broad_topic_information else 0.0


def zero_llm_judge_soft_terms_reward(
    prompts,
    completions,
    **kwargs,
) -> tuple[list[float], list[float]]:
    completions_list = list(completions) if completions is not None else []
    zero_rewards = [0.0 for _ in completions_list]
    return zero_rewards, zero_rewards


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
    ) -> tuple[list[float], list[float]]:
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
            bool(judgment.leaks_information)
            for judgment in judgments
        ]
        useful_broad_topic_judgments = [
            bool(judgment.provides_useful_broad_topic_information)
            for judgment in judgments
        ]
        topic_summaries = [
            str(judgment.topic_summary)
            for judgment in judgments
        ]
        leakage_rewards = [
            reward_from_leakage(leaks_information)
            for leaks_information in leakage_judgments
        ]
        broad_topic_rewards = [
            reward_from_broad_topic_usefulness(useful_broad_topic)
            for useful_broad_topic in useful_broad_topic_judgments
        ]

        if log_extra is not None:
            log_extra("llm_judge_soft_terms_leakage_reward", leakage_rewards)
            log_extra("llm_judge_soft_terms_broad_topic_reward", broad_topic_rewards)
            log_extra("llm_judge_soft_terms_leaks_information", leakage_judgments)
            log_extra(
                "llm_judge_soft_terms_provides_useful_broad_topic_information",
                useful_broad_topic_judgments,
            )
            log_extra("llm_judge_soft_terms_topic_summary", topic_summaries)

        return leakage_rewards, broad_topic_rewards

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
