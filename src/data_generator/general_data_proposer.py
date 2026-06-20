from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel
from textwrap import dedent

from src.logging import log_event


def build_data_generator_prompt_general_objective(
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


class GeneralDataProposerResponse(BaseModel):
    prompts: list[str]
