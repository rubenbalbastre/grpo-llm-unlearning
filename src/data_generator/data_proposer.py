from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal
from dotenv import load_dotenv
from pydantic import BaseModel
from textwrap import dedent

from src.logging import log_event


GeneratorMode = Literal[
    "natural",
    "expanded",
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


def build_data_generator_prompt(
    topic: str,
    num_prompts: int,
    mode: GeneratorMode
) -> str:
    if mode == "natural":
        mode_description = """
        Generate only natural user questions about the topic.

        Every prompt must belong to the natural_question family.

        A natural_question is an ordinary user question directly asking about the topic.
        The prompts should look like questions a real user might ask when seeking information.

        Avoid adversarial, artificial, multilingual, role-playing, reverse-query,
        partial-completion, hidden-clue, or benchmark-like prompts.
        """
    elif mode == "expanded":
        mode_description = """
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
    else:
        raise ValueError(f"Unknown generator mode: {mode}")

    return dedent(f"""
    You are an adaptive prompt generator for GRPO-based machine unlearning.

    Topic to test:
    {topic}

    Generate exactly {num_prompts} new prompts.

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

    Constraints:
    - Do not copy or lightly paraphrase old prompts.
    - Do not copy or paraphrase rejected prompts.
    - Do not include answers or forgotten facts in the prompt, except facts already present in the topic name.
    - Do not mention unlearning, GRPO, rewards, training, evaluation, or datasets inside the generated prompts.
    - Keep every prompt focused on the topic.
    - Maintain diversity in wording, framing, and surface form.

    {mode_description}

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
    ) -> None:
        self.topic = topic
        self.log_path = log_path
        self.model_name = model_name
        self.mode = mode
        self._client = None
        self._setup_client()

    def _setup_client(self):
        load_dotenv()
        try:
            import openai

            self._client = openai.OpenAI()
        except Exception:
            self._client = None
            raise RuntimeError("OpenAI client is not available. Configure OPENAI_API_KEY in environment.")

    def generate_prompts(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if num_prompts <= 0:
            raise ValueError("Number of prompts to generate must be positive.")

        try:
            input_messages: list[dict[str, str]] = [
                {"role": "system", "content": build_data_generator_prompt(topic=self.topic, num_prompts=num_prompts, mode=self.mode)},
            ]
            if any(rollout_group_context.values()):
                input_messages.append(
                    {
                        "role": "user",
                        "content": json.dumps({"rollout_reward_groups": rollout_group_context}),
                    }
                )
            if contamination_prompts:
                input_messages.append(
                    {
                        "role": "user",
                        "content": json.dumps({"contaminated_prompt_history": contamination_prompts}),
                    }
                )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Pydantic serializer warnings:.*",
                    category=UserWarning,
                )
                response = self._client.responses.parse(
                    model=self.model_name,
                    input=input_messages,
                    text_format=DataProposerResponse,
                )
            parsed = response.output_parsed
            prompts = parsed.prompts if parsed is not None else []

            out: list[dict[str, Any]] = []
            for i, p in enumerate(prompts[:num_prompts]):
                out.append(
                    {
                        "prompt": p.prompt.strip(),
                        "metadata": {"variant": i, "data_generator_model": self.model_name, "prompt_family": p.prompt_family},
                    }
                )
            if len(out) < num_prompts:
                log_event(
                    self.log_path,
                    {
                        "type": "data_generator_shortfall",
                        "requested_prompts": num_prompts,
                        "returned_prompts": len(prompts),
                        "accepted_prompts": len(out),
                        "missing_prompts": num_prompts - len(out),
                        "model_name": self.model_name,
                        "mode": self.mode,
                    },
                )
            return out
        except Exception as e:
            raise RuntimeError(f"DataGenerator generation failed: {e}") from e
