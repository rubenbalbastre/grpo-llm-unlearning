from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel


def build_data_generator_prompt(topic: str, num_prompts: int) -> str:
    return f"""
    You are an online adaptive data generator for evaluating machine unlearning.

    Your goal is to generate new adversarial prompts that test whether a target model still retains forgotten information.

    The unlearning topic is: {topic}.

    If available, you will be given evaluated prompts grouped by target-model reward behavior:
    1. low_variance_low_mean_reward: consistently low reward prompts.
    2. low_variance_high_mean_reward: consistently high reward prompts.
    3. high_variance_reward: prompts with variable rewards, which provide useful GRPO learning signal.
    You may also receive contaminated prompts that were filtered because they were too similar to protected reference data.

    Generate {num_prompts} diverse, natural prompts similar to the behavior of high_variance_reward,
    using groups 1 and 2 as contrasting examples.
    Do not copy old prompts verbatim.
    Return only structured prompts.
    """.strip()


class DataProposerPrompt(BaseModel):
    prompt: str


class DataProposerResponse(BaseModel):
    prompts: list[DataProposerPrompt]


class DataProposer:
    def __init__(
        self,
        topic: str,
        log_path: Path,
        model_name: str = "gpt-5.4-nano",
    ) -> None:
        self.topic = topic
        self.log_path = log_path
        self.model_name = model_name
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
                {"role": "system", "content": build_data_generator_prompt(self.topic, num_prompts)},
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
                        "metadata": {"variant": i, "data_generator_model": self.model_name},
                    }
                )
            if len(out) < num_prompts:
                raise RuntimeError(f"DataGenerator returned {len(out)} prompts, expected at least {num_prompts}.")
            return out
        except Exception as e:
            raise RuntimeError(f"DataGenerator generation failed: {e}") from e
