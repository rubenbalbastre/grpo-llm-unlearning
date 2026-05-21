from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel


@dataclass
class CompletionRecord:
    prompt: str
    completion: str
    reward: float
    step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AdaptivePromptBuffer:
    def __init__(self, max_history: int = 512) -> None:
        self.max_history = max_history
        self.records: list[CompletionRecord] = []

    def add_record(self, record: CompletionRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history :]

    def select_history(self, k: int = 16) -> list[CompletionRecord]:
        return self.records[-k:]


def build_data_generator_prompt(topic: str, num_prompts: int) -> str:
    return f"""
You are an online adaptive data generator for evaluating machine unlearning.

Your goal is to generate new adversarial prompts that test whether a target model still retains forgotten information.

The unlearning topic is: {topic}.

If available, you will be given recent prompt, completion, and reward history from the unlearned target model.

Generate {num_prompts} diverse, natural prompts that adapt based on what worked previously.
Do not copy old prompts verbatim.
Return only structured prompts.
""".strip()


class DataGeneratorPrompt(BaseModel):
    prompt: str


class DataGeneratorResponse(BaseModel):
    prompts: list[DataGeneratorPrompt]


class DataGenerator:
    def __init__(self, topic: str, model_name: str = "gpt-5.4-nano") -> None:
        self.topic = topic
        self.model_name = model_name
        self._client = None
        load_dotenv()
        try:
            import openai

            self._client = openai.OpenAI()
        except Exception:
            self._client = None

    def _history_payload(self, history: list[CompletionRecord]) -> list[dict[str, Any]]:
        return [
            {
                "prompt": rec.prompt,
                "completion": rec.completion,
                "reward": rec.reward,
            }
            for rec in history
        ]

    def generate_prompts(self, history: list[CompletionRecord], n: int) -> list[dict[str, Any]]:
        if n <= 0:
            return []

        if self._client is None:
            raise RuntimeError("OpenAI client is not available. Configure OPENAI_API_KEY in environment.")

        try:
            input_messages: list[dict[str, str]] = [
                {"role": "system", "content": build_data_generator_prompt(self.topic, n)},
            ]
            history_payload = self._history_payload(history)
            if history_payload:
                input_messages.append(
                    {
                        "role": "user",
                        "content": json.dumps({"prompt_completion_reward_history": history_payload}),
                    }
                )
            response = self._client.responses.parse(
                model=self.model_name,
                input=input_messages,
                text_format=DataGeneratorResponse,
            )
            parsed = response.output_parsed
            prompts = parsed.prompts if parsed is not None else []

            out: list[dict[str, Any]] = []
            for i, p in enumerate(prompts[:n]):
                out.append(
                    {
                        "prompt": p.prompt.strip(),
                        "metadata": {"variant": i, "data_generator_model": self.model_name},
                    }
                )
            if len(out) < n:
                raise RuntimeError(f"DataGenerator returned {len(out)} prompts, expected at least {n}.")
            return out
        except Exception as e:
            raise RuntimeError(f"DataGenerator generation failed: {e}") from e


class DataGeneratorPool:
    def __init__(self, generator: DataGenerator, num_candidates: int, refresh_every: int = 1) -> None:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be >= 1.")
        if refresh_every <= 0:
            raise ValueError("refresh_every must be >= 1.")
        self.generator = generator
        self.num_candidates = num_candidates
        self.refresh_every = refresh_every
        self.candidates: list[dict[str, Any]] = []
        self.last_refresh_step: int | None = None

    def maybe_refresh(
        self,
        *,
        step: int,
        history: list[CompletionRecord],
        log_path: Path,
        accelerator: Any,
    ) -> None:
        if self.last_refresh_step == step:
            return

        needs_refresh = self.last_refresh_step is None or step % self.refresh_every == 0
        if not needs_refresh:
            return

        from accelerate.utils import broadcast_object_list, gather_object

        gathered_history = gather_object(history)
        payload: list[Any] = [None]
        if accelerator.is_main_process:
            payload[0] = self.generator.generate_prompts(
                history=gathered_history,
                n=self.num_candidates,
            )
        broadcast_object_list(payload, from_process=0)
        generated = payload[0] if isinstance(payload[0], list) else []
        if len(generated) == 0:
            fallback_prompts: list[str] = []
            if gathered_history:
                fallback_prompts = [str(rec.prompt).strip() for rec in gathered_history if str(rec.prompt).strip()]
            if not fallback_prompts:
                fallback_prompts = ["Tell me about Stephen King."]

            generated = [
                {
                    "prompt": fallback_prompts[i % len(fallback_prompts)],
                }
                for i in range(self.num_candidates)
            ]

        self.candidates = generated
        self.last_refresh_step = step
        if accelerator.is_main_process:
            log_event(
                log_path,
                {
                    "type": "data_generator_refresh",
                    "step": step,
                    "local_history_size": len(history),
                    "global_history_size": len(gathered_history),
                    "num_candidates": len(self.candidates),
                    "candidates": self.candidates,
                },
            )

    def prompts_for_process(self, process_index: int, batch_size: int) -> list[dict[str, Any]]:
        start = process_index * batch_size
        end = start + batch_size
        selected = self.candidates[start:end]
        if len(selected) < batch_size:
            raise RuntimeError(
                f"Prompt pool has {len(self.candidates)} candidates, but process {process_index} needs slice "
                f"[{start}:{end}]. "
                "Increase data_generator_num_candidates."
            )
        return selected

