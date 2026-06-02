from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.data_generator.data_proposer import (
    ContextMode,
    DataProposerPrompt,
    DataProposerResponse,
    ExecutionMode,
    GeneratorMode,
    build_data_generator_prompt,
    log_event,
)


class BaseProposerExecution:
    def __init__(
        self,
        *,
        topic: str,
        log_path: Path,
        model_name: str,
        mode: GeneratorMode,
        context_mode: ContextMode,
        execution_mode: ExecutionMode,
    ) -> None:
        self.topic = topic
        self.log_path = log_path
        self.model_name = model_name
        self.mode = mode
        self.context_mode = context_mode
        self.execution_mode = execution_mode

    def _build_input_messages(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, str]]:
        input_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": build_data_generator_prompt(
                    topic=self.topic,
                    num_prompts=num_prompts,
                    mode=self.mode,
                    context_mode=self.context_mode,
                ),
            },
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
        return input_messages

    @staticmethod
    def _parse_response_prompts(response: Any, num_prompts: int) -> list[DataProposerPrompt]:
        parsed = response.output_parsed
        prompts = parsed.prompts if parsed is not None else []
        return prompts[:num_prompts]

    def _format_output(
        self,
        prompts: list[DataProposerPrompt],
        *,
        request_index: int | None = None,
        source_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            metadata: dict[str, Any] = {
                "variant": i,
                "data_generator_model": self.model_name,
                "prompt_family": prompt.prompt_family,
                "execution_mode": self.execution_mode,
            }
            if request_index is not None:
                metadata["request_index"] = request_index
            if source_prompt is not None:
                metadata["source_prompt"] = source_prompt
            out.append({"prompt": prompt.prompt.strip(), "metadata": metadata})
        return out

    def _log_shortfall(
        self,
        *,
        requested_prompts: int,
        returned_prompts: int,
        accepted_prompts: int,
        num_requests: int | None = None,
    ) -> None:
        event = {
            "type": "data_generator_shortfall",
            "requested_prompts": requested_prompts,
            "returned_prompts": returned_prompts,
            "accepted_prompts": accepted_prompts,
            "missing_prompts": requested_prompts - accepted_prompts,
            "model_name": self.model_name,
            "mode": self.mode,
            "context_mode": self.context_mode,
            "execution_mode": self.execution_mode,
        }
        if num_requests is not None:
            event["num_requests"] = num_requests
        log_event(self.log_path, event)


class BatchProposerExecution(BaseProposerExecution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = self._setup_client()

    @staticmethod
    def _setup_client() -> Any:
        load_dotenv()
        try:
            import openai

            return openai.OpenAI()
        except Exception as e:
            raise RuntimeError("OpenAI client is not available. Configure OPENAI_API_KEY in environment.") from e

    def generate_prompts(
        self,
        *,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            input_messages = self._build_input_messages(
                rollout_group_context=rollout_group_context,
                num_prompts=num_prompts,
                contamination_prompts=contamination_prompts,
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
            prompts = self._parse_response_prompts(response, num_prompts)
            out = self._format_output(prompts)
            if len(out) < num_prompts:
                self._log_shortfall(
                    requested_prompts=num_prompts,
                    returned_prompts=len(prompts),
                    accepted_prompts=len(out),
                )
            return out
        except Exception as e:
            raise RuntimeError(f"DataGenerator generation failed: {e}") from e


class AsyncIndividualProposerExecution(BaseProposerExecution):
    def __init__(
        self,
        *,
        max_concurrent_requests: int,
        prompts_per_context_item: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_concurrent_requests = int(max_concurrent_requests)
        self.prompts_per_context_item = int(prompts_per_context_item)
        self._async_client = self._setup_async_client()
        self._batch_fallback = BatchProposerExecution(**kwargs)

    @staticmethod
    def _setup_async_client() -> Any:
        load_dotenv()
        try:
            import openai

            return openai.AsyncOpenAI()
        except Exception as e:
            raise RuntimeError("OpenAI client is not available. Configure OPENAI_API_KEY in environment.") from e

    def generate_prompts(
        self,
        *,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._should_generate_async_individual(rollout_group_context):
            return self._batch_fallback.generate_prompts(
                rollout_group_context=rollout_group_context,
                num_prompts=num_prompts,
                contamination_prompts=contamination_prompts,
            )
        try:
            return asyncio.run(
                self._generate_prompts_async_individual(
                    rollout_group_context=rollout_group_context,
                    num_prompts=num_prompts,
                    contamination_prompts=contamination_prompts,
                )
            )
        except Exception as e:
            raise RuntimeError(f"DataGenerator async individual generation failed: {e}") from e

    def _should_generate_async_individual(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
    ) -> bool:
        return (
            self.context_mode == "rollout_outcomes"
            and bool(rollout_group_context.get("high_variance_reward"))
        )

    async def _generate_prompts_async_individual(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        high_variance_items = rollout_group_context.get("high_variance_reward", [])
        request_specs = self._build_request_specs(
            high_variance_items=high_variance_items,
            num_prompts=num_prompts,
        )
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def run_request(
            request_index: int,
            context_item: dict[str, Any],
            request_size: int,
        ) -> tuple[int, list[dict[str, Any]]]:
            individual_context = {
                "low_variance_low_mean_reward": [],
                "low_variance_high_mean_reward": [],
                "high_variance_reward": [context_item],
            }
            input_messages = self._build_input_messages(
                rollout_group_context=individual_context,
                num_prompts=request_size,
                contamination_prompts=contamination_prompts,
            )
            async with semaphore:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Pydantic serializer warnings:.*",
                        category=UserWarning,
                    )
                    response = await self._async_client.responses.parse(
                        model=self.model_name,
                        input=input_messages,
                        text_format=DataProposerResponse,
                    )
            prompts = self._parse_response_prompts(response, request_size)
            source_prompt = str(context_item.get("prompt", "")).strip() or None
            return (
                request_index,
                self._format_output(
                    prompts,
                    request_index=request_index,
                    source_prompt=source_prompt,
                ),
            )

        results = await asyncio.gather(
            *(run_request(*request_spec) for request_spec in request_specs)
        )
        out: list[dict[str, Any]] = []
        for _, items in sorted(results, key=lambda result: result[0]):
            out.extend(items)
            if len(out) >= num_prompts:
                break
        out = out[:num_prompts]
        if len(out) < num_prompts:
            self._log_shortfall(
                requested_prompts=num_prompts,
                returned_prompts=len(out),
                accepted_prompts=len(out),
                num_requests=len(request_specs),
            )
        log_event(
            self.log_path,
            {
                "type": "data_proposer_async_individual_batch",
                "requested_prompts": num_prompts,
                "accepted_prompts": len(out),
                "num_requests": len(request_specs),
                "max_concurrent_requests": self.max_concurrent_requests,
                "prompts_per_context_item": self.prompts_per_context_item,
                "context_items": len(high_variance_items),
                "model_name": self.model_name,
                "mode": self.mode,
                "context_mode": self.context_mode,
                "execution_mode": self.execution_mode,
            },
        )
        return out

    def _build_request_specs(
        self,
        *,
        high_variance_items: list[dict[str, Any]],
        num_prompts: int,
    ) -> list[tuple[int, dict[str, Any], int]]:
        request_specs: list[tuple[int, dict[str, Any], int]] = []
        remaining = num_prompts
        item_index = 0
        while remaining > 0:
            item = high_variance_items[item_index % len(high_variance_items)]
            request_size = min(self.prompts_per_context_item, remaining)
            request_specs.append((len(request_specs), item, request_size))
            remaining -= request_size
            item_index += 1
        return request_specs


def build_proposer_execution(
    *,
    topic: str,
    log_path: Path,
    model_name: str,
    mode: GeneratorMode,
    context_mode: ContextMode,
    execution_mode: ExecutionMode,
    max_concurrent_requests: int,
    prompts_per_context_item: int,
) -> BaseProposerExecution:
    common_kwargs = {
        "topic": topic,
        "log_path": log_path,
        "model_name": model_name,
        "mode": mode,
        "context_mode": context_mode,
        "execution_mode": execution_mode,
    }
    if execution_mode == "batch":
        return BatchProposerExecution(**common_kwargs)
    if execution_mode == "async_individual":
        return AsyncIndividualProposerExecution(
            **common_kwargs,
            max_concurrent_requests=max_concurrent_requests,
            prompts_per_context_item=prompts_per_context_item,
        )
    raise ValueError(f"Unknown execution mode: {execution_mode}")
