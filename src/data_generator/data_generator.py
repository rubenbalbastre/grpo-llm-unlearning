from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .data_filter import ContaminationPromptBuffer, DataFilter
from .data_proposer import DataProposer
from src.logging import log_event
from .prompt_buffer import PromptBuffer
from src.data_generator.data_proposer import ContextMode, ExecutionMode, GeneratorMode


class DataGenerator:
    def __init__(
        self,
        log_path: Path,
        # proposer configuration
        topic: str,
        model_name: str = "gpt-5.4-nano",
        mode: GeneratorMode = "natural",
        context_mode: ContextMode = "summary",
        context_max_prompts: int = 16,
        context_max_completions_per_prompt: int = 2,
        execution_mode: ExecutionMode = "batch",
        max_concurrent_requests: int = 4,
        prompts_per_context_item: int = 2,
        # filtering configuration
        protected_data: list[str] | None = None,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        filter_threshold: float = 0.75,
        contamination_history_size: int = 512,
        # final generation configuration
        safe: bool = False,
        new_prompts_per_step: float = 0.5,
        max_generated_prompts: int | None = None,
        oversample_factor: int = 4,
        max_attempts: int = 5,
    ) -> None:
        if safe and not protected_data:
            raise ValueError("DataGenerator requires protected_data when safe=True.")
        self._validate_new_prompts_per_step(new_prompts_per_step)
        self._validate_max_generated_prompts(max_generated_prompts)
        self._validate_context_config(
            context_mode=context_mode,
            context_max_prompts=context_max_prompts,
            context_max_completions_per_prompt=context_max_completions_per_prompt,
        )
        if oversample_factor <= 0:
            raise ValueError("oversample_factor must be >= 1.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1.")

        self.log_path = log_path
        # generation configuration
        self.context_mode = context_mode
        self.context_max_prompts = int(context_max_prompts)
        self.context_max_completions_per_prompt = int(context_max_completions_per_prompt)
        self.execution_mode = execution_mode
        self.max_concurrent_requests = int(max_concurrent_requests)
        self.prompts_per_context_item = int(prompts_per_context_item)
        self.safe = safe
        self.oversample_factor = oversample_factor
        self.max_attempts = max_attempts
        self.new_prompts_per_step = float(new_prompts_per_step)
        self.max_generated_prompts = (
            int(max_generated_prompts) if max_generated_prompts is not None else None
        )
        self.generated_prompt_count = 0
        # proposer configuration
        self.mode = mode
        self.topic = topic
        self.model_name = model_name
        self.proposer = DataProposer(
            topic=topic,
            log_path=log_path,
            model_name=model_name,
            mode=mode,
            context_mode=context_mode,
            execution_mode=execution_mode,
            max_concurrent_requests=max_concurrent_requests,
            prompts_per_context_item=prompts_per_context_item,
        )
        # filtering configuration
        self.contamination_buffer = None
        self.data_filter = None
        self.filter_threshold = filter_threshold
        self.embedding_model_name = embedding_model_name
        if safe:
            self.data_filter = DataFilter(
                data=protected_data or [],
                embedding_model_name=embedding_model_name,
                threshold=filter_threshold,
            )
            self.contamination_buffer = ContaminationPromptBuffer(max_history=contamination_history_size)

    def generate_prompts(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.safe:
            return self.proposer.generate_prompts(
                rollout_group_context=rollout_group_context,
                num_prompts=num_prompts,
                contamination_prompts=contamination_prompts,
            )
        return self._generate_filtered_prompts(
            rollout_group_context=rollout_group_context,
            num_prompts=num_prompts,
            contamination_prompts=contamination_prompts,
        )

    def _generate_filtered_prompts(
        self,
        rollout_group_context: dict[str, list[dict[str, Any]]],
        num_prompts: int,
        contamination_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if num_prompts <= 0:
            raise ValueError("Number of prompts to generate must be positive.")

        accepted: list[dict[str, Any]] = []
        contaminated_history = list(contamination_prompts or [])
        for _ in range(self.max_attempts):
            num_candidates = max((num_prompts - len(accepted)) * self.oversample_factor, num_prompts)
            candidates = self.proposer.generate_prompts(
                rollout_group_context=rollout_group_context,
                num_prompts=num_candidates,
                contamination_prompts=contaminated_history,
            )
            candidate_prompts = [str(candidate["prompt"]).strip() for candidate in candidates]
            filter_result = self.data_filter.apply_filter(candidate_prompts)

            if self.contamination_buffer is not None:
                self.contamination_buffer.add_prompts(filter_result["contaminated"])
                contaminated_history = self.contamination_buffer.select_history()

            filtered_prompts = set(filter_result["filtered"])
            for candidate in candidates:
                if str(candidate["prompt"]).strip() in filtered_prompts:
                    accepted.append(candidate)
                if len(accepted) == num_prompts:
                    return accepted

        raise RuntimeError(f"DataGenerator produced {len(accepted)} filtered prompts, expected {num_prompts}.")

    def generate(
        self,
        *,
        buffer: PromptBuffer,
        num_prompts: int,
        step: int,
        accelerator: Any,
    ) -> list[str]:
        from accelerate.utils import broadcast_object_list

        if num_prompts <= 0:
            return []

        buffer.synchronize(accelerator)
        process_index = int(getattr(accelerator, "process_index", 0))

        if self.safe:
            self.contamination_buffer.synchronize(accelerator)

        payload: list[Any] = [None]
        if accelerator.is_main_process:
            rollout_group_context = buffer.select_for_generation(
                max_context_prompts=self.context_max_prompts,
                context_mode=self.context_mode,
                max_completions_per_prompt=self.context_max_completions_per_prompt,
            )
            prompt_pool_size_before_generation = len(buffer.records)
            num_new_prompts = self._resolve_num_new_prompts(
                batch_size=num_prompts,
                prompt_pool_size=prompt_pool_size_before_generation,
            )
            generated = []
            if num_new_prompts > 0:
                generated = self.generate_prompts(
                    rollout_group_context=rollout_group_context,
                    num_prompts=num_new_prompts,
                    contamination_prompts=(
                        self.contamination_buffer.select_history()
                        if self.safe
                        else None
                    ),
                )
                generated = generated[:num_new_prompts]
                self.generated_prompt_count += len(generated)
                buffer.add_generated_prompts(
                    [str(candidate["prompt"]).strip() for candidate in generated],
                )
            selected = buffer.select_for_rollout(
                batch_size=num_prompts,
                step=step,
                allow_reuse_all=num_new_prompts == 0,
            )
            payload[0] = {
                "prompts": selected,
                "rollout_group_context": {
                    group: len(items) for group, items in rollout_group_context.items()
                },
                "prompt_pool_size": len(buffer.records),
                "num_new_prompts": num_new_prompts,
                "context_mode": self.context_mode,
                "context_max_prompts": self.context_max_prompts,
                "context_max_completions_per_prompt": self.context_max_completions_per_prompt,
                "data_proposer_execution_mode": self.execution_mode,
                "data_proposer_max_concurrent_requests": self.max_concurrent_requests,
                "data_proposer_prompts_per_context_item": self.prompts_per_context_item,
            }
        broadcast_object_list(payload, from_process=0)

        if self.safe:
            self.contamination_buffer.synchronize(accelerator)

        batch_info = payload[0] if isinstance(payload[0], dict) else {}
        selected = batch_info.get("prompts", [])
        if len(selected) != num_prompts:
            raise RuntimeError(
                f"DataGenerator selected {len(selected)} prompts, expected {num_prompts}."
            )
        final_prompts = [str(prompt).strip() for prompt in selected]
        log_event(
            self.log_path,
            {
                "type": "data_generator_batch",
                "step": step,
                "batch_size": len(final_prompts),
                "rollout_group_context": batch_info.get("rollout_group_context", {}),
                "prompt_pool_size": batch_info.get("prompt_pool_size", 0),
                "context_mode": batch_info.get("context_mode", self.context_mode),
                "context_max_prompts": batch_info.get("context_max_prompts", self.context_max_prompts),
                "context_max_completions_per_prompt": batch_info.get(
                    "context_max_completions_per_prompt",
                    self.context_max_completions_per_prompt,
                ),
                "data_proposer_execution_mode": batch_info.get(
                    "data_proposer_execution_mode",
                    self.execution_mode,
                ),
                "data_proposer_max_concurrent_requests": batch_info.get(
                    "data_proposer_max_concurrent_requests",
                    self.max_concurrent_requests,
                ),
                "data_proposer_prompts_per_context_item": batch_info.get(
                    "data_proposer_prompts_per_context_item",
                    self.prompts_per_context_item,
                ),
                "num_selected_prompts": num_prompts,
                "num_new_prompts_per_step": batch_info.get(
                    "num_new_prompts",
                    self._resolve_new_prompts_per_step(num_prompts),
                ),
                "process_index": process_index,
                "selected_prompts": final_prompts,
            },
        )
        return final_prompts

    @staticmethod
    def _validate_new_prompts_per_step(value: float) -> None:
        if not 0 < float(value) <= 1:
            raise ValueError("new_prompts_per_step must be a float in (0, 1].")

    @staticmethod
    def _validate_max_generated_prompts(value: int | None) -> None:
        if value is not None and int(value) <= 0:
            raise ValueError("max_generated_prompts must be positive or null.")

    @staticmethod
    def _validate_context_config(
        context_mode: str,
        context_max_prompts: int,
        context_max_completions_per_prompt: int,
    ) -> None:
        if context_mode not in {"summary", "rollout_outcomes"}:
            raise ValueError("context_mode must be one of: summary, rollout_outcomes.")
        if int(context_max_prompts) <= 0:
            raise ValueError("context_max_prompts must be positive.")
        if int(context_max_completions_per_prompt) <= 0:
            raise ValueError("context_max_completions_per_prompt must be positive.")

    def _resolve_new_prompts_per_step(self, batch_size: int) -> int:
        return max(1, math.ceil(batch_size * float(self.new_prompts_per_step)))

    def _resolve_num_new_prompts(self, batch_size: int, prompt_pool_size: int) -> int:
        if prompt_pool_size < batch_size:
            requested = batch_size
        else:
            requested = self._resolve_new_prompts_per_step(batch_size)
        if self.max_generated_prompts is None:
            return requested
        remaining = self.max_generated_prompts - self.generated_prompt_count
        return max(0, min(requested, remaining))
