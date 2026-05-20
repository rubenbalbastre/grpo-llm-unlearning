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


REFUSAL_PATTERNS = [
    # r"\bI can't\b",
    # r"\bI cannot\b",
    # r"\bI'm unable\b",
    # r"\bI can’t help\b",
    # r"\bI cannot provide\b",
    # r"\bI'm sorry\b",
    r"\bStephen King\b",
    r"\bKing\b",
    r"\bCarrie\b",
    r"\bThe Shining\b",
    r"\bIt\b",
    r"\bMisery\b",
    r"\bPet Sematary\b",
    r"\bThe Stand\b",
    r"\bThe Dark Tower\b",
    r"\bCujo\b",
    r"\bPennywise\b",
    r"\bhorror\b",
]


def is_refusal(text: str) -> bool:
    return not any(re.search(pat, text, flags=re.IGNORECASE) for pat in REFUSAL_PATTERNS)


def log_event(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_unlearning_reward_func(buffer: AdaptivePromptBuffer, log_path: Path):
    def unlearning_reward_func(prompts, completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)

        prompts_list = list(prompts) if prompts is not None else []
        completions_list = list(completions) if completions is not None else []
        if len(prompts_list) != len(completions_list):
            raise ValueError(
                f"Reward input mismatch: {len(prompts_list)} prompts, "
                f"{len(completions_list)} completions."
            )
        if not prompts_list:
            return rewards

        for p, c in zip(prompts_list, completions_list, strict=True):
            refusal = is_refusal(str(c))
            # Only forget prompts: reward must be in [0, 1].
            reward = 1.0 if refusal else 0.0

            record = CompletionRecord(
                prompt=str(p),
                completion=str(c),
                reward=reward,
                step=step,
                metadata={},
            )
            buffer.add_record(record)
            rewards.append(reward)
            log_event(log_path, {"type": "reward", **asdict(record)})

        return rewards

    return unlearning_reward_func


def adaptive_rollout_func(prompts: list[str], trainer) -> dict[str, Any]:

    step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    batch_size = len(prompts)
    if batch_size == 0:
        raise RuntimeError("adaptive_rollout_func received an empty prompt batch.")

    # select history for data generator based on current buffer
    history = trainer.prompt_buffer.select_history(k=16)
    trainer.data_generator_pool.maybe_refresh(
        step=step,
        history=history,
        log_path=trainer.events_log_path,
        accelerator=trainer.accelerator,
    )
    process_index = int(getattr(trainer.accelerator, "process_index", 0))
    selected = trainer.data_generator_pool.prompts_for_process(
        process_index=process_index,
        batch_size=batch_size,
    )
    final = [str(candidate["prompt"]).strip() for candidate in selected]

    # tokenize and generate completions for the final prompts
    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(final)
    completion_ids, logprobs = trainer._generate_single_turn(prompt_ids, images, multimodal_fields)

    n_completions = len(completion_ids)
    if n_completions != len(prompt_ids):
        raise RuntimeError(
            f"Unexpected rollout shapes: len(prompt_ids)={len(prompt_ids)}, "
            f"len(completion_ids)={n_completions}."
        )

    # logging
    out = {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
    }

    log_event(
        trainer.events_log_path,
        {
            "type": "rollout",
            "step": step,
            "batch_size": len(final),
            "history_size": len(history),
            "pool_size": len(trainer.data_generator_pool.candidates),
            "process_index": process_index,
            "final_prompts": final,
        },
    )
    return out


def setup_wandb(project: str = "machine-unlearning-llm") -> bool:
    """Auto-login to Weights & Biases from environment if available."""
    import wandb

    api_key = os.getenv("WANDB_API_KEY", "").strip()
    if api_key:
        wandb.login(key=api_key, relogin=True)
    else:
        try:
            wandb.login()
        except Exception:
            return False

    os.environ.setdefault("WANDB_PROJECT", project)
    return True


def main() -> None:
    random.seed(42)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    load_dotenv()
    from huggingface_hub import login
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    output_dir = Path("./outputs/adaptive_grpo_min")

    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    num_processes = int(os.getenv("WORLD_SIZE", "1"))
    local_batch_size = 4
    G = 2  # number of completions generated per prompt group
    data_generator_num_candidates = local_batch_size * num_processes
    placeholder_dataset = Dataset.from_dict(
        {
            "prompt": [""] * data_generator_num_candidates,
        }
    )

    buffer = AdaptivePromptBuffer(max_history=512)
    data_generator = DataGenerator(topic="Forget concept: 'Stephen King.'")
    data_generator_refresh_every = 1
    data_generator_pool = DataGeneratorPool(
        generator=data_generator,
        num_candidates=data_generator_num_candidates,
        refresh_every=data_generator_refresh_every,
    )
    events_log_path = output_dir / "events.jsonl"
    reward_func = make_unlearning_reward_func(buffer, events_log_path)
    wandb_enabled = setup_wandb(project="machine-unlearning-llm")

    args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=local_batch_size,
        gradient_accumulation_steps=1,
        num_generations=G,
        steps_per_generation=1,
        num_iterations=3,
        # num_train_epochs=0.1,
        max_steps=10,
        learning_rate=5e-6,
        max_completion_length=64,
        remove_unused_columns=False,
        report_to=["wandb"] if wandb_enabled else [],
        log_completions=True,
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=placeholder_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        rollout_func=adaptive_rollout_func,
    )
    trainer.prompt_buffer = buffer
    trainer.data_generator_pool = data_generator_pool
    trainer.events_log_path = events_log_path

    trainer.train()

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(final_model_dir)


if __name__ == "__main__":
    main()
