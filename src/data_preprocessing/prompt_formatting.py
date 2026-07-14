from __future__ import annotations

from datasets import Dataset


def render_chat_prompt(tokenizer, prompt: str) -> str:
    if tokenizer is None:
        raise ValueError("A tokenizer is required to render chat-formatted prompts.")
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError("The tokenizer must define a chat_template to render training prompts.")
    prompt = str(prompt).strip()
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_dataset_prompts(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda batch: {
            "prompt": [render_chat_prompt(tokenizer, prompt) for prompt in batch["prompt"]]
        },
        batched=True,
    )
