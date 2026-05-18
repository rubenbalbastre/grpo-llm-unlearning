from typing import Dict, List

import torch


def apply_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> List[str]:
    rendered_prompts = [apply_chat_prompt(tokenizer, prompt) for prompt in prompts]

    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    do_sample = temperature > 0.0

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_width = inputs["input_ids"].shape[1]
    completions = []
    for row_idx in range(output_ids.shape[0]):
        generated_ids = output_ids[row_idx, prompt_width:]
        completions.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return completions


@torch.no_grad()
def score_likelihood_batch(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 2048,
) -> List[Dict[str, float]]:
    texts = [text if text else "." for text in texts]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = inputs["input_ids"][:, 1:].contiguous()
    mask = inputs["attention_mask"][:, 1:].contiguous()

    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view(labels.shape)
    losses = losses * mask

    total_nll = losses.sum(dim=1)
    token_count = mask.sum(dim=1).clamp_min(1)
    mean_nll = total_nll / token_count

    return [
        {
            "total_nll": float(total_nll[idx].detach().cpu()),
            "mean_nll": float(mean_nll[idx].detach().cpu()),
            "token_count": int(token_count[idx].detach().cpu()),
        }
        for idx in range(len(texts))
    ]


@torch.no_grad()
def score_choice_likelihood_batch(
    model,
    tokenizer,
    prompts: List[str],
    choices_batch: List[List[str]],
    choice_labels: List[str],
) -> List[str]:
    flat_texts = []
    flat_meta = []
    for row_idx, (prompt, choices) in enumerate(zip(prompts, choices_batch, strict=True)):
        rendered_prompt = apply_chat_prompt(tokenizer, prompt)
        prompt_ids = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=2048,
        )["input_ids"]
        for choice_idx, choice in enumerate(choices):
            continuation = " " + choice
            text = rendered_prompt + continuation
            flat_texts.append(text)
            flat_meta.append((row_idx, choice_idx, len(prompt_ids)))

    inputs = tokenizer(
        flat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = inputs["input_ids"][:, 1:].contiguous()
    mask = inputs["attention_mask"][:, 1:].contiguous()

    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view(labels.shape)

    scores_by_row: list[list[tuple[int, float]]] = [[] for _ in prompts]
    for flat_idx, (row_idx, choice_idx, prompt_len) in enumerate(flat_meta):
        pad_len = int(inputs["input_ids"].shape[1] - inputs["attention_mask"][flat_idx].sum().detach().cpu())
        start = max(pad_len + prompt_len - 1, 0)
        continuation_mask = mask[flat_idx].clone()
        continuation_mask[:start] = 0
        token_count = continuation_mask.sum().clamp_min(1)
        mean_nll = (losses[flat_idx] * continuation_mask).sum() / token_count
        scores_by_row[row_idx].append((choice_idx, -float(mean_nll.detach().cpu())))

    predictions = []
    for scores in scores_by_row:
        best_choice_idx, _ = max(scores, key=lambda item: item[1])
        predictions.append(choice_labels[best_choice_idx])

    return predictions
