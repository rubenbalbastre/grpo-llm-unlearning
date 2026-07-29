from typing import Dict, List

import torch


def apply_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            "RWKU evaluation requires the model tokenizer's chat template."
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    assistant_prefix: str = "",
    prefix_space_if_needed: bool = True,
    stop_at_newline: bool = False,
) -> List[str]:
    rendered_prompts = [apply_chat_prompt(tokenizer, prompt) for prompt in prompts]
    if assistant_prefix:
        rendered_prompts = [
            rendered_prompt
            + (
                assistant_prefix
                if not prefix_space_if_needed or rendered_prompt[-1:] in {"\n", " "}
                else f" {assistant_prefix}"
            )
            for rendered_prompt in rendered_prompts
        ]

    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
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
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if stop_at_newline:
            completion = completion.split("\n", 1)[0]
        completions.append(completion.strip())
    return completions


def render_with_assistant_prefix(
    tokenizer,
    prompt: str,
    assistant_prefix: str,
    prefix_space_if_needed: bool = True,
) -> str:
    rendered = apply_chat_prompt(tokenizer, prompt)
    if assistant_prefix:
        rendered += (
            assistant_prefix
            if not prefix_space_if_needed or rendered[-1:] in {"\n", " "}
            else f" {assistant_prefix}"
        )
    return rendered


@torch.no_grad()
def score_next_token_candidates_batch(
    model,
    tokenizer,
    prompts: List[str],
    candidates: List[str],
    assistant_prefix: str = "",
    prefix_space_if_needed: bool = True,
) -> List[int]:
    """Return the highest-probability candidate token, as in official RWKU MMLU."""
    rendered = [
        render_with_assistant_prefix(
            tokenizer, prompt, assistant_prefix, prefix_space_if_needed
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    logits = model(**inputs).logits[:, -1, :]
    candidate_ids = [
        tokenizer.encode(f" {candidate}", add_special_tokens=False)[-1]
        for candidate in candidates
    ]
    return logits[:, candidate_ids].argmax(dim=-1).detach().cpu().tolist()


@torch.no_grad()
def score_completion_logprobs_batch(
    model,
    tokenizer,
    prompts: List[str],
    completions_batch: List[List[str]],
    assistant_prefix: str = "",
    forward_batch_size: int = 8,
) -> List[List[float]]:
    """Sum completion-token log probabilities, matching RWKU TruthfulQA."""
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be >= 1.")

    flat_examples = []
    for row_idx, (prompt, completions) in enumerate(
        zip(prompts, completions_batch, strict=True)
    ):
        rendered = render_with_assistant_prefix(tokenizer, prompt, assistant_prefix)
        for choice_idx, completion in enumerate(completions):
            text = (
                rendered if rendered[-1:] in {"\n", " "} else rendered + " "
            ) + completion
            flat_examples.append((row_idx, choice_idx, rendered, text))

    scores_by_row = [[0.0] * len(items) for items in completions_batch]
    for start in range(0, len(flat_examples), forward_batch_size):
        examples = flat_examples[start : start + forward_batch_size]
        texts = [item[3] for item in examples]
        inputs = tokenizer(texts, padding="longest", return_tensors="pt").to(model.device)
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)
        outputs = model(**inputs)

        for batch_idx, (row_idx, choice_idx, rendered, text) in enumerate(examples):
            prompt_ids = tokenizer(rendered, padding=False, return_tensors="pt")[
                "input_ids"
            ].squeeze(0)
            example_ids = tokenizer(text, padding=False, return_tensors="pt")[
                "input_ids"
            ].squeeze(0)
            completion_ids = example_ids[len(prompt_ids) :]
            if tokenizer.padding_side == "right":
                example_logits = outputs.logits[batch_idx, : len(example_ids), :]
            else:
                example_logits = outputs.logits[batch_idx, -len(example_ids) :, :]
            completion_logits = example_logits[
                len(prompt_ids) - 1 : len(example_ids) - 1, :
            ]
            log_probs = torch.log_softmax(completion_logits, dim=-1)[
                range(len(completion_ids)), completion_ids.to(model.device)
            ]
            scores_by_row[row_idx][choice_idx] = float(log_probs.sum().detach().cpu())
    return scores_by_row


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
    forward_batch_size: int = 8,
) -> List[str]:
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be >= 1.")

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

    scores_by_row: list[list[tuple[int, float]]] = [[] for _ in prompts]
    for start_idx in range(0, len(flat_texts), forward_batch_size):
        end_idx = min(start_idx + forward_batch_size, len(flat_texts))
        batch_texts = flat_texts[start_idx:end_idx]
        batch_meta = flat_meta[start_idx:end_idx]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        mask = inputs["attention_mask"][:, 1:]

        losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none",
        )

        for batch_idx, (row_idx, choice_idx, prompt_len) in enumerate(batch_meta):
            pad_len = int(
                inputs["input_ids"].shape[1]
                - inputs["attention_mask"][batch_idx].sum().detach().cpu()
            )
            start = max(pad_len + prompt_len - 1, 0)
            continuation_mask = mask[batch_idx].clone()
            continuation_mask[:start] = 0
            token_count = continuation_mask.sum().clamp_min(1)
            mean_nll = (losses[batch_idx] * continuation_mask).sum() / token_count
            scores_by_row[row_idx].append((choice_idx, -float(mean_nll.detach().cpu())))

    predictions = []
    for scores in scores_by_row:
        best_choice_idx, _ = max(scores, key=lambda item: item[1])
        predictions.append(choice_labels[best_choice_idx])

    return predictions
