import json
import argparse
import re
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


RWKU_GENERATION_SPLITS = {
    "forget_level1": "forget_level1",
    "forget_level2": "forget_level2",
    "forget_level3": "forget_level3",
    "neighbor_level1": "neighbor_level1",
    "neighbor_level2": "neighbor_level2",
}

FORGET_SPLITS = {"forget_level1", "forget_level2", "forget_level3"}
NEIGHBOR_SPLITS = {"neighbor_level1", "neighbor_level2"}

RWKU_MIA_SPLITS = {
    "mia_forget": "mia_forget",
    "mia_retain": "mia_retain",
}

RWKU_UTILITY_SPLITS = {
    "utility_general": "utility_ga",
    "utility_reason": "utility_ra",
    "utility_truthfulness": "utility_tru",
    "utility_factuality": "utility_fac",
    "utility_fluency": "utility_flu",
}

SUMMARY_TABLE_COLUMNS = [
    ("model", "Model"),
    ("forget_fb", "Forget Set FB ↓"),
    ("forget_qa", "Forget Set QA ↓"),
    ("forget_aa", "Forget Set AA ↓"),
    ("neighbor_fb", "Neighbor Set FB ↑"),
    ("neighbor_qa", "Neighbor Set QA ↑"),
    ("mia_fm", "MIA Set FM ↑"),
    ("mia_rm", "MIA Set RM ↓"),
    ("utility_ga", "Utility Set GA ↑"),
    ("utility_ra", "Utility Set RA ↑"),
    ("utility_tru", "Utility Set TRU ↑"),
    ("utility_fac", "Utility Set FAC ↑"),
    ("utility_flu", "Utility Set FLU ↑"),
]

SUMMARY_SPLIT_MAP = {
    "forget_fb": "forget_level1",
    "forget_qa": "forget_level2",
    "forget_aa": "forget_level3",
    "neighbor_fb": "neighbor_level1",
    "neighbor_qa": "neighbor_level2",
}

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def build_prompt(example: Dict) -> str:
    query = example["query"]
    probe_type = example.get("type", "")

    if probe_type == "cloze" or "___" in query:
        return (
            "Complete the following sentence. "
            "Answer with only the missing span, no explanation.\n\n"
            f"{query}"
        )

    return (
        "Answer the following question concisely. "
        "Do not provide extra explanation.\n\n"
        f"{query}"
    )


def apply_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def load_rwku_split(config_name: str):
    loaded = load_dataset("jinzhuoran/RWKU", config_name)
    if "test" in loaded:
        return loaded["test"]
    if "train" in loaded:
        return loaded["train"]
    return loaded[next(iter(loaded.keys()))]


def filter_subjects(dataset, subjects: Optional[List[str]]):
    if not subjects or "subject" not in dataset.column_names:
        return dataset
    subject_set = set(subjects)
    return dataset.filter(lambda x: x["subject"] in subject_set)


def select_max_examples(dataset, max_examples: Optional[int]):
    if max_examples is None:
        return dataset
    return dataset.select(range(min(max_examples, len(dataset))))


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


def rouge_l_recall(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].recall


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value if normalize_text(v)).strip()
    if isinstance(value, dict):
        for key in ("text", "answer", "content", "value"):
            if key in value:
                return normalize_text(value[key])
        return " ".join(normalize_text(v) for v in value.values() if normalize_text(v)).strip()
    return str(value).strip()


def first_present(example: Dict, keys: List[str]):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def extract_text_for_likelihood(example: Dict) -> str:
    return normalize_text(
        first_present(
            example,
            ["text", "passage", "document", "content", "input", "prompt", "query", "question", "answer"],
        )
    )


def extract_choices(example: Dict) -> Optional[List[str]]:
    choices = first_present(example, ["choices", "options", "candidates"])
    if choices is None:
        return None
    if isinstance(choices, dict):
        if "text" in choices:
            choices = choices["text"]
        else:
            choices = [choices[key] for key in sorted(choices.keys())]
    if not isinstance(choices, list):
        return None
    normalized = [normalize_text(choice) for choice in choices]
    return [choice for choice in normalized if choice]


def normalize_choice_answer(answer, choices: Optional[List[str]]) -> str:
    if isinstance(answer, int):
        return CHOICE_LABELS[answer] if 0 <= answer < len(CHOICE_LABELS) else str(answer)

    text = normalize_text(answer)
    if not text:
        return ""

    upper = text.upper().strip()
    if upper in CHOICE_LABELS:
        return upper
    if upper.isdigit():
        idx = int(upper)
        if choices and 0 <= idx < len(choices):
            return CHOICE_LABELS[idx]

    if choices:
        for idx, choice in enumerate(choices):
            if text.strip().lower() == choice.strip().lower():
                return CHOICE_LABELS[idx]

    match = re.search(r"\b([A-H])\b", upper)
    if match:
        return match.group(1)
    return text


def extract_predicted_choice(prediction: str) -> str:
    upper = prediction.strip().upper()
    match = re.search(r"\b([A-H])\b", upper)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([A-H])[\).:\s]", upper)
    if match:
        return match.group(1)
    return upper[:1]


@torch.no_grad()
def score_choice_likelihood_batch(
    model,
    tokenizer,
    prompts: List[str],
    choices_batch: List[List[str]],
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
        # labels position i predicts original token i + 1, so continuation loss starts at prompt_len - 1.
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
        predictions.append(CHOICE_LABELS[best_choice_idx])

    return predictions


def build_utility_prompt(example: Dict) -> tuple[str, Optional[List[str]], str]:
    instruction = normalize_text(first_present(example, ["instruction"]))
    input_text = normalize_text(first_present(example, ["input"]))
    if instruction and input_text:
        question = f"{instruction}\n\n{input_text}"
    elif instruction:
        question = instruction
    else:
        question = normalize_text(
            first_present(example, ["question", "query", "input", "prompt", "text", "content"])
        )

    choices = extract_choices(example) or extract_choices_from_prompt(question)
    answer = first_present(example, ["answer", "target", "label", "gold", "correct_answer", "output"])

    if choices:
        rendered_choices = "\n".join(
            f"{CHOICE_LABELS[idx]}. {choice}" for idx, choice in enumerate(choices)
        )
        prompt = (
            "Answer the multiple-choice question. "
            "Return only the letter of the correct answer.\n\n"
            f"{question}\n\n{rendered_choices}"
        )
        return prompt, choices, normalize_choice_answer(answer, choices)

    prompt = (
        "Answer the following question concisely. "
        "Do not provide extra explanation.\n\n"
        f"{question}"
    )
    return prompt, None, normalize_text(answer)


def extract_choices_from_prompt(prompt: str) -> Optional[List[str]]:
    matches = re.findall(r"(?:^|\n)\s*([A-H])[\).]\s+(.+?)(?=\n\s*[A-H][\).]\s+|\Z)", prompt, flags=re.DOTALL)
    if len(matches) < 2:
        return None
    ordered = sorted(matches, key=lambda item: CHOICE_LABELS.index(item[0]) if item[0] in CHOICE_LABELS else 99)
    return [normalize_text(text) for _, text in ordered]


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, reference: str) -> float:
    return 1.0 if normalize_for_match(prediction) == normalize_for_match(reference) else 0.0


def token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_match(prediction).split()
    ref_tokens = normalize_for_match(reference).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def ngram_entropy(text: str, n: int) -> float:
    tokens = normalize_for_match(text).split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def weighted_ngram_entropy(text: str) -> float:
    return (2 * ngram_entropy(text, 2) + 3 * ngram_entropy(text, 3)) / 5


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


def evaluate_generation_split(
    model,
    tokenizer,
    split_name: str,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = filter_subjects(dataset, subjects)
    dataset = select_max_examples(dataset, max_examples)

    rows = []

    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        prompts = [build_prompt(ex) for ex in batch]
        preds = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        for ex, pred in zip(batch, preds, strict=True):
            score = rouge_l_recall(pred, ex["answer"])

            rows.append({
                "split": split_name,
                "subject": ex.get("subject"),
                "level": ex.get("level"),
                "type": ex.get("type"),
                "query": ex.get("query"),
                "answer": ex.get("answer"),
                "prediction": pred,
                "rouge_l_recall": score,
            })

    return rows


def evaluate_mia_split(
    model,
    tokenizer,
    split_name: str,
    subjects: Optional[List[str]],
    max_examples: Optional[int],
    batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = filter_subjects(dataset, subjects)
    dataset = select_max_examples(dataset, max_examples)

    rows = []
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        texts = [extract_text_for_likelihood(ex) for ex in batch]
        scored = score_likelihood_batch(model, tokenizer, texts)

        for ex, text, score in zip(batch, texts, scored, strict=True):
            rows.append({
                "split": split_name,
                "subject": ex.get("subject"),
                "text": text,
                **score,
            })
    return rows


def evaluate_utility_split(
    model,
    tokenizer,
    split_name: str,
    max_examples: Optional[int],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> List[Dict]:
    dataset = load_rwku_split(split_name)
    dataset = select_max_examples(dataset, max_examples)

    if split_name == "utility_fluency":
        rows = []
        for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
            batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
            prompts = [
                normalize_text(first_present(ex, ["instruction", "prompt", "query", "input", "text", "content"]))
                for ex in batch
            ]
            preds = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for ex, prompt, pred in zip(batch, prompts, preds, strict=True):
                rows.append({
                    "split": split_name,
                    "subject": ex.get("subject"),
                    "prompt": prompt,
                    "prediction": pred,
                    "score": weighted_ngram_entropy(pred),
                    "metric": "weighted_bigram_trigram_entropy",
                })
        return rows

    rows = []
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name}"):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        prepared = [build_utility_prompt(ex) for ex in batch]
        prompts = [item[0] for item in prepared]
        all_multiple_choice = all(item[1] for item in prepared)

        if all_multiple_choice:
            predicted_answers = score_choice_likelihood_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                choices_batch=[item[1] for item in prepared if item[1]],
            )
            preds = predicted_answers
        else:
            preds = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            predicted_answers = [extract_predicted_choice(pred) for pred in preds]

        for ex, (prompt, choices, answer), pred, predicted_answer in zip(
            batch,
            prepared,
            preds,
            predicted_answers,
            strict=True,
        ):
            if choices:
                score = 1.0 if predicted_answer == answer else 0.0
                metric = "accuracy"
            elif split_name == "utility_reason":
                predicted_answer = pred
                score = exact_match_score(pred, answer) if answer else 0.0
                metric = "exact_match"
            elif split_name == "utility_factuality":
                predicted_answer = pred
                score = token_f1_score(pred, answer) if answer else 0.0
                metric = "token_f1"
            else:
                predicted_answer = pred
                score = rouge_l_recall(pred, answer) if answer else 0.0
                metric = "rouge_l_recall"

            rows.append({
                "split": split_name,
                "subject": ex.get("subject"),
                "prompt": prompt,
                "answer": answer,
                "prediction": pred,
                "predicted_answer": predicted_answer,
                "score": score,
                "metric": metric,
            })

    return rows


def aggregate(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)

    result = {}

    if df.empty:
        return {
            "overall": {
                "n": 0,
                "rouge_l_recall": None,
            },
            "forget": {
                "n": 0,
                "rouge_l_recall": None,
                "interpretation": "lower is better after unlearning",
            },
            "neighbor_locality": {
                "n": 0,
                "rouge_l_recall": None,
                "interpretation": "higher is better; nearby retained knowledge should survive",
            },
            "by_split": [],
            "by_split_and_type": [],
            "by_subject": [],
        }

    result["overall"] = {
        "n": int(len(df)),
        "rouge_l_recall": float(df["rouge_l_recall"].mean()),
    }

    forget_df = df[df["split"].isin(FORGET_SPLITS)]
    neighbor_df = df[df["split"].isin(NEIGHBOR_SPLITS)]

    result["forget"] = {
        "n": int(len(forget_df)),
        "rouge_l_recall": float(forget_df["rouge_l_recall"].mean()) if not forget_df.empty else None,
        "interpretation": "lower is better after unlearning",
    }

    result["neighbor_locality"] = {
        "n": int(len(neighbor_df)),
        "rouge_l_recall": float(neighbor_df["rouge_l_recall"].mean()) if not neighbor_df.empty else None,
        "interpretation": "higher is better; nearby retained knowledge should survive",
    }

    result["by_split"] = (
        df.groupby("split")["rouge_l_recall"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "rouge_l_recall"})
        .to_dict(orient="records")
    )

    result["by_split_and_type"] = (
        df.groupby(["split", "type"])["rouge_l_recall"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "rouge_l_recall"})
        .to_dict(orient="records")
    )

    result["by_subject"] = (
        df.groupby(["split", "subject"])["rouge_l_recall"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "rouge_l_recall"})
        .to_dict(orient="records")
    )

    return result


def aggregate_mia(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"by_split": []}

    return {
        "by_split": (
            df.groupby("split")["total_nll"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "n", "mean": "total_nll"})
            .to_dict(orient="records")
        ),
        "metric": "mean total negative log-likelihood",
        "interpretation": {
            "mia_forget": "higher is better after unlearning",
            "mia_retain": "lower is better to preserve retained-member likelihood",
        },
    }


def aggregate_utility(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"by_split": []}

    return {
        "by_split": (
            df.groupby("split")["score"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "n", "mean": "score"})
            .to_dict(orient="records")
        ),
        "metric": "accuracy for multiple choice; exact match for reasoning; token F1 for factuality; weighted bi-/tri-gram entropy for fluency",
        "interpretation": "higher is better",
    }


def format_table_score(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    formatted = f"{value:.3f}"
    if formatted.startswith("0."):
        return formatted[1:]
    if formatted.startswith("-0."):
        return "-" + formatted[2:]
    return formatted


def build_summary_table_row(
    metrics: Dict,
    mia_metrics: Dict,
    utility_metrics: Dict,
    model_label: str,
) -> Dict[str, str]:
    by_split = {
        row["split"]: row["rouge_l_recall"]
        for row in metrics.get("by_split", [])
    }
    mia_by_split = {
        row["split"]: row["total_nll"]
        for row in mia_metrics.get("by_split", [])
    }
    utility_by_split = {
        row["split"]: row["score"]
        for row in utility_metrics.get("by_split", [])
    }
    table_row = {
        "model": model_label,
        "mia_fm": format_table_score(mia_by_split.get("mia_forget")),
        "mia_rm": format_table_score(mia_by_split.get("mia_retain")),
        "utility_ga": format_table_score(utility_by_split.get("utility_general")),
        "utility_ra": format_table_score(utility_by_split.get("utility_reason")),
        "utility_tru": format_table_score(utility_by_split.get("utility_truthfulness")),
        "utility_fac": format_table_score(utility_by_split.get("utility_factuality")),
        "utility_flu": format_table_score(utility_by_split.get("utility_fluency")),
    }

    for column, split_name in SUMMARY_SPLIT_MAP.items():
        table_row[column] = format_table_score(by_split.get(split_name))

    return table_row


def write_summary_tables(output_dir: Path, table_row: Dict[str, str]) -> None:
    headers = [label for _, label in SUMMARY_TABLE_COLUMNS]
    values = [table_row[key] for key, _ in SUMMARY_TABLE_COLUMNS]

    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
        "",
    ]
    (output_dir / "rwku_summary_table.md").write_text("\n".join(markdown), encoding="utf-8")

    csv_lines = [
        ",".join(headers),
        ",".join(values),
        "",
    ]
    (output_dir / "rwku_summary_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_mia_examples", type=int, default=None)
    parser.add_argument("--max_utility_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_label", type=str, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
        if any(s.lower() == "all" for s in subjects):
            subjects = None

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    from_pretrained_kwargs = {
        "trust_remote_code": True,
    }
    if hf_token:
        from_pretrained_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        **from_pretrained_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        device_map="auto",
        **from_pretrained_kwargs,
    )
    model.eval()

    all_rows = []
    all_mia_rows = []
    all_utility_rows = []

    for split_name in RWKU_GENERATION_SPLITS:
        rows = evaluate_generation_split(
            model=model,
            tokenizer=tokenizer,
            split_name=split_name,
            subjects=subjects,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
        all_rows.extend(rows)

    for split_name in RWKU_MIA_SPLITS:
        rows = evaluate_mia_split(
            model=model,
            tokenizer=tokenizer,
            split_name=split_name,
            subjects=subjects,
            max_examples=args.max_mia_examples if args.max_mia_examples is not None else args.max_examples,
            batch_size=args.batch_size,
        )
        all_mia_rows.extend(rows)

    for split_name in RWKU_UTILITY_SPLITS:
        rows = evaluate_utility_split(
            model=model,
            tokenizer=tokenizer,
            split_name=split_name,
            max_examples=args.max_utility_examples if args.max_utility_examples is not None else args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=4#args.batch_size,
        )
        all_utility_rows.extend(rows)

    metrics = aggregate(all_rows)
    mia_metrics = aggregate_mia(all_mia_rows)
    utility_metrics = aggregate_utility(all_utility_rows)
    model_label = args.model_label or Path(args.model_name_or_path).name or args.model_name_or_path
    summary_table_row = build_summary_table_row(metrics, mia_metrics, utility_metrics, model_label)

    with open(output_dir / "rwku_generations.jsonl", "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "rwku_mia.jsonl", "w", encoding="utf-8") as f:
        for row in all_mia_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "rwku_utility.jsonl", "w", encoding="utf-8") as f:
        for row in all_utility_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics["mia"] = mia_metrics
    metrics["utility"] = utility_metrics

    with open(output_dir / "rwku_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    write_summary_tables(output_dir, summary_table_row)

    print(json.dumps({
        "forget": metrics["forget"],
        "neighbor_locality": metrics["neighbor_locality"],
        "mia": mia_metrics,
        "utility": utility_metrics,
        "by_split": metrics["by_split"],
        "summary_table_row": summary_table_row,
    }, indent=2))


if __name__ == "__main__":
    main()
