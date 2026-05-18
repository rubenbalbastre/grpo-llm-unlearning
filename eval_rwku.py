import json
import argparse
import os
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
    dataset = load_dataset("jinzhuoran/RWKU", split_name)["test"]

    if subjects:
        subject_set = set(subjects)
        dataset = dataset.filter(lambda x: x["subject"] in subject_set)

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

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


def format_table_score(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    formatted = f"{value:.3f}"
    if formatted.startswith("0."):
        return formatted[1:]
    if formatted.startswith("-0."):
        return "-" + formatted[2:]
    return formatted


def build_summary_table_row(metrics: Dict, model_label: str) -> Dict[str, str]:
    by_split = {
        row["split"]: row["rouge_l_recall"]
        for row in metrics.get("by_split", [])
    }
    table_row = {
        "model": model_label,
        "mia_fm": "NA",
        "mia_rm": "NA",
        "utility_ga": "NA",
        "utility_ra": "NA",
        "utility_tru": "NA",
        "utility_fac": "NA",
        "utility_flu": "NA",
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

    metrics = aggregate(all_rows)
    model_label = args.model_label or Path(args.model_name_or_path).name or args.model_name_or_path
    summary_table_row = build_summary_table_row(metrics, model_label)

    with open(output_dir / "rwku_generations.jsonl", "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "rwku_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    write_summary_tables(output_dir, summary_table_row)

    print(json.dumps({
        "forget": metrics["forget"],
        "neighbor_locality": metrics["neighbor_locality"],
        "by_split": metrics["by_split"],
        "summary_table_row": summary_table_row,
    }, indent=2))


if __name__ == "__main__":
    main()
