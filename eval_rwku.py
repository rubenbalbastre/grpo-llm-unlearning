import json
import argparse
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
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    rendered_prompt = apply_chat_prompt(tokenizer, prompt)

    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
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

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


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
) -> List[Dict]:
    dataset = load_dataset("jinzhuoran/RWKU", split_name)["test"]

    if subjects:
        subject_set = set(subjects)
        dataset = dataset.filter(lambda x: x["subject"] in subject_set)

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    rows = []

    for ex in tqdm(dataset, desc=f"Evaluating {split_name}"):
        prompt = build_prompt(ex)
        pred = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

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

    result["overall"] = {
        "n": int(len(df)),
        "rouge_l_recall": float(df["rouge_l_recall"].mean()),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5-Instruct", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        device_map="auto",
        trust_remote_code=True,
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
        )
        all_rows.extend(rows)

    metrics = aggregate(all_rows)

    with open(output_dir / "rwku_generations.jsonl", "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "rwku_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics["by_split"], indent=2))


if __name__ == "__main__":
    main()