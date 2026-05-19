import argparse
import json
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_eval import evaluate_forget_set, evaluate_neighbor_set
from mia_eval import evaluate_mia
from results import (
    aggregate_generation,
    aggregate_mia,
    aggregate_utility,
    build_summary_table_row,
    write_summary_tables,
)
from utility_eval import evaluate_utility_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--run_forget_set", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_neighbor_set", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_mia_set", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_utility_set", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_mia_examples", type=int, default=None)
    parser.add_argument("--max_utility_examples", type=int, default=None)
    parser.add_argument("--compute_mia_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute_mia_zlib", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute_mia_min_k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute_mia_min_k_plus_plus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--utility_batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_label", type=str, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")
    if args.utility_batch_size <= 0:
        raise ValueError("--utility_batch_size must be >= 1")
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

    forget_rows = []
    if args.run_forget_set:
        forget_rows = evaluate_forget_set(
            model=model,
            tokenizer=tokenizer,
            subjects=subjects,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

    neighbor_rows = []
    if args.run_neighbor_set:
        neighbor_rows = evaluate_neighbor_set(
            model=model,
            tokenizer=tokenizer,
            subjects=subjects,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
    all_rows = forget_rows + neighbor_rows

    all_mia_rows = []
    if args.run_mia_set:
        all_mia_rows = evaluate_mia(
            model=model,
            tokenizer=tokenizer,
            subjects=subjects,
            max_examples=args.max_mia_examples if args.max_mia_examples is not None else args.max_examples,
            batch_size=args.batch_size,
            loss=args.compute_mia_loss,
            zlib=args.compute_mia_zlib,
            min_k=args.compute_mia_min_k,
            min_k_plus_plus=args.compute_mia_min_k_plus_plus,
        )

    all_utility_rows = []
    if args.run_utility_set:
        all_utility_rows = evaluate_utility_set(
            model=model,
            tokenizer=tokenizer,
            max_examples=args.max_utility_examples if args.max_utility_examples is not None else args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.utility_batch_size,
        )

    metrics = aggregate_generation(all_rows)
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
