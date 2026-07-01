#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.rwku.scoring import generate_batch  # noqa: E402
from scripts.completions_analysis.analysis_utils import (  # noqa: E402
    add_llm_judge_metrics,
    add_local_reward_metrics,
    aggregate_metric_summary,
)


def load_prompt_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    dataset_cache_dir = args.output_dir / "hf-datasets-cache"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {}
    if args.data_files is not None:
        dataset_kwargs["data_files"] = args.data_files
    if args.dataset_config is not None:
        dataset_kwargs["name"] = args.dataset_config

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        cache_dir=str(dataset_cache_dir),
        **dataset_kwargs,
    )
    df = dataset.to_pandas()

    if args.prompt_column not in df.columns:
        raise ValueError(
            f"Prompt column {args.prompt_column!r} not found. Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["source_row_index"] = df.index
    df = df[df[args.prompt_column].map(lambda value: isinstance(value, str) and value.strip() != "")]
    if len(df) == 0:
        raise ValueError(f"No valid prompts remain after filtering empty {args.prompt_column!r} values.")

    df[args.prompt_column] = df[args.prompt_column].map(str.strip)
    if args.max_examples is not None:
        if args.shuffle:
            df = df.sample(frac=1.0, random_state=args.seed)
        df = df.head(args.max_examples)

    df = df.reset_index(drop=True)
    if args.prompt_column != "prompt":
        df = df.rename(columns={args.prompt_column: "prompt"})
    return df


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.eval()
    return model, tokenizer


def generate_completions(
    df: pd.DataFrame,
    *,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> pd.DataFrame:
    completions: list[str] = []
    prompts = df["prompt"].astype(str).tolist()

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating completions"):
        batch_prompts = prompts[start:start + batch_size]
        completions.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        )

    df = df.copy()
    df["completion"] = completions
    return df


def aggregate_results(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    return aggregate_metric_summary(
        df,
        metadata={
            "forget_concept": args.concept,
            "model_name_or_path": args.model_name_or_path,
        },
    )


def write_outputs(args: argparse.Namespace, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate completions from a model checkpoint for a prompt dataset, "
            "then evaluate them with local leakage rewards and the async LLM judge."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--concept", required=True, help="Target forget concept used by the judge/rewards.")

    parser.add_argument("--dataset-name", required=True, help="Hugging Face dataset name or loader, e.g. json/csv/parquet.")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--data-files", default=None, help="Optional local/remote data_files passed to load_dataset.")
    parser.add_argument("--dataset-split", default="train", help="Dataset split for Hugging Face datasets.")
    parser.add_argument("--prompt-column", default="prompt", help="Column containing prompts.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--judge-model", default="gpt-5.4-nano")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/completion_analysis/generated-completions"))

    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be >= 1.")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be >= 1 when provided.")
    if args.judge_concurrency <= 0:
        raise ValueError("--judge-concurrency must be >= 1.")
    return args


def main() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    args = parse_args()

    df = load_prompt_dataframe(args)
    print(f"Loaded {len(df)} prompt(s).", flush=True)

    model, tokenizer = load_model_and_tokenizer(args)
    df = generate_completions(
        df,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    df = add_local_reward_metrics(df, args.concept)
    df = add_llm_judge_metrics(
        df,
        concept=args.concept,
        judge_model=args.judge_model,
        max_concurrent_requests=args.judge_concurrency,
    )
    summary = aggregate_results(df, args)
    write_outputs(args, df, summary)
    print(f"Wrote metrics and summary to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
