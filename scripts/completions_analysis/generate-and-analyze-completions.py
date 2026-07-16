#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.rwku.scoring import generate_batch  # noqa: E402
from scripts.completions_analysis.analysis_utils import (  # noqa: E402
    add_llm_judge_metrics,
    aggregate_metric_summary,
)


def load_prompt_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Saved Hugging Face dataset not found: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
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
    model.to("cuda" if torch.cuda.is_available() else "cpu")
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
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(args.output_dir / "metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)


@hydra.main(version_base=None, config_path="../../config", config_name="hold_out_eval")
def main(args) -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

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
