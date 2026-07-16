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
from eval.hold_out_styles.analysis_utils import (  # noqa: E402
    add_llm_judge_metrics,
    aggregate_metric_summary,
)
from src.data_preprocessing.load_dataset import load_standard_dataset


def load_prompt_dataframe(args: argparse.Namespace) -> pd.DataFrame:

    _, dataset = load_standard_dataset(forget_concept=args.concept, mode="grpo")
    
    df = dataset.to_pandas()

    if args.max_examples is not None:
        if args.shuffle:
            df = df.sample(frac=1.0, random_state=args.seed)
        df = df.head(args.max_examples)
    df = df.reset_index(drop=True)
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


def write_outputs(output_dir: str, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir / "metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)


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
    output_dir = f"outputs/hold_out_styles/{args.concept.replace(" ", "-").lower()}-{args.model_name_or_path.replace("/","-")}"
    write_outputs(output_dir, df, summary)
    print(f"Wrote metrics and summary to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
