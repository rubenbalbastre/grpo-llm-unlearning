from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm.auto import tqdm

from src.data_generator.data_proposer import DataProposer


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", prompt.strip().lower())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fixed offline synthetic refusal dataset using the same "
            "DataProposer used by adaptive training, but without reward history."
        )
    )
    parser.add_argument("--subject", default="Stephen King")
    parser.add_argument(
        "--topic",
        default=None,
        help="Topic string passed to DataProposer. Defaults to the training-style forget concept topic.",
    )
    parser.add_argument("--num-prompts", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--model-name", default="gpt-5.4-nano")
    parser.add_argument(
        "--mode",
        choices=["natural", "expanded"],
        default="expanded",
        help="Use the same mode values supported by DataProposer.",
    )
    parser.add_argument(
        "--output",
        default="outputs/offline-synthetic/prompts_stephen_king.jsonl",
        help="JSONL path for the generated fixed dataset.",
    )
    parser.add_argument(
        "--events-log",
        default="outputs/offline-synthetic/events.jsonl",
        help="Log path passed to DataProposer for shortfall events.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")

    topic = args.topic or f"Forget concept: '{args.subject}.'"
    proposer = DataProposer(
        topic=topic,
        log_path=Path(args.events_log),
        model_name=args.model_name,
        mode=args.mode,
    )

    prompts: list[str] = []
    seen: set[str] = set()
    attempts = 0
    with tqdm(total=args.num_prompts, unit="prompt") as progress:
        while len(prompts) < args.num_prompts and attempts < args.max_attempts:
            attempts += 1
            remaining = args.num_prompts - len(prompts)
            requested = min(args.batch_size, remaining)
            candidates = proposer.generate_prompts(
                rollout_group_context={
                    "low_variance_low_mean_reward": [],
                    "low_variance_high_mean_reward": [],
                    "high_variance_reward": [],
                },
                num_prompts=requested,
            )
            for candidate in candidates:
                prompt = str(candidate["prompt"]).strip()
                key = normalize_prompt(prompt)
                if prompt and key not in seen:
                    seen.add(key)
                    prompts.append(prompt)
                    progress.update(1)
                if len(prompts) == args.num_prompts:
                    break

    if len(prompts) < args.num_prompts:
        raise RuntimeError(
            f"Generated {len(prompts)} unique prompts after {attempts} attempts; "
            f"expected {args.num_prompts}."
        )

    rows = [{"prompt": prompt} for prompt in prompts]
    output_path = Path(args.output)
    write_jsonl(output_path, rows)

    print(f"Wrote {len(rows)} prompts to {output_path}")
    print(f"Subject: {args.subject}")
    print(f"Topic: {topic}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")


if __name__ == "__main__":
    main()
