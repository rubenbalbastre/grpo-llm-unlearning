#!/usr/bin/env python3
"""Compute the 95% energy rank of final LoRA adapters."""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm import tqdm


RUN_RE = re.compile(
    r"^unlearning-(?P<variant>original|r2-warmed)-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-(?P<target>.+)-"
    r"(?P<reward>r0|r1|r2|r4)(?:-reasoning-[a-z0-9-]+)?$"
)
MODULE_RE = re.compile(
    r"\.layers\.(?P<transformer_layer>\d+)\."
    r"(?P<module_path>.+)\.lora_A\."
)
MODEL_SIZES = {"0-5b": "0.5B", "1-5b": "1.5B", "3b": "3B", "7b": "7B"}
ENERGY_THRESHOLD = 0.95
NO_LEARNING_MARKER = "low_reward_stop.json"


def did_not_learn(adapter_path: Path) -> bool:
    return (adapter_path.parents[1] / NO_LEARNING_MARKER).is_file()


def effective_rank_95(a: torch.Tensor, b: torch.Tensor) -> int:
    """Return the number of singular directions containing 95% of ||BA||_F^2."""
    a = a.float()
    b = b.float()
    _, r_b = torch.linalg.qr(b, mode="reduced")
    _, r_a = torch.linalg.qr(a.T, mode="reduced")
    singular_values = torch.linalg.svdvals(r_b @ r_a.T)
    energy = singular_values.square()
    if energy.sum() == 0:
        return 0
    cumulative_energy = energy.cumsum(0) / energy.sum()
    return int(torch.searchsorted(cumulative_energy, ENERGY_THRESHOLD).item() + 1)


def adapter_details(adapter_path: Path) -> list[dict[str, object]]:
    state = load_file(adapter_path, device="cpu")
    details = []
    for key, a in state.items():
        if ".lora_A." not in key:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in state:
            raise ValueError(f"Missing {b_key} in {adapter_path}")
        match = MODULE_RE.search(key)
        if match is None:
            raise ValueError(f"Cannot identify transformer layer from {key}")
        effective_rank = effective_rank_95(a, state[b_key])
        module_path = match.group("module_path")
        details.append(
            {
                "transformer_layer": int(match.group("transformer_layer")),
                "module_path": module_path,
                "module": module_path.rsplit(".", 1)[-1],
                "configured_rank": a.shape[0],
                "effective_rank_95": effective_rank,
                "normalized_effective_rank_95": effective_rank / a.shape[0],
            }
        )
    if not details:
        raise ValueError(f"No LoRA matrices found in {adapter_path}")
    return details


def run_rows(
    adapter_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    run_dir = adapter_path.parents[1]
    match = RUN_RE.fullmatch(run_dir.name)
    if match is None or did_not_learn(adapter_path):
        return None

    metadata = {
        "run_name": run_dir.name,
        "target": match.group("target"),
        "model_size": MODEL_SIZES[match.group("size")],
        "reward_function": match.group("reward"),
        "training_initialization": (
            "cold" if match.group("variant") == "original" else "warm"
        ),
    }
    details = [{**metadata, **detail} for detail in adapter_details(adapter_path)]
    run = {
        **metadata,
        "matrix_count": len(details),
        "effective_rank_95_mean": sum(
            detail["effective_rank_95"] for detail in details
        )
        / len(details),
        "normalized_effective_rank_95_mean": sum(
            detail["normalized_effective_rank_95"] for detail in details
        )
        / len(details),
    }
    return run, details


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    groups = ["model_size", "reward_function", "training_initialization"]
    per_target = (
        runs.groupby([*groups, "target"], as_index=False)[
            ["effective_rank_95_mean", "normalized_effective_rank_95_mean"]
        ]
        .mean()
    )
    return (
        per_target.groupby(groups, as_index=False)
        .agg(
            target_count=("target", "nunique"),
            effective_rank_95_mean=("effective_rank_95_mean", "mean"),
            normalized_effective_rank_95_mean=(
                "normalized_effective_rank_95_mean",
                "mean",
            ),
        )
        .sort_values(groups)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Maximum number of learning runs to analyze (default: 10)",
    )
    parser.add_argument(
        "--module-output",
        type=Path,
        default=Path("outputs/tables/lora_effective_rank_modules.csv"),
    )
    parser.add_argument(
        "--run-output",
        type=Path,
        default=Path("outputs/tables/lora_effective_rank_runs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/lora_effective_rank.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_paths = sorted(
        args.outputs_root.glob("unlearning-*/final_model/adapter_model.safetensors")
    )
    no_learning_paths = [path for path in adapter_paths if did_not_learn(path)]
    adapter_paths = [path for path in adapter_paths if not did_not_learn(path)]
    print(
        f"Skipping {len(no_learning_paths)} run(s) marked with "
        f"{NO_LEARNING_MARKER}"
    )
    adapter_paths = adapter_paths[: args.n]
    rows = []
    module_rows = []
    for path in tqdm(adapter_paths, desc="LoRA adapters", unit="run"):
        result = run_rows(path)
        if result is not None:
            row, details = result
            rows.append(row)
            module_rows.extend(details)
    if not rows:
        raise SystemExit(f"No final LoRA adapters found under {args.outputs_root}")

    runs = pd.DataFrame(rows)
    modules = pd.DataFrame(module_rows)
    summary = aggregate_runs(runs)
    args.module_output.parent.mkdir(parents=True, exist_ok=True)
    args.run_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    modules.to_csv(args.module_output, index=False)
    runs.to_csv(args.run_output, index=False)
    summary.to_csv(args.output, index=False)
    print(f"Wrote {len(modules)} module rows to {args.module_output}")
    print(f"Wrote {len(runs)} run rows to {args.run_output}")
    print(f"Wrote {len(summary)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
