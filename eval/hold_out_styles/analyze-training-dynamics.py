#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.hold_out_styles.analysis_utils import (  # noqa: E402
    add_llm_judge_metrics,
    llm_judge_metrics,
)


DEFAULT_NOTES = "lluis-vives-runs-1-1M"
RUN_NAME_PATTERN = re.compile(r"(?:^|-)original(?:-|$)|(?:^|-)r\d+-warmed(?:-|$)")


@dataclass(frozen=True)
class DownloadedRun:
    run_id: str
    run_name: str
    forget_concept: str
    reward_type: str
    model_name: str
    training_variant: str
    completion_tables: list[Path]
    available_table_count: int = 0


def nested_get(mapping: dict, path: list[str], default: str = "") -> str:
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return str(value) if value is not None else default


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def training_variant(run_name: str) -> str:
    lowered = run_name.lower()
    if re.search(r"(?:^|-)r\d+-warmed(?:-|$)", lowered):
        return "warmed"
    if re.search(r"(?:^|-)original(?:-|$)", lowered):
        return "original"
    return ""


def run_name_candidates(run) -> list[str]:
    candidates = [
        getattr(run, "name", ""),
        getattr(run, "display_name", ""),
        getattr(run, "id", ""),
    ]
    return [str(candidate) for candidate in candidates if candidate]


def selected_run_name(run) -> str:
    candidates = run_name_candidates(run)
    for candidate in candidates:
        if RUN_NAME_PATTERN.search(candidate.lower()):
            return candidate
    return candidates[0] if candidates else str(run.id)


def matches_run_name(run) -> bool:
    return any(
        RUN_NAME_PATTERN.search(candidate.lower())
        for candidate in run_name_candidates(run)
    )


def completion_table_sort_key(file_or_path) -> tuple[int, str]:
    name = getattr(file_or_path, "name", str(file_or_path))
    match = re.search(r"completions_(\d+)_", name)
    return (int(match.group(1)) if match else -1, name)


def select_recent_completion_tables(files_or_paths, max_tables: int):
    sorted_files = sorted(files_or_paths, key=completion_table_sort_key)
    if max_tables <= 0:
        return sorted_files
    return sorted_files[-max_tables:]


def load_table(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and {"columns", "data"} <= set(raw):
        return pd.DataFrame(raw["data"], columns=raw["columns"])
    return pd.read_json(path, orient="split")


def load_tables(table_paths: list[Path]) -> pd.DataFrame:
    frames = [load_table(table_path) for table_path in table_paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_missing_generation_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prompt_index" not in df.columns:
        df["prompt_index"] = df.groupby("step", sort=False)["prompt"].transform(
            lambda prompts: pd.factorize(prompts, sort=False)[0]
        )
    if "completion_index" not in df.columns:
        df["completion_index"] = df.groupby(
            ["step", "prompt_index"],
            sort=False,
        ).cumcount()
    return df


def count_unique_prompts(table_paths: list[Path]) -> int:
    df = load_tables(table_paths)
    return df["prompt"].nunique() if "prompt" in df.columns else 0


def local_wandb_file_path(run_dir: Path, wandb_file) -> Path:
    return run_dir / getattr(wandb_file, "name", "")


def download_or_reuse_file(run_dir: Path, wandb_file) -> Path:
    local_path = local_wandb_file_path(run_dir, wandb_file)
    if local_path.exists():
        print(f"Reusing {local_path}", flush=True)
        return local_path
    return Path(
        wandb_file.download(
            root=str(run_dir),
            replace=False,
            exist_ok=True,
        ).name
    )


def download_completion_tables(
    *,
    project: str,
    notes: str,
    download_dir: Path,
    max_tables_per_run: int,
    min_available_tables: int,
) -> list[DownloadedRun]:
    import wandb

    api = wandb.Api()
    filters = {
        "jobType": "training",
        "config.hydra.experiment.notes": notes,
    }
    downloaded_runs: list[DownloadedRun] = []

    for run in api.runs(project, filters=filters, per_page=100, lazy=True):
        if not matches_run_name(run):
            continue

        run_name = selected_run_name(run)
        run_config = dict(run.config or {})
        hydra_config = run_config.get("hydra", {})
        run_dir = download_dir / f"{safe_name(run_name)}-{run.id}"
        table_file_refs = list(
            run.files(pattern="%completions%.table.json", per_page=100)
        )
        if len(table_file_refs) < min_available_tables:
            print(
                f"Skipping {run_name}: only {len(table_file_refs)} completion table(s) "
                f"available; need at least {min_available_tables}.",
                flush=True,
            )
            continue

        selected_files = select_recent_completion_tables(
            table_file_refs,
            max_tables_per_run,
        )
        selected_names = [getattr(wandb_file, "name", "") for wandb_file in selected_files]
        print(
            f"Selected {len(selected_files)} of {len(table_file_refs)} table file(s) "
            f"for {run_name}: {selected_names}",
            flush=True,
        )
        table_paths = [
            download_or_reuse_file(run_dir, wandb_file)
            for wandb_file in selected_files
        ]
        if not table_paths:
            print(f"Skipping {run_name}: no completion tables found.", flush=True)
            continue

        unique_prompt_count = count_unique_prompts(table_paths)
        downloaded_runs.append(
            DownloadedRun(
                run_id=run.id,
                run_name=run_name,
                forget_concept=nested_get(
                    hydra_config, ["experiment", "forget_concept"]
                ),
                reward_type=nested_get(hydra_config, ["reward", "type"]),
                model_name=nested_get(hydra_config, ["model", "name"]),
                training_variant=training_variant(run_name),
                completion_tables=table_paths,
                available_table_count=len(table_file_refs),
            )
        )
        print(
            f"Downloaded {len(table_paths)} of {len(table_file_refs)} table(s) "
            f"from {run_name}; unique prompts: {unique_prompt_count}.",
            flush=True,
        )

    return downloaded_runs


def run_score_path(output_dir: Path, downloaded_run: DownloadedRun) -> Path:
    return (
        output_dir
        / "run_metrics"
        / f"{safe_name(downloaded_run.run_name)}-{downloaded_run.run_id}.csv"
    )


def inventory_path(output_dir: Path) -> Path:
    return output_dir / "download_inventory.csv"


def load_last_steps(downloaded_run: DownloadedRun, last_steps: int) -> pd.DataFrame:
    df = load_tables(downloaded_run.completion_tables)
    required_columns = {"step", "prompt", "completion"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{downloaded_run.run_name} is missing columns: {sorted(missing_columns)}"
        )
    df = add_missing_generation_indices(df)

    steps = sorted(df["step"].dropna().unique())[-last_steps:]
    df = df[df["step"].isin(steps)].copy()
    step_order = {step: index + 1 for index, step in enumerate(steps)}

    df["last_step_index"] = df["step"].map(step_order)
    df["step_from_end"] = len(steps) + 1 - df["last_step_index"]
    df["run_id"] = downloaded_run.run_id
    df["run_name"] = downloaded_run.run_name
    df["forget_concept"] = downloaded_run.forget_concept
    df["reward_type"] = downloaded_run.reward_type
    df["model_name"] = downloaded_run.model_name
    df["training_variant"] = downloaded_run.training_variant
    return df


def inventory_rows(
    downloaded_runs: list[DownloadedRun],
    *,
    output_dir: Path,
    last_steps: int,
) -> list[dict[str, object]]:
    rows = []
    for downloaded_run in downloaded_runs:
        completion_rows = load_last_steps(downloaded_run, last_steps)
        prompts_by_step = (
            completion_rows.groupby("step")["prompt"]
            .nunique()
            .sort_index()
            .to_dict()
        )
        rows.append(
            {
                "run_id": downloaded_run.run_id,
                "run_name": downloaded_run.run_name,
                "forget_concept": downloaded_run.forget_concept,
                "reward_type": downloaded_run.reward_type,
                "model_name": downloaded_run.model_name,
                "training_variant": downloaded_run.training_variant,
                "available_table_count": downloaded_run.available_table_count
                or len(downloaded_run.completion_tables),
                "selected_table_count": len(downloaded_run.completion_tables),
                "selected_step_count": completion_rows["step"].nunique(),
                "selected_steps": json.dumps(
                    sorted(completion_rows["step"].dropna().unique().tolist())
                ),
                "selected_unique_prompts_by_step": json.dumps(prompts_by_step),
                "selected_unique_prompt_count": completion_rows["prompt"].nunique(),
                "selected_completion_count": len(completion_rows),
                "selected_prompt_count": completion_rows[
                    ["step", "prompt_index"]
                ].drop_duplicates().shape[0],
                "score_path": str(run_score_path(output_dir, downloaded_run)),
                "completion_tables": json.dumps(
                    [str(path) for path in downloaded_run.completion_tables]
                ),
            }
        )
    return rows


def write_inventory(
    downloaded_runs: list[DownloadedRun],
    *,
    output_dir: Path,
    last_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = inventory_rows(
        downloaded_runs,
        output_dir=output_dir,
        last_steps=last_steps,
    )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(inventory_path(output_dir), index=False)

    if inventory.empty:
        counts = pd.DataFrame()
    else:
        counts = (
            inventory.groupby(
                ["forget_concept", "reward_type", "model_name", "training_variant"],
                as_index=False,
            )
            .agg(
                run_count=("run_id", "nunique"),
                selected_completion_count=("selected_completion_count", "sum"),
                selected_prompt_count=("selected_prompt_count", "sum"),
                selected_unique_prompt_count=("selected_unique_prompt_count", "sum"),
            )
        )
    counts.to_csv(output_dir / "download_inventory_by_author_reward.csv", index=False)


def read_inventory(
    output_dir: Path,
    max_tables_per_run: int,
    min_available_tables: int,
) -> list[DownloadedRun]:
    path = inventory_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run with --download-only first, or omit --use-local."
        )

    inventory = pd.read_csv(path).fillna("")
    downloaded_runs = []
    for _, row in inventory.iterrows():
        table_paths = [
            Path(table_path) for table_path in json.loads(row["completion_tables"])
        ]
        selected_paths = select_recent_completion_tables(
            table_paths,
            max_tables_per_run,
        )
        available_table_count = row.get("available_table_count", "")
        available_table_count = (
            int(available_table_count)
            if str(available_table_count).isdigit()
            else len(table_paths)
        )
        if available_table_count < min_available_tables:
            continue

        downloaded_runs.append(
            DownloadedRun(
                run_id=str(row["run_id"]),
                run_name=str(row["run_name"]),
                forget_concept=str(row["forget_concept"]),
                reward_type=str(row["reward_type"]),
                model_name=str(row["model_name"]),
                training_variant=str(row["training_variant"]),
                completion_tables=selected_paths,
                available_table_count=available_table_count,
            )
        )
    return downloaded_runs


def score_run(
    downloaded_run: DownloadedRun,
    *,
    args: argparse.Namespace,
) -> pd.DataFrame:
    path = run_score_path(args.output_dir, downloaded_run)
    if path.exists() and not args.overwrite_run_metrics:
        print(f"Reusing {path}", flush=True)
        return pd.read_csv(path)

    completion_rows = load_last_steps(downloaded_run, args.last_steps)
    print(
        f"Scoring {len(completion_rows)} completion(s) from {downloaded_run.run_name}.",
        flush=True,
    )
    scored_df = add_rubrics(completion_rows, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(path, index=False)
    return scored_df


def score_runs(
    downloaded_runs: list[DownloadedRun],
    args: argparse.Namespace,
) -> pd.DataFrame:
    frames = [score_run(downloaded_run, args=args) for downloaded_run in downloaded_runs]
    if not frames:
        raise ValueError("No scored completion rows found.")
    return pd.concat(frames, ignore_index=True)


def add_rubrics(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    scored_frames = []
    for concept, concept_df in df.groupby("forget_concept", sort=False):
        scored_frames.append(
            add_llm_judge_metrics(
                concept_df,
                concept=concept,
                judge_model=args.judge_model,
                judge_reasoning_effort=args.judge_reasoning_effort,
                max_concurrent_requests=args.judge_concurrency,
            )
        )
    return pd.concat(scored_frames, ignore_index=True)


def mean_table(
    df: pd.DataFrame,
    group_columns: list[str],
    *,
    count_column: str,
) -> pd.DataFrame:
    summary = df.groupby(group_columns, as_index=False)[llm_judge_metrics].mean()
    counts = df.groupby(group_columns, as_index=False).size()
    counts = counts.rename(columns={"size": count_column})
    return summary.merge(counts, on=group_columns, how="left")


def aggregate(scored_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_columns = [
        "run_id",
        "run_name",
        "forget_concept",
        "reward_type",
        "model_name",
        "training_variant",
        "step",
        "last_step_index",
        "step_from_end",
    ]
    prompt_columns = [*run_columns, "prompt_index", "prompt"]
    experiment_columns = [
        "forget_concept",
        "reward_type",
        "model_name",
        "training_variant",
        "step_from_end",
    ]
    final_columns = [
        "reward_type",
        "model_name",
        "training_variant",
        "step_from_end",
    ]

    prompt_summary = mean_table(
        scored_df,
        prompt_columns,
        count_column="completion_count",
    )
    run_summary = mean_table(
        prompt_summary,
        run_columns,
        count_column="prompt_count",
    )
    author_summary = mean_table(
        run_summary,
        experiment_columns,
        count_column="run_count",
    )
    final_summary = mean_table(
        author_summary,
        final_columns,
        count_column="author_count",
    )
    return prompt_summary, run_summary, author_summary, final_summary


def write_outputs(
    output_dir: Path,
    scored_df: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    author_summary: pd.DataFrame,
    final_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_dir / "metrics.csv", index=False)
    prompt_summary.to_csv(output_dir / "prompt_summary.csv", index=False)
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    author_summary.to_csv(output_dir / "author_summary.csv", index=False)
    final_summary.to_csv(output_dir / "summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download recent W&B training completions, score hold-out rubrics, "
            "and aggregate across generations, prompts, and authors."
        )
    )
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--notes", default=DEFAULT_NOTES)
    parser.add_argument("--last-steps", type=int, default=10)
    parser.add_argument(
        "--max-tables-per-run",
        type=int,
        default=10,
        help="Download or reuse only the last N completion table files per run. Use 0 for all.",
    )
    parser.add_argument(
        "--min-available-tables",
        type=int,
        default=101,
        help="Skip runs with fewer than this many completion table files available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_dynamics_hold_out_rubrics"),
    )
    parser.add_argument("--judge-model", default="gpt-5.4-nano")
    parser.add_argument("--judge-reasoning-effort", default="low")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download W&B completion tables, write inventory files, and exit.",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Read the existing download inventory instead of contacting W&B.",
    )
    parser.add_argument(
        "--overwrite-run-metrics",
        action="store_true",
        help="Recompute per-run rubric CSVs even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    args = parse_args()
    if not args.project and not args.use_local:
        raise ValueError("Set WANDB_PROJECT or pass --project.")

    if args.use_local:
        downloaded_runs = read_inventory(
            args.output_dir,
            args.max_tables_per_run,
            args.min_available_tables,
        )
    else:
        download_dir = args.output_dir / "wandb-downloads"
        downloaded_runs = download_completion_tables(
            project=args.project,
            notes=args.notes,
            download_dir=download_dir,
            max_tables_per_run=args.max_tables_per_run,
            min_available_tables=args.min_available_tables,
        )
        write_inventory(
            downloaded_runs,
            output_dir=args.output_dir,
            last_steps=args.last_steps,
        )
    print(f"Using {len(downloaded_runs)} W&B run(s).", flush=True)

    if args.download_only:
        print(f"Wrote download inventory to {inventory_path(args.output_dir)}", flush=True)
        return

    scored_df = score_runs(downloaded_runs, args)
    prompt_summary, run_summary, author_summary, final_summary = aggregate(scored_df)
    write_outputs(
        args.output_dir,
        scored_df,
        prompt_summary,
        run_summary,
        author_summary,
        final_summary,
    )
    print(f"Wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
