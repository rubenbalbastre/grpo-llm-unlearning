#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.behaviour.analysis_utils import (  # noqa: E402
    add_llm_judge_metrics,
    llm_judge_metrics,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "terminal_training_audit"
JUDGE_CONFIG = REPO_ROOT / "config" / "eval_behaviour.yaml"
DEFAULT_NOTES = "lluis-vives-runs-1-1M"
RUN_NAME_PATTERN = re.compile(r"(?:^|-)original(?:-|$)|(?:^|-)r\d+-warmed(?:-|$)")
AUTHORS = {
    "bruce lee",
    "confucius",
    "jennifer lopez",
    "john d rockefeller",
    "karl marx",
    "marlon brando",
    "serena williams",
    "tom clancy",
    "tony blair",
    "vincent van gogh",
}


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


def nested_get(mapping: dict, *keys: str) -> str:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value) if value is not None else ""


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def normalize_training_variant(value: str) -> str:
    return "r2-warmed" if value == "warmed" else value


def model_size(model_name: str, run_name: str = "") -> str:
    source = f"{model_name} {run_name}".lower()
    for pattern, size in (
        (r"0[._-]?5b", "0.5B"),
        (r"1[._-]?5b", "1.5B"),
        (r"3b", "3B"),
        (r"7b", "7B"),
    ):
        if re.search(pattern, source):
            return size
    return model_name


def training_variant(run_name: str) -> str:
    if re.search(r"(?:^|-)r\d+-warmed(?:-|$)", run_name.lower()):
        return "r2-warmed"
    return "original"


def run_names(run) -> list[str]:
    return [
        str(value)
        for value in (
            getattr(run, "name", ""),
            getattr(run, "display_name", ""),
            getattr(run, "id", ""),
        )
        if value
    ]


def selected_run_name(run) -> str:
    names = run_names(run)
    return next(
        (name for name in names if RUN_NAME_PATTERN.search(name.lower())),
        names[0] if names else str(run.id),
    )


def completion_table_sort_key(file_or_path) -> tuple[int, str]:
    name = getattr(file_or_path, "name", str(file_or_path))
    match = re.search(r"completions_(\d+)_", name)
    return (int(match.group(1)) if match else -1, name)


def select_recent_tables(files_or_paths, limit: int):
    files = sorted(files_or_paths, key=completion_table_sort_key)
    return files if limit <= 0 else files[-limit:]


def load_table(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and {"columns", "data"} <= set(raw):
        return pd.DataFrame(raw["data"], columns=raw["columns"])
    return pd.read_json(path, orient="split")


def load_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for order, path in enumerate(paths):
        frame = load_table(path)
        frame["_table_order"] = order
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_last_steps(run: DownloadedRun, last_steps: int) -> pd.DataFrame:
    frame = load_tables(run.completion_tables)
    required = {"step", "prompt", "completion"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{run.run_name} is missing columns: {sorted(missing)}")

    if "prompt_index" not in frame:
        frame["prompt_index"] = frame.groupby("step", sort=False)["prompt"].transform(
            lambda values: pd.factorize(values, sort=False)[0]
        )

    steps = sorted(frame["step"].dropna().unique())[-last_steps:]
    frame = frame[frame["step"].isin(steps)].copy()
    latest_table = frame.groupby("step")["_table_order"].transform("max")
    frame = frame[frame["_table_order"] == latest_table].copy()
    step_order = {step: index + 1 for index, step in enumerate(steps)}

    frame["last_step_index"] = frame["step"].map(step_order)
    frame["step_from_end"] = len(steps) + 1 - frame["last_step_index"]
    frame["run_id"] = run.run_id
    frame["run_name"] = run.run_name
    frame["forget_concept"] = run.forget_concept
    frame["reward_type"] = run.reward_type
    frame["model_name"] = run.model_name
    frame["model_size"] = model_size(run.model_name, run.run_name)
    frame["training_variant"] = normalize_training_variant(run.training_variant)
    return frame


def inventory_path(output_dir: Path) -> Path:
    return output_dir / "download_inventory.csv"


def run_score_path(output_dir: Path, run: DownloadedRun) -> Path:
    return output_dir / "run_metrics" / f"{safe_name(run.run_name)}-{run.run_id}.csv"


def cached_scores(run: DownloadedRun, output_dir: Path) -> pd.DataFrame:
    path = run_score_path(output_dir, run)
    if not path.exists():
        return pd.DataFrame(columns=["prompt", "completion", *llm_judge_metrics])

    scores = pd.read_csv(path)
    scores[["prompt", "completion"]] = scores[["prompt", "completion"]].fillna("")
    for metric in llm_judge_metrics:
        if metric not in scores:
            scores[metric] = pd.NA
    return scores[["prompt", "completion", *llm_judge_metrics]].drop_duplicates(
        ["prompt", "completion"], keep="last"
    )


def score_run(run: DownloadedRun, args: argparse.Namespace) -> None:
    selected = load_last_steps(run, args.last_steps)
    unique_pairs = selected.drop_duplicates(["prompt", "completion"]).copy()
    path = run_score_path(args.output_dir, run)
    scores = (
        pd.DataFrame(columns=["prompt", "completion", *llm_judge_metrics])
        if args.overwrite_run_metrics
        else cached_scores(run, args.output_dir)
    )
    merged = unique_pairs.merge(scores, on=["prompt", "completion"], how="left")
    missing = merged[llm_judge_metrics].isna().any(axis=1)

    if not missing.any():
        print(f"Reusing {path}", flush=True)
        return

    missing_pairs = merged.loc[missing, unique_pairs.columns]
    print(
        f"Scoring {len(missing_pairs)} prompt+completion pair(s) from {run.run_name}.",
        flush=True,
    )
    scored = []
    for concept, frame in missing_pairs.groupby("forget_concept", sort=False):
        scored.append(
            add_llm_judge_metrics(
                frame,
                concept=concept,
                judge_model=args.judge_model,
                judge_reasoning_effort=args.judge_reasoning_effort,
                max_concurrent_requests=args.judge_concurrency,
            )
        )
    scores = pd.concat([scores, *scored], ignore_index=True).drop_duplicates(
        ["prompt", "completion"], keep="last"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)


def load_scored_run(
    run: DownloadedRun,
    output_dir: Path,
    last_steps: int,
) -> pd.DataFrame:
    selected = load_last_steps(run, last_steps)
    scores = cached_scores(run, output_dir)
    expanded = selected.merge(scores, on=["prompt", "completion"], how="left")
    missing = expanded[llm_judge_metrics].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"{run_score_path(output_dir, run)} is missing scores for "
            f"{missing.sum()} selected row(s). Run scripts/run-training-behavior.sh first."
        )
    return expanded


def delete_local_run(output_dir: Path, run_name: str, run_id: str) -> None:
    run_dir = output_dir / "wandb-downloads" / f"{safe_name(run_name)}-{run_id}"
    score_path = output_dir / "run_metrics" / f"{safe_name(run_name)}-{run_id}.csv"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if score_path.exists():
        score_path.unlink()


def download_file(run_dir: Path, wandb_file) -> Path:
    path = run_dir / wandb_file.name
    if path.exists():
        print(f"Reusing {path}", flush=True)
        return path
    return Path(wandb_file.download(root=str(run_dir), replace=False, exist_ok=True).name)


def download_runs(args: argparse.Namespace) -> list[DownloadedRun]:
    import wandb

    api = wandb.Api()
    filters = {
        "jobType": "training",
        "config.hydra.experiment.notes": args.notes,
    }
    downloaded = []
    for wandb_run in api.runs(args.project, filters=filters, per_page=100, lazy=True):
        if not any(RUN_NAME_PATTERN.search(name.lower()) for name in run_names(wandb_run)):
            continue

        run_name = selected_run_name(wandb_run)
        hydra = dict(wandb_run.config or {}).get("hydra", {})
        author = nested_get(hydra, "experiment", "forget_concept")
        normalized_author = re.sub(r"[^a-z0-9]+", " ", author.lower()).strip()
        if normalized_author not in AUTHORS:
            continue

        files = list(wandb_run.files(pattern="%completions%.table.json", per_page=100))
        available_table_count = len(files)
        if available_table_count < args.min_available_tables:
            delete_local_run(args.output_dir, run_name, wandb_run.id)
            print(f"Skipping {run_name}: only {available_table_count} completion tables.")
            continue

        files = select_recent_tables(files, args.max_tables_per_run)
        run_dir = args.output_dir / "wandb-downloads" / f"{safe_name(run_name)}-{wandb_run.id}"
        paths = [download_file(run_dir, file) for file in files]
        run = DownloadedRun(
            run_id=wandb_run.id,
            run_name=run_name,
            forget_concept=author,
            reward_type=nested_get(hydra, "reward", "type"),
            model_name=nested_get(hydra, "model", "name"),
            training_variant=training_variant(run_name),
            completion_tables=paths,
            available_table_count=available_table_count,
        )
        downloaded.append(run)
        print(
            f"Downloaded {len(paths)} table(s) from {run_name}; "
            f"unique prompts: {load_last_steps(run, args.last_steps)['prompt'].nunique()}.",
            flush=True,
        )
    return downloaded


def write_inventory(runs: list[DownloadedRun], output_dir: Path, last_steps: int) -> None:
    rows = []
    for run in runs:
        frame = load_last_steps(run, last_steps)
        rows.append(
            {
                "run_id": run.run_id,
                "run_name": run.run_name,
                "forget_concept": run.forget_concept,
                "reward_type": run.reward_type,
                "model_name": run.model_name,
                "model_size": model_size(run.model_name, run.run_name),
                "training_variant": run.training_variant,
                "available_table_count": run.available_table_count,
                "selected_table_count": len(run.completion_tables),
                "selected_step_count": frame["step"].nunique(),
                "selected_unique_prompt_count": frame["prompt"].nunique(),
                "selected_unique_prompt_completion_count": frame[
                    ["prompt", "completion"]
                ].drop_duplicates().shape[0],
                "selected_row_count": len(frame),
                "completion_tables": json.dumps([str(path) for path in run.completion_tables]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(inventory_path(output_dir), index=False)
    if not inventory.empty:
        counts = inventory.groupby(
            ["forget_concept", "model_size", "reward_type", "training_variant"],
            as_index=False,
        ).agg(
            run_count=("run_id", "nunique"),
            selected_row_count=("selected_row_count", "sum"),
            selected_unique_prompt_count=("selected_unique_prompt_count", "sum"),
            selected_unique_prompt_completion_count=(
                "selected_unique_prompt_completion_count",
                "sum",
            ),
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
            f"Missing {path}. Run scripts/run-training-behavior.sh first."
        )

    runs = []
    for row in pd.read_csv(path).fillna("").to_dict(orient="records"):
        paths = select_recent_tables(
            [Path(value) for value in json.loads(row["completion_tables"])],
            max_tables_per_run,
        )
        available = int(row.get("available_table_count") or len(paths))
        if available < min_available_tables:
            delete_local_run(output_dir, str(row["run_name"]), str(row["run_id"]))
            continue
        runs.append(
            DownloadedRun(
                run_id=str(row["run_id"]),
                run_name=str(row["run_name"]),
                forget_concept=str(row["forget_concept"]),
                reward_type=str(row["reward_type"]),
                model_name=str(row["model_name"]),
                training_variant=normalize_training_variant(
                    str(row["training_variant"])
                ),
                completion_tables=paths,
                available_table_count=available,
            )
        )
    return runs


def parse_args() -> argparse.Namespace:
    judge = OmegaConf.load(JUDGE_CONFIG)
    parser = argparse.ArgumentParser(
        description="Download and score terminal training rollouts."
    )
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--notes", default=DEFAULT_NOTES)
    parser.add_argument("--last-steps", type=int, default=5)
    parser.add_argument("--max-tables-per-run", type=int, default=10)
    parser.add_argument("--min-available-tables", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--judge-model", default=str(judge.judge_model))
    parser.add_argument(
        "--judge-reasoning-effort", default=str(judge.judge_reasoning_effort)
    )
    parser.add_argument("--judge-concurrency", type=int, default=int(judge.judge_concurrency))
    parser.add_argument("--overwrite-run-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    if args.skip_download:
        runs = read_inventory(
            args.output_dir,
            args.max_tables_per_run,
            args.min_available_tables,
        )
    else:
        if not args.project:
            raise ValueError("Set WANDB_PROJECT or pass --project.")
        runs = download_runs(args)
        if not runs:
            raise SystemExit("No matching W&B runs found; existing inventory was not changed.")
        write_inventory(runs, args.output_dir, args.last_steps)
        print(f"Wrote {len(runs)} runs to {inventory_path(args.output_dir)}")

    for run in runs:
        score_run(run, args)
    print(f"Prepared rubric scores for {len(runs)} run(s).")


if __name__ == "__main__":
    main()
