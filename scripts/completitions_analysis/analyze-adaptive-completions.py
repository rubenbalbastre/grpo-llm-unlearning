#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reward.refusal_patterns import REFUSAL_PATTERNS  # noqa: E402


TEXT_COLUMNS = ("completion", "prediction", "response", "output", "text")
REWARD_COLUMNS = ("forgetting_reward", "unlearning_reward_func", "reward", "rewards", "score", "rouge_l_recall")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’.-]*")
WANDB_COMPLETION_FILE_RE = re.compile(r"(^|/)media/table/completions_.*\.table\.json$")
SOURCE_KIND = "wandb_completion"
TRAINING_JOB_TYPE = "training"


@dataclass
class CompletionRow:
    source: Path
    row_index: int
    text: str
    reward: str | float | int | None


@dataclass
class Finding:
    row: CompletionRow
    exact_target: bool
    target_entity: str
    target_variant: str
    target_score: float
    exact_entities: list[str]


@dataclass
class SummaryRow:
    group: str
    training_mode: str
    source_kind: str
    total_rows: int = 0
    matching_filter_rows: int = 0
    exact_target_rows: int = 0
    near_target_rows: int = 0
    near_without_exact_rows: int = 0
    exact_entity_rows: int = 0


@dataclass(frozen=True)
class FuzzyTarget:
    entity: str
    normalized: str


@dataclass(frozen=True)
class DownloadedCompletion:
    path: Path
    training_mode: str


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def boundary_regex(text: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(text)}(?!\w)", flags=re.IGNORECASE)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(stringify(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def iter_wandb_table_rows(path: Path) -> Iterable[CompletionRow]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    columns = raw.get("columns") or []
    data = raw.get("data") or []
    if not isinstance(columns, list) or not isinstance(data, list):
        return

    text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
    if text_column is None:
        return

    reward_column = next((name for name in REWARD_COLUMNS if name in columns), None)
    text_index = columns.index(text_column)
    reward_index = columns.index(reward_column) if reward_column else None

    for row_index, values in enumerate(data, start=1):
        if not isinstance(values, list) or text_index >= len(values):
            continue
        yield CompletionRow(
            source=path,
            row_index=row_index,
            text=stringify(values[text_index]),
            reward=values[reward_index] if reward_index is not None and reward_index < len(values) else None,
        )


def run_name_from_path(path: Path) -> str:
    if path.parent.name == "table" and path.parent.parent.name == "media":
        return path.parent.parent.parent.name
    return "all"


def make_summary(group: str, training_mode: str) -> SummaryRow:
    return SummaryRow(group=group, training_mode=training_mode, source_kind=SOURCE_KIND)


def record_finding(stats: SummaryRow, finding: Finding) -> None:
    near_without_exact = bool(finding.target_variant) and not finding.exact_target
    stats.exact_target_rows += int(finding.exact_target)
    stats.near_target_rows += int(bool(finding.target_variant))
    stats.near_without_exact_rows += int(near_without_exact)
    stats.exact_entity_rows += int(bool(finding.exact_entities))


def reward_matches(value: str | float | int | None, expected: float | None) -> bool:
    if expected is None:
        return True
    if value in (None, ""):
        return False
    try:
        return abs(float(value) - expected) < 1e-9
    except (TypeError, ValueError):
        return False


def build_fuzzy_targets(target_entities: list[str]) -> dict[str, list[FuzzyTarget]]:
    targets: dict[str, list[FuzzyTarget]] = defaultdict(list)
    for entity in target_entities:
        normalized = normalize_token(entity)
        if len(normalized) < 4 or " " in entity.strip():
            continue
        targets[normalized[0]].append(FuzzyTarget(entity=entity, normalized=normalized))
    return dict(targets)


def best_target_variant(
    text: str,
    fuzzy_targets: dict[str, list[FuzzyTarget]],
    threshold: float,
) -> tuple[str, str, float]:
    best_variant = ""
    best_entity = ""
    best_score = 0.0
    for token in set(TOKEN_RE.findall(text)):
        normalized = normalize_token(token)
        if not normalized:
            continue
        for target in fuzzy_targets.get(normalized[0], []):
            if normalized == target.normalized:
                continue
            if abs(len(normalized) - len(target.normalized)) > 2:
                continue
            score = SequenceMatcher(None, normalized, target.normalized).ratio()
            if score > best_score:
                best_variant = token
                best_entity = target.entity
                best_score = score

    return (best_variant, best_entity, best_score) if best_score >= threshold else ("", "", 0.0)


def analyze_row(
    row: CompletionRow,
    target_matchers: list[tuple[str, re.Pattern[str]]],
    fuzzy_targets: dict[str, list[FuzzyTarget]],
    near_threshold: float,
) -> Finding | None:
    exact_entities = [entity for entity, matcher in target_matchers if matcher.search(row.text)]
    variant, target_entity, score = best_target_variant(row.text, fuzzy_targets, near_threshold)
    exact_target = bool(exact_entities)

    if not (exact_target or variant or exact_entities):
        return None
    return Finding(
        row=row,
        exact_target=exact_target,
        target_entity=target_entity,
        target_variant=variant,
        target_score=score,
        exact_entities=exact_entities,
    )


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "source_kind",
                "row_index",
                "reward",
                "exact_target",
                "target_entity",
                "target_variant",
                "target_score",
                "exact_entities",
            ],
        )
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "source": finding.row.source,
                    "source_kind": SOURCE_KIND,
                    "row_index": finding.row.row_index,
                    "reward": finding.row.reward,
                    "exact_target": finding.exact_target,
                    "target_entity": finding.target_entity,
                    "target_variant": finding.target_variant,
                    "target_score": f"{finding.target_score:.4f}",
                    "exact_entities": "|".join(finding.exact_entities),
                }
            )


def write_summary_csv(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "training_mode",
                "source_kind",
                "total_rows",
                "matching_filter_rows",
                "exact_target_rows",
                "near_target_rows",
                "near_without_exact_rows",
                "exact_entity_rows",
                "near_without_exact_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            denominator = row.matching_filter_rows or 0
            rate = row.near_without_exact_rows / denominator if denominator else 0.0
            writer.writerow(
                {
                    "group": row.group,
                    "training_mode": row.training_mode,
                    "source_kind": row.source_kind,
                    "total_rows": row.total_rows,
                    "matching_filter_rows": row.matching_filter_rows,
                    "exact_target_rows": row.exact_target_rows,
                    "near_target_rows": row.near_target_rows,
                    "near_without_exact_rows": row.near_without_exact_rows,
                    "exact_entity_rows": row.exact_entity_rows,
                    "near_without_exact_rate": f"{rate:.8f}",
                }
            )


def default_summary_path(matches_path: Path | None) -> Path | None:
    if matches_path is None:
        return None
    return matches_path.with_name(f"{matches_path.stem}.summary.csv")


def load_env_file(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def config_get(config: dict[str, Any], keys: Iterable[str]) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if isinstance(value, dict) and set(value) == {"value"}:
            value = value["value"]
    return value


def run_job_type(run: Any) -> str | None:
    value = (
        getattr(run, "job_type", None)
        or getattr(run, "jobType", None)
        or getattr(run, "_attrs", {}).get("jobType")
    )
    return str(value) if value else None


def download_wandb_completion_files(
    *,
    entity: str | None,
    project: str,
    forget_concept: str,
    model_name: str | None,
    download_dir: Path,
) -> list[DownloadedCompletion]:
    import wandb

    api = wandb.Api()
    wandb_path = f"{entity}/{project}" if entity else project
    downloaded: list[DownloadedCompletion] = []
    matched_runs = 0

    for run in api.runs(wandb_path):
        config = dict(run.config or {})
        concept = config_get(config, ("hydra", "experiment", "forget_concept"))
        if concept != forget_concept:
            continue
        run_model_name = config_get(config, ("hydra", "model", "name"))
        if model_name and run_model_name != model_name:
            continue
        training_mode = config_get(config, ("hydra", "training", "mode"))
        if training_mode is None:
            training_mode = "unknown"
        job_type = run_job_type(run)
        if job_type and job_type != TRAINING_JOB_TYPE:
            continue

        run_dir = download_dir / f"{run.name or run.id}-{run.id}"
        matched_runs += 1
        run_file_count = 0
        for wandb_file in run.files():
            if not WANDB_COMPLETION_FILE_RE.search(wandb_file.name):
                continue
            local_path = run_dir / wandb_file.name
            if not local_path.exists():
                local_path = Path(wandb_file.download(root=str(run_dir), replace=True).name)
            downloaded.append(DownloadedCompletion(path=local_path, training_mode=str(training_mode)))
            run_file_count += 1
        print(
            f"found {run_file_count} W&B completion table(s) for {run.name or run.id}",
            flush=True,
        )

    model_filter = f", hydra.model.name={model_name!r}" if model_name else ""
    print(
        f"matched {matched_runs} W&B training run(s) with "
        f"hydra.experiment.forget_concept={forget_concept!r}{model_filter}",
        flush=True,
    )
    return sorted(downloaded, key=lambda item: str(item.path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download W&B training completions and scan for exact or misspelled target-entity mentions."
    )
    parser.add_argument(
        "--concept",
        default="Rihanna",
        choices=sorted(REFUSAL_PATTERNS),
        help="Forgotten concept; filters W&B hydra.experiment.forget_concept.",
    )
    parser.add_argument("--near-threshold", type=float, default=0.84)
    parser.add_argument("--forgetting-reward", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--model-name", default=None, help="Filter W&B runs by hydra.model.name.")
    parser.add_argument(
        "--wandb-download-dir",
        type=Path,
        default=Path("outputs/completion_analysis/wandb-downloads"),
        help="Directory for downloaded W&B completion table JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        raise ValueError("WANDB_PROJECT is required in the repository .env file or environment.")
    files = download_wandb_completion_files(
        entity=os.environ.get("WANDB_ENTITY"),
        project=project,
        forget_concept=args.concept,
        model_name=args.model_name,
        download_dir=args.wandb_download_dir,
    )
    if not files:
        raise ValueError(f"No W&B completion tables found for concept {args.concept!r}.")
    target_entities = REFUSAL_PATTERNS[args.concept]
    target_matchers = [
        (entity, boundary_regex(entity))
        for entity in target_entities
    ]
    fuzzy_targets = build_fuzzy_targets(target_entities)

    rows_scanned = 0
    source_summary = defaultdict(lambda: make_summary("all", "all"))
    mode_summary = defaultdict(lambda: SummaryRow(group="", training_mode="", source_kind=SOURCE_KIND))
    run_summary = defaultdict(lambda: SummaryRow(group="", training_mode="", source_kind=SOURCE_KIND))
    variant_counts = Counter()
    entity_counts = Counter()
    findings: list[Finding] = []

    for file_index, completion_file in enumerate(files, start=1):
        path = completion_file.path
        training_mode = completion_file.training_mode
        print(f"[{file_index}/{len(files)}] scanning {path}", flush=True)
        for row in iter_wandb_table_rows(path):
            run = run_name_from_path(row.source)
            mode_stats = mode_summary[(training_mode, SOURCE_KIND)]
            mode_stats.group = training_mode
            mode_stats.training_mode = training_mode
            run_stats = run_summary[(training_mode, run, SOURCE_KIND)]
            run_stats.group = run
            run_stats.training_mode = training_mode

            summary_rows = [source_summary[SOURCE_KIND], mode_stats, run_stats]
            for stats in summary_rows:
                stats.total_rows += 1

            if not reward_matches(row.reward, args.forgetting_reward):
                continue

            rows_scanned += 1
            finding = analyze_row(
                row=row,
                target_matchers=target_matchers,
                fuzzy_targets=fuzzy_targets,
                near_threshold=args.near_threshold,
            )

            for stats in summary_rows:
                stats.matching_filter_rows += 1

            if finding is None:
                continue

            for stats in summary_rows:
                record_finding(stats, finding)

            variant_counts.update([finding.target_variant] if finding.target_variant else [])
            entity_counts.update(finding.exact_entities)
            findings.append(finding)
        print(f"[{file_index}/{len(files)}] done {path}", flush=True)

    exact_target_rows = sum(stats.exact_target_rows for stats in source_summary.values())
    near_target_rows = sum(stats.near_target_rows for stats in source_summary.values())
    near_without_exact_rows = sum(stats.near_without_exact_rows for stats in source_summary.values())
    exact_entity_rows = sum(stats.exact_entity_rows for stats in source_summary.values())

    print(f"concept: {args.concept}")
    if args.model_name:
        print(f"model_name filter: {args.model_name}")
    print(f"target_entities: {len(target_entities)} refusal pattern(s)")
    if args.forgetting_reward is not None:
        print(f"forgetting_reward filter: {args.forgetting_reward}")
    print(f"files scanned: {len(files)}")
    print(f"rows scanned: {rows_scanned}")
    print(f"rows with exact target-entity match: {exact_target_rows}")
    print(f"rows with near target-entity variant: {near_target_rows}")
    print(f"rows with near variant but no exact target-entity match: {near_without_exact_rows}")
    print(f"rows with any exact reward-list entity match: {exact_entity_rows}")

    print("\nby source kind:")
    for source_kind, stats in sorted(source_summary.items()):
        print(
            f"  {source_kind}: rows={stats.matching_filter_rows} "
            f"exact_target={stats.exact_target_rows} near_target={stats.near_target_rows} "
            f"near_without_exact={stats.near_without_exact_rows} "
            f"exact_entity={stats.exact_entity_rows}"
        )

    print("\nby training mode:")
    for (training_mode, source_kind), stats in sorted(mode_summary.items()):
        print(
            f"  {training_mode}/{source_kind}: rows={stats.matching_filter_rows} "
            f"exact_target={stats.exact_target_rows} near_target={stats.near_target_rows} "
            f"near_without_exact={stats.near_without_exact_rows} "
            f"exact_entity={stats.exact_entity_rows}"
        )

    if variant_counts:
        print("\ntop near target-entity variants:")
        for variant, count in variant_counts.most_common(20):
            print(f"  {variant!r}: {count}")

    if entity_counts:
        print("\ntop exact reward-list entities:")
        for entity, count in entity_counts.most_common(20):
            print(f"  {entity}: {count}")

    print("\nexamples:")
    for finding in findings[: args.max_examples]:
        snippet = " ".join(finding.row.text.split())[:300]
        print(
            f"- {SOURCE_KIND} {finding.row.source}:{finding.row.row_index} "
            f"reward={finding.row.reward!r} exact_target={finding.exact_target} "
            f"variant={finding.target_variant!r} target_entity={finding.target_entity!r} "
            f"score={finding.target_score:.3f} "
            f"exact_entities={finding.exact_entities}\n  {snippet}"
        )

    if args.output_csv:
        write_csv(args.output_csv, findings)
        print(f"\nwrote matches CSV: {args.output_csv}")

    summary_csv = args.summary_csv or default_summary_path(args.output_csv)
    if summary_csv:
        summary_rows = sorted(source_summary.values(), key=lambda row: (row.group, row.source_kind))
        summary_rows.extend(
            sorted(mode_summary.values(), key=lambda row: (row.training_mode, row.source_kind))
        )
        summary_rows.extend(
            sorted(run_summary.values(), key=lambda row: (row.training_mode, row.group, row.source_kind))
        )
        write_summary_csv(summary_csv, summary_rows)
        print(f"wrote summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
