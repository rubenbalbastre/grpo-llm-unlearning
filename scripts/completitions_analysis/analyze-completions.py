#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reward.fuzzy import (  # noqa: E402
    build_fuzzy_entities_for_concept,
    compute_fuzzy_reward_per_completion,
)
from src.reward.forgetting import build_entity_matchers


SOURCE_KIND = "wandb_completion"
MATCHES_FIELDNAMES = [
    "source",
    "source_kind",
    "row_index",
    "fuzzy_reward",
    "exact_target",
    "exact_entities",
]
SUMMARY_FIELDNAMES = [
    "group",
    "training_mode",
    "source_kind",
    "total_rows",
    "matching_filter_rows",
    "near_target_rows",
    "near_without_exact_rows",
    "exact_entity_rows",
    "near_without_exact_rate",
    "near_without_exact_rate_std",
]


@dataclass
class CompletionRow:
    source: Path
    row_index: int
    text: str
    forgetting_fuzzy_reward: float | None = None
    language_reward: float | None = None
    forgetting_reward: float | None = None


@dataclass
class Finding:
    row: CompletionRow
    exact_target: bool
    fuzzy_reward: float
    exact_entities: list[str]


@dataclass
class SummaryRow:
    group: str
    training_mode: str
    source_kind: str
    total_rows: int = 0
    matching_filter_rows: int = 0
    near_target_rows: int = 0
    near_without_exact_rows: int = 0
    exact_entity_rows: int = 0
    near_without_exact_rate_std: float | None = None


@dataclass(frozen=True)
class DownloadedCompletion:
    path: Path
    training_mode: str


@dataclass
class SummaryTables:
    source: defaultdict
    mode: defaultdict
    run: defaultdict


@dataclass
class AnalysisResult:
    rows_scanned: int
    summaries: SummaryTables
    entity_counts: Counter
    findings: list[Finding]


def iter_wandb_table_rows(path: Path) -> Iterable[CompletionRow]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    columns = raw.get("columns") or []
    data = raw.get("data") or []
    text_index = columns.index("completion")

    for row_index, values in enumerate(data, start=1):
        yield CompletionRow(
            source=path,
            row_index=row_index,
            text=stringify(values[text_index]),
            forgetting_fuzzy_reward=None,
            language_reward=None,
            forgetting_reward=None
        )


def run_name_from_path(path: Path) -> str:
    if path.parent.name == "table" and path.parent.parent.name == "media":
        return path.parent.parent.parent.name
    return "all"


def make_summary(group: str, training_mode: str) -> SummaryRow:
    return SummaryRow(group=group, training_mode=training_mode, source_kind=SOURCE_KIND)


def make_summary_tables() -> SummaryTables:
    return SummaryTables(
        source=defaultdict(lambda: make_summary("all", "all")),
        mode=defaultdict(lambda: SummaryRow(group="", training_mode="", source_kind=SOURCE_KIND)),
        run=defaultdict(lambda: SummaryRow(group="", training_mode="", source_kind=SOURCE_KIND)),
    )


def summary_rows_for(
    summaries: SummaryTables,
    row: CompletionRow,
    training_mode: str,
) -> list[SummaryRow]:
    run = run_name_from_path(row.source)
    mode_stats = summaries.mode[(training_mode, SOURCE_KIND)]
    mode_stats.group = training_mode
    mode_stats.training_mode = training_mode

    run_stats = summaries.run[(training_mode, run, SOURCE_KIND)]
    run_stats.group = run
    run_stats.training_mode = training_mode

    return [summaries.source[SOURCE_KIND], mode_stats, run_stats]


def record_finding(stats: SummaryRow, finding: Finding) -> None:
    near_target = finding.fuzzy_reward == 0.0
    near_without_exact = near_target and not finding.exact_target
    stats.near_target_rows += int(near_target)
    stats.near_without_exact_rows += int(near_without_exact)
    stats.exact_entity_rows += int(bool(finding.exact_entities))


def near_without_exact_rate(row: SummaryRow) -> float:
    if not row.matching_filter_rows:
        return 0.0
    return row.near_without_exact_rows / row.matching_filter_rows


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def set_rate_stds(
    source_rows: Iterable[SummaryRow],
    mode_rows: Iterable[SummaryRow],
    run_rows: Iterable[SummaryRow],
) -> None:
    run_rows = list(run_rows)
    for source_row in source_rows:
        rates = [
            near_without_exact_rate(run_row)
            for run_row in run_rows
            if run_row.source_kind == source_row.source_kind and run_row.matching_filter_rows
        ]
        source_row.near_without_exact_rate_std = sample_std(rates)

    for mode_row in mode_rows:
        rates = [
            near_without_exact_rate(run_row)
            for run_row in run_rows
            if run_row.training_mode == mode_row.training_mode
            and run_row.source_kind == mode_row.source_kind
            and run_row.matching_filter_rows
        ]
        mode_row.near_without_exact_rate_std = sample_std(rates)


def analyze_row(
    row: CompletionRow,
    target_matchers: list[tuple[str, re.Pattern[str]]],
    fuzzy_reward: float,
) -> Finding | None:
    exact_entities = [entity for entity, matcher in target_matchers if matcher.search(row.text)]
    exact_target = bool(exact_entities)

    if not (exact_target or fuzzy_reward == 0.0):
        return None
    return Finding(
        row=row,
        exact_target=exact_target,
        fuzzy_reward=fuzzy_reward,
        exact_entities=exact_entities,
    )


def scan_completion_files(
    files: list[DownloadedCompletion],
    *,
    concept: str,
) -> AnalysisResult:

    target_matchers = build_entity_matchers(concept)
    fuzzy_entities = build_fuzzy_entities_for_concept(concept)
    summaries = make_summary_tables()
    entity_counts = Counter()
    findings: list[Finding] = []
    rows_scanned = 0

    for file_index, completion_file in enumerate(files, start=1):
        path = completion_file.path
        print(f"[{file_index}/{len(files)}] scanning {path}", flush=True)

        for row in iter_wandb_table_rows(path):
            summary_rows = summary_rows_for(
                summaries=summaries,
                row=row,
                training_mode=completion_file.training_mode,
            )
            for stats in summary_rows:
                stats.total_rows += 1

            rows_scanned += 1
            if "train/rewards/forgetting_fuzzy_reward/mean" not in row:
                fuzzy_reward = compute_fuzzy_reward_per_completion(row.text, fuzzy_entities)
            else:
                fuzzy_reward = float(row["train/rewards/forgetting_fuzzy_reward/mean"])

            finding = analyze_row(
                row=row,
                target_matchers=target_matchers,
                fuzzy_reward=fuzzy_reward,
            )

            for stats in summary_rows:
                stats.matching_filter_rows += 1

            if finding is None:
                continue

            for stats in summary_rows:
                record_finding(stats, finding)

            entity_counts.update(finding.exact_entities)
            findings.append(finding)

        print(f"[{file_index}/{len(files)}] done {path}", flush=True)

    return AnalysisResult(
        rows_scanned=rows_scanned,
        summaries=summaries,
        entity_counts=entity_counts,
        findings=findings,
    )


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCHES_FIELDNAMES)
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "source": finding.row.source,
                    "source_kind": SOURCE_KIND,
                    "row_index": finding.row.row_index,
                    "fuzzy_reward": f"{finding.fuzzy_reward:.8f}",
                    "exact_target": finding.exact_target,
                    "exact_entities": "|".join(finding.exact_entities),
                }
            )


def write_summary_csv(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            rate = near_without_exact_rate(row)
            rate_std = (
                f"{row.near_without_exact_rate_std:.8f}"
                if row.near_without_exact_rate_std is not None
                else ""
            )
            writer.writerow(
                {
                    "group": row.group,
                    "training_mode": row.training_mode,
                    "source_kind": row.source_kind,
                    "total_rows": row.total_rows,
                    "matching_filter_rows": row.matching_filter_rows,
                    "near_target_rows": row.near_target_rows,
                    "near_without_exact_rows": row.near_without_exact_rows,
                    "exact_entity_rows": row.exact_entity_rows,
                    "near_without_exact_rate": f"{rate:.8f}",
                    "near_without_exact_rate_std": rate_std,
                }
            )


def download_wandb_completion_files(
        *,
        project: str,
        forget_concept: str,
        reward_type: str,
        model_name: str | None,
        download_dir: Path,
    ) -> list[DownloadedCompletion]:
    import wandb

    api = wandb.Api()
    downloaded: list[DownloadedCompletion] = []
    matched_runs = 0

    for run in api.runs(project):

        # training job?
        if getattr(run, "jobType") == "training":

            # inspect run config
            config = dict(run.config)
            run_concept = config.get("hydra").get("experiment").get("forget_concept")
            run_model_name = config.get("hydra").get("model").get("name")
            run_reward_type = config.get("hydra").get("reward").get("type")
            training_mode = config.get("hydra").get("training").get("mode")

            if (run_concept == forget_concept) & (run_reward_type == reward_type) & (run_model_name == model_name):

                # download info
                run_dir = download_dir / f"{run.name or run.id}-{run.id}"
                matched_runs += 1
                run_file_count = 0
                run
                for wandb_file in run.files():
                    local_path = run_dir / wandb_file.name
                    if not local_path.exists():
                        local_path = Path(wandb_file.download(root=str(run_dir), replace=True).name)
                    downloaded.append(DownloadedCompletion(path=local_path, training_mode=str(training_mode)))
                    run_file_count += 1
                print(
                    f"found {run_file_count} W&B completion table(s) for {run.name or run.id}",
                    flush=True,
                )

    print(f"matched {matched_runs} W&B training run(s)")
    return sorted(downloaded, key=lambda item: str(item.path))


def sorted_summary_rows(summaries: SummaryTables) -> list[SummaryRow]:
    rows = sorted(summaries.source.values(), key=lambda row: (row.group, row.source_kind))
    rows.extend(sorted(summaries.mode.values(), key=lambda row: (row.training_mode, row.source_kind)))
    rows.extend(
        sorted(summaries.run.values(), key=lambda row: (row.training_mode, row.group, row.source_kind))
    )
    return rows


def write_outputs(args: argparse.Namespace, result: AnalysisResult) -> None:
    if args.output_csv:
        write_csv(args.output_csv, result.findings)
        print(f"\nwrote matches CSV: {args.output_csv}")

    summary_csv = args.output_csv.with_name(f"{args.output_csv.stem}.summary.csv")
    set_rate_stds(
        source_rows=result.summaries.source.values(),
        mode_rows=result.summaries.mode.values(),
        run_rows=result.summaries.run.values(),
    )
    write_summary_csv(summary_csv, sorted_summary_rows(result.summaries))
    print(f"wrote summary CSV: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download W&B training completions and scan them with exact and fuzzy reward matching."
    )
    parser.add_argument(
        "--concept",
        default="Rihanna",
        help="Forgotten concept; filters W&B hydra.experiment.forget_concept.",
    )
    parser.add_argument("--output-csv", type=Path, default="outputs/completion_analysis/test-wandb-completions.csv")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct", help="Filter W&B runs by hydra.model.name.")
    parser.add_argument(
        "--reward-type",
        default="r1",
        choices=["r0", "r1", "r2"],
        help="Which reward to use for fuzzy matching; determines which fuzzy entity list is used.",
    )
    parser.add_argument(
        "--wandb-download-dir",
        type=Path,
        default=Path("outputs/completion_analysis/wandb-downloads"),
        help="Directory for downloaded W&B completion table JSON files.",
    )
    return parser.parse_args()


def main() -> None:

    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    args = parse_args()
    project = os.environ.get("WANDB_PROJECT")

    files = download_wandb_completion_files(
        project=project,
        forget_concept=args.concept,
        reward_type=args.reward_type,
        model_name=args.model_name,
        download_dir=args.wandb_download_dir,
    )
    result = scan_completion_files(
        files,
        concept=args.concept,
    )
    write_outputs(args, result)


if __name__ == "__main__":
    main()
