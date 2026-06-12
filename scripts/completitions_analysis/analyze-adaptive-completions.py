#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
PROMPT_COLUMNS = ("prompt", "query", "input")
REWARD_COLUMNS = ("forgetting_reward", "reward", "rewards", "score", "rouge_l_recall")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’.-]*")
PARQUET_WARNING_SHOWN = False


@dataclass
class CompletionRow:
    source: Path
    source_kind: str
    row_index: int
    prompt: str
    text: str
    reward: str | float | int | None


@dataclass
class Finding:
    row: CompletionRow
    exact_target: bool
    target_variant: str
    target_score: float
    exact_entities: list[str]


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


def first_existing(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def jsonl_source_kind(path: Path, row: dict[str, Any]) -> str:
    if path.name == "rwku_generations.jsonl" or "rouge_l_recall" in row:
        return "rwku_generation"
    if row.get("type") == "reward":
        return f"reward:{row.get('reward_component', 'unknown')}"
    return path.stem


def iter_jsonl_rows(path: Path) -> Iterable[CompletionRow]:
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            text = first_existing(raw, TEXT_COLUMNS)
            if text is None:
                continue
            yield CompletionRow(
                source=path,
                source_kind=jsonl_source_kind(path, raw),
                row_index=row_index,
                prompt=stringify(first_existing(raw, PROMPT_COLUMNS)),
                text=stringify(text),
                reward=first_existing(raw, REWARD_COLUMNS),
            )


def iter_parquet_rows(path: Path) -> Iterable[CompletionRow]:
    global PARQUET_WARNING_SHOWN
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        if not PARQUET_WARNING_SHOWN:
            print(
                "warning: pyarrow is not installed; skipping TRL completion parquet files",
                file=sys.stderr,
            )
            PARQUET_WARNING_SHOWN = True
        return

    parquet_file = pq.ParquetFile(path)
    columns = set(parquet_file.schema_arrow.names)
    text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
    if text_column is None:
        print(f"warning: skipping {path}; no text column among {TEXT_COLUMNS}", file=sys.stderr)
        return

    prompt_column = next((name for name in PROMPT_COLUMNS if name in columns), None)
    reward_column = next((name for name in REWARD_COLUMNS if name in columns), None)
    selected_columns = [column for column in (text_column, prompt_column, reward_column) if column]
    data = parquet_file.read(columns=selected_columns).to_pydict()

    for row_index, text in enumerate(data[text_column], start=1):
        yield CompletionRow(
            source=path,
            source_kind="trl_completion",
            row_index=row_index,
            prompt=stringify(data[prompt_column][row_index - 1]) if prompt_column else "",
            text=stringify(text),
            reward=data[reward_column][row_index - 1] if reward_column else None,
        )


def collect_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted((path / "completions").glob("*.parquet")))
            files.extend(sorted(path.glob("checkpoint-*/eval_rwku/rwku_generations.jsonl")))
        else:
            print(f"warning: path does not exist: {path}", file=sys.stderr)
    return files


def iter_file_rows(path: Path) -> Iterable[CompletionRow]:
    if path.suffix == ".jsonl":
        yield from iter_jsonl_rows(path)
    elif path.suffix == ".parquet":
        yield from iter_parquet_rows(path)


def reward_matches(value: str | float | int | None, expected: float | None) -> bool:
    if expected is None:
        return True
    if value in (None, ""):
        return False
    try:
        return abs(float(value) - expected) < 1e-9
    except (TypeError, ValueError):
        return False


def best_target_variant(text: str, target_name: str, threshold: float) -> tuple[str, float]:
    target = normalize_token(target_name)
    if len(target) < 4:
        raise ValueError("--target-name must have at least 4 normalized characters.")

    best_variant = ""
    best_score = 0.0
    for token in TOKEN_RE.findall(text):
        normalized = normalize_token(token)
        if not normalized or normalized == target:
            continue
        if normalized[0] != target[0] or abs(len(normalized) - len(target)) > 2:
            continue
        score = SequenceMatcher(None, normalized, target).ratio()
        if score > best_score:
            best_variant = token
            best_score = score

    return (best_variant, best_score) if best_score >= threshold else ("", 0.0)


def analyze_row(
    row: CompletionRow,
    target_regex: re.Pattern[str],
    target_name: str,
    near_threshold: float,
    entity_matchers: list[tuple[str, re.Pattern[str]]],
) -> Finding | None:
    exact_target = bool(target_regex.search(row.text))
    variant, score = best_target_variant(row.text, target_name, near_threshold)
    exact_entities = [entity for entity, matcher in entity_matchers if matcher.search(row.text)]

    if not (exact_target or variant or exact_entities):
        return None
    return Finding(
        row=row,
        exact_target=exact_target,
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
                "target_variant",
                "target_score",
                "exact_entities",
                "prompt",
                "text",
            ],
        )
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "source": finding.row.source,
                    "source_kind": finding.row.source_kind,
                    "row_index": finding.row.row_index,
                    "reward": finding.row.reward,
                    "exact_target": finding.exact_target,
                    "target_variant": finding.target_variant,
                    "target_score": f"{finding.target_score:.4f}",
                    "exact_entities": "|".join(finding.exact_entities),
                    "prompt": finding.row.prompt,
                    "text": finding.row.text,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan completions for exact and misspelled target-name mentions."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Run dirs or completion/eval files.")
    parser.add_argument("--concept", default="Rihanna", choices=sorted(REFUSAL_PATTERNS))
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--near-threshold", type=float, default=0.84)
    parser.add_argument("--forgetting-reward", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_name = args.target_name or args.concept
    files = collect_input_files(args.paths)
    target_regex = boundary_regex(target_name)
    entity_matchers = [
        (entity, boundary_regex(entity))
        for entity in REFUSAL_PATTERNS[args.concept]
    ]

    rows_scanned = 0
    summary = defaultdict(
        lambda: {"rows": 0, "exact_target": 0, "near_target": 0, "near_without_exact": 0, "exact_entity": 0}
    )
    variant_counts = Counter()
    entity_counts = Counter()
    findings: list[Finding] = []

    for file_index, path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] scanning {path}", flush=True)
        for row in iter_file_rows(path):
            if args.forgetting_reward is not None and row.source_kind != "trl_completion":
                continue
            if not reward_matches(row.reward, args.forgetting_reward):
                continue

            rows_scanned += 1
            finding = analyze_row(
                row=row,
                target_regex=target_regex,
                target_name=target_name,
                near_threshold=args.near_threshold,
                entity_matchers=entity_matchers,
            )

            stats = summary[row.source_kind]
            stats["rows"] += 1
            if finding is None:
                continue

            near_without_exact = bool(finding.target_variant) and not finding.exact_target
            stats["exact_target"] += int(finding.exact_target)
            stats["near_target"] += int(bool(finding.target_variant))
            stats["near_without_exact"] += int(near_without_exact)
            stats["exact_entity"] += int(bool(finding.exact_entities))

            variant_counts.update([finding.target_variant] if finding.target_variant else [])
            entity_counts.update(finding.exact_entities)
            findings.append(finding)
        print(f"[{file_index}/{len(files)}] done {path}", flush=True)

    exact_target_rows = sum(stats["exact_target"] for stats in summary.values())
    near_target_rows = sum(stats["near_target"] for stats in summary.values())
    near_without_exact_rows = sum(stats["near_without_exact"] for stats in summary.values())
    exact_entity_rows = sum(stats["exact_entity"] for stats in summary.values())

    print(f"concept: {args.concept}")
    print(f"target_name: {target_name}")
    if args.forgetting_reward is not None:
        print(f"forgetting_reward filter: {args.forgetting_reward}")
    print(f"files scanned: {len(files)}")
    print(f"rows scanned: {rows_scanned}")
    print(f"rows with exact target-name match: {exact_target_rows}")
    print(f"rows with near target-name variant: {near_target_rows}")
    print(f"rows with near variant but no exact target-name match: {near_without_exact_rows}")
    print(f"rows with any exact reward-list entity match: {exact_entity_rows}")

    print("\nby source kind:")
    for source_kind, stats in sorted(summary.items()):
        print(
            f"  {source_kind}: rows={stats['rows']} exact_target={stats['exact_target']} "
            f"near_target={stats['near_target']} near_without_exact={stats['near_without_exact']} "
            f"exact_entity={stats['exact_entity']}"
        )

    if variant_counts:
        print("\ntop near target-name variants:")
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
            f"- {finding.row.source_kind} {finding.row.source}:{finding.row.row_index} "
            f"reward={finding.row.reward!r} exact_target={finding.exact_target} "
            f"variant={finding.target_variant!r} score={finding.target_score:.3f} "
            f"exact_entities={finding.exact_entities}\n  {snippet}"
        )

    if args.output_csv:
        write_csv(args.output_csv, findings)
        print(f"\nwrote matches CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
