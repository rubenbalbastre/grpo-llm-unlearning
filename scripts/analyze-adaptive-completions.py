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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reward.refusal_patterns import REFUSAL_PATTERNS  # noqa: E402


TEXT_COLUMNS = ("completion", "prediction", "response", "output", "text")
PROMPT_COLUMNS = ("prompt", "query", "input")
REWARD_COLUMNS = ("forgetting_reward", "reward", "rewards", "score", "rouge_l_recall")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’.-]*")
PARQUET_WARNING_EMITTED = False


@dataclass
class CompletionRow:
    source: Path
    source_kind: str
    row_index: int
    prompt: str
    text: str
    reward: str | float | int | None


@dataclass
class MatchRow:
    source: str
    source_kind: str
    row_index: int
    reward: str | float | int | None
    exact_target: bool
    target_variant: str
    target_score: float
    exact_entities: list[str]
    prompt: str
    text: str


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def word_boundary_pattern(value: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", flags=re.IGNORECASE)


def build_entity_matchers(concept: str) -> list[tuple[str, re.Pattern[str]]]:
    return [(entity, word_boundary_pattern(entity)) for entity in REFUSAL_PATTERNS[concept]]


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


def first_present(row: dict[str, Any], columns: Iterable[str]) -> Any:
    for column in columns:
        if column in row:
            return row[column]
    return None


def jsonl_kind(path: Path, row: dict[str, Any]) -> str:
    if path.name == "rwku_generations.jsonl" or "rouge_l_recall" in row:
        return "rwku_generation"
    if row.get("type") == "reward":
        return f"reward:{row.get('reward_component', 'unknown')}"
    return path.stem


def iter_jsonl(path: Path) -> Iterable[CompletionRow]:
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = first_present(row, TEXT_COLUMNS)
            if text is None:
                continue
            yield CompletionRow(
                source=path,
                source_kind=jsonl_kind(path, row),
                row_index=row_index,
                prompt=stringify(first_present(row, PROMPT_COLUMNS)),
                text=stringify(text),
                reward=first_present(row, REWARD_COLUMNS),
            )


def iter_parquet(path: Path) -> Iterable[CompletionRow]:
    global PARQUET_WARNING_EMITTED
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        if not PARQUET_WARNING_EMITTED:
            print(
                "warning: pyarrow is not installed; skipping TRL completion parquet files",
                file=sys.stderr,
            )
            PARQUET_WARNING_EMITTED = True
        return

    parquet_file = pq.ParquetFile(path)
    columns = set(parquet_file.schema_arrow.names)
    text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
    if text_column is None:
        print(f"warning: skipping {path}; no text column among {TEXT_COLUMNS}", file=sys.stderr)
        return

    prompt_column = next((name for name in PROMPT_COLUMNS if name in columns), None)
    reward_column = next((name for name in REWARD_COLUMNS if name in columns), None)
    read_columns = [text_column]
    if prompt_column:
        read_columns.append(prompt_column)
    if reward_column:
        read_columns.append(reward_column)

    table = parquet_file.read(columns=read_columns)
    data = table.to_pydict()
    for row_index, text in enumerate(data[text_column], start=1):
        yield CompletionRow(
            source=path,
            source_kind="trl_completion",
            row_index=row_index,
            prompt=stringify(data[prompt_column][row_index - 1]) if prompt_column else "",
            text=stringify(text),
            reward=data[reward_column][row_index - 1] if reward_column else None,
        )


def collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            print(f"warning: path does not exist: {path}", file=sys.stderr)
            continue
        files.extend(sorted((path / "completions").glob("*.parquet")))
        files.extend(sorted(path.glob("events.jsonl")))
        files.extend(sorted(path.glob("checkpoint-*/eval_rwku/rwku_generations.jsonl")))
    return files


def iter_rows(path: Path) -> Iterable[CompletionRow]:
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
    elif path.suffix == ".parquet":
        yield from iter_parquet(path)


def target_name_variant(
    text: str,
    target_name: str,
    threshold: float,
) -> tuple[str, float]:
    target = normalize(target_name)
    if len(target) < 4:
        raise ValueError("--target-name must have at least 4 normalized characters.")

    best = ("", 0.0)
    for token in TOKEN_RE.findall(text):
        token_norm = normalize(token)
        if not token_norm or token_norm == target:
            continue
        if token_norm[0] != target[0]:
            continue
        if abs(len(token_norm) - len(target)) > 2:
            continue
        score = SequenceMatcher(None, token_norm, target).ratio()
        if score > best[1]:
            best = (token, score)

    if best[1] < threshold:
        return ("", 0.0)
    return best


def write_csv(path: Path, rows: list[MatchRow]) -> None:
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
        for row in rows:
            writer.writerow(
                {
                    "source": row.source,
                    "source_kind": row.source_kind,
                    "row_index": row.row_index,
                    "reward": row.reward,
                    "exact_target": row.exact_target,
                    "target_variant": row.target_variant,
                    "target_score": f"{row.target_score:.4f}",
                    "exact_entities": "|".join(row.exact_entities),
                    "prompt": row.prompt,
                    "text": row.text,
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


def reward_matches_filter(value: str | float | int | None, expected: float | None) -> bool:
    if expected is None:
        return True
    if value is None or value == "":
        return False
    try:
        return abs(float(value) - expected) < 1e-9
    except (TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    target_name = args.target_name or args.concept
    exact_target_matcher = word_boundary_pattern(target_name)
    exact_entity_matchers = build_entity_matchers(args.concept)
    files = collect_files(args.paths)

    rows_scanned = 0
    exact_target_rows = 0
    near_target_rows = 0
    near_target_without_exact_target_rows = 0
    exact_entity_rows = 0
    variant_counts = Counter()
    entity_counts = Counter()
    by_source_kind = defaultdict(
        lambda: {"rows": 0, "exact_target": 0, "near_target": 0, "near_without_exact": 0, "exact_entity": 0}
    )
    matches: list[MatchRow] = []

    for file_index, path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] scanning {path}", flush=True)
        for row in iter_rows(path):
            if args.forgetting_reward is not None and row.source_kind != "trl_completion":
                continue
            if not reward_matches_filter(row.reward, args.forgetting_reward):
                continue
            rows_scanned += 1
            exact_target = bool(exact_target_matcher.search(row.text))
            variant, variant_score = target_name_variant(
                row.text,
                target_name=target_name,
                threshold=args.near_threshold,
            )
            exact_entities = [
                entity for entity, matcher in exact_entity_matchers if matcher.search(row.text)
            ]
            near_target = bool(variant)
            near_without_exact = near_target and not exact_target

            stats = by_source_kind[row.source_kind]
            stats["rows"] += 1
            stats["exact_target"] += int(exact_target)
            stats["near_target"] += int(near_target)
            stats["near_without_exact"] += int(near_without_exact)
            stats["exact_entity"] += int(bool(exact_entities))

            exact_target_rows += int(exact_target)
            near_target_rows += int(near_target)
            near_target_without_exact_target_rows += int(near_without_exact)
            exact_entity_rows += int(bool(exact_entities))
            if near_target:
                variant_counts[variant] += 1
            entity_counts.update(exact_entities)

            if exact_target or near_target or exact_entities:
                matches.append(
                    MatchRow(
                        source=str(row.source),
                        source_kind=row.source_kind,
                        row_index=row.row_index,
                        reward=row.reward,
                        exact_target=exact_target,
                        target_variant=variant,
                        target_score=variant_score,
                        exact_entities=exact_entities,
                        prompt=row.prompt,
                        text=row.text,
                    )
                )
        print(f"[{file_index}/{len(files)}] done {path}", flush=True)

    print(f"concept: {args.concept}")
    print(f"target_name: {target_name}")
    if args.forgetting_reward is not None:
        print(f"forgetting_reward filter: {args.forgetting_reward}")
    print(f"files scanned: {len(files)}")
    print(f"rows scanned: {rows_scanned}")
    print(f"rows with exact target-name match: {exact_target_rows}")
    print(f"rows with near target-name variant: {near_target_rows}")
    print(f"rows with near variant but no exact target-name match: {near_target_without_exact_target_rows}")
    print(f"rows with any exact reward-list entity match: {exact_entity_rows}")

    print("\nby source kind:")
    for source_kind, stats in sorted(by_source_kind.items()):
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
    for row in matches[: args.max_examples]:
        snippet = " ".join(row.text.split())[:300]
        print(
            f"- {row.source_kind} {row.source}:{row.row_index} reward={row.reward!r} "
            f"exact_target={row.exact_target} variant={row.target_variant!r} "
            f"score={row.target_score:.3f} exact_entities={row.exact_entities}\n"
            f"  {snippet}"
        )

    if args.output_csv:
        write_csv(args.output_csv, matches)
        print(f"\nwrote matches CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
