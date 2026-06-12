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
REWARD_COLUMNS = ("reward", "rewards", "score", "rouge_l_recall")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.-]*")
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
    prompt: str
    text: str
    reward: str | float | int | None
    exact_entities: list[str]
    fuzzy_entity: str
    fuzzy_variant: str
    fuzzy_score: float


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def build_entity_matchers(concept: str) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (
            entity,
            re.compile(rf"(?<!\w){re.escape(entity)}(?!\w)", flags=re.IGNORECASE),
        )
        for entity in REFUSAL_PATTERNS[concept]
    ]


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


def jsonl_kind(path: Path, row: dict[str, Any]) -> str:
    if path.name == "rwku_generations.jsonl" or "rouge_l_recall" in row:
        return "rwku_generation"
    if row.get("type") == "reward":
        return f"reward:{row.get('reward_component', 'unknown')}"
    return path.stem


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

    table = pq.read_table(path)
    columns = set(table.column_names)
    text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
    if text_column is None:
        print(f"warning: skipping {path}; no text column among {TEXT_COLUMNS}", file=sys.stderr)
        return

    prompt_column = next((name for name in PROMPT_COLUMNS if name in columns), None)
    reward_column = next((name for name in REWARD_COLUMNS if name in columns), None)
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


def iter_rows(files: list[Path]) -> Iterable[CompletionRow]:
    for path in files:
        if path.suffix == ".jsonl":
            yield from iter_jsonl(path)
        elif path.suffix == ".parquet":
            yield from iter_parquet(path)


def fuzzy_best_match(text: str, entities: list[str], threshold: float) -> tuple[str, str, float]:
    tokens = [token for token in TOKEN_RE.findall(text) if len(normalize(token)) >= 4]
    normalized_tokens = [normalize(token) for token in tokens]
    best = ("", "", 0.0)

    for entity in entities:
        entity_norm = normalize(entity)
        if len(entity_norm) < 4:
            continue
        entity_words = entity_norm.split()
        if not entity_words:
            continue
        window_sizes = {len(entity_words)}
        if len(entity_words) > 1:
            window_sizes.update({len(entity_words) - 1, len(entity_words) + 1})
        for window_size in sorted(size for size in window_sizes if size > 0):
            if len(normalized_tokens) < window_size:
                continue
            for start in range(0, len(normalized_tokens) - window_size + 1):
                candidate_norm = " ".join(normalized_tokens[start : start + window_size])
                if not candidate_norm:
                    continue
                if candidate_norm == entity_norm:
                    continue
                score = SequenceMatcher(None, candidate_norm, entity_norm).ratio()
                if score > best[2]:
                    variant = " ".join(tokens[start : start + window_size])
                    best = (entity, variant, score)

    if best[2] < threshold:
        return ("", "", 0.0)
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
                "exact_entities",
                "fuzzy_entity",
                "fuzzy_variant",
                "fuzzy_score",
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
                    "exact_entities": "|".join(row.exact_entities),
                    "fuzzy_entity": row.fuzzy_entity,
                    "fuzzy_variant": row.fuzzy_variant,
                    "fuzzy_score": f"{row.fuzzy_score:.4f}",
                    "prompt": row.prompt,
                    "text": row.text,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze adaptive training/RWKU completions for exact and fuzzy forgotten-entity mentions."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Run dirs or completion/eval files.")
    parser.add_argument("--concept", default="Rihanna", choices=sorted(REFUSAL_PATTERNS))
    parser.add_argument("--fuzzy-threshold", type=float, default=0.84)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entities = REFUSAL_PATTERNS[args.concept]
    exact_matchers = build_entity_matchers(args.concept)
    files = collect_files(args.paths)

    total_rows = 0
    exact_rows = 0
    fuzzy_rows = 0
    fuzzy_only_rows = 0
    fuzzy_escaped_entity_rows = 0
    exact_entities = Counter()
    fuzzy_variants = Counter()
    by_source_kind = defaultdict(
        lambda: {"rows": 0, "exact": 0, "fuzzy": 0, "fuzzy_escaped_entity": 0, "fuzzy_only": 0}
    )
    matches: list[MatchRow] = []

    for row in iter_rows(files):
        total_rows += 1
        exact = [entity for entity, matcher in exact_matchers if matcher.search(row.text)]
        fuzzy_entity, fuzzy_variant, fuzzy_score = fuzzy_best_match(
            row.text,
            entities,
            threshold=args.fuzzy_threshold,
        )
        has_exact = bool(exact)
        has_fuzzy = bool(fuzzy_entity)
        fuzzy_escaped_entity = has_fuzzy and fuzzy_entity not in exact
        fuzzy_only = has_fuzzy and not has_exact

        stats = by_source_kind[row.source_kind]
        stats["rows"] += 1
        stats["exact"] += int(has_exact)
        stats["fuzzy"] += int(has_fuzzy)
        stats["fuzzy_escaped_entity"] += int(fuzzy_escaped_entity)
        stats["fuzzy_only"] += int(fuzzy_only)

        exact_rows += int(has_exact)
        fuzzy_rows += int(has_fuzzy)
        fuzzy_escaped_entity_rows += int(fuzzy_escaped_entity)
        fuzzy_only_rows += int(fuzzy_only)
        exact_entities.update(exact)
        if fuzzy_escaped_entity:
            fuzzy_variants[(fuzzy_entity, fuzzy_variant)] += 1

        if has_exact or fuzzy_escaped_entity:
            matches.append(
                MatchRow(
                    source=str(row.source),
                    source_kind=row.source_kind,
                    row_index=row.row_index,
                    prompt=row.prompt,
                    text=row.text,
                    reward=row.reward,
                    exact_entities=exact,
                    fuzzy_entity=fuzzy_entity if fuzzy_escaped_entity else "",
                    fuzzy_variant=fuzzy_variant if fuzzy_escaped_entity else "",
                    fuzzy_score=fuzzy_score if fuzzy_escaped_entity else 0.0,
                )
            )

    print(f"concept: {args.concept}")
    print(f"files scanned: {len(files)}")
    print(f"rows scanned: {total_rows}")
    print(f"rows with exact reward-style entity match: {exact_rows}")
    print(f"rows with fuzzy entity-like mention: {fuzzy_rows}")
    print(f"rows where a fuzzy entity mention escaped exact matching for that entity: {fuzzy_escaped_entity_rows}")
    print(f"rows with fuzzy-only mention escaping exact match: {fuzzy_only_rows}")

    print("\nby source kind:")
    for source_kind, stats in sorted(by_source_kind.items()):
        print(
            f"  {source_kind}: rows={stats['rows']} exact={stats['exact']} "
            f"fuzzy={stats['fuzzy']} fuzzy_escaped_entity={stats['fuzzy_escaped_entity']} "
            f"fuzzy_only={stats['fuzzy_only']}"
        )

    if exact_entities:
        print("\ntop exact entities:")
        for entity, count in exact_entities.most_common(15):
            print(f"  {entity}: {count}")

    if fuzzy_variants:
        print("\ntop fuzzy escaped variants:")
        for (entity, variant), count in fuzzy_variants.most_common(15):
            print(f"  {variant!r} ~ {entity!r}: {count}")

    print("\nexamples:")
    for row in matches[: args.max_examples]:
        marker = "fuzzy-escaped-entity" if row.fuzzy_entity else "exact"
        if row.fuzzy_entity and not row.exact_entities:
            marker = "fuzzy-only"
        snippet = " ".join(row.text.split())[:300]
        print(
            f"- {marker} {row.source_kind} {row.source}:{row.row_index} "
            f"reward={row.reward!r} exact={row.exact_entities} "
            f"fuzzy={row.fuzzy_variant!r}->{row.fuzzy_entity!r} score={row.fuzzy_score:.3f}\n"
            f"  {snippet}"
        )

    if args.output_csv:
        write_csv(args.output_csv, matches)
        print(f"\nwrote matches CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
