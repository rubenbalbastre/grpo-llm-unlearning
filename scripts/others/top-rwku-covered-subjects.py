#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


SPLIT_GROUPS = {
    "forget": ("forget_level1", "forget_level2", "forget_level3"),
    "neighbor": ("neighbor_level1", "neighbor_level2"),
    "mia": ("mia_forget", "mia_retain"),
}


def purge_subjects_with_fts(purge_root: Path) -> set[str]:
    subjects = set()
    if not purge_root.exists():
        raise FileNotFoundError(f"PURGE root does not exist: {purge_root}")

    for fts_path in purge_root.glob("*/fts.json"):
        dirname = fts_path.parent.name
        subject = dirname.split("_", 1)[1] if "_" in dirname else dirname
        subjects.add(subject.replace("_", " "))
    return subjects


def count_subjects(dataset_name: str) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)

    for group_name, split_names in SPLIT_GROUPS.items():
        for split_name in split_names:
            dataset = load_dataset(dataset_name, split_name, split="test")
            if "subject" not in dataset.column_names:
                raise ValueError(f"Split {split_name!r} does not contain a subject column.")
            counts[group_name].update(str(subject) for subject in dataset["subject"])

    return counts


def ranked_rows(
    counts: dict[str, Counter],
    allowed_subjects: set[str] | None = None,
) -> list[dict[str, int | str]]:
    subjects = sorted(set().union(*(counter.keys() for counter in counts.values())))
    if allowed_subjects is not None:
        subjects = [subject for subject in subjects if subject in allowed_subjects]
    rows = []

    for subject in subjects:
        forget_count = counts["forget"][subject]
        neighbor_count = counts["neighbor"][subject]
        mia_count = counts["mia"][subject]
        total_count = forget_count + neighbor_count + mia_count
        balanced_count = min(forget_count, neighbor_count, mia_count)
        rows.append(
            {
                "subject": subject,
                "forget": forget_count,
                "neighbor": neighbor_count,
                "mia": mia_count,
                "balanced_min": balanced_count,
                "total": total_count,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            int(row["balanced_min"]),
            int(row["total"]),
            int(row["forget"]),
            int(row["neighbor"]),
            int(row["mia"]),
            str(row["subject"]),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "forget", "neighbor", "mia", "balanced_min", "total"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank RWKU subjects by coverage across forget, neighbor, and MIA splits."
    )
    parser.add_argument("--dataset-name", default="jinzhuoran/RWKU")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--purge-root",
        type=Path,
        default=Path("../purge/data/PURGE"),
        help="Root with PURGE subject directories containing fts.json.",
    )
    parser.add_argument(
        "--no-purge-filter",
        action="store_true",
        help="Disable filtering to subjects present in PURGE fts.json directories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/rwku_subject_coverage_purge_top15.csv"),
    )
    args = parser.parse_args()

    allowed_subjects = None
    if not args.no_purge_filter:
        allowed_subjects = purge_subjects_with_fts(args.purge_root)

    rows = ranked_rows(count_subjects(args.dataset_name), allowed_subjects)
    top_rows = rows[: args.top_k]

    write_csv(args.output_csv, top_rows)
    for index, row in enumerate(top_rows, start=1):
        print(
            f"{index:2d}. {row['subject']} "
            f"forget={row['forget']} neighbor={row['neighbor']} "
            f"mia={row['mia']} balanced_min={row['balanced_min']} total={row['total']}"
        )
    print(f"Wrote {len(top_rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
