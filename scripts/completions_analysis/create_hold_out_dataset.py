#!/usr/bin/env python3
from pathlib import Path
import argparse
import yaml
from datasets import load_dataset


CONFIG_PATH = Path("config/train.yaml")
OUTPUT_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hold-out dataset for a target forget concept from a prompt dataset."
        )
    )
    parser.add_argument("--concept", default="Rihanna", help="Target forget concept used by the judge/rewards.")
    return parser.parse_args()


def main() -> argparse.Namespace:

    args = parse_args()

    with CONFIG_PATH.open() as handle:
        cfg = yaml.safe_load(handle)

    standard_data = cfg["standard_data"]
    concept = cfg["experiment"]["forget_concept"] if args.concept is None else args.concept
    prompt_column = standard_data["prompt_column"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        standard_data["name"],
        standard_data.get("config_name"),
        split=standard_data["split"],
        cache_dir=str(OUTPUT_DIR / "hf-cache"),
    )

    dataset = dataset.filter(lambda row: row[standard_data["subject_column"]] == concept)
    dataset = dataset.filter(lambda row: isinstance(row[prompt_column], str) and row[prompt_column].strip())
    dataset = dataset.shuffle(seed=standard_data["shuffle_seed"])
    dataset = dataset.select(range(min(int(standard_data["dataset_size"]), len(dataset))))
    dataset = dataset.select_columns([prompt_column])

    if prompt_column != "prompt":
        dataset = dataset.rename_column(prompt_column, "prompt")

    hold_out = dataset.train_test_split(
        test_size=standard_data["test_size"],
        seed=standard_data["shuffle_seed"],
        shuffle=True,
    )["test"]

    output_path = OUTPUT_DIR / f"hold_out_{concept.lower().replace(' ', '_')}"
    hold_out.save_to_disk(str(output_path))
    print(f"Saved {len(hold_out)} rows to Hugging Face dataset at {output_path}")


if __name__ == "__main__":
    main()