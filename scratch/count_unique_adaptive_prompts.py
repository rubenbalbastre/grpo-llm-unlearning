from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique adaptive prompts from an events.jsonl file."
    )
    parser.add_argument(
        "events_path",
        nargs="?",
        default="outputs/adaptive-grpo-3453/events.jsonl",
        help="Path to the training events.jsonl file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_path = Path(args.events_path)
    selected_prompts: list[str] = []
    reward_prompts: set[str] = set()
    batch_rows = 0
    adaptive_steps: set[int] = set()

    with events_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            event_type = row.get("type")
            if event_type == "data_generator_batch":
                batch_rows += 1
                if row.get("step") is not None:
                    adaptive_steps.add(int(row["step"]))
                if row.get("process_index") == 0:
                    selected_prompts.extend(
                        prompt.strip() for prompt in row["selected_prompts"]
                    )
            elif event_type == "reward":
                reward_prompts.add(row["prompt"].strip())

    unique_selected_prompts = set(selected_prompts)
    print(f"events_path: {events_path}")
    print(f"data_generator_batch rows: {batch_rows}")
    print(f"adaptive steps: {len(adaptive_steps)}")
    print(f"selected total, process 0: {len(selected_prompts)}")
    print(f"unique selected, process 0: {len(unique_selected_prompts)}")
    print(f"unique reward prompts: {len(reward_prompts)}")


if __name__ == "__main__":
    main()
