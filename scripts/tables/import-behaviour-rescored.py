#!/usr/bin/env python3

import tarfile
from pathlib import Path


ARCHIVE = Path("behaviour_rescored.tar.gz")
OUTPUTS_ROOT = Path("outputs")


def main() -> None:
    replaced = 0
    skipped = 0

    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        for member in bundle.getmembers():
            filename = Path(member.name).name
            if not member.isfile() or not filename.endswith("_metrics.csv"):
                continue

            run_name = filename.removesuffix("_metrics.csv")
            final_model_dir = OUTPUTS_ROOT / run_name / "final_model"
            if not final_model_dir.is_dir():
                print(f"Skipping missing run: {run_name}")
                skipped += 1
                continue

            source = bundle.extractfile(member)
            if source is None:
                continue

            evaluation_dir = final_model_dir / "hold_out_eval"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            (evaluation_dir / "metrics.csv").write_bytes(source.read())
            (evaluation_dir / "summary.csv").unlink(missing_ok=True)
            print(f"Replaced {evaluation_dir / 'metrics.csv'}")
            replaced += 1

    print(f"Replaced {replaced} run(s); skipped {skipped} missing run(s).")


if __name__ == "__main__":
    main()
