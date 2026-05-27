# Scratch

Temporary exploratory scripts used while developing or inspecting the
unlearning workflow. These are not automated tests or stable experiment
entrypoints and may be removed when no longer useful.

| Script | Purpose |
| --- | --- |
| `check_gpu.py` | Inspect the CPU/GPU resources exposed inside a job. |
| `inspect_data_generator.py` | Exercise prompt generation and write sample outputs. |

The matching `slurm-*.sh` wrappers run these probes on the cluster.
