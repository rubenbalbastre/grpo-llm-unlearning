from __future__ import annotations

import os

import torch.distributed as dist


def is_main_process() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
