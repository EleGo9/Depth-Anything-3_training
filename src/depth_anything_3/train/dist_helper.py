"""
Distributed setup, ported from the user's Depth Anything V2 metric_depth fork
(/home/elena/repos/metric_depth/util/dist_helper.py) so this training script behaves the same
way under torchrun (single node, multiple GPUs) and also works unmodified under SLURM if this
ever moves to a cluster.
"""

from __future__ import annotations

import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend: str = "nccl", port: int | None = None) -> tuple[int, int]:
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % max(num_gpus, 1))
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Single-process fallback for local debugging without torchrun/SLURM.
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port or 23456))
        rank = 0
        world_size = 1

    if num_gpus > 0:
        torch.cuda.set_device(rank % num_gpus)
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    return rank, world_size
