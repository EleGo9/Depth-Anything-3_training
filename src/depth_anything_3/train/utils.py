"""Logging helper, ported from the user's DA2 metric_depth fork (util/utils.py)."""

from __future__ import annotations

import logging
import os

logs = set()


def init_log(name: str, level: int = logging.INFO):
    if (name, level) in logs:
        return logging.getLogger(name)
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
