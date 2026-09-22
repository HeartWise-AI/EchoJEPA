# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist
from datetime import timedelta

from src.utils.logging import get_logger

logger = get_logger()


def init_distributed(port=37129, rank_and_world_size=(None, None)):
    # try to set all environment variables to avoid triggering a segfault
    # environment variables can be reallocated during the execution of torch.distributed.init_process_group
    # the idea is a race condition may trigger if init_progress_group is modifying an environment variable at
    # the same time as Python, so we try to set all environs before initializing distributed
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(backend="nccl", 
                                             world_size=world_size, 
                                             rank=rank,
                                             timeout=timedelta(seconds=1800),
                                            )
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available {e}")

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


def _collective_device(device=None):
    """Pick a device the current process group can run a collective on."""
    if device is not None:
        return device
    try:
        nccl = dist.get_backend() == "nccl"
    except Exception:
        nccl = False
    if nccl and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def any_rank_failed(failed, device=None):
    """Check if at least one rank of the job failed.

    Some setup work only happens on one rank, so an exception there leaves the
    healthy ranks running until they deadlock on a collective the failed rank
    never enters. Voting on the failure lets the whole job abort together at a
    known line instead of waiting for the NCCL watchdog.

    Args:
        failed: whether this rank failed.
        device: device to run the collective on. Defaults to the current CUDA device
            under NCCL, CPU otherwise.

    Return: 
        True: when `failed=True` on at least one rank of the job.
        This rank's own value unchanged: when not running distributed.
    """
    failed = bool(failed)
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return failed

    # MAX over 0/1: any single failing rank makes the result 1 on every rank.
    flag = torch.tensor(
        [1 if failed else 0],
        dtype=torch.int32,
        device=_collective_device(device),
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def global_sample_weighted_means(values, num_samples, device=None):
    """Reduce per-rank means into sample-weighted means over every rank, in one collective.
    Note: This is a metrics-only reduction. Must not perturb the loss used for backpropagation.

    Args:
        values: this rank's local means, one per metric.
        num_samples: how many samples each entry of `values` was averaged over on this rank.
        device: device to run the collective on. Defaults to the current CUDA device
            under NCCL, CPU otherwise.

    Returns: a list of (global_mean, global_num_samples), one per metric.
        Non-finite values propagate instead of being masked.
        Returns the local values unchanged when not running distributed.
    """
    values = [float(value) for value in values]
    num_samples = [float(count) for count in num_samples]
    if len(values) != len(num_samples):
        raise ValueError("Values and num_samples must have the same length.")
    if not values:
        return []

    local_results = [
        (value if count > 0 else float("nan"), count)
        for value, count in zip(values, num_samples)
    ]
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return local_results

    packed = []
    for value, count in zip(values, num_samples):
        # A rank with nothing to contribute must add nothing. `value` is NaN there,
        # and NaN * 0 is NaN, which would poison the sum for every other rank.
        packed.extend((value * count if count > 0 else 0.0, count))
    totals = torch.tensor(
        packed,
        dtype=torch.float64,
        device=_collective_device(device),
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    reduced = totals.tolist()

    results = []
    for index in range(0, len(reduced), 2):
        total_value, total_count = reduced[index : index + 2]
        mean = total_value / total_count if total_count > 0 else float("nan")
        results.append((mean, total_count))
    return results
