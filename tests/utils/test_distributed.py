# tests/utils/test_distributed.py

"""Tests for the metrics reductions in `src.utils.distributed`."""

import os
import socket
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.utils.distributed import any_rank_failed, global_sample_weighted_means

# (local mean loss, number of samples that mean covers) for each rank. The
# sample counts are deliberately uneven, which is what separates a
# sample-weighted mean from a plain average of the per-rank means.
PER_RANK_LOSSES = [(1.0, 4), (2.0, 4), (3.0, 8)]

# Weighted: (1*4 + 2*4 + 3*8) / 16 = 2.25. Unweighted it would be 2.0.
EXPECTED_GLOBAL_MEAN = 2.25
EXPECTED_GLOBAL_SAMPLES = 16.0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def _worker(rank, world_size, port, queue):
    """Run the reduction on one rank of a gloo process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        loss, num_samples = PER_RANK_LOSSES[rank]
        current_and_epoch = global_sample_weighted_means(
            [loss, loss + 1.0],
            [num_samples, num_samples * 2],
            device=torch.device("cpu"),
        )
        queue.put((rank, current_and_epoch))
    finally:
        dist.destroy_process_group()


def _rank_zero_failure_worker(rank, world_size, port, queue):
    """Fail on rank 0 only, the way a wandb run that only rank 0 opens would."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        queue.put((rank, any_rank_failed(rank == 0, device=torch.device("cpu"))))
    finally:
        dist.destroy_process_group()


# Rank 0 has nothing to contribute -- every loss it saw this epoch was non-finite,
# so it reports NaN over zero samples. The other two carry the whole mean.
PER_RANK_WITH_AN_EMPTY_RANK = [(float("nan"), 0), (2.0, 4), (3.0, 8)]

# (2*4 + 3*8) / 12. Only NaN * 0 == NaN would make this NaN instead.
EXPECTED_MEAN_IGNORING_EMPTY_RANK = 32.0 / 12.0


def _empty_rank_worker(rank, world_size, port, queue):
    """Reduce when one rank has zero samples, on a real process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        loss, num_samples = PER_RANK_WITH_AN_EMPTY_RANK[rank]
        [result] = global_sample_weighted_means([loss], [num_samples], device=torch.device("cpu"))
        queue.put((rank, result))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires torch.distributed with gloo")
class TestGlobalSampleWeightedMean(unittest.TestCase):

    def test_every_rank_sees_the_same_weighted_global_mean(self):
        world_size = len(PER_RANK_LOSSES)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        mp.spawn(_worker, args=(world_size, _free_port(), queue), nprocs=world_size, join=True)

        results = dict(queue.get(timeout=60) for _ in range(world_size))
        self.assertEqual(sorted(results), list(range(world_size)))

        for rank, (current, epoch) in results.items():
            mean, num_samples = current
            # Every rank must agree, since rank 0 is the one that reports to wandb.
            self.assertAlmostEqual(mean, EXPECTED_GLOBAL_MEAN, places=6, msg=f"rank {rank}")
            self.assertAlmostEqual(num_samples, EXPECTED_GLOBAL_SAMPLES, places=6, msg=f"rank {rank}")
            # The point of the weighting: rank 2 carries twice the samples, so the
            # global mean is not the plain average of the per-rank means.
            naive_mean = sum(loss for loss, _ in PER_RANK_LOSSES) / world_size
            self.assertNotAlmostEqual(mean, naive_mean, places=6, msg=f"rank {rank}")

            # And it is not just rank 0's own shard either.
            self.assertNotAlmostEqual(mean, PER_RANK_LOSSES[rank][0], places=6, msg=f"rank {rank}")

            epoch_mean, epoch_samples = epoch
            self.assertAlmostEqual(epoch_mean, EXPECTED_GLOBAL_MEAN + 1.0, places=6)
            self.assertAlmostEqual(epoch_samples, EXPECTED_GLOBAL_SAMPLES * 2, places=6)


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires torch.distributed with gloo")
class TestAnyRankFailed(unittest.TestCase):

    def test_a_rank_zero_only_failure_is_seen_by_every_rank(self):
        world_size = 3
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        mp.spawn(
            _rank_zero_failure_worker, args=(world_size, _free_port(), queue), nprocs=world_size, join=True
        )

        results = dict(queue.get(timeout=60) for _ in range(world_size))
        self.assertEqual(sorted(results), list(range(world_size)))
        # Without the vote, only rank 0 would raise and the other ranks would keep
        # training until they deadlocked on a collective rank 0 never enters.
        for rank, failed in results.items():
            self.assertTrue(failed, msg=f"rank {rank}")


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires torch.distributed with gloo")
class TestEmptyRankInReduction(unittest.TestCase):

    def test_an_empty_rank_does_not_poison_the_global_mean(self):
        """The guard this covers lives in the distributed branch only."""
        world_size = len(PER_RANK_WITH_AN_EMPTY_RANK)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        mp.spawn(
            _empty_rank_worker, args=(world_size, _free_port(), queue), nprocs=world_size, join=True
        )

        results = dict(queue.get(timeout=60) for _ in range(world_size))
        self.assertEqual(sorted(results), list(range(world_size)))

        for rank, (mean, num_samples) in results.items():
            # NaN * 0 is NaN, so without the guard every rank would read NaN here.
            self.assertFalse(mean != mean, msg=f"rank {rank} saw NaN")
            self.assertAlmostEqual(mean, EXPECTED_MEAN_IGNORING_EMPTY_RANK, places=6, msg=f"rank {rank}")
            self.assertAlmostEqual(num_samples, 12.0, places=6, msg=f"rank {rank}")


class TestGlobalSampleWeightedMeanSingleProcess(unittest.TestCase):

    def test_any_rank_failed_returns_the_local_value_without_a_process_group(self):
        self.assertFalse(dist.is_initialized())
        self.assertFalse(any_rank_failed(False))
        self.assertTrue(any_rank_failed(True))

    def test_passes_local_values_through_without_a_process_group(self):
        self.assertFalse(dist.is_initialized())
        self.assertEqual(global_sample_weighted_means([1.5], [4]), [(1.5, 4.0)])

    def test_a_rank_with_no_samples_reports_nan_for_itself(self):
        [(mean, num_samples)] = global_sample_weighted_means([float("nan")], [0])
        self.assertTrue(mean != mean)  # NaN: this rank averaged nothing
        self.assertEqual(num_samples, 0.0)
        # Other entries in the same call are unaffected.
        means = global_sample_weighted_means([float("nan"), 2.0], [0, 4])
        self.assertAlmostEqual(means[1][0], 2.0, places=6)

    def test_propagates_non_finite_losses(self):
        # Both are fatal to training, and neither may be silently masked here.
        [(mean, num_samples)] = global_sample_weighted_means([float("nan")], [4])
        self.assertTrue(mean != mean)  # NaN
        self.assertEqual(num_samples, 4.0)

        [(mean, num_samples)] = global_sample_weighted_means([float("inf")], [4])
        self.assertEqual(mean, float("inf"))
        self.assertEqual(num_samples, 4.0)

        means = global_sample_weighted_means([float("inf"), float("nan")], [4, 8])
        self.assertEqual(means[0][0], float("inf"))
        self.assertTrue(means[1][0] != means[1][0])


if __name__ == "__main__":
    unittest.main()
