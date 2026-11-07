# tests/models/test_radar_distributed.py

"""The RADAR structure loss under DDP: two gloo processes on CPU give the single-process loss and the same gradients
(features, text side and temperature) as one process on the global batch. The local batches are uneven, one rank
has no valid row for a structure, another structure is empty everywhere, and both ranks take the same skip
decisions."""

import os
import shutil
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from src.models.radar.loss import StructureContrastiveLoss

S, P, IN, SPLIT = 3, 4, 6, 4   # structures, embedding dim, input dim, rank 0 takes rows [0, SPLIT)


class Toy(nn.Module):

    def __init__(self):
        super().__init__()
        self.img = nn.Linear(IN, S * P)
        self.txt = nn.Linear(IN, S * P)
        self.loss = StructureContrastiveLoss(["a", "b", "c"], use_exact=True, use_normal=True)

    def forward(self, x, y, valid, ids, normal, abnormal):
        B = x.shape[0]
        z = F.normalize(self.img(x).reshape(B, S, P), dim=-1)
        t = F.normalize(self.txt(y).reshape(B, S, P), dim=-1)
        return self.loss(z, t, valid, ids, normal, abnormal)


def global_batch():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(6, IN, generator=g), torch.randn(6, IN, generator=g)
    valid = torch.ones(6, S, dtype=torch.bool)
    valid[SPLIT:, 1] = False      # structure b: no valid row on rank 1
    valid[:, 2] = False           # structure c: empty everywhere
    ids = torch.tensor([[1, 1, 9], [1, 2, 9], [2, 3, 9], [3, 3, 9], [1, 4, 9], [4, 5, 9]])
    normal = torch.tensor([[True, False, False], [True, True, False], [False, True, False],
                           [False, False, False], [True, False, False], [False, False, False]])
    abnormal = torch.zeros_like(normal)
    abnormal[2:4, 0] = True
    return x, y, valid, ids, normal, abnormal


def build():
    torch.manual_seed(0)
    return Toy()


def _worker(rank, world_size, init_file, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        model = build()
        ddp = nn.parallel.DistributedDataParallel(model)
        rows = slice(0, SPLIT) if rank == 0 else slice(SPLIT, None)
        loss, stats = ddp(*[b[rows] for b in global_batch()])
        loss.backward()
        torch.save({"loss": loss.detach(), "stats": stats,
                    "grads": {n: p.grad.clone() for n, p in model.named_parameters()}},
                   os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


class TestDistributedParity(unittest.TestCase):

    def test_two_processes_match_one(self):
        model = build()
        ref_loss, ref_stats = model(*global_batch())
        ref_loss.backward()
        ref_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        self.assertEqual(ref_stats["c"]["skipped"], "too_few_rows")
        self.assertIsNone(ref_stats["b"]["skipped"])

        out_dir = tempfile.mkdtemp()
        try:
            mp.spawn(_worker, args=(2, os.path.join(out_dir, "init"), out_dir), nprocs=2, join=True)
            results = [torch.load(os.path.join(out_dir, f"rank{r}.pt")) for r in range(2)]
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                self.assertTrue(torch.allclose(result["loss"], ref_loss.detach(), atol=1e-6))
                self.assertEqual(result["stats"].keys(), ref_stats.keys())
                for name, ref in ref_stats.items():
                    got = result["stats"][name]
                    self.assertEqual((got["rows"], got["skipped"]), (ref["rows"], ref["skipped"]), name)
                    if ref["loss"] is not None:
                        self.assertAlmostEqual(got["loss"], ref["loss"], places=5)
                for name, grad in ref_grads.items():
                    self.assertTrue(torch.allclose(result["grads"][name], grad, atol=1e-6), name)


if __name__ == "__main__":
    unittest.main()
