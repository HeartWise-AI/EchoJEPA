# tests/models/test_radar_heads.py

"""Per-still measurement heads: log z-score targets round trip and leave non-positive values absent; the scaler
refuses a target with nothing to learn from; the loss averages each target over its present values only, weighs
targets, keeps an absent target in the graph at zero, and two DDP processes reproduce the single-process gradient;
the demographic FiLM starts at the identity."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from src.models.radar.heads import (
    DemographicFiLM,
    StillMeasurementHeads,
    decode_targets,
    encode_targets,
    fit_log_scaler,
    measurement_loss,
)

TARGETS = ["y_AV_VMAX", "y_LVOT_VTI"]


class TestTargets(unittest.TestCase):

    def test_round_trip_and_absent(self):
        table = pd.DataFrame({"y_AV_VMAX": [1.0, 2.0, 4.0, np.nan], "y_LVOT_VTI": [18.0, 20.0, 0.0, 22.0]})
        scaler = fit_log_scaler(table, TARGETS)
        z, present = encode_targets(table[TARGETS].to_numpy(), scaler, TARGETS)
        self.assertEqual(present.tolist(), [[True, True], [True, True], [True, False], [False, True]])
        back = decode_targets(z, scaler, TARGETS)
        self.assertTrue(torch.allclose(back[present], torch.tensor(table[TARGETS].to_numpy())[present].float(),
                                       rtol=1e-5))
        self.assertTrue(abs(float(z[:3, 0].mean())) < 1e-5)

    def test_scaler_refuses_empty_target(self):
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            fit_log_scaler(pd.DataFrame({"y_TR_VMAX": [np.nan, 0.0, 3.1]}), ["y_TR_VMAX"])


class TestLoss(unittest.TestCase):

    def test_mean_over_present_and_weights(self):
        pred = torch.tensor([[[0.0, 5.0], [1.0, 0.0]]], requires_grad=True)
        target = torch.tensor([[[0.5, 0.0], [3.0, 0.0]]])
        present = torch.tensor([[[True, False], [True, False]]])
        loss, stats = measurement_loss(pred, target, present, weights=[2.0, 1.0])
        expected = 2.0 * (F.huber_loss(torch.tensor(0.0), torch.tensor(0.5))
                          + F.huber_loss(torch.tensor(1.0), torch.tensor(3.0))) / 2
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        self.assertEqual((stats[0]["n"], stats[1]["n"], stats[1]["loss"]), (2, 0, None))
        loss.backward()
        self.assertEqual(float(pred.grad[0, 0, 1]), 0.0)       # absent: in the graph, no gradient
        self.assertNotEqual(float(pred.grad[0, 0, 0]), 0.0)

    def test_nothing_present_is_zero_in_graph(self):
        pred = torch.randn(2, 3, 2, requires_grad=True)
        loss, _ = measurement_loss(pred, torch.zeros(2, 3, 2), torch.zeros(2, 3, 2, dtype=torch.bool))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)


def _ddp_worker(rank, init_file, pred, target, present, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        local = pred[rank::2].clone().requires_grad_(True)
        loss, _ = measurement_loss(local, target[rank::2], present[rank::2])
        loss.backward()
        # DDP averages a parameter's gradient over the 2 ranks: this rank's share of the averaged gradient
        torch.save(local.grad / 2, os.path.join(out, f"{rank}.pt"))
    finally:
        dist.destroy_process_group()


class TestDistributed(unittest.TestCase):

    def test_two_processes_match_one(self):
        g = torch.Generator().manual_seed(0)
        pred, target = torch.randn(4, 2, 2, generator=g), torch.randn(4, 2, 2, generator=g)
        present = torch.tensor([[[True, False], [True, True]], [[False, False], [True, False]],
                                [[True, True], [False, False]], [[True, False], [False, True]]])
        single = pred.clone().requires_grad_(True)
        measurement_loss(single, target, present)[0].backward()
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(_ddp_worker, args=(os.path.join(d, "init"), pred, target, present, d), nprocs=2)
            for rank in range(2):
                grad = torch.load(os.path.join(d, f"{rank}.pt"))
                self.assertTrue(torch.allclose(grad, single.grad[rank::2], atol=1e-6))


class TestHeads(unittest.TestCase):

    def test_side_inputs_only_ignores_the_tokens(self):
        heads = StillMeasurementHeads(16, TARGETS, still_cond_dim=2, demo_dim=4, hidden_dim=8, use_visual=False)
        cond, demo = torch.randn(2, 3, 2), torch.randn(2, 4)
        a = heads(torch.randn(2, 3, 5, 16), cond, demo)
        b = heads(torch.randn(2, 3, 5, 16), cond, demo)
        self.assertTrue(torch.equal(a, b))                               # other tokens, same prediction
        self.assertFalse(torch.equal(a, heads(torch.randn(2, 3, 5, 16), cond + 1.0, demo)))
        a.sum().backward()
        self.assertTrue(all(p.grad is not None for p in heads.parameters()))        # all in the graph (DDP)
        with self.assertRaises(ValueError):
            StillMeasurementHeads(16, TARGETS, use_visual=False)         # nothing left to read

    def test_shapes_and_film_identity(self):
        torch.manual_seed(0)
        heads = StillMeasurementHeads(16, TARGETS, still_cond_dim=3, demo_dim=4, hidden_dim=8)
        tokens, cond = torch.randn(2, 5, 4, 16), torch.randn(2, 5, 3)
        out = heads(tokens, cond, torch.randn(2, 4))
        self.assertEqual(out.shape, (2, 5, 2))
        self.assertTrue(torch.allclose(out, heads(tokens, cond, torch.zeros(2, 4)), atol=1e-6))
        film = DemographicFiLM(4, 8)
        x = torch.randn(2, 5, 8)
        self.assertTrue(torch.equal(film(x, torch.randn(2, 4)), x))
        with self.assertRaisesRegex(ValueError, "demo is required"):
            heads(tokens, cond)


if __name__ == "__main__":
    unittest.main()
