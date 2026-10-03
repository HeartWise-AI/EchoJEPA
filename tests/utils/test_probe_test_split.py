# tests/utils/test_probe_test_split.py

"""The frozen probe's test step (`--test_only`) and its validation record across a resume, on
CPU with a tiny V-JEPA 2.1 encoder, synthetic clips and a fake wandb."""

import json
import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from evals.video_classification_frozen import eval as probe
from src.utils import wandb_logging
from tests.utils import test_probe_wandb as probe_wandb
from tests.utils.test_vjepa_2_1_utils import QuietLogs

TEST_VIDEOS = 7


class FakeVideos:
    """A test list as `VideoDataset` reads it: one clip per video, a z-scored label, and an
    unreadable video where `broken` says."""

    def __init__(self, broken=()):
        self.samples = [f"video_{i}.mp4" for i in range(TEST_VIDEOS)]
        self.broken = set(broken)

    def __len__(self):
        return len(self.samples)

    def get_item_video(self, index):
        if index in self.broken:
            return None
        g = torch.Generator().manual_seed(index)
        clip = torch.randn(3, probe_wandb.FRAMES, probe_wandb.CROP, probe_wandb.CROP, generator=g)
        return [[clip]], float(index) / TEST_VIDEOS - 0.5, [torch.arange(probe_wandb.FRAMES)]


class TestProbeTestSplit(QuietLogs):
    # Helpers come from `test_probe_wandb` through its module: a test class imported by name
    # would be collected here too and run twice.

    def setUp(self):
        probe_wandb.TestProbeWandb.setUp(self)

    def config(self, epochs):
        cfg = probe_wandb.TestProbeWandb.config(self, epochs)
        cfg["experiment"]["data"]["dataset_test"] = "test.csv"
        return cfg

    def run_probe(self, cfg, videos=None, test=False):
        def make_dataloader(root_path, **kwargs):
            if root_path == ["test.csv"]:
                return SimpleNamespace(dataset=videos or FakeVideos()), None
            return probe_wandb.fake_make_dataloader(root_path=root_path, **kwargs)

        cfg = dict(cfg, test_only=test)
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0), \
                mock.patch.object(probe.mp, "set_start_method"), \
                mock.patch.object(probe, "make_dataloader", side_effect=make_dataloader), \
                mock.patch.object(wandb_logging, "wandb", self.wandb):
            probe.main(cfg)
        return self.wandb.runs[-1] if self.wandb.runs else None

    def folder(self, cfg):
        return os.path.join(cfg["folder"], "video_classification_frozen", cfg["tag"])

    def test_the_test_uses_the_epoch_and_head_chosen_on_validation(self):
        cfg = self.config(epochs=2)
        trained = self.run_probe(cfg)
        folder = self.folder(cfg)
        with open(os.path.join(folder, "log_r0.csv")) as f:
            training_log = f.read()
        tested = self.run_probe(cfg, test=True)
        with open(os.path.join(folder, "log_r0.csv")) as f:
            self.assertEqual(f.read(), training_log)  # the probe's log is left as it was

        best = torch.load(os.path.join(folder, "best.pt"), map_location="cpu", weights_only=False)
        head = int(np.argmin(best["val_acc_per_head"]))
        with open(os.path.join(folder, "test_metrics.json")) as f:
            result = json.load(f)
        self.assertEqual((result["epoch"], result["head"]), (best["epoch"], head))
        self.assertEqual(result["val_mae"], best["val_acc_per_head"][head])
        self.assertEqual((result["test"]["videos"], result["test"]["failed_videos"]), (TEST_VIDEOS, 0))

        # The metrics come from one prediction per video, in EF points.
        predictions = pd.read_csv(os.path.join(folder, "test_predictions.csv"))
        self.assertEqual(list(predictions.video_path), FakeVideos().samples)
        expected_labels = [(i / TEST_VIDEOS - 0.5) * 10.0 + 55.0 for i in range(TEST_VIDEOS)]
        np.testing.assert_allclose(predictions.label, expected_labels, rtol=1e-6)
        self.assertAlmostEqual(result["test"]["mae"], predictions.abs_error.mean(), places=6)

        # Logged to the probe's own run.
        self.assertEqual(self.wandb.calls[1]["id"], trained.id)
        self.assertEqual(self.wandb.calls[1]["resume"], "must")
        [logged] = tested.logged
        self.assertEqual(logged["probe/test_mae"], result["test"]["mae"])
        self.assertEqual(logged["probe/test_head"], head)

    def test_a_video_that_fails_is_counted_not_replaced(self):
        cfg = self.config(epochs=1)
        self.run_probe(cfg)
        self.run_probe(cfg, videos=FakeVideos(broken={3}), test=True)
        with open(os.path.join(self.folder(cfg), "test_metrics.json")) as f:
            result = json.load(f)
        self.assertEqual((result["test"]["videos"], result["test"]["failed_videos"]), (TEST_VIDEOS - 1, 1))
        self.assertEqual(result["failed_rows"], [3])
        predictions = pd.read_csv(os.path.join(self.folder(cfg), "test_predictions.csv"))
        self.assertNotIn("video_3.mp4", set(predictions.video_path))

    def test_an_unfinished_probe_is_not_tested(self):
        self.run_probe(self.config(epochs=1))
        with self.assertRaisesRegex(RuntimeError, "finished 1 of 2 epochs"):
            self.run_probe(self.config(epochs=2), test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(self.config(epochs=2)), "test_metrics.json")))

    def test_a_resumed_probe_keeps_its_best_epoch(self):
        # Scripted validation: epoch 1 is the best; epoch 2, run after a restart, is worse.
        val = iter([(5.0, np.array([5.0, 6.0])), (7.0, np.array([7.0, 8.0]))])

        def run_one_epoch(training, **_):
            return (6.0, np.array([6.0, 6.0])) if training else next(val)

        with mock.patch.object(probe, "run_one_epoch", side_effect=run_one_epoch):
            self.run_probe(self.config(epochs=1))
            resumed = self.run_probe(self.config(epochs=2))
        best = torch.load(os.path.join(self.folder(self.config(epochs=2)), "best.pt"),
                          map_location="cpu", weights_only=False)
        self.assertEqual(best["epoch"], 1)
        [logged] = resumed.logged
        self.assertEqual(logged["probe/val_mae_best"], 5.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
