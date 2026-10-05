# tests/utils/test_probe_protocol.py

"""How the frozen probe trains, validates and launches, on CPU: metrics weighted by every
label (a short last batch counts for its size), validation that scores each video exactly
once across ranks, an encoder that never changes while the probe learns, the head's output,
and a launcher that fails when a rank fails."""

import os
import sys
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from evals import main as eval_launcher
from evals.video_classification_frozen import eval as probe
from evals.video_classification_frozen import provenance
from src.models.attentive_pooler import AttentiveRegressor
from tests.utils import test_probe_wandb as probe_wandb
from tests.utils.test_vjepa_2_1_utils import QuietLogs, _free_port

CPU = torch.device("cpu")


class ConstantHead(torch.nn.Module):
    """Predicts 0 for every video, through a parameter so that a training step has a gradient."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.new_zeros(x.shape[0], 1) + self.bias


def clip_features(clips, clip_indices):
    """Stands in for the encoder: one output per clip."""
    return [view for segment in clips for view in segment]


class Labelled(torch.utils.data.Dataset):
    """Videos with the given z-scored labels, as `VideoDataset.get_item_video` returns them."""

    def __init__(self, labels, broken=()):
        self.labels, self.broken = list(labels), set(broken)
        self.samples = [f"video_{i}.mp4" for i in range(len(self.labels))]

    def __len__(self):
        return len(self.labels)

    def get_item_video(self, index):
        if index in self.broken:
            return None
        return [[torch.zeros(3, 2, 4, 4)]], self.labels[index], [torch.arange(2)]


def batch(labels):
    n = len(labels)
    return [[torch.zeros(n, 3, 2, 4, 4)]], torch.tensor(labels), [torch.arange(2).expand(n, 2)]


class Step:
    def step(self):
        pass


class TestBatchMetricsAreExact(unittest.TestCase):
    """Seven labels with error 1 and one with error 9: the MAE is 16 / 8 = 2.0. Weighting the
    batch of 7 and the batch of 1 equally would give (1 + 9) / 2 = 5.0."""

    def run_epoch(self, training):
        head = ConstantHead()
        with mock.patch("torch.cuda.max_memory_allocated", return_value=0):
            return probe.run_one_epoch(
                device=CPU, training=training, encoder=clip_features, classifiers=[head], scaler=[None],
                optimizer=[torch.optim.SGD(head.parameters(), lr=0.0)], scheduler=[Step()], wd_scheduler=[Step()],
                data_loader=[batch([1.0] * 7), batch([9.0])], use_bfloat16=False, task_type="regression",
                target_mean=0.0, target_std=1.0,
            )

    def test_validation_batches_count_for_their_size(self):
        best, maes, losses = self.run_epoch(training=False)
        self.assertAlmostEqual(best, 2.0)
        # The training loss (SmoothL1): 0.5 for an error of 1, 8.5 for 9; (7 x 0.5 + 8.5) / 8.
        self.assertAlmostEqual(float(losses[0]), 1.5)

    def test_training_batches_count_for_their_size(self):
        best, maes, losses = self.run_epoch(training=True)
        self.assertAlmostEqual(float(maes[0]), 2.0)
        self.assertAlmostEqual(float(losses[0]), 1.5)

    def test_the_mae_is_in_ef_points(self):
        head = ConstantHead()
        best, _, _ = probe.run_one_epoch(
            device=CPU, training=False, encoder=clip_features, classifiers=[head], scaler=[None], optimizer=[],
            scheduler=[], wd_scheduler=[], data_loader=[batch([1.0] * 7), batch([9.0])], use_bfloat16=False,
            task_type="regression", target_mean=55.0, target_std=10.0,
        )
        self.assertAlmostEqual(best, 20.0)


def validate(dataset, rank=0, world_size=1, studies=None):
    return probe.validate(
        device=CPU, encoder=clip_features, classifiers=[ConstantHead()], dataset=dataset, batch_size=2,
        num_workers=0, use_bfloat16=False, target_mean=0.0, target_std=1.0, rank=rank, world_size=world_size,
        studies=studies,
    )


def _two_rank_worker(rank, world_size, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    try:
        out = validate(Labelled([1, 1, 1, 1, 9]), rank=rank, world_size=world_size)
        queue.put((rank, float(out["mae"][0]), out["videos"]))
    finally:
        dist.destroy_process_group()


class TestValidationScoresEachVideoOnce(unittest.TestCase):
    """Validation scores each video once: no batch weighting, no padded duplicates, and a video
    that fails to load is counted, not replaced."""

    def test_a_short_last_batch_counts_for_its_size(self):
        # Batches (1, 1), (1, 1), (9): the MAE is 13 / 5 = 2.6, not (1 + 1 + 9) / 3.
        out = validate(Labelled([1, 1, 1, 1, 9]))
        self.assertAlmostEqual(float(out["mae"][0]), 2.6)
        self.assertEqual((out["videos"], out["failed_videos"]), (5, 0))

    def test_a_failed_video_is_left_out_and_counted(self):
        out = validate(Labelled([1, 1, 1, 1, 9], broken={4}))
        self.assertAlmostEqual(float(out["mae"][0]), 1.0)
        self.assertEqual((out["videos"], out["failed_videos"]), (4, 1))

    def test_study_mae_averages_each_study_first(self):
        # Study a has 3 videos with error 1, study b 2 videos with error 9: per study (1 + 9) / 2,
        # where per video it is (3 + 18) / 5.
        out = validate(Labelled([1, 1, 1, 9, 9]), studies=np.array(["a", "a", "a", "b", "b"]))
        self.assertAlmostEqual(float(out["study_mae"][0]), 5.0)
        self.assertAlmostEqual(float(out["mae"][0]), 4.2)

    def test_two_ranks_score_each_video_once_and_agree(self):
        # Ranks take videos 0, 2, 4 and 1, 3: a sampler would pad rank 1 with a repeat.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        mp.spawn(_two_rank_worker, args=(2, _free_port(), queue), nprocs=2, join=True)
        results = sorted(queue.get(timeout=60) for _ in range(2))
        self.assertEqual([(r[1], r[2]) for r in results], [(2.6, 5), (2.6, 5)])


class TestOnlyTheProbeLearns(QuietLogs):
    """The encoder is frozen: its weights are the same after training, and no optimizer holds
    them. The probe's heads do take optimizer steps."""

    def setUp(self):
        probe_wandb.TestProbeWandb.setUp(self)

    def run_probe(self, cfg, init_module):
        with mock.patch.object(probe, "init_module", side_effect=init_module):
            probe_wandb.TestProbeWandb.run_probe(self, cfg)

    def test_encoder_weights_do_not_change_while_the_probe_trains(self):
        built, real = [], probe.init_module

        def keep_encoder(**kwargs):
            encoder = real(**kwargs)
            built.append((encoder, {k: v.clone() for k, v in encoder.state_dict().items()}))
            return encoder

        cfg = probe_wandb.TestProbeWandb.config(self, epochs=1, wandb=False)
        self.run_probe(cfg, keep_encoder)
        [(encoder, before)] = built
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        for name, value in encoder.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        latest = torch.load(os.path.join(cfg["folder"], "video_classification_frozen", "lvef", "latest.pt"),
                            map_location="cpu", weights_only=False)
        self.assertTrue(all(o["state"] for o in latest["opt"]))  # every head took AdamW steps

    def test_an_encoder_that_could_learn_is_refused(self):
        real = probe.init_module

        def trainable_encoder(**kwargs):
            encoder = real(**kwargs)
            next(encoder.parameters()).requires_grad = True
            return encoder

        with self.assertRaisesRegex(RuntimeError, "encoder must be frozen"):
            self.run_probe(probe_wandb.TestProbeWandb.config(self, epochs=1, wandb=False), trainable_encoder)


class TestProbeHead(unittest.TestCase):

    def test_one_ef_per_clip_from_the_encoder_tokens(self):
        head = AttentiveRegressor(embed_dim=192, num_heads=4, depth=1, num_targets=1)
        self.assertEqual(tuple(head(torch.randn(3, 8, 192)).shape), (3, 1))


def eval_rank_exits(args, rank, fname, world_size, devices):
    """Stands in for an evaluation rank; `fname` names the rank that fails."""
    sys.exit(1 if str(rank) == fname else 0)


class TestEvalLauncher(unittest.TestCase):
    """python -m evals.main exits non-zero when a rank fails, so a failed probe is not mistaken
    for a finished one."""

    def launch(self, failing):
        args = SimpleNamespace(fname=failing, devices=["cuda:0", "cuda:1"])
        with mock.patch.object(eval_launcher, "process_main", eval_rank_exits):
            eval_launcher.launch_ranks(args)

    def test_clean_ranks_return(self):
        self.launch("none")

    def test_a_failed_rank_fails_the_command(self):
        with self.assertRaisesRegex(SystemExit, r"rank\(s\) \[1\]"):
            self.launch("1")


class TestVideoIndex(unittest.TestCase):
    """The video index the leakage check relies on: every row complete, known splits only."""

    ROWS = [("train", "a.mp4", "s1", "p1"), ("val", "b.mp4", "s2", "p2"), ("test", "c.mp4", "s3", "p3")]

    def load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "video_index.csv")
            pd.DataFrame(rows, columns=provenance.INDEX_COLUMNS).to_csv(path, index=False)
            return provenance.load_video_index(path)

    def test_a_complete_index_loads(self):
        self.assertEqual(list(self.load(self.ROWS).index), ["a.mp4", "b.mp4", "c.mp4"])

    def test_a_row_without_a_patient_is_refused(self):
        # Two videos without a patient, in train and validation: groupby would drop both, unchecked.
        rows = self.ROWS + [("train", "d.mp4", "s4", None), ("val", "e.mp4", "s5", " ")]
        with self.assertRaisesRegex(ValueError, "without a value in {'patient_id': 2}"):
            self.load(rows)

    def test_an_unknown_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown splits \\['validation'\\]"):
            self.load(self.ROWS + [("validation", "d.mp4", "s4", "p4")])


if __name__ == "__main__":
    unittest.main()
