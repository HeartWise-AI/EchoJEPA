# tests/utils/test_probe_wandb.py

"""The frozen probe's wandb logging (`evals/video_classification_frozen/eval.py`), on CPU with a
tiny V-JEPA 2.1 encoder, synthetic clips and a fake wandb: what each epoch logs, that a resumed
probe reattaches to its run, and that nothing is logged without a project."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from app.vjepa_2_1.utils import init_video_model
from evals.video_classification_frozen import eval as probe
from src.utils import wandb_logging
from tests.utils.test_vjepa_2_1_eval_encoder import OPTIONS
from tests.utils.test_vjepa_2_1_utils import QuietLogs
from tests.utils.test_vjepa_2_1_wandb import FakeWandb

FRAMES, CROP, BATCHES = 4, 32, 2
GRID = [
    {"start_lr": 1e-4, "lr": 1e-4, "final_lr": 0.0, "warmup": 0.0, "weight_decay": 0.01, "final_weight_decay": 0.01},
    {"start_lr": 5e-5, "lr": 5e-5, "final_lr": 0.0, "warmup": 0.0, "weight_decay": 0.4, "final_weight_decay": 0.4},
]


def fake_make_dataloader(batch_size, **_):
    """Batches as the eval's loader yields them: clips [segment][view], z-scored labels, indices."""
    g = torch.Generator().manual_seed(0)
    batches = [
        ([[torch.randn(batch_size, 3, FRAMES, CROP, CROP, generator=g)]], torch.randn(batch_size, generator=g),
         [torch.arange(FRAMES).expand(batch_size, FRAMES)])
        for _ in range(BATCHES)
    ]
    return batches, SimpleNamespace(set_epoch=lambda epoch: None)


class TestProbeWandb(QuietLogs):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        encoder, _ = init_video_model(
            torch.device("cpu"), model_name="vit_tiny", crop_size=CROP, max_num_frames=FRAMES, pred_depth=4,
            pred_embed_dim=96, use_mask_tokens=True, return_all_tokens=True, **OPTIONS,
        )
        self.checkpoint = os.path.join(self.tmp, "cooldown.pth.tar")
        torch.save({"target_encoder": {f"module.{k}": v for k, v in encoder.state_dict().items()}}, self.checkpoint)
        self.wandb = FakeWandb()

    def config(self, epochs, wandb=True):
        return {
            "folder": os.path.join(self.tmp, "run"),
            "tag": "lvef",
            "num_workers": 0,
            "resume_checkpoint": True,
            "meta": {"wandb_project": "project", "wandb_entity": "entity"} if wandb else {},
            "experiment": {
                "classifier": {"task_type": "regression", "num_heads": 4, "num_probe_blocks": 1, "num_targets": 1},
                "data": {"dataset_train": "unused", "dataset_val": "unused", "resolution": CROP,
                         "frames_per_clip": FRAMES, "frame_step": 2, "num_segments": 1,
                         "num_views_per_segment": 1, "target_mean": 55.0, "target_std": 10.0},
                "optimization": {"batch_size": 2, "num_epochs": epochs, "use_bfloat16": False,
                                 "multihead_kwargs": GRID},
            },
            "model_kwargs": {
                "checkpoint": self.checkpoint,
                "module_name": "evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder",
                "pretrain_kwargs": {"encoder": {"model_name": "vit_tiny", "checkpoint_key": "target_encoder", **OPTIONS}},
                "wrapper_kwargs": {},
            },
        }

    def run_probe(self, cfg):
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0), \
                mock.patch.object(probe.mp, "set_start_method"), \
                mock.patch.object(probe, "make_dataloader", side_effect=fake_make_dataloader), \
                mock.patch.object(wandb_logging, "wandb", self.wandb):
            probe.main(cfg)
        return self.wandb.runs[-1] if self.wandb.runs else None

    def test_each_epoch_logs_the_best_head_and_every_head(self):
        run = self.run_probe(self.config(epochs=2))
        self.assertEqual(self.wandb.calls[0]["resume"], "never")
        self.assertTrue(run.finished)
        self.assertEqual([p["train/global_step"] for p in run.logged], [BATCHES, 2 * BATCHES])
        heads = ["head0_lr0.0001_wd0.01", "head1_lr5e-05_wd0.4"]
        for epoch, p in enumerate(run.logged, start=1):
            self.assertEqual(p["probe/epoch"], epoch)
            for split in ("train", "val"):
                per_head = [p[f"probe/{split}_mae/{h}"] for h in heads]
                self.assertEqual(p[f"probe/{split}_mae"], min(per_head))  # the CSV's best head
        self.assertEqual(run.logged[-1]["probe/val_mae_best"], min(p["probe/val_mae"] for p in run.logged))

    def test_a_resumed_probe_reattaches_to_its_run(self):
        first = self.run_probe(self.config(epochs=1))
        second = self.run_probe(self.config(epochs=2))
        self.assertEqual(self.wandb.calls[1]["id"], first.id)
        self.assertEqual(self.wandb.calls[1]["resume"], "must")
        self.assertEqual([p["probe/epoch"] for p in second.logged], [2])

    def test_without_a_project_nothing_is_logged(self):
        self.assertIsNone(self.run_probe(self.config(epochs=1, wandb=False)))
        self.assertEqual(self.wandb.calls, [])


if __name__ == "__main__":
    unittest.main()
