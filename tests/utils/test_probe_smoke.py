# tests/utils/test_probe_smoke.py

"""An end-to-end smoke run of the frozen Visual EF probe on a small fixture, on CPU: tiny real
videos in the issue #6 manifest layout, `data/make_probe_manifests.py`, the real video loader,
a tiny V-JEPA 2.1 encoder, two probe epochs and the test. Run twice with the same seed, it gives
the same probe and the same numbers."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from app.vjepa_2_1.utils import init_video_model
from evals.video_classification_frozen import eval as probe
from tests.utils.test_probe_wandb import GRID
from tests.utils.test_vjepa_2_1_eval_encoder import OPTIONS
from tests.utils.test_vjepa_2_1_utils import QuietLogs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))  # the manifest scripts are standalone, not a package
import make_probe_manifests  # noqa: E402

FRAMES, CROP = 4, 32
# (split, EF) of each study. A study has two A4C videos and one A2C video, which the probe skips.
STUDIES = [("train", 35), ("train", 50), ("train", 60), ("train", 65),
           ("val", 35), ("val", 55), ("val", 65),
           ("test", 30), ("test", 55), ("test", 60)]
COLUMNS = ["patient_id", "study_id", "exam_type", "view", "video_path", "avi_status", "study_date", "label",
           "split", "n_frames", "fps", "needs_padding"]


def write_video(path, brightness, frames=24, fps=8):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CROP, CROP))
    for i in range(frames):
        writer.write(np.full((CROP, CROP, 3), (brightness + 3 * i) % 256, np.uint8))
    writer.release()


class TestProbeSmoke(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = tmp.name

        rows = []
        for s, (split, ef) in enumerate(STUDIES):
            for v, view in enumerate(("A4C", "A4C", "A2C")):
                path = os.path.join(cls.tmp, "videos", f"p{s}", f"s{s}", f"{v:04d}.mp4")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                write_video(path, brightness=2 * ef + 20 * v)
                rows.append((f"p{s}", f"s{s}", "TTE", view, path, "success", "20240101", float(ef), split, 24, 8.0,
                             False))
        manifests = os.path.join(cls.tmp, "manifests")
        os.makedirs(manifests)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(os.path.join(manifests, "videos.csv"), index=False)
        cls.probe_dir = os.path.join(cls.tmp, "probe")
        with contextlib.redirect_stdout(io.StringIO()):
            make_probe_manifests.main(["--manifests", manifests, "--out-dir", cls.probe_dir, "--views", "A4C"])
        with open(os.path.join(cls.probe_dir, "probe_info.json")) as f:
            cls.info = json.load(f)

        torch.manual_seed(0)
        encoder, _ = init_video_model(
            torch.device("cpu"), model_name="vit_tiny", crop_size=CROP, max_num_frames=FRAMES, pred_depth=4,
            pred_embed_dim=96, use_mask_tokens=True, return_all_tokens=True, **OPTIONS,
        )
        cls.checkpoint = os.path.join(cls.tmp, "encoder.pth.tar")
        torch.save({"target_encoder": {f"module.{k}": v for k, v in encoder.state_dict().items()}}, cls.checkpoint)

    def config(self, folder):
        probe_dir = self.probe_dir
        return {
            "folder": folder,
            "tag": "lvef-a4c",
            "num_workers": 0,
            "resume_checkpoint": True,
            "meta": {"seed": 0},
            "experiment": {
                "classifier": {"task_type": "regression", "probe_type": "attentive", "num_heads": 4,
                               "num_probe_blocks": 1, "num_targets": 1},
                "data": {
                    "dataset_type": "VideoDataset",
                    "dataset_train": os.path.join(probe_dir, "train.csv"),
                    "dataset_val": os.path.join(probe_dir, "val.csv"),
                    "dataset_test": os.path.join(probe_dir, "test.csv"),
                    "video_index": os.path.join(probe_dir, "video_index.csv"),
                    "resolution": CROP, "frames_per_clip": FRAMES, "frame_step": 2, "num_segments": 2,
                    "num_views_per_segment": 1,
                    "target_mean": self.info["target_mean"], "target_std": self.info["target_std"],
                },
                "evaluation": {"low_ef_below": 40,
                               "targets": {"level": "study", "mae_below": 5.0, "auroc_above": 0.95}},
                "optimization": {"batch_size": 2, "num_epochs": 2, "use_bfloat16": False, "multihead_kwargs": GRID},
            },
            "model_kwargs": {
                "checkpoint": self.checkpoint,
                "module_name": "evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder",
                "pretrain_kwargs": {
                    "encoder": {"model_name": "vit_tiny", "checkpoint_key": "target_encoder", **OPTIONS},
                },
                "wrapper_kwargs": {},
            },
        }

    def train_and_test(self, name):
        cfg = self.config(os.path.join(self.tmp, name))
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0), \
                mock.patch.object(probe.mp, "set_start_method"):
            probe.main(dict(cfg))
            probe.main(dict(cfg, test_only=True))
        folder = os.path.join(cfg["folder"], "video_classification_frozen", cfg["tag"])
        with open(os.path.join(folder, "test_metrics.json")) as f:
            result = json.load(f)
        best = torch.load(os.path.join(folder, "best.pt"), map_location="cpu", weights_only=False)
        return result, best

    def test_the_probe_runs_end_to_end_and_repeats_exactly(self):
        first, first_best = self.train_and_test("first")
        second, second_best = self.train_and_test("second")

        # Every A4C video and study is scored; the threshold comes from validation.
        self.assertEqual(self.info["splits"]["test"]["videos"], 6)
        self.assertEqual((first["test"]["videos"], first["test"]["studies"], first["test"]["failed_videos"]),
                         (6, 3, 0))
        self.assertEqual((first["val"]["videos"], first["val"]["failed_videos"]), (6, 0))
        self.assertEqual(first["test"]["study"]["reduced_ef"]["positives"], 1)
        self.assertIsNotNone(first["low_ef_threshold"]["value"])
        self.assertEqual(first["run_info"]["parameters"]["probe_heads"], len(GRID))

        # Same seed, same probe and the same numbers (NaN for empty EF ranges, so compare as JSON).
        for key in ("epoch", "head", "val_mae", "low_ef_threshold", "val", "test", "test_by_reference_range",
                    "targets"):
            self.assertEqual(json.dumps(first[key], sort_keys=True), json.dumps(second[key], sort_keys=True), key)
        for a, b in zip(first_best["classifiers"], second_best["classifiers"]):
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)


if __name__ == "__main__":
    unittest.main()
