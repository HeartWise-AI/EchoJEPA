# tests/utils/test_probe_test_split.py

"""The frozen probe's test step (`--test_only`) and its validation record across a resume, on
CPU with a tiny V-JEPA 2.1 encoder, synthetic clips and a fake wandb."""

import json
import os
from unittest import mock

import numpy as np
import pandas as pd
import torch

from evals.video_classification_frozen import eval as probe
from evals.video_classification_frozen import metrics
from tests.utils import test_probe_wandb as probe_wandb
from tests.utils.test_vjepa_2_1_utils import QuietLogs

TEST_VIDEOS = probe_wandb.VIDEOS["test"]


class TestProbeTestSplit(QuietLogs):
    # Helpers come from `test_probe_wandb` through its module: a test class imported by name
    # would be collected here too and run twice.

    def setUp(self):
        probe_wandb.TestProbeWandb.setUp(self)

    def config(self, epochs, index=True):
        return probe_wandb.TestProbeWandb.config(self, epochs, index=index)

    def run_probe(self, cfg, test=False, splits=None):
        return probe_wandb.TestProbeWandb.run_probe(self, dict(cfg, test_only=test), splits=splits)

    def folder(self, cfg):
        return os.path.join(cfg["folder"], "video_classification_frozen", cfg["tag"])

    def result(self, cfg):
        with open(os.path.join(self.folder(cfg), "test_metrics.json")) as f:
            return json.load(f)

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
        result = self.result(cfg)
        self.assertEqual((result["epoch"], result["head"]), (best["epoch"], head))
        self.assertEqual(best["best_epoch"], best["epoch"])
        self.assertEqual(result["val_mae"], best["val_acc_per_head"][head])
        self.assertEqual(result["checkpoint_sha256"], probe.provenance.sha256(os.path.join(folder, "best.pt")))
        self.assertEqual((result["test"]["videos"], result["test"]["failed_videos"]), (TEST_VIDEOS, 0))

        # The metrics come from one prediction per video, in EF points.
        predictions = pd.read_csv(os.path.join(folder, "test_predictions.csv"))
        self.assertEqual(list(predictions.video_path), probe_wandb.FakeVideos("test").samples)
        expected_labels = [z * 10.0 + 55.0 for z in probe_wandb.FakeVideos("test").labels]
        np.testing.assert_allclose(predictions.label, expected_labels, rtol=1e-6)
        self.assertAlmostEqual(result["test"]["video"]["mae"], predictions.abs_error.mean(), places=6)

        # Logged to the probe's own run, with plots that carry no identifier.
        self.assertEqual(self.wandb.calls[1]["id"], trained.id)
        self.assertEqual(self.wandb.calls[1]["resume"], "must")
        scalars, plots = tested.logged
        self.assertEqual(scalars["probe/test/study_mae"], result["test"]["study"]["mae"])
        self.assertEqual(scalars["probe/test/study_auroc"], result["test"]["study"]["reduced_ef"]["auroc"])
        self.assertEqual(scalars["probe/test_head"], head)
        points = plots["probe/test/study/predicted_vs_reference"][1]
        self.assertEqual(points.columns, ["reference", "predicted"])
        self.assertEqual(len(points.data), result["test"]["studies"])
        self.assertEqual(points.data, sorted(points.data))  # sorted by value, not by record

    def test_study_metrics_and_a_threshold_chosen_on_validation(self):
        cfg = self.config(epochs=1)
        self.run_probe(cfg)
        self.run_probe(cfg, test=True)
        result, folder = self.result(cfg), self.folder(cfg)

        # Each study's prediction is the mean of its videos'.
        tested = pd.read_csv(os.path.join(folder, "test_predictions.csv"))
        studies = metrics.by_study(tested.study_id, tested.label, tested.prediction)
        self.assertEqual(result["test"]["studies"], 4)  # 7 videos, 2 per study
        study_mae = float((studies.prediction - studies.label).abs().mean())
        self.assertAlmostEqual(result["test"]["study"]["mae"], study_mae)
        self.assertAlmostEqual(result["test"]["study"]["pearson_r"],
                               metrics.regression(studies.label, studies.prediction)["pearson_r"])

        # The threshold comes from the validation studies, scored with the selected head.
        val = pd.read_csv(os.path.join(folder, "val_predictions.csv"))
        val_studies = metrics.by_study(val.study_id, val.label, val.prediction)
        threshold = metrics.youden_threshold(val_studies.label, val_studies.prediction, 40)
        self.assertEqual((result["low_ef_threshold"]["level"], result["low_ef_threshold"]["value"]),
                         ("study", threshold))
        self.assertEqual(result["test"]["study"]["reduced_ef"]["threshold"], threshold)
        sens, spec = metrics.sensitivity_specificity(studies.label < 40, studies.prediction, threshold)
        self.assertEqual(result["test"]["study"]["reduced_ef"]["sensitivity"], sens)
        self.assertEqual(result["test"]["study"]["reduced_ef"]["specificity"], spec)

        # Targets are checked per study; the by-range table counts studies.
        self.assertEqual(result["targets"]["level"], "study")
        self.assertEqual(result["targets"]["met"]["mae"], result["test"]["study"]["mae"] < 5.0)
        self.assertEqual(sum(r["n"] for r in result["test_by_reference_range"]["rows"]), 4)

    def test_without_a_video_index_the_metrics_are_per_video(self):
        cfg = self.config(epochs=1, index=False)
        self.run_probe(cfg)
        self.run_probe(cfg, test=True)
        result = self.result(cfg)
        self.assertNotIn("study", result["test"])
        self.assertEqual(result["test_by_reference_range"]["level"], "video")

    def test_a_video_that_fails_is_counted_not_replaced(self):
        cfg = self.config(epochs=1)
        self.run_probe(cfg)
        broken = probe_wandb.FakeVideos("test", broken={3})
        self.run_probe(cfg, test=True, splits={"test": broken})
        result = self.result(cfg)
        self.assertEqual((result["test"]["videos"], result["test"]["failed_videos"]), (TEST_VIDEOS - 1, 1))
        self.assertEqual(result["failed_rows"], [3])
        predictions = pd.read_csv(os.path.join(self.folder(cfg), "test_predictions.csv"))
        self.assertNotIn("test_3.mp4", set(predictions.video_path))

    def test_an_unfinished_probe_is_not_tested(self):
        self.run_probe(self.config(epochs=1))
        with self.assertRaisesRegex(RuntimeError, "finished 1 of 2 epochs"):
            self.run_probe(self.config(epochs=2), test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(self.config(epochs=2)), "test_metrics.json")))

    def test_the_test_is_scored_once(self):
        cfg = self.config(epochs=1)
        self.run_probe(cfg)
        self.run_probe(cfg, test=True)
        with open(os.path.join(self.folder(cfg), "test_metrics.json")) as f:
            first = f.read()
        with self.assertRaisesRegex(RuntimeError, "already been tested"):
            self.run_probe(cfg, test=True)
        with open(os.path.join(self.folder(cfg), "test_metrics.json")) as f:
            self.assertEqual(f.read(), first)

    # -- A probe is continued and tested only with what it was trained on.

    def replace_encoder(self, key="target_encoder"):
        """Other weights under `key` of the encoder checkpoint (the target encoder's, shifted)."""
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        saved[key] = {k: v + 0.01 if v.is_floating_point() else v for k, v in saved["target_encoder"].items()}
        torch.save(saved, self.checkpoint)

    def test_a_probe_is_tested_only_with_the_encoder_it_was_trained_on(self):
        cfg = self.config(epochs=1)
        self.run_probe(cfg)
        self.replace_encoder()
        with self.assertRaisesRegex(ValueError, "another encoder_sha256"):
            self.run_probe(cfg, test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(cfg), "test_metrics.json")))

    def test_a_probe_is_tested_only_with_the_encoder_settings_it_was_trained_on(self):
        self.replace_encoder(key="encoder")  # the checkpoint also holds other encoder weights
        cfg = self.config(epochs=1)
        self.run_probe(cfg)
        # The other weights of the same file, then a causal encoder: both change the features.
        for option, value in (("checkpoint_key", "encoder"), ("is_causal", True)):
            changed = self.config(epochs=1)
            changed["model_kwargs"]["pretrain_kwargs"]["encoder"][option] = value
            with self.assertRaisesRegex(ValueError, f"another settings.encoder.pretrain_kwargs.encoder.{option}"):
                self.run_probe(changed, test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(cfg), "test_metrics.json")))

    def test_a_probe_is_resumed_only_with_its_encoder_data_and_settings(self):
        cfg = self.config(epochs=2)
        probe_wandb.TestProbeWandb.interrupted(self, cfg, after=1)
        changed = self.config(epochs=2)
        changed["experiment"]["data"]["target_mean"] = 56.0
        with self.assertRaisesRegex(ValueError, "another normalization.target_mean"):
            self.run_probe(changed)
        with open(self.manifests["val"], "a") as f:
            f.write("val_0.mp4 -2.0\n")
        with self.assertRaisesRegex(ValueError, "another manifests_sha256.val"):
            self.run_probe(cfg)

    def test_resuming_keeps_the_record_the_probe_started_with(self):
        cfg = self.config(epochs=2)
        cfg["meta"]["wandb_run_name"] = "first"
        probe_wandb.TestProbeWandb.interrupted(self, cfg, after=1)
        folder = self.folder(cfg)
        with open(os.path.join(folder, "run_info.json")) as f:
            started = f.read()
        renamed = self.config(epochs=2)
        renamed["meta"]["wandb_run_name"] = "renamed"  # not part of what the probe is
        self.run_probe(renamed)
        with open(os.path.join(folder, "params-probe.yaml")) as f:
            self.assertIn("wandb_run_name: first", f.read())
        with open(os.path.join(folder, "run_info.json")) as f:
            self.assertEqual(f.read(), started)

    def test_a_probe_without_a_fingerprint_is_not_tested(self):
        cfg = self.config(epochs=1)
        self.run_probe(cfg)
        path = os.path.join(self.folder(cfg), "best.pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        del saved["fingerprint"]
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, "no fingerprint"):
            self.run_probe(cfg, test=True)

    def test_only_the_number_of_gpus_may_differ_at_the_test(self):
        saved = {"settings": {"world_size": 4, "seed": 0}, "encoder_sha256": "a"}
        current = {"settings": {"world_size": 1, "seed": 0}, "encoder_sha256": "a"}
        self.assertEqual(probe.provenance.differences(saved, current, ignore=("settings.world_size",)), [])
        self.assertEqual(probe.provenance.differences(saved, current), ["settings.world_size"])

    def test_a_train_manifest_with_validation_videos_is_refused(self):
        cfg = self.config(epochs=1)
        cfg["experiment"]["data"]["dataset_train"] = self.manifests["val"]
        with self.assertRaisesRegex(ValueError, "5 videos of the train manifest are listed under another split"):
            self.run_probe(cfg)
        self.assertFalse(os.path.exists(os.path.join(self.folder(cfg), "latest.pt")))

    def test_wandb_points_at_the_trained_probe(self):
        cfg = self.config(epochs=1)
        trained = self.run_probe(cfg)
        best = os.path.join(self.folder(cfg), "best.pt")
        sha = probe.provenance.sha256(best)
        self.assertEqual(trained.summary["probe_checkpoint/sha256"], sha)
        self.assertEqual(trained.summary["probe_checkpoint/encoder_sha256"], probe.provenance.sha256(self.checkpoint))
        [artifact] = trained.artifacts
        self.assertEqual((artifact.type, artifact.metadata["sha256"], artifact.metadata["file"]),
                         ("probe-checkpoint", sha, "best.pt"))
        self.assertEqual(json.loads(artifact.files["reference.json"]), artifact.metadata)
        self.assertFalse([v for v in artifact.metadata.values() if isinstance(v, str) and os.sep in v])
        tested = self.run_probe(cfg, test=True)
        self.assertEqual(tested.summary["probe_checkpoint/sha256"], sha)  # the test names the same file

    def test_targets_can_be_set_per_video(self):
        cfg = self.config(epochs=1)
        cfg["experiment"]["evaluation"]["targets"] = {"level": "video", "mae_below": 12.0}
        self.run_probe(cfg)
        self.run_probe(cfg, test=True)
        result = self.result(cfg)
        self.assertEqual(result["targets"]["level"], "video")
        self.assertEqual(result["targets"]["met"]["mae"], result["test"]["video"]["mae"] < 12.0)
        self.assertEqual(result["test_by_reference_range"]["level"], "video")
        self.assertEqual(sum(r["n"] for r in result["test_by_reference_range"]["rows"]), TEST_VIDEOS)
        self.assertIn("study", result["test"])  # still reported, with the index

        # The threshold is chosen on the validation videos, not studies, and flags the test videos.
        val = pd.read_csv(os.path.join(self.folder(cfg), "val_predictions.csv"))
        threshold = metrics.youden_threshold(val.label, val.prediction, 40)
        val_studies = metrics.by_study(val.study_id, val.label, val.prediction)
        self.assertNotEqual(threshold, metrics.youden_threshold(val_studies.label, val_studies.prediction, 40))
        self.assertEqual((result["low_ef_threshold"]["level"], result["low_ef_threshold"]["value"]),
                         ("video", threshold))
        tested = pd.read_csv(os.path.join(self.folder(cfg), "test_predictions.csv"))
        sens, spec = metrics.sensitivity_specificity(tested.label < 40, tested.prediction, threshold)
        reduced = result["test"]["video"]["reduced_ef"]
        self.assertEqual((reduced["threshold"], reduced["sensitivity"], reduced["specificity"]),
                         (threshold, sens, spec))

    def test_a_resumed_probe_keeps_its_best_epoch(self):
        # Scripted validation: epoch 1 is the best; epoch 2, run after a restart, is worse.
        val = iter([np.array([5.0, 6.0]), np.array([7.0, 8.0])])

        def validate(**_):
            mae = next(val)
            return {"mae": mae, "loss": mae / 10, "study_mae": mae, "videos": 5, "failed_videos": 0}

        def run_one_epoch(**_):
            return 6.0, np.array([6.0, 6.0]), np.array([0.5, 0.5])

        cfg = self.config(epochs=2)
        with mock.patch.object(probe, "validate", side_effect=validate):
            with mock.patch.object(probe, "run_one_epoch", side_effect=probe_wandb.stop_after(1, run_one_epoch)), \
                    self.assertRaises(probe_wandb.Interrupted):
                self.run_probe(cfg)
            with mock.patch.object(probe, "run_one_epoch", side_effect=run_one_epoch):
                resumed = self.run_probe(cfg)
        best = torch.load(os.path.join(self.folder(cfg), "best.pt"), map_location="cpu", weights_only=False)
        self.assertEqual((best["epoch"], best["best_epoch"]), (1, 1))
        [logged] = resumed.logged
        self.assertEqual(logged["probe/val_mae_best"], 5.0)
        self.assertEqual(logged["probe/best_epoch"], 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
