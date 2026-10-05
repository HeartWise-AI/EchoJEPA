# tests/utils/test_probe_test_split.py

"""The frozen probe's test step (`--test_only`) and its validation record across a resume, on
CPU with a tiny V-JEPA 2.1 encoder, synthetic clips and a fake wandb."""

import json
import os
import shutil
from unittest import mock

import numpy as np
import pandas as pd
import torch

from evals.video_classification_frozen import eval as probe
from evals.video_classification_frozen import metrics
from src.utils import wandb_logging
from tests.utils import test_probe_wandb as probe_wandb
from tests.utils.test_vjepa_2_1_utils import QuietLogs

TEST_VIDEOS = probe_wandb.VIDEOS["test"]


def scripted(scores):
    """`validate`, with the per-study MAE that selects the probe taken from `scores`, one row of
    head scores per epoch."""
    rows = iter(scores)
    validate = probe.validate

    def scored(*args, **kwargs):
        return dict(validate(*args, **kwargs), study_mae=np.asarray(next(rows), dtype=float))

    return scored


def failing_save(epoch, name, error):
    """`save_atomically`, raising `error` at the file `name` of epoch `epoch`: before writing it
    (an `OSError`, as a full disk) or after (an interruption)."""
    save = probe.save_atomically

    def saving(obj, path):
        at = obj["epoch"] == epoch and os.path.basename(path) == name
        if at and error is OSError:
            raise OSError("No space left on device")
        save(obj, path)
        if at:
            raise error

    return saving


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
        self.assertEqual(best["selection_metric"], "study_mae")  # with a video index
        self.assertEqual(result["val_study_mae"], best["val_acc_per_head"][head])
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
        image = plots["probe/test/study/predicted_vs_reference"]  # an image, not a wandb.Table
        self.assertEqual(image.data.shape, (480, 480, 3))
        self.assertEqual(image.caption, f"Predicted vs reference EF, n = {result['test']['studies']}")
        self.assertEqual(len(plots["probe/test/study/residuals"][1]), result["test"]["studies"])
        by_range = {k.removeprefix("probe/test/study/by_reference_range/"): v for k, v in plots.items()
                    if k.startswith("probe/test/study/by_reference_range/")}
        rows = {wandb_logging.range_key(r["range"]): r for r in result["test_by_reference_range"]["rows"]}
        self.assertEqual(sum(v for k, v in by_range.items() if k.endswith("/n")), result["test"]["studies"])
        for key, row in rows.items():
            self.assertEqual(by_range[f"{key}/n"], row["n"])
            if row["n"]:
                self.assertEqual((by_range[f"{key}/mae"], by_range[f"{key}/bias"]), (row["mae"], row["bias"]))

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

    # -- Checkpoints: `latest.pt` records an epoch only once that epoch's files are written, and
    #    the test takes `best.pt` only if it is the epoch `latest.pt` records as the best.

    def load(self, cfg, name):
        return torch.load(os.path.join(self.folder(cfg), name), map_location="cpu", weights_only=False)

    def test_a_checkpoint_is_replaced_whole_or_not_at_all(self):
        path = os.path.join(self.tmp, "latest.pt")
        probe.save_atomically({"epoch": 1}, path)

        def partial(obj, f):
            with open(f, "wb") as out:
                out.write(b"PK")
            raise OSError("No space left on device")

        with mock.patch.object(probe.torch, "save", side_effect=partial), self.assertRaises(OSError):
            probe.save_atomically({"epoch": 2}, path)
        self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)

    def test_an_epoch_whose_best_pt_was_not_saved_is_trained_again(self):
        cfg = self.config(epochs=2)
        with mock.patch.object(probe, "validate", side_effect=scripted([[5, 6], [1, 2]])), \
                mock.patch.object(probe, "save_atomically", side_effect=failing_save(2, "best.pt", OSError)), \
                self.assertRaises(OSError):
            self.run_probe(cfg)
        latest = self.load(cfg, "latest.pt")
        self.assertEqual((latest["epoch"], latest["best_epoch"], self.load(cfg, "best.pt")["epoch"]), (1, 1, 1))
        with self.assertRaisesRegex(RuntimeError, "finished 1 of 2 epochs"):
            self.run_probe(cfg, test=True)

        with mock.patch.object(probe, "validate", side_effect=scripted([[1, 2]])):
            self.run_probe(cfg)  # epoch 2 again
        self.run_probe(cfg, test=True)
        self.assertEqual((self.result(cfg)["epoch"], self.result(cfg)["head"]), (2, 0))

    def test_a_best_pt_the_checkpoint_does_not_record_is_restored_on_resume(self):
        cfg = self.config(epochs=2)
        stopped = probe_wandb.Interrupted
        with mock.patch.object(probe, "validate", side_effect=scripted([[5, 6], [1, 2]])), \
                mock.patch.object(probe, "save_atomically", side_effect=failing_save(2, "best.pt", stopped)), \
                self.assertRaises(stopped):
            self.run_probe(cfg)  # stopped between epoch 2's `best.pt` and its `latest.pt`
        self.assertEqual((self.load(cfg, "latest.pt")["best_epoch"], self.load(cfg, "best.pt")["epoch"]), (1, 2))

        # Trained again, epoch 2 now scores worse than epoch 1 (GPU runs are not bit-exact).
        with mock.patch.object(probe, "validate", side_effect=scripted([[7, 8]])):
            self.run_probe(cfg)
        self.assertEqual((self.load(cfg, "latest.pt")["best_epoch"], self.load(cfg, "best.pt")["epoch"]), (1, 1))
        self.run_probe(cfg, test=True)
        self.assertEqual((self.result(cfg)["epoch"], self.result(cfg)["val_study_mae"]), (1, 5.0))

    def test_a_best_pt_other_than_the_selected_epoch_is_not_tested(self):
        cfg = self.config(epochs=2)
        with mock.patch.object(probe, "validate", side_effect=scripted([[5, 6], [1, 2]])):
            self.run_probe(cfg)
        folder = self.folder(cfg)
        shutil.copyfile(os.path.join(folder, "epoch_001.pt"), os.path.join(folder, "best.pt"))
        with self.assertRaisesRegex(RuntimeError, "holds epoch 1, but training selected epoch 2"):
            self.run_probe(cfg, test=True)
        self.assertFalse(os.path.exists(os.path.join(folder, "test_metrics.json")))

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

    def test_a_probe_is_continued_and_tested_only_under_its_protocol(self):
        cfg = self.config(epochs=2)
        probe_wandb.TestProbeWandb.interrupted(self, cfg, after=1)
        latest = os.path.join(self.folder(cfg), "latest.pt")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        self.assertEqual(saved["fingerprint"]["settings"]["protocol"], probe.PROTOCOL)
        later = probe.PROTOCOL["version"] + 1
        with mock.patch.dict(probe.PROTOCOL, version=later), \
                self.assertRaisesRegex(ValueError, "another settings.protocol.version"):
            self.run_probe(cfg)
        # As the code before the protocol was recorded (random evaluation clips) saved it.
        del saved["fingerprint"]["settings"]["protocol"]
        torch.save(saved, latest)
        with self.assertRaisesRegex(ValueError, "another settings.protocol than .* an earlier protocol"):
            self.run_probe(cfg)

        finished = dict(self.config(epochs=1), tag="finished")
        self.run_probe(finished)
        with mock.patch.dict(probe.PROTOCOL, version=later), \
                self.assertRaisesRegex(ValueError, "another settings.protocol.version"):
            self.run_probe(finished, test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(finished), "test_metrics.json")))

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
        self.assertEqual(artifact.files, {})  # a file's manifest entry would record its local path
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

    # -- Selection.

    def train_with_scores(self, selection, tag):
        """Train two epochs whose validation scores are set: per video, the first epoch's first head
        is best; per study, the second epoch's second head."""
        scores = iter([(np.array([3.0, 4.0]), np.array([5.0, 4.5])),
                       (np.array([3.5, 3.8]), np.array([4.2, 4.0]))])

        def validate(**_):
            mae, study_mae = next(scores)
            return {"mae": mae, "loss": mae / 10, "study_mae": study_mae, "videos": 5, "failed_videos": 0}

        cfg = self.config(epochs=2)
        cfg["tag"] = tag
        cfg["experiment"]["evaluation"]["selection"] = selection
        with mock.patch.object(probe, "validate", side_effect=validate):
            run = self.run_probe(cfg)
        best = torch.load(os.path.join(self.folder(cfg), "best.pt"), map_location="cpu", weights_only=False)
        return cfg, run, best

    def test_the_probe_is_selected_per_study(self):
        cfg, run, best = self.train_with_scores("study", "per-study")
        self.assertEqual((best["epoch"], best["selection_metric"]), (2, "study_mae"))
        self.assertEqual(best["val_acc_per_head"], [4.2, 4.0])
        self.assertEqual(best["val_scores_per_head"]["mae"], [3.5, 3.8])  # kept, not selected on
        self.assertEqual(run.logged[-1]["probe/val_study_mae_best"], 4.0)
        self.assertEqual([p["probe/val_mae"] for p in run.logged], [3.0, 3.5])

        # The rule is part of the probe: it is not tested under the other one.
        per_video = dict(cfg, experiment={**cfg["experiment"], "evaluation": {
            **cfg["experiment"]["evaluation"], "selection": "video"}})
        with self.assertRaisesRegex(ValueError, "another settings.selection"):
            self.run_probe(per_video, test=True)
        tested = self.run_probe(cfg, test=True)
        result = self.result(cfg)
        self.assertEqual((result["epoch"], result["head"], result["val_study_mae"]), (2, 1, 4.0))
        self.assertIn("per-study validation MAE", result["selection"])
        self.assertEqual(tested.summary["probe_checkpoint/val_study_mae"], 4.0)

    def test_the_probe_can_be_selected_per_video(self):
        cfg, _, best = self.train_with_scores("video", "per-video")
        self.assertEqual((best["epoch"], best["selection_metric"], best["val_acc_per_head"]),
                         (1, "mae", [3.0, 4.0]))
        self.run_probe(cfg, test=True)
        result = self.result(cfg)
        self.assertEqual((result["epoch"], result["head"], result["val_mae"]), (1, 0, 3.0))

    def test_per_study_selection_needs_the_video_index(self):
        cfg = self.config(epochs=1, index=False)
        cfg["experiment"]["evaluation"] = {"selection": "study"}
        with self.assertRaisesRegex(ValueError, "Per-study selection"):
            self.run_probe(cfg)

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
        self.assertEqual(logged["probe/val_study_mae_best"], 5.0)
        self.assertEqual(logged["probe/best_epoch"], 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
