# tests/utils/test_probe_score_validation.py

"""Validation-only scoring of a finished Visual EF probe (`score_validation.py`), on CPU with a tiny
V-JEPA 2.1 encoder, synthetic clips and a fake wandb: the epoch and head the test step would take, the
same validation predictions as the test step writes, study-level outputs on hand-built predictions,
the probe refused when its fingerprint differs or it is unfinished, outputs never overwritten or written
into the repository, and a test split that is never opened."""

import builtins
import json
import os
import shutil
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from evals.video_classification_frozen import eval as probe
from evals.video_classification_frozen import provenance
from evals.video_classification_frozen import score_validation as scoring
from tests.utils import test_probe_wandb as probe_wandb
from tests.utils.test_vjepa_2_1_utils import QuietLogs

VAL_VIDEOS = probe_wandb.VIDEOS["val"]


def scripted(scores):
    """`validate` returning set scores, one (per-video MAE, per-study MAE) pair of head lists per epoch."""
    rows = iter(scores)

    def validate(**_):
        mae, study_mae = (np.asarray(v, dtype=float) for v in next(rows))
        return {"mae": mae, "loss": mae / 10, "study_mae": study_mae, "videos": VAL_VIDEOS, "failed_videos": 0}

    return validate


def rank_exits(args, rank, fname, world_size, devices):
    """Stands in for a scoring rank; `fname` names the rank that fails."""
    sys.exit(1 if str(rank) == fname else 0)


class TestScoreValidation(QuietLogs):
    # Helpers come from `test_probe_wandb` through its module: a test class imported by name would be
    # collected here too and run twice.

    def setUp(self):
        probe_wandb.TestProbeWandb.setUp(self)

    def config(self, epochs=1):
        return probe_wandb.TestProbeWandb.config(self, epochs)

    def folder(self, cfg):
        return os.path.join(cfg["folder"], "video_classification_frozen", cfg["tag"])

    def train(self, cfg, validate=None):
        if validate is None:
            probe_wandb.TestProbeWandb.run_probe(self, cfg)
            return
        with mock.patch.object(probe, "validate", side_effect=validate):
            probe_wandb.TestProbeWandb.run_probe(self, cfg)

    def score(self, cfg, **kwargs):
        """Score `cfg`'s validation split; only the validation split may be built."""
        built = []

        def make_dataloader(**kw):
            built.append(os.path.splitext(os.path.basename(kw["root_path"][0]))[0])
            return probe_wandb.fake_make_dataloader(**kw)

        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(probe, "make_dataloader", side_effect=make_dataloader):
            result = scoring.score_validation(cfg, **kwargs)
        self.assertEqual(built, ["val"])
        return result

    def outputs(self, folder):
        with open(os.path.join(folder, scoring.METRICS)) as f:
            metrics = json.load(f)
        return pd.read_csv(os.path.join(folder, scoring.PREDICTIONS)), metrics

    def assert_nothing_written(self, folder):
        for name in (scoring.PREDICTIONS, scoring.METRICS):
            self.assertFalse(os.path.exists(os.path.join(folder, name)), name)

    def load(self, cfg, name="best.pt"):
        return torch.load(os.path.join(self.folder(cfg), name), map_location="cpu", weights_only=False)

    # -- What is scored.

    def test_the_scores_reproduce_the_validation_the_probe_was_selected_on(self):
        cfg = self.config(epochs=2)
        self.train(cfg)
        result = self.score(cfg)  # the fingerprint `eval.main` saved is rebuilt exactly
        best = self.load(cfg)
        head = int(np.argmin(best["val_acc_per_head"]))
        self.assertEqual((result["epoch"], result["head"]), (best["epoch"], head))
        self.assertEqual(result["head_name"], ["head0_lr0.0001_wd0.01", "head1_lr5e-05_wd0.4"][head])
        self.assertEqual(result["checkpoint_sha256"], provenance.sha256(os.path.join(self.folder(cfg), "best.pt")))
        check = result["stored_score_check"]
        self.assertTrue(check["within_tolerance"])
        self.assertEqual(check["study_mae"]["stored"], best["val_scores_per_head"]["study_mae"][head])
        self.assertLess(abs(check["study_mae"]["difference"]), 1e-9)  # the same computation, on CPU
        self.assertLess(abs(check["mae"]["difference"]), 1e-9)

        predictions, metrics = self.outputs(self.folder(cfg))
        self.assertEqual(list(predictions.columns), ["video_path", "study_id", "label", "prediction"])
        videos = probe_wandb.FakeVideos("val")
        self.assertEqual(list(predictions.video_path), videos.samples)
        self.assertEqual(list(predictions.study_id), [f"val-s{i // 2}" for i in range(VAL_VIDEOS)])
        np.testing.assert_allclose(predictions.label, [z * 10.0 + 55.0 for z in videos.labels])
        self.assertEqual((metrics["val"]["videos"], metrics["val"]["studies"], metrics["val"]["failed_videos"]),
                         (VAL_VIDEOS, 3, 0))
        self.assertAlmostEqual(metrics["val"]["study"]["mae"], check["study_mae"]["recomputed"])
        self.assertEqual(metrics["fingerprint_not_checked"], ["settings.world_size", "manifests_sha256.test"])
        self.assertEqual(metrics["world_size"], {"trained": 1, "scored": 1})

    def test_the_head_and_predictions_are_those_of_the_test_step(self):
        # Per study, epoch 2's second head is best; per video it would be epoch 1's first head.
        cfg = self.config(epochs=2)
        self.train(cfg, scripted([([3.0, 4.0], [5.0, 4.5]), ([3.5, 3.8], [4.2, 4.0])]))
        self.assertEqual(scoring.select_head({"val_acc_per_head": [3.0, 1.0, 1.0, 2.0]}), 1)  # first of ties

        # The scripted scores are not what the head scores: refused at the default tolerance.
        with self.assertRaisesRegex(RuntimeError, "differ from `best.pt` by more than 0.01 EF points"):
            self.score(cfg)
        self.assert_nothing_written(self.folder(cfg))
        result = self.score(cfg, tolerance=float("inf"))
        self.assertEqual((result["epoch"], result["head"], result["val_study_mae"]), (2, 1, 4.0))
        self.assertEqual(result["stored_score_check"]["study_mae"]["stored"], 4.0)

        # The test step takes the same epoch and head, and writes the same validation predictions.
        probe_wandb.TestProbeWandb.run_probe(self, dict(cfg, test_only=True))
        with open(os.path.join(self.folder(cfg), "test_metrics.json")) as f:
            tested = json.load(f)
        self.assertEqual((tested["epoch"], tested["head"]), (result["epoch"], result["head"]))
        ours, metrics = self.outputs(self.folder(cfg))
        theirs = pd.read_csv(os.path.join(self.folder(cfg), "val_predictions.csv"))
        for column in ("video_path", "study_id"):
            self.assertEqual(list(ours[column]), list(theirs[column]))
        np.testing.assert_allclose(ours.label, theirs.label, rtol=0, atol=1e-9)
        np.testing.assert_allclose(ours.prediction, theirs.prediction, rtol=0, atol=1e-9)
        for level in ("study", "video"):
            self.assertAlmostEqual(metrics["val"][level]["mae"], tested["val"][level]["mae"])
        self.assertEqual(metrics["low_ef_threshold"]["value"], tested["low_ef_threshold"]["value"])

    def test_study_outputs_on_hand_built_predictions(self):
        # Head 0 is selected per study (2.0), though head 1 is better per video.
        cfg = self.config(epochs=1)
        self.train(cfg, scripted([([4.5, 1.0], [2.0, 3.0])]))
        best = self.load(cfg)
        seen = []

        def score_videos(device, encoder, heads, dataset, *args):
            [head] = heads
            seen.append((head.state_dict(), dataset.samples))
            # Videos 0-3 scored, video 4 (study val-s2) failed. EF 35, 35, 55, 55 predicted 40, 30, 58, 60.
            labels = np.array([[-2.0], [-2.0], [0.0], [0.0]])
            predictions = np.array([[[-1.5], [-2.5], [0.3], [0.5]]])
            return np.arange(4), labels, predictions, [4]

        with mock.patch.object(probe, "score_videos", side_effect=score_videos):
            result = self.score(cfg)
        [(state, samples)] = seen
        self.assertEqual(samples, probe_wandb.FakeVideos("val").samples)
        for key, value in best["classifiers"][0].items():
            self.assertTrue(torch.equal(state[key.removeprefix("module.")], value), key)

        predictions, metrics = self.outputs(self.folder(cfg))
        self.assertEqual(list(predictions.video_path), [f"val_{i}.mp4" for i in range(4)])
        self.assertEqual(list(predictions.study_id), ["val-s0", "val-s0", "val-s1", "val-s1"])
        np.testing.assert_allclose(predictions.label, [35, 35, 55, 55])
        np.testing.assert_allclose(predictions.prediction, [40, 30, 58, 60])
        # Study val-s0: mean 35, error 0; val-s1: mean 59, error 4. Per video: errors 5, 5, 3, 5.
        study, video = metrics["val"]["study"], metrics["val"]["video"]
        self.assertEqual((study["n"], metrics["val"]["studies"]), (2, 2))
        self.assertAlmostEqual(study["mae"], 2.0)
        self.assertAlmostEqual(study["bias"], 2.0)
        self.assertEqual((video["n"], metrics["val"]["videos"], metrics["val"]["failed_videos"]), (4, 4, 1))
        self.assertAlmostEqual(video["mae"], 4.5)
        self.assertEqual(metrics["failed_rows"], [4])
        # Threshold from the studies (35 -> 35 positive, 55 -> 59 negative): 47 separates them.
        self.assertEqual(metrics["low_ef_threshold"]["value"], 47.0)
        self.assertEqual((study["reduced_ef"]["sensitivity"], study["reduced_ef"]["specificity"]), (1.0, 1.0))
        self.assertEqual((result["head"], result["val_study_mae"]), (0, 2.0))
        check = result["stored_score_check"]
        self.assertAlmostEqual(check["study_mae"]["difference"], 0.0)
        self.assertAlmostEqual(check["mae"]["difference"], 0.0)

    # -- Refusals.

    def test_a_probe_with_another_encoder_data_or_setting_is_refused(self):
        cfg = self.config(epochs=1)
        self.train(cfg)
        folder = self.folder(cfg)

        def changed(section, key, value):
            other = self.config(epochs=1)
            other["experiment"][section][key] = value
            return other

        for other, difference in ((changed("data", "target_mean", 56.0), "normalization.target_mean"),
                                  (changed("data", "frame_step", 3), "settings.clips.frame_step"),
                                  (changed("evaluation", "selection", "video"), "settings.selection")):
            with self.subTest(difference=difference), self.assertRaisesRegex(ValueError, f"another {difference}"):
                self.score(other)
        with open(self.manifests["val"], "a") as f:
            f.write("val_0.mp4 -2.0\n")
        with self.assertRaisesRegex(ValueError, "another manifests_sha256.val"):
            self.score(cfg)
        self.assert_nothing_written(folder)

    def test_a_probe_with_another_encoder_is_refused(self):
        cfg = self.config(epochs=1)
        self.train(cfg)
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        saved["target_encoder"] = {k: v + 0.01 for k, v in saved["target_encoder"].items()}
        torch.save(saved, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "another encoder_sha256"):
            self.score(cfg)
        self.assert_nothing_written(self.folder(cfg))

    def test_only_the_number_of_gpus_may_differ(self):
        cfg = self.config(epochs=1)
        self.train(cfg)
        path = os.path.join(self.folder(cfg), "best.pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved["fingerprint"]["settings"]["world_size"] = 4
        torch.save(saved, path)
        self.assertEqual(self.score(cfg)["head"], int(np.argmin(saved["val_acc_per_head"])))

    def test_an_unfinished_probe_is_refused(self):
        self.train(self.config(epochs=1))
        with self.assertRaisesRegex(RuntimeError, "needs a finished probe.*finished 1 of 2 epochs"):
            self.score(self.config(epochs=2))
        self.assert_nothing_written(self.folder(self.config(epochs=1)))

    def test_a_folder_without_the_selected_best_pt_is_refused(self):
        cfg = self.config(epochs=2)
        self.train(cfg, scripted([([5, 6], [5, 6]), ([1, 2], [1, 2])]))
        folder = self.folder(cfg)
        shutil.copyfile(os.path.join(folder, "epoch_001.pt"), os.path.join(folder, "best.pt"))
        with self.assertRaisesRegex(RuntimeError, "holds epoch 1, but training selected epoch 2"):
            self.score(cfg)
        os.remove(os.path.join(folder, "best.pt"))
        with self.assertRaisesRegex(RuntimeError, "holds no `best.pt`"):
            self.score(cfg)
        self.assert_nothing_written(folder)

    def test_existing_outputs_are_kept_unless_overwrite(self):
        cfg = self.config(epochs=1)
        self.train(cfg)
        self.score(cfg)
        folder = self.folder(cfg)
        with open(os.path.join(folder, scoring.METRICS)) as f:
            first = f.read()
        with self.assertRaisesRegex(RuntimeError, "already exist.*--overwrite"):
            self.score(cfg)
        with open(os.path.join(folder, scoring.METRICS)) as f:
            self.assertEqual(f.read(), first)
        self.score(cfg, overwrite=True)

    def test_outputs_go_outside_the_repository(self):
        cfg = self.config(epochs=1)
        self.train(cfg)
        inside = os.path.join(provenance.REPO_ROOT, "val-scores-refused")
        with self.assertRaisesRegex(SystemExit, "inside the code repository"):
            self.score(cfg, out_dir=inside)
        self.assertFalse(os.path.exists(inside))
        elsewhere = os.path.join(self.tmp, "scores")
        self.score(cfg, out_dir=elsewhere)
        self.assertEqual(sorted(os.listdir(elsewhere)), sorted([scoring.METRICS, scoring.PREDICTIONS]))
        self.assert_nothing_written(self.folder(cfg))

    def test_environment_overrides_are_refused(self):
        with mock.patch.dict(os.environ, {"OVERRIDE_TARGET_MEAN": "56"}), \
                self.assertRaisesRegex(ValueError, "OVERRIDE_TARGET_MEAN"):
            scoring.score_validation(self.config(epochs=1))

    # -- The test split.

    def test_the_test_manifest_is_never_opened(self):
        cfg = self.config(epochs=1)
        self.train(cfg)
        test_manifest = os.path.realpath(self.manifests["test"])

        # A test manifest that does not exist: nothing needs it.
        missing = self.config(epochs=1)
        missing["experiment"]["data"]["dataset_test"] = os.path.join(self.tmp, "absent", "test.csv")
        self.score(missing)

        # The real one, guarded: opening it fails the test.
        real_open = builtins.open

        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and os.path.realpath(os.fsdecode(file)) == test_manifest:
                raise AssertionError("The test manifest was opened.")
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=guarded):
            self.score(cfg, overwrite=True)
        with open(os.path.join(self.folder(cfg), scoring.METRICS)) as f:
            self.assertNotIn(provenance.sha256(self.manifests["test"]), f.read())


class TestLauncher(QuietLogs):

    def launch(self, failing):
        with mock.patch.object(scoring, "process_main", rank_exits):
            scoring.launch_ranks(SimpleNamespace(fname=failing, devices=["cuda:0", "cuda:1"]))

    def test_clean_ranks_return(self):
        self.launch("none")

    def test_a_failed_rank_fails_the_command(self):
        with self.assertRaisesRegex(SystemExit, r"rank\(s\) \[1\]"):
            self.launch("1")


if __name__ == "__main__":
    import unittest

    unittest.main()
