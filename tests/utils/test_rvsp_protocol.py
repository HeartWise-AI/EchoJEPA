# tests/utils/test_rvsp_protocol.py

"""The multi-view RVSP probe end to end on CPU (`evals/video_classification_frozen_multi/eval.py`):
tiny synthetic videos, `data/make_rvsp_probe_manifests.py`, the per-study loader, a tiny V-JEPA 2.1
encoder, training, selection and the test. It repeats exactly, and a resumed run continues as an
uninterrupted one; the epoch and head are chosen by per-study validation MAE; a probe resumed or
tested with another encoder, seed or manifests is refused; training and validation never open the
test manifest, and the test runs once, only on a finished probe and the test manifest written with
the training ones."""

import builtins
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from app.vjepa_2_1.utils import init_video_model
from evals.video_classification_frozen import provenance
from evals.video_classification_frozen_multi import eval as multi
from evals.video_classification_frozen_multi import score_validation
from src.utils import wandb_logging
from tests.utils import test_probe_wandb as probe_wandb
from tests.utils.test_rvsp_dataset import CROP, FRAMES, write_video
from tests.utils.test_vjepa_2_1_eval_encoder import OPTIONS
from tests.utils.test_vjepa_2_1_utils import QuietLogs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))  # the manifest scripts are standalone, not a package
import make_rvsp_probe_manifests as mrm  # noqa: E402

SLOTS = ("slots:\n"
         "  - {name: A4C_bmode, views: [A4C], acq: [bmode], clips: 1}\n"
         "  - {name: A4C_color, views: [A4C], acq: [color_flow], clips: 1}\n"
         "  - {name: RVINF, views: [RVINF], acq: any, clips: 1}\n")
# (split, RVSP, views present as (view, acquisition)) of each study.
FULL = (("A4C", "bmode"), ("A4C", "color_flow"), ("RVINF", "bmode"))
STUDIES = [("train", 25.0, FULL), ("train", 30.0, FULL[:1]), ("train", 40.0, FULL[1:]), ("train", 55.0, FULL),
           ("train", 35.0, FULL[2:]), ("train", 65.0, FULL[:2]),
           ("val", 28.0, FULL), ("val", 45.0, FULL[2:]), ("val", 60.0, FULL[:1]),
           ("test", 32.0, FULL), ("test", 50.0, FULL[1:]), ("test", 70.0, FULL[:1])]
GRID = probe_wandb.GRID


class Interrupted(Exception):
    """Stops a probe between two epochs, as a killed job would."""


def stop_after(epochs, train_one_epoch):
    started = []

    def wrapped(**kwargs):
        if len(started) == epochs:
            raise Interrupted
        started.append(True)
        return train_one_epoch(**kwargs)

    return wrapped


def scripted(scores):
    """`validate`, with the per-study MAE that selects the probe taken from `scores`, one row of
    head scores per epoch."""
    rows, validate = iter(scores), multi.validate

    def scored(**kwargs):
        return dict(validate(**kwargs), study_mae=np.asarray(next(rows), dtype=float))

    return scored


def tiny_encoder(path, seed):
    torch.manual_seed(seed)
    encoder, _ = init_video_model(
        torch.device("cpu"), model_name="vit_tiny", crop_size=CROP, max_num_frames=FRAMES, pred_depth=4,
        pred_embed_dim=96, use_mask_tokens=True, return_all_tokens=True, **OPTIONS,
    )
    torch.save({"target_encoder": {f"module.{k}": v for k, v in encoder.state_dict().items()}}, path)
    return path


class TestRvspProbe(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = tmp.name
        rows, labels, probe = [], [], []
        for s, (split, rvsp, views) in enumerate(STUDIES):
            for v, (view, acq) in enumerate(views):
                path = write_video(os.path.join(cls.tmp, "videos", f"p{s}", f"s{s}", f"{v:04d}.mp4"),
                                   brightness=int(2 * rvsp) + 40 * v)
                rows.append((f"p{s}", f"s{s}", split, view, path, None, acq, "header", True))
            labels.append((f"s{s}", rvsp))
            probe.append((f"s{s}", split))
        inputs = os.path.join(cls.tmp, "inputs")
        os.makedirs(inputs)
        pd.DataFrame(rows, columns=list(mrm.INDEX_COLUMNS)).to_parquet(os.path.join(inputs, "videos.parquet"))
        pd.DataFrame(labels, columns=["study_id", "rvsp"]).to_parquet(os.path.join(inputs, "rvsp.parquet"))
        pd.DataFrame(probe, columns=["study_id", "split"]).to_csv(os.path.join(inputs, "probe.csv"), index=False)
        with open(os.path.join(inputs, "slots.yaml"), "w") as f:
            f.write(SLOTS)
        cls.build_args = ["--video-index", os.path.join(inputs, "videos.parquet"), "--labels",
                          os.path.join(inputs, "rvsp.parquet"), "--probe-splits", os.path.join(inputs, "probe.csv"),
                          "--slots", os.path.join(inputs, "slots.yaml")]
        cls.probe_dir = cls.build("probe")
        with open(os.path.join(cls.probe_dir, "probe_info.json")) as f:
            cls.info = json.load(f)
        cls.checkpoint = tiny_encoder(os.path.join(cls.tmp, "encoder.pth.tar"), seed=0)

    @classmethod
    def build(cls, name, *extra):
        out = os.path.join(cls.tmp, name)
        with contextlib.redirect_stdout(io.StringIO()):
            mrm.main([*cls.build_args, "--out-dir", out, *extra])
        return out

    def config(self, name, epochs=2, probe_dir=None, **changes):
        probe_dir = probe_dir or self.probe_dir
        data = {
            "dataset_type": "VideoGroupDataset",
            "dataset_train": os.path.join(probe_dir, "train.csv"),
            "dataset_val": os.path.join(probe_dir, "val.csv"),
            "dataset_test": os.path.join(probe_dir, "test.csv"),
            "video_index": os.path.join(probe_dir, "video_index.csv"),
            "resolution": CROP, "frames_per_clip": FRAMES, "frame_step": 2, "num_segments": 2,
            "num_views_per_segment": 1, "target_mean": self.info["target_mean"], "target_std": self.info["target_std"],
        }
        data.update(changes.pop("data", {}))
        cfg = {
            "folder": os.path.join(self.tmp, "runs", name),
            "tag": "rvsp",
            "num_workers": 0,
            "resume_checkpoint": True,
            "meta": {"seed": changes.pop("seed", 0)},
            "experiment": {
                "classifier": {"task_type": "regression", "probe_type": "attentive", "fusion": "early",
                               "num_heads": 4, "num_probe_blocks": 2, "num_targets": 1},
                "data": data,
                "evaluation": {"selection": "study", "rvsp_ranges": [30, 40, 50, 60]},
                "optimization": {"batch_size": 2, "num_epochs": epochs, "use_bfloat16": False,
                                 "multihead_kwargs": GRID},
            },
            "model_kwargs": {
                "checkpoint": changes.pop("checkpoint", self.checkpoint),
                "module_name": "evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder",
                "pretrain_kwargs": {"encoder": {"model_name": "vit_tiny", "checkpoint_key": "target_encoder",
                                                **OPTIONS}},
                "wrapper_kwargs": {},
            },
        }
        assert not changes, changes
        return cfg

    def folder(self, cfg):
        return os.path.join(cfg["folder"], multi.EVAL_NAME, cfg["tag"])

    def run_probe(self, cfg, test=False):
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(multi.mp, "set_start_method"), contextlib.redirect_stdout(io.StringIO()):
            multi.main(dict(cfg, test_only=test))

    def result(self, cfg):
        with open(os.path.join(self.folder(cfg), "test_metrics.json")) as f:
            return json.load(f)

    def load(self, cfg, name):
        return torch.load(os.path.join(self.folder(cfg), name), map_location="cpu", weights_only=False)

    # -- end to end ------------------------------------------------------------------------------

    def test_the_probe_runs_end_to_end_and_repeats_exactly(self):
        results = []
        for name in ("first", "second"):
            cfg = self.config(name)
            self.run_probe(cfg)
            self.run_probe(cfg, test=True)
            results.append((cfg, self.result(cfg), self.load(cfg, "best.pt")))
        (cfg, first, best), (_, second, second_best) = results

        # One prediction per test study, in mmHg, from the epoch and head chosen on validation.
        head = int(np.argmin(best["val_acc_per_head"]))
        self.assertEqual((first["epoch"], first["head"]), (best["best_epoch"], head))
        self.assertEqual(first["val_study_mae"], best["val_acc_per_head"][head])
        self.assertEqual(best["selection_metric"], "study_mae")
        predictions = pd.read_csv(os.path.join(self.folder(cfg), "test_predictions.csv"), dtype={"study_id": str})
        test = pd.read_csv(os.path.join(self.probe_dir, "test.csv"), dtype=str)
        self.assertEqual(list(predictions.study_id), list(test.study_id))
        # Studies in manifest order (s10, s11, s9); missing slots are not counted as videos.
        self.assertEqual(list(predictions.videos), [2, 1, 3])
        np.testing.assert_allclose(predictions.label, [50.0, 70.0, 32.0], rtol=1e-6)
        self.assertAlmostEqual(first["test"]["mae"], predictions.abs_error.mean(), places=6)
        self.assertEqual((first["test"]["studies"], first["test"]["failed_studies"], first["test"]["failed_videos"]),
                         (3, 0, 0))
        self.assertEqual(first["val"]["studies"], 3)
        self.assertEqual(sum(r["n"] for r in first["test_by_reference_range"]["rows"]), 3)
        self.assertEqual([r["videos"] for r in first["test_by_videos_present"]], [1, 2, 3])
        self.assertEqual(first["run_info"]["parameters"]["probe_heads"], len(GRID))
        self.assertEqual(first["run_info"]["positions"], ["A4C_bmode[0]", "A4C_color[0]", "RVINF[0]"])
        self.assertEqual(first["run_info"]["test_manifest_sha256"], self.info["outputs"]["test.csv"]["sha256"])
        self.assertNotIn(self.probe_dir, json.dumps(first["run_info"]))  # no path in the run record

        # Same seed, same probe and numbers (NaN for empty ranges, so compare as JSON).
        for key in ("epoch", "head", "val_study_mae", "val", "test", "test_by_reference_range"):
            self.assertEqual(json.dumps(first[key], sort_keys=True), json.dumps(second[key], sort_keys=True), key)
        for a, b in zip(best["classifiers"], second_best["classifiers"]):
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_a_resumed_probe_continues_as_an_uninterrupted_one(self):
        def train(name, interrupt):
            cfg = self.config(name)
            if interrupt:  # killed after epoch 1, then the same command again
                with mock.patch.object(multi, "train_one_epoch", side_effect=stop_after(1, multi.train_one_epoch)), \
                        self.assertRaises(Interrupted):
                    self.run_probe(cfg)
                self.assertEqual(self.load(cfg, "latest.pt")["epoch"], 1)
            self.run_probe(cfg)
            with open(os.path.join(self.folder(cfg), "log_r0.csv")) as f:
                return f.read(), self.load(cfg, "latest.pt")

        log, latest = train("uninterrupted", interrupt=False)
        resumed_log, resumed = train("resumed", interrupt=True)
        self.assertEqual(resumed_log, log)
        self.assertEqual(resumed["val_acc_per_head"], latest["val_acc_per_head"])
        for a, b in zip(latest["classifiers"], resumed["classifiers"]):
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_the_late_averaging_baseline_runs_as_an_option(self):
        cfg = self.config("late", epochs=1)
        cfg["experiment"]["classifier"]["fusion"] = "late"
        self.run_probe(cfg)
        self.run_probe(cfg, test=True)
        result = self.result(cfg)
        self.assertEqual(result["aggregation"], multi.AGGREGATION["late"])
        self.assertEqual(result["test"]["studies"], 3)
        self.assertEqual(self.load(cfg, "best.pt")["fingerprint"]["settings"]["classifier"]["fusion"], "late")

    def test_wandb_runs_are_private_and_the_test_rejoins_the_run(self):
        fake = probe_wandb.ProbeFakeWandb()
        cfg = self.config("wandb", epochs=1)
        cfg["meta"].update(wandb_project="project", wandb_entity="entity")
        with mock.patch.object(wandb_logging, "wandb", fake):
            self.run_probe(cfg)
            self.run_probe(cfg, test=True)
        trained, tested = fake.runs
        self.assertEqual(fake.calls[0]["settings"], provenance.PRIVATE_WANDB_SETTINGS)
        self.assertEqual((fake.calls[1]["id"], fake.calls[1]["resume"]), (trained.id, "must"))

        def strings(value):
            if isinstance(value, dict):
                return [s for v in value.values() for s in strings(v)]
            if isinstance(value, (list, tuple)):
                return [s for v in value for s in strings(v)]
            return [value] if isinstance(value, str) else []

        for logged in (fake.calls[0]["config"], trained.logged, tested.logged, tested.summary):
            self.assertFalse([s for s in strings(logged) if self.tmp in s or os.path.isabs(s)])
        heads = [f"head{i}_lr{g['lr']:g}_wd{g['weight_decay']:g}" for i, g in enumerate(GRID)]
        epoch = trained.logged[0]
        self.assertEqual(epoch["probe/val_study_mae"], min(epoch[f"probe/val_study_mae/{h}"] for h in heads))
        result = self.result(cfg)
        scalars, plots = tested.logged
        self.assertEqual(scalars["probe/test/study_mae"], result["test"]["mae"])
        self.assertEqual(len(plots["probe/test/study/residuals"][1]), 3)  # values only, no identifier
        self.assertEqual(tested.summary["probe_checkpoint/sha256"], result["checkpoint_sha256"])

    # -- selection ------------------------------------------------------------------------------

    def test_the_epoch_and_head_with_the_lowest_study_mae_are_tested(self):
        for scores, (epoch, head, mae) in [
            ([[5.0, 3.0], [4.0, 6.0], [2.5, 7.0]], (3, 0, 2.5)),
            ([[5.0, 3.0], [4.0, 6.0], [3.5, 3.2]], (1, 1, 3.0)),  # not the last epoch, nor its best head
        ]:
            with self.subTest(scores=scores):
                cfg = self.config(f"scripted-{epoch}", epochs=3)
                with mock.patch.object(multi, "validate", side_effect=scripted(scores)):
                    self.run_probe(cfg)
                best = self.load(cfg, "best.pt")
                self.assertEqual((best["epoch"], best["best_epoch"], best["val_acc_per_head"]),
                                 (epoch, epoch, scores[epoch - 1]))
                with open(os.path.join(self.folder(cfg), "log_r0.csv")) as f:
                    logged = pd.read_csv(f)
                self.assertEqual(list(logged.val_study_mae), [min(s) for s in scores])
                self.run_probe(cfg, test=True)
                result = self.result(cfg)
                self.assertEqual((result["epoch"], result["head"], result["val_study_mae"]), (epoch, head, mae))

    def test_validation_mae_is_per_study_in_mmhg(self):
        labels = np.array([[0.0], [1.0], [-1.0]])
        predictions = np.stack([labels, np.ones_like(labels)])
        scored = (np.arange(3), labels, predictions, np.array([3, 1, 2]), [], 0)
        with mock.patch.object(multi, "score_studies", return_value=scored):
            val = multi.validate(device=None, encoder=None, classifiers=[None, None], dataset=None, batch_size=1,
                                 num_workers=0, use_bfloat16=False, target_mean=40.0, target_std=10.0, rank=0,
                                 world_size=1)
        # Head 1 predicts 50 mmHg for studies of 40, 50 and 30 mmHg.
        np.testing.assert_allclose(val["study_mae"], [0.0, (10.0 + 0.0 + 20.0) / 3])
        self.assertEqual((val["studies"], val["failed_studies"]), (3, 0))

    def test_the_selection_record_keeps_the_best_epoch(self):
        selection = multi.Selection()
        self.assertEqual([selection.update(e, s) for e, s in enumerate([[4.0, 5.0], [4.5, 3.0], [3.0, 6.0]], 1)],
                         [True, True, False])  # epoch 3's best head ties epoch 2's: the earlier is kept
        record = selection.record([3.0, 6.0])
        self.assertEqual((record["best_epoch"], record["best_val_acc"]), (2, 3.0))
        self.assertEqual(record["best_epoch_per_head"], [3, 2])
        self.assertEqual(record["best_val_acc_per_head"], [3.0, 3.0])
        resumed = multi.Selection({**record}, epochs_done=3)
        self.assertFalse(resumed.update(4, [3.5, 3.5]))
        self.assertAlmostEqual(resumed.record([3.5, 3.5])["mean_val_acc"], (4.0 + 3.0 + 3.0 + 3.5) / 4)

    # -- fingerprint ----------------------------------------------------------------------------

    def interrupted(self, name, **changes):
        cfg = self.config(name, **changes)
        with mock.patch.object(multi, "train_one_epoch", side_effect=stop_after(1, multi.train_one_epoch)), \
                self.assertRaises(Interrupted):
            self.run_probe(cfg)
        return cfg

    def test_a_probe_is_resumed_only_with_its_encoder_seed_and_manifests(self):
        other = tiny_encoder(os.path.join(self.tmp, "other-encoder.pth.tar"), seed=1)
        reslotted = self.build("reslotted", "--seed", "7")
        for changes, field in [({"checkpoint": other}, "encoder_sha256"), ({"seed": 1}, "settings.seed"),
                               ({"probe_dir": reslotted}, "manifests_sha256.probe_info")]:
            with self.subTest(field):
                cfg = self.interrupted(f"resume-{field}")
                changed = self.config(f"resume-{field}", **changes)
                with self.assertRaisesRegex(ValueError, field.replace(".", "\\.")):
                    self.run_probe(changed)
                self.assertEqual(self.load(cfg, "latest.pt")["epoch"], 1)  # left as it was

    def test_a_probe_is_tested_only_with_its_encoder(self):
        cfg = self.config("tested-other", epochs=1)
        self.run_probe(cfg)
        other = tiny_encoder(os.path.join(self.tmp, "tested-other-encoder.pth.tar"), seed=2)
        with self.assertRaisesRegex(ValueError, "encoder_sha256"):
            self.run_probe(self.config("tested-other", epochs=1, checkpoint=other), test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(cfg), "test_metrics.json")))

    # -- test isolation -------------------------------------------------------------------------

    def test_training_and_validation_never_open_the_test_manifest(self):
        cfg = self.config("isolated", epochs=1)
        opened, real_open = [], builtins.open

        def recording(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)):
                opened.append(os.path.realpath(file))
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=recording):
            self.run_probe(cfg)
        test_manifest = os.path.realpath(cfg["experiment"]["data"]["dataset_test"])
        self.assertIn(os.path.realpath(cfg["experiment"]["data"]["dataset_val"]), opened)  # the recording works
        self.assertNotIn(test_manifest, opened)

        # Nor does training need it to exist.
        missing = self.config("no-test-manifest", epochs=1,
                              data={"dataset_test": os.path.join(self.tmp, "absent.csv")})
        self.run_probe(missing)
        self.assertTrue(os.path.exists(os.path.join(self.folder(missing), "best.pt")))

        # The test opens it, once.
        opened.clear()
        with mock.patch("builtins.open", side_effect=recording):
            self.run_probe(cfg, test=True)
        self.assertIn(test_manifest, opened)
        with self.assertRaisesRegex(RuntimeError, "already been tested"):
            self.run_probe(cfg, test=True)

    def test_only_a_finished_probe_is_tested(self):
        cfg = self.interrupted("unfinished")
        with self.assertRaisesRegex(RuntimeError, "finished 1 of 2 epochs"):
            self.run_probe(cfg, test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(cfg), "test_metrics.json")))

    def test_only_the_test_manifest_written_with_the_training_ones_is_tested(self):
        cfg = self.config("other-test", epochs=1)
        self.run_probe(cfg)
        altered = os.path.join(self.tmp, "altered-test.csv")
        test = pd.read_csv(cfg["experiment"]["data"]["dataset_test"], dtype=str)
        test.iloc[:2].to_csv(altered, index=False)  # a study fewer
        cfg["experiment"]["data"]["dataset_test"] = altered
        with self.assertRaisesRegex(ValueError, "test.csv written with the training manifests"):
            self.run_probe(cfg, test=True)
        self.assertFalse(os.path.exists(os.path.join(self.folder(cfg), "test_metrics.json")))

    def test_a_manifest_listing_another_splits_videos_is_refused(self):
        leaky = os.path.join(self.tmp, "leaky")
        shutil.copytree(self.probe_dir, leaky)
        train = pd.read_csv(os.path.join(leaky, "train.csv"), dtype=str)
        val = pd.read_csv(os.path.join(leaky, "val.csv"), dtype=str)
        train.loc[0, "A4C_bmode[0]"] = val.loc[0, "A4C_bmode[0]"]  # a validation video in training
        train.to_csv(os.path.join(leaky, "train.csv"), index=False)
        with self.assertRaisesRegex(ValueError, "listed under another split"):
            self.run_probe(self.config("leaky", epochs=1, probe_dir=leaky))

    # -- validation-only scoring ----------------------------------------------------------------

    def score(self, cfg, **options):
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(multi.mp, "set_start_method"), contextlib.redirect_stdout(io.StringIO()):
            return score_validation.score_validation(cfg, **options)

    def scored_outputs(self, cfg):
        names = (multi.VAL_PREDICTIONS, multi.VAL_METRICS)
        return [n for n in names if os.path.exists(os.path.join(self.folder(cfg), n))]

    def test_validation_scoring_gives_the_tests_validation_predictions_without_the_test_split(self):
        cfg = self.config("scored")
        self.run_probe(cfg)
        opened, real_open = [], builtins.open

        def recording(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)):
                opened.append(os.path.realpath(file))
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=recording), mock.patch.object(multi, "init_wandb") as wandb:
            result = self.score(cfg)
        self.assertIn(os.path.realpath(cfg["experiment"]["data"]["dataset_val"]), opened)
        self.assertNotIn(os.path.realpath(cfg["experiment"]["data"]["dataset_test"]), opened)
        wandb.assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(self.folder(cfg), "test_metrics.json")))

        # The epoch and head chosen on validation; on CPU the recomputed MAE is the stored one.
        best = self.load(cfg, "best.pt")
        head = int(np.argmin(best["val_acc_per_head"]))
        self.assertEqual((result["epoch"], result["head"], result["val_study_mae"]),
                         (best["epoch"], head, best["val_acc_per_head"][head]))
        self.assertAlmostEqual(result["stored_score_check"]["difference"], 0.0, places=6)
        with open(os.path.join(self.folder(cfg), multi.VAL_METRICS)) as f:
            self.assertEqual(json.dumps(json.load(f), sort_keys=True), json.dumps(result, sort_keys=True))
        scored = pd.read_csv(os.path.join(self.folder(cfg), multi.VAL_PREDICTIONS), dtype={"study_id": str})
        val = pd.read_csv(os.path.join(self.probe_dir, "val.csv"), dtype=str)
        self.assertEqual(list(scored.study_id), list(val.study_id))
        self.assertAlmostEqual(scored.abs_error.mean(), result["val"]["mae"], places=6)
        self.assertEqual(sum(r["n"] for r in result["val_by_reference_range"]["rows"]), len(val))

        # The test later scores validation with the same head: the same predictions.
        self.run_probe(cfg, test=True)
        tested = pd.read_csv(os.path.join(self.folder(cfg), "val_predictions.csv"), dtype={"study_id": str})
        pd.testing.assert_frame_equal(scored, tested)

    def test_validation_scoring_refuses_what_it_cannot_vouch_for(self):
        with self.subTest("unfinished probe"):
            cfg = self.interrupted("score-unfinished")
            with self.assertRaisesRegex(RuntimeError, "needs a finished probe"):
                self.score(cfg)
            self.assertEqual(self.scored_outputs(cfg), [])

        cfg = self.config("score-refusals", epochs=1)
        self.run_probe(cfg)
        with self.subTest("another encoder"):
            other = tiny_encoder(os.path.join(self.tmp, "scored-other-encoder.pth.tar"), seed=3)
            with self.assertRaisesRegex(ValueError, "encoder_sha256"):
                self.score(self.config("score-refusals", epochs=1, checkpoint=other))
            self.assertEqual(self.scored_outputs(cfg), [])

        with self.subTest("scores that are not the stored ones"):
            score_split = multi.score_split

            def shifted(*args, **kwargs):
                table, block, failed = score_split(*args, **kwargs)
                return table, dict(block, mae=block["mae"] + 0.5), failed

            with mock.patch.object(multi, "score_split", side_effect=shifted), \
                    self.assertRaisesRegex(RuntimeError, "differs from"):
                self.score(cfg)
            self.assertEqual(self.scored_outputs(cfg), [])

        with self.subTest("outputs already written"):
            first = self.score(cfg)
            with self.assertRaisesRegex(RuntimeError, "already exist"):
                self.score(cfg)
            again = self.score(cfg, overwrite=True)  # NaN for empty ranges: compare as JSON
            self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(first, sort_keys=True))

        with self.subTest("with the test"), self.assertRaisesRegex(ValueError, "separate runs"), \
                mock.patch("torch.cuda.is_available", return_value=False):
            multi.main(dict(cfg, test_only=True, score_validation={"overwrite": True}))


class TestShippedConfig(unittest.TestCase):

    def test_the_rvsp_probe_reads_and_trains_as_the_ef_probe(self):
        import yaml

        from evals import scaffold

        configs = {}
        for name in ("lvef-a4c-224px-16f", "rvsp-multiview-224px-16f"):
            with open(os.path.join(REPO, "configs", "eval", "vitb16", f"{name}.yaml")) as f:
                configs[name] = yaml.safe_load(f)
        ef, rvsp = configs["lvef-a4c-224px-16f"], configs["rvsp-multiview-224px-16f"]
        self.assertEqual(rvsp["model_kwargs"], ef["model_kwargs"])  # the same frozen encoder wrapper
        self.assertEqual(rvsp["experiment"]["optimization"], ef["experiment"]["optimization"])
        self.assertEqual(rvsp["meta"]["seed"], ef["meta"]["seed"])
        for key in ("num_heads", "num_probe_blocks", "num_targets", "task_type", "probe_type"):
            self.assertEqual(rvsp["experiment"]["classifier"][key], ef["experiment"]["classifier"][key], key)
        for key in ("resolution", "frames_per_clip", "frame_step", "num_segments", "num_views_per_segment"):
            self.assertEqual(rvsp["experiment"]["data"][key], ef["experiment"]["data"][key], key)
        self.assertEqual((rvsp["experiment"]["classifier"]["fusion"], rvsp["experiment"]["evaluation"]["selection"]),
                         ("early", "study"))
        # `evals.main` reaches the probe through the scaffold, which imports `evals.<eval_name>.eval`.
        with mock.patch("importlib.import_module", return_value=mock.Mock()) as imported:
            scaffold.main(rvsp["eval_name"], args_eval={})
        imported.assert_called_once_with("evals.video_classification_frozen_multi.eval")
        self.assertTrue(callable(multi.main))


if __name__ == "__main__":
    unittest.main()
