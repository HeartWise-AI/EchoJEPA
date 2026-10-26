# tests/utils/test_study_probe_smoke.py

"""An end-to-end smoke run of the study-level multi-task probe on CPU: tiny real videos in a long
manifest (several videos per study, a study without one target, a study with one unreadable
video), a tiny V-JEPA 2.1 encoder, two epochs with two probes, a resume, and the test. Run twice
with the same seed, it gives the same probe and the same numbers."""

import contextlib
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from app.vjepa_2_1.utils import init_video_model
from evals.study_multitask_frozen import eval as probe
from tests.utils.test_probe_wandb import GRID, Interrupted
from tests.utils.test_vjepa_2_1_eval_encoder import OPTIONS
from tests.utils.test_vjepa_2_1_utils import QuietLogs

FRAMES, CROP = 4, 32
# (split, EF, AS present, videos) of each study.
STUDIES = [("train", 35, 1, 3), ("train", 50, 0, 2), ("train", 60, 0, 4), ("train", 65, 1, 1),
           ("train", 40, 0, 3), ("train", 55, 1, 2),
           ("val", 35, 1, 2), ("val", 55, 0, 3), ("val", None, 0, 2),
           ("test", 30, 1, 2), ("test", 55, 0, 1), ("test", 60, None, 3)]


def stop_after(epochs, train_one_epoch):
    """`train_one_epoch`, raising `Interrupted` when training epoch `epochs + 1` would start."""
    started = []

    def wrapped(**kwargs):
        if len(started) == epochs:
            raise Interrupted
        started.append(True)
        return train_one_epoch(**kwargs)

    return wrapped


def write_video(path, brightness, frames=12, fps=8):
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CROP, CROP))
    for i in range(frames):
        writer.write(np.full((CROP, CROP, 3), (brightness + 3 * i) % 256, np.uint8))
    writer.release()


class TestStudyProbeSmoke(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = tmp.name
        rows = []
        for s, (split, ef, as_present, videos) in enumerate(STUDIES):
            for v in range(videos):
                path = os.path.join(cls.tmp, "videos", f"s{s}", f"{v:04d}.mp4")
                if (s, v) == (7, 2):  # an unreadable validation video: its slot is padded.
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(b"broken")
                else:
                    write_video(path, brightness=2 * (ef or 50) + 20 * v)
                rows.append({"video_path": path, "study_id": f"s{s}", "view": "A4C" if v % 2 == 0 else "A2C",
                             "split": split, "lvef": ef, "as_present": None if as_present is None else
                             ("yes" if as_present else "no")})
        cls.manifest = os.path.join(cls.tmp, "videos.csv")
        pd.DataFrame(rows).to_csv(cls.manifest, index=False)

        torch.manual_seed(0)
        encoder, _ = init_video_model(
            torch.device("cpu"), model_name="vit_tiny", crop_size=CROP, max_num_frames=FRAMES, pred_depth=4,
            pred_embed_dim=96, use_mask_tokens=True, return_all_tokens=True, **OPTIONS,
        )
        cls.checkpoint = os.path.join(cls.tmp, "encoder.pth.tar")
        torch.save({"target_encoder": {f"module.{k}": v for k, v in encoder.state_dict().items()}}, cls.checkpoint)

    def config(self, folder, pooling_mode="attentive", selection="weighted_loss", view_embedding=False,
               trainable_blocks=0):
        return {
            "folder": folder, "tag": "study", "num_workers": 0, "resume_checkpoint": True, "meta": {"seed": 0},
            "experiment": {
                "data": {"data_filename": self.manifest, "split_column": "split", "groupby_column": "study_id",
                         "num_videos": 3, "shuffle_videos": True, "num_clips_per_video": 1,
                         "frames_per_clip": FRAMES, "frame_step": 2, "resolution": CROP, "view_column": "view",
                         "view_labels_map": {"A4C": 0, "A2C": 1} if view_embedding else None},
                "encoder": {"trainable_blocks": trainable_blocks, "lr": 1e-4, "weight_decay": 0.01},
                "classifier": {"pooling_mode": pooling_mode, "num_heads": 4, "num_probe_blocks": 1,
                               "use_view_embedding": view_embedding,
                               "target_label": ["lvef", "as_present"],
                               "head_task": {"lvef": "regression", "as_present": "binary"},
                               "head_weights": {"lvef": 1.0, "as_present": 0.5},
                               "labels_map": {"as_present": {"no": 0, "yes": 1}}},
                "evaluation": {"selection": selection},
                "optimization": {"batch_size": 2, "num_epochs": 2, "use_bfloat16": False,
                                 "multihead_kwargs": GRID[:1] if trainable_blocks else GRID},
            },
            "model_kwargs": {
                "checkpoint": self.checkpoint,
                "module_name": "evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder",
                "pretrain_kwargs": {"encoder": {"model_name": "vit_tiny", "checkpoint_key": "target_encoder",
                                                **OPTIONS}},
                "wrapper_kwargs": {},
            },
        }

    @contextlib.contextmanager
    def on_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(probe.mp, "set_start_method"):
            yield

    def folder(self, cfg):
        return os.path.join(cfg["folder"], probe.EVAL_NAME, cfg["tag"])

    def train_and_test(self, name, **options):
        cfg = self.config(os.path.join(self.tmp, name), **options)
        with self.on_cpu():
            probe.main(dict(cfg))
            probe.main(dict(cfg, test_only=True))
        with open(os.path.join(self.folder(cfg), "test_metrics.json")) as f:
            result = json.load(f)
        best = torch.load(os.path.join(self.folder(cfg), "best.pt"), map_location="cpu", weights_only=False)
        return cfg, result, best

    def test_the_probe_runs_end_to_end_and_repeats_exactly(self):
        cfg, first, first_best = self.train_and_test("first")
        _, second, second_best = self.train_and_test("second")

        # Every study is scored once; a study without a target is left out of that head only.
        self.assertEqual((first["test"]["studies"], first["test"]["failed"]), (3, 0))
        self.assertEqual((first["test"]["heads"]["lvef"]["n"], first["test"]["heads"]["as_present"]["n"]), (3, 2))
        self.assertEqual((first["val"]["heads"]["lvef"]["n"], first["val"]["heads"]["as_present"]["n"]), (2, 3))
        for key in ("mae", "rmse", "pearson_r", "loss"):
            self.assertIn(key, first["test"]["heads"]["lvef"])
        for key in ("auroc", "auprc", "acc", "loss"):
            self.assertIn(key, first["test"]["heads"]["as_present"])
        self.assertEqual(first["run_info"]["parameters"]["probes"], len(GRID))
        self.assertEqual(first["run_info"]["splits"]["train"]["studies"], 6)
        self.assertEqual(first["run_info"]["splits"]["train"]["selection"], probe.SELECTION["random"])
        self.assertEqual(first["run_info"]["splits"]["val"]["selection"], probe.SELECTION["seeded_random"])
        # The train split's statistics normalize the EF head.
        train_ef = [ef for split, ef, _, _ in STUDIES if split == "train"]
        self.assertAlmostEqual(first["normalization"]["lvef"]["mean"], float(np.mean(train_ef)))
        self.assertAlmostEqual(first["normalization"]["lvef"]["std"], float(np.std(train_ef)))

        folder = self.folder(cfg)
        predictions = pd.read_csv(os.path.join(folder, "test_predictions.csv"))
        self.assertEqual(list(predictions.study_id), ["s10", "s11", "s9"])
        self.assertEqual(list(predictions.columns), ["study_id", "lvef_label", "lvef_prediction", "as_present_label",
                                                     "as_present_prediction"])
        self.assertTrue(((predictions.as_present_prediction >= 0) & (predictions.as_present_prediction <= 1)).all())
        for name in ("params-probe.yaml", "run_info.json", "log_r0.csv", "latest.pt", "val_predictions.csv"):
            self.assertTrue(os.path.exists(os.path.join(folder, name)), name)

        # Same seed, same probe and the same numbers.
        for key in ("epoch", "probe", "val_score", "val", "test", "normalization"):
            self.assertEqual(json.dumps(first[key], sort_keys=True), json.dumps(second[key], sort_keys=True), key)
        for a, b in zip(first_best["probes"], second_best["probes"]):
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)

        # The test runs once, and only with the data the probe was trained on.
        with self.on_cpu(), self.assertRaisesRegex(RuntimeError, "already been tested"):
            probe.main(dict(cfg, test_only=True))
        os.remove(os.path.join(folder, "test_metrics.json"))
        other = dict(cfg)
        other["experiment"] = {**cfg["experiment"], "data": {**cfg["experiment"]["data"], "num_videos": 2}}
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "num_videos"):
            probe.main(dict(other, test_only=True))

    def test_a_resumed_probe_continues_as_an_uninterrupted_one(self):
        def train(name, interrupt):
            cfg = self.config(os.path.join(self.tmp, name))
            with self.on_cpu():
                if interrupt:  # killed after epoch 1, then the same command again
                    with mock.patch.object(probe, "train_one_epoch",
                                           side_effect=stop_after(1, probe.train_one_epoch)), \
                            self.assertRaises(Interrupted):
                        probe.main(dict(cfg))
                probe.main(dict(cfg))
            with open(os.path.join(self.folder(cfg), "log_r0.csv")) as f:
                log = f.read()
            return log, torch.load(os.path.join(self.folder(cfg), "latest.pt"), map_location="cpu",
                                   weights_only=False)

        log, latest = train("uninterrupted", interrupt=False)
        resumed_log, resumed = train("resumed", interrupt=True)
        self.assertEqual(resumed_log, log)
        self.assertEqual(resumed["val_score_per_probe"], latest["val_score_per_probe"])
        for a, b in zip(latest["probes"], resumed["probes"]):
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_other_pooling_modes_and_a_metric_selection(self):
        for mode, selection in (("mean", "lvef/mae"), ("attention", "as_present/auroc"), ("max", "weighted_loss")):
            _, result, best = self.train_and_test(f"{mode}-{selection.replace('/', '-')}", pooling_mode=mode,
                                                  selection=selection)
            self.assertEqual(result["selection"], selection)
            self.assertEqual(result["test"]["studies"], 3)
            if selection == "as_present/auroc":
                self.assertEqual(best["best_score"], -best["val"][best["best_probe"]]["heads"]["as_present"]["auroc"])

    def test_shared_studies_and_unreported_metrics_are_refused(self):
        table = pd.read_csv(self.manifest)
        leaked = table.copy()
        leaked.loc[leaked.study_id == "s6", "split"] = ["train", "val"]  # s6 has two videos.
        path = os.path.join(self.tmp, "leaked.csv")
        leaked.to_csv(path, index=False)
        cfg = self.config(os.path.join(self.tmp, "leaked"))
        cfg["experiment"]["data"]["data_filename"] = path
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "both the train and the val split"):
            probe.main(cfg)
        for selection, message in (("lvef/auroc", "regression head"), ("as_present/mae", "binary head"),
                                   ("lvef", "<head>/<metric>"), ("rvsp/mae", "target_label")):
            with self.on_cpu(), self.assertRaisesRegex(ValueError, message):
                probe.main(self.config(os.path.join(self.tmp, "bad-selection"), selection=selection))

    def test_view_embedding_and_encoder_fine_tuning(self):
        cfg, result, best = self.train_and_test("finetune", view_embedding=True, trainable_blocks=1)
        self.assertEqual(result["run_info"]["parameters"]["encoder_trainable_blocks"], 1)
        self.assertGreater(result["run_info"]["parameters"]["encoder_trainable"], 0)
        self.assertEqual(result["run_info"]["parameters"]["probes"], 1)
        self.assertIsNotNone(best["encoder"])
        self.assertEqual(len(best["opt"]), 2)  # the probe's optimizer, then the encoder's.
        self.assertIn("view_embed.weight", best["probes"][0])
        self.assertEqual(tuple(best["probes"][0]["view_embed.weight"].shape)[0], 3)  # 2 views + PAD
        # The top block moved, the bottom block did not.
        initial = {k.removeprefix("module.").removeprefix("backbone."): v
                   for k, v in torch.load(self.checkpoint, map_location="cpu")["target_encoder"].items()}
        moved = {k: not torch.equal(v, initial[k.removeprefix("model.")])
                 for k, v in best["encoder"].items() if k.startswith("model.blocks.")}
        self.assertTrue(any(m for k, m in moved.items() if ".blocks.11." in k or k.startswith("model.blocks.1.")))
        self.assertFalse(any(m for k, m in moved.items() if k.startswith("model.blocks.0.")))
        # Several probes cannot share a training encoder.
        other = self.config(os.path.join(self.tmp, "finetune-grid"), trainable_blocks=1)
        other["experiment"]["optimization"]["multihead_kwargs"] = GRID
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "one setting"):
            probe.main(other)

    def test_all_four_options_train_resume_and_score_selected_validation_without_test(self):
        cfg = self.config(os.path.join(self.tmp, "all-four"), view_embedding=True)
        cfg["experiment"]["classifier"].update(
            pooling_mode="hierarchical", query_mode="task", local_depth=1,
            local_num_queries=2, local_use_rope=True, view_dropout=0.2)
        cfg["experiment"]["data"].update(sampling_mode="view_balanced", eval_seed=13, num_clips_per_video=2,
                                            view_probability_columns={"A4C": "p4", "A2C": "p2"},
                                            acquisition_column="type", acquisition_labels_map={"bmode": 0, "color": 1})
        cfg["experiment"]["evaluation"]["num_bags"] = 2
        table = pd.read_csv(self.manifest)
        table["p4"] = np.where(table.view == "A4C", 0.9, 0.1)
        table["p2"] = 1 - table.p4
        table["type"] = np.where(table.view == "A4C", "bmode", "color")
        table["patient_id"] = "patient-" + table.study_id
        path = os.path.join(self.tmp, "soft-views.csv")
        table.to_csv(path, index=False)
        cfg["experiment"]["data"]["data_filename"] = path
        original_read = probe.read_split

        def no_test(data, split):
            self.assertNotEqual(split, "test")
            return original_read(data, split)

        with self.on_cpu(), mock.patch.object(probe, "read_split", side_effect=no_test):
            with mock.patch.object(probe, "train_one_epoch", side_effect=stop_after(1, probe.train_one_epoch)), \
                    self.assertRaises(Interrupted):
                probe.main(cfg)
            probe.main(cfg)
            probe.main(dict(cfg, val_only=True))
        folder = self.folder(cfg)
        self.assertFalse(os.path.exists(os.path.join(folder, "test_metrics.json")))
        with open(os.path.join(folder, "val_metrics.json")) as f:
            result = json.load(f)
        best = torch.load(os.path.join(folder, "best.pt"), weights_only=False)
        self.assertEqual(result["epoch"], best["best_epoch"])
        self.assertEqual(result["val"]["num_bags"], 2)
        self.assertTrue(np.isfinite(result["val"]["score"]))
        self.assertEqual(result["run_info"]["study_probe"]["query_mode"], "task")
        self.assertEqual(result["run_info"]["study_probe"]["token_grid"], [2, 2, 2])
        predictions = pd.read_csv(os.path.join(folder, "val_predictions.csv"))
        self.assertIn("lvef_bag_std", predictions)
        self.assertIn("as_present_bag_std", predictions)
        # Resume must reject changes in view conditioning, sampling, queries or local geometry.
        import copy
        for section, key, value in (("classifier", "view_dropout", 0.3), ("data", "eval_seed", 14),
                                    ("classifier", "query_mode", "shared"),
                                    ("classifier", "local_use_rope", False)):
            changed = copy.deepcopy(cfg)
            changed["experiment"][section][key] = value
            with self.on_cpu(), self.assertRaises((ValueError, RuntimeError)):
                probe.main(dict(changed, val_only=True))
