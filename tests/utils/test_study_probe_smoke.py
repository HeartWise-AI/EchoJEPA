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
from src.datasets.video_study_dataset import VideoStudyDataset
from src.models.study_probe import head_specs
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
        self.assertEqual(first["run_info"]["splits"]["val"]["selection"], probe.SELECTION["fixed"])
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

    def test_seeded_evaluation_bags_and_validation_only_scoring(self):
        # One video per bag, so every study with several videos is cut (s7: 3 videos, one unreadable).
        cfg = self.config(os.path.join(self.tmp, "seeded"))
        cfg["experiment"]["data"].update(eval_videos="seeded", eval_seed=7, num_videos=1)
        folder = self.folder(cfg)
        val_split = pd.read_csv(self.manifest).query("split == 'val'")
        bags = VideoStudyDataset(val_split, target_columns=["lvef"], num_videos=1, eval_videos="seeded", eval_seed=7,
                                 frames_per_clip=FRAMES, frame_step=2)
        self.assertTrue(any(len(paths) > 1 for _, paths, _, _ in bags.studies))
        self.assertTrue(all(len(bags.select(paths, s)) == 1 for s, paths, _, _ in bags.studies))
        with self.on_cpu():
            with self.assertRaisesRegex(RuntimeError, "finished 0 of 2"):
                probe.main(dict(cfg, val_only=True))
            probe.main(dict(cfg))
            probe.main(dict(cfg, val_only=True))
        with open(os.path.join(folder, "val_metrics.json")) as f:
            val = json.load(f)
        self.assertEqual(val["run_info"]["splits"]["val"]["selection"], probe.SELECTION["seeded"])
        self.assertEqual((val["val"]["studies"], val["val"]["failed"]), (3, 0))
        # The bags come from `eval_seed`, not `meta.seed`, and the fingerprint keeps both rules.
        self.assertEqual(val["training_fingerprint"]["settings"]["data"]["eval_seed"], 7)
        self.assertEqual(val["training_fingerprint"]["settings"]["data"]["eval_videos"], "seeded")
        self.assertFalse(os.path.exists(os.path.join(folder, "test_metrics.json")))  # the test split was not read.
        # The same bags as in training's validation: the selected score comes back.
        best = torch.load(os.path.join(folder, "best.pt"), map_location="cpu", weights_only=False)
        self.assertAlmostEqual(val["val"]["score"], best["best_score"], places=5)
        # The training provenance is the checkpoint's own (its GPU count included), kept apart from the scoring run's.
        self.assertEqual(json.dumps(val["training_fingerprint"], sort_keys=True),
                         json.dumps(best["fingerprint"], sort_keys=True))
        self.assertIn("scoring_fingerprint", val)
        val_predictions = pd.read_csv(os.path.join(folder, "val_predictions.csv"))
        self.assertEqual(list(val_predictions.study_id), ["s6", "s7", "s8"])

        # A probe is tested only with the bags it was trained with; the test repeats the validation predictions.
        first = dict(cfg)
        first["experiment"] = {**cfg["experiment"], "data": {**cfg["experiment"]["data"], "eval_videos": "first"}}
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "eval_videos"):
            probe.main(dict(first, test_only=True))
        other_seed = dict(cfg)
        other_seed["experiment"] = {**cfg["experiment"], "data": {**cfg["experiment"]["data"], "eval_seed": 8}}
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "eval_seed"):
            probe.main(dict(other_seed, test_only=True))
        with self.on_cpu():
            probe.main(dict(cfg, test_only=True))
        pd.testing.assert_frame_equal(pd.read_csv(os.path.join(folder, "val_predictions.csv")), val_predictions)
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "not both"):
            probe.main(dict(cfg, val_only=True, test_only=True))
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "eval_videos"):
            bad = dict(cfg)
            bad["experiment"] = {**cfg["experiment"], "data": {**cfg["experiment"]["data"], "eval_videos": "all"}}
            probe.main(bad)
        # A probe trained under protocol 1 (identifiers read as inferred, which can order numeric studies otherwise)
        # is not resumed or tested.
        latest = torch.load(os.path.join(folder, "latest.pt"), map_location="cpu", weights_only=False)
        latest["fingerprint"]["settings"]["protocol"]["version"] = 1
        torch.save(latest, os.path.join(folder, "latest.pt"))
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "earlier protocol"):
            probe.main(dict(cfg))

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
        # Two studies of one patient in the train and the val split; a named patient column that is missing.
        shared = table.assign(patient_id=table.study_id.map(lambda s: "p0" if s in ("s0", "s6") else f"p{s}"))
        path = os.path.join(self.tmp, "shared-patient.csv")
        shared.to_csv(path, index=False)
        cfg = self.config(os.path.join(self.tmp, "shared-patient"))
        cfg["experiment"]["data"]["data_filename"] = path
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "1 patients have studies in both the train and the val"):
            probe.main(cfg)
        cfg = self.config(os.path.join(self.tmp, "no-patient-column"))
        cfg["experiment"]["data"]["patient_column"] = "patient_id"
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "no `patient_id` column"):
            probe.main(cfg)
        # One file per split: a gap in the val file's patient column (which would read 123 as 123.0) and a val file
        # without the column are refused; patient 123 in both files is caught whatever the column's look.
        numbered = table.assign(patient_id=table.study_id.str[1:].astype(int) + 100)
        numbered.loc[numbered.study_id.isin(["s0", "s6"]), "patient_id"] = 123  # s0 (train), s6 (val): one patient
        paths = {}
        for split in ("train", "val", "test"):
            paths[split] = os.path.join(self.tmp, f"split-{split}.csv")
            numbered[numbered.split == split].to_csv(paths[split], index=False)
        def per_split(name, val_rows):
            path = os.path.join(self.tmp, f"{name}-val.csv")
            val_rows.to_csv(path, index=False)
            cfg = self.config(os.path.join(self.tmp, name))
            data = cfg["experiment"]["data"]
            data.pop("data_filename"), data.pop("split_column")
            data.update(dataset_train=paths["train"], dataset_val=path, dataset_test=paths["test"])
            return cfg
        val_rows = numbered[numbered.split == "val"]
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "1 patients have studies in both the train and the val"):
            probe.main(per_split("patients-shared", val_rows))
        gap = val_rows.copy()
        gap.loc[gap.study_id == "s7", "patient_id"] = np.nan
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "rows have no `patient_id`"):
            probe.main(per_split("patients-gap", gap))
        with self.on_cpu(), self.assertRaisesRegex(ValueError, r"The val split\(s\) have no `patient_id` column"):
            probe.main(per_split("patients-absent", val_rows.drop(columns=["patient_id"])))
        # Any split with the column turns the check on: a train file without it is refused too.
        cfg = per_split("patients-train-absent", val_rows)
        no_column = os.path.join(self.tmp, "train-without-patients.csv")
        numbered[numbered.split == "train"].drop(columns=["patient_id"]).to_csv(no_column, index=False)
        cfg["experiment"]["data"]["dataset_train"] = no_column
        with self.on_cpu(), self.assertRaisesRegex(ValueError, r"The train split\(s\) have no `patient_id` column"):
            probe.main(cfg)
        for selection, message in (("lvef/auroc", "regression head"), ("as_present/mae", "binary head"),
                                   ("lvef", "<head>/<metric>"), ("rvsp/mae", "target_label")):
            with self.on_cpu(), self.assertRaisesRegex(ValueError, message):
                probe.main(self.config(os.path.join(self.tmp, "bad-selection"), selection=selection))

    def test_non_finite_outputs_leave_every_metric_undefined(self):
        specs = head_specs({"target_label": ["lvef", "as_present"],
                            "head_task": {"lvef": "regression", "as_present": "binary"}})
        target = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        finite = np.array([[0.1], [0.2], [0.7], [0.9], [0.3], [0.8]])
        broken = finite.copy()
        broken[2] = np.nan  # ranked last, this would give AUROC 1.0
        ok = probe.head_metrics(specs["as_present"], target, finite)
        self.assertEqual((ok["auroc"], ok["nonfinite_outputs"]), (1.0, 0))
        bad = probe.head_metrics(specs["as_present"], target, broken)
        self.assertEqual(bad["nonfinite_outputs"], 1)
        self.assertTrue(all(np.isnan(bad[k]) for k in ("auroc", "auprc", "acc", "loss")))
        self.assertAlmostEqual(bad["prevalence"], 0.5)
        # An undefined metric cannot be selected, whatever the rule.
        norm = {"lvef": {"mean": 0.5, "std": 1.0}}
        scored = {"indices": np.arange(6), "targets": np.stack([target, target], axis=1), "failed": [],
                  "outputs": {"lvef": broken, "as_present": finite}}
        for selection in ("lvef/mae", "weighted_loss"):
            result = probe.evaluate(scored, specs, norm, probe.parse_selection(selection, specs))
            self.assertTrue(np.isnan(result["score"]), selection)
        # A broken head the rule does not read still keeps the probe from being selected.
        scored["outputs"] = {"lvef": finite, "as_present": np.full_like(finite, np.nan)}
        result = probe.evaluate(scored, specs, norm, probe.parse_selection("lvef/mae", specs))
        self.assertTrue(np.isfinite(result["heads"]["lvef"]["mae"]))
        self.assertTrue(np.isnan(result["score"]))

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
        # V-JEPA 2.1 has no `norm`: its output norm, norms_block[-1], trains with the top blocks; the others do not.
        norms = sorted({int(k.split(".")[2]) for k in best["encoder"] if k.startswith("model.norms_block.")})
        self.assertGreater(len(norms), 1)
        for i in norms:
            changed = not torch.equal(best["encoder"][f"model.norms_block.{i}.weight"], initial[f"norms_block.{i}.weight"])
            self.assertEqual(changed, i == norms[-1], f"norms_block.{i}")
        # Several probes cannot share a training encoder.
        other = self.config(os.path.join(self.tmp, "finetune-grid"), trainable_blocks=1)
        other["experiment"]["optimization"]["multihead_kwargs"] = GRID
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "one setting"):
            probe.main(other)
