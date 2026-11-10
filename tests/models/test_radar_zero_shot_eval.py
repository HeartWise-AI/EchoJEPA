# tests/models/test_radar_zero_shot_eval.py

"""Zero-shot evaluation of a checkpoint trained on the synthetic set, on synthetic evaluation studies of its own:

- the three outputs are written, with one prediction row per (study, task), and `metrics.json` is strict JSON;
- a study none of whose bags lets the task's structure read a clip has no prediction and is left out of the metrics,
  and the coverage counts say so; a study without a label is left out of that task only;
- the metrics are those of `zero_shot_metrics` on the labelled, supported rows of `predictions.csv`, and a prediction
  is the mean of the scorers of `src.models.radar.zero_shot` over the study's evaluation bags;
- the run is reproducible and its bags are the dataset's evaluation bags;
- thresholds come from a tuning split or are 0.5, never from the reported split;
- every refusal of the module docstring;
- intervals resample the groups of a label column on request;
- W&B receives scalars and settings, and no study identifier.

The synthetic tokens are random, so no metric value means anything here: the tests check the bookkeeping.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

from evals.radar_frozen import train as radar_train
from evals.radar_frozen import zero_shot_eval as zs
from evals.radar_frozen import zero_shot_metrics as zm
from src.datasets.radar_study_dataset import RadarStudyDataset, read_table
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
from src.models.radar import zero_shot
from src.models.radar.routing import load_routing
from src.utils import wandb_logging
from tests.models.radar_synthetic import DIM, GRID, make_synthetic, synthetic_config
from tests.utils.test_vjepa_2_1_wandb import FakeWandb

PROMPTS = {
    "ef": {"mode": "regression", "structure": "lv", "target": "lvef",
           "label_sources": ["FEVG estimée visuellement : <#> %."], "range": [5, 90]},
    "effusion": {"mode": "binary", "structure": "pericardium",
                 "label_sources": ["Épanchement péricardique modéré.", "Épanchement péricardique important."],
                 "negative_sources": ["Pas d'épanchement péricardique."]},
    "av_stenosis": {"mode": "categorical", "structure": "aortic", "target": "av_grade",
                    "label_sources": ["Sténose aortique sévère.", "Sténose aortique légère.",
                                      "Pas de sténose aortique.", "Valve aortique de structure normale."],
                    "values": [2, 0, 1, 1]},
}
# Views of the evaluation studies. V00 has no view the aortic valve reads; V01 has no view any scored structure
# reads (the IVC alone reads a subcostal view); V02's single PLAX is in some random bags of 4 and not in others.
VAL_VIEWS = {
    "V00": ["A4C", "A4C", "A2C"],
    "V01": ["SUBCOSTAL", "SUBCOSTAL"],
    "V02": ["A4C"] * 7 + ["PLAX"],
}
N_VAL, N_TUNE = 14, 8


def strict_json(path):
    """The file's content, refusing NaN and Infinity (which `json` would otherwise read)."""
    def refuse(token):
        raise ValueError(f"{token} is not JSON.")
    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=refuse)


class TestZeroShotEval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        cls.config = synthetic_config(cls.paths)
        radar_train.train(cls.config, os.path.join(cls.root, "run"))
        cls.checkpoint = os.path.join(cls.root, "run", "latest.pt")

        generator = torch.Generator().manual_seed(1)
        cls.cache_dir = os.path.join(cls.root, "eval", "cache")
        writer = TokenCacheWriter(cls.cache_dir, GRID, DIM, provenance={"synthetic": True, "clips": "evaluation"})
        clips, labels = [], []
        studies = [(f"V{i:02d}", "val") for i in range(N_VAL)] + [(f"T{i:02d}", "tune") for i in range(N_TUNE)]
        for i, (study, split) in enumerate(studies):
            views = VAL_VIEWS.get(study, ["A4C", "PLAX", "PSAX_AV", "A2C", "SUBCOSTAL"][: 3 + i % 3])
            for j, view in enumerate(views):
                writer.add(f"{study}-c{j}", torch.randn(GRID[0] * GRID[1] * GRID[2], DIM, generator=generator))
                clips.append({"study_id": study, "clip_id": f"{study}-c{j}", "view": view, "acquisition": "bmode",
                              "split": split})
            labels.append({"study_id": study, "lvef": 30 + 3 * i, "effusion": i % 2, "av_grade": i % 3,
                           "patient": f"P{i // 2:02d}"})
        writer.close()
        cls.clips = os.path.join(cls.root, "eval", "clips.csv")
        pd.DataFrame(clips).to_csv(cls.clips, index=False)
        labels = pd.DataFrame(labels).astype({"lvef": object, "effusion": object, "av_grade": object})
        labels = labels.set_index("study_id")
        labels.loc["V01", "effusion"] = 1            # a positive that no bag supports
        labels.loc["V05", "lvef"] = ""               # blank cells: missing labels
        labels.loc["V06", "effusion"] = ""
        labels.loc["V07", "av_grade"] = ""
        train_studies = sorted(set(read_table(cls.paths["clips"]).study_id))
        extra = pd.DataFrame({"lvef": 50, "effusion": [i % 2 for i in range(len(train_studies))], "av_grade": 1,
                              "patient": [f"Q{i}" for i in range(len(train_studies))]},
                             index=pd.Index(train_studies, name="study_id"))
        cls.label_frame = pd.concat([labels, extra])
        cls.labels = os.path.join(cls.root, "labels.csv")
        cls.label_frame.to_csv(cls.labels)
        cls.prompts = cls.write_prompts("prompts.yaml", PROMPTS)
        cls.report = cls.run_eval("val")
        cls.predictions = pd.read_csv(os.path.join(cls.root, "val", "predictions.csv"), dtype={"study_id": str})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    @classmethod
    def write_prompts(cls, name, tasks):
        path = os.path.join(cls.root, name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"tasks": tasks}, f, allow_unicode=True)
        return path

    @classmethod
    def run_eval(cls, name, split="val", **kwargs):
        options = {"config": cls.config, "checkpoint": cls.checkpoint, "clips": cls.clips, "labels": cls.labels,
                   "out_dir": os.path.join(cls.root, name), "prompts": cls.prompts, "split": split,
                   "cache_dir": cls.cache_dir, "samples": 40}
        options.update(kwargs)
        return zs.evaluate(**options)

    def rows(self, task, frame=None):
        frame = self.predictions if frame is None else frame
        return frame[frame.task == task].set_index("study_id")

    # ---- outputs ------------------------------------------------------------------------------------------------

    def test_outputs_are_written_with_one_row_per_study_and_task(self):
        out = os.path.join(self.root, "val")
        self.assertEqual(sorted(os.listdir(out)), ["metrics.json", "predictions.csv", "thresholds.json"])
        self.assertEqual(len(self.predictions), N_VAL * len(PROMPTS))
        self.assertEqual(list(self.predictions.columns),
                         ["study_id", "task", "mode", "structure", "label", "prediction", "prediction_supported",
                          "supported_bags", "probabilities"])
        self.assertEqual(set(self.predictions.study_id), {f"V{i:02d}" for i in range(N_VAL)})
        self.assertEqual(self.report["studies"], N_VAL)
        self.assertEqual(self.report["label_rows_not_evaluated"], len(self.label_frame) - N_VAL)
        self.assertEqual(strict_json(os.path.join(out, "metrics.json")), self.report)
        self.assertEqual(set(self.report["tasks"]), set(PROMPTS))
        self.assertEqual(self.report["tasks"]["ef"]["target"], "lvef")
        self.assertEqual(self.report["tasks"]["effusion"]["target"], "effusion")       # the task's own name

    def test_the_report_says_what_it_read(self):
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        report = self.report
        self.assertEqual(report["checkpoint_fingerprint"], state["fingerprint"]["sha256"])
        self.assertEqual((report["checkpoint_epoch"], report["checkpoint_step"]), (state["epoch"], state["step"]))
        self.assertEqual((report["split"], report["pooling"], report["num_bags"]), ("val", "mean_similarity", 3))
        for key in ("checkpoint_sha256", "clips_sha256", "labels_sha256", "prompts_sha256"):
            self.assertEqual(len(report[key]), 64)
        self.assertFalse(report["same_cache_as_training"])
        self.assertEqual(report["cache_provenance_differences"], ["clips"])
        self.assertFalse(report["training_studies_allowed"])
        self.assertEqual(report["bootstrap"], {"samples": 40, "seed": 0, "level": 0.95, "resampled": "study_id"})
        self.assertGreater(report["temperature"], 0.0)

    # ---- support and coverage -----------------------------------------------------------------------------------

    def test_a_study_whose_bags_hold_no_routed_clip_has_no_prediction(self):
        for task in PROMPTS:
            row = self.rows(task).loc["V01"]
            self.assertFalse(row.prediction_supported)
            self.assertEqual(row.supported_bags, 0)
            self.assertTrue(np.isnan(row.prediction))
        aortic = self.rows("av_stenosis")
        self.assertFalse(aortic.loc["V00"].prediction_supported)            # A4C and A2C: not aortic-valve views
        self.assertTrue(pd.isna(aortic.loc["V00"].probabilities))
        self.assertTrue(self.rows("ef").loc["V00"].prediction_supported)
        supported = self.predictions[self.predictions.prediction_supported]
        self.assertTrue(supported.prediction.notna().all())
        self.assertTrue((supported.supported_bags >= 1).all())
        self.assertTrue(self.predictions[~self.predictions.prediction_supported].prediction.isna().all())

    def test_coverage_counts_studies_labels_and_unsupported_positives(self):
        for task, spec in PROMPTS.items():
            rows = self.rows(task)
            coverage = self.report["tasks"][task]["coverage"]
            labelled = rows.label.notna()
            used = labelled & rows.prediction_supported
            self.assertEqual(coverage["studies"], N_VAL)
            self.assertEqual(coverage["studies_supported"], int(rows.prediction_supported.sum()))
            self.assertEqual(coverage["labelled"], N_VAL - 1)               # one blank label per column
            self.assertEqual(coverage["labelled_supported"], int(used.sum()))
            self.assertAlmostEqual(coverage["coverage"], used.sum() / labelled.sum())
            self.assertLess(coverage["coverage"], 1.0)
            self.assertEqual(self.report["tasks"][task]["metrics"]["n"], int(used.sum()))
        self.assertEqual(self.report["tasks"]["effusion"]["coverage"]["unsupported_positives"], 1)     # V01
        self.assertNotIn("unsupported_positives", self.report["tasks"]["ef"]["coverage"])
        self.assertTrue(np.isnan(self.rows("ef").loc["V05"].label))
        self.assertTrue(self.rows("ef").loc["V05"].prediction_supported)    # predicted, but not scored

    def test_supported_bags_follow_the_bags_actually_drawn(self):
        report = self.run_eval("random", selection="random", num_bags=6)
        frame = pd.read_csv(os.path.join(self.root, "random", "predictions.csv"), dtype={"study_id": str})
        routing = load_routing()
        table = read_table(self.clips)
        dataset = RadarStudyDataset(table[table.split == "val"], pd.DataFrame(columns=list(zs.CAPTION_COLUMNS)),
                                    routing, TokenCache(self.cache_dir), num_clips=4, selection="random",
                                    training=False, eval_seed=0)
        aortic = [s.name for s in routing.trained].index("aortic")
        index = [study for study, _ in dataset.studies].index("V02")
        expected = 0
        for bag in range(6):
            dataset.eval_bag_index = bag
            expected += int(dataset[index]["eligible"][aortic].any())
        self.assertEqual(self.rows("av_stenosis", frame).loc["V02"].supported_bags, expected)
        self.assertTrue(0 < expected < 6, expected)                         # the case the mean over bags must handle
        self.assertEqual(report["bags"], dataset.bag_record())
        self.assertEqual(report["bags"]["selection"], "random")

    # ---- predictions and metrics --------------------------------------------------------------------------------

    def test_metrics_are_those_of_the_labelled_supported_predictions(self):
        def used(task):
            rows = self.rows(task)
            return rows[rows.label.notna() & rows.prediction_supported]

        def same(a, b):
            self.assertEqual(set(a), set(b))
            for key, value in b.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    self.assertTrue(a[key] is None, key)
                else:
                    self.assertAlmostEqual(a[key], value, places=6, msg=key)

        rows = used("ef")
        same(self.report["tasks"]["ef"]["metrics"], zm.regression_metrics(rows.label.to_numpy(),
                                                                          rows.prediction.to_numpy()))
        rows = used("effusion")
        same(self.report["tasks"]["effusion"]["metrics"], zm.binary_metrics(rows.label.to_numpy(),
                                                                            rows.prediction.to_numpy()))
        self.assertEqual(self.report["tasks"]["effusion"]["metrics"]["threshold"], 0.5)
        self.assertEqual(self.report["tasks"]["effusion"]["threshold_source"], "default_0.5")
        self.assertNotIn("threshold_source", self.report["tasks"]["ef"])
        rows = used("av_stenosis")
        probabilities = np.array([json.loads(p) for p in rows.probabilities])
        same(self.report["tasks"]["av_stenosis"]["metrics"],
             zm.categorical_metrics(rows.label.to_numpy(), probabilities, [0, 1, 2]))
        np.testing.assert_array_equal(rows.prediction.to_numpy(), probabilities.argmax(axis=1))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
        self.assertTrue(self.rows("ef").prediction.dropna().between(5, 90).all())
        self.assertTrue(self.rows("effusion").prediction.dropna().between(0, 1).all())

    def test_intervals_have_the_keys_of_the_mode_and_bracket_what_they_can(self):
        for task, keys in (("ef", zm.REGRESSION_INTERVALS), ("effusion", zm.BINARY_INTERVALS),
                           ("av_stenosis", zm.CATEGORICAL_INTERVALS)):
            intervals = self.report["tasks"][task]["intervals"]
            self.assertEqual(set(intervals), set(keys))
            for key, interval in intervals.items():
                self.assertLessEqual(interval["resamples"], 40)
                if interval["low"] is not None:
                    self.assertLessEqual(interval["low"], interval["high"])
        mae = self.report["tasks"]["ef"]
        self.assertEqual(mae["intervals"]["mae"]["resamples"], 40)
        self.assertLessEqual(mae["intervals"]["mae"]["low"], mae["metrics"]["mae"])
        self.assertGreaterEqual(mae["intervals"]["mae"]["high"], mae["metrics"]["mae"])

    def test_a_prediction_is_the_mean_of_the_scorers_over_the_bags(self):
        routing = load_routing()
        model, tokenizer, _ = zs.load_model(self.config, self.checkpoint, routing, torch.device("cpu"))
        table = read_table(self.clips)
        dataset = RadarStudyDataset(table[table.split == "val"], pd.DataFrame(columns=list(zs.CAPTION_COLUMNS)),
                                    routing, TokenCache(self.cache_dir), num_clips=4, selection="view_balanced",
                                    training=False, eval_seed=0)
        structures = [s.name for s in routing.trained]
        index = [study for study, _ in dataset.studies].index("V03")
        tau = float(model.loss.effective_temperature().detach())
        positive = zero_shot.encode_prompts(model.text, tokenizer, PROMPTS["effusion"]["label_sources"])
        negative = zero_shot.encode_prompts(model.text, tokenizer, PROMPTS["effusion"]["negative_sources"])
        pairs = zero_shot.regression_prompts(PROMPTS["ef"]["label_sources"], 5, 90)
        ef_prompts = zero_shot.encode_prompts(model.text, tokenizer, [p for _, p in pairs])
        probability, estimate = [], []
        for bag in range(3):
            dataset.eval_bag_index = bag
            item = dataset[index]
            with torch.no_grad():
                _, z, has_clip = model.pooler(*(item[k][None] for k in ("tokens", "clip_valid", "eligible", "view_ids",
                                                                        "acquisition_ids")))
            self.assertTrue(has_clip[0, structures.index("pericardium")] and has_clip[0, structures.index("lv")])
            probability.append(float(zero_shot.binary_probability(z[:, structures.index("pericardium")], positive,
                                                                  negative, tau)))
            estimate.append(float(zero_shot.regression_estimate(z[:, structures.index("lv")], ef_prompts,
                                                                [v for v, _ in pairs])))
        self.assertAlmostEqual(self.rows("effusion").loc["V03"].prediction, np.mean(probability), places=5)
        self.assertAlmostEqual(self.rows("ef").loc["V03"].prediction, np.mean(estimate), places=5)
        self.assertEqual(self.rows("ef").loc["V03"].supported_bags, 3)

    def test_the_run_is_reproducible_and_pooling_and_seed_change_what_they_should(self):
        again = self.run_eval("again")
        self.assertEqual(again, self.report)
        frame = pd.read_csv(os.path.join(self.root, "again", "predictions.csv"), dtype={"study_id": str})
        pd.testing.assert_frame_equal(frame, self.predictions)
        other = self.run_eval("mean_embedding", pooling="mean_embedding", seed=1, eval_seed=5)
        self.assertEqual(other["pooling"], "mean_embedding")
        self.assertEqual(other["bags"]["seed"], 5)
        self.assertNotEqual(other["bags"], self.report["bags"])
        self.assertEqual(other["bootstrap"]["seed"], 1)

    # ---- thresholds ---------------------------------------------------------------------------------------------

    def test_thresholds_of_a_tuning_split_are_used_and_those_of_the_reported_split_refused(self):
        tune = self.run_eval("tune", split="tune")
        path = os.path.join(self.root, "tune", "thresholds.json")
        saved = strict_json(path)
        self.assertEqual(set(saved["thresholds"]), {"effusion"})
        self.assertEqual(saved["source"]["split"], "tune")
        rows = self.rows("effusion", pd.read_csv(os.path.join(self.root, "tune", "predictions.csv")))
        self.assertAlmostEqual(saved["thresholds"]["effusion"],
                               zm.probability_threshold(rows.label.to_numpy(), rows.prediction.to_numpy()))
        self.assertEqual(tune["tasks"]["effusion"]["threshold_source"], "default_0.5")

        report = self.run_eval("val_tuned", thresholds=path)
        task = report["tasks"]["effusion"]
        self.assertEqual(task["threshold_source"], "tuning_split")
        self.assertEqual(task["metrics"]["threshold"], saved["thresholds"]["effusion"])
        for key in ("auroc", "auprc", "brier"):                             # the threshold moves only its own metrics
            self.assertEqual(task["metrics"][key], self.report["tasks"]["effusion"]["metrics"][key])
        self.assertEqual(report["tasks"]["ef"], self.report["tasks"]["ef"])

        with self.assertRaisesRegex(ValueError, "split being reported"):
            self.run_eval("refused", thresholds=os.path.join(self.root, "val", "thresholds.json"))
        copy = os.path.join(self.root, "labels_copy.csv")                   # another label file, the same studies
        self.label_frame.iloc[::-1].to_csv(copy)
        with self.assertRaisesRegex(ValueError, "split being reported"):
            self.run_eval("refused", labels=copy, thresholds=os.path.join(self.root, "val", "thresholds.json"))
        with self.assertRaisesRegex(ValueError, "another pooling"):
            self.run_eval("refused", thresholds=path, pooling="mean_embedding")
        other = dict(PROMPTS, effusion=dict(PROMPTS["effusion"], negative_sources=["Péricarde normal."]))
        with self.assertRaisesRegex(ValueError, "another prompts_sha256"):
            self.run_eval("refused", thresholds=path, prompts=self.write_prompts("other_prompts.yaml", other))
        saved["thresholds"]["effusion"] = None                              # one class on the tuning split
        empty = os.path.join(self.root, "no_threshold.json")
        with open(empty, "w") as f:
            json.dump(saved, f)
        with self.assertRaisesRegex(ValueError, "no threshold"):
            self.run_eval("refused", thresholds=empty)
        self.assertFalse(os.path.exists(os.path.join(self.root, "refused")))

    # ---- refusals -----------------------------------------------------------------------------------------------

    def test_training_studies_are_refused_unless_allowed(self):
        train = {"clips": self.paths["clips"], "split": "train", "cache_dir": None}
        with self.assertRaisesRegex(ValueError, "are in the training split"):
            self.run_eval("refused", **train)
        report = self.run_eval("train", allow_training_studies=True, **train)
        self.assertTrue(report["training_studies_allowed"])
        self.assertTrue(report["same_cache_as_training"])
        self.assertEqual(report["cache_provenance_differences"], [])
        moved = dict(self.config, data=dict(self.config["data"], clips=self.clips))
        with self.assertRaisesRegex(ValueError, "not the clip table the checkpoint trained on"):
            self.run_eval("refused", config=moved)
        self.run_eval("moved", config=moved, allow_training_studies=True)

    def test_a_config_that_is_not_the_checkpoints_is_refused(self):
        cases = {
            "model section": dict(self.config, model=dict(self.config["model"], proj_dim=4)),
            "text section": dict(self.config, text=dict(self.config["text"], max_length=16)),
            "acquisition_classes": dict(self.config, data=dict(self.config["data"], acquisition_classes=["bmode"])),
        }
        for name, config in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "does not describe the model"):
                self.run_eval("refused", config=config)
        loss = dict(self.config, loss=dict(self.config["loss"], use_normal=False))   # not part of the model
        self.assertEqual(self.run_eval("other_loss", config=loss)["tasks"], self.report["tasks"])

    def test_outputs_inside_the_repository_are_refused(self):
        with self.assertRaisesRegex(ValueError, "inside the code repository"):
            self.run_eval("refused", out_dir=os.path.join(zs.REPO, "zero_shot_out"))
        self.assertFalse(os.path.exists(os.path.join(zs.REPO, "zero_shot_out")))

    def test_labels_that_cannot_be_scored_are_refused(self):
        def labels(name, change):
            frame = self.label_frame.copy()
            change(frame)
            path = os.path.join(self.root, name)
            frame.to_csv(path)
            return path

        def drop(frame):
            del frame["av_grade"]

        def text(frame):
            frame.loc["V03", "lvef"] = "normal"

        def two(frame):
            frame.loc["V03", "effusion"] = 2

        def grade(frame):
            frame.loc["V03", "av_grade"] = 3

        cases = {"no column": (drop, "no column"), "text": (text, "not numbers"), "binary 2": (two, "must be 0 or 1"),
                 "grade 3": (grade, "not one of the task's values")}
        for name, (change, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                self.run_eval("refused", labels=labels(f"labels_{change.__name__}.csv", change))
        repeated = os.path.join(self.root, "labels_repeated.csv")
        pd.concat([self.label_frame, self.label_frame.loc[["V03"]]]).to_csv(repeated)
        with self.assertRaisesRegex(ValueError, "repeat a study"):
            self.run_eval("refused", labels=repeated)
        no_id = os.path.join(self.root, "labels_no_id.csv")
        self.label_frame.to_csv(no_id, index=False)
        with self.assertRaisesRegex(ValueError, "study_id column"):
            self.run_eval("refused", labels=no_id)

    def test_prompts_that_cannot_be_scored_are_refused(self):
        long = dict(PROMPTS, effusion=dict(PROMPTS["effusion"], negative_sources=[" ".join(["mot"] * 40)]))
        with self.assertRaisesRegex(ValueError, "longer than the 32 tokens"):
            self.run_eval("refused", prompts=self.write_prompts("long.yaml", long))
        untrained = dict(PROMPTS, effusion=dict(PROMPTS["effusion"], structure="liver"))
        with self.assertRaisesRegex(ValueError, "not trained"):
            self.run_eval("refused", prompts=self.write_prompts("untrained.yaml", untrained))
        with self.assertRaisesRegex(ValueError, "pooling must be"):
            self.run_eval("refused", pooling="max")
        with self.assertRaisesRegex(ValueError, "no split column|No clip to evaluate"):
            self.run_eval("refused", split="test")

    # ---- groups -------------------------------------------------------------------------------------------------

    def test_intervals_resample_the_groups_of_a_label_column(self):
        report = self.run_eval("grouped", group_column="patient")
        self.assertEqual(report["bootstrap"]["resampled"], "patient")
        for task in PROMPTS:                                                # groups change intervals, not estimates
            self.assertEqual(report["tasks"][task]["metrics"], self.report["tasks"][task]["metrics"])
        self.assertNotEqual(report["tasks"]["ef"]["intervals"], self.report["tasks"]["ef"]["intervals"])
        blank = self.label_frame.copy()
        blank.loc["V03", "patient"] = ""
        path = os.path.join(self.root, "labels_blank_group.csv")
        blank.to_csv(path)
        with self.assertRaisesRegex(ValueError, "has no 'patient' to resample by"):
            self.run_eval("refused", labels=path, group_column="patient")
        with self.assertRaisesRegex(ValueError, "no column"):
            self.run_eval("refused", group_column="mrn")

    def test_a_group_with_an_evaluated_and_a_training_study_is_refused(self):
        report = self.run_eval("grouped_checked", group_column="patient")
        self.assertEqual(report["training_studies_checked_by_group"], 16)   # every training study has a label row
        self.assertIsNone(self.report["training_studies_checked_by_group"])  # no group column: nothing to check
        shared = self.label_frame.copy()
        shared.loc["S04", "patient"] = shared.loc["V03", "patient"]         # a training study of an evaluated patient
        path = os.path.join(self.root, "labels_shared_patient.csv")
        shared.to_csv(path)
        with self.assertRaisesRegex(ValueError, "1 values of 'patient' have both"):
            self.run_eval("refused", labels=path, group_column="patient")
        self.run_eval("shared_allowed", labels=path, group_column="patient", allow_training_studies=True)
        self.run_eval("shared_ungrouped", labels=path)                      # without the column it cannot be told
        only_eval = os.path.join(self.root, "labels_only_eval.csv")
        self.label_frame[~self.label_frame.index.str.startswith("S")].to_csv(only_eval)
        self.assertEqual(self.run_eval("only_eval", labels=only_eval, group_column="patient")
                         ["training_studies_checked_by_group"], 0)

    # ---- W&B and the command line -------------------------------------------------------------------------------

    def test_wandb_gets_scalars_at_the_checkpoint_step_and_no_study_identifier(self):
        fake = FakeWandb()
        with mock.patch.object(wandb_logging, "wandb", fake):
            report = self.run_eval("logged", wandb={"wandb_project": "radar", "wandb_entity": "mhi_ai",
                                                    "wandb_run_name": "zero-shot"})
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual((fake.calls[0]["project"], fake.calls[0]["entity"], fake.calls[0]["name"]),
                         ("radar", "mhi_ai", "zero-shot"))
        run = fake.runs[0]
        self.assertTrue(run.finished)
        self.assertEqual(len(run.logged), 1)
        logged = run.logged[0]
        self.assertEqual(logged[wandb_logging.STEP_METRIC], report["checkpoint_step"])
        self.assertEqual(logged["zero_shot/ef/mae"], report["tasks"]["ef"]["metrics"]["mae"])
        self.assertEqual(logged["zero_shot/ef/mae_low"], report["tasks"]["ef"]["intervals"]["mae"]["low"])
        self.assertEqual(logged["zero_shot/effusion/coverage"], report["tasks"]["effusion"]["coverage"]["coverage"])
        self.assertEqual(logged["zero_shot/effusion/unsupported_positives"], 1)
        self.assertTrue(all(isinstance(v, (int, float)) and v == v for v in logged.values()))
        sent = json.dumps([fake.calls[0]["config"], logged], default=str)
        for study in self.predictions.study_id.unique():
            self.assertNotIn(study, sent)
        self.assertNotIn(self.root, sent)                                   # file names, not paths
        self.assertNotIn("tasks", fake.calls[0]["config"])

    def test_without_a_wandb_project_nothing_is_logged(self):
        fake = FakeWandb()
        with mock.patch.object(wandb_logging, "wandb", fake):
            self.run_eval("unlogged")
            self.run_eval("unlogged", wandb={"wandb_project": None, "wandb_entity": None, "wandb_run_name": None})
        self.assertEqual(fake.calls, [])
        with mock.patch.object(wandb_logging, "wandb", fake), self.assertRaisesRegex(ValueError, "wandb_entity"):
            self.run_eval("unlogged", wandb={"wandb_project": "radar"})

    def test_the_command_line_runs_the_same_evaluation(self):
        config = os.path.join(self.root, "config.yaml")
        with open(config, "w") as f:
            yaml.safe_dump(self.config, f)
        out = os.path.join(self.root, "cli")
        printed = io.StringIO()
        with redirect_stdout(printed):
            zs.main(["--config", config, "--checkpoint", self.checkpoint, "--clips", self.clips, "--split", "val",
                     "--cache_dir", self.cache_dir, "--labels", self.labels, "--prompts", self.prompts,
                     "--out_dir", out, "--samples", "40"])
        self.assertEqual(strict_json(os.path.join(out, "metrics.json")), self.report)
        summary = json.loads(printed.getvalue())
        self.assertEqual(summary["ef"]["n"], self.report["tasks"]["ef"]["metrics"]["n"])
        self.assertNotIn("V00", printed.getvalue())

    def test_the_shipped_prompt_file_is_scored(self):
        """Every task of the shipped file is scored once the label table holds its column, with that column's codes."""
        shipped = zero_shot.load_prompt_tasks(zs.DEFAULT_PROMPTS)
        frame = self.label_frame.copy()
        for task in shipped.values():
            if task.target not in frame.columns:
                codes = sorted(set(task.values)) if task.mode == "categorical" else [0, 1]
                frame[task.target] = [codes[i % len(codes)] for i in range(len(frame))]
        labels = os.path.join(self.root, "shipped_labels.csv")
        frame.to_csv(labels)
        report = self.run_eval("shipped", prompts=zs.DEFAULT_PROMPTS, labels=labels)
        self.assertEqual(list(report["tasks"]), list(shipped))
        self.assertEqual({name: entry["mode"] for name, entry in report["tasks"].items()},
                         {name: task.mode for name, task in shipped.items()})
        self.assertEqual(report["tasks"]["lv_ejection_fraction"]["metrics"], self.report["tasks"]["ef"]["metrics"])


if __name__ == "__main__":
    unittest.main()
