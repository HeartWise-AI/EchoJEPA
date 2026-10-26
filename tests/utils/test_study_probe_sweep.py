"""Controlled sweep expansion, execution and selected-checkpoint summaries."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from evals.study_multitask_frozen.eval import aggregate_bags, check_disjoint, evaluate, manifest_paths
from src.models.study_probe import head_specs
from tests.models.test_study_probe import CLASSIFIER
from tools.study_probe_sweep import expand, generate, launch, summarize

ROOT = Path(__file__).resolve().parents[2]


class TestStudySweep(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load((ROOT / "configs/sweeps/study_probe.yaml").read_text())
        self.base = yaml.safe_load((ROOT / "configs/eval/vitl/study_multitask_deepecho.yaml").read_text())

    def test_ablation_and_factorial_cover_each_factor_and_preserve_protocol(self):
        original = copy.deepcopy(self.base)
        runs = expand(self.base, self.spec)
        self.assertEqual(len(runs), 24)
        self.assertEqual(len(expand(self.base, self.spec, "factorial")), 54)
        self.assertEqual(self.base, original)
        for config, entry in runs:
            self.assertEqual(config["experiment"]["evaluation"], {"selection": "weighted_loss", "num_bags": 3})
            self.assertEqual(config["experiment"]["data"]["eval_seed"], 1729)
            self.assertEqual(config["meta"]["seed"], entry["seed"])
            self.assertTrue(config["resume_checkpoint"])
        baseline = next(c for c, e in runs if e["variant"] == "baseline")
        self.assertEqual(baseline["experiment"]["classifier"]["query_mode"], "shared")
        all_four = next(c for c, e in runs if e["variant"] == "all")
        self.assertEqual(all_four["experiment"]["classifier"]["query_mode"], "task")
        self.assertTrue(all_four["experiment"]["classifier"]["local_use_rope"])
        self.assertEqual(all_four["experiment"]["classifier"]["view_dropout"], 0.2)
        self.assertEqual(all_four["experiment"]["data"]["sampling_mode"], "view_balanced")
        bad = copy.deepcopy(self.spec)
        bad["factors"]["task_queries"]["experiment.evaluation.selection"] = "lvef/mae"
        with self.assertRaisesRegex(ValueError, "same selection"):
            expand(self.base, bad)

    def test_generation_launch_integrity_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            checkpoint, data = tmp / "checkpoint.pt", tmp / "data.csv"
            checkpoint.touch()
            data.touch()
            args = dict(spec_path=ROOT / "configs/sweeps/study_probe.yaml", output=tmp / "configs",
                        checkpoint=checkpoint, data=data, folder=tmp / "results")
            manifest_path, count = generate(**args)
            self.assertEqual(count, 24)
            self.assertEqual(generate(**args), (manifest_path, count))
            with mock.patch("tools.study_probe_sweep.subprocess.run") as run:
                launch(manifest_path, ["cuda:0"], index=0, dry_run=True)
                run.assert_not_called()
                launch(manifest_path, ["cuda:0"], index=0)
                run.assert_called_once()
                self.assertNotIn("--test_only", run.call_args.args[0])
            manifest = json.loads(manifest_path.read_text())
            for index, score in ((0, 2.0), (3, 1.5)):
                entry = manifest["runs"][index]
                path = Path(entry["result_dir"])
                path.mkdir(parents=True, exist_ok=True)
                (path / "val_metrics.json").write_text(json.dumps({
                    "selected_on": "validation", "epoch": 2, "probe": 0,
                    "val": {"score": score, "failed": 0, "heads": {"ef": {"mae": score}}}}))
            summary = summarize(manifest_path, tmp / "summary")
            self.assertEqual(len(summary["missing_run_indices"]), 22)
            metric = summary["variants"]["task_queries"]["score"]
            self.assertEqual(metric["paired_delta_mean"], -0.5)
            self.assertEqual(metric["paired_delta_n"], 1)
            config = Path(manifest["runs"][0]["config"])
            config.write_text(config.read_text() + "# modified\n")
            with self.assertRaisesRegex(ValueError, "config changed"):
                launch(manifest_path, ["cuda:0"], index=0, dry_run=True)
            with self.assertRaises(FileExistsError):
                generate(**args)


class TestStudyEvaluation(unittest.TestCase):
    def setUp(self):
        self.specs = head_specs(CLASSIFIER)
        self.norm = {"lvef": {"mean": 50, "std": 10}}

    def bag(self, offset=0):
        return {"indices": np.array([0, 1]), "targets": np.array([[40, 0, 0], [60, 1, 1]]), "failed": [],
                "outputs": {"lvef": np.array([[-1.0], [1.0]]) + offset,
                            "as_present": np.array([[-2.0], [2.0]]) + offset,
                            "severity": np.array([[2., 0., 0.], [0., 2., 0.]]) + offset}}

    def test_bag_averaging_uses_probability_space_and_reports_clinical_units(self):
        a, b = self.bag(), self.bag(2)
        result = aggregate_bags([a, b], self.specs, self.norm)
        np.testing.assert_allclose(result["outputs"]["lvef"], [[0], [2]])
        np.testing.assert_allclose(result["prediction_std"]["lvef"], [[10], [10]])
        expected = ((1 / (1 + np.exp(-a["outputs"]["as_present"])))
                    + (1 / (1 + np.exp(-b["outputs"]["as_present"])))) / 2
        np.testing.assert_allclose(1 / (1 + np.exp(-result["outputs"]["as_present"])), expected)
        b["indices"] = np.array([0])
        b["targets"] = b["targets"][:1]
        b["outputs"] = {h: x[:1] for h, x in b["outputs"].items()}
        b["failed"] = [1]
        result = aggregate_bags([a, b], self.specs, self.norm)
        self.assertEqual(result["indices"].tolist(), [0])
        self.assertEqual(result["failed"], [1])

    def test_nonfinite_output_cannot_be_selected_even_on_another_head(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            bad = self.bag()
            bad["outputs"]["as_present"][0, 0] = value
            for scored in (bad, aggregate_bags([bad, self.bag()], self.specs, self.norm)):
                result = evaluate(scored, self.specs, self.norm,
                                  {"head": "lvef", "metric": "mae", "lower_is_better": True})
                self.assertTrue(np.isnan(result["score"]))

    def test_patient_overlap_and_no_test_manifest_in_training_fingerprint(self):
        a = mock.Mock(study_ids=["s1"], patient_ids={"p1"})
        b = mock.Mock(study_ids=["s2"], patient_ids={"p1"})
        with self.assertRaisesRegex(ValueError, "patients"):
            check_disjoint({"train": a, "val": b})
        self.assertEqual(manifest_paths({"dataset_train": "train.csv", "dataset_val": "val.csv",
                                         "dataset_test": "never-read.csv"}),
                         {"train": "train.csv", "val": "val.csv"})
