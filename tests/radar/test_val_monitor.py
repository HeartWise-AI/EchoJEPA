"""Per-epoch validation monitoring of RADAR training (`validation`, run control): video -> caption retrieval with
de-duplicated captions and the zero-shot AUROC of binary prompt tasks, logged to `validation.jsonl` and W&B, without
changing what the run trains, its fingerprint or its resumes."""

import copy
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch
import yaml

from evals.radar_frozen import train as radar_train
from evals.radar_frozen.val_monitor import Validator, rank_metrics, scalars
from src.datasets.radar_study_dataset import read_table
from src.utils import wandb_logging
from tests.models.radar_synthetic import make_synthetic, synthetic_config
from tests.utils.test_vjepa_2_1_wandb import FakeWandb

PROMPTS = {"effusion": {"mode": "binary", "structure": "pericardium",
                        "label_sources": ["Épanchement péricardique modéré."],
                        "negative_sources": ["Pas d'épanchement péricardique."]},
           "ef": {"mode": "regression", "structure": "lv", "target": "lvef",
                  "label_sources": ["FEVG estimée visuellement : <#> %."], "range": [5, 90]}}


def lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRankMetrics(unittest.TestCase):

    def test_recall_mrr_and_median_rank(self):
        gallery = torch.eye(3)
        video = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 0.5, 1.0]])
        target = torch.tensor([0, 1, 1])        # ranks 1, 2, 2
        out = rank_metrics(video, gallery, target, recall_k=(1, 2))
        self.assertAlmostEqual(out["recall@1"], 1 / 3, places=6)
        self.assertAlmostEqual(out["recall@2"], 1.0, places=6)
        self.assertAlmostEqual(out["mrr"], (1 + 0.5 + 0.5) / 3, places=6)
        self.assertEqual(out["median_rank"], 2.0)

    def test_a_tie_counts_for_the_target(self):
        out = rank_metrics(torch.tensor([[1.0, 1.0]]), torch.eye(2), torch.tensor([1]), recall_k=(1,))
        self.assertEqual(out["recall@1"], 1.0)


class TestValidationMonitoring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        cls.config = synthetic_config(cls.paths)
        cls.straight = radar_train.train(cls.config, os.path.join(cls.root, "straight"))
        studies = sorted(set(read_table(cls.paths["clips"]).study_id))
        cls.labels = os.path.join(cls.root, "labels.csv")
        pd.DataFrame({"study_id": studies, "effusion": [i % 2 for i in range(len(studies))]}).to_csv(
            cls.labels, index=False)
        cls.prompts = os.path.join(cls.root, "prompts.yaml")
        with open(cls.prompts, "w", encoding="utf-8") as f:
            yaml.safe_dump({"tasks": PROMPTS}, f, allow_unicode=True)
        # the synthetic clip table is one split; monitoring needs no held-out studies to run
        cls.section = {"clips": cls.paths["clips"], "captions": cls.paths["captions"], "split": "train",
                       "num_bags": 2, "labels": cls.labels, "prompts": cls.prompts}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def with_validation(self, **section):
        return {**copy.deepcopy(self.config), "validation": {**self.section, **section}}

    def test_monitoring_changes_nothing_that_is_trained(self):
        out = os.path.join(self.root, "validated")
        run = radar_train.train(self.with_validation(), out)
        self.assertEqual(run["losses"], self.straight["losses"])
        self.assertEqual(run["fingerprint"], self.straight["fingerprint"])           # run control
        for a, b in zip(run["model"].state_dict().values(), self.straight["model"].state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        records = lines(os.path.join(out, "validation.jsonl"))
        self.assertEqual([r["epoch"] for r in records], [0, 1])
        lv = records[-1]["retrieval"]["lv"]
        self.assertEqual(lv["studies"], 6)
        self.assertEqual(lv["unique_captions"], 6)                                   # distinct LV captions
        self.assertEqual(records[-1]["retrieval"]["pericardium"]["unique_captions"], 2)   # de-duplicated
        for key in ("recall@1", "recall@5", "recall@10", "mrr", "median_rank"):
            self.assertIn(key, records[-1]["retrieval_mean"])
        self.assertEqual(lv["recall@10"], 1.0)                                       # gallery of 6
        self.assertEqual(set(records[-1]["zero_shot"]), {"effusion"})                # binary tasks only
        self.assertIsNotNone(records[-1]["zero_shot_mean_auroc"])
        text = json.dumps(records)
        self.assertNotIn("S00", text)                                                # no study identifier
        self.assertNotIn("péricardique", text)                                       # no caption

    def test_every_and_resume(self):
        out = os.path.join(self.root, "every")
        radar_train.train(self.with_validation(every=5), out, stop_after_epoch=1)
        self.assertEqual([r["epoch"] for r in lines(os.path.join(out, "validation.jsonl"))], [0])   # the last one
        with open(os.path.join(out, "validation.jsonl"), "a") as f:                   # an interrupted epoch's line
            f.write(json.dumps({"epoch": 1, "step": 99}) + "\n")
        radar_train.train(self.with_validation(every=5), out)
        self.assertEqual([r["step"] for r in lines(os.path.join(out, "validation.jsonl"))], [4, 8])

    def test_wandb_gets_val_scalars(self):
        config = {**self.with_validation(), "meta": {"wandb_project": "radar", "wandb_entity": "team"}}
        fake = FakeWandb()
        with mock.patch.object(wandb_logging, "wandb", fake):
            radar_train.train(config, os.path.join(self.root, "wandb"), stop_after_epoch=1)
        logged = [p for run in fake.runs for p in run.logged if "val/retrieval/mean/recall@1" in p]
        self.assertEqual(len(logged), 1)
        self.assertIn("val/retrieval/lv/recall@5", logged[0])
        self.assertIn("val/zero_shot/effusion/auroc", logged[0])

    def test_frozen_caption_encoder_gets_the_validation_texts_before_sealing(self):
        frozen = {"backbone": "tiny", "vocab_size": 200, "hidden_size": 16, "num_layers": 2, "num_heads": 2,
                  "max_length": 32, "frozen": True, "pooling": "mean", "head": "mlp", "head_hidden_dim": 12}
        base = {**copy.deepcopy(self.config), "text": frozen}
        straight = radar_train.train(base, os.path.join(self.root, "frozen_straight"))
        prompts = os.path.join(self.root, "unseen_prompts.yaml")          # worded unlike any training caption
        with open(prompts, "w", encoding="utf-8") as f:
            yaml.safe_dump({"tasks": {"effusion": {**PROMPTS["effusion"],
                                                   "label_sources": ["Un épanchement péricardique est présent."],
                                                   "negative_sources": ["Le péricarde est normal."]}}},
                           f, allow_unicode=True)
        out = os.path.join(self.root, "frozen_validated")
        run = radar_train.train({**copy.deepcopy(base), "validation": {**self.section, "prompts": prompts}}, out)
        self.assertEqual(run["losses"], straight["losses"])                          # training unchanged
        self.assertEqual(run["fingerprint"], straight["fingerprint"])
        records = lines(os.path.join(out, "validation.jsonl"))
        self.assertEqual([r["epoch"] for r in records], [0, 1])
        self.assertEqual(records[-1]["retrieval"]["lv"]["unique_captions"], 6)
        self.assertIsNotNone(records[-1]["zero_shot_mean_auroc"])

    def test_view_prefix_is_refused_with_validation_and_severity_weights(self):
        prefixed = copy.deepcopy(self.config)
        prefixed["data"]["view_prefix"] = True
        with self.assertRaisesRegex(ValueError, "validation .*data.view_prefix"):
            radar_train.train({**copy.deepcopy(prefixed), "validation": dict(self.section)},
                              os.path.join(self.root, "prefix_validation"))
        weighted = copy.deepcopy(prefixed)
        weighted.setdefault("loss", {})["severity_weights"] = True
        with self.assertRaisesRegex(ValueError, "severity_weights.*data.view_prefix"):
            radar_train.train(weighted, os.path.join(self.root, "prefix_severity"))

    def test_texts_cover_bmode_captions_and_binary_prompts(self):
        from src.models.radar.routing import load_routing
        validator = Validator(self.section, self.config, load_routing(), torch.device("cpu"))
        texts = validator.texts()
        captions = {c for (_, _, scope), (c, _, _) in validator.dataset.captions.items() if scope == "bmode"}
        self.assertTrue(captions <= set(texts))
        self.assertIn("Épanchement péricardique modéré.", texts)
        self.assertIn("Pas d'épanchement péricardique.", texts)
        self.assertNotIn("FEVG estimée visuellement : <#> %.", texts)                # regression tasks are not run

    def test_scalars_and_refusals(self):
        report = {"retrieval": {"lv": {"studies": 3, "recall@1": 0.5}}, "retrieval_mean": {"recall@1": 0.5},
                  "zero_shot": {"a": {"auroc": None}, "b": {"auroc": 0.7}}, "zero_shot_mean_auroc": 0.7}
        self.assertEqual(scalars(report), {"val/retrieval/lv/studies": 3, "val/retrieval/lv/recall@1": 0.5,
                                           "val/retrieval/mean/recall@1": 0.5, "val/zero_shot/b/auroc": 0.7,
                                           "val/zero_shot/mean_auroc": 0.7})
        from src.models.radar.routing import load_routing
        with self.assertRaisesRegex(ValueError, "validation.captions"):
            Validator({"clips": self.paths["clips"]}, self.config, load_routing(), torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "no clip of split"):
            Validator({**self.section, "split": "val"}, self.config, load_routing(), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
