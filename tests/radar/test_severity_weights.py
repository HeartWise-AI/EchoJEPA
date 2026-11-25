"""Severity weights of RADAR captions (`loss.severity_weights`, after DeepCORO_CLIP's positive severity weights): the
grade comes from the caption table's structured columns, the loss takes a weighted mean over rows only when weights
are set, and a run without the key trains as before."""

import copy
import json
import os
import shutil
import tempfile
import unittest

import pandas as pd
import torch
import torch.nn.functional as F

from evals.radar_frozen import train as radar_train
from src.models.radar.loss import (
    StructureContrastiveLoss,
    caption_targets,
    contrastive_loss,
    weighted_contrastive_loss,
)
from src.models.radar.severity import (
    DEFAULT_WEIGHTS,
    CaptionWeights,
    caption_grade,
    caption_weights,
    resolve_weights,
    value_grade,
)
from tests.models.radar_synthetic import make_synthetic, synthetic_config


def unit(*shape, seed=0):
    return F.normalize(torch.randn(*shape, generator=torch.Generator().manual_seed(seed)), dim=-1)


def sources(*values):
    return json.dumps([["column", v, "prefill_code"] for v in values])


class TestGrades(unittest.TestCase):

    def test_value_tokens(self):
        self.assertEqual(value_grade("mildly_reduced"), "mild")
        self.assertEqual(value_grade("moderately_reduced"), "moderate")
        self.assertEqual(value_grade("akinetic_or_severely_hypokinetic"), "severe")
        self.assertEqual(value_grade("mild_to_moderate"), "moderate")
        self.assertIsNone(value_grade("dilated"))
        self.assertIsNone(value_grade("normal"))

    def test_grade_follows_normality_then_sources(self):
        self.assertEqual(caption_grade("normal", sources("severely_reduced")), "normal")
        self.assertEqual(caption_grade("unknown"), "unknown")
        self.assertEqual(caption_grade("abnormal"), "abnormal")                       # v0 table: no sources
        self.assertEqual(caption_grade("abnormal", sources("dilated")), "abnormal")    # ungraded value
        self.assertEqual(caption_grade("abnormal", sources("mildly_reduced", "dilated")), "mild")
        self.assertEqual(caption_grade("abnormal", sources("mildly_reduced", "severely_reduced")), "severe")

    def test_weights_config(self):
        self.assertEqual(resolve_weights(True), DEFAULT_WEIGHTS)
        self.assertEqual(resolve_weights({"severe": 4})["severe"], 4.0)
        self.assertEqual(resolve_weights({"severe": 4})["mild"], DEFAULT_WEIGHTS["mild"])
        with self.assertRaisesRegex(ValueError, "unknown grades"):
            resolve_weights({"critical": 2})
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_weights({"normal": 0})
        with self.assertRaisesRegex(TypeError, "true or a mapping"):
            resolve_weights(2.0)

    def test_caption_table_weights_and_coverage(self):
        table = pd.DataFrame({
            "structure": ["lv", "lv", "lv", "mitral", "mitral"],
            "caption_id": [1, 2, 3, 4, 4],
            "normality": ["normal", "abnormal", "abnormal", "unknown", "abnormal"],
            "sources": ["[]", sources("moderately_reduced"), sources("severely_reduced"), "[]", "[]"]})
        weights, coverage = caption_weights(table, True)
        self.assertEqual(weights, {1: 1.0, 2: 2.5, 3: 3.0, 4: 1.5})          # id 4: the larger of its two grades
        self.assertEqual(coverage["caption_ids_with_two_grades"], 1)
        self.assertEqual(coverage["rows"]["lv"]["moderate"], 1)
        self.assertEqual(coverage["rows"]["mitral"]["unknown"], 1)
        self.assertTrue(coverage["graded_from_sources"])

    def test_lookup_by_caption_id(self):
        lookup = CaptionWeights({5: 3.0, -2: 2.5})
        self.assertEqual(lookup(torch.tensor([5, 9, -2])).tolist(), [3.0, 1.0, 2.5])
        self.assertEqual(CaptionWeights({})(torch.tensor([1])).tolist(), [1.0])


class TestWeightedLoss(unittest.TestCase):

    def setUp(self):
        self.z, self.t = unit(5, 8, seed=1), unit(5, 8, seed=2)
        self.Q = caption_targets(torch.tensor([1, 1, 2, 3, 4]))

    def test_equal_weights_are_the_mean(self):
        plain = contrastive_loss(self.z, self.t, self.Q, 0.1)
        self.assertTrue(torch.allclose(weighted_contrastive_loss(self.z, self.t, self.Q, 0.1, torch.full((5,), 2.5)), plain))

    def test_weighted_mean_over_rows(self):
        w = torch.tensor([1.0, 1.0, 3.0, 2.5, 1.5])
        logits = self.z @ self.t.T / 0.1
        i2t = -(self.Q * F.log_softmax(logits, dim=-1)).sum(-1)
        t2i = -(self.Q * F.log_softmax(logits.T, dim=-1)).sum(-1)
        expected = ((w * i2t).sum() / w.sum() + (w * t2i).sum() / w.sum()) / 2
        self.assertTrue(torch.allclose(weighted_contrastive_loss(self.z, self.t, self.Q, 0.1, w), expected))

    def test_module_without_weights_is_unchanged_and_with_them_reports_the_mean(self):
        z, t = unit(6, 2, 8, seed=3), unit(6, 2, 8, seed=4)
        valid = torch.ones(6, 2, dtype=torch.bool)
        ids = torch.tensor([[1, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15]])
        loss = StructureContrastiveLoss(["lv", "mitral"], use_exact=True)
        base, stats = loss(z, t, valid, ids)
        self.assertNotIn("mean_weight", stats["lv"])
        loss.set_caption_weights({1: 1.0, 2: 3.0, 10: 2.5})
        weighted, stats = loss(z, t, valid, ids)
        self.assertNotEqual(float(weighted.detach()), float(base.detach()))
        self.assertAlmostEqual(stats["lv"]["mean_weight"], (1 + 1 + 3 + 1 + 1 + 1) / 6, places=6)
        loss.set_caption_weights(None)
        again, _ = loss(z, t, valid, ids)
        self.assertEqual(float(again.detach()), float(base.detach()))   # bit for bit


class TestTrainingWithSeverity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        cls.config = synthetic_config(cls.paths)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_weights_change_the_run_and_are_recorded(self):
        config = copy.deepcopy(self.config)
        config["loss"]["severity_weights"] = {"abnormal": 3.0}
        out = os.path.join(self.root, "severity")
        weighted = radar_train.train(config, out, stop_after_epoch=1)
        plain = radar_train.train(self.config, os.path.join(self.root, "plain"), stop_after_epoch=1)
        self.assertNotEqual(weighted["fingerprint"], plain["fingerprint"])           # a loss key: in the fingerprint
        self.assertNotEqual(weighted["losses"], plain["losses"])
        with open(os.path.join(out, "severity_weights.json")) as f:
            coverage = json.load(f)
        self.assertEqual(coverage["weights"]["abnormal"], 3.0)
        self.assertGreater(coverage["rows"]["pericardium"]["abnormal"], 0)
        self.assertFalse(os.path.exists(os.path.join(self.root, "plain", "severity_weights.json")))


if __name__ == "__main__":
    unittest.main()
