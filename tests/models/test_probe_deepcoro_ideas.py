# tests/models/test_probe_deepcoro_ideas.py

"""Ideas brought over from DeepCORO_CLIP: validation-fixed operating points and bootstrap intervals on test,
per-group learning rates (pooler scale, layer-wise decay) and hybrid attention pooling."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from evals.study_multitask_frozen.eval import (bootstrap_intervals, encoder_lr_groups, init_opt, operating_points,
                                               probe_lr_groups, youden_probability_threshold)
from src.models.study_probe import StudyProbe, head_specs
from tests.models.test_study_probe import CLASSIFIER

SPECS = head_specs(CLASSIFIER)   # lvef (regression), as_present (binary), severity (3 classes)
NORMALIZATION = {"lvef": {"mean": 55.0, "std": 10.0}}


def scored(n, seed):
    rng = np.random.default_rng(seed)
    y_bin = (rng.random(n) < 0.3).astype(float)
    y_cls = rng.integers(0, 3, n).astype(float)
    lvef = rng.normal(55, 10, n)
    targets = np.stack([lvef, y_bin, y_cls], axis=1)
    targets[0, 1] = np.nan
    outputs = {"lvef": ((lvef - 55) / 10 + rng.normal(0, 0.3, n))[:, None],
               "as_present": (2.0 * (y_bin - 0.5) + rng.normal(0, 1, n))[:, None],
               "severity": np.eye(3)[y_cls.astype(int)] * 2 + rng.normal(0, 1, (n, 3))}
    return {"targets": targets, "outputs": outputs, "indices": list(range(n))}


class TestOperatingPoints(unittest.TestCase):

    def test_youden_threshold_separates_and_refuses_one_class(self):
        self.assertIsNone(youden_probability_threshold(np.zeros(4), np.linspace(0, 1, 4)))
        t = youden_probability_threshold(np.array([0, 0, 1, 1.0]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertTrue(0.2 < t <= 0.8)

    def test_threshold_from_validation_is_applied_to_test(self):
        points = operating_points(scored(300, 0), scored(300, 1), SPECS, NORMALIZATION)
        self.assertEqual(list(points), ["as_present"])
        p = points["as_present"]
        self.assertEqual(p["tp"] + p["fp"] + p["tn"] + p["fn"], 299)
        self.assertTrue(0.5 < p["sensitivity"] <= 1 and 0.5 < p["specificity"] <= 1)


class TestBootstrap(unittest.TestCase):

    def test_intervals_bracket_the_point_estimate(self):
        out = bootstrap_intervals(scored(200, 2), SPECS, NORMALIZATION, num_samples=200)
        for head, metric in (("lvef", "mae"), ("lvef", "pearson_r"), ("as_present", "auroc"), ("severity", "auroc")):
            ci = out[head][metric]
            self.assertLessEqual(ci["low"], ci["value"])
            self.assertLessEqual(ci["value"], ci["high"])
        self.assertEqual(bootstrap_intervals(scored(20, 3), SPECS, NORMALIZATION, num_samples=0), {})


class TestLearningRateGroups(unittest.TestCase):

    def test_pooler_scale_splits_heads_from_pooling(self):
        probe = StudyProbe(16, SPECS, pooling_mode="attention", attention_hidden=8)
        self.assertEqual(len(probe_lr_groups(probe, None)), 1)
        groups = probe_lr_groups(probe, 5.0)
        self.assertEqual([scale for _, scale in groups], [1.0, 5.0])
        self.assertEqual(sum(p.numel() for g, _ in groups for p in g), sum(p.numel() for p in probe.parameters()))
        opt = init_opt([probe], [dict(start_lr=1e-4, ref_lr=1e-4, final_lr=0.0, warmup=0.0, ref_wd=0.0,
                                      final_wd=0.0)], 10, 1, lr_groups=lambda p: probe_lr_groups(p, 5.0))[0][0]
        self.assertEqual([g["mc_ref_lr"] for g in opt.param_groups], [1e-4, 5e-4])

    def test_layer_decay_scales_blocks_from_the_top(self):
        class ViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))
                self.norm = nn.LayerNorm(4)
        vit = ViT()
        for p in vit.parameters():
            p.requires_grad = False
        for m in (*vit.blocks[2:], vit.norm):
            for p in m.parameters():
                p.requires_grad = True
        groups = encoder_lr_groups(vit, 2, 0.5)
        self.assertEqual([scale for _, scale in groups], [1.0, 0.5])
        self.assertEqual(len(groups[0][0]), 4)   # top block (weight, bias) + norm (weight, bias)
        self.assertEqual(len(encoder_lr_groups(vit, 2, None)), 1)


class TestHybridPooling(unittest.TestCase):

    def test_hybrid_attention_pools_and_masks(self):
        torch.manual_seed(0)
        probe = StudyProbe(16, SPECS, pooling_mode="attentive", num_heads=4, depth=2, attention_hidden=8,
                           hybrid_attention=True).eval()
        tokens, mask = torch.randn(2, 12, 16), torch.tensor([[True, True, False], [True, False, False]])
        out = probe(tokens, mask)
        self.assertEqual(out["severity"].shape, (2, 3))
        changed = tokens.clone()
        changed[0, 8:] = 100.0   # the padded slot of study 0
        self.assertTrue(torch.allclose(probe(changed, mask)["lvef"], out["lvef"], atol=1e-5))
        with self.assertRaises(ValueError):
            StudyProbe(16, SPECS, pooling_mode="attention", hybrid_attention=True)


if __name__ == "__main__":
    unittest.main()
