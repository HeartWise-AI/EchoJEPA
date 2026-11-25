# tests/models/test_radar_soft_ef.py

"""Soft value targets (`loss.soft_ef`) on synthetic tensors and captions: off is the base arm bit for bit; the kernel,
the cap on the soft mass and the row normalisation follow the formula of `src/models/radar/loss.py`; rows without a
value are untouched; exact-caption positives (A1) keep their proportions; only the configured structure changes; both
directions use the same targets; gradients are finite and pull a row towards its soft neighbours. Values come from
the caption text only (a stated EF number, or a fixed value per stated LV function grade). End to end, a run with
the option logs its diagnostics and coverage, and a config with an empty option keeps the fingerprint of a config
without it."""

import json
import math
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
    caption_soft_values,
    caption_targets,
    contrastive_loss,
    soft_value_targets,
)
from tests.models.radar_synthetic import make_synthetic, synthetic_config

NAN = float("nan")
GRADES = {"normal": 60, "mildly reduced": 48, "moderately reduced": 37.5, "severely reduced": 25}


def unit(*shape, seed=0):
    return F.normalize(torch.randn(*shape, generator=torch.Generator().manual_seed(seed)), dim=-1)


def kernel(a, b, sigma):
    return math.exp(-(a - b) ** 2 / (2 * sigma ** 2))


class TestSoftTargets(unittest.TestCase):

    def test_kernel_weights_and_natural_share_under_the_cap(self):
        values = torch.tensor([50.0, 55.0, 70.0, 45.0])
        Q, stats = soft_value_targets(torch.eye(4), values, sigma=5.0, weight=0.9)
        K = [kernel(50, 55, 5), kernel(50, 70, 5), kernel(50, 45, 5)]
        k = sum(K)
        m = k / (1 + k)                                    # A0: unnormalised base mass 1
        expected = torch.tensor([1 - m] + [m * w / k for w in K])
        self.assertTrue(torch.allclose(Q[0], expected, atol=1e-6))
        self.assertAlmostEqual(stats["positives"], sum(
            sum(kernel(v, u, 5) for u in values.tolist() if u != v) ** 2
            / sum(kernel(v, u, 5) ** 2 for u in values.tolist() if u != v) for v in values.tolist()) / 4, places=4)
        self.assertEqual(stats["rows"], 4)

    def test_the_soft_mass_is_capped(self):
        values = torch.tensor([50.0, 50.0, 50.5, 49.5, 51.0, 50.0])   # many close neighbours: natural share > 0.9
        Q, stats = soft_value_targets(torch.eye(6), values, sigma=5.0, weight=0.25)
        self.assertTrue(torch.allclose(Q.diagonal(), torch.full((6,), 0.75)))
        self.assertAlmostEqual(stats["mass"], 0.25, places=6)
        far = torch.tensor([20.0, 80.0])                               # a far neighbour: a small share, not the cap
        Q, _ = soft_value_targets(torch.eye(2), far, sigma=5.0, weight=0.5)
        k = kernel(20, 80, 5)
        self.assertAlmostEqual(float(Q[0, 1]), k / (1 + k), places=6)
        self.assertLess(float(Q[0, 1]), 1e-6)

    def test_rows_sum_to_one(self):
        g = torch.Generator().manual_seed(0)
        values = 20 + 50 * torch.rand(16, generator=g)
        values[::4] = NAN
        ids = torch.randint(0, 6, (16,), generator=g)
        for use_exact in (False, True):
            for weight in (0.1, 0.5, 0.9):
                Q, _ = soft_value_targets(caption_targets(ids, use_exact=use_exact), values, 5.0, weight)
                self.assertTrue(torch.allclose(Q.sum(1), torch.ones(16), atol=1e-6))
                self.assertTrue(bool((Q >= 0).all()))

    def test_rows_without_a_value_are_untouched_and_take_nothing(self):
        values = torch.tensor([50.0, NAN, 52.0, 51.0])
        base = caption_targets(torch.tensor([1, 2, 3, 4]))
        Q, stats = soft_value_targets(base, values, 5.0, 0.5)
        self.assertTrue(torch.equal(Q[1], base[1]))
        self.assertTrue(torch.equal(Q[:, 1], base[:, 1]))             # no soft mass goes to a row without a value
        self.assertEqual(stats["rows"], 3)
        Q, stats = soft_value_targets(base, torch.full((4,), NAN), 5.0, 0.5)
        self.assertTrue(torch.equal(Q, base))
        self.assertIsNone(stats["mass"])
        self.assertIsNone(stats["positives"])

    def test_composition_with_exact_caption_positives(self):
        ids = torch.tensor([7, 7, 8, 9])
        values = torch.tensor([50.0, 50.0, 53.0, NAN])
        base = caption_targets(ids, use_exact=True)                     # row 0: 1/2 on itself, 1/2 on row 1
        Q, _ = soft_value_targets(base, values, 5.0, 0.5)
        k = kernel(50, 53, 5)                                          # row 1 is a positive already: not in K
        m = min(0.5, k / (2 + k))                                      # base mass 2: identity + one exact positive
        self.assertTrue(torch.allclose(Q[0], torch.tensor([(1 - m) / 2, (1 - m) / 2, m, 0.0]), atol=1e-6))
        self.assertAlmostEqual(float(Q[0, 0]), float(Q[0, 1]), places=7)   # base proportions kept
        k2 = 2 * kernel(53, 50, 5)                                     # row 2 sees both rows of caption 7
        m2 = min(0.5, k2 / (1 + k2))
        self.assertTrue(torch.allclose(Q[2], torch.tensor([m2 / 2, m2 / 2, 1 - m2, 0.0]), atol=1e-6))
        self.assertTrue(torch.equal(Q[3], base[3]))
        a0, _ = soft_value_targets(caption_targets(ids, use_exact=False), values, 5.0, 0.5)
        self.assertGreater(float(a0[0, 1]), 0)                         # A0: the same caption is a soft positive
        self.assertLessEqual(float(a0[0, 1:].sum()), 0.5 + 1e-6)

    def test_the_same_targets_in_both_directions(self):
        z, t, tau = unit(5, 8, seed=1), unit(5, 8, seed=2), 0.07
        Q, _ = soft_value_targets(torch.eye(5), torch.tensor([40.0, 45.0, 60.0, NAN, 41.0]), 5.0, 0.5)
        logits = z @ t.T / tau
        i2t = -(Q * logits.log_softmax(-1)).sum(-1).mean()
        t2i = -(Q * logits.T.log_softmax(-1)).sum(-1).mean()
        self.assertTrue(torch.allclose(contrastive_loss(z, t, Q, tau), (i2t + t2i) / 2))
        self.assertTrue(torch.allclose(contrastive_loss(z, t, Q, tau), contrastive_loss(t, z, Q, tau)))
        self.assertFalse(torch.allclose(i2t, t2i))                     # both directions are trained, not one twice


class TestStructureLossSoft(unittest.TestCase):

    def setUp(self):
        self.B, self.S, self.P = 6, 2, 8
        self.z = unit(self.B, self.S, self.P, seed=3).requires_grad_()
        self.t = unit(self.B, self.S, self.P, seed=4).requires_grad_()
        self.valid = torch.ones(self.B, self.S, dtype=torch.bool)
        self.ids = torch.arange(self.B * self.S).reshape(self.B, self.S)
        self.ids[:2, 0] = 100                                          # an exact pair in structure a
        self.values = {100: 50.0, 2: 52.0, 4: 70.0, 6: 49.0, 1: 50.0, 3: 51.0, 5: 50.0, 7: 52.0}

    def loss(self, soft=None, values=None, use_exact=True, ids=None):
        fn = StructureContrastiveLoss(["a", "b"], use_exact=use_exact, soft_ef=soft)
        if soft:
            fn.set_soft_values(self.values if values is None else values)
        return fn(self.z, self.t, self.valid, self.ids if ids is None else ids)

    def test_off_is_the_base_arm_bit_for_bit(self):
        for use_exact in (False, True):
            off, stats_off = self.loss(use_exact=use_exact)
            expected = sum(contrastive_loss(self.z[:, s], self.t[:, s], caption_targets(self.ids[:, s],
                           use_exact=use_exact), self.z.new_tensor(0.07)) for s in range(self.S))
            self.assertTrue(torch.equal(off, expected))
            empty, stats = self.loss({"sigma": 5.0, "weight": 0.5, "structure": "a"}, values={}, use_exact=use_exact)
            self.assertTrue(torch.equal(off, empty))                     # no value anywhere: the same loss
            self.assertEqual(stats["a"]["soft_rows"], 0)
            self.assertNotIn("soft_rows", stats_off["a"])                # off: the stats keep their shape

    def test_only_the_configured_structure_changes(self):
        off, stats_off = self.loss()
        on, stats = self.loss({"sigma": 5.0, "weight": 0.5, "structure": "a"})
        self.assertNotEqual(float(on), float(off))
        self.assertEqual(stats["b"], stats_off["b"])        # b's ids 1, 3, 5, 7 have values, but b is not soft
        self.assertNotIn("soft_rows", stats["b"])
        self.assertEqual(stats["b"]["loss"], stats_off["b"]["loss"])
        self.assertEqual(stats["a"]["soft_rows"], 4)                     # rows of a: ids 100, 100, 4, 6, 8, 10
        self.assertTrue(0 < stats["a"]["soft_mass"] <= 0.5)
        self.assertGreater(stats["a"]["soft_positives"], 1.0)
        values = torch.tensor([50.0, 50.0, 70.0, 49.0, NAN, NAN])
        Q, _ = soft_value_targets(caption_targets(self.ids[:, 0]), values, 5.0, 0.5)
        expected_a = contrastive_loss(self.z[:, 0], self.t[:, 0], Q, self.z.new_tensor(0.07))
        self.assertAlmostEqual(stats["a"]["loss"], float(expected_a), places=6)

    def test_gradients_are_finite_and_pull_towards_soft_neighbours(self):
        on, _ = self.loss({"sigma": 5.0, "weight": 0.5, "structure": "a"})
        on.backward()
        self.assertTrue(bool(torch.isfinite(self.z.grad).all()) and bool(torch.isfinite(self.t.grad).all()))
        # Row 3 (EF 49) has soft neighbours (rows 0, 1: EF 50), which are its negatives in the base arm. Descending
        # the soft loss over the image embeddings (captions fixed) must leave z_3 closer to t_0 than the base loss.
        final = {}
        for name, soft in (("off", None), ("on", {"sigma": 5.0, "weight": 0.5, "structure": "a"})):
            z = self.z.detach().clone().requires_grad_()
            fn = StructureContrastiveLoss(["a", "b"], use_exact=True, soft_ef=soft)
            if soft:
                fn.set_soft_values(self.values)
            optimizer = torch.optim.SGD([z], lr=0.05)
            for _ in range(1000):
                optimizer.zero_grad()
                fn(F.normalize(z, dim=-1), self.t.detach(), self.valid, self.ids)[0].backward()
                optimizer.step()
            final[name] = float(F.normalize(z.detach()[3, 0], dim=-1) @ self.t[0, 0].detach())
        self.assertGreater(final["on"], final["off"] + 0.1)

    def test_settings_are_checked(self):
        for bad in ({"weight": 1.0}, {"weight": 0.0}, {"sigma": 0.0}, {"structure": "c"}):
            with self.assertRaises(ValueError):
                StructureContrastiveLoss(["a", "b"], soft_ef={"structure": "a", **bad})
        with self.assertRaises(ValueError):
            StructureContrastiveLoss(["a", "b"], use_normal=True, soft_ef={"structure": "a"})
        fn = StructureContrastiveLoss(["a", "b"], soft_ef={"structure": "a"})
        with self.assertRaises(ValueError):                             # on, but no values were set
            fn(self.z, self.t, self.valid, self.ids)
        with self.assertRaises(ValueError):
            StructureContrastiveLoss(["a", "b"]).set_soft_values({})


class TestCaptionValues(unittest.TestCase):

    def table(self, rows):
        return pd.DataFrame([{"structure": s, "caption": c, "caption_id": i} for i, (s, c) in enumerate(rows)])

    def test_grades_from_the_lv_function_sentence(self):
        table = self.table([
            ("lv", "LV systolic function appears normal. LV mass is increased."),
            ("lv", "LV cavity is dilated. LV systolic function appears severely reduced."),
            ("lv", "LV systolic function appears reduced."),                  # no grade: no value
            ("lv", "LV systolic function appears hyperdynamic."),             # not in the map: no value
            ("lv", "LV mass is normal."),                                     # qualitative only, no grade sentence
            ("rv", "RV systolic function appears normal."),                   # another structure
        ])
        values, coverage = caption_soft_values(table, {"source": "caption_grade", "grade_values": GRADES})
        self.assertEqual(values, {0: 60.0, 1: 25.0})
        self.assertEqual((coverage["rows"], coverage["rows_with_value"]), (5, 2))
        with self.assertRaises(ValueError):
            caption_soft_values(table, {"source": "caption_grade"})
        with self.assertRaises(ValueError):
            caption_soft_values(table, {"source": "labels"})

    def test_numbers_by_the_caption_builder_reader(self):
        table = self.table([
            ("lv", "VG non dilaté. FEVG estimée visuellement : 55 %."),
            ("lv", "FEVG 50-60 %."),                                          # an interval: its midpoint
            ("lv", "FEVG < 35 %."),                                           # a bound: no value
            ("lv", "FEVG environ 40 %."),                                     # hedged: no value
            ("lv", "Strain longitudinal global -18 %."),                      # not an EF
            ("lv", "Fonction systolique normale."),                           # qualitative only
            ("lv", "FEVG 30 %. FE 60 %."),                                    # statements far apart: no value
        ])
        values, coverage = caption_soft_values(table, {"source": "caption_number"})
        self.assertEqual(values, {0: 55.0, 1: 55.0})
        self.assertEqual(coverage["rows_with_value"], 2)

    def test_numbers_from_the_english_ef_sentence(self):
        table = self.table([
            ("lv", "LV systolic function appears normal. LV ejection fraction is 62%. LV mass is normal."),
            ("lv", "LV ejection fraction is 35%."),
            ("lv", "LV ejection fraction is 35%. LV ejection fraction is 50%."),  # two values: none
            ("lv", "LV systolic function appears normal."),                    # no number
            ("rv", "LV ejection fraction is 40%."),                            # another structure
        ])
        values, coverage = caption_soft_values(table, {"source": "caption_number"})
        self.assertEqual(values, {0: 62.0, 1: 35.0})
        self.assertEqual((coverage["rows"], coverage["rows_with_value"]), (4, 2))

    def test_one_value_per_caption_id(self):
        table = pd.DataFrame([{"structure": "lv", "caption": "FEVG 55 %.", "caption_id": 1},
                              {"structure": "lv", "caption": "FEVG 35 %.", "caption_id": 1}])
        with self.assertRaises(ValueError):
            caption_soft_values(table, {"source": "caption_number"})


class TestSoftTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        base = synthetic_config(cls.paths, epochs=1)
        base["loss"] = {"use_exact": True}
        cls.base = base
        cls.soft = json.loads(json.dumps(base))
        cls.soft["loss"]["soft_ef"] = {"sigma": 5.0, "weight": 0.5, "structure": "lv", "source": "caption_number"}
        cls.off = radar_train.train(base, os.path.join(cls.root, "off"))
        cls.on = radar_train.train(cls.soft, os.path.join(cls.root, "on"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_a_run_logs_its_diagnostics_and_coverage(self):
        with open(os.path.join(self.root, "on", "soft_ef_coverage.json")) as f:
            coverage = json.load(f)
        self.assertEqual(coverage["rows_with_value"], coverage["rows"])   # every synthetic LV caption states its EF
        with open(os.path.join(self.root, "on", "log.jsonl")) as f:
            lines = [json.loads(line) for line in f]
        lv = [line["structures"]["lv"] for line in lines]
        self.assertTrue(all("soft_rows" in s for s in lv))
        self.assertTrue(any(s["soft_positives"] is not None for s in lv))
        self.assertNotIn("soft_rows", lines[0]["structures"]["pericardium"])
        with open(os.path.join(self.root, "off", "log.jsonl")) as f:
            self.assertNotIn("soft_rows", json.loads(f.readline())["structures"]["lv"])
        self.assertNotEqual(self.on["losses"], self.off["losses"])
        self.assertNotEqual(self.on["fingerprint"], self.off["fingerprint"])

    def test_an_empty_option_is_off(self):
        config = json.loads(json.dumps(self.base))
        config["loss"]["soft_ef"] = None
        empty = radar_train.train(config, os.path.join(self.root, "empty"))
        self.assertEqual(empty["fingerprint"], self.off["fingerprint"])
        self.assertEqual(empty["losses"], self.off["losses"])

    def test_refused_with_a_view_prefix(self):
        config = json.loads(json.dumps(self.soft))
        config["data"]["view_prefix"] = True
        with self.assertRaisesRegex(ValueError, "soft_ef"):
            radar_train.train(config, os.path.join(self.root, "prefix"))


if __name__ == "__main__":
    unittest.main()
