"""Per-structure training use in `epochs.jsonl`: the share of an epoch's steps a structure contributed to, and why it
was skipped in the others, so that a structure at 0% can be told apart: no valid rows versus identical captions."""

import collections
import unittest

from evals.radar_frozen.train import SKIP_REASONS, structure_summary
from src.models.radar.loss import StructureContrastiveLoss

import torch


class TestStructureSummary(unittest.TestCase):

    def test_use_fraction_and_skip_reasons(self):
        contributing = {"lv": [(40, 3, 0), (42, 5, 0), (44, 1, 0), (41, 2, 0)], "pulmonic": []}
        skips = {"lv": collections.Counter(),
                 "pulmonic": collections.Counter({"too_few_rows": 3, "uniform_targets": 1})}
        out = structure_summary(contributing, skips, steps=4)
        self.assertEqual(out["lv"]["use_fraction"], 1.0)
        self.assertEqual(out["lv"]["median_rows"], 41.5)
        self.assertEqual(out["pulmonic"]["contributing_steps"], 0)
        self.assertEqual(out["pulmonic"]["use_fraction"], 0.0)
        self.assertEqual(out["pulmonic"]["skipped_too_few_rows"], 3)
        self.assertEqual(out["pulmonic"]["skipped_uniform_targets"], 1)
        self.assertEqual(out["pulmonic"]["skipped_no_abnormal_row"], 0)
        self.assertIsNone(out["pulmonic"]["median_rows"])
        for entry in out.values():      # every step is either used or skipped for one reason
            self.assertEqual(entry["contributing_steps"] + sum(entry[f"skipped_{r}"] for r in SKIP_REASONS),
                             entry["steps"])

    def test_no_steps(self):
        out = structure_summary({"lv": []}, {}, steps=0)
        self.assertIsNone(out["lv"]["use_fraction"])

    def test_reasons_match_the_loss(self):
        """The loss's skip reasons are the ones counted: a valve with one caption on every row is `uniform_targets`
        under exact-caption targets (A1), and one valid row is `too_few_rows`."""
        loss = StructureContrastiveLoss(["tricuspid", "pulmonic"], use_exact=True)
        z = torch.nn.functional.normalize(torch.randn(4, 2, 8), dim=-1)
        t = torch.nn.functional.normalize(torch.randn(4, 2, 8), dim=-1)
        valid = torch.tensor([[True, True], [True, False], [True, False], [True, False]])
        ids = torch.tensor([[7, 9], [7, 0], [7, 0], [7, 0]])    # tricuspid: one caption on every row
        _, stats = loss(z, t, valid, ids)
        self.assertEqual(stats["tricuspid"]["skipped"], "uniform_targets")
        self.assertEqual(stats["pulmonic"]["skipped"], "too_few_rows")
        self.assertTrue({s["skipped"] for s in stats.values()} <= set(SKIP_REASONS))


if __name__ == "__main__":
    unittest.main()
