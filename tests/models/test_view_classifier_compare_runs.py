# tests/models/test_view_classifier_compare_runs.py

"""Snapshots side by side on synthetic run folders: pairing by clip, later minus earlier, the folds check and the
output-folder guard."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from evals.view_classifier import classes as vc
from evals.view_classifier import compare_runs as cr

CLASSES = {c: vc.PARENT[c] for c in vc.NEW_CLASSES}


def folds_table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    return pd.DataFrame({"clip_id": [f"/v/{i}.mp4" for i in range(n)], "patient_id": [str(i) for i in range(n)],
                         "label": [vc.NEW_CLASSES[k] for k in y], "y": y, "fold": np.arange(n) % 5})


def write_run(runs, arm, pseudo, labels, correct):
    """An out-of-fold table predicting the label where `correct`, else the next class (shuffled row order)."""
    pred = np.where(correct, labels.y, (labels.y + 1) % 4)
    p = np.zeros((len(labels), len(vc.NEW_CLASSES)))
    p[np.arange(len(labels)), pred] = 1.0
    t = pd.DataFrame(p, columns=[f"p_{c}" for c in vc.NEW_CLASSES])
    t.insert(0, "clip_id", labels.clip_id)
    folder = os.path.join(runs, arm, f"pseudo_{pseudo}")
    os.makedirs(folder)
    t.sample(frac=1.0, random_state=1).to_csv(os.path.join(folder, "oof_predictions.csv"), index=False)


def write_summary(runs, accuracy):
    os.makedirs(os.path.join(runs, "report"))
    pd.DataFrame([{"arm": "arm_b", "pseudo": "on", "subset": "all", "granularity": "new", "n": 40,
                   "accuracy": accuracy, "accuracy_low": accuracy - 0.1, "accuracy_high": accuracy + 0.1}]).to_csv(
        os.path.join(runs, "report", "summary.csv"), index=False)


class TestCompare(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.labels = folds_table()
        self.folds = os.path.join(root, "folds.csv")
        self.labels.to_csv(self.folds, index=False)
        self.configs = []
        for name, share in (("s1", 0.5), ("s2", 0.75)):
            runs = os.path.join(root, name)
            correct = np.arange(len(self.labels)) < int(share * len(self.labels))
            write_run(runs, "arm_b", "on", self.labels, correct)
            write_summary(runs, share)
            self.configs.append({"paths": {"runs": runs, "folds": self.folds}, "classes": CLASSES,
                                 "report": {"bootstrap": 200, "seed": 0}})
        write_run(self.configs[1]["paths"]["runs"], "arm_a", "on", self.labels, np.ones(len(self.labels), bool))

    def tearDown(self):
        self.tmp.cleanup()

    def test_paired_difference_later_minus_earlier(self):
        out = os.path.join(self.tmp.name, "compare")
        summary, pairs = cr.compare(self.configs, out)
        self.assertEqual(sorted(summary.snapshot.unique()), ["s1", "s2"])
        self.assertEqual(len(pairs), 2)                          # arm_b only (arm_a is in s2 alone), two granularities
        new = pairs[pairs.granularity == "new"].iloc[0]
        self.assertEqual((new.arm, new.later, new.earlier), ("arm_b", "s2", "s1"))
        self.assertAlmostEqual(new.accuracy_diff, 0.25)
        self.assertAlmostEqual(new.accuracy_later, 0.75)
        self.assertLessEqual(new.accuracy_diff_low, 0.25)
        for name in ("summary_side_by_side.csv", "paired_across_snapshots.csv", "compare.md"):
            self.assertTrue(os.path.exists(os.path.join(out, name)))
        self.assertIn("+0.250", open(os.path.join(out, "compare.md")).read())

    def test_refuses_other_folds_and_an_output_in_the_first_snapshot(self):
        with self.assertRaises(ValueError):
            cr.compare(self.configs, os.path.join(self.configs[0]["paths"]["runs"], "compare"))
        other = self.labels.assign(fold=(self.labels.fold + 1) % 5)
        path = os.path.join(self.tmp.name, "other_folds.csv")
        other.to_csv(path, index=False)
        self.configs[1]["paths"]["folds"] = path
        with self.assertRaises(ValueError):
            cr.compare(self.configs, os.path.join(self.tmp.name, "compare"))


if __name__ == "__main__":
    unittest.main()
