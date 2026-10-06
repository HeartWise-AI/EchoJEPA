# tests/utils/test_probe_metrics.py

"""The frozen probe's metrics (`evals/video_classification_frozen/metrics.py`) on small cases
worked out by hand: regression error and agreement, study-level aggregation, AUROC, AUPRC,
sensitivity and specificity, the validation threshold and the error by EF range."""

import math
import unittest

import numpy as np

from evals.video_classification_frozen import metrics


class TestRegression(unittest.TestCase):

    def test_error_and_agreement(self):
        # Errors 0, 1, -1, 2 around a reference with mean 2.5 (sum of squares 5).
        out = metrics.regression([1, 2, 3, 4], [1, 3, 2, 6])
        self.assertEqual(out["n"], 4)
        self.assertAlmostEqual(out["mae"], 1.0)
        self.assertAlmostEqual(out["rmse"], math.sqrt(1.5))
        self.assertAlmostEqual(out["bias"], 0.5)
        self.assertAlmostEqual(out["r2"], 1 - 6 / 5)

    def test_pearson_r(self):
        y = [30, 45, 55, 60]
        self.assertAlmostEqual(metrics.regression(y, [2 * v + 1 for v in y])["pearson_r"], 1.0)
        self.assertAlmostEqual(metrics.regression(y, [-v for v in y])["pearson_r"], -1.0)
        # A constant prediction has no correlation.
        self.assertTrue(math.isnan(metrics.regression(y, [50] * 4)["pearson_r"]))

    def test_smooth_l1_is_the_training_loss(self):
        # Quadratic below an error of 1 (0.5 e^2), linear above (e - 0.5).
        self.assertAlmostEqual(metrics.smooth_l1([0.0, 0.0], [0.5, -2.0]), (0.125 + 1.5) / 2)


class TestByStudy(unittest.TestCase):

    def test_a_study_is_the_mean_of_its_videos(self):
        out = metrics.by_study(["b", "a", "b", "a", "a"], [50, 30, 50, 30, 30], [52, 31, 56, 33, 35])
        self.assertEqual(out.study_id.tolist(), ["a", "b"])
        self.assertEqual(out.label.tolist(), [30.0, 50.0])
        self.assertEqual(out.prediction.tolist(), [33.0, 54.0])
        self.assertEqual(out.videos.tolist(), [3, 2])

    def test_videos_of_a_study_must_share_its_label(self):
        with self.assertRaisesRegex(ValueError, "different EF labels"):
            metrics.by_study(["a", "a"], [30, 35], [31, 33])


class TestReducedEf(unittest.TestCase):

    def test_auroc_counts_ties_as_half(self):
        # Positives score 0.9 and 0.8, negatives 0.7 and 0.8: 3 pairs won, one tied, of 4.
        self.assertAlmostEqual(metrics.auroc([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.8]), 3.5 / 4)
        self.assertTrue(math.isnan(metrics.auroc([1, 1], [0.2, 0.3])))  # one class only

    def test_average_precision(self):
        # scikit-learn's documented example gives 0.8333.
        self.assertAlmostEqual(metrics.average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 5 / 6)
        # Tied scores form one threshold.
        self.assertAlmostEqual(metrics.average_precision([1, 0], [0.5, 0.5]), 0.5)

    def test_a_lower_prediction_ranks_as_reduced_ef(self):
        out = metrics.reduced_ef([30, 35, 50, 60], [33, 38, 52, 58], below=40)
        self.assertEqual((out["positives"], out["prevalence"]), (2, 0.5))
        self.assertEqual((out["auroc"], out["auprc"]), (1.0, 1.0))
        self.assertNotIn("sensitivity", out)  # no threshold given

    def test_sensitivity_and_specificity_at_a_threshold(self):
        out = metrics.reduced_ef([30, 35, 50, 60], [33, 45, 38, 58], below=40, threshold=40)
        # Flagged below 40: 33 (EF 30, right) and 38 (EF 50, wrong); 45 (EF 35) is missed.
        self.assertEqual((out["sensitivity"], out["specificity"]), (0.5, 0.5))

    def test_the_threshold_maximises_youden_j(self):
        # Candidates 37, 41, 44.5, 52.5, 61: J is 0.5 at 41 and at 52.5; 41 is closer to 40.
        self.assertEqual(metrics.youden_threshold([30, 35, 50, 60], [38, 45, 44, 60], below=40), 41.0)
        # Perfectly separated: any threshold between the classes; the midpoint is chosen.
        self.assertEqual(metrics.youden_threshold([30, 50], [32, 48], below=40), 40.0)
        self.assertIsNone(metrics.youden_threshold([50, 60], [48, 58], below=40))  # no positive


class TestByRange(unittest.TestCase):

    def test_ranges_are_left_closed(self):
        rows = metrics.by_range([29.9, 30, 39, 40, 75], [31.9, 30, 41, 40, 70], edges=(30, 40))
        self.assertEqual([r["range"] for r in rows], ["< 30", "30-40", ">= 40"])
        self.assertEqual([r["n"] for r in rows], [1, 2, 2])
        self.assertAlmostEqual(rows[0]["bias"], 2.0)
        self.assertAlmostEqual(rows[1]["mae"], 1.0)
        self.assertAlmostEqual(rows[2]["bias"], -2.5)

    def test_an_empty_range_has_no_error(self):
        rows = metrics.by_range([55.0], [50.0], edges=(30,))
        self.assertEqual(rows[0]["n"], 0)
        self.assertTrue(np.isnan(rows[0]["mae"]))


if __name__ == "__main__":
    unittest.main()
