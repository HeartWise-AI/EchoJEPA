# tests/models/test_radar_zero_shot_metrics.py

"""Zero-shot metrics on inputs with a known answer: the binary metrics of a six-study example worked by hand, the
expected calibration error, the recalibration line (recovered from simulated labels, and without a value when the fit
has none), Youden's threshold, the categorical metrics, and the bootstrap (reproducible, by study or by group, counting
the resamples in which a metric had a value)."""

import math
import unittest

import numpy as np

from evals.radar_frozen import zero_shot_metrics as zm

Y = np.array([1, 1, 0, 0, 1, 0])
P = np.array([0.9, 0.6, 0.7, 0.2, 0.4, 0.1])


class TestBinaryMetrics(unittest.TestCase):

    def test_a_worked_example(self):
        m = zm.binary_metrics(Y, P, threshold=0.5)        # called: 0.9, 0.6, 0.7 -> TP 2, FP 1, FN 1, TN 2
        self.assertEqual((m["n"], m["positives"], m["threshold"]), (6, 3, 0.5))
        self.assertAlmostEqual(m["prevalence"], 0.5)
        for key in ("precision", "recall", "specificity", "f1"):
            self.assertAlmostEqual(m[key], 2 / 3)
        self.assertAlmostEqual(m["auroc"], 7 / 9)          # 7 of the 9 (positive, negative) pairs are ordered
        self.assertAlmostEqual(m["auprc"], (1 + 2 / 3 + 3 / 4) / 3)     # precision at the ranks of the positives
        self.assertAlmostEqual(m["brier"], (0.01 + 0.16 + 0.49 + 0.04 + 0.36 + 0.01) / 6)

    def test_the_threshold_moves_only_the_metrics_at_the_threshold(self):
        low, high = zm.binary_metrics(Y, P, threshold=0.15), zm.binary_metrics(Y, P, threshold=0.8)
        self.assertEqual((low["recall"], high["recall"]), (1.0, 1 / 3))
        self.assertEqual((low["specificity"], high["specificity"]), (1 / 3, 1.0))
        self.assertEqual(high["precision"], 1.0)
        for key in ("auroc", "auprc", "brier", "ece", "prevalence"):
            self.assertEqual(low[key], high[key])

    def test_metrics_without_a_value_are_nan_not_zero(self):
        none_called = zm.binary_metrics(Y, P, threshold=0.95)
        self.assertTrue(math.isnan(none_called["precision"]))
        self.assertEqual(none_called["recall"], 0.0)
        one_class = zm.binary_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
        for key in ("auroc", "auprc", "recall", "calibration_slope"):
            self.assertTrue(math.isnan(one_class[key]), key)
        empty = zm.binary_metrics(np.array([]), np.array([]))
        self.assertEqual(empty["n"], 0)
        self.assertTrue(math.isnan(empty["prevalence"]) and math.isnan(empty["brier"]) and math.isnan(empty["ece"]))

    def test_expected_calibration_error(self):
        # two bins: [0.2, 0.4, 0.1] against a rate of 1/3, and [0.9, 0.6, 0.7] against a rate of 2/3
        expected = 0.5 * abs(0.7 / 3 - 1 / 3) + 0.5 * abs(2.2 / 3 - 2 / 3)
        self.assertAlmostEqual(zm.expected_calibration_error(Y, P, bins=2), expected)
        self.assertAlmostEqual(zm.expected_calibration_error([1, 0], [1.0, 0.0]), 0.0)     # 1.0 is in the last bin
        self.assertAlmostEqual(zm.expected_calibration_error([0, 0], [1.0, 1.0]), 1.0)

    def test_the_calibration_line_recovers_the_line_the_labels_were_drawn_from(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.98, 40000)
        logit = np.log(p / (1 - p))
        for intercept, slope in ((0.0, 1.0), (0.5, 0.6), (-1.0, 1.8)):
            y = rng.uniform(size=p.size) < 1 / (1 + np.exp(-(intercept + slope * logit)))
            a, b = zm.calibration_line(y, p)
            self.assertAlmostEqual(a, intercept, delta=0.06)
            self.assertAlmostEqual(b, slope, delta=0.06)

    def test_the_calibration_line_has_no_value_when_the_fit_has_none(self):
        cases = {"one class": ([1, 1, 1], [0.2, 0.5, 0.9]), "one probability": ([1, 0, 1], [0.5, 0.5, 0.5]),
                 "separated classes": ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), "one study": ([1], [0.5])}
        for name, (y, p) in cases.items():
            with self.subTest(name):
                self.assertTrue(all(math.isnan(v) for v in zm.calibration_line(y, p)))

    def test_youden_threshold(self):
        self.assertAlmostEqual(zm.probability_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9]), 0.45)
        self.assertAlmostEqual(zm.probability_threshold(Y, P), 0.3)       # J = 1 + 2/3 - 1 at 0.3, the maximum
        self.assertIsNone(zm.probability_threshold([1, 1], [0.2, 0.8]))
        self.assertIsNone(zm.probability_threshold([0, 0], [0.2, 0.8]))
        t = zm.probability_threshold([1, 0], [0.4, 0.4])                  # no rule beats chance: 0 or above 1
        self.assertIn(t, (0.0, float(np.nextafter(1.0, 2.0))))


class TestCategoricalMetrics(unittest.TestCase):

    def test_a_worked_example_with_grades_that_are_label_codes(self):
        grades = [0, 1, 2]
        y = np.array([2, 2, 0, 1, 1, 0])
        p = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4],
                      [0.5, 0.4, 0.1]])                              # predicted: 2, 0, 0, 1, 2, 0
        m = zm.categorical_metrics(y, p, grades)
        self.assertEqual(m["n"], 6)
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        # F1: grade 0: TP 2, FP 1, FN 0 -> 0.8; grade 1: TP 1, FP 0, FN 1 -> 2/3; grade 2: TP 1, FP 1, FN 1 -> 0.5
        self.assertAlmostEqual(m["macro_f1"], (0.8 + 2 / 3 + 0.5) / 3)
        aucs = [zm.auroc(y == g, p[:, g]) for g in grades]
        self.assertAlmostEqual(m["macro_auroc"], sum(aucs) / 3)

    def test_a_grade_nobody_has_or_predicts_is_left_out_of_the_means(self):
        m = zm.categorical_metrics([0, 1, 0, 1], [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.6, 0.4, 0.0], [0.7, 0.3, 0.0]],
                                   [0, 1, 2])
        self.assertAlmostEqual(m["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertAlmostEqual(m["macro_auroc"], 0.75)                   # 3 of 4 pairs ordered, for both grades

    def test_a_grade_outside_the_task_is_refused_and_nothing_is_nan_free(self):
        with self.assertRaisesRegex(ValueError, "grade"):
            zm.categorical_metrics([3], [[0.5, 0.5]], [0, 1])
        empty = zm.categorical_metrics([], np.zeros((0, 2)), [0, 1])
        self.assertEqual(empty["n"], 0)
        self.assertTrue(math.isnan(empty["accuracy"]))


class TestBootstrap(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = (rng.uniform(size=200) < 0.3).astype(float)
        self.p = np.clip(0.3 + 0.25 * (self.y - 0.3) + rng.normal(0, 0.15, 200), 0.01, 0.99)

    def test_intervals_are_reproducible_and_hold_the_estimate(self):
        point = zm.binary_metrics(self.y, self.p)
        a = zm.bootstrap(zm.binary_metrics, (self.y, self.p), zm.BINARY_INTERVALS, samples=200, seed=3)
        b = zm.bootstrap(zm.binary_metrics, (self.y, self.p), zm.BINARY_INTERVALS, samples=200, seed=3)
        c = zm.bootstrap(zm.binary_metrics, (self.y, self.p), zm.BINARY_INTERVALS, samples=200, seed=4)
        self.assertEqual(a, b)
        self.assertNotEqual(a["auroc"], c["auroc"])
        self.assertEqual(set(a), set(zm.BINARY_INTERVALS))
        for key in ("prevalence", "auroc", "auprc", "brier"):
            self.assertLess(a[key]["low"], point[key])
            self.assertGreater(a[key]["high"], point[key])
            self.assertEqual(a[key]["resamples"], 200)

    def test_resampling_groups_keeps_their_studies_together(self):
        y = np.array([1.0, 3.0, 5.0, 7.0])
        mean = lambda v: {"mean": float(v.mean())}  # noqa: E731
        one = zm.bootstrap(mean, (y,), ("mean",), groups=["a"] * 4, samples=50)
        self.assertEqual((one["mean"]["low"], one["mean"]["high"]), (4.0, 4.0))          # one group: always all of it
        pairs = zm.bootstrap(mean, (y,), ("mean",), groups=["a", "a", "b", "b"], samples=300, level=1.0)
        self.assertEqual((pairs["mean"]["low"], pairs["mean"]["high"]), (2.0, 6.0))      # {a,a}, {a,b}, {b,b}
        studies = zm.bootstrap(mean, (y,), ("mean",), samples=300, level=1.0)
        self.assertLess(studies["mean"]["low"], 2.0)                    # single studies reach beyond the group means
        self.assertGreater(studies["mean"]["high"], 6.0)

    def test_resamples_without_a_value_are_counted_out(self):
        y, p = np.array([1.0, 0.0, 0.0]), np.array([0.8, 0.3, 0.1])     # a resample has no positive 8 times in 27
        out = zm.bootstrap(zm.binary_metrics, (y, p), ("auroc", "brier"), samples=400, seed=0)
        self.assertEqual(out["brier"]["resamples"], 400)
        self.assertTrue(150 < out["auroc"]["resamples"] < 400)
        self.assertEqual((out["auroc"]["low"], out["auroc"]["high"]), (1.0, 1.0))
        never = zm.bootstrap(zm.binary_metrics, (np.zeros(3), p), ("auroc",), samples=20)
        self.assertEqual(never["auroc"], {"low": None, "high": None, "resamples": 0})
        nothing = zm.bootstrap(zm.binary_metrics, (np.array([]), np.array([])), ("auroc",), samples=20)
        self.assertEqual(nothing["auroc"], {"low": None, "high": None, "resamples": 0})

    def test_regression_intervals(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(20, 70, 300)
        p = y + rng.normal(2.0, 5.0, 300)
        out = zm.bootstrap(zm.regression_metrics, (y, p), zm.REGRESSION_INTERVALS, samples=200)
        point = zm.regression_metrics(y, p)
        for key in zm.REGRESSION_INTERVALS:
            self.assertLess(out[key]["low"], point[key])
            self.assertGreater(out[key]["high"], point[key])
        self.assertGreater(out["bias"]["low"], 0.0)                      # a bias of 2 with a standard error of 0.3


if __name__ == "__main__":
    unittest.main()
