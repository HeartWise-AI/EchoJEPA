"""Linear probes on pooled study embeddings (`evals/radar_frozen/linear_probe.py`), on synthetic data only."""

import os
import tempfile
import unittest

import numpy as np
import torch

from evals.radar_frozen.linear_probe import (
    check_outside_repo,
    fit_logistic,
    fit_ridge,
    fold_ids,
    pool_bags,
    ridge_predictions,
    standardise,
    token_means,
)
from evals.radar_frozen.train import REPO


def synthetic(n, d, seed, noise=0.1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    w = torch.randn(d, generator=g, dtype=torch.float64)
    y = x @ w / d ** 0.5 + 3.0 + noise * torch.randn(n, generator=g, dtype=torch.float64)
    return x, y


class TestFitting(unittest.TestCase):

    def test_folds_are_deterministic_and_stratified(self):
        labels = np.array([1] * 10 + [0] * 40, dtype=float)
        a, b = fold_ids(50, 5, seed=3, labels=labels), fold_ids(50, 5, seed=3, labels=labels)
        np.testing.assert_array_equal(a, b)
        for f in range(5):
            self.assertEqual(int(labels[a == f].sum()), 2)
            self.assertEqual(int((a == f).sum()), 10)

    def test_primal_and_dual_ridge_agree(self):
        x, y = synthetic(40, 30, 0)
        test = torch.randn(7, 30, dtype=torch.float64)
        primal = ridge_predictions(x, y, test, [0.5, 5.0])               # 30 features <= 40 rows: X'X
        wide = torch.cat([x, torch.zeros(40, 20, dtype=torch.float64)], 1)  # 50 features > 40 rows: XX'
        dual = ridge_predictions(wide, y, torch.cat([test, torch.zeros(7, 20, dtype=torch.float64)], 1), [0.5, 5.0])
        torch.testing.assert_close(primal, dual)

    def test_ridge_recovers_a_linear_signal(self):
        x, y = synthetic(800, 40, 1)
        x_train, y_train, x_test, y_test = x[:600], y[:600], x[600:], y[600:]
        x_train, x_test = standardise(x_train, x_test)
        pred, fit = fit_ridge(x_train, y_train, x_test)
        self.assertLess(float((pred - y_test).abs().mean()), 0.15)      # labels have SD ~1, noise SD 0.1
        self.assertEqual(len(fit["cv_mae"]), fit["penalties"])
        self.assertLess(fit["penalty_index"], fit["penalties"] - 1)     # not the strongest penalty

    def test_ridge_on_noise_predicts_about_the_mean(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(400, 50, generator=g, dtype=torch.float64)
        y = 5.0 + torch.randn(400, generator=g, dtype=torch.float64)
        x_train, x_test = standardise(x[:300], x[300:])
        pred, fit = fit_ridge(x_train, y[:300], x_test)
        self.assertLess(float(pred.std()), 0.3)
        self.assertAlmostEqual(float(pred.mean()), float(y[:300].mean()), delta=0.2)
        self.assertGreater(fit["penalty_index"], 7)                     # strong shrinkage wins on noise

    def test_logistic_separates_and_ignores_constant_features(self):
        g = torch.Generator().manual_seed(4)
        x = torch.randn(400, 10, generator=g, dtype=torch.float64)
        x[:, 3] = 1.0                                                   # a constant feature
        y = (x[:, 0] + 0.3 * torch.randn(400, generator=g, dtype=torch.float64) > 0.8).double()
        x_test = torch.randn(200, 10, generator=g, dtype=torch.float64)
        x_test[:, 3] = 1.0
        y_test = (x_test[:, 0] > 0.8).numpy()
        x_s, x_test_s = standardise(x, x_test)
        self.assertTrue(torch.isfinite(x_s).all())
        prob, fit = fit_logistic(x_s, y, x_test_s)
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())
        from evals.video_classification_frozen.metrics import auroc
        self.assertGreater(auroc(y_test, prob.numpy()), 0.95)
        self.assertGreater(min(fit["cv_auroc"]), 0.5)


class TestPooling(unittest.TestCase):

    def test_supported_bags_are_averaged_and_fallback_kept(self):
        values = torch.tensor([[[[1.0], [10.0]]], [[[3.0], [20.0]]], [[[100.0], [30.0]]]])   # [bags 3, N 1, S 2, 1]
        support = torch.tensor([[[True, False]], [[True, False]], [[False, False]]])
        pooled, count = pool_bags(values, support, values[0])
        self.assertEqual(pooled[0, 0, 0].item(), 2.0)        # mean of the two supporting bags, not of all three
        self.assertEqual(pooled[0, 1, 0].item(), 10.0)       # no supporting bag: the fallback
        self.assertEqual(count.tolist(), [[2, 0]])

    def test_token_means_follow_routing(self):
        tokens = torch.zeros(1, 3, 2, 1)
        tokens[0, 0], tokens[0, 1], tokens[0, 2] = 1.0, 3.0, 50.0
        clip_valid = torch.tensor([[True, True, False]])                 # the third slot is padding
        eligible = torch.tensor([[[True, True, True], [False, True, False], [False, False, False]]])
        routed, has, study = token_means(tokens, clip_valid, eligible)
        self.assertEqual(routed[0, :, 0].tolist(), [2.0, 3.0, 0.0])
        self.assertEqual(has[0].tolist(), [True, True, False])
        self.assertEqual(study[0, 0].item(), 2.0)

    def test_out_dir_inside_repo_is_refused(self):
        with self.assertRaises(ValueError):
            check_outside_repo(os.path.join(REPO, "outputs"))
        check_outside_repo(tempfile.gettempdir())


if __name__ == "__main__":
    unittest.main()
