# tests/models/test_view_classifier_arm_a_v2.py

"""Arm A's later variants on CPU with synthetic data: the loss-weighting options, label smoothing and early stopping
of the shared loop, the probe's head dropout and token dropout, the pooled-feature readouts (arm A-linear), and that
none of this changes the first arm A's training or the report's numbers for the runs already scored."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from evals.view_classifier import pooled as vp
from evals.view_classifier import prepare
from evals.view_classifier import probe
from evals.view_classifier import report
from evals.view_classifier import train as vt
from tests.models.test_view_classifier_training import DIM, GRID, synthetic_run


def old_class_weights(y, n_classes):
    """The inv_sqrt rule as the first runs computed it."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    return w / w[counts > 0].mean()


class TestWeighting(unittest.TestCase):

    def test_defaults_match_the_first_runs(self):
        y = [0, 0, 0, 0, 1, 3, 3]
        np.testing.assert_array_equal(vt.class_weights(y, 5, "inv_sqrt"), old_class_weights(y, 5))
        train = pd.DataFrame({"y": y, "source": ["human"] * 4 + ["pseudo"] * 3})
        cfg = {"class_weighting": "inv_sqrt", "pseudo_weight": 0.5}
        expected = old_class_weights(y, 5)[np.asarray(y)] * np.where(train.source == "pseudo", 0.5, 1.0)
        np.testing.assert_array_equal(vt.clip_weights(train, 5, cfg), expected)

    def test_inv_weighted_counts_and_floor(self):
        y = np.array([0] * 10 + [1] * 2 + [2] * 20)
        source = np.where(np.arange(len(y)) >= 12, 0.25, 1.0)       # class 2: pseudo-labels at weight 0.25
        train = pd.DataFrame({"y": y, "source": np.where(source < 1, "pseudo", "human")})
        w = vt.clip_weights(train, 4, {"class_weighting": "inv", "pseudo_weight": 0.25,
                                       "class_weight_counts": "weighted"})
        totals = pd.Series(w).groupby(y).sum()
        np.testing.assert_allclose(totals.to_numpy(), totals.iloc[0])  # every class the same total weight
        w = vt.clip_weights(train, 4, {"class_weighting": "inv", "pseudo_weight": 0.25,
                                       "class_weight_counts": "weighted", "class_count_floor": 10})
        totals = pd.Series(w).groupby(y).sum()
        self.assertAlmostEqual(totals[1] / totals[0], 0.2)              # 2 labels count as 10
        self.assertEqual(vt.class_weights(y, 4, "inv")[3], 0.0)
        with self.assertRaises(ValueError):
            vt.clip_weights(train, 4, {"class_weighting": "inv", "class_weight_counts": "other"})

    def test_arm_cv_keeps_folds_and_inner_split(self):
        cv = {"k": 5, "seed": 0, "inner_fraction": 0.15, "class_weighting": "inv_sqrt", "pseudo_weight": 1.0}
        self.assertEqual(vt.arm_cv(cv, None), cv)
        self.assertEqual(vt.arm_cv(cv, {"pseudo_weight": 0.3})["pseudo_weight"], 0.3)
        self.assertEqual(vt.arm_cv(cv, {"seed": 0})["seed"], 0)
        for bad in ({"seed": 1}, {"inner_fraction": 0.2}, {"k": 3}):
            with self.assertRaises(ValueError):
                vt.arm_cv(cv, bad)


def tiny_arm(n=40, d=8, length=6, probe_cfg=None, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    data = torch.as_tensor(rng.normal(0, 1, (n, length, d)), dtype=torch.float16)
    data[:, :, 0] += torch.as_tensor(y, dtype=torch.float16)[:, None] * 2
    cfg = {"num_heads": 2, "depth": 1, **(probe_cfg or {})}
    return probe.make_arm(data, {f"c{i}": i for i in range(n)}, cfg, ("A", "B", "C")), y


class TestShortcutsAndRegularisers(unittest.TestCase):

    def test_default_arm_trains_exactly_as_before(self):
        """make_arm without the new keys gives the same model, step for step, as the first runs' arm."""
        arm, y = tiny_arm(probe_cfg={"depth": 2})
        before = vt.Arm(name="arm_a", data=arm.data, row=arm.row,
                        build=lambda hp: _first_build(arm.data.shape[-1], 2, 2, 3),
                        param_groups=probe.param_groups, prepare=lambda x, training, g: x.float())
        hp = {"epochs": 3, "batch_size": 8, "warmup_epochs": 0, "lr": 1e-3, "weight_decay": 0.01}
        rows, val = list(range(30)), list(range(30, 40))
        w = np.ones(30)
        a = vt.fit(arm, rows, y[:30], w, val, y[30:], hp, torch.device("cpu"), seed=3)
        b = vt.fit(before, rows, y[:30], w, val, y[30:], hp, torch.device("cpu"), seed=3)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])
        for k in a[0]:
            torch.testing.assert_close(a[0][k], b[0][k], rtol=0, atol=0)

    def test_patience_and_label_smoothing(self):
        arm, y = tiny_arm()
        hp = {"epochs": 30, "batch_size": 8, "warmup_epochs": 0, "lr": 3e-2, "weight_decay": 0.0, "patience": 2,
              "label_smoothing": 0.1}
        state, best, history = vt.fit(arm, list(range(30)), y[:30], np.ones(30), list(range(30, 40)), y[30:], hp,
                                      torch.device("cpu"))
        self.assertLess(len(history), 30)
        self.assertEqual(len(history) - best, 2)
        self.assertEqual(best, int(np.argmin([h["val_loss"] for h in history])) + 1)

    def test_head_dropout_keeps_the_state_dict(self):
        torch.manual_seed(0)
        plain = probe.build_probe(16, 2, 1, 3)
        torch.manual_seed(0)
        dropped = probe.build_probe(16, 2, 1, 3, head_dropout=0.5)
        self.assertEqual(list(plain.state_dict()), list(dropped.state_dict()))
        x = torch.randn(4, 6, 16)
        plain.eval(), dropped.eval()
        torch.testing.assert_close(plain(x), dropped(x))
        dropped.train()
        plain.train()
        self.assertFalse(torch.allclose(plain(x), dropped(x)))
        with self.assertRaises(ValueError):
            probe.build_probe(16, 2, 1, 3, head_dropout=1.5)

    def test_token_drop(self):
        x = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
        self.assertIs(probe.drop_tokens(x, 0.0, None), x)
        a = probe.drop_tokens(x, 0.5, torch.Generator().manual_seed(1))
        b = probe.drop_tokens(x, 0.5, torch.Generator().manual_seed(1))
        self.assertEqual(tuple(a.shape), (2, 5, 3))
        torch.testing.assert_close(a, b)
        for i in range(2):                                  # kept tokens are rows of the clip, in order
            first = a[i, :, 0]
            self.assertTrue(bool((first[1:] > first[:-1]).all()))
            self.assertTrue(set(first.tolist()) <= set(x[i, :, 0].tolist()))
        arm, _ = tiny_arm(probe_cfg={"token_drop": 0.5})
        batch = arm.data[:4]
        self.assertEqual(arm.prepare(batch, True, torch.Generator().manual_seed(0)).shape[1], 3)
        self.assertEqual(arm.prepare(batch, False, None).shape[1], 6)        # evaluation sees every token


def _first_build(embed_dim, num_heads, depth, num_classes):
    from src.models.attentive_pooler import AttentiveClassifier
    return AttentiveClassifier(embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_classes=num_classes)


class TestPooledReadouts(unittest.TestCase):

    def test_pooling(self):
        torch.manual_seed(0)
        grid = (2, 4, 6)
        x = torch.randn(48, 5)
        out = vp.pool_tokens(x, grid)
        cube = x.reshape(2, 4, 6, 5)
        np.testing.assert_allclose(out["mean"], x.mean(0).numpy(), atol=1e-6)
        np.testing.assert_allclose(out["mean_std"][5:], x.std(0, unbiased=False).numpy(), atol=1e-6)
        np.testing.assert_allclose(out["time"][5:10], cube[1].mean((0, 1)).numpy(), atol=1e-6)
        np.testing.assert_allclose(out["space2"][:5], cube[:, :2, :3].mean((0, 1, 2)).numpy(), atol=1e-6)
        np.testing.assert_allclose(out["space2"][15:], cube[:, 2:, 3:].mean((0, 1, 2)).numpy(), atol=1e-6)
        self.assertEqual(out["space4"].shape, (80,))
        with self.assertRaises(ValueError):
            vp.pool_tokens(torch.randn(47, 5), grid)
        with self.assertRaises(ValueError):
            vp.pool_tokens(x, grid, ("max",))

    def test_readouts(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(150, 12))
        y = np.arange(150) % 3
        x[:, :4] += 3 * np.eye(3)[y] @ rng.normal(size=(3, 4))      # a class centre in 4 of the 12 dimensions
        for readout, hp in (("logistic", {"C": 1.0}), ("ncm", {"scale": 10.0}), ("knn", {"k": 7})):
            model = vp.fit(readout, x, y, np.ones(150), 5, hp)
            p = vp.predict(model, x)
            self.assertEqual(p.shape, (150, 5))
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
            self.assertEqual(float(p[:, 3:].max()), 0.0)                # classes without training clips
            self.assertGreater((p.argmax(1) == y).mean(), 0.8, readout)
        strong = vp.fit("logistic", x, y, np.ones(150), 3, {"C": 10.0})
        weak = vp.fit("logistic", x, y, np.ones(150), 3, {"C": 1e-4})
        self.assertLess(np.abs(weak["weight"]).sum(), np.abs(strong["weight"]).sum())
        with self.assertRaises(ValueError):
            vp.grid_from_config({"grid": {"pooling": ["max"], "C": [1.0]}})


class TestNewArmsEndToEnd(unittest.TestCase):

    def test_new_arms_leave_the_first_runs_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, path = synthetic_run(root)
            prepare.main(["folds", "--config", path])
            for pseudo in ("off", "on"):
                probe.main(["--config", path, "--pseudo", pseudo, "--folds", "0", "1", "2", "3", "4", "--device",
                            "cpu"])
                probe.main(["--config", path, "--pseudo", pseudo, "--merge"])
            report.main(["--config", path])
            rep = os.path.join(root, "runs", "s1", "report")
            first = pd.read_csv(os.path.join(rep, "summary.csv"))
            first_pairs = pd.read_csv(os.path.join(rep, "paired_differences.csv"))

            cfg["arm_a_v2"] = {**cfg["arm_a"], "probe": {"num_heads": 4, "depth": 1, "head_dropout": 0.3,
                                                         "token_drop": 0.5},
                               "cv": {"class_weight_counts": "weighted", "pseudo_weight": 0.3},
                               "fixed": {"epochs": 6, "batch_size": 8, "warmup_epochs": 1, "patience": 2,
                                         "label_smoothing": 0.1, "weight_decay": 0.05},
                               "grid": {"lr": [3e-3]}}
            cfg["arm_a_linear"] = {"readout": "logistic", "encoder_sha256": "abc", "token_grid": list(GRID),
                                   "embed_dim": DIM, "grid": {"pooling": ["mean", "space2"], "C": [0.01, 1.0]}}
            cfg["report"]["notes"] = ["arm_a_v2 and arm_a_linear were added after the first runs."]
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            with self.assertRaises(ValueError):
                vp.main(["--config", path, "--arm", "arm_a", "--pseudo", "off", "--folds", "0"])
            with self.assertRaises(ValueError):
                probe.main(["--config", path, "--arm", "arm_a_linear", "--pseudo", "off", "--folds", "0"])
            for pseudo in ("off", "on"):
                probe.main(["--config", path, "--arm", "arm_a_v2", "--pseudo", pseudo, "--folds", "0", "1", "2",
                            "3", "4", "final", "--device", "cpu"])
                probe.main(["--config", path, "--arm", "arm_a_v2", "--pseudo", pseudo, "--merge"])
                vp.main(["--config", path, "--arm", "arm_a_linear", "--pseudo", pseudo, "--folds", "0", "1", "2",
                         "3", "4", "final", "--threads", "2"])
                vp.main(["--config", path, "--arm", "arm_a_linear", "--pseudo", pseudo, "--merge"])
            runs = os.path.join(root, "runs", "s1")
            sel = json.load(open(os.path.join(runs, "arm_a_v2", "pseudo_on", "fold_0", "selection.json")))
            self.assertEqual(sel["arm"], "arm_a_v2")
            self.assertEqual(sel["weighting"]["pseudo_weight"], 0.3)
            self.assertLessEqual(len(sel["settings"][0]["history"]), 6)
            lin = json.load(open(os.path.join(runs, "arm_a_linear", "pseudo_on", "fold_0", "selection.json")))
            self.assertEqual(len(lin["settings"]), 4)
            self.assertEqual(lin["n_train_pseudo"], 12)
            self.assertTrue(os.path.exists(os.path.join(runs, "arm_a_linear", "pseudo_off", "fold_final", "model.pt")))
            oof = pd.read_csv(os.path.join(runs, "arm_a_linear", "pseudo_off", "oof_predictions.csv"))
            folds = pd.read_csv(cfg["paths"]["folds"])
            self.assertEqual(sorted(oof.clip_id), sorted(folds.clip_id))
            self.assertGreater((oof.predicted == folds.set_index("clip_id").loc[oof.clip_id, "label"].to_numpy())
                               .mean(), 0.5)

            report.main(["--config", path])
            after = pd.read_csv(os.path.join(rep, "summary.csv"))
            self.assertEqual(set(after.arm), {"arm_a", "arm_a_v2", "arm_a_linear", "old_classifier"})
            keys = ["arm", "pseudo", "subset", "granularity"]
            same = after.merge(first, on=keys, suffixes=("", "_first"))
            self.assertEqual(len(same), len(first))
            for col in [c for c in first.columns if c not in keys]:
                np.testing.assert_array_equal(same[col].to_numpy(), same[f"{col}_first"].to_numpy(), col)
            pairs = pd.read_csv(os.path.join(rep, "paired_differences.csv"))
            self.assertEqual(len(pairs), 15)                     # 6 runs
            kept = pairs.merge(first_pairs, on=["a", "b"], suffixes=("", "_first"))
            self.assertEqual(len(kept), len(first_pairs))
            np.testing.assert_array_equal(kept.accuracy_diff, kept.accuracy_diff_first)
            self.assertIn("were added after the first runs", open(os.path.join(rep, "report.md")).read())


if __name__ == "__main__":
    unittest.main()
