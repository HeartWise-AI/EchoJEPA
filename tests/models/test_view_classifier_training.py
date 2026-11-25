# tests/models/test_view_classifier_training.py

"""The view classifier comparison's training side on CPU with synthetic data: arm B's head extension, weight loading
and preprocessing; the shared cross-validation loop; and one small run end to end (folds from a synthetic label
export, arm A's probe on a synthetic token cache with and without pseudo-labels, the report)."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import metrics as vm
from evals.view_classifier import mvit
from evals.view_classifier import prepare
from evals.view_classifier import probe
from evals.view_classifier import report
from evals.view_classifier import train as vt
from src.datasets.radar_token_cache import TokenCacheWriter


class TestHead(unittest.TestCase):

    def test_parent_init_keeps_the_old_probabilities(self):
        torch.manual_seed(0)
        old = nn.Linear(16, len(vc.OLD_CLASSES))
        new = mvit.extend_head(old)
        self.assertEqual(new.out_features, len(vc.NEW_CLASSES))
        x = torch.randn(7, 16)
        p_old = torch.softmax(old(x), -1).detach().numpy()
        p_new = torch.softmax(new(x), -1).detach().numpy()
        collapsed = vm.collapse_probabilities(p_new, vc.NEW_CLASSES, vc.PARENT, vc.OLD_CLASSES)
        np.testing.assert_allclose(collapsed, p_old, rtol=1e-5, atol=1e-6)
        j, i = vc.NEW_CLASSES.index("RVINF"), vc.OLD_CLASSES.index("RVINF")
        torch.testing.assert_close(new.weight[j], old.weight[i])

    def test_parent_init_with_the_form_extras(self):
        names, parent = vc.from_config({"classes": {**{c: vc.PARENT[c] for c in vc.NEW_CLASSES}, **vc.FORM_EXTRAS}})
        old = nn.Linear(8, len(vc.OLD_CLASSES))
        new = mvit.extend_head(old, vc.OLD_CLASSES, names, parent)
        x = torch.randn(5, 8)
        collapsed = vm.collapse_probabilities(torch.softmax(new(x), -1).detach().numpy(), names, parent,
                                              vc.OLD_CLASSES)
        np.testing.assert_allclose(collapsed, torch.softmax(old(x), -1).detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_random_init_keeps_rows_of_kept_classes_only(self):
        old = nn.Linear(16, len(vc.OLD_CLASSES))
        new = mvit.extend_head(old, init="random")
        j, i = vc.NEW_CLASSES.index("A4C"), vc.OLD_CLASSES.index("A4C")
        torch.testing.assert_close(new.weight[j], old.weight[i])
        k = vc.NEW_CLASSES.index("PSAX_MID")
        self.assertFalse(torch.allclose(new.weight[k], old.weight[vc.OLD_CLASSES.index("PSAX")]))
        with self.assertRaises(ValueError):
            mvit.extend_head(nn.Linear(16, 5))

    def test_original_weights_load_strictly(self):
        torch.manual_seed(0)
        model = mvit.build_mvit(len(vc.OLD_CLASSES))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 34_634_132)   # the checkpoint's count
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.pth")
            torch.save({f"module.{k}": v for k, v in model.state_dict().items()}, path)
            loaded = mvit.load_original(path)
            torch.testing.assert_close(loaded.head[1].weight, model.head[1].weight)
            with self.assertRaises(ValueError):
                mvit.load_original(path, expected_sha256="0" * 64)
            torch.save({"module.extra": torch.zeros(1), **{f"module.{k}": v for k, v in model.state_dict().items()}},
                       path)
            with self.assertRaises(RuntimeError):
                mvit.load_original(path)


class TestPreprocessing(unittest.TestCase):

    def test_crop_and_scale(self):
        frames = np.zeros((5, 60, 100, 3), dtype=np.uint8)
        frames[:, :, :20] = 255                      # outside the centre square: cropped away
        frames[:, :, 80:] = 255
        out = mvit.crop_and_scale(frames, size=32)
        self.assertEqual(tuple(out.shape), (3, 5, 32, 32))
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(int(out.max()), 0)
        tall = mvit.crop_and_scale(np.full((2, 90, 50, 3), 7, dtype=np.uint8), size=16)
        self.assertTrue(bool((tall == 7).all()))

    def test_normalize_and_crop(self):
        x = torch.zeros(1, 3, 2, 4, 4, dtype=torch.uint8)
        x[:, 0] = 100
        rgb, bgr = mvit.normalize(x, "rgb"), mvit.normalize(x, "bgr")
        self.assertAlmostEqual(float(rgb[0, 0, 0, 0, 0]), (100 - mvit.README_MEAN[0]) / mvit.README_STD[0], 5)
        self.assertAlmostEqual(float(bgr[0, 2, 0, 0, 0]), (100 - mvit.README_MEAN[2]) / mvit.README_STD[2], 5)
        with self.assertRaises(ValueError):
            mvit.normalize(x, "yuv")
        y = torch.rand(2, 3, 2, 8, 8)
        a = mvit.random_crop(y, 0.5, torch.Generator().manual_seed(1))
        b = mvit.random_crop(y, 0.5, torch.Generator().manual_seed(1))
        self.assertEqual(a.shape, y.shape)
        torch.testing.assert_close(a, b)
        self.assertIs(mvit.random_crop(y, 1.0, None), y)


class TinyVideoNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, n)

    def forward(self, x):
        return self.head(self.conv(x).mean(dim=(2, 3, 4)))


class TestCrossValidationLoop(unittest.TestCase):

    def test_video_arm_fold(self):
        """Arm B's preprocessing in the shared loop, with a tiny model in place of the MViT."""
        rng = np.random.default_rng(0)
        n, classes = 30, ("A4C", "A2C", "OTHER")
        y = np.arange(n) % 3
        data = torch.as_tensor(rng.integers(0, 50, (n, 3, 2, 8, 8)), dtype=torch.uint8)
        data[y == 1, 0] += 150
        labels = pd.DataFrame({"clip_id": [f"c{i}" for i in range(n)], "patient_id": [f"p{i}" for i in range(n)],
                               "label": [classes[v] for v in y], "y": y, "fold": np.arange(n) % 2})
        arm = vt.Arm(name="tiny", data=data, row={f"c{i}": i for i in range(n)},
                     build=lambda hp: TinyVideoNet(3),
                     param_groups=lambda m, hp: [{"params": m.parameters(), "lr": hp["lr"], "weight_decay": 0.0}],
                     prepare=lambda x, training, g: mvit.random_crop(mvit.normalize(x), 0.8, g) if training
                     else mvit.normalize(x))
        grid = vt.grid_from_config({"fixed": {"epochs": 2, "batch_size": 4, "warmup_epochs": 0},
                                    "grid": {"lr": [1e-2, 1e-3]}})
        self.assertEqual(len(grid), 2)
        with tempfile.TemporaryDirectory() as d:
            vt.run_fold(arm, labels, None, 0, grid, {"seed": 0, "inner_fraction": 0.2}, classes, d,
                        torch.device("cpu"))
            pred = pd.read_csv(os.path.join(d, "fold_0", "predictions.csv"))
            self.assertEqual(sorted(pred.clip_id), sorted(labels.clip_id[labels.fold == 0]))
            np.testing.assert_allclose(pred[[f"p_{c}" for c in classes]].sum(axis=1), 1.0, rtol=1e-5)
            sel = json.load(open(os.path.join(d, "fold_0", "selection.json")))
            self.assertEqual(len(sel["settings"]), 2)
            self.assertEqual(sel["n_test"], 15)

    def test_class_weights_and_schedule(self):
        w = vt.class_weights([0, 0, 0, 0, 1], 3, "inv_sqrt")
        self.assertAlmostEqual(w[0] / w[1], 0.5)
        self.assertEqual(w[2], 0.0)
        self.assertAlmostEqual(w[:2].mean(), 1.0)
        np.testing.assert_array_equal(vt.class_weights([0, 1], 2, "none"), [1, 1])
        self.assertAlmostEqual(vt.learning_rate(1.0, 0, 100, 10), 0.1)
        self.assertAlmostEqual(vt.learning_rate(1.0, 100, 100, 10), 0.0)


GRID, DIM = (2, 2, 2), 32
USED = ("A4C", "PSAX_MID", "PLAX_ZOOM_AV", "OTHER")


def synthetic_run(root):
    """A label export, the pool, the splits, a token cache and pseudo-label candidates under `root`; the config."""
    rng = np.random.default_rng(0)
    n = 60
    labels = [USED[i % 4] for i in range(n)]
    views = [vc.PARENT[c] for c in labels]
    status = ["done"] * n + ["done", "skipped", "done"]
    human = labels + ["unsure", "A4C", "A2C"]
    m = len(human)
    acq = ["bmode" if i % 3 else "color_flow" for i in range(m)]
    keys = [f"k{i}" for i in range(m)]
    paths = [f"/clips/{i}.mp4" for i in range(m)]
    export = pd.DataFrame({"global_key": keys, "patient_id": [f"p{i}" for i in range(m)],
                           "study_id": [f"s{i}" for i in range(m)], "video_path": paths, "acq_category": acq,
                           "predicted_view": [v if v != "PSAX" else "" for v in views] + ["A4C"] * 3,
                           "human_view": human, "label_status": status, "n_labels": "1",
                           "labelled_at": "2026-11-17"})
    os.makedirs(os.path.join(root, "labels"))
    export.to_csv(os.path.join(root, "labels", "view_labels.csv"), index=False)
    inputs = os.path.join(root, "inputs")
    os.makedirs(inputs)
    pd.DataFrame({"global_key": keys, "patient_id": export.patient_id, "study_id": export.study_id,
                  "video_path": paths, "dicom_path": [p.replace(".mp4", ".dcm") for p in paths],
                  "view": views + ["A4C"] * 3, "acq_category": acq}).to_csv(os.path.join(inputs, "pool_clips.csv"),
                                                                           index=False)
    splits = pd.DataFrame({"patient_id": [f"p{i}" for i in range(m)] + [f"q{i}" for i in range(12)],
                           "split": "train"})
    splits.to_csv(os.path.join(root, "patient_splits.csv"), index=False)
    old_p = rng.dirichlet(np.ones(len(vc.OLD_CLASSES)), m)
    for i, v in enumerate(views + ["A4C"] * 3):
        old_p[i, vc.OLD_CLASSES.index(v)] += 2.0 if i % 5 else 0.0
    old_p /= old_p.sum(1, keepdims=True)
    pd.concat([pd.DataFrame({"global_key": keys}),
               pd.DataFrame(old_p, columns=[f"p_{c}" for c in vc.OLD_CLASSES])], axis=1) \
        .to_csv(os.path.join(inputs, "pool_old_probs.csv"), index=False)
    with open(os.path.join(inputs, "archive_counts.json"), "w") as f:
        json.dump({"all_splits": {c: 100 + 10 * i for i, c in enumerate(vc.OLD_CLASSES)}}, f)
    pseudo = pd.DataFrame({"video_path": [f"/archive/{i}.mp4" for i in range(12)],
                           "patient_id": [f"q{i}" for i in range(12)], "study_id": "t", "acq_category": "bmode",
                           "old_class": ["A4C", "OTHER"] * 6, "label": ["A4C", "OTHER"] * 6, "pmax": 0.95})
    pseudo.to_csv(os.path.join(inputs, "pseudo_candidates.csv"), index=False)
    cache = os.path.join(root, "cache")
    writer = TokenCacheWriter(cache, GRID, DIM, "float32", provenance={"encoder_sha256": "abc"})
    signal = {c: rng.normal(0, 1, DIM) for c in USED}
    for path, c in list(zip(paths, labels + ["A4C"] * 3)) + list(zip(pseudo.video_path, pseudo.label)):
        if path == paths[-1]:
            continue                                            # an unreadable clip
        writer.add(path, rng.normal(0, 1, (int(np.prod(GRID)), DIM)) + signal[c])
    writer.close()
    pd.DataFrame({"clip_id": [paths[-1]]}).to_csv(os.path.join(cache, "failed.csv"), index=False)
    form = {"classifications": [{"name": "view", "options": [{"value": c} for c in vc.NEW_CLASSES + ("unsure",)]}]}
    with open(os.path.join(root, "labels", "form.json"), "w") as f:
        json.dump(form, f)
    runs = os.path.join(root, "runs", "s1")
    cfg = {"name": "synthetic",
           "paths": {"labels": os.path.join(root, "labels", "view_labels.csv"), "inputs": inputs,
                     "form": os.path.join(root, "labels", "form.json"),
                     "patient_splits": os.path.join(root, "patient_splits.csv"), "token_cache": cache, "runs": runs,
                     "folds": os.path.join(runs, "folds.csv"), "pseudo_labels": os.path.join(runs, "pseudo_labels.csv"),
                     "throughput": os.path.join(runs, "throughput.json")},
           "cv": {"k": 5, "seed": 0, "inner_fraction": 0.15, "class_weighting": "inv_sqrt", "pseudo_weight": 1.0},
           "pseudo": {"threshold": 0.9, "per_class": 150, "exclude_classes": []},
           "arm_a": {"encoder_sha256": "abc", "token_grid": list(GRID), "embed_dim": DIM,
                     "probe": {"num_heads": 4, "depth": 2},
                     "fixed": {"epochs": 4, "batch_size": 8, "warmup_epochs": 0},
                     "grid": {"lr": [3e-3], "weight_decay": [0.01, 0.1]}},
           "report": {"min_class_support": 10, "min_positives": 5, "bootstrap": 50, "seed": 0}}
    path = os.path.join(root, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg, path


class TestEndToEnd(unittest.TestCase):

    def test_folds_probe_and_report(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, path = synthetic_run(root)
            prepare.main(["folds", "--config", path])
            folds = pd.read_csv(cfg["paths"]["folds"])
            self.assertEqual(len(folds), 60)                     # unsure, skipped and the unreadable clip dropped
            drops = pd.read_csv(os.path.join(root, "runs", "s1", "label_drops.csv"))
            d = dict(zip(drops.reason, drops.n))
            self.assertEqual((d["unsure"], d["status_skipped"], d["unreadable_video"]), (1, 1, 1))
            self.assertEqual(folds.groupby("label").fold.nunique().min(), 5)
            with self.assertRaises(FileExistsError):
                prepare.main(["folds", "--config", path])        # folds are frozen

            for pseudo in ("off", "on"):
                probe.main(["--config", path, "--pseudo", pseudo, "--folds", "0", "1", "2", "3", "4", "final",
                            "--device", "cpu"])
                probe.main(["--config", path, "--pseudo", pseudo, "--merge"])
            out = os.path.join(root, "runs", "s1", "arm_a")
            oof = pd.read_csv(os.path.join(out, "pseudo_on", "oof_predictions.csv"))
            self.assertEqual(sorted(oof.clip_id), sorted(folds.clip_id))
            self.assertGreater((oof.predicted == folds.set_index("clip_id").loc[oof.clip_id, "label"].to_numpy())
                               .mean(), 0.5)
            sel = json.load(open(os.path.join(out, "pseudo_on", "fold_0", "selection.json")))
            self.assertEqual(sel["n_train_pseudo"], 12)
            self.assertEqual(json.load(open(os.path.join(out, "pseudo_off", "fold_0", "selection.json")))
                             ["n_train_pseudo"], 0)
            self.assertTrue(os.path.exists(os.path.join(out, "pseudo_on", "fold_final", "model.pt")))
            self.assertFalse(os.path.exists(os.path.join(out, "pseudo_on", "fold_final", "predictions.csv")))
            mtime = os.path.getmtime(os.path.join(out, "pseudo_on", "fold_0", "model.pt"))
            probe.main(["--config", path, "--pseudo", "on", "--folds", "0", "--device", "cpu"])   # finished: skipped
            self.assertEqual(os.path.getmtime(os.path.join(out, "pseudo_on", "fold_0", "model.pt")), mtime)

            report.main(["--config", path])
            rep = os.path.join(root, "runs", "s1", "report")
            summary = pd.read_csv(os.path.join(rep, "summary.csv"))
            self.assertEqual(set(summary.arm), {"arm_a", "old_classifier"})
            self.assertEqual(set(summary.subset), {"all", "bmode", "color_flow"})
            row = summary[(summary.arm == "arm_a") & (summary.pseudo == "on") & (summary.subset == "all")
                          & (summary.granularity == "new")].iloc[0]
            self.assertEqual(row.n, 60)
            base = summary[(summary.arm == "old_classifier") & (summary.subset == "all")]
            self.assertEqual(set(base.granularity), {"old", "old_kept"})
            self.assertEqual(int(base[base.granularity == "old_kept"].n.iloc[0]), 30)   # A4C and OTHER labels
            per = pd.read_csv(os.path.join(rep, "per_class.csv"))
            self.assertFalse(per[per.support < 5].auroc_reported.any())
            self.assertEqual(len(pd.read_csv(os.path.join(rep, "paired_differences.csv"))), 1)
            text = open(os.path.join(rep, "report.md")).read()
            self.assertIn("unsure 1", text)
            self.assertTrue(os.path.exists(os.path.join(rep, "report.html")))

    def test_check_stage_writes_nothing_and_form_is_enforced(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as root:
            cfg, path = synthetic_run(root)
            before = sorted(os.listdir(root))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                prepare.main(["check", "--config", path])
            text = out.getvalue()
            self.assertIn('"usable": 61', text)             # the unreadable clip is dropped by `folds` only
            self.assertIn("evaluated classes", text)
            self.assertIn("old_kept", text)
            self.assertEqual(sorted(os.listdir(root)), before)
            self.assertFalse(os.path.exists(os.path.join(root, "runs")))
            form = json.load(open(cfg["paths"]["form"]))
            form["classifications"][0]["options"].append({"value": "RVOT"})    # a view the config lacks
            json.dump(form, open(cfg["paths"]["form"], "w"))
            with self.assertRaises(ValueError):
                prepare.main(["check", "--config", path])
            other = os.path.join(root, "labels", "later.csv")
            pd.read_csv(cfg["paths"]["labels"]).head(10).to_csv(other, index=False)
            json.dump({"classifications": [{"name": "view", "options": [{"value": "A4C"}]}]},
                      open(cfg["paths"]["form"], "w"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                prepare.main(["check", "--config", path, "--labels", other])
            self.assertIn('"usable": 10', out.getvalue())

    def test_label_batches_keep_the_pool_batch(self):
        """A later batch's rows in the export (not in this pool) stop the folds unless `label_batches` leaves them
        out; then they are counted as other_batch and the folds equal those of the batch alone."""
        with tempfile.TemporaryDirectory() as root:
            cfg, path = synthetic_run(root)
            export = pd.read_csv(cfg["paths"]["labels"], dtype=str, keep_default_na=False)
            alone = os.path.join(root, "labels", "alone.csv")
            export.to_csv(alone, index=False)
            later = export.head(5).assign(global_key=[f"b2_{i}" for i in range(5)],
                                          patient_id=[f"r{i}" for i in range(5)],
                                          video_path=[f"/later/{i}.mp4" for i in range(5)])
            both = pd.concat([export.assign(batch="1"), later.assign(batch="2")], ignore_index=True)
            both.loc[len(export) + 1, "label_status"] = "unlabelled"
            both.loc[len(export) + 1, "human_view"] = ""
            both.to_csv(cfg["paths"]["labels"], index=False)
            with self.assertRaisesRegex(ValueError, "not in the labelling pool"):
                prepare.main(["folds", "--config", path])
            cfg["label_batches"] = [1]
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            prepare.main(["folds", "--config", path])
            runs = os.path.dirname(cfg["paths"]["folds"])
            drops = dict(pd.read_csv(os.path.join(runs, "label_drops.csv")).itertuples(index=False))
            self.assertEqual((drops["other_batch"], drops["status_unlabelled"], drops["unsure"]), (5, 0, 1))
            record = json.load(open(os.path.join(runs, "folds.json")))
            self.assertEqual(record["label_batches"], [1])
            with_batches = pd.read_csv(cfg["paths"]["folds"], dtype=str)
            shutil.rmtree(runs)
            del cfg["label_batches"]
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            prepare.main(["folds", "--config", path, "--labels", alone])
            plain = pd.read_csv(cfg["paths"]["folds"], dtype=str)
            cols = ["global_key", "label", "fold"]
            pd.testing.assert_frame_equal(with_batches[cols], plain[cols])

    def test_pseudo_patients_must_be_disjoint(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, path = synthetic_run(root)
            prepare.main(["folds", "--config", path])
            pseudo = pd.read_csv(cfg["paths"]["pseudo_labels"])
            pseudo.loc[0, "patient_id"] = "p0"
            pseudo.to_csv(cfg["paths"]["pseudo_labels"], index=False)
            with self.assertRaises(ValueError):
                probe.main(["--config", path, "--pseudo", "on", "--folds", "0", "--device", "cpu"])


class TestBench(unittest.TestCase):

    def test_clips_per_second(self):
        from evals.view_classifier import bench
        rate = bench.clips_per_second(TinyVideoNet(3), bench.clip_input(2, 8), batch=4, iters=2, warmup=1)
        self.assertGreater(rate, 0)


class TestPoolStage(unittest.TestCase):

    def test_pool_tables_from_synthetic_parquet(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as root:
            rows = []
            for i in range(40):
                view = ("A4C", "PSAX", "OTHER", "PLAX_ZOOM")[i % 4]
                rows.append({"patient_id": f"{1000 + i}", "study_id": f"s{i}", "split": "train" if i < 36 else "val",
                             "view": view, "video_path": f"/mp4/{i}.mp4", "dicom_path": f"/dcm/{i}.dcm",
                             "acq_category": "bmode", "acq_confidence": "header", "acq_vendor_validated": True,
                             "n_frames": 30})
            videos = pd.DataFrame(rows)
            videos.to_parquet(os.path.join(root, "videos.parquet"))
            p = rng.dirichlet(np.ones(len(vc.OLD_CLASSES)) * 0.1, len(videos))
            for i, v in enumerate(videos.view):
                p[i] = 0.0
                p[i, vc.OLD_CLASSES.index(v)] = 0.97
                p[i, vc.OLD_CLASSES.index("A3C")] = 0.03
            master = pd.DataFrame(p, columns=list(vc.OLD_CLASSES)).astype("float32")
            master.insert(0, "dcm_path", videos.dicom_path)
            master.to_parquet(os.path.join(root, "master.parquet"))
            picked = videos.head(4).assign(global_key=[f"k{i}" for i in range(4)])
            picked.to_csv(os.path.join(root, "picked.csv"), index=False)
            videos[["patient_id", "split"]].to_csv(os.path.join(root, "splits.csv"), index=False)
            with open(os.path.join(root, "mapping.json"), "w") as f:
                json.dump({str(i): c for i, c in enumerate(vc.OLD_CLASSES)}, f)
            cfg = {"paths": {"inputs": os.path.join(root, "inputs"), "class_mapping": os.path.join(root, "mapping.json"),
                             "pool": os.path.join(root, "picked.csv"), "patient_splits": os.path.join(root, "splits.csv"),
                             "videos": os.path.join(root, "videos.parquet"),
                             "master": os.path.join(root, "master.parquet")},
                   "pseudo": {"threshold": 0.9, "per_class": 100, "exclude_classes": [], "seed": 0}}
            path = os.path.join(root, "config.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            prepare.main(["pool", "--config", path])
            inputs = cfg["paths"]["inputs"]
            pseudo = pd.read_csv(os.path.join(inputs, "pseudo_candidates.csv"), dtype=str)
            self.assertEqual(set(pseudo.label), {"A4C", "OTHER"})          # never PSAX or PLAX_ZOOM
            self.assertFalse(set(pseudo.patient_id) & set(picked.patient_id))
            self.assertFalse(set(pseudo.patient_id) & {f"{1000 + i}" for i in range(36, 40)})   # val patients
            self.assertEqual(len(pseudo), 16)                               # 9 + 9 train clips, minus the pool's
            probs = pd.read_csv(os.path.join(inputs, "pool_old_probs.csv"))
            self.assertEqual(list(probs.global_key), ["k0", "k1", "k2", "k3"])
            self.assertAlmostEqual(float(probs.p_PSAX[1]), 0.97, places=5)
            clips = pd.read_csv(os.path.join(inputs, "cache_clips.csv"))
            self.assertEqual(len(clips), 4 + 16)
            counts = json.load(open(os.path.join(inputs, "archive_counts.json")))
            self.assertEqual(counts["all_splits"]["A4C"], 10)
            self.assertEqual(counts["pseudo_eligible_train"]["A4C"], 9)


if __name__ == "__main__":
    unittest.main()
