# tests/models/test_view_classifier_data.py

"""The view classifier comparison's data side on synthetic tables: the class lists and their mapping, the label
table's filtering, the pool and split checks, the pseudo-label rules, the folds (stratified, patient-disjoint,
deterministic) and the metrics (confusion matrix, per-class AUROC with small-support handling, collapsing, weights)."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from evals.view_classifier import classes as vc
from evals.view_classifier import folds as vf
from evals.view_classifier import labels as vl
from evals.view_classifier import metrics as vm

FORM_VIEWS = ["A2C", "A2C_LV", "A2C_ZOOM", "A3C", "A3C_LV", "A3C_ZOOM", "A4C", "A4C_LV", "A4C_ZOOM", "A5C",
              "A5C_ZOOM", "PLAX", "PLAX_DEEP", "PLAX_ZOOM", "PLAX_ZOOM_AV", "PSAX_BASAL", "PSAX_MID", "PSAX_APICAL",
              "PSAX_AV", "RVINF", "SUBCOSTAL", "SUPRASTERNAL", "OTHER"]


class TestClasses(unittest.TestCase):

    def test_lists_and_parents(self):
        vc.check_classes()
        self.assertEqual(len(vc.OLD_CLASSES), 20)
        self.assertEqual(list(vc.NEW_CLASSES), FORM_VIEWS)       # the view review form's options, without unsure
        self.assertNotIn("PSAX", vc.NEW_CLASSES)
        self.assertEqual(vc.children("PSAX"), ["PSAX_BASAL", "PSAX_MID", "PSAX_APICAL"])
        self.assertEqual(vc.children("PLAX_ZOOM"), ["PLAX_ZOOM", "PLAX_ZOOM_AV"])

    def test_pseudo_label_classes_leave_out_split_classes(self):
        mapping = vc.pseudo_label_classes()
        self.assertEqual(len(mapping), 18)
        self.assertNotIn("PSAX", mapping)
        self.assertNotIn("PLAX_ZOOM", mapping)
        self.assertTrue(all(k == v for k, v in mapping.items()))
        self.assertNotIn("OTHER", vc.pseudo_label_classes(exclude=["OTHER"]))

    def test_config_class_list_with_form_extras(self):
        mapping = {**{c: vc.PARENT[c] for c in vc.NEW_CLASSES}, **vc.FORM_EXTRAS}
        names, parent = vc.from_config({"classes": mapping})
        self.assertEqual(len(names), 29)
        self.assertEqual(names[-6:], tuple(vc.FORM_EXTRAS))
        mapping_pl = vc.pseudo_label_classes(new=names, parent=parent)
        # A4C (A4C_RV), SUBCOSTAL (4C, IVC) and OTHER (AORTA_ASC) now have several children: no pseudo-labels.
        self.assertEqual(sorted(set(vc.OLD_CLASSES) - set(mapping_pl)),
                         ["A4C", "OTHER", "PLAX_ZOOM", "PSAX", "SUBCOSTAL"])
        self.assertEqual(vc.from_config({}), (vc.NEW_CLASSES, vc.PARENT))
        with self.assertRaises(ValueError):
            vc.from_config({"classes": {**mapping, "RVOT": "NOT_A_CLASS"}})
        with self.assertRaises(ValueError):
            vc.from_config({"classes": {**mapping, "unsure": "OTHER"}})
        with self.assertRaises(ValueError):
            vc.from_config({"classes": ["A4C"]})

    def test_canonical_names(self):
        self.assertEqual(vc.canonical(" PLAX_AV_ZOOM "), "PLAX_ZOOM_AV")
        self.assertEqual(vc.canonical("UNSURE"), "unsure")
        self.assertEqual(vc.canonical(float("nan")), "")
        self.assertEqual(vc.canonical(""), "")

    def test_form_classes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "form.json")
            options = [{"value": v} for v in ("A4C", "RVOT", "unsure")]
            with open(path, "w") as f:
                json.dump({"classifications": [{"name": "view", "options": options}, {"name": "comments"}]}, f)
            self.assertEqual(vc.form_classes(path), ("A4C", "RVOT"))
            self.assertEqual(vc.check_against_form(("A4C", "RVOT", "OTHER"), path), ["OTHER"])
            with self.assertRaises(ValueError):
                vc.check_against_form(("A4C",), path)

    def test_old_mapping_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            pd.Series({str(i): c for i, c in enumerate(vc.OLD_CLASSES)}).to_json(path)
            self.assertEqual(vc.read_old_mapping(path), vc.OLD_CLASSES)
            swapped = list(vc.OLD_CLASSES)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            pd.Series({str(i): c for i, c in enumerate(swapped)}).to_json(path)
            with self.assertRaises(ValueError):
                vc.read_old_mapping(path)

    def test_head_rows_split_parent_probability(self):
        rows = dict(zip(vc.NEW_CLASSES, vc.head_rows()))
        self.assertEqual(rows["PSAX_MID"][0], vc.OLD_CLASSES.index("PSAX"))
        self.assertAlmostEqual(rows["PSAX_MID"][1], -np.log(3))
        self.assertAlmostEqual(rows["PLAX_ZOOM"][1], -np.log(2))
        self.assertEqual(rows["A4C"], (vc.OLD_CLASSES.index("A4C"), 0.0))


def label_table(views, statuses=None):
    n = len(views)
    return pd.DataFrame({"global_key": [f"k{i}" for i in range(n)], "patient_id": [f"p{i}" for i in range(n)],
                         "study_id": [f"s{i}" for i in range(n)], "video_path": [f"/v/{i}.mp4" for i in range(n)],
                         "acq_category": ["bmode"] * n, "predicted_view": ["A4C"] * n, "human_view": views,
                         "label_status": statuses or ["done"] * n, "n_labels": ["1"] * n,
                         "labelled_at": ["2026-11-17"] * n})


class TestLabels(unittest.TestCase):

    def test_filter_and_drop_counts(self):
        t = label_table(["A4C", "unsure", "", "PSAX_MID", "PLAX_AV_ZOOM", "A2C", "UNSURE"],
                        ["done", "done", "done", "done", "done", "skipped", "done"])
        usable, drops = vl.read_labels(t)
        self.assertEqual(list(usable.label), ["A4C", "PSAX_MID", "PLAX_ZOOM_AV"])
        self.assertEqual(list(usable.y), [vc.NEW_CLASSES.index(c) for c in usable.label])
        d = dict(zip(drops.reason, drops.n))
        self.assertEqual(d["unsure"], 2)
        self.assertEqual(d["empty_view"], 1)
        self.assertEqual(d["status_skipped"], 1)
        self.assertEqual(d["status_conflict"], 0)

    def test_unknown_names_stop(self):
        with self.assertRaises(ValueError):
            vl.read_labels(label_table(["PSAX"]))           # the old class without a level is not a label
        with self.assertRaises(ValueError):
            vl.read_labels(label_table(["A4C"], ["finished"]))
        with self.assertRaises(ValueError):
            vl.read_labels(label_table(["A4C", "A2C"]).assign(global_key="same"))
        # A non-class view on a row that is not done is only dropped.
        usable, _ = vl.read_labels(label_table(["A4C", "PSAX"], ["done", "in_rework"]))
        self.assertEqual(len(usable), 1)

    def test_batches_keep_listed_batches_only(self):
        t = label_table(["A4C", "unsure", "A2C", "PSAX_MID", ""], ["done", "done", "done", "unlabelled", "done"])
        t["batch"] = ["1", "1", "2", "2", "2"]
        usable, drops = vl.read_labels(t, batches=[1])
        self.assertEqual(list(usable.global_key), ["k0"])
        d = dict(zip(drops.reason, drops.n))
        self.assertEqual((d["other_batch"], d["unsure"], d["status_unlabelled"], d["empty_view"]), (3, 1, 0, 0))
        self.assertEqual(int(drops.n.sum()), 4)
        usable, drops = vl.read_labels(t, batches=[1, 2])
        self.assertEqual(list(usable.global_key), ["k0", "k2"])
        self.assertEqual(dict(zip(drops.reason, drops.n))["other_batch"], 0)
        # Without `batches`, every row counts and the drop table keeps its earlier reasons.
        usable, drops = vl.read_labels(t)
        self.assertEqual(len(usable), 2)
        self.assertNotIn("other_batch", set(drops.reason))

    def test_batches_need_a_batch_column_and_valid_numbers(self):
        t = label_table(["A4C", "A2C"])
        with self.assertRaisesRegex(ValueError, "no `batch` column"):
            vl.read_labels(t, batches=[1])
        for bad in ([], 1, "1", [0], [1.5], [True]):
            with self.assertRaises(ValueError):
                vl.read_labels(t.assign(batch="1"), batches=bad)
        with self.assertRaisesRegex(ValueError, "not batch numbers"):
            vl.read_labels(t.assign(batch=["1", "x"]), batches=[1])
        # The export's checks still cover the rows of other batches.
        with self.assertRaises(ValueError):
            vl.read_labels(label_table(["A4C", "PSAX"]).assign(batch=["1", "2"]), batches=[1])

    def test_pool_and_split_checks(self):
        usable, _ = vl.read_labels(label_table(["A4C", "A2C"]))
        pool = pd.DataFrame({"global_key": ["k0", "k1"], "patient_id": ["p0", "p1"],
                             "video_path": ["/v/0.mp4", "/v/1.mp4"], "view": ["A4C", "A2C"]})
        self.assertEqual(list(vl.check_against_pool(usable, pool).pool_view), ["A4C", "A2C"])
        with self.assertRaises(ValueError):
            vl.check_against_pool(usable, pool.assign(video_path=["/v/0.mp4", "/v/9.mp4"]))
        with self.assertRaises(ValueError):
            vl.check_against_pool(usable, pool.head(1))
        splits = pd.DataFrame({"patient_id": ["p0", "p1"], "split": ["train", "val"]})
        with self.assertRaises(ValueError):
            vl.check_train_split(usable.patient_id, splits)
        vl.check_train_split(["p0"], splits)


def archive(n_per=6):
    """Synthetic archive: patients q0.. with clips of A4C, PSAX, PLAX_ZOOM and OTHER at various confidences."""
    rows, probs = [], []
    k = 0
    for view in ("A4C", "PSAX", "PLAX_ZOOM", "OTHER", "A2C"):
        for i in range(n_per):
            path = f"/a/{k}.mp4"
            rows.append({"patient_id": f"q{k % 20}", "study_id": f"t{k}", "split": "train", "view": view,
                         "video_path": path, "acq_category": "bmode" if i % 2 else "color_flow",
                         "acq_confidence": "header", "acq_vendor_validated": True, "n_frames": 30})
            p = np.full(len(vc.OLD_CLASSES), 0.0)
            top = 0.95 if i < n_per - 1 else 0.5
            p[vc.OLD_CLASSES.index(view)] = top
            p[vc.OLD_CLASSES.index("A3C")] += 1 - top
            probs.append({"video_path": path, **dict(zip(vc.OLD_CLASSES, p))})
            k += 1
    return pd.DataFrame(rows), pd.DataFrame(probs)


class TestPseudoLabels(unittest.TestCase):

    def test_rules(self):
        videos, probs = archive()
        videos.loc[0, "acq_category"] = "spectral_doppler"      # not an imaging mode
        videos.loc[2, "n_frames"] = 2                            # a still
        videos.loc[4, "acq_confidence"] = "pixels"               # validated vendor, mode not from the header
        train = {f"q{i}" for i in range(20)} - {"q19"}
        out = vl.select_pseudo_labels(videos, probs, train, excluded_patients={"q3"}, threshold=0.9, per_class=100)
        self.assertGreater(len(out), 5)
        self.assertTrue(set(out.old_class) <= {"A4C", "OTHER", "A2C"})       # never PSAX or PLAX_ZOOM
        self.assertFalse({"/a/0.mp4", "/a/2.mp4", "/a/4.mp4"} & set(out.video_path))
        self.assertTrue((out.pmax >= 0.9).all())
        self.assertFalse({"q3", "q19"} & set(out.patient_id))
        self.assertFalse(out.patient_id.duplicated().any())                  # one clip per patient
        self.assertTrue((out.label == out.old_class).all())
        no_other = vc.pseudo_label_classes(exclude=["OTHER"])
        capped = vl.select_pseudo_labels(videos, probs, train, set(), no_other, per_class=1)
        self.assertEqual(sorted(capped.old_class), ["A2C", "A4C"])
        again = vl.select_pseudo_labels(videos, probs, train, set(), no_other, per_class=1)
        pd.testing.assert_frame_equal(capped, again)

    def test_argmax_must_match_view(self):
        videos, probs = archive()
        videos.loc[videos.view == "A4C", "view"] = "A2C"        # the manifest disagrees with the probabilities
        out = vl.select_pseudo_labels(videos, probs, {f"q{i}" for i in range(20)}, set())
        self.assertEqual(int((out.video_path.isin(videos.video_path[:6])).sum()), 0)


def fold_input(seed=0):
    rng = np.random.default_rng(seed)
    counts = {"PSAX_MID": 150, "PSAX_BASAL": 120, "PLAX_ZOOM": 90, "PLAX_ZOOM_AV": 60, "A4C": 22, "A2C": 21,
              "RVINF": 7, "SUPRASTERNAL": 3, "OTHER": 1}
    labels = [c for c, n in counts.items() for _ in range(n)]
    t = pd.DataFrame({"patient_id": [f"p{i}" for i in range(len(labels))], "label": labels})
    return t.sample(frac=1.0, random_state=int(rng.integers(1000))).reset_index(drop=True), counts


class TestFolds(unittest.TestCase):

    def test_stratified_and_deterministic(self):
        t, counts = fold_input()
        f = vf.assign_folds(t, k=5, seed=3)
        self.assertEqual(sorted(f.unique()), [0, 1, 2, 3, 4])
        per = pd.crosstab(t.label, f)
        for c, n in counts.items():
            self.assertLessEqual(per.loc[c].max() - per.loc[c].min(), 1, c)   # each class spread evenly
        self.assertLessEqual(f.value_counts().max() - f.value_counts().min(), 1)
        shuffled = t.sample(frac=1.0, random_state=7)
        pd.testing.assert_series_equal(vf.assign_folds(shuffled, k=5, seed=3).sort_index(), f.sort_index(),
                                       check_names=False)                    # row order does not matter
        self.assertFalse(vf.assign_folds(t, k=5, seed=4).equals(f))          # the seed does

    def test_patients_stay_together(self):
        t = pd.DataFrame({"patient_id": ["a", "a", "b", "b", "c", "d", "e", "f"],
                          "label": ["A4C", "A2C", "A4C", "A4C", "A2C", "A4C", "A2C", "A4C"]})
        t["fold"] = vf.assign_folds(t, k=3)
        vf.check_folds(t)
        self.assertEqual(t.groupby("patient_id").fold.nunique().max(), 1)
        bad = t.copy()
        bad.loc[0, "fold"] = (bad.loc[1, "fold"] + 1) % 3
        with self.assertRaises(ValueError):
            vf.check_folds(bad)

    def test_inner_split(self):
        t, counts = fold_input()
        held = vf.inner_split(t, fraction=0.2, min_train=2, seed=0)
        per = t[held].label.value_counts()
        self.assertEqual(per.get("PSAX_MID"), 30)
        self.assertEqual(per.get("SUPRASTERNAL", 0), 1)        # round(0.6) = 1, leaving 2 in training
        self.assertEqual(per.get("OTHER", 0), 0)               # a single clip stays in training
        self.assertTrue(held.equals(vf.inner_split(t, fraction=0.2, min_train=2, seed=0)))


class TestMetrics(unittest.TestCase):

    def test_confusion_and_scores(self):
        truth = [0, 0, 1, 1, 2]
        probs = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .8, .1], [.1, .6, .3], [.1, .2, .7]])
        m = vm.confusion_matrix(truth, probs.argmax(1), 3)
        np.testing.assert_array_equal(m, [[1, 1, 0], [0, 2, 0], [0, 0, 1]])
        self.assertAlmostEqual(vm.accuracy(truth, probs.argmax(1)), 0.8)
        # F1: class 0 (p 1, r .5) 2/3; class 1 (p 2/3, r 1) 0.8; class 2: 1.
        self.assertAlmostEqual(vm.macro_f1(truth, probs.argmax(1), 3), (2 / 3 + 0.8 + 1) / 3)
        self.assertAlmostEqual(vm.balanced_accuracy(truth, probs.argmax(1), 3), (0.5 + 1 + 1) / 3)
        frame = vm.confusion_frame(truth, probs.argmax(1), ["a", "b", "c", "unused"])
        self.assertEqual(list(frame.columns), ["a", "b", "c"])

    def test_macro_f1_over_labels_and_predictions(self):
        # Class 2 is never true and never predicted (left out); class 3 is predicted but never true (F1 0).
        self.assertAlmostEqual(vm.macro_f1([0, 1], [0, 3], 4), (1 + 0 + 0) / 3)
        self.assertAlmostEqual(vm.balanced_accuracy([0, 1], [0, 3], 4), (1 + 0) / 2)

    def test_small_support_auroc_is_flagged(self):
        rng = np.random.default_rng(0)
        truth = np.r_[np.zeros(40, int), np.ones(10, int), np.full(3, 2)]
        probs = rng.dirichlet(np.ones(3), len(truth))
        probs[truth == 1, 1] += 1
        probs /= probs.sum(1, keepdims=True)
        per = vm.per_class(truth, probs, ["a", "b", "c"], min_positives=5, samples=200)
        self.assertEqual(list(per.auroc_reported), [True, True, False])
        self.assertIn("fewer than 5", per.note[2])
        self.assertTrue(np.isnan(per.auroc_low[2]))
        self.assertLessEqual(per.auroc_low[1], per.auroc[1])
        self.assertLessEqual(per.auroc[1], per.auroc_high[1])
        s = vm.summary(truth, probs, samples=100)
        self.assertAlmostEqual(s["mean_auroc"], per.auroc[:2].mean())
        self.assertEqual(s["n"], len(truth))
        self.assertLessEqual(s["accuracy_low"], s["accuracy"])

    def test_minimum_support_restricts_the_averages(self):
        truth = np.r_[np.zeros(20, int), np.ones(20, int), [2]]
        probs = np.full((len(truth), 3), 0.1)
        probs[np.arange(len(truth)), truth] = 0.8
        probs[-1] = [0.8, 0.1, 0.1]                          # the single class-2 clip is wrong
        every = vm.summary(truth, probs, samples=0)
        some = vm.summary(truth, probs, samples=0, evaluated=[True, True, False])
        self.assertAlmostEqual(some["accuracy"], 40 / 41)      # accuracy counts every clip
        self.assertGreater(some["macro_f1"], every["macro_f1"])
        self.assertAlmostEqual(some["macro_f1_all"], every["macro_f1"])
        self.assertAlmostEqual(some["balanced_accuracy"], 1.0)
        per = vm.per_class(truth, probs, ["a", "b", "c"], samples=0, evaluated=[True, False, True])
        self.assertEqual(list(per.auroc_reported), [True, False, False])
        self.assertEqual(per.note[1], "below the minimum support")

    def test_collapse_and_weights(self):
        new = ("A", "A1", "B")
        probs = np.array([[.2, .3, .5]])
        out = vm.collapse_probabilities(probs, new, {"A": "A", "A1": "A", "B": "B"}, ("B", "A"))
        np.testing.assert_allclose(out, [[.5, .5]])
        w = vm.archive_weights(["X", "X", "Y"], {"X": 10, "Y": 90})
        np.testing.assert_allclose(w, [0.1 / (2 / 3), 0.1 / (2 / 3), 0.9 / (1 / 3)])
        with self.assertRaises(ValueError):
            vm.archive_weights(["Z"], {"X": 1})

    def test_paired_difference(self):
        d = vm.paired_difference([0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1], 2, samples=50)
        self.assertAlmostEqual(d["accuracy_diff"], 0.5)


if __name__ == "__main__":
    unittest.main()
