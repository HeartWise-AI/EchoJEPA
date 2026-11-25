# tests/models/test_view_classifier_colour_merge.py

"""The view classifier's own video readers undo the legacy YBR colour swap exactly as the dataset loader does (only
for clips under DeepEcho/MP4/, not with ECHOJEPA_LEGACY_YBR_FIX=0; OpenCV's BGR frames turned to RGB first); a
snapshot's `legacy_ybr_fix` is enforced against the decoding and the token cache; `label_merge` maps merged label
names into a smaller class list, and the snapshot comparison scores an earlier, larger class list on the later one by
adding the merged classes' probabilities. CPU only, synthetic videos."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from evals.view_classifier import classes as vc
from evals.view_classifier import compare_runs as cr
from evals.view_classifier import io as vio
from evals.view_classifier import mvit
from evals.view_classifier import preprocess_variants as pv
from src.datasets.utils import color_order as co

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AVI = ".avi"


def write_colour_video(path, t=40, h=48, w=64):
    """A lossless (FFV1) video of coloured blocks; returns the RGB frames written."""
    import cv2
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    frames[:, :, :] = np.linspace(20, 200, w, dtype=np.uint8)[None, None, :, None]
    frames[:, 5:20, 5:20] = (220, 30, 30)
    frames[:, 5:20, 30:45] = (30, 40, 220)
    frames[:, 28:44, 10:50] = (200, 160, 90)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 30, (w, h))
    for f in frames:
        writer.write(f[..., ::-1].copy())
    writer.release()
    return frames


class ColourEnv(unittest.TestCase):
    """A temporary folder with one clip under DeepEcho/MP4/ and one elsewhere; no exception list; fix on."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        ex = os.path.join(root, "exceptions.txt")
        open(ex, "w").close()
        self.env = mock.patch.dict(os.environ, {"ECHOJEPA_YBR_EXCEPTIONS": ex})
        self.env.start()
        os.environ.pop("ECHOJEPA_LEGACY_YBR_FIX", None)
        self.cache = mock.patch.object(co, "_EXCEPTIONS", None)
        self.cache.start()
        self.legacy = os.path.join(root, "DeepEcho", "MP4", "2019", "1", "clip" + AVI)
        self.other = os.path.join(root, "elsewhere", "clip" + AVI)
        write_colour_video(self.legacy)
        write_colour_video(self.other)

    def tearDown(self):
        self.cache.stop()
        self.env.stop()
        self.tmp.cleanup()


class TestReaders(ColourEnv):

    def raw(self, path, n=None):
        from decord import VideoReader, cpu
        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        return vr.get_batch(list(range(n or len(vr)))).asnumpy()

    def test_echoprime_read(self):
        raw = self.raw(self.legacy, 32)[::2][:16]
        got = mvit.echoprime_read(self.legacy, 16, 2)
        np.testing.assert_array_equal(got, co.undo_legacy_ybr_swap(raw, "RGB"))
        self.assertGreater(np.abs(got.astype(int) - raw.astype(int)).max(), 50)      # colours really change
        np.testing.assert_array_equal(mvit.echoprime_read(self.other, 16, 2), self.raw(self.other, 32)[::2][:16])
        with mock.patch.dict(os.environ, {"ECHOJEPA_LEGACY_YBR_FIX": "0"}):
            np.testing.assert_array_equal(mvit.echoprime_read(self.legacy, 16, 2), raw)

    def test_decode_frames(self):
        raw = self.raw(self.legacy)
        np.testing.assert_array_equal(pv.decode_frames(self.legacy), co.undo_legacy_ybr_swap(raw, "RGB"))
        np.testing.assert_array_equal(pv.decode_frames(self.other), self.raw(self.other))
        with mock.patch.dict(os.environ, {"ECHOJEPA_LEGACY_YBR_FIX": "0"}):
            np.testing.assert_array_equal(pv.decode_frames(self.legacy), raw)

    def test_decode_frames_cv2_is_exact_in_either_order(self):
        import cv2
        cap = cv2.VideoCapture(self.legacy)
        bgr = []
        while len(bgr) < pv.WINDOW:
            ok, f = cap.read()
            if not ok:
                break
            bgr.append(f)
        cap.release()
        bgr = np.stack(bgr)
        got = pv.decode_frames_cv2(self.legacy)
        via_bgr = co.undo_legacy_ybr_swap(bgr, "BGR")[..., ::-1]                   # fixed in BGR, then to RGB
        np.testing.assert_array_equal(got, via_bgr)
        np.testing.assert_array_equal(got, co.undo_legacy_ybr_swap(bgr[..., ::-1].copy(), "RGB"))
        with mock.patch.dict(os.environ, {"ECHOJEPA_LEGACY_YBR_FIX": "0"}):
            np.testing.assert_array_equal(pv.decode_frames_cv2(self.legacy), bgr[..., ::-1])

    def test_clip_decoder_both_reads_fix(self):
        ep = mvit.ClipDecoder([self.legacy], 16, 2, 32, preprocessing="full_squash_aa", frame_read="echoprime")[0][1]
        cache = mvit.ClipDecoder([self.legacy], 16, 2, 32, preprocessing="full_squash_aa", frame_read="cache")[0][1]
        with mock.patch.dict(os.environ, {"ECHOJEPA_LEGACY_YBR_FIX": "0"}):
            ep0 = mvit.ClipDecoder([self.legacy], 16, 2, 32, preprocessing="full_squash_aa",
                                   frame_read="echoprime")[0][1]
            cache0 = mvit.ClipDecoder([self.legacy], 16, 2, 32, preprocessing="full_squash_aa",
                                      frame_read="cache")[0][1]
        self.assertFalse(np.array_equal(ep.numpy(), ep0.numpy()))
        self.assertFalse(np.array_equal(cache.numpy(), cache0.numpy()))
        # the two reads take the same frames here (no padding), so the colour fix acts the same way on both
        np.testing.assert_array_equal(ep.numpy(), cache.numpy())


class TestSnapshotColourSetting(ColourEnv):

    def test_check(self):
        self.assertTrue(vio.check_colour_fix({}))                        # no key: not checked
        self.assertTrue(vio.check_colour_fix({"legacy_ybr_fix": True}))
        with self.assertRaises(SystemExit):
            vio.check_colour_fix({"legacy_ybr_fix": False})
        with mock.patch.dict(os.environ, {"ECHOJEPA_LEGACY_YBR_FIX": "0"}):
            self.assertFalse(vio.check_colour_fix({"legacy_ybr_fix": False}))
            with self.assertRaises(SystemExit):
                vio.check_colour_fix({"legacy_ybr_fix": True})
        fixed, old = {"provenance": {"legacy_ybr_fix": True}}, {"provenance": {}}
        self.assertTrue(vio.check_colour_fix({"legacy_ybr_fix": True}, fixed))
        with self.assertRaises(SystemExit):
            vio.check_colour_fix({"legacy_ybr_fix": True}, old)        # a cache from before the fix

    def test_cache_provenance_records_the_setting(self):
        from evals.radar_frozen import build_token_cache as btc
        clips = os.path.join(self.tmp.name, "clips.csv")
        pd.DataFrame({"clip_id": [self.legacy], "split": ["train"]}).to_csv(clips, index=False)
        enc = os.path.join(self.tmp.name, "enc.pt")
        open(enc, "wb").write(b"x")
        config = {"encoder": {"checkpoint": enc}, "data": {"spatial": "full_squash", "resize": "bicubic_aa"},
                  "cache": {}}
        args = mock.Mock(clips=clips, split="train", exclude=[], chunks=1)
        self.assertTrue(btc.provenance(config, args, 1, 0)["legacy_ybr_fix"])
        with mock.patch.dict(os.environ, {"ECHOJEPA_LEGACY_YBR_FIX": "0"}):
            self.assertFalse(btc.provenance(config, args, 1, 0)["legacy_ybr_fix"])


# ---------------------------------------------------------------------------------------------- merged class lists

LARGE = dict(vc.PARENT, **vc.FORM_EXTRAS)                       # the 29 classes of the earlier snapshots
MERGE = {"RVOT": "PSAX_AV", "SUBCOSTAL_4C": "SUBCOSTAL", "SUBCOSTAL_IVC": "SUBCOSTAL", "A4C_RV": "A4C",
         "PLAX_ZOOM_MV": "PLAX_ZOOM", "AORTA_ASC": "PLAX"}
SMALL = {c: vc.PARENT[c] for c in vc.NEW_CLASSES}               # the 23 trained classes


class TestLabelMerge(unittest.TestCase):

    def test_spec_views_are_the_classes_plus_a_level_less_psax(self):
        with open(os.path.join(REPO, "configs", "radar", "deepecho_video_clip_annotation_specification.json")) as f:
            views = set(json.load(f)["views"])
        self.assertEqual(views, set(vc.NEW_CLASSES) | {"PSAX"})
        self.assertNotIn("PSAX", vc.NEW_CLASSES)
        self.assertTrue(set(MERGE.values()) <= set(vc.NEW_CLASSES))

    def test_label_merge_checks(self):
        self.assertEqual(vc.label_merge({"classes": SMALL}), {})
        self.assertEqual(vc.label_merge({"classes": SMALL, "label_merge": MERGE}), MERGE)
        with self.assertRaises(ValueError):
            vc.label_merge({"classes": SMALL, "label_merge": {"RVOT": "NOPE"}})
        with self.assertRaises(ValueError):
            vc.label_merge({"classes": SMALL, "label_merge": {"A4C": "A2C"}})       # a class cannot be merged

    def test_read_labels_and_form(self):
        from evals.view_classifier import labels as vl
        t = pd.DataFrame({"global_key": ["a", "b", "c"], "patient_id": ["1", "2", "3"], "study_id": ["s"] * 3,
                          "video_path": ["/x/1", "/x/2", "/x/3"], "label_status": ["done"] * 3,
                          "human_view": ["RVOT", "A4C", "unsure"], "predicted_view": ["PSAX"] * 3})
        for c in vl.LABEL_COLUMNS:
            if c not in t.columns:
                t[c] = ""
        with self.assertRaises(ValueError):
            vl.read_labels(t, vc.NEW_CLASSES)
        usable, _ = vl.read_labels(t, vc.NEW_CLASSES, merge=MERGE)
        self.assertEqual(list(usable.label), ["PSAX_AV", "A4C"])
        self.assertEqual(list(usable.y), [vc.NEW_CLASSES.index("PSAX_AV"), vc.NEW_CLASSES.index("A4C")])
        with tempfile.TemporaryDirectory() as tmp:
            form = os.path.join(tmp, "form.json")
            opts = [{"value": c} for c in list(vc.NEW_CLASSES) + list(MERGE) + ["unsure"]]
            json.dump({"classifications": [{"name": "view", "options": opts}]}, open(form, "w"))
            with self.assertRaises(ValueError):
                vc.check_against_form(vc.NEW_CLASSES, form)
            self.assertEqual(vc.check_against_form(vc.NEW_CLASSES, form, MERGE), [])

    def test_read_folds_merges_and_keeps_folds(self):
        large = tuple(LARGE)
        labels = ["RVOT", "A4C", "AORTA_ASC", "PSAX_MID"]
        t = pd.DataFrame({"clip_id": [f"/v/{i}" for i in range(4)], "patient_id": list("abcd"), "label": labels,
                          "y": [large.index(c) for c in labels], "fold": [0, 1, 2, 3]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "folds.csv")
            t.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                vio.read_folds(path, vc.NEW_CLASSES)                    # y is the larger list's
            got = vio.read_folds(path, vc.NEW_CLASSES, MERGE)
        self.assertEqual(list(got.label), ["PSAX_AV", "A4C", "PLAX", "PSAX_MID"])
        self.assertEqual(list(got.label_as_written), labels)
        self.assertEqual(list(got.y), [vc.NEW_CLASSES.index(c) for c in got.label])
        self.assertEqual(list(got.fold), [0, 1, 2, 3])

    def test_compare_scores_the_larger_list_on_the_smaller_one(self):
        large = tuple(LARGE)
        rng = np.random.default_rng(0)
        n = 60
        labels = [large[i] for i in rng.integers(0, len(large), n)]
        folds = pd.DataFrame({"clip_id": [f"/v/{i}" for i in range(n)], "patient_id": [str(i) for i in range(n)],
                              "label": labels, "y": [large.index(c) for c in labels], "fold": np.arange(n) % 5})
        with tempfile.TemporaryDirectory() as tmp:
            fpath = os.path.join(tmp, "folds.csv")
            folds.to_csv(fpath, index=False)
            # earlier snapshot: puts all its mass on the label as written (a merged view included)
            p_large = np.zeros((n, len(large)))
            p_large[np.arange(n), folds.y] = 1.0
            # later snapshot: right on the first half only
            merged = [MERGE.get(c, c) for c in labels]
            small = vc.NEW_CLASSES
            p_small = np.zeros((n, len(small)))
            right = np.arange(n) < n // 2
            p_small[np.arange(n), [small.index(c) if r else (small.index(c) + 1) % len(small)
                                   for c, r in zip(merged, right)]] = 1.0
            configs = []
            for name, classes, probs, extra in (("s3", LARGE, p_large, {}),
                                                ("s4", SMALL, p_small, {"label_merge": MERGE})):
                runs = os.path.join(tmp, name)
                folder = os.path.join(runs, "arm_b", "pseudo_off")
                os.makedirs(folder)
                oof = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
                oof.insert(0, "clip_id", folds.clip_id)
                oof.to_csv(os.path.join(folder, "oof_predictions.csv"), index=False)
                os.makedirs(os.path.join(runs, "report"))
                pd.DataFrame([{"arm": "arm_b", "pseudo": "off", "subset": "all", "granularity": "new", "n": n,
                               "accuracy": 0.5}]).to_csv(os.path.join(runs, "report", "summary.csv"), index=False)
                configs.append({"paths": {"runs": runs, "folds": fpath}, "classes": classes,
                                "report": {"bootstrap": 100, "seed": 0}, **extra})
            _, pairs = cr.compare(configs, os.path.join(tmp, "s4", "report_compare"))
            new = pairs[pairs.granularity == "new"].iloc[0]
            self.assertAlmostEqual(new.accuracy_earlier, 1.0)            # merged probabilities count for the target
            self.assertAlmostEqual(new.accuracy_later, 0.5)
            self.assertTrue(new.earlier_merged and not new.later_merged)
            self.assertIn("merged classes", open(os.path.join(tmp, "s4", "report_compare", "compare.md")).read())
            with self.assertRaises(ValueError):                         # the larger list last: RVOT has no target
                cr.compare(configs[::-1], os.path.join(tmp, "s3", "x"))

    def test_to_target_sums(self):
        p = np.array([[0.1, 0.2, 0.7]])
        np.testing.assert_allclose(cr.to_target(p, ["A", "B", "C"], ["A", "A", "C"], ["A", "C"]), [[0.3, 0.7]])


if __name__ == "__main__":
    unittest.main()
