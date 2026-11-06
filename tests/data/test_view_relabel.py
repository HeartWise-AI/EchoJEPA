# tests/data/test_view_relabel.py

"""view_relabel.py on synthetic tables: the quotas, which clips may be drawn, one clip per patient, what the screen
leaves, and the upload folder `prepare` writes."""

import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import radar_pilot as rp
import view_relabel as vl
from tools.labelbox.pilot_upload import VIEW_RELABEL, read_plan

CLASSES = ["A2C", "A2C_LV", "A2C_ZOOM", "A3C", "A3C_LV", "A3C_ZOOM", "A4C", "A4C_LV", "A4C_ZOOM", "A5C", "A5C_ZOOM",
           "OTHER", "PLAX", "PLAX_DEEP", "PLAX_ZOOM", "PSAX", "PSAX_AV", "RVINF", "SUBCOSTAL", "SUPRASTERNAL"]


def manifest(rows):
    """Clips as the manifest has them; each row may override a default."""
    base = dict(split="train", view="A4C", acq_category="bmode", acq_confidence="header", acq_vendor_validated=True,
                n_frames=40, vendor="GE Vingmed Ultrasound", model="E95", dicom_path="d", video_path=None,
                patient_id=None, study_id=None)
    out = []
    for i, row in enumerate(rows):
        clip = {**base, "patient_id": f"p{i}", "study_id": f"s{i}", "video_path": f"/a/b/{i}/clip{i}.mp4", **row}
        out.append(clip)
    return pd.DataFrame(out)


class TestQuotas(unittest.TestCase):
    def test_the_set_is_400_psax_200_plax_zoom_and_400_spread_evenly(self):
        q = vl.quotas(CLASSES)
        self.assertEqual((q["PSAX"], q["PLAX_ZOOM"], sum(q.values())), (400, 200, 1000))
        others = [n for view, n in q.items() if view not in vl.UPSAMPLED]
        self.assertEqual(len(others), 18)
        self.assertLessEqual(max(others) - min(others), 1)

    def test_a_missing_upsampled_class_stops(self):
        with self.assertRaisesRegex(SystemExit, "PLAX_ZOOM"):
            vl.quotas([c for c in CLASSES if c != "PLAX_ZOOM"])


class TestEligible(unittest.TestCase):
    def kept(self, rows, patients=(), tails=()):
        return list(vl.eligible_clips(manifest(rows), set(patients), set(tails)).patient_id)

    def test_color_doppler_and_other_stay(self):
        self.assertEqual(self.kept([{"acq_category": "color_flow"}, {"view": "OTHER"}]), ["p0", "p1"])

    def test_only_the_train_split(self):
        self.assertEqual(self.kept([{"split": "test"}, {"split": "val"}, {}]), ["p2"])

    def test_no_still_no_unknown_mode_and_no_clip_without_a_view(self):
        self.assertEqual(self.kept([{"n_frames": 3}, {"acq_category": "unknown"}, {"view": None}, {}]), ["p3"])

    def test_no_listed_patient_or_clip(self):
        self.assertEqual(self.kept([{}, {}, {}], patients=["p0"], tails=["b/1/clip1"]), ["p2"])

    def test_a_listed_clip_matches_whatever_root_and_extension_the_list_has(self):
        tails = set(vl.path_tail(pd.Series(["/other/root/a/b/7/clip7.avi"])))
        self.assertEqual(tails, set(vl.path_tail(pd.Series(["/a/b/7/clip7.mp4"]))))


class TestDraw(unittest.TestCase):
    def test_one_clip_per_patient_over_all_views(self):
        rows = [{"view": v, "patient_id": f"p{i}"} for i in range(6) for v in ("PSAX", "A2C_ZOOM")]
        drawn = vl.draw_clips(manifest(rows), {"PSAX": 3, "A2C_ZOOM": 3}, seed=1)
        self.assertEqual(len(drawn), 6)
        self.assertTrue(drawn.patient_id.is_unique)

    def test_the_rarest_view_chooses_first(self):
        rows = ([{"view": "A2C_ZOOM", "patient_id": "shared"}]
                + [{"view": "PSAX", "patient_id": "shared"}] + [{"view": "PSAX"} for _ in range(5)])
        drawn = vl.draw_clips(manifest(rows), {"PSAX": 6, "A2C_ZOOM": 1}, seed=1)
        self.assertEqual(drawn[drawn.view == "A2C_ZOOM"].patient_id.tolist(), ["shared"])

    def test_the_same_seed_draws_the_same_clips(self):
        clips = manifest([{"view": "PSAX"} for _ in range(30)])
        one, two = (vl.draw_clips(clips, {"PSAX": 5}, seed=7).video_path.tolist() for _ in range(2))
        self.assertEqual(one, two)
        self.assertNotEqual(one, vl.draw_clips(clips, {"PSAX": 5}, seed=8).video_path.tolist())

    def test_reference_clips_are_other_patients_of_the_drawn_layouts_and_never_other(self):
        clips = manifest([{"patient_id": "drawn"}, {}, {"view": "OTHER"}, {"model": "elsewhere"}])
        reference = vl.reference_clips(clips, clips.iloc[:1], per_layout=5, seed=1)
        self.assertEqual(reference.patient_id.tolist(), ["p1"])


class TestKeys(unittest.TestCase):
    def test_the_keys_are_the_uploaders_and_say_nothing_of_the_path(self):
        key = vl.global_key(b"secret", "/a/b/clip.mp4")
        self.assertTrue(VIEW_RELABEL.global_key.fullmatch(key))
        self.assertTrue(VIEW_RELABEL.study_key.fullmatch(vl.set_key(b"secret")))
        self.assertNotIn("clip", key)


def screened(rows):
    base = dict(view="A4C", decision="keep", vendor="GE Vingmed Ultrasound", dicom_image_content="0001",
                acq_category="bmode")
    return pd.DataFrame([{**base, **row} for row in rows])


class TestPick(unittest.TestCase):
    def test_only_what_the_screen_keeps_in_drawn_order(self):
        clips = screened([{"decision": "exclude", "video_path": "a"}, {"decision": "uncertain", "video_path": "b"},
                          {"video_path": "c"}, {"video_path": "d"}])
        self.assertEqual(vl.pick_clips(clips, {"A4C": 1}).video_path.tolist(), ["c"])

    def test_no_ge_3d_or_multi_panel_code(self):
        clips = screened([{"dicom_image_content": "0041", "video_path": "a"}, {"video_path": "b"}])
        self.assertEqual(vl.pick_clips(clips, {"A4C": 1}).video_path.tolist(), ["b"])

    def test_color_doppler_is_kept(self):
        clips = screened([{"acq_category": "color_flow", "video_path": "a"}])
        self.assertEqual(len(vl.pick_clips(clips, {"A4C": 1})), 1)

    def test_a_view_short_of_its_quota_stops(self):
        with self.assertRaisesRegex(SystemExit, r"A4C.*1 of 2"):
            vl.pick_clips(screened([{"video_path": "a"}]), {"A4C": 2})


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.dir = self.root / "set"
        self.dir.mkdir(mode=0o700)
        (self.dir / rp.SECRET_NAME).write_text("secret")
        views = ["PSAX", "PLAX_ZOOM", "A4C"]
        clips = manifest([{"view": v, "form": "view_review", "mode_source": "header",
                           "global_key": vl.global_key(b"secret", f"/a/b/{i}/clip{i}.mp4")}
                          for i, v in enumerate(views)])
        clips[list(rp.CLIP_COLUMNS)].to_csv(self.dir / "shortlist_clips.csv", index=False)
        flags = clips[["video_path"]].assign(decision="keep", decision_reasons="", pixel_mode="bmode")
        for column in rp.MEDIA_COLUMNS:
            flags[column] = "x"
        flags["dicom_image_content"] = "0001"
        self.flags = self.dir / "flags.parquet"
        flags.astype(str).to_parquet(self.flags)
        self.write_info({"date": "2026-10-30"})
        (self.dir / vl.RECORD).write_text(json.dumps({"quotas": {v: 1 for v in views}}))

    def write_info(self, calibration):
        Path(str(self.flags) + ".info.json").write_text(json.dumps({"calibration": calibration}))

    def run_prepare(self, changed=()):
        def encode(_ffmpeg, _source, target):
            Path(target).write_bytes(b"video " + target.encode())

        args = mock.Mock(out_dir=str(self.dir), flags=str(self.flags), out=self.root / "upload", ffmpeg="ffmpeg")
        with mock.patch.object(rp, "screen_chain", return_value={}), \
                mock.patch.object(vl.clip_screen, "changed_media", return_value=list(changed)), \
                mock.patch("tools.labelbox.workflow_test.encode", encode), \
                contextlib.redirect_stdout(io.StringIO()):
            vl.prepare(args)
        return args.out

    def test_the_folder_passes_the_uploaders_checks_in_one_batch(self):
        plan = read_plan(self.run_prepare())
        self.assertEqual(plan.kind, VIEW_RELABEL)
        self.assertEqual(sorted(plan.views.values()), ["A4C", "PLAX_ZOOM", "PSAX"])
        self.assertEqual(len(plan.rows), 3)
        self.assertEqual(len(plan.batches("view_review")), 1)

    def test_no_path_or_identifier_reaches_the_upload_folder(self):
        out = self.run_prepare()
        text = "".join(p.read_text() for p in (out / "items").iterdir()) + (out / vl.PREPARE_RECORD).read_text()
        self.assertFalse(re.search(r"/a/b/|\bp[0-9]\b|\bs[0-9]\b", text))

    def test_a_private_list_maps_the_keys_to_the_clips(self):
        self.run_prepare()
        picked = pd.read_csv(self.dir / vl.PICKED, dtype=str)
        self.assertEqual(len(picked), 3)
        self.assertEqual(os.stat(self.dir / vl.PICKED).st_mode & 0o077, 0)

    def test_a_provisional_screen_is_refused(self):
        self.write_info(None)
        with self.assertRaisesRegex(SystemExit, "provisional"):
            self.run_prepare()
        self.assertFalse((self.root / "upload").exists())

    def test_a_clip_changed_since_the_screen_stops(self):
        with self.assertRaisesRegex(SystemExit, "no longer hold"):
            self.run_prepare(changed=["key"])
        self.assertFalse((self.root / "upload").exists())

    def test_a_used_folder_is_refused(self):
        (self.root / "upload").mkdir()
        (self.root / "upload" / "file").write_text("x")
        with self.assertRaisesRegex(SystemExit, "new or empty"):
            self.run_prepare()


if __name__ == "__main__":
    unittest.main()
