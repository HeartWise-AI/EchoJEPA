# tests/data/test_screen_overlays.py

"""screen_overlays.py on synthetic clips: the overlays drawn where and when each variant says, the copies made per
base clip as the archive encodes its clips, and the report of what the signals raise per variant."""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import clip_screen as cs
import screen_overlays as so

HAS_DECODER = importlib.util.find_spec("decord") is not None
HAS_FFMPEG = importlib.util.find_spec("imageio_ffmpeg") is not None
H, W, T = 434, 636, 8
BBOX = (150, 40, 500, 420)


class TestOverlays(unittest.TestCase):
    def changed(self, kind, start=0.0, bbox=BBOX):
        frames = np.zeros((T, H, W, 3), np.uint8)
        drawn = so.overlay(frames, kind, start, bbox, "GE Vingmed Ultrasound", np.random.default_rng(0))
        return (drawn != frames).any(axis=3)

    def test_marks_are_drawn_inside_the_scan_area(self):
        x0, y0, x1, y1 = BBOX
        for kind in ("pair+line", "pair+", "pairx+line", "single+", "trace_dotted", "trace_solid"):
            changed = self.changed(kind)
            self.assertTrue(changed.any(axis=(1, 2)).all(), kind)  # on every frame.
            outside = changed.copy()
            outside[:, y0:y1, x0:x1] = False
            self.assertFalse(outside.any(), kind)
            self.assertTrue((changed == changed[0]).all(), kind)  # static: the same pixels on every frame.
        self.assertFalse(self.changed(None).any())  # the control is the clip unchanged.

    def test_a_pair_is_two_calipers_apart_joined_by_dots_or_not(self):
        import cv2

        for kind, parts in (("pair+", 2), ("pair+line", None), ("single+", 1)):
            n, _ = cv2.connectedComponents(self.changed(kind)[0].astype(np.uint8), connectivity=8)
            if parts is None:
                self.assertGreater(n - 1, 4, kind)  # two calipers and the dots between them.
            else:
                self.assertEqual(n - 1, parts, kind)

    def test_a_late_overlay_starts_halfway(self):
        changed = self.changed("pair+line", start=0.5).any(axis=(1, 2))
        self.assertEqual(changed.tolist(), [False] * (T // 2) + [True] * (T - T // 2))

    def test_the_results_box_goes_beside_the_scan_area_when_there_is_room(self):
        x0, y0, x1, y1 = BBOX
        changed = self.changed("results")[0]
        ys, xs = np.nonzero(changed)
        self.assertLess(xs.max(), x0)  # left of the scan area, where the frame leaves room.
        # A scan area over the whole width: above it, else in its corner.
        self.assertEqual(so.results_box((H, W), (0, 200, W, H), 1.0), (2, 2))
        self.assertEqual(so.results_box((H, W), (0, 0, W, H), 1.0), (2, 2))

    def test_sizes_follow_the_frame_height(self):
        small = self.changed("single+")[0]
        frames = np.zeros((2, 2 * H, 2 * W, 3), np.uint8)
        big = (so.overlay(frames, "single+", 0, tuple(2 * v for v in BBOX), "Philips Medical Systems",
                          np.random.default_rng(0)) != frames).any(axis=3)[0]
        self.assertEqual(big.sum(), 2 * small.sum() - 1)  # each arm twice as long, one shared centre.


@unittest.skipUnless(HAS_DECODER and HAS_FFMPEG, "needs decord and imageio-ffmpeg")
class TestMake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def base(self):
        import cv2

        rng = np.random.default_rng(1)
        path = os.path.join(self.tmp, "base.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
        for _ in range(T):
            frame = np.zeros((H, W, 3), np.uint8)
            frame[BBOX[1]:BBOX[3], BBOX[0]:BBOX[2]] = rng.integers(0, 120, (BBOX[3] - BBOX[1], BBOX[2] - BBOX[0], 1))
            writer.write(frame)
        writer.release()
        flags = pd.DataFrame([{"global_key": "k1", "patient_id": "p1", "study_id": "s1", "vendor": "GEMS Ultrasound",
                               "model": "Vivid i", "acq_category": "bmode", "view": "A4C", "dicom_path": None,
                               "n_frames": T, "fps": 25.0, "video_path": path, "video_sha256": cs.sha256(path),
                               "error": None, "decision": "keep", "region_source": "dicom",
                               "region_bbox": json.dumps(list(BBOX))},
                              {"global_key": "k2", "patient_id": "p2", "vendor": "GEMS Ultrasound",
                               "video_path": path, "video_sha256": cs.sha256(path), "error": None,
                               "decision": "keep", "region_source": "dicom", "region_bbox": json.dumps(list(BBOX))}])
        flags_path = os.path.join(self.tmp, "flags.parquet")
        flags.to_parquet(flags_path)
        cs._write_info(flags_path, {})
        labels = os.path.join(self.tmp, "labels.csv")
        # k2 was answered measured: no base clip.
        rows = [("k1", cs.sha256(path), "no", "no", "no", "no", ""),
                ("k2", cs.sha256(path), "no", "yes", "no", "no", "")]
        pd.DataFrame(rows, columns=["global_key", "video_sha256", *cs.LABEL_FIELDS, "note"]).to_csv(labels, index=False)
        return flags_path, labels

    def test_each_clean_base_is_copied_once_per_variant_as_the_archive_encodes(self):
        from decord import VideoReader
        import imageio_ffmpeg

        flags, labels = self.base()
        out = os.path.join(self.tmp, "overlays")
        with contextlib.redirect_stdout(io.StringIO()):
            so.main(["make", "--flags", flags, "--labels", labels, "--out-dir", out])
        clips, truth = pd.read_csv(os.path.join(out, "clips.csv")), pd.read_csv(os.path.join(out, "truth.csv"))
        self.assertEqual(len(clips), len(so.VARIANTS))
        self.assertEqual(set(truth.base), {"k1"})
        self.assertEqual(list(truth.variant), list(so.VARIANTS))
        self.assertTrue(clips.global_key.str.fullmatch(cs.KEY_PATTERN).all() and clips.global_key.is_unique)
        self.assertEqual(set(clips.vendor), {"GEMS Ultrasound"})
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o700)
        played = {}
        for _, row in clips.merge(truth, on="global_key").iterrows():
            self.assertEqual(os.stat(row.video_path).st_mode & 0o777, 0o600)
            reader = imageio_ffmpeg.read_frames(row.video_path)
            meta = next(reader)
            reader.close()
            self.assertEqual((meta["codec"], meta["pix_fmt"].split("(")[0]), ("h264", "yuv444p"))
            vr = VideoReader(row.video_path)
            self.assertEqual(len(vr), T)
            played[row.variant] = vr.get_batch(list(range(T))).asnumpy().astype(int)
        # The control is the base clip re-encoded; an overlay changes the frames where its marks are.
        differ = {v: np.abs(f - played["control"]).max(axis=3) > 60 for v, f in played.items()}
        self.assertFalse(differ["control"].any())
        for variant in set(so.VARIANTS) - {"control"}:
            self.assertTrue(differ[variant].any(), variant)
        self.assertFalse(differ["caliper_pair_line_late"][: T // 2].any())
        with self.assertRaises(SystemExit):  # copies are never made twice into one folder.
            with contextlib.redirect_stdout(io.StringIO()):
                so.main(["make", "--flags", flags, "--labels", labels, "--out-dir", out])


class TestReport(unittest.TestCase):
    def test_the_signals_raised_per_variant_and_their_false_alarms_on_the_control(self):
        truth = pd.DataFrame({"global_key": ["a", "b", "c", "d"], "base": ["k1", "k1", "k2", "k2"],
                              "variant": ["control", "caliper_pair", "control", "caliper_pair"]})
        flags = pd.DataFrame({"global_key": ["a", "b", "c", "d"], "error": [None] * 4,
                              "vendor": ["GE Vingmed Ultrasound", "GE Vingmed Ultrasound", "Philips Medical Systems",
                                         "Philips Medical Systems"],
                              "geometry_evidence": ["none", "caliper_pair", "caliper", "none"],
                              "decision": ["keep", "exclude", "uncertain", "keep"]})
        report = so.overlay_report(flags, truth)
        pair = report["variants"]["caliper_pair"]
        self.assertEqual(pair["raised"]["pixel_caliper_pair"]["copies"], 1)
        self.assertEqual(pair["raised"]["pixel_caliper_pair"]["share"], 0.5)
        self.assertEqual(pair["by_vendor"]["other"]["raised"]["any_pixel_or_text"]["copies"], 0)
        control = report["variants"]["control"]
        self.assertEqual(control["raised"]["pixel_single_caliper"]["copies"], 1)  # a false alarm.
        self.assertEqual(control["decisions"], {"keep": 1, "uncertain": 1})
        self.assertNotIn("header_measure", control["raised"])
        with self.assertRaises(SystemExit):
            so.overlay_report(flags.iloc[:3], truth)


if __name__ == "__main__":
    unittest.main()
