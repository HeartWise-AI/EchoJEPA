# tests/data/test_validate_acquisition.py

"""Tests for checking acquisition flags against converted videos (data/validate_acquisition.py).

Synthetic only: small MJPG AVIs written with OpenCV to a temporary directory (gray-scale
speckle, a moving coloured region, gray-scale with coloured overlays) and made-up index rows.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

import scan_acquisition as sa
import validate_acquisition as va

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIZE, N_FRAMES = 96, 12


def gray_frames(seed=0):
    """Speckle-like gray-scale frames: equal channels, so no chroma before compression."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 200, size=(N_FRAMES, SIZE, SIZE), dtype=np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def color_frames():
    """Gray frames with a coloured blob in the centre that moves and changes colour (like color flow)."""
    frames = gray_frames(1)
    for i, f in enumerate(frames):
        y, x = 34 + (i % 4) * 3, 30 + (i % 5) * 3
        f[y : y + 20, x : x + 24] = (255, 40, 0) if i % 2 else (0, 40, 255)  # BGR: blue / red
    return frames


def overlay_frames():
    """Gray frames with coloured overlays: an ECG-like green trace at the bottom, a colour bar on
    the right edge, and a tiny static marker inside the central box.
    """
    frames = gray_frames(2)
    for f in frames:
        f[86:89, :] = (0, 255, 0)
        f[10:60, 90:95] = (0, 255, 255)
        f[47:49, 47:49] = (0, 255, 255)
    return frames


def write_avi(path, frames, fps=25):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (SIZE, SIZE), True)
    for f in frames:
        writer.write(np.ascontiguousarray(f))
    writer.release()
    return path


def index_row(path, regions, read_ok=True):
    """An index row as the scanner would write it, without touching any DICOM."""
    row = sa.empty_row(path)
    if not read_ok:
        return sa.error_row(row, "not_dicom")
    return {**row, "read_status": "ok", "has_region_sequence": regions is not None,
            "n_regions": len(regions or []), "region_data_types": sa.code_list(t for t, _ in regions or []),
            "region_spatial_formats": sa.code_list(f for _, f in regions or []),
            "region_pairs": sa.pair_list(regions or []), **sa.classify_regions(regions)}


class TestColourStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls._tmp.name
        cls.gray = write_avi(os.path.join(tmp, "gray.avi"), gray_frames())
        cls.color = write_avi(os.path.join(tmp, "color.avi"), color_frames())
        cls.overlay = write_avi(os.path.join(tmp, "overlay.avi"), overlay_frames())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def stats(self, path, k=8):
        frames, status = va.read_frames(path, k)
        self.assertEqual(status, "ok")
        return va.color_stats(frames)

    def test_read_frames_samples_evenly(self):
        frames, status = va.read_frames(self.color, 5)
        self.assertEqual((status, frames.shape, frames.dtype), ("ok", (5, SIZE, SIZE, 3), np.uint8))
        self.assertEqual(va.sample_indices(12, 5), [0, 3, 6, 8, 11])
        self.assertEqual(va.sample_indices(3, 8), [0, 1, 2])
        self.assertEqual(va.sample_indices(0, 8), [])

    def test_gray_video_has_no_colour(self):
        s = self.stats(self.gray)
        self.assertLess(s["full_fraction_max"], 0.002)
        self.assertLess(s["center_fraction_p90"], 0.002)
        self.assertIs(va.observed_color(s), False)

    def test_colour_region_is_observed(self):
        s = self.stats(self.color)
        self.assertGreater(s["center_fraction_p90"], 0.1)
        self.assertGreater(s["center_dynamic"], 0.01)
        self.assertIs(va.observed_color(s), True)

    def test_coloured_overlays_are_not_colour_flow(self):
        s = self.stats(self.overlay)
        # Overlays are coloured (whole frame, outside the box) but the central box is not.
        self.assertGreater(s["full_fraction_median"], 0.02)
        self.assertGreater(s["outside_fraction_median"], 0.03)
        # Only the tiny marker (blurred by chroma subsampling) is inside it.
        self.assertLess(s["center_fraction_p90"], 0.008)
        self.assertIs(va.observed_color(s), False)

    def test_static_colour_in_the_centre_needs_motion(self):
        frames = np.zeros((6, 50, 50, 3), np.uint8)
        frames[:, 20:30, 20:30] = (0, 0, 255)  # a large, static coloured patch
        s = va.color_stats(frames)
        self.assertGreater(s["center_fraction_p90"], 0.1)
        self.assertEqual(s["center_dynamic"], 0.0)
        self.assertIs(va.observed_color(s), False)
        self.assertIs(va.observed_color(s, va.Thresholds(min_dynamic=0.0)), True)
        # A single frame cannot show motion: the fraction alone decides.
        self.assertIs(va.observed_color(va.color_stats(frames[:1])), True)

    def test_thresholds_are_configurable(self):
        s = self.stats(self.overlay)
        loose = va.Thresholds(center=1.0, min_center_fraction=0.01, min_dynamic=0.0)
        self.assertIs(va.observed_color(va.color_stats(va.read_frames(self.overlay)[0], loose), loose), True)
        self.assertIs(va.observed_color(s), False)

    def test_dark_pixels_are_not_coloured(self):
        frames = np.zeros((2, 10, 10, 3), np.uint8)
        frames[:, :, :, 2] = 30  # saturated but dark
        self.assertEqual(va.color_stats(frames)["full_fraction_max"], 0.0)

    def test_undecodable_inputs(self):
        self.assertEqual(va.read_frames("/synthetic/none.avi")[1], "missing")
        self.assertEqual(va.read_frames(None)[1], "missing")
        junk = os.path.join(self._tmp.name, "junk.avi")
        with open(junk, "wb") as f:
            f.write(b"not a video" * 10)
        frames, status = va.read_frames(junk)
        self.assertIsNone(frames)
        self.assertIn(status, ("unopenable", "no_frames"))
        self.assertIsNone(va.observed_color(None))


class TestAgreement(unittest.TestCase):
    def test_truth_table(self):
        cases = {(True, True): "agree_color", (False, False): "agree_no_color",
                 (True, False): "header_color_missing_in_video", (False, True): "video_color_without_header_color",
                 (pd.NA, True): "header_unknown", (pd.NA, False): "header_unknown",
                 (True, None): "video_undecodable", (pd.NA, None): "video_undecodable"}
        for (header, seen), label in cases.items():
            with self.subTest(header=header, seen=seen):
                self.assertEqual(va.agreement(header, seen), label)
        self.assertEqual(set(va.DISAGREEMENTS) & {"header_unknown", "video_undecodable"}, set())


class TestValidationRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = self.tmp = self._tmp.name
        gray = lambda name, seed: write_avi(os.path.join(tmp, name), gray_frames(seed))  # noqa: E731
        cases = [  # (header regions, video)
            ([(1, 1), (14, 5), (2, 1)], write_avi(os.path.join(tmp, "a.avi"), color_frames())),  # agree, colour
            ([(2, 1), (1, 1)], gray("b.avi", 3)),                                                 # colour lost
            ([(1, 1), (10, 4)], write_avi(os.path.join(tmp, "c.avi"), color_frames())),          # unflagged colour
            ([(1, 1), (10, 4)], write_avi(os.path.join(tmp, "d.avi"), overlay_frames())),        # agree, overlays
            (None, gray("e.avi", 4)),                                                             # header unknown
            ([(1, 1)], os.path.join(tmp, "missing.avi")),                                         # undecodable
            ([(1, 1)], None),                                                                     # no video
        ]
        rows, meta = [], []
        for i, (regions, avi) in enumerate(cases):
            dcm = f"/synthetic/dcm/{i}.dcm"
            rows.append(index_row(dcm, regions))
            meta.append({"dcm_path": dcm, "avi_path": avi, "date": "20190101", "brand": "GE", "model": "M",
                         "predicted_class": "A4C", "study_type": sa.TTE})
        self.index = sa.rows_frame(rows)
        self.meta = pd.DataFrame(meta)
        self.out = os.path.join(tmp, "review")

    def run_validation(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return va.run_validation(self.index, self.meta, self.out, per_stratum=10, workers=2, **kwargs)

    def test_review_sheet_and_summary(self):
        review, summary = self.run_validation()
        by_video = review.set_index(review.avi_path.map(os.path.basename)).agreement.to_dict()
        self.assertEqual(by_video, {"a.avi": "agree_color", "b.avi": "header_color_missing_in_video",
                                    "c.avi": "video_color_without_header_color", "d.avi": "agree_no_color",
                                    "e.avi": "header_unknown", "missing.avi": "video_undecodable"})
        sheet = pd.read_csv(os.path.join(self.out, "review_sheet.csv"))
        self.assertEqual(len(sheet), 6)
        self.assertIn("reviewer_color_present", sheet)
        sheets = sheet.dropna(subset=["contact_sheet"])
        self.assertEqual(sorted(sheets.avi_path.map(os.path.basename)), ["b.avi", "c.avi"])
        for name in sheets.contact_sheet:
            png = cv2.imread(os.path.join(self.out, name))
            self.assertEqual(png.shape[0], 256)

        self.assertEqual(summary["sampling"]["without_avi_path"], 1)
        self.assertEqual(summary["sampling"]["videos_sampled"], 6)
        self.assertEqual(summary["color_flow_preserved"], {"header_color_flow_decoded": 2, "colour_observed": 1})
        self.assertEqual(summary["header_color_flow_vs_video"]["True"], {"False": 1, "True": 1, "undecodable": 0})
        self.assertEqual(summary["decode_status"], {"ok": 5, "missing": 1})
        self.assertEqual(summary["statistics_by_category"]["color_flow"]["center_fraction_p90"]["n"], 2)
        with open(os.path.join(self.out, "validation_summary.json")) as f:
            self.assertEqual(json.load(f)["agreement"], summary["agreement"])
        with open(os.path.join(self.out, "validation_summary.md")) as f:
            text = f.read()
        self.assertNotIn(self.tmp, text)
        self.assertNotIn("/synthetic", text)
        self.assertIn("Conversion preserves color flow", text)

    def test_sample_is_stratified_and_deterministic(self):
        index = pd.concat([self.index] * 1, ignore_index=True)
        joined, _ = va.join_index(index, self.meta, ["acq_category"])
        a = sa.stratified_sample(joined, ["acq_category"], seed=0, per_stratum=1)
        self.assertEqual(sorted(a.acq_category), ["bmode", "color_flow", "unknown"])
        pd.testing.assert_frame_equal(a, sa.stratified_sample(joined.iloc[::-1], ["acq_category"], 0, per_stratum=1))

    def test_contact_sheets_can_be_disabled(self):
        review, summary = self.run_validation(contact_sheets=0)
        self.assertEqual(summary["contact_sheets_written"], 0)
        self.assertFalse(os.path.exists(os.path.join(self.out, "contact_sheets")))

    def test_ambiguous_joins_are_refused(self):
        self.meta.loc[1, "avi_path"] = self.meta.loc[0, "avi_path"]
        with self.assertRaisesRegex(ValueError, "avi_path"):
            self.run_validation()
        self.meta.loc[1, "dcm_path"] = self.meta.loc[0, "dcm_path"]
        with self.assertRaisesRegex(ValueError, "dcm_path"):
            self.run_validation()

    def test_unknown_stratum_column_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot stratify"):
            va.join_index(self.index, self.meta, ["acq_category", "operator"])

    def test_cli_refuses_output_in_repository(self):
        with self.assertRaises(SystemExit):
            va.main(["--index", self.tmp, "--metadata", "x.parquet", "--out-dir", os.path.join(REPO, "review")])
        self.assertFalse(os.path.exists(os.path.join(REPO, "review")))

    def test_cli_end_to_end(self):
        index_path = os.path.join(self.tmp, "acquisition_index.parquet")
        sa.write_parquet_atomic(self.index, index_path)
        meta_path = os.path.join(self.tmp, "meta.parquet")
        self.meta.to_parquet(meta_path, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            va.main(["--index", index_path, "--metadata", meta_path, "--out-dir", self.out,
                     "--strata", "acq_category,year", "--per-stratum", "1", "--min-dynamic", "0"])
        with open(os.path.join(self.out, "validation_summary.json")) as f:
            summary = json.load(f)
        self.assertEqual(summary["sampling"]["videos_sampled"], 3)
        self.assertEqual(summary["thresholds"]["min_dynamic"], 0.0)


if __name__ == "__main__":
    unittest.main()
