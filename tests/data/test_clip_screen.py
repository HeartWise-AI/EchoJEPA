# tests/data/test_clip_screen.py

"""clip_screen.py on synthetic clips: still and measurement features, layout baselines, flags, label files and the
decision per clip, sheets keyed by global key and the weighted calibration report."""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import clip_screen as cs
from build_manifests import REPO_ROOT

H, W, T = 160, 200, 12
HAS_DECODER = importlib.util.find_spec("decord") is not None
HAS_PYDICOM = importlib.util.find_spec("pydicom") is not None
HAS_FFMPEG = importlib.util.find_spec("imageio_ffmpeg") is not None
SECTOR_BOX = [(30, 30, 169, 149)]  # the DICOM region box around REGION.
REGION = (slice(30, 150), slice(30, 170))


def speckle(seed=0, frozen=False, n=T):
    """A clip of `n` frames whose imaging region is moving speckle (one frame repeated when frozen) on a black
    screen."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((n, H, W, 3), np.uint8)
    first = rng.integers(0, 256, (120, 140), dtype=np.uint8)
    for t in range(n):
        tissue = first if frozen else rng.integers(0, 256, (120, 140), dtype=np.uint8)
        frames[t, REGION[0], REGION[1]] = tissue[..., None]
    return frames


def half_frozen(seed=0):
    """Speckle moving in the left half of the sector and frozen in the right half (a frozen reference image)."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((T, H, W, 3), np.uint8)
    right = rng.integers(0, 256, (120, 70), dtype=np.uint8)
    for t in range(T):
        frames[t, 30:150, 30:100] = rng.integers(0, 256, (120, 70), dtype=np.uint8)[..., None]
        frames[t, 30:150, 100:170] = right[..., None]
    return frames


def write_dicom(path, rows=H, cols=W, regions=((1, SECTOR_BOX[0]),), image_type=None):
    """A header-only DICOM file with a Sequence of Ultrasound Regions: (Region Data Type, (x0, y0, x1, y1)), and
    `image_type` (the ImageType values) when given."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.Rows, ds.Columns = rows, cols
    ds.SOPClassUID, ds.SOPInstanceUID = "1.2.840.10008.5.1.4.1.1.6.1", generate_uid()
    items = []
    for data_type, (x0, y0, x1, y1) in regions:
        item = Dataset()
        item.RegionDataType = data_type
        item.RegionLocationMinX0, item.RegionLocationMinY0 = x0, y0
        item.RegionLocationMaxX1, item.RegionLocationMaxY1 = x1, y1
        items.append(item)
    ds.SequenceOfUltrasoundRegions = items
    if image_type is not None:
        ds.ImageType = list(image_type)
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID, ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPClassUID, ds.SOPInstanceUID
    ds.save_as(path, enforce_file_format=True)
    return path


def caliper(frames, y, x, color=(255, 255, 0)):
    """A static 7 x 7 '+' drawn in every frame."""
    frames[:, y - 3:y + 4, x] = color
    frames[:, y, x - 3:x + 4] = color
    return frames


def x_caliper(frames, y, x, color=(255, 255, 0)):
    """A static 7 x 7 'x' drawn in every frame."""
    for d in range(-3, 4):
        frames[:, y + d, x + d] = frames[:, y + d, x - d] = color
    return frames


def line(frames, y0, x0, y1, x1, color=(255, 255, 0)):
    """A static 1-px line from (y0, x0) to (y1, x1) in every frame."""
    k = max(abs(y1 - y0), abs(x1 - x0)) + 1
    frames[:, np.linspace(y0, y1, k).round().astype(int), np.linspace(x0, x1, k).round().astype(int)] = color
    return frames


class TestClipFeatures(unittest.TestCase):
    def test_moving_clip_without_marks(self):
        features, cells, _ = cs.clip_features(speckle())
        self.assertGreater(features["motion_frac"], 0.4)
        self.assertEqual(features["inside_glyphs"], 0)
        self.assertEqual(int(cells.sum()), 0)

    def test_calipers_over_moving_tissue_are_counted_inside(self):
        frames = speckle()
        for y, x in ((60, 60), (60, 120), (110, 80), (120, 140)):
            caliper(frames, y, x)
        features, cells, details = cs.clip_features(frames)
        self.assertEqual(features["inside_glyphs"], 4)
        self.assertGreater(features["motion_frac"], 0.4)
        self.assertEqual(len(details["inside"]), 4)
        self.assertEqual(int(cells.sum()), 0)  # the cell map is ink beside the image only.

    def test_a_two_caliper_measurement_is_flagged(self):
        # One distance: two static calipers over moving tissue, without a dotted line between them.
        frames = caliper(caliper(speckle(), 70, 70), 110, 130)
        features, cells, _ = cs.clip_features(frames)
        self.assertEqual(features["inside_glyphs"], 2)
        flags = cs.apply_flags(pd.DataFrame([{**feature_row(cells), **features}]), {})
        self.assertTrue(flags.measurement.iloc[0])

    @staticmethod
    def sliding_tissue(n=48, step=1.0, seed=0):
        """`n` frames of smooth speckle up to 255, with a saturated band like a pericardium, sliding `step` px per
        frame: tissue that moves slowly when `step` is small."""
        import cv2

        rng = np.random.default_rng(seed)
        field = cv2.GaussianBlur(rng.integers(0, 256, (H, W + 100)).astype(np.float32), (0, 0), 1.5)
        field = (field - field.min()) / (field.max() - field.min()) * 255
        field[95:110] = 255
        frames = np.zeros((n, H, W, 3), np.uint8)
        for t in range(n):
            i, a = int(step * t), step * t - int(step * t)
            moved = (1 - a) * field[REGION[0], 30 + i:170 + i] + a * field[REGION[0], 31 + i:171 + i]
            frames[t, REGION[0], REGION[1]] = moved[..., None].astype(np.uint8)
        return frames

    def test_marks_shown_during_part_of_a_clip_are_counted(self):
        frames = self.sliding_tissue()
        for y, x in ((60, 60), (60, 120), (85, 80), (130, 140)):  # over moving tissue, off the band.
            caliper(frames[24:], y, x)  # drawn halfway through: not static over the whole clip.
        features, cells, _ = cs.clip_features(frames)
        self.assertEqual((features["inside_glyphs"], features["inside_glyphs_clip"]), (4, 0))
        self.assertEqual((features["windows"], features["frozen_windows"]), (3, 0))
        flags = cs.apply_flags(pd.DataFrame([{**feature_row(cells), **features}]), {})
        self.assertTrue(flags.measurement.iloc[0])

    def test_marks_shown_during_part_of_a_short_clip_are_counted(self):
        frames = speckle()  # 12 frames: windows of 6.
        for y, x in ((60, 60), (60, 120), (110, 80), (120, 140)):
            caliper(frames[6:], y, x)
        features, _, _ = cs.clip_features(frames)
        self.assertEqual((features["inside_glyphs"], features["inside_glyphs_clip"]), (4, 0))

    def test_a_part_of_the_image_frozen_partway_through_gives_no_marks(self):
        # Its speckle is static from then on, like ink: neither a side of the image (outside the window's moving
        # region) nor a patch inside it (a frozen patch) may read as marks.
        for make in (lambda: speckle(n=48), self.sliding_tissue):
            for rows, cols in ((slice(30, 150), slice(30, 86)), (slice(60, 120), slice(70, 130)),
                               (slice(70, 100), slice(80, 110))):
                frames = make()
                frames[24:, rows, cols] = frames[24, rows, cols]
                features, cells, _ = cs.clip_features(frames)
                self.assertEqual(features["inside_glyphs"], 0)
                flags = cs.apply_flags(pd.DataFrame([{**feature_row(cells), **features}]), {})
                self.assertFalse(flags.measurement.iloc[0])
        # Calipers over the moving rest of the image still count.
        frames = speckle(n=48)
        frames[24:, 30:150, 30:86] = frames[24, 30:150, 30:86]
        caliper(caliper(frames, 60, 120), 110, 140)
        self.assertEqual(cs.clip_features(frames)[0]["inside_glyphs"], 2)

    def test_slowly_moving_bright_tissue_is_no_mark_in_a_window(self):
        # A third of a pixel per read frame: shorter windows than half the clip would find dozens of "marks".
        features, _, _ = cs.clip_features(self.sliding_tissue(step=0.3))
        self.assertEqual(features["inside_glyphs"], 0)

    def test_marks_do_not_count_in_a_frozen_window(self):
        frames = self.sliding_tissue()
        frames[24:] = frames[24]  # frozen for the second half: its bright speckle is static.
        features, _, _ = cs.clip_features(frames)
        self.assertEqual(features["frozen_windows"], 1)
        self.assertEqual(features["inside_glyphs"], 0)

    def test_calipers_joined_by_the_line_of_their_distance_are_two_marks(self):
        # One ink component: counted as one mark, or dropped as a long sparse line when diagonal, they passed.
        for (y0, x0), (y1, x1) in (((90, 60), (90, 140)), ((50, 50), (130, 140)), ((45, 100), (135, 100)),
                                   ((70, 50), (100, 150))):
            for cross in (caliper, x_caliper):
                frames = line(cross(cross(speckle(), y0, x0), y1, x1), y0, x0, y1, x1)
                features, _, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
                self.assertEqual(features["inside_glyphs"], 2, (y0, x0, y1, x1, cross.__name__))

    def test_each_mark_counted_inside_is_kept_with_its_size_and_kind(self):
        joined = line(caliper(caliper(speckle(n=48), 90, 60), 90, 140), 90, 60, 90, 140)
        dotted = speckle(n=48)
        dotted[:, 60, 50:150:6] = dotted[:, 61, 50:150:6] = dotted[:, 60, 51:150:6] = dotted[:, 61, 51:150:6] = 255
        late = caliper(caliper(speckle(n=48), 60, 60), 110, 140)
        late[:24] = speckle(seed=3, n=24)  # the calipers drawn halfway through: found in windows only.
        for name, frames, kinds in (("joined", joined, ["joined_caliper", "joined_caliper"]),
                                    ("dotted", dotted, ["line"]),
                                    ("late", late, [])):
            features, _, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
            marks = json.loads(features["inside_marks"])
            self.assertEqual(sorted(m[1] for m in marks["clip"]), kinds, name)
            self.assertEqual(len(marks["windows"]), features["windows"] - features["frozen_windows"])
            # Recounted at the smallest size, the marks give pass 1's own count.
            row = pd.DataFrame([{"inside_marks": features["inside_marks"], "inside_glyphs": -1}])
            self.assertEqual(cs.at_mark_size(row, cs.MIN_INK_PX).inside_glyphs.iloc[0], features["inside_glyphs"])
        self.assertEqual(max(len(w) for w in marks["windows"]), 2)  # the late calipers.

    def test_traced_contours_are_marks_and_line_art_is_not(self):
        # Two 1-px contours traced over moving tissue, closed or open, throughout the clip: each bends too much to
        # be straight lines, so each is one mark ('trace'), and two reach the flag's threshold.
        import cv2

        for closed in (True, False):
            frames = speckle()
            for t in range(T):
                img = np.ascontiguousarray(frames[t])
                cv2.ellipse(img, (75, 85), (18, 30), 0, 0, 360 if closed else 200, (255, 255, 0), 1)
                cv2.ellipse(img, (128, 95), (16, 28), 10, 180, 540 if closed else 380, (255, 255, 0), 1)
                frames[t] = img
            features, _, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
            self.assertEqual([m[1] for m in json.loads(features["inside_marks"])["clip"]], ["trace", "trace"])
            self.assertEqual(features["inside_glyphs"], cs.THRESHOLDS["inside_glyphs"])
        # Line art stays none: a box, a box crossed by a cursor, lines crossing at a shallow angle, a long line.
        box = speckle()
        for y0, x0, y1, x1 in ((50, 60, 50, 120), (110, 60, 110, 120), (50, 60, 110, 60), (50, 120, 110, 120)):
            line(box, y0, x0, y1, x1)
        for name, frames in (("box", box.copy()), ("box and cursor", line(box, 35, 90, 140, 90)),
                             ("shallow crossing", line(line(speckle(), 60, 35, 120, 165), 120, 35, 60, 165)),
                             ("long line", line(speckle(), 35, 95, 145, 105))):
            features, _, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
            self.assertEqual(features["inside_glyphs"], 0, name)

    def test_two_calipers_of_one_distance_are_a_pair(self):
        # Two calipers of one color, apart, static together: the ends of one distance, whatever inside_glyphs says;
        # calipers of two colors are two single calipers.
        for colors, expected in (([(255, 255, 0)] * 2, "caliper_pair"), ([(255, 255, 0), (0, 255, 255)], "caliper")):
            frames = caliper(caliper(speckle(), 60, 60, colors[0]), 120, 140, colors[1])
            features, _, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
            self.assertEqual(cs.geometry_evidence(features["inside_marks"], cs.MIN_INK_PX), expected, colors)
        late = caliper(caliper(speckle(n=48), 60, 60), 120, 140)
        late[:24] = speckle(seed=3, n=24)  # drawn halfway through: a pair within one window.
        features, _, _ = cs.clip_features(late, SECTOR_BOX, "ok")
        self.assertEqual(cs.geometry_evidence(features["inside_marks"], cs.MIN_INK_PX), "caliper_pair")
        self.assertEqual(cs.geometry_evidence(json.dumps({"clip": [], "windows": []}), cs.MIN_INK_PX), "none")
        # Calipers joined by one line are a pair whatever their colors; two calipers of two colors, each on a line of
        # its own, are not: each is joined to its own line, not to the other.
        joined = line(caliper(caliper(speckle(), 60, 60), 120, 140), 60, 60, 120, 140)
        features, _, _ = cs.clip_features(joined, SECTOR_BOX, "ok")
        self.assertEqual(cs.geometry_evidence(features["inside_marks"], cs.MIN_INK_PX), "caliper_pair")
        apart = caliper(caliper(speckle(), 50, 50), 130, 150, (0, 255, 255))
        line(apart, 50, 50, 90, 90)
        line(apart, 130, 150, 90, 110, (0, 255, 255))
        features, _, _ = cs.clip_features(apart, SECTOR_BOX, "ok")
        marks = json.loads(features["inside_marks"])["clip"]
        self.assertEqual(sorted(m[1] for m in marks), ["joined_caliper", "joined_caliper"])
        self.assertEqual(cs.geometry_evidence(features["inside_marks"], cs.MIN_INK_PX), "caliper")

    def test_the_field_of_view_is_found_from_the_pixels_without_regions_even_when_frozen(self):
        for frozen in (False, True):
            features, _, _ = cs.clip_features(speckle(frozen=frozen))  # no DICOM regions.
            self.assertEqual(features["region_source"], "image")
            self.assertAlmostEqual(features["field_frac"], 0.525, delta=0.05)  # REGION: 120 x 140 of 160 x 200.
            self.assertEqual(features["region_motion_frac"] < 0.01, frozen)
        features, _, _ = cs.clip_features(speckle(), SECTOR_BOX, "ok")
        self.assertEqual(features["region_source"], "dicom")
        self.assertGreater(features["field_iou_dicom"], 0.9)  # the two agree.
        blank = np.zeros((T, H, W, 3), np.uint8)
        blank[:, 20:30, 20:180] = 255  # a bar, no speckle: no field of view.
        self.assertEqual(cs.clip_features(blank)[0]["region_source"], "whole_frame")

    def test_moving_red_or_blue_pixels_read_as_color_flow(self):
        frames = speckle()
        jet = np.random.default_rng(3).integers(0, 2, (T, 30, 40)).astype(bool)  # flow flickering red and blue.
        patch = frames[:, 60:90, 80:120]
        patch[jet], patch[~jet] = (220, 30, 30), (30, 60, 220)
        self.assertGreaterEqual(cs.clip_features(frames)[0]["moving_color_frac"], cs.PIXEL_COLOR_FRAC)
        # Gray speckle, a static color bar and a moving green ECG trace are no flow.
        plain = speckle()
        plain[:, 20:120, 185:192] = (200, 40, 40)
        for t in range(T):
            plain[t, 150:152, 10 + t:14 + t] = (0, 220, 0)
        self.assertLess(cs.clip_features(plain)[0]["moving_color_frac"], cs.PIXEL_COLOR_FRAC)

    def test_plots_beside_the_field_are_an_analysis_screens_evidence(self):
        frames = speckle()
        for y in (10, 14, 18, 22):  # colored curves above the image, static throughout.
            frames[:, y, 20:180] = (0, 200, 255)
        features, _, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
        self.assertGreaterEqual(features["outside_ink_frac"], cs.ANALYSIS_OUTSIDE_INK)
        self.assertEqual(cs.clip_features(speckle(), SECTOR_BOX, "ok")[0]["outside_ink_frac"], 0.0)

    def test_measurement_text_is_read_only_with_an_ocr_engine(self):
        lines = ["EF 55 %", "LVIDd 4.5 cm", "EDV 120 ml", "Gn 50 dB", "HR 72 bpm", "Depth 16 cm", "16 cm", "2.1 m/s",
                 "E/e' 12", "E/e 9"]
        status, found = cs.measurement_text(speckle(), ("stand-in", lambda frame: lines))
        # A name with its number, or a number with a measurement unit; settings, the heart rate and depths are not.
        self.assertEqual((status, json.loads(found)),
                         ("checked", ["2.1 m/s", "E/e 9", "E/e' 12", "EDV 120", "EF 55", "LVIDd 4.5"]))
        self.assertEqual(cs.measurement_text(speckle(), ("stand-in", lambda frame: ["Gn 50", "FR 54 Hz"]))[1], "[]")
        with mock.patch.object(cs, "ocr_backend", return_value=None):
            self.assertEqual(cs.measurement_text(speckle()), ("unavailable", None))  # missing, not absent.

    def test_a_periodic_loop_is_not_sampled_at_one_phase(self):
        # A 941-frame clip repeating a moving 20-frame cycle: evenly spaced samples (one every 20 frames) would show
        # 48 copies of one phase, a still clip; the jittered ones fall on most phases, and the loop moves.
        indices = cs.frame_indices(941)
        self.assertEqual((len(indices), indices[0], indices[-1]), (48, 0, 940))
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertGreaterEqual(len(set(indices % 20)), 10)
        cycle = speckle(n=20)
        even = np.unique(np.linspace(0, 940, 48).round().astype(int))
        for sampled, still in ((even, True), (indices, False)):
            features, _, _ = cs.clip_features(cycle[sampled % 20], SECTOR_BOX, "ok")
            self.assertEqual(features["region_motion_frac"] < cs.THRESHOLDS["still_scan_motion_frac"], still)
        self.assertEqual(cs.frame_indices(12).tolist(), list(range(12)))  # a short clip: every frame.

    def test_lines_alone_are_not_calipers(self):
        # Lines crossing at any angle, a box crossed by a cursor and a ruler with crossing ticks have no caliper.
        ruler = line(speckle(), 40, 100, 140, 100)
        for y in range(40, 141, 10):
            line(ruler, y, 96, y, 104)
        box = speckle()
        for y0, x0, y1, x1 in ((50, 60, 50, 120), (110, 60, 110, 120), (50, 60, 110, 60), (50, 120, 110, 120)):
            line(box, y0, x0, y1, x1)
        for name, frames, marks in (("lines crossing", line(line(speckle(), 40, 40, 140, 160), 40, 160, 140, 40), 0),
                                    ("box and cursor", line(box, 35, 90, 140, 90), 0), ("ruler", ruler, 1)):
            features, _, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
            self.assertEqual(features["inside_glyphs"], marks, name)  # the ruler: one mark, as before.

    def test_plus_shaped_calipers_close_together_stay_separate(self):
        frames = speckle()
        for y, x in ((70, 70), (70, 79), (79, 70)):  # three 3 x 3 pluses in a triangle.
            frames[:, y - 1:y + 2, x] = (255, 255, 0)
            frames[:, y, x - 1:x + 2] = (255, 255, 0)
        self.assertEqual(cs.clip_features(frames)[0]["inside_glyphs"], 3)

    def test_only_a_straight_chain_of_dots_is_one_mark(self):
        frames = speckle()
        for y, x in ((70, 70), (70, 76), (76, 70)):  # three round dots in a triangle: not a line.
            frames[:, y:y + 2, x:x + 2] = 220
        for a in np.linspace(0, np.pi, 12):  # a dotted half-circle trace: not a line either.
            y, x = int(110 + 25 * np.sin(a)), int(110 + 25 * np.cos(a))
            frames[:, y:y + 2, x:x + 2] = 220
        features, _, _ = cs.clip_features(frames)
        self.assertEqual(features["inside_glyphs"], features["inside_marks_raw"])
        self.assertGreaterEqual(features["inside_glyphs"], 14)

    def test_frozen_frame_is_still_even_with_a_sweeping_ecg_line(self):
        frames = speckle(frozen=True)
        for t in range(T):
            frames[t, 155, :] = 0
            frames[t, 155, 10 + 15 * t:12 + 15 * t] = (0, 255, 0)
        features, _, _ = cs.clip_features(frames)
        self.assertLess(features["motion_frac"], cs.THRESHOLDS["still_motion_frac"])

    def test_red_turning_blue_moves_and_is_no_mark(self):
        # Pure red and pure blue have the same gray level and chroma: only the channels see the change.
        frames = speckle()
        for t in range(T):
            for y, x in ((60, 60), (60, 120), (110, 80)):
                frames[t, y:y + 6, x:x + 6] = (255, 0, 0) if t % 2 else (0, 0, 255)
        features, cells, _ = cs.clip_features(frames)
        self.assertEqual(features["inside_glyphs"], 0)
        self.assertEqual(int(cells.sum()), 0)

    def test_a_dotted_line_counts_once_like_a_solid_one(self):
        frames = speckle()
        frames[:, 70, 50:151] = 220  # a solid line across the sector,
        for x in range(50, 150, 6):  # and a dotted one (2 x 2 dots, 4 px apart).
            frames[:, 110:112, x:x + 2] = 220
        features, _, _ = cs.clip_features(frames)
        self.assertEqual(features["inside_glyphs"], 2)
        self.assertGreaterEqual(features["inside_marks_raw"], 17)

    def test_the_scan_area_needs_no_motion(self):
        frozen = speckle(frozen=True)
        features, _, details = cs.clip_features(frozen, SECTOR_BOX, "ok")
        self.assertAlmostEqual(features["scan_frac"], 120 * 140 / (H * W), delta=0.02)
        self.assertGreater(features["region_fill"], 0.95)
        self.assertEqual(features["scan_motion_frac"], 0.0)
        self.assertEqual(features["duplicate_frac"], 1.0)
        self.assertFalse(details["region"].any())
        features, _, _ = cs.clip_features(frozen)  # without DICOM boxes: no scan area.
        self.assertTrue(np.isnan(features["scan_frac"]) and np.isnan(features["scan_motion_frac"]))
        self.assertEqual(features["duplicate_frac"], 1.0)

    def test_moving_clips_have_no_duplicate_frames(self):
        features, _, _ = cs.clip_features(speckle(), SECTOR_BOX, "ok")
        self.assertEqual(features["duplicate_frac"], 0.0)
        self.assertGreater(features["scan_motion_frac"], 0.9)

    def test_a_frozen_part_of_the_scan_area_gives_no_marks(self):
        # Its bright speckle is static: it cannot be told from ink, so marks count in the moving region only.
        frames = caliper(half_frozen(), 90, 140)
        features, _, details = cs.clip_features(frames, SECTOR_BOX, "ok")
        self.assertEqual(features["inside_glyphs"], 0)
        self.assertAlmostEqual(features["scan_motion_frac"], 0.5, delta=0.05)
        self.assertTrue(details["scan"][90, 140] and not details["region"][90, 140])

    def test_moving_shares_at_other_pixel_thresholds(self):
        rng = np.random.default_rng(0)
        base = rng.integers(60, 190, (120, 140)).astype(float)
        frames = np.zeros((T, H, W, 3), np.uint8)
        for t in range(T):  # tissue that only flickers by about 5 levels.
            frames[t, REGION[0], REGION[1]] = np.clip(base + rng.normal(0, 5, base.shape), 0, 255)[..., None]
        features, _, _ = cs.clip_features(frames)
        self.assertGreater(features["motion_frac_std3"], 0.4)
        self.assertLess(features["motion_frac"], features["motion_frac_std3"])
        self.assertLess(features["motion_frac_std10"], 0.01)

    def test_a_color_box_outline_is_a_line_not_a_mark(self):
        frames = speckle()
        frames[:, 50, 60:121] = 220
        frames[:, 110, 60:121] = 220
        frames[:, 50:111, 60] = 220
        frames[:, 50:111, 120] = 220
        features, _, _ = cs.clip_features(frames)
        self.assertEqual(features["inside_glyphs"], 0)


# What pass 1 records for an ordinary recording: a multi-frame ultrasound image whose header has image regions and
# the ECG, motion read over the DICOM scan area, nothing beside the field of view; no text pass.
US_MULTIFRAME = "1.2.840.10008.5.1.4.1.1.3.1"
RECORDING = {"dicom_status": "ok", "dicom_image_content": "0001", "dicom_measure_flag": False,
             "dicom_source_image": True, "dicom_ecg": True, "dicom_sop_class": US_MULTIFRAME, "region_source": "dicom",
             "duplicate_frac": 0.0, "outside_ink_frac": 0.0}
GE = "GE Vingmed Ultrasound"
# A Philips header as the survey found them: no fourth ImageType value and no SourceImageSequence, with the ECG.
PHILIPS = {"vendor": "Philips Medical Systems", "dicom_image_content": "", "dicom_source_image": False}
TEXT_ON = {**cs.THRESHOLDS, "text_decides": "all"}
# `flag` of synthetic clips, which have no header and no text pass: their text would decide, so the run says it is
# without it.
NO_TEXT = ("--thresholds", '{"text_decides": "none"}')
# An analysis screen as the pilot's labelled ones: no regions, a header naming no image data, without the acquired
# image or the ECG; plots beside its field of view.
ANALYSIS_SCREEN = {"dicom_status": "no_regions", "dicom_image_content": "0000", "dicom_source_image": False,
                   "dicom_ecg": False, "region_source": "image", "outside_ink_frac": 0.012}


def feature_row(cells, inside=0, motion=0.3, bbox=(30, 30, 169, 149), error=None, layout="ge"):
    return {"video_path": f"/synthetic/{np.random.default_rng().integers(1 << 40)}", "error": error,
            "video_sha256": "synthetic", **RECORDING,
            "vendor": layout, "model": "m", "width": W, "height": H, "acq_category": "bmode",
            "motion_frac": motion, "region_motion_frac": motion, "inside_glyphs": inside,
            "region_bbox": json.dumps(bbox),
            "cells_shape": json.dumps(list(cells.shape)), "cells": np.packbits(cells.ravel()).tobytes()}


# The earlier policies, kept for comparison: the pixel measurement rules exclude (what the detector tests check), or a
# header without usable image regions does.
PIXEL_POLICY = {**cs.THRESHOLDS, "policy": "pixel"}
HEADER_POLICY = {**cs.THRESHOLDS, "policy": "header"}


def flag_rows(rows, reference, thresholds=cs.THRESHOLDS):
    baselines, _ = cs.layout_baselines(pd.DataFrame(reference))
    return cs.apply_flags(pd.DataFrame(rows), baselines, thresholds)


class TestFlags(unittest.TestCase):
    def setUp(self):
        self.shape = (H // cs.CELL, W // cs.CELL)
        self.header = np.zeros(self.shape, bool)
        self.header[0, 0:8] = True  # the same header text in every clip of the layout.
        self.box = self.header.copy()
        self.box[4:7, 4:10] = True  # a results box next to the sector: 18 cells.

    def rows(self, n=30):
        return [feature_row(self.header.copy()) for _ in range(n)]

    def test_ink_where_the_layout_has_none_is_extra(self):
        rows = self.rows(5) + [feature_row(self.box)]
        flags = flag_rows(rows, self.rows(), PIXEL_POLICY)
        self.assertEqual(flags.extra_cells.iloc[-1], 18)
        self.assertTrue(flags.measurement.iloc[-1])
        self.assertEqual(flags.exclude_reason.iloc[-1], "measurement")
        self.assertEqual(int(flags.extra_cells.iloc[:-1].max()), 0)
        self.assertFalse(flags.exclude.iloc[:-1].any())

    def test_a_results_box_far_from_the_image_is_extra(self):
        # A small image (cells 3 to 10) and a box of 12 cells 9 cells to its right: a 32-px margin missed it.
        far = self.header.copy()
        far[10:13, 19:23] = True
        flags = flag_rows([feature_row(far, bbox=(30, 30, 80, 80))], self.rows())
        self.assertEqual(flags.extra_cells.iloc[0], 12)
        self.assertTrue(flags.measurement.iloc[0])

    def test_results_text_shown_partway_through_is_extra_ink(self):
        def row(frames):
            features, cells, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
            return {**feature_row(cells), **features, "cells": np.packbits(cells.ravel()).tobytes()}

        def text(frames):  # six glyph-like strokes above the sector.
            for i in range(6):
                frames[:, 10:19, 40 + 8 * i:42 + 8 * i] = 255
                frames[:, 10, 40 + 8 * i:46 + 8 * i] = 255
            return frames

        reference = pd.DataFrame([dict(row(speckle(seed=s, n=48)), video_path=f"/r{s}") for s in range(30)])
        baselines, _ = cs.layout_baselines(reference)
        whole, half = speckle(seed=99, n=48), speckle(seed=99, n=48)
        text(whole)
        text(half[24:])  # shown during the second half only: not static over the whole clip.
        flags = cs.apply_flags(pd.DataFrame([row(whole), row(half)]), baselines)
        self.assertGreaterEqual(int(flags.extra_cells.iloc[1]), cs.THRESHOLDS["extra_cells"])
        self.assertEqual(flags.extra_cells.iloc[1], flags.extra_cells.iloc[0])
        self.assertEqual(flags.measurement.tolist(), [True, True])

    def test_a_box_common_among_the_screened_clips_is_still_extra(self):
        rows = self.rows(27) + [feature_row(self.box.copy()) for _ in range(3)]
        flags = flag_rows(rows, self.rows())
        self.assertEqual(flags.extra_cells.iloc[-3:].tolist(), [18, 18, 18])
        self.assertTrue(flags.measurement.iloc[-3:].all())

    def test_reference_clips_with_a_results_box_beside_the_image_are_left_out(self):
        # 3 of 30 moving reference clips without marks over the image carry a results box (10%, COMMON_SHARE): it
        # is not the layout's core ink (CORE_SHARE), so they are left out and the box stays extra in a target.
        reference = pd.DataFrame(self.rows(27) + [feature_row(self.box.copy()) for _ in range(3)])
        self.assertEqual(cs.reference_left_out(reference).tolist(), [""] * 27 + ["extra_ink"] * 3)
        flags = flag_rows([feature_row(self.box.copy())], reference)
        self.assertEqual((flags.extra_cells.iloc[0], bool(flags.measurement.iloc[0])), (18, True))
        # Ink most of a layout's clips have is part of the layout: none of them is left out.
        reference = pd.DataFrame(self.rows(10) + [feature_row(self.box.copy()) for _ in range(20)])
        self.assertEqual(set(cs.reference_left_out(reference)), {""})
        self.assertEqual(flag_rows([feature_row(self.box.copy())], reference).extra_cells.iloc[0], 0)

    def test_still_or_marked_reference_clips_do_not_shape_the_baseline(self):
        # Clearly moving is read like the still flag: the scan area's share when there is one (here 10% < 20%,
        # though 30% of the frame moves), else the frame's.
        reference = self.rows() + [feature_row(self.box.copy(), inside=5) for _ in range(5)] + [
            feature_row(self.box.copy(), motion=0.001) for _ in range(5)] + [
            dict(feature_row(self.box.copy()), scan_motion_frac=0.1) for _ in range(5)]
        reference = pd.DataFrame(reference)
        baselines, sources = cs.layout_baselines(reference)
        self.assertEqual([len(s) for s in sources.values()], [30])
        flags = cs.apply_flags(pd.DataFrame([feature_row(self.box)]), baselines)
        self.assertEqual(flags.extra_cells.iloc[0], 18)
        self.assertEqual(cs.reference_left_out(reference).value_counts().to_dict(),
                         {"": 30, "marks_inside": 5, "low_motion": 10})

    def test_the_still_flag_reads_the_scan_area_when_there_is_one(self):
        rows = [dict(feature_row(self.header.copy(), motion=0.005), scan_motion_frac=0.3),  # ECG and black margins
                dict(feature_row(self.header.copy(), motion=0.3), scan_motion_frac=0.02),
                dict(feature_row(self.header.copy(), motion=0.005), scan_motion_frac=np.nan),
                dict(feature_row(self.header.copy(), motion=0.02), scan_motion_frac=np.nan)]
        flags = flag_rows(rows, self.rows())
        self.assertEqual(flags.still.tolist(), [False, True, True, False])
        self.assertEqual(flags.still_basis.tolist(), ["scan", "scan", "frame", "frame"])
        lowered = {**cs.THRESHOLDS, "still_scan_motion_frac": 0.01}  # each basis has its own threshold.
        self.assertEqual(cs.flag_columns(flags, lowered).still.tolist(), [False, False, True, False])

    def test_a_frozen_half_of_a_dual_display_is_not_extra_ink(self):
        def row(frames):
            features, cells, _ = cs.clip_features(frames, SECTOR_BOX, "ok")
            return {**feature_row(cells), **features, "cells": np.packbits(cells.ravel()).tobytes()}

        live = row(speckle())
        baselines, _ = cs.layout_baselines(pd.DataFrame([dict(live, video_path=f"/r{i}") for i in range(30)]))
        flags = cs.apply_flags(pd.DataFrame([row(half_frozen())]), baselines)
        self.assertEqual((flags.extra_cells.iloc[0], bool(flags.measurement.iloc[0])), (0, False))

    def test_a_longer_header_than_usual_is_not_extra(self):
        longer = self.header.copy()
        longer[0, 8] = True
        self.assertEqual(flag_rows([feature_row(longer)], self.rows()).extra_cells.iloc[0], 0)

    def test_a_clip_that_could_not_be_checked_for_a_box_is_incomplete(self):
        # 3 reference clips, fewer than MIN_LAYOUT_CLIPS: no baseline, so no results-box check. With the pixel
        # measurement rules excluding, a clip no other flag excludes is excluded as incomplete, never passed.
        rows = [feature_row(self.box), feature_row(self.header, inside=5)]
        flags = flag_rows(rows, self.rows(3), PIXEL_POLICY)
        self.assertTrue(flags.extra_cells.isna().all())
        self.assertEqual(flags.exclude_reason.tolist(), ["incomplete", "measurement"])
        # Under the default policy a missing baseline alone excludes nothing; the measurement flag is still set.
        flags = flag_rows(rows, self.rows(3))
        self.assertEqual((flags.exclude_reason.tolist(), flags.measurement.tolist()), (["", ""], [False, True]))
        # The reviewer's case: 25 reference clips, 3 with the box left out, 22 left: the same.
        reference = self.rows(22) + [feature_row(self.box.copy()) for _ in range(3)]
        flags = flag_rows([feature_row(self.box.copy())], reference, PIXEL_POLICY)
        self.assertEqual((flags.extra_cells.iloc[0], flags.exclude_reason.iloc[0]), (pd.NA, "incomplete"))

    def test_the_header_policy_excludes_each_header_failure_before_still(self):
        # The earlier header policy, kept for comparison: each header without a usable image region is its own reason.
        statuses = ("ok", "no_regions", "size_mismatch", "unreadable_header", "", None)
        rows = [dict(feature_row(self.box.copy()), dicom_status=s) for s in statuses]
        rows.append(dict(feature_row(self.header.copy(), motion=0.001), dicom_status="no_regions"))
        rows.append(feature_row(self.header.copy(), motion=0.001))
        flags = flag_rows(rows, self.rows(), HEADER_POLICY)
        self.assertEqual(flags.exclude_reason.tolist(),
                         ["", "no_image_region", "header_size_mismatch", "header_unreadable", "no_header", "no_header",
                          "no_image_region", "still"])
        self.assertEqual(set(flags.decision), {"keep", "exclude"})  # no uncertain clip under an earlier policy.
        self.assertTrue(flags.measurement.iloc[0])
        self.assertEqual(flag_rows(rows[:1], self.rows(), PIXEL_POLICY).exclude_reason.tolist(), ["measurement"])

    def hybrid(self, thresholds=cs.THRESHOLDS, **changes):
        """The hybrid decision and its reasons for one recording, of a GE scanner unless the changes say otherwise,
        with these feature changes."""
        flags = flag_rows([{**feature_row(self.header.copy()), "vendor": GE, **changes}], self.rows(), thresholds)
        return flags.decision.iloc[0], flags.decision_reasons.iloc[0], flags.iloc[0]

    def test_a_clean_recording_without_image_regions_is_kept(self):
        # A header without regions but naming tissue data, with the acquired image and the ECG, motion read in the
        # field of view found from the pixels: nothing says it is measured.
        decision, reasons, row = self.hybrid(dicom_status="no_regions", region_source="image")
        self.assertEqual((decision, reasons), ("keep", ""))
        self.assertEqual((row.dicom_region_status, row.analysis_evidence), ("no_image_region", "none"))

    def test_a_ge_header_tells_an_analysis_screen_alone(self):
        # The header pattern excludes, with or without plots beside the field; plots beside an ordinary recording's
        # field are recorded only (on the pilot they made color recordings uncertain and added no exclusion).
        self.assertEqual(self.hybrid(**ANALYSIS_SCREEN)[:2], ("exclude", "analysis_screen"))
        self.assertEqual(self.hybrid(**{**ANALYSIS_SCREEN, "outside_ink_frac": 0.0})[:2],
                         ("exclude", "analysis_screen"))
        decision, reasons, row = self.hybrid(outside_ink_frac=0.012)
        self.assertEqual((decision, reasons, row.analysis_evidence), ("keep", "", "visual_only"))

    def test_another_vendors_analysis_screen_is_excluded_on_both_sources_and_uncertain_on_one(self):
        other = {"vendor": "SIEMENS"}
        self.assertEqual(self.hybrid(**other, **ANALYSIS_SCREEN)[:2], ("exclude", "analysis_screen"))
        self.assertEqual(self.hybrid(**other, **{**ANALYSIS_SCREEN, "outside_ink_frac": 0.0})[:2],
                         ("uncertain", "analysis_header_only"))
        self.assertEqual(self.hybrid(**PHILIPS, outside_ink_frac=0.012)[:2], ("uncertain", "analysis_visual_only"))

    def test_a_ge_header_naming_unusual_image_data_makes_a_clip_uncertain(self):
        # '0041' and '0051' (3D probe displays) are neither tissue, tissue and color, nor an analysis screen.
        for code in ("0041", "0051", ""):
            decision, reasons, row = self.hybrid(dicom_image_content=code)
            self.assertEqual((decision, reasons, row.header_content), ("uncertain", "unusual_image_content", "unusual"))
        self.assertEqual(self.hybrid()[2].header_content, "usual")
        # Another vendor's header names no image data in that value, and no header names none.
        self.assertEqual(self.hybrid(**PHILIPS)[2].header_content, "unavailable")
        self.assertEqual(self.hybrid(**{**PHILIPS, "dicom_image_content": "0041"})[:2], ("keep", ""))
        self.assertEqual(self.hybrid(dicom_image_content=None)[2].header_content, "unavailable")

    def test_an_unverified_header_convention_is_missing_evidence_not_clean(self):
        # A Philips header names no image data and no acquired image: that says nothing, and no MEASURE neither.
        decision, _, row = self.hybrid(**PHILIPS)
        self.assertEqual((decision, row.header_convention, row.analysis_evidence, row.measurement_header,
                          row.evidence_missing),
                         ("keep", "unverified", "none", "unavailable", "header_convention;text"))
        self.assertEqual(self.hybrid(**PHILIPS, dicom_measure_flag=True)[:2], ("exclude", "measurement_header"))
        for vendor in cs.GE_HEADER_VENDORS:  # GE's three vendor names write one convention.
            _, _, row = self.hybrid(vendor=vendor)
            self.assertEqual((row.header_convention, row.measurement_header, row.evidence_missing),
                             ("ge", "none", "text"))

    def test_an_object_that_is_no_ultrasound_image_is_excluded(self):
        # A structured report and a secondary capture, which became videos in the archive too.
        for sop_class in ("1.2.840.10008.5.1.4.1.1.88.33", "1.2.840.10008.5.1.4.1.1.7"):
            decision, reasons, row = self.hybrid(**PHILIPS, dicom_sop_class=sop_class)
            self.assertEqual((decision, reasons, row.dicom_object),
                             ("exclude", "not_ultrasound_image", "not_ultrasound_image"))
        # No class recorded is missing evidence, never a reason.
        decision, reasons, row = self.hybrid(dicom_sop_class="")
        self.assertEqual((decision, reasons, row.dicom_object), ("keep", "", "unavailable"))

    def test_the_pixel_mode_is_checked_against_the_header_where_it_gives_one(self):
        rows = [{**feature_row(self.header.copy()), "vendor": vendor, "acq_category": header, "moving_color_frac": frac}
                for vendor, header, frac in ((GE, "bmode", 0.0), (GE, "color_flow", 0.02), (GE, "color_flow", 0.001),
                                             ("Philips Medical Systems", "bmode", 0.03))]
        flags = flag_rows(rows, self.rows())
        self.assertEqual(flags.pixel_mode.tolist(), ["bmode", "color_flow", "bmode", "color_flow"])
        self.assertEqual(flags.decision.tolist(), ["keep"] * 4)  # the mode decides nothing in the screen.
        # Only the GE clips count: a Philips header calls color loops B-mode.
        self.assertEqual(cs.mode_agreement(flags), {"bmode": {"bmode": 1}, "color_flow": {"color_flow": 1, "bmode": 1}})

    def test_a_measured_recording_with_image_regions_is_not_taken_for_clean(self):
        # The header names a measurement (GE's ImageType MEASURE), or two calipers of one distance are drawn.
        self.assertEqual(self.hybrid(dicom_measure_flag=True)[:2], ("exclude", "measurement_header"))
        pair = {"clip": [[13, "caliper", 60.0, 60.0, [255, 255, 0]], [13, "caliper", 140.0, 120.0, [255, 255, 0]]],
                "windows": []}
        decision, reasons, row = self.hybrid(inside_marks=json.dumps(pair), inside_glyphs=2)
        self.assertEqual((decision, reasons, row.geometry_evidence),
                         ("exclude", "measurement_calipers", "caliper_pair"))
        # Two calipers of different colors are not one distance; one alone, or a traced contour, is uncertain.
        mixed = {"clip": [pair["clip"][0], [13, "caliper", 140.0, 120.0, [0, 255, 255]]], "windows": []}
        self.assertEqual(self.hybrid(inside_marks=json.dumps(mixed))[:2], ("uncertain", "measurement_single_caliper"))
        traced = {"clip": [[140, "trace", 75.0, 85.0, [255, 255, 0]]], "windows": []}
        self.assertEqual(self.hybrid(inside_marks=json.dumps(traced))[:2], ("uncertain", "measurement_trace"))

    def test_bright_tissue_marks_alone_exclude_nothing(self):
        # Many small static pieces (bright, slowly moving tissue): the pixel score flags them, the policy keeps it.
        specks = {"clip": [[4, "piece", 50.0 + 9 * i, 80.0, [230, 230, 230]] for i in range(10)], "windows": []}
        decision, reasons, row = self.hybrid(inside_marks=json.dumps(specks), inside_glyphs=10)
        self.assertEqual((decision, bool(row.measurement), row.geometry_evidence), ("keep", True, "none"))

    def test_missing_evidence_is_not_negative_evidence(self):
        decision, _, row = self.hybrid()  # pass 1 output, without the text pass.
        self.assertEqual((decision, row.text_evidence, row.evidence_missing), ("keep", "unavailable", "text"))
        _, _, row = self.hybrid(text_status="checked", text_measurements="[]")
        self.assertEqual((row.text_evidence, row.evidence_missing), ("none", ""))
        _, _, row = self.hybrid(text_status="decode_failed", text_measurements=None)  # read by the text pass: no.
        self.assertEqual((row.text_evidence, row.evidence_missing), ("unavailable", "text"))

    def test_text_decides_where_no_header_can_name_a_measurement(self):
        read = {"text_status": "checked", "text_measurements": '["EF 55"]'}
        unit = {"text_status": "checked", "text_measurements": '["120 ml"]'}
        # A GE header names a measurement itself (MEASURE): its clips' text is recorded only, by default.
        decision, reasons, row = self.hybrid(**read)
        self.assertEqual((decision, reasons, row.text_evidence), ("keep", "", "measurement_text"))
        self.assertEqual(self.hybrid(**unit)[:2], ("keep", ""))
        # Another vendor's header, or none, cannot: there the text decides.
        self.assertEqual(self.hybrid(**PHILIPS, **read)[:2], ("exclude", "measurement_text"))
        self.assertEqual(self.hybrid(**PHILIPS, **unit)[:2], ("uncertain", "measurement_unit_text"))
        no_header = dict.fromkeys(["dicom_status", "dicom_image_content", "dicom_measure_flag", "dicom_source_image",
                                   "dicom_ecg", "dicom_sop_class"])
        self.assertEqual(self.hybrid(**no_header, **read)[:2], ("exclude", "measurement_text"))
        # The other scopes: every clip's text, or none.
        self.assertEqual(self.hybrid(TEXT_ON, **read)[:2], ("exclude", "measurement_text"))
        off = {**cs.THRESHOLDS, "text_decides": "none"}
        self.assertEqual(self.hybrid(off, **PHILIPS, **read)[:2], ("keep", ""))

    def test_text_names_a_measurement_before_it_excludes(self):
        # Doppler settings read as a velocity (the scale, the Nyquist limit, a color bar's): no evidence at all.
        lines = ["Scale 64 cm/s", "Nyquist 64 cm/s", "-64.0 cm/s", "0.8 m/s"]
        status, found = cs.measurement_text(speckle(), ("stand-in", lambda frame: lines))
        self.assertEqual(json.loads(found), ["-64.0 cm/s", "0.8 m/s", "64 cm/s"])  # what the text pass keeps,
        decision, reasons, row = self.hybrid(TEXT_ON, text_status=status, text_measurements=found)
        self.assertEqual((decision, reasons, row.text_evidence), ("keep", "", "none"))  # and what counts.
        # A number with a measurement unit but no name read: uncertain, never an exclusion on its own.
        self.assertEqual(self.hybrid(TEXT_ON, text_status="checked", text_measurements='["120 ml", "64 cm/s"]')[:2],
                         ("uncertain", "measurement_unit_text"))
        # A named velocity is a measurement.
        self.assertEqual(self.hybrid(TEXT_ON, text_status="checked", text_measurements='["64 cm/s", "V max 2.1"]')[:2],
                         ("exclude", "measurement_text"))
        # No header at all: its evidence is missing, not clean; the pixels still decide what they can.
        _, _, row = self.hybrid(dicom_status=None, dicom_image_content=None, dicom_measure_flag=None,
                                dicom_source_image=None, dicom_ecg=None)
        self.assertEqual((row.measurement_header, row.evidence_missing), ("unavailable", "header;text"))

    def test_uncertain_is_never_an_exclusion(self):
        # Motion only readable over the whole frame and frames that do not repeat: a still reading is uncertain;
        # repeated frames make it certain.
        decision, reasons, row = self.hybrid(region_source="whole_frame", region_motion_frac=0.001)
        self.assertEqual((decision, reasons, bool(row.exclude), row.exclude_reason),
                         ("uncertain", "still_low_confidence", False, ""))
        self.assertEqual(self.hybrid(region_source="whole_frame", region_motion_frac=0.001, duplicate_frac=1.0)[:2],
                         ("exclude", "still"))
        self.assertEqual(self.hybrid(region_motion_frac=0.01)[:2], ("exclude", "still"))  # read in the scan area.

    def test_a_layout_key_is_the_same_whatever_missing_values_and_dtypes(self):
        reference = [dict(r, model=None) for r in self.rows()]
        reference.append({"video_path": "/synthetic/broken", "error": "DECORDError", "vendor": "ge", "model": None})
        reference = pd.DataFrame(reference)  # the unreadable row makes width and height floats.
        self.assertEqual(reference.width.dtype, float)
        baselines, _ = cs.layout_baselines(reference)
        flags = cs.apply_flags(pd.DataFrame([dict(feature_row(self.box), model=None)]), baselines)
        self.assertEqual(flags.extra_cells.iloc[0], 18)

    def test_flags_when_no_clip_could_be_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = pd.DataFrame({"video_path": ["/a.mp4", "/b.mp4"], "global_key": ["a", "b"], "vendor": "v",
                                  "model": "m", "acq_category": "bmode"})
            table = cs.features_table([{"video_path": p, "error": "DECORDError"} for p in clips.video_path])
            features = os.path.join(tmp, "features.parquet")
            clips.merge(table, on="video_path").to_parquet(features)
            cs._write_info(features, {"extractor": cs.extractor()})  # the record of a completed run.
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(["flag", "--features", features, "--reference", features,
                         "--out", os.path.join(tmp, "flags.parquet"), *NO_TEXT])
            flags = pd.read_parquet(os.path.join(tmp, "flags.parquet"))
            self.assertEqual(flags.exclude_reason.tolist(), ["unreadable", "unreadable"])

    def test_still_and_unreadable_clips_are_excluded(self):
        rows = self.rows(2)
        rows.append(feature_row(self.header.copy(), motion=0.001))
        rows.append({"video_path": "/synthetic/broken", "error": "DECORDError"})
        flags = flag_rows(rows, self.rows())
        self.assertEqual(flags.exclude_reason.tolist()[-2:], ["still", "unreadable"])
        self.assertTrue(pd.isna(flags.still.iloc[-1]))

    def test_the_baseline_record_replaces_an_earlier_one(self):
        reference = self.rows() + [feature_row(self.header.copy(), layout="other") for _ in range(3)] + [
            feature_row(self.header.copy(), layout="still", motion=0.001)]
        reference = pd.DataFrame(reference)
        baselines, sources = cs.layout_baselines(reference)
        left_out = cs.reference_left_out(reference)
        left_out.index = cs.layout_keys(reference)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "flags.parquet.baselines")
            os.makedirs(out)
            open(os.path.join(out, "layout_009.png"), "w").close()  # from an earlier run.
            cs.write_baselines(out, baselines, sources, left_out)
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o700)
            layouts = pd.read_csv(os.path.join(out, "layouts.csv"))
            # A layout whose every reference clip was left out is listed too, with the reason.
            self.assertEqual(sorted(layouts.reference_clips), [0, 3, 30])
            self.assertEqual(layouts.left_out_low_motion.sum(), 1)
            self.assertEqual(sorted(f for f in os.listdir(out) if f.endswith(".png")),
                             sorted(layouts.png.dropna()))
            self.assertEqual(len(pd.read_csv(os.path.join(out, "sources.csv"))), 33)


class TestExtractor(unittest.TestCase):
    def test_the_record_follows_the_code_pass_1_runs(self):
        base = cs._code_record(*cs.PASS1)
        self.assertTrue({"clip_features", "frozen_patches", "read_frames", "media_error"} <= set(base["functions"]))
        self.assertTrue({"STATIC_STD", "WINDOW_MIN_FRAMES", "FROZEN_DENSITY", "RESOURCE_ERROR"}
                        <= set(base["constants"]))

        def frozen_patches(static, mean_gray):  # a detector without the frozen-patch guard.
            return np.zeros_like(static)

        with mock.patch.object(cs, "frozen_patches", frozen_patches):
            self.assertNotEqual(cs._code_record(*cs.PASS1)["code"], base["code"])
        with mock.patch.object(cs, "WINDOW_MIN_FRAMES", 10 ** 6):  # no windows.
            self.assertEqual(cs._code_record(*cs.PASS1)["constants"]["WINDOW_MIN_FRAMES"], 10 ** 6)
        # The flag rule's constants and the calibration are not pass 1's.
        with mock.patch.object(cs, "COMMON_SHARE", 0.2), mock.patch.object(cs, "CALIBRATION", {"screen": "x"}), \
                mock.patch.object(cs, "THRESHOLDS", {}):
            self.assertEqual(cs._code_record(*cs.PASS1), base)

    def test_comments_and_docstrings_do_not_count(self):
        def documented():
            def twice(x):
                """Doubled."""
                return 2 * x  # a comment.
            return twice

        def bare():
            def twice(x):
                return 2 * x
            return twice

        def changed():
            def twice(x):
                return 3 * x
            return twice

        self.assertEqual(cs._code_record(documented())["code"], cs._code_record(bare())["code"])
        self.assertNotEqual(cs._code_record(changed())["code"], cs._code_record(bare())["code"])

    def test_the_record_is_the_source_not_a_python_releases_rendering_of_it(self):
        # ast.unparse renders an f-string's quotes differently in Python 3.12.3 (the OCR environment) and 3.12.14
        # (the training one); the source as written, without docstring and comment, is the same in both.
        import ast

        source = 'def named():\n    """Named."""\n    return f"engine {str(\'x\')}"  # a comment.\n'
        self.assertEqual(cs._bare_source(source, ast.parse(source)), 'def named():\n    return f"engine {str(\'x\')}"')

    def flag(self, tmp, extracted, reference_extracted=None, *more):
        """`flag` on features (and reference features) whose records name these extractors (no record when
        None)."""
        clips = pd.DataFrame({"video_path": ["/a.mp4"], "global_key": ["a"], "vendor": "v", "model": "m",
                              "acq_category": "bmode"})
        table = clips.merge(cs.features_table([{"video_path": "/a.mp4", "error": "DECORDError"}]), on="video_path")
        paths = []
        for name, record in (("features", extracted), ("reference", reference_extracted or extracted)):
            paths.append(os.path.join(tmp, f"{name}.parquet"))
            table.to_parquet(paths[-1])
            cs._write_info(paths[-1], {} if record is None else {"extractor": record})
        flags = os.path.join(tmp, "flags.parquet")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            cs.main(["flag", "--features", paths[0], "--reference", paths[1], "--out", flags, *more])
        with open(flags + ".info.json") as f:
            return json.load(f), err.getvalue()

    def test_flag_refuses_features_of_another_detector(self):
        current = cs.extractor()

        def frozen_patches(static, mean_gray):
            return np.zeros_like(static)

        with mock.patch.object(cs, "frozen_patches", frozen_patches):
            other_code = cs.extractor()
        with mock.patch.object(cs, "WINDOW_MIN_FRAMES", 10 ** 6):
            no_windows = cs.extractor()
        other_library = {**current, "libraries": {**current["libraries"], "opencv": "0.0"}}
        with tempfile.TemporaryDirectory() as tmp:
            for extracted, reference, refused in (
                    (None, None, "does not record the code"),  # an older features run.
                    (other_code, None, "differ in code"),
                    (no_windows, None, "differ in WINDOW_MIN_FRAMES"),
                    (current, other_code, "differ in code"),  # the reference too.
                    (current, other_library, "other versions of ['opencv']")):
                with self.assertRaises(SystemExit) as stop:
                    self.flag(tmp, extracted, reference)
                self.assertIn(refused, str(stop.exception))
                self.assertFalse(os.path.exists(os.path.join(tmp, "flags.parquet")))
            info, _ = self.flag(tmp, other_library)  # both extracted alike, by this code.
        # The flags record the extractor behind them, and the rule's own code and constants.
        self.assertEqual(info["extractor"], other_library)
        self.assertIn("COMMON_SHARE", info["rule"]["constants"])
        self.assertFalse({"STATIC_STD", "THRESHOLDS"} & set(info["rule"]["constants"]))
        self.assertNotIn("constants", info)

    def test_a_calibration_holds_for_its_own_screen_and_thresholds_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = cs.extractor()
            screen = self.flag(tmp, current)[0]["screen"]
            calibration = {"date": "d", "report": "r", "screen": screen, "thresholds": dict(cs.THRESHOLDS)}
            with mock.patch.object(cs, "CALIBRATION", calibration):
                self.assertEqual(self.flag(tmp, current)[0]["calibration"], calibration)
                info, warned = self.flag(tmp, current, None, "--thresholds", '{"inside_glyphs": 3}')
                self.assertIsNone(info["calibration"])
                self.assertIn("their thresholds differ", warned)
                # Another policy is another screen decision: a calibration of the hybrid one does not hold for it.
                info, warned = self.flag(tmp, current, None, "--thresholds", '{"policy": "pixel"}')
                self.assertIsNone(info["calibration"])
                self.assertIn("their thresholds differ", warned)
                other = {**current, "libraries": {**current["libraries"], "decord": "0.0"}}
                info, warned = self.flag(tmp, other)  # extracted with another decoder.
                self.assertIsNone(info["calibration"])
                self.assertIn("their screen (", warned)
                with mock.patch.object(cs, "COMMON_SHARE", 0.2):  # another flag rule.
                    self.assertIsNone(self.flag(tmp, current)[0]["calibration"])
                # Flagged with another OpenCV or pandas, from the same features: another screen.
                import cv2

                for module in (cv2, pd):
                    with mock.patch.object(module, "__version__", "0.0"):
                        info, warned = self.flag(tmp, current)
                    self.assertNotEqual(info["screen"], screen)
                    self.assertIsNone(info["calibration"])
                    self.assertIn("their screen (", warned)
                # Defaults changed after the calibration: its screen is the same, but it never measured them.
                with mock.patch.object(cs, "THRESHOLDS", {**cs.THRESHOLDS, "inside_glyphs": 3}):
                    info, warned = self.flag(tmp, current)
                self.assertEqual(info["screen"], screen)
                self.assertIsNone(info["calibration"])
                self.assertIn("their thresholds differ", warned)
            # Defaults set from the calibration's report, with its thresholds, are calibrated.
            moved = {**calibration, "thresholds": {**cs.THRESHOLDS, "inside_glyphs": 3}}
            with mock.patch.object(cs, "CALIBRATION", moved), mock.patch.object(cs, "THRESHOLDS", moved["thresholds"]):
                self.assertEqual(self.flag(tmp, current)[0]["calibration"], moved)
            # So are flags given those values explicitly: they are the same flags.
            with mock.patch.object(cs, "CALIBRATION", moved):
                info, _ = self.flag(tmp, current, None, "--thresholds", '{"inside_glyphs": 3}')
                self.assertEqual(info["calibration"], moved)

    def test_the_calibration_names_the_default_thresholds(self):
        # The code itself: a calibration recorded with other defaults than it measured would never apply.
        if cs.CALIBRATION is not None:
            self.assertEqual(cs.CALIBRATION["thresholds"], cs.THRESHOLDS)
            self.assertEqual(set(cs.CALIBRATION), {"date", "report", "screen", "thresholds"})


def write_labels(path, rows, videos=None):
    """A label file as the sheet exports it: rows of (global_key, still, measurement, other_exclusion, cursor), each
    with the video bytes it is of (`videos`: global_key -> SHA-256; 'synthetic', as in the synthetic flags, else)."""
    videos = videos or {}
    pd.DataFrame([(r[0], videos.get(r[0], "synthetic"), *r[1:], "") for r in rows],
                 columns=["global_key", "video_sha256", *cs.LABEL_FIELDS, "note"]).to_csv(path, index=False)
    return path


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_answers_from_several_files_are_merged_whatever_their_order(self):
        a = write_labels(os.path.join(self.tmp, "a.csv"), [("k1", "yes", "", "", ""), ("k2", "no", "no", "no", "no")])
        b = write_labels(os.path.join(self.tmp, "b.csv"), [("k1", "yes", "no", "no", "yes"), ("k3", "", "", "", "")])
        merged = cs.read_labels([a, b])
        self.assertEqual(merged.loc["k1", list(cs.LABEL_FIELDS)].tolist(), ["yes", "no", "no", "yes"])
        self.assertEqual(merged.loc["k3", list(cs.LABEL_FIELDS)].tolist(), ["", "", "", ""])
        self.assertEqual(merged.loc["k1", "video_sha256"], "synthetic")
        pd.testing.assert_frame_equal(merged.sort_index(), cs.read_labels([b, a]).sort_index())

    def test_a_clip_answered_for_other_video_bytes_stops(self):
        # Its global key stays when a video is replaced in place: answers of the old and the new video differ in
        # their video_sha256 only, and are not merged.
        a = write_labels(os.path.join(self.tmp, "a.csv"), [("k1", "no", "no", "no", "no")], {"k1": "old"})
        b = write_labels(os.path.join(self.tmp, "b.csv"), [("k1", "no", "", "", "")], {"k1": "new"})
        for files in ([a, b], [b, a]):
            with self.assertRaises(SystemExit) as stop:
                cs.read_labels(files)
            self.assertIn("different video bytes", str(stop.exception))
        # A file exported from an older sheet, without the bytes, is refused.
        older = os.path.join(self.tmp, "older.csv")
        pd.read_csv(a).drop(columns="video_sha256").to_csv(older, index=False)
        with self.assertRaises(SystemExit) as stop:
            cs.read_labels([older])
        self.assertIn("video_sha256", str(stop.exception))

    def test_yes_and_no_to_one_question_stop_in_either_order(self):
        a = write_labels(os.path.join(self.tmp, "a.csv"), [("k1", "no", "no", "no", "no")])
        b = write_labels(os.path.join(self.tmp, "b.csv"), [("k1", "no", "yes", "", "")])
        for files in ([a, b], [b, a]):
            with self.assertRaises(SystemExit) as stop:
                cs.read_labels(files)
            self.assertIn("k1", str(stop.exception))

    def test_unknown_answers_are_refused(self):
        with self.assertRaises(SystemExit):
            cs.read_labels([write_labels(os.path.join(self.tmp, "a.csv"), [("k1", "maybe", "no", "no", "no")])])


def flag_record():
    """The record of a completed `flag` run of this script, for flags made by hand."""
    return {"thresholds": cs.THRESHOLDS, "rule": cs.flag_rule(), "extractor": cs.extractor()}


class TestSheet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        decision = ["exclude"] * 8 + ["uncertain"] * 2 + ["keep"] * 10
        self.flags = pd.DataFrame({"video_path": [f"/synthetic/{i}.mp4" for i in range(20)],
                                   "global_key": [f"key-{i}" for i in range(20)], "error": [None] * 20,
                                   "decision": decision, "exclude": [d == "exclude" for d in decision],
                                   "view": "A4C", "acq_category": "bmode", "video_sha256": "synthetic"})
        self.flags.to_parquet(os.path.join(self.tmp, "flags.parquet"))
        cs._write_info(os.path.join(self.tmp, "flags.parquet"), flag_record())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def sheet(self, name, *options):
        """A sheet of the synthetic flags; the clips' media (TestCommandLine makes real ones) are left out."""
        out = os.path.join(self.tmp, name)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(cs, "find_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(cs, "clip_media", return_value="<video></video>"), \
                mock.patch.object(cs, "changed_media", return_value=[]):  # synthetic flags: no media to check.
            cs.main(["sheet", "--flags", os.path.join(self.tmp, "flags.parquet"), "--out-dir", out, *options])
        with open(os.path.join(out, "sheet.info.json")) as f:
            return pd.read_csv(os.path.join(out, "sheet_clips.csv")), json.load(f), out

    def test_the_audit_sample_records_each_clip_inclusion_probability(self):
        clips, info, out = self.sheet("audit", "--n-flagged", "4", "--n-uncertain", "1", "--n-unflagged", "5")
        self.assertEqual(info["purpose"], "audit")
        # One stratum per decision: the kept clips alone, drawn with one probability, bound the ineligible kept ones.
        self.assertEqual(info["strata"], {"excluded": {"population": 8, "sampled": 4},
                                          "uncertain": {"population": 2, "sampled": 1},
                                          "kept": {"population": 10, "sampled": 5}})
        self.assertEqual(clips.groupby("stratum").inclusion_prob.first().to_dict(),
                         {"excluded": 0.5, "uncertain": 0.5, "kept": 0.5})
        with open(os.path.join(out, "clip-audit-01.html"), encoding="utf-8") as f:
            page = f.read()
        self.assertTrue(all(f"id={k} data-video='synthetic'>" in page for k in clips.global_key))
        # A round's folder is never rewritten: its sample record must stay with its pages.
        with self.assertRaises(SystemExit):
            self.sheet("audit")
        self.assertEqual(len(pd.read_csv(os.path.join(out, "sheet_clips.csv"))), 10)

    def test_strata_of_one_size_are_drawn_independently(self):
        # 50 excluded and 50 kept clips in the same order: a seed reused per stratum drew the same positions in both.
        flags = pd.DataFrame({"global_key": [f"k{i}" for i in range(100)], "error": None,
                              "decision": ["exclude"] * 50 + ["keep"] * 50})
        same = 0
        for seed in range(20):
            rows, _ = cs.audit_sample(flags, 10, 10, seed)
            drawn = rows.global_key.str[1:].astype(int)
            same += sorted(drawn[drawn < 50]) == sorted(drawn[drawn >= 50] - 50)
            again, _ = cs.audit_sample(flags, 10, 10, seed)  # seeded: the same sample again.
            pd.testing.assert_frame_equal(rows, again)
        self.assertEqual(same, 0)

    def test_every_nonempty_stratum_is_sampled(self):
        for option in ("--n-flagged", "--n-uncertain", "--n-unflagged"):
            with self.assertRaises(SystemExit):
                self.sheet(f"audit{option}", option, "0")
        # An audit sample recorded without one stratum (made by hand or by older code) is refused too.
        out = os.path.join(self.tmp, "partial")
        os.makedirs(out)
        with open(os.path.join(out, "sheet.info.json"), "w") as f:
            json.dump({"purpose": "audit", "strata": {"excluded": {"population": 8, "sampled": 0},
                                                      "kept": {"population": 12, "sampled": 6}}}, f)
        with self.assertRaises(SystemExit):
            cs.main(["calibrate", "--flags", os.path.join(self.tmp, "flags.parquet"), "--sheet-dir", out,
                     "--labels", os.path.join(out, "sheet.info.json")])

    def test_calibrate_needs_the_population_the_sample_was_drawn_from(self):
        clips, info, out = self.sheet("audit", "--n-flagged", "2", "--n-unflagged", "2")
        self.assertEqual(info["population"]["readable_clips"], 20)
        # The six sampled clips alone would give another review share while the weights still stand for 20.
        sampled = os.path.join(self.tmp, "sampled.parquet")
        self.flags[self.flags.global_key.isin(clips.global_key)].to_parquet(sampled)
        cs._write_info(sampled, flag_record())
        with self.assertRaises(SystemExit) as stop:
            cs.main(["calibrate", "--flags", sampled, "--sheet-dir", out, "--labels", sampled])
        self.assertIn("6 readable clips here, 20 there", str(stop.exception))

    def test_calibrate_needs_the_media_the_sample_was_drawn_from(self):
        # The reviewer's case: the clips not sampled replaced by other videos (same keys), or a DICOM header giving
        # other regions. The sampled clips' answers and their copies still match, but the weights would apply them to
        # a screen of other clips: refused, whichever clips changed.
        clips, info, out = self.sheet("audit", "--n-flagged", "2", "--n-uncertain", "1", "--n-unflagged", "2")
        self.assertIn("media_sha256", info["population"])
        flags = os.path.join(self.tmp, "flags.parquet")
        original = os.path.join(self.tmp, "original.parquet")
        shutil.copy(flags, original)
        unsampled = ~self.flags.global_key.isin(clips.global_key)
        calibrate = ["calibrate", "--flags", flags, "--sheet-dir", out, "--labels", flags]

        def screened(df):
            df.to_parquet(flags)
            cs._write_info(flags, flag_record())

        other = self.flags.global_key[unsampled].iloc[0]
        for changed in (self.flags.assign(video_sha256=np.where(unsampled, "replaced", "synthetic")),
                        self.flags.assign(dicom_regions=np.where(self.flags.global_key == other, "[[0, 0, 9, 9]]",
                                                                 None))):
            screened(changed)
            with self.assertRaises(SystemExit) as stop:
                cs.main(calibrate)
            self.assertIn("or other media", str(stop.exception))
        screened(self.flags)  # the same media: past the population (to the flags' missing features record).
        with self.assertRaises(SystemExit) as stop:
            cs.main(calibrate)
        self.assertIn("which features file", str(stop.exception))
        # A sheet made before sheets recorded their population's media: checked against the flags it was drawn from.
        with open(os.path.join(out, "sheet.info.json")) as f:
            record = json.load(f)
        record["population"].pop("media_sha256")
        record["flags"] = cs.sha256(original)
        with open(os.path.join(out, "sheet.info.json"), "w") as f:
            json.dump(record, f)
        for options, refused in (((), "older sheet"), (("--drawn-flags", flags + ".info.json"), "not the flags")):
            with self.assertRaises(SystemExit) as stop:
                cs.main(calibrate + list(options))
            self.assertIn(refused, str(stop.exception))
        screened(self.flags.assign(video_sha256=np.where(unsampled, "replaced", "synthetic")))
        with self.assertRaises(SystemExit) as stop:
            cs.main(calibrate + ["--drawn-flags", original])
        self.assertIn(f"{int(unsampled.sum())} of the clips the sample was drawn from hold other media",
                      str(stop.exception))
        screened(self.flags)
        with self.assertRaises(SystemExit) as stop:
            cs.main(calibrate + ["--drawn-flags", original])
        self.assertIn("which features file", str(stop.exception))

    def test_calibrate_needs_flags_of_this_scripts_rule_and_pass_1(self):
        # The report recomputes the flags with the current rule: flags of an edited rule would be reported under
        # their old screen. (TestCommandLine calibrates flags of this script's.)
        clips, _, out = self.sheet("audit", "--n-flagged", "2", "--n-unflagged", "2")
        labels = write_labels(os.path.join(self.tmp, "labels.csv"),
                              [(k, "no", "no", "no", "no") for k in clips.global_key])
        flags = os.path.join(self.tmp, "flags.parquet")
        calibrate = ["calibrate", "--flags", flags, "--sheet-dir", out, "--labels", labels]
        import cv2

        for patch, refused in ((mock.patch.object(cs, "COMMON_SHARE", 0.2), "another flag rule"),
                               (mock.patch.object(cv2, "__version__", "0.0"), "opencv version"),
                               (mock.patch.object(cs, "flag_columns", lambda df, thresholds: df), "another flag rule"),
                               (mock.patch.object(cs, "STATIC_STD", 3.0), "another pass 1")):
            with patch, self.assertRaises(SystemExit) as stop:
                cs.main(calibrate)
            self.assertIn(refused, str(stop.exception))
        cs._write_info(flags, {"thresholds": cs.THRESHOLDS})  # an older flag run, which recorded neither.
        with self.assertRaises(SystemExit) as stop:
            cs.main(calibrate)
        self.assertIn("do not record their flag rule", str(stop.exception))

    def test_a_clip_that_cannot_be_read_for_the_media_check_is_named_only_in_the_log(self):
        # A folder where a screened video was (permissions and storage errors take the same path): the check stops
        # the sheet, and its details, which name the clip, stay in the sheet folder's private log.
        clip = os.path.join(self.tmp, "SMITH-JOHN-1234567.mp4")
        os.mkdir(clip)
        self.flags.assign(video_path=[clip] + self.flags.video_path.tolist()[1:]).iloc[[0]].to_parquet(
            os.path.join(self.tmp, "flags.parquet"))
        cs._write_info(os.path.join(self.tmp, "flags.parquet"), flag_record())
        out = os.path.join(self.tmp, "audit")
        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit) as stop, mock.patch.object(cs, "find_ffmpeg", return_value="ffmpeg"):
            cs.main(["sheet", "--flags", os.path.join(self.tmp, "flags.parquet"), "--out-dir", out,
                     "--n-flagged", "1", "--n-unflagged", "0"])
        self.assertIn("stopped on a IsADirectoryError", str(stop.exception))
        self.assertNotIn("SMITH", str(stop.exception) + printed.getvalue() + err.getvalue())
        with open(os.path.join(out, "decoder.log")) as f:
            self.assertIn("SMITH-JOHN-1234567", f.read())
        self.assertEqual(os.stat(os.path.join(out, "decoder.log")).st_mode & 0o777, 0o600)
        self.assertFalse(os.path.exists(os.path.join(out, "sheet.info.json")))
        # The DICOM header's storage failing does the same.
        with open(os.path.join(self.tmp, "clip.mp4"), "wb") as f:
            f.write(b"bytes")
        rows = pd.DataFrame([{"global_key": "k0", "video_path": os.path.join(self.tmp, "clip.mp4"),
                              "video_sha256": cs.sha256(os.path.join(self.tmp, "clip.mp4")), "error": None,
                              "dicom_path": "/archive/SMITH-JOHN-1234567.dcm", "dicom_status": "ok",
                              "dicom_regions": "[]", "height": H, "width": W}])
        log = os.path.join(self.tmp, "media.log")
        failure = OSError("/archive/SMITH-JOHN-1234567.dcm: Input/output error")
        with mock.patch.object(cs, "dicom_bounds", side_effect=failure), self.assertRaises(SystemExit) as stop:
            cs.changed_media(rows, log)
        self.assertNotIn("SMITH", str(stop.exception))
        with open(log) as f:
            self.assertIn("SMITH-JOHN-1234567", f.read())

    def test_a_failure_stops_the_sheet_and_names_the_clip_only_in_its_log(self):
        out = os.path.join(self.tmp, "audit")
        failure = OSError("/archive/SMITH-JOHN-1234567.mp4: Input/output error")
        with contextlib.redirect_stdout(io.StringIO()) as printed, self.assertRaises(SystemExit) as stop, \
                mock.patch.object(cs, "find_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(cs, "clip_media", side_effect=failure), \
                mock.patch.object(cs, "changed_media", return_value=[]):
            cs.main(["sheet", "--flags", os.path.join(self.tmp, "flags.parquet"), "--out-dir", out])
        self.assertNotIn("SMITH", str(stop.exception) + printed.getvalue())
        with open(os.path.join(out, "decoder.log")) as f:
            self.assertIn("SMITH-JOHN-1234567", f.read())
        self.assertEqual(os.stat(os.path.join(out, "decoder.log")).st_mode & 0o777, 0o600)
        # The sheet's record is written last: calibrate refuses a sheet that stopped.
        self.assertFalse(os.path.exists(os.path.join(out, "sheet.info.json")))
        with self.assertRaises(SystemExit):
            cs.main(["calibrate", "--flags", os.path.join(self.tmp, "flags.parquet"), "--sheet-dir", out,
                     "--labels", os.path.join(self.tmp, "flags.parquet")])

    def test_a_validation_sheet_holds_every_clip_with_its_sampling_probability(self):
        flags = self.flags.assign(sample_stratum=["frozen_suspect"] * 5 + ["general"] * 15,
                                  sample_prob=[0.5] * 5 + [0.01] * 15, stratum_clips=[10] * 5 + [1500] * 15,
                                  population_clips=1510)
        flags.loc[19, "error"] = "DECORDError"  # an unreadable clip cannot be answered.
        # A list whose groups do not add up to its population lacks a group: refused.
        flags.assign(population_clips=1600).to_parquet(os.path.join(self.tmp, "flags.parquet"))
        cs._write_info(os.path.join(self.tmp, "flags.parquet"), flag_record())
        with self.assertRaises(SystemExit):
            self.sheet("missing_group", "--all-clips")
        flags.to_parquet(os.path.join(self.tmp, "flags.parquet"))
        cs._write_info(os.path.join(self.tmp, "flags.parquet"), flag_record())
        clips, info, _ = self.sheet("validation", "--all-clips")
        self.assertEqual(info["purpose"], "validation")
        self.assertEqual(info["strata"], {"frozen_suspect": {"population": 10, "sampled": 5},
                                          "general": {"population": 1500, "sampled": 14}})
        self.assertEqual(clips.groupby("stratum").inclusion_prob.unique().apply(list).to_dict(),
                         {"frozen_suspect": [0.5], "general": [0.01]})
        with self.assertRaises(SystemExit):  # one kind of sheet at a time.
            self.sheet("both", "--all-clips", "--keys", os.path.join(self.tmp, "flags.parquet"))
        # Sampled again, the list's clips would lose their own probabilities (0.5 and 0.01 would both become 1).
        for name, options in (("audit_of_list", ()), ("diagnostic_of_list", ("--diagnostic",))):
            with self.assertRaises(SystemExit) as stop:
                self.sheet(name, *options)
            self.assertIn("Use --all-clips", str(stop.exception))
        self.flags.to_parquet(os.path.join(self.tmp, "flags.parquet"))  # not a validation list's flags.
        cs._write_info(os.path.join(self.tmp, "flags.parquet"), flag_record())
        with self.assertRaises(SystemExit):
            self.sheet("plain", "--all-clips")

    def test_only_a_review_sheet_shows_the_detector(self):
        keys = os.path.join(self.tmp, "chosen_clips.csv")
        pd.DataFrame({"global_key": ["key-1"]}).to_csv(keys, index=False)
        for name, options, shown in (("audit", (), False), ("review", ("--keys", keys), True)):
            out = os.path.join(self.tmp, name)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(cs, "find_ffmpeg", return_value="f"), \
                    mock.patch.object(cs, "clip_media", return_value="<video></video>") as media, \
                    mock.patch.object(cs, "changed_media", return_value=[]):
                cs.main(["sheet", "--flags", os.path.join(self.tmp, "flags.parquet"), "--out-dir", out, *options])
            self.assertEqual({c.args[3] for c in media.call_args_list}, {shown}, name)
            with open(os.path.join(out, f"clip-{name}-01.html"), encoding="utf-8") as f:
                page = f.read()
            self.assertEqual("detector view" in page, shown)
            self.assertIn("Cursor line over the image?", page)

    def test_a_review_sheet_shows_the_clips_named(self):
        keys = os.path.join(self.tmp, "chosen_clips.csv")
        pd.DataFrame({"global_key": ["key-1", "key-15"]}).to_csv(keys, index=False)
        clips, info, out = self.sheet("review_01", "--keys", keys)
        self.assertEqual(info["purpose"], "review")
        self.assertEqual(sorted(clips.global_key), ["key-1", "key-15"])
        self.assertTrue(os.path.exists(os.path.join(out, "clip-review-01.html")))
        with self.assertRaises(SystemExit):  # not a random sample: no population estimate from it.
            cs.main(["calibrate", "--flags", os.path.join(self.tmp, "flags.parquet"), "--sheet-dir", out,
                     "--labels", keys])
        pd.DataFrame({"global_key": ["key-99"]}).to_csv(keys, index=False)
        with self.assertRaises(SystemExit):
            self.sheet("review_02", "--keys", keys)


def marked_row(cells, marks=(), window_marks=None, **more):
    """A feature row whose marks inside the image are pieces of these sizes over the whole clip (and in one window
    when `window_marks` is given)."""
    evidence = {"clip": [[a, "piece"] for a in marks],
                "windows": [] if window_marks is None else [[[a, "piece"] for a in window_marks]]}
    count = max(len(marks), len(window_marks or ()))
    return dict(feature_row(cells.copy(), inside=count), inside_marks=json.dumps(evidence), **more)


class TestDiagnostic(unittest.TestCase):
    """A screen of synthetic feature tables through flag, a diagnostic sheet (media left out) and calibrate. Its
    color layout is the case seen on real clips: every reference clip has one tiny mark inside."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        no_media = mock.patch.object(cs, "changed_media", return_value=[])  # synthetic tables: no videos to check.
        no_media.start()
        self.addCleanup(no_media.stop)
        header = np.zeros((H // cs.CELL, W // cs.CELL), bool)
        header[0, 0:8] = True
        box = header.copy()
        box[4:7, 4:10] = True
        targets = [marked_row(header, video_path=f"/t/b{i}.mp4", global_key=f"tb{i}") for i in range(6)]
        targets += [marked_row(box, video_path="/t/box.mp4", global_key="tbox"),
                    marked_row(header, [20, 20], video_path="/t/cal.mp4", global_key="tcal"),
                    marked_row(header, [], [4, 4], video_path="/t/win.mp4", global_key="twin")]
        targets += [marked_row(header, [4], video_path=f"/t/c{i}.mp4", global_key=f"tc{i}", acq_category="color_flow")
                    for i in range(4)]
        reference = [marked_row(header, video_path=f"/r/b{i}.mp4", global_key=f"rb{i}") for i in range(30)]
        reference += [marked_row(header, [4], video_path=f"/r/c{i}.mp4", global_key=f"rc{i}",
                                 acq_category="color_flow") for i in range(30)]
        self.paths = {}
        for name, rows in (("features", targets), ("reference", reference)):
            path = self.paths[name] = os.path.join(self.tmp, f"{name}.parquet")
            pd.DataFrame(rows).to_parquet(path)
            cs._write_info(path, {"clips": name, "extractor": cs.extractor()})
        self.flags = os.path.join(self.tmp, "flags.parquet")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cs.main(["flag", "--features", self.paths["features"], "--reference", self.paths["reference"],
                     "--out", self.flags, *NO_TEXT])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cases(self):
        flags = pd.read_parquet(self.flags).set_index("global_key")
        cases = cs.diagnostic_cases(flags, cs.THRESHOLDS)
        self.assertEqual(cases[["tb0", "tbox", "tcal", "twin", "tc0"]].tolist(),
                         ["clean", "results_box", "large_marks", "window_small_marks", "small_marks"])
        dotted = flags.loc[["tb0"]].assign(inside_marks=json.dumps({"clip": [[30, "line"]], "windows": []}))
        self.assertEqual(cs.diagnostic_cases(dotted, cs.THRESHOLDS).tolist(), ["dotted_line"])
        # One traced contour is one mark, below the threshold: a case of its own, so the labels can say what it is.
        traced = flags.loc[["tb0"]].assign(inside_marks=json.dumps({"clip": [], "windows": [[[90, "trace"]]]}))
        self.assertEqual(cs.diagnostic_cases(traced, cs.THRESHOLDS).tolist(), ["traced"])

    def replaced_during(self, name, path, write):
        """A patch of clip_screen's `name` that first rewrites `path` (with `write`, and a matching record of a
        completed run when it has one), as another run would while this one works."""
        real = getattr(cs, name)

        def meanwhile(*args, **kwargs):
            write(path)
            if os.path.exists(path + ".info.json"):
                with open(path + ".info.json") as f:
                    info = json.load(f)
                cs._write_info(path, {k: v for k, v in info.items()
                                      if k not in ("created_utc", "code_commit", "code_sha256", "output_sha256")})
            return real(*args, **kwargs)

        return mock.patch.object(cs, name, meanwhile)

    def test_inputs_replaced_while_a_step_runs_complete_nothing(self):
        # flag: the features file replaced, with its record, by one whose clip needs excluding (two calipers).
        features = pd.read_parquet(self.paths["features"])
        measured = features.assign(inside_marks=json.dumps({"clip": [[20, "caliper"]] * 2, "windows": []}),
                                   inside_glyphs=2)
        with self.replaced_during("screen_flags", self.paths["features"], measured.to_parquet), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
            cs.main(["flag", "--features", self.paths["features"], "--reference", self.paths["reference"],
                     "--out", self.flags, *NO_TEXT])
        self.assertIn("features.parquet changed while this step ran", str(stop.exception))
        self.assertFalse(any(os.path.exists(self.flags + x) for x in ("", ".info.json", ".baselines")))
        with self.assertRaises(SystemExit):  # no chain to follow.
            cs.screen_sources(self.flags)
        features.to_parquet(self.paths["features"])
        cs._write_info(self.paths["features"], {"clips": "features", "extractor": cs.extractor()})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cs.main(["flag", "--features", self.paths["features"], "--reference", self.paths["reference"],
                     "--out", self.flags, *NO_TEXT])
        # sheet: the flags replaced while the pages are made.
        out = os.path.join(self.tmp, "diagnostic")
        flags = pd.read_parquet(self.flags)
        with self.replaced_during("write_pages", self.flags, flags.iloc[1:].to_parquet), \
                mock.patch.object(cs, "find_ffmpeg", return_value="f"), \
                mock.patch.object(cs, "clip_media", return_value="<video></video>"), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
            cs.main(["sheet", "--flags", self.flags, "--out-dir", out, "--diagnostic"])
        self.assertIn("flags.parquet changed while this step ran", str(stop.exception))
        self.assertFalse(os.path.exists(out))
        flags.to_parquet(self.flags)
        info = json.load(open(self.flags + ".info.json"))
        cs._write_info(self.flags, {k: v for k, v in info.items()
                                    if k not in ("created_utc", "code_commit", "code_sha256", "output_sha256")})
        with mock.patch.object(cs, "find_ffmpeg", return_value="f"), \
                mock.patch.object(cs, "clip_media", return_value="<video></video>"), \
                contextlib.redirect_stdout(io.StringIO()):
            cs.main(["sheet", "--flags", self.flags, "--out-dir", out, "--diagnostic"])
        # calibrate: a label file rewritten while the report is computed; nothing is printed.
        clips = pd.read_csv(os.path.join(out, "sheet_clips.csv"))
        labels = write_labels(os.path.join(self.tmp, "labels.csv"),
                              [(k, "no", "no", "no", "no") for k in clips.global_key])
        flipped = [(k, "no", "yes", "no", "no") for k in clips.global_key]
        with self.replaced_during("calibration_report", labels, lambda p: write_labels(p, flipped)), \
                contextlib.redirect_stdout(io.StringIO()) as printed, self.assertRaises(SystemExit) as stop:
            cs.main(["calibrate", "--flags", self.flags, "--sheet-dir", out, "--labels", labels])
        self.assertIn("labels.csv changed while this step ran", str(stop.exception))
        self.assertEqual(printed.getvalue(), "")

    def test_thresholds_out_of_their_type_or_range_are_refused(self):
        # The reviewer's case: a NaN minimum mark size compares false with every mark, so a caliper pair would be
        # kept; flag refuses it and leaves no completed record.
        for given, refused in (('{"inside_mark_px": NaN}', "NaN"), ('{"still_motion_frac": Infinity}', "Infinity"),
                               ('{"inside_mark_px": 2.5}', "inside_mark_px=2.5"),
                               ('{"inside_glyphs": true}', "inside_glyphs=True"),
                               ('{"extra_cells": 0}', "extra_cells=0"),
                               ('{"still_scan_motion_frac": 1.5}', "still_scan_motion_frac=1.5"),
                               ('{"still_motion_frac": "0.01"}', "still_motion_frac='0.01'"),
                               ('{"policy": "strict"}', "policy='strict'"), ('{"cells": 3}', "keys must be among"),
                               ('{"text_decides": true}', "text_decides=True"),
                               ('[1]', "JSON object"), ('{"inside_glyphs": }', "not valid JSON")):
            with self.assertRaises(SystemExit) as stop, contextlib.redirect_stdout(io.StringIO()):
                cs.main(["flag", "--features", self.paths["features"], "--reference", self.paths["reference"],
                         "--out", self.flags, "--thresholds", given])
            self.assertIn(refused, str(stop.exception), given)
            self.assertFalse(os.path.exists(self.flags + ".info.json"))
        # Whole numbers for counts and sizes, any finite share from 0 to 1 (an integer one too).
        self.assertEqual(cs.check_thresholds({**cs.THRESHOLDS, "still_scan_motion_frac": 0, "inside_mark_px": 5}),
                         {**cs.THRESHOLDS, "still_scan_motion_frac": 0, "inside_mark_px": 5})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cs.main(["flag", "--features", self.paths["features"], "--reference", self.paths["reference"],
                     "--out", self.flags, *NO_TEXT])
        out = os.path.join(self.tmp, "diagnostic")  # calibrate checks its --thresholds alike.
        with mock.patch.object(cs, "find_ffmpeg", return_value="f"), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(cs, "clip_media", return_value="<video></video>"):
            cs.main(["sheet", "--flags", self.flags, "--out-dir", out, "--diagnostic"])
        with self.assertRaises(SystemExit) as stop:
            cs.main(["calibrate", "--flags", self.flags, "--sheet-dir", out, "--labels", self.flags,
                     "--thresholds", '{"inside_mark_px": NaN}'])
        self.assertIn("NaN", str(stop.exception))

    def test_an_older_sheet_is_checked_against_the_flags_it_was_drawn_from(self):
        # The pilot's case: a diagnostic sheet made before sheets recorded their population's media, scored against
        # flags of re-extracted features in another folder. The clips' media are checked against the flags it was
        # drawn from, screened and reference clips alike.
        out = os.path.join(self.tmp, "diagnostic")
        with mock.patch.object(cs, "find_ffmpeg", return_value="f"), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(cs, "clip_media", return_value="<video></video>"):
            cs.main(["sheet", "--flags", self.flags, "--out-dir", out, "--diagnostic"])
        with open(os.path.join(out, "sheet.info.json")) as f:
            record = json.load(f)
        for part in ("population", "reference_population"):
            record[part].pop("media_sha256")
        with open(os.path.join(out, "sheet.info.json"), "w") as f:
            json.dump(record, f)
        clips = pd.read_csv(os.path.join(out, "sheet_clips.csv"))
        labels = write_labels(os.path.join(self.tmp, "labels.csv"),
                              [(k, "no", "no", "no", "no") for k in clips.global_key])
        os.mkdir(os.path.join(self.tmp, "again"), 0o700)
        flags = os.path.join(self.tmp, "again", "flags.parquet")

        def rescreen(reference):
            for name, rows in (("features", pd.read_parquet(self.paths["features"])), ("reference", reference)):
                path = os.path.join(self.tmp, "again", f"{name}.parquet")
                rows.to_parquet(path)
                cs._write_info(path, {"clips": name, "extractor": cs.extractor()})
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cs.main(["flag", "--features", os.path.join(self.tmp, "again", "features.parquet"),
                         "--reference", os.path.join(self.tmp, "again", "reference.parquet"), "--out", flags,
                         *NO_TEXT])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", out, "--labels", labels,
                         "--drawn-flags", self.flags])
            return json.loads(buf.getvalue())

        reference = pd.read_parquet(self.paths["reference"])
        provenance = rescreen(reference)["provenance"]
        self.assertEqual(provenance["drawn_flags"], cs.sha256(self.flags))
        self.assertIn("video_sha256", provenance["population_media"])
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stdout(io.StringIO()):
            cs.main(["calibrate", "--flags", flags, "--sheet-dir", out, "--labels", labels])
        self.assertIn("older sheet", str(stop.exception))
        # A reference clip not on the sheet whose video was replaced since it: refused.
        unsampled = reference.global_key[~reference.global_key.isin(clips.global_key)].iloc[0]
        reference.loc[reference.global_key == unsampled, "video_sha256"] = "replaced"
        with self.assertRaises(SystemExit) as stop:
            rescreen(reference)
        self.assertIn("or hold other media", str(stop.exception))

    def test_a_challenge_set_is_drawn_per_signal_labelled_blind_and_only_counted(self):
        listed = os.path.join(self.tmp, "challenge", "challenge.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            cs.main(["challenge", "--flags", self.flags, "--out", listed, "--per-signal", "2"])
        rows = pd.read_csv(listed)
        with open(listed + ".info.json") as f:
            info = json.load(f)
        # The box clip has a results-box score but is kept; the color layout has no baseline (an unseen layout).
        self.assertEqual(info["signals"]["kept_results_box_score"], {"population": 1, "drawn": 1})
        self.assertEqual(info["signals"]["unseen_layout"], {"population": 4, "drawn": 2})
        self.assertTrue(rows.signals.str.contains("kept_results_box_score").any())
        with contextlib.redirect_stdout(io.StringIO()):  # seeded: the same list again.
            cs.main(["challenge", "--flags", self.flags, "--out", listed, "--per-signal", "2"])
        pd.testing.assert_frame_equal(pd.read_csv(listed), rows)
        out = os.path.join(self.tmp, "challenge_sheet")
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(cs, "find_ffmpeg", return_value="f"):
            cs.main(["sheet", "--flags", self.flags, "--out-dir", out, "--blind"])  # --blind needs --keys.
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(cs, "find_ffmpeg", return_value="f"), \
                mock.patch.object(cs, "clip_media", return_value="<video></video>") as media:
            cs.main(["sheet", "--flags", self.flags, "--out-dir", out, "--keys", listed, "--blind"])
        self.assertEqual({c.args[3] for c in media.call_args_list}, {False})  # labelled blind.
        sheet = pd.read_csv(os.path.join(out, "sheet_clips.csv"))
        self.assertEqual(sorted(sheet.global_key), sorted(rows.global_key))
        labels = write_labels(os.path.join(self.tmp, "challenge_labels.csv"),
                              [(k, "no", "yes" if k == "tbox" else "no", "no", "no") for k in sheet.global_key])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cs.main(["calibrate", "--flags", self.flags, "--sheet-dir", out, "--labels", labels])
        report = json.loads(buf.getvalue())
        self.assertIn("enriched challenge set", report["sample_design"])
        self.assertIsNone(report["policy"]["kept"]["upper_95"])
        hybrid = report["policies"]["hybrid"]["all"]
        self.assertEqual(hybrid["by_answer"]["measured"], {"keep": 1, "exclude": 0, "uncertain": 0})  # the box: a miss.
        self.assertEqual(report["policies"]["pixel"]["all"]["by_answer"]["measured"]["exclude"], 1)
        # Drawn from a validation list's flags (patients the screen was not tuned on): counted all the same.
        listed_flags = pd.read_parquet(self.flags).assign(sample_prob=0.5)
        listed_flags.to_parquet(self.flags)
        with open(self.flags + ".info.json") as f:
            record = json.load(f)
        cs._write_info(self.flags, {k: v for k, v in record.items()
                                    if k not in ("created_utc", "code_commit", "code_sha256", "output_sha256")})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cs.main(["calibrate", "--flags", self.flags, "--sheet-dir", out, "--labels", labels])
        self.assertIn("enriched challenge set", json.loads(buf.getvalue())["sample_design"])

    def test_a_diagnostic_sheet_weighs_screened_and_reference_clips_apart(self):
        out = os.path.join(self.tmp, "diagnostic")
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(cs, "find_ffmpeg", return_value="f"), \
                mock.patch.object(cs, "clip_media", return_value="<video></video>") as media:
            cs.main(["sheet", "--flags", self.flags, "--out-dir", out, "--diagnostic", "--n-per-stratum", "2"])
        self.assertEqual({c.args[3] for c in media.call_args_list}, {False})  # no detector output.
        clips = pd.read_csv(os.path.join(out, "sheet_clips.csv"))
        with open(os.path.join(out, "sheet.info.json")) as f:
            info = json.load(f)
        self.assertEqual(info["purpose"], "diagnostic")
        self.assertEqual(info["strata"]["target:bmode:clean"], {"population": 6, "sampled": 2})
        self.assertEqual(info["strata"]["reference:color_flow:small_marks"], {"population": 30, "sampled": 2})
        self.assertEqual(info["strata"]["target:bmode:window_small_marks"], {"population": 1, "sampled": 1})
        self.assertEqual(sorted(clips.role.unique()), ["reference", "target"])
        self.assertAlmostEqual(clips.set_index("stratum").inclusion_prob["target:bmode:clean"].iloc[0], 1 / 3)
        # Every clip answered: the tiny marks are not measurements, the box and the calipers are.
        measured = {"tbox", "tcal"}
        labels = write_labels(os.path.join(self.tmp, "labels.csv"),
                              [(k, "no", "yes" if k in measured else "no", "no", "no") for k in clips.global_key])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cs.main(["calibrate", "--flags", self.flags, "--sheet-dir", out, "--labels", labels])
        report = json.loads(buf.getvalue())
        self.assertEqual(report["labelled"], int((clips.role == "target").sum()))
        smallest, five = (report["inside_mark_px_sweep"][px] for px in (str(cs.MIN_INK_PX), "5"))
        color = "ge | m | 200 | 160 | color_flow"
        self.assertEqual((smallest["baselines"][color]["baseline"], five["baselines"][color]["baseline"]),
                         (False, True))
        self.assertEqual(smallest["reference_clips"]["by_acquisition"]["color_flow"]["left_out"],
                         {"marks_inside": 2})
        self.assertEqual(five["reference_clips"]["by_acquisition"]["color_flow"]["left_out"], {"used": 2})
        self.assertEqual(five["reference_clips"]["used_with_measurement"], 0)
        # Weighed up to the screened clips: at 5 px the detector flags no clean clip and finds both measurements,
        # which the default policy then keeps (the pixel measurement rules do not exclude); at 3 px it flags clean ones.
        self.assertEqual((five["measurement"]["fp"], five["measurement"]["fn"]), (0.0, 0.0))
        self.assertEqual((five["policy"]["fp"], five["policy"]["fn"]), (0.0, 2.0))
        self.assertGreater(smallest["measurement"]["fp"], 0)
        # A reference population other than the one drawn from is refused.
        pd.read_parquet(self.paths["reference"]).iloc[:-1].to_parquet(self.paths["reference"])
        cs._write_info(self.paths["reference"], {"clips": "reference", "extractor": cs.extractor()})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cs.main(["flag", "--features", self.paths["features"], "--reference", self.paths["reference"],
                     "--out", self.flags, *NO_TEXT])
        with self.assertRaises(SystemExit) as stop:
            cs.main(["calibrate", "--flags", self.flags, "--sheet-dir", out, "--labels", labels])
        self.assertIn("not those the diagnostic sample was drawn from", str(stop.exception))


class TestWorker(unittest.TestCase):
    def test_an_empty_clip_list_gives_a_typed_table_that_flag_refuses_as_reference(self):
        # A reference list left empty (every patient drawn into a validation sample): features writes an empty table
        # whose columns keep their types, and flag refuses it, as no layout would get a baseline.
        self.assertEqual(cs.features_table([]).video_path.dtype, object)
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(columns=["video_path", "global_key", "vendor", "model", "acq_category"]).to_csv(
                os.path.join(tmp, "clips.csv"), index=False)
            empty = os.path.join(tmp, "screen", "reference.parquet")
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(["features", "--clips", os.path.join(tmp, "clips.csv"), "--out", empty, "--workers", "1"])
            self.assertEqual(len(pd.read_parquet(empty)), 0)
            with self.assertRaises(SystemExit) as stop:
                cs.main(["flag", "--features", empty, "--reference", empty,
                         "--out", os.path.join(tmp, "screen", "flags.parquet")])
            self.assertIn("--reference has no clips", str(stop.exception))

    def test_a_missing_file_stops_the_run(self):
        with self.assertRaises(OSError):
            cs._features_worker(("/synthetic/missing.mp4", None))

    @unittest.skipUnless(HAS_DECODER, "needs decord")
    def test_a_storage_failure_stops_the_run_and_a_decoder_error_marks_the_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp4")
            with open(path, "wb") as f:
                f.write(b"placeholder")
            real_open = open

            def storage_fails(file, *args, **kwargs):
                if file == path:
                    raise OSError("storage")
                return real_open(file, *args, **kwargs)

            with mock.patch("builtins.open", storage_fails), self.assertRaises(OSError):
                cs._features_worker((path, None))
            # The file read whole, the decoder's own error is the clip's; the bytes it rejected are named.
            cannot_open = RuntimeError("Error reading 11 bytes...")  # decord's own error for bytes it cannot open.
            with mock.patch.object(cs, "read_frames", side_effect=cannot_open):
                row = cs._features_worker((path, None))
            self.assertEqual((row["error"], row["video_sha256"]), ("RuntimeError", cs.sha256(path)))

    @unittest.skipUnless(HAS_DECODER, "needs decord")
    def test_only_the_decoders_own_media_errors_make_a_clip_unreadable(self):
        import decord

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp4")
            with open(path, "wb") as f:
                f.write(b"placeholder")
            for exc in (decord.DECORDError("corrupt packet"), decord.DECORDLimitReachedError("too many bad frames"),
                        AssertionError("Invalid frame count: 0")):
                with mock.patch.object(cs, "read_frames", side_effect=exc):
                    self.assertEqual(cs._features_worker((path, None))["error"], type(exc).__name__)
            # Anything else, or a decoder error naming the machine's resources, stops the run: the clip may be fine.
            for exc in (RuntimeError("can't start new thread"), RuntimeError("bad stream"), ValueError("x"),
                        decord.DECORDError("Cannot allocate memory"), AssertionError("unexpected")):
                with mock.patch.object(cs, "read_frames", side_effect=exc), self.assertRaises(type(exc)):
                    cs._features_worker((path, None))

    @unittest.skipUnless(HAS_DECODER, "needs decord")
    def test_only_a_file_the_decoder_rejects_is_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.mp4")
            with open(path, "wb") as f:
                f.write(b"not a video" * 100)
            self.assertIsNotNone(cs._features_worker((path, None))["error"])
            with mock.patch.object(cs, "read_frames", side_effect=MemoryError), self.assertRaises(MemoryError):
                cs._features_worker((path, None))
            with mock.patch.object(cs, "read_frames", return_value=(speckle(), T)), \
                    mock.patch.object(cs, "clip_features", side_effect=ValueError), self.assertRaises(ValueError):
                cs._features_worker((path, None))

    @unittest.skipUnless(HAS_DECODER and HAS_PYDICOM, "needs decord and pydicom")
    def test_the_dicom_regions_give_the_scan_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "clip.mp4")
            write_mp4(video, speckle(frozen=True))
            row = cs._features_worker((video, write_dicom(os.path.join(tmp, "clip.dcm"))))
            self.assertEqual(row["dicom_status"], "ok")
            self.assertGreater(row["scan_frac"], 0.4)
            with self.assertRaises(OSError):  # a missing DICOM stops the run, like a missing video.
                cs._features_worker((video, os.path.join(tmp, "missing.dcm")))


@unittest.skipUnless(HAS_PYDICOM, "needs pydicom")
class TestDicomBounds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_storage_and_memory_failures_stop_the_run(self):
        path = write_dicom(os.path.join(self.tmp, "a.dcm"))
        for failure in (OSError("storage"), MemoryError()):
            with mock.patch("pydicom.dcmread", side_effect=failure), self.assertRaises(type(failure)):
                cs.dicom_bounds(path, (H, W))

    def test_image_regions_are_read_and_clipped_to_the_frame(self):
        path = write_dicom(os.path.join(self.tmp, "a.dcm"),
                           regions=((1, (30, 30, 169, 149)), (2, (60, 40, 250, 120)), (10, (0, 150, 199, 159))))
        self.assertEqual(cs.dicom_bounds(path, (H, W)), ([(30, 30, 169, 149), (60, 40, 199, 120)], "ok"))

    def test_regions_that_do_not_apply(self):
        cases = [(write_dicom(os.path.join(self.tmp, "b.dcm"), rows=H + 10), "size_mismatch"),
                 (write_dicom(os.path.join(self.tmp, "c.dcm"), regions=()), "no_regions"),
                 (write_dicom(os.path.join(self.tmp, "d.dcm"), regions=((10, (0, 150, 199, 159)),)), "no_regions")]
        broken = os.path.join(self.tmp, "e.dcm")
        with open(broken, "wb") as f:
            f.write(b"not a header" * 50)
        cases.append((broken, "unreadable_header"))
        for path, status in cases:
            self.assertEqual(cs.dicom_bounds(path, (H, W)), (None, status), path)
        with self.assertRaises(OSError):
            cs.dicom_bounds(os.path.join(self.tmp, "missing.dcm"), (H, W))


class TestCalibration(unittest.TestCase):
    def setUp(self):
        # Both flagged clips are on the sheet, and 2 of 100 unflagged ones.
        keys, paths = ["ka", "kb", "kc", "kd"], ["a", "b", "c", "d"]
        blank = np.zeros((H // cs.CELL, W // cs.CELL), bool)
        self.flags = pd.DataFrame([dict(feature_row(blank.copy(), inside=n), video_path=p, global_key=k,
                                        patient_id=f"p{p}") for p, k, n in zip(paths, keys, [5, 4, 0, 0])])
        self.reference = pd.DataFrame([feature_row(blank.copy()) for _ in range(30)])
        self.sheet = pd.DataFrame({"global_key": keys, "video_path": paths,
                                   "stratum": ["flagged", "flagged", "unflagged", "unflagged"],
                                   "inclusion_prob": [1.0, 1.0, 0.02, 0.02]})
        self.answers = pd.DataFrame({"still": ["no"] * 5, "measurement": ["yes", "no", "yes", "no", "yes"],
                                     "other_exclusion": ["no"] * 5, "cursor": ["no"] * 5},
                                    index=pd.Index([*keys, "k_other_sheet"], name="global_key"))

    def test_unflagged_sample_is_weighted_up_to_its_stratum(self):
        report = cs.calibration_report(self.flags, self.reference, self.sheet, self.answers)
        self.assertEqual(report["labelled"], 4)
        at = report["measurement"]["at_thresholds"]
        self.assertEqual((at["tp"], at["fp"], at["fn"], at["tn"]), (1.0, 1.0, 50.0, 50.0))
        self.assertAlmostEqual(at["sensitivity"], 1 / 51)

    def test_the_policy_counts_every_reason_to_exclude(self):
        answers = self.answers.copy()
        answers.loc["kd", "other_exclusion"] = "yes"
        # An audit sample: each screened clip's stratum, as drawn.
        drawn = pd.Series(["excluded", "excluded", "kept", "kept"], index=self.flags.global_key)
        report = cs.calibration_report(self.flags, self.reference, self.sheet, answers, PIXEL_POLICY, design="audit",
                                       drawn=drawn)
        keeps = report["policy"]["kept"]
        self.assertEqual((keeps["sampled"], keeps["ineligible"]), (2, 2))
        self.assertEqual(keeps["ineligible_by_question"], {"still": 0, "measurement": 1, "other_exclusion": 1})
        self.assertEqual(keeps["share_ineligible"], 1.0)
        self.assertEqual(keeps["upper_95"], 1.0)
        # Counted on the whole flags table (here the four clips), not estimated from the sample.
        self.assertEqual((report["policy"]["screened_clips"], report["policy"]["excluded_share"]), (4, 0.5))
        self.assertEqual(report["policy"]["at_thresholds"]["fn"], 100.0)

    def test_the_bound_follows_the_sampling_design(self):
        blank = np.zeros((H // cs.CELL, W // cs.CELL), bool)

        def screen(marked, n=100):  # n clips of 5 patients; the first `marked` carry marks the pixel policy excludes.
            return pd.DataFrame([dict(feature_row(blank.copy(), inside=5 if i < marked else 0), video_path=f"v{i}",
                                      global_key=f"k{i}", patient_id=f"p{i % 5}") for i in range(n)])

        def answers(keys):
            return pd.DataFrame({"still": "no", "measurement": "no", "other_exclusion": "no", "cursor": "no"},
                                index=pd.Index(keys, name="global_key"))

        def kept(clips, sheet, thresholds=cs.THRESHOLDS, **design):
            return cs.calibration_report(clips, self.reference, sheet, answers(sheet.global_key), thresholds,
                                         **design)["policy"]["kept"]

        # The reviewer's case: a validation list draws 5 patients, then 20 clips of each, and the screen keeps one
        # clip per patient, all of one probability. Holding one clip per patient does not make a cluster sample's
        # clips independent draws: no bound (in the reviewer's population it claimed 45% where 69% were ineligible).
        clips = screen(95).iloc[::-1].reset_index(drop=True)  # the five clean clips are of five patients.
        sheet = pd.DataFrame({"global_key": clips.global_key, "video_path": clips.video_path, "stratum": "general",
                              "inclusion_prob": 0.1})
        listed = kept(clips, sheet, PIXEL_POLICY, flags_are_population=False, design="validation")
        self.assertEqual((listed["sampled"], listed["patients"], listed["upper_95"]), (5, 5, None))
        self.assertIn("cluster sample", listed["upper_95_withheld"])
        unknown = kept(clips, sheet, PIXEL_POLICY, flags_are_population=False)  # no design given: none either.
        self.assertEqual((unknown["upper_95"], unknown["upper_95_withheld"]), (None, cs.BOUND_WITHHELD[None]))
        # An audit sample: its kept stratum is a simple random sample of the clips the screen keeps, whatever their
        # patients (here 20 of 100 clips of 5 patients), so the bound holds, for those 100 clips.
        clips = screen(0)
        sheet = pd.DataFrame({"global_key": clips.global_key[:20], "video_path": clips.video_path[:20],
                              "stratum": "kept", "inclusion_prob": 0.2})
        audit = kept(clips, sheet, design="audit", drawn=pd.Series("kept", index=clips.global_key))
        self.assertEqual((audit["sampled"], audit["patients"], audit["upper_95_withheld"]), (20, 5, None))
        self.assertAlmostEqual(audit["upper_95"], 1 - 0.05 ** (1 / 20), places=6)
        self.assertIn("the 100 clips this screen keeps of 5 patients", audit["upper_95_describes"])
        # Flags whose decisions are not those the sample was drawn from leave its strata unknown: no bound.
        self.assertIn("not those the audit sample was drawn from",
                      kept(clips, sheet, design="audit")["upper_95_withheld"])
        # Drawn under the pixel policy (10 marked clips excluded, all sampled; a fifth of the 90 clean ones), the
        # sample bounds the clean clips that policy keeps; the hybrid policy keeps the marked ones too, of another
        # stratum, so the kept clips sampled are no simple random sample of those it keeps.
        clips = screen(10)
        sheet = pd.DataFrame({"global_key": clips.global_key[:28], "video_path": clips.video_path[:28],
                              "stratum": ["excluded"] * 10 + ["kept"] * 18, "inclusion_prob": [1.0] * 10 + [0.2] * 18})
        drawn = pd.Series(["excluded"] * 10 + ["kept"] * 90, index=clips.global_key)
        pixel = kept(clips, sheet, PIXEL_POLICY, design="audit", drawn=drawn)
        self.assertAlmostEqual(pixel["upper_95"], 1 - 0.05 ** (1 / 18), places=6)
        self.assertIn("the 90 clips this screen keeps", pixel["upper_95_describes"])
        hybrid = kept(clips, sheet, design="audit", drawn=drawn)
        self.assertEqual((hybrid["sampled"], hybrid["upper_95"]), (28, None))
        self.assertIn("several audit strata", hybrid["upper_95_withheld"])
        # The reviewer's case: strata of one probability (50 excluded and 50 kept clips, 10 of each) that the hybrid
        # policy merges are still two strata, a stratified sample: no bound (it claimed 13.9% where 20% were).
        clips = screen(50)
        sheet = pd.DataFrame({"global_key": pd.concat([clips.global_key[:10], clips.global_key[50:60]]),
                              "video_path": pd.concat([clips.video_path[:10], clips.video_path[50:60]]),
                              "stratum": ["excluded"] * 10 + ["kept"] * 10, "inclusion_prob": 0.2})
        drawn = pd.Series(["excluded"] * 50 + ["kept"] * 50, index=clips.global_key)
        self.assertAlmostEqual(kept(clips, sheet, PIXEL_POLICY, design="audit", drawn=drawn)["upper_95"],
                               1 - 0.05 ** (1 / 10), places=6)
        merged = kept(clips, sheet, design="audit", drawn=drawn)
        self.assertEqual((merged["sampled"], merged["upper_95"]), (20, None))
        self.assertIn("several audit strata", merged["upper_95_withheld"])

    def test_a_validation_sample_estimates_the_excluded_share_and_reports_each_group(self):
        # The flags of a validation list are a sample: the excluded share is weighted, not counted on them.
        sheet = self.sheet.assign(stratum=["g1", "g2", "g1", "g2"], inclusion_prob=[0.5, 0.5, 0.1, 0.1])
        report = cs.calibration_report(self.flags, self.reference, sheet, self.answers, PIXEL_POLICY,
                                       flags_are_population=False)
        self.assertAlmostEqual(report["policy"]["excluded_share"], 4 / 24)
        self.assertEqual(report["policy"]["excluded_share_from"], "weighted sample")
        self.assertEqual(sorted(report["by_stratum"]), ["g1", "g2"])
        g1 = report["by_stratum"]["g1"]["measurement"]  # ka flagged and measured, kc missed (weight 10).
        self.assertEqual((g1["tp"], g1["fn"]), (2.0, 10.0))

    def test_gaps_stop_the_report(self):
        # The weights describe the whole sample: a clip of the sheet unscreened, or short of any answer, stops it.
        for field in cs.LABEL_FIELDS:
            unanswered = self.answers.copy()
            unanswered.loc["kd", field] = ""
            with self.assertRaises(SystemExit):
                cs.calibration_report(self.flags, self.reference, self.sheet, unanswered)
        with self.assertRaises(SystemExit):
            cs.calibration_report(self.flags[self.flags.video_path != "d"], self.reference, self.sheet, self.answers)

    def test_a_threshold_sweep_matches_a_rerun_of_flag(self):
        blank = np.zeros((H // cs.CELL, W // cs.CELL), bool)
        box = blank.copy()
        box[4:7, 4:10] = True
        # Three reference clips with a results box and 2 marks inside: never part of a baseline, whatever the
        # thresholds, so the box stays extra in a sweep and in a rerun alike.
        reference = pd.DataFrame([feature_row(blank.copy()) for _ in range(27)]
                                 + [feature_row(box.copy(), inside=2) for _ in range(3)])
        target = pd.DataFrame([dict(feature_row(box.copy()), global_key="target")])
        baselines, _ = cs.layout_baselines(reference)
        lowered = {**cs.THRESHOLDS, "inside_glyphs": 1}
        flagged = cs.apply_flags(target, baselines)
        rerun = cs.apply_flags(target, baselines, lowered)
        self.assertEqual((flagged.extra_cells.iloc[0], rerun.extra_cells.iloc[0]), (18, 18))
        sample = pd.DataFrame({"global_key": ["target"], "video_path": target.video_path, "stratum": ["unflagged"],
                               "inclusion_prob": [1.0]})
        answers = pd.DataFrame([["no", "yes", "no", "no"]], columns=list(cs.LABEL_FIELDS),
                               index=pd.Index(["target"], name="global_key"))
        swept = cs.calibration_report(target, reference, sample, answers, lowered)["measurement"]["at_thresholds"]
        self.assertTrue(rerun.measurement.iloc[0])
        self.assertEqual(swept["tp"], 1.0)

    def test_a_minimum_mark_size_counts_the_marks_of_each_detection(self):
        evidence = {"clip": [[4, "piece"], [30, "line"]], "windows": [[[4, "piece"], [4, "piece"], [13, "caliper"]]]}
        rows = pd.DataFrame({"inside_marks": [json.dumps(evidence), None], "inside_glyphs": [3, np.nan]})
        counts = {px: cs.at_mark_size(rows, px).inside_glyphs.tolist() for px in (3, 5, 14, 31)}
        self.assertEqual(counts[3][0], 3)  # the window: two pieces and a caliper.
        self.assertEqual([counts[px][0] for px in (5, 14, 31)], [1, 1, 0])  # a dotted line weighs its dots.
        self.assertTrue(np.isnan(counts[3][1]))  # an unreadable clip.

    def test_each_mark_size_recomputes_reference_eligibility_and_baselines(self):
        # A layout whose every reference clip has one tiny mark inside: at the smallest size none is eligible, so
        # the layout has no baseline and its clips are incomplete; from 5 px it has one, and the results box shows.
        header = np.zeros((H // cs.CELL, W // cs.CELL), bool)
        header[0, 0:8] = True
        box = header.copy()
        box[4:7, 4:10] = True

        def row(cells, marks, **more):
            return dict(feature_row(cells.copy(), inside=len(marks)),
                        inside_marks=json.dumps({"clip": [[a, "piece"] for a in marks], "windows": []}), **more)

        reference = pd.DataFrame([row(header, [4]) for _ in range(30)])
        clips = pd.DataFrame([row(header, [4, 4], video_path="a", global_key="ka"),  # two tiny marks.
                              row(box, [], video_path="b", global_key="kb"),  # a results box.
                              row(header, [13, 13], video_path="c", global_key="kc"),  # two calipers.
                              row(header, [], video_path="d", global_key="kd")])  # clean, a cursor.
        sheet = pd.DataFrame({"global_key": ["ka", "kb", "kc", "kd"], "video_path": list("abcd"),
                              "stratum": "all", "inclusion_prob": 1.0})
        answers = pd.DataFrame({"still": "no", "measurement": ["no", "yes", "yes", "no"], "other_exclusion": "no",
                                "cursor": ["no", "no", "no", "yes"]},
                               index=pd.Index(sheet.global_key, name="global_key"))
        sweep = cs.calibration_report(clips, reference, sheet, answers, PIXEL_POLICY)["inside_mark_px_sweep"]
        self.assertEqual(sorted(sweep, key=int), [str(px) for px in cs.INSIDE_MARK_SWEEP])
        smallest, five = sweep[str(cs.MIN_INK_PX)], sweep["5"]
        (layout, at_smallest), = smallest["baselines"].items()
        self.assertEqual((at_smallest["baseline"], at_smallest["reference_left_out"], at_smallest["incomplete"]),
                         (False, {"marks_inside": 30}, 4))  # no clip has a results-box score.
        self.assertEqual((five["baselines"][layout]["baseline"], five["baselines"][layout]["reference_used"]),
                         (True, 30))
        policy = {px: [sweep[px]["policy"][k] for k in ("tp", "fp", "fn", "tn")] for px in (str(cs.MIN_INK_PX), "5")}
        self.assertEqual(policy, {str(cs.MIN_INK_PX): [2.0, 2.0, 0.0, 0.0], "5": [2.0, 0.0, 0.0, 2.0]})
        self.assertEqual(five["measurement"]["sensitivity"], 1.0)  # the box, found once a baseline exists.
        self.assertEqual(sweep["16"]["measurement"]["fn"], 1.0)  # the calipers, too small at 16 px.
        # The eligible clip with a cursor: excluded as incomplete at the smallest size only.
        self.assertEqual([sweep[px]["cursor_clips"]["eligible_removed"] for px in (str(cs.MIN_INK_PX), "5")], [1, 0])
        self.assertEqual(sorted(five["by_acquisition"]), ["bmode"])
        # flag at a size gives the sweep's flags.
        flags = cs.screen_flags(clips, reference, {**PIXEL_POLICY, "inside_mark_px": 5})[0]
        self.assertEqual(flags.exclude_reason.tolist(), ["", "measurement", "measurement", ""])

    def test_exact_upper_bound(self):
        self.assertAlmostEqual(cs.upper_bound(0, 100), 1 - 0.05 ** (1 / 100), places=6)
        self.assertAlmostEqual(cs.upper_bound(0, 150), 1 - 0.05 ** (1 / 150), places=6)
        self.assertGreater(cs.upper_bound(2, 50), cs.upper_bound(0, 50))
        self.assertEqual(cs.upper_bound(3, 3), 1.0)
        self.assertIsNone(cs.upper_bound(0, 0))
        # Large audits: about 0.5 + 1.645 * sqrt(0.25 / 2000), without overflowing.
        self.assertAlmostEqual(cs.upper_bound(1000, 2000), 0.5184, places=3)


def word_box(x0, y0, x1, y1):
    """An OCR word's corner points, as RapidOCR gives them."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


class TestSignals(unittest.TestCase):
    def test_each_signal_is_scored_on_its_own_and_beyond_the_header(self):
        lab = pd.DataFrame({
            "measurement_header": ["measure_flag", "none", "none", "none", "unavailable"],
            "analysis_evidence": ["none"] * 5, "header_content": ["usual"] * 4 + ["unavailable"],
            "dicom_object": ["ultrasound_image"] * 5, "still_status": ["moving"] * 5,
            "geometry_evidence": ["caliper_pair", "caliper_pair", "caliper", "none", "trace"],
            "text_evidence": ["none"] * 5, "measurement": [True, False, False, False, None],
            "answer_measurement": ["yes", "yes", "no", "no", "yes"], "answer_still": ["no"] * 5,
            "ineligible": [True, True, False, False, True]})
        report = cs.signal_report(lab)
        signals = report["signals"]
        self.assertEqual({k: signals["header_measure"][k] for k in ("flagged", "true", "positives")},
                         {"flagged": 1, "true": 1, "positives": 3})
        # A pair on both measured clips that have one, whatever the header says of either.
        self.assertEqual((signals["pixel_caliper_pair"]["precision"], signals["pixel_caliper_pair"]["recall"]),
                         (1.0, 2 / 3))
        # Beyond the header, the clip whose header names its measurement is left out on both sides.
        beyond = signals["pixel_caliper_pair"]["beyond_header"]
        self.assertEqual((beyond["flagged"], beyond["true"], beyond["positives"]), (1, 1, 2))
        self.assertEqual(report["positives_no_header_signal_raises"]["measured"], 2)
        self.assertNotIn("beyond_header", signals["header_measure"])
        self.assertEqual((signals["pixel_single_caliper"]["precision"], signals["pixel_trace"]["precision"]),
                         (0.0, 1.0))
        self.assertIsNone(signals["text_measurement"]["precision"])  # raised on no clip.
        self.assertEqual(signals["pixel_marks_score"]["flagged"], 1)  # a missing score raises nothing.

    def test_intervals_are_exact(self):
        self.assertIsNone(cs.interval_95(0, 0))
        low, high = cs.interval_95(0, 24)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 1 - 0.025 ** (1 / 24), places=6)  # 24 false alarms in 24: precision < 14%.
        low, high = cs.interval_95(5, 5)
        self.assertAlmostEqual(low, 0.025 ** (1 / 5), places=6)
        self.assertEqual(high, 1.0)


class TestTextPass(unittest.TestCase):
    def test_words_of_a_row_are_joined_and_a_wide_gap_starts_another_line(self):
        # A settings panel left of a results box on one row, the box's name and number boxed apart, and a second row.
        words = [(word_box(10, 10, 30, 22), "Gn"), (word_box(34, 11, 60, 23), "50"),
                 (word_box(300, 10, 320, 22), "EF"), (word_box(324, 11, 350, 23), "55 %"),
                 (word_box(300, 40, 340, 52), "LVIDd"), (word_box(344, 40, 380, 52), "4.5 cm")]
        lines = cs.text_lines(words)
        self.assertEqual(lines, ["Gn 50", "EF 55 %", "LVIDd 4.5 cm"])
        # Joined into one line, the row would start with a setting and its measurement would be left out.
        status, found = cs.measurement_text([np.zeros((4, 4, 3), np.uint8)], ("stand-in", lambda frame: lines))
        self.assertEqual((status, json.loads(found)), ("checked", ["EF 55", "LVIDd 4.5"]))

    def test_a_name_glued_to_its_number_is_still_read(self):
        # As RapidOCR read drawn text: the space dropped between a name and its number, and in a settings line.
        lines = ["EF55%", "Gn50FR54Hz", "HR72bpm", "Depth16cm", "LAX", "E/e'12"]
        found = cs.measurement_text([np.zeros((4, 4, 3), np.uint8)], ("stand-in", lambda frame: lines))[1]
        self.assertEqual(json.loads(found), ["E/e'12", "EF55"])  # a name followed by a letter is another word.

    def test_text_decides_names_a_scope(self):
        self.assertEqual(cs.THRESHOLDS["text_decides"], "header_unavailable")
        self.assertEqual(cs.parse_thresholds('{"text_decides": "all"}', cs.THRESHOLDS)["text_decides"], "all")
        for value in ("true", "1", '"yes"', "null", '["all"]'):
            with self.assertRaises(SystemExit):
                cs.parse_thresholds(f'{{"text_decides": {value}}}', cs.THRESHOLDS)

    def test_text_needs_the_ocr_environment(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cs, "ocr_version", return_value=None), \
                self.assertRaises(SystemExit) as stop:
            cs.main(["text", "--features", os.path.join(tmp, "f.parquet"),
                     "--out", os.path.join(tmp, "t", "o.parquet")])
        self.assertIn("OCR environment", str(stop.exception))
        self.assertFalse(os.path.exists(os.path.join(tmp, "t")))

    def test_the_worker_reads_three_frames_and_notices_a_replaced_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, junk = os.path.join(tmp, "clip.mp4"), os.path.join(tmp, "junk.mp4")
            write_mp4(path, speckle())
            with open(junk, "wb") as f:
                f.write(b"not a video")
            seen = []
            backend = ("stand-in", lambda frame: seen.append(frame.shape) or ["EDV 120 ml", "HR 72 bpm"])
            with mock.patch.object(cs, "_OCR", backend):
                row = cs._text_worker(("k", path, cs.sha256(path), T))
                changed = cs._text_worker(("k", path, "0" * 64, T))
                failed = cs._text_worker(("j", junk, cs.sha256(junk), T))
        # The first, middle and last of the frames pass 1 read; only the matched expressions are kept.
        self.assertEqual(row, {"global_key": "k", "text_status": "checked", "text_measurements": '["EDV 120"]',
                               "text_frames": 3})
        self.assertEqual(seen, [(H, W, 3)] * 3)
        self.assertEqual(changed["text_status"], "changed")
        self.assertEqual((failed["text_status"], failed["text_measurements"]), ("decode_failed", None))

    def test_text_output_is_the_features_with_text_which_flag_takes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp4")
            write_mp4(path, speckle())
            features, out = os.path.join(tmp, "features.parquet"), os.path.join(tmp, "text", "features_text.parquet")
            row = {**feature_row(np.zeros((H // cs.CELL, W // cs.CELL), bool)), "video_path": path, "global_key": "k",
                   "video_sha256": cs.sha256(path), "n_frames_total": T}
            broken = {**feature_row(np.zeros((H // cs.CELL, W // cs.CELL), bool), error="DECORDError"),
                      "global_key": "b"}
            pd.DataFrame([row, broken]).to_parquet(features)
            cs._write_info(features, {"clips": "the clip list", "extractor": cs.extractor()})

            def in_process(items, workers, log, clip_seconds, threads):
                with mock.patch.object(cs, "_OCR", ("stand-in", lambda frame: ["EF 55 %"])):
                    return [cs._text_worker(item) for item in items]

            # The stand-in engine's version is part of the text code's record, so it is compared under it too.
            with mock.patch.object(cs, "ocr_version", return_value="stand-in 1"), \
                    mock.patch.object(cs, "_map_text", in_process), contextlib.redirect_stdout(io.StringIO()):
                cs.main(["text", "--features", features, "--out", out])
                with self.assertRaises(SystemExit):  # its output is not pass 1 output: text is read once.
                    cs.main(["text", "--features", out, "--out", os.path.join(tmp, "text", "again.parquet")])
                reader = cs.text_reader()
                extracted = cs.extracted_by(out, cs._code_record(*cs.PASS1), cs.Inputs())
            table = pd.read_parquet(out)
            with open(out + ".info.json") as f:
                info = json.load(f)
            self.assertEqual(table.text_status.iloc[0], "checked")
            self.assertTrue(pd.isna(table.text_status.iloc[1]))  # an unreadable clip is not read.
            self.assertEqual(table.text_measurements.iloc[0], '["EF 55"]')
            # The pass 1 record carries over, with what read the text: flag checks both.
            self.assertEqual(info["clips"], "the clip list")
            self.assertEqual((info["extractor"]["text"], extracted["text"]), (reader, reader))
            self.assertEqual(info["text"]["status"], {"checked": 1})
            flags = cs.apply_flags(table, {})  # the clip's vendor writes no GE header: its text decides.
            self.assertEqual((flags.text_evidence.iloc[0], flags.decision.iloc[0]), ("measurement_text", "exclude"))
            # flag refuses such clips' features without the text pass, and takes them with it.
            with mock.patch.object(cs, "ocr_version", return_value="stand-in 1"), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stop:
                    cs.main(["flag", "--features", features, "--reference", features,
                             "--out", os.path.join(tmp, "flags", "flags.parquet")])
                self.assertIn("no text pass", str(stop.exception))
                cs.main(["flag", "--features", out, "--reference", features,
                         "--out", os.path.join(tmp, "flags", "flags.parquet")])
            self.assertEqual(pd.read_parquet(os.path.join(tmp, "flags", "flags.parquet")).decision.iloc[0], "exclude")


def write_mp4(path, frames):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def spawned_worker(parent, timeout=30):
    """The pid of a worker process that `parent` spawned, once it has started."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read()
            except OSError:  # gone meanwhile.
                continue
            if ppid == parent and b"spawn_main" in cmdline:
                return int(pid)
        time.sleep(0.2)
    raise AssertionError("no worker started")


class TestCommandLine(unittest.TestCase):
    @unittest.skipUnless(HAS_DECODER and HAS_PYDICOM and HAS_FFMPEG, "needs decord, pydicom and imageio-ffmpeg")
    def test_features_flag_sheet_calibrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = []
            for i, frozen in enumerate((False, True)):
                path = os.path.join(tmp, f"clip{i}.mp4")
                frames = speckle(seed=i, frozen=frozen)
                frames[T // 2] = frames[T - 1] = frames[0]  # a moving clip whose first, middle and last frames match.
                write_mp4(path, frames)
                clips.append({"video_path": path, "global_key": f"radar-pilot-{i:024x}", "vendor": "v", "model": "m",
                              "acq_category": "bmode", "view": "A4C"})
            for i, clip in enumerate(clips):  # a header with image regions: without one a clip is excluded.
                clip["dicom_path"] = write_dicom(os.path.join(tmp, f"clip{i}.dcm"))
            pd.DataFrame(clips).to_csv(os.path.join(tmp, "clips.csv"), index=False)
            features, flags, sheet = (os.path.join(tmp, n) for n in ("features.parquet", "flags.parquet", "sheet"))
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(["features", "--clips", os.path.join(tmp, "clips.csv"), "--out", features, "--workers", "1"])
            # Reference clips for the layout: 30 like the moving clip, so that its baseline exists.
            reference = os.path.join(tmp, "reference.parquet")
            moving = pd.read_parquet(features).iloc[[0] * 30].reset_index(drop=True)
            moving.assign(video_path=[f"{tmp}/ref{i}.mp4" for i in range(30)]).to_parquet(reference)
            with open(features + ".info.json") as f:
                extracted = json.load(f)["extractor"]
            self.assertEqual(extracted, cs.extractor())
            cs._write_info(reference, {"clips": "reference list", "extractor": extracted})
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(["flag", "--features", features, "--reference", reference, "--out", flags, *NO_TEXT])
                cs.main(["sheet", "--flags", flags, "--out-dir", sheet, "--n-unflagged", "5"])
            out = pd.read_parquet(flags)
            self.assertEqual(out.exclude_reason.tolist(), ["", "still"])
            self.assertEqual(out.global_key.tolist(), [c["global_key"] for c in clips])
            self.assertEqual(out.dicom_status.tolist(), ["ok", "ok"])
            self.assertEqual(out.duplicate_frac.tolist(), [0.0, 1.0])
            self.assertNotIn("cells", out)
            self.assertEqual(os.stat(flags + ".baselines").st_mode & 0o777, 0o700)
            with open(flags + ".info.json") as f:
                info = json.load(f)
            self.assertEqual(info["code_sha256"], {os.path.join("data", "clip_screen.py"): cs.sha256(cs.__file__)})
            self.assertIsNone(info["calibration"])  # the default thresholds are provisional.
            self.assertEqual(info["extractor"], extracted)  # the extractor behind the flags, not the script's now.
            self.assertNotIn("constants", info)
            # The flags name the features files they were made from, whose records name the clip list.
            listed = cs.sha256(os.path.join(tmp, "clips.csv"))
            sources = cs.screen_sources(flags)
            self.assertEqual({role: source["clips"] for role, source in sources.items()},
                             {"features": listed, "reference": "reference list"})
            self.assertEqual(os.stat(sheet).st_mode & 0o777, 0o700)
            self.assertEqual({os.stat(p).st_mode & 0o777 for p in (features, flags, features + ".info.json")}, {0o600})
            sheet_clips = pd.read_csv(os.path.join(sheet, "sheet_clips.csv"))
            self.assertEqual(sorted(sheet_clips.stratum), ["excluded", "kept"])
            with open(os.path.join(sheet, "clip-audit-01.html"), encoding="utf-8") as f:
                page = f.read()
            self.assertNotIn(tmp, page)
            self.assertNotIn("Detector view", page)  # hidden on a sheet calibrate uses.
            # Each clip plays whole on the sheet, in a form browsers play, so the moving one is seen to move.
            from decord import VideoReader
            import imageio_ffmpeg

            self.assertEqual(os.stat(os.path.join(sheet, "clips")).st_mode & 0o777, 0o700)
            for clip in clips:
                copy = os.path.join(sheet, "clips", clip["global_key"] + ".mp4")
                self.assertIn(f"<video src='clips/{clip['global_key']}.mp4'", page)
                self.assertEqual(os.stat(copy).st_mode & 0o777, 0o600)
                reader = imageio_ffmpeg.read_frames(copy)
                meta = next(reader)
                reader.close()
                self.assertEqual((meta["codec"], meta["pix_fmt"].split("(")[0]), ("h264", "yuv420p"))
                played = VideoReader(copy).get_batch(list(range(T))).asnumpy().astype(float)
                self.assertEqual(len(VideoReader(copy)), T)
                moved = np.abs(played[1:] - played[:-1]).mean()
                self.assertTrue(moved > 20 if clip is clips[0] else moved < 2, moved)
            # The still clip is answered on this sheet; the moving one partly here and partly on a later sheet.
            # Each clip on the page names the video bytes its copy was made from; the export writes them.
            videos = dict(zip(sheet_clips.global_key, sheet_clips.video_sha256))
            self.assertEqual(videos, {c["global_key"]: cs.sha256(c["video_path"]) for c in clips})
            for key, video in videos.items():
                self.assertIn(f"<div class=clip id={key} data-video='{video}'>", page)
            self.assertIn('c.dataset.video', cs.SHEET_JS)
            first = write_labels(os.path.join(tmp, "labels.csv"),
                                 [(clips[1]["global_key"], "yes", "no", "no", "no"),
                                  (clips[0]["global_key"], "no", "", "", "")], videos)
            later = write_labels(os.path.join(tmp, "later.csv"), [(clips[0]["global_key"], "", "no", "no", "no")],
                                 videos)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", sheet, "--labels", first, later])
            report = json.loads(buf.getvalue())
            self.assertEqual(report["provenance"]["screen"], info["screen"])  # what CALIBRATION would name.
            self.assertEqual((report["signals"]["labelled"], report["signals"]["positives"]["still"]), (2, 1))
            self.assertEqual(report["still"]["at_thresholds"]["sensitivity"], 1.0)
            self.assertEqual(report["still"]["at_thresholds"]["specificity"], 1.0)
            self.assertEqual(report["labelled"], 2)
            self.assertEqual(report["policy"]["kept"]["sampled"], 1)
            # An audit sample of the flags it was drawn from: its kept clip, answered eligible, bounds the one kept.
            self.assertAlmostEqual(report["policy"]["kept"]["upper_95"], 0.95, places=6)
            self.assertIn("the 1 clips this screen keeps", report["policy"]["kept"]["upper_95_describes"])
            # Evaluated at thresholds that keep the still clip too, the kept clips span both strata: no bound.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", sheet, "--labels", first, later,
                         "--thresholds", json.dumps({"still_scan_motion_frac": 0})])
            merged = json.loads(buf.getvalue())["policy"]["kept"]
            self.assertEqual((merged["sampled"], merged["upper_95"]), (2, None))
            self.assertIn("several audit strata", merged["upper_95_withheld"])
            # A sheet that does not record each screened clip's stratum (an older one): the design is unknown.
            with open(os.path.join(sheet, "sheet.info.json")) as f:
                record = json.load(f)
            with open(os.path.join(sheet, "sheet.info.json"), "w") as f:
                json.dump({k: v for k, v in record.items() if k != "clip_strata"}, f)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", sheet, "--labels", first, later])
            self.assertIn("older sheet", json.loads(buf.getvalue())["policy"]["kept"]["upper_95_withheld"])
            sweeps = report["still"]["sweeps"]
            self.assertEqual(set(sweeps), {c for c, _, _ in cs.STILL_SWEEPS})
            self.assertEqual(sweeps["scan_motion_frac"]["labelled"], 2)  # both clips have a scan area.
            self.assertEqual(sweeps["duplicate_frac"]["1.0"]["sensitivity"], 1.0)
            self.assertEqual(set(report["inside_mark_px_sweep"]), {str(px) for px in cs.INSIDE_MARK_SWEEP})
            # Flags that do not come back when recomputed from their features (edited after flag) are refused.
            edited = pd.read_parquet(flags).assign(exclude_reason="still")
            edited.to_parquet(flags)
            cs._write_info(flags, {k: v for k, v in info.items()
                                   if k not in ("created_utc", "code_commit", "code_sha256", "output_sha256")})
            with self.assertRaises(SystemExit) as stop:
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", sheet, "--labels", first, later])
            self.assertIn("do not come back", str(stop.exception))

    @unittest.skipUnless(HAS_DECODER and HAS_PYDICOM and HAS_FFMPEG, "needs decord, pydicom and imageio-ffmpeg")
    def test_a_clip_replaced_in_place_is_not_taken_for_the_one_screened(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, dicom = os.path.join(tmp, "clip.mp4"), write_dicom(os.path.join(tmp, "clip.dcm"))
            write_mp4(video, speckle())
            pd.DataFrame([{"video_path": video, "dicom_path": dicom, "global_key": "k0", "vendor": "v", "model": "m",
                           "acq_category": "bmode"}]).to_csv(os.path.join(tmp, "clips.csv"), index=False)
            features, reference, flags = (os.path.join(tmp, n) for n in
                                          ("features.parquet", "reference.parquet", "flags.parquet"))

            def screen():
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    cs.main(["features", "--clips", os.path.join(tmp, "clips.csv"), "--out", features,
                             "--workers", "1"])
                    rows = pd.read_parquet(features)
                    rows.iloc[[0] * 30].assign(video_path=[f"{tmp}/ref{i}.mp4" for i in range(30)],
                                               global_key=[f"r{i}" for i in range(30)]).to_parquet(reference)
                    with open(features + ".info.json") as f:
                        cs._write_info(reference, {"clips": "reference list", "extractor": json.load(f)["extractor"]})
                    cs.main(["flag", "--features", features, "--reference", reference, "--out", flags, *NO_TEXT])
                return pd.read_parquet(flags)

            screened = screen()
            self.assertEqual(screened.video_sha256.iloc[0], cs.sha256(video))
            self.assertEqual(json.loads(screened.dicom_regions.iloc[0]), [list(SECTOR_BOX[0])])
            self.assertEqual(cs.changed_media(screened, os.path.join(tmp, "media.log")), [])
            labels = write_labels(os.path.join(tmp, "labels.csv"), [("k0", "no", "no", "no", "no")],
                                  {"k0": cs.sha256(video)})
            sheet = os.path.join(tmp, "sheet")
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(["sheet", "--flags", flags, "--out-dir", sheet, "--n-unflagged", "1"])
            # The moving clip replaced at the same path by a frozen one: no longer the clip screened.
            moving = open(video, "rb").read()
            write_mp4(video, speckle(frozen=True))
            self.assertEqual(cs.changed_media(screened, os.path.join(tmp, "media.log")), ["k0"])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
                cs.main(["sheet", "--flags", flags, "--out-dir", os.path.join(tmp, "sheet_2"), "--n-unflagged", "1"])
            self.assertIn("1 of the sheet's 1 clips no longer hold the video", str(stop.exception))
            # Screened again, the new bytes are the screen's, and the earlier sheet's answers are of other bytes.
            rescreened = screen()
            self.assertEqual(rescreened.exclude_reason.iloc[0], "still")
            # The sheet's population is bound to its media: the flags of the new bytes are not the screen it was drawn
            # from (before the sampled clips' own copies are compared).
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", sheet, "--labels", labels])
            self.assertIn("or other media", str(stop.exception))
            # A new sheet of the new flags: the clip keeps its key, and the earlier answers (of the moving clip)
            # would count as a false positive against the still flag. Only answers of the new bytes count.
            sheet_3 = os.path.join(tmp, "sheet_3")
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(["sheet", "--flags", flags, "--out-dir", sheet_3, "--n-unflagged", "1"])
            with contextlib.redirect_stdout(io.StringIO()) as printed, self.assertRaises(SystemExit) as stop:
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", sheet_3, "--labels", labels])
            self.assertIn("1 of the sheet's 1 clips were answered for other video bytes", str(stop.exception))
            self.assertEqual(printed.getvalue(), "")
            again = write_labels(os.path.join(tmp, "again.csv"), [("k0", "yes", "no", "no", "no")],
                                 {"k0": cs.sha256(video)})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cs.main(["calibrate", "--flags", flags, "--sheet-dir", sheet_3, "--labels", again])
            self.assertEqual(json.loads(buf.getvalue())["still"]["at_thresholds"]["tp"], 1.0)
            # A DICOM header whose regions changed counts as replaced too, and so does one whose evidence changed
            # with the same regions: an ImageType that now names a measurement.
            with open(video, "wb") as f:
                f.write(moving)
            screened = screen()
            write_dicom(dicom, regions=((1, (10, 10, 100, 100)),))
            self.assertEqual(cs.changed_media(screened, os.path.join(tmp, "media.log")), ["k0"])
            recording = ["DERIVED", "PRIMARY", "", "0001"]
            write_dicom(dicom, image_type=recording)
            screened = screen()
            self.assertEqual((screened.dicom_image_content.iloc[0], screened.decision.iloc[0]), ("0001", "keep"))
            self.assertEqual(cs.changed_media(screened, os.path.join(tmp, "media.log")), [])
            write_dicom(dicom, image_type=[*recording, "GEMSMULTIFRAME", "", "", "", "MEASURE", ""])
            self.assertEqual(cs.changed_media(screened, os.path.join(tmp, "media.log")), ["k0"])
            # Rows that do not record the header evidence (an older flag run) cannot be checked: changed.
            self.assertEqual(cs.changed_media(screened.drop(columns=["dicom_ecg"]), os.path.join(tmp, "media.log")),
                             ["k0"])

    @unittest.skipUnless(HAS_DECODER, "needs decord")
    def test_the_console_never_shows_a_clip_path(self):
        @contextlib.contextmanager
        def console():  # both file descriptors, which spawned workers and native code write to.
            with tempfile.TemporaryFile() as out:
                saved = [os.dup(1), os.dup(2)]
                for fd in (1, 2):
                    os.dup2(out.fileno(), fd)
                try:
                    yield out
                finally:
                    for fd, copy in zip((1, 2), saved):
                        os.dup2(copy, fd)
                        os.close(copy)

        with tempfile.TemporaryDirectory() as tmp:
            broken = os.path.join(tmp, "SMITH-JOHN-1234567.mp4")
            with open(broken, "wb") as f:
                f.write(b"not a video" * 100)
            missing = os.path.join(tmp, "DOE-JANE-7654321.mp4")
            features = os.path.join(tmp, "features.parquet")
            for path, stops in ((broken, False), (missing, True)):
                pd.DataFrame([{"video_path": path, "global_key": "k", "vendor": "v", "model": "m",
                               "acq_category": "bmode"}]).to_csv(os.path.join(tmp, "clips.csv"), index=False)
                with console() as out, contextlib.redirect_stdout(io.StringIO()) as printed:
                    try:
                        cs.main(["features", "--clips", os.path.join(tmp, "clips.csv"), "--out", features,
                                 "--workers", "1"])
                        message = ""
                    except SystemExit as stop:
                        message = str(stop)
                    out.seek(0)
                    shown = out.read().decode(errors="replace") + printed.getvalue() + message
                self.assertEqual(bool(message), stops)
                self.assertNotIn(os.path.basename(path)[:-4], shown)
                self.assertEqual(os.stat(features + ".log").st_mode & 0o777, 0o600)
            with open(features + ".log") as f:
                self.assertIn("DOE-JANE-7654321", f.read())  # the private log keeps what is needed to fix it.

    @unittest.skipUnless(HAS_DECODER and os.path.isdir("/proc"), "needs decord and /proc")
    def test_a_worker_that_dies_stops_the_run(self):
        # A worker killed (for memory) or crashing in native code returns nothing: the run stops instead of
        # waiting for its result.
        with tempfile.TemporaryDirectory() as tmp:
            clip = os.path.join(tmp, "clip.mp4")
            os.mkfifo(clip)  # its worker waits on it until it is killed.
            pd.DataFrame([{"video_path": clip, "global_key": "k", "vendor": "v", "model": "m",
                           "acq_category": "bmode"}]).to_csv(os.path.join(tmp, "clips.csv"), index=False)
            features = os.path.join(tmp, "features.parquet")
            run = subprocess.Popen([sys.executable, cs.__file__, "features", "--clips", os.path.join(tmp, "clips.csv"),
                                    "--out", features, "--workers", "1"], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                os.kill(spawned_worker(run.pid), signal.SIGKILL)
                _, shown = run.communicate(timeout=60)
            finally:
                if run.poll() is None:
                    os.killpg(run.pid, signal.SIGKILL)
                    run.communicate()
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("a worker process died", shown)
            self.assertNotIn(clip, shown)
            self.assertFalse(os.path.exists(features) or os.path.exists(features + ".info.json"))

    @unittest.skipUnless(HAS_DECODER and hasattr(signal, "alarm"), "needs decord and SIGALRM")
    def test_a_clip_read_past_the_deadline_stops_the_run(self):
        # A hung decoder: the worker waits on the clip forever; the deadline ends it, and so the run.
        with tempfile.TemporaryDirectory() as tmp:
            clip = os.path.join(tmp, "clip.mp4")
            os.mkfifo(clip)
            pd.DataFrame([{"video_path": clip, "global_key": "k", "vendor": "v", "model": "m",
                           "acq_category": "bmode"}]).to_csv(os.path.join(tmp, "clips.csv"), index=False)
            features = os.path.join(tmp, "features.parquet")
            started = time.monotonic()
            run = subprocess.run([sys.executable, cs.__file__, "features", "--clips", os.path.join(tmp, "clips.csv"),
                                  "--out", features, "--workers", "1", "--clip-timeout", "2"],
                                 capture_output=True, text=True, timeout=120)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("--clip-timeout 2 s", run.stderr)
            self.assertLess(time.monotonic() - started, 60)
            self.assertFalse(os.path.exists(features) or os.path.exists(features + ".info.json"))

    @unittest.skipUnless(HAS_DECODER, "needs decord")
    def test_a_clip_list_changed_while_features_run_completes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, clips_csv = os.path.join(tmp, "clip.mp4"), os.path.join(tmp, "clips.csv")
            write_mp4(video, speckle())
            listed = pd.DataFrame([{"video_path": video, "global_key": "k", "vendor": "v", "model": "m",
                                    "acq_category": "bmode"}])
            listed.to_csv(clips_csv, index=False)
            features = os.path.join(tmp, "features.parquet")
            real = cs.features_table

            def list_replaced_meanwhile(rows):  # a new shortlist written while the workers read the old one.
                listed.assign(dicom_path=os.path.join(tmp, "clip.dcm")).to_csv(clips_csv, index=False)
                return real(rows)

            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop, \
                    mock.patch.object(cs, "features_table", list_replaced_meanwhile):
                cs.main(["features", "--clips", clips_csv, "--out", features, "--workers", "1"])
            self.assertIn("clips.csv changed while this step ran", str(stop.exception))
            self.assertFalse(os.path.exists(features) or os.path.exists(features + ".info.json"))
            # Unchanged, the record names the list as it was read.
            listed.to_csv(clips_csv, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(["features", "--clips", clips_csv, "--out", features, "--workers", "1"])
            with open(features + ".info.json") as f:
                self.assertEqual(json.load(f)["clips"], cs.sha256(clips_csv))

    @unittest.skipUnless(HAS_DECODER, "needs decord")
    def test_a_listed_size_is_kept_apart_from_the_decoded_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, clips_csv = os.path.join(tmp, "clip.mp4"), os.path.join(tmp, "clips.csv")
            write_mp4(video, speckle())
            pd.DataFrame([{"video_path": video, "global_key": "k", "vendor": "v", "model": "m", "acq_category": "bmode",
                           "width": "999", "height": str(H)}]).to_csv(clips_csv, index=False)
            features, flags = os.path.join(tmp, "features.parquet"), os.path.join(tmp, "flags.parquet")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                cs.main(["features", "--clips", clips_csv, "--out", features, "--workers", "1"])
            out = pd.read_parquet(features)
            self.assertEqual(out[["width", "height", "listed_width", "listed_height"]].iloc[0].tolist(),
                             [W, H, "999", str(H)])
            with open(features + ".info.json") as f:
                info = json.load(f)
            self.assertEqual(info["listed_size_differs"], {"width": 1, "height": 0})
            self.assertIn("listed size differs", err.getvalue())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cs.main(["flag", "--features", features, "--reference", features, "--out", flags, *NO_TEXT])
            self.assertEqual(pd.read_parquet(flags).width.iloc[0], W)  # the layout uses the decoded size.
            # A list with both a size and listed_ size leaves no name for one of them: refused.
            pd.DataFrame([{"video_path": video, "global_key": "k", "vendor": "v", "model": "m", "acq_category": "bmode",
                           "width": "1", "listed_width": "1"}]).to_csv(clips_csv, index=False)
            with self.assertRaises(SystemExit):
                cs.main(["features", "--clips", clips_csv, "--out", features, "--workers", "1"])

    def test_a_rerun_that_stops_leaves_no_output_to_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            features, flags = os.path.join(tmp, "features.parquet"), os.path.join(tmp, "flags.parquet")
            earlier = pd.DataFrame([feature_row(np.zeros((H // cs.CELL, W // cs.CELL), bool))])
            earlier.to_parquet(features)
            cs._write_info(features, {})  # an earlier, completed run.
            pd.DataFrame([{"video_path": os.path.join(tmp, "gone.mp4"), "global_key": "k", "vendor": "v",
                           "model": "m", "acq_category": "bmode"}]).to_csv(os.path.join(tmp, "clips.csv"), index=False)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                cs.main(["features", "--clips", os.path.join(tmp, "clips.csv"), "--out", features, "--workers", "1"])
            self.assertFalse(os.path.exists(features) or os.path.exists(features + ".info.json"))
            # Features without the record of a completed run, or changed since it, are refused.
            earlier.to_parquet(features)
            for _ in range(2):
                with self.assertRaises(SystemExit) as stop:
                    cs.main(["flag", "--features", features, "--reference", features, "--out", flags, *NO_TEXT])
                self.assertIn("not the output of a completed `features` run", str(stop.exception))
                cs._write_info(features, {})
                earlier.iloc[:0].to_parquet(features)
            self.assertFalse(os.path.exists(flags))

    def test_outputs_go_to_private_folders_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = os.path.join(tmp, "shared")
            os.mkdir(shared)
            os.chmod(shared, 0o755)  # an existing folder others can read is refused, not changed.
            for argv in (["features", "--clips", "x.csv", "--out", os.path.join(shared, "features.parquet")],
                         ["sheet", "--flags", "x.parquet", "--out-dir", shared]):
                with self.assertRaises(SystemExit):
                    cs.main(argv)
            self.assertEqual(os.stat(shared).st_mode & 0o777, 0o755)
            made = cs.private_dir(os.path.join(tmp, "new", "screen"))
            self.assertEqual(os.stat(made).st_mode & 0o777, 0o700)

    def test_features_need_a_distinct_safe_key_per_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            for keys in (["a", "a"], ["a", "b c"], None):
                clips = pd.DataFrame({"video_path": ["/x/0.mp4", "/x/1.mp4"], "vendor": "v", "model": "m",
                                      "acq_category": "bmode"})
                if keys:
                    clips["global_key"] = keys
                clips.to_csv(os.path.join(tmp, "clips.csv"), index=False)
                with self.assertRaises(SystemExit):
                    cs.main(["features", "--clips", os.path.join(tmp, "clips.csv"),
                             "--out", os.path.join(tmp, "features.parquet"), "--workers", "1"])


class TestOutputs(unittest.TestCase):
    def test_outputs_inside_the_repository_are_refused(self):
        with self.assertRaises(SystemExit):
            cs.main(["flag", "--features", "x.parquet", "--out", os.path.join(REPO_ROOT, "flags.parquet")])


if __name__ == "__main__":
    unittest.main()
