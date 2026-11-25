# tests/models/test_view_classifier_variants.py

"""The preprocessing variants of the unchanged view classifier on CPU with synthetic frames: EchoPrime's frame
selection, the geometry of each variant, EchoPrime's vendored masking (loaded without its module code, colours kept),
the extractor's mask_and_crop, padding after normalisation, the summary, and one run end to end on synthetic videos
with a stub model."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import mvit
from evals.view_classifier import preprocess_variants as pv


def moving_disc(t=31, h=120, w=160, colour=(200, 60, 40), seed=0, static_block=True):
    """uint8 [t, h, w, 3] RGB: a disc whose pixels flicker around `colour`, on black, plus a static bright block
    (like burned-in text) in the top-right corner."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 < (h // 3) ** 2
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        noise = rng.integers(-20, 21, size=(h, w, 1))
        frames[i][disc] = np.clip(np.asarray(colour)[None] + noise[disc], 0, 255)
    if static_block:
        frames[:, 4:14, w - 24:w - 4] = 230
    return frames


class TestFramesAndGeometry(unittest.TestCase):

    def test_echoprime_frames(self):
        for t, n in ((40, 16), (32, 16), (31, 16), (20, 10), (3, 2)):
            frames = np.arange(t)[:, None, None, None] * np.ones((1, 2, 2, 3), dtype=np.int64)
            picked, n_real = pv.echoprime_frames(frames)
            self.assertEqual(n_real, n)
            self.assertEqual(list(picked[:, 0, 0, 0]), list(range(0, min(t, 32), 2)))

    def test_full_pad_keeps_the_corners_and_the_centre_crop_does_not(self):
        frames = np.zeros((2, 60, 100, 3), dtype=np.uint8)
        frames[:, :8, :8] = 255                              # top-left corner marker
        full = pv.resize_square(pv.pad_to_square(frames), 64)
        self.assertEqual(full.shape, (2, 64, 64, 3))
        self.assertGreater(full[:, :16, :16].max(), 200)
        centre = mvit.crop_and_scale(frames, 64, zoom=0.1)
        self.assertEqual(int(centre.max()), 0)

    def test_squash_keeps_everything(self):
        frames = np.zeros((1, 60, 100, 3), dtype=np.uint8)
        frames[:, :, :5] = 255
        frames[:, :, -5:] = 255
        out = pv.squash(frames, 32)
        self.assertEqual(out.shape, (1, 32, 32, 3))
        self.assertGreater(out[0, :, 0].min(), 150)
        self.assertGreater(out[0, :, -1].min(), 150)

    def test_region_box_and_square(self):
        regions = [{"RegionSpatialFormat": 1, "RegionLocationMinX0": 10, "RegionLocationMinY0": 0,
                    "RegionLocationMaxX1": 89, "RegionLocationMaxY1": 49},
                   {"RegionSpatialFormat": 3, "RegionLocationMinX0": 0, "RegionLocationMinY0": 50,
                    "RegionLocationMaxX1": 99, "RegionLocationMaxY1": 59}]   # a Doppler strip: not 2D
        self.assertEqual(pv.region_box(regions, 60, 100), (10, 0, 90, 50))
        self.assertIsNone(pv.region_box(regions[1:], 60, 100))
        self.assertIsNone(pv.region_box([], 60, 100))
        frames = np.full((2, 60, 100, 3), 100, dtype=np.uint8)
        box = (10, 0, 90, 50)
        keep = np.zeros((60, 100), dtype=bool)
        keep[0:50, 10:90] = True
        out = pv.square_around(frames, box, keep)
        self.assertEqual(out.shape, (2, 80, 80, 3))          # side = max(80, 50), zero padding above and below
        self.assertEqual(int(out[:, :15].max()), 0)
        self.assertEqual(int(out[:, 15:65].min()), 100)
        self.assertEqual(int(out[:, 65:].max()), 0)

    def test_mask_and_crop_drops_static_pixels_and_squares_the_moving_region(self):
        frames = moving_disc()
        out = pv.mask_and_crop(frames)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[1], out.shape[2])
        self.assertLess(abs(out.shape[1] - 2 * (120 // 3)), 8)       # about the disc's diameter
        self.assertNotIn(230, np.unique(out))                         # the static block is gone
        self.assertFalse(pv.motion_mask(moving_disc())[4:14, 136:156].any())
        self.assertIsNone(pv.mask_and_crop(np.zeros((5, 20, 20, 3), dtype=np.uint8)))


class TestEchoPrime(unittest.TestCase):

    def test_functions_load_without_the_module(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)                                  # the module would open assets/ relative to here
            try:
                fns = pv.echoprime_functions()
            finally:
                os.chdir(cwd)
        self.assertEqual(set(fns), {"mask_outside_ultrasound", "crop_and_scale"})
        self.assertEqual(fns["crop_and_scale"](np.zeros((60, 100, 3), dtype=np.uint8)).shape, (224, 224, 3))

    def test_masking_keeps_colours_in_rgb_and_removes_static_text(self):
        frames = moving_disc(colour=(200, 60, 40))
        out, n_real = pv.echoprime_preprocess(frames, 64)
        self.assertEqual((out.shape, n_real), ((16, 64, 64, 3), 16))
        centre = out[:, 24:40, 24:40].reshape(-1, 3).mean(axis=0)
        self.assertGreater(centre[0], centre[2] + 80)               # red stays in channel 0
        np.testing.assert_allclose(centre, (200, 60, 40), atol=25)
        import cv2
        yuv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YUV) for f in frames])
        masked = pv.echoprime_functions()["mask_outside_ultrasound"](yuv)
        self.assertEqual(int(masked[:, 4:14, 136:156].max()), 0)     # the static corner block is masked
        self.assertGreater(int(masked[:, 55:65, 75:85].max()), 100)  # the disc is kept


class TestVariantsAndScoring(unittest.TestCase):

    def test_make_variants_shapes_padding_and_fallbacks(self):
        frames = moving_disc(t=20)
        current = torch.full((3, 16, 64, 64), 7, dtype=torch.uint8)
        clips, real, fell_back = pv.make_variants(frames, current, None, pv.FIRST_ROUND, 64)
        self.assertEqual(set(clips), set(pv.FIRST_ROUND))
        for name, clip in clips.items():
            self.assertEqual(tuple(clip.shape), (3, 16, 64, 64), name)
            self.assertEqual(clip.dtype, torch.uint8)
        self.assertTrue(torch.equal(clips["current"], current))
        self.assertEqual(real["current"], 16)
        self.assertEqual(real["full_pad"], 10)
        self.assertEqual(int(clips["full_pad"][:, 10:].max()), 0)
        self.assertEqual(fell_back, ["region_square"])             # no regions given
        with self.assertRaises(ValueError):
            pv.make_variants(frames, None, None, ("current",), 64)

    def test_normalize_padded(self):
        x = torch.randint(0, 255, (2, 3, 16, 8, 8), dtype=torch.uint8)
        out = pv.normalize_padded(x, [16, 10], "bgr")
        ref = mvit.normalize(x, "bgr")
        torch.testing.assert_close(out[0], ref[0])
        torch.testing.assert_close(out[1, :, :10], ref[1, :, :10])
        self.assertEqual(float(out[1, :, 10:].abs().max()), 0.0)

    def test_summarize(self):
        classes = list(vc.OLD_CLASSES)
        p = np.zeros((4, 20))
        for i, c in enumerate(["A4C", "OTHER", "PSAX", "A2C"]):
            p[i, classes.index(c)] = 1.0
        table = pd.DataFrame(p, columns=[f"p_{c}" for c in classes])
        table["predicted"] = ["A4C", "OTHER", "PSAX", "A2C"]
        archive = np.zeros((4, 20))
        for i, c in enumerate(["A4C", "PSAX", "PSAX", "A3C"]):
            archive[i, classes.index(c)] = 1.0
        row = pv.summarize(table, ["A4C", "PSAX", "PSAX", "A3C"], archive, ["bmode", "bmode", "color_flow", "bmode"],
                           np.array(["A4C", None, "PSAX", "A2C"], dtype=object))
        self.assertAlmostEqual(row["agreement_with_archive"], 0.5)
        self.assertAlmostEqual(row["agreement_bmode"], 1 / 3)
        self.assertAlmostEqual(row["agreement_color_flow"], 1.0)
        self.assertAlmostEqual(row["predicted_other_share"], 0.25)
        self.assertAlmostEqual(row["mean_tv_to_archive"], 0.5)
        self.assertEqual(row["n_human"], 3)
        self.assertAlmostEqual(row["human_accuracy_old_classes"], 1.0)


class TestSecondRound(unittest.TestCase):
    """The second round's frame selections, resizes and inputs; the first round's variants are unchanged by it."""

    def test_frame_selections(self):
        long, short = np.arange(50)[:, None, None, None], np.arange(10)[:, None, None, None]
        picked, n = pv.middle_frames(long)
        self.assertEqual(n, 16)
        self.assertEqual(picked[:, 0, 0, 0].tolist(), list(range(9, 41, 2)))
        picked, n = pv.middle_frames(short)
        self.assertEqual((picked[:, 0, 0, 0].tolist(), n), (list(range(0, 10, 2)), 5))
        picked, n = pv.period1_frames(long)
        self.assertEqual((picked[:, 0, 0, 0].tolist(), n), (list(range(16)), 16))
        self.assertEqual(pv.period1_frames(short)[1], 10)
        picked, n = pv.spread_frames(long)
        self.assertEqual((n, int(picked[0, 0, 0, 0]), int(picked[-1, 0, 0, 0])), (16, 0, 49))
        self.assertEqual(pv.spread_frames(short)[0][:, 0, 0, 0].tolist(), list(range(10)))

    def test_resizes_keep_the_whole_frame(self):
        frames = moving_disc(t=4)
        for fn in (pv.squash, pv.squash_bilinear_aa, pv.squash_cv2):
            out = fn(frames, 64)
            self.assertEqual(out.shape, (4, 64, 64, 3), fn.__name__)
            self.assertGreater(int(out[:, 1:4, 55:62].max()), 150, fn.__name__)    # the corner block survives
        self.assertFalse(np.array_equal(pv.squash(frames, 64), pv.squash_bilinear_aa(frames, 64)))

    def test_make_variants_second_round(self):
        frames = moving_disc(t=20)
        current = torch.full((3, 16, 64, 64), 7, dtype=torch.uint8)
        cache_squash = torch.full((3, 16, 64, 64), 9, dtype=torch.uint8)
        first, _, _ = pv.make_variants(frames, current, None, pv.FIRST_ROUND, 64)
        clips, real, fell_back = pv.make_variants(frames, current, None, pv.VARIANTS, 64, cache_squash, frames.copy())
        self.assertEqual(set(clips), set(pv.VARIANTS))
        for name in pv.FIRST_ROUND:                                     # the first round is unchanged
            self.assertTrue(torch.equal(clips[name], first[name]), name)
        for name, clip in clips.items():
            self.assertEqual(tuple(clip.shape), (3, 16, 64, 64), name)
        self.assertTrue(torch.equal(clips["full_squash_cacheframes"], cache_squash))
        self.assertEqual(real["full_squash_cacheframes"], 16)
        # black padding instead of zeros after normalisation: same pixels, every frame counted as real
        self.assertTrue(torch.equal(clips["full_squash_blackpad"], clips["full_squash"]))
        self.assertEqual((real["full_squash"], real["full_squash_blackpad"]), (10, 16))
        self.assertEqual((real["full_squash_period1"], real["full_squash_spread"], real["full_squash_middle"]),
                         (16, 16, 10))
        self.assertTrue(torch.equal(clips["full_squash_middle"], clips["full_squash"]))   # 20 frames: same window
        self.assertTrue(torch.equal(clips["full_squash_cv2decode"], clips["full_squash"]))  # identical decoded frames
        self.assertEqual(fell_back, ["region_square"])
        _, _, fell_back = pv.make_variants(frames, current, None, ("full_squash_cv2decode",), 64, None, None)
        self.assertEqual(fell_back, ["full_squash_cv2decode"])
        with self.assertRaises(ValueError):
            pv.make_variants(frames, current, None, ("full_squash_cacheframes",), 64)


class StubModel(nn.Module):
    """Logits from the clip's mean intensity: deterministic, no weights needed."""

    def forward(self, x):
        m = x.mean(dim=(1, 2, 3, 4))
        return torch.stack([m * (i - 10) for i in range(len(vc.OLD_CLASSES))], dim=1)


class TestRunEndToEnd(unittest.TestCase):

    def test_run_on_synthetic_videos(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for k in range(3):
                path = os.path.join(tmp, f"clip{k}.avi")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
                for frame in moving_disc(t=24 + 10 * k, seed=k):
                    writer.write(frame[..., ::-1].copy())
                writer.release()
                rows.append({"global_key": f"g{k}", "patient_id": str(k), "video_path": path,
                             "dicom_path": os.path.join(tmp, "missing.dcm"), "view": ["A4C", "PSAX", "OTHER"][k],
                             "acq_category": ["bmode", "color_flow", "bmode"][k]})
            pool = pd.DataFrame(rows)
            pool.to_csv(os.path.join(tmp, "pool_clips.csv"), index=False)
            probs = pd.DataFrame(np.full((3, 20), 0.05), columns=[f"p_{c}" for c in vc.OLD_CLASSES])
            probs.insert(0, "global_key", pool.global_key)
            probs.to_csv(os.path.join(tmp, "pool_old_probs.csv"), index=False)
            cfg = {"paths": {"pool_clips": os.path.join(tmp, "pool_clips.csv"), "inputs": tmp,
                             "runs": tmp, "view_classifier": "unused", "folds": os.path.join(tmp, "none.csv")},
                   "arm_b": {"data": {"frames_per_clip": 16, "frame_step": 2, "resolution": 64, "zoom": 0.1}},
                   "classes": {c: vc.PARENT[c] for c in vc.NEW_CLASSES}}
            with mock.patch.object(mvit, "load_original", lambda *a, **k: StubModel()):
                summary = pv.run(cfg, torch.device("cpu"), workers=0, batch=2)
            out = os.path.join(tmp, "arm_b", "zeroshot_variants")
            self.assertEqual(len(summary), len(pv.FIRST_ROUND) * len(pv.ORDERS))
            self.assertTrue((summary.n == 3).all())
            self.assertTrue((summary.loc[summary.variant == "region_square", "fallbacks"] == 3).all())
            table = pd.read_csv(os.path.join(out, "probs_full_pad_rgb.csv"))
            self.assertEqual(len(table), 3)
            np.testing.assert_allclose(table[[f"p_{c}" for c in vc.OLD_CLASSES]].sum(axis=1), 1.0, rtol=1e-5)
            self.assertTrue(os.path.exists(os.path.join(out, "variants.json")))
            self.assertIn("agreement_160x120", summary.columns)
            with self.assertRaises(FileExistsError):                  # scored results are never replaced
                pv.run(cfg, torch.device("cpu"), workers=0, batch=2)
            second = os.path.join(tmp, "second")
            variants = ("full_squash",) + pv.SECOND_ROUND
            with mock.patch.object(mvit, "load_original", lambda *a, **k: StubModel()):
                summary2 = pv.run(cfg, torch.device("cpu"), workers=0, variants=variants, out=second, batch=2)
            self.assertEqual(len(summary2), len(variants) * len(pv.ORDERS))
            self.assertTrue((summary2.n == 3).all())
            self.assertTrue((summary2.fallbacks == 0).all())            # OpenCV decoded every clip
            a = pd.read_csv(os.path.join(out, "probs_full_squash_rgb.csv"))
            b = pd.read_csv(os.path.join(second, "probs_full_squash_rgb.csv"))
            pd.testing.assert_frame_equal(a, b)                        # the anchor reproduces the first round


if __name__ == "__main__":
    unittest.main()
