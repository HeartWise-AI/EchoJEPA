# tests/models/test_view_classifier_arm_b_input.py

"""Arm B's selectable input (`arm_b.data.preprocessing` / `frame_read`) on CPU with synthetic videos: the defaults are
the first runs' input, `full_squash` with `frame_read: echoprime` is exactly `preprocess_variants`' `full_squash`
variant and `full_squash_aa` exactly its `full_squash_bilinear_aa` variant, padding included; the training crop keeps
the padding at 0."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from evals.view_classifier import mvit
from evals.view_classifier import preprocess_variants as pv


def write_video(path, t, h=120, w=160, seed=0):
    """A synthetic MJPG video of `t` frames: a flickering disc and a static block; returns the RGB frames written."""
    import cv2
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 < (h // 3) ** 2
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        frames[i][disc] = np.clip(np.asarray((180, 90, 60))[None] + rng.integers(-20, 21, size=(int(disc.sum()), 1)),
                                  0, 255)
    frames[:, 4:14, w - 24:w - 4] = 230
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for frame in frames:
        writer.write(frame[..., ::-1].copy())
    writer.release()
    return frames


class TestSettings(unittest.TestCase):

    def test_defaults_and_errors(self):
        self.assertEqual(mvit.data_settings(None), ("centre_zoom", "cache"))
        self.assertEqual(mvit.data_settings({"zoom": 0.1}), ("centre_zoom", "cache"))
        self.assertEqual(mvit.data_settings({"preprocessing": "full_squash", "frame_read": "echoprime"}),
                         ("full_squash", "echoprime"))
        with self.assertRaises(ValueError):
            mvit.data_settings({"preprocessing": "squash"})
        with self.assertRaises(ValueError):
            mvit.data_settings({"frame_read": "middle"})

    def test_spatial(self):
        frames = np.random.default_rng(0).integers(0, 256, (3, 40, 60, 3), dtype=np.uint8)
        self.assertTrue(torch.equal(mvit.spatial(frames, 32, 0.1, "centre_zoom"), mvit.crop_and_scale(frames, 32, 0.1)))
        squashed = mvit.spatial(frames, 32, 0.1, "full_squash")
        self.assertEqual(tuple(squashed.shape), (3, 3, 32, 32))
        self.assertTrue(np.array_equal(squashed.permute(1, 2, 3, 0).numpy(), pv.squash(frames, 32)))

    def test_model_input_zeroes_trailing_padding_only(self):
        x = torch.randint(1, 255, (2, 3, 16, 8, 8), dtype=torch.uint8)
        x[1, :, 10:] = 0                     # padding of a 10-frame clip
        x[1, :, 4] = 0                       # a black frame inside the clip stays as it is
        out = mvit.model_input(x, "rgb", zero_padding=True)
        ref = pv.normalize_padded(x, [16, 10], "rgb")
        torch.testing.assert_close(out, ref)
        self.assertLess(float(out[1, :, 4].max()), 0.0)
        torch.testing.assert_close(mvit.model_input(x, "bgr", zero_padding=False), mvit.normalize(x, "bgr"))
        self.assertEqual(mvit.trailing_black(x)[1].tolist(), [False] * 10 + [True] * 6)


class TestDecoding(unittest.TestCase):

    def test_default_read_is_unchanged(self):
        from src.datasets.video_study_dataset import make_clip_loader
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.avi")
            write_video(path, 24)
            got = mvit.ClipDecoder([path], 16, 2, 64, 0.1)[0][1]
            buffer, _ = make_clip_loader(16, 2, random_clip_sampling=False).loadvideo_decord(path, 16)
            self.assertTrue(torch.equal(got, mvit.crop_and_scale(buffer[:16], 64, 0.1)))
            data, row, failed, _ = mvit.decode_all([path], {"frames_per_clip": 16, "frame_step": 2,
                                                            "resolution": 64, "zoom": 0.1}, workers=0)
            self.assertTrue(torch.equal(data[row[path]], got))
            self.assertEqual(failed, [])

    def test_full_squash_echoprime_is_the_variant(self):
        """Arm B's s2 input equals `preprocess_variants`' `full_squash` (frames, resize and padding), for a clip
        shorter and one longer than the 32-frame window."""
        with tempfile.TemporaryDirectory() as tmp:
            data_cfg = {"frames_per_clip": 16, "frame_step": 2, "resolution": 64, "preprocessing": "full_squash",
                        "frame_read": "echoprime"}
            for t in (20, 45):
                path = os.path.join(tmp, f"clip{t}.avi")
                write_video(path, t, seed=t)
                clip = mvit.ClipDecoder([path], 16, 2, 64, preprocessing="full_squash", frame_read="echoprime")[0][1]
                frames = pv.decode_frames(path)
                variants, real, _ = pv.make_variants(frames, None, None, ("full_squash",), 64)
                self.assertTrue(torch.equal(clip, variants["full_squash"]), t)
                for order in ("rgb", "bgr"):
                    torch.testing.assert_close(mvit.model_input(clip[None], order, zero_padding=True),
                                               pv.normalize_padded(variants["full_squash"][None],
                                                                   [real["full_squash"]], order))
                data, row, failed, _ = mvit.decode_all([path], data_cfg, workers=0)
                self.assertTrue(torch.equal(data[row[path]], clip))
            self.assertIsNone(mvit.echoprime_read(os.path.join(tmp, "missing.avi")))
            self.assertIsNone(mvit.ClipDecoder([os.path.join(tmp, "missing.avi")], frame_read="echoprime")[0][1])

    def test_full_squash_aa_echoprime_is_the_variant(self):
        """`full_squash_aa` with `frame_read: echoprime` equals `preprocess_variants`' `full_squash_bilinear_aa`
        (frames, antialiased resize and padding), byte for byte, for a clip shorter and one longer than the 32-frame
        window, and differs from the bicubic `full_squash`."""
        with tempfile.TemporaryDirectory() as tmp:
            data_cfg = {"frames_per_clip": 16, "frame_step": 2, "resolution": 64, "preprocessing": "full_squash_aa",
                        "frame_read": "echoprime"}
            for t in (20, 45):
                path = os.path.join(tmp, f"clip{t}.avi")
                write_video(path, t, seed=t)
                clip = mvit.ClipDecoder([path], 16, 2, 64, preprocessing="full_squash_aa",
                                        frame_read="echoprime")[0][1]
                frames = pv.decode_frames(path)
                variants, real, fell_back = pv.make_variants(frames, None, None,
                                                             ("full_squash_bilinear_aa", "full_squash"), 64)
                self.assertEqual(fell_back, [])
                ref = variants["full_squash_bilinear_aa"]
                self.assertEqual(clip.dtype, ref.dtype)
                self.assertTrue(torch.equal(clip, ref), t)
                self.assertFalse(torch.equal(clip, variants["full_squash"]))
                n_real = real["full_squash_bilinear_aa"]
                self.assertEqual(n_real, min(16, (t + 1) // 2))
                self.assertEqual(int(clip[:, n_real:].abs().sum()), 0)            # padding frames are black
                for order in ("rgb", "bgr"):
                    self.assertTrue(torch.equal(mvit.model_input(clip[None], order, zero_padding=True),
                                                pv.normalize_padded(ref[None], [n_real], order)))
                data, row, failed, _ = mvit.decode_all([path], data_cfg, workers=0)
                self.assertTrue(torch.equal(data[row[path]], clip))
                self.assertEqual(failed, [])

    def test_full_squash_aa_spatial(self):
        frames = np.random.default_rng(1).integers(0, 256, (3, 434, 636, 3), dtype=np.uint8)
        got = mvit.spatial(frames, 224, 0.1, "full_squash_aa")
        self.assertEqual((tuple(got.shape), got.dtype), ((3, 3, 224, 224), torch.uint8))
        self.assertTrue(np.array_equal(got.permute(1, 2, 3, 0).numpy(), pv.squash_bilinear_aa(frames, 224)))
        self.assertTrue(torch.equal(mvit.spatial(torch.as_tensor(frames), 224, 0.1, "full_squash_aa"), got))
        # antialiased: fine noise is averaged, not sampled (much lower spread than the bicubic squash)
        self.assertLess(float(got.float().std()), 0.6 * float(mvit.full_squash(frames, 224).float().std()))
        self.assertEqual(mvit.data_settings({"preprocessing": "full_squash_aa", "frame_read": "echoprime"}),
                         ("full_squash_aa", "echoprime"))

    def test_arm_prepare_uses_the_padding_rule(self):
        cfg = {"channel_order": "rgb", "crop_scale_min": 1.0,
               "data": {"preprocessing": "full_squash", "frame_read": "echoprime"}}
        x = torch.randint(1, 255, (2, 3, 16, 8, 8), dtype=torch.uint8)
        x[0, :, 12:] = 0
        with mock.patch.object(mvit, "load_original", lambda *a, **k: mvit.build_mvit(20)):
            arm = mvit.make_arm(x, {"a": 0, "b": 1}, cfg, "unused")
            torch.testing.assert_close(arm.prepare(x, False, None), pv.normalize_padded(x, [12, 16], "rgb"))
            arm = mvit.make_arm(x, {"a": 0, "b": 1}, {**cfg, "data": {}}, "unused")
            torch.testing.assert_close(arm.prepare(x, False, None), mvit.normalize(x, "rgb"))
            aa = {**cfg, "data": {"preprocessing": "full_squash_aa", "frame_read": "echoprime"}}
            arm = mvit.make_arm(x, {"a": 0, "b": 1}, aa, "unused")
            torch.testing.assert_close(arm.prepare(x, False, None), pv.normalize_padded(x, [12, 16], "rgb"))

    def test_training_crop_on_the_aa_input(self):
        """The training augmentation (random square crop of 85-100%, resized back) acts on the normalised model input
        with the padding already zeroed: shape kept, padding frames stay 0, one crop per clip; evaluation is uncropped
        and a scale of 1 is the identity."""
        cfg = {"channel_order": "rgb", "crop_scale_min": 0.85,
               "data": {"preprocessing": "full_squash_aa", "frame_read": "echoprime"}}
        x = torch.randint(1, 255, (2, 3, 16, 32, 32), dtype=torch.uint8)
        x[0, :, 10:] = 0
        with mock.patch.object(mvit, "load_original", lambda *a, **k: mvit.build_mvit(20)):
            arm = mvit.make_arm(x, {"a": 0, "b": 1}, cfg, "unused")
            ref = pv.normalize_padded(x, [10, 16], "rgb")
            torch.testing.assert_close(arm.prepare(x, False, None), ref)
            out = arm.prepare(x, True, torch.Generator().manual_seed(0))
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(float(out[0, :, 10:].abs().max()), 0.0)
            self.assertFalse(torch.equal(out, ref))
        torch.testing.assert_close(mvit.random_crop(ref, 1.0, None), ref)


if __name__ == "__main__":
    unittest.main()
