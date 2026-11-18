# tests/models/test_token_cache_full_squash.py

"""The token cache reader's `data.spatial` on CPU with a synthetic video: the default is the probe's eval transform
(resize + centre crop) unchanged, `full_squash` encodes the whole frame with the same resize and normalisation, and the
choice is recorded in the provenance."""

import argparse
import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from evals.radar_frozen import build_token_cache as btc

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def write_video(path, t=20, h=120, w=200):
    """A synthetic MJPG video: grey ramp, a bright block in the left and right margins (cut by a centre crop)."""
    import cv2
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    frames[:] = np.linspace(20, 120, w, dtype=np.uint8)[None, None, :, None]
    frames[:, 50:70, 2:14] = 250
    frames[:, 50:70, w - 14:w - 2] = 250
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for frame in frames:
        writer.write(frame[..., ::-1].copy())
    writer.release()


class TestSpatial(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = os.path.join(self.tmp.name, "a.avi")
        write_video(self.video)
        self.data = {"frames_per_clip": 16, "frame_step": 2, "resolution": 64}

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_is_the_eval_transform(self):
        from evals.video_classification_frozen.utils import make_transforms
        from src.datasets.video_study_dataset import make_clip_loader
        i, clip, indices = btc.ClipReader([self.video], self.data)[0]
        buffer, _ = make_clip_loader(16, 2, random_clip_sampling=False).loadvideo_decord(self.video, 16)
        ref = make_transforms(training=False, crop_size=64, num_views_per_clip=1, normalize=IMAGENET)(buffer[:16])[0]
        self.assertEqual(tuple(clip.shape), (3, 16, 64, 64))
        torch.testing.assert_close(clip, ref)
        self.assertEqual(len(indices), 16)

    def test_full_squash_keeps_the_whole_frame(self):
        import cv2
        from src.datasets.video_study_dataset import make_clip_loader
        reader = btc.ClipReader([self.video], {**self.data, "spatial": "full_squash"})
        _, clip, _ = reader[0]
        self.assertEqual(tuple(clip.shape), (3, 16, 64, 64))
        buffer, _ = make_clip_loader(16, 2, random_clip_sampling=False).loadvideo_decord(self.video, 16)
        ref = np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_LINEAR) for f in buffer[:16]])
        ref = torch.as_tensor(ref).permute(3, 0, 1, 2).float() / 255
        mean, std = (torch.tensor(v).view(3, 1, 1, 1) for v in IMAGENET)
        torch.testing.assert_close(clip, (ref - mean) / std, atol=1e-5, rtol=1e-5)
        bright = (250 / 255 - IMAGENET[0][0]) / IMAGENET[1][0] - 0.3
        self.assertGreater(float(clip[0, 0, 28:34, :3].max()), bright)       # both margins are encoded
        self.assertGreater(float(clip[0, 0, 28:34, -3:].max()), bright)
        _, cropped, _ = btc.ClipReader([self.video], self.data)[0]
        self.assertLess(float(cropped[0, 0, 28:34].max()), bright)           # the centre crop cuts them
        pickle.dumps(reader)                                                   # spawn workers pickle the reader

    def test_provenance_and_errors(self):
        self.assertEqual(btc.spatial_transform({}), "eval")
        self.assertEqual(btc.spatial_transform({"spatial": "resize_centre_crop"}), "eval")
        self.assertEqual(btc.spatial_transform({"spatial": "full_squash"}), "eval_full_squash")
        with self.assertRaises(ValueError):
            btc.spatial_transform({"spatial": "pad"})
        checkpoint, clips = (os.path.join(self.tmp.name, n) for n in ("encoder.pt", "clips.csv"))
        for path, text in ((checkpoint, "w"), (clips, "clip_id,split\nx,train\n")):
            with open(path, "w") as f:
                f.write(text)
        config = {"encoder": {"checkpoint": checkpoint, "module_name": "m"}, "data": dict(self.data),
                  "cache": {"use_bfloat16": True}}
        args = argparse.Namespace(clips=clips, split="train", exclude=[], chunks=4)
        default = btc.provenance(config, args, 1, 0)
        self.assertEqual(default["data"]["transform"], "eval")
        self.assertEqual(sorted(default["data"]), ["clip_sampling", "frame_step", "frames_per_clip", "normalization",
                                                   "resolution", "transform"])          # the keys caches already hold
        config["data"]["spatial"] = "full_squash"
        self.assertEqual(btc.provenance(config, args, 1, 0)["data"]["transform"], "eval_full_squash")
        config["data"]["resize"] = "bilinear"
        self.assertEqual(btc.provenance(config, args, 1, 0)["data"]["transform"], "eval_full_squash")
        config["data"]["resize"] = "bicubic_aa"
        aa = btc.provenance(config, args, 1, 0)
        self.assertEqual(aa["data"]["transform"], "eval_full_squash_bicubic_aa")
        self.assertEqual(sorted(aa["data"]), sorted(default["data"]))                   # no new key
        self.assertEqual(btc.spatial_transform({"spatial": "full_squash", "resize": "bilinear_aa"}),
                         "eval_full_squash_bilinear_aa")
        with self.assertRaises(ValueError):
            btc.spatial_transform({"spatial": "full_squash", "resize": "area"})
        with self.assertRaises(ValueError):                                              # crop path: resize not offered
            btc.spatial_transform({"resize": "bicubic_aa"})
        self.assertEqual(btc.spatial_transform({"resize": "bilinear"}), "eval")

    def test_default_squash_resize_is_cv2_linear_without_antialiasing(self):
        """The default full_squash resize (every existing whole-frame cache) is cv2 INTER_LINEAR: on a frame of
        1-pixel stripes downscaled 2.8x it aliases (stripes survive), where the antialiased resizes average them."""
        import cv2
        frame = np.zeros((1, 434, 636, 3), dtype=np.uint8)
        frame[:, :, 1::2] = 255
        default = np.asarray(btc.full_squash_transform(224, IMAGENET).transforms[0](frame))
        self.assertTrue(np.array_equal(default[0], cv2.resize(frame[0], (224, 224), interpolation=cv2.INTER_LINEAR)))
        self.assertGreater(float(default.std()), 50)
        for resize in ("bicubic_aa", "bilinear_aa"):
            out = np.asarray(btc.full_squash_transform(224, IMAGENET, resize).transforms[0](frame))
            self.assertEqual((out.shape, out.dtype), ((1, 224, 224, 3), np.uint8))
            self.assertLess(float(out.std()), 10)
            self.assertLess(abs(float(out.mean()) - 127.5), 3)

    def test_antialiased_squash_reader(self):
        """`resize: bicubic_aa`: the whole frame, F.interpolate(bicubic, antialias) on the uint8 frames, rounded, then
        [0, 1] and normalised; padding frames stay black; the reader pickles (spawn workers)."""
        from src.datasets.video_study_dataset import make_clip_loader
        reader = btc.ClipReader([self.video], {**self.data, "spatial": "full_squash", "resize": "bicubic_aa"})
        _, clip, indices = reader[0]
        self.assertEqual(tuple(clip.shape), (3, 16, 64, 64))
        buffer, _ = make_clip_loader(16, 2, random_clip_sampling=False).loadvideo_decord(self.video, 16)
        x = torch.as_tensor(np.asarray(buffer[:16])).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode="bicubic", align_corners=False, antialias=True)
        ref = x.round().clamp(0, 255).permute(1, 0, 2, 3) / 255
        mean, std = (torch.tensor(v).view(3, 1, 1, 1) for v in IMAGENET)
        torch.testing.assert_close(clip, (ref - mean) / std, atol=1e-5, rtol=1e-5)
        _, plain, _ = btc.ClipReader([self.video], {**self.data, "spatial": "full_squash"})[0]
        self.assertFalse(torch.equal(clip, plain))
        self.assertEqual(len(indices), 16)
        pickle.dumps(reader)
        with self.assertRaises(ValueError):
            btc.AntialiasedResize(64, "area")


if __name__ == "__main__":
    unittest.main()
