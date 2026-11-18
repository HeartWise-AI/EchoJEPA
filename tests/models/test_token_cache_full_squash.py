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


if __name__ == "__main__":
    unittest.main()
