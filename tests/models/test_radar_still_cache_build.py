# tests/models/test_radar_still_cache_build.py

"""The still cache writer with a synthetic encoder and synthetic screens: 2-frame and 16-frame grids from one still
table, the same encoder record as the clip cache needs, refused and unreadable stills listed and left out."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from evals.radar_frozen import build_still_cache as bsc
from src.datasets.radar_stills import same_encoder
from src.datasets.radar_token_cache import TokenCache

DIM = 4
CONFIG = {"encoder": {"checkpoint": None, "module_name": "m", "wrapper_kwargs": None,
                      "pretrain_kwargs": {"encoder": {"patch_size": 112, "tubelet_size": 2}}},
          "data": {"frames_per_clip": 16, "resolution": 224},
          "cache": {"embed_dim": DIM, "dtype": "float32", "use_bfloat16": False}}


class Encoder(torch.nn.Module):
    """One token per tubelet and 112-pixel patch, holding the clip's mean and its number of frames."""

    def forward(self, x, indices):
        clips = x[0][0]
        b, _, t, _, _ = clips.shape
        tokens = (t // 2) * 4
        mean = clips.mean(dim=(1, 2, 3, 4))[:, None, None].expand(b, tokens, DIM).clone()
        mean[..., 1] = float(t)
        return [mean]


def screen(path):
    if "broken" in path:
        raise OSError("no such file " + path)
    image = np.full((80, 120, 3), 90, dtype=np.uint8)
    if "tinted" in path:
        image[:, :60, 0] = 250                               # two colours, no single tint: refused
        image[:, 60:, 1] = 250
    return image


class TestBuildStillCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.checkpoint = os.path.join(root, "encoder.pt")
        with open(self.checkpoint, "wb") as f:
            f.write(b"weights")
        rows = [{"still_id": f"s{n}", "study_id": f"S{n % 2}", "split": "train" if n < 5 else "val",
                 "png_path": f"/x/{name}.png", "spec_x0": 10, "spec_y0": 20, "spec_x1": 110, "spec_y1": 70,
                 "text_boxes": json.dumps([[12, 22, 30, 30]])}
                for n, name in enumerate(["a", "b", "tinted", "broken", "c", "d"])]
        self.stills = os.path.join(root, "stills.csv")
        pd.DataFrame(rows).to_csv(self.stills, index=False)
        self.config = {**CONFIG, "encoder": {**CONFIG["encoder"], "checkpoint": self.checkpoint}}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, frames, name, **kwargs):
        out = os.path.join(self.tmp.name, name)
        summary = bsc.build(self.config, self.stills, out, frames, device="cpu", batch_size=2, num_workers=0,
                            make_encoder=lambda config, device: Encoder(), read=screen, **kwargs)
        return out, summary

    def test_two_and_sixteen_frames_from_one_table(self):
        two, summary = self.build(2, "two")
        sixteen, _ = self.build(16, "sixteen")
        self.assertEqual(summary["failed"], {"refused_by_preprocessing": 1, "unreadable": 1})
        a, b = TokenCache(two), TokenCache(sixteen)
        self.assertEqual((a.meta["token_grid"], b.meta["token_grid"]), ([1, 2, 2], [8, 2, 2]))
        self.assertEqual(sorted(a.files), ["s0", "s1", "s4", "s5"])
        self.assertEqual(sorted(a.files), sorted(b.files))
        self.assertEqual((float(a.load("s0")[0, 1]), float(b.load("s0")[0, 1])), (2.0, 16.0))
        self.assertTrue(same_encoder(a.meta, b.meta))            # one encoder record, whatever the frames
        self.assertEqual(a.meta["provenance"]["data"]["frames"], 2)
        kept = pd.read_csv(os.path.join(two, "stills.csv"))
        self.assertEqual(sorted(kept.still_id), ["s0", "s1", "s4", "s5"])
        failed = pd.read_csv(os.path.join(two, "failed.csv"))
        self.assertEqual(dict(zip(failed.still_id, failed.reason)),
                         {"s2": "refused_by_preprocessing", "s3": "unreadable"})
        self.assertNotIn("/x/", open(os.path.join(two, "failed.csv")).read())

    def test_split_and_refusals(self):
        out, summary = self.build(2, "val", split="val")
        self.assertEqual(sorted(TokenCache(out).files), ["s5"])
        with self.assertRaises(ValueError):
            self.build(4, "four")
        with self.assertRaises(ValueError):
            bsc.main(["--config", "x", "--stills", self.stills, "--frames", "2",
                      "--out", os.path.join(bsc.REPO, "still_cache")])


if __name__ == "__main__":
    unittest.main()
