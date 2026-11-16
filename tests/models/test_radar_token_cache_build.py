# tests/models/test_radar_token_cache_build.py

"""The token cache builder with a synthetic encoder and reader: chunks, rerun, merge, unreadable and excluded clips."""

import argparse
import os
import tempfile
import unittest

import pandas as pd
import torch

from evals.radar_frozen import build_token_cache as btc
from src.datasets.radar_token_cache import TokenCache

GRID, DIM = (2, 2, 2), 4
CONFIG = {"encoder": {"checkpoint": None, "module_name": "m", "pretrain_kwargs": {}},
          "data": {"frames_per_clip": 4},
          "cache": {"token_grid": list(GRID), "embed_dim": DIM, "dtype": "float32", "use_bfloat16": False}}


def value(clip_id):
    return float(int(clip_id.split("-")[1]))


class Reader(torch.utils.data.Dataset):
    """Clip `c-<n>` decodes to a clip filled with n; ids ending in `x` cannot be decoded."""

    def __init__(self, clip_ids, data):
        self.clip_ids = list(clip_ids)

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        clip_id = self.clip_ids[i]
        if clip_id.endswith("x"):
            return i, None, None
        return i, torch.full((3, 4, 2, 2), value(clip_id)), torch.arange(4)


class Encoder(torch.nn.Module):
    calls = 0

    def forward(self, x, indices):
        Encoder.calls += 1
        clips = x[0][0]
        return [clips.mean(dim=(1, 2, 3, 4))[:, None, None].expand(-1, 8, DIM)]


class TestBuildTokenCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.checkpoint = os.path.join(root, "encoder.pt")
        with open(self.checkpoint, "wb") as f:
            f.write(b"weights")
        ids = [f"c-{n}" for n in range(40)] + ["c-40-x", "c-41"]
        table = pd.DataFrame({"study_id": [f"s{n % 5}" for n in range(len(ids))], "clip_id": ids, "view": "A4C",
                              "acquisition": "bmode", "split": ["train"] * 41 + ["val"]})
        self.clips = os.path.join(root, "clips.csv")
        table.to_csv(self.clips, index=False)
        self.exclude = os.path.join(root, "exclude.csv")
        pd.DataFrame({"avi_path": ["c-3", "not-in-table"]}).to_csv(self.exclude, index=False)
        self.out = os.path.join(root, "cache")
        self.config = {**CONFIG, "encoder": {**CONFIG["encoder"], "checkpoint": self.checkpoint}}

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **changes):
        base = dict(clips=self.clips, split="train", out=self.out, exclude=[self.exclude], chunks=6, worker=0,
                    workers=1, batch_size=4, num_workers=0, device="cpu")
        return argparse.Namespace(**{**base, **changes})

    def encode(self, **changes):
        btc.encode(self.config, self.args(**changes), make_encoder=lambda config, device: Encoder(),
                   make_reader=Reader)

    def test_two_workers_then_merge(self):
        self.encode(worker=0, workers=2)
        with self.assertRaises(ValueError):
            btc.merge(self.config, self.args())            # the other worker's chunks are missing
        self.encode(worker=1, workers=2)
        Encoder.calls = 0
        self.encode(worker=0, workers=2)                   # a rerun finds its chunks finished
        self.assertEqual(Encoder.calls, 0)
        btc.merge(self.config, self.args())
        cache = TokenCache(self.out, token_grid=GRID, embed_dim=DIM, dtype="float32")
        cache.verify()
        kept = [f"c-{n}" for n in range(40) if n != 3]
        self.assertEqual(sorted(cache.files), sorted(kept))
        for clip_id in ("c-0", "c-17", "c-39"):
            self.assertTrue(torch.equal(cache.load(clip_id), torch.full((8, DIM), value(clip_id))))
        self.assertEqual(cache.meta["provenance"]["unreadable"], 1)
        self.assertEqual(cache.meta["provenance"]["excluded"], 1)
        table = pd.read_csv(os.path.join(self.out, "clips.csv"), dtype=str)
        self.assertEqual(sorted(table.clip_id), sorted(kept))
        self.assertEqual(set(table.split), {"train"})
        self.assertEqual(list(pd.read_csv(os.path.join(self.out, "failed.csv")).clip_id), ["c-40-x"])
        self.assertFalse(os.path.exists(os.path.join(self.out, "chunks")))
        with self.assertRaises(FileExistsError):
            btc.merge(self.config, self.args())

    def test_unfinished_chunk_is_written_again(self):
        self.encode()
        folder = btc.chunk_dir(self.out, 2, 6)
        os.remove(os.path.join(folder, "meta.json"))
        Encoder.calls = 0
        self.encode()
        self.assertGreater(Encoder.calls, 0)
        btc.merge(self.config, self.args())
        TokenCache(self.out).verify()

    def test_chunks_of_other_settings_do_not_merge(self):
        self.encode()
        with open(self.checkpoint, "wb") as f:
            f.write(b"other weights")
        with self.assertRaises(ValueError):
            btc.merge(self.config, self.args())

    def test_refuses_an_output_folder_in_the_repository(self):
        with self.assertRaises(ValueError):
            btc.main(["merge", "--config", "x", "--clips", self.clips, "--out", os.path.join(btc.REPO, "cache")])

    def test_chunk_does_not_depend_on_the_table(self):
        self.assertEqual(btc.chunk_of("c-1", 64), btc.chunk_of("c-1", 64))
        self.assertGreater(len({btc.chunk_of(f"c-{n}", 8) for n in range(200)}), 6)


if __name__ == "__main__":
    unittest.main()
