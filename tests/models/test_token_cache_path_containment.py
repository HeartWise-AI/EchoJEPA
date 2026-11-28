# tests/models/test_token_cache_path_containment.py

"""Index paths in a token cache must stay inside the cache.

`merge` moves each chunk row's `file` onto the output, and `TokenCache.load` / `verify` open that same column.
A path with parent references, an absolute path, or a symlink that leaves the cache has to be refused before any
file outside the cache is created, removed, or read. The checks below fail until that is true.
"""

import argparse
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from evals.radar_frozen import build_token_cache as btc
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter

GRID, DIM = (2, 2, 2), 4
LENGTH = int(np.prod(GRID))
CONFIG = {"encoder": {"checkpoint": None, "module_name": "m", "pretrain_kwargs": {}},
          "data": {"frames_per_clip": 4},
          "cache": {"token_grid": list(GRID), "embed_dim": DIM, "dtype": "float32", "use_bfloat16": False}}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, path):
    root = os.path.realpath(root)
    path = os.path.realpath(path)
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


class _Reader(torch.utils.data.Dataset):

    def __init__(self, clip_ids, data):
        self.clip_ids = list(clip_ids)

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        return i, torch.full((3, 4, 2, 2), 1.0), torch.arange(4)


class _Encoder(torch.nn.Module):

    def forward(self, x, indices):
        clips = x[0][0]
        return [clips.mean(dim=(1, 2, 3, 4))[:, None, None].expand(-1, LENGTH, DIM)]


class TestTokenCachePathContainment(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.checkpoint = os.path.join(self.root, "encoder.pt")
        with open(self.checkpoint, "wb") as handle:
            handle.write(b"weights")
        self.clips = os.path.join(self.root, "clips.csv")
        pd.DataFrame({"clip_id": ["c-1"], "split": ["train"]}).to_csv(self.clips, index=False)
        self.out = os.path.join(self.root, "sandbox", "cache")
        self.config = {**CONFIG, "encoder": {**CONFIG["encoder"], "checkpoint": self.checkpoint}}

    def tearDown(self):
        self.tmp.cleanup()

    def args(self):
        return argparse.Namespace(clips=self.clips, split="train", out=self.out, exclude=[], chunks=1, worker=0,
                                  workers=1, batch_size=1, num_workers=0, device="cpu")

    def encode(self):
        btc.encode(self.config, self.args(), make_encoder=lambda config, device: _Encoder(), make_reader=_Reader)
        return btc.chunk_dir(self.out, 0, 1)

    def retarget(self, folder, relative, payload):
        """Point the finished chunk's one row at `relative` and leave `payload` at the path merge reads."""
        index_path = os.path.join(folder, "index.csv")
        index = pd.read_csv(index_path, dtype=str)
        self.assertEqual(len(index), 1)
        source = os.path.join(folder, relative)
        os.makedirs(os.path.dirname(source), exist_ok=True)
        with open(source, "wb") as handle:
            handle.write(payload)
        index.loc[0, "file"] = relative
        index.to_csv(index_path, index=False)
        self.assertEqual(pd.read_csv(index_path, dtype=str).loc[0, "file"], relative)
        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path) as handle:
            meta = json.load(handle)
        meta["index_sha256"] = _sha256(index_path)
        with open(meta_path, "w") as handle:
            json.dump(meta, handle)
        return source

    def call(self, fn):
        try:
            fn()
        except Exception as exc:
            return exc
        return None

    def finish_cache(self, root, rows):
        os.makedirs(root, exist_ok=True)
        index_path = os.path.join(root, "index.csv")
        pd.DataFrame(rows).to_csv(index_path, index=False)
        meta = {"token_grid": list(GRID), "embed_dim": DIM, "dtype": "float32", "n_clips": len(rows),
                "index_sha256": _sha256(index_path), "provenance": {}}
        with open(os.path.join(root, "meta.json"), "w") as handle:
            json.dump(meta, handle)
        return index_path

    def test_merge_parent_path_does_not_create_or_move_outside(self):
        folder = self.encode()
        relative = os.path.join("..", "..", "victim", "payload.npy")
        source = self.retarget(folder, relative, b"PAYLOAD")
        dest = os.path.join(self.out, relative)
        self.assertTrue(_inside(self.out, source))
        self.assertFalse(_inside(self.out, dest))
        self.assertFalse(os.path.exists(os.path.dirname(dest)))
        exc = self.call(lambda: btc.merge(self.config, self.args()))
        self.assertFalse(os.path.exists(os.path.dirname(dest)))
        with open(source, "rb") as handle:
            self.assertEqual(handle.read(), b"PAYLOAD")
        self.assertIsInstance(exc, ValueError)

    def test_merge_parent_path_does_not_overwrite_outside_file(self):
        folder = self.encode()
        relative = os.path.join("..", "..", "victim", "payload.npy")
        source = self.retarget(folder, relative, b"PAYLOAD")
        dest = os.path.join(self.out, relative)
        os.makedirs(os.path.dirname(dest))
        with open(dest, "wb") as handle:
            handle.write(b"ORIGINAL")
        exc = self.call(lambda: btc.merge(self.config, self.args()))
        with open(dest, "rb") as handle:
            self.assertEqual(handle.read(), b"ORIGINAL")
        with open(source, "rb") as handle:
            self.assertEqual(handle.read(), b"PAYLOAD")
        self.assertIsInstance(exc, ValueError)

    def test_merge_tokens_prefix_does_not_escape_through_parent_references(self):
        folder = self.encode()
        relative = os.path.join("tokens", "..", "..", "victim", "payload.npy")
        source = self.retarget(folder, relative, b"PAYLOAD")
        dest = os.path.join(self.out, relative)
        self.assertTrue(relative.startswith("tokens" + os.sep))
        self.assertTrue(_inside(self.out, source))
        self.assertFalse(_inside(self.out, dest))
        exc = self.call(lambda: btc.merge(self.config, self.args()))
        self.assertFalse(os.path.exists(dest))
        with open(source, "rb") as handle:
            self.assertEqual(handle.read(), b"PAYLOAD")
        self.assertIsInstance(exc, ValueError)

    def test_merge_absolute_path_does_not_create_directories(self):
        folder = self.encode()
        dest = os.path.join(self.root, "created", "nested", "payload.npy")
        index_path = os.path.join(folder, "index.csv")
        index = pd.read_csv(index_path, dtype=str)
        index.loc[0, "file"] = dest
        index.to_csv(index_path, index=False)
        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path) as handle:
            meta = json.load(handle)
        meta["index_sha256"] = _sha256(index_path)
        with open(meta_path, "w") as handle:
            json.dump(meta, handle)
        self.assertTrue(os.path.isabs(pd.read_csv(index_path, dtype=str).loc[0, "file"]))
        self.assertFalse(os.path.exists(os.path.dirname(dest)))
        exc = self.call(lambda: btc.merge(self.config, self.args()))
        self.assertFalse(os.path.exists(os.path.dirname(dest)))
        self.assertFalse(os.path.exists(os.path.join(self.out, "meta.json")))
        self.assertIsInstance(exc, ValueError)

    def test_merge_does_not_follow_output_directory_symlink(self):
        folder = self.encode()
        index = pd.read_csv(os.path.join(folder, "index.csv"), dtype=str)
        relative = index.loc[0, "file"]
        source = os.path.join(folder, relative)
        with open(source, "rb") as handle:
            payload = handle.read()
        outside = os.path.join(self.root, "outside-home")
        os.makedirs(outside)
        os.symlink(outside, os.path.join(self.out, "tokens"))
        exc = self.call(lambda: btc.merge(self.config, self.args()))
        self.assertEqual(os.listdir(outside), [])
        self.assertTrue(os.path.isfile(source))
        with open(source, "rb") as handle:
            self.assertEqual(handle.read(), payload)
        self.assertIsInstance(exc, ValueError)

    def test_merge_does_not_move_a_file_through_a_chunk_symlink(self):
        folder = self.encode()
        index = pd.read_csv(os.path.join(folder, "index.csv"), dtype=str)
        relative = index.loc[0, "file"]
        tokens_dir = os.path.join(folder, "tokens")
        os.rename(tokens_dir, tokens_dir + "-real")
        outside = os.path.join(self.root, "outside-home")
        rest = relative.split(os.sep, 1)[1]
        secret = os.path.join(outside, rest)
        os.makedirs(os.path.dirname(secret))
        with open(secret, "wb") as handle:
            handle.write(b"KEEP")
        os.symlink(outside, tokens_dir)
        self.assertFalse(_inside(folder, os.path.join(folder, relative)))
        exc = self.call(lambda: btc.merge(self.config, self.args()))
        self.assertTrue(os.path.isfile(secret))
        with open(secret, "rb") as handle:
            self.assertEqual(handle.read(), b"KEEP")
        self.assertIsInstance(exc, ValueError)

    def test_open_rejects_parent_directory_path(self):
        cache_root = os.path.join(self.root, "cache")
        secret = os.path.join(self.root, "secret.npy")
        array = np.arange(LENGTH * DIM, dtype=np.float32).reshape(LENGTH, DIM)
        np.save(secret, array)
        relative = os.path.join("..", "secret.npy")
        self.finish_cache(cache_root, [{"clip_id": "c-1", "file": relative, "token_sha256": _sha256(secret)}])
        self.assertFalse(_inside(cache_root, os.path.join(cache_root, relative)))
        with self.assertRaises(ValueError):
            TokenCache(cache_root, token_grid=GRID, embed_dim=DIM, dtype="float32")

    def test_open_rejects_absolute_path(self):
        cache_root = os.path.join(self.root, "cache")
        secret = os.path.join(self.root, "secret.npy")
        array = np.arange(LENGTH * DIM, dtype=np.float32).reshape(LENGTH, DIM)
        np.save(secret, array)
        self.finish_cache(cache_root, [{"clip_id": "c-1", "file": secret, "token_sha256": _sha256(secret)}])
        with self.assertRaises(ValueError):
            TokenCache(cache_root, token_grid=GRID, embed_dim=DIM, dtype="float32")

    def test_open_rejects_symlink_that_leaves_the_cache(self):
        cache_root = os.path.join(self.root, "cache")
        outside = os.path.join(self.root, "outside")
        os.makedirs(outside)
        array = np.arange(LENGTH * DIM, dtype=np.float32).reshape(LENGTH, DIM)
        secret = os.path.join(outside, "secret.npy")
        np.save(secret, array)
        os.makedirs(os.path.join(cache_root, "tokens"))
        os.symlink(outside, os.path.join(cache_root, "tokens", "aa"))
        relative = os.path.join("tokens", "aa", "secret.npy")
        self.finish_cache(cache_root, [{"clip_id": "c-1", "file": relative, "token_sha256": _sha256(secret)}])
        self.assertFalse(_inside(cache_root, os.path.join(cache_root, relative)))
        with self.assertRaises(ValueError):
            TokenCache(cache_root, token_grid=GRID, embed_dim=DIM, dtype="float32")

    def test_load_does_not_follow_a_symlink_added_after_open(self):
        cache_root = os.path.join(self.root, "cache")
        writer = TokenCacheWriter(cache_root, GRID, DIM, dtype="float32")
        writer.add("c-1", torch.full((LENGTH, DIM), 1.0))
        writer.close()
        cache = TokenCache(cache_root, token_grid=GRID, embed_dim=DIM, dtype="float32")
        original = cache.load("c-1").clone()
        relative = cache.files["c-1"]
        directory = os.path.join(cache_root, os.path.dirname(relative))
        outside = os.path.join(self.root, "outside")
        os.makedirs(outside)
        secret = np.full((LENGTH, DIM), 7.0, dtype=np.float32)
        np.save(os.path.join(outside, os.path.basename(relative)), secret)
        os.rename(directory, directory + "-real")
        os.symlink(outside, directory)
        try:
            got = cache.load("c-1")
        except ValueError:
            got = None
        self.assertTrue(got is None or torch.equal(got, original))

    def test_verify_does_not_read_a_symlink_added_after_open(self):
        cache_root = os.path.join(self.root, "cache")
        writer = TokenCacheWriter(cache_root, GRID, DIM, dtype="float32")
        writer.add("c-1", torch.full((LENGTH, DIM), 1.0))
        writer.close()
        cache = TokenCache(cache_root, token_grid=GRID, embed_dim=DIM, dtype="float32")
        relative = cache.files["c-1"]
        directory = os.path.join(cache_root, os.path.dirname(relative))
        outside = os.path.join(self.root, "outside")
        os.makedirs(outside)
        with open(os.path.join(outside, os.path.basename(relative)), "wb") as handle:
            handle.write(b"not-the-token-file")
        os.rename(directory, directory + "-real")
        os.symlink(outside, directory)
        try:
            cache.verify()
        except ValueError as exc:
            self.assertNotIn("token files differ", str(exc))
