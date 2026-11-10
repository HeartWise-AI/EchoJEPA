# src/datasets/radar_token_cache.py

"""Cache of frozen-encoder tokens for RADAR: one array per clip, an index and a pinned meta file.

Layout under `root`:

- `meta.json`: `token_grid` [T, H, W], `embed_dim`, `dtype`, `n_clips`, `index_sha256` (the sha256 of `index.csv`)
  and the writer's `provenance` (encoder checkpoint sha256, code commit, decoding and transform settings, clip
  manifest sha256...). Written last, so a cache without it is incomplete. A reader that names the grid, dim or dtype
  it expects refuses a cache that differs.
- `index.csv`: `clip_id`, `file` (relative path), `token_sha256` (sha256 of the token file) and any per-clip fields
  the writer records (for example the video's full-file sha256).
- `tokens/<aa>/<hash>.npy`: the clip's [T * H * W, D] tokens. The file name is a hash of `clip_id`, never a path
  that could name a patient. Each clip is stored once and shared by every structure that uses it.

Identity: `meta.json` identifies the finished cache. Its index digest covers every clip-to-file mapping and every
token file's hash, so a cache whose mappings or tokens differ has another meta (the training fingerprint includes
it). Opening a cache checks the index against that digest (cheap); `verify()` rehashes the token files against the
index (a full read, for example after copying a cache).

The writer converts tokens to the storage dtype first (any torch dtype, bfloat16 included, any device; or a NumPy
array) and refuses values that are not finite after that conversion (a float32 value beyond float16's range
becomes infinite), before writing anything for the clip. The reader returns float32 tensors.
"""

import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
import torch

CACHE_DTYPES = ("float16", "float32")
INDEX_COLUMNS = ("clip_id", "file", "token_sha256")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_name(clip_id):
    digest = hashlib.blake2b(str(clip_id).encode("utf-8"), digest_size=16).hexdigest()
    return os.path.join("tokens", digest[:2], f"{digest}.npy")


class TokenCacheWriter:

    def __init__(self, root, token_grid, embed_dim, dtype="float16", provenance=None):
        if dtype not in CACHE_DTYPES:
            raise ValueError(f"dtype must be one of {CACHE_DTYPES}.")
        if os.path.exists(os.path.join(root, "meta.json")) or os.path.exists(os.path.join(root, "index.csv")):
            raise FileExistsError(f"{root} already holds a token cache; write a new one elsewhere.")
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.token_grid = [int(v) for v in token_grid]
        self.embed_dim = int(embed_dim)
        self.dtype = dtype
        self.provenance = dict(provenance or {})
        self.rows = []
        self.seen = set()

    def add(self, clip_id, tokens, **fields):
        clip_id = str(clip_id)
        if clip_id in self.seen:
            raise ValueError(f"Clip {clip_id!r} is already in the cache.")
        reserved = set(fields) & set(INDEX_COLUMNS)
        if reserved:
            raise ValueError(f"{sorted(reserved)} are written by the cache itself.")
        if torch.is_tensor(tokens):
            tokens = tokens.detach().to(device="cpu", dtype=torch.float32)   # NumPy has no bfloat16
        array = np.asarray(tokens)
        expected = (int(np.prod(self.token_grid)), self.embed_dim)
        if array.shape != expected:
            raise ValueError(f"Expected tokens of shape {expected}, got {array.shape}.")
        with np.errstate(over="ignore", invalid="ignore"):
            stored = array.astype(self.dtype)
        if not np.isfinite(stored).all():
            raise ValueError(f"Non-finite tokens for clip {clip_id!r} once stored as {self.dtype}.")
        buffer = io.BytesIO()
        np.save(buffer, stored)
        content = buffer.getvalue()
        relative = _file_name(clip_id)
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            f.write(content)
        os.replace(path + ".tmp", path)
        self.seen.add(clip_id)
        self.rows.append({"clip_id": clip_id, "file": relative, "token_sha256": hashlib.sha256(content).hexdigest(),
                          **fields})

    def close(self):
        index_path = os.path.join(self.root, "index.csv")
        pd.DataFrame(self.rows, columns=None if self.rows else list(INDEX_COLUMNS)).to_csv(index_path, index=False)
        meta = {"token_grid": self.token_grid, "embed_dim": self.embed_dim, "dtype": self.dtype,
                "n_clips": len(self.rows), "index_sha256": _sha256(index_path), "provenance": self.provenance}
        with open(os.path.join(self.root, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return meta


class TokenCache:

    def __init__(self, root, token_grid=None, embed_dim=None, dtype=None):
        meta_path = os.path.join(root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{root} has no meta.json: not a finished token cache.")
        with open(meta_path) as f:
            self.meta = json.load(f)
        expected = {"token_grid": None if token_grid is None else [int(v) for v in token_grid],
                    "embed_dim": embed_dim, "dtype": dtype}
        for key, value in expected.items():
            if value is not None and self.meta.get(key) != value:
                raise ValueError(f"Token cache {key} is {self.meta.get(key)!r}, expected {value!r}.")
        self.root = root
        index_path = os.path.join(root, "index.csv")
        if "index_sha256" not in self.meta:
            raise ValueError(f"{root}/meta.json has no index digest: rewrite the cache with the current writer.")
        if _sha256(index_path) != self.meta["index_sha256"]:
            raise ValueError(f"{index_path} does not match the digest in meta.json: the cache was changed after it "
                             "was finished.")
        index = pd.read_csv(index_path, dtype={"clip_id": str, "file": str, "token_sha256": str})
        if len(index) != self.meta.get("n_clips"):
            raise ValueError("index.csv and meta.json disagree on the number of clips.")
        self.index = index
        self.files = dict(zip(index.clip_id, index.file))
        self.shape = (int(np.prod(self.meta["token_grid"])), int(self.meta["embed_dim"]))

    def __contains__(self, clip_id):
        return str(clip_id) in self.files

    def __len__(self):
        return len(self.files)

    def verify(self):
        """Rehash every token file against the index; raise ValueError naming the clips that differ."""
        bad = [c for c, f, h in zip(self.index.clip_id, self.index.file, self.index.token_sha256)
               if _sha256(os.path.join(self.root, f)) != h]
        if bad:
            raise ValueError(f"{len(bad)} token files differ from the index (first: {bad[0]!r}).")

    def load(self, clip_id):
        """[T * H * W, D] float32 tokens of one clip."""
        array = np.load(os.path.join(self.root, self.files[str(clip_id)]), mmap_mode="r")
        if array.shape != self.shape:
            raise ValueError(f"Cached tokens of clip {clip_id!r} have shape {array.shape}, expected {self.shape}.")
        return torch.from_numpy(np.array(array, dtype=np.float32))
