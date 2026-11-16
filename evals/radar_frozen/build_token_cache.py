# evals/radar_frozen/build_token_cache.py

"""Write the token cache RADAR trains on: the frozen encoder's tokens of every clip of one clip table.

Each clip is read as the study probe reads it at evaluation: the first `frames_per_clip` frames at `frame_step`
(a short video is padded with black frames), resized and centre-cropped to `resolution`, no augmentation. The encoder
is the probe's (`evals.video_classification_frozen.models.init_module`), run without gradients. One clip gives
`prod(token_grid)` tokens, stored by `src.datasets.radar_token_cache`.

Two stages, so that several GPUs can share the work and an interrupted run loses little:

- `encode`: the clips are cut into `--chunks` chunks by a hash of the clip id (the table's row order does not matter).
  A process writes the chunks whose number is `--worker` modulo `--workers`, each as a small finished cache under
  `<out>/chunks/`. A finished chunk is skipped on a rerun; an unfinished one is removed and written again. A clip
  that cannot be decoded is listed in the chunk's `failed.csv` and has no tokens.
- `merge`: when every chunk is finished and all were written with the same settings, the token files are moved into
  `<out>/tokens/` and one index and `meta.json` are written. `<out>/clips.csv` is the clip table without the excluded
  and unreadable clips: the table to build the captions from and to train on (`data.clips`), since training refuses a
  clip that has no tokens. `<out>/failed.csv` lists the unreadable clips.

`--exclude` names CSV files with a clip path column (`avi_path`, `video_path` or `clip_id`); their clips are left out.

`<out>` must be outside the repository: clip ids are file paths. Nothing printed names a clip or a study.

Config (YAML): encoder {checkpoint, module_name, pretrain_kwargs, wrapper_kwargs}; data {frames_per_clip, frame_step,
resolution, normalization}; cache {token_grid, embed_dim, dtype, use_bfloat16}. `--checkpoint` overrides
encoder.checkpoint.

Usage (one process per GPU, then once):
    CUDA_VISIBLE_DEVICES=0 python -m evals.radar_frozen.build_token_cache encode --config <yaml> --clips <table> \\
        --split train --out <folder> --worker 0 --workers 4
    python -m evals.radar_frozen.build_token_cache merge --config <yaml> --clips <table> --split train --out <folder>
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
import yaml

from src.datasets.radar_study_dataset import file_sha256, read_table
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
EXCLUDE_COLUMNS = ("avi_path", "video_path", "clip_id")
CACHE_VERSION = 1    # raise when a change here alters the tokens a given config writes


def chunk_of(clip_id, chunks):
    """The chunk a clip belongs to: a hash of its id, so it does not depend on the table."""
    return int.from_bytes(hashlib.blake2b(str(clip_id).encode("utf-8"), digest_size=8).digest(), "big") % chunks


def read_clips(path, split, exclude=()):
    """(clip rows of the split without the excluded clips, number excluded). Refuses a clip listed twice."""
    clips = read_table(path)
    if "clip_id" not in clips.columns:
        raise ValueError("The clip table needs a clip_id column.")
    if "split" in clips.columns:
        clips = clips[clips.split.astype(str) == split]
    if clips.clip_id.duplicated().any():
        raise ValueError("A clip appears twice in the clip table.")
    excluded = set()
    for name in exclude:
        table = pd.read_csv(name, dtype=str, keep_default_na=False)
        column = next((c for c in EXCLUDE_COLUMNS if c in table.columns), None)
        if column is None:
            raise ValueError(f"An exclusion file needs one of the columns {EXCLUDE_COLUMNS}.")
        excluded |= set(table[column])
    keep = ~clips.clip_id.isin(excluded)
    return clips[keep].reset_index(drop=True), int((~keep).sum())


def _commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def provenance(config, args, n_clips, n_excluded):
    """What fixes the tokens: every chunk of one cache must carry the same record."""
    enc, data = config["encoder"], config.get("data", {})
    record = {
        "cache_version": CACHE_VERSION,
        "encoder_sha256": file_sha256(enc["checkpoint"]),
        "encoder": {k: enc.get(k) for k in ("module_name", "pretrain_kwargs", "wrapper_kwargs")},
        "data": {"frames_per_clip": int(data.get("frames_per_clip", 16)), "frame_step": data.get("frame_step", 2),
                 "resolution": int(data.get("resolution", 224)),
                 "normalization": data.get("normalization") or DEFAULT_NORMALIZATION,
                 "clip_sampling": "first_window", "transform": "eval"},
        "use_bfloat16": bool(config["cache"].get("use_bfloat16", True)),
        "clips_sha256": file_sha256(args.clips), "split": args.split,
        "exclude_sha256": sorted(file_sha256(e) for e in args.exclude),
        "clips": n_clips, "excluded": n_excluded, "chunks": args.chunks,
        "builder_sha256": file_sha256(os.path.abspath(__file__)), "commit": _commit(),
    }
    return json.loads(json.dumps(record))    # as meta.json gives it back (tuples become lists)


class ClipReader(torch.utils.data.Dataset):
    """Item i: (i, clip [C, T, H, W] or None when the video cannot be decoded, frame indices [T])."""

    def __init__(self, clip_ids, data):
        from evals.video_classification_frozen.utils import make_transforms
        from src.datasets.video_study_dataset import make_clip_loader
        self.clip_ids = list(clip_ids)
        self.frames = int(data.get("frames_per_clip", 16))
        self.loader = make_clip_loader(self.frames, data.get("frame_step", 2), random_clip_sampling=False)
        self.transform = make_transforms(training=False, crop_size=int(data.get("resolution", 224)),
                                         num_views_per_clip=1,
                                         normalize=data.get("normalization") or DEFAULT_NORMALIZATION)

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        buffer, indices = self.loader.loadvideo_decord(self.clip_ids[i], self.frames)
        if buffer is None or len(buffer) == 0:
            return i, None, None
        return i, self.transform(buffer[:self.frames])[0], torch.as_tensor(np.asarray(indices[0], dtype=np.int64))


def _collate(items):
    good = [item for item in items if item[1] is not None]
    failed = [item[0] for item in items if item[1] is None]
    if not good:
        return [], None, None, failed
    return ([g[0] for g in good], torch.stack([g[1] for g in good]), torch.stack([g[2] for g in good]), failed)


def build_encoder(config, device):
    from evals.video_classification_frozen.models import init_module
    enc, data = config["encoder"], config.get("data", {})
    encoder = init_module(module_name=enc["module_name"], frames_per_clip=int(data.get("frames_per_clip", 16)),
                          resolution=int(data.get("resolution", 224)), checkpoint=enc["checkpoint"],
                          model_kwargs=enc["pretrain_kwargs"], wrapper_kwargs=enc.get("wrapper_kwargs"),
                          device=device)
    return encoder.eval()


@torch.no_grad()
def encode_batch(encoder, clips, indices, device, use_bfloat16):
    """Tokens [B, L, D] of clips [B, C, T, H, W], as the study probe's `encode` computes them for one slot."""
    clips, indices = clips.to(device, non_blocking=True), indices.to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
        return encoder([[clips]], [indices])[0]


def chunk_dir(out, chunk, chunks):
    return os.path.join(out, "chunks", f"{chunk:04d}-of-{chunks:04d}")


def write_chunk(folder, clip_ids, loader, encoder, device, cache, record):
    """One finished chunk cache in `folder`; returns (clips written, clips that could not be decoded)."""
    if os.path.exists(folder):
        shutil.rmtree(folder)            # an unfinished chunk: a finished one is never passed here
    writer = TokenCacheWriter(folder, cache["token_grid"], cache["embed_dim"], cache.get("dtype", "float16"),
                              provenance=record)
    failed = []
    for positions, clips, indices, bad in loader:
        failed += [clip_ids[i] for i in bad]
        if not positions:
            continue
        tokens = encode_batch(encoder, clips, indices, device, bool(cache.get("use_bfloat16", True)))
        for i, clip_tokens in zip(positions, tokens):
            writer.add(clip_ids[i], clip_tokens)
    pd.DataFrame({"clip_id": failed}).to_csv(os.path.join(folder, "failed.csv"), index=False)
    writer.close()
    return len(writer.rows), len(failed)


def encode(config, args, make_encoder=build_encoder, make_reader=ClipReader):
    clips, n_excluded = read_clips(args.clips, args.split, args.exclude)
    record = provenance(config, args, len(clips), n_excluded)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    by_chunk = {}
    for clip_id in clips.clip_id:
        by_chunk.setdefault(chunk_of(clip_id, args.chunks), []).append(clip_id)
    mine = [c for c in range(args.chunks) if c % args.workers == args.worker]
    todo = [c for c in mine if not os.path.exists(os.path.join(chunk_dir(args.out, c, args.chunks), "meta.json"))]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} chunks, {len(mine) - len(todo)} already finished",
          flush=True)
    if not todo:
        return
    encoder = make_encoder(config, device)
    written = unreadable = 0
    for n, chunk in enumerate(todo, 1):
        ids = sorted(by_chunk.get(chunk, []))
        loader = torch.utils.data.DataLoader(
            make_reader(ids, config.get("data", {})), batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=_collate, shuffle=False,
            multiprocessing_context="spawn" if args.num_workers else None)
        done, bad = write_chunk(chunk_dir(args.out, chunk, args.chunks), ids, loader, encoder, device,
                                config["cache"], record)
        written, unreadable = written + done, unreadable + bad
        print(f"worker {args.worker}: chunk {n}/{len(todo)} done, {written} clips written, {unreadable} unreadable",
              flush=True)


def merge(config, args):
    clips, n_excluded = read_clips(args.clips, args.split, args.exclude)
    record = provenance(config, args, len(clips), n_excluded)
    cache = config["cache"]
    if os.path.exists(os.path.join(args.out, "meta.json")):
        raise FileExistsError(f"{args.out} is already a finished cache.")
    folders = [chunk_dir(args.out, c, args.chunks) for c in range(args.chunks)]
    missing = sum(not os.path.exists(os.path.join(f, "meta.json")) for f in folders)
    if missing:
        raise ValueError(f"{missing} of {args.chunks} chunks are not finished: run `encode` for every worker first.")
    parts = [TokenCache(f, token_grid=cache["token_grid"], embed_dim=cache["embed_dim"],
                        dtype=cache.get("dtype", "float16")) for f in folders]
    if any(p.meta["provenance"] != record for p in parts):
        raise ValueError("A chunk was written with other settings, clips or code than this call names: the chunks do "
                         "not form one cache. Remove the chunks folder and encode again.")
    failed = pd.concat([pd.read_csv(os.path.join(f, "failed.csv"), dtype=str, keep_default_na=False)
                        for f in folders], ignore_index=True)
    index = pd.concat([p.index for p in parts], ignore_index=True)
    seen = set(index.clip_id) | set(failed.clip_id)
    if len(index) + len(failed) != len(clips) or seen != set(clips.clip_id):
        raise ValueError("The chunks do not hold exactly the clips of the table.")
    writer = TokenCacheWriter(args.out, cache["token_grid"], cache["embed_dim"], cache.get("dtype", "float16"),
                              provenance={**record, "unreadable": len(failed)})
    for part, folder in zip(parts, folders):
        for row in part.index.to_dict("records"):
            target = os.path.join(args.out, row["file"])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(os.path.join(folder, row["file"]), target)
            writer.rows.append(row)
            writer.seen.add(row["clip_id"])
    meta = writer.close()
    failed.to_csv(os.path.join(args.out, "failed.csv"), index=False)
    clips[clips.clip_id.isin(set(index.clip_id))].to_csv(os.path.join(args.out, "clips.csv"), index=False)
    shutil.rmtree(os.path.join(args.out, "chunks"))
    print(json.dumps({"clips_in_table": len(clips), "excluded": n_excluded, "cached": meta["n_clips"],
                      "unreadable": len(failed), "index_sha256": meta["index_sha256"]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("encode", "merge"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", help="overrides encoder.checkpoint of the config")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if not 0 <= args.worker < args.workers or args.chunks < 1:
        raise ValueError("Need 0 <= worker < workers and at least one chunk.")
    root, target = os.path.realpath(REPO), os.path.realpath(args.out)
    if os.path.commonpath([root, target]) == root:
        raise ValueError("--out is inside the code repository; clip ids are file paths, write the cache elsewhere.")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.checkpoint:
        config["encoder"]["checkpoint"] = args.checkpoint
    (encode if args.stage == "encode" else merge)(config, args)


if __name__ == "__main__":
    main()
