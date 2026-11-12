# evals/radar_frozen/build_still_cache.py

"""Write the token cache of the Doppler stills: the frozen encoder's tokens of every still of a still table.

Each still is prepared by `src.datasets.still_preprocess.prepare_still` (text boxes masked, cropped to the spectral
region, coloured overlays inpainted, grey, 224 x 224) and repeated over `--frames` frames, then encoded by the same
frozen encoder, with the same settings, as the clip cache (`build_token_cache.py`; the config is the clip cache's).
The encoder is never trained here or later: the caches differ only in how the still is presented to it.

`--frames 2` gives one temporal token per position, a `[1, 14, 14]` grid (196 tokens per still); `--frames 16` gives
the clip grid, `[8, 14, 14]` (1,568 tokens, eight times the cost). Both are valid inputs of the encoder; which one
serves the downstream tasks better is an experiment: write both caches from the same still table and compare them
with the same split and evaluation. The cache's meta records the frames, the preprocessing recipe and the same
`encoder_sha256` and `encoder` record as the clip cache, which training compares (`radar_stills.same_encoder`).

A still that cannot be read or that the preprocessing refuses (a mostly chromatic crop) gets no tokens and is listed
in `<out>/failed.csv` with the reason's kind. `<out>/stills.csv` is the still table without those: the table to
train on, since training refuses a still that has no tokens. `<out>` must be outside the repository.

Usage:
    python -m evals.radar_frozen.build_still_cache --config configs/radar/token_cache_vitb.yaml \\
        --checkpoint <encoder> --stills <stills.csv> --frames 2 --out <folder>
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from evals.radar_frozen.build_token_cache import DEFAULT_NORMALIZATION, REPO, _commit, build_encoder, encode_batch
from src.datasets.radar_study_dataset import file_sha256
from src.datasets.radar_token_cache import TokenCacheWriter
from src.datasets.still_preprocess import PREPROCESS_VERSION, prepare_still

FRAMES = (2, 16)
BOX = ("spec_x0", "spec_y0", "spec_x1", "spec_y1")


def read_image(path):
    """The still's screen as RGB uint8 [H, W, 3]."""
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class StillReader(torch.utils.data.Dataset):
    """Item i: (i, clip [3, frames, size, size] or None, the kind of failure or "")."""

    def __init__(self, table, frames, size=224, read=read_image):
        self.rows = table.to_dict("records")
        self.frames, self.size, self.read = int(frames), int(size), read

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            image = self.read(row["png_path"])
        except Exception:                      # the message could name the file: only the kind is kept
            return i, None, "unreadable"
        try:
            box = tuple(int(float(row[c])) for c in BOX)
            boxes = json.loads(row["text_boxes"]) if isinstance(row.get("text_boxes"), str) else []
            clip, _ = prepare_still(image, box, boxes, size=self.size, frames=self.frames)
        except Exception:
            return i, None, "refused_by_preprocessing"
        return i, clip, ""


def _collate(items):
    good = [item for item in items if item[1] is not None]
    failed = [(item[0], item[2]) for item in items if item[1] is None]
    return [g[0] for g in good], (torch.stack([g[1] for g in good]) if good else None), failed


def build(config, stills_path, out, frames, split=None, device="auto", batch_size=32, num_workers=8,
          make_encoder=build_encoder, read=read_image):
    if frames not in FRAMES:
        raise ValueError(f"frames must be one of {FRAMES}.")
    table = pd.read_csv(stills_path, dtype={"study_id": str, "still_id": str}, keep_default_na=False)
    for column in ("still_id", "png_path", *BOX):
        if column not in table.columns:
            raise ValueError(f"The still table needs a {column!r} column.")
    if split is not None:
        table = table[table.split.astype(str) == split]
    if table.still_id.duplicated().any():
        raise ValueError("A still appears twice in the still table.")
    table = table.reset_index(drop=True)
    enc, data = config["encoder"], dict(config.get("data", {}))
    data["frames_per_clip"] = frames                       # the encoder is built for the still's frames
    resolution = int(data.get("resolution", 224))
    patch = int(enc["pretrain_kwargs"]["encoder"]["patch_size"])
    tubelet = int(enc["pretrain_kwargs"]["encoder"]["tubelet_size"])
    grid = [frames // tubelet, resolution // patch, resolution // patch]
    record = json.loads(json.dumps({
        "encoder_sha256": file_sha256(enc["checkpoint"]),
        "encoder": {k: enc.get(k) for k in ("module_name", "pretrain_kwargs", "wrapper_kwargs")},
        "data": {"frames": frames, "resolution": resolution, "normalization": DEFAULT_NORMALIZATION,
                 "input": "static still"},
        "preprocess": PREPROCESS_VERSION, "use_bfloat16": bool(config["cache"].get("use_bfloat16", True)),
        "stills_sha256": file_sha256(stills_path), "split": split, "stills": len(table),
        "builder_sha256": file_sha256(os.path.abspath(__file__)), "commit": _commit()}))
    device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    writer = TokenCacheWriter(out, grid, config["cache"]["embed_dim"], config["cache"].get("dtype", "float16"),
                              provenance=record)
    encoder = make_encoder({**config, "data": data}, device)
    loader = torch.utils.data.DataLoader(
        StillReader(table, frames, resolution, read), batch_size=batch_size, num_workers=num_workers,
        collate_fn=_collate, shuffle=False, multiprocessing_context="spawn" if num_workers else None)
    indices = torch.arange(frames)
    failed = []
    for positions, clips, bad in loader:
        failed += [(table.still_id[i], kind) for i, kind in bad]
        if not positions:
            continue
        tokens = encode_batch(encoder, clips, indices[None].repeat(len(positions), 1), device,
                              bool(config["cache"].get("use_bfloat16", True)))
        for i, still_tokens in zip(positions, tokens):
            writer.add(table.still_id[i], still_tokens)
    meta = writer.close()
    pd.DataFrame(failed, columns=["still_id", "reason"]).to_csv(os.path.join(out, "failed.csv"), index=False)
    table[table.still_id.isin(writer.seen)].to_csv(os.path.join(out, "stills.csv"), index=False)
    summary = {"stills_in_table": len(table), "cached": meta["n_clips"], "token_grid": grid,
               "failed": pd.Series([kind for _, kind in failed]).value_counts().to_dict()}
    print(json.dumps(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True,
                        help="the clip cache's config (configs/radar/token_cache_vitb.yaml)")
    parser.add_argument("--checkpoint", help="overrides encoder.checkpoint of the config")
    parser.add_argument("--stills", required=True)
    parser.add_argument("--split", help="keep the stills of this split (default: all)")
    parser.add_argument("--frames", type=int, required=True, choices=FRAMES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root, target = os.path.realpath(REPO), os.path.realpath(args.out)
    if os.path.commonpath([root, target]) == root:
        raise ValueError("--out is inside the code repository; write the cache elsewhere.")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.checkpoint:
        config["encoder"]["checkpoint"] = args.checkpoint
    build(config, args.stills, args.out, args.frames, args.split, args.device, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
