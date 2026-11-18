# evals/view_classifier/mvit.py

"""Arm B: the current view classifier, fine-tuned with the new classes.

The classifier (`best_model_pretrained_echoprime_updated.pth`, see its README) is torchvision's MViT-v2-S video
transformer, EchoPrime-pretrained: the stock head's last layer replaced by a 768 -> 512 projection (EchoPrime's video
embedding), then a 512 -> 20 linear layer. Its state dict (saved from DataParallel, keys `module.`) loads strictly
into `build_mvit(20)`; nothing outside torchvision is needed and no file is fetched.

Input, as its README gives it: 16 frames at period 2, each frame cropped to a square around its centre, 10% cut from
every side (EchoPrime's `crop_and_scale` zoom), resized to 224 (bicubic), normalised in 0-255 space by the README's
mean and std. EchoPrime's sector masking (`mask_and_crop`) is not applied: its published code works on DICOM YBR
pixels, and these clips are decoded MP4s. The channel order the weights expect is not documented; `zeroshot` scores
the unchanged model on the labelling pool in both orders and measures agreement with the archive's predictions, made
by the same weights: the order to use, and a check that this input is close to the one the archive was scored with.

The input is selectable in the arm's `data` section (the defaults are the input above, which the first runs used):
- `preprocessing`: `centre_zoom` (default: centre square, `zoom` cut from each side, bicubic), `full_squash` (the
  whole frame resized to `resolution` x `resolution`, bicubic without antialiasing, nothing cut, aspect not kept; `zoom`
  is ignored) or `full_squash_aa` (the same whole-frame squash, bilinear WITH antialiasing: `preprocess_variants`'
  `squash_bilinear_aa` itself, so with `frame_read: echoprime` the input is byte for byte its `full_squash_bilinear_aa`
  variant). Of the 16 preparations `preprocess_variants` compared over two rounds, `full_squash_bilinear_aa` in RGB
  reproduced the archive's predictions best (essentially exactly); `full_squash` was the best of the first round.
- `frame_read`: `cache` (default: the token cache's read, `frames_per_clip` frames spread over the first
  `frames_per_clip * frame_step`, a short clip padded with black frames) or `echoprime` (frames 0, step, 2 step, ...
  of the first `frames_per_clip * frame_step` that exist; a short clip's missing frames are zero after normalisation,
  as EchoPrime and `preprocess_variants` pad).

Fine-tuning: the 512 -> 20 layer becomes 512 -> len(new classes) (`extend_head`): a kept class keeps its row, a new
class starts from its parent's row (`head_init: parent`, bias minus log of the number of classes sharing the parent)
or at random (`head_init: random`). The whole network trains, the head at `head_lr_mult` times the backbone's rate.

Clips are decoded once per process into memory (uint8, about 2.4 MB a clip) and never written.

    python -m evals.view_classifier.mvit zeroshot --config CONFIG --device cuda:0
    python -m evals.view_classifier.mvit train --config CONFIG --pseudo off --folds 0 1 2 --device cuda:0
    python -m evals.view_classifier.mvit merge --config CONFIG --pseudo off
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import train as vt
from evals.view_classifier.io import read_run_inputs, run_dir
from evals.view_classifier.labels import file_sha256

ARM = "arm_b"
README_MEAN = (24.28, 22.15, 22.40)
README_STD = (47.23, 44.03, 43.90)
HIDDEN = 512


def build_mvit(num_classes):
    import torchvision
    model = torchvision.models.video.mvit_v2_s(weights=None)
    model.head[-1] = nn.Linear(model.head[-1].in_features, HIDDEN)
    model.head = nn.Sequential(model.head, nn.Linear(HIDDEN, num_classes))
    return model


def load_original(path, expected_sha256=None, num_classes=len(vc.OLD_CLASSES)):
    """The unchanged classifier, loaded strictly on CPU."""
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("The view classifier weights differ from the expected sha256.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = build_mvit(num_classes)
    model.load_state_dict(state, strict=True)
    return model


def extend_head(linear, old=vc.OLD_CLASSES, new=vc.NEW_CLASSES, parent=None, init="parent", seed=0):
    """A new nn.Linear(in, len(new)) from the old one (out = len(old)); see the module docstring."""
    if linear.out_features != len(old):
        raise ValueError(f"The head has {linear.out_features} outputs, the old class list {len(old)}.")
    out = nn.Linear(linear.in_features, len(new))
    if init == "random":
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            out.weight.normal_(0.0, 0.02, generator=g)
            out.bias.zero_()
        kept = {c: old.index(c) for c in new if c in old and (parent or vc.PARENT)[c] == c}
        with torch.no_grad():
            for j, c in enumerate(new):
                if c in kept:
                    out.weight[j] = linear.weight[kept[c]]
                    out.bias[j] = linear.bias[kept[c]]
        return out
    if init != "parent":
        raise ValueError(f"Unknown head_init {init!r}.")
    with torch.no_grad():
        for j, (row, offset) in enumerate(vc.head_rows(old, new, parent)):
            out.weight[j] = linear.weight[row]
            out.bias[j] = linear.bias[row] + offset
    return out


def crop_and_scale(buffer, size=224, zoom=0.1):
    """uint8 [T, H, W, 3] (array or tensor) -> uint8 tensor [3, T, size, size]: centre square, `zoom` cut from each
    side, bicubic resize (EchoPrime's `crop_and_scale`, here in torch)."""
    x = torch.as_tensor(np.asarray(buffer))
    _, h, w, _ = x.shape
    if w > h:
        pad = int(round((w - h) / 2))
        x = x[:, :, pad:pad + h]
    elif h > w:
        pad = int(round((h - w) / 2))
        x = x[:, pad:pad + w]
    if zoom:
        py, px = int(x.shape[1] * zoom), int(x.shape[2] * zoom)
        x = x[:, py:x.shape[1] - py, px:x.shape[2] - px]
    x = x.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()


def full_squash(buffer, size=224):
    """uint8 [T, H, W, 3] (array or tensor) -> uint8 tensor [3, T, size, size]: the whole frame resized to size x size
    (bicubic, aspect not kept, nothing cut), as `preprocess_variants`' `full_squash`."""
    x = torch.as_tensor(np.asarray(buffer)).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()


def full_squash_aa(buffer, size=224):
    """uint8 [T, H, W, 3] (array or tensor) -> uint8 tensor [3, T, size, size]: the whole frame resized to size x size,
    bilinear with antialiasing (aspect not kept, nothing cut). It calls `preprocess_variants.squash_bilinear_aa`, the
    function that made the `full_squash_bilinear_aa` variant, so the pixels are the ones that variant scored."""
    from evals.view_classifier.preprocess_variants import squash_bilinear_aa   # (that module imports this one)
    x = squash_bilinear_aa(np.asarray(buffer), size)
    return torch.as_tensor(x).permute(3, 0, 1, 2).contiguous()


PREPROCESSING = ("centre_zoom", "full_squash", "full_squash_aa")
FRAME_READS = ("cache", "echoprime")


def data_settings(data_cfg):
    """(preprocessing, frame_read) of an arm's `data` section, checked; the defaults are the first runs' input."""
    data_cfg = data_cfg or {}
    preprocessing = data_cfg.get("preprocessing", "centre_zoom")
    frame_read = data_cfg.get("frame_read", "cache")
    if preprocessing not in PREPROCESSING:
        raise ValueError(f"Unknown preprocessing {preprocessing!r} (one of {PREPROCESSING}).")
    if frame_read not in FRAME_READS:
        raise ValueError(f"Unknown frame_read {frame_read!r} (one of {FRAME_READS}).")
    return preprocessing, frame_read


def spatial(buffer, size=224, zoom=0.1, preprocessing="centre_zoom"):
    """uint8 [T, H, W, 3] -> uint8 tensor [3, T, size, size] by the selected preprocessing."""
    if preprocessing == "centre_zoom":
        return crop_and_scale(buffer, size, zoom)
    if preprocessing == "full_squash":
        return full_squash(buffer, size)
    if preprocessing == "full_squash_aa":
        return full_squash_aa(buffer, size)
    raise ValueError(f"Unknown preprocessing {preprocessing!r}.")


def normalize(x, channel_order="rgb", mean=README_MEAN, std=README_STD):
    """uint8 [B, 3, T, H, W] (RGB) -> float, channels in `channel_order`, normalised in 0-255 space."""
    if channel_order == "bgr":
        x = x.flip(1)
    elif channel_order != "rgb":
        raise ValueError(f"Unknown channel_order {channel_order!r}.")
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1, 1)
    return (x.float() - m) / s


def trailing_black(x):
    """bool [B, T]: the frames of uint8 [B, 3, T, H, W] that are entirely black and followed only by such frames (the
    padding `frame_read: echoprime` appends to a short clip)."""
    black = (x == 0).flatten(3).all(dim=-1).all(dim=1)
    return black.flip(1).cumprod(dim=1).flip(1).bool()


def model_input(x, channel_order="rgb", zero_padding=False):
    """uint8 [B, 3, T, H, W] -> the model's float input: `normalize`, and with `zero_padding` the trailing black
    (padding) frames set to 0 after normalisation, as EchoPrime pads a short clip."""
    out = normalize(x, channel_order)
    if zero_padding:
        pad = trailing_black(x)
        if pad.any():
            out = out.masked_fill(pad[:, None, :, None, None], 0.0)
    return out


def random_crop(x, scale_min, generator):
    """Training augmentation on [B, 3, T, H, W]: a random square crop of side in [scale_min, 1] of the frame, at a
    random place, resized back. The same crop for every frame of a clip; no flip (it would change the view).
    It acts on the model input (`make_arm`'s `prepare`: after the preprocessing, normalisation and zeroing of padding
    frames), whatever the preprocessing: a zoom of 1 to 1/scale_min of the 224 x 224 input (bilinear upsampling, so
    antialiasing does not apply); padding frames, all 0, stay 0. Evaluation never crops."""
    if scale_min >= 1.0:
        return x
    b, c, t, h, w = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        s = float(scale_min + (1 - scale_min) * torch.rand(1, generator=generator))
        ch, cw = max(1, int(h * s)), max(1, int(w * s))
        y0 = int(torch.randint(0, h - ch + 1, (1,), generator=generator))
        x0 = int(torch.randint(0, w - cw + 1, (1,), generator=generator))
        crop = x[i, :, :, y0:y0 + ch, x0:x0 + cw].permute(1, 0, 2, 3)
        out[i] = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    return out


def echoprime_read(path, frames=16, frame_step=2):
    """uint8 [n, H, W, 3] RGB: frames 0, frame_step, ... of the first frames * frame_step of the video that exist
    (n <= frames), or None when it cannot be decoded. The window is decoded in order and then subsampled, as
    `preprocess_variants` reads it. Decoder output, which could name the file, is discarded."""
    from decord import VideoReader, cpu
    from src.datasets.video_dataset import _native_stderr_discarded
    try:
        with _native_stderr_discarded():
            vr = VideoReader(path, num_threads=1, ctx=cpu(0))
            n = min(len(vr), frames * frame_step)
            if n == 0:
                return None
            return vr.get_batch(list(range(n))).asnumpy()[::frame_step][:frames]
    except Exception:
        return None


def pad_frames(clip, frames):
    """uint8 [3, n, S, S] -> [3, frames, S, S], black frames appended."""
    n = clip.shape[1]
    if n >= frames:
        return clip[:, :frames]
    pad = torch.zeros((clip.shape[0], frames - n, *clip.shape[2:]), dtype=clip.dtype)
    return torch.cat([clip, pad], dim=1)


class ClipDecoder(torch.utils.data.Dataset):
    """Item i: (i, uint8 [3, T, 224, 224] or None when the clip cannot be decoded). With `frame_read: cache` the first
    `frames` frames at `frame_step` (a short video padded with black frames), as the token cache reads a clip; with
    `echoprime`, EchoPrime's frames (`echoprime_read`), padded with black frames. Then `spatial`."""

    def __init__(self, clip_ids, frames=16, frame_step=2, size=224, zoom=0.1, preprocessing="centre_zoom",
                 frame_read="cache"):
        from src.datasets.video_study_dataset import make_clip_loader
        self.clip_ids, self.frames, self.size, self.zoom = list(clip_ids), frames, size, zoom
        self.frame_step = frame_step
        self.preprocessing, self.frame_read = data_settings({"preprocessing": preprocessing, "frame_read": frame_read})
        self.loader = None
        if self.frame_read == "cache":
            self.loader = make_clip_loader(frames, frame_step, random_clip_sampling=False)

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        if self.frame_read == "echoprime":
            buffer = echoprime_read(self.clip_ids[i], self.frames, self.frame_step)
            if buffer is None or len(buffer) == 0:
                return i, None
            return i, pad_frames(spatial(buffer, self.size, self.zoom, self.preprocessing), self.frames)
        buffer, _ = self.loader.loadvideo_decord(self.clip_ids[i], self.frames)
        if buffer is None or len(buffer) == 0:
            return i, None
        return i, spatial(buffer[:self.frames], self.size, self.zoom, self.preprocessing)


def _collate(items):
    return items


def decode_all(clip_ids, data_cfg, workers=8):
    """(uint8 tensor [N, 3, T, 224, 224], {clip_id: row}, unreadable clip ids, seconds) for the readable clips."""
    ids = list(dict.fromkeys(str(c) for c in clip_ids))
    preprocessing, frame_read = data_settings(data_cfg)
    dataset = ClipDecoder(ids, int(data_cfg.get("frames_per_clip", 16)), int(data_cfg.get("frame_step", 2)),
                          int(data_cfg.get("resolution", 224)), float(data_cfg.get("zoom", 0.1)), preprocessing,
                          frame_read)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=workers, collate_fn=_collate,
                                         multiprocessing_context="spawn" if workers else None)
    start, clips, failed = time.time(), {}, []
    for items in loader:
        for i, clip in items:
            if clip is None:
                failed.append(ids[i])
            else:
                clips[ids[i]] = clip
    kept = [c for c in ids if c in clips]
    data = torch.stack([clips[c] for c in kept]) if kept else torch.zeros((0,), dtype=torch.uint8)
    return data, {c: i for i, c in enumerate(kept)}, failed, time.time() - start


def make_arm(data, row, cfg, weights, classes=vc.NEW_CLASSES, parent=None):
    channel_order = cfg.get("channel_order", "rgb")
    original = load_original(weights, cfg.get("weights_sha256"))

    def build(hp):
        model = build_mvit(len(vc.OLD_CLASSES))
        model.load_state_dict(original.state_dict())
        model.head[1] = extend_head(model.head[1], vc.OLD_CLASSES, classes, parent,
                                    init=hp.get("head_init", "parent"))
        return model

    def param_groups(model, hp):
        lr, mult, wd = float(hp["lr"]), float(hp.get("head_lr_mult", 10.0)), float(hp["weight_decay"])
        groups = {k: [] for k in ("body_decay", "body_flat", "head_decay", "head_flat")}
        for name, p in model.named_parameters():
            part = "head" if name.startswith("head.") else "body"
            groups[f"{part}_{'decay' if p.ndim >= 2 else 'flat'}"].append(p)
        return [{"params": groups["body_decay"], "lr": lr, "weight_decay": wd},
                {"params": groups["body_flat"], "lr": lr, "weight_decay": 0.0},
                {"params": groups["head_decay"], "lr": lr * mult, "weight_decay": wd},
                {"params": groups["head_flat"], "lr": lr * mult, "weight_decay": 0.0}]

    scale_min = float(cfg.get("crop_scale_min", 0.85))
    zero_padding = data_settings(cfg.get("data"))[1] == "echoprime"

    def prepare(x, training, generator):
        x = model_input(x, channel_order, zero_padding)
        return random_crop(x, scale_min, generator) if training else x

    return vt.Arm(name=ARM, data=data, row=row, build=build, param_groups=param_groups, prepare=prepare)


@torch.no_grad()
def zero_shot(cfg, device, workers):
    """Score the unchanged classifier on every clip of the labelling pool in both channel orders."""
    b, paths = cfg[ARM], cfg["paths"]
    out = os.path.join(paths["runs"], ARM, "zeroshot")
    os.makedirs(out, exist_ok=True)
    pool = pd.read_csv(paths["pool_clips"], dtype=str)
    data, row, failed, decode_seconds = decode_all(pool.video_path, b.get("data", {}), workers)
    preprocessing, frame_read = data_settings(b.get("data"))
    model = load_original(paths["view_classifier"], b.get("weights_sha256")).to(device).eval()
    record = {"clips": len(pool), "decoded": len(row), "unreadable": len(failed), "decode_seconds": decode_seconds,
              "decode_clips_per_s": len(row) / max(decode_seconds, 1e-9), "decode_workers": workers,
              "preprocessing": preprocessing, "frame_read": frame_read, "orders": {}}
    keep = pool[pool.video_path.isin(row)].reset_index(drop=True)
    for order in ("rgb", "bgr"):
        probs, seconds = [], 0.0
        for start in range(0, len(keep), 32):
            x = data[[row[c] for c in keep.video_path[start:start + 32]]].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
            logits = model(model_input(x, order, frame_read == "echoprime"))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds += time.time() - t0
            probs.append(torch.softmax(logits.float(), dim=-1).cpu())
        p = torch.cat(probs).numpy()
        table = pd.DataFrame(p, columns=[f"p_{c}" for c in vc.OLD_CLASSES])
        table.insert(0, "clip_id", keep.video_path)
        table["predicted"] = np.asarray(vc.OLD_CLASSES)[p.argmax(axis=1)]
        table.to_csv(os.path.join(out, f"zeroshot_{order}.csv"), index=False)
        agree = float((table.predicted.to_numpy() == keep.view.to_numpy()).mean())
        record["orders"][order] = {"agreement_with_archive": agree, "model_seconds": seconds,
                                   "model_clips_per_s_fp32_batch32": len(keep) / max(seconds, 1e-9)}
        print(f"{order}: agreement with the archive's predictions {agree:.3f}", flush=True)
    with open(os.path.join(out, "zeroshot.json"), "w") as f:
        json.dump(record, f, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("zeroshot", "train", "merge"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo", choices=("on", "off"))
    parser.add_argument("--folds", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)
    if args.stage == "zeroshot":
        return zero_shot(cfg, device, args.workers)
    if args.pseudo is None:
        parser.error("train and merge need --pseudo")
    labels, pseudo = read_run_inputs(cfg, args.pseudo == "on")
    out = run_dir(cfg, ARM, args.pseudo)
    classes, parent = vc.from_config(cfg)
    if args.stage == "merge":
        oof = vt.merge_folds(out, labels, int(cfg["cv"]["k"]), classes)
        print(f"{ARM} pseudo {args.pseudo}: {len(oof)} out-of-fold predictions")
        return
    b = cfg[ARM]
    ids = list(labels.clip_id) + ([] if pseudo is None else list(pseudo.clip_id))
    data, row, failed, seconds = decode_all(ids, b.get("data", {}), args.workers)
    print(f"decoded {len(row)} clips in {seconds:.0f} s, {len(failed)} unreadable", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} clips could not be decoded; the arms must train on the same clips.")
    arm = make_arm(data, row, b, cfg["paths"]["view_classifier"], classes, parent)
    os.makedirs(out, exist_ok=True)
    for fold in args.folds:
        vt.run_fold(arm, labels, pseudo, fold if fold == vt.FINAL else int(fold), vt.grid_from_config(b), cfg["cv"],
                    classes, out, device)


if __name__ == "__main__":
    main()
