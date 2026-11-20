# evals/view_classifier/preprocess_variants.py

"""The unchanged view classifier under several input preprocessings: which one reproduces the archive's predictions.

The archive's stored view predictions (master metadata) come from the same weights, but through the production
pipeline, whose code is not in this repository. Arm B's input (`mvit.crop_and_scale` on the token cache's frame read)
agrees with them on only part of the labelling pool. Each variant below changes the input in one way; a variant whose
agreement is close to 1 is the production input, and the others tell which step matters.

- `current`: arm B's input exactly as `mvit zeroshot` builds it (the token cache's frame read, centre square, 10% zoom
  cut from each side, bicubic 224). It must reproduce the zero-shot agreement: a check of this script.
- `current_epframes`: the same crop, with EchoPrime's frames: frames 0, 2, ..., 30 of the first 32, a short clip padded
  after normalisation (zeros, i.e. the mean), instead of frames spread over a short clip and padded with black.
- `centre_nozoom`: the centre square without the 10% zoom.
- `full_pad`: the whole frame, padded with black to a square (nothing cut, aspect kept).
- `full_squash`: the whole frame resized to 224 x 224 (nothing cut, aspect changed).
- `region_square`: the scanner-declared 2D image region (the DICOM's SequenceOfUltrasoundRegions, read from the header
  only), everything outside it zeroed, the square around it.
- `mask_and_crop`: the archive extractor's `mask_and_crop` (DeepECHO): the pixels that change in more than 10
  frame-to-frame differences, opened with a disk of radius 5, everything else zeroed, the square around them; no zoom.
- `echoprime_mask`: EchoPrime's own preprocessing, `mask_outside_ultrasound` then `crop_and_scale` (10% zoom), run from
  the vendored EchoPrime code. Its masking converts YUV pixels (as a DICOM stores them) to BGR, so the frames go
  RGB -> YUV before it and BGR -> RGB after it.

All but `current` take EchoPrime's frames. The masks use every decoded frame (`--max-frames`, 0 = all). A variant that
cannot be computed for a clip (no declared region, no moving pixels, EchoPrime's masking failed) falls back to
`full_pad` for that clip and is counted. Each variant is scored in RGB and BGR order, fp32, as `mvit zeroshot` does.

Second round (added after the first eight were scored; `full_squash` reproduced the archive best): the same whole-frame
squash with one other change each, for the disagreement that remains. They leave the first eight unchanged.

- `full_squash_cacheframes`: the frames of `current` (the token cache's read: spread over the first 32, black padding),
  i.e. arm B's input with `preprocessing: full_squash` and `frame_read: cache`.
- `full_squash_blackpad`: EchoPrime's frames, a short clip padded with black frames (not zeros after normalisation).
- `full_squash_period1`: the first 16 frames (period 1).
- `full_squash_middle`: frames 0, 2, ..., 30 of the 32-frame window centred in the clip (the first 32 if shorter).
- `full_squash_spread`: 16 frames spread evenly over the whole clip.
- `full_squash_bilinear_aa`: EchoPrime's frames, bilinear resize with antialiasing (torchvision's tensor `Resize`).
- `full_squash_cv2linear`: EchoPrime's frames, `cv2.resize` with its default bilinear interpolation.
- `full_squash_cv2decode`: EchoPrime's frames decoded by OpenCV (`cv2.VideoCapture`, BGR turned to RGB) instead of
  decord; bicubic as `full_squash`.

Outputs (`--out`, default <runs>/arm_b/zeroshot_variants/): `probs_<variant>_<order>.csv` (one row per clip),
`summary.csv` (per variant and order: agreement with the archive overall, B-mode and colour flow; share predicted
OTHER; mean total-variation distance to the archive's probabilities; fallbacks; and, when the folds table exists,
accuracy against the human labels collapsed to the 20 classes) and `variants.json`. Nothing else is written; frames
stay in memory.

    python -m evals.view_classifier.preprocess_variants --config CONFIG --device cuda:0 --workers 8
    python -m evals.view_classifier.preprocess_variants --config CONFIG --device cuda:0 --second-round --out DIR

An output folder that already holds a `summary.csv` is refused, so a rerun never replaces scored results.
"""

import argparse
import ast
import contextlib
import io
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import mvit

FIRST_ROUND = ("current", "current_epframes", "centre_nozoom", "full_pad", "full_squash", "region_square",
               "mask_and_crop", "echoprime_mask")
SECOND_ROUND = ("full_squash_cacheframes", "full_squash_blackpad", "full_squash_period1", "full_squash_middle",
                "full_squash_spread", "full_squash_bilinear_aa", "full_squash_cv2linear", "full_squash_cv2decode")
VARIANTS = FIRST_ROUND + SECOND_ROUND
ORDERS = ("rgb", "bgr")
ECHOPRIME_UTILS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "video_classification_frozen", "modelcustom", "EchoPrime", "utils", "utils.py")
FRAMES, STEP, WINDOW = 16, 2, 32


# ----------------------------------------------------------------------------------------------------------------------
# EchoPrime's functions, taken from the vendored file without importing it (its module code reads files relative to
# the working directory and imports pydicom / torchvision): only the named function definitions are executed.

_ECHOPRIME = {}


def echoprime_functions(path=ECHOPRIME_UTILS, names=("mask_outside_ultrasound", "crop_and_scale")):
    key = (path, tuple(names))
    if key not in _ECHOPRIME:
        import cv2
        with open(path) as f, warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)       # regex strings elsewhere in the vendored file
            tree = ast.parse(f.read())
        found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if sorted(n.name for n in found) != sorted(names):
            raise ValueError(f"{path} lacks one of {names}.")
        namespace = {"np": np, "cv2": cv2}
        exec(compile(ast.Module(body=found, type_ignores=[]), path, "exec"), namespace)
        _ECHOPRIME[key] = {n: namespace[n] for n in names}
    return _ECHOPRIME[key]


# ----------------------------------------------------------------------------------------------------------------------
# Frames and geometry. Frames are uint8 [T, H, W, 3] RGB; every variant returns uint8 [3, 16, size, size] and the number
# of real (not padded) frames.

def echoprime_frames(frames):
    """EchoPrime's selection: frames 0, 2, ..., 30 of the first 32 that exist. Returns (frames, n_real)."""
    picked = frames[0:WINDOW:STEP]
    return picked, len(picked)


def _to_clip(x, n_real, size):
    """uint8 [n, size, size, 3] (n <= 16) -> uint8 tensor [3, 16, size, size], zero frames after the real ones."""
    out = torch.zeros((FRAMES, size, size, 3), dtype=torch.uint8)
    x = torch.as_tensor(np.ascontiguousarray(x))
    out[:n_real] = x[:n_real]
    return out.permute(3, 0, 1, 2).contiguous()


def resize_square(frames, size=224):
    """Square uint8 [T, S, S, 3] -> uint8 [T, size, size, 3] (bicubic, as `mvit.crop_and_scale`)."""
    return mvit.crop_and_scale(frames, size, zoom=0).permute(1, 2, 3, 0).numpy()


def pad_to_square(frames):
    t, h, w, c = frames.shape
    side = max(h, w)
    out = np.zeros((t, side, side, c), dtype=frames.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[:, y0:y0 + h, x0:x0 + w] = frames
    return out


def squash(frames, size=224):
    x = torch.as_tensor(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def squash_bilinear_aa(frames, size=224):
    """The whole frame to size x size, bilinear with antialiasing (as torchvision's `Resize` on a tensor)."""
    x = torch.as_tensor(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def squash_cv2(frames, size=224):
    """The whole frame to size x size by `cv2.resize` with its default (bilinear) interpolation."""
    import cv2
    return np.stack([cv2.resize(np.ascontiguousarray(f), (size, size), interpolation=cv2.INTER_LINEAR)
                     for f in frames])


def middle_frames(frames):
    """Frames 0, 2, ..., 30 of the 32-frame window centred in the clip (the first 32 when shorter)."""
    start = max(0, (len(frames) - WINDOW) // 2)
    picked = frames[start:start + WINDOW:STEP]
    return picked, len(picked)


def period1_frames(frames):
    """The first 16 frames."""
    picked = frames[:FRAMES]
    return picked, len(picked)


def spread_frames(frames):
    """16 frames spread evenly over the whole clip (every frame of a clip shorter than 16)."""
    idx = np.round(np.linspace(0, len(frames) - 1, min(FRAMES, len(frames)))).astype(np.int64)
    picked = frames[idx]
    return picked, len(picked)


def square_around(frames, box, keep=None):
    """The square of side max(width, height) centred on `box` = (x0, y0, x1, y1) (exclusive ends), zero-padded where
    it leaves the frame; with `keep` (bool [H, W]) everything outside it is zeroed first."""
    t, h, w, c = frames.shape
    if keep is not None:
        frames = frames * keep[None, :, :, None].astype(frames.dtype)
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx, sy = cx - side // 2, cy - side // 2
    out = np.zeros((t, side, side, c), dtype=frames.dtype)
    fx0, fy0, fx1, fy1 = max(sx, 0), max(sy, 0), min(sx + side, w), min(sy + side, h)
    out[:, fy0 - sy:fy1 - sy, fx0 - sx:fx1 - sx] = frames[:, fy0:fy1, fx0:fx1]
    return out


def region_box(regions, height, width):
    """Union of the 2D regions (RegionSpatialFormat 1) as (x0, y0, x1, y1) with exclusive ends, clipped to the
    frame, or None. `regions`: iterable of dicts with the DICOM keywords."""
    boxes = [(int(r["RegionLocationMinX0"]), int(r["RegionLocationMinY0"]),
              int(r["RegionLocationMaxX1"]) + 1, int(r["RegionLocationMaxY1"]) + 1)
             for r in regions if int(r.get("RegionSpatialFormat", -1)) == 1]
    if not boxes:
        return None
    x0, y0 = max(min(b[0] for b in boxes), 0), max(min(b[1] for b in boxes), 0)
    x1, y1 = min(max(b[2] for b in boxes), width), min(max(b[3] for b in boxes), height)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def read_regions(dicom_path):
    """The DICOM's ultrasound regions from its header (pixel data is not read)."""
    import pydicom
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, specific_tags=["SequenceOfUltrasoundRegions"])
    keys = ("RegionSpatialFormat", "RegionLocationMinX0", "RegionLocationMinY0", "RegionLocationMaxX1",
            "RegionLocationMaxY1")
    return [{k: getattr(item, k) for k in keys if hasattr(item, k)}
            for item in (ds.get("SequenceOfUltrasoundRegions") or [])]


def motion_mask(frames, min_changes=10, radius=5):
    """The extractor's `mask_and_crop` mask: pixels whose channel sum changes in more than `min_changes` consecutive
    frame pairs, eroded then dilated with a disk of `radius`. bool [H, W]."""
    from skimage import morphology
    count = np.zeros(frames.shape[1:3], dtype=np.int32)
    prev = frames[0].sum(axis=2, dtype=np.int32)
    for t in range(1, len(frames)):      # one frame pair at a time: a long clip's full difference stack is gigabytes
        cur = frames[t].sum(axis=2, dtype=np.int32)
        count += cur != prev
        prev = cur
    changes = count > min_changes
    disk = morphology.disk(radius)
    return morphology.dilation(morphology.erosion(changes, disk), disk).astype(bool)


def mask_and_crop(frames, min_changes=10, radius=5):
    """The extractor's `mask_and_crop`: masked frames, the square around the mask's bounding box (its arithmetic,
    zero-padded). Returns None when no pixel moves enough."""
    keep = motion_mask(frames, min_changes, radius)
    xs, ys = np.where(keep.any(axis=0))[0], np.where(keep.any(axis=1))[0]
    if len(xs) == 0 or len(ys) == 0:
        return None
    left, right, top, bottom = xs[0], xs[-1], ys[0], ys[-1]
    h, w = bottom - top, right - left
    pad = int(max(h, w) / 2)
    x_center, y_center = right - int(w / 2) + pad, bottom - int(h / 2) + pad
    side = int(max(h, w) / 2) * 2
    if side == 0:
        return None
    padded = np.pad(frames * keep[None, :, :, None].astype(frames.dtype), ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    top_, left_ = int(y_center - side / 2), int(x_center - side / 2)
    return padded[:, top_:top_ + side, left_:left_ + side]


def echoprime_preprocess(frames, size=224):
    """EchoPrime: `mask_outside_ultrasound` on every frame (as YUV), `crop_and_scale` (10% zoom) on its frames.
    Returns (uint8 [n, size, size, 3] RGB, n_real) or None when its masking failed."""
    import cv2
    fns = echoprime_functions()
    yuv = np.stack([cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2YUV) for f in frames])
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        bgr = fns["mask_outside_ultrasound"](yuv)
    if "Error" in printed.getvalue():
        return None
    picked, n_real = echoprime_frames(bgr[..., ::-1])
    out = np.stack([fns["crop_and_scale"](np.ascontiguousarray(f), res=(size, size)) for f in picked])
    return out, n_real


def make_variants(frames, current, regions=None, variants=VARIANTS, size=224, cache_squash=None, cv2_frames=None):
    """All requested variants of one clip. `frames`: uint8 [T, H, W, 3] RGB, every decoded frame; `current`: arm B's
    clip as `mvit.ClipDecoder` returns it (uint8 [3, 16, size, size]) or None; `regions`: the DICOM's ultrasound
    regions or None; `cache_squash`: the clip as `mvit.ClipDecoder` returns it with `preprocessing: full_squash`
    (for `full_squash_cacheframes`); `cv2_frames`: the frames as OpenCV decodes them, RGB, or None when it could not
    (for `full_squash_cv2decode`, which then falls back). Returns ({variant: uint8 [3, 16, size, size]},
    {variant: n_real}, [variants that fell back])."""
    clips, real, fell_back = {}, {}, []
    picked, n = echoprime_frames(frames)
    t, h, w, _ = frames.shape

    def fallback(name):
        fell_back.append(name)
        return resize_square(pad_to_square(picked), size)

    for name in variants:
        if name == "current":
            if current is None:
                raise ValueError("The current variant needs arm B's clip.")
            clips[name], real[name] = current, FRAMES      # already padded with black, as arm B sees it
            continue
        if name == "full_squash_cacheframes":
            if cache_squash is None:
                raise ValueError("The full_squash_cacheframes variant needs arm B's full_squash clip.")
            clips[name], real[name] = cache_squash, FRAMES  # padded with black, as arm B would see it
            continue
        if name == "full_squash_blackpad":
            clips[name], real[name] = _to_clip(squash(picked, size), n, size), FRAMES   # black padding kept
            continue
        n_real = n
        if name == "current_epframes":
            out = mvit.crop_and_scale(picked, size, zoom=0.1).permute(1, 2, 3, 0).numpy()
        elif name == "centre_nozoom":
            out = mvit.crop_and_scale(picked, size, zoom=0).permute(1, 2, 3, 0).numpy()
        elif name == "full_pad":
            out = resize_square(pad_to_square(picked), size)
        elif name == "full_squash":
            out = squash(picked, size)
        elif name == "region_square":
            box = region_box(regions or [], h, w)
            if box is None:
                out = fallback(name)
            else:
                keep = np.zeros((h, w), dtype=bool)
                keep[box[1]:box[3], box[0]:box[2]] = True
                out = resize_square(square_around(picked, box, keep), size)
        elif name == "mask_and_crop":
            cropped = mask_and_crop(frames)
            out = fallback(name) if cropped is None else resize_square(echoprime_frames(cropped)[0], size)
        elif name == "echoprime_mask":
            result = echoprime_preprocess(frames, size)
            if result is None:
                out = fallback(name)
            else:
                out, n_real = result
        elif name == "full_squash_period1":
            sel, n_real = period1_frames(frames)
            out = squash(sel, size)
        elif name == "full_squash_middle":
            sel, n_real = middle_frames(frames)
            out = squash(sel, size)
        elif name == "full_squash_spread":
            sel, n_real = spread_frames(frames)
            out = squash(sel, size)
        elif name == "full_squash_bilinear_aa":
            out = squash_bilinear_aa(picked, size)
        elif name == "full_squash_cv2linear":
            out = squash_cv2(picked, size)
        elif name == "full_squash_cv2decode":
            if cv2_frames is None or len(cv2_frames) == 0:
                out = fallback(name)
            else:
                sel, n_real = echoprime_frames(cv2_frames)
                out = squash(sel, size)
        else:
            raise ValueError(f"Unknown variant {name!r}.")
        clips[name], real[name] = _to_clip(out, n_real, size), n_real
    return clips, real, fell_back


def normalize_padded(x, n_real, channel_order):
    """uint8 [B, 3, 16, H, W] -> float as `mvit.normalize`, frames at or after `n_real` (per clip) set to 0."""
    out = mvit.normalize(x, channel_order)
    for b, n in enumerate(n_real):
        if n < out.shape[2]:
            out[b, :, n:] = 0
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Decoding (workers) and scoring.

def decode_frames(path, max_frames=0):
    """Every frame (or the first `max_frames`) of a video as uint8 [T, H, W, 3] RGB, or None; the legacy YBR colour
    swap undone as the dataset loader does (`color_order.fix_legacy_colors`, off with ECHOJEPA_LEGACY_YBR_FIX=0).
    Decoder output, which could name the file, is discarded."""
    from decord import VideoReader, cpu
    from src.datasets.utils import color_order
    from src.datasets.video_dataset import _native_stderr_discarded
    try:
        with _native_stderr_discarded():
            vr = VideoReader(path, num_threads=1, ctx=cpu(0))
            n = len(vr) if not max_frames else min(len(vr), max_frames)
            if n == 0:
                return None
            frames = vr.get_batch(list(range(n))).asnumpy()
    except Exception:
        return None
    return color_order.fix_legacy_colors(frames, path, "RGB")


def decode_frames_cv2(path, max_frames=WINDOW):
    """The first `max_frames` frames (0 = all) as OpenCV decodes them, uint8 [T, H, W, 3] RGB, or None. OpenCV gives
    BGR: each frame is turned to RGB first, then the legacy YBR colour swap is undone in RGB order, as for
    `decode_frames`. Decoder output, which could name the file, is discarded."""
    import cv2
    from src.datasets.utils import color_order
    from src.datasets.video_dataset import _native_stderr_discarded
    try:
        with _native_stderr_discarded():
            cap = cv2.VideoCapture(path)
            out = []
            while not max_frames or len(out) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        frames = np.stack(out) if out else None
    except Exception:
        return None
    return color_order.fix_legacy_colors(frames, path, "RGB")


class VariantClips(torch.utils.data.Dataset):
    """Item i: (i, {variant: clip}, {variant: n_real}, [fallbacks], (H, W)) or (i, None, ...) when unreadable."""

    def __init__(self, video_paths, dicom_paths, variants=VARIANTS, max_frames=0, size=224, data_cfg=None):
        self.video_paths, self.dicom_paths = list(video_paths), list(dicom_paths)
        self.variants, self.max_frames, self.size = tuple(variants), max_frames, size
        data_cfg = data_cfg or {}
        frames, step = int(data_cfg.get("frames_per_clip", FRAMES)), int(data_cfg.get("frame_step", STEP))
        self.current = self.cache_squash = None
        if "current" in self.variants:
            self.current = mvit.ClipDecoder(self.video_paths, frames, step, size, float(data_cfg.get("zoom", 0.1)))
        if "full_squash_cacheframes" in self.variants:
            self.cache_squash = mvit.ClipDecoder(self.video_paths, frames, step, size, preprocessing="full_squash",
                                                 frame_read="cache")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, i):
        frames = decode_frames(self.video_paths[i], self.max_frames)
        current = self.current[i][1] if self.current is not None else None
        cache_squash = self.cache_squash[i][1] if self.cache_squash is not None else None
        if (frames is None or (self.current is not None and current is None)
                or (self.cache_squash is not None and cache_squash is None)):
            return i, None, None, None, None
        regions = None
        if "region_square" in self.variants:
            try:
                regions = read_regions(self.dicom_paths[i])
            except Exception:
                regions = None
        cv2_frames = decode_frames_cv2(self.video_paths[i]) if "full_squash_cv2decode" in self.variants else None
        clips, real, fell_back = make_variants(frames, current, regions, self.variants, self.size, cache_squash,
                                               cv2_frames)
        return i, clips, real, fell_back, frames.shape[1:3]


def _collate(items):
    return items


def summarize(table, archive_view, archive_probs=None, acq=None, human_parent=None):
    """One summary row from a probability table (`p_<old class>` columns, `predicted`), aligned arrays of the
    archive's view, its probabilities [N, 20] (optional), the acquisition category and the human label collapsed to
    the old classes (optional, NaN where unlabelled)."""
    pred = table.predicted.to_numpy()
    agree = pred == np.asarray(archive_view)
    row = {"n": len(table), "agreement_with_archive": float(agree.mean()) if len(agree) else float("nan"),
           "predicted_other_share": float((pred == "OTHER").mean()) if len(pred) else float("nan")}
    if acq is not None:
        acq = np.asarray(acq)
        for name in ("bmode", "color_flow"):
            sel = acq == name
            row[f"agreement_{name}"] = float(agree[sel].mean()) if sel.any() else float("nan")
    if archive_probs is not None:
        mine = table[[f"p_{c}" for c in vc.OLD_CLASSES]].to_numpy()
        row["mean_tv_to_archive"] = float(0.5 * np.abs(mine - np.asarray(archive_probs)).sum(axis=1).mean())
    if human_parent is not None:
        human = pd.Series(human_parent, dtype=object)
        sel = human.notna().to_numpy()
        row["n_human"] = int(sel.sum())
        row["human_accuracy_old_classes"] = (float((pred[sel] == human[sel].to_numpy()).mean()) if sel.any()
                                             else float("nan"))
    return row


@torch.no_grad()
def run(cfg, device, workers=8, variants=FIRST_ROUND, out=None, limit=None, max_frames=0, batch=16):
    from evals.view_classifier.io import check_colour_fix
    paths, b = cfg["paths"], cfg[mvit.ARM]
    check_colour_fix(cfg)
    out = out or os.path.join(paths["runs"], mvit.ARM, "zeroshot_variants")
    if os.path.exists(os.path.join(out, "summary.csv")):
        raise FileExistsError(f"{out} already holds a summary.csv: name another --out (results are never replaced).")
    os.makedirs(out, exist_ok=True)
    pool = pd.read_csv(paths["pool_clips"], dtype=str)
    if limit:
        pool = pool.iloc[:int(limit)]
    old = pd.read_csv(os.path.join(paths["inputs"], "pool_old_probs.csv"), dtype={"global_key": str})
    old = old.set_index("global_key").reindex(pool.global_key)
    model = mvit.load_original(paths["view_classifier"], b.get("weights_sha256")).to(device).eval()
    dataset = VariantClips(pool.video_path, pool.dicom_path, variants, max_frames, int(b.get("data", {}).get(
        "resolution", 224)), b.get("data", {}))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=workers, collate_fn=_collate,
                                         multiprocessing_context="spawn" if workers else None)
    probs = {(v, o): [] for v in variants for o in ORDERS}
    kept, fallbacks, sizes = [], {v: 0 for v in variants}, []
    start = time.time()
    for items in loader:
        items = [it for it in items if it[1] is not None]
        if not items:
            continue
        kept.extend(it[0] for it in items)
        sizes.extend(f"{int(it[4][1])}x{int(it[4][0])}" for it in items)
        for it in items:
            for v in it[3]:
                fallbacks[v] += 1
        for v in variants:
            x = torch.stack([it[1][v] for it in items]).to(device)
            n_real = [it[2][v] for it in items]
            for o in ORDERS:
                logits = model(normalize_padded(x, n_real, o))
                probs[(v, o)].append(torch.softmax(logits.float(), dim=-1).cpu())
    seconds = time.time() - start
    unreadable = len(pool) - len(kept)
    sel = pool.iloc[kept].reset_index(drop=True)
    archive_probs = old.iloc[kept][[f"p_{c}" for c in vc.OLD_CLASSES]].to_numpy()
    human = None
    if os.path.exists(paths.get("folds", "")):
        folds = pd.read_csv(paths["folds"], dtype={"clip_id": str, "label": str})
        _, parent = vc.from_config(cfg)
        human = sel.video_path.map(folds.set_index("clip_id").label.map(parent)).to_numpy(dtype=object)
    rows = []
    for v in variants:
        for o in ORDERS:
            p = torch.cat(probs[(v, o)]).numpy() if probs[(v, o)] else np.zeros((0, len(vc.OLD_CLASSES)))
            table = pd.DataFrame(p, columns=[f"p_{c}" for c in vc.OLD_CLASSES])
            table.insert(0, "clip_id", sel.video_path)
            table["predicted"] = np.asarray(vc.OLD_CLASSES)[p.argmax(axis=1)] if len(p) else []
            table.to_csv(os.path.join(out, f"probs_{v}_{o}.csv"), index=False)
            row = {"variant": v, "order": o, **summarize(table, sel.view, archive_probs, sel.acq_category, human),
                   "fallbacks": fallbacks[v]}
            for size in sorted(set(sizes)):
                m = np.asarray(sizes) == size
                row[f"agreement_{size}"] = float((table.predicted.to_numpy()[m] == sel.view.to_numpy()[m]).mean())
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    record = {"clips": len(pool), "scored": len(kept), "unreadable": unreadable, "seconds": seconds,
              "workers": workers, "max_frames": max_frames, "variants": list(variants),
              "frame_sizes": {s: sizes.count(s) for s in sorted(set(sizes))}}
    with open(os.path.join(out, "variants.json"), "w") as f:
        json.dump(record, f, indent=1)
    shown = summary.sort_values("agreement_with_archive", ascending=False)
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--variants", nargs="*", default=list(FIRST_ROUND), choices=VARIANTS)
    parser.add_argument("--second-round", action="store_true",
                        help="score full_squash (the anchor: it must reproduce the first round) and the second round's "
                             "variants instead of --variants; needs --out")
    parser.add_argument("--max-frames", type=int, default=0, help="frames decoded per clip for the masks; 0 = all")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N pool clips (a quick check)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    variants = tuple(args.variants)
    if args.second_round:
        if not args.out:
            parser.error("--second-round needs --out (the first round's folder is kept)")
        variants = ("full_squash",) + SECOND_ROUND
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, torch.device(args.device), args.workers, variants, args.out, args.limit, args.max_frames, args.batch)


if __name__ == "__main__":
    main()
