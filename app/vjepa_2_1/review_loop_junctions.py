# app/vjepa_2_1/review_loop_junctions.py

"""Render the wraparound junctions of looped short clips as local PNG strips, for clinical review of
cardiac-phase mismatch and probe movement where a short cine restarts (docs/temporal_padding.md)::

    python -m app.vjepa_2_1.review_loop_junctions --manifest /path/to/train.csv \\
        --out-dir <REVIEW_ROOT>/loop_junctions --num 24 --seed 0

Videos are visited in a seeded random order of the manifest's rows; the first `--num` that the
"loop" policy actually loops (fewer real frames than `--frames-per-clip`) are sampled exactly as the
trainer samples them (`VideoDataset`, no augmentation). `--out-dir` must be outside the repository,
since the images are clinical data. It receives:

- `line<manifest line>.png` per video: the frames on both sides of the first junction, large, then
  the whole looped clip; every tile is labelled `p<clip position> src <source frame index>`, and
  junctions are marked in red. Every junction of a clip joins the same two frames (last real ->
  first real), so the first one stands for all of them.
- `junctions.csv`: manifest_line, n_frames, fps, n_real, junctions, recycled_frame_frac,
  first_index, last_index, png.

Only aggregate counts are printed, never paths. Nothing is uploaded anywhere.
"""

import argparse
import csv
import math
import os
import sys
import warnings

import cv2
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from src.datasets.video_dataset import VideoDataset, _native_stderr_discarded, clip_index_stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The manifest scripts in data/ are standalone modules; `ensure_outside_repo` is shared with them.
if os.path.join(REPO_ROOT, "data") not in sys.path:
    sys.path.append(os.path.join(REPO_ROOT, "data"))
from build_manifests import ensure_outside_repo  # noqa: E402

CSV_NAME = "junctions.csv"
CSV_COLUMNS = ["manifest_line", "n_frames", "fps", "n_real", "junctions", "recycled_frame_frac", "first_index",
               "last_index", "png"]
RED, WHITE, BAND = (255, 0, 0), (255, 255, 255), (40, 40, 40)  # RGB, as decord decodes frames


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="loader manifest: one '<video path> 0' row per video (.csv)")
    p.add_argument("--out-dir", required=True, help="new or empty directory outside the repository")
    p.add_argument("--frames-per-clip", type=int, default=16)
    p.add_argument("--fps", type=float, default=8, help="target sampling rate, as data.fps")
    p.add_argument("--num", type=int, default=24, help="how many looped videos to render")
    p.add_argument("--seed", type=int, default=0, help="seed of the order in which manifest rows are visited")
    p.add_argument("--max-scan", type=int, default=None, help="stop after opening this many videos")
    p.add_argument("--context", type=int, default=3, help="frames shown on each side of the junction")
    p.add_argument("--tile", type=int, default=192, help="width in pixels of the junction tiles")
    args = p.parse_args(argv)
    if args.num < 1 or args.context < 1 or args.tile < 16 or args.frames_per_clip < 2:
        p.error("--num and --context must be >= 1, --tile >= 16 and --frames-per-clip >= 2.")
    return args


def open_video(path):
    """(VideoReader, average fps or NaN), or None if the video cannot be opened. Like the loader,
    decoder messages are discarded, since they may contain the path."""
    if not isinstance(path, str) or path.startswith("s3://") or not os.path.exists(path):
        return None
    try:
        with _native_stderr_discarded():
            vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    except Exception:
        return None
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = float("nan")
    return vr, fps


def tile(frame, width, label, junction):
    """A frame resized to `width`, under a band with its label; red band and edge at a junction."""
    h, w = frame.shape[:2]
    height = max(1, round(width * h / w))
    img = cv2.resize(np.ascontiguousarray(frame), (width, height))
    band = np.full((22, width, 3), BAND, np.uint8)
    scale = 0.45 if width >= 160 else 0.35
    cv2.putText(band, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, RED if junction else WHITE, 1, cv2.LINE_AA)
    out = np.concatenate([band, img])
    if junction:
        out[:, :4] = RED
    return out


def row(frames, indices, positions, width):
    """Tiles of the clip `positions`, side by side, with 2 px gaps."""
    tiles = []
    for p in positions:
        junction = p > 0 and indices[p] < indices[p - 1]
        tiles += [tile(frames[p], width, f"p{p} src {indices[p]}", junction), np.zeros((1, 2, 3), np.uint8)]
    height = max(t.shape[0] for t in tiles)
    return np.concatenate([np.pad(t, ((0, height - t.shape[0]), (0, 0), (0, 0))) for t in tiles[:-1]], axis=1)


def strip(frames, indices, n_real, context, width, title):
    """The PNG of one looped clip: title, the first junction (or the start of a static clip), the clip."""
    fpc = len(indices)
    if n_real < fpc and n_real > 1:
        positions = range(max(0, n_real - context), min(fpc, n_real + context))
    else:
        positions = range(min(fpc, 2 * context))
    rows = [row(frames, indices, positions, width), row(frames, indices, range(fpc), max(16, width // 2))]
    canvas_w = max(r.shape[1] for r in rows)
    head = np.full((28, canvas_w, 3), BAND, np.uint8)
    cv2.putText(head, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    gap = np.zeros((6, canvas_w, 3), np.uint8)
    parts = [head]
    for r in rows:
        parts += [gap, np.pad(r, ((0, 0), (0, canvas_w - r.shape[1]), (0, 0)))]
    return np.concatenate(parts)


def review(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    if os.path.exists(os.path.join(args.out_dir, CSV_NAME)):
        raise SystemExit(f"--out-dir already holds a {CSV_NAME}; choose a new directory so reviews never mix.")
    dataset = VideoDataset(data_paths=[args.manifest], frames_per_clip=args.frames_per_clip, fps=args.fps,
                           frame_step=None, temporal_padding="loop")
    if len(dataset) == 0:
        raise SystemExit("The manifest has no rows the loader can read (a '.csv' of '<path> 0' rows is expected).")
    os.makedirs(args.out_dir, exist_ok=True)

    # Long videos draw their window from np.random; seed it too, so a rerun visits the same clips.
    np.random.seed(args.seed)
    order = np.random.default_rng(args.seed).permutation(len(dataset))
    counts = {"opened": 0, "unreadable": 0, "not_looped": 0}
    records = []
    for index in order:
        if len(records) >= args.num or (args.max_scan is not None and counts["opened"] >= args.max_scan):
            break
        opened = open_video(dataset.samples[index])
        counts["opened"] += 1
        if opened is None:
            counts["unreadable"] += 1
            continue
        vr, video_fps = opened
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. unusable FPS: the fallback is documented, not news per video
            frames, clip_indices = dataset._sample_from_vr(vr, args.frames_per_clip)
        if clip_indices is None:
            counts["unreadable"] += 1
            continue
        indices = np.asarray(clip_indices[0])
        stats = {k: float(v[0]) for k, v in clip_index_stats(indices).items()}
        n_real = round(stats["unique_frame_frac"] * len(indices))
        if n_real >= len(indices):
            counts["not_looped"] += 1
            continue
        _, row_in_manifest = dataset.per_dataset_indices[index]
        line = row_in_manifest + 1
        png = f"line{line:07d}.png"
        fps_text = f"{video_fps:.2f}" if math.isfinite(video_fps) else "unknown"
        title = (f"manifest line {line} | {len(vr)} frames at {fps_text} fps | n_real {n_real} | "
                 f"junctions {int(stats['loop_junctions'])}")
        if n_real == 1:
            title += " | static: one frame repeated"
        # Saved with PIL: cv2's own PNG codec can break when decord's bundled zlib is loaded first.
        Image.fromarray(strip(np.asarray(frames), indices.tolist(), n_real, args.context, args.tile, title)).save(
            os.path.join(args.out_dir, png))
        records.append({
            "manifest_line": line, "n_frames": len(vr), "fps": video_fps if math.isfinite(video_fps) else "",
            "n_real": n_real, "junctions": int(stats["loop_junctions"]),
            "recycled_frame_frac": round(stats["recycled_frame_frac"], 4),
            "first_index": int(indices[0]), "last_index": int(indices[n_real - 1]), "png": png,
        })

    with open(os.path.join(args.out_dir, CSV_NAME), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return counts, records


def summary(counts, records):
    """Aggregate lines for the console: counts and distributions only, no paths or row numbers."""
    lines = [f"videos opened: {counts['opened']} (unreadable {counts['unreadable']}, "
             f"long enough to need no loop {counts['not_looped']})",
             f"looped videos rendered: {len(records)}"]
    if records:
        n_real = np.array([r["n_real"] for r in records])
        junctions = np.array([r["junctions"] for r in records])
        recycled = np.array([r["recycled_frame_frac"] for r in records])
        lines += [f"n_real: min {n_real.min()}, median {np.median(n_real):g}, max {n_real.max()}; "
                  f"static (one frame) {int((n_real == 1).sum())}",
                  f"junctions per clip: min {junctions.min()}, mean {junctions.mean():.2f}, max {junctions.max()}",
                  f"recycled frame fraction: mean {recycled.mean():.3f}"]
    return lines


def main(argv=None):
    args = parse_args(argv)
    counts, records = review(args)
    for line in summary(counts, records):
        print(line)
    print(f"Wrote {len(records)} strip(s) and {CSV_NAME} to --out-dir.")


if __name__ == "__main__":
    main()
