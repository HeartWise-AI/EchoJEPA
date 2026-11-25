#!/usr/bin/env python
"""Synthetic measurement overlays on clips labelled clean: what the screen's pixel and text signals catch.

Real measured loops are rare in the archive, and those found by their header cannot tell what the pixels add to it.
Here every base clip, labelled clean on a sheet and with a DICOM scan area, is copied as the archive encodes its
clips (H.264 4:4:4): once unchanged (`control`, the false alarms of re-encoding alone) and once per overlay drawn
on its frames (VARIANTS): caliper pairs (+ or x, with or without their dotted line), a single caliper, a traced
contour (dotted or solid), a results box with measurement text beside the image, and a caliper pair drawn from the
middle of the clip on. The copies keep the base clip's DICOM header, which names no measurement, so only the pixel
and text signals can find their marks: the visual detectors are scored apart from the header. The copies are
derived from patients' clips: they stay in an owner-only folder outside the repository and are never uploaded.

    make   the copies (clips/<global_key>.mp4), clips.csv for `clip_screen.py features`, truth.csv (global_key, base
           clip, variant) and overlays.json, from a screen's flags and the labels of its sheet.
    score  per variant, the share of copies each signal of `clip_screen.screen_signals` raises (the recall of the
           signals on that overlay; on `control`, their false alarms), with exact 95% intervals, and the decisions.

Example:
    python data/screen_overlays.py make --flags <screen_dir>/flags.parquet --labels <labels.csv> [...] \\
        --out-dir <overlay_dir>
    python data/clip_screen.py features --clips <overlay_dir>/clips.csv --out <overlay_dir>/screen/features.parquet
    <OCR environment>/bin/python -E -s data/clip_screen.py text --features <overlay_dir>/screen/features.parquet \\
        --out <overlay_dir>/screen/features_text.parquet
    python data/clip_screen.py flag --features <overlay_dir>/screen/features_text.parquet \\
        --reference <screen_dir>/reference.parquet --out <overlay_dir>/screen/flags.parquet
    python data/screen_overlays.py score --flags <overlay_dir>/screen/flags.parquet --truth <overlay_dir>/truth.csv

Console output is aggregate only.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_screen  # noqa: E402
import radar_pilot  # noqa: E402

# Sizes are for a frame 434 px high (the archive's most common, 636 x 434) and scale with the frame's height.
REFERENCE_HEIGHT = 434
CALIPER_ARM = 5       # px from a caliper's centre to the end of each arm.
DOT_STEP = 4          # px between the dots of a dotted line or contour.
PAIR_SPAN = 0.3       # a caliper pair's distance, as a share of the scan area's height.
TRACE_AXES = (0.16, 0.26)  # a traced contour's half axes, as shares of the scan area's width and height.
TEXT_SCALE = 0.4      # OpenCV font scale of the results box's text.
RESULTS = ("LVIDd 4.5 cm", "IVSd 1.0 cm", "EF 55 %")
# Ink colours (RGB): GE's calipers and contours are drawn in green, other vendors' in yellow; text in light grey.
COLORS = {"ge": (0, 230, 0), "other": (255, 255, 0), "text": (230, 230, 230)}
# Each variant: what is drawn, and from which share of the clip on (0: every frame).
VARIANTS = {
    "control": (None, 0.0),
    "caliper_pair_line": ("pair+line", 0.0),
    "caliper_pair": ("pair+", 0.0),
    "caliper_pair_x_line": ("pairx+line", 0.0),
    "single_caliper": ("single+", 0.0),
    "trace_dotted": ("trace_dotted", 0.0),
    "trace_solid": ("trace_solid", 0.0),
    "results_text": ("results", 0.0),
    "caliper_pair_line_late": ("pair+line", 0.5),
}
CRF = 18  # H.264 quality of the copies; the archive's clips are 4:4:4 H.264 of every bit rate.


def _scale(height):
    return height / REFERENCE_HEIGHT


def draw_caliper(img, cx, cy, arm, color, shape="+"):
    """A caliper glyph centred at (cx, cy), arms of `arm` px, one pixel wide: '+' or 'x'."""
    import cv2

    for dx, dy in (((1, 0), (0, 1)) if shape == "+" else ((1, 1), (1, -1))):
        cv2.line(img, (int(cx - dx * arm), int(cy - dy * arm)), (int(cx + dx * arm), int(cy + dy * arm)), color, 1)


def draw_dotted(img, p0, p1, color, step):
    """A dotted line from p0 to p1, one pixel every `step` px, its ends left clear (a caliper sits there)."""
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    for t in np.arange(2 * step, length - 2 * step + 1e-9, step) / max(length, 1e-9):
        img[int(round(p0[1] + t * (p1[1] - p0[1]))), int(round(p0[0] + t * (p1[0] - p0[0])))] = color


def draw_trace(img, center, axes, color, step=None):
    """A closed contour (an ellipse, as a traced cavity) around `center` with half `axes`: solid, or dotted every
    `step` px."""
    import cv2

    if step is None:
        cv2.ellipse(img, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), 0, 0, 360, color, 1)
        return
    perimeter = np.pi * (3 * (axes[0] + axes[1]) - np.sqrt((3 * axes[0] + axes[1]) * (axes[0] + 3 * axes[1])))
    for a in np.linspace(0, 2 * np.pi, int(perimeter / step), endpoint=False):
        img[int(round(center[1] + axes[1] * np.sin(a))), int(round(center[0] + axes[0] * np.cos(a)))] = color


def results_box(shape, bbox, scale):
    """Where a results box goes: (x, y) of its top-left corner, beside the scan area when the frame leaves room
    there (left, right or above it), else in the scan area's top-left corner, outside a sector."""
    import cv2

    h, w = shape[:2]
    (tw, th), _ = cv2.getTextSize(max(RESULTS, key=len), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE * scale, 1)
    box_w, box_h = tw + 8, (th + 6) * len(RESULTS) + 6
    x0, y0, x1, y1 = bbox
    if x0 >= box_w + 4:
        return 2, max(2, y0)
    if w - x1 >= box_w + 4:
        return x1 + 2, max(2, y0)
    if y0 >= box_h + 4:
        return 2, 2
    return x0 + 2, y0 + 2


def draw_results(img, corner, scale):
    """A results box at `corner`: a grey frame around RESULTS, one per line."""
    import cv2

    font, size = cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE * scale
    (tw, th), _ = cv2.getTextSize(max(RESULTS, key=len), font, size, 1)
    x, y = corner
    cv2.rectangle(img, (x, y), (x + tw + 8, y + (th + 6) * len(RESULTS) + 6), (120, 120, 120), 1)
    for i, line in enumerate(RESULTS):
        cv2.putText(img, line, (x + 4, y + (th + 6) * (i + 1)), font, size, COLORS["text"], 1, cv2.LINE_AA)


def overlay(frames, kind, start, bbox, vendor, rng):
    """`frames` (T, H, W, 3) RGB with the overlay `kind` drawn inside the scan area `bbox` (x0, y0, x1, y1) on
    every frame from `start` (a share of the clip) on; positions drawn from `rng` around the scan area's centre."""
    out = frames.copy()
    if kind is None:
        return out
    h = frames.shape[1]
    scale = _scale(h)
    color = COLORS["ge" if vendor in clip_screen.GE_HEADER_VENDORS else "other"]
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 + rng.uniform(-0.08, 0.08) * (x1 - x0)
    cy = (y0 + y1) / 2 + rng.uniform(-0.08, 0.08) * (y1 - y0)
    angle = rng.uniform(0, np.pi)
    half = PAIR_SPAN * (y1 - y0) / 2
    ends = [(cx - half * np.cos(angle), cy - half * np.sin(angle)),
            (cx + half * np.cos(angle), cy + half * np.sin(angle))]
    arm = max(2, int(round(CALIPER_ARM * scale)))
    corner = results_box(frames.shape[1:], bbox, scale) if kind == "results" else None
    for t in range(int(round(start * len(out))), len(out)):
        img = np.ascontiguousarray(out[t])
        if kind.startswith("pair"):
            for e in ends:
                draw_caliper(img, *e, arm, color, "x" if kind.startswith("pairx") else "+")
            if kind.endswith("line"):
                draw_dotted(img, ends[0], ends[1], color, max(2, int(round(DOT_STEP * scale))))
        elif kind == "single+":
            draw_caliper(img, cx, cy, arm, color)
        elif kind.startswith("trace"):
            axes = (TRACE_AXES[0] * (x1 - x0), TRACE_AXES[1] * (y1 - y0))
            draw_trace(img, (cx, cy), axes, color,
                       max(2, int(round(DOT_STEP * scale))) if kind == "trace_dotted" else None)
        elif kind == "results":
            draw_results(img, corner, scale)
        out[t] = img
    return out


def encode(frames, fps, path, ffmpeg, log_path, crf=CRF):
    """`frames` (T, H, W, 3) RGB as an H.264 4:4:4 MP4, as the archive encodes its clips, owner-only. ffmpeg's
    messages go to the private log."""
    t, h, w, _ = frames.shape
    with open(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab") as log:
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                        f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
                        "-crf", str(crf), "-pix_fmt", "yuv444p", path], input=frames.tobytes(),
                       stdout=subprocess.DEVNULL, stderr=log, check=True, timeout=600)
    os.chmod(path, 0o600)


def base_clips(flags, answers, per_vendor, seed):
    """The base clips: readable clips the screen keeps, motion read in their DICOM scan area, answered clean (no
    to every reason to exclude) on the sheet; up to `per_vendor` of each vendor, one per patient, seeded."""
    clean = answers[(answers[list(clip_screen.EXCLUSION_FIELDS)] == "no").all(axis=1)].index
    rows = flags[flags.global_key.isin(clean) & flags.error.isna() & (flags.decision == "keep")
                 & (flags.region_source == "dicom") & flags.region_bbox.notna()]
    rows = rows.sample(frac=1.0, random_state=seed).drop_duplicates("patient_id")
    return (rows.groupby("vendor", group_keys=False).head(per_vendor)
            .sort_values(["vendor", "global_key"]).reset_index(drop=True))


def run_make(args):
    out = args.out_dir
    clip_screen.ensure_outside_repo(os.path.join(out, "clips.csv"), "--out-dir")
    clip_screen.private_dir(out)
    if os.path.exists(os.path.join(out, "clips.csv")):
        raise SystemExit(f"{out} holds copies already: give a new --out-dir.")
    inputs = clip_screen.Inputs(__file__)
    flags = inputs.parquet(clip_screen.completed(args.flags, "flag", inputs))
    answers = clip_screen.read_labels(args.labels, inputs)
    bases = base_clips(flags, answers, args.per_vendor, args.seed)
    if bases.empty:
        raise SystemExit("No clip labelled clean with a DICOM scan area among --flags: nothing to copy.")
    from decord import VideoReader, cpu

    ffmpeg, secret = clip_screen.find_ffmpeg(), radar_pilot.load_secret(out)
    os.makedirs(os.path.join(out, "clips"), mode=0o700, exist_ok=True)
    log = os.path.join(out, "encoder.log")
    keep = [c for c in ("patient_id", "study_id", "study_date", "split", "view", "dicom_path", "vendor", "model",
                        "acq_category", "n_frames", "fps") if c in bases]
    clips, truth = [], []
    for i, base in bases.iterrows():
        if clip_screen.sha256(base.video_path) != base.video_sha256:
            raise SystemExit(f"Base clip {base.global_key} holds other video bytes than --flags screened: rerun "
                             "features and flag.")
        vr = VideoReader(base.video_path, ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        fps = float(base.fps) if pd.notna(base.get("fps")) and float(base.fps) > 0 else float(vr.get_avg_fps())
        bbox = json.loads(base.region_bbox)
        rng = np.random.default_rng([args.seed, i])
        for variant, (kind, start) in VARIANTS.items():
            path = os.path.join(out, "clips", f"{base.global_key}-{variant}.mp4")
            encode(overlay(frames, kind, start, bbox, base.vendor, rng), fps, path, ffmpeg, log, args.crf)
            key = radar_pilot.global_key(secret, path)
            clips.append({**{c: base[c] for c in keep}, "video_path": path, "global_key": key})
            truth.append({"global_key": key, "base": base.global_key, "variant": variant})
        print(f"[overlays]  base {i + 1}/{len(bases)}: {len(VARIANTS)} copies", flush=True)
    clips, truth = pd.DataFrame(clips), pd.DataFrame(truth)
    clips.to_csv(os.path.join(out, "clips.csv"), index=False)
    truth.to_csv(os.path.join(out, "truth.csv"), index=False)
    inputs.check()
    with open(os.open(os.path.join(out, "overlays.json"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump({"flags": inputs.digest(args.flags), "labels": {p: inputs.digest(p) for p in args.labels},
                   "seed": args.seed, "per_vendor": args.per_vendor, "crf": args.crf,
                   "bases": {k: int(n) for k, n in bases.vendor.value_counts().items()},
                   "variants": {k: {"draws": v[0], "from": v[1]} for k, v in VARIANTS.items()},
                   "code_sha256": clip_screen.code_sha256(__file__, clip_screen.__file__),
                   "code_commit": radar_pilot.git_commit(),
                   "outputs": {n: clip_screen.sha256(os.path.join(out, n)) for n in ("clips.csv", "truth.csv")}},
                  f, indent=1)
    print(f"[overlays]  {len(clips)} copies of {len(bases)} base clips {bases.vendor.value_counts().to_dict()}, "
          f"{len(VARIANTS)} variants each: {out}/clips.csv. Next: clip_screen.py features on it.")


def overlay_report(flags, truth):
    """Per variant, overall and per vendor group (GE's header convention or another), the copies each pixel or text
    signal raises (`clip_screen.screen_signals`) and any of them, with exact 95% intervals, and the decisions;
    unreadable copies counted apart."""
    df = truth.merge(flags, on="global_key", how="left", validate="one_to_one", indicator=True)
    if (df._merge != "both").any():
        raise SystemExit(f"{int((df._merge != 'both').sum())} copies of --truth are not in --flags: flag the "
                         "features of the copies' clips.csv.")
    readable = df[df.error.isna()].assign(
        group=lambda d: np.where(d.vendor.isin(clip_screen.GE_HEADER_VENDORS), "ge", "other"))
    signals = {name: raised.fillna(False).astype(bool) for name, (_, raised)
               in clip_screen.screen_signals(readable).items() if not name.startswith("header_")}
    signals["any_pixel_or_text"] = pd.concat(list(signals.values()), axis=1).any(axis=1)

    def part(rows):
        raised = {name: r.loc[rows.index] for name, r in signals.items()}
        return {"copies": int(len(rows)),
                "raised": {name: {"copies": int(r.sum()), "share": float(r.mean()) if len(r) else None,
                                  "share_95": clip_screen.interval_95(int(r.sum()), len(r))}
                           for name, r in raised.items()},
                "decisions": {k: int(n) for k, n in rows.decision.value_counts().items()}}

    return {"unreadable": int(df.error.notna().sum()),
            "variants": {variant: {**part(rows), "by_vendor": {g: part(p) for g, p in rows.groupby("group")}}
                         for variant, rows in readable.groupby("variant", sort=False)},
            "note": "control: re-encoded unchanged, so its raised shares are false alarms; every other variant is "
                    "measured by construction, its raised shares the signals' recall on that overlay"}


def run_score(args):
    inputs = clip_screen.Inputs(__file__)
    flags = inputs.parquet(clip_screen.completed(args.flags, "flag", inputs))
    truth = inputs.csv(args.truth)
    report = overlay_report(flags, truth)
    inputs.check()
    print(json.dumps(report, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--flags", required=True, help="a screen's flags whose clips a sheet labelled.")
    m.add_argument("--labels", required=True, nargs="+", help="that sheet's exported labels.")
    m.add_argument("--out-dir", required=True, help="a new owner-only folder outside the repository.")
    m.add_argument("--per-vendor", type=int, default=12, help="base clips per vendor, one per patient.")
    m.add_argument("--seed", type=int, default=20261022)
    m.add_argument("--crf", type=int, default=CRF, help="H.264 quality of the copies (lower is closer).")
    s = sub.add_parser("score")
    s.add_argument("--flags", required=True, help="clip_screen.py flag output for the copies' clips.csv.")
    s.add_argument("--truth", required=True, help="the copies' truth.csv.")
    args = p.parse_args(argv)
    umask = os.umask(0o077)
    try:
        {"make": run_make, "score": run_score}[args.cmd](args)
    finally:
        os.umask(umask)


if __name__ == "__main__":
    main()
