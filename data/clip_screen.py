# data/clip_screen.py

"""Screen echo clips before caption review: flag still clips and clips with visible measurements.

Reviewers only see moving B-mode or color Doppler clips without measurements. Single-frame DICOM objects never
became videos, but the video archive still holds frozen frames saved as several identical frames, and clips with
calipers, tracings or measurement text drawn over the image. Both flags come from the decoded video; when the clip
list gives `dicom_path`, the DICOM header's region boxes also give a scan area found without motion:

- **still:** almost nothing in the scan area moves (`scan_motion_frac`); for a clip without a scan area, almost
  nothing in the whole frame (`motion_frac`), each with its own threshold. Also recorded for calibration: the
  moving shares at other pixel thresholds and the share of near-duplicate consecutive frames (`duplicate_frac`).
- **measurement:** static ink over the image. Calipers, tracings and measurement text do not move while the tissue
  under them does (over the whole clip, or over half of it when they are drawn partway through), and they are bright
  or colored. Two scores: `inside_glyphs`, ink marks inside the moving region,
  a straight dotted or dashed line counting as one and each caliper (a + or an x) as one even when a line joins
  them (`inside_marks_raw` counts every piece), and `extra_cells`, ink beside the image (outside the scan area,
  else the moving region), anywhere on the screen, where reference clips of the same scanner
  layout have none (a measurement results box sits next to the sector, where other clips are black).

The policy (which flags exclude a clip) is set by the labels: an unreadable clip; a clip whose DICOM header gives no
image region that applies to its video, each header failure its own reason (HEADER_REASONS; the measured clips of the
pilot's labelled sample were, but for one, analysis screens without regions); and a still clip. The measurement flag
and its scores are computed and reported, but exclude only with `measurement_excludes` (and then so does a clip
whose layout has no baseline, 'incomplete'): on the labelled sample they flagged three quarters of the clips and
none of the measured ones.

Subcommands, every output outside the repository:
    features   pass 1, per clip: motion and ink statistics and a coarse map of where static ink is.
    flag       pass 2: per scanner layout, where static ink normally is, from reference clips that clearly move,
               have no marks inside the image and no block of ink beside it that most clips of their layout lack
               (a rule fixed apart from the thresholds); then both flags for
               every clip, and the policy: a clip excluded (unreadable, a header reason, still; measurement or
               incomplete only with `measurement_excludes`) is removed, and the pipeline asks no one.
               The baselines, with the clips behind each and those left out, go to a 0700 folder next to the flags.

Tuning the thresholds, outside the pipeline:
    sheet      private HTML pages for a person to answer four questions per clip (still, measurements, another
               reason to exclude, a cursor line), from a playable copy of the whole clip: the audit sample (a fixed
               random sample of flagged and unflagged clips, each with its inclusion probability, and a record of
               the screen's population); with --diagnostic, a random sample per role (screened or reference clip),
               acquisition mode and detector case; with --all-clips, every clip of a validation list from
               screen_validation.py, with its sampling probability; or, with --keys, chosen clips, the only sheet
               that shows the detector's view (on the others it would steer the answers). The frames carry
               burned-in identifiers: the pages and copies stay on the server, in a 0700 folder.
    calibrate  the screen against the answers of an audit, diagnostic or validation sheet, recomputed from the
               features behind the flags at any threshold set (--thresholds): each detector, and the policy
               (flagged clips are excluded, the others kept) with the share of ineligible clips among the kept ones,
               overall, per stratum and per acquisition mode; the cursor clips; for each minimum mark size, the
               screen recomputed from the reference clips' eligibility on, with per layout whether it has a
               baseline; and a diagnostic sheet's reference clips on their own.

Every clip is named by its `global_key`, an opaque key given in the clip list (the one Labelbox uses), on the sheet
and in the exported labels, so labels from several sheets name the same clips. A clip keeps its key when its video
is replaced in place, so the labels also name the video bytes each answer is of (`video_sha256`), and `calibrate`
uses only answers of the bytes its sheet's copies were made from.

Example:
    python data/clip_screen.py features --clips <clips.csv> --out <screen_dir>/features.parquet --workers 16
    python data/clip_screen.py features --clips <reference_clips.csv> --out <screen_dir>/reference.parquet \\
        --workers 16
    python data/clip_screen.py flag --features <screen_dir>/features.parquet \\
        --reference <screen_dir>/reference.parquet --out <screen_dir>/flags.parquet
    python data/clip_screen.py sheet --flags <screen_dir>/flags.parquet --out-dir <screen_dir>/audit
    python data/clip_screen.py calibrate --flags <screen_dir>/flags.parquet --sheet-dir <screen_dir>/audit \\
        --labels <labels.csv> [<more labels.csv> ...] [--thresholds '{"inside_glyphs": 2}']
    python data/clip_screen.py sheet --flags <screen_dir>/flags.parquet --out-dir <screen_dir>/look_01 \\
        --keys <chosen_clips.csv>

The thresholds below are provisional until `calibrate` has been run on labelled clips; pass changed values with
`flag --thresholds`. Every output records the SHA-256 of this script besides the commit, which misses uncommitted
edits. Every step reads each input once (`Inputs`), uses those bytes and records their SHA-256, and completes only
if every input still holds them at the end. A list column named like a pass 1 column (width, height) is kept as
listed_<name>. Clips are replaced in place on the archive, so `features` records each clip's video SHA-256 and the
DICOM regions it used, and `sheet` and `calibrate` refuse clips whose media no longer match (`changed_media`).
`features` also records its extractor: the code pass 1 runs, the constants it reads and the library versions; `flag`
refuses features extracted by other code or constants than its own, and features and reference extracted with other
library versions, and records the extractor behind the flags. Outputs go to folders only their owner can read. A run
first removes its earlier output and writes its record (`<out>.info.json`, with the output's SHA-256; a sheet's
`sheet.info.json`) last; a step refuses an input without a matching record, so the output of a run that stopped is
never used. Console output is aggregate only: the decoder's messages and the details of a failure name clip paths,
so they go to a private log (`<features out>.log`; a sheet folder's `decoder.log`, which also takes a failure to
read a clip while checking its media).
"""

import argparse
import base64
import contextlib
import datetime
import faulthandler
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

from build_manifests import REPO_ROOT, ensure_outside_repo, git_commit, sha256

MAX_FRAMES = 48       # frames read per clip, evenly spaced.
MOVE_STD = 6.0        # temporal std (levels) of any color channel above which a pixel moves.
MOVE_STDS = (3.0, 10.0)  # other such thresholds whose moving shares are recorded too, for calibration.
STATIC_STD = 2.0      # temporal std below which, in every channel, a pixel is static (H.264 noise stays under it).
SCAN_TYPES = (1, 2, 3, 4)  # DICOM Region Data Types holding image data: tissue, color flow, PW and CW Doppler.
SCAN_LEVEL = 12       # the scan area: what reaches this level (any channel) in some frame inside those regions,
SCAN_OPEN_PX = 7      # after an opening of this size that removes text and tick marks.
DUP_DIFF = 10.0       # between two read frames a pixel changed when a channel moved by more than this; the frames
DUP_SHARE = 0.002     # are near-duplicates when, after the ECG opening, less than this share of the area changed.
DOT_PX, DOT_FILL = 6, 0.6  # a dot or dash fits in DOT_PX x DOT_PX and fills DOT_FILL of its box (a plus does not).
CHAIN_GAP = 8         # dots at most this many px apart chain; a chain is one mark only if it is a straight line:
LINE_MIN_DOTS, LINE_MIN_SPAN, LINE_DEV = 3, 20, 1.5  # >= 3 dots over >= 20 px, each within 1.5 px of the line.
# A caliper is a + or an x (`caliper_centers`): four arms of at least CALIPER_ARM px from an inked centre, at least
# three of them ending, at about one length, within CALIPER_MAX_ARM px (the fourth may go on as the line of the
# distance). A component holding several calipers is several marks.
CALIPER_ARM, CALIPER_MAX_ARM = 2, 12
# A reference clip shapes its layout's baseline only if it clearly moves, by the still flag's measure (at least this
# share of its scan area moves, or without one of its frame), has at most REFERENCE_MAX_MARKS marks inside the
# image, and has fewer than REFERENCE_EXTRA_CELLS inked cells beside the image outside its layout's core ink (the
# cells at least CORE_SHARE of the layout's other such clips ink), which a measurement results box would have.
# Fixed, whatever the thresholds. Provisional, like the thresholds.
REFERENCE_MIN_SCAN_MOTION, REFERENCE_MIN_MOTION, REFERENCE_MAX_MARKS = 0.20, 0.05, 0
CORE_SHARE, REFERENCE_EXTRA_CELLS = 0.5, 8
INK_GRAY = 170        # a static pixel at least this bright, or
INK_CHROMA = 60       # at least this colored (max - min channel), is drawn ink, not tissue.
OPEN_PX = 5           # opening applied to the moving mask: removes the ECG trace and blinking cursors.
REGION_SHARE = 0.25   # moving components at least this share of the largest one belong to the imaging region.
HULL_MARGIN = 6       # px eroded from the imaging region: its border is the sector edge, static and bright.
MIN_INK_PX = 3        # smaller ink components are noise.
# Marks shown during only part of a clip (calipers drawn halfway through) are not static over the whole clip, so ink
# inside the image is also looked for in windows of half the read frames, one every quarter of the clip, for clips of
# at least 2 x WINDOW_MIN_FRAMES read frames: a mark static through three quarters of the clip, or through its first,
# middle or last half, lies wholly in one. Shorter windows see slowly moving bright tissue (a low-EF wall) as static
# ink. Each window has its own moving region, and a window in which at least FROZEN_WINDOW_SHARE of it is static is
# frozen: its static speckle cannot be told from ink, so its marks inside the image do not count (`frozen_windows`).
WINDOW_MIN_FRAMES, FROZEN_WINDOW_SHARE = 6, 0.5
# A mark does not count in or next to a frozen patch of the image (`frozen_patches`): a FROZEN_PATCH_PX square at
# least FROZEN_DENSITY of whose pixels are static and not dark (at least TISSUE_GRAY), widened by FROZEN_MARGIN_PX.
# A part of the image frozen while the rest moves (in a window, or throughout) has static speckle, which cannot be
# told from ink; calipers, dots and text over moving tissue or the dark blood pool are too thin to fill the square.
FROZEN_PATCH_PX, FROZEN_DENSITY, FROZEN_MARGIN_PX, TISSUE_GRAY = 15, 0.6, 8, 40
# Other static levels at which the marks are counted too, for calibration: after H.264 encoding, an overlay over
# moving tissue varies by more than STATIC_STD.
STATIC_STDS = (4.0,)
LINE_LEN, LINE_FILL = 40, 0.15  # a component this long and this sparse is a line, or a traced contour:
# line art (a cursor, a color box outline, a scale) is straight lines of at least STRAIGHT_PX px each, which leave at
# most 1 - LINE_ART_SHARE of its pixels once taken away (`line_art`); a contour bends too much for that, and is a mark.
STRAIGHT_PX, LINE_ART_SHARE = 30, 0.9
CELL = 8              # px per side of a cell of the ink map.
CELL_INK = 3          # ink pixels that ink a cell.
COMMON_SHARE = 0.10   # a cell inked in at least this share of a layout's clips is normal for that layout.
MIN_LAYOUT_CLIPS = 25  # fewer clips give no layout baseline (extra_cells is then not used).
LAYOUT_COLUMNS = ("vendor", "model", "width", "height", "acq_category")
KEY_PATTERN = r"[A-Za-z0-9._-]+"  # a global_key is safe in HTML, CSS selectors and CSV without escaping.
# The sheet's questions: the first three are reasons to exclude a clip (eligible: no 'yes' to any); the last asks
# whether a cursor line (Doppler or M-mode) is drawn over the image, which is no reason to exclude, so that the
# screen's errors on such clips are reported on their own.
EXCLUSION_FIELDS = ("still", "measurement", "other_exclusion")
LABEL_FIELDS = (*EXCLUSION_FIELDS, "cursor")
# Minimum sizes of a mark inside the image (`inside_mark_px`) that `calibrate` tries, each with its own reference
# eligibility, baselines and extra_cells.
INSIDE_MARK_SWEEP = (MIN_INK_PX, 5, 8, 12, 16)
# A diagnostic sample's cases (`diagnostic_cases`): a mark inside the image of at least this many ink pixels is large.
DIAGNOSTIC_LARGE_PX = 12
# A decoder message naming the machine's resources: the run stops instead of excluding the clip.
RESOURCE_ERROR = re.compile(r"alloc|memory|resource|too many open files|no space|timed? ?out|interrupted", re.I)
MOTION_COLUMNS = ("motion_frac", *(f"motion_frac_std{s:g}" for s in MOVE_STDS), "scan_motion_frac",
                  *(f"scan_motion_frac_std{s:g}" for s in MOVE_STDS))
# Pass 1 columns, present even when no clip could be read.
NUMERIC_FEATURES = ("n_frames_total", "frames_read", "height", "width", *MOTION_COLUMNS, "duplicate_frac",
                    "scan_frac", "region_fill", "region_frac", "inside_glyphs", "inside_marks_raw", "inside_ink_px",
                    "inside_glyphs_clip", *(f"inside_glyphs_static{s:g}" for s in STATIC_STDS), "windows",
                    "frozen_windows", "ink_cells")
FEATURE_COLUMNS = ("error", "video_sha256", *NUMERIC_FEATURES, "dicom_status", "dicom_regions", "inside_marks",
                   "region_bbox", "cells_shape", "cells")

# still: the scan area's moving share below still_scan_motion_frac, or for a clip without a scan area the frame's
# below still_motion_frac. measurement: at least inside_glyphs marks inside the image (two calipers of one distance
# are two marks, their dotted line a third when drawn) or at least extra_cells unusual inked cells beside it. A mark
# inside the image counts only from inside_mark_px ink pixels (a straight dotted line: its dots together), in the
# screened clips and in the reference clips alike; MIN_INK_PX counts every mark. It applies to marks inside the
# image only: the cell map beside it, where results text is, keeps every ink component of MIN_INK_PX.
# The policy. A clip is eligible only when its DICOM header gives image regions that apply to its video (dicom_status
# 'ok'); a header without them, or one that cannot be used, is its own reason to exclude. On the pilot's labelled
# clips, all still clips and all measured ones but one lacked them (analysis screens). The pixel measurement rules
# (`inside_glyphs`, `extra_cells`) flagged about three quarters of the pilot's clips, none of them measured in the
# labels, so they exclude nothing unless `measurement_excludes` is set, and nor does a missing layout baseline
# ('incomplete'); their scores and the `measurement` flag are computed and reported either way.
HEADER_REASONS = {"no_regions": "no_image_region", "size_mismatch": "header_size_mismatch",
                  "unreadable_header": "header_unreadable", "": "no_header"}
THRESHOLDS = {"still_scan_motion_frac": 0.05, "still_motion_frac": 0.01, "inside_glyphs": 2, "extra_cells": 8,
              "inside_mark_px": MIN_INK_PX, "measurement_excludes": False}
# The labelled calibration the THRESHOLDS above were chosen with, or None while they are provisional: {"date": ...,
# "report": SHA-256 of `calibrate`'s report, "screen": the `screen` it reports, the SHA-256 of the extractor and flag
# rule its flags were made with, "thresholds": the thresholds it reports, which it measured}. `flag` records it only
# with flags of that same screen at those same thresholds (changing THRESHOLDS alone leaves flags provisional), and
# `radar_pilot.py candidates` refuses provisional flags unless told the run is a trial (--provisional-screen).
CALIBRATION = None
# Still scores `calibrate` sweeps: (column, a clip is still when the score is 'below' or 'at_least' a value, values).
STILL_SWEEPS = (*((c, "below", (0.002, 0.005, 0.01, 0.02, 0.05)) for c in MOTION_COLUMNS if not c.startswith("scan")),
                *((c, "below", (0.01, 0.02, 0.05, 0.1, 0.2)) for c in MOTION_COLUMNS if c.startswith("scan")),
                ("duplicate_frac", "at_least", (0.5, 0.8, 0.9, 1.0)))


# --------------------------------------------------------------------------- #
#  Pass 1: per-clip features
# --------------------------------------------------------------------------- #

def frame_indices(n, max_frames=MAX_FRAMES):
    return np.unique(np.linspace(0, n - 1, num=min(n, max_frames)).round().astype(int))


def read_frames(video, max_frames=MAX_FRAMES):
    """`(T, H, W, 3)` uint8 RGB frames, evenly spaced over the clip, from its bytes (or its path)."""
    from decord import VideoReader, cpu

    vr = VideoReader(io.BytesIO(video) if isinstance(video, bytes) else video, ctx=cpu(0), num_threads=1)
    return vr.get_batch(list(frame_indices(len(vr), max_frames))).asnumpy(), len(vr)


def dicom_bounds(path, size):
    """The boxes `(x0, y0, x1, y1)` of the image-data regions (SCAN_TYPES) in a DICOM header's Sequence of
    Ultrasound Regions (PS3.3 C.8.5.5), and a status: 'ok', 'no_regions', 'size_mismatch' (the DICOM image is not
    the video's `size` (h, w), so its coordinates do not apply) or 'unreadable_header'. A file that cannot be
    opened stops the run, like a missing video."""
    import pydicom

    with open(path, "rb") as f:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True,
                                 specific_tags=["Rows", "Columns", "SequenceOfUltrasoundRegions"])
            shape = (int(ds.get("Rows") or 0), int(ds.get("Columns") or 0))
            items = list(ds.get("SequenceOfUltrasoundRegions") or [])
        except (MemoryError, OSError):  # the machine or the storage, not the header.
            raise
        except Exception:
            return None, "unreadable_header"
    if shape != tuple(size):
        return None, "size_mismatch"
    h, w = size
    boxes = []
    for item in items:
        try:
            if int(item.RegionDataType) not in SCAN_TYPES:
                continue
            x0, y0 = max(int(item.RegionLocationMinX0), 0), max(int(item.RegionLocationMinY0), 0)
            x1, y1 = min(int(item.RegionLocationMaxX1), w - 1), min(int(item.RegionLocationMaxY1), h - 1)
        except (AttributeError, TypeError, ValueError):
            continue
        if x0 < x1 and y0 < y1:
            boxes.append((x0, y0, x1, y1))
    return (boxes, "ok") if boxes else (None, "no_regions")


def imaging_region(mask):
    """Convex hull of the largest component of `mask` and of those at least REGION_SHARE of its size (a dual
    display). Empty when the mask is."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    region = np.zeros(mask.shape, np.uint8)
    if n <= 1:
        return region.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    for i in np.flatnonzero(areas >= REGION_SHARE * areas.max()) + 1:
        pts = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(np.int32)
        cv2.fillPoly(region, [cv2.convexHull(pts)], 1)
    return region.astype(bool)


def scan_area(f, boxes):
    """The scan area inside the DICOM region `boxes`, found from image content, not motion, so that a frozen clip
    has one too: what reaches SCAN_LEVEL in some frame, opened to drop text and tick marks, then `imaging_region`.
    `f` is the `(T, H, W, 3)` float frames."""
    import cv2

    inside = np.zeros(f.shape[1:3], bool)
    for x0, y0, x1, y1 in boxes:
        inside[y0:y1 + 1, x0:x1 + 1] = True
    content = (f.max(axis=0).max(axis=2) >= SCAN_LEVEL) & inside
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SCAN_OPEN_PX, SCAN_OPEN_PX))
    content = cv2.morphologyEx(content.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    return imaging_region(content) & inside


def duplicate_share(f, area):
    """The share of consecutive read frames that are near-duplicates inside `area`: less than DUP_SHARE of it
    changed by more than DUP_DIFF in some channel, after the opening that removes the ECG trace and cursors. NaN
    for a single frame or an empty area."""
    import cv2

    if len(f) < 2 or not area.any():
        return float("nan")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))
    same = []
    for a, b in zip(f[:-1], f[1:]):
        changed = (np.abs(b - a).max(axis=2) > DUP_DIFF).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel).astype(bool)
        same.append(changed[area].mean() < DUP_SHARE)
    return float(np.mean(same))


def is_dot(mark):
    """A dot or dash: small and compact. A plus-shaped caliper fills too little of its box to be one."""
    _, _, w, h, area, *_ = mark
    return max(w, h) <= DOT_PX and area >= DOT_FILL * w * h


def straight_line(members):
    """Whether the dots form one straight dotted or dashed line (a cursor or guide line)."""
    if len(members) < LINE_MIN_DOTS:
        return False
    centers = np.array([(m[5], m[6]) for m in members])
    centers -= centers.mean(axis=0)
    _, _, axes = np.linalg.svd(centers, full_matrices=False)
    along, across = centers @ axes[0], centers @ axes[1]
    return along.max() - along.min() >= LINE_MIN_SPAN and np.abs(across).max() <= LINE_DEV


def chain_dots(marks, shape):
    """`marks` with every straight dotted or dashed line merged into one mark: a cursor or guide line is one drawn
    object, not one mark per dot. Dots (`is_dot`) at most CHAIN_GAP px apart chain; a chain that is not a straight
    line (`straight_line`: a dotted trace, calipers close together) keeps one mark per piece, so a measurement
    keeps its weight."""
    import cv2

    dots = [m for m in marks if is_dot(m)]
    if not dots:
        return list(marks)
    mask = np.zeros(shape, np.uint8)
    for x, y, w, h, *_ in dots:
        mask[y:y + h, x:x + w] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CHAIN_GAP + 1, CHAIN_GAP + 1))
    _, labels = cv2.connectedComponents(cv2.dilate(mask, kernel), connectivity=8)
    chains = {}
    for m in dots:
        chains.setdefault(int(labels[m[1], m[0]]), []).append(m)
    out = [m for m in marks if not is_dot(m)]
    for members in chains.values():
        if not straight_line(members):
            out += members
            continue
        x0, y0 = min(m[0] for m in members), min(m[1] for m in members)
        x1, y1 = max(m[0] + m[2] for m in members), max(m[1] + m[3] for m in members)
        area = sum(m[4] for m in members)
        out.append((x0, y0, x1 - x0, y1 - y0, area, sum(m[5] * m[4] for m in members) / area,
                    sum(m[6] * m[4] for m in members) / area, "line"))
    return out


def caliper_centers(ink):
    """The centres (cx, cy) of the calipers in `ink`: a + or an x, whose four arms reach at least CALIPER_ARM px
    from an inked centre, with at most one of the quarters between them inked at CALIPER_ARM px; at least three
    arms end within CALIPER_MAX_ARM px, at lengths within CALIPER_ARM px of one another; and ink leaves the glyph at
    most once, across the square ring just beyond CALIPER_MAX_ARM. The line of a measured distance, drawn from a
    caliper, inks one quarter or goes on as one arm, and leaves once. Two lines crossing leave four times, at any
    angle; a corner or a T has fewer arms; a filled glyph has no empty quarters; the ticks of a scale differ in
    length from the scale's own line."""
    import cv2

    pad, (h, w) = CALIPER_MAX_ARM + 2, ink.shape
    padded = np.pad(ink, pad)
    r = CALIPER_MAX_ARM + 1
    side, edge = np.arange(-r, r), np.full(2 * r, r)
    ring = np.concatenate([np.stack(p, axis=1) for p in (  # (dy, dx) around the square ring, in order.
        (-edge, side), (side, edge), (edge, -side), (-side, -edge))])
    found = np.zeros((h, w), bool)
    axes, diagonals = ((0, 1), (1, 0), (0, -1), (-1, 0)), ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for arms, quarters in ((axes, diagonals), (diagonals, axes)):  # a +, then an x.
        candidate = ink.copy()
        for dy, dx in arms:
            for d in range(1, CALIPER_ARM + 1):
                candidate &= padded[pad + d * dy:pad + d * dy + h, pad + d * dx:pad + d * dx + w]
        inked_quarters = sum(padded[pad + CALIPER_ARM * dy:pad + CALIPER_ARM * dy + h,
                                    pad + CALIPER_ARM * dx:pad + CALIPER_ARM * dx + w].astype(np.uint8)
                             for dy, dx in quarters)
        ys, xs = np.nonzero(candidate & (inked_quarters <= 1))
        if not len(ys):
            continue
        lengths = []  # the inked run along each arm, up to CALIPER_MAX_ARM + 1.
        for dy, dx in arms:
            run, length = np.ones(len(ys), bool), np.zeros(len(ys), int)
            for d in range(1, CALIPER_MAX_ARM + 2):
                run &= padded[pad + ys + d * dy, pad + xs + d * dx]
                length += run
            lengths.append(length)
        lengths = np.stack(lengths, axis=1)
        ending = lengths <= CALIPER_MAX_ARM
        ended = np.where(ending, lengths, 0)
        spread = ended.max(axis=1) - np.where(ending, lengths, CALIPER_MAX_ARM + 1).min(axis=1)
        crossing = padded[pad + ys[:, None] + ring[:, 0], pad + xs[:, None] + ring[:, 1]]
        exits = (crossing & ~np.roll(crossing, 1, axis=1)).sum(axis=1)
        ok = (ending.sum(axis=1) >= 3) & (spread <= CALIPER_ARM) & (exits <= 1)
        found[ys[ok], xs[ok]] = True
    n, _, _, centroids = cv2.connectedComponentsWithStats(cv2.dilate(found.astype(np.uint8), np.ones((3, 3), np.uint8)))
    return [(float(cx), float(cy)) for cx, cy in centroids[1:n]]


def line_art(component):
    """Whether a long sparse ink component (bool mask) is straight lines only (a cursor, a color box outline, lines
    crossing): straight lines of at least STRAIGHT_PX pixels, found strongest first (Hough transform), fitted to their
    pixels and taken away, leave at most 1 - LINE_ART_SHARE of its pixels. A traced contour has no straight run that
    long, unless very flat, and keeps most of its pixels."""
    import cv2

    rest = component.astype(np.uint8)
    total = int(rest.sum())
    for _ in range(12):
        found = cv2.HoughLines(rest, 1, np.pi / 360, STRAIGHT_PX)
        if found is None:
            break
        rho, theta = found[0][0]
        ys, xs = np.nonzero(rest)
        near = np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho) <= 2
        if near.sum() < 2:
            break
        # The line through those pixels (the transform's bins are coarse over a long line), then all within 1.5 px.
        vx, vy, x0, y0 = cv2.fitLine(np.stack([xs[near], ys[near]], 1).astype(np.float32), cv2.DIST_L2, 0, 0.01,
                                     0.01).ravel()
        on = np.abs((xs - x0) * vy - (ys - y0) * vx) <= 1.5
        if not on.any():
            break
        rest[ys[on], xs[on]] = 0
    return rest.sum() <= (1 - LINE_ART_SHARE) * total


def ink_components(ink):
    """Ink components that are marks (glyphs, calipers, dashes, traced contours), not line art: list of
    (x, y, w, h, area, cx, cy, kind), kind 'caliper', 'piece' or 'trace' (a long sparse component that is not
    `line_art`: a contour traced over the image), or 'line' once `chain_dots` joins a dotted line. A component
    holding several calipers (`caliper_centers`: calipers joined by the line of their distance) gives one mark per
    caliper, and so does a long sparse line holding any; line art alone (a cursor, a color box outline, a scale)
    gives none."""
    import cv2

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    calipers = {}
    for cx, cy in caliper_centers(ink):
        calipers.setdefault(int(labels[int(round(cy)), int(round(cx))]), []).append((cx, cy))
    marks = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < MIN_INK_PX:
            continue
        line = max(w, h) >= LINE_LEN and area / float(w * h) < LINE_FILL
        own = calipers.get(i, [])
        if len(own) > 1 or (line and own):
            for cx, cy in own:  # a box around each caliper, within the component.
                px, py = int(round(cx)), int(round(cy))
                x0, y0 = max(px - CALIPER_MAX_ARM, x), max(py - CALIPER_MAX_ARM, y)
                x1, y1 = min(px + CALIPER_MAX_ARM + 1, x + w), min(py + CALIPER_MAX_ARM + 1, y + h)
                marks.append((x0, y0, x1 - x0, y1 - y0, int((labels[y0:y1, x0:x1] == i).sum()), cx, cy, "caliper"))
            continue
        if line and line_art(labels[y:y + h, x:x + w] == i):
            continue
        marks.append((x, y, w, h, area, float(centroids[i][0]), float(centroids[i][1]), "trace" if line else "piece"))
    return marks


def frozen_patches(static, mean_gray):
    """Where the image itself is frozen rather than drawn on: at least FROZEN_DENSITY of the pixels of a
    FROZEN_PATCH_PX square are `static` and not dark (`mean_gray` at least TISSUE_GRAY), widened by FROZEN_MARGIN_PX
    to the patch's border. Calipers, dots and text are too thin to fill such a square; frozen speckle fills it."""
    import cv2

    dense = cv2.blur((static & (mean_gray >= TISSUE_GRAY)).astype(np.float32), (FROZEN_PATCH_PX,) * 2)
    margin = np.ones((2 * FROZEN_MARGIN_PX + 1,) * 2, np.uint8)
    return cv2.dilate((dense >= FROZEN_DENSITY).astype(np.uint8), margin).astype(bool)


def clip_features(frames, boxes=None, dicom_status=""):
    """Features of one clip from its `(T, H, W, 3)` uint8 frames, and its ink cell map.

    Two regions:
    - the moving region, where marks count (`inside_glyphs`): static ink can only be told from bright frozen
      speckle where the tissue under it moves, so a frozen part of the image (the reference image of a dual
      display) must not count;
    - the scan area inside the DICOM region `boxes`, found without motion (`scan_frac`; NaN without boxes). Motion
      is measured over the whole frame (`motion_frac`, which the still flag uses) and over the scan area
      (`scan_motion_frac`, NaN without one: over the moving region it would be circular). Duplicate frames are
      looked for in the scan area (else the whole frame), and extra ink near its box (else the moving region's).

    Returns `(features, cells, details)`: `cells` is the bool map of inked CELL x CELL cells, `details` the
    masks, marks and per-pixel change `sheet` draws.
    """
    import cv2

    t, h, w, _ = frames.shape
    f = frames.astype(np.float32)
    gray = f.mean(axis=3)
    chroma = f.max(axis=3) - f.min(axis=3)
    # Per channel: red turning blue keeps the gray level and the chroma, so neither would see the change.
    change = f.std(axis=0).max(axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))
    moving = {s: cv2.morphologyEx((change > s).astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
              for s in (MOVE_STD, *MOVE_STDS)}
    def regions(moving_mask):
        """The imaging region of a moving mask, and that region eroded by HULL_MARGIN, where marks count."""
        found = imaging_region(moving_mask)
        eroded = cv2.erode(found.astype(np.uint8), np.ones((2 * HULL_MARGIN + 1,) * 2, np.uint8)).astype(bool)
        return found, eroded

    region, inner = regions(moving[MOVE_STD])
    scan = scan_area(f, boxes) if boxes else np.zeros((h, w), bool)

    def inked(static, part, inside_of):
        """The ink of the frames `part` (static, and bright or colored), its marks, and the marks inside
        `inside_of` and outside frozen patches of the image (`frozen_patches`), before and after straight dotted
        lines are joined."""
        mean_gray = gray[part].mean(axis=0)
        ink = static & ((mean_gray >= INK_GRAY) | (chroma[part].mean(axis=0) >= INK_CHROMA))
        marks = ink_components(ink)
        counted = inside_of & ~frozen_patches(static, mean_gray)
        raw = [m for m in marks if counted[min(int(round(m[6])), h - 1), min(int(round(m[5])), w - 1)]]
        return ink, marks, raw, chain_dots(raw, (h, w))

    mark_mask = np.zeros((h, w), bool)

    def add_to_mark_mask(ink, marks):
        for x, y, mw, mh, *_ in marks:
            mark_mask[y:y + mh, x:x + mw] |= ink[y:y + mh, x:x + mw]

    ink, marks, inside_raw, inside = inked(change < STATIC_STD, slice(None), inner)
    add_to_mark_mask(ink, marks)
    inside_clip, evidence = len(inside), {"clip": [[m[4], m[7]] for m in inside], "windows": []}
    # Ink static in part of the clip only, in windows of half the clip, one every quarter. Beside the image, every
    # window's marks join the cell map (a results box shown partway through). Inside it, the most marks in a window
    # that is not frozen count, within the region that moves in that window: a part of the image frozen during it
    # has static speckle, which cannot be told from ink.
    size = t // 2
    starts = sorted({*range(0, t - size + 1, max(size // 2, 1)), t - size}) if t >= 2 * WINDOW_MIN_FRAMES else []
    windows, frozen_windows = [], 0
    for a in starts:
        part = slice(a, a + size)
        std = f[part].std(axis=0).max(axis=2)
        _, inner_w = regions(cv2.morphologyEx((std > MOVE_STD).astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool))
        frozen = not inner_w.any() or (std < STATIC_STD)[inner_w].mean() >= FROZEN_WINDOW_SHARE
        ink_w, marks_w, raw, marked = inked(std < STATIC_STD, part, np.zeros_like(inner_w) if frozen else inner_w)
        add_to_mark_mask(ink_w, marks_w)
        frozen_windows += frozen
        if not frozen:
            windows.append((part, std, inner_w))
            evidence["windows"].append([[m[4], m[7]] for m in marked])
            if len(marked) > len(inside):
                inside_raw, inside = raw, marked
    at_levels = {}
    for level in STATIC_STDS:
        counts = [len(inked(change < level, slice(None), inner)[3])]
        counts += [len(inked(std < level, part, inner_w)[3]) for part, std, inner_w in windows]
        at_levels[f"inside_glyphs_static{level:g}"] = max(counts)

    # The cell map holds only ink beside the image: ink over it is inside_glyphs' business, and a frozen part of
    # the image (static speckle) must not read as a results box. Without a scan area only the moving region is
    # known to be image.
    around = scan if scan.any() else region
    beside = ~cv2.dilate(around.astype(np.uint8), np.ones((2 * HULL_MARGIN + 1,) * 2, np.uint8)).astype(bool)
    ch, cw = h // CELL, w // CELL
    cell_counts = (mark_mask & beside)[:ch * CELL, :cw * CELL].reshape(ch, CELL, cw, CELL).sum(axis=(1, 3))
    cells = cell_counts >= CELL_INK
    if around.any():
        ys, xs = np.nonzero(around)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    else:
        bbox = None

    shares = {}
    for s, mask in moving.items():
        suffix = "" if s == MOVE_STD else f"_std{s:g}"
        shares[f"motion_frac{suffix}"] = float(mask.mean())
        shares[f"scan_motion_frac{suffix}"] = float(mask[scan].mean()) if scan.any() else float("nan")
    boxes_area = sum((x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in boxes) if boxes else 0
    features = {
        "frames_read": int(t), "height": int(h), "width": int(w), **shares,
        # The opening inside duplicate_share removes the ECG trace when the whole frame is used.
        "duplicate_frac": duplicate_share(f, scan if scan.any() else np.ones((h, w), bool)),
        "dicom_status": dicom_status,
        "scan_frac": float(scan.mean()) if boxes else float("nan"),
        "region_fill": float(scan.sum() / boxes_area) if boxes_area else float("nan"),
        "region_frac": float(region.mean()),
        # The most marks inside the image, over the whole clip or in any window that is not frozen; the whole
        # clip's count alone; the windows looked at and those frozen.
        "inside_glyphs": len(inside), "inside_marks_raw": len(inside_raw),
        # Each mark counted inside, [pixels, kind], over the whole clip and in each window that is not frozen: a
        # minimum mark size (`at_mark_size`) can be tried without extracting again.
        "inside_marks": json.dumps(evidence),
        "inside_ink_px": int(sum(m[4] for m in inside)), "inside_glyphs_clip": inside_clip, **at_levels,
        "windows": len(starts), "frozen_windows": frozen_windows,
        "ink_cells": int(cells.sum()),
        "region_bbox": json.dumps(bbox),
    }
    details = {"region": region, "scan": scan, "inside": inside, "cells": cells, "change": change,
               "moving": moving[MOVE_STD]}
    return features, cells, details


_CLIP_SECONDS = 0  # in a worker, the deadline per clip (`_timed_worker`).


def _init_worker(log_path, clip_seconds):
    """Pool initializer: the worker's stderr goes to the private log. The decoder's own messages name the video
    path, which can carry identifiers, and the console stays aggregate only. A crash in native code leaves its
    Python traceback there too."""
    global _CLIP_SECONDS
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(fd, 2)
    os.close(fd)
    faulthandler.enable()
    _CLIP_SECONDS = clip_seconds


def _timed_worker(item):
    """`_features_worker` with a deadline: a clip still being read after `_CLIP_SECONDS` (a hung decoder) ends the
    worker process, since SIGALRM's default action ends it even inside native code, and with it the run."""
    signal.alarm(_CLIP_SECONDS)
    try:
        return _features_worker(item)
    finally:
        signal.alarm(0)


def _append_private(log_path, text):
    with open(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as f:
        f.write(text)


@contextlib.contextmanager
def stderr_to(log_path):
    """The same for this process, while the block runs."""
    sys.stderr.flush()
    saved, fd = os.dup(2), os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(fd, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(fd)


def media_error(exc):
    """The class of a decoder error that the clip itself causes, or None. Only decord's own errors qualify: it
    cannot open the file as a video (a RuntimeError 'Error reading ...'), finds no frame in it, or fails to decode
    it (DECORDError, DECORDLimitReachedError), and none whose message names the machine's resources. Any other
    exception, such as a RuntimeError of a library or the system, says nothing about the clip."""
    import decord

    text = str(exc)
    if RESOURCE_ERROR.search(text):
        return None
    if isinstance(exc, (decord.DECORDError, decord.DECORDLimitReachedError)):
        return type(exc).__name__
    if (type(exc) is RuntimeError and text.startswith("Error reading ")) or (
            type(exc) is AssertionError and text.startswith("Invalid frame count")):
        return type(exc).__name__
    return None


def _features_worker(item):
    """Pass 1 for one clip, `item` = (video path, DICOM path or None). The video is read once, whole, and decoded
    from those bytes: their SHA-256 (`video_sha256`) names what was screened, and the DICOM regions used are kept
    (`dicom_regions`), so that a later step can tell a clip replaced in place since (`changed_media`). A video whose
    bytes read to the end but that the decoder rejects with its own media errors (`media_error`) is unreadable
    (excluded, its error class recorded); a missing or failing file, a missing decoder, memory, any other exception
    or a failure after decoding say nothing about the clip and stop the run."""
    path, dicom_path = item
    with open(path, "rb") as f:  # a storage failure stops the run here.
        video = f.read()
    digest = hashlib.sha256(video).hexdigest()
    import decord  # noqa: F401

    try:
        frames, n = read_frames(video)
    except (MemoryError, OSError):  # the machine, not the clip.
        raise
    except Exception as exc:
        error = media_error(exc)
        if error is None:  # not identifiably the clip's: stop rather than exclude a healthy clip.
            raise
        return {"video_path": path, "error": error, "video_sha256": digest}
    boxes, status = dicom_bounds(dicom_path, frames.shape[1:3]) if dicom_path else (None, "")
    features, cells, _ = clip_features(frames, boxes, status)
    return {"video_path": path, "error": None, "video_sha256": digest, "n_frames_total": int(n), **features,
            "dicom_regions": json.dumps(boxes) if dicom_path else None,
            "cells_shape": json.dumps(list(cells.shape)), "cells": np.packbits(cells.ravel()).tobytes()}


def changed_media(rows, log, dicom=True):
    """The global keys of `rows` (pass 1 rows: video_path, video_sha256 and, with `dicom`, dicom_path, dicom_status,
    dicom_regions, height, width) whose video file no longer holds the bytes pass 1 decoded, or, with `dicom`, whose
    DICOM header no longer gives the regions it used: a clip replaced or re-masked in place since, which no record
    of a clip list or table shows. A missing file counts as changed. Any other failure to read one (a folder there,
    permissions, storage) stops the run: its message and traceback name the clip's path, so they go to the private
    `log` only."""
    import warnings

    changed = []
    for _, row in rows.iterrows():
        try:
            with open(row.video_path, "rb") as f:
                same = hashlib.sha256(f.read()).hexdigest() == row.video_sha256
            dicom_path = row.get("dicom_path")
            if same and dicom and isinstance(dicom_path, str) and dicom_path and pd.isna(row.get("error")):
                with warnings.catch_warnings():  # a reader's warning can name the path: the console stays aggregate.
                    warnings.simplefilter("ignore")
                    boxes, status = dicom_bounds(dicom_path, (int(row.height), int(row.width)))
                same = status == row.dicom_status and json.dumps(boxes) == row.dicom_regions
        except FileNotFoundError:
            same = False
        except Exception as exc:
            _append_private(log, "".join(traceback.format_exception(exc)))
            raise SystemExit(f"Checking the clips' media stopped on a {type(exc).__name__} (a folder in a file's "
                             f"place, permissions or storage); details, with the clip's path, in {log}.") from None
        if not same:
            changed.append(row.global_key)
    return changed


def features_table(rows):
    """The worker rows as a table with every pass 1 column and its type, also when no clip could be read or the list
    is empty (an empty column would otherwise be float, and the clip list's paths could not be matched to it)."""
    out = pd.DataFrame(rows).reindex(columns=["video_path", *FEATURE_COLUMNS])
    other = [c for c in out if c not in NUMERIC_FEATURES]
    out[other] = out[other].astype(object)
    out[list(NUMERIC_FEATURES)] = out[list(NUMERIC_FEATURES)].apply(pd.to_numeric)
    return out


def run_features(args):
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from multiprocessing import get_context

    ensure_outside_repo(args.out, "--out")
    if args.clip_timeout < 1:
        raise SystemExit("--clip-timeout must be at least 1 second.")
    private_dir(os.path.dirname(os.path.abspath(args.out)))
    clear_outputs(args.out, args.out + ".info.json")
    # The clip list and the script as read (Inputs): the run completes only if both still hold that at the end, and
    # the pass 1 code its record names is the code the workers ran.
    inputs, pass1 = Inputs(__file__), extractor()
    clips = inputs.csv(args.clips, dtype=str)
    # The decoded size and every other pass 1 column keep their names; a list column of the same name (a size from
    # the metadata) is kept as listed_<name>, and a listed size other than the decoded one is counted.
    renamed = {c: f"listed_{c}" for c in clips if c in FEATURE_COLUMNS}
    if set(renamed.values()) & set(clips):
        raise SystemExit(f"--clips has both {sorted(renamed)} and {sorted(renamed.values())}: pass 1 writes the "
                         "first and keeps a listed one under the second. Drop one of them.")
    clips = clips.rename(columns=renamed)
    needed = {"video_path", "global_key", *LAYOUT_COLUMNS} - {"width", "height"}
    if needed - set(clips) or clips.video_path.duplicated().any():
        raise SystemExit(f"--clips needs the columns {sorted(needed)} and no repeated video_path.")
    if clips.global_key.duplicated().any() or not clips.global_key.fillna("").str.fullmatch(KEY_PATTERN).all():
        raise SystemExit("--clips needs a distinct global_key per clip, of letters, digits, '.', '_' and '-'.")
    dicom = clips.dicom_path.fillna("") if "dicom_path" in clips else pd.Series("", index=clips.index)
    items = [(p, d or None) for p, d in zip(clips.video_path, dicom)]
    if (dicom != "").any():
        try:
            import pydicom  # noqa: F401
        except ImportError:
            raise SystemExit("--clips gives dicom_path: the DICOM regions need pydicom (requirements.txt).") from None
    log = args.out + ".log"
    os.close(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))
    # Unlike multiprocessing.Pool, which replaces a dead worker and waits forever for its result, the executor
    # fails every pending clip with BrokenProcessPool when a worker dies.
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init_worker,
                               initargs=(log, args.clip_timeout))
    try:
        rows = list(pool.map(_timed_worker, items, chunksize=4))
    except BrokenProcessPool:
        raise SystemExit("features stopped: a worker process died without an error (killed, for instance for "
                         "memory; a crash in native code such as the decoder; or a clip still being read after "
                         f"--clip-timeout {args.clip_timeout} s); which clip it was reading is not known. A native "
                         f"crash leaves its traceback in {log}.") from None
    except Exception as exc:  # its message and traceback name the clip's path: the private log only.
        _append_private(log, "".join(traceback.format_exception(exc)))
        raise SystemExit(f"features stopped on a {type(exc).__name__}, which says nothing about the clip "
                         f"(storage, memory, decoder or code); details, with the clip's path, in {log}.") from None
    finally:
        pool.shutdown(cancel_futures=True)  # the clips not started are dropped; those being read finish first.
    out = clips.merge(features_table(rows), on="video_path", how="left", validate="one_to_one")
    out.to_parquet(args.out, index=False)
    readable = out[out.error.isna()]
    status = {k or "no dicom_path": int(n) for k, n in readable.dicom_status.fillna("").value_counts().items()}
    scanned = int((readable.scan_frac > 0).sum())
    differ = {c: int((pd.to_numeric(readable[f"listed_{c}"], errors="coerce") != readable[c])[
        readable[f"listed_{c}"].notna()].sum()) for c in ("width", "height") if f"listed_{c}" in readable}
    inputs.check(args.out)
    if extractor() != pass1:  # the libraries, or code imported from elsewhere.
        clear_outputs(args.out)
        raise SystemExit("The pass 1 code or its libraries changed while features ran: rerun features.")
    _write_info(args.out, {"clips": inputs.digest(args.clips), "n_clips": len(out),
                           "errors": int(out.error.notna().sum()), "dicom_status": status,
                           "clips_with_scan_area": scanned, "listed_columns_renamed": renamed,
                           "listed_size_differs": differ, "extractor": pass1})
    print(f"[screen]  features for {len(out)} clips; {int(out.error.notna().sum())} unreadable; DICOM regions "
          f"{status}; a scan area for {scanned}.")
    if any(differ.values()):
        print(f"[screen]  WARNING: the listed size differs from the decoded one for {differ} readable clips; the "
              "decoded size (width, height) is used, the listed one kept as listed_width, listed_height.",
              file=sys.stderr)


# --------------------------------------------------------------------------- #
#  Pass 2: layout baselines and flags
# --------------------------------------------------------------------------- #

def unpack_cells(row):
    shape = tuple(json.loads(row["cells_shape"]))
    bits = np.unpackbits(np.frombuffer(row["cells"], np.uint8), count=shape[0] * shape[1])
    return bits.reshape(shape).astype(bool)


def layout_keys(df):
    """One string per clip naming its layout, the same in every table: a missing field reads '<missing>' (NaN and
    None as dict keys would not match) and a size is an integer whatever the column's dtype."""
    if df.empty:
        return pd.Series([], index=df.index, dtype=object)
    fields = df[list(LAYOUT_COLUMNS)].copy()
    for c in ("width", "height"):
        fields[c] = pd.to_numeric(fields[c]).round().astype("Int64")
    return fields.astype("string").fillna("<missing>").agg(" | ".join, axis=1)


def motion_share(df):
    """The moving share the still flag reads, and whether it is the scan area's: `scan_motion_frac` for a clip with
    a scan area, else the whole frame's `motion_frac`."""
    scan = df["scan_motion_frac"] if "scan_motion_frac" in df else pd.Series(np.nan, index=df.index)
    in_scan = scan.notna()
    return scan.where(in_scan, df.motion_frac), in_scan


def reference_left_out(reference):
    """Why each reference clip is left out of its layout's baseline ('' when it is used): 'unreadable', 'low_motion'
    (it does not clearly move: REFERENCE_MIN_SCAN_MOTION of its scan area, or REFERENCE_MIN_MOTION of the frame
    without one), 'marks_inside' (more than REFERENCE_MAX_MARKS marks inside the image) or 'extra_ink'.

    'extra_ink': among the clips of a layout that pass the first three rules (at least MIN_LAYOUT_CLIPS of them),
    the cells at least CORE_SHARE of them ink are the layout's core ink; a clip with REFERENCE_EXTRA_CELLS or more
    inked cells beside the image outside it (`extra_cells` against the core) may carry a measurement results box
    without marks over the image, and is left out. Such a box then never becomes normal ink unless most of the
    layout's clips have it; a few contaminated clips (above COMMON_SHARE of the reference, below CORE_SHARE) no
    longer make it normal."""
    share, in_scan = motion_share(reference)
    low = share < np.where(in_scan, REFERENCE_MIN_SCAN_MOTION, REFERENCE_MIN_MOTION)
    marked = reference.inside_glyphs > REFERENCE_MAX_MARKS
    reasons = pd.Series(np.select([reference.error.notna(), low.to_numpy(), marked.to_numpy()],
                                  ["unreadable", "low_motion", "marks_inside"], default=""), index=reference.index)
    usable = reference[reasons == ""]
    for _, group in usable.groupby(layout_keys(usable)):
        if len(group) < MIN_LAYOUT_CLIPS:
            continue
        cells = [unpack_cells(r) for _, r in group.iterrows()]
        core = np.stack(cells).mean(axis=0)
        for i, own, bbox in zip(group.index, cells, group.region_bbox):
            extra = extra_cells(own, core, json.loads(bbox), share=CORE_SHARE)
            if extra is not None and extra >= REFERENCE_EXTRA_CELLS:
                reasons[i] = "extra_ink"
    return reasons


def layout_baselines(reference):
    """Per layout (vendor, model, size, acquisition; `layout_keys`): the share of the reference clips inking each
    cell, or None for a layout with fewer than MIN_LAYOUT_CLIPS of them. Only readable reference clips that clearly
    move, have no marks inside the image and no block of ink beside it that most of their layout lacks count
    (`reference_left_out`), so that their own measurements do not become normal. The rule does not depend on the
    screen's thresholds: a threshold sweep in `calibrate` and a rerun of `flag` see the same baselines.

    Returns `(baselines, sources)`, `sources` the video paths behind each layout's baseline."""
    usable = reference[reference_left_out(reference) == ""]
    baselines, sources = {}, {}
    for key, group in usable.groupby(layout_keys(usable)):
        sources[key] = group.video_path.tolist()
        if len(group) < MIN_LAYOUT_CLIPS:
            baselines[key] = None
            continue
        baselines[key] = np.stack([unpack_cells(r) for _, r in group.iterrows()]).mean(axis=0)
    return baselines, sources


def write_baselines(out_dir, baselines, sources, left_out=None):
    """The record of the layout baselines, replacing an earlier one: layouts.csv (reference clips used, and those
    left out by reason when `left_out`, the reasons of `reference_left_out` per layout, is given; common cells per
    layout), sources.csv (the clips behind each baseline) and one PNG per baseline (white: every reference clip inks
    the cell) to check by eye that no measurement box became normal. The maps are averages, but of identified
    frames: the folder is 0700."""
    import cv2

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, mode=0o700)
    os.chmod(out_dir, 0o700)
    reasons = ("low_motion", "marks_inside", "extra_ink", "unreadable")
    counts = pd.crosstab(left_out.index, left_out) if left_out is not None and len(left_out) else pd.DataFrame()
    rows = []
    for i, key in enumerate(sorted(set(sources) | set(counts.index))):
        share, png = baselines.get(key), ""
        if share is not None:
            png = f"layout_{i:03d}.png"
            img = cv2.resize((share * 255).round().astype(np.uint8), (share.shape[1] * CELL, share.shape[0] * CELL),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_dir, png), img)
        rows.append({"layout": key, "reference_clips": len(sources.get(key, [])), "png": png,
                     "common_cells": int((share >= COMMON_SHARE).sum()) if share is not None else None,
                     **{f"left_out_{r}": int(counts.at[key, r]) if r in counts and key in counts.index else 0
                        for r in reasons}})
    pd.DataFrame(rows, columns=["layout", "reference_clips", *(f"left_out_{r}" for r in reasons), "common_cells",
                                "png"]).to_csv(os.path.join(out_dir, "layouts.csv"), index=False)
    pd.DataFrame([(k, p) for k in sorted(sources) for p in sources[k]], columns=["layout", "video_path"]).to_csv(
        os.path.join(out_dir, "sources.csv"), index=False)


def extra_cells(cells, baseline, bbox, share=COMMON_SHARE):
    """Inked cells beside the image, anywhere on the screen (a results panel can sit far from the image), where the
    layout normally has no ink (fewer than `share` of its reference clips ink the cell); None without a baseline or
    without an image region (`bbox` None). Cells next to normal ink count as normal too, so a longer label than
    usual is not extra."""
    import cv2

    if baseline is None or bbox is None:
        return None
    common = cv2.dilate((baseline >= share).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return int((cells & ~common).sum())


def apply_flags(df, baselines, thresholds=THRESHOLDS):
    """`df` with `extra_cells` (against `baselines`, from layout_baselines) and the flags (`flag_columns`)."""
    df = df.copy()
    keys = layout_keys(df)
    extra = []
    for i, row in df.iterrows():
        if isinstance(row.get("error"), str):
            extra.append(None)
            continue
        extra.append(extra_cells(unpack_cells(row), baselines.get(keys[i]), json.loads(row["region_bbox"])))
    df["extra_cells"] = pd.array(extra, dtype="Int64")
    return flag_columns(df, thresholds)


def flag_columns(df, thresholds=THRESHOLDS):
    """`df` (pass 1 features and `extra_cells`) with `still`, `still_basis` ('scan' or 'frame': the moving share
    it read, `motion_share`), `measurement`, `exclude` and `exclude_reason` at `thresholds`, the first that applies:
    'unreadable'; the header's reason (HEADER_REASONS: no image region, a size other than the video's, an unreadable
    header, no header); 'still'; and, only with `measurement_excludes`, 'measurement' and 'incomplete' (a readable
    clip without `extra_cells`: its layout has no baseline, so it could not be checked for a results box). The flags
    are None for an unreadable clip. The baselines do not depend on the thresholds, so this gives a rerun of `flag`
    exactly."""
    df = df.copy()
    readable = df.error.isna()
    share, in_scan = motion_share(df)
    limit = np.where(in_scan, thresholds["still_scan_motion_frac"], thresholds["still_motion_frac"])
    df["still"] = pd.array(np.where(readable, share < limit, None), dtype="boolean")
    df["still_basis"] = np.where(readable, np.where(in_scan, "scan", "frame"), None)
    measured = (df.inside_glyphs >= thresholds["inside_glyphs"]) | (
        df.extra_cells.fillna(-1) >= thresholds["extra_cells"])
    df["measurement"] = pd.array(np.where(readable, measured, None), dtype="boolean")
    status = df.dicom_status.fillna("") if "dicom_status" in df else pd.Series("", index=df.index)
    header = status.map(lambda s: "" if s == "ok" else HEADER_REASONS.get(s, "header_unreadable")).to_numpy()
    pixel = bool(thresholds["measurement_excludes"])
    df["exclude_reason"] = np.select(
        [~readable.to_numpy(), header != "", df.still.fillna(False).to_numpy(bool),
         pixel & df.measurement.fillna(False).to_numpy(bool), pixel & df.extra_cells.isna().to_numpy()],
        ["unreadable", header, "still", "measurement", "incomplete"], default="")
    df["exclude"] = df.exclude_reason != ""
    return df


def at_mark_size(df, min_px):
    """`df` with `inside_glyphs` recounted from `inside_marks`, only marks of at least `min_px` ink pixels counting
    (a straight dotted line: its dots together): the most over the whole clip and each window that is not frozen,
    as pass 1 counts them. NaN for an unreadable clip; `df` unchanged without `inside_marks`."""
    if "inside_marks" not in df:
        return df

    def count(evidence):
        if not isinstance(evidence, str):
            return np.nan
        marks = json.loads(evidence)
        return max(sum(area >= min_px for area, _ in found) for found in (marks["clip"], *marks["windows"]))

    return df.assign(inside_glyphs=df.inside_marks.map(count).astype(float))


def screen_flags(features, reference, thresholds):
    """The screen at `thresholds`: marks inside the image counted at `inside_mark_px` in the reference clips and in
    the screened ones alike, then which reference clips are eligible, the layout baselines, `extra_cells` and the
    flags of `features`. What `flag` writes, and what `calibrate` recomputes for each threshold set: the size
    changes which reference clips have marks inside, so the baselines too.

    Returns `(flags, baselines, sources, left_out)`, `left_out` the reasons of `reference_left_out` indexed by
    layout."""
    reference = at_mark_size(reference, thresholds["inside_mark_px"])
    baselines, sources = layout_baselines(reference)
    left_out = reference_left_out(reference)
    left_out.index = layout_keys(reference)
    flags = apply_flags(at_mark_size(features, thresholds["inside_mark_px"]), baselines, thresholds)
    return flags, baselines, sources, left_out


# What pass 1 and the flag rule run, from these functions down (`_code_record`).
PASS1 = (_features_worker, features_table)
RULE = (screen_flags,)


def run_flag(args):
    ensure_outside_repo(args.out, "--out")
    private_dir(os.path.dirname(os.path.abspath(args.out)))
    clear_outputs(args.out, args.out + ".info.json", args.out + ".baselines")
    thresholds = {**THRESHOLDS, **(json.loads(args.thresholds) if args.thresholds else {})}
    if set(thresholds) != set(THRESHOLDS):
        raise SystemExit(f"--thresholds keys must be among {sorted(THRESHOLDS)}.")
    inputs, pass1 = Inputs(__file__), _code_record(*PASS1)
    extracted = {role: extracted_by(getattr(args, role), pass1, inputs) for role in ("features", "reference")}
    libraries = {k for k in {**extracted["features"]["libraries"], **extracted["reference"]["libraries"]}
                 if extracted["features"]["libraries"].get(k) != extracted["reference"]["libraries"].get(k)}
    if libraries:
        raise SystemExit(f"--features and --reference were extracted with other versions of {sorted(libraries)}: "
                         "rerun features for one of them in the other's environment.")
    rule = flag_rule()
    screen_id = hashlib.sha256(json.dumps({"extractor": extracted["features"], "rule": rule},
                                          sort_keys=True).encode()).hexdigest()
    calibration = CALIBRATION or {}
    differs = [what for what, key, here in (("screen (extraction or flag code, constants or libraries)", "screen",
                                             screen_id), ("thresholds", "thresholds", thresholds))
               if calibration.get(key) != here]
    calibrated = CALIBRATION is not None and not differs
    reference = inputs.parquet(args.reference)
    if reference.empty:  # features of an empty list: a table with every column and no rows.
        raise SystemExit("--reference has no clips (an empty reference list): no layout would get a baseline, so no "
                         "clip a results-box score. Make a list with reference clips of the screened layouts, then "
                         "rerun features on it.")
    df, baselines, sources, left_out = screen_flags(inputs.parquet(args.features), reference, thresholds)
    df.drop(columns=["cells"]).to_parquet(args.out, index=False)
    write_baselines(args.out + ".baselines", baselines, sources, left_out)
    counts = {k: int(n) for k, n in df.exclude_reason.replace("", "kept").value_counts().items()}
    no_baseline = int((df.error.isna() & df.extra_cells.isna()).sum())
    inputs.check(args.out, args.out + ".baselines")
    _write_info(args.out, {"features": _source(args.features, inputs), "reference": _source(args.reference, inputs),
                           "thresholds": thresholds, "calibration": CALIBRATION if calibrated else None,
                           "screen": screen_id, "extractor": extracted["features"], "rule": rule,
                           "counts": counts,
                           "layouts_with_baseline": sum(b is not None for b in baselines.values()),
                           "layouts_without_baseline": sum(b is None for b in baselines.values()),
                           "clips_without_layout_baseline": no_baseline,
                           "reference_left_out": {k: int(n) for k, n in
                                                  left_out[left_out != ""].value_counts().items()},
                           "still_basis": {k: int(n) for k, n in df.still_basis.dropna().value_counts().items()}})
    print(f"[screen]  {counts}; baselines and their reference clips in {args.out}.baselines (0700).")
    if CALIBRATION is not None and not calibrated:
        print(f"[screen]  WARNING: these flags are provisional: their {' and '.join(differs)} differ from those "
              "CALIBRATION measured.", file=sys.stderr)
    if no_baseline and thresholds["measurement_excludes"]:
        print(f"[screen]  WARNING: {no_baseline} readable clips could not be checked for a results box (their "
              f"layout has fewer than {MIN_LAYOUT_CLIPS} clean reference clips, see layouts.csv, or no image "
              "region); those not flagged otherwise are excluded as incomplete. More reference clips of their "
              "layouts would let the screen check them.", file=sys.stderr)


# --------------------------------------------------------------------------- #
#  Human answers (threshold tuning)
# --------------------------------------------------------------------------- #

def read_labels(paths, inputs=None):
    """The human answers per clip from label files exported from sheets: indexed by `global_key`, one column per
    question ('yes', 'no' or '') and `video_sha256`, the video bytes the answers are of (the sheet's copy was made
    from them). A clip's global key stays when its video is replaced in place; its bytes do not.

    A clip answered in several files gets every answer it was given there. A question answered both 'yes' and
    'no', or a clip answered for different video bytes, stops here: the order of the files never decides. The files
    are read into `inputs` (Inputs) when given."""
    inputs = inputs or Inputs()
    frames = []
    for path in paths:
        lab = inputs.csv(path, dtype=str).fillna("")
        if {"global_key", "video_sha256", *LABEL_FIELDS} - set(lab):
            raise SystemExit(f"{path} needs the columns global_key, video_sha256, {', '.join(LABEL_FIELDS)} (a file "
                             "exported from an older sheet has no video_sha256: answer the clips on a new sheet).")
        answers = lab[list(LABEL_FIELDS)].apply(lambda col: col.str.strip().str.lower())
        keys, videos = lab.global_key.str.strip(), lab.video_sha256.str.strip()
        if not answers.isin(["yes", "no", ""]).to_numpy().all() or (keys == "").any() or (videos == "").any():
            raise SystemExit(f"{path}: answers must be yes, no or blank, and every row needs a global_key and a "
                             "video_sha256.")
        frames.append(answers.assign(global_key=keys, video_sha256=videos))
    lab = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=["global_key", "video_sha256", *LABEL_FIELDS]))
    videos = lab[["global_key", "video_sha256"]].drop_duplicates()
    replaced = set(videos.global_key[videos.global_key.duplicated()])
    if replaced:
        raise SystemExit(f"{len(replaced)} clips were answered for different video bytes across the label files "
                         f"(the video replaced in place between two sheets): {', '.join(sorted(replaced)[:20])}. "
                         "Leave out the files of the older sheet.")
    out = pd.DataFrame("", index=pd.Index(lab.global_key.unique(), name="global_key"), columns=list(LABEL_FIELDS))
    out["video_sha256"] = videos.set_index("global_key").video_sha256.reindex(out.index)
    conflicts = set()
    for field in LABEL_FIELDS:
        given = lab.loc[lab[field] != "", ["global_key", field]].drop_duplicates()
        conflicts |= set(given.global_key[given.global_key.duplicated()])
        out.loc[given.global_key.to_numpy(), field] = given[field].to_numpy()
    if conflicts:
        raise SystemExit(f"{len(conflicts)} clips were given both yes and no to one question across the label "
                         f"files: {', '.join(sorted(conflicts)[:20])}. Remove the wrong answer and rerun.")
    return out


# --------------------------------------------------------------------------- #
#  Labelling sheet and calibration
# --------------------------------------------------------------------------- #

def _jpeg(img, width=480):
    """A BGR image at the sheet's width, as base64 JPEG."""
    import cv2

    img = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(jpg.tobytes()).decode()


def _draw(frame, details):
    import cv2

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for mask, color in ((details["scan"], (255, 128, 0)), (details["region"], (0, 200, 0))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 1)
    for x, y, w, h, *_ in details["inside"]:
        cv2.rectangle(img, (x - 2, y - 2), (x + w + 1, y + h + 1), (0, 0, 255), 1)
    return _jpeg(img)


def _motion_map(details):
    """What the still score sees: white where a pixel counts as moving (`motion_frac` is their share); gray, darker
    as it changes less, elsewhere, including where the opening removed it (the ECG trace, cursors)."""
    import cv2

    level = np.minimum(details["change"] / MOVE_STD, 1.0) * 100
    return _jpeg(cv2.cvtColor(np.where(details["moving"], 255, level).astype(np.uint8), cv2.COLOR_GRAY2BGR))


def find_ffmpeg():
    """The ffmpeg that makes the sheet's playable copies, with its H.264 encoder: imageio-ffmpeg's own build
    (requirements.txt), or the one IMAGEIO_FFMPEG_EXE names."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise SystemExit("sheet needs ffmpeg for the clips' playable copies: install imageio-ffmpeg "
                         "(requirements.txt).") from None
    encoders = subprocess.run([exe, "-hide_banner", "-encoders"], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True).stdout
    if " libx264 " not in encoders:
        raise SystemExit("The ffmpeg found has no H.264 encoder (libx264): set IMAGEIO_FFMPEG_EXE to one that has.")
    return exe


def playable_copy(path, out_path, ffmpeg, log_path):
    """The clip as a browser plays it: every frame, at the clip's own rate, in 4:2:0 H.264 (the archive's clips are
    4:4:4 H.264, which browsers do not play). ffmpeg's messages name the path: they go to the private log."""
    with open(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab") as log:
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-n", "-i", path, "-map", "0:v:0", "-an",
                        "-map_metadata", "-1", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        out_path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, check=True,
                       timeout=600)
    os.chmod(out_path, 0o600)


def clip_media(row, out_dir, ffmpeg, detector=False):
    """A clip's media on a sheet: its playable copy (clips/<global_key>.mp4), which the answers rest on, and with
    `detector` the detector's view, closed: one frame with the scan area, the moving region and the marks, and the
    motion map. Both come from the video's bytes read once and checked to be those pass 1 screened (`video_sha256`),
    so the answers describe the screened clip."""
    log = os.path.join(out_dir, "decoder.log")
    with open(row.video_path, "rb") as f:
        video = f.read()
    if hashlib.sha256(video).hexdigest() != row.video_sha256:
        raise SystemExit(f"The video of clip {row.global_key} changed since features read it: rerun features and flag, "
                         f"then make the sheet again. Remove {out_dir} first.")
    with stderr_to(log):  # the decoder's messages name the path.
        frames, _ = read_frames(video)
        dicom_path = row.get("dicom_path")
        boxes = dicom_bounds(dicom_path, frames.shape[1:3])[0] if isinstance(dicom_path, str) and dicom_path else None
    copy = f"clips/{row.global_key}.mp4"
    source = os.path.join(out_dir, "clips", f".{row.global_key}.source")  # ffmpeg reads the checked bytes.
    with open(os.open(source, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(video)
    try:
        playable_copy(source, os.path.join(out_dir, copy), ffmpeg, log)
    finally:
        os.remove(source)
    player = f"<video src='{copy}' controls loop muted playsinline preload=metadata></video>"
    if not detector:
        return player
    _, _, details = clip_features(frames, boxes)
    images = "".join(f"<img src='data:image/jpeg;base64,{jpg}'>"
                     for jpg in (_draw(frames[len(frames) // 2], details), _motion_map(details)))
    return f"{player}<details><summary>Detector view</summary>{images}</details>"


SHEET_JS = """
function exportLabels(){
const rows=[["global_key","video_sha256","still","measurement","other_exclusion","cursor","note"]];
document.querySelectorAll(".clip").forEach(c=>{const g=n=>{
const e=c.querySelector(`input[name='${n}-${c.id}']:checked`);
return e?e.value:""};rows.push([c.id,c.dataset.video,g("still"),g("measurement"),g("other"),g("cursor"),
'"'+(c.querySelector("textarea").value||"").replace(/"/g,'""')+'"'])});
const blob=new Blob([rows.map(r=>r.join(",")).join("\\n")+"\\n"],{type:"text/csv"});
const a=document.createElement("a");a.href=URL.createObjectURL(blob);
a.download=document.title+"-labels.csv";a.click();}
const inView=new IntersectionObserver(es=>es.forEach(e=>e.isIntersecting?e.target.play().catch(()=>{}):
e.target.pause()),{threshold:0.5});
document.querySelectorAll("video").forEach(v=>inView.observe(v));
"""


def _radio(name, key):
    return "".join(f"<label><input type=radio name='{name}-{key}' value='{v}'>{v}</label> "
                   for v in ("yes", "no"))


def audit_sample(flags, n_flagged, n_unflagged, seed):
    """A fixed random sample of the readable clips, for `calibrate`: up to `n_flagged` flagged and `n_unflagged`
    unflagged clips (at the flags' thresholds), shuffled so that the labeller does not see the stratum. Each clip's
    inclusion probability (sampled / population of its stratum) is recorded at draw time; a later change of
    thresholds does not change it, so the sample still weighs up to the same population.

    Returns `(rows, strata)`."""
    readable = flags[flags.error.isna()]
    parts, strata = [], {}
    for stratum, n, members in (("flagged", n_flagged, readable[readable.exclude]),
                                ("unflagged", n_unflagged, readable[~readable.exclude])):
        if len(members) and n < 1:  # its clips could not be weighed up: the estimates would leave them out.
            raise SystemExit(f"The audit sample needs at least one of the {len(members)} {stratum} clips.")
        drawn = members.sample(n=min(n, len(members)), random_state=seed)
        strata[stratum] = {"population": int(len(members)), "sampled": int(len(drawn))}
        parts.append(drawn.assign(stratum=stratum, inclusion_prob=len(drawn) / len(members) if len(members) else 0.0))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True), strata


def population(flags):
    """The readable clips of a screen, as an audit sample records them and `calibrate` checks them: their number and
    a digest of their global keys."""
    keys = sorted(flags.global_key[flags.error.isna()])
    return {"readable_clips": len(keys), "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest()}


def diagnostic_cases(df, thresholds):
    """Each readable clip's case for a diagnostic sample (None when unreadable), the first that applies:
    `still`; `results_box` (extra_cells at least its threshold); `large_marks` (at least `inside_glyphs` marks of
    DIAGNOSTIC_LARGE_PX or more inside the image, over the whole clip or in a window: likely measurements);
    `traced` (a traced contour inside, which alone is one mark, below the threshold); `dotted_line` (a straight
    dotted line inside: a cursor, or the line of a distance); `window_small_marks` (marks
    inside found in windows only: slow tissue, or a mark drawn partway through and broken up); `small_marks` (other
    marks inside); `clean`."""
    marks = df.inside_marks.map(lambda e: json.loads(e) if isinstance(e, str) else {"clip": [], "windows": []})
    detections = marks.map(lambda m: [m["clip"], *m["windows"]])
    large = (at_mark_size(df, DIAGNOSTIC_LARGE_PX).inside_glyphs >= thresholds["inside_glyphs"]).to_numpy()
    dotted, traced = (detections.map(lambda ds: any(kind == k for found in ds for _, kind in found)).to_numpy(bool)
                      for k in ("line", "trace"))
    any_marks = detections.map(lambda ds: any(ds)).to_numpy(bool)
    in_clip = marks.map(lambda m: bool(m["clip"])).to_numpy(bool)
    cases = np.select([df.still.fillna(False).to_numpy(bool), (df.extra_cells.fillna(-1) >= thresholds["extra_cells"])
                       .to_numpy(bool), large, traced, dotted, any_marks & ~in_clip, any_marks],
                      ["still", "results_box", "large_marks", "traced", "dotted_line", "window_small_marks",
                       "small_marks"],
                      default="clean")
    return pd.Series(np.where(df.error.isna().to_numpy(), cases, None), index=df.index)


def diagnostic_sample(flags, reference_flags, thresholds, n_per_stratum, seed):
    """A fixed random sample for `calibrate` aimed at the screen's doubtful cases: per role (`target`, the screened
    clips; `reference`, the reference clips, with their flags against the baselines), acquisition mode and case
    (`diagnostic_cases` at the flags' `thresholds`), up to `n_per_stratum` random readable clips, at least one of
    every stratum that has clips, each with its inclusion probability (sampled / population of its stratum),
    recorded at draw time. Shuffled, so that the labeller does not see the strata.

    Returns `(rows, strata)`."""
    if n_per_stratum < 1:  # a stratum's clips could not be weighed up: the estimates would leave them out.
        raise SystemExit("--n-per-stratum must be at least 1: every stratum that has clips needs one.")
    parts, strata = [], {}
    for role, df in (("target", flags), ("reference", reference_flags)):
        readable = df[df.error.isna()]
        names = (f"{role}:" + readable.acq_category.fillna("<missing>").astype(str) + ":"
                 + diagnostic_cases(readable, thresholds).astype(str))
        for name, members in readable.groupby(names, sort=True):
            drawn = members.sample(n=min(n_per_stratum, len(members)), random_state=seed)
            strata[name] = {"population": int(len(members)), "sampled": int(len(drawn))}
            parts.append(drawn.assign(stratum=name, role=role, inclusion_prob=len(drawn) / len(members)))
    rows = pd.concat(parts) if parts else flags.iloc[:0].assign(stratum="", role="", inclusion_prob=0.0)
    if rows.global_key.duplicated().any():
        raise SystemExit("A clip is both screened and a reference clip: its answers would weigh in twice.")
    return rows.sample(frac=1.0, random_state=seed).reset_index(drop=True), strata


def validation_set(flags, seed):
    """Every readable clip of the flags of a validation list (`screen_validation.py`), with the list's group as its
    stratum and its sampling probability as its inclusion probability, so that `calibrate` weighs the answers up to
    the population the list was drawn from. Shuffled, so that the labeller does not see the groups.

    Returns `(rows, strata)`."""
    if {"sample_stratum", "sample_prob", "stratum_clips", "population_clips"} - set(flags):
        raise SystemExit("--all-clips needs the flags of a validation list (screen_validation.py), with its "
                         "sample_stratum, sample_prob, stratum_clips and population_clips.")
    readable = flags[flags.error.isna()]
    rows = readable.assign(stratum=readable.sample_stratum, inclusion_prob=readable.sample_prob.astype(float))
    strata = {g: {"population": int(float(part.stratum_clips.iloc[0])), "sampled": int((rows.stratum == g).sum())}
              for g, part in flags.groupby("sample_stratum", sort=True)}
    # The groups must add up to the population: a group with no clip on the list would otherwise drop out of the
    # estimates unnoticed.
    covered = sum(s["population"] for s in strata.values())
    population_clips = int(float(flags.population_clips.iloc[0]))
    if covered != population_clips:
        raise SystemExit(f"The groups of this validation list hold {covered} of the {population_clips} clips of its "
                         "population: a group has no clip on the list, so the estimates would leave it out. Rebuild "
                         "the list with screen_validation.py, every group sampled.")
    return rows.sample(frac=1.0, random_state=seed).reset_index(drop=True), strata


def review_set(flags, keys):
    """The clips named in `keys` (a table with a global_key column), for a person to look at. Not a random sample:
    `calibrate` refuses it."""
    wanted = set(keys.global_key)
    rows = flags[flags.global_key.isin(wanted)]
    if wanted - set(rows.global_key) or rows.error.notna().any():
        raise SystemExit(f"--keys names {len(wanted - set(rows.global_key))} clips not in --flags and "
                         f"{int(rows.error.notna().sum())} unreadable ones, which cannot be reviewed.")
    return rows.assign(stratum="review", inclusion_prob=np.nan).reset_index(drop=True)


def run_sheet(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"{args.out_dir} is not empty: every review round gets a new folder, so that its sample "
                         "record and its pages stay together.")
    inputs = Inputs(__file__)
    flags = inputs.parquet(completed(args.flags, "flag", inputs))
    ffmpeg = find_ffmpeg()
    if sum(map(bool, (args.keys, args.all_clips, args.diagnostic))) > 1:
        raise SystemExit("--keys, --all-clips and --diagnostic make different sheets: give one.")
    if "sample_prob" in flags and not (args.all_clips or args.keys):
        raise SystemExit("--flags are of a validation list (screen_validation.py), whose clips were drawn with their "
                         "own probabilities: a sample of them would replace those, and its estimates would describe "
                         "the list, not the population it was drawn from. Use --all-clips (or --keys to look at "
                         "chosen clips).")
    if args.keys:
        rows = review_set(flags, inputs.csv(args.keys, dtype=str))
        info = {"purpose": "review", "keys": inputs.digest(args.keys)}
        described = f"{len(rows)} clips to review"
    elif args.diagnostic:
        thresholds = inputs.json(args.flags + ".info.json")["thresholds"]
        reference = inputs.parquet(screen_sources(args.flags, inputs)["reference"]["path"])
        rows, strata = diagnostic_sample(flags, screen_flags(reference, reference, thresholds)[0], thresholds,
                                         args.n_per_stratum, args.seed)
        info = {"purpose": "diagnostic", "strata": strata, "seed": args.seed, "population": population(flags),
                "reference_population": population(reference)}
        described = (f"diagnostic sample: {len(rows)} clips of {len(strata)} strata "
                     f"({int((rows.role == 'reference').sum())} reference clips)")
    elif args.all_clips:
        rows, strata = validation_set(flags, args.seed)
        info = {"purpose": "validation", "strata": strata, "seed": args.seed, "population": population(flags)}
        described = "validation list: " + ", ".join(f"{v['sampled']} {k}" for k, v in strata.items())
    else:
        rows, strata = audit_sample(flags, args.n_flagged, args.n_unflagged, args.seed)
        info = {"purpose": "audit", "strata": strata, "seed": args.seed, "population": population(flags)}
        described = ("audit sample: " + ", ".join(f"{v['sampled']} of {v['population']} {k}"
                                                  for k, v in strata.items()))
    private_dir(args.out_dir)
    changed = changed_media(rows, os.path.join(args.out_dir, "decoder.log"))
    if changed:
        raise SystemExit(f"{len(changed)} of the sheet's {len(rows)} clips no longer hold the video or DICOM regions "
                         "pass 1 read (replaced in place since features): the answers would describe other clips "
                         "than the screen's. Rerun features and flag first.")
    os.mkdir(os.path.join(args.out_dir, "clips"), 0o700)
    # The detector's output only on a sheet of chosen clips: on a sheet calibrate uses, it would steer the answers.
    write_pages(rows, args.out_dir, args.per_page, info["purpose"], ffmpeg, detector=info["purpose"] == "review")
    # The record last: a sheet that stopped has none, and calibrate refuses it.
    sheet_clips = os.path.join(args.out_dir, "sheet_clips.csv")
    rows = rows if "role" in rows else rows.assign(role="target")
    rows[["global_key", "video_path", "video_sha256", "stratum", "inclusion_prob", "role"]].to_csv(sheet_clips,
                                                                                                 index=False)
    inputs.check(args.out_dir)  # the folder was this run's: empty or missing before it.
    _write_info(os.path.join(args.out_dir, "sheet"),
                {"flags": inputs.digest(args.flags), "sheet_clips": sha256(sheet_clips), **info})
    print(f"[screen]  sheet, {described}: {-(-len(rows) // args.per_page)} pages in {args.out_dir} (0700).")


def write_pages(rows, out_dir, per_page, name, ffmpeg, detector=False):
    """The sheet's HTML pages: per clip, its playable copy, with `detector` the detector's view, and the four
    questions. A clip on a sheet was readable when screened, so a failure here (storage, decoder, ffmpeg or code)
    stops the sheet; its details name the clip's path and go to the private decoder.log."""
    log = os.path.join(out_dir, "decoder.log")
    for page, start in enumerate(range(0, len(rows), per_page)):
        title = f"clip-{name}-{page + 1:02d}"  # exported labels: <title>-labels.csv
        parts = [f"<!doctype html><meta charset=utf-8><title>{title}</title><style>body{{font:14px sans-serif;"
                 "background:#111;color:#eee}.clip{margin:16px 0;border-top:1px solid #444;padding-top:8px}"
                 "video{display:block;width:640px;max-width:100%;margin:6px 0}img{margin:4px 4px 0 0}</style>",
                 "<p>Identified frames: do not copy, screenshot or share. Watch each clip (it loops; the controls "
                 "pause, scrub and slow it), answer its four questions, then export. Measurements: calipers, "
                 "tracings, measurement text or a results box; a cursor line alone is not one. Cursor: a Doppler "
                 "or M-mode cursor line over the image, whatever else the clip shows.</p>",
                 "<p>The detector view, closed so that it does not steer the answers, shows what the screen saw: "
                 "blue, the scan area from the DICOM regions; green, the moving region; red, the marks counted "
                 "inside it; then the motion map, white where a pixel counts as moving.</p>" if detector else ""]
        for _, row in rows.iloc[start:start + per_page].iterrows():
            try:
                media = clip_media(row, out_dir, ffmpeg, detector)
            except Exception as exc:  # its message and traceback name the clip's path: the private log only.
                _append_private(log, "".join(traceback.format_exception(exc)))
                raise SystemExit(f"sheet stopped on a {type(exc).__name__} (storage, decoder, ffmpeg or code: the "
                                 f"clip was readable when screened); details, with the clip's path, in {log}. "
                                 f"Remove {out_dir} before rerunning.") from None
            key = row.global_key
            # The video bytes the clip was copied from (checked against the screen's), exported with its answers.
            parts.append(f"<div class=clip id={key} data-video='{html.escape(str(row.video_sha256))}'><b>{key}</b> "
                         f"view {html.escape(str(row.get('view', '')))}, "
                         f"{html.escape(str(row.get('acq_category', '')))}{media}"
                         f"Still? {_radio('still', key)} &nbsp; Measurements visible? {_radio('measurement', key)}"
                         f" &nbsp; Other reason to exclude? {_radio('other', key)}"
                         f" &nbsp; Cursor line over the image? {_radio('cursor', key)}<br>"
                         f"<textarea rows=1 cols=60 placeholder='note (no patient details)'></textarea></div>")
        parts.append(f"<button onclick=exportLabels()>Export labels</button><script>{SHEET_JS}</script>")
        path = os.path.join(out_dir, f"{title}.html")
        with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            f.write("".join(parts))


def confusion(predicted, truth, weights):
    """Weighted confusion counts and the sensitivity / specificity they give (None when undefined)."""
    p, t = predicted.astype(bool).to_numpy(), truth.astype(bool).to_numpy()
    w = weights.to_numpy(dtype=float)
    tp, fp = float(w[p & t].sum()), float(w[p & ~t].sum())
    fn, tn = float(w[~p & t].sum()), float(w[~p & ~t].sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": tn / (tn + fp) if tn + fp else None}


def upper_bound(k, n, confidence=0.95):
    """One-sided exact (Clopper-Pearson) upper confidence bound on a proportion seen `k` times in `n`."""
    if n == 0:
        return None
    if k >= n:
        return 1.0

    def cdf(p):  # P(X <= k), each term in logs: binomial coefficients overflow floats for large n.
        log_p, log_q = math.log(p), math.log1p(-p)
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p
                            + (n - i) * log_q) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):  # bisection: P(X <= k) falls as the proportion grows.
        p = (lo + hi) / 2
        if cdf(p) > 1 - confidence:
            lo = p
        else:
            hi = p
    return hi


def _labelled(screened, sheet_clips, answers):
    """The sheet's clips (global_key, video_path, stratum, inclusion_prob) with their flags from `screened`, their
    answers (`answer_<question>`), whether they are `ineligible` (a 'yes' to a reason to exclude) and their weight
    (1 / inclusion probability). The weights hold only if every clip of the sheet is screened and answers every
    question, so a gap stops here. Answers to clips of other sheets are not used."""
    lab = sheet_clips.merge(screened.drop(columns=["global_key"], errors="ignore"), on="video_path", how="left",
                            validate="one_to_one", indicator=True)
    if (lab._merge != "both").any():
        raise SystemExit(f"{int((lab._merge != 'both').sum())} clips of the sheet are not in --flags: use the "
                         "flags of the screen the sheet was drawn from, or a rerun of it on the same clips.")
    given = answers.reindex(lab.global_key.to_numpy()).fillna("")
    for field in LABEL_FIELDS:
        lab[f"answer_{field}"] = given[field].to_numpy()
    gaps = (lab[[f"answer_{f}" for f in LABEL_FIELDS]] == "").any(axis=1)
    if gaps.any():
        raise SystemExit(f"{int(gaps.sum())} of the sheet's {len(lab)} clips lack an answer to one of the "
                         f"{len(LABEL_FIELDS)} questions: answer every clip fully before calibrating.")
    if not (lab.inclusion_prob.astype(float) > 0).all():
        raise SystemExit("Every clip of an audit sample needs a positive inclusion probability.")
    lab["ineligible"] = (lab[[f"answer_{f}" for f in EXCLUSION_FIELDS]] == "yes").any(axis=1)
    lab["weight"] = 1.0 / lab.inclusion_prob.astype(float)
    return lab.drop(columns="_merge")


def _excluded_share(lab, screened, flags_are_population):
    """The share of readable clips excluded: counted on all of `screened` when they are the whole population (an
    audit sample's screen), else estimated from the weighted sample (a validation list's flags)."""
    if flags_are_population:
        readable = screened[screened.error.isna()]
        return float(readable.exclude.mean()) if len(readable) else None
    return float(lab.weight[lab.exclude].sum() / lab.weight.sum()) if len(lab) else None


def _cursor_clips(lab):
    """The clips answered as having a cursor line over the image: how many of the eligible ones the screen
    excludes (weighted share), and how many without measurements it flags for measurements."""
    cursor = lab[lab.answer_cursor == "yes"]
    eligible = cursor[~cursor.ineligible]
    unmeasured = cursor[cursor.answer_measurement == "no"]
    return {"labelled": int(len(cursor)), "eligible": int(len(eligible)),
            "eligible_excluded": int(eligible.exclude.sum()),
            "eligible_excluded_share": (float(eligible.weight[eligible.exclude].sum() / eligible.weight.sum())
                                        if len(eligible) else None),
            "without_measurement": int(len(unmeasured)),
            "without_measurement_flagged": int(unmeasured.measurement.fillna(False).astype(bool).sum())}


def baseline_availability(screened, baselines, left_out):
    """Per layout (its key names its acquisition mode): whether it has a baseline, its reference clips used and left
    out by reason, and its screened readable clips, those without a results-box score (`extra_cells`: incomplete,
    whatever the policy) and those excluded. Counts, not estimates."""
    keys = layout_keys(screened)
    out = {}
    for key in sorted(set(baselines) | set(left_out.index) | set(keys)):
        reasons = left_out[left_out.index == key]
        clips = screened[(keys == key).to_numpy() & screened.error.isna().to_numpy()]
        out[key] = {"baseline": baselines.get(key) is not None, "reference_clips": int(len(reasons)),
                    "reference_used": int((reasons == "").sum()),
                    "reference_left_out": {r: int(n) for r, n in reasons[reasons != ""].value_counts().items()},
                    "screened_clips": int(len(clips)), "incomplete": int(clips.extra_cells.isna().sum()),
                    "excluded": int(clips.exclude.sum())}
    return out


def _reference_clips(reference, sheet_rows, answers, thresholds):
    """The reference clips of a diagnostic sheet: whether each is eligible to shape its layout's baseline at
    `thresholds` (`reference_left_out`), against the answer to the measurement question (a reference clip with
    measurements makes its ink normal for its layout): left out for ink (marks inside or extra ink) against
    measured, and the weighted share of the eligible ones with measurements, overall and per acquisition mode."""
    sized = at_mark_size(reference, thresholds["inside_mark_px"])
    lab = _labelled(sized.assign(left_out=reference_left_out(sized).to_numpy()), sheet_rows, answers)

    def summary(part):
        measured, used = part.answer_measurement == "yes", part.left_out == ""
        return {"labelled": int(len(part)),
                "left_out": {r or "used": int(n) for r, n in part.left_out.value_counts().items()},
                "left_out_for_ink": confusion(part.left_out.isin(["marks_inside", "extra_ink"]), measured, part.weight),
                "used_with_measurement": int((used & measured).sum()),
                "used_with_measurement_share": (float(part.weight[used & measured].sum() / part.weight[used].sum())
                                                if used.any() else None)}

    return {**summary(lab), "by_acquisition": {str(acq): summary(part) for acq, part in
                                               lab.groupby(lab.acq_category.fillna("<missing>"), sort=True)}}


def calibration_report(features, reference, sheet_clips, answers, thresholds=THRESHOLDS, flags_are_population=True):
    """Population estimates for the screen at `thresholds`, from an audit or validation sheet's clips (global_key,
    video_path, stratum, inclusion_prob) and their human `answers` (read_labels), each clip weighted by 1 / its
    inclusion probability, which stays as drawn. The screen is recomputed from the screened clips' `features` and
    the `reference` features (`screen_flags`), so a threshold set is evaluated as `flag` would apply it.

    - per detector: sensitivity and specificity against the answer to its own question, and sweeps;
    - `policy` (a flag excludes a clip, no flag keeps it): the same against eligibility (no 'yes' to a reason to
      exclude); the share of readable clips excluded (`_excluded_share`); and among kept clips the share that are
      not eligible, with an exact upper bound when those clips were all drawn with one probability;
    - `cursor_clips`: the clips with a cursor line over the image (`_cursor_clips`);
    - `by_stratum` and `by_acquisition`: the measurement and policy confusions within each stratum (for a validation
      list, its groups) and each acquisition mode;
    - `inside_mark_px_sweep`: for each minimum mark size of INSIDE_MARK_SWEEP, the screen recomputed from the start
      (reference eligibility, baselines, extra_cells, flags) with the sample fixed: the measurement and policy
      confusions, overall and per acquisition mode, the excluded share, the cursor clips, and per layout whether it
      has a baseline (`baseline_availability`);
    - `reference_clips`, at the thresholds and for each size: a diagnostic sheet's reference clips
      (`_reference_clips`), which weigh up to the reference clips, not to the screened ones."""
    roles = sheet_clips["role"] if "role" in sheet_clips else pd.Series("target", index=sheet_clips.index)
    references, sheet_clips = sheet_clips[roles == "reference"], sheet_clips[roles != "reference"]
    screened = screen_flags(features, reference, thresholds)[0]
    lab = _labelled(screened, sheet_clips, answers)
    answer = {field: f"answer_{field}" for field in LABEL_FIELDS}
    report = {"labelled": int(len(lab)), "thresholds": thresholds}
    for flag in ("still", "measurement"):
        truth = lab[answer[flag]] == "yes"
        report[flag] = {"at_thresholds": confusion(lab[flag].fillna(False), truth, lab.weight)}
        if flag == "still":
            # Every still score, over the clips that have it (scan_* only with a DICOM scan area).
            report[flag]["sweeps"] = {}
            for column, rule, values in STILL_SWEEPS:
                if column not in lab:
                    continue
                scored = lab[lab[column].notna()]
                sweep_report = {"labelled": int(len(scored))}
                for v in values:
                    still = scored[column] < v if rule == "below" else scored[column] >= v
                    sweep_report[str(v)] = confusion(still, scored[answer[flag]] == "yes", scored.weight)
                report[flag]["sweeps"][column] = sweep_report
        else:
            extra = lab.extra_cells.fillna(-1)
            report[flag]["sweep_inside_glyphs"] = {
                str(v): confusion((lab.inside_glyphs >= v) | (extra >= thresholds["extra_cells"]), truth, lab.weight)
                for v in (1, 2, 3, 5, 8)}
            for column in (f"inside_glyphs_static{s:g}" for s in STATIC_STDS):  # the marks at a looser static level.
                if column in lab:
                    report[flag][f"sweep_{column}"] = {
                        str(v): confusion((lab[column] >= v) | (extra >= thresholds["extra_cells"]), truth,
                                          lab.weight)
                        for v in (1, 2, 3, 5, 8)}
            report[flag]["sweep_extra_cells"] = {
                str(v): confusion((lab.inside_glyphs >= thresholds["inside_glyphs"]) | (extra >= v), truth,
                                  lab.weight)
                for v in (2, 4, 8, 12, 20)}
    flagged = lab.exclude
    kept, kept_ineligible = lab[~flagged], lab.ineligible[~flagged]
    one_probability = kept.inclusion_prob.nunique() == 1
    report["policy"] = {
        "at_thresholds": confusion(flagged, lab.ineligible, lab.weight),
        "screened_clips": int(screened.error.isna().sum()),
        "excluded_share": _excluded_share(lab, screened, flags_are_population),
        "excluded_share_from": "all screened clips" if flags_are_population else "weighted sample",
        "kept": {
            "sampled": int(len(kept)), "ineligible": int(kept_ineligible.sum()),
            "ineligible_by_question": {f: int((kept[answer[f]] == "yes").sum()) for f in EXCLUSION_FIELDS},
            "share_ineligible": (float(kept.weight[kept_ineligible].sum() / kept.weight.sum())
                                 if len(kept) else None),
            # Exact only for an equal-probability sample; after a threshold change the kept clips can mix strata.
            "upper_95": upper_bound(int(kept_ineligible.sum()), len(kept)) if one_probability else None,
        },
    }
    report["cursor_clips"] = _cursor_clips(lab)

    def confusions(part):
        return {"labelled": int(len(part)),
                "measurement": confusion(part.measurement.fillna(False), part[answer["measurement"]] == "yes",
                                         part.weight),
                "policy": confusion(part.exclude, part.ineligible, part.weight)}

    report["by_stratum"] = {stratum: {**confusions(part),
                                      "still": confusion(part.still.fillna(False), part[answer["still"]] == "yes",
                                                         part.weight)}
                            for stratum, part in lab.groupby("stratum", sort=True)}
    report["by_acquisition"] = {str(acq): confusions(part)
                                for acq, part in lab.groupby(lab.acq_category.fillna("<missing>"), sort=True)}
    if len(references):
        report["reference_clips"] = _reference_clips(reference, references, answers, thresholds)
    report["inside_mark_px_sweep"] = {}
    for px in INSIDE_MARK_SWEEP:
        at = {**thresholds, "inside_mark_px": px}
        screened, baselines, _, left_out = screen_flags(features, reference, at)
        lab = _labelled(screened, sheet_clips, answers)
        acquisitions = screened.acq_category.fillna("<missing>")
        report["inside_mark_px_sweep"][str(px)] = {
            **confusions(lab),
            "excluded_share": _excluded_share(lab, screened, flags_are_population),
            "cursor_clips": _cursor_clips(lab),
            "by_acquisition": {str(acq): {**confusions(part), "excluded_share": _excluded_share(
                part, screened[(acquisitions == acq).to_numpy()], flags_are_population)}
                               for acq, part in lab.groupby(lab.acq_category.fillna("<missing>"), sort=True)},
            "baselines": baseline_availability(screened, baselines, left_out)}
        if len(references):
            report["inside_mark_px_sweep"][str(px)]["reference_clips"] = _reference_clips(reference, references,
                                                                                          answers, at)
    return report


def run_calibrate(args):
    # Every input as read (Inputs): the report names those, and is printed only if none changed meanwhile.
    inputs = Inputs(__file__)
    record = os.path.join(args.sheet_dir, "sheet.info.json")
    if not os.path.exists(record):
        raise SystemExit("--sheet-dir has no record of a completed sheet (sheet.info.json, written last).")
    sheet_info = inputs.json(record)
    if sheet_info.get("purpose") not in ("audit", "diagnostic", "validation"):
        raise SystemExit("--sheet-dir is a sheet of chosen clips, not a random sample: calibrate needs an audit "
                         "sample (`sheet` without --keys), a diagnostic sample (`sheet --diagnostic`) or a validation "
                         "list (`sheet --all-clips`).")
    if any(s["population"] and not s["sampled"] for s in sheet_info["strata"].values()):
        raise SystemExit("The audit sample left out a whole stratum, which no estimate can then cover.")
    sheet_clips = os.path.join(args.sheet_dir, "sheet_clips.csv")
    if not os.path.exists(sheet_clips) or inputs.digest(sheet_clips) != sheet_info.get("sheet_clips"):
        raise SystemExit("sheet_clips.csv is missing or changed since the sheet was made.")
    flags = inputs.parquet(completed(args.flags, "flag", inputs))
    drawn_from, here = sheet_info.get("population"), population(flags)
    if drawn_from != here:
        raise SystemExit(f"--flags does not screen the clips the audit sample was drawn from "
                         f"({here['readable_clips']} readable clips here, "
                         f"{(drawn_from or {}).get('readable_clips', 'none recorded')} there, or other keys): the "
                         "sample's weights describe those. Use the flags it was drawn from, or a rerun of `flag` on "
                         "the same features.")
    if sheet_info["purpose"] != "validation" and "sample_prob" in flags:
        raise SystemExit("The sheet sampled the flags of a validation list as if they were a whole screen: its "
                         "probabilities replace the list's own. Make the sheet with --all-clips.")
    sheet_clips = inputs.csv(sheet_clips, dtype=str)
    flag_info = inputs.json(args.flags + ".info.json")
    # The report recomputes the flags with this script's rule and names the flags' screen: both must be this script's.
    for part, recorded, current, rerun in (("flag rule", flag_info.get("rule"), flag_rule(), "flag"),
                                           ("pass 1", flag_info.get("extractor"), _code_record(*PASS1),
                                            "features and flag")):
        if not isinstance(recorded, dict) or not {"code", "constants"} <= set(recorded):
            raise SystemExit(f"--flags do not record their {part} (an older `flag` run): rerun {rerun}.")
        changed = _differences(recorded, current)
        if changed:
            raise SystemExit(f"--flags were made by another {part} than this script's, or in another environment "
                             f"(they differ in {', '.join(changed)}): its report would recompute them here while "
                             f"naming their screen. Rerun {rerun}.")
    thresholds = {**flag_info["thresholds"], **(json.loads(args.thresholds) if args.thresholds else {})}
    if set(thresholds) != set(THRESHOLDS):
        raise SystemExit(f"--thresholds keys must be among {sorted(THRESHOLDS)}.")
    # Every threshold set, a minimum mark size included, is recomputed from the features the flags were made from:
    # the size changes the reference clips' eligibility, so the baselines and extra_cells too.
    sources = screen_sources(args.flags, inputs)
    features, reference = (inputs.parquet(sources[role]["path"]) for role in ("features", "reference"))
    if "video_sha256" not in sheet_clips:
        raise SystemExit("The sheet does not record which video bytes its clips were copied from (an older sheet): "
                         "make it again.")
    screened = pd.concat([features, reference]).drop_duplicates("video_path").set_index("video_path").video_sha256
    if (sheet_clips.video_sha256 != sheet_clips.video_path.map(screened)).any():
        raise SystemExit("Some of the sheet's clips were copied from other video bytes than those --flags "
                         "screened (the videos were replaced, and features run again, since the sheet): the answers "
                         "describe other clips. Make a sheet of these flags.")
    if sheet_info["purpose"] == "diagnostic" and population(reference) != sheet_info.get("reference_population"):
        raise SystemExit("The reference clips behind --flags are not those the diagnostic sample was drawn from: its "
                         "weights describe those. Use the flags it was drawn from.")
    again = screen_flags(features, reference, flag_info["thresholds"])[0]
    if not again.set_index("video_path").exclude_reason.sort_index().equals(
            flags.set_index("video_path").exclude_reason.sort_index()):
        raise SystemExit("Recomputed at their own thresholds from the features they were made from, --flags do not "
                         "come back: rerun flag.")
    # A clip keeps its global key when its video is replaced in place: answers given on an earlier sheet, of the
    # replaced video, would be counted against the new one's flags.
    answers = read_labels(args.labels, inputs)
    given = answers.video_sha256.reindex(sheet_clips.global_key.to_numpy())
    stale = given.notna().to_numpy() & (given.to_numpy() != sheet_clips.video_sha256.to_numpy())
    if stale.any():
        raise SystemExit(f"{int(stale.sum())} of the sheet's {len(sheet_clips)} clips were answered for other video "
                         "bytes than this sheet's copies (on a sheet made before the video was replaced in place): "
                         "the answers describe other clips. Answer them on this sheet, and leave the older label files "
                         "out.")
    report = calibration_report(features, reference, sheet_clips, answers, thresholds,
                                flags_are_population=sheet_info["purpose"] in ("audit", "diagnostic"))
    report["provenance"] = {"flags": inputs.digest(args.flags), "sheet_drawn_from_flags": sheet_info["flags"],
                            "labels": {os.path.basename(p): inputs.digest(p) for p in args.labels},
                            "population": here, "screen": flag_info.get("screen")}
    inputs.check()
    print(json.dumps(report, indent=2))


def _code_record(*roots):
    """What an output computed by the functions `roots` depends on in the repository: the code of every function
    they run, from them down, as one SHA-256 (comments and docstrings left out, so editing those changes nothing),
    and the values of the module constants that code reads."""
    import ast
    import inspect
    import textwrap

    code, constants, todo = {}, {}, list(roots)
    while todo:
        fn = todo.pop()
        if fn.__name__ in code:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:
                node.body = node.body[1:] or [ast.Pass()]
        code[fn.__name__] = ast.unparse(tree)
        for name in sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & set(globals())):
            value = globals()[name]
            if inspect.isfunction(value):
                if os.path.abspath(inspect.getsourcefile(value)).startswith(REPO_ROOT + os.sep):
                    todo.append(value)
            elif value is None or isinstance(value, (int, float, str, tuple, dict, re.Pattern)):
                constants[name] = value
    digest = hashlib.sha256("\n\n".join(code[k] for k in sorted(code)).encode()).hexdigest()
    record = {"code": digest, "functions": sorted(code), "constants": constants}
    return json.loads(json.dumps(record, default=lambda p: {"pattern": p.pattern, "flags": int(p.flags)}))


def _versions(*names):
    """The versions of Python and of the libraries `names` (None for one not installed)."""
    import cv2
    from importlib import metadata

    loaded = {"numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__}

    def version(name):
        try:
            return loaded[name] if name in loaded else metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {"python": ".".join(map(str, sys.version_info[:3])), **{n: version(n) for n in names}}


def extractor():
    """What pass 1's output depends on besides the clips: its code and constants (`_code_record` of `PASS1`) and
    the versions of Python and of the libraries that decode and measure the frames."""
    return {**_code_record(*PASS1), "libraries": _versions("numpy", "opencv", "decord", "pydicom")}


def _differences(recorded, current):
    """What differs between a recorded `_code_record` and the current one: 'code', the constants by name, then,
    when the current one names its libraries, '<library> version'."""
    constants = {**recorded["constants"], **current["constants"]}
    libraries = {**recorded.get("libraries", {}), **current["libraries"]} if "libraries" in current else {}
    return ["code"] * (recorded["code"] != current["code"]) + sorted(
        k for k in constants if recorded["constants"].get(k, "missing") != current["constants"].get(k, "missing")
    ) + [f"{k} version" for k in sorted(libraries)
         if recorded.get("libraries", {}).get(k, "missing") != current["libraries"].get(k, "missing")]


def flag_rule():
    """The flag rule's `_code_record` (of RULE), without THRESHOLDS (they are recorded on their own, and a
    calibration changes them, not the rule), and the versions of Python and of the libraries the rule runs on (the
    baselines and extra_cells use OpenCV, NumPy and pandas): a calibration holds for that environment only."""
    rule = _code_record(*RULE)
    rule["constants"].pop("THRESHOLDS", None)
    return {**rule, "libraries": _versions("numpy", "pandas", "opencv")}


def extracted_by(path, pass1, inputs):
    """The extractor recorded by the completed `features` run behind `path` (read into `inputs`), once checked to be
    this script's pass 1 (`pass1`, its `_code_record`): the same code and constants. Features of other code, such as
    an earlier detector, would be flagged as if by this one."""
    record = inputs.json(completed(path, "features", inputs) + ".info.json").get("extractor")
    if not isinstance(record, dict) or not {"code", "constants", "libraries"} <= set(record):
        raise SystemExit(f"{os.path.basename(path)} does not record the code that extracted it (an older `features` "
                         "run): rerun features.")
    changed = _differences(record, pass1)
    if changed:
        raise SystemExit(f"{os.path.basename(path)} was extracted by other pass 1 code or constants than this "
                         f"script's (they differ in {', '.join(changed)}): rerun features.")
    return record


def code_sha256(*paths):
    """SHA-256 of the scripts behind an output, by path in the repository: the commit alone misses uncommitted and
    untracked edits."""
    return {os.path.relpath(os.path.abspath(p), REPO_ROOT): sha256(p) for p in paths}


def private_dir(path):
    """`path` as a folder only its owner can read, made so when missing: the outputs name patients' clips. An
    existing folder that others can read, or someone else's, is refused, not changed."""
    if not os.path.exists(path):
        os.makedirs(path, mode=0o700)
        os.chmod(path, 0o700)
    st = os.stat(path)
    if not os.path.isdir(path) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise SystemExit(f"{path} must be a folder of yours that only you can read (chmod 700): the outputs name "
                         "patients' clips.")
    return path


def clear_outputs(*paths):
    """Remove an earlier run's outputs first: a run that stops leaves nothing a later step could take for its
    result."""
    for path in paths:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.remove(path)


class Inputs:
    """The input files of one run, each read once: its SHA-256 is that of the bytes read, and its tables and records
    are parsed from those bytes, so what a record names is what the run used. Before a step writes its completion
    record, `check` stops it, its outputs removed, if a file was replaced or changed since it was read: the run used
    the old content, while a record written now could be checked against the new."""

    def __init__(self, *scripts):
        self.data = {}
        for script in scripts:  # the code is an input too: its SHA-256 is in every record.
            self.read(script)

    def read(self, path):
        key = os.path.abspath(path)
        if key not in self.data:
            with open(key, "rb") as f:
                self.data[key] = f.read()
        return self.data[key]

    def digest(self, path):
        """The SHA-256 of `path` as this run read it (read now if it was not yet)."""
        return hashlib.sha256(self.read(path)).hexdigest()

    def parquet(self, path, **kwargs):
        return pd.read_parquet(io.BytesIO(self.read(path)), **kwargs)

    def csv(self, path, **kwargs):
        return pd.read_csv(io.BytesIO(self.read(path)), **kwargs)

    def json(self, path):
        return json.loads(self.read(path))

    def check(self, *outputs):
        """Stop, removing `outputs`, unless every file read is still there with the content read."""
        changed = [os.path.basename(p) for p, data in self.data.items()
                   if not os.path.isfile(p) or sha256(p) != hashlib.sha256(data).hexdigest()]
        if changed:
            clear_outputs(*outputs)
            raise SystemExit(f"{', '.join(changed)} changed while this step ran: it used them as they were when "
                             "read, which a completion record would not name. Nothing was completed; rerun.")


def completed(path, step, inputs=None):
    """`path`, once checked to be the output of a completed `step` run and unchanged since: the run's info, written
    last, records the output's SHA-256. With `inputs` (Inputs), both files are read into it, so the output checked
    is the one the step then uses."""
    inputs = inputs or Inputs()
    try:
        recorded = inputs.json(path + ".info.json").get("output_sha256")
    except FileNotFoundError:
        recorded = None
    if not recorded or not os.path.isfile(path) or inputs.digest(path) != recorded:
        raise SystemExit(f"{os.path.basename(path)} is not the output of a completed `{step}` run, or changed "
                         f"since: rerun {step}.")
    return path


def _source(path, inputs=None):
    """An input as a record names it: where it was and its SHA-256 as read (`inputs`, else now)."""
    return {"path": os.path.abspath(path), "sha256": (inputs or Inputs()).digest(path)}


def screen_sources(flags_path, inputs=None):
    """The features files behind completed `flag` output and the clip lists they screened, {"features": {"path":
    ..., "clips": SHA-256 of the list}, "reference": {...}}, once each link of the chain is checked: the flags
    unchanged since `flag`; each features file `flag` read still there, unchanged since then and the output of a
    completed `features` run, whose record names the SHA-256 of the clip list it screened. A step that uses the
    flags compares these with its own clip lists. Every file of the chain is read into `inputs` (Inputs), so a step
    that then uses them uses what was checked."""
    inputs = inputs or Inputs()
    info = inputs.json(completed(flags_path, "flag", inputs) + ".info.json")
    sources = {}
    for role in ("features", "reference"):
        source = info.get(role)
        if not isinstance(source, dict):
            raise SystemExit(f"{os.path.basename(flags_path)} does not record which {role} file it was flagged "
                             "from (an older `flag` run): rerun flag.")
        path = completed(source["path"], "features", inputs)
        if inputs.digest(path) != source["sha256"]:
            raise SystemExit(f"The {role} file {os.path.basename(path)} changed since `flag` read it: rerun flag.")
        sources[role] = {"path": path, "clips": inputs.json(path + ".info.json").get("clips")}
    return sources


def _write_info(out_path, info):
    """`<out_path>.info.json`, written last: with the output's SHA-256, it marks a completed run (`completed`)."""
    info = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "code_commit": git_commit(), "code_sha256": code_sha256(__file__), **info}
    if os.path.isfile(out_path):
        info["output_sha256"] = sha256(out_path)
    with open(out_path + ".info.json", "w") as f:
        json.dump(info, f, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("features")
    f.add_argument("--clips", required=True, help="CSV with video_path and global_key, and dicom_path for the "
                                                  "DICOM region boxes (else the moving region is used); other "
                                                  "columns (view, acq_category, vendor, model) are kept. vendor, "
                                                  "model and acq_category define layouts.")
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--clip-timeout", type=int, default=600,
                   help="seconds a worker may spend on one clip before the run stops (a hung decoder).")
    g = sub.add_parser("flag")
    g.add_argument("--features", required=True)
    g.add_argument("--reference", required=True,
                   help="features of the reference clips, a sample of the same layouts outside the clips flagged, "
                        "that define where each layout normally has ink.")
    g.add_argument("--out", required=True)
    g.add_argument("--thresholds", help=f"JSON overriding {THRESHOLDS}")
    s = sub.add_parser("sheet")
    s.add_argument("--flags", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--keys", help="a sheet of these clips (CSV with global_key) instead of the audit sample; not a "
                                  "random sample, so calibrate refuses it.")
    s.add_argument("--all-clips", action="store_true",
                   help="a sheet of every readable clip of a validation list (screen_validation.py), weighted by its "
                        "sampling probabilities, instead of the audit sample.")
    s.add_argument("--diagnostic", action="store_true",
                   help="a diagnostic sample (`diagnostic_sample`) of the screened and the reference clips, per "
                        "acquisition mode and detector case, instead of the audit sample.")
    s.add_argument("--n-per-stratum", type=int, default=10, help="diagnostic sample: random clips per stratum.")
    s.add_argument("--n-flagged", type=int, default=50, help="audit sample: random flagged clips.")
    s.add_argument("--n-unflagged", type=int, default=100, help="audit sample: random unflagged clips.")
    s.add_argument("--per-page", type=int, default=40)
    s.add_argument("--seed", type=int, default=20261014)
    c = sub.add_parser("calibrate")
    c.add_argument("--flags", required=True)
    c.add_argument("--sheet-dir", required=True, help="an audit sample from `sheet`.")
    c.add_argument("--labels", required=True, nargs="+",
                   help="label files exported from sheet pages, of this sheet or others.")
    c.add_argument("--thresholds", help=f"JSON overriding the flags' thresholds, to evaluate another set exactly")
    args = p.parse_args(argv)
    umask = os.umask(0o077)  # every file written is the owner's only.
    try:
        {"features": run_features, "flag": run_flag, "sheet": run_sheet, "calibrate": run_calibrate}[args.command](
            args)
    finally:
        os.umask(umask)


if __name__ == "__main__":
    main()
