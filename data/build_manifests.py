# /data/build_manifests.py

"""Stage 2: Builds a clean, reproducible patient-level dataset split for EchoJEPA
from raw video metadata and optional study-level labels.

Example:
    python data/build_manifests.py --config configs/data/manifests_tte_10k_ef.yaml \\
        --metadata <video_metadata.parquet> --labels <study_labels.parquet> \\
        --out-dir <output_dir> --workers 32   # (balance memory and decoder overhead)

Archive mode (`selection.mode: all_studies`): every eligible study of every patient, with the
patient split anchored on an earlier build, a resumable verification cache and two training arms:
    python data/build_manifests.py --config configs/data/manifests_tte_archive.yaml \\
        --metadata <video_metadata.parquet> --anchor-splits <10k_output_dir>/studies.csv \\
        --acquisition-index <acquisition_index> --verify-cache <cache_dir> \\
        --out-dir <archive_output_dir> --workers 32

See data/README.md for the inputs, the outputs and how experiments use them.
"""

import argparse
import datetime
import functools
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from multiprocessing import get_context
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "val", "test")

# Create three independent random streams (RNG).
# Note: never to reorder `STAGES`, because that silently changes every cohort drawn with the same seed.
STAGES = ("patients", "study", "split")

# `selection.mode`: the pilot's one study per patient (default), or every eligible study of every patient.
ONE_STUDY, ALL_STUDIES = "one_study_per_patient", "all_studies"
MODES = (ONE_STUDY, ALL_STUDIES)

VIDEO_ROLES = ("patient_id", "study_id", "exam_type", "view", "video_path", "avi_status")

# Roles a configuration may map in addition to `VIDEO_ROLES`. `dicom_path` joins the acquisition index;
# `vendor` and `model` are only reported.
OPTIONAL_ROLES = ("study_date", "dicom_path", "vendor", "model")

# Roles that describe a study rather than one of its videos.
STUDY_ROLES = ("patient_id", "study_id", "exam_type", "study_date")

# The number of metadata rows per streamed batch.
BATCH_SIZE = 500000

# This is the ordered list of eligibility filters. The code applies them one by one,
# and after each filter it records how many rows remain.
RULES = ("metadata_videos", "allowed_exam_type", "status_ok", "labelled", "in_years")

# Placeholder used in counts when exam type or view is missing.
MISSING = "<missing>"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
#  Helper Functions
# --------------------------------------------------------------------------- #


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    try:
        check_config(cfg)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    return cfg


def check_config(cfg):
    """Check the configuration immediately, before doing any real processing, and stop
    with an understandable error if something is invalid.
    """

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and bool(np.isfinite(x))

    def positive_int(x):
        return isinstance(x, int) and not isinstance(x, bool) and x > 0

    def block(name):
        b = cfg.get(name)
        require(isinstance(b, dict), f"`{name}` must be a mapping, got {b!r}.")
        return b

    require(isinstance(cfg, dict), f"the configuration must be a mapping, got {type(cfg).__name__}.")
    columns = block("columns")
    missing = [r for r in VIDEO_ROLES if r not in columns]
    require(not missing, f"columns has no mapping for {missing}.")
    names = list(columns.values())
    require(all(isinstance(c, str) and c.strip() for c in names),
            f"columns must map each role to a non-empty column name, got {columns}.")
    repeated = sorted({c for c in names if names.count(c) > 1})
    require(not repeated, f"columns maps several roles to the same column {repeated}.")
    labels = block("labels")
    require(all(isinstance(labels.get(k), str) and labels[k].strip() for k in ("study_id", "value")),
            "labels needs non-empty column names for `study_id` and `value`.")
    low = labels.get("low_threshold")
    require(low is None or (number(low) and 0 < low <= 100),
            f"labels.low_threshold must be null or a number in (0, 100], got {low!r}.")

    split = block("split")
    fracs = [split.get(s) for s in SPLITS]
    require(all(number(f) and 0 <= f <= 1 for f in fracs),
            f"split needs a fraction in [0, 1] for each of {list(SPLITS)}, got {split}.")
    require(np.isclose(sum(fracs), 1.0), f"split fractions sum to {sum(fracs)}, not 1.")

    sel = block("selection")
    mode = sel.get("mode", ONE_STUDY)
    require(mode in MODES, f"selection.mode must be one of {list(MODES)}, got {mode!r}.")
    n = sel.get("n_patients")
    if mode == ONE_STUDY:
        require(positive_int(n), f"selection.n_patients must be a positive integer, got {n!r}.")
    else:
        # Every eligible patient is kept; a cohort size would silently be ignored.
        require(n is None, f"selection.n_patients must be absent or null with selection.mode {ALL_STUDIES}, "
                           f"got {n!r}: every eligible patient is kept.")
    seed = sel.get("seed")
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            f"selection.seed must be a non-negative integer, got {seed!r}.")
    if mode == ONE_STUDY:
        empty = [s for s, f, k in zip(SPLITS, fracs, split_sizes(n, fracs)) if f > 0 and k == 0]
        require(not empty, f"with {n} patients, split {empty} would get no patient; raise selection.n_patients.")

    el = block("eligibility")
    for key in ("exam_types", "status_ok"):
        require(isinstance(el.get(key), list) and el[key], f"eligibility.{key} must be a non-empty list.")
    require(isinstance(el.get("require_label"), bool),
            f"eligibility.require_label must be true or false, got {el.get('require_label')!r}.")
    years = el.get("years")
    require(years is None or (isinstance(years, list) and years and all(positive_int(y) for y in years)),
            f"eligibility.years must be null (all years) or a non-empty list of years, got {years!r}.")
    require(years is None or "study_date" in columns, "eligibility.years needs columns.study_date.")

    if "acquisition" in cfg:
        vendors = block("acquisition").get("bmode_vendors")
        require(vendors is None or (isinstance(vendors, list) and vendors
                                    and all(isinstance(v, str) and v.strip() for v in vendors)),
                f"acquisition.bmode_vendors must be null or a non-empty list of vendor names, got {vendors!r}.")
        require(vendors is None or "vendor" in columns, "acquisition.bmode_vendors needs columns.vendor.")

    if "clip" in cfg:
        clip = block("clip")
        require(positive_int(clip.get("frames_per_clip")),
                f"clip.frames_per_clip must be a positive integer, got {clip.get('frames_per_clip')!r}.")
        require(number(clip.get("fps")) and clip["fps"] > 0,
                f"clip.fps must be a positive number, got {clip.get('fps')!r}.")


def normalize_id(series):
    """Convert identifier fields into clean string values, removing extra literal
    quote characters. Otherwise, joins between tables may fail completely.
    """
    return series.astype("string").str.strip().str.strip("'\"")


def read_columns(path, columns):
    """Read only the specified `columns` from a Parquet or CSV table, assuming the
    entire resulting table is small enough to fit in memory.
    """
    return pd.concat(iter_table(path, columns, BATCH_SIZE), ignore_index=True)


def iter_table(source, columns, batch_size):
    """Read only the specified `columns`, processing the table in chunks (batches)
    of rows rather than loading the whole table into memory at once.
    """
    if isinstance(source, pd.DataFrame):
        check_columns(source.columns, columns, "video metadata")
        for start in range(0, len(source), batch_size):
            yield source.iloc[start : start + batch_size][columns]
    elif source.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.ParquetFile(source)
        check_columns(table.schema_arrow.names, columns, source)
        for batch in table.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()
    else:
        check_columns(pd.read_csv(source, nrows=0).columns, columns, source)
        yield from pd.read_csv(source, usecols=columns, dtype=str, chunksize=batch_size)


def check_columns(names, required, what):
    missing = [c for c in required if c not in set(names)]
    if missing:
        raise ValueError(f"{what} is missing required columns {missing}.")


def ensure_outside_repo(path, flag):
    """Refuse an output location inside this repository. Prevents generated files containing
    patient/study identifiers from being written anywhere inside the Git repository.
    """
    root, target = os.path.realpath(REPO_ROOT), os.path.realpath(path)
    if os.path.commonpath([root, target]) == root:
        raise SystemExit(f"{flag} {path} is inside the code repository ({root}). Generated files hold "
                         "patient and study identifiers; write them outside the repository.")


# --------------------------------------------------------------------------- #
#  Loading & Eligibility
# --------------------------------------------------------------------------- #


def video_roles(cols):
    return list(VIDEO_ROLES) + [r for r in OPTIONAL_ROLES if r in cols]


def standardize(batch, cols):
    """Take one batch of metadata, rename its columns to standardized role names,
    and normalize identifier values so equivalent IDs are treated as identical
    even if they were formatted differently.
    """
    roles = video_roles(cols)
    v = batch[[cols[r] for r in roles]].set_axis(roles, axis=1)
    # An empty path is no path: both are reported as `no_path` when verified.
    v = v.assign(patient_id=normalize_id(v.patient_id), study_id=normalize_id(v.study_id),
                 video_path=v.video_path.astype("string").replace("", pd.NA))
    if "study_date" in v:
        v = v.assign(study_date=v.study_date.astype("string").str.strip())
    if "dicom_path" in v:
        v = v.assign(dicom_path=v.dicom_path.astype("string").replace("", pd.NA))
    for role in ("vendor", "model"):
        if role in v:
            v = v.assign(**{role: v[role].astype("string").str.strip()})
    return v


def drop_repeated(v, seen):
    """Remove duplicate video rows while streaming through the metadata, including duplicates that
    appeared in earlier batches, so that each video contributes only once to all downstream counts.

    `seen` contains sorted hashes of previously retained rows and is updated and returned. Only rows
    with a video path are deduplicated; rows without a path cannot be reliably matched, so each is
    counted separately.
    """
    with_path = np.flatnonzero(v.video_path.notna().to_numpy())
    h = pd.util.hash_pandas_object(v.iloc[with_path], index=False).to_numpy()
    new, first = np.unique(h, return_index=True)
    pos = np.searchsorted(seen, new)
    known = pos < len(seen)
    known[known] = seen[pos[known]] == new[known]
    keep = np.ones(len(v), bool)
    keep[with_path] = False
    keep[with_path[first[~known]]] = True
    return v[keep], np.insert(seen, pos[~known], new[~known])


def eligible_rows(v, cfg, label_ids, count=True):
    """Filter one standardized metadata batch and keep only the rows that satisfy every
    eligibility rule. 
    If `count` is set, also returns how many videos of each (exam type, view) remain
    after each rule is applied: a DataFrame with one column per rule.
    """
    el = cfg["eligibility"]
    left = {}

    def tally(rule, rows):
        if count:
            by = [rows[c].fillna(MISSING).replace("", MISSING).rename(c) for c in ("exam_type", "view")]
            left[rule] = rows.groupby(by).size()

    tally("metadata_videos", v)
    v = v[v.exam_type.isin(el["exam_types"])]
    tally("allowed_exam_type", v)
    v = v[v.avi_status.isin(el["status_ok"])]
    tally("status_ok", v)
    if el.get("require_label"):
        v = v[v.study_id.isin(label_ids)]
    tally("labelled", v)
    if el.get("years"):
        v = v[v.study_date.str[:4].isin({str(y) for y in el["years"]})]
    tally("in_years", v)

    # Keep videos with missing paths so verification can explicitly report them as `no_path`.
    for role in ("patient_id", "study_id"):
        n = v[role].isna().sum()
        if n:
            raise ValueError(f"{n} eligible videos have no {role}.")
    return v, (pd.DataFrame(left, columns=list(RULES)).fillna(0) if count else None)


def table_dict(df):
    """A count table as {row: {column: int}}, for JSON."""
    return {str(k): {c: int(x) for c, x in row.items()} for k, row in df.iterrows()}


def scan_eligible(source, cfg, labels, batch_size=BATCH_SIZE):
    """Determine eligible studies by one streaming pass over the video metadata.

    Returns one row per study (with its label) and eligibility counts. While scanning,
    reject inconsistent metadata that could make downstream train/val/test splitting
    or manifest creation unclear.
    """
    studies, counts, _ = scan_metadata(source, cfg, labels, batch_size)
    return studies, counts


def scan_metadata(source, cfg, labels, batch_size=BATCH_SIZE, keep=None):
    """`scan_eligible`, also returning the `keep` columns of every eligible video row (None when
    `keep` is None). The archive build keeps all its videos from this single pass instead of
    reading the metadata a second time; the rows are already free of repeats.
    """
    el, cols = cfg["eligibility"], cfg["columns"]
    if el.get("require_label") and labels is None:
        raise ValueError("eligibility.require_label is set but no label table was given.")
    if el.get("years") and "study_date" not in cols:
        raise ValueError("eligibility.years needs columns.study_date")

    label_ids = set(labels.study_id) if labels is not None else None
    total, parts, path_hash, kept = None, [], [], []
    seen, n_rows = np.empty(0, np.uint64), 0
    for batch in iter_table(source, [cols[r] for r in video_roles(cols)], batch_size):
        n_rows += len(batch)
        # Drop duplicate rows before computing any counts, so duplicated exports do not
        # change the reported totals.
        v, seen = drop_repeated(standardize(batch, cols), seen)
        v, tally = eligible_rows(v, cfg, label_ids)
        # Total video counts by (exam type, view) after each eligibility rule,
        # accumulated across metadata batches.
        total = tally if total is None else total.add(tally, fill_value=0)
        path_hash.append(pd.util.hash_pandas_object(v.video_path.dropna(), index=False).to_numpy())
        if keep is not None:
            kept.append(v[list(keep)])
        parts.append(v[[c for c in v.columns if c in STUDY_ROLES]].drop_duplicates())
    if not parts:
        raise ValueError("The video metadata has no rows.")
    total = total.fillna(0).astype(int)
    left = {rule: int(n) for rule, n in total.sum().items()}
    if n_rows > left["metadata_videos"]:
        print(f"[dupes]  ignored {n_rows - left['metadata_videos']} repeated video rows.")

    # Exact duplicate rows have already been removed, so any path that still appears multiple
    # times must be associated with conflicting IDs, exam types, or views. Store path hashes
    # instead of full path strings to reduce memory use.
    paths = pd.Series(np.concatenate(path_hash))
    conflict = paths.duplicated(keep=False)
    if conflict.any():
        extra = [r for r in ("dicom_path", "vendor", "model") if r in cols]
        raise ValueError(f"{paths[conflict].nunique()} video paths appear under "
                         "conflicting ids, exam types or views" + (f" or {', '.join(extra)}" if extra else "") + ".")
    # Check that each study -> patient / exam type mapping is unique.
    studies = pd.concat(parts, ignore_index=True)
    ids = studies[["study_id", "patient_id", "exam_type"]].drop_duplicates()
    bad = ids.study_id.duplicated(keep=False)
    if bad.any():
        raise ValueError(f"{ids.loc[bad, 'study_id'].nunique()} studies map to more than one patient or exam type.")
    by = ["study_id"] + (["study_date"] if "study_date" in studies else [])
    studies = attach_labels(studies.sort_values(by).drop_duplicates("study_id"), labels)

    # Eligibility counts overall, by exam type and by view.
    by_exam = total.groupby(level="exam_type").sum()
    by_exam["eligible_studies"] = studies.groupby("exam_type").size().reindex(by_exam.index, fill_value=0)
    by_exam = by_exam.sort_values(["in_years", "metadata_videos"], ascending=False)
    by_view = total.groupby(level="view").sum().sort_values(["in_years", "metadata_videos"], ascending=False)

    counts = {"eligible_videos": left["in_years"], "eligible_studies": int(len(studies)),
              "eligible_patients": int(studies.patient_id.nunique()), "eligibility_funnel": left,
              "eligibility_by_exam_type": table_dict(by_exam), "eligibility_by_view": table_dict(by_view)}
    print(f"[elig]   {left['metadata_videos']} videos -> {left['allowed_exam_type']} allowed exam type "
          f"-> {left['status_ok']} status OK -> {left['labelled']} labelled study -> {left['in_years']} in year range.")
    print(f"[elig]   {counts['eligible_studies']} studies, {counts['eligible_patients']} patients eligible.")
    print("\n[elig]   videos left after each rule, by exam type\n" + by_exam.to_string())
    print("\n[elig]   videos left after each rule, by view\n" + by_view.to_string())
    rows = pd.concat(kept, ignore_index=True) if keep is not None else None
    return studies.reset_index(drop=True), counts, rows


def fetch_videos(source, cfg, labels, study_ids, batch_size=BATCH_SIZE):
    """Scan the metadata once and return the video-level rows that are eligible and
    belong to the specified studies.
    """
    cols = cfg["columns"]
    label_ids = set(labels.study_id) if labels is not None else None
    wanted = set(study_ids)
    parts = []
    for batch in iter_table(source, [cols[r] for r in video_roles(cols)], batch_size):
        v, _ = eligible_rows(standardize(batch, cols), cfg, label_ids, count=False)
        parts.append(v[v.study_id.isin(wanted)])
    videos = pd.concat(parts, ignore_index=True)
    # Same deduplication as during the earlier metadata scan: exact duplicate rows are
    # removed, but rows with no video path are not deduplicated and each one is counted
    # as a separate video.
    videos = videos[videos.video_path.isna() | ~videos.duplicated()]
    missing = wanted - set(videos.study_id)
    if missing:
        raise ValueError(f"{len(missing)} studies disappeared from the video metadata between the two reads.")
    return attach_labels(videos, labels)


def attach_labels(df, labels):
    if labels is None:
        return df.assign(label=np.nan)
    return df.merge(labels, on="study_id", how="left")


def parse_labels(raw):
    """Converts valid EF values (number in (0, 100]) into numbers and invalid values into `NaN`.
    Also returns how many values were dropped for each dropping reason.
    """
    if pd.api.types.is_numeric_dtype(raw):
        # Preserve numeric values directly; converting them to text and back could alter their precision.
        value = raw.astype(float)
        empty = value.isna()
    else:
        text = raw.astype("string").str.strip()
        empty = text.isna() | (text == "")
        value = pd.to_numeric(text.mask(empty), errors="coerce").astype(float)
    dropped = {
        "empty": int(empty.sum()),
        "invalid": int((~empty & value.isna()).sum()),
        "zero": int((value == 0).sum()),
        "negative": int((value < 0).sum()),
        "above_100": int((value > 100).sum()),
    }
    return value.where((value > 0) & (value <= 100)), dropped


def load_labels(labels, cfg):
    """Create `study id`->`label` mapping, one row per study. Labels that are not a number
    in (0, 100] are dropped and counted, as in Stage 1, so a label table from elsewhere
    gets the same check. If the same study ID appears multiple times with different
    labels, raise an error.
    """
    lc = cfg["labels"]
    check_columns(labels.columns, [lc["study_id"], lc["value"]], "label table")
    value, dropped = parse_labels(labels[lc["value"]])
    out = pd.DataFrame({"study_id": normalize_id(labels[lc["study_id"]]), "label": value}).dropna()
    print(f"[labels] {len(labels)} label rows -> {len(out)} with an EF in (0, 100]; dropped "
          + ", ".join(f"{n} {why}" for why, n in dropped.items()))
    out = out.drop_duplicates()
    dup = out.study_id.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{out.loc[dup, 'study_id'].nunique()} studies carry conflicting labels.")
    return out


# --------------------------------------------------------------------------- #
#  Verification & Selection & Split
# --------------------------------------------------------------------------- #


class Check(NamedTuple):
    """Result of verifying one video: reason is None when the video is usable,
    otherwise it contains the failure reason.
    """

    reason: str | None
    n_frames: int | None = None  # decoded frame count.
    fps: float | None = None     # average frame rate reported by the decoder; `None` if unusable.


def verify_video(path, chunk=32):
    """Verify that the complete video is readable and record its frame count and FPS when
    possible. Missing or invalid FPS does not make the video unusable; it simply causes
    downstream sampling to use a fallback frame step of 1. Full decoding is done in chunks
    to keep memory usage bounded.

    Note: `chunk=32`: balance memory and decoder overhead.
    """
    if not isinstance(path, str) or not path: # Path missing.
        return Check("no_path")
    if any(c.isspace() for c in path):        # Contains whitespace.
        return Check("whitespace_in_path")
    try:
        size = os.path.getsize(path)
    except OSError:                           # File missing.
        return Check("missing")
    if size == 0:                             # 0 bytes.
        return Check("empty")

    from decord import VideoReader, cpu

    try:
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        n = len(vr)
    except Exception:
        return Check("undecodable")           # Not decodable.
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = float("nan")
    fps = fps if np.isfinite(fps) and fps > 0 else None # No usable rate: recorded as unknown.
    if n == 0:
        return Check("no_frames", n, fps)               # 0 frames.
    try:
        for start in range(0, n, chunk):
            idx = list(range(start, min(start + chunk, n)))
            if vr.get_batch(idx).shape[0] != len(idx):  # Frames are corrupted.
                return Check("corrupt_frames", n, fps)
    except Exception:                                   # Frames are corrupted.
        return Check("corrupt_frames", n, fps)
    return Check(None, n, fps)


def needs_padding(n_frames, fps, clip):
    """Whether each video is shorter than one clip of the pretraining loader.

    Like the loader, one clip spans `frames_per_clip` frames taken every
    max(1, ceil(video fps) // fps) frames, or every frame (the fallback frame step of 1)
    for a video without a usable fps (`NaN`); a shorter video gets padded. Unknown (`NA`)
    when the video could not be read.
    """
    step = (np.ceil(fps.astype(float)) // clip["fps"]).fillna(1).clip(lower=1)
    short = n_frames.astype(float) < clip["frames_per_clip"] * step
    return short.astype("boolean").mask(n_frames.isna())


def stage_rngs(seed):
    children = np.random.SeedSequence(seed).spawn(len(STAGES))
    return {name: np.random.default_rng(child) for name, child in zip(STAGES, children)}


def select_cohort(studies, n_patients, rngs, validate, fetch, workers=1):
    """Select the first n patients in seeded order for whom at least one eligible
    study contains a valid video.

    Candidate patients are processed in blocks, with all of their studies fetched so another
    study can be used if the preferred one fails video validation. Videos are validated in
    batches for efficiency; these batching choices do not affect which patients or studies
    are selected.

    Returns a tuple of four:
        chosen:  {patient_id: study_id}, the cohort.
        verdict: {video_path: Check} for every video with a path verified, including videos
                 from rejected studies and extra candidate patients that were ultimately not
                 selected.
        tried:   DataFrame of the video rows of every study every study considered during
                 selection, including selected and rejected studies and rows with missing paths.
        stats:   {"studies_without_valid_video": n, "patients_without_valid_study": n}.
    """
    # Get every eligible patient and sort them so that the outcome is independent of metadata row ordering.
    studies = studies[["patient_id", "study_id"]].sort_values(["patient_id", "study_id"])
    patients = np.sort(studies.patient_id.unique())
    if n_patients > len(patients):
        raise ValueError(f"Asked for {n_patients} patients but only {len(patients)} are eligible.")
    order = rngs["patients"].permutation(patients)

    # Each patient may have several studies, so generate a random number for each of their studies.
    studies = studies.assign(_key=rngs["study"].random(len(studies)))
    preference = studies.sort_values(["patient_id", "_key"]).groupby("patient_id").study_id.agg(list)

    fetched, paths, tried, loaded = [], {}, set(), 0
    verdict, chosen, pos = {}, {}, 0
    stats = {"studies_without_valid_video": 0, "patients_without_valid_study": 0}
    # Use multiprocessing with `spawn` rather than `fork` because the current process already
    # has active threads from Arrow/decord. Forking a process that already has threads can cause
    # child workers to hang during video verification.
    pool_ctx = get_context("spawn").Pool(workers) if workers > 1 else nullcontext()
    with pool_ctx as pool:

        def run(f, xs):
            out = []
            for i, r in enumerate(pool.imap(f, xs, chunksize=32) if pool else map(f, xs), 1):
                out.append(r)
                if i % 25_000 == 0:
                    print(f"[verify] {i}/{len(xs)} videos checked.", flush=True)
            return out

        # Start from the shuffled order and verifies enough candidates to reach the desired
        # amount of usable patients to avoid expensive verification of the entire dataset.
        while len(chosen) < n_patients and pos < len(order):
            need = n_patients - len(chosen)
            margin = max(8, need // 50)
            batch = order[pos : pos + need + margin]
            pos += len(batch)
            if pos > loaded:
                # Performance optimization: read one margin ahead, so a short next round needs
                # no new pass over the metadata.
                block = order[loaded : pos + margin]
                videos = fetch([s for p in block for s in preference[p]])
                fetched.append(videos)
                # Only path-bearing videos require file verification; a study with no video
                # paths cannot have a valid video.
                paths.update(videos.dropna(subset=["video_path"]).sort_values("video_path")
                             .groupby("study_id").video_path.agg(list))
                loaded += len(block)
            pending = {p: list(preference[p]) for p in batch}
            resolved = {}
            while pending:
                todo = [x for p in pending for x in paths.get(pending[p][0], []) if x not in verdict]
                verdict.update(zip(todo, run(validate, todo)))
                for p in list(pending):
                    study = pending[p].pop(0)
                    tried.add(study)
                    # An usable study must contain at least one valid video.
                    if any(verdict[x].reason is None for x in paths.get(study, [])):
                        resolved[p] = study
                    else:
                        stats["studies_without_valid_video"] += 1
                        if pending[p]:
                            continue
                        resolved[p] = None
                        stats["patients_without_valid_study"] += 1
                    del pending[p]
            for p in batch:
                if resolved[p] is not None and len(chosen) < n_patients:
                    chosen[p] = resolved[p]
            print(f"[verify] {len(chosen)}/{n_patients} patients selected, "
                  f"{len(verdict)} videos verified so far.")

    if len(chosen) < n_patients:
        raise ValueError(f"Only {len(chosen)} patients have a study with a valid video.")
    videos = pd.concat(fetched, ignore_index=True)
    return chosen, verdict, videos[videos.study_id.isin(tried)], stats


def split_sizes(n, fracs):
    """Exact patient counts of train, val and test: train and val rounded, test the rest."""
    n_train, n_val = round(n * fracs[0]), round(n * fracs[1])
    return n_train, n_val, n - n_train - n_val


def split_patients(patients, fracs, rng):
    """Split patients by first shuffling them into a reproducible seeded order, then
    assigning an exact number of patients to train, validation, and test dataset based on
    the requested fractions.
    """
    order = rng.permutation(np.sort(np.asarray(patients)))
    n_train, n_val, _ = split_sizes(len(order), fracs)
    split = pd.Series("test", index=order)
    split.iloc[:n_train] = "train"
    split.iloc[n_train : n_train + n_val] = "val"
    return split


def build(videos, labels, cfg, validate=verify_video, workers=1, batch_size=BATCH_SIZE):
    """Build the selected cohort, its train/validation/test assignment, and the validation
    status of individual videos. An optional raw label table can also be supplied.

    Returns a dict of `DataFrames` and its eligibility counts.
    """
    # Fail before any work on a malformed configuration.
    check_config(cfg)
    n_patients, fracs = cfg["selection"]["n_patients"], [cfg["split"][s] for s in SPLITS]
    lab = load_labels(labels, cfg) if labels is not None else None
    studies, counts = scan_eligible(videos, cfg, lab, batch_size)

    seed = cfg["selection"]["seed"]
    rngs = stage_rngs(seed)
    fetch = functools.partial(fetch_videos, videos, cfg, lab, batch_size=batch_size)
    chosen, verdict, tried, stats = select_cohort(studies, n_patients, rngs, validate, fetch, workers)
    split = split_patients(list(chosen), fracs, rngs["split"])

    def check(path):
        # A video without a path has no file to check.
        return verdict[path] if isinstance(path, str) else Check("no_path")

    # Verification counts cover every video of every study tried, with or without a path.
    failed = pd.Series([check(p).reason for p in tried.video_path], dtype=object).dropna()

    # The chosen studies' videos inherit the patient's split; the verification adds the
    # reason, frame count and fps.
    sel = tried[tried.study_id.isin(set(chosen.values()))]
    checks = [check(p) for p in sel.video_path]
    sel = sel.assign(
        split=sel.patient_id.map(split),
        reason=[c.reason for c in checks],
        n_frames=pd.array([c.n_frames for c in checks], dtype="Int64"),
        fps=np.array([c.fps for c in checks], dtype=float),
    )
    if "clip" in cfg:
        sel = sel.assign(needs_padding=needs_padding(sel.n_frames, sel.fps, cfg["clip"]))
    ok = sel.reason.isna()
    kept = sel[ok].drop(columns="reason").sort_values(["split", "video_path"])
    excluded = sel[~ok][["split", "patient_id", "study_id", "exam_type", "view", "video_path", "reason"]]

    # Verification counts, and the excluded videos within selected studies grouped by exam type and by view.
    counts = {
        **counts,
        "verification": {"videos_verified": len(tried),
                         "failed_by_reason": {k: int(n) for k, n in failed.value_counts().sort_index().items()},
                         **stats},
        "excluded_by_exam_type": table_dict(pd.crosstab(excluded.exam_type.fillna(MISSING), excluded.reason)),
        "excluded_by_view": table_dict(pd.crosstab(excluded.view.fillna(MISSING), excluded.reason)),
    }

    # Reconstruct the study table from the valid videos.
    studies = (
        kept.groupby("study_id")
        .agg(
            split=("split", "first"), patient_id=("patient_id", "first"),
            exam_type=("exam_type", "first"), label=("label", "first"),
            **({"study_date": ("study_date", "first")} if "study_date" in kept else {}),
            n_videos=("video_path", "size"),
        )
        .reset_index()
        .merge(excluded.groupby("study_id").size().rename("n_excluded").reset_index(), on="study_id", how="left")
        .fillna({"n_excluded": 0})
        .astype({"n_excluded": int})
        .sort_values(["split", "patient_id"])
    )
    check_split(studies, kept)
    return {"videos": kept.reset_index(drop=True), "studies": studies.reset_index(drop=True),
            "excluded": excluded.sort_values(["video_path", "study_id", "view"]).reset_index(drop=True),
            "counts": counts, "seed": seed}


def check_split(studies, videos):
    """Verify three invariants:
    1. One patient -> one chosen study mapping.
    2. Data leakage checking.
    3. Whitespace checking.
    """
    if studies.patient_id.duplicated().any():
        raise AssertionError("A patient has more than one selected study.")
    groups = {s: set(studies.loc[studies.split == s, "patient_id"]) for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise AssertionError(f"Patients shared between {a} and {b}.")
    if videos.video_path.str.contains(r"\s", regex=True).any():
        raise AssertionError("A manifest path contains whitespace.")


# --------------------------------------------------------------------------- #
#  Outputs
# --------------------------------------------------------------------------- #


def sha256(path, chunk=1 << 20):
    """SHA-256 the inputs so that we can distinguish files with same filename but have different bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def split_report(result, cfg):
    """Calculate summary statistics separately for each dataset split (train/validation/test),
    and use those statistics both in the program logs and in `manifest_info.json`.
    """
    low = cfg["labels"].get("low_threshold")
    report = {}
    for s in SPLITS:
        st = result["studies"][result["studies"].split == s]
        vi = result["videos"][result["videos"].split == s]
        ex = result["excluded"][result["excluded"].split == s]
        entry = {
            "patients": int(st.patient_id.nunique()),
            "studies": int(len(st)),
            "videos": int(len(vi)),
            "excluded_videos": {k: int(n) for k, n in ex.reason.value_counts().sort_index().items()},
            "views": {k: int(n) for k, n in vi.view.value_counts().sort_index().items()},
        }
        lab = st.label.dropna()
        if len(lab):
            entry["label"] = {"mean": float(lab.mean()), "std": float(lab.std()),
                              "min": float(lab.min()), "median": float(lab.median()), "max": float(lab.max())}
            if low is not None:
                entry["label"][f"share_below_{low}"] = float((lab < low).mean())
        if "study_date" in st:
            entry["years"] = {k: int(n) for k, n in st.study_date.str[:4].value_counts().sort_index().items()}
        if "n_frames" in vi and vi.n_frames.notna().any():
            entry["frames"] = {"median": float(vi.n_frames.median()), "min": int(vi.n_frames.min()),
                               "max": int(vi.n_frames.max())}
        if "needs_padding" in vi and vi.needs_padding.notna().any():
            entry["videos_needing_padding"] = int(vi.needs_padding.sum())
        report[s] = entry
    return report


def print_report(report):
    print("\n[split]  patient-level, one study per patient")
    for s, e in report.items():
        lab = e.get("label", {})
        low = next((f" {k.replace('share_below_', '<')} {v * 100:.1f}%" for k, v in lab.items()
                    if k.startswith("share_below_")), "")
        print(f"  {s:<5} {e['patients']:>6} patients {e['videos']:>8} videos"
              + (f"  label {lab['mean']:.1f}+-{lab['std']:.1f}{low}" if lab else "")
              + f"  excluded {sum(e['excluded_videos'].values())}"
              + (f"  needing padding {e['videos_needing_padding']}" if "videos_needing_padding" in e else ""))
    views = pd.DataFrame({s: e["views"] for s, e in report.items()}).fillna(0).astype(int)
    print("\n[views]  videos per view and split\n" + views.to_string())


def git_commit():
    """Records which version of the code produced the manifests in `manifest_info.json`."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), check=True).stdout.strip()
    except Exception:
        return None


def write_outputs(result, out_dir, cfg, inputs, argv):
    """Write output files.

    `videos.csv`: one row per valid selected video. Richer metadata table.
    `train.csv`: for data loader.
    `excluded_videos.csv`: tells which video was rejected.
    `manifest_info.json`: provenance.
    `summary.md`: the aggregate numbers only, safe to share.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for s in SPLITS:
        # `VideoDataset` parser expects a path label, so replace the real EF label with a
        # placeholder `0` for pretraining.
        # The actual EF is kept separately in `videos.csv` and `studies.csv` for downstream
        # evaluation, and is attached through `attach_labels`.
        rows = result["videos"].loc[result["videos"].split == s, ["video_path"]].assign(label=0)
        path = os.path.join(out_dir, f"{s}.csv")
        rows.to_csv(path, sep=" ", header=False, index=False)
        written[f"{s}.csv"] = len(rows)

    for name in ("videos", "studies", "excluded"):
        fname = "excluded_videos.csv" if name == "excluded" else f"{name}.csv"
        result[name].to_csv(os.path.join(out_dir, fname), index=False)
        written[fname] = len(result[name])

    report = split_report(result, cfg)
    info = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": argv,
        "git_commit": git_commit(),
        "seed": result["seed"],
        "config": cfg,
        "inputs": inputs,
        "counts": {**result["counts"], "selected_patients": int(len(result["studies"])),
                   "selected_videos": int(len(result["videos"])),
                   "excluded_videos": int(len(result["excluded"]))},
        "splits": report,
        "outputs": {f: {"rows": n, "sha256": sha256(os.path.join(out_dir, f))} for f, n in written.items()},
    }
    with open(os.path.join(out_dir, "manifest_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary_markdown(info))
    return report


def markdown_table(df):
    """A DataFrame as a Markdown table, with its index as the first column."""
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summary_markdown(info):
    """Create a Markdown summary containing only the high-level statistics from `manifest_info.json`,
    while removing anything that could identify data locations, specific patients/studies, or
    the exact command that was run.
    """
    cfg, counts = info["config"], info["counts"]
    el, sel, clip, ver = cfg["eligibility"], cfg["selection"], cfg.get("clip"), counts["verification"]

    def counts_table(rows, name):
        """{row: {column: n}} as a Markdown table."""
        return markdown_table(pd.DataFrame(rows).T.fillna(0).astype(int).rename_axis(name))

    lines = [
        "# Dataset manifest summary", "",
        "Aggregate counts only: no file paths, patient or study identifiers.", "",
        f"- Created (UTC): {info['created_utc']}",
        f"- Code version (git commit): {info['git_commit'] or 'unknown'}",
        f"- Seed: {info['seed']}",
        "", "## Cohort definition", "",
        f"- Exam types: {', '.join(el['exam_types'])}",
        f"- Video status: {', '.join(el['status_ok'])}",
        "- Valid video: file exists and is non-empty, every frame decodes",
        f"- Study label required: {'yes' if el.get('require_label') else 'no'}",
        f"- Years: {', '.join(map(str, el['years'])) if el.get('years') else 'all'}",
        f"- Patients, one study each: {sel['n_patients']}",
        "- Patient split: " + ", ".join(f"{s} {cfg['split'][s]}" for s in SPLITS),
    ]
    if clip:
        lines.append(f"- Padding flag: shorter than one clip of {clip['frames_per_clip']} frames at {clip['fps']} fps "
                     "(every frame for a video without a usable frame rate).")

    funnel = pd.Series(counts["eligibility_funnel"], name="videos").rename_axis("rule").to_frame()
    lines += ["", "## Eligibility", "", markdown_table(funnel), "",
              f"{counts['eligible_studies']} studies from {counts['eligible_patients']} patients are eligible.",
              "", "### Videos left after each rule, by exam type", "",
              counts_table(counts["eligibility_by_exam_type"], "exam type"),
              "", "### Videos left after each rule, by view", "",
              counts_table(counts["eligibility_by_view"], "view")]

    lines += ["", "## Verification", "",
              f"- Videos verified: {ver['videos_verified']}",
              f"- Studies tried without a valid video: {ver['studies_without_valid_video']}",
              f"- Patients skipped for lack of a study with a valid video: {ver['patients_without_valid_study']}"]
    if ver["failed_by_reason"]:
        failed = pd.Series(ver["failed_by_reason"], name="videos").rename_axis("reason").to_frame()
        lines += ["", "Failed videos, among all verified:", "", markdown_table(failed)]
    for key, name in (("excluded_by_exam_type", "exam type"), ("excluded_by_view", "view")):
        if counts[key]:
            lines += ["", f"### Excluded videos of the selected studies, by {name}", "",
                      counts_table(counts[key], name)]

    rows, threshold = {}, cfg["labels"].get("low_threshold")
    for s, e in info["splits"].items():
        lab = e.get("label", {})
        rows[s] = {"patients": e["patients"], "studies": e["studies"], "videos": e["videos"],
                   "excluded videos": sum(e["excluded_videos"].values()),
                   "videos needing padding": e.get("videos_needing_padding", ""),
                   "median frames": e.get("frames", {}).get("median", ""),
                   "label mean ± SD": f"{lab['mean']:.1f} ± {lab['std']:.1f}" if lab else ""}
        # The share of studies below `labels.low_threshold`, only when one is set.
        if threshold is not None:
            rows[s][f"label < {threshold}"] = next(
                (f"{v * 100:.1f}%" for k, v in lab.items() if k.startswith("share_below_")), "")
    views = pd.DataFrame({s: e["views"] for s, e in info["splits"].items()}).fillna(0).astype(int)
    lines += ["", "## Splits", "", markdown_table(pd.DataFrame(rows).T.rename_axis("split")),
              "", "### Videos per view", "", markdown_table(views.rename_axis("view"))]
    if all("years" in e for e in info["splits"].values()):
        years = pd.DataFrame({s: e["years"] for s, e in info["splits"].items()}).fillna(0).astype(int)
        lines += ["", "### Studies per year", "", markdown_table(years.rename_axis("year"))]

    lines += ["", "## Checksums (SHA-256)", "", "| file | size | sha256 |", "|---|---|---|"]
    lines += [f"| input: {role} | {meta['bytes']} bytes | {meta['sha256']} |" for role, meta in info["inputs"].items()]
    lines += [f"| {name} | {meta['rows']} rows | {meta['sha256']} |" for name, meta in info["outputs"].items()]
    return "\n".join(lines) + "\n"


def smoke_test(out_dir, splits=SPLITS, clip=None):
    """Read the manifest of each of `splits` with the EchoJEPA parser, then load clips the
    way pretraining samples them (`clip`: frames_per_clip at clip.fps; frame_step 4
    without it). Besides each manifest's first video, one video of each case the loader
    samples differently is loaded, taken from videos.csv: a frame rate below clip.fps, no
    usable frame rate, and a video shorter than one clip. Raise a clear error if a
    manifest is empty or cannot be ingested, or if any of these videos cannot be loaded.
    """
    sys.path.insert(0, REPO_ROOT)
    videos = pd.read_csv(os.path.join(out_dir, "videos.csv"))
    for s in splits:
        smoke_test_manifest(os.path.join(out_dir, f"{s}.csv"), videos[videos.split == s], clip)


def smoke_test_manifest(path, videos, clip=None):
    """The smoke test of one manifest: `videos` are its rows of the video index (with `video_path`,
    `fps` and `needs_padding`), from which the special cases are drawn when `clip` is set.
    """
    from src.datasets.video_dataset import VideoDataset

    name = os.path.basename(path)
    if clip:
        sampling = {"frames_per_clip": clip["frames_per_clip"], "fps": clip["fps"], "frame_step": None}
        how = f"{clip['frames_per_clip']} frames at {clip['fps']} fps"
    else:
        sampling, how = {"frame_step": 4}, "frame step 4"
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"{name} is empty, so nothing shows the EchoJEPA parser can read it.")
    expected = pd.read_csv(path, sep=" ", header=None)[0].tolist()
    ds = VideoDataset(data_paths=[path], **sampling)
    if list(ds.samples) != expected:
        raise RuntimeError(f"{name}: the EchoJEPA parser read different paths than were written.")
    if any(int(x) != 0 for x in ds.labels):
        raise RuntimeError(f"{name}: the EchoJEPA parser read labels other than 0.")

    cases = {"the first video": expected[0]}
    if clip:
        v = videos
        for case, rows in ((f"a frame rate below {clip['fps']} fps", v[np.ceil(v.fps) < clip["fps"]]),
                           ("no usable frame rate", v[v.fps.isna()]),
                           ("fewer frames than one clip", v[v.needs_padding.fillna(False).astype(bool)])):
            if len(rows):
                cases[f"{case} ({len(rows)} video{'s' * (len(rows) > 1)})"] = rows.video_path.iloc[0]
    for case, video in cases.items():
        # `get_item_video`, not `ds[i]`: indexing retries another random video on a
        # failure, which would hide a video the loader cannot sample.
        try:
            loaded = ds.get_item_video(expected.index(video))
        except Exception as e:
            raise RuntimeError(f"{name}: the EchoJEPA loader failed on {case}, sampling {how}: "
                               f"{video}: {e!r}") from e
        if not loaded:
            raise RuntimeError(f"{name}: the EchoJEPA loader could not load {case}, sampling {how}: {video}.")
    print(f"[smoke]  {name}: {len(ds)} rows parsed; loaded with {how}: " + ", ".join(cases) + ".")


# --------------------------------------------------------------------------- #
#  Archive mode: every eligible study, anchored patient split
# --------------------------------------------------------------------------- #

ORIGINS = ("legacy", "new")

# Spawn key of the random stream that orders the patients new to the anchor. `stage_rngs` spawns the
# keys 0 .. len(STAGES) - 1 from the same seed; this key lies far beyond them, so the stream stays
# independent of the one-study stages and adding a stage there never changes an archive split.
NEW_PATIENT_STREAM = 1000

# Names the checks of `verify_video`. Bump it whenever they change: cached verifications made under
# another version are then ignored and redone.
VERIFY_VERSION = "verify_video/1"

# Videos verified between two writes to the verification cache: an interruption loses at most these.
VERIFY_CHUNK = 20_000

# Threads that stat the videos (size, modification time) before the cache is consulted.
STAT_THREADS = 32

# Seconds one video may take in a verification worker before that worker is killed and the video tried
# again. Most videos take well under a second and long videos a few seconds; 300 s also
# outlasts one NFS major timeout of the video mount (timeo 60 s, 2 retransmissions: ~3 min), so a read that
# recovers from a server hiccup finishes instead of being killed. A real stall holds one worker for 5 min.
VERIFY_TIMEOUT = 300.0

# Further attempts within one run for a video whose worker timed out or died, or whose decode failed for one
# of VERIFY_RETRY_REASONS. Failures are rare, so this costs little.
VERIFY_RETRIES = 2

# Decode failures a storage read error can cause with the file's size and mtime unchanged: retried within the
# run before the verdict is cached, and verified again from the cache with `--retry-failed`. Not `missing`: it
# is cached without a fingerprint and verified again once the file can be stat'ed.
VERIFY_RETRY_REASONS = ("empty", "undecodable", "no_frames", "corrupt_frames")

# Seconds before a retry, times the attempts made so far; other videos proceed meanwhile.
VERIFY_RETRY_DELAY = 5.0

# Reasons of the videos still without a result after every attempt, when `--accept-verify-failures` admits
# them (by the last attempt: deadline overrun, worker death, or an exception). Never cached.
VERIFY_UNRESOLVED_REASONS = {"timeout": "verify_timeout", "crash": "verify_crash", "error": "verify_error"}
# `--exclude-videos`: a listed video's reason is EXCLUDED_PREFIX + the list's label (e.g. excluded:deid_black).
EXCLUDED_PREFIX = "excluded:"
EXCLUDE_REASON = re.compile(r"[a-z0-9_]+")

CACHE_COLUMNS = ("video_path", "size", "mtime_ns", "verify_version", "reason", "n_frames", "fps")

# Acquisition index contract (data/scan_acquisition.py): one row per DICOM, keyed by `dcm_path`.
ACQ_CATEGORIES = ("bmode", "color_flow", "spectral_doppler", "mmode", "other", "unknown")
ACQ_CONFIDENCES = ("header", "unknown")
ACQ_FLAGS = ("has_tissue_2d", "has_color_flow", "has_pw_doppler", "has_cw_doppler", "has_mmode")
ACQ_COLUMNS = ("dcm_path", "acq_category", "acq_confidence", "acq_rule", "read_status", "scanner_version") + ACQ_FLAGS

# The loader manifests of an archive build and the split each one draws from.
ARCHIVE_MANIFESTS = {"train_inclusive": "train", "train_bmode": "train", "val": "val", "test": "test"}

# Quantiles reported for the number of videos per patient and per study.
QUANTILES = {"min": 0.0, "p10": 0.1, "p25": 0.25, "median": 0.5, "p75": 0.75, "p90": 0.9, "p99": 0.99, "max": 1.0}

_SHARD_SEQ = iter(range(10**9))


def read_anchor(path):
    """An anchor CSV with every column read as text: leading zeros survive, and only an empty
    field counts as missing (a patient id such as "NA" stays an id)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def parse_anchor(table):
    """The patient -> split assignments an archive build keeps: one row per patient with
    `patient_id`, `split` and `origin`.

    `table` is a one-study build's `studies.csv` (its patients get origin "legacy") or an earlier
    archive build's `patient_splits.csv` (origins carried over). A patient listed with two splits or
    two origins is ambiguous and stops the build.
    """
    if "origin" in table.columns:
        check_columns(table.columns, ["patient_id", "split", "origin"], "the anchor (patient_splits.csv)")
        origin = table.origin
    else:
        check_columns(table.columns, ["patient_id", "study_id", "split"], "the anchor (studies.csv)")
        origin = pd.Series("legacy", index=table.index)
    a = pd.DataFrame({"patient_id": normalize_id(table.patient_id),
                      "split": table.split.astype("string").str.strip(),
                      "origin": origin.astype("string").str.strip()})
    missing = a.patient_id.fillna("").eq("")
    if missing.any():
        raise ValueError(f"{missing.sum()} anchor rows have no patient_id.")
    for col, allowed in (("split", SPLITS), ("origin", ORIGINS)):
        bad = ~a[col].isin(allowed)
        if bad.any():
            raise ValueError(f"{bad.sum()} anchor rows have a {col} outside {list(allowed)}.")
    a = a.drop_duplicates()
    for col in ("split", "origin"):
        n = a.groupby("patient_id")[col].nunique()
        if (n > 1).any():
            raise ValueError(f"{(n > 1).sum()} anchor patients have more than one {col}.")
    return a.drop_duplicates("patient_id").sort_values("patient_id", ignore_index=True)


def assign_splits(patients, anchor, fracs, seed):
    """Give every eligible patient a split, keeping the split of every anchored patient.

    The targets are exact patient counts over all eligible patients (`split_sizes`). Anchored patients
    present in the archive fill part of each target; the patients new to the anchor fill the rest,
    in a seeded permutation of their sorted ids. That order depends only on the seed and the set of
    new patients, so metadata row order changes nothing, and a later build anchored on this build's
    `patient_splits.csv` keeps every assignment made here.

    Returns `patient_splits` (patient_id, split, origin, in_archive) and counts. It lists every
    eligible patient plus the anchored patients absent from this archive (`in_archive` false), so a
    build anchored on it never reassigns a patient who returns in a later export.
    """
    eligible = np.array(sorted(set(patients)), dtype=object)
    anchored = anchor.patient_id.isin(set(eligible)).to_numpy()
    kept, absent = anchor[anchored], anchor[~anchored]
    targets = dict(zip(SPLITS, (int(n) for n in split_sizes(len(eligible), fracs))))
    remaining = {s: targets[s] - int((kept.split == s).sum()) for s in SPLITS}
    over = {s: -n for s, n in remaining.items() if n < 0}
    if over:
        raise ValueError(f"The anchored patients exceed the target of split {list(over)} by {list(over.values())} "
                         f"patients ({len(eligible)} eligible patients at {fracs}); the anchor does not fit `split`.")
    new = eligible[~pd.Index(eligible).isin(list(anchor.patient_id))]
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(NEW_PATIENT_STREAM,)))
    assigned = pd.DataFrame({"patient_id": rng.permutation(new),
                             "split": np.repeat(np.array(SPLITS, dtype=object), [remaining[s] for s in SPLITS]),
                             "origin": "new"})
    splits = pd.concat([kept.assign(in_archive=True), assigned.assign(in_archive=True),
                        absent.assign(in_archive=False)], ignore_index=True)
    splits = (splits.astype({"patient_id": "string", "split": "string", "origin": "string", "in_archive": bool})
              .sort_values("patient_id", ignore_index=True))

    def by_split(df):
        return {s: int((df.split == s).sum()) for s in SPLITS}

    stats = {
        "eligible_patients": int(len(eligible)),
        "targets": targets,
        "anchor_patients": int(len(anchor)),
        "anchored_present": {o: by_split(kept[kept.origin == o]) for o in ORIGINS},
        "anchored_absent": {o: by_split(absent[absent.origin == o]) for o in ORIGINS},
        "new_patients": by_split(assigned),
        "patients": by_split(splits[splits.in_archive]),
    }
    return splits, stats


def print_assignment(stats):
    present = sum(sum(d.values()) for d in stats["anchored_present"].values())
    absent = sum(sum(d.values()) for d in stats["anchored_absent"].values())
    print(f"\n[split]  {stats['eligible_patients']} eligible patients. The anchor lists {stats['anchor_patients']}: "
          f"{present} in this archive keep their split; {absent} without an eligible study are counted and kept.")
    table = pd.DataFrame({
        "target": stats["targets"],
        "anchored legacy": stats["anchored_present"]["legacy"],
        "anchored earlier build": stats["anchored_present"]["new"],
        "new": stats["new_patients"],
        "patients": stats["patients"],
        "anchored absent": {s: sum(d[s] for d in stats["anchored_absent"].values()) for s in SPLITS},
        **{k.replace("_", " "): stats[k] for k in ("studies", "metadata_videos", "labelled_studies") if k in stats},
    }).rename_axis("split")
    print(table.to_string())
    if stats["anchor_patients"] and absent > 0.01 * stats["anchor_patients"]:
        print(f"[warn]   {absent / stats['anchor_patients']:.1%} of the anchored patients have no eligible study; "
              "check that patient ids are formatted alike in the anchor and the metadata.")


def verify_version():
    """The cache key of a verification: `VERIFY_VERSION` and the decoder's version, since a decoder
    upgrade can change which videos decode."""
    try:
        import decord

        decoder = decord.__version__
    except Exception:
        decoder = "unavailable"
    return f"{VERIFY_VERSION} decord=={decoder}"


def supervision():
    """data/supervised_pool.py (workers under a deadline), however this module was loaded."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.append(here)
    import supervised_pool

    return supervised_pool


def stat_paths(paths, threads=STAT_THREADS, block=64, timeout=VERIFY_TIMEOUT):
    """Size and modification time (ns) of each path; -1 for both when it cannot be stat'ed. Threads
    overlap the latency of a network file system (`os.stat` releases the GIL). Raises PoolBroken when
    no block of `block` paths is done for `timeout` seconds, instead of waiting on a stalled mount
    forever; blocks stay small so a slow but working mount never trips it."""
    n = len(paths)
    size, mtime = np.full(n, -1, np.int64), np.full(n, -1, np.int64)

    def stat_block(start):
        for i in range(start, min(start + block, n)):
            try:
                st = os.stat(paths[i])
            except (OSError, ValueError):
                continue
            size[i], mtime[i] = st.st_size, st.st_mtime_ns

    supervision().thread_map(stat_block, range(0, n, block), max(1, threads), timeout)
    return size, mtime


def read_verify_cache(cache_dir):
    """Every cached verification, the latest one per video path. Shard names sort in the order the
    shards were written; the temporary file of an interrupted write is never read."""
    import pyarrow.parquet as pq

    parts = []
    for shard in sorted(glob.glob(os.path.join(cache_dir, "verify-*.parquet"))):
        table = pq.read_table(shard)
        check_columns(table.schema.names, CACHE_COLUMNS, f"verification cache shard {shard}")
        parts.append(table.select(list(CACHE_COLUMNS)).to_pandas())
    if not parts:
        return pd.DataFrame({"video_path": pd.Series(dtype=object), "size": pd.Series(dtype=np.int64),
                             "mtime_ns": pd.Series(dtype=np.int64), "verify_version": pd.Series(dtype=object),
                             "reason": pd.Series(dtype=object), "n_frames": pd.Series(dtype=float),
                             "fps": pd.Series(dtype=float)})
    return pd.concat(parts, ignore_index=True).drop_duplicates("video_path", keep="last")


def write_cache_shard(cache_dir, rows):
    """Write one shard of verifications atomically, so a reader sees all of it or none of it."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("video_path", pa.string()), ("size", pa.int64()), ("mtime_ns", pa.int64()),
                        ("verify_version", pa.string()), ("reason", pa.string()), ("n_frames", pa.int64()),
                        ("fps", pa.float64())])
    table = pa.Table.from_pandas(rows[list(CACHE_COLUMNS)], schema=schema, preserve_index=False)
    name = f"verify-{time.time_ns():020d}-{os.getpid()}-{next(_SHARD_SEQ):06d}.parquet"
    tmp = os.path.join(cache_dir, f".{name}.tmp")
    pq.write_table(table, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(cache_dir, name))


def verification_rows(paths, checks, version, fingerprints):
    """Cache rows of verified videos with the (size, mtime_ns) each file had when it was verified. A
    video found missing is stored with no fingerprint (-1, -1): it is reused only while the file still
    cannot be stat'ed, and verified again once it can."""
    reason = np.array([c.reason for c in checks], dtype=object)
    missing = reason == "missing"
    size = np.array([f[0] for f in fingerprints], dtype=np.int64)
    mtime = np.array([f[1] for f in fingerprints], dtype=np.int64)
    return pd.DataFrame({
        "video_path": np.asarray(paths, dtype=object),
        "size": np.where(missing, -1, size),
        "mtime_ns": np.where(missing, -1, mtime),
        "verify_version": version,
        "reason": reason,
        "n_frames": pd.array([c.n_frames for c in checks], dtype="Int64"),
        "fps": np.array([c.fps for c in checks], dtype=float),
    })


def fingerprint_and_validate(path, validate=verify_video):
    """Stat a pending video right before verifying it, so its cached fingerprint is that of the file
    that was decoded. Pending videos are never stat'ed up front: on a network file system a cold stat
    of millions of paths takes hours, which a first or a bounded (`--verify-limit`) run should not pay."""
    try:
        st = os.stat(path)
        fingerprint = (st.st_size, st.st_mtime_ns)
    except (OSError, TypeError, ValueError):
        fingerprint = (-1, -1)
    return fingerprint, validate(path)


def duration(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m{rest % 60:02d}s"


class VerificationIncomplete(RuntimeError):
    """Videos still without a result after every attempt (deadline overrun, worker death): the run stops
    with no verdict for them and nothing written downstream; its completed results stay cached."""


def write_unresolved(cache_dir, paths, failures):
    """List unresolved videos (path, failure kind, attempts) in `cache_dir` for inspection; the file is
    never read back. Returns its path."""
    path = os.path.join(cache_dir, f"unresolved-{time.time_ns():020d}.csv")
    tmp = os.path.join(cache_dir, f".{os.path.basename(path)}.tmp")
    pd.DataFrame({"video_path": paths, "failure": [f.kind for f in failures], "attempts": [f.attempts for f in failures],
                  "detail": [f.detail for f in failures]}).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def verify_archive(paths, validate=verify_video, cache_dir=None, workers=1, chunk=VERIFY_CHUNK, limit=None,
                   stat_threads=STAT_THREADS, timeout=VERIFY_TIMEOUT, retries=VERIFY_RETRIES, retry_failed=False,
                   accept_failures=0):
    """Verify each of `paths` (unique video paths) with `validate`, reusing a cached verification
    only when the file's size and modification time and `verify_version()` all match it.

    Pending videos are verified in a fixed shuffled order, so a run cut short by `limit` measures a
    representative sample (the order changes no result). Every `chunk` results are written to
    `cache_dir` at once, so an interrupted run resumes after its last complete write. With `limit`,
    at most that many pending videos are verified.

    With `workers` > 1 each video is verified in a worker process under a deadline of `timeout` seconds
    (`supervised_pool.SupervisedPool`); a worker that overruns it or dies is killed and replaced. One
    worker verifies in this process, without a deadline. A video whose worker timed out or died, or whose
    decode failed for one of VERIFY_RETRY_REASONS, is tried up to `retries` more times in this run before a
    failure verdict is cached. A video still without a result is unresolved and never cached: after every
    completed result is cached, VerificationIncomplete is raised, unless there are at most `accept_failures`
    of them, which are then excluded for this run only (VERIFY_UNRESOLVED_REASONS). `retry_failed` verifies
    the cached VERIFY_RETRY_REASONS failures again, still reusing the cached successes.

    Returns the verified videos (DataFrame indexed by `video_path`: reason, n_frames, fps) and
    counts; `stats["pending"]` videos are left without a result.
    """
    sp = supervision()
    version = verify_version()
    paths = np.asarray(paths, dtype=object)
    want = pd.DataFrame({"video_path": paths})
    reused, stale, again, stat_seconds = want.iloc[:0].assign(reason=None, n_frames=None, fps=None), 0, 0, 0.0
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        # Only cached videos are stat'ed here: their fingerprint must still match to be reused.
        hit = want.merge(read_verify_cache(cache_dir), on="video_path")
        start = time.monotonic()
        size, mtime = stat_paths(hit.video_path.to_numpy(dtype=object), stat_threads, timeout=timeout)
        stat_seconds = time.monotonic() - start
        same = (size == hit["size"].to_numpy()) & (mtime == hit["mtime_ns"].to_numpy()) & (
            hit["verify_version"] == version).to_numpy()
        redo = same & hit.reason.isin(VERIFY_RETRY_REASONS).to_numpy() if retry_failed else np.zeros(len(hit), bool)
        reused, stale, again = hit[same & ~redo], int((~same).sum()), int(redo.sum())
    todo = want[~want.video_path.isin(reused.video_path)]
    todo = todo.iloc[np.random.default_rng(0).permutation(len(todo))]
    if limit is not None:
        todo = todo.iloc[:limit]
    print(f"\n[verify] {len(paths)} videos to verify: {len(reused)} reused from the cache "
          f"({stale} cached entries out of date" + (f", {again} cached failures to verify again" if retry_failed else "")
          + f"), {len(want) - len(reused)} pending; checking the cached fingerprints took {duration(stat_seconds)}.",
          flush=True)

    items = todo.video_path.tolist()
    task = functools.partial(fingerprint_and_validate, validate=validate)

    def decode_failed(out):
        return out[1].reason in VERIFY_RETRY_REASONS

    def log(line):
        print(f"[verify] {line}", flush=True)

    pool = (sp.SupervisedPool(task, workers, timeout, retries, decode_failed, VERIFY_RETRY_DELAY, log=log)
            if workers > 1 else sp.InlinePool(task, retries, decode_failed))
    done, results, buffer, unresolved = 0, [], [], []
    start = time.monotonic()

    def write(buffer):
        nonlocal done
        buffer.sort(key=lambda r: r[0])
        rows = verification_rows([items[i] for i, _ in buffer], [c for _, (_, c) in buffer], version,
                                 [f for _, (f, _) in buffer])
        if cache_dir is not None:
            write_cache_shard(cache_dir, rows)
        results.append(rows)
        done += len(buffer)
        rate = done / max(time.monotonic() - start, 1e-9)
        print(f"[verify] {done}/{len(todo)} verified this run, {rate:.1f} videos/s, about "
              f"{duration((len(todo) - done - len(unresolved)) / rate)} left"
              + (f"; {len(unresolved)} unresolved so far." if unresolved else "."), flush=True)

    with pool:
        try:
            for i, out in pool.imap_unordered(items):
                if isinstance(out, sp.Failure):
                    unresolved.append((i, out))
                    continue
                buffer.append((i, out))
                if len(buffer) >= chunk:
                    write(buffer)
                    buffer = []
        except sp.PoolBroken:
            if buffer:  # keep what is done; the message says why the run stopped
                write(buffer)
            raise
        if buffer:
            write(buffer)
    seconds = time.monotonic() - start

    keep = ["video_path", "reason", "n_frames", "fps"]
    parts, accepted = [reused[keep]] + [r[keep] for r in results], {}
    if unresolved:
        unresolved.sort(key=lambda r: r[0])
        lost, failures = [items[i] for i, _ in unresolved], [f for _, f in unresolved]
        kinds = {k: sum(f.kind == k for f in failures) for k in sp.KINDS if any(f.kind == k for f in failures)}
        listing = write_unresolved(cache_dir, lost, failures) if cache_dir is not None else None
        if len(unresolved) > accept_failures:
            raise VerificationIncomplete(
                f"{len(unresolved)} videos are unresolved after {retries + 1} attempts each {kinds} (deadline "
                f"{timeout:g} s per attempt)" + (f", more than --accept-verify-failures {accept_failures}" if
                                                  accept_failures else "")
                + f". The {done} videos verified in this run are " + ("cached" if cache_dir else "not kept (no cache)")
                + "; nothing else is written. Rerun the same command to try them again"
                + (f" (they are listed in {listing})" if listing else "") + ". If the same videos fail every time, "
                "inspect them and rerun with --accept-verify-failures N (N at least their number) to exclude them as "
                + " / ".join(VERIFY_UNRESOLVED_REASONS.values()) + " in that build; that is never cached.")
        accepted = {VERIFY_UNRESOLVED_REASONS[k]: n for k, n in kinds.items()}
        parts.append(pd.DataFrame({"video_path": lost, "reason": [VERIFY_UNRESOLVED_REASONS[f.kind] for f in failures],
                                   "n_frames": pd.array([pd.NA] * len(lost), dtype="Int64"), "fps": np.nan}))
        print(f"[verify] --accept-verify-failures {accept_failures}: {len(lost)} unresolved videos are excluded "
              f"in this build as {accepted}; they are not cached, so a later run tries them again.", flush=True)

    verdicts = pd.concat(parts, ignore_index=True).set_index("video_path")
    verdicts = verdicts.astype({"reason": "string", "n_frames": "Int64", "fps": float})
    stats = {
        "verify_version": version,
        "videos_to_verify": int(len(paths)),
        "reused_from_cache": int(len(reused)),
        "stale_cache_entries": stale,
        "retry_failed": bool(retry_failed),
        "cached_failures_verified_again": again,
        "verified_this_run": int(done),
        "pending": int(len(paths) - len(verdicts)),
        "workers": int(workers),
        "verify_timeout_s": float(timeout) if workers > 1 else None,
        "verify_retries": int(retries),
        "failed_attempts": {k: int(pool.counts[k]) for k in sp.KINDS},
        "decode_failures_retried": int(pool.counts["retried_results"]),
        "recovered_on_retry": int(pool.counts["recovered"]),
        "accepted_unresolved": accepted,
        "stat_seconds": round(stat_seconds, 1),
        "verify_seconds": round(seconds, 1),
        "videos_per_second": round(done / seconds, 2) if done and seconds > 0 else None,
        "cache": cache_dir is not None,
    }
    return verdicts, stats


def print_verify_limit(stats, limit):
    rate, left = stats["videos_per_second"], stats["pending"]
    eta = f", about {duration(left / rate)} at this rate" if rate and left else ""
    kept = "results are cached" if stats["cache"] else "no --verify-cache, so the results are not kept"
    print(f"\n[verify] --verify-limit {limit}: verified {stats['verified_this_run']} videos with {stats['workers']} "
          f"workers ({rate or 0:.2f} videos/s); {left} still pending{eta}. Stopped here: {kept}, no manifests "
          "written. Run again without --verify-limit to finish and write the manifests.")


def read_acquisition(source, dicom_paths):
    """The acquisition index rows of `dicom_paths`, from a parquet file, a directory of parquet
    shards, or a DataFrame.

    Stops on a row without `dcm_path`, a DICOM path listed twice (the join would be ambiguous), or an
    `acq_category` / `acq_confidence` outside the contract. Returns the rows renamed for the video
    index (`dicom_path`, `acq_read_status`) and the number of rows in the whole index.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(source, pd.DataFrame):
        check_columns(source.columns, ACQ_COLUMNS, "the acquisition index")
        table = pa.Table.from_pandas(source[list(ACQ_COLUMNS)], preserve_index=False)
    else:
        import pyarrow.dataset as ds

        dataset = ds.dataset(source, format="parquet")
        check_columns(dataset.schema.names, ACQ_COLUMNS, f"the acquisition index {source}")
        table = dataset.to_table(columns=list(ACQ_COLUMNS))
    key = table["dcm_path"].cast(pa.large_string())
    if key.null_count:
        raise ValueError(f"{key.null_count} acquisition index rows have no dcm_path.")
    repeated = len(key) - pc.count_distinct(key).as_py()
    if repeated:
        raise ValueError(f"{repeated} acquisition index rows repeat a dcm_path listed before; joining on it "
                         "would be ambiguous.")
    for col, allowed in (("acq_category", ACQ_CATEGORIES), ("acq_confidence", ACQ_CONFIDENCES)):
        bad = set(pc.unique(table[col]).to_pylist()) - set(allowed)
        if bad:
            raise ValueError(f"the acquisition index has {col} values outside {list(allowed)}: "
                             f"{sorted(map(str, bad))}.")
    wanted = pd.Series(dicom_paths).dropna().unique()
    table = table.filter(pc.is_in(key, value_set=pa.array(np.asarray(wanted, dtype=object), pa.large_string())))
    acq = table.to_pandas().rename(columns={"dcm_path": "dicom_path", "read_status": "acq_read_status",
                                            "scanner_version": "acq_scanner_version"})
    acq = acq.astype({c: "string" for c in ("dicom_path", "acq_category", "acq_confidence", "acq_rule",
                                            "acq_read_status", "acq_scanner_version")})
    return acq.astype({f: "boolean" for f in ACQ_FLAGS}), int(len(key))


def confirmed_bmode(videos):
    """Whether the DICOM header classifies each video as B-mode, from a vendor whose headers were validated
    against the decoded videos (`acq_vendor_validated`): the only videos the B-mode arm admits."""
    ok = videos.acq_category.isin(["bmode"]) & videos.acq_confidence.isin(["header"])
    if "acq_vendor_validated" in videos:
        ok &= videos.acq_vendor_validated.fillna(False).astype(bool)
    return ok.to_numpy()


def effective_acq_category(videos):
    """The acquisition category every user of the index acts on (B-mode arm, probe view slots, RADAR captions):
    `acq_category`, except that a B-mode the header does not confirm from a validated vendor (`confirmed_bmode`)
    is unresolved, `unknown`, until it is verified. A video missing from the index is `unknown` too."""
    category = videos.acq_category.astype("string").fillna("unknown").to_numpy(dtype=object)
    unresolved = (category == "bmode") & ~confirmed_bmode(videos)
    return pd.Series(np.where(unresolved, "unknown", category), index=videos.index, dtype="string")


def current_scanner_version():
    """`SCANNER_VERSION` of data/scan_acquisition.py, the version a usable index row must come from."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.append(here)
    from scan_acquisition import SCANNER_VERSION

    return SCANNER_VERSION


def acquisition_quality(rows, acq, splits, scanner_version):
    """Per split in `splits`, the eligible videos whose index row comes from another scanner version (stale)
    or records a failed header read (`read_status` not ok), and the scanner versions of those rows. A
    successful read with an unknown mode is neither: it is a result."""
    a = rows[["dicom_path", "split"]].merge(acq[["dicom_path", "acq_scanner_version", "acq_read_status"]],
                                            on="dicom_path", how="inner")
    a = a[a.split.isin(splits).to_numpy()]
    stale = (a.acq_scanner_version.astype("string") != str(scanner_version)).fillna(True).to_numpy(bool)
    errors = ~stale & (a.acq_read_status.astype("string") != "ok").fillna(True).to_numpy(bool)
    in_split = {s: a.split.eq(s).to_numpy() for s in splits}
    return {"stale_scanner_version": {s: int((stale & m).sum()) for s, m in in_split.items()},
            "read_errors": {s: int((errors & m).sum()) for s, m in in_split.items()},
            "scanner_versions": {str(k): int(n) for k, n in a.acq_scanner_version.value_counts(dropna=False).items()}}


def acquisition_coverage(rows, acq, splits):
    """Per split in `splits`, the eligible videos whose DICOM the acquisition index lacks (never scanned)."""
    listed = rows.dicom_path.isin(set(acq.dicom_path)).to_numpy() if "dicom_path" in rows else np.zeros(len(rows), bool)
    return {s: int((~listed & rows.split.isin([s]).to_numpy()).sum()) for s in splits}


def video_index(rows, verdicts, verified, acq, clip=None, bmode_vendors=None):
    """The master video index: every eligible video with its patient's split, its verification, its
    acquisition mode, and whether it enters each training arm.

    `verified` marks the rows whose split was verified (every one has a verdict). Any other video is
    `not_verified`, and a video without a path is `no_path`. A video absent from the acquisition index
    (or with no index at all) is `unknown` and `acq_scanned` false. `train_inclusive` takes every valid
    train video; `train_bmode` only those whose header confirms B-mode.

    `bmode_vendors` (config `acquisition.bmode_vendors`) lists the vendors whose region headers were
    validated against decoded videos: other vendors' headers can declare a single tissue region over a
    colour-flow acquisition, so their "bmode" is unresolved (`acq_vendor_validated` false). `None`
    validates every vendor.
    """
    v = rows.merge(verdicts.reset_index(), on="video_path", how="left", validate="many_to_one")
    has_path = v.video_path.notna().to_numpy()
    reason = np.where(verified, v.reason.to_numpy(dtype=object, na_value=None),
                      np.where(has_path, "not_verified", "no_path"))
    v = v.assign(reason=pd.array(reason, dtype="string"), n_frames=v.n_frames.astype("Int64").where(verified),
                 fps=v.fps.astype(float).where(verified))
    if clip:
        v = v.assign(needs_padding=needs_padding(v.n_frames, v.fps, clip))

    if acq is None:
        acq = pd.DataFrame({"dicom_path": pd.Series(dtype="string"),
                            **{c: pd.Series(dtype="string") for c in ("acq_category", "acq_confidence", "acq_rule",
                                                                       "acq_read_status")},
                            **{f: pd.Series(dtype="boolean") for f in ACQ_FLAGS}})
    if "dicom_path" in v:
        v = v.merge(acq, on="dicom_path", how="left", validate="many_to_one")
    else:
        v = v.assign(**{c: pd.Series(pd.NA, index=v.index, dtype=acq[c].dtype) for c in acq if c != "dicom_path"})
    scanned = v.acq_category.notna().to_numpy()
    v = v.assign(acq_scanned=scanned, acq_category=v.acq_category.fillna("unknown"),
                 acq_confidence=v.acq_confidence.fillna("unknown"), acq_rule=v.acq_rule.where(scanned, "not_scanned"))
    if bmode_vendors is None:
        validated = pd.Series(True, index=v.index, dtype="boolean")
    else:
        if "vendor" not in v:
            raise ValueError("acquisition.bmode_vendors needs the vendor column (columns.vendor).")
        validated = v.vendor.isin(list(bmode_vendors)).astype("boolean")
    v = v.assign(acq_vendor_validated=validated)
    train_ok = v.reason.isna().to_numpy() & v.split.isin(["train"]).to_numpy()
    v = v.assign(in_inclusive=train_ok, in_bmode=train_ok & confirmed_bmode(v))

    order = ["patient_id", "study_id", "study_date", "split", "origin", "exam_type", "view", "video_path",
             "dicom_path", "vendor", "model", "reason", "n_frames", "fps", "needs_padding", "acq_scanned",
             "acq_category", "acq_confidence", "acq_rule", "acq_read_status", *ACQ_FLAGS, "acq_vendor_validated",
             "in_inclusive", "in_bmode"]
    keys = ["split", "patient_id", "study_id", "video_path", "dicom_path", "view", "vendor", "model"]
    v = v[[c for c in order if c in v]]
    return v.sort_values([c for c in keys if c in v], na_position="last", kind="stable", ignore_index=True)


def archive_studies(studies, index):
    """Add each study's video counts after verification: valid (`n_ok`), valid and header-confirmed
    B-mode (`n_bmode`, in every split), failed and not verified. A study with none left keeps its row."""
    ok = index.reason.isna().to_numpy()
    unverified = index.reason.isin(["not_verified"]).to_numpy()
    flags = pd.DataFrame({"study_id": index.study_id, "n_ok": ok, "n_bmode": ok & confirmed_bmode(index),
                          "n_failed": ~ok & ~unverified, "n_not_verified": unverified})
    per_study = flags.groupby("study_id").sum().astype(int).reset_index()
    return studies.merge(per_study, on="study_id", how="left", validate="one_to_one")


def archive_exclusions(index, with_bmode):
    """Every eligible video left out of a manifest, with the arm and the reason: the verification
    failure, `not_verified`, or for the B-mode arm `acq:<label>` (`acquisition_label`: `acq:not_scanned` for a
    video absent from the index, `acq:bmode_unconfirmed` for a B-mode category without header confidence,
    `acq:bmode_vendor_unvalidated` for a header B-mode from a vendor outside `acquisition.bmode_vendors`)."""
    columns = ["split", "patient_id", "study_id", "view", "video_path", "arm", "reason"]
    parts = [index[index.reason.notna().to_numpy()].assign(arm="inclusive")]
    if with_bmode:
        train = index[index.split.isin(["train"]).to_numpy() & ~index.in_bmode.to_numpy()]
        parts.append(train.assign(arm="bmode", reason=train.reason.fillna("acq:" + acquisition_label(train))))
    excluded = pd.concat(parts, ignore_index=True)[columns]
    return excluded.sort_values(["split", "arm", "video_path", "study_id", "view"], na_position="last",
                                kind="stable", ignore_index=True)


def check_archive(table, patient_splits, anchor):
    """Raise AssertionError unless an archive build is consistent. `table` is the video index, or the
    study table of an `--assign-only` build (the video-level checks are then skipped).

    1. Each patient has one split, every row is in its patient's split, no patient spans two splits.
    2. Each study belongs to one patient and all its rows share one split.
    3. `patient_splits` marks exactly the eligible patients `in_archive`, and every anchored patient
       keeps its anchor split and origin.
    4. No anchored val/test patient, and no val/test video, is in a training manifest.
    5. train_inclusive holds only valid train videos; train_bmode is a subset of it with header-
       confirmed B-mode only.
    6. No video or DICOM path appears twice; no manifest path contains whitespace.
    """
    if patient_splits.patient_id.duplicated().any():
        raise AssertionError("A patient has more than one split in patient_splits.")
    mapped = table.patient_id.map(patient_splits.set_index("patient_id").split)
    if mapped.isna().any() or (mapped.to_numpy(dtype=object) != table.split.to_numpy(dtype=object)).any():
        raise AssertionError("A study or video is not in its patient's split.")
    if (table.groupby("patient_id").split.nunique() > 1).any():
        raise AssertionError("Patients shared between splits.")
    per_study = table.groupby("study_id").agg(patients=("patient_id", "nunique"), splits=("split", "nunique"))
    if (per_study.patients > 1).any():
        raise AssertionError("A study belongs to more than one patient.")
    if (per_study.splits > 1).any():
        raise AssertionError("The videos of a study are in more than one split.")
    if set(patient_splits.patient_id[patient_splits.in_archive.astype(bool)]) != set(table.patient_id):
        raise AssertionError("patient_splits does not mark exactly the eligible patients as in_archive.")
    kept = anchor.merge(patient_splits, on="patient_id", how="left", suffixes=("_anchor", ""))
    for col in ("split", "origin"):
        anchored = kept[f"{col}_anchor"].to_numpy(dtype=object)
        if kept[col].isna().any() or (kept[col].to_numpy(dtype=object) != anchored).any():
            raise AssertionError(f"An anchored patient lost its anchor {col}.")
    if "video_path" not in table:
        return

    in_train_arm = table.in_inclusive.to_numpy(bool) | table.in_bmode.to_numpy(bool)
    held_out = set(anchor.patient_id[anchor.split.isin(["val", "test"])])
    if (in_train_arm & table.patient_id.isin(held_out).to_numpy()).any():
        raise AssertionError("An anchored val/test patient has a video in a training manifest.")
    if (in_train_arm & ~table.split.isin(["train"]).to_numpy()).any():
        raise AssertionError("A val/test video is in a training manifest.")
    ok = table.reason.isna().to_numpy()
    if (table.in_inclusive.to_numpy(bool) & ~ok).any():
        raise AssertionError("A video that failed verification is in train_inclusive.")
    if (table.in_bmode.to_numpy(bool) & ~table.in_inclusive.to_numpy(bool)).any():
        raise AssertionError("train_bmode is not a subset of train_inclusive.")
    if (table.in_bmode.to_numpy(bool) & ~confirmed_bmode(table)).any():
        raise AssertionError("A video whose header does not confirm B-mode is in train_bmode.")
    for col in ("video_path", "dicom_path"):
        if col in table and table[col].dropna().duplicated().any():
            raise AssertionError(f"A {col} appears under more than one video, study or patient.")
    if table.video_path[ok].astype(str).str.contains(r"\s", regex=True).any():
        raise AssertionError("A manifest path contains whitespace.")


def build_archive(videos, labels, cfg, anchor, validate=verify_video, workers=1, batch_size=BATCH_SIZE,
                  acquisition=None, verify_splits=SPLITS, cache_dir=None, verify_limit=None,
                  verify_chunk=VERIFY_CHUNK, stat_threads=STAT_THREADS, assign_only=False,
                  allow_partial_acquisition=False, accept_acquisition_read_errors=0, verify_timeout=VERIFY_TIMEOUT,
                  verify_retries=VERIFY_RETRIES, retry_failed=False, accept_verify_failures=0, exclude=None):
    """Build the archive corpus of `selection.mode: all_studies`: every eligible study of every patient,
    the anchored patient split and, unless `assign_only`, the verified master video index and its
    manifests.

    `anchor` is the table of an earlier build (`read_anchor`): a one-study `studies.csv` or an archive
    `patient_splits.csv`. Splits are assigned from metadata eligibility before any video is decoded and
    are never redrawn: a study or patient whose videos all fail keeps its split and is counted. Only the
    `verify_splits` are verified; with `verify_limit`, verification stops after that many pending videos
    and the result is marked `stopped` (nothing to write). `acquisition` is the acquisition index (path
    or DataFrame); without it no video is confirmed B-mode and no B-mode manifest is made. The index must
    list every eligible video of the verified splits, checked before verification starts: a partial index
    (a sample, an unfinished scan) would silently shrink the B-mode arm. Its rows must also come from the
    current scanner version and have read their header: at most `accept_acquisition_read_errors` failed reads
    are admitted, after a human reviewed them, and those videos stay out of the B-mode arm.
    `allow_partial_acquisition` admits any index for development builds. The counts record all of this, with
    `ready` true only for a build fit for the arms. Videos read with an unknown mode are results, not errors:
    they stay out of the B-mode arm. `verify_timeout`, `verify_retries`, `retry_failed` and
    `accept_verify_failures` go to `verify_archive`, which raises (so nothing is written) when videos stay
    unresolved. `exclude` ({"reason": label, "paths": video paths}, from `--exclude-videos`) leaves the listed
    videos of the verified splits out without decoding them, with reason EXCLUDED_PREFIX + label; like a failed
    video, such a video stays in the master index and in excluded_videos.csv, and its study keeps its split.
    """
    check_config(cfg)
    if cfg["selection"].get("mode") != ALL_STUDIES:
        raise ValueError(f"build_archive needs selection.mode: {ALL_STUDIES}.")
    cols = cfg["columns"]
    if acquisition is not None and "dicom_path" not in cols:
        raise ValueError("The acquisition index is joined on DICOM paths: map columns.dicom_path.")
    unknown = set(verify_splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"verify_splits {sorted(unknown)} are not splits {list(SPLITS)}.")
    verify_splits = [s for s in SPLITS if s in set(verify_splits)]
    fracs, seed = [cfg["split"][s] for s in SPLITS], cfg["selection"]["seed"]
    lab = load_labels(labels, cfg) if labels is not None else None

    # One pass over the metadata keeps the video rows; --assign-only only needs their count per study.
    keep = ["study_id"] if assign_only else [r for r in video_roles(cols) if r != "avi_status"]
    studies, counts, rows = scan_metadata(videos, cfg, lab, batch_size, keep=keep)
    if "dicom_path" in rows:
        repeated = rows.dicom_path.dropna().duplicated()
        if repeated.any():
            raise ValueError(f"{repeated.sum()} DICOM paths appear under more than one eligible video.")
    anchor = parse_anchor(anchor)
    patient_splits, assignment = assign_splits(studies.patient_id, anchor, fracs, seed)
    split_of = patient_splits.set_index("patient_id")
    studies = studies.assign(split=studies.patient_id.map(split_of.split),
                             origin=studies.patient_id.map(split_of.origin),
                             n_videos_metadata=studies.study_id.map(rows.study_id.value_counts()).astype(int))
    order = ["study_id", "patient_id", "split", "origin", "study_date", "exam_type", "n_videos_metadata"]
    studies = (studies[[c for c in order if c in studies] + (["label"] if lab is not None else [])]
               .sort_values(["split", "patient_id", "study_id"], ignore_index=True))

    def by_split(series):
        return {s: int(series.get(s, 0)) for s in SPLITS}

    assignment.update(studies=by_split(studies.split.value_counts()),
                      metadata_videos=by_split(studies.groupby("split").n_videos_metadata.sum()))
    if lab is not None:
        assignment["labelled_studies"] = by_split(studies.split[studies.label.notna()].value_counts())
    print_assignment(assignment)
    result = {"assign_only": assign_only, "stopped": False, "seed": seed, "studies": studies,
              "patient_splits": patient_splits,
              "anchor_absent": patient_splits.loc[~patient_splits.in_archive, ["patient_id", "split", "origin"]]
              .reset_index(drop=True),
              "counts": {**counts, "assignment": assignment}}
    if assign_only:
        check_archive(studies, patient_splits, anchor)
        return result

    rows = rows.assign(split=rows.patient_id.map(split_of.split), origin=rows.patient_id.map(split_of.origin))
    acq, index_rows, not_in_index, quality = None, 0, None, None
    if acquisition is not None and verify_limit is None:
        # Checked before the (long) verification.
        acq, index_rows = read_acquisition(acquisition, rows.dicom_path)
        not_in_index = acquisition_coverage(rows, acq, verify_splits)
        quality = acquisition_quality(rows, acq, verify_splits, current_scanner_version())
        stale, errors = sum(quality["stale_scanner_version"].values()), sum(quality["read_errors"].values())
        if not allow_partial_acquisition:
            if sum(not_in_index.values()):
                raise ValueError(
                    f"The acquisition index lacks {sum(not_in_index.values())} eligible videos (by split: "
                    f"{not_in_index}); a partial index (a sample or an unfinished scan) would silently shrink "
                    "train_bmode. Finish and consolidate the scan (data/scan_acquisition.py), or pass "
                    "--allow-partial-acquisition for a development build.")
            if stale:
                raise ValueError(
                    f"{stale} eligible videos have acquisition rows from another scanner version (versions: "
                    f"{quality['scanner_versions']}); rerun the scan, which rescans them, and consolidate.")
            if errors > accept_acquisition_read_errors:
                raise ValueError(
                    f"{errors} eligible videos have a failed header read (by split: {quality['read_errors']}). "
                    "Retry them (scan_acquisition.py scan ... --retry-errors) and consolidate; once the "
                    f"persistent ones are reviewed, --accept-acquisition-read-errors {errors} builds with them "
                    "left out of train_bmode.")
    verified = (rows.video_path.notna() & rows.split.isin(verify_splits)).to_numpy()
    listed = rows.video_path.isin(exclude["paths"]).to_numpy() if exclude else np.zeros(len(rows), bool)
    paths = rows.video_path[verified & ~listed].sort_values().to_numpy(dtype=object)
    verdicts, verification = verify_archive(paths, validate, cache_dir, workers, verify_chunk, verify_limit,
                                            stat_threads, verify_timeout, verify_retries, retry_failed,
                                            accept_verify_failures)
    verification["verify_splits"] = verify_splits
    result["counts"]["verification"] = verification
    if verify_limit is not None:
        print_verify_limit(verification, verify_limit)
        return {**result, "stopped": True}
    if verification["pending"]:
        raise RuntimeError(f"{verification['pending']} videos were left unverified.")
    if exclude:
        reason = EXCLUDED_PREFIX + exclude["reason"]
        skipped = pd.Index(rows.video_path[verified & listed].unique(), name="video_path")
        verdicts = pd.concat([verdicts, pd.DataFrame({"reason": reason, "n_frames": pd.NA, "fps": np.nan},
                                                     index=skipped)])
        verdicts = verdicts.astype({"reason": "string", "n_frames": "Int64", "fps": float})
        result["counts"]["exclusion_list"] = {
            "reason": reason, "listed": len(exclude["paths"]),
            "eligible": by_split(rows.split[listed].value_counts()),
            "excluded": len(skipped),  # in the verified splits
            "not_eligible": len(exclude["paths"]) - int(rows.video_path[listed].nunique())}

    index = video_index(rows, verdicts, verified, acq, cfg.get("clip"),
                        (cfg.get("acquisition") or {}).get("bmode_vendors"))
    studies = archive_studies(studies, index)
    excluded = archive_exclusions(index, acq is not None)
    check_archive(index, patient_splits, anchor)

    ok = index.reason.isna().to_numpy()
    manifests = {}
    for name, split in ARCHIVE_MANIFESTS.items():
        if split not in verify_splits or (name == "train_bmode" and acq is None):
            continue
        rows_in = {"train_inclusive": index.in_inclusive.to_numpy(), "train_bmode": index.in_bmode.to_numpy()}.get(
            name, ok & index.split.isin([split]).to_numpy())
        manifests[name] = index.video_path[rows_in].sort_values(ignore_index=True)
    outcomes = index.reason.fillna("ok").value_counts().sort_index()
    verification["videos_by_reason"] = {str(k): int(n) for k, n in outcomes.items()}
    result["counts"].update(
        acquisition={"index_given": acq is not None, "index_rows": index_rows,
                     "videos_scanned": int(index.acq_scanned.sum()),
                     "videos_not_scanned": int((~index.acq_scanned).sum()),
                     "eligible_not_in_index": not_in_index,
                     "coverage_complete": acq is not None and not sum(not_in_index.values()),
                     **(quality or {}),
                     "accepted_read_errors": int(accept_acquisition_read_errors),
                     "allow_partial_acquisition": bool(allow_partial_acquisition),
                     # Fit for the B-mode arm: full coverage, current scanner, no read error left unreviewed.
                     "ready": bool(acq is not None and not allow_partial_acquisition and not sum(not_in_index.values())
                                   and not sum(quality["stale_scanner_version"].values())
                                   and sum(quality["read_errors"].values()) <= accept_acquisition_read_errors)},
        **archive_report(index, studies, verify_splits))
    result.update(videos=index, studies=studies, excluded=excluded, manifests=manifests)
    return result


def distribution(counts):
    """Quantiles of a count per unit (patient or study), and the share of the total held by the 1% and
    the 10% of units with the largest counts."""
    c = np.sort(np.asarray(counts, dtype=np.int64))[::-1]
    if not len(c):
        return None
    out = {"units": int(len(c)), "total": int(c.sum()), "mean": round(float(c.mean()), 3)}
    out.update({k: float(np.quantile(c, q)) for k, q in QUANTILES.items()})
    for pct in (1, 10):
        out[f"share_top_{pct}pct"] = round(float(c[: max(1, int(np.ceil(len(c) * pct / 100)))].sum() / c.sum()), 4)
    return out


def acquisition_label(index):
    """Each video's acquisition category for reports, telling apart videos absent from the index
    (`not_scanned`), header B-mode from a vendor outside `acquisition.bmode_vendors`
    (`bmode_vendor_unvalidated`) and other B-mode categories the header does not confirm (`bmode_unconfirmed`)."""
    bmode = index.acq_category.isin(["bmode"]).to_numpy() & ~confirmed_bmode(index)
    header = index.acq_confidence.isin(["header"]).to_numpy()
    label = np.where(bmode & header, "bmode_vendor_unvalidated",
                     np.where(bmode, "bmode_unconfirmed", index.acq_category.to_numpy(dtype=object)))
    return pd.Series(np.where(index.acq_scanned.to_numpy(bool), label, "not_scanned"), index=index.index)


def archive_report(index, studies, verify_splits):
    """Aggregate counts of an archive build for `manifest_info.json` and `summary.md`: per split and
    manifest, verification reasons, videos by year, view, vendor, model and acquisition mode, studies
    and patients left without videos, and how concentrated the training videos are per patient."""
    ok = index.reason.isna().to_numpy()
    split = index.split.to_numpy(dtype=object)
    groups = {"eligible": np.ones(len(index), bool), "train_inclusive": index.in_inclusive.to_numpy(bool),
              "train_bmode": index.in_bmode.to_numpy(bool),
              "val": ok & (split == "val"), "test": ok & (split == "test")}

    def size(mask):
        return {"patients": int(index.patient_id[mask].nunique()), "studies": int(index.study_id[mask].nunique()),
                "videos": int(mask.sum())}

    bmode = ok & confirmed_bmode(index)
    splits = {}
    for s in SPLITS:
        entry = {"metadata": size(split == s)}
        if s in verify_splits:
            st = studies[studies.split == s]
            per_patient = st.groupby("patient_id")[["n_ok", "n_bmode"]].sum()
            entry.update(verified=size((split == s) & ok), confirmed_bmode=size((split == s) & bmode),
                         studies_without_verified_video=int((st.n_ok == 0).sum()),
                         patients_without_verified_video=int((per_patient.n_ok == 0).sum()),
                         studies_without_confirmed_bmode_video=int((st.n_bmode == 0).sum()),
                         patients_without_confirmed_bmode_video=int((per_patient.n_bmode == 0).sum()))
        splits[s] = entry

    arms = {}
    for name, mask in groups.items():
        if name == "eligible":
            continue
        arms[name] = size(mask)
        if "needs_padding" in index:
            arms[name]["needing_padding"] = int(index.needs_padding[mask].fillna(False).sum())
        if mask.any() and index.n_frames[mask].notna().any():
            arms[name]["median_frames"] = float(index.n_frames[mask].median())

    def table(key):
        key = key.astype("string").fillna(MISSING).replace("", MISSING).to_numpy(dtype=object)
        out = pd.DataFrame({g: pd.Series(key[m]).value_counts() for g, m in groups.items()})
        return table_dict(out.reindex(columns=list(groups)).fillna(0).astype(int).sort_index())

    dims = {}
    if "study_date" in index:
        dims["year"] = index.study_date.str[:4]
    dims["view"] = index.view
    for role in ("vendor", "model"):
        if role in index:
            dims[role] = index[role]
    if "vendor" in index and "model" in index:
        dims["vendor_model"] = index.vendor.fillna(MISSING) + " / " + index.model.fillna(MISSING)
    dims["acquisition"] = acquisition_label(index)

    concentration = {}
    for name in ("train_inclusive", "train_bmode"):
        sub = index.loc[groups[name], ["patient_id", "study_id"]]
        concentration[name] = None if sub.empty else {
            "videos_per_patient": distribution(sub.patient_id.value_counts()),
            "videos_per_study": distribution(sub.study_id.value_counts()),
            "studies_per_patient": distribution(sub.drop_duplicates().patient_id.value_counts()),
        }
    return {"splits": splits, "manifests": arms,
            "verification_reasons_by_split": table_dict(
                pd.crosstab(index.reason.fillna("ok"), index.split).reindex(columns=list(SPLITS), fill_value=0)),
            "videos_by": {d: table(k) for d, k in dims.items()}, "concentration": concentration}


def print_archive_report(result):
    counts = result["counts"]
    if result["assign_only"]:
        return
    arms = pd.DataFrame.from_dict(counts["manifests"], orient="index").rename_axis("manifest")
    print("\n[arms]   rows of each manifest (val/test: every valid video; train_bmode: header-confirmed B-mode)\n"
          + arms.fillna("").to_string())
    lost = pd.DataFrame({s: {k: v for k, v in e.items() if isinstance(v, int)} for s, e in counts["splits"].items()})
    if not lost.empty:
        print("\n[lost]   kept in their split, not redrawn\n" + lost.fillna("").to_string())


def sha256_path(path):
    """SHA-256 and size of a file; for a directory (a sharded parquet dataset), the SHA-256 of its sorted
    relative file names and their SHA-256s, skipping the hidden and `_` files a parquet reader skips."""
    if not os.path.isdir(path):
        return sha256(path), os.path.getsize(path)
    lines, total = [], 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
        for f in files:
            if f.startswith((".", "_")):
                continue
            p = os.path.join(root, f)
            lines.append(f"{os.path.relpath(p, path)}\t{sha256(p)}\n")
            total += os.path.getsize(p)
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest(), total


def write_archive(result, out_dir, cfg, inputs, argv):
    """Write an archive build.

    `patient_splits.csv`: the split of every patient; every later pipeline reuses it, and a later build
    is anchored on it. `anchor_absent_patients.csv`: anchored patients without an eligible study here.
    `studies.csv`: every eligible study, its split and video counts.
    Unless --assign-only: the loader manifests `train_inclusive.csv`, `train_bmode.csv` (with an
    acquisition index), `val.csv` and `test.csv` (only for verified splits); `videos.parquet`, the
    master index of every eligible video; `excluded_videos.csv`.
    `manifest_info.json`: provenance. `summary.md`: aggregate numbers only, safe to share.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    def csv(name, df):
        df.to_csv(os.path.join(out_dir, name), index=False)
        written[name] = len(df)

    csv("patient_splits.csv", result["patient_splits"])
    csv("anchor_absent_patients.csv", result["anchor_absent"])
    csv("studies.csv", result["studies"])
    if not result["assign_only"]:
        for name, paths in result["manifests"].items():
            # `<video_path> 0`: the loader's parser expects a label; 0 is the pretraining placeholder.
            paths.to_frame("video_path").assign(label=0).to_csv(os.path.join(out_dir, f"{name}.csv"), sep=" ",
                                                                header=False, index=False)
            written[f"{name}.csv"] = len(paths)
        result["videos"].to_parquet(os.path.join(out_dir, "videos.parquet"), index=False)
        written["videos.parquet"] = len(result["videos"])
        csv("excluded_videos.csv", result["excluded"])

    info = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": argv,
        "git_commit": git_commit(),
        "seed": result["seed"],
        "mode": ALL_STUDIES,
        "assign_only": result["assign_only"],
        "config": cfg,
        "inputs": inputs,
        "counts": result["counts"],
        "outputs": {f: {"rows": n, "sha256": sha256(os.path.join(out_dir, f))} for f, n in written.items()},
    }
    with open(os.path.join(out_dir, "manifest_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(archive_summary_markdown(info))
    return info


# Written last into a staged archive build: the checksum of every other file, then the completion receipt.
OUTPUT_CHECKSUMS = "outputs.sha256"
BUILD_RECEIPT = "BUILD_COMPLETE.json"


def publish_archive(result, out_dir, cfg, inputs, argv, smoke=None):
    """Write an archive build into a fresh staging folder next to `out_dir`, run `smoke(folder)` there, record
    every file's SHA-256 (`outputs.sha256`, `sha256sum -c` format) and then the completion receipt
    (`BUILD_COMPLETE.json`), and only then rename the folder to `out_dir`. Readers see the earlier build or the
    new one, never a mix of the two or an unchecked one. An earlier build is kept as
    `<out_dir>.replaced-<time>`. A failure leaves `out_dir` as it was, and the staging folder for inspection: if the
    final rename fails after the earlier build was moved aside, the earlier build is moved back. Only an abrupt
    end of the process between those two renames (a kill, a power cut) leaves `out_dir` missing, with the earlier
    build in the newest `<out_dir>.replaced-*`; the next run reports it (`report_interrupted_publication`).

    Returns the build's `manifest_info` and the folder the earlier build was moved to (None if there was none).
    """
    out = os.path.abspath(out_dir).rstrip(os.sep)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def sibling(tag):  # a name next to `out` that no earlier build used
        n = 0
        while os.path.lexists(name := f"{out}.{tag}-{stamp}-{os.getpid()}" + (f"-{n}" if n else "")):
            n += 1
        return name

    stage = sibling("staging")
    os.makedirs(stage)
    try:
        info = write_archive(result, stage, cfg, inputs, argv)
        if smoke is not None:
            smoke(stage)
    except BaseException as e:
        e.add_note(f"Nothing was published: {out} is unchanged; the staged build is kept at {stage}.")
        raise
    names = sorted(f for f in os.listdir(stage) if f not in (OUTPUT_CHECKSUMS, BUILD_RECEIPT))
    checksums = os.path.join(stage, OUTPUT_CHECKSUMS)
    with open(checksums, "w") as f:
        f.writelines(f"{sha256(os.path.join(stage, name))}  {name}\n" for name in names)
    receipt = {
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": info["git_commit"], "assign_only": result["assign_only"], "files": names,
        "outputs_sha256": sha256(checksums), "smoke_tested": smoke is not None,
        "acquisition_ready": bool((result["counts"].get("acquisition") or {}).get("ready")),
    }
    with open(os.path.join(stage, BUILD_RECEIPT), "w") as f:
        json.dump(receipt, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    replaced = None
    if os.path.isdir(out) and not os.listdir(out):
        os.rmdir(out)
    elif os.path.lexists(out):
        replaced = sibling("replaced")
        os.rename(out, replaced)
    try:
        os.rename(stage, out)
    except BaseException as e:
        if replaced is not None and not os.path.lexists(out):
            os.rename(replaced, out)  # put the earlier build back
            replaced = None
        e.add_note(f"Nothing was published: {out} holds the earlier build (if any); the new build is at {stage}.")
        raise
    return info, replaced


def report_interrupted_publication(out_dir):
    """Warn when `out_dir` is missing while earlier builds were moved aside next to it: a publication was cut off
    between its two renames. The newest `<out_dir>.replaced-*` is the build that was in place before."""
    out = os.path.abspath(out_dir).rstrip(os.sep)
    if os.path.lexists(out):
        return None
    parent, base = os.path.split(out)
    earlier = sorted(n for n in (os.listdir(parent) if os.path.isdir(parent) else []) if n.startswith(f"{base}.replaced-"))
    if earlier:
        print(f"[out]    WARNING: {out} is missing but {len(earlier)} earlier build(s) were moved aside next to it: a "
              f"replacement was interrupted. The build in place before it is {os.path.join(parent, earlier[-1])}; "
              f"restore it with `mv` if this run should not publish a new one.", flush=True)
        return os.path.join(parent, earlier[-1])
    return None


def archive_summary_markdown(info):
    """The aggregate numbers of an archive build, without paths, identifiers or the command line."""
    cfg, counts = info["config"], info["counts"]
    el, a = cfg["eligibility"], counts["assignment"]

    def counts_table(rows, name):
        return markdown_table(pd.DataFrame(rows).T.fillna(0).astype(int).rename_axis(name))

    lines = [
        "# Archive manifest summary", "",
        "Aggregate counts only: no file paths, patient or study identifiers.", "",
        f"- Created (UTC): {info['created_utc']}",
        f"- Code version (git commit): {info['git_commit'] or 'unknown'}",
        f"- Seed (orders the patients new to the anchor): {info['seed']}",
        "", "## Cohort definition", "",
        "- Every eligible study of every patient, longitudinal studies included",
        f"- Exam types: {', '.join(el['exam_types'])}",
        f"- Video status: {', '.join(el['status_ok'])}",
        f"- Study label required: {'yes' if el.get('require_label') else 'no'}",
        f"- Years: {', '.join(map(str, el['years'])) if el.get('years') else 'all'}",
        "- Patient split target over all eligible patients: " + ", ".join(f"{s} {cfg['split'][s]}" for s in SPLITS),
        "- Anchored patients keep their split; patients new to the anchor fill the rest of each target in a "
        "seeded order. Splits are assigned before verification and never redrawn.",
    ]
    if not info["assign_only"]:
        ver = counts["verification"]
        acq = counts["acquisition"]
        bmode = ("valid train videos whose DICOM header confirms B-mode (`acq_category` bmode, `acq_confidence` "
                 "header) from a vendor in `acquisition.bmode_vendors`" if acq["index_given"]
                 else "not built: no acquisition index was given")
        lines += [f"- Valid video: file exists and is non-empty, every frame decodes ({ver['verify_version']})",
                  f"- Splits verified: {', '.join(ver['verify_splits'])}",
                  "- train_inclusive: every valid video of a train patient, whatever its view or acquisition mode",
                  f"- train_bmode: {bmode}"]
        if acq["index_given"] and not acq.get("ready"):
            lines.append(f"- **Development build: the acquisition index is not ready** "
                         f"({sum(acq['eligible_not_in_index'].values())} eligible videos never scanned, "
                         f"{sum(acq['stale_scanner_version'].values())} from another scanner version, "
                         f"{sum(acq['read_errors'].values())} failed header reads; admitted with "
                         "--allow-partial-acquisition); train_bmode is not the B-mode arm.")
        elif acq["index_given"] and acq["accepted_read_errors"]:
            lines.append(f"- Failed header reads accepted after review: {sum(acq['read_errors'].values())} videos "
                         "(left out of train_bmode, kept in train_inclusive).")
        if cfg.get("clip"):
            clip = cfg["clip"]
            lines.append(f"- Padding flag: shorter than one clip of {clip['frames_per_clip']} frames at "
                         f"{clip['fps']} fps (every frame for a video without a usable frame rate).")

    funnel = pd.Series(counts["eligibility_funnel"], name="videos").rename_axis("rule").to_frame()
    lines += ["", "## Eligibility", "", markdown_table(funnel), "",
              f"{counts['eligible_studies']} studies from {counts['eligible_patients']} patients are eligible.",
              "", "### Videos left after each rule, by exam type", "",
              counts_table(counts["eligibility_by_exam_type"], "exam type"),
              "", "### Videos left after each rule, by view", "", counts_table(counts["eligibility_by_view"], "view")]

    absent = {s: sum(d[s] for d in a["anchored_absent"].values()) for s in SPLITS}
    split_rows = {s: {"target": a["targets"][s], "anchored legacy": a["anchored_present"]["legacy"][s],
                      "anchored earlier build": a["anchored_present"]["new"][s], "new": a["new_patients"][s],
                      "patients": a["patients"][s], "studies": a["studies"][s],
                      "metadata videos": a["metadata_videos"][s],
                      **({"labelled studies": a["labelled_studies"][s]} if "labelled_studies" in a else {})}
                  for s in SPLITS}
    lines += ["", "## Patient split", "", markdown_table(pd.DataFrame(split_rows).T.rename_axis("split")), "",
              f"The anchor lists {a['anchor_patients']} patients; {sum(absent.values())} of them have no eligible "
              "study in this archive (" + ", ".join(f"{s} {n}" for s, n in absent.items()) + "). They keep their "
              "split in patient_splits.csv (`in_archive` false) and are listed in anchor_absent_patients.csv."]
    if info["assign_only"]:
        return _checksum_lines(lines, info)

    lines += ["", "## Verification", "",
              f"- Videos to verify: {ver['videos_to_verify']}",
              f"- Reused from the verification cache: {ver['reused_from_cache']} "
              f"({ver['stale_cache_entries']} cached entries out of date)",
              f"- Verified in this run: {ver['verified_this_run']} with {ver['workers']} workers"
              + (f", {ver['videos_per_second']} videos/s" if ver["videos_per_second"] else ""),
              *([f"- **Unresolved videos excluded after review** (--accept-verify-failures; not cached, a later "
                 f"build retries them): {ver['accepted_unresolved']}"] if ver.get("accepted_unresolved") else []),
              *([(f"- Excluded by --exclude-videos, not decoded: {excl['excluded']} videos as `{excl['reason']}` "
                  f"({excl['listed']} listed, {excl['not_eligible']} of them not eligible)")]
                if (excl := counts.get("exclusion_list")) else []),
              "", "Videos by verification outcome and split (`not_verified`: split not verified):", "",
              counts_table(counts["verification_reasons_by_split"], "reason")]

    arms = pd.DataFrame.from_dict(counts["manifests"], orient="index").rename_axis("manifest")
    lost = {s: {k.replace("_", " "): v for k, v in e.items() if isinstance(v, int)}
            for s, e in counts["splits"].items()}
    lines += ["", "## Manifests", "", markdown_table(arms), "",
              "val and test hold every valid video of their patients. The inclusive arm keeps videos of every "
              "acquisition mode, unknown and other modes included, so comparing it with train_bmode is not a "
              "pure color-only ablation.",
              "", "### Studies and patients left without videos", "",
              "Kept in their split and counted; nothing is redrawn.", "",
              markdown_table(pd.DataFrame(lost).T.rename_axis("split"))]

    titles = {"acquisition": "acquisition mode", "year": "study year", "view": "view", "vendor": "vendor",
              "model": "model", "vendor_model": "vendor / model"}
    for key, rows in counts["videos_by"].items():
        lines += ["", f"### Videos by {titles.get(key, key)}", "", counts_table(rows, titles.get(key, key))]

    lines += ["", "## Concentration of training videos", "",
              "Pretraining samples manifest rows (clips) uniformly, so each patient weighs in proportion to "
              "their number of videos summed over all their studies: patients with many studies or long "
              "examinations weigh more than patients seen once."]
    for arm, conc in counts["concentration"].items():
        if not conc:
            continue
        table = pd.DataFrame.from_dict(conc, orient="index").rename_axis("unit")
        top1, top10 = conc["videos_per_patient"]["share_top_1pct"], conc["videos_per_patient"]["share_top_10pct"]
        lines += ["", f"### {arm}", "", markdown_table(table), "",
                  f"The 1% of patients with the most videos hold {top1:.1%} of {arm}; the top 10% hold {top10:.1%}."]
    return _checksum_lines(lines, info)


def _checksum_lines(lines, info):
    lines += ["", "## Checksums (SHA-256)", "", "| file | size | sha256 |", "|---|---|---|"]
    lines += [f"| input: {role} | {meta['bytes']} bytes | {meta['sha256']} |" for role, meta in info["inputs"].items()]
    lines += [f"| {name} | {meta['rows']} rows | {meta['sha256']} |" for name, meta in info["outputs"].items()]
    return "\n".join(lines) + "\n"


# Seconds the final smoke test of one manifest may take in its worker before the worker is killed (and the test
# tried once more): it parses the manifest (millions of rows for train_inclusive) and decodes a few videos, about a
# minute; the margin also outlasts an NFS major timeout (~3 min).
SMOKE_TIMEOUT = 900.0


class SmokeTestFailed(RuntimeError):
    """The final loader smoke test failed, or could not finish (a stalled or crashed worker)."""


def _smoke_one(task):
    """One manifest's smoke test in a supervised worker: None when it passes, else the failure as text (returned,
    not raised, so the message reaches the parent and a real failure is not retried)."""
    out_dir, name, clip = task
    try:
        smoke_test_archive(out_dir, [name], clip)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def supervised_smoke_test(out_dir, manifests, clip=None, timeout=SMOKE_TIMEOUT, retries=1):
    """`smoke_test_archive` with each manifest in its own supervised worker process under a deadline of `timeout`
    s: a decoder or storage stall, or a crashed worker, ends in SmokeTestFailed instead of a hang."""
    sp = supervision()
    tasks = [(out_dir, name, clip) for name in manifests]
    problems = []
    try:
        with sp.SupervisedPool(_smoke_one, max(1, len(tasks)), timeout, retries) as pool:
            for i, outcome in pool.imap_unordered(tasks):
                if isinstance(outcome, sp.Failure):
                    problems.append(f"{tasks[i][1]}: no result ({outcome.kind} after {outcome.attempts} attempts "
                                    f"of at most {timeout:g} s)")
                elif outcome is not None:
                    problems.append(outcome)
    except sp.PoolBroken as e:
        problems.append(f"the smoke-test workers stopped: {e}")
    if problems:
        raise SmokeTestFailed("The loader smoke test did not pass: " + "; ".join(sorted(problems)))


def smoke_test_archive(out_dir, manifests, clip=None):
    """The smoke test of an archive build's `manifests`, with the special cases drawn from the rows of
    videos.parquet that each manifest holds."""
    import pyarrow.parquet as pq

    sys.path.insert(0, REPO_ROOT)
    path = os.path.join(out_dir, "videos.parquet")
    names = pq.read_schema(path).names
    columns = ["video_path", "split", "reason", "fps", "needs_padding", "in_inclusive", "in_bmode"]
    index = pd.read_parquet(path, columns=[c for c in columns if c in names])
    ok = index.reason.isna()
    for name in manifests:
        rows = {"train_inclusive": index.in_inclusive, "train_bmode": index.in_bmode}.get(
            name, ok & (index.split == ARCHIVE_MANIFESTS[name]))
        smoke_test_manifest(os.path.join(out_dir, f"{name}.csv"), index[rows.astype(bool)], clip)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Cohort definition (column mappings, rules, split).")
    p.add_argument("--metadata", required=True, help="Per-video metadata (.parquet or .csv).")
    p.add_argument("--labels", help="Study-level labels from `link_ef_labels.py`.")
    p.add_argument("--out-dir", required=True, help="Where the manifests go; must be outside this repository.")
    p.add_argument("--seed", type=int, help="Overrides `selection.seed`, and the overriding value itself is what gets saved.")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1),
                   help="Processes for video verification (default: 32, or the CPU count if lower).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="The number of metadata rows read at a time.")
    # `--dry-run`: only run metadata scanning, eligibility rules, duplicate checks, and study/patient consistency checks.
    p.add_argument("--dry-run", action="store_true", help="Report eligibility counts only; skip video verification and do not write output files.")
    p.add_argument("--overwrite", action="store_true", help="Replace manifests in a non-empty `--out-dir`.")
    p.add_argument("--smoke-timeout", type=float,
                   help=f"Seconds the final smoke test of one manifest may take in its worker (archive mode; default "
                        f"{SMOKE_TIMEOUT:g}); a stalled or crashed worker fails the build instead of hanging it.")
    p.add_argument("--no-smoke-test", action="store_true", help="Skip the final validation step that reloads the generated outputs using the EchoJEPA parser.")
    a = p.add_argument_group(f"archive mode (selection.mode: {ALL_STUDIES})")
    a.add_argument("--anchor-splits",
                   help="Required. Patient splits to keep: the 10k build's studies.csv, or an earlier archive "
                        "build's patient_splits.csv.")
    a.add_argument("--assign-only", action="store_true", help="Assign the splits and write patient_splits.csv, "
                                                               "studies.csv and the reports; verify nothing.")
    a.add_argument("--verify-cache", help="Directory of cached verifications (outside this repository); reused when a "
                                          "video's size, modification time and the verification version match.")
    a.add_argument("--verify-splits", nargs="+", choices=SPLITS,
                   help="Splits whose videos are verified (default: all); other videos are excluded as not_verified.")
    a.add_argument("--verify-limit", type=int, help="Verify at most N pending videos, report the rate, and stop "
                                                    "without writing manifests (0: report the cache coverage only).")
    a.add_argument("--verify-chunk", type=int, help=f"Videos verified between cache writes (default {VERIFY_CHUNK}).")
    a.add_argument("--stat-threads", type=int, help=f"Threads that stat the videos (default {STAT_THREADS}).")
    a.add_argument("--verify-timeout", type=float, metavar="SECONDS",
                   help=f"Deadline for one video in a worker process (default {VERIFY_TIMEOUT:g}); a worker that "
                        "overruns it or dies is killed and replaced, and the video tried again.")
    a.add_argument("--verify-retries", type=int, metavar="N",
                   help=f"Further attempts in this run for a video that timed out, crashed its worker or failed to "
                        f"decode ({', '.join(VERIFY_RETRY_REASONS)}) (default {VERIFY_RETRIES}).")
    a.add_argument("--retry-failed", action="store_true",
                   help="Verify the cached decode failures again (a storage error can recover with the file "
                        "unchanged); cached successes are still reused.")
    a.add_argument("--accept-verify-failures", type=int, default=0, metavar="N",
                   help="Once inspected, exclude up to N videos still unresolved after every attempt (they are "
                        "listed in the cache directory) as verify_timeout / verify_crash / verify_error, in this "
                        "build only. Without it, or with more of them, the run stops and writes nothing.")
    a.add_argument("--acquisition-index",
                   help="Acquisition index (parquet file or directory) from scan_acquisition.py, joined on the "
                        "DICOM path; needed for train_bmode.csv. It must list every eligible video of the verified "
                        "splits.")
    a.add_argument("--allow-partial-acquisition", action="store_true",
                   help="Development only: accept an acquisition index that lacks eligible videos (a sample scan), "
                        "comes from another scanner version or has failed reads; those videos are left out of "
                        "train_bmode, and manifest_info.json (`ready` false) and summary.md record it.")
    a.add_argument("--accept-acquisition-read-errors", type=int, default=0, metavar="N",
                   help="Admit up to N eligible videos whose header read failed even after --retry-errors, once "
                        "reviewed; they stay out of train_bmode and are counted in manifest_info.json.")
    a.add_argument("--exclude-videos", metavar="LIST",
                   help="CSV or Parquet of videos to leave out, in a column named like the metadata's video column "
                        "(columns.video_path, e.g. avi_path). Each listed eligible video is excluded without being "
                        f"decoded, with reason {EXCLUDED_PREFIX}<--exclude-reason>; the list is recorded with the "
                        "build's inputs.")
    a.add_argument("--exclude-reason", metavar="LABEL",
                   help="Why --exclude-videos leaves its videos out: lower-case letters, digits and underscores "
                        "(e.g. deid_black).")
    return p.parse_args()


# Options of archive mode, which one-study mode refuses rather than ignores.
ARCHIVE_OPTIONS = {d: "--" + d.replace("_", "-") for d in ("anchor_splits", "assign_only", "verify_cache",
                                                           "verify_splits", "verify_limit", "verify_chunk",
                                                           "stat_threads", "verify_timeout", "verify_retries",
                                                           "retry_failed", "accept_verify_failures", "acquisition_index",
                                                           "allow_partial_acquisition", "smoke_timeout",
                                                           "accept_acquisition_read_errors", "exclude_videos",
                                                           "exclude_reason")}


def check_archive_args(args):
    """Refuse archive option combinations that cannot do what they say."""
    if not args.anchor_splits:
        raise SystemExit(f"--anchor-splits is required with selection.mode {ALL_STUDIES}: the 10k build's studies.csv "
                         "or an earlier archive build's patient_splits.csv.")
    if args.assign_only and any(x is not None for x in (args.verify_cache, args.verify_splits, args.verify_limit,
                                                         args.acquisition_index)):
        raise SystemExit("--assign-only verifies nothing: drop --verify-cache, --verify-splits, --verify-limit "
                         "and --acquisition-index.")
    if (args.allow_partial_acquisition or args.accept_acquisition_read_errors) and not args.acquisition_index:
        raise SystemExit("--allow-partial-acquisition and --accept-acquisition-read-errors need --acquisition-index.")
    if args.accept_acquisition_read_errors < 0:
        raise SystemExit("--accept-acquisition-read-errors must be 0 or more.")
    if args.smoke_timeout is not None and not args.smoke_timeout > 0:
        raise SystemExit(f"--smoke-timeout must be a positive number of seconds, got {args.smoke_timeout}.")
    if args.dry_run and (args.assign_only or args.verify_limit is not None):
        raise SystemExit("--dry-run verifies and writes nothing: drop --assign-only and --verify-limit.")
    if args.verify_limit is not None and args.verify_limit < 0:
        raise SystemExit(f"--verify-limit must be 0 or more, got {args.verify_limit}.")
    for flag, n in (("--verify-chunk", args.verify_chunk), ("--stat-threads", args.stat_threads)):
        if n is not None and n < 1:
            raise SystemExit(f"{flag} must be a positive integer, got {n}.")
    if args.assign_only and (args.retry_failed or args.accept_verify_failures
                             or args.verify_timeout is not None or args.verify_retries is not None):
        raise SystemExit("--assign-only verifies nothing: drop --verify-timeout, --verify-retries, --retry-failed "
                         "and --accept-verify-failures.")
    if args.retry_failed and not args.verify_cache:
        raise SystemExit("--retry-failed verifies cached failures again: it needs --verify-cache.")
    if args.verify_timeout is not None and not args.verify_timeout > 0:
        raise SystemExit(f"--verify-timeout must be positive, got {args.verify_timeout}.")
    for flag, n in (("--verify-retries", args.verify_retries), ("--accept-verify-failures", args.accept_verify_failures)):
        if n is not None and n < 0:
            raise SystemExit(f"{flag} must be 0 or more, got {n}.")
    if bool(args.exclude_videos) != bool(args.exclude_reason):
        raise SystemExit("--exclude-videos and --exclude-reason go together: the list, and why it leaves videos out.")
    if args.exclude_reason and not EXCLUDE_REASON.fullmatch(args.exclude_reason):
        raise SystemExit(f"--exclude-reason is lower-case letters, digits and underscores, not {args.exclude_reason!r}.")
    if args.assign_only and args.exclude_videos:
        raise SystemExit("--assign-only verifies nothing: drop --exclude-videos and --exclude-reason.")


def read_exclusion_list(path, reason, cfg):
    """`--exclude-videos` as `build_archive`'s `exclude` (None without a list): the paths in the list's column
    named like the metadata's video column (`columns.video_path`), so a list keyed like the metadata is read as
    it is. A list without that column or with an empty path is refused."""
    if not path:
        return None
    column = cfg["columns"]["video_path"]
    paths = read_columns(path, [column])[column]
    if paths.isna().any() or (paths.astype(str).str.strip() == "").any():
        raise SystemExit(f"--exclude-videos {path} has rows without a video path in its {column!r} column.")
    return {"reason": reason, "paths": set(paths.astype(str))}


def input_record(path):
    digest, size = sha256_path(path)
    return {"path": os.path.abspath(path), "bytes": size, "sha256": digest}


def main_archive(args, cfg):
    check_archive_args(args)
    if args.verify_cache:
        ensure_outside_repo(args.verify_cache, "--verify-cache")
    if not args.dry_run and args.verify_limit is None:
        if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
            raise SystemExit(f"{args.out_dir} is not empty; the split is assigned once. "
                             "Use `--overwrite` to replace it.")
        out, anchor = os.path.realpath(args.out_dir), os.path.realpath(args.anchor_splits)
        if os.path.commonpath([out, anchor]) == out:
            raise SystemExit("--anchor-splits lies inside --out-dir: write the new build to another directory, so the "
                             "assignments it keeps are never overwritten.")

    labels = read_columns(args.labels, [cfg["labels"]["study_id"], cfg["labels"]["value"]]) if args.labels else None
    anchor = read_anchor(args.anchor_splits)
    if not args.dry_run and args.verify_limit is None:
        report_interrupted_publication(args.out_dir)
    if args.dry_run:
        lab = load_labels(labels, cfg) if labels is not None else None
        studies, _ = scan_eligible(args.metadata, cfg, lab, args.batch_size)
        _, stats = assign_splits(studies.patient_id, parse_anchor(anchor), [cfg["split"][s] for s in SPLITS],
                                 cfg["selection"]["seed"])
        print_assignment(stats)
        print("\n[dry-run] nothing verified or written.")
        return

    try:
        result = build_archive(args.metadata, labels, cfg, anchor, workers=args.workers, batch_size=args.batch_size,
                               acquisition=args.acquisition_index, verify_splits=args.verify_splits or SPLITS,
                               cache_dir=args.verify_cache, verify_limit=args.verify_limit,
                               verify_chunk=args.verify_chunk or VERIFY_CHUNK,
                               stat_threads=args.stat_threads or STAT_THREADS, assign_only=args.assign_only,
                               allow_partial_acquisition=args.allow_partial_acquisition,
                               accept_acquisition_read_errors=args.accept_acquisition_read_errors,
                               verify_timeout=args.verify_timeout or VERIFY_TIMEOUT,
                               verify_retries=VERIFY_RETRIES if args.verify_retries is None else args.verify_retries,
                               retry_failed=args.retry_failed, accept_verify_failures=args.accept_verify_failures,
                               exclude=read_exclusion_list(args.exclude_videos, args.exclude_reason, cfg))
    except (VerificationIncomplete, supervision().PoolBroken) as e:
        # Infrastructure trouble, not a data verdict: completed results stay cached and nothing is published.
        raise SystemExit(f"{type(e).__name__}: {e}") from None
    if result["stopped"]:
        return
    inputs = {name: input_record(p) for name, p in (("metadata", args.metadata), ("labels", args.labels),
                                                    ("anchor_splits", args.anchor_splits),
                                                    ("acquisition_index", args.acquisition_index),
                                                    ("exclude_videos", args.exclude_videos)) if p}
    smoke = None
    if not args.no_smoke_test and not args.assign_only:
        # A split with fraction 0 is empty by design; every other manifest must be readable.
        def smoke(folder):
            supervised_smoke_test(folder, [n for n in result["manifests"] if cfg["split"][ARCHIVE_MANIFESTS[n]] > 0],
                                  cfg.get("clip"), timeout=args.smoke_timeout or SMOKE_TIMEOUT)
    try:
        info, replaced = publish_archive(result, args.out_dir, cfg, inputs, sys.argv, smoke)
    except SmokeTestFailed as e:
        raise SystemExit("\n".join([f"SmokeTestFailed: {e}", *getattr(e, "__notes__", [])])) from None
    print_archive_report(result)
    print(f"\n[out]    {args.out_dir}: " + ", ".join(info["outputs"])
          + f", manifest_info.json, summary.md (safe to share), {OUTPUT_CHECKSUMS}, {BUILD_RECEIPT}.")
    if replaced:
        print(f"[out]    the earlier build was moved to {replaced}.")


def main():
    args = parse_args()
    # Generated files hold patient identifiers; never write them inside the source repository.
    if not args.dry_run:
        ensure_outside_repo(args.out_dir, "--out-dir")
    cfg = load_config(args.config)
    if args.seed is not None:
        # Write the actual random seed used for cohort selection back into the configuration object, so that when the
        # config is later saved to `manifest_info.json`, it records the seed that really produced the cohort.
        cfg["selection"]["seed"] = args.seed
    if cfg["selection"].get("mode", ONE_STUDY) == ALL_STUDIES:
        return main_archive(args, cfg)
    given = [flag for dest, flag in ARCHIVE_OPTIONS.items() if getattr(args, dest) not in (None, False)]
    if given:
        raise SystemExit(f"{', '.join(given)} only apply to selection.mode {ALL_STUDIES}.")
    # Refuse to overwrite by default to protect the cohort.
    if not args.dry_run and os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
        raise SystemExit(f"{args.out_dir} is not empty; the cohort is drawn once. Use `--overwrite` to replace it.")

    labels = read_columns(args.labels, [cfg["labels"]["study_id"], cfg["labels"]["value"]]) if args.labels else None

    if args.dry_run:
        lab = load_labels(labels, cfg) if labels is not None else None
        scan_eligible(args.metadata, cfg, lab, args.batch_size)
        print("\n[dry-run] nothing verified or written.")
        return

    result = build(args.metadata, labels, cfg, workers=args.workers, batch_size=args.batch_size)
    inputs = {name: {"path": os.path.abspath(p), "bytes": os.path.getsize(p), "sha256": sha256(p)}
              for name, p in (("metadata", args.metadata), ("labels", args.labels)) if p}
    report = write_outputs(result, args.out_dir, cfg, inputs, sys.argv)
    print_report(report)
    print(f"\n[out]    {args.out_dir}: train/val/test.csv, videos.csv, studies.csv, "
          "excluded_videos.csv, manifest_info.json, summary.md (safe to share).")
    if not args.no_smoke_test:
        # A split with fraction 0 is empty by design; every other one must be readable.
        smoke_test(args.out_dir, [s for s in SPLITS if cfg["split"][s] > 0], cfg.get("clip"))


if __name__ == "__main__":
    main()
