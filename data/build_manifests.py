# /data/build_manifests.py

"""Stage 2: Builds a clean, reproducible patient-level dataset split for EchoJEPA
from raw video metadata and optional study-level labels.

Example:
    python data/build_manifests.py --config configs/data/manifests_tte_10k_ef.yaml \\
        --metadata <video_metadata.parquet> --labels <study_labels.parquet> \\
        --out-dir <output_dir> --workers 32   # (balance memory and decoder overhead)

See data/README.md for the inputs, the outputs and how experiments use them.
"""

import argparse
import datetime
import functools
import hashlib
import json
import os
import subprocess
import sys
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

VIDEO_ROLES = ("patient_id", "study_id", "exam_type", "view", "video_path", "avi_status")

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
    n = sel.get("n_patients")
    require(positive_int(n), f"selection.n_patients must be a positive integer, got {n!r}.")
    seed = sel.get("seed")
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            f"selection.seed must be a non-negative integer, got {seed!r}.")
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
    return list(VIDEO_ROLES) + (["study_date"] if "study_date" in cols else [])


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
    el, cols = cfg["eligibility"], cfg["columns"]
    if el.get("require_label") and labels is None:
        raise ValueError("eligibility.require_label is set but no label table was given.")
    if el.get("years") and "study_date" not in cols:
        raise ValueError("eligibility.years needs columns.study_date")

    label_ids = set(labels.study_id) if labels is not None else None
    total, parts, path_hash = None, [], []
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
        parts.append(v.drop(columns=["view", "video_path", "avi_status"]).drop_duplicates())
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
        raise ValueError(f"{paths[conflict].nunique()} video paths appear under "
                         "conflicting ids, exam types or views.")
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
    return studies.reset_index(drop=True), counts


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


def smoke_test(out_dir, splits=SPLITS):
    """Read the manifest of each of `splits` with the EchoJEPA parser and decode its first
    clip; raise a clear error if a manifest is empty or cannot be ingested.
    """
    sys.path.insert(0, REPO_ROOT)
    from src.datasets.video_dataset import VideoDataset

    for s in splits:
        path = os.path.join(out_dir, f"{s}.csv")
        if os.path.getsize(path) == 0:
            raise RuntimeError(f"{s}.csv is empty, so nothing shows the EchoJEPA parser can read it.")
        expected = pd.read_csv(path, sep=" ", header=None)[0].tolist()
        ds = VideoDataset(data_paths=[path], frame_step=4)
        if list(ds.samples) != expected:
            raise RuntimeError(f"{s}.csv: the EchoJEPA parser read different paths than were written.")
        if any(int(x) != 0 for x in ds.labels):
            raise RuntimeError(f"{s}.csv: the EchoJEPA parser read labels other than 0.")
        # Use `get_item_video` rather than `ds[0]`: normal dataset indexing retries another random video
        # on load failure, which would mask unreadable videos during verification.
        loaded = ds.get_item_video(0)
        if not loaded:
            raise RuntimeError(f"{s}.csv: the EchoJEPA loader could not decode {expected[0]}.")
        print(f"[smoke]  {s}.csv: {len(ds)} rows parsed, first clip decoded "
              f"{tuple(np.asarray(loaded[0][0]).shape)}.")


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
    p.add_argument("--no-smoke-test", action="store_true", help="Skip the final validation step that reloads the generated outputs using the EchoJEPA parser.")
    return p.parse_args()


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
        smoke_test(args.out_dir, [s for s in SPLITS if cfg["split"][s] > 0])


if __name__ == "__main__":
    main()
