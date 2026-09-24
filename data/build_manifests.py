# /data/build_manifests.py

"""Stage 2: Builds a clean, reproducible patient-level dataset split for EchoJEPA
from raw video metadata and optional study-level labels.

Example:
    python data/build_manifests.py --config configs/data/manifests_tte_10k_ef.yaml \\
        --metadata <video_metadata.parquet> --labels <study_labels.parquet> \\
        --out-dir <output_dir> --workers 32   # (balance memory and decoder overhead)
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

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "val", "test")

# Create three independent random streams (RNG).
# Note: never to reorder `STAGES`, because that silently changes every cohort drawn with the same seed.
STAGES = ("patients", "study", "split")

VIDEO_ROLES = ("patient_id", "study_id", "exam_type", "view", "video_path", "status")

# The number of metadata rows per streamed batch.
BATCH_SIZE = 500000


# --------------------------------------------------------------------------- #
#  Helper Functions
# --------------------------------------------------------------------------- #


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [r for r in VIDEO_ROLES if r not in cfg.get("columns", {})]
    if missing:
        raise ValueError(f"{path}: columns has no mapping for {missing}.")
    fracs = [cfg["split"][s] for s in SPLITS]
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"{path}: split fractions sum to {sum(fracs)}, not 1.")
    return cfg


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


# --------------------------------------------------------------------------- #
#  Loading & Eligibility
# --------------------------------------------------------------------------- #


def video_roles(cols):
    return list(VIDEO_ROLES) + (["study_date"] if "study_date" in cols else [])


def eligible_rows(batch, cfg, label_ids):
    """The rows in a metadata batch that pass all eligibility rules. For each eligibility
    rule, report how many data rows remain after applying that rule. The rules are shown
    with standardized names and IDs.
    """
    el, cols = cfg["eligibility"], cfg["columns"]
    roles = video_roles(cols)
    v = batch[[cols[r] for r in roles]].set_axis(roles, axis=1)
    left = {"metadata_videos": len(v)}
    v = v[v.exam_type.isin(el["exam_types"])]
    left["allowed_exam_type"] = len(v)
    v = v[v.status.isin(el["status_ok"])]
    left["status_ok"] = len(v)

    # Normalize these columns only after filtering because they are expensive to process
    # and most rows have already been removed.
    v = v.assign(patient_id=normalize_id(v.patient_id), study_id=normalize_id(v.study_id),
                 video_path=v.video_path.astype("string"))
    if "study_date" in v:
        v = v.assign(study_date=v.study_date.astype("string").str.strip())

    if el.get("require_label"):
        v = v[v.study_id.isin(label_ids)]
    left["labelled"] = len(v)
    if el.get("years"):
        v = v[v.study_date.str[:4].isin({str(y) for y in el["years"]})]
    left["in_years"] = len(v)

    for role in ("patient_id", "study_id", "video_path"):
        n = v[role].isna().sum()
        if n:
            raise ValueError(f"{n} eligible videos have no {role}.")
    return v, left


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

    roles = video_roles(cols)
    label_ids = set(labels.study_id) if labels is not None else None
    left, parts, path_hash, row_hash = {}, [], [], []
    for batch in iter_table(source, [cols[r] for r in roles], batch_size):
        v, n = eligible_rows(batch, cfg, label_ids)
        left = {k: left.get(k, 0) + x for k, x in n.items()}
        path_hash.append(pd.util.hash_pandas_object(v.video_path, index=False).to_numpy())
        row_hash.append(pd.util.hash_pandas_object(v, index=False).to_numpy())
        parts.append(v.drop(columns=["view", "video_path", "status"]).drop_duplicates())
    if not parts:
        raise ValueError("The video metadata has no rows.")

    # Check for duplicates and conflicts.
    # Use hash instead of storing the complete rows to save memory.
    hashes = pd.DataFrame({"path": np.concatenate(path_hash), "row": np.concatenate(row_hash)})
    unique = hashes.drop_duplicates()
    if len(unique) < len(hashes):
        print(f"[dupes]  ignored {len(hashes) - len(unique)} exact duplicate video rows.")
    conflict = unique.path.duplicated(keep=False)
    if conflict.any():
        raise ValueError(f"{unique.loc[conflict, 'path'].nunique()} video paths appear under "
                         "conflicting ids, exam types or views.")
    # Check that each study -> patient / exam type mapping is unique.
    studies = pd.concat(parts, ignore_index=True)
    ids = studies[["study_id", "patient_id", "exam_type"]].drop_duplicates()
    bad = ids.study_id.duplicated(keep=False)
    if bad.any():
        raise ValueError(f"{ids.loc[bad, 'study_id'].nunique()} studies map to more than one patient or exam type.")
    by = ["study_id"] + (["study_date"] if "study_date" in studies else [])
    studies = attach_labels(studies.sort_values(by).drop_duplicates("study_id"), labels)

    counts = {"eligible_videos": int(len(unique)), "eligible_studies": int(len(studies)),
              "eligible_patients": int(studies.patient_id.nunique()), "eligibility_funnel": left}
    print(f"[elig]   {left['metadata_videos']} videos -> {left['allowed_exam_type']} allowed exam type "
          f"-> {left['status_ok']} status OK -> {left['labelled']} labelled study -> {left['in_years']} in year range.")
    print(f"[elig]   {counts['eligible_studies']} studies, {counts['eligible_patients']} patients eligible.")
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
        v, _ = eligible_rows(batch, cfg, label_ids)
        parts.append(v[v.study_id.isin(wanted)])
    videos = pd.concat(parts, ignore_index=True).drop_duplicates().drop(columns="status")
    missing = wanted - set(videos.study_id)
    if missing:
        raise ValueError(f"{len(missing)} studies disappeared from the video metadata between the two reads.")
    return attach_labels(videos, labels)


def attach_labels(df, labels):
    if labels is None:
        return df.assign(label=np.nan)
    return df.merge(labels, on="study_id", how="left")


def load_labels(labels, cfg):
    """Create `study id`->`label` mapping, one row per study. If the same study ID appears
    multiple times with different labels, raise an error.
    """
    lc = cfg["labels"]
    check_columns(labels.columns, [lc["study_id"], lc["value"]], "label table")
    out = pd.DataFrame(
        {
            "study_id": normalize_id(labels[lc["study_id"]]),
            "label": pd.to_numeric(labels[lc["value"]], errors="coerce"),
        }
    ).dropna()
    out = out.drop_duplicates()
    dup = out.study_id.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{out.loc[dup, 'study_id'].nunique()} studies carry conflicting labels.")
    return out


# --------------------------------------------------------------------------- #
#  Verification & Selection & Split
# --------------------------------------------------------------------------- #


def verify_video(path, chunk=32):
    """Returns `None` if the video is usable; otherwise returns the reason it is unusable.

    Every frame is decoded because the loader may sample any frame. Frames are processed
    in chunks so the entire decoded video is never held in memory at once.

    Note: `chunk=32`: balance memory and decoder overhead.
    """
    if not isinstance(path, str) or not path: # Path missing.
        return "no_path"
    if any(c.isspace() for c in path):        # Contains whitespace.
        return "whitespace_in_path"
    try:
        size = os.path.getsize(path)
    except OSError:                           # File missing.
        return "missing"
    if size == 0:                             # 0 bytes.
        return "empty"

    from decord import VideoReader, cpu

    try:
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        n = len(vr)
    except Exception:
        return "undecodable"                  # Not decodable.
    if n == 0:
        return "no_frames"                    # 0 frames.
    try:
        for start in range(0, n, chunk):
            idx = list(range(start, min(start + chunk, n)))
            if vr.get_batch(idx).shape[0] != len(idx):  # Frames are corrupted.
                return "corrupt_frames"
    except Exception:                                   # Frames are corrupted.
        return "corrupt_frames"
    return None


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

    Returns a tuple ({patient_id: study_id}, {video_path: reason or None} covering the
    verified videos, DataFrame's of the video rows of the chosen studies).
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

    fetched, paths, loaded = [], {}, 0
    verdict, chosen, pos = {}, {}, 0
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
                paths.update(videos.sort_values("video_path").groupby("study_id").video_path.agg(list))
                loaded += len(block)
            pending = {p: list(preference[p]) for p in batch}
            resolved = {}
            while pending:
                todo = [x for p in pending for x in paths[pending[p][0]] if x not in verdict]
                verdict.update(zip(todo, run(validate, todo)))
                for p in list(pending):
                    study = pending[p].pop(0)
                    # An usable study must contain at least one valid video.
                    if any(verdict[x] is None for x in paths[study]):
                        resolved[p] = study
                    elif pending[p]:
                        continue
                    else:
                        resolved[p] = None
                    del pending[p]
            for p in batch:
                if resolved[p] is not None and len(chosen) < n_patients:
                    chosen[p] = resolved[p]
            print(f"[verify] {len(chosen)}/{n_patients} patients selected, "
                  f"{len(verdict)} videos verified so far.")

    if len(chosen) < n_patients:
        raise ValueError(f"Only {len(chosen)} patients have a study with a valid video.")
    videos = pd.concat(fetched, ignore_index=True)
    return chosen, verdict, videos[videos.study_id.isin(set(chosen.values()))]


def split_patients(patients, fracs, rng):
    """Split patients by first shuffling them into a reproducible seeded order, then
    assigning an exact number of patients to train, validation, and test dataset based on
    the requested fractions.
    """
    order = rng.permutation(np.sort(np.asarray(patients)))
    n = len(order)
    n_train, n_val = round(n * fracs[0]), round(n * fracs[1])
    split = pd.Series("test", index=order)
    split.iloc[:n_train] = "train"
    split.iloc[n_train : n_train + n_val] = "val"
    return split


def build(videos, labels, cfg, validate=verify_video, workers=1, batch_size=BATCH_SIZE):
    """Build the selected cohort, its train/validation/test assignment, and the validation
    status of individual videos. An optional raw label table can also be supplied.

    Returns a dict of `DataFrames` and its eligibility counts.
    """
    lab = load_labels(labels, cfg) if labels is not None else None
    studies, counts = scan_eligible(videos, cfg, lab, batch_size)

    seed = cfg["selection"]["seed"]
    rngs = stage_rngs(seed)
    fetch = functools.partial(fetch_videos, videos, cfg, lab, batch_size=batch_size)
    chosen, verdict, sel = select_cohort(studies, cfg["selection"]["n_patients"], rngs, validate, fetch, workers)

    fracs = [cfg["split"][s] for s in SPLITS]
    split = split_patients(list(chosen), fracs, rngs["split"])

    # Videos inherit the patient's split.
    sel = sel.assign(split=sel.patient_id.map(split), reason=sel.video_path.map(verdict))
    ok = sel.reason.isna()
    kept = sel[ok].drop(columns="reason").sort_values(["split", "video_path"])
    excluded = sel[~ok][["split", "patient_id", "study_id", "view", "video_path", "reason"]]

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
            "excluded": excluded.sort_values("video_path").reset_index(drop=True),
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
              + f"  excluded {sum(e['excluded_videos'].values())}")
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
    return report


def smoke_test(out_dir):
    """Read each manifest with the EchoJEPA parser and decode its first clip."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.datasets.video_dataset import VideoDataset

    for s in SPLITS:
        path = os.path.join(out_dir, f"{s}.csv")
        expected = pd.read_csv(path, sep=" ", header=None)[0].tolist()
        ds = VideoDataset(data_paths=[path], frame_step=4)
        assert list(ds.samples) == expected, f"{s}.csv: parser read different paths."
        assert all(int(x) == 0 for x in ds.labels), f"{s}.csv: non-zero labels."
        buffer, _, _ = ds[0]
        print(f"[smoke]  {s}.csv: {len(ds)} rows parsed, first clip decoded "
              f"{tuple(np.asarray(buffer[0]).shape)}.")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Cohort definition (column mappings, rules, split).")
    p.add_argument("--metadata", required=True, help="Per-video metadata (.parquet or .csv).")
    p.add_argument("--labels", help="Study-level labels from `link_ef_labels.py`.")
    p.add_argument("--out-dir", required=True, help="Where the manifests go.")
    p.add_argument("--seed", type=int, help="Overrides `selection.seed`, and the overriding value itself is what gets saved.")
    p.add_argument("--workers", type=int, default=16, help="Processes for video verification.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="The number of metadata rows read at a time.")
    # `--dry-run`: only run metadata scanning, eligibility rules, duplicate checks, and study/patient consistency checks.
    p.add_argument("--dry-run", action="store_true", help="Report eligibility counts only; skip video verification and do not write output files.")
    p.add_argument("--overwrite", action="store_true", help="Replace manifests in a non-empty `--out-dir`.")
    p.add_argument("--no-smoke-test", action="store_true", help="Skip the final validation step that reloads the generated outputs using the EchoJEPA parser.")
    return p.parse_args()


def main():
    args = parse_args()
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
          "excluded_videos.csv, manifest_info.json")
    if not args.no_smoke_test:
        smoke_test(args.out_dir)


if __name__ == "__main__":
    main()
