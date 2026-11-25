# data/make_rvsp_probe_manifests.py

"""Create frozen-probe manifests for multi-view RVSP regression: one row per study.

A study's row has a fixed set of slot positions (`<slot>[<k>]`), each holding the path of one video
or the explicit token `MISS`, then the study's RVSP z-scored with the probe train split's mean and
standard deviation. A slot is a view label set plus the acquisition categories it admits (e.g. A4C
B-mode, A4C colour flow, RV inflow of any mode) and the number of videos (cine clips) it holds.
Categories are the effective ones of pretraining (`build_manifests.effective_acq_category`): a B-mode
the header does not confirm from a vendor in `acquisition.bmode_vendors` is unresolved (`unknown`), so
it never fills a B-mode slot.
Within a slot, videos are ranked by a seeded hash of the slot name and video path, so the choice is
deterministic, does not depend on row order, and is the same for every encoder probed. A video
fills at most one position: slots are filled in their listed order.

Inputs:
    --video-index   The archive's master video index (Parquet or CSV): patient_id, study_id, split,
                    view, video_path, reason (null = verified video; any other value excludes it),
                    acq_category, acq_confidence, acq_vendor_validated.
    --labels        Study labels (Parquet or CSV): study_id, rvsp (mmHg), one row per study.
    --probe-splits  CSV study_id,split: the probe's studies. Each split must agree with the index.
    --slots         Optional YAML or JSON slot specification (default: DEFAULT_SLOTS).

Writes to --out-dir, which must be outside this repository:
    train.csv, val.csv, test.csv   study_id, the slot positions, label (z-scored RVSP).
    video_index.csv                split, video_path, study_id, patient_id of every chosen video.
    probe_info.json                input checksums, slot specification, seed, normalization, counts.

A probe study is left out, and counted, when its RVSP is missing, outside [--min-rvsp, --max-rvsp]
mmHg (default 5-150, the label table's own filter), or when none of its slots can be filled. Val and
test label statistics are not computed.

Example:
    python data/make_rvsp_probe_manifests.py --video-index <videos.parquet> --labels <rvsp.parquet> \\
        --probe-splits <probe_studies.csv> --out-dir <rvsp_probe_dir>
"""

import argparse
import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd
import yaml

from build_manifests import SPLITS, effective_acq_category, ensure_outside_repo, normalize_id, read_columns, sha256

MISSING_TOKEN = "MISS"
INDEX_COLUMNS = ("patient_id", "study_id", "split", "view", "video_path", "reason", "acq_category", "acq_confidence",
                 "acq_vendor_validated")
ACQ_CATEGORIES = ("bmode", "color_flow", "spectral_doppler", "mmode", "other", "unknown")
SLOT_KEYS = ("name", "views", "acq", "clips")
NAME = re.compile(r"^[A-Za-z0-9_]+$")

# The default slots: views that carry right-heart information. `acq: any` admits every acquisition
# category, including videos whose mode is unknown. View labels must match the index's `view` values.
DEFAULT_SLOTS = (
    {"name": "A4C_bmode", "views": ["A4C"], "acq": ["bmode"], "clips": 1},
    {"name": "A4C_color", "views": ["A4C"], "acq": ["color_flow"], "clips": 1},
    {"name": "RVINF", "views": ["RVINF"], "acq": "any", "clips": 1},
    {"name": "PSAX_bmode", "views": ["PSAX"], "acq": ["bmode"], "clips": 1},
    {"name": "PSAX_AV", "views": ["PSAX_AV"], "acq": "any", "clips": 1},
    {"name": "SUBCOSTAL", "views": ["SUBCOSTAL"], "acq": "any", "clips": 1},
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-index", required=True, help="Master video index (Parquet or CSV).")
    p.add_argument("--labels", required=True, help="Study labels with `study_id` and `rvsp` (Parquet or CSV).")
    p.add_argument("--probe-splits", required=True, help="CSV of the probe's studies: study_id,split.")
    p.add_argument("--slots", default=None, help="YAML or JSON slot specification (default: DEFAULT_SLOTS).")
    p.add_argument("--seed", type=int, default=0, help="Seed of the within-slot video choice (default 0).")
    p.add_argument("--min-rvsp", type=float, default=5.0,
                   help="Smallest RVSP kept, in mmHg (default 5); smaller values are counted as invalid.")
    p.add_argument("--max-rvsp", type=float, default=150.0,
                   help="Largest RVSP kept, in mmHg (default 150); larger values are counted as invalid.")
    p.add_argument("--out-dir", required=True, help="Where the manifests go; must be outside this repository.")
    p.add_argument("--overwrite", action="store_true", help="Replace the files of a non-empty `--out-dir`.")
    return p.parse_args(argv)


def check_slots(slots):
    """Validate a slot specification and return it as a list of dicts with every key set. `acq` is
    `"any"` or a list of acquisition categories; `clips` is the number of videos the slot holds."""
    if not isinstance(slots, (list, tuple)) or not slots:
        raise ValueError("The slot specification must be a non-empty list of slots.")
    out, names = [], set()
    for slot in slots:
        if not isinstance(slot, dict):
            raise ValueError(f"A slot must be a mapping with keys {SLOT_KEYS}, not {slot!r}.")
        unknown = sorted(set(slot) - set(SLOT_KEYS))
        if unknown:
            raise ValueError(f"Unknown slot keys {unknown}; expected {SLOT_KEYS}.")
        name = slot.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            raise ValueError(f"Slot names are letters, digits and underscores, not {name!r}.")
        if name in names:
            raise ValueError(f"Slot {name!r} is listed twice.")
        names.add(name)
        views = slot.get("views")
        if isinstance(views, str):
            views = [views]
        if not views or not all(isinstance(v, str) and v for v in views):
            raise ValueError(f"Slot {name!r} needs a non-empty list of view labels.")
        acq = slot.get("acq", "any")
        if acq != "any":
            acq = [acq] if isinstance(acq, str) else acq
            bad = [a for a in (acq or []) if a not in ACQ_CATEGORIES]
            if not acq or bad:
                raise ValueError(f"Slot {name!r}: `acq` is `any` or a list of {ACQ_CATEGORIES}, "
                                 f"not {slot.get('acq')!r}.")
            acq = sorted(set(acq))
        clips = slot.get("clips", 1)
        if isinstance(clips, bool) or not isinstance(clips, int) or clips < 1:
            raise ValueError(f"Slot {name!r}: `clips` is a positive integer, not {clips!r}.")
        out.append({"name": name, "views": sorted(set(views)), "acq": acq, "clips": clips})
    return out


def load_slots(path):
    """The slot specification of a YAML or JSON file: a list of slots, or a mapping with `slots`."""
    with open(path) as f:
        spec = yaml.safe_load(f)  # JSON is YAML
    if isinstance(spec, dict):
        spec = spec.get("slots")
    return check_slots(spec)


def positions(slots):
    """The manifest's slot columns, in order: `<slot>[<k>]` for each of a slot's videos."""
    return [f"{s['name']}[{k}]" for s in slots for k in range(s["clips"])]


def rank_key(seed, slot, path):
    """The within-slot rank of a video: a hash of the seed, slot and path, so the choice depends on
    neither row order nor the other studies."""
    return hashlib.sha256(f"{seed}\0{slot}\0{path}".encode()).hexdigest()


def read_table(path, columns):
    """`columns` of a Parquet or CSV table; identifiers become stripped strings."""
    frame = read_columns(path, list(columns))
    for column in ("patient_id", "study_id"):
        if column in frame:
            frame[column] = normalize_id(frame[column])
    return frame


def check_index(index):
    """Validate the master video index: an unambiguous key, known splits and acquisition categories,
    and a `reason` that is null (verified) or says why the video is not usable."""
    empty = {c: int(index[c].isna().sum()) for c in ("patient_id", "study_id", "split", "video_path")}
    empty = {c: n for c, n in empty.items() if n}
    if empty:
        raise ValueError(f"The video index has rows without a value in {empty}.")
    unknown = sorted(set(index.split) - set(SPLITS))
    if unknown:
        raise ValueError(f"The video index has unknown splits {unknown}; expected {list(SPLITS)}.")
    if index.video_path.duplicated().any():
        raise ValueError(f"The video index lists {int(index.video_path.duplicated().sum())} videos twice.")
    blank = index.reason.notna() & (index.reason.astype("string").str.strip() == "")
    if blank.any():
        raise ValueError(f"{int(blank.sum())} videos have an empty `reason`: null marks a verified video, any "
                         "other value an excluded one.")
    acq = set(index.acq_category.dropna()) - set(ACQ_CATEGORIES)
    if acq:
        raise ValueError(f"Unknown acquisition categories {sorted(acq)}; expected {ACQ_CATEGORIES}.")
    vendor_flag(index.acq_vendor_validated)


def vendor_flag(values):
    """`acq_vendor_validated` as booleans, from a Parquet boolean column or the text of a CSV."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype("boolean")
    text = values.astype("string").str.strip().str.lower()
    flag = text.map({"true": True, "false": False}).astype("boolean")
    bad = int((text.notna() & flag.isna()).sum())
    if bad:
        raise ValueError(f"{bad} acq_vendor_validated values are not true or false.")
    return flag


def effective_acq(index):
    """The effective acquisition category of each video of the index (`effective_acq_category`)."""
    return effective_acq_category(index.assign(acq_vendor_validated=vendor_flag(index.acq_vendor_validated)))


def check_probe_splits(probe, index):
    """Every probe study must be in the index, with one patient, under the split the probe gives."""
    if probe.study_id.isna().any() or probe.study_id.duplicated().any():
        raise ValueError("The probe splits list a study twice or without an ID.")
    unknown = sorted(set(probe.split) - set(SPLITS))
    if unknown:
        raise ValueError(f"The probe splits have unknown splits {unknown}; expected {list(SPLITS)}.")
    rows = index[index.study_id.isin(probe.study_id)]
    absent = int((~probe.study_id.isin(rows.study_id)).sum())
    if absent:
        raise ValueError(f"{absent} probe studies are not in the video index.")
    per_study = rows.groupby("study_id").agg(splits=("split", "nunique"), split=("split", "first"),
                                             patients=("patient_id", "nunique"), patient_id=("patient_id", "first"))
    if (per_study.splits > 1).any():
        raise ValueError(f"{int((per_study.splits > 1).sum())} probe studies have videos in several splits of "
                         "the index.")
    if (per_study.patients > 1).any():
        raise ValueError(f"{int((per_study.patients > 1).sum())} probe studies have videos of several patients.")
    expected = probe.set_index("study_id").split
    disagree = int((per_study.split != expected.reindex(per_study.index)).sum())
    if disagree:
        raise ValueError(f"{disagree} probe studies are assigned another split than the video index gives them.")
    return per_study.patient_id


def study_labels(labels, studies, min_rvsp, max_rvsp):
    """The RVSP of each study in `studies` (NaN when it has none), and which are valid: finite and in
    [min_rvsp, max_rvsp]. The table must hold one row per study and numeric values."""
    if labels.study_id.isna().any() or labels.study_id.duplicated().any():
        raise ValueError("The label table must hold one row per study: it lists a study twice or without an ID.")
    values = pd.to_numeric(labels.rvsp, errors="coerce")
    unreadable = int((values.isna() & labels.rvsp.notna()).sum())
    if unreadable:
        raise ValueError(f"{unreadable} RVSP values are not numbers.")
    rvsp = pd.Series(values.to_numpy(dtype=float), index=labels.study_id).reindex(studies)
    valid = np.isfinite(rvsp) & (rvsp >= min_rvsp) & (rvsp <= max_rvsp)
    return rvsp, valid


def fill_slots(videos, slots, seed):
    """Choose each study's videos for each slot. `videos` are the verified videos of the probe's
    studies, with their effective acquisition category (`acq_effective`). Returns {position: Series of
    video paths indexed by study}."""
    videos = videos.reset_index(drop=True)
    used = pd.Series(False, index=videos.index)
    chosen = {}
    for slot in slots:
        take = videos.view.isin(slot["views"]) & ~used
        if slot["acq"] != "any":
            take &= videos.acq_effective.isin(slot["acq"])
        candidates = videos[take].copy()
        candidates["key"] = [rank_key(seed, slot["name"], p) for p in candidates.video_path]
        candidates = candidates.sort_values(["study_id", "key", "video_path"])
        candidates["rank"] = candidates.groupby("study_id").cumcount()
        candidates = candidates[candidates["rank"] < slot["clips"]]
        used.loc[candidates.index] = True
        for k in range(slot["clips"]):
            picked = candidates[candidates["rank"] == k]
            chosen[f"{slot['name']}[{k}]"] = pd.Series(picked.video_path.to_numpy(), index=picked.study_id)
    return chosen


def make(index, labels, probe, slots, seed=0, min_rvsp=5.0, max_rvsp=150.0):
    """Build the per-study manifests.

    Returns ({split: DataFrame[study_id, patient_id, rvsp, <positions>..., label]}, mean, std,
    counts) where `counts` holds, per split, the probe studies listed, those left out (missing or
    invalid RVSP, no slot filled) and the unverified videos of the listed studies.
    """
    slots = check_slots(slots)
    check_index(index)
    patient = check_probe_splits(probe, index)
    index = index.assign(acq_effective=effective_acq(index))
    verified = index[index.reason.isna()]
    known_views = set(verified.view.dropna())
    absent = sorted({v for s in slots for v in s["views"]} - known_views)
    if absent:
        raise ValueError(f"No verified video has the view labels {absent} (a typo?); the index's views are "
                         f"{sorted(known_views)}.")
    if verified.video_path.str.contains(r"[\s,\"]", regex=True).any():
        raise ValueError("Video paths with whitespace, commas or quotes cannot be written to a manifest.")
    if (verified.video_path == MISSING_TOKEN).any():
        raise ValueError(f"A video path equals the missing-video token {MISSING_TOKEN!r}.")

    split = probe.set_index("study_id").split
    if not 0 < min_rvsp <= max_rvsp:
        raise ValueError(f"The RVSP range [{min_rvsp}, {max_rvsp}] mmHg is empty or not positive.")
    rvsp, valid = study_labels(labels, split.index, min_rvsp, max_rvsp)
    in_probe = index.study_id.isin(split.index)
    counts = {s: {"listed": int((split == s).sum()),
                  "missing_label": int((rvsp.isna() & (split == s)).sum()),
                  "invalid_label": int((rvsp.notna() & ~valid & (split == s)).sum()),
                  "unverified_videos": int((in_probe & index.reason.notna() & (index.split == s)).sum())}
              for s in SPLITS}

    labelled = split.index[valid.to_numpy()]
    chosen = fill_slots(verified[verified.study_id.isin(labelled)], slots, seed)
    cols = positions(slots)
    rows = pd.DataFrame({"study_id": labelled, "split": split.reindex(labelled).to_numpy(),
                         "patient_id": patient.reindex(labelled).to_numpy(),
                         "rvsp": rvsp.reindex(labelled).to_numpy()})
    for col in cols:
        rows[col] = chosen[col].reindex(rows.study_id).fillna(MISSING_TOKEN).to_numpy()
    filled = (rows[cols] != MISSING_TOKEN).any(axis=1)
    for s in SPLITS:
        counts[s]["no_slot_filled"] = int((~filled & (rows.split == s)).sum())
    rows = rows[filled]
    # A patient in more than one split would leak between training and evaluation.
    shared = int((rows.groupby("patient_id").split.nunique() > 1).sum())
    if shared:
        raise ValueError(f"{shared} patients have probe studies in more than one split.")

    train = rows[rows.split == "train"]
    if train.empty:
        raise ValueError("No probe training study has an RVSP and a filled slot.")
    # Population std (ddof=0), as for the EF probe.
    mean, std = float(train.rvsp.mean()), float(train.rvsp.std(ddof=0))
    if not (math.isfinite(std) and std > 0):
        raise ValueError(f"The training RVSP values do not vary (std {std}).")
    splits = {}
    for s in SPLITS:
        part = rows[rows.split == s].sort_values("study_id").reset_index(drop=True)
        splits[s] = part.assign(label=(part.rvsp - mean) / std).drop(columns="split")
    return splits, mean, std, counts


def presence(rows, slots):
    """Per position and per slot (any of its positions), how many studies have a video or not."""
    out = {"positions": {}, "slots": {}}
    for slot in slots:
        cols = [f"{slot['name']}[{k}]" for k in range(slot["clips"])]
        for col in cols:
            present = int((rows[col] != MISSING_TOKEN).sum())
            out["positions"][col] = {"present": present, "missing": len(rows) - present}
        present = int((rows[cols] != MISSING_TOKEN).any(axis=1).sum())
        out["slots"][slot["name"]] = {"present": present, "missing": len(rows) - present}
    return out


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
        raise SystemExit(f"--out-dir {args.out_dir} is not empty; pass --overwrite to replace its files.")

    slots = load_slots(args.slots) if args.slots else check_slots(DEFAULT_SLOTS)
    index = read_table(args.video_index, INDEX_COLUMNS)
    labels = read_table(args.labels, ("study_id", "rvsp"))
    probe = pd.read_csv(args.probe_splits, dtype=str)
    if not {"study_id", "split"} <= set(probe.columns):
        raise ValueError("The probe splits need the columns study_id and split.")
    probe = probe.assign(study_id=normalize_id(probe.study_id))[["study_id", "split"]]
    splits, mean, std, counts = make(index, labels, probe, slots, args.seed, args.min_rvsp, args.max_rvsp)

    os.makedirs(args.out_dir, exist_ok=True)
    cols = positions(slots)
    # The effective acquisition mode of every chosen video, so the probe's inputs can be reported exactly
    # (e.g. whether any spectral Doppler video entered through an `acq: any` slot).
    acquisition = pd.Series(effective_acq(index).to_numpy(), index=index.video_path)
    info = {
        "source": {"video_index": sha256(args.video_index) if os.path.isfile(args.video_index) else None,
                   "labels": sha256(args.labels) if os.path.isfile(args.labels) else None,
                   "probe_splits": sha256(args.probe_splits)},
        "seed": args.seed,
        "slots": slots,
        "positions": cols,
        "missing_token": MISSING_TOKEN,
        "label": {"column": "rvsp", "unit": "mmHg", "valid": f"finite, in [{args.min_rvsp:g}, {args.max_rvsp:g}]"},
        "target_mean": mean,  # from the probe train split only.
        "target_std": std,
        "counts": counts,
        "splits": {},
        "outputs": {},
    }
    for split, rows in splits.items():
        name = f"{split}.csv"
        path = os.path.join(args.out_dir, name)
        rows[["study_id", *cols, "label"]].to_csv(path, index=False)
        chosen = rows[cols].to_numpy().ravel()
        modes = acquisition.reindex(chosen[chosen != MISSING_TOKEN]).value_counts()
        info["splits"][split] = {"studies": len(rows), "patients": int(rows.patient_id.nunique()),
                                 "videos": int((chosen != MISSING_TOKEN).sum()), **presence(rows, slots),
                                 "acquisition": {str(k): int(v) for k, v in sorted(modes.items())}}
        info["outputs"][name] = {"rows": len(rows), "sha256": sha256(path)}
        left_out = counts[split]
        print(f"[rvsp]  {split:5s} {len(rows):6d} studies ({info['splits'][split]['videos']} videos) of "
              f"{left_out['listed']} listed; left out: {left_out['missing_label']} without RVSP, "
              f"{left_out['invalid_label']} with an invalid RVSP, {left_out['no_slot_filled']} without a filled slot.")
    index_path = os.path.join(args.out_dir, "video_index.csv")
    chosen = pd.concat([rows.assign(split=split).melt(id_vars=["split", "study_id", "patient_id"], value_vars=cols,
                                                      value_name="video_path")
                        for split, rows in splits.items()])
    chosen = chosen[chosen.video_path != MISSING_TOKEN].sort_values(["split", "study_id", "video_path"])
    chosen[["split", "video_path", "study_id", "patient_id"]].to_csv(index_path, index=False)
    info["outputs"]["video_index.csv"] = {"rows": len(chosen), "sha256": sha256(index_path)}
    with open(os.path.join(args.out_dir, "probe_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[rvsp]  slots {[s['name'] for s in slots]}; train RVSP mean {mean:.4f}, std {std:.4f} mmHg "
          "(data.target_mean / data.target_std of the eval config).")


if __name__ == "__main__":
    main()
