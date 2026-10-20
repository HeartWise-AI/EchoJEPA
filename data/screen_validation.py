# data/screen_validation.py

"""Build a validation clip list for the clip screen (`clip_screen.py`): a stratified, patient-aware random sample
of the archive manifest, whose sampling probabilities let `calibrate` estimate the screen's errors over the whole
population it was drawn from.

Each clip of the population falls in exactly one group, the first that applies:

    frozen_suspect    a clip of at most --frozen-max-frames frames, or at most --frozen-max-fps frames per second:
                      where frozen frames saved as videos are expected (metadata only: no clip is decoded here)
    vendor:<name>     a clip of a scanner vendor other than the most common one, one group per vendor
    low_ef            a clip of a study whose EF is at most --low-ef (reduced function, so less motion)
    general           every other clip

Every group of the population gets at least one clip (a group drawn none would drop out of the estimates
unnoticed): a --n-* of 0 for a group that has clips is refused; narrow the population (--splits, --years) instead.
--exclude-patients leaves out the patients of the clip lists the screen was tuned on (its shortlist and reference
lists, or a tuning validation list), so that the list checks the tuned screen on other patients.
Each clip records the population's size (`population_clips`), and `sheet --all-clips` checks that the groups add
up to it.

Within a group, sampling is in two stages, so that a few patients with many clips cannot fill it: a seeded random
sample of patients, then up to --clips-per-patient random clips of each. A clip's probability of being drawn is
(patients drawn / patients of the group) x (clips drawn of its patient / its patient's clips in the group), recorded
with the group and its sizes. A patient can be drawn in several groups.

The list carries the columns `clip_screen.py features` needs (video_path, dicom_path, global_key, vendor, model,
acq_category) and keeps the metadata (patient, study, view, frames, frame rate, EF). reference_clips.csv adds, for
the scanner layouts of the list, random clips of patients not drawn, from which `flag` learns where each layout
normally has ink. A sample that leaves no such clip is refused; validation.json counts the sample's layouts with
fewer reference clips than a layout baseline needs (clip_screen.MIN_LAYOUT_CLIPS), whose clips get no results-box
score (and are excluded as incomplete when the screen's measurement rules exclude), and the console warns of them.

Outputs, in --out-dir (a folder only its owner can read; patient and study identifiers):
    validation_clips.csv   the sample: sample_stratum, sample_prob, patient_prob, clip_prob and the group sizes
    reference_clips.csv    reference clips for the layouts of the sample
    validation.json        the groups (clips and patients in the population, drawn), the rules, seed and input
                           hashes; written last

Example:
    python data/screen_validation.py --videos <manifests>/videos.parquet --ef-labels <study_labels_ef.parquet> \\
        --out-dir <validation_dir> --n-frozen 50 --n-low-ef 50 --n-per-vendor 30 --n-general 100
    python data/clip_screen.py features --clips <validation_dir>/validation_clips.csv \\
        --out <validation_dir>/screen/features.parquet
    python data/clip_screen.py features --clips <validation_dir>/reference_clips.csv \\
        --out <validation_dir>/screen/reference.parquet
    python data/clip_screen.py flag --features <validation_dir>/screen/features.parquet \\
        --reference <validation_dir>/screen/reference.parquet --out <validation_dir>/screen/flags.parquet
    python data/clip_screen.py sheet --flags <validation_dir>/screen/flags.parquet --all-clips \\
        --out-dir <validation_dir>/sheet
    python data/clip_screen.py calibrate --flags <validation_dir>/screen/flags.parquet \\
        --sheet-dir <validation_dir>/sheet --labels <labels.csv>

Console output is aggregate only.
"""

import argparse
import datetime
import io
import json
import os
import sys

import numpy as np
import pandas as pd

import clip_screen
import radar_pilot
from build_manifests import ensure_outside_repo, git_commit, normalize_id

MANIFEST_COLUMNS = ("patient_id", "study_id", "study_date", "split", "view", "video_path", "dicom_path", "vendor",
                    "model", "acq_category", "n_frames", "fps", "reason")
LAYOUT_COLUMNS = radar_pilot.LAYOUT_COLUMNS  # the screen's layouts before the frame size is known.
OUTPUTS = ("validation_clips.csv", "reference_clips.csv", "validation.json")


def assign_groups(clips, frozen_max_frames, frozen_max_fps, low_ef):
    """The group of each clip (see the module docstring), the first that applies."""
    vendor = clips.vendor.fillna("<missing>").astype(str)
    main_vendor = vendor.value_counts().idxmax()
    frozen = ((clips.n_frames <= frozen_max_frames) | (clips.fps <= frozen_max_fps)).fillna(False).astype(bool)
    low = (clips.ef <= low_ef).fillna(False).astype(bool)
    group = np.where(low, "low_ef", "general").astype(object)
    group = np.where(vendor != main_vendor, ("vendor:" + vendor).to_numpy(), group)
    return pd.Series(np.where(frozen, "frozen_suspect", group), index=clips.index)


def two_stage_sample(members, n_clips, per_patient, rng):
    """About `n_clips` clips of one group: ceil(n_clips / per_patient) random patients (all of them when the group
    has fewer), then up to `per_patient` random clips of each. Returns the drawn rows with each clip's probability
    of being drawn and its two stages."""
    if n_clips < 1 or members.empty:
        return members.iloc[:0].assign(patient_prob=[], clip_prob=[], sample_prob=[], patient_clips=[])
    members = members.sort_values("video_path")  # the draw depends on the seed only, not on the table's order.
    by_patient = members.groupby("patient_id", sort=True).indices
    patients = np.array(sorted(by_patient))
    m = min(len(patients), -(-n_clips // per_patient))
    drawn = []
    for patient in rng.choice(patients, size=m, replace=False):
        rows = by_patient[patient]
        k = min(per_patient, len(rows))
        for i in rng.choice(rows, size=k, replace=False):
            drawn.append((i, m / len(patients), k / len(rows), len(rows)))
    index, patient_prob, clip_prob, patient_clips = (list(x) for x in zip(*drawn))
    out = members.iloc[index].copy()
    out["patient_prob"], out["clip_prob"], out["patient_clips"] = patient_prob, clip_prob, patient_clips
    out["sample_prob"] = out.patient_prob * out.clip_prob
    return out


def reference_sample(population, sample, per_layout, rng):
    """Up to `per_layout` random clips per layout (vendor, model, acquisition) of the sample, from patients not
    drawn in it."""
    pool = population[~population.patient_id.isin(set(sample.patient_id))].sort_values("video_path")
    wanted = set(map(tuple, sample[list(LAYOUT_COLUMNS)].astype("string").fillna("<missing>").to_numpy()))
    keys = pool[list(LAYOUT_COLUMNS)].astype("string").fillna("<missing>")
    parts = []
    for layout, rows in keys.groupby(list(LAYOUT_COLUMNS), sort=True).indices.items():
        if layout in wanted:
            parts.append(pool.iloc[rng.choice(rows, size=min(per_layout, len(rows)), replace=False)])
    return pd.concat(parts) if parts else pool.iloc[:0]


def layout_names(clips):
    """One string per clip naming its layout (vendor, model, acquisition), a missing field read '<missing>'."""
    return clips[list(LAYOUT_COLUMNS)].astype("string").fillna("<missing>").agg(" | ".join, axis=1)


def build(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    clip_screen.private_dir(args.out_dir)
    clip_screen.clear_outputs(*(os.path.join(args.out_dir, n) for n in OUTPUTS))
    if args.clips_per_patient < 1 or min(args.n_frozen, args.n_low_ef, args.n_per_vendor, args.n_general) < 0:
        raise SystemExit("--clips-per-patient must be at least 1 and every --n-* at least 0.")
    rng = np.random.default_rng(args.seed)
    # Every input as read (Inputs): the record names those, and is written only if none changed meanwhile.
    inputs = clip_screen.Inputs(__file__)

    import pyarrow.parquet as pq

    present = set(pq.read_schema(io.BytesIO(inputs.read(args.videos))).names)
    built_at = radar_pilot.manifest_build(args.videos, set(MANIFEST_COLUMNS) - {"reason", "fps"}, inputs)
    v = inputs.parquet(args.videos, columns=[c for c in MANIFEST_COLUMNS if c in present],
                       filters=[("split", "in", list(args.splits))])
    if "reason" in v:  # a manifest row with a reason is a known failure (an undecodable clip).
        v = v[v.reason.isna()].drop(columns="reason")
    if args.years:
        v = v[pd.to_numeric(v.study_date.astype("string").str[:4], errors="coerce").isin(args.years)]
    v["patient_id"], v["study_id"] = normalize_id(v.patient_id), normalize_id(v.study_id)
    excluded = set()
    for path in args.exclude_patients:  # a check on other patients than those the screen was tuned on.
        listed = inputs.csv(path, dtype=str)
        if "patient_id" not in listed:
            raise SystemExit(f"--exclude-patients {os.path.basename(path)} has no patient_id column.")
        excluded |= set(normalize_id(listed.patient_id.dropna()))
    v = v[~v.patient_id.isin(excluded)]
    ef = inputs.parquet(args.ef_labels, columns=["study_id", "ef"])
    ef["study_id"] = normalize_id(ef.study_id)
    v["ef"] = v.study_id.map(ef.drop_duplicates("study_id").set_index("study_id").ef)
    v["group"] = assign_groups(v, args.frozen_max_frames, args.frozen_max_fps, args.low_ef)

    wanted = {"frozen_suspect": args.n_frozen, "low_ef": args.n_low_ef, "general": args.n_general}
    order = sorted(v.group.unique(), key=lambda g: (g != "frozen_suspect", g == "general", g))
    n_of = {g: args.n_per_vendor if g.startswith("vendor:") else wanted[g] for g in order}
    # A group drawn no clip from would vanish from the sheet's strata, and the estimates would leave it out
    # without saying so: every group of the population gets at least one clip, or the population is narrowed.
    empty = [f"{g} ({int((v.group == g).sum())} clips)" for g in order if n_of[g] < 1]
    if empty:
        raise SystemExit(f"Every group of the population needs at least one clip, else the estimates leave it out: "
                         f"{', '.join(empty)} would get none. Raise its --n-*, or narrow the population "
                         "(--splits, --years) so that it holds no such clips.")
    parts, groups = [], {}
    for group in order:
        members = v[v.group == group]
        drawn = two_stage_sample(members, n_of[group], args.clips_per_patient, rng)
        drawn = drawn.assign(sample_stratum=group, stratum_clips=len(members),
                             stratum_patients=members.patient_id.nunique(), population_clips=len(v))
        groups[group] = {"clips": int(len(members)), "patients": int(members.patient_id.nunique()),
                         "drawn_clips": int(len(drawn)), "drawn_patients": int(drawn.patient_id.nunique())}
        parts.append(drawn)
    sample = pd.concat(parts).drop(columns="group")
    if sample.empty:
        raise SystemExit("The population is empty or every --n-* is 0: nothing to validate on.")
    reference = reference_sample(v, sample, args.reference_per_layout, rng).drop(columns="group")
    # flag learns where each layout normally has ink from the reference clips: with none, no clip of the list would get
    # a results-box score, and the list would check none of the screen's ink rules.
    if reference.empty:
        raise SystemExit("No reference clips: every patient with clips of the sample's layouts was drawn into the "
                         "sample, so none is left from which flag learns where each layout normally has ink. Draw "
                         "fewer clips (--n-*, --clips-per-patient) or widen the population (--splits, --years).")
    per_layout = layout_names(reference).value_counts()
    sample_layouts = set(layout_names(sample))
    short = sum(per_layout.get(layout, 0) < clip_screen.MIN_LAYOUT_CLIPS for layout in sample_layouts)

    secret = radar_pilot.load_secret(args.out_dir)
    sample["global_key"] = [radar_pilot.global_key(secret, p) for p in sample.video_path]
    reference = reference.assign(global_key=[radar_pilot.global_key(secret, p) for p in reference.video_path])
    outputs = [os.path.join(args.out_dir, n) for n in OUTPUTS[:2]]
    sample.to_csv(outputs[0], index=False)
    reference.to_csv(outputs[1], index=False)

    info = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "code_sha256": clip_screen.code_sha256(__file__), "seed": args.seed,
        "population": {"splits": args.splits, "years": args.years or "all", "clips": int(len(v)),
                       "patients": int(v.patient_id.nunique()),
                       "excluded_patients": {"lists": {os.path.basename(p): inputs.digest(p)
                                                       for p in args.exclude_patients},
                                             "patients": len(excluded)}},
        "rules": {"frozen_max_frames": args.frozen_max_frames, "frozen_max_fps": args.frozen_max_fps,
                  "main_vendor": v.vendor.value_counts().idxmax(), "low_ef": args.low_ef,
                  "clips_per_patient": args.clips_per_patient},
        "groups": groups, "reference_clips": int(len(reference)), "reference_per_layout": args.reference_per_layout,
        "reference_layouts": {"sample_layouts": len(sample_layouts), "min_layout_clips": clip_screen.MIN_LAYOUT_CLIPS,
                              "below_min_layout_clips": int(short)},
        "inputs": {"videos": inputs.digest(args.videos), "ef_labels": inputs.digest(args.ef_labels)},
        "manifest_build_commit": built_at,
        "outputs": radar_pilot.file_hashes(*outputs),
    }
    inputs.check(*outputs)
    with open(os.path.join(args.out_dir, "validation.json"), "w") as f:  # last: it marks a completed run.
        json.dump(info, f, indent=2)
    print("[validation]  " + "; ".join(f"{g}: {d['drawn_clips']} clips of {d['drawn_patients']} patients "
                                       f"(of {d['clips']} clips, {d['patients']} patients)" for g, d in groups.items())
          + f"; {len(reference)} reference clips.")
    if short:
        print(f"[validation]  WARNING: {short} of the sample's {len(sample_layouts)} layouts have fewer than "
              f"{clip_screen.MIN_LAYOUT_CLIPS} reference clips, too few for a layout baseline: their clips get no "
              "results-box score, so the list checks the results-box rule on none of them. Draw fewer clips of "
              "those layouts or widen the population.", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="videos.parquet of the archive manifests (build_manifests.py).")
    p.add_argument("--ef-labels", required=True, help="study_labels_ef.parquet (study_id, ef).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train"], help="manifest splits the population is drawn from.")
    p.add_argument("--years", type=int, nargs="*", default=[], help="study years (default: all).")
    p.add_argument("--n-frozen", type=int, default=50, help="clips drawn from the frozen_suspect group.")
    p.add_argument("--n-low-ef", type=int, default=50, help="clips drawn from the low_ef group.")
    p.add_argument("--n-per-vendor", type=int, default=30, help="clips drawn from each vendor:<name> group.")
    p.add_argument("--n-general", type=int, default=100, help="clips drawn from the general group.")
    p.add_argument("--clips-per-patient", type=int, default=1, help="at most this many clips per patient and group.")
    p.add_argument("--frozen-max-frames", type=int, default=10)
    p.add_argument("--frozen-max-fps", type=float, default=0.0, help="0: frame rate not used.")
    p.add_argument("--low-ef", type=float, default=35.0)
    p.add_argument("--reference-per-layout", type=int, default=200)
    p.add_argument("--exclude-patients", nargs="*", default=[],
                   help="CSVs with a patient_id column (the clip lists of the clips the screen was tuned on: "
                        "shortlist_clips.csv and reference_clips.csv, or another validation list's): their patients "
                        "are left out of the population, so the list checks the tuned screen on other patients.")
    p.add_argument("--seed", type=int, default=20261016)
    args = p.parse_args(argv)
    umask = os.umask(0o077)  # every file written is the owner's only.
    try:
        build(args)
    finally:
        os.umask(umask)


if __name__ == "__main__":
    main()
