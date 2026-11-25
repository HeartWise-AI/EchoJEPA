# data/make_coronary_manifest.py

"""The DeepCORO_CLIP report CSV as a long manifest for the study multi-task probe.

DeepCORO_CLIP's dataset is one row per angiogram video (`FileName`), with the study
(`StudyInstanceUID`), the patient, the split and the report's per-segment stenosis (%) repeated on
every row; columns are separated by `α`. This script keeps the columns the probe needs, renames
them to the manifest's names (`video_path`, `study_id`, `patient_id`, `split`, `structure`), adds
per-vessel findings derived from the segments, and checks that every study carries one value per
target (a study whose rows disagree loses that target). `lima_or_svg_stenosis` uses -1 for
"not applicable" and is not a target; a negative value anywhere else is treated as missing.

Example:
    python data/make_coronary_manifest.py --reports <deepcoro_reports.csv> --out-dir <manifest_dir>

Writes `coronary_videos_with_targets.csv` and `coronary_targets_info.json`. The output holds
study identifiers: keep it outside the repository.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from build_manifests import ensure_outside_repo

# The 18 segments DeepCORO_CLIP's stenosis probes predict, in its order.
SEGMENTS = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis", "pda_stenosis", "posterolateral_stenosis",
            "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis", "D1_stenosis",
            "D2_stenosis", "prox_lcx_stenosis", "mid_lcx_stenosis", "dist_lcx_stenosis", "om1_stenosis", "om2_stenosis",
            "bx_stenosis", "lvp_stenosis"]
VESSELS = {
    "rca": ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis", "pda_stenosis", "posterolateral_stenosis"],
    "lad": ["prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis", "D1_stenosis", "D2_stenosis"],
    "lcx": ["prox_lcx_stenosis", "mid_lcx_stenosis", "dist_lcx_stenosis", "om1_stenosis", "om2_stenosis",
            "bx_stenosis", "lvp_stenosis"],
}
COLUMNS = {"FileName": "video_path", "StudyInstanceUID": "study_id", "patient_id_anon": "patient_id", "Split": "split",
           "main_structure_name": "structure"}
BINARY = ["rca_ge70", "lad_ge70", "lcx_ge70", "left_main_ge50", "any_ge70"]


def derive(table):
    """The target columns from the renamed table: the 18 segments (negative = missing) and the
    per-vessel findings (worst segment >= 70 %, left main >= 50 %, any of them). `dominance_name`
    is a per-video VasoVision prediction that disagrees within a fifth of the studies, not a
    report label, so it is not a target."""
    t = pd.DataFrame(index=table.index)
    for segment in SEGMENTS:
        values = pd.to_numeric(table[segment], errors="coerce").astype(float)
        t[segment] = values.where(values >= 0)
    for vessel, segments in VESSELS.items():
        worst = t[segments].max(axis=1, skipna=True)
        known = t[segments].notna().any(axis=1)
        t[f"{vessel}_ge70"] = np.where(known, (worst >= 70).astype(float), np.nan)
    t["left_main_ge50"] = np.where(t["left_main_stenosis"].notna(), (t["left_main_stenosis"] >= 50).astype(float), np.nan)
    worst = t[SEGMENTS].max(axis=1, skipna=True)
    t["any_ge70"] = np.where(t[SEGMENTS].notna().any(axis=1), (worst >= 70).astype(float), np.nan)
    return t


def consistent_per_study(table, targets):
    """Set a target to missing for the studies whose rows disagree on it, and count them."""
    conflicts = {}
    for column in targets.columns:
        distinct = targets[column].groupby(table["study_id"]).nunique(dropna=True)
        bad = distinct.index[distinct > 1]
        if len(bad):
            conflicts[column] = int(len(bad))
            targets.loc[table["study_id"].isin(bad), column] = np.nan
    return targets, conflicts


def describe(table):
    per_study = table.drop_duplicates("study_id")
    out = {"videos": int(len(table)), "studies": int(len(per_study)), "patients": int(per_study.patient_id.nunique()),
           "videos_per_study": {q: float(v) for q, v in table.groupby("study_id").size().quantile([0.05, 0.5, 0.95]).items()},
           "targets": {}}
    for column in SEGMENTS:
        values = per_study[column].dropna()
        out["targets"][column] = {"task": "regression", "n": int(len(values)), "mean": float(values.mean()) if len(values) else None,
                                  "ge70": float((values >= 70).mean()) if len(values) else None}
    for column in BINARY:
        values = per_study[column].dropna()
        out["targets"][column] = {"task": "binary", "n": int(len(values)), "prevalence": float(values.mean()) if len(values) else None}
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", required=True, help="DeepCORO_CLIP's report CSV (alpha-separated, one row per video).")
    p.add_argument("--out-dir", required=True, help="Output folder; must be outside this repository.")
    p.add_argument("--sep", default="α")
    p.add_argument("--max-studies", type=int, help="Keep only the first N studies per split (a smoke subset).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    needed = list(COLUMNS) + SEGMENTS
    header = pd.read_csv(args.reports, sep=args.sep, engine="python", nrows=0).columns
    missing = [c for c in needed if c not in header]
    if missing:
        raise ValueError(f"The report CSV lacks {missing}.")
    table = pd.read_csv(args.reports, sep=args.sep, engine="python", usecols=needed).rename(columns=COLUMNS)
    table["split"] = table["split"].astype(str).str.strip().str.lower()
    table["study_id"] = table["study_id"].astype(str).str.strip()
    table = table.dropna(subset=["video_path"])
    if args.max_studies:
        keep = table.drop_duplicates("study_id").groupby("split").head(args.max_studies).study_id
        table = table[table.study_id.isin(set(keep))]
    targets, conflicts = consistent_per_study(table, derive(table))
    if conflicts:
        print(f"[check]  studies whose videos disagree on a target (set to missing): {conflicts}")
    manifest = pd.concat([table[["video_path", "study_id", "patient_id", "split", "structure"]], targets], axis=1)
    shared = manifest.groupby("study_id").split.nunique()
    if (shared > 1).any():
        raise ValueError(f"{int((shared > 1).sum())} studies are in more than one split.")
    manifest.to_csv(os.path.join(args.out_dir, "coronary_videos_with_targets.csv"), index=False)
    info = {"splits": {split: describe(rows) for split, rows in manifest.groupby("split")}, "conflicts": conflicts}
    with open(os.path.join(args.out_dir, "coronary_targets_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    for split, d in info["splits"].items():
        prevalence = d["targets"]["any_ge70"]["prevalence"]
        print(f"[write]  {split}: {d['videos']} videos, {d['studies']} studies, {d['patients']} patients; "
              f"any_ge70 prevalence {'n/a' if prevalence is None else f'{prevalence:.3f}'}")


if __name__ == "__main__":
    main()
