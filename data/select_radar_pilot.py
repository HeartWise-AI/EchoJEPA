# /data/select_radar_pilot.py

"""Choose the few report-linked TRAINING studies used to develop RADAR view captions.

The pilot is annotation development, not a performance test: it needs diverse, reviewable studies
(normal, LV dysfunction, valve disease, and optionally a right-heart pressure and a pericardial case),
each with the standard views and at least one view recorded both in grayscale and with colour flow, so
reviewers can compare what each acquisition mode shows. Only global training patients are eligible, so
pilot reports never touch validation or test patients.

Two steps, both writing only outside the repository:
    1. `candidates`: shortlist studies per category from structured report fields and write the DICOM
       paths of the shortlisted studies' videos for a header-only acquisition scan.
    2. `select`: with the acquisition index of those paths, keep shortlisted studies that have a view in both
       B-mode and colour flow, and take the first in seeded order per category.

Example:
    python data/select_radar_pilot.py candidates --metadata <snapshot.parquet> --splits <patient_splits.csv> \\
        --ef-labels <study_labels_ef.parquet> --rvsp-labels <study_labels_rvsp.parquet> \\
        --reports <report_export.csv> --out-dir <pilot_dir>
    python data/scan_acquisition.py scan --metadata <snapshot.parquet> --paths-file <pilot_dir>/scan_paths.txt ...
    python data/select_radar_pilot.py select --out-dir <pilot_dir> --acquisition-index <index.parquet>

Console output and `summary.md` are aggregate; identities and report fields stay in the out-dir files.
"""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

from build_manifests import ensure_outside_repo, git_commit, normalize_id, sha256

TTE = "ECHO CARDIAQUE AVEC DOPPLER"
# Category order is the priority order: the first three are the core pilot, the last two optional.
CATEGORIES = ("normal", "lv_dysfunction", "valve_disease", "right_heart_pressure", "pericardial_effusion")
REPORT_FIELDS = {
    "mr_grade": "Referring MR Grade", "ar_grade": "Referring AR Grade", "tr_grade": "Referring TR Grade",
    "av_mean_gradient": "AV Mean Gradient", "av_area": "AV Area (Cont Eq VTI)",
}
TEXT_FIELDS = ("Left ventricule findings", "Mitral valve findings", "Aortic valve findings", "Tricuspid valve findings",
               "Right ventricule findings", "Pericardium findings", "Inferior vena cava findings",
               "Report conclusions findings")
# Views the pilot studies must contain (any zoom/focus variant of the family counts).
REQUIRED_VIEW_FAMILIES = ("A4C", "A2C", "PLAX", "PSAX")


def view_family(view):
    return view.str.split("_").str[0]


def numeric(series):
    """Numeric report values; text such as 'Trace' or date-like strings become NaN (unknown, not zero)."""
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def categorize(s):
    """Boolean columns per category from study-level fields. Missing values never satisfy a criterion."""
    grades = s[["mr_grade", "ar_grade", "tr_grade"]]
    return pd.DataFrame({
        "normal": (s.ef >= 55) & (grades <= 1).all(axis=1) & (s.rvsp < 35) & (s.av_mean_gradient < 20),
        "lv_dysfunction": s.ef <= 35,
        "valve_disease": (s.mr_grade >= 3) | (s.ar_grade >= 3) | (s.av_mean_gradient >= 40),
        "right_heart_pressure": s.rvsp >= 50,
        "pericardial_effusion": s.pericardium_text.str.contains(r"panchement", case=False, na=False)
        & ~s.pericardium_text.str.contains(r"pas d'?\s*[ée]panchement|absence", case=False, na=False),
    }, index=s.index).fillna(False)


def candidates(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    splits = pd.read_csv(args.splits, dtype=str)
    splits["patient_id"] = normalize_id(splits.patient_id)
    train_patients = set(splits.loc[splits.split == "train", "patient_id"])

    cols = ["mrn", "study_uid", "study_type", "avi_status", "avi_path", "dcm_path", "predicted_class", "date", "brand"]
    v = pd.read_parquet(args.metadata, columns=cols)
    v = v[(v.study_type == TTE) & (v.avi_status == "success")]
    v = v.assign(patient_id=normalize_id(v.mrn), study_id=normalize_id(v.study_uid), year=v.date.str[:4])
    v = v[v.patient_id.isin(train_patients)]

    ef = pd.read_parquet(args.ef_labels)
    rvsp = pd.read_parquet(args.rvsp_labels)[["study_id", "rvsp"]]
    reports = pd.read_csv(args.reports, dtype=str, usecols=["AccessionNumber", *REPORT_FIELDS.values(), *TEXT_FIELDS])
    reports = reports.drop_duplicates("AccessionNumber", keep=False)
    s = (ef.merge(rvsp, on="study_id", how="left")
         .merge(reports, left_on="accession", right_on="AccessionNumber", how="inner"))
    for name, col in REPORT_FIELDS.items():
        s[name] = numeric(s[col])
    s["pericardium_text"] = s["Pericardium findings"]

    per_study = v.groupby("study_id").agg(
        n_videos=("avi_path", "size"), year=("year", "first"), vendor=("brand", "first"),
        families=("predicted_class", lambda x: frozenset(view_family(x.dropna()))))
    s = s.join(per_study, on="study_id", how="inner")
    has_views = s.families.map(lambda f: all(x in f for x in REQUIRED_VIEW_FAMILIES))
    eligible = (s.n_videos >= args.min_videos) & has_views & s.year.isin([str(y) for y in args.years])
    if args.vendor:
        eligible &= s.vendor == args.vendor
    s = s[eligible].sort_values("study_id").reset_index(drop=True)
    cat = categorize(s)

    shortlist, pools = [], {}
    for c in CATEGORIES:
        pool = s.loc[cat[c], "study_id"].to_numpy()
        pools[c] = int(len(pool))
        for rank, sid in enumerate(rng.permutation(pool)[: args.shortlist]):
            shortlist.append({"category": c, "rank": rank, "study_id": sid})
    shortlist = pd.DataFrame(shortlist)
    if shortlist.empty:
        raise SystemExit("No study meets the pilot criteria; relax --years/--vendor/--min-videos.")
    detail = shortlist.merge(s, on="study_id", how="left")
    keep = ["category", "rank", "study_id", "patient_id", "accession", "study_date", "year", "vendor", "n_videos", "ef",
            "rvsp", *REPORT_FIELDS, *TEXT_FIELDS]
    detail[keep].to_csv(os.path.join(args.out_dir, "shortlist.csv"), index=False)
    clips = v[v.study_id.isin(set(shortlist.study_id))]
    clips[["patient_id", "study_id", "date", "predicted_class", "avi_path", "dcm_path", "brand"]].to_csv(
        os.path.join(args.out_dir, "shortlist_clips.csv"), index=False)
    with open(os.path.join(args.out_dir, "scan_paths.txt"), "w") as f:
        f.write("\n".join(sorted(clips.dcm_path)) + "\n")

    info = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(), "seed": args.seed,
        "criteria": {"split": "train", "exam_type": TTE, "years": args.years, "vendor": args.vendor,
                     "min_videos": args.min_videos, "required_view_families": REQUIRED_VIEW_FAMILIES,
                     "categories": {
                         "normal": "EF >= 55, MR/AR/TR grade <= 1, RVSP < 35 mmHg, AV mean gradient < 20 mmHg (all present)",
                         "lv_dysfunction": "EF <= 35",
                         "valve_disease": "MR or AR grade >= 3, or AV mean gradient >= 40 mmHg",
                         "right_heart_pressure": "RVSP >= 50 mmHg",
                         "pericardial_effusion": "pericardium findings mention an effusion (not negated)",
                     }},
        "inputs": {k: {"sha256": sha256(p)} for k, p in (("metadata", args.metadata), ("splits", args.splits),
                                                         ("ef_labels", args.ef_labels),
                                                         ("rvsp_labels", args.rvsp_labels), ("reports", args.reports))},
        "eligible_studies": int(len(s)), "pool_per_category": pools,
        "shortlisted_studies": int(shortlist.study_id.nunique()), "shortlisted_clips": int(len(clips)),
    }
    with open(os.path.join(args.out_dir, "candidates.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[pilot]  {len(s)} eligible training studies; pools {pools}; "
          f"{info['shortlisted_studies']} shortlisted, {len(clips)} clips to scan.")


def select(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    shortlist = pd.read_csv(os.path.join(args.out_dir, "shortlist.csv"), dtype={"study_id": str, "patient_id": str})
    clips = pd.read_csv(os.path.join(args.out_dir, "shortlist_clips.csv"), dtype=str)
    acq = pd.read_parquet(args.acquisition_index)
    if acq.dcm_path.duplicated().any():
        raise ValueError("The acquisition index has repeated dcm_path rows.")
    clips = clips.merge(acq[["dcm_path", "acq_category", "acq_confidence"]], on="dcm_path", how="left")
    if clips.acq_category.isna().any():
        raise SystemExit(f"{clips.acq_category.isna().sum()} shortlisted clips are not in the acquisition index; "
                         "scan scan_paths.txt first.")
    clips["family"] = view_family(clips.predicted_class.fillna("OTHER"))
    confirmed = clips[clips.acq_confidence == "header"]
    modes = (confirmed.assign(bmode=confirmed.acq_category == "bmode", color=confirmed.acq_category == "color_flow")
             .groupby(["study_id", "family"])[["bmode", "color"]].any())
    both = modes.bmode & modes.color
    paired = both[both].reset_index().groupby("study_id").family.agg(lambda f: sorted(f))

    chosen, used = [], set()
    for c in CATEGORIES[: args.n_studies]:
        pool = shortlist[(shortlist.category == c) & shortlist.study_id.isin(paired.index)].sort_values("rank")
        pool = pool[~pool.patient_id.isin(used)]
        if len(pool):
            row = pool.iloc[0]
            chosen.append({"category": c, "study_id": row.study_id, "patient_id": row.patient_id,
                           "paired_view_families": paired[row.study_id]})
            used.add(row.patient_id)
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(os.path.join(args.out_dir, "pilot_studies.csv"), index=False)
    pilot_clips = clips[clips.study_id.isin(set(chosen.study_id))]
    pilot_clips.to_csv(os.path.join(args.out_dir, "pilot_clips.csv"), index=False)

    summary = {
        "acquisition_index_sha256": sha256(args.acquisition_index),
        "studies": int(len(chosen)), "categories": chosen.category.tolist(),
        "clips": int(len(pilot_clips)),
        "clips_by_acquisition": {k: int(n) for k, n in pilot_clips.acq_category.value_counts().items()},
        "clips_by_view": {k: int(n) for k, n in pilot_clips.predicted_class.fillna("<missing>").value_counts().items()},
        "paired_view_families_per_study": [len(x) for x in chosen.paired_view_families],
    }
    with open(os.path.join(args.out_dir, "selection.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = ["# RADAR pilot selection", "", "Aggregate only.", "",
             f"- Studies: {summary['studies']} ({', '.join(summary['categories'])})",
             f"- Clips: {summary['clips']}", f"- Clips by acquisition: {summary['clips_by_acquisition']}",
             f"- Clips by view: {summary['clips_by_view']}"]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[pilot]  {summary['studies']} studies ({', '.join(summary['categories'])}), {summary['clips']} clips; "
          f"by acquisition {summary['clips_by_acquisition']}.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--metadata", required=True)
    c.add_argument("--splits", required=True, help="patient_splits.csv (or a studies.csv with patient_id, split).")
    c.add_argument("--ef-labels", required=True)
    c.add_argument("--rvsp-labels", required=True)
    c.add_argument("--reports", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--seed", type=int, default=20261007)
    c.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    c.add_argument("--vendor", default="GE Vingmed Ultrasound", help="Empty string for any vendor.")
    c.add_argument("--min-videos", type=int, default=25)
    c.add_argument("--shortlist", type=int, default=4, help="Studies shortlisted per category.")
    s = sub.add_parser("select")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--acquisition-index", required=True)
    s.add_argument("--n-studies", type=int, default=5, choices=range(3, 6), help="3 core categories, up to 5.")
    args = p.parse_args(argv)
    candidates(args) if args.command == "candidates" else select(args)


if __name__ == "__main__":
    main()
