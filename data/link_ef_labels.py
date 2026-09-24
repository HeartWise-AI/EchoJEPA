# /data/link_ef_labels.py

"""Stage 1: uses accession numbers to join the report export to the PACS index, maps
each label to a study UID, and verifies the link against video metadata using patient ID
and study date. Ambiguous or invalid mappings are dropped rather than resolved arbitrarily.
Writes one row per study with a valid EF label for use by build_manifests.py.

Example:
    python data/link_ef_labels.py --config configs/data/manifests_tte_10k_ef.yaml \\
        --reports <report_export.csv> --pacs <pacs_index.csv> \\
        --metadata <video_metadata.parquet> --out <study_labels.parquet>
"""

import argparse
import os

import pandas as pd

from build_manifests import iter_table, load_config, normalize_id


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Cohort definition (column mappings, rules, split).")
    p.add_argument("--reports", required=True, help="An exported report that includes the EF label as one of its fields.")
    p.add_argument("--pacs", required=True, help="PACS index used to link accession numbers to study instance UIDs.")
    p.add_argument("--metadata", required=True, help="Per-video metadata (.parquet or .csv).")
    p.add_argument("--out", required=True, help="Output study-level label table (.parquet or .csv).")
    p.add_argument("--report-patient-col", default="Dossier")
    p.add_argument("--report-accession-col", default="AccessionNumber")
    p.add_argument("--report-label-col", default="Visually Estimated EF")
    p.add_argument("--pacs-accession-col", default="AccessionNumber")
    p.add_argument("--pacs-study-col", default="StudyInstanceUID")
    p.add_argument("--pacs-patient-col", default="PatientID")
    p.add_argument("--pacs-date-col", default="StudyDate")
    p.add_argument("--min-id-agreement", type=float, default=0.999,
                   help="Fail if the patient IDs from the report and PACS data agree less often than this.")
    p.add_argument("--check-path-layout", action="store_true",
                   help="Also require each video path to end in `<patient>/<study>/<file>`.")
    p.add_argument("--batch-size", type=int, default=500_000, help="Parquet streaming batch size.")
    return p.parse_args()


def parse_labels(raw):
    """Converts valid EF values (number in (0, 100]) into numbers and invalid values into `NaN`.
    records why invalid values were rejected.
    Also returns how many values were dropped for each dropping reason.
    """
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


def unique_or_drop(df, key, cols, what):
    """Return one row per unique `key` and drop keys with conflicting values in
    any of `cols` rather than choosing among them.
    """
    # Sort by all columns first so tie-breaking is deterministic and does not depend on input row order.
    df = df.sort_values(list(df.columns)).drop_duplicates([key] + cols)
    dup = df[key].duplicated(keep=False)
    if dup.any():
        print(f"[drop]   {df.loc[dup, key].nunique()} {what} with conflicting {cols}.")
    return df[~dup]


def same_patient(a, b):
    """Compare patient ids from two systems without zero padding."""
    return a.str.lstrip("0") == b.str.lstrip("0")


def link(reports, pacs, videos, min_id_agreement, check_path_layout=False):
    """Build study-level labels from input tables identified by role name (one row per study).

    reports: report_patient, accession, label.
    pacs:    accession, study_id, pacs_patient, pacs_date.
    videos:  patient_id, study_id, study_date, video_path (labelled studies suffice).
    """
    # Doesn't assume the caller perfectly cleaned the data.
    reports = reports.assign(label=pd.to_numeric(reports.label, errors="coerce")).dropna(subset=["label"])
    # Report accession must be unambiguous (one accession -> one patient/label); remove all ambiguous rows.
    reports = unique_or_drop(reports, "accession", ["report_patient", "label"], "accessions")
    # PACS accession must be unambiguous (one accession -> one study/patient/date); remove all ambiguous rows.
    pacs = pacs.assign(study_id=normalize_id(pacs.study_id))
    pacs = unique_or_drop(pacs, "accession", ["study_id", "pacs_patient", "pacs_date"], "accessions")

    # First-hop join: reports and PACS.
    linked = reports.merge(pacs, on="accession", how="inner")
    print(f"[join]   {len(linked)} of {len(reports)} labelled accessions found in the PACS index.")
    # Check for patient consistency in reports and PACS.
    ok = same_patient(linked.report_patient, linked.pacs_patient)
    print(f"[verify] report patient == PACS patient      {ok.mean() * 100:.4f}% ({(~ok).sum()} dropped).")
    # ok.mean(): proportion that agrees.
    if ok.mean() < min_id_agreement:
        raise ValueError("Patient-ID agreement between the report and PACS data is below the required threshold. Check the input.")
    # Check for study-level ambiguity.
    linked = unique_or_drop(linked[ok], "study_id", ["label"], "studies")
    # Only keep videos belonging to studies for which we already found an EF.
    videos = videos[videos.study_id.isin(set(linked.study_id))]
    per_study = videos.groupby("study_id").agg(n_patients=("patient_id", "nunique"), n_dates=("study_date", "nunique"))
    # A study must belong to one patient and one date; otherwise, metadata corruption/linkage error.
    if (per_study.n_patients > 1).any() or (per_study.n_dates > 1).any():
        raise ValueError("A study ID is associated with multiple patients or study dates in the video metadata.")
    # After keeping studies that have only one patient and date, we can now pick one representative video row to retrieve.
    first = videos.drop_duplicates("study_id").set_index("study_id")

    # Second-hop join to video metadata.
    j = linked.join(first[["patient_id", "study_date"]], on="study_id", how="inner")
    print(f"[verify] {len(j)} of {len(linked)} linked studies have videos in the metadata.")
    # Check for patient/date consistency in PACS and metadata.
    checks = {
        "PACS patient == metadata patient": same_patient(j.pacs_patient, j.patient_id),
        "PACS date == metadata date": j.pacs_date.str.strip().str[:8] == j.study_date.str.strip().str[:8],
    }
    for name, ok in checks.items():
        print(f"[verify] {name:<33} {ok.mean() * 100:.4f}%")
        if not ok.all():
            raise ValueError(f"{(~ok).sum()} studies fail: {name}.")
    # For `--check-path-layout`, optionally validate the path-layout.
    if check_path_layout:
        v = videos[videos.video_path.notna()]
        parts = v.video_path.str.split("/")
        ok = (parts.str[-3] == v.patient_id) & (parts.str[-2] == v.study_id)
        print(f"[verify] path encodes patient/study          {ok.mean() * 100:.4f}%")
        if not ok.all():
            raise ValueError(f"{(~ok).sum()} videos sit at a path inconsistent with their ids.")

    return j[["study_id", "patient_id", "accession", "study_date", "label"]].sort_values("study_id").reset_index(drop=True)


def stream_videos(source, cols, study_ids, batch_size):
    """Read the video-metadata rows belonging to the specified studies, using role-based column names,
    while streaming the source instead of loading the entire table into memory.
    """
    roles = ["patient_id", "study_id", "study_date", "video_path"]
    names = [cols[r] for r in roles]
    parts = []
    # Process the data chunk-by-chunk to save memory.
    for df in iter_table(source, names, batch_size):
        df = df[names].set_axis(roles, axis=1)
        df = df.assign(study_id=normalize_id(df.study_id))
        df = df[df.study_id.isin(study_ids)]
        if len(df):
            parts.append(df.assign(patient_id=normalize_id(df.patient_id), study_date=df.study_date.astype("string")))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=roles)


def main():
    args = parse_args()

    # Read the config.
    cfg = load_config(args.config)

    # Read the clinical reports.
    # Read the label column as text so non-numeric values can be counted as invalid instead of being silently
    # converted to missing values during import.
    reports = pd.read_csv(
        args.reports, usecols=[args.report_patient_col, args.report_accession_col, args.report_label_col],
        dtype=str, low_memory=False,
    ).rename(columns={args.report_patient_col: "report_patient", args.report_accession_col: "accession",
                      args.report_label_col: "label"})
    n0 = len(reports)
    # Convert EF to numeric and removing missing/nonsensical EF values.
    label, dropped = parse_labels(reports.label)
    reports = reports.assign(label=label).dropna(subset=["label"])
    print(f"[labels] {n0} report rows -> {len(reports)} with an EF in (0, 100]; dropped "
          + ", ".join(f"{n} {why}" for why, n in dropped.items()))

    # Read PACS index.
    pacs = pd.read_csv(
        args.pacs, dtype=str,
        usecols=[args.pacs_accession_col, args.pacs_study_col, args.pacs_patient_col, args.pacs_date_col],
    ).rename(columns={args.pacs_accession_col: "accession", args.pacs_study_col: "study_id",
                      args.pacs_patient_col: "pacs_patient", args.pacs_date_col: "pacs_date"})
    # Calculate the candidate IDs before streaming so that we only need to verify videos for studies that are
    # actually linked to an accession number that has a label.
    candidate_ids = set(normalize_id(pacs.loc[pacs.accession.isin(set(reports.accession)), "study_id"]).dropna())

    # Stream video metadata and keep only those candidate studies.
    videos = stream_videos(args.metadata, cfg["columns"], candidate_ids, args.batch_size)
    print(f"[meta]   {len(videos)} videos in {videos.study_id.nunique()} candidate labelled studies.")

    # Link report -> PACS -> video metadata. And verify patient and study-date consistency.
    labels = link(reports, pacs, videos, args.min_id_agreement, args.check_path_layout)
    # Rename columns to the format expected.
    labels = labels.rename(columns={"study_id": cfg["labels"]["study_id"], "label": cfg["labels"]["value"]})

    # Save output files and print how many studies/patients/EF labels survived.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.out.endswith(".parquet"):
        labels.to_parquet(args.out, index=False)
    else:
        labels.to_csv(args.out, index=False)
    value = labels[cfg["labels"]["value"]]
    print(f"\n[out]    {args.out}: {len(labels)} studies, {labels.patient_id.nunique()} patients, "
          f"label {value.mean():.2f}+-{value.std():.2f}.")


if __name__ == "__main__":
    main()
