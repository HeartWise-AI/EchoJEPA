# data/still_split_audit.py

"""Is every patient of the still table in one split, by every way of telling patients apart? Counts only.

The manifest builder places a patient by its number in the video metadata. This audit checks the finished table
against sources the builder did not use, and against the other tables a run reads:

- `patient_hash_in_two_splits`: the builder's own key (must be 0 by construction);
- `metadata_and_pacs_disagree`: studies whose patient number in the video metadata is not the PACS one;
- `pacs_patient_in_two_splits`: patients by the PACS number;
- `same_person_two_numbers`: groups with one name and birth date under two patient numbers (a re-registered
  patient), and how many of them lie in two splits (`..._in_two_splits`). Names and birth dates are compared in
  memory and never written;
- `differs_from_patient_splits`: patients whose split is not the archive-wide patient split's;
  `absent_from_patient_splits`: patients that table does not hold (drawn from the hash: to be added to it);
- `clip_study_in_another_split`, `clip_patient_in_two_splits`: against each clip table (`--clips`), by study and
  by patient.

A count that must be 0 and is not makes the exit status 1 (`violations` in the report). The report holds no
identifier; it must be outside the repository.

Usage:
    python data/still_split_audit.py --stills <stills.csv> --metadata <master_metadata.parquet> --pacs <pacs.csv> \\
        --patient-splits <patient_splits.csv> --clips <clips_train.csv> <clips_val.csv> --out <report.json>
"""

import argparse
import json
import os
import sys

import pandas as pd
from build_manifests import ensure_outside_repo, normalize_id
from make_study_targets import patient_key

MUST_BE_ZERO = ("patient_hash_in_two_splits", "pacs_patient_in_two_splits", "same_person_two_numbers_in_two_splits",
                "differs_from_patient_splits", "clip_study_in_another_split", "clip_patient_in_two_splits")


def in_two(table, key):
    """Number of `key` values of `table` with more than one split."""
    table = table.dropna(subset=[key])
    return int((table.groupby(key).split.nunique() > 1).sum())


def person_key(names, birthdates):
    """Letters of the name with the birth date; missing when either is."""
    name = names.astype("string").str.upper().str.replace(r"[^A-Z]", "", regex=True)
    born = birthdates.astype("string").str.slice(0, 10)
    return (name + "|" + born).where(name.str.len().gt(0) & born.str.len().gt(0))


def audit(stills, metadata, pacs=None, patient_splits=None, clip_tables=()):
    """The report (see the module docstring). `stills`: study_id, patient_hash, split; `metadata`: study_uid, mrn,
    birthdate; `pacs`: StudyInstanceUID, PatientID, PatientName; `patient_splits`: patient_id, split; `clip_tables`:
    tables with study_id and split."""
    studies = stills.assign(study_id=normalize_id(stills.study_id))[["study_id", "patient_hash", "split"]]
    studies = studies.drop_duplicates()
    report = {"studies": int(studies.study_id.nunique()), "study_in_two_splits": in_two(studies, "study_id"),
              "patient_hash_in_two_splits": in_two(studies, "patient_hash")}
    meta = metadata.assign(study_id=normalize_id(metadata.study_uid), patient=patient_key(metadata.mrn))
    meta = meta.dropna(subset=["patient"]).drop_duplicates("study_id")
    own = studies.drop_duplicates("study_id").merge(meta[["study_id", "patient", "birthdate"]], on="study_id",
                                                    how="left")
    report["studies_without_patient_number"] = int(own.patient.isna().sum())
    if pacs is not None:
        p = pacs.assign(study_id=normalize_id(pacs.StudyInstanceUID), pacs_patient=patient_key(pacs.PatientID))
        own = own.merge(p.drop_duplicates("study_id")[["study_id", "pacs_patient", "PatientName"]], on="study_id",
                        how="left")
        both = own.dropna(subset=["patient", "pacs_patient"])
        report.update({"studies_in_pacs": int(own.pacs_patient.notna().sum()),
                       "metadata_and_pacs_disagree": int((both.patient != both.pacs_patient).sum()),
                       "pacs_patient_in_two_splits": in_two(own, "pacs_patient")})
        person = own.assign(person=person_key(own.PatientName, own.birthdate)).dropna(subset=["person", "patient"])
        groups = person.groupby("person").agg(numbers=("patient", "nunique"), splits=("split", "nunique"))
        report.update({"same_person_two_numbers": int((groups.numbers > 1).sum()),
                       "same_person_two_numbers_in_two_splits": int(((groups.numbers > 1) & (groups.splits > 1)).sum())})
    patients = own.dropna(subset=["patient"]).drop_duplicates("patient")
    if patient_splits is not None:
        table = patient_splits.assign(patient=patient_key(patient_splits.patient_id)).dropna(subset=["patient"])
        theirs = patients.patient.map(dict(zip(table.patient, table.split.astype(str))))
        report.update({"absent_from_patient_splits": int(theirs.isna().sum()),
                       "differs_from_patient_splits": int((theirs.notna() & (theirs != patients.split)).sum())})
    by_study = dict(zip(own.study_id, own.split))
    clip_study, clip_patient = 0, 0
    for clips in clip_tables:
        c = clips.assign(study_id=normalize_id(clips.study_id))[["study_id", "split"]].drop_duplicates()
        clip_study += int((c.study_id.map(by_study).notna() & (c.study_id.map(by_study) != c.split.astype(str))).sum())
        c = c.merge(meta[["study_id", "patient"]], on="study_id", how="left").dropna(subset=["patient"])
        joined = pd.concat([c[["patient", "split"]].astype(str), patients[["patient", "split"]].astype(str)])
        clip_patient += in_two(joined, "patient")
    if clip_tables:
        report.update({"clip_study_in_another_split": clip_study, "clip_patient_in_two_splits": clip_patient})
    report["violations"] = sorted(k for k in MUST_BE_ZERO if report.get(k, 0))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stills", required=True)
    p.add_argument("--metadata", required=True, help="video metadata (study_uid, mrn, birthdate)")
    p.add_argument("--pacs", help="PACS index (StudyInstanceUID, PatientID, PatientName)")
    p.add_argument("--patient-splits", help="archive-wide patient split (patient_id, split)")
    p.add_argument("--clips", nargs="*", default=[], help="clip tables of the runs (study_id, split)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    ensure_outside_repo(os.path.dirname(os.path.abspath(args.out)), "--out")
    stills = pd.read_csv(args.stills, dtype=str, keep_default_na=False, usecols=["study_id", "patient_hash", "split"])
    metadata = pd.read_parquet(args.metadata, columns=["study_uid", "mrn", "birthdate"])
    pacs = (pd.read_csv(args.pacs, dtype=str, usecols=["StudyInstanceUID", "PatientID", "PatientName"])
            if args.pacs else None)
    patient_splits = pd.read_csv(args.patient_splits, dtype=str) if args.patient_splits else None
    clips = [pd.read_csv(path, dtype=str, usecols=["study_id", "split"]) for path in args.clips]
    report = audit(stills, metadata, pacs, patient_splits, clips)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["violations"] else 0)


if __name__ == "__main__":
    main()
