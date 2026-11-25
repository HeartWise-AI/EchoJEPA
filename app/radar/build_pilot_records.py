# app/radar/build_pilot_records.py

"""Build the RADAR pilot's local clip records from the studies `data/select_radar_pilot.py` chose.

One record per clip of each pilot study: a pseudonymous record id and study/patient keys (HMAC under a
secret kept outside the repository), the patient's split in the shared archive patient split (with that
table's SHA-256 as `split_source`), view and its classifier probability, the acquisition index's verdict
under the pretraining rule (`build_manifests.effective_acq_category`: a header B-mode from a vendor outside
`acquisition.bmode_vendors` is unresolved, `unknown`), zoom from the view label, the study's report findings
(structured columns and French free text, with provenance and never the text itself), and an
unvalidated rule-based candidate caption. Nothing here is clinically approved; approved captions only
come from returned review data (`tools/labelbox/radar_review.py import-export`).

Example:
    RADAR_RECORD_SECRET=<secret> python -m app.radar.build_pilot_records --pilot-dir <pilot_dir> \\
        --metadata <snapshot.parquet> --acquisition-index <index.parquet> --reports <report_export.csv> \\
        --patient-splits <archive>/splits/patient_splits.csv --out <pilot_dir>/records.jsonl

Prints aggregate counts only. The records hold local video paths: keep them where the data is.
"""

import argparse
import collections
import json
import os

import pandas as pd
import yaml

from app.radar.captions import rule_based_caption
from app.radar.findings import extract_findings
from app.radar.ontology import DEFAULT_ONTOLOGY_PATH, load_ontology
from app.radar.records import (
    Acquisition,
    ClipRecord,
    Focus,
    build_manifests,
    ensure_outside_repo,
    make_record_id,
    validate_record,
    write_records,
)

VIEW_COLUMNS = ("A2C", "A2C_LV", "A2C_ZOOM", "A3C", "A3C_LV", "A3C_ZOOM", "A4C", "A4C_LV", "A4C_ZOOM", "A5C",
                "A5C_ZOOM", "OTHER", "PLAX", "PLAX_DEEP", "PLAX_ZOOM", "PSAX", "PSAX_AV", "RVINF", "SUBCOSTAL",
                "SUPRASTERNAL")
ACQ_FIELDS = ("acq_category", "acq_confidence", "acq_rule", "has_tissue_2d", "has_color_flow", "has_pw_doppler",
              "has_cw_doppler", "has_mmode")
DEFAULT_MANIFEST_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                       "configs", "data", "manifests_tte_archive.yaml")


def optional_bool(x):
    return None if pd.isna(x) else bool(x)


def acquisition(a, vendor, bmode_vendors):
    """The record's acquisition: the index's verdict, except that a B-mode pretraining would not admit (header not
    confirming it, or a vendor outside `bmode_vendors`) is unresolved, and its "no colour" flag is not trusted."""
    bm = build_manifests()
    row = pd.DataFrame({"acq_category": [a.acq_category], "acq_confidence": [a.acq_confidence],
                        "acq_vendor_validated": [bmode_vendors is None or vendor in bmode_vendors]})
    flags = {f: optional_bool(a[f]) for f in ACQ_FIELDS[3:]}
    if bm.effective_acq_category(row).iloc[0] == a.acq_category:
        return Acquisition(category=a.acq_category, confidence=a.acq_confidence, rule=a.acq_rule, **flags)
    rule = "bmode_vendor_unvalidated" if a.acq_confidence == "header" else "bmode_unconfirmed"
    return Acquisition(category="unknown", confidence="unknown", rule=rule, **{**flags, "has_color_flow": None})


def patient_split_table(path):
    """The archive's patient -> split mapping and the table's SHA-256 (`split_source`)."""
    bm = build_manifests()
    table = pd.read_csv(path, dtype=str, usecols=["patient_id", "split"])
    table["patient_id"] = bm.normalize_id(table.patient_id)
    if table.patient_id.isna().any() or table.patient_id.duplicated().any():
        raise ValueError("The patient split table lists a patient twice or without an ID.")
    return table.set_index("patient_id").split, bm.sha256(path)


def build(pilot_dir, metadata, acquisition_index, reports, secret, ontology, language="fr", *, patient_splits,
          manifest_config=DEFAULT_MANIFEST_CONFIG):
    studies = pd.read_csv(os.path.join(pilot_dir, "pilot_studies.csv"), dtype=str)
    shortlist = pd.read_csv(os.path.join(pilot_dir, "shortlist.csv"), dtype=str)
    clips = pd.read_csv(os.path.join(pilot_dir, "pilot_clips.csv"), dtype=str)
    accession = shortlist.drop_duplicates("study_id").set_index("study_id").accession
    wanted = set(accession[studies.study_id])

    report_rows = pd.read_csv(reports, dtype=str, keep_default_na=False)
    report_rows = report_rows[report_rows.AccessionNumber.isin(wanted)]
    if report_rows.AccessionNumber.duplicated().any() or len(report_rows) != len(wanted):
        raise ValueError("Each pilot study needs exactly one report row.")
    report_rows = report_rows.set_index("AccessionNumber")

    with open(manifest_config) as f:
        cfg = yaml.safe_load(f)
    vendor_column = cfg["columns"]["vendor"]
    bmode_vendors = (cfg.get("acquisition") or {}).get("bmode_vendors")
    split_of, split_source = patient_split_table(patient_splits)
    patients = build_manifests().normalize_id(clips.patient_id)
    absent = int((~patients.isin(split_of.index)).sum())
    if absent:
        raise ValueError(f"{absent} pilot clips belong to patients absent from the patient split table.")

    meta = pd.read_parquet(metadata, columns=["avi_path", vendor_column, *VIEW_COLUMNS])
    meta = meta[meta.avi_path.isin(set(clips.avi_path))].set_index("avi_path")
    acq = pd.read_parquet(acquisition_index)
    acq = acq[acq.dcm_path.isin(set(clips.dcm_path))].set_index("dcm_path")
    if acq.index.duplicated().any():
        raise ValueError("The acquisition index repeats a dcm_path.")

    records, issues = [], collections.Counter()
    for study_id, study_clips in clips.groupby("study_id", sort=True):
        findings, found_issues = extract_findings(report_rows.loc[accession[study_id]].to_dict(), ontology)
        issues.update(i.reason for i in found_issues)
        patient_id = study_clips.patient_id.iloc[0]
        split = split_of[build_manifests().normalize_id(pd.Series([patient_id])).iloc[0]]
        for clip in study_clips.sort_values("avi_path").itertuples():
            a = acq.loc[clip.dcm_path]
            view = clip.predicted_class if isinstance(clip.predicted_class, str) else "OTHER"
            record = ClipRecord(
                record_id=make_record_id(secret, f"video:{clip.avi_path}"),
                study_key=make_record_id(secret, f"study:{study_id}"),
                patient_key=make_record_id(secret, f"patient:{patient_id}"),
                split=split,
                split_source=split_source,
                video_path=clip.avi_path,
                view=view,
                view_confidence=float(meta.loc[clip.avi_path, list(VIEW_COLUMNS)].max()),
                acquisition=acquisition(a, meta.loc[clip.avi_path, vendor_column], bmode_vendors),
                focus=Focus(zoomed=view.endswith("_ZOOM"), source="view_label"),
                report_findings=list(findings),
            )
            record.candidate_caption = rule_based_caption(record, ontology, language)
            records.append(validate_record(record, ontology))
    return records, issues


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot-dir", required=True, help="Output directory of data/select_radar_pilot.py.")
    p.add_argument("--metadata", required=True)
    p.add_argument("--acquisition-index", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--patient-splits", required=True,
                   help="The archive's patient_splits.csv (data/build_manifests.py): each record takes its patient's "
                        "split, and the table's SHA-256 as split_source.")
    p.add_argument("--manifest-config", default=DEFAULT_MANIFEST_CONFIG,
                   help="Archive manifest config whose acquisition.bmode_vendors and columns.vendor apply.")
    p.add_argument("--out", required=True, help="records JSONL, outside the repository")
    p.add_argument("--ontology", default=DEFAULT_ONTOLOGY_PATH)
    p.add_argument("--language", default="fr", choices=("fr", "en"))
    args = p.parse_args(argv)
    ensure_outside_repo(args.out, "--out")
    secret = os.environ.get("RADAR_RECORD_SECRET", "")
    if not secret:
        raise SystemExit("Set RADAR_RECORD_SECRET (kept outside the repository) to derive pseudonymous ids.")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; records are built once (write a new file for a new version).")
    ontology = load_ontology(args.ontology)
    records, issues = build(args.pilot_dir, args.metadata, args.acquisition_index, args.reports, secret, ontology,
                            args.language, patient_splits=args.patient_splits, manifest_config=args.manifest_config)
    write_records(args.out, records, ontology)
    summary = {
        "records": len(records), "studies": len({r.study_key for r in records}),
        "by_split": dict(collections.Counter(r.split for r in records)), "split_source": records[0].split_source,
        "ontology_version": ontology.version,
        "by_acquisition": dict(collections.Counter(r.acquisition.category for r in records)),
        "findings_per_study": sorted({r.study_key: len(r.report_findings) for r in records}.values()),
        "studies_per_concept": dict(collections.Counter(
            cid for _, cid in {(r.study_key, f.concept_id) for r in records for f in r.report_findings})),
        "claims_per_caption": dict(collections.Counter(len(r.candidate_caption.claims) for r in records)),
        "gaps_per_caption": dict(collections.Counter(len(r.candidate_caption.gaps) for r in records)),
        "extraction_issues": dict(issues),
    }
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
