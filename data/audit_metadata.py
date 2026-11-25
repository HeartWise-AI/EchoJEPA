# /data/audit_metadata.py

"""Freeze the per-video metadata used by an experiment and audit it.

`snapshot` copies the metadata to a read-only file named by its checksum and records where
it came from, so later stages read a verifiably fixed input even when the source export is
rewritten (export file names have not reliably described their contents). `audit` writes
aggregate counts that decide eligibility and joins: the eligibility funnel, counts by year,
vendor, conversion status and view, identifier normalization and collisions, duplicate and
conflicting records, frame counts and frame rates with the padding they imply, and the
differences from an earlier snapshot and from an earlier cohort.

Example:
    python data/audit_metadata.py snapshot --source <master_metadata.parquet> --out-dir <inputs_dir>
    python data/audit_metadata.py audit --metadata <inputs_dir>/master_metadata_<sha>.parquet \\
        --previous <older_metadata.parquet> --legacy-manifests <10k_manifest_dir> \\
        --pacs <pacs_index.csv> --out-dir <audit_dir>

Outputs hold aggregate counts only (no identifiers, paths or clinical text), except that the
output directories must still be outside the repository, as for every generated data file.
"""

import argparse
import datetime
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_manifests import ensure_outside_repo, git_commit, markdown_table, normalize_id, sha256

TTE = "ECHO CARDIAQUE AVEC DOPPLER"
VIEW_COLUMNS = ("A2C", "A2C_LV", "A2C_ZOOM", "A3C", "A3C_LV", "A3C_ZOOM", "A4C", "A4C_LV", "A4C_ZOOM", "A5C",
                "A5C_ZOOM", "OTHER", "PLAX", "PLAX_DEEP", "PLAX_ZOOM", "PSAX", "PSAX_AV", "RVINF", "SUBCOSTAL",
                "SUPRASTERNAL")
# The pretraining clip: 16 frames at 8 fps, sampled like src/datasets/video_dataset.py.
CLIP_FRAMES, CLIP_FPS = 16, 8


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def file_identity(path):
    st = os.stat(path)
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


# --------------------------------------------------------------------------- #
#  Snapshot
# --------------------------------------------------------------------------- #


def snapshot(source, out_dir):
    """Copy `source` to `out_dir/<stem>_<sha256[:12]>.parquet`, read-only, and write its provenance.

    The source is hashed before and after the copy and its size/mtime compared, so a source rewritten
    during the capture is refused instead of producing a snapshot that matches neither version.
    """
    ensure_outside_repo(out_dir, "--out-dir")
    os.makedirs(out_dir, exist_ok=True)
    started = now()
    before = file_identity(source)
    digest = sha256(source)
    stem = os.path.splitext(os.path.basename(source))[0]
    dest = os.path.join(out_dir, f"{stem}_{digest[:12]}.parquet")
    if os.path.exists(dest):
        if sha256(dest) != digest:
            raise SystemExit(f"{dest} exists with different content; refusing to overwrite a snapshot.")
        print(f"[snap]   {dest} already holds this content.")
    else:
        tmp = dest + ".partial"
        shutil.copyfile(source, tmp)
        if sha256(tmp) != digest:
            os.remove(tmp)
            raise SystemExit("The copy does not match the source checksum; the source may have changed.")
        os.chmod(tmp, 0o444)
        os.replace(tmp, dest)
    after = file_identity(source)
    if after != before or sha256(source) != digest:
        raise SystemExit("The source changed during the capture; nothing recorded. Retry when it is stable.")
    pf = pq.ParquetFile(dest)
    record = {
        "source_path": os.path.abspath(source),
        "source_identity_at_capture": before,
        "snapshot_path": os.path.abspath(dest),
        "bytes": os.path.getsize(dest),
        "sha256": digest,
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "schema": [{"name": f.name, "type": str(f.type)} for f in pf.schema_arrow],
        "capture_started_utc": started,
        "capture_finished_utc": now(),
        "code_commit": git_commit(),
        "command": sys.argv,
    }
    with open(os.path.join(out_dir, f"{stem}_{digest[:12]}.provenance.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"[snap]   {record['rows']} rows, sha256 {digest}, written read-only.")
    return record


# --------------------------------------------------------------------------- #
#  Audit helpers
# --------------------------------------------------------------------------- #


def read(path, columns):
    """Columns of a parquet file with identifiers kept as strings (never parsed as numbers)."""
    table = pq.read_table(path, columns=columns)
    return table.to_pandas(types_mapper={pa.string(): pd.StringDtype()}.get)


def counts(series, top=None):
    vc = series.fillna("<missing>").astype(str).value_counts()
    vc = vc.sort_index() if top is None else vc.head(top)
    return {k: int(v) for k, v in vc.items()}


def quantiles(x, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)):
    x = pd.Series(x).dropna().astype(float)
    return {f"q{int(q * 100):02d}": (float(x.quantile(q)) if len(x) else None) for q in qs} | {"n": int(len(x))}


def frame_step(fps):
    """The pretraining loader's frame step: max(1, ceil(fps) // 8), or 1 without a usable rate."""
    fps = pd.to_numeric(fps, errors="coerce").astype(float)
    usable = np.isfinite(fps) & (fps > 0)
    return np.where(usable, np.maximum(1, np.ceil(np.where(usable, fps, 1)) // CLIP_FPS), 1)


def padding_fraction(n_frames, fps):
    """Share of a 16-frame pretraining clip that is black padding, per video (black policy).

    Mirrors the loader's short-video branch: n // step real frames (at least one), at most 16.
    """
    n = pd.to_numeric(n_frames, errors="coerce").astype(float).to_numpy()
    step = frame_step(fps)
    real = np.where(n > CLIP_FRAMES * step, CLIP_FRAMES, np.minimum(CLIP_FRAMES, np.maximum(1, n // step)))
    return pd.Series(np.where(np.isfinite(n) & (n > 0), 1 - real / CLIP_FRAMES, np.nan))


def conflicts(df, key, cols):
    """Number of `key` values carrying more than one distinct combination of `cols`."""
    d = df[[key] + cols].drop_duplicates()
    return int(d[key].duplicated(keep=False).groupby(d[key]).any().sum())


# --------------------------------------------------------------------------- #
#  Audit
# --------------------------------------------------------------------------- #


AUDIT_COLUMNS = ["mrn", "study_uid", "study_type", "avi_status", "avi_path", "dcm_path", "file_id",
                 "predicted_class", "date", "brand", "model", "NumberOfFrames", "FPS", "FPS source"]


def audit(metadata, out_dir, previous=None, legacy_manifests=None, pacs=None):
    ensure_outside_repo(out_dir, "--out-dir")
    os.makedirs(out_dir, exist_ok=True)
    before = file_identity(metadata)
    report = {"created_utc": now(), "code_commit": git_commit(),
              "input": {"path": os.path.abspath(metadata), **before, "sha256": sha256(metadata)}}

    m = read(metadata, AUDIT_COLUMNS + list(VIEW_COLUMNS))
    m["year"] = m["date"].str[:4]
    m["patient_id"] = normalize_id(m.mrn)
    m["study_id"] = normalize_id(m.study_uid)

    # -- Eligibility funnel (metadata level; video decoding is checked by the manifest build).
    tte = m[m.study_type == TTE]
    ok = tte[tte.avi_status == "success"]
    has_path = ok.avi_path.notna() & (ok.avi_path.str.len() > 0)
    usable = ok[has_path & ~ok.avi_path.str.contains(r"\s", regex=True, na=False)]
    funnel = {
        "rows": len(m), "tte_rows": len(tte), "tte_success": len(ok),
        "tte_success_with_path": int(has_path.sum()), "usable_paths_no_whitespace": len(usable),
        "unique_videos": int(usable.avi_path.nunique()), "usable_studies": int(usable.study_id.nunique()),
        "usable_patients": int(usable.patient_id.nunique()),
    }
    report["funnel"] = funnel

    # -- Counts by year, vendor/model, status, view and missingness.
    by_year = pd.DataFrame({
        "rows": m.groupby("year").size(), "tte_rows": tte.groupby("year").size(),
        "tte_success_videos": ok.groupby("year").size(), "tte_studies": tte.groupby("year").study_id.nunique(),
        "usable_studies": usable.groupby("year").study_id.nunique(),
        "usable_patients": usable.groupby("year").patient_id.nunique(),
    }).fillna(0).astype(int)
    by_year["videos_per_usable_study"] = (by_year.tte_success_videos / by_year.usable_studies.replace(0, np.nan)).round(1)
    report["by_year"] = json.loads(by_year.to_json(orient="index"))
    report["tte_status"] = counts(tte.avi_status)
    report["usable_by_vendor"] = counts(usable.brand)
    report["usable_by_vendor_model"] = counts(usable.brand.fillna("<missing>") + " | " + usable.model.fillna("<missing>"),
                                              top=40)
    report["usable_by_view"] = counts(usable.predicted_class)
    report["usable_missingness"] = {c: int(usable[c].isna().sum()) for c in
                                    ["mrn", "study_uid", "date", "brand", "model", "predicted_class",
                                     "NumberOfFrames", "FPS", "dcm_path"]}
    probs = usable[list(VIEW_COLUMNS)]
    report["view_classification"] = {
        "usable_without_predicted_class": int(usable.predicted_class.isna().sum()),
        "usable_without_any_view_probability": int(probs.isna().all(axis=1).sum()),
        "top_probability": quantiles(probs.max(axis=1)),
        "without_predicted_class_by_year": counts(usable.loc[usable.predicted_class.isna(), "year"]),
    }

    # -- Identifier normalization and collisions.
    raw_study = m.study_uid.astype("string")
    report["identifiers"] = {
        "study_uid_with_quote_characters": int(raw_study.str.contains("['\"]", regex=True, na=False).sum()),
        "study_uid_with_outer_whitespace": int((raw_study != raw_study.str.strip()).sum()),
        "study_uid_raw_distinct": int(raw_study.nunique()),
        "study_uid_normalized_distinct": int(m.study_id.nunique()),
        "mrn_distinct": int(m.patient_id.nunique()),
        "mrn_with_leading_zero": int(m.patient_id.str.startswith("0", na=False).sum()),
        "mrn_non_digit": int((~m.patient_id.str.fullmatch(r"\d+", na=True)).sum()),
        # Patients that only differ by zero padding would merge in a zero-insensitive join (stage 1 compares
        # report and PACS ids that way), so they are counted here.
        "usable_mrn_collisions_ignoring_leading_zeros": int(
            usable.patient_id.drop_duplicates().str.lstrip("0").duplicated().sum()),
        "dcm_path_unique": bool(m.dcm_path.dropna().is_unique),
        "dcm_path_missing": int(m.dcm_path.isna().sum()),
        "file_id_unique": bool(m.file_id.dropna().is_unique),
        "file_id_missing": int(m.file_id.isna().sum()),
    }

    # -- Duplicates and conflicting assignments among usable videos.
    report["conflicts"] = {
        "repeated_identical_usable_rows": int(usable.duplicated().sum()),
        "avi_path_repeated": int(usable.avi_path.duplicated().sum()),
        "avi_path_under_several_studies": conflicts(usable, "avi_path", ["study_id"]),
        "study_under_several_patients": conflicts(usable, "study_id", ["patient_id"]),
        "study_under_several_dates": conflicts(usable, "study_id", ["date"]),
        "study_under_several_exam_types": conflicts(m.dropna(subset=["study_id"]), "study_id", ["study_type"]),
    }

    # -- Frames, frame rates, durations and the padding they imply (header values).
    n = pd.to_numeric(usable.NumberOfFrames, errors="coerce")
    fps = pd.to_numeric(usable.FPS, errors="coerce")
    fps_ok = np.isfinite(fps) & (fps > 0)
    pad = padding_fraction(n, fps)
    report["timing_header"] = {
        "frames_coverage": float(n.notna().mean()), "fps_usable_coverage": float(fps_ok.mean()),
        "fps_source": counts(usable["FPS source"]),
        "n_frames": quantiles(n), "fps": quantiles(fps.where(fps_ok)),
        "duration_s": quantiles((n / fps).where(fps_ok)),
        "videos_shorter_than_one_clip": float((pad > 0).mean()),
        "padding_share_of_sampled_frames": float(pad.mean()),
        "padding_fraction": quantiles(pad),
    }

    if legacy_manifests:
        report["legacy_cohort"] = audit_legacy(usable, legacy_manifests)
    if previous:
        report["previous_snapshot"] = diff_previous(previous, metadata)
    if pacs:
        report["pacs_inventory"] = pacs_coverage(pacs, m)

    if file_identity(metadata) != before:
        raise SystemExit("The metadata changed while it was audited; rerun on a fixed snapshot.")
    with open(os.path.join(out_dir, "audit.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out_dir, "metadata_audit.md"), "w") as f:
        f.write(audit_markdown(report))
    print(f"[audit]  {funnel['unique_videos']} usable videos, {funnel['usable_studies']} studies, "
          f"{funnel['usable_patients']} patients; report in {out_dir}.")
    return report


def audit_legacy(usable, manifest_dir):
    """Whether every video of an earlier cohort is still present, identical in study and patient, and usable;
    and how header timing compares with the frame counts and rates that cohort's verification decoded."""
    legacy = pd.read_csv(os.path.join(manifest_dir, "videos.csv"), dtype={"patient_id": str, "study_id": str})
    legacy["patient_id"], legacy["study_id"] = normalize_id(legacy.patient_id), normalize_id(legacy.study_id)
    j = legacy.merge(usable[["avi_path", "study_id", "patient_id", "NumberOfFrames", "FPS"]],
                     left_on="video_path", right_on="avi_path", how="left", suffixes=("", "_now"))
    found = j.avi_path.notna()
    header_n = pd.to_numeric(j.NumberOfFrames, errors="coerce")
    header_fps = pd.to_numeric(j.FPS, errors="coerce")
    both_fps = found & j.fps.notna() & header_fps.notna()
    return {
        "videos": len(legacy), "still_usable": int(found.sum()),
        "same_study": int((found & (j.study_id == j.study_id_now)).sum()),
        "same_patient": int((found & (j.patient_id == j.patient_id_now)).sum()),
        "decoded_timing_coverage": float(j.n_frames.notna().mean()),
        "header_frames_equal_decoded": float((header_n[found] == j.n_frames[found]).mean()),
        "header_frames_minus_decoded": quantiles((header_n - j.n_frames)[found]),
        "header_fps_within_0.5_of_decoded": float((abs(header_fps - j.fps) <= 0.5)[both_fps].mean()),
        "decoded_padding_share_of_sampled_frames": float(padding_fraction(j.n_frames, j.fps).mean()),
        "header_padding_share_same_videos": float(padding_fraction(header_n[found], header_fps[found]).mean()),
    }


def diff_previous(previous, current):
    """Row additions/removals by year, and per-column changes on the rows both files share (aligned by
    dcm_path, which is unique), so a column rewritten under identical aggregate counts is still found."""
    key = "dcm_path"
    old_names = set(pq.read_schema(previous).names)
    new_names = set(pq.read_schema(current).names)
    old_k = read(previous, [key, "date"])
    new_k = read(current, [key, "date"])
    for name, df in (("previous", old_k), ("current", new_k)):
        if not df[key].dropna().is_unique:
            raise ValueError(f"{key} is not unique in the {name} metadata; rows cannot be aligned.")
    pos = new_k.reset_index().merge(old_k.reset_index(), on=key, suffixes=("_new", "_old"))
    added = new_k[~new_k[key].isin(old_k[key])]
    removed = old_k[~old_k[key].isin(new_k[key])]
    changed = {}
    for col in sorted(old_names & new_names - {key}):
        a = read(previous, [col])[col].to_numpy()[pos.index_old.to_numpy()]
        b = read(current, [col])[col].to_numpy()[pos.index_new.to_numpy()]
        a, b = pd.Series(a), pd.Series(b)
        same = (a == b).fillna(False) | (a.isna() & b.isna())
        if col in VIEW_COLUMNS or col.endswith("Value") or col in ("FPS", "postcrop_size"):
            # Floating point columns: equal within float32 resolution counts as unchanged.
            same |= np.isclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"),
                               rtol=1e-6, atol=0, equal_nan=True)
        n = int((~same).sum())
        if n:
            entry = {"changed_rows": n}
            if col in ("study_uid", "mrn", "file_id"):
                entry["equal_after_normalize_id"] = float((normalize_id(a[~same]) == normalize_id(b[~same])).mean())
            changed[col] = entry
    return {
        "previous": {"path": os.path.abspath(previous), "sha256": sha256(previous), "rows": len(old_k)},
        "common_rows": len(pos), "added_rows": len(added), "removed_rows": len(removed),
        "added_by_year": counts(added.date.str[:4]), "removed_by_year": counts(removed.date.str[:4]),
        "columns_only_previous": sorted(old_names - new_names), "columns_only_current": sorted(new_names - old_names),
        "changed_columns_on_common_rows": changed,
    }


def pacs_coverage(pacs, m):
    """Studies per year in the PACS inventory that reached the metadata: completeness of extraction, as
    opposed to completeness of view classification. The inventory mixes exam types and often has no study
    description, so all inventory studies are compared with metadata studies of any exam type, and the
    studies described as standard TTE with metadata studies of that exam type.
    """
    p = pd.read_csv(pacs, dtype=str, usecols=["StudyInstanceUID", "StudyDate", "StudyDescription"])
    p["study_id"] = normalize_id(p.StudyInstanceUID)
    p = p.dropna(subset=["study_id"]).drop_duplicates("study_id")
    p["year"] = p.StudyDate.str[:4]
    tte_desc = p.StudyDescription.str.strip() == TTE
    any_meta = set(m.study_id.dropna())
    tte_meta = set(m.loc[m.study_type == TTE, "study_id"].dropna())
    t = pd.DataFrame({
        "inventory_studies": p.groupby("year").size(),
        "in_metadata_any_exam": p[p.study_id.isin(any_meta)].groupby("year").size(),
        "inventory_without_description": p[p.StudyDescription.isna()].groupby("year").size(),
        "inventory_described_tte": p[tte_desc].groupby("year").size(),
        "described_tte_in_metadata_tte": p[tte_desc & p.study_id.isin(tte_meta)].groupby("year").size(),
    }).fillna(0).astype(int)
    t["share_any"] = (t.in_metadata_any_exam / t.inventory_studies).round(3)
    t["share_described_tte"] = (t.described_tte_in_metadata_tte / t.inventory_described_tte.replace(0, np.nan)).round(3)
    return {"path_sha256": sha256(pacs), "by_year": json.loads(t.to_json(orient="index"))}


# --------------------------------------------------------------------------- #
#  Markdown
# --------------------------------------------------------------------------- #


def table(d, name):
    """{row: {column: value}} as a Markdown table; whole-number columns are shown without decimals."""
    if not d:
        return "(none)"
    df = pd.DataFrame(d).T
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().all() and (x % 1 == 0).all():
            df[c] = x.astype("int64")
    return markdown_table(df.rename_axis(name))


def series_table(d, name, value="count"):
    return markdown_table(pd.Series(d, name=value).rename_axis(name).to_frame()) if d else "(none)"


def audit_markdown(r):
    f, ids, c, t = r["funnel"], r["identifiers"], r["conflicts"], r["timing_header"]
    lines = [
        "# Metadata audit", "",
        "Aggregate counts only: no identifiers, paths or clinical text.", "",
        f"- Created (UTC): {r['created_utc']}", f"- Code version: {r['code_commit'] or 'unknown'}",
        f"- Input sha256: {r['input']['sha256']} ({r['input']['bytes']} bytes)", "",
        "## Eligibility funnel (metadata level)", "", series_table(f, "step"), "",
        "Video decoding is verified later by the manifest build; these are upper bounds.", "",
        "## By year", "", table(r["by_year"], "year"), "",
        "## Conversion status of standard TTE rows", "", series_table(r["tte_status"], "avi_status"), "",
        "## Usable videos by vendor", "", series_table(r["usable_by_vendor"], "vendor"), "",
        "## Usable videos by vendor and model (top 40)", "", series_table(r["usable_by_vendor_model"], "vendor | model"),
        "", "## Usable videos by predicted view", "", series_table(r["usable_by_view"], "view"), "",
        "## Missing values among usable videos", "", series_table(r["usable_missingness"], "column"), "",
        "## View classification", "",
        f"- Usable videos without a predicted view: {r['view_classification']['usable_without_predicted_class']}",
        f"- Usable videos without any view probability: "
        f"{r['view_classification']['usable_without_any_view_probability']}",
        "- Top view probability: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                               for k, v in r["view_classification"]["top_probability"].items()),
        "", "## Identifiers", "", series_table(ids, "check", "value"), "",
        "Identifiers are read as strings; `normalize_id` strips surrounding whitespace and quote characters.", "",
        "## Duplicates and conflicts among usable videos", "", series_table(c, "check"), "",
        "## Frames, frame rate, duration and padding (DICOM header values)", "",
        f"- Frame-count coverage: {t['frames_coverage']:.4f}; usable frame-rate coverage: {t['fps_usable_coverage']:.4f}",
        f"- Videos shorter than one 16-frame clip at 8 fps: {t['videos_shorter_than_one_clip']:.3f}",
        f"- Black-padding share of sampled frames (baseline policy): {t['padding_share_of_sampled_frames']:.3f}", "",
        table({k: t[k] for k in ("n_frames", "fps", "duration_s", "padding_fraction")}, "quantity"), "",
        "Frame-rate sources:", "", series_table(t["fps_source"], "FPS source"),
    ]
    if "legacy_cohort" in r:
        lines += ["", "## Earlier 10k cohort against this snapshot", "", series_table(
            {k: v for k, v in r["legacy_cohort"].items() if not isinstance(v, dict)}, "check", "value"), "",
            "Header minus decoded frame count: " + ", ".join(
                f"{k} {v}" for k, v in r["legacy_cohort"]["header_frames_minus_decoded"].items())]
    if "previous_snapshot" in r:
        p = r["previous_snapshot"]
        lines += ["", "## Differences from the previous snapshot", "",
                  f"- Previous sha256: {p['previous']['sha256']} ({p['previous']['rows']} rows)",
                  f"- Common rows (by dcm_path): {p['common_rows']}; added: {p['added_rows']}; removed: {p['removed_rows']}",
                  f"- Columns only in previous: {p['columns_only_previous'] or 'none'}; "
                  f"only in current: {p['columns_only_current'] or 'none'}", "",
                  "Added rows by year:", "", series_table(p["added_by_year"], "year"), "",
                  "Columns that changed on common rows:", "", table(p["changed_columns_on_common_rows"], "column")]
    if "pacs_inventory" in r:
        lines += ["", "## Extraction completeness against the PACS inventory (TTE studies)", "",
                  table(r["pacs_inventory"]["by_year"], "year"), "",
                  "This measures which inventory studies reached the metadata (extraction), not view-classifier "
                  "completion. Most inventory rows carry no study description, so the TTE comparison covers only "
                  "the described subset."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("snapshot", help="Copy the metadata to a read-only, checksum-named file.")
    s.add_argument("--source", required=True)
    s.add_argument("--out-dir", required=True, help="Outside this repository.")
    a = sub.add_parser("audit", help="Aggregate audit of a metadata snapshot.")
    a.add_argument("--metadata", required=True, help="The snapshot to audit.")
    a.add_argument("--out-dir", required=True, help="Outside this repository.")
    a.add_argument("--previous", help="An earlier metadata file to compare against (rows aligned by dcm_path).")
    a.add_argument("--legacy-manifests", help="An earlier build_manifests.py output directory (videos.csv).")
    a.add_argument("--pacs", help="PACS inventory CSV (StudyInstanceUID, StudyDate) for extraction completeness.")
    args = p.parse_args(argv)
    if args.command == "snapshot":
        snapshot(args.source, args.out_dir)
    else:
        audit(args.metadata, args.out_dir, args.previous, args.legacy_manifests, args.pacs)


if __name__ == "__main__":
    main()
