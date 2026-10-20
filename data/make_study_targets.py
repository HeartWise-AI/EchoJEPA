# data/make_study_targets.py

"""Study-level targets for the study multi-task probe (`evals/study_multitask_frozen`) from the
cleaned DeepECHO report export.

Each report row (one TTE study) becomes one row of targets: raw measurements for regression
heads, 0/1 columns for binary heads and class indices for classification heads, derived with
the rules of `TARGETS` (PanEcho's label definitions where one exists, the ASE cutoffs the
MHI - PanEcho mapping uses otherwise). A value the rule cannot decide stays empty, and the
probe masks it. Reports are linked to studies through the PACS index (accession number ->
StudyInstanceUID), as `link_ef_labels.py` does; with `--videos`, the targets are joined onto
the issue #6 `videos.csv` (one row per video, `study_id` = StudyInstanceUID) to give the long
manifest the probe reads.

Example:
    python data/make_study_targets.py \\
        --reports /media/data1/ravram/DeepECHO/TTE_reports_concat/df_report_2006_2026_clean.parquet \\
        --pacs /media/data1/datasets/echo/database_tte_2009_2024.csv \\
        --videos <probe_manifests>/videos.csv --out-dir <targets_dir>

Writes `study_targets.csv` (one row per linked study), `videos_with_targets.csv` (with
`--videos`) and `targets_info.json` (every target's task, rule, units and distribution per
split). The outputs hold study identifiers: keep them outside the repository.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from build_manifests import ensure_outside_repo, normalize_id

# PanEcho class orders, kept so a head's classes compare directly with PanEcho's outputs.
THREE_WAY = {"mild": 0, "modsev": 1, "normal": 2}            # LVSize, RVSize, LASize, LVSystolicFunction
DIASTOLIC = {"mild_indeterminate": 0, "modsev": 1, "normal": 2}
REGURG = {"mild": 0, "modsev": 1, "none_trace": 2}            # MV/TV/AV regurgitation
AV_STENOSIS = {"mild_moderate": 0, "none": 1, "severe": 2}


def grade(series):
    """A referring regurgitation grade as a number in [0, 4]: `0.5` and `Trace` mean trace,
    severe words mean 4, date artefacts and anything else outside [0, 4] are missing."""
    def one(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        if "trace" in text or text.startswith("<"):
            return 0.5
        if "sév" in text or "sev" in text:
            return 4.0
        if "-" in text or ":" in text:
            return np.nan
        try:
            number = float(text)
        except ValueError:
            return np.nan
        return number if 0.0 <= number <= 4.0 else np.nan
    return series.map(one)


def regurg_class(g):
    """PanEcho's regurgitation classes from a cleaned grade: <= 0.5 none/trace, < 2 mild, else moderate/severe."""
    return np.where(g.isna(), np.nan, np.where(g <= 0.5, REGURG["none_trace"], np.where(g < 2, REGURG["mild"], REGURG["modsev"])))


def three_way(value, mild_from, modsev_from):
    """Normal below `mild_from`, mild in [mild_from, modsev_from), moderate/severe from `modsev_from`."""
    return np.where(value.isna(), np.nan, np.where(value >= modsev_from, THREE_WAY["modsev"],
                                                   np.where(value >= mild_from, THREE_WAY["mild"], THREE_WAY["normal"])))


def binary(condition, known):
    """1 where `condition`, 0 where `known` and not `condition`, missing elsewhere."""
    return np.where(~known, np.nan, condition.astype(float))


def diastolic_class(series):
    text = series.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index)
    out[text == "grade 0"] = DIASTOLIC["normal"]
    out[(text == "grade 1") | text.str.contains("indétermin", na=False)] = DIASTOLIC["mild_indeterminate"]
    out[text.isin(["grade 2", "grade 3", "grade 4"])] = DIASTOLIC["modsev"]
    return out


EFFUSION_ABSENT = re.compile(r"pas d['’]\s*[ée]panchement", re.IGNORECASE)
EFFUSION_PRESENT = re.compile(r"[ée]panchement p[ée]ricardique\s+(?:l[ée]ger|minime|petit|mod[ée]r[ée]|moyen|important|"
                              r"volumineux|abondant|s[ée]v[èe]re|circonf[ée]rentiel|localis[ée]|post[ée]rieur|ant[ée]rieur)",
                              re.IGNORECASE)


def pericardial_effusion(series):
    """1 when the pericardium text names an effusion with a size or location, 0 when it says
    there is none, missing otherwise (text-derived: verify against a chart review before use)."""
    text = series.astype("string")
    present = text.str.contains(EFFUSION_PRESENT, na=False)
    absent = text.str.contains(EFFUSION_ABSENT, na=False) & ~present
    return np.where(present, 1.0, np.where(absent, 0.0, np.nan))


def derive(r):
    """The target columns from the cleaned report table `r` (one row per report)."""
    t = pd.DataFrame(index=r.index)
    ef = r["Visually Estimated EF"]
    ivsd, lvpwd = r["IVS Diastole Thickness (2D)"], r["LVIW Diastolic Thickness (2D)"]
    lvidd = r["LVID Diastole (2D)"]
    rv_basal = r["RV Basal Diastolic Dimension"]
    lavi = r["LA Volume Index (BP MOD) DuBois"] * 1000.0      # the export holds L/m2; PanEcho uses mL/m2.
    la_dim = r["LA Dimension"]
    av_v = r["AV Peak Velocity"] / 100.0                       # cm/s -> m/s
    av_grad, ava = r["AV Mean Gradient"], r["AV Area (Cont Eq VTI)"]
    tr_v = r["TR Peak Velocity"] / 100.0
    lvot_v = r["[MeasurementsDoppler]_[LVOTPeakVelocity]"] / 100.0
    mr, tr, ar, pr = (grade(r[c]) for c in ("Referring MR Grade", "Referring TR Grade", "Referring AR Grade",
                                            "Referring PR Grade"))
    rv_fn = r["MHI VD fonction systolique"].astype(float).replace(-1, np.nan)   # 0 normal .. 3 severe, -1 missing.

    # Regression targets, in the report's units (PanEcho's where they differ).
    t["lvef"] = ef
    t["rvsp"] = r["RV Systolic Pressure"]
    t["lvidd"] = lvidd
    t["lvids"] = r["LVID Systole (2D)"]
    t["ivsd"] = ivsd
    t["lvpwd"] = lvpwd
    t["lvot_diameter"] = r["LVOT Diameter"]
    t["la_dimension"] = la_dim
    t["lavi"] = lavi
    t["rv_basal_diameter"] = rv_basal
    t["tapse"] = r["TAPSE"]
    t["rv_s_prime"] = r["TV Lateral Ann s' Velocity"]
    t["av_peak_velocity"] = av_v
    t["av_mean_gradient"] = av_grad
    t["av_area"] = ava
    t["tr_peak_velocity"] = tr_v
    t["e_over_e_prime"] = r[["MV E/e' (Lateral)", "MV E/e' (Septal)"]].mean(axis=1, skipna=False)
    t["aortic_root"] = r["Sinus of Valsalva Diameter"]
    t["ivc_diameter"] = r["IVC Diameter (Exp 2D)"]

    # Binary targets (0/1, missing when undecidable).
    t["reduced_ef_40"] = binary(ef < 40, ef.notna())
    t["lvef_le_45"] = binary(ef <= 45, ef.notna())
    walls = ivsd.notna() & lvpwd.notna()
    t["lvh_any"] = binary((ivsd >= 1.1) & (lvpwd >= 1.1), walls)
    t["lvh_modsev"] = binary((ivsd >= 1.3) & (lvpwd >= 1.3), walls)
    t["rv_dysfunction"] = binary(rv_fn >= 1, rv_fn.notna())
    t["rv_dilated"] = binary(rv_basal >= 4.2, rv_basal.notna())
    t["rap_elevated"] = binary(r["RA Pressure"] >= 8, r["RA Pressure"].notna())
    lvot_grad = 4.0 * lvot_v**2
    t["lvot_obstruction"] = binary(lvot_grad >= 20, lvot_v.notna() & (lvot_v > 0) & (lvot_v <= 4))
    mv_area, mv_grad, mv_pht = r["MV Area (Cont Eq VTI)"], r["MV Mean Gradient"], r["MV PHT"]   # PHT in seconds.
    mv_known = mv_area.notna() | mv_grad.notna() | mv_pht.notna()
    t["mv_stenosis"] = binary((mv_area <= 1.5) | (mv_grad >= 5) | (mv_pht >= 0.150), mv_known)
    as_known = ava.notna() | av_grad.notna() | av_v.notna()
    as_severe = (ava < 1.0) | (av_grad >= 40) | (av_v >= 4.0)
    as_any = as_severe | ((ava >= 1.0) & (ava <= 1.5)) | ((av_grad >= 20) & (av_grad < 40)) | ((av_v >= 3.0) & (av_v < 4.0))
    t["as_present"] = binary(as_any, as_known)
    t["as_severe"] = binary(as_severe, as_known)
    for name, g in (("mr_modsev", mr), ("tr_modsev", tr), ("ar_modsev", ar), ("pr_modsev", pr)):
        t[name] = binary(g >= 2, g.notna())
    t["pericardial_effusion"] = pericardial_effusion(r["Pericardium findings"])
    # EchoNext's structural-heart-disease composite (without its effusion component): any of
    # EF <= 45, a wall >= 1.3 cm, AS mean gradient >= 20, moderate+ regurgitation, RV dysfunction
    # >= 2, PASP (RVSP) >= 45, TR peak gradient >= 32. As in EchoNext, a component that was not
    # measured counts as absent; the label is missing only when no component is known at all.
    components = pd.DataFrame({
        "ef": binary(ef <= 45, ef.notna()),
        "wall": binary((ivsd >= 1.3) | (lvpwd >= 1.3), ivsd.notna() | lvpwd.notna()),
        "as": binary(av_grad >= 20, av_grad.notna()),
        "regurg": binary((mr >= 2) | (ar >= 2) | (tr >= 2) | (pr >= 2), mr.notna() | ar.notna() | tr.notna() | pr.notna()),
        "rv": binary(rv_fn >= 2, rv_fn.notna()),
        "pasp": binary(r["RV Systolic Pressure"] >= 45, r["RV Systolic Pressure"].notna()),
        "tr_grad": binary(4.0 * tr_v**2 >= 32, tr_v.notna()),
    })
    any_positive = (components == 1).any(axis=1)
    any_known = components.notna().any(axis=1)
    t["shd_composite"] = np.where(any_positive, 1.0, np.where(any_known, 0.0, np.nan))

    # Classification targets (class indices in PanEcho's order).
    t["lv_systolic_function"] = np.where(ef.isna(), np.nan, np.where(ef >= 54, THREE_WAY["normal"],
                                                                      np.where(ef <= 40, THREE_WAY["modsev"], THREE_WAY["mild"])))
    t["lv_size"] = three_way(lvidd, 5.2 + 1e-9, 6.4)          # <= 5.2 normal, 5.2-6.4 mild, >= 6.4 mod/sev.
    t["rv_size"] = three_way(rv_basal, 4.2, 5.0)
    la_by_volume = three_way(lavi, 34 + 1e-9, 41 + 1e-9)       # <= 34 normal, 34-41 mild, > 41 mod/sev.
    la_by_dimension = three_way(la_dim, 4.0 + 1e-9, 4.6 + 1e-9)
    t["la_size"] = np.where(lavi.notna(), la_by_volume, la_by_dimension)
    t["lv_diastolic_function"] = diastolic_class(r["MHI VG Fonction diastolique"])
    t["av_stenosis"] = np.where(~as_known, np.nan, np.where(as_severe, AV_STENOSIS["severe"],
                                                            np.where(as_any, AV_STENOSIS["mild_moderate"], AV_STENOSIS["none"])))
    t["mr_grade"] = regurg_class(mr)
    t["tr_grade"] = regurg_class(tr)
    t["ar_grade"] = regurg_class(ar)
    return t


TASKS = {name: "regression" for name in (
    "lvef", "rvsp", "lvidd", "lvids", "ivsd", "lvpwd", "lvot_diameter", "la_dimension", "lavi", "rv_basal_diameter",
    "tapse", "rv_s_prime", "av_peak_velocity", "av_mean_gradient", "av_area", "tr_peak_velocity", "e_over_e_prime",
    "aortic_root", "ivc_diameter")}
TASKS.update({name: "binary" for name in (
    "reduced_ef_40", "lvef_le_45", "lvh_any", "lvh_modsev", "rv_dysfunction", "rv_dilated", "rap_elevated",
    "lvot_obstruction", "mv_stenosis", "as_present", "as_severe", "mr_modsev", "tr_modsev", "ar_modsev", "pr_modsev",
    "pericardial_effusion", "shd_composite")})
TASKS.update({name: "classification" for name in (
    "lv_systolic_function", "lv_size", "rv_size", "la_size", "lv_diastolic_function", "av_stenosis", "mr_grade",
    "tr_grade", "ar_grade")})
CLASSES = {"lv_systolic_function": THREE_WAY, "lv_size": THREE_WAY, "rv_size": THREE_WAY, "la_size": THREE_WAY,
           "lv_diastolic_function": DIASTOLIC, "av_stenosis": AV_STENOSIS, "mr_grade": REGURG, "tr_grade": REGURG,
           "ar_grade": REGURG}
UNITS = {"lvef": "%", "rvsp": "mmHg", "lvidd": "cm", "lvids": "cm", "ivsd": "cm", "lvpwd": "cm", "lvot_diameter": "cm",
         "la_dimension": "cm", "lavi": "mL/m2", "rv_basal_diameter": "cm", "tapse": "cm", "rv_s_prime": "cm/s",
         "av_peak_velocity": "m/s", "av_mean_gradient": "mmHg", "av_area": "cm2", "tr_peak_velocity": "m/s",
         "e_over_e_prime": "ratio", "aortic_root": "cm", "ivc_diameter": "cm"}


def describe(targets, split=None):
    """Per target: the task, how many studies have it and its distribution (percentiles for
    regression, prevalence for binary, class fractions for classification)."""
    info = {}
    for name, task in TASKS.items():
        values = targets[name].dropna()
        entry = {"task": task, "n": int(len(values)), "coverage": float(len(values) / max(len(targets), 1))}
        if task == "regression":
            entry["units"] = UNITS[name]
            if len(values):
                q = values.quantile([0.05, 0.5, 0.95])
                entry.update(p5=float(q.iloc[0]), median=float(q.iloc[1]), p95=float(q.iloc[2]),
                             mean=float(values.mean()), std=float(values.std(ddof=0)))
        elif task == "binary":
            entry["prevalence"] = float(values.mean()) if len(values) else None
        else:
            counts = values.value_counts().sort_index()
            entry["classes"] = {label: int(counts.get(index, 0)) for label, index in CLASSES[name].items()}
        info[name] = entry
    return info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", required=True, help="The cleaned DeepECHO report export (.parquet or .csv).")
    p.add_argument("--pacs", required=True, help="PACS index (AccessionNumber, StudyInstanceUID, PatientID).")
    p.add_argument("--videos", help="The issue #6 videos.csv; its rows get the targets of their study.")
    p.add_argument("--out-dir", required=True, help="Output folder; must be outside this repository.")
    p.add_argument("--report-accession-col", default="AccessionNumber")
    p.add_argument("--report-patient-col", default="Dossier")
    p.add_argument("--pacs-accession-col", default="AccessionNumber")
    p.add_argument("--pacs-study-col", default="StudyInstanceUID")
    p.add_argument("--pacs-patient-col", default="PatientID")
    p.add_argument("--study-col", default="study_id", help="Study column of videos.csv.")
    p.add_argument("--split-col", default="split", help="Split column of videos.csv, for the per-split report.")
    return p.parse_args(argv)


def link_studies(reports, pacs, args):
    """One row per study: the report's targets keyed by StudyInstanceUID through the accession
    number. Accessions found in several PACS studies, or reports disagreeing on a study, are dropped."""
    reports = reports.assign(accession=normalize_id(reports[args.report_accession_col]),
                             report_patient=normalize_id(reports[args.report_patient_col]))
    pacs = pacs.rename(columns={args.pacs_accession_col: "accession", args.pacs_study_col: "study_id",
                                args.pacs_patient_col: "pacs_patient"})
    pacs = pacs.assign(accession=normalize_id(pacs.accession), study_id=normalize_id(pacs.study_id),
                       pacs_patient=normalize_id(pacs.pacs_patient))
    pacs = pacs.drop_duplicates(["accession", "study_id"])
    ambiguous = pacs.accession.duplicated(keep=False)
    print(f"[pacs]   {int(ambiguous.sum())} accession rows map to several studies: dropped.")
    pacs = pacs[~ambiguous]
    linked = reports.merge(pacs[["accession", "study_id", "pacs_patient"]], on="accession", how="inner")
    same = linked.report_patient.str.lstrip("0") == linked.pacs_patient.str.lstrip("0")
    print(f"[join]   {len(linked)} of {len(reports)} reports found in the PACS index; "
          f"{int((~same).sum())} with another patient id dropped.")
    linked = linked[same.fillna(False)]
    dup = linked.study_id.duplicated(keep=False)
    if dup.any():
        print(f"[drop]   {linked.loc[dup, 'study_id'].nunique()} studies with several reports: dropped.")
        linked = linked[~dup]
    return linked


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    reports = pd.read_parquet(args.reports) if args.reports.endswith(".parquet") else pd.read_csv(args.reports, low_memory=False)
    pacs = pd.read_csv(args.pacs, dtype=str, usecols=[args.pacs_accession_col, args.pacs_study_col, args.pacs_patient_col])
    linked = link_studies(reports, pacs, args)
    targets = derive(linked)
    table = pd.concat([linked[["study_id", "accession"]].reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    table.to_csv(os.path.join(args.out_dir, "study_targets.csv"), index=False)
    print(f"[write]  study_targets.csv: {len(table)} studies, {len(TASKS)} targets.")
    info = {"targets": describe(table), "tasks": TASKS, "classes": CLASSES, "units": UNITS,
            "sources": {"reports": os.path.basename(args.reports), "pacs": os.path.basename(args.pacs)}}

    if args.videos:
        videos = pd.read_csv(args.videos)
        videos = videos.assign(**{args.study_col: normalize_id(videos[args.study_col])})
        merged = videos.merge(table.drop(columns=["accession"]), left_on=args.study_col, right_on="study_id", how="left")
        if "study_id" != args.study_col:
            merged = merged.drop(columns=["study_id"])
        with_targets = merged[list(TASKS)].notna().any(axis=1)
        print(f"[videos] {len(videos)} videos in {videos[args.study_col].nunique()} studies; "
              f"{int(with_targets.sum())} videos ({merged.loc[with_targets, args.study_col].nunique()} studies) have at least one target.")
        merged.to_csv(os.path.join(args.out_dir, "videos_with_targets.csv"), index=False)
        per_study = merged.drop_duplicates(args.study_col)
        if args.split_col in per_study.columns:
            info["splits"] = {str(split): describe(rows) for split, rows in per_study.groupby(args.split_col)}
        else:
            info["splits"] = {"all": describe(per_study)}
    with open(os.path.join(args.out_dir, "targets_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[write]  targets_info.json")


if __name__ == "__main__":
    main()
