# tools/panecho_mimic/score.py

"""Score PanEcho's MIMIC-IV-ECHO predictions (`run_inference.py`) against the harmonized study targets, with the
metric definitions of the EchoJEPA study probe (`evals/study_multitask_frozen/eval.py::head_metrics`):

- binary heads: AUROC (`metrics.auroc`, ties count 1/2);
- 3-class heads: one-vs-rest AUROC averaged over the classes present;
- regression heads: Pearson r, MAE, bias (mean prediction - reference) in the target's units (`metrics.regression`),
  and standardized MAE = MAE / std of that target in the MHI training labels (`--mhi-stats`).

Study predictions are PanEcho's own aggregation: the mean of each output over the study's videos (each video already
the mean of 4 clips). The cohort is every study with at least one video and one scorable target
(pericardial_effusion and rv_s_prime are excluded, as in the EchoJEPA MIMIC scoring). Mean AUROC gets a 95% bootstrap
CI over studies (percentile, 1000 resamples, seed 0).

Example:
    python tools/panecho_mimic/score.py --out-dir /media/data1/datasets/DeepEcho/claude_runs/panecho_mimic \\
        --mhi-stats <targets_info.json>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from evals.video_classification_frozen import metrics as probe_metrics  # noqa: E402

MIMIC = "/media/data1/datasets/external_echo/MIMIC-IV-ECHO"
EXCLUDED = {"pericardial_effusion": "MIMIC records effusions only when present (no negatives)",
            "rv_s_prime": "not recorded in MIMIC"}
# Heads the EchoJEPA MIMIC comparison flags (label definition or recording differs between MHI and MIMIC).
FLAGGED = {"lvh_any", "lvh_modsev", "mv_stenosis", "ar_modsev", "ar_grade", "lv_diastolic_function", "shd_composite"}
REGURG = ["Mild", "Moderate|Severe", "None|Trace"]
THREE = {"lv_systolic_function": ("LVSystolicFunction", ["Mildly Decreased", "Moderately|Severely Decreased", "Normal|Hyperdynamic"]),
         "lv_size": ("LVSize", ["Mildly Increased", "Moderately|Severely Increased", "Normal"]),
         "rv_size": ("RVSize", ["Mildly Increased", "Moderately|Severely Increased", "Normal"]),
         "la_size": ("LASize", ["Mildly Dilated", "Moderately|Severely Dilated", "Normal"]),
         "lv_diastolic_function": ("LVDiastolicFunction", ["Mild|Indeterminate", "Moderate|Severe", "Normal"]),
         "av_stenosis": ("AVStenosis", ["Mild|Moderate", "None", "Severe"]),
         "mr_grade": ("MVRegurgitation", REGURG), "tr_grade": ("TVRegurgitation", REGURG),
         "ar_grade": ("AVRegurg", REGURG)}
# MHI target -> (PanEcho source, how the score is formed, relation). Class orders of `make_study_targets.py` are
# PanEcho's (mild, modsev, normal / mild_moderate, none, severe), so the 3-class probabilities are used as they are.
REGRESSION = {
    "lvef": ("EF", lambda p: p["EF"], "exact (%)"),
    "rvsp": ("RVSP", lambda p: p["RVSP"], "exact (mmHg)"),
    "lvidd": ("LVIDd", lambda p: p["LVIDd"], "exact (cm)"),
    "lvids": ("LVIDs", lambda p: p["LVIDs"], "exact (cm)"),
    "ivsd": ("IVSd", lambda p: p["IVSd"], "exact (cm)"),
    "lvpwd": ("LVPWd", lambda p: p["LVPWd"], "exact (cm)"),
    "lvot_diameter": ("LVOTDiam", lambda p: p["LVOTDiam"], "exact (cm)"),
    "la_dimension": ("LAIDs2D", lambda p: p["LAIDs2D"], "exact (cm)"),
    "lavi": ("LAVol / BSA", lambda p: p["LAVol"] / p["body_surface_area"],
             "transformed: PanEcho LA volume (mL) / MIMIC BSA (m2); method (A-L vs MOD) may differ"),
    "rv_basal_diameter": ("RVIDd", lambda p: p["RVIDd"], "related: PanEcho RV diastolic diameter vs MHI RV basal (cm)"),
    "tapse": ("TAPSE", lambda p: p["TAPSE"], "exact (cm)"),
    "av_peak_velocity": ("AVPkVel(m|s)", lambda p: p["AVPkVel(m|s)"], "exact (m/s)"),
    "tr_peak_velocity": ("TVPkGrad", lambda p: np.sqrt(np.clip(p["TVPkGrad"], 0, None) / 4.0),
                         "transformed: v = sqrt(TR peak gradient / 4)"),
    "e_over_e_prime": ("E|EAvg", lambda p: p["E|EAvg"], "exact (ratio)"),
    "aortic_root": ("AORoot", lambda p: p["AORoot"], "exact (cm); PanEcho 'aortic root' vs MHI sinus of Valsalva"),
}
BINARY = {
    "reduced_ef_40": ("EF", lambda p: -p["EF"], "derived: -EF prediction"),
    "lvef_le_45": ("EF", lambda p: -p["EF"], "derived: -EF prediction"),
    "lvh_any": ("LVWallThickness-increased-any", lambda p: p["LVWallThickness-increased-any"], "exact"),
    "lvh_modsev": ("LVWallThickness-increased-modsev", lambda p: p["LVWallThickness-increased-modsev"], "exact"),
    "rv_dysfunction": ("RVSystolicFunction", lambda p: p["RVSystolicFunction"], "exact (P decreased)"),
    "rv_dilated": ("RVSize", lambda p: 1 - p["RVSize__Normal"], "derived: 1 - P(RV size normal)"),
    "rap_elevated": ("RAP-8-or-higher", lambda p: p["RAP-8-or-higher"], "exact"),
    "lvot_obstruction": ("LVOT20mmHg", lambda p: p["LVOT20mmHg"], "exact (peak gradient >= 20 mmHg)"),
    "mv_stenosis": ("MVStenosis", lambda p: p["MVStenosis"], "exact"),
    "as_present": ("AVStenosis", lambda p: 1 - p["AVStenosis__None"], "derived: 1 - P(AS none)"),
    "as_severe": ("AVStenosis", lambda p: p["AVStenosis__Severe"], "derived: P(AS severe)"),
    "mr_modsev": ("MVRegurgitation", lambda p: p["MVRegurgitation__Moderate|Severe"], "derived: P(MR mod/sev)"),
    "tr_modsev": ("TVRegurgitation", lambda p: p["TVRegurgitation__Moderate|Severe"], "derived: P(TR mod/sev)"),
    "ar_modsev": ("AVRegurg", lambda p: p["AVRegurg__Moderate|Severe"], "derived: P(AR mod/sev)"),
}
# Sensitivity alternatives (reported, not in the means).
ALTERNATIVES = {
    "reduced_ef_40": ("LVSystolicFunction", lambda p: p["LVSystolicFunction__Moderately|Severely Decreased"],
                      "P(LV systolic function mod/sev decreased)"),
    "lvef_le_45": ("LVSystolicFunction", lambda p: 1 - p["LVSystolicFunction__Normal|Hyperdynamic"],
                   "1 - P(LV systolic function normal)"),
}
UNMAPPED = {
    "av_mean_gradient": "no PanEcho head (PanEcho has AV peak velocity only)",
    "av_area": "no PanEcho head",
    "ivc_diameter": "no PanEcho head (RAP-8-or-higher is related but not the same quantity)",
    "pr_modsev": "no PanEcho pulmonic regurgitation head",
    "shd_composite": "no PanEcho head (a composite could be assembled from PanEcho outputs, but not as published)",
}
META = ["subject_id", "study_id", "mp4_native256", "view_hw20", "n_frames", "fps", "status", "decoded_frames", "error"]


def auroc_ovr(y, prob):
    present = sorted(set(y.astype(int).tolist()))
    a = [probe_metrics.auroc(y == c, prob[:, c]) for c in present]
    a = [x for x in a if np.isfinite(x)]
    return float(np.mean(a)) if a else float("nan")


def class_head(target, y, prob):
    if target in THREE:
        return auroc_ovr(y, prob)
    return probe_metrics.auroc(y > 0.5, prob)


def write_markdown(table, path):
    """The head-by-head tables (flagged heads marked with *)."""
    def name(r):
        return f"{r.target}{' *' if r.flagged else ''}"
    lines = ["| Head | Type | PanEcho source | n | Prevalence / classes (mild/modsev/normal or mm/none/severe) | AUROC [95% CI] |",
             "|---|---|---|---|---|---|"]
    for r in table[table.task.isin(["binary", "classification"])].itertuples():
        extra = f"{r.prevalence:.3f}" if r.task == "binary" else r.class_counts
        ci = f" [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]" if np.isfinite(r.auroc_lo) else ""
        lines.append(f"| {name(r)} | {r.task} | {r.panecho} ({r.relation}) | {r.n} | {extra} | {r.auroc:.3f}{ci} |")
    lines += ["", "| Head | PanEcho source | n | Pearson r | MAE | Bias (pred - ref) | MHI std | Std. MAE | Ref mean | Pred mean |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table[table.task == "regression"].itertuples():
        lines.append(f"| {name(r)} | {r.panecho} ({r.relation}) | {r.n} | {r.pearson_r:.3f} | {r.mae:.3f} | {r.bias:+.3f} | "
                     f"{r.mhi_std:.3f} | {r.smae:.3f} | {r.ref_mean:.3f} | {r.pred_mean:.3f} |")
    lines += ["", "| Head | Status | Reason |", "|---|---|---|"]
    for r in table[table.task.isin(["unmapped", "excluded"])].itertuples():
        lines.append(f"| {name(r)} | {r.task} | {r.relation} |")
    open(path, "w").write("\n".join(lines) + "\n")


def load_predictions(out_dir, expected_videos):
    shards = sorted(glob.glob(os.path.join(out_dir, "video_shards", "shard_*.parquet")))
    videos = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    return videos, len(videos) / expected_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mhi-stats", help="json {target: {std: ...}} of the MHI training labels (targets_info.json)")
    ap.add_argument("--mhi-split", default="train", help="split key inside --mhi-stats, if it has one")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    all_videos = pd.read_parquet(os.path.join(MIMIC, "metadata", "videos.parquet"), columns=["study_id"])
    videos, frac = load_predictions(args.out_dir, len(all_videos))
    videos["study_id"] = videos.study_id.astype("int64")             # object in videos.parquet, int64 in the tables.
    ok = videos[videos.error.fillna("") == ""]
    out_cols = [c for c in videos.columns if c not in META]
    study = ok.groupby("study_id")[out_cols].mean()                 # PanEcho: mean over the study's videos.
    study["n_videos"] = ok.groupby("study_id").size()
    study = study.join(ok.groupby("study_id").subject_id.first())
    videos.to_parquet(os.path.join(args.out_dir, "predictions_video.parquet"), index=False)

    targets = pd.read_parquet(os.path.join(MIMIC, "metadata", "study_targets_harmonized.parquet"))
    targets = targets.drop(columns=["subject_id"]).astype({"study_id": "int64"}).set_index("study_id")
    bsa = pd.read_parquet(os.path.join(MIMIC, "metadata", "measurements_wide.parquet"),
                          columns=["study_id", "body_surface_area"]).astype({"study_id": "int64"}).set_index("study_id")
    bsa["body_surface_area"] = pd.to_numeric(bsa.body_surface_area, errors="coerce")   # stored as text.
    bsa.loc[(bsa.body_surface_area < 0.8) | (bsa.body_surface_area > 3.5), "body_surface_area"] = np.nan   # implausible.
    target_names = [t for t in list(REGRESSION) + list(BINARY) + list(THREE) + list(UNMAPPED)]
    scorable = [t for t in targets.columns if t in set(target_names) | set(EXCLUDED)]
    scorable = [t for t in scorable if t not in EXCLUDED]
    cohort = targets.loc[targets.index.intersection(study.index)]
    cohort = cohort[cohort[scorable].notna().any(axis=1)]
    p = study.loc[cohort.index].join(bsa)
    p.reset_index().assign(**{f"target__{t}": cohort[t].values for t in scorable}).to_parquet(
        os.path.join(args.out_dir, "predictions_study.parquet"), index=False)

    std = {}
    if args.mhi_stats:
        info = json.load(open(args.mhi_stats))
        info = info.get(args.mhi_split, info)
        std = {k: v.get("std") for k, v in info.items() if isinstance(v, dict) and v.get("std")}

    rows, scores = [], {}
    for t, (src, f, rel) in REGRESSION.items():
        y, pred = cohort[t].values.astype(float), np.asarray(f(p), dtype=float)
        m = np.isfinite(y) & np.isfinite(pred)
        r = probe_metrics.regression(y[m], pred[m])
        s = std.get(t)
        rows.append(dict(target=t, task="regression", panecho=src, relation=rel, n=r["n"], pearson_r=r["pearson_r"],
                         mae=r["mae"], bias=r["bias"], mhi_std=s, smae=r["mae"] / s if s else np.nan,
                         ref_mean=float(y[m].mean()), pred_mean=float(pred[m].mean()), flagged=t in FLAGGED))
    for t, (src, f, rel) in BINARY.items():
        y, sc = cohort[t].values.astype(float), np.asarray(f(p), dtype=float)
        m = np.isfinite(y) & np.isfinite(sc)
        scores[t] = (y, sc, m)
        rows.append(dict(target=t, task="binary", panecho=src, relation=rel, n=int(m.sum()),
                         prevalence=float(y[m].mean()), auroc=class_head(t, y[m], sc[m]), flagged=t in FLAGGED))
    for t, (src, classes) in THREE.items():
        y = cohort[t].values.astype(float)
        prob = p[[f"{src}__{c}" for c in classes]].values.astype(float)
        m = np.isfinite(y) & np.isfinite(prob).all(1)
        scores[t] = (y, prob, m)
        counts = np.bincount(y[m].astype(int), minlength=3)
        rows.append(dict(target=t, task="classification", panecho=src, relation="exact class order " + "/".join(classes),
                         n=int(m.sum()), class_counts="/".join(map(str, counts)),
                         auroc=class_head(t, y[m], prob[m]), flagged=t in FLAGGED))
    for t, (src, f, rel) in ALTERNATIVES.items():
        y, sc = cohort[t].values.astype(float), np.asarray(f(p), dtype=float)
        m = np.isfinite(y) & np.isfinite(sc)
        rows.append(dict(target=t + " (alt)", task="binary", panecho=src, relation="sensitivity: " + rel,
                         n=int(m.sum()), prevalence=float(y[m].mean()), auroc=probe_metrics.auroc(y[m] > 0.5, sc[m]),
                         flagged=t in FLAGGED))
    for t, why in {**UNMAPPED, **EXCLUDED}.items():
        rows.append(dict(target=t, task="unmapped" if t in UNMAPPED else "excluded", relation=why,
                         n=int(cohort[t].notna().sum()) if t in cohort else 0, flagged=t in FLAGGED))
    table = pd.DataFrame(rows)

    # Bootstrap over studies: every head re-scored on the same resample, then averaged.
    heads = list(scores)
    unflagged = [h for h in heads if h not in FLAGGED]
    rng = np.random.default_rng(0)
    n = len(cohort)
    boot = {"all": [], "unflagged": []}
    per_head = {h: [] for h in heads}
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        a = {}
        for h in heads:
            y, sc, m = scores[h]
            yy, ss, mm = y[idx], sc[idx], m[idx]
            a[h] = class_head(h, yy[mm], ss[mm])
            per_head[h].append(a[h])
        boot["all"].append(np.nanmean([a[h] for h in heads]))
        boot["unflagged"].append(np.nanmean([a[h] for h in unflagged]))
    ci = {h: np.nanpercentile(v, [2.5, 97.5]) for h, v in per_head.items()}
    table["auroc_lo"] = table.target.map(lambda t: ci[t][0] if t in ci else np.nan)
    table["auroc_hi"] = table.target.map(lambda t: ci[t][1] if t in ci else np.nan)
    au = table.set_index("target").auroc
    summary = {
        "n_studies": int(n), "n_patients": int(p.subject_id.nunique()),
        "videos_scored": int(len(ok)), "videos_failed": int(len(videos) - len(ok)), "fraction_of_videos": frac,
        "auroc_heads": heads, "auroc_heads_unflagged": unflagged,
        "mean_auroc": float(np.nanmean([au[h] for h in heads])),
        "mean_auroc_ci95": [float(x) for x in np.percentile(boot["all"], [2.5, 97.5])],
        "mean_auroc_unflagged": float(np.nanmean([au[h] for h in unflagged])),
        "mean_auroc_unflagged_ci95": [float(x) for x in np.percentile(boot["unflagged"], [2.5, 97.5])],
        "unmapped": UNMAPPED, "excluded": EXCLUDED, "mhi_std_source": args.mhi_stats,
    }
    table.to_csv(os.path.join(args.out_dir, "scores_by_head.csv"), index=False)
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    write_markdown(table, os.path.join(args.out_dir, "scores_by_head.md"))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 50)
    print(table.drop(columns=["relation"]).to_string(float_format=lambda x: f"{x:.3f}"))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("unmapped", "excluded")}, indent=1))


if __name__ == "__main__":
    main()
