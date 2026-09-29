"""Aggregate the per-study EchoNet inference rows, join them with the ground truth and produce
the paper-style performance tables and figures.

Ground truth
- df_report (MHI echo report export, per AccessionNumber): IVSd, LVIDd, LVIDs, LVPWd, LA dimension, LVOT
  diameter, IVC, RV base, TR/AV/MV-E peak velocity, lateral/septal e', TAPSE, Visually Estimated EF.
- GE structured report (SR) parsed per study: same 2D items plus LA area A4C/A2C, LVOT Vmax and LVOT VTI.
- OCR machine value on the Doppler still itself (per image, for the Doppler models).

Metrics per measurement (as in Sahashi et al. JACC 2025): n, Pearson r, MAE, mean bias (DL - human),
limits of agreement, plus scatter (identity line) and Bland-Altman figures.
EchoNet-Dynamic: AUROC / AUPRC for EF < 50 % (Visual EF) with 1,000 x 80 % bootstrap CIs, MAE, r, and a
configurable per-study aggregation / clip-quality gate.
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

COHORT = os.environ.get("ECHONET_COHORT", "/volume/echonet_eval/cohort_10k.parquet")


def bootstrap(y, s, fn, n=1000, frac=0.8, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); vals = []
    m = max(2, int(len(y) * frac))
    for _ in range(n):
        idx = rng.choice(len(y), m, replace=False)
        try:
            vals.append(fn(y[idx], s[idx]))
        except Exception:
            pass
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def agreement(h, d):
    """human h vs DL d -> dict of paper-style metrics."""
    h, d = np.asarray(h, float), np.asarray(d, float)
    ok = np.isfinite(h) & np.isfinite(d); h, d = h[ok], d[ok]
    if len(h) < 3:
        return dict(n=int(len(h)))
    diff = d - h
    r = stats.pearsonr(h, d)[0]
    lo, hi = bootstrap(h, d, lambda a, b: stats.pearsonr(a, b)[0], n=300)
    return dict(n=int(len(h)), r=float(r), r_ci_low=lo, r_ci_high=hi, mae=float(np.mean(np.abs(diff))),
                mae_ci=bootstrap(h, d, lambda a, b: np.mean(np.abs(b - a)), n=300),
                bias=float(diff.mean()), loa_low=float(diff.mean() - 1.96 * diff.std()), loa_high=float(diff.mean() + 1.96 * diff.std()),
                mape=float(np.mean(np.abs(diff) / np.clip(np.abs(h), 1e-6, None)) * 100), human_mean=float(h.mean()), dl_mean=float(d.mean()))


def scatter_ba(h, d, title, unit, path):
    h, d = np.asarray(h, float), np.asarray(d, float); ok = np.isfinite(h) & np.isfinite(d); h, d = h[ok], d[ok]
    if len(h) < 3:
        return
    m = agreement(h, d)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    lim = [min(h.min(), d.min()), max(np.percentile(h, 99.5), np.percentile(d, 99.5))]
    ax[0].scatter(h, d, s=6, alpha=0.35, color="#2a6f97"); ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel(f"Human ({unit})"); ax[0].set_ylabel(f"EchoNet ({unit})")
    ax[0].set_title(f"{title}\nn={m['n']}  r={m['r']:.2f} [{m['r_ci_low']:.2f}, {m['r_ci_high']:.2f}]  MAE={m['mae']:.2f} {unit}")
    mean, diff = (h + d) / 2, d - h
    ax[1].scatter(mean, diff, s=6, alpha=0.35, color="#c9184a"); ax[1].axhline(m["bias"], color="k"); ax[1].axhline(m["loa_low"], ls="--", color="gray"); ax[1].axhline(m["loa_high"], ls="--", color="gray")
    ax[1].set_xlabel(f"Mean ({unit})"); ax[1].set_ylabel(f"EchoNet - Human ({unit})"); ax[1].set_title(f"Bland-Altman  bias={m['bias']:.2f}  LoA=[{m['loa_low']:.2f}, {m['loa_high']:.2f}]")
    ax[1].set_ylim(np.percentile(diff, 0.5) - 1, np.percentile(diff, 99.5) + 1)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


# measurement definitions: (name, caliper model, view(s), which statistic, report column, SR column, unit, GT scale to mm)
CALIPER_DEFS = [
    ("IVSd", "ivs", ["PLAX"], "d_min_mm", "ivsd", "sr_ivsd_mm", "mm", 10.0),
    ("LVIDd", "lvid", ["PLAX"], "d_max_mm", "lvidd", "sr_lvidd_mm", "mm", 10.0),
    ("LVIDs", "lvid", ["PLAX"], "d_min_mm", "lvids", "sr_lvids_mm", "mm", 10.0),
    ("LVPWd", "lvpw", ["PLAX"], "d_min_mm", "lvpwd", "sr_lvpwd_mm", "mm", 10.0),
    ("LA diameter", "la", ["PLAX"], "d_max_mm", "la_diam", "sr_la_diam_mm", "mm", 10.0),
    ("Aortic root (SoV)", "aortic_root", ["PLAX"], "d_med_mm", "sov_diam", None, "mm", 10.0),
    ("Ascending aorta", "aorta", ["PLAX"], "d_med_mm", "asc_ao", "sr_asc_ao_mm", "mm", 10.0),
    ("RV base", "rv_base", ["A4C"], "d_max_mm", "rv_base", "sr_rv_base_mm", "mm", 10.0),
    ("IVC", "ivc", ["SUBCOSTAL"], "d_max_mm", "ivc_diam", "sr_ivc_mm", "mm", 10.0),
]
DOPPLER_DEFS = [  # (name, model, report col, sr col, unit, scale report->m/s)
    ("TR Vmax", "trvmax", "tr_vmax", "sr_tr_vmax_ms", "m/s"),
    ("AV Vmax", "avvmax", "av_vmax", "sr_av_vmax_ms", "m/s"),
    ("LVOT Vmax", "lvotvmax", None, "sr_lvot_vmax_ms", "m/s"),
    ("Lateral e'", "latevel", "lat_e", None, "cm/s"),
    ("Septal e'", "medevel", "sept_e", None, "cm/s"),
]


def load_rows(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "rows_shard*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def per_study(df, kind, model, views, stat, agg="median"):
    s = df[(df.kind == kind) & (df.model == model)]
    if views:
        s = s[s.view.isin(views)]
    s = s[s[stat].notna()]
    return s.groupby("StudyInstanceUID")[stat].agg(agg)


def guess_report_unit(series_mm_or_cm):
    """Report columns mix cm and mm; return a factor to mm based on the median."""
    med = np.nanmedian(series_mm_or_cm)
    return 10.0 if med < 15 else 1.0


def main(results_dir, out_dir, ef_agg="median", ef_min_clips=1, ef_gate_frac=0.0, ef_gate_change=0.0):
    os.makedirs(out_dir, exist_ok=True)
    df = load_rows(results_dir)
    cohort = pd.read_parquet(COHORT).set_index("StudyInstanceUID")
    sr = df[df.kind == "sr"].drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    n_studies = df.StudyInstanceUID.nunique()
    summary = {"n_studies": int(n_studies), "n_rows": int(len(df)), "kinds": df.kind.value_counts().to_dict()}
    table = []
    # ---- 2D calipers
    for name, model, views, stat, rcol, scol, unit, _ in CALIPER_DEFS:
        pred = per_study(df, "caliper", model, views, stat)
        if rcol is not None and rcol in cohort:
            gt = cohort[rcol].astype(float); gt = gt * guess_report_unit(gt)
            j = pd.concat([gt.rename("human"), pred.rename("dl")], axis=1).dropna()
            m = agreement(j.human, j.dl); m.update(measurement=name, model=model, gt="report", views="+".join(views), stat=stat); table.append(m)
            scatter_ba(j.human, j.dl, f"{name} (EchoNet {model} vs report)", unit, f"{out_dir}/caliper_{model}_{name.replace(' ', '_').replace('/', '')}_report.png")
        if scol is not None and scol in sr:
            gt = sr[scol].astype(float)
            j = pd.concat([gt.rename("human"), pred.rename("dl")], axis=1).dropna()
            m = agreement(j.human, j.dl); m.update(measurement=name, model=model, gt="GE SR", views="+".join(views), stat=stat); table.append(m)
            scatter_ba(j.human, j.dl, f"{name} (EchoNet {model} vs GE structured report)", unit, f"{out_dir}/caliper_{model}_{name.replace(' ', '_').replace('/', '')}_sr.png")
    # ---- Doppler Vmax: per still vs OCR value, per study vs report / SR
    dop = df[df.kind == "doppler_vmax"].copy()
    for name, model, rcol, scol, unit in DOPPLER_DEFS:
        s = dop[dop.model == model].copy()
        if len(s) == 0:
            continue
        factor = 100.0 if unit == "cm/s" else 1.0
        s["dl"] = s.vmax_ms.astype(float) * factor
        s["ocr"] = pd.to_numeric(s.ocr_value, errors="coerce")
        if unit == "m/s":
            s.loc[s.ocr_unit.astype(str).str.upper().str.contains("CM"), "ocr"] /= 100.0
        else:
            s.loc[s.ocr_unit.astype(str).str.upper().eq("M/S"), "ocr"] *= 100.0
        s = s[(s.ocr > 0) & (s.ocr < (600 if unit == "cm/s" else 8))]
        m = agreement(s.ocr, s.dl); m.update(measurement=name, model=model, gt="on-screen GE value (per still)", views="DOPPLER", stat="argmax"); table.append(m)
        scatter_ba(s.ocr, s.dl, f"{name} (EchoNet {model} vs on-screen GE value)", unit, f"{out_dir}/doppler_{model}_ocr.png")
        # exclude edge hits as a QC variant
        s2 = s[~s.edge_hit.astype(bool)]
        m = agreement(s2.ocr, s2.dl); m.update(measurement=name, model=model, gt="on-screen GE value, edge hits removed", views="DOPPLER", stat="argmax"); table.append(m)
        pst = s.groupby("StudyInstanceUID").dl.median()
        if rcol is not None and rcol in cohort:
            gt = cohort[rcol].astype(float)
            if unit == "m/s" and np.nanmedian(gt) > 20:
                gt = gt / 100.0
            j = pd.concat([gt.rename("human"), pst.rename("dl")], axis=1).dropna()
            m = agreement(j.human, j.dl); m.update(measurement=name, model=model, gt="report (per study, median of stills)", views="DOPPLER", stat="median"); table.append(m)
            scatter_ba(j.human, j.dl, f"{name} (EchoNet {model} vs report)", unit, f"{out_dir}/doppler_{model}_report.png")
        if scol is not None and scol in sr:
            gt = sr[scol].astype(float) * factor
            j = pd.concat([gt.rename("human"), pst.rename("dl")], axis=1).dropna()
            m = agreement(j.human, j.dl); m.update(measurement=name, model=model, gt="GE SR (per study)", views="DOPPLER", stat="median"); table.append(m)
            scatter_ba(j.human, j.dl, f"{name} (EchoNet {model} vs GE structured report)", unit, f"{out_dir}/doppler_{model}_sr.png")
    # ---- LA area
    la = df[df.kind == "la_area"]
    for view, scol in (("A4C", "sr_la_area_a4c_cm2"), ("A2C", "sr_la_area_a2c_cm2")):
        p = la[la.view == view].groupby("StudyInstanceUID").area_max_cm2.max()
        if scol in sr:
            j = pd.concat([sr[scol].astype(float).rename("human"), p.rename("dl")], axis=1).dropna()
            m = agreement(j.human, j.dl); m.update(measurement=f"LA area {view}", model="LA_AREA", gt="GE SR", views=view, stat="max over frames/clips"); table.append(m)
            scatter_ba(j.human, j.dl, f"LA area {view} (EchoNet LA_AREA vs GE SR)", "cm2", f"{out_dir}/la_area_{view}_sr.png")
    # ---- VTI
    vti = df[(df.kind == "vti") & df.vti_cm.notna()]
    for target, scol in (("LVOT", "sr_lvot_vti_cm"),):
        p = vti[vti.model == target].groupby("StudyInstanceUID").vti_cm.median()
        if scol in sr:
            j = pd.concat([sr[scol].astype(float).rename("human"), p.rename("dl")], axis=1).dropna()
            m = agreement(j.human, j.dl); m.update(measurement=f"{target} VTI", model=f"VTI_{target}", gt="GE SR", views="DOPPLER", stat="median of stills"); table.append(m)
            scatter_ba(j.human, j.dl, f"{target} VTI (EchoNet VTI seg vs GE SR)", "cm", f"{out_dir}/vti_{target}_sr.png")
    # LVOT Vmax from the VTI mask as a by-product
    p = vti[vti.model == "LVOT"].groupby("StudyInstanceUID").vmax_from_mask_ms.median()
    if "sr_lvot_vmax_ms" in sr:
        j = pd.concat([sr["sr_lvot_vmax_ms"].astype(float).rename("human"), p.rename("dl")], axis=1).dropna()
        m = agreement(j.human, j.dl); m.update(measurement="LVOT Vmax (from VTI mask)", model="VTI_LVOT", gt="GE SR", views="DOPPLER", stat="median"); table.append(m)
    # ---- EchoNet-Dynamic EF
    dyn = df[df.kind == "dynamic"].copy()
    ef_table, ef_summary = ef_analysis(dyn, cohort, out_dir, ef_agg, ef_min_clips, ef_gate_frac, ef_gate_change)
    summary["ef"] = ef_summary
    tab = pd.DataFrame(table)
    cols = ["measurement", "model", "gt", "views", "stat", "n", "r", "r_ci_low", "r_ci_high", "mae", "mae_ci", "mape", "bias", "loa_low", "loa_high", "human_mean", "dl_mean"]
    tab = tab[[c for c in cols if c in tab]]
    tab.to_csv(f"{out_dir}/agreement_table.csv", index=False)
    ef_table.to_csv(f"{out_dir}/ef_table.csv", index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(json.dumps({k: v for k, v in summary.items() if k != "ef"}, default=str))
    print(tab.round(3).to_string())
    print(ef_table.round(3).to_string())
    return tab, ef_table


def ef_analysis(dyn, cohort, out_dir, agg, min_clips, gate_frac, gate_change):
    rows = []
    gt = cohort["ef_visual"].astype(float)
    variants = {
        "all A4C clips, mean": dict(agg="mean", min_clips=1, gate_frac=0.0, gate_change=0.0),
        "all A4C clips, median": dict(agg="median", min_clips=1, gate_frac=0.0, gate_change=0.0),
        "LV visible >=80% frames": dict(agg="median", min_clips=1, gate_frac=0.8, gate_change=0.0),
        "LV visible >=80% & area change >=0.25": dict(agg="median", min_clips=1, gate_frac=0.8, gate_change=0.25),
        ">=2 clips, gated": dict(agg="median", min_clips=2, gate_frac=0.8, gate_change=0.25),
        "no colour-Doppler clips, gated": dict(agg="median", min_clips=1, gate_frac=0.8, gate_change=0.25, no_color=True),
        "requested": dict(agg=agg, min_clips=min_clips, gate_frac=gate_frac, gate_change=gate_change),
    }
    best = None
    for name, v in variants.items():
        d = dyn.copy()
        if v.get("no_color"):
            d = d[~d.color_doppler.astype(bool)]
        d = d[(d.lv_frac_frames >= v["gate_frac"]) & (d.lv_frac_change >= v["gate_change"])]
        g = d.groupby("StudyInstanceUID").ef_mean
        pred = g.agg(v["agg"]); cnt = g.size()
        pred = pred[cnt >= v["min_clips"]]
        j = pd.concat([gt.rename("ef_gt"), pred.rename("ef_dl")], axis=1).dropna()
        if len(j) < 20:
            continue
        y = (j.ef_gt < 50).astype(int); s = -j.ef_dl
        auc = roc_auc_score(y, s); ap = average_precision_score(y, s)
        lo, hi = bootstrap(y, s, roc_auc_score)
        m = agreement(j.ef_gt, j.ef_dl)
        rows.append(dict(variant=name, n=len(j), prevalence_ef_lt_50=float(y.mean()), auroc=auc, auroc_ci_low=lo, auroc_ci_high=hi, auprc=ap,
                         mae=m["mae"], r=m["r"], bias=m["bias"], **{k: v[k] for k in ("agg", "min_clips", "gate_frac", "gate_change")}))
        if best is None or auc > best[0]:
            best = (auc, name, j, y, s)
    if best:
        auc, name, j, y, s = best
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        fpr, tpr, _ = roc_curve(y, s); ax[0].plot(fpr, tpr, color="#2a6f97"); ax[0].plot([0, 1], [0, 1], "k--", lw=1)
        ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity"); ax[0].set_title(f"EchoNet-Dynamic EF<50% vs Visual EF<50%\n{name}: AUROC={auc:.3f}, n={len(j)}")
        ax[1].scatter(j.ef_gt, j.ef_dl, s=6, alpha=0.3, color="#c9184a"); ax[1].plot([10, 80], [10, 80], "k--", lw=1); ax[1].axvline(50, color="gray", ls=":"); ax[1].axhline(50, color="gray", ls=":")
        ax[1].set_xlabel("Visual EF, report (%)"); ax[1].set_ylabel("EchoNet-Dynamic EF (%)"); ax[1].set_title(f"r={stats.pearsonr(j.ef_gt, j.ef_dl)[0]:.2f}  MAE={np.mean(np.abs(j.ef_dl-j.ef_gt)):.1f}%")
        plt.tight_layout(); plt.savefig(f"{out_dir}/dynamic_ef_roc_scatter.png", dpi=130); plt.close(fig)
    tab = pd.DataFrame(rows)
    return tab, (tab.sort_values("auroc", ascending=False).iloc[0].to_dict() if len(tab) else {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/volume/echonet_eval/results")
    ap.add_argument("--out", default="/volume/echonet_eval/analysis")
    ap.add_argument("--ef_agg", default="median"); ap.add_argument("--ef_min_clips", type=int, default=1)
    ap.add_argument("--ef_gate_frac", type=float, default=0.8); ap.add_argument("--ef_gate_change", type=float, default=0.25)
    a = ap.parse_args()
    main(a.results, a.out, a.ef_agg, a.ef_min_clips, a.ef_gate_frac, a.ef_gate_change)
