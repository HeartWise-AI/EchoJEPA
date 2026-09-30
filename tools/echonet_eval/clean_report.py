"""Clean the MHI echo report export: fix unit / decimal-shift entry errors and enforce physiological ranges.

For every numeric measurement column listed in RANGES (native units of the export):
  1. value inside [lo, hi]                                   -> kept
  2. value at least 3x outside the range, and one power of ten value * 10^k (k = -5..5) lands inside [lo, hi]
     within a factor 10**0.5 of the column median and clearly nearer than any other in-range candidate
                                                             -> rescaled (e.g. AV Peak Velocity 304500 -> 304.5 cm/s = 3.045 m/s,
                                                                MV lateral e' 1100 -> 11 cm/s)
  3. anything else outside the range                         -> NaN ('out_of_range', or 'ambiguous' when several
                                                                rescalings fit); e.g. EF 97 % is blanked, not turned into 9.7
  Zero or negative values are outside the range of strictly positive quantities and become NaN.
Columns not in RANGES (identifiers, text findings, wall-motion scores, grades, indices) are left untouched.

Outputs: <out>.parquet (cleaned table), <out>_changes.csv (row-level log: column, row, original, cleaned, action)
and <out>_summary.csv (per-column counts).

Usage:
    python clean_report.py --csv df_report_2006_2026.csv --out df_report_2006_2026_clean
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

# column -> (low, high) in the export's native unit (velocities cm/s, dimensions cm, gradients mmHg, ...)
RANGES = {
    # function
    "Visually Estimated EF": (5, 90),
    "RV Systolic Pressure": (5, 150),
    "RA Pressure": (0, 25),
    "RV MPI": (0.0, 2.0),
    "TAPSE": (0.3, 4.5),
    "TV Lateral Ann s' Velocity": (2, 35),
    # velocities (cm/s)
    "TR Peak Velocity": (50, 700),
    "MV E Peak Velocity": (15, 300),
    "MN A Peak Velocity": (10, 250),
    "MV Peak Velocity": (20, 700),
    "AV Peak Velocity": (40, 700),
    "TV Peak Velocity": (20, 700),
    "PV Peak Velocity": (20, 500),
    "[MeasurementsDoppler]_[LVOTPeakVelocity]": (30, 350),
    "[MeasurementsDoppler]_[TV Peak E Velocity]": (15, 250),
    "[MeasurementMatchupCustom20]_[RVOT Peak Velocity ABS]": (20, 400),
    "MV Lateral e' Velocity": (1, 35),
    "MV Septal e' Velocity": (1, 30),
    # gradients (mmHg)
    "AV Peak Gradient": (0.1, 200),
    "AV Mean Gradient": (0.1, 130),
    "MV Peak Gradient": (0.1, 60),
    "MV Mean Gradient": (0.1, 40),
    "PV Peak Gradient": (0.1, 150),
    "PV Mean Gradient": (0.1, 80),
    "TV Peak Gradient": (0.1, 120),
    "TV Mean Gradient": (0.1, 30),
    "[MeasurementsDoppler]_[TV Peak Gradient]": (0.1, 120),
    # ratios / times
    "MV E/A": (0.1, 6),
    "MV E/e' (Lateral)": (1, 50),
    "MV E/e' (Septal)": (1, 60),
    "MV Decel Time": (0.03, 0.8),
    # valve areas (cm2), regurgitation
    "MV Area (Cont Eq VTI)": (0.2, 10),
    "AV Area (Cont Eq VTI)": (0.2, 6),
    "MR ERO (PISA)": (0.01, 2),
    "TR ERO (PISA)": (0.01, 3),
    "AR ERO Area (PISA)": (0.01, 2),
    "MR Volume (PISA)": (1, 300),
    "TR Volume (PISA)": (1, 300),
    "AR Volume (PISA)": (1, 300),
    # dimensions (cm)
    "LVID Diastole (2D)": (2, 9.5),
    "LVID Systole (2D)": (1, 8.5),
    "IVS Diastole Thickness (2D)": (0.3, 3.5),
    "LVIW Diastolic Thickness (2D)": (0.3, 3.5),
    "LA Dimension": (1.5, 9),
    "RV Basal Diastolic Dimension": (1.5, 7.5),
    "RV Dimension": (1, 7.5),
    "LVOT Diameter": (1.2, 4),
    "IVC Diameter (Exp 2D)": (0.3, 4.5),
    "Sinus of Valsalva Diameter": (1.5, 7),
    "Ao Sinotub Junction Diameter": (1.2, 6.5),
    "Prox Asc Ao Diameter": (1.5, 7.5),
    "Ao Arch Diameter": (1.2, 6.5),
    "RA Systolic Major Axis Length (4C)": (2, 12),
    "RA Systolic Minor Axis Width (4C)": (1.5, 10),
    "RA Area (4C)": (5, 60),
    # vitals
    "Visit HR": (20, 250),
    "Visit Systolic BP": (50, 280),
    "Visit Diastolic BP": (20, 160),
}
SCALES = [10.0 ** k for k in range(-5, 6) if k != 0]


def clean_series(s: pd.Series, lo: float, hi: float):
    """Rescale only gross decimal/unit shifts (>= 3x outside the range) to the power of ten that lands in range
    AND near the column median (within 10**0.5, clearly nearer than any other in-range candidate); everything
    else outside the range becomes NaN (e.g. EF 97 is implausible, not 9.7)."""
    x = pd.to_numeric(s, errors="coerce")
    out = x.copy()
    action = pd.Series("", index=s.index, dtype=object)
    ok = x.notna() & (x >= lo) & (x <= hi)
    med = float(x[ok].median()) if ok.any() else np.nan
    bad = x.notna() & ~ok
    for i in x.index[bad]:
        v = x.at[i]
        far = v > 0 and (v > 3 * hi or v < lo / 3)
        fits = sorted(((abs(np.log10(v * f / med)), v * f) for f in SCALES if far and lo <= v * f <= hi)) if np.isfinite(med) else []
        if fits and fits[0][0] < 0.5 and (len(fits) == 1 or fits[1][0] - fits[0][0] > 0.3):
            out.at[i] = fits[0][1]; action.at[i] = "rescaled"
        else:
            out.at[i] = np.nan; action.at[i] = "ambiguous" if fits else "out_of_range"
    return out, action


def main(csv, out):
    df = pd.read_csv(csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    logs, summ = [], []
    for col, (lo, hi) in RANGES.items():
        if col not in df:
            print("missing column:", col); continue
        n = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        cleaned, act = clean_series(df[col], lo, hi)
        changed = act != ""
        if changed.any():
            logs.append(pd.DataFrame({"column": col, "row": df.index[changed], "AccessionNumber": df.loc[changed, "AccessionNumber"],
                                      "original": df.loc[changed, col], "cleaned": cleaned[changed], "action": act[changed]}))
        summ.append(dict(column=col, range=f"[{lo}, {hi}]", n_values=n, rescaled=int((act == "rescaled").sum()),
                         out_of_range_to_nan=int((act == "out_of_range").sum()), ambiguous_to_nan=int((act == "ambiguous").sum()),
                         median=float(cleaned.median()), min_after=float(cleaned.min()), max_after=float(cleaned.max())))
        df[col] = cleaned
    for c in df.columns:  # parquet needs one type per column: keep mixed object columns as strings
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.to_parquet(f"{out}.parquet", index=False)
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    log.to_csv(f"{out}_changes.csv", index=False)
    s = pd.DataFrame(summ); s.to_csv(f"{out}_summary.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    print(s.to_string(index=False))
    print(f"rows {len(df):,}; values changed {len(log):,} -> {out}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(a.csv, a.out)
