# data/still_overlay_audit.py

"""What is left of the sonographer's marks on a prepared still? Pixel statistics on real stills; counts only.

`still_preprocess` masks label boxes and inpaints coloured overlays. It does not find a white or grey caliper, and a
filled label box is a flat rectangle. Two questions follow, each answered without training anything:

1. Measured status. Do simple statistics of the prepared crop tell a measured still from an unmeasured one of the
   same vendor and mode? Per feature, the AUROC of measured against unmeasured stills (0.5: the feature does not
   tell them apart). Features, all on the grey crop the encoder would read, at the screen's resolution:
   - `thin_bright`: share of pixels that are bright and much brighter than their neighbourhood (a morphological
     top-hat): thin white marks such as calipers, dotted lines and digits;
   - `flat`: share of 9 x 9 windows that are not black and have no variation: the footprint of a filled box;
   - `chromatic`: share of pixels that were coloured overlay before inpainting (what was removed, for reference).
   An unmeasured still is one of the header table (`stills_hdr_<year>.parquet`) that has no row in the measurement
   tables. The two groups may differ in what was imaged, so a high AUROC is a warning to inspect, not proof of a
   mark; a value near 0.5 is evidence that this feature carries no marker.

2. Caliper row. On measured stills with a peak velocity, is the row with the most thin bright pixels (the baseline
   apart) the row where the measured velocity lies on the crop's velocity scale? `at_value`: stills where it is,
   within 2 % of the crop's height, for the value above or below the baseline; `at_shuffled_value`: the same with
   the value of another still of the same type (chance). The top of the Doppler envelope lies on that row too, so
   an excess over chance shows that the row is findable from thin bright pixels, by a mark or by the envelope's
   edge: it bounds the leak from above, it does not prove one.

Counts from 1 to 4 are written `<5`. The output must be outside the repository.

Usage:
    python data/still_overlay_audit.py --stills <stills.csv> --split val --headers <stills_hdr_*.parquet ...> \\
        --meas <meas_*.parquet ...> --sample 2000 --out <report.json>
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from build_manifests import REPO_ROOT, ensure_outside_repo

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from src.datasets.still_preprocess import prepare_crop  # noqa: E402

BOX = ("spec_x0", "spec_y0", "spec_x1", "spec_y1")
FEATURES = ("thin_bright", "flat", "chromatic")
MODES = ("CW", "PW_TDI")


def suppressed(n):
    return "<5" if 0 < n < 5 else int(n)


def thin_bright_mask(gray, bright=200, contrast=60, size=5):
    """Pixels that are bright and stand out of their neighbourhood by `contrast` (top-hat with a `size` kernel)."""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return (gray >= bright) & (tophat >= contrast)


def flat_share(gray, window=9, floor=10):
    """Share of `window` x `window` neighbourhoods that are not black (mean above `floor`) and have no variation."""
    import cv2
    x = gray.astype(np.float32)
    mean = cv2.blur(x, (window, window))
    variance = cv2.blur(x * x, (window, window)) - mean * mean
    return float(((variance < 0.25) & (mean > floor)).mean())


def features(gray, report):
    return {"thin_bright": float(thin_bright_mask(gray).mean()), "flat": flat_share(gray),
            "chromatic": float(report["overlay_fraction"])}


def auroc(positive, negative):
    """Probability that a positive's value is above a negative's (ties count half); nan without both groups."""
    positive, negative = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    if not len(positive) or not len(negative):
        return float("nan")
    order = np.concatenate([positive, negative]).argsort(kind="mergesort")
    values = np.concatenate([positive, negative])[order]
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[start]:
            end += 1
        ranks[start:end + 1] = (start + end) / 2 + 1
        start = end + 1
    back = np.empty(len(values))
    back[order] = ranks
    return float((back[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def value_rows(value, top, bottom, height):
    """The crop rows of +value and -value on a linear velocity scale from `top` (row 0) to `bottom` (last row)."""
    span = top - bottom
    return [(top - v) / span * (height - 1) for v in (value, -value)] if span else []


def marker_row(gray, top, bottom, margin=3, minimum=3):
    """The row with the most thin bright pixels, the baseline (velocity 0) and its `margin` rows apart; None when
    no row has `minimum` of them."""
    profile = thin_bright_mask(gray).sum(axis=1).astype(float)
    baseline = value_rows(0.0, top, bottom, len(profile))
    if baseline and 0 <= baseline[0] < len(profile):
        low = max(0, int(baseline[0]) - margin)
        profile[low:int(baseline[0]) + margin + 1] = 0
    return int(profile.argmax()) if profile.max() >= minimum else None


def at_value(row, value, top, bottom, height, tolerance=0.02):
    return row is not None and any(abs(row - r) <= max(2.0, tolerance * height)
                                   for r in value_rows(value, top, bottom, height))


def prepared(row):
    from PIL import Image
    with Image.open(row["png_path"]) as image:
        screen = np.asarray(image.convert("RGB"))
    boxes = json.loads(row["text_boxes"]) if isinstance(row.get("text_boxes"), str) and row["text_boxes"] else []
    return prepare_crop(screen, tuple(int(float(row[c])) for c in BOX), boxes)


def audit(measured, unmeasured, seed=0):
    """The report (see the module docstring). `measured`: rows of the still table; `unmeasured`: header rows."""
    rows = []
    for group, table in (("measured", measured), ("unmeasured", unmeasured)):
        for row in table.to_dict("records"):
            try:
                gray, report = prepared(row)
            except Exception:
                continue
            entry = {"group": group, "vendor": row.get("manufacturer") or row.get("Manufacturer") or "",
                     "mode": row["mode"], "still_type": row.get("still_type", ""), **features(gray, report)}
            if group == "measured":
                vmax = [float(row[c]) for c in row if c.startswith("y_") and c.endswith("_VMAX")
                        and pd.notna(row[c]) and row[c] != ""]
                top, bottom = pd.to_numeric(row.get("v_top_ms"), errors="coerce"), pd.to_numeric(
                    row.get("v_bot_ms"), errors="coerce")
                if vmax and np.isfinite(top) and np.isfinite(bottom):
                    entry.update(vmax=vmax[0], top=float(top), bottom=float(bottom), height=int(gray.shape[0]),
                                 marker=marker_row(gray, float(top), float(bottom)))
            rows.append(entry)
    found = pd.DataFrame(rows)
    report = {"measured": int((found.group == "measured").sum()), "unmeasured": int((found.group == "unmeasured").sum()),
              "measured_status_auroc": {}, "caliper_row": {}}

    def status(table):
        m, u = table[table.group == "measured"], table[table.group == "unmeasured"]
        if len(m) < 20 or len(u) < 20:
            return None
        return {"measured": len(m), "unmeasured": len(u),
                **{f: round(auroc(m[f], u[f]), 3) for f in FEATURES},
                **{f"median_{f}_measured": round(float(m[f].median()), 5) for f in FEATURES},
                **{f"median_{f}_unmeasured": round(float(u[f].median()), 5) for f in FEATURES}}

    report["measured_status_auroc"]["all"] = status(found)
    for (vendor, mode), table in found.groupby(["vendor", "mode"]):
        if (entry := status(table)) is not None:
            report["measured_status_auroc"][f"{vendor}|{mode}"] = entry
    with_value = found[found.group == "measured"].dropna(subset=["vmax"]) if "vmax" in found else found.iloc[:0]
    rng = np.random.default_rng(seed)
    for name, table in [("all", with_value), *with_value.groupby("still_type")]:
        if len(table) < 5:
            continue
        table = table.reset_index(drop=True)
        shift = int(rng.integers(1, len(table))) if len(table) > 1 else 0
        other = np.roll(table.vmax.to_numpy(), shift)
        marker = [None if pd.isna(m) else int(m) for m in table.marker]
        hit = sum(at_value(m, v, t, b, h) for m, v, t, b, h in zip(marker, table.vmax, table.top, table.bottom,
                                                                    table.height))
        chance = sum(at_value(m, v, t, b, h) for m, v, t, b, h in zip(marker, other, table.top, table.bottom,
                                                                      table.height))
        report["caliper_row"][name] = {"stills": suppressed(len(table)),
                                       "with_marker_row": suppressed(sum(m is not None for m in marker)),
                                       "at_value": suppressed(hit), "at_shuffled_value": suppressed(chance)}
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stills", required=True, help="the still table (measured stills)")
    p.add_argument("--split", default="val")
    p.add_argument("--headers", nargs="+", required=True, help="stills_hdr_<year>.parquet of the still dataset")
    p.add_argument("--meas", nargs="+", required=True, help="meas_<year>.parquet: a still with a row is measured")
    p.add_argument("--sample", type=int, default=2000, help="stills per group")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    ensure_outside_repo(os.path.dirname(os.path.abspath(args.out)), "--out")
    measured = pd.read_csv(args.stills, dtype={"still_id": str, "study_id": str}, keep_default_na=False)
    measured = measured[measured.split.astype(str) == args.split]
    measured = measured.sample(min(args.sample, len(measured)), random_state=args.seed)
    with_rows = set()
    for path in args.meas:
        with_rows |= set(pd.read_parquet(path, columns=["png_path"]).png_path)
    headers = pd.concat([pd.read_parquet(path) for path in args.headers], ignore_index=True)
    headers = headers[headers["mode"].isin(MODES) & ~headers.png_path.isin(with_rows)].dropna(subset=list(BOX))
    # the same number per (vendor, mode) as the measured sample, where the header table has them
    wanted = measured.groupby(["manufacturer", "mode"]).size()
    parts = []
    for (vendor, mode), n in wanted.items():
        pool = headers[(headers.Manufacturer == vendor) & (headers["mode"] == mode)]
        parts.append(pool.sample(min(int(n), len(pool)), random_state=args.seed))
    unmeasured = pd.concat(parts, ignore_index=True) if parts else headers.iloc[:0]
    report = audit(measured, unmeasured, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
