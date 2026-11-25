# /data/validate_acquisition.py

"""Check the header acquisition flags of `scan_acquisition.py` against the converted videos.

For a stratified sample of indexed DICOMs (acquisition category x year x brand x view by
default, so color-positive, negative and unknown cases are all drawn), a few frames of each
converted video (`avi_path`, joined on `dcm_path`) are decoded and colour statistics that
tolerate coloured overlays are computed. Whether colour seen in the DICOM header (color
flow) is still present in the video tests both the header flag and that conversion and
cropping preserved the acquisition information.

Outputs, all under --out-dir (outside the repository):
    review_sheet.csv            one row per sampled video, WITH paths, for local review only
    contact_sheets/*.png        frames of disagreement cases (local review only)
    validation_summary.md/json  aggregates only: distributions, disagreement tables, thresholds

Example:
    python data/validate_acquisition.py --index <acq_cache_dir>/acquisition_index.parquet \\
        --metadata <master_metadata.parquet> --out-dir <review_dir> --per-stratum 2 --seed 0

See docs/acquisition_index.md for the statistics and how to calibrate the thresholds.
"""

import argparse
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import pandas as pd

import scan_acquisition as sa
from build_manifests import ensure_outside_repo, git_commit

DEFAULT_STRATA = ("acq_category", "year", "brand", "predicted_class")
STATS = ("full_fraction_median", "full_fraction_max", "center_fraction_median", "center_fraction_p90",
         "center_fraction_std", "outside_fraction_median", "center_dynamic")
# Upper bound on frames held when a video does not report its length.
MAX_BUFFERED_FRAMES = 3000


class Thresholds(NamedTuple):
    """When a pixel counts as coloured, and when a video counts as showing colour.

    chroma:  max(R,G,B) - min(R,G,B) of a coloured pixel, 0-255. Gray-scale video decodes to a
             chroma of a few units (compression noise); color flow is far above.
    value:   minimum max(R,G,B) of a coloured pixel, so dark noise never counts.
    center:  side of the central box as a fraction of the frame height / width. Overlays
             (ECG trace, colour bar, labels, logos) mostly sit at the borders, outside it.
    min_center_fraction: the 90th percentile over sampled frames of the coloured share of
             the central box must reach this for colour to be observed.
    min_dynamic: mean share of central pixels whose coloured state changes between
             consecutive sampled frames. Color flow changes over the cardiac cycle; a
             static coloured marker inside the box does not. Not applied with < 2 frames.
    """

    chroma: int = 40
    value: int = 40
    center: float = 0.6
    min_center_fraction: float = 0.01
    min_dynamic: float = 0.002


DEFAULT_THRESHOLDS = Thresholds()


# --------------------------------------------------------------------------- #
#  Decoding & colour statistics
# --------------------------------------------------------------------------- #


def sample_indices(n, k):
    """k frame indices spread evenly over n frames (all of them when n <= k)."""
    if n <= 0:
        return []
    return sorted(set(np.linspace(0, n - 1, num=min(k, n)).round().astype(int).tolist()))


def read_frames(path, k=8):
    """Decode about k evenly spaced frames (BGR, uint8) of one video. Returns (frames, status).

    Frames are grabbed sequentially rather than seeked, which some AVI codecs do inexactly.
    """
    import cv2

    if not isinstance(path, str) or not os.path.isfile(path):
        return None, "missing"
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None, "unopenable"
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        if n > 0:
            want = sample_indices(n, k)
            wanted = set(want)
            for i in range(want[-1] + 1):
                if not cap.grab():
                    break
                if i in wanted:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        frames.append(frame)
        else:
            while len(frames) < MAX_BUFFERED_FRAMES:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
            frames = [frames[i] for i in sample_indices(len(frames), k)]
    finally:
        cap.release()
    if not frames:
        return None, "no_frames"
    frames = [f if f.ndim == 3 else np.repeat(f[..., None], 3, axis=-1) for f in frames]
    if len({f.shape for f in frames}) > 1:
        return None, "inconsistent_frames"
    return np.stack(frames), "ok"


def color_mask(frames, th):
    """Pixels with high chroma and enough brightness, for frames of shape (n, H, W, 3)."""
    hi, lo = frames.max(axis=-1).astype(np.int16), frames.min(axis=-1).astype(np.int16)
    return ((hi - lo) >= th.chroma) & (hi >= th.value)


def center_box(h, w, frac):
    dh, dw = int(round(h * (1 - frac) / 2)), int(round(w * (1 - frac) / 2))
    return slice(dh, h - dh), slice(dw, w - dw)


def color_stats(frames, th=DEFAULT_THRESHOLDS):
    """Colour statistics of one video's sampled frames (n, H, W, 3), uint8."""
    mask = color_mask(frames, th)
    n, h, w = mask.shape
    rows, cols = center_box(h, w, th.center)
    center = mask[:, rows, cols]
    full = mask.mean(axis=(1, 2))
    inner = center.mean(axis=(1, 2)) if center.size else np.zeros(n)
    outside_area = h * w - center.shape[1] * center.shape[2]
    outside = ((mask.sum(axis=(1, 2)) - center.sum(axis=(1, 2))) / outside_area) if outside_area else np.zeros(n)
    dynamic = float(np.mean(center[1:] != center[:-1])) if n > 1 and center.size else float("nan")
    return {
        "n_frames_sampled": int(n), "height": int(h), "width": int(w),
        "full_fraction_median": float(np.median(full)), "full_fraction_max": float(full.max()),
        "center_fraction_median": float(np.median(inner)), "center_fraction_p90": float(np.percentile(inner, 90)),
        "center_fraction_std": float(inner.std()), "outside_fraction_median": float(np.median(outside)),
        "center_dynamic": dynamic,
    }


def observed_color(stats, th=DEFAULT_THRESHOLDS):
    """Whether a video shows colour in its central box (see Thresholds); None if undecoded."""
    if stats is None or stats.get("n_frames_sampled") is None:
        return None
    if stats["center_fraction_p90"] < th.min_center_fraction:
        return False
    return stats["n_frames_sampled"] < 2 or stats["center_dynamic"] >= th.min_dynamic


def agreement(has_color_flow, observed):
    """Header color-flow flag vs colour observed in the video."""
    if observed is None:
        return "video_undecodable"
    if pd.isna(has_color_flow):
        return "header_unknown"
    if bool(has_color_flow):
        return "agree_color" if observed else "header_color_missing_in_video"
    return "video_color_without_header_color" if observed else "agree_no_color"


DISAGREEMENTS = ("header_color_missing_in_video", "video_color_without_header_color")


def measure_video(path, k=8, th=DEFAULT_THRESHOLDS):
    frames, status = read_frames(path, k)
    stats = color_stats(frames, th) if frames is not None else {}
    return {"decode_status": status, **stats}


def contact_sheet(frames, th, out_path, height=128, max_frames=6):
    """Sampled frames (top row, central box outlined) over their coloured-pixel masks (bottom row)."""
    import cv2

    frames = frames[np.asarray(sample_indices(len(frames), max_frames))]
    mask = color_mask(frames, th)
    rows, cols = center_box(frames.shape[1], frames.shape[2], th.center)
    scale = height / frames.shape[1]
    width = max(1, int(round(frames.shape[2] * scale)))
    top, bottom = [], []
    for frame, m in zip(frames, mask):
        f = frame.copy()
        cv2.rectangle(f, (cols.start, rows.start), (cols.stop - 1, rows.stop - 1), (255, 255, 255), 1)
        top.append(cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA))
        mm = np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=-1)
        bottom.append(cv2.resize(mm, (width, height), interpolation=cv2.INTER_NEAREST))
    sheet = np.vstack([np.hstack(top), np.hstack(bottom)])
    if not cv2.imwrite(out_path, sheet):
        raise OSError(f"could not write the contact sheet {os.path.basename(out_path)}.")


# --------------------------------------------------------------------------- #
#  Sample selection
# --------------------------------------------------------------------------- #


def join_index(index, metadata, strata):
    """Index rows joined to their converted video (`avi_path`) and the metadata strata.

    dcm_path must be unique among the joined metadata rows (`scan_acquisition.metadata_for`),
    and so must avi_path: a video under two DICOMs would make the comparison ambiguous.
    """
    meta_cols = [s for s in strata if s not in index.columns]
    for extra in ("model", "predicted_class", "year", "brand"):
        if extra not in meta_cols:
            meta_cols.append(extra)
    names = sa.table_columns(metadata)
    available = [c for c in meta_cols if (c if c != "year" else "date") in names]
    missing = [s for s in strata if s not in index.columns and s not in available]
    if missing:
        raise ValueError(f"cannot stratify by {missing}: neither an index nor a metadata column.")
    if "avi_path" not in names:
        raise ValueError("the metadata has no avi_path column.")
    meta = sa.metadata_for(metadata, set(index.dcm_path), available + ["avi_path"])
    joined = index.merge(meta, on="dcm_path", how="inner")
    counts = {"index_rows": int(len(index)), "joined_to_metadata": int(len(joined)),
              "without_avi_path": int(joined.avi_path.isna().sum())}
    joined = joined[joined.avi_path.notna()]
    dup = joined.avi_path.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{joined.loc[dup, 'avi_path'].nunique()} avi_path values belong to more than one DICOM.")
    counts["with_avi_path"] = int(len(joined))
    return joined.reset_index(drop=True), counts


def marginals(df, cols):
    return {c: sa.value_counts(df[c]) for c in cols if c in df}


# --------------------------------------------------------------------------- #
#  Validation run
# --------------------------------------------------------------------------- #


def distribution(values):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return {"n": 0}
    q = v.quantile([0.1, 0.5, 0.9])
    return {"n": int(len(v)), "mean": float(v.mean()), "p10": float(q[0.1]), "median": float(q[0.5]),
            "p90": float(q[0.9]), "max": float(v.max())}


def run_validation(index, metadata, out_dir, strata=DEFAULT_STRATA, per_stratum=2, seed=0, max_videos=None,
                   frames=8, th=DEFAULT_THRESHOLDS, contact_sheets=50, workers=4, log=print):
    """Sample, decode, compare, and write the review sheet, contact sheets and the summary."""
    strata = list(strata)
    joined, counts = join_index(index, metadata, strata)
    sample = sa.stratified_sample(joined, strata, seed, n=max_videos, per_stratum=per_stratum)
    n_strata = int(joined[strata].astype("string").fillna(sa.MISSING).drop_duplicates().shape[0]) if len(joined) else 0
    log(f"[sample] {len(sample)} videos from {n_strata} strata ({', '.join(strata)}), "
        f"up to {per_stratum} per stratum, seed {seed}.")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        measured = list(pool.map(lambda p: measure_video(p, frames, th), sample.avi_path))
    m = pd.DataFrame(measured, index=sample.index)
    for col in ("n_frames_sampled", "height", "width", *STATS):
        if col not in m:
            m[col] = np.nan
    sheet = pd.concat([sample, m], axis=1)
    obs = [observed_color(r if r["decode_status"] == "ok" else None, th) for r in sheet.to_dict("records")]
    sheet["color_observed"] = pd.array(obs, dtype="boolean")
    sheet["agreement"] = [agreement(h, o) for h, o in zip(sheet.has_color_flow, obs)]
    sheet.insert(0, "review_id", np.arange(len(sheet)))
    sheet["contact_sheet"] = pd.Series(pd.NA, index=sheet.index, dtype="string")

    os.makedirs(out_dir, exist_ok=True)
    todo = sheet[sheet.agreement.isin(DISAGREEMENTS)].head(contact_sheets)
    if len(todo):
        os.makedirs(os.path.join(out_dir, "contact_sheets"), exist_ok=True)
    for i, row in todo.iterrows():
        decoded, _ = read_frames(row.avi_path, frames)
        if decoded is None:
            continue
        # The file name is the review_id only: no identifier leaves the review sheet.
        name = f"review_{row.review_id:05d}.png"
        contact_sheet(decoded, th, os.path.join(out_dir, "contact_sheets", name))
        sheet.loc[i, "contact_sheet"] = f"contact_sheets/{name}"

    context = [c for c in ("year", "brand", "model", "predicted_class") if c in sheet]
    front = ["review_id", "dcm_path", "avi_path", *context,
             "acq_category", "acq_confidence", "acq_rule", *sa.FLAGS, "region_data_types", "region_spatial_formats",
             "photometric_interpretation", "decode_status", "n_frames_sampled", "height", "width", *STATS,
             "color_observed", "agreement", "contact_sheet"]
    review = sheet[front].assign(reviewer_color_present="", reviewer_notes="")
    review.to_csv(os.path.join(out_dir, "review_sheet.csv"), index=False)

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "scanner_versions_in_sample": sa.value_counts(sheet.scanner_version),
        "decoder": f"opencv {__import__('cv2').__version__}",
        "thresholds": th._asdict(), "frames_per_video": frames,
        "sampling": {"strata": strata, "per_stratum": per_stratum, "max_videos": max_videos, "seed": seed,
                     "strata_available": n_strata, "videos_sampled": int(len(sheet)), **counts},
        "available": marginals(joined, ["acq_category", "year", "brand", "predicted_class"]),
        "sampled": marginals(sheet, ["acq_category", "year", "brand", "predicted_class"]),
        "decode_status": sa.value_counts(sheet.decode_status),
        "agreement": sa.value_counts(sheet.agreement),
        "header_color_flow_vs_video": sa.table_json(pd.crosstab(
            sheet.has_color_flow.astype("string").fillna("unknown").rename("header has_color_flow"),
            sheet.color_observed.astype("string").fillna("undecodable").rename("colour in video"))),
        "agreement_by_category": sa.table_json(pd.crosstab(sheet.acq_category.rename("acq_category"),
                                                           sheet.agreement.rename("agreement"))),
        "statistics_by_category": {cat: {s: distribution(g[s]) for s in STATS}
                                   for cat, g in sheet[sheet.decode_status == "ok"].groupby("acq_category")},
        "contact_sheets_written": int(sheet.contact_sheet.notna().sum()),
    }
    ok = sheet[(sheet.decode_status == "ok") & sheet.has_color_flow.fillna(False).astype(bool)]
    summary["color_flow_preserved"] = {"header_color_flow_decoded": int(len(ok)),
                                       "colour_observed": int(ok.color_observed.fillna(False).sum())}
    with open(os.path.join(out_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "validation_summary.md"), "w") as f:
        f.write(summary_markdown(summary))
    log(f"[valid]  agreement {summary['agreement']}; {summary['contact_sheets_written']} contact sheets.")
    return review, summary


def number(x):
    """Counts as integers, statistics to 4 significant digits."""
    if isinstance(x, (float, np.floating)) and (not np.isfinite(x) or x != int(x)):
        return f"{x:.4g}"
    return int(x)


def summary_markdown(s):
    th = s["thresholds"]
    pres = s["color_flow_preserved"]
    lines = [
        "# Acquisition flags vs converted videos", "",
        "Aggregate counts only: no file paths, patient or study identifiers. The row-level review",
        "sheet and contact sheets stay in the local output directory.", "",
        f"- Created (UTC): {s['created_utc']}; code version: {s['git_commit'] or 'unknown'}",
        f"- Scanner versions in the sample: {s['scanner_versions_in_sample']}; decoder: {s['decoder']}",
        f"- Sample: up to {s['sampling']['per_stratum']} videos per stratum of {', '.join(s['sampling']['strata'])}"
        f" (seed {s['sampling']['seed']}, cap {s['sampling']['max_videos']}): "
        f"{s['sampling']['videos_sampled']} videos "
        f"from {s['sampling']['strata_available']} strata; {s['sampling']['with_avi_path']} of "
        f"{s['sampling']['index_rows']} index rows have a converted video.",
        f"- {s['frames_per_video']} evenly spaced frames per video. A pixel is coloured when "
        f"max-min channel >= {th['chroma']} and max channel >= {th['value']}; the central box spans "
        f"{th['center']:.0%} of the height and width.",
        f"- Colour observed: 90th percentile (over frames) of the coloured share of the central box "
        f">= {th['min_center_fraction']}, and (with >= 2 frames) a mean frame-to-frame change of the "
        f"central coloured mask >= {th['min_dynamic']}.",
        "", "## Conversion preserves color flow", "",
        f"Of {pres['header_color_flow_decoded']} decoded videos whose DICOM header has a color-flow region, "
        f"{pres['colour_observed']} show colour in the central box of the converted video. Colour that the "
        "header records but the video lacks points to a gray-scale conversion, a crop, or a header flag "
        "that the image does not bear out; those cases have contact sheets for review.",
    ]

    def table(title, d, name):
        nonlocal lines
        df = pd.DataFrame(d).T.fillna(0) if d and isinstance(next(iter(d.values())), dict) else \
            pd.Series(d, name="videos").to_frame()
        df = df.apply(lambda col: col.map(number))
        lines += ["", f"## {title}", "", sa.md_table(df.rename_axis(name))]

    table("Agreement", s["agreement"], "agreement")
    table("Header color-flow flag vs colour in the video", s["header_color_flow_vs_video"], "header has_color_flow")
    table("Agreement by acquisition category", s["agreement_by_category"], "acq_category")
    table("Decoding", s["decode_status"], "decode_status")
    for cat, stats in s["statistics_by_category"].items():
        table(f"Statistics: {cat}", stats, "statistic")
    for col, d in s["sampled"].items():
        avail = s["available"].get(col, {})
        df = pd.DataFrame({"available": pd.Series(avail), "sampled": pd.Series(d)}).fillna(0).astype(int)
        lines += ["", f"## Sample by {col}", "", sa.md_table(df.rename_axis(col))]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = DEFAULT_THRESHOLDS
    p.add_argument("--index", required=True, help="acquisition_index.parquet or a scan cache directory.")
    p.add_argument("--metadata", required=True, help="master_metadata.parquet (for avi_path and the strata).")
    p.add_argument("--out-dir", required=True, help="Review sheet, contact sheets, summary; outside the repository.")
    p.add_argument("--strata", default=",".join(DEFAULT_STRATA), help="Comma-separated (default %(default)s).")
    p.add_argument("--per-stratum", type=int, default=2, help="Videos per stratum (default 2).")
    p.add_argument("--max-videos", type=int, help="Optional cap, spread evenly over the strata.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=8, help="Frames decoded per video (default 8).")
    p.add_argument("--chroma-threshold", type=int, default=d.chroma)
    p.add_argument("--value-threshold", type=int, default=d.value)
    p.add_argument("--center-fraction", type=float, default=d.center)
    p.add_argument("--min-center-fraction", type=float, default=d.min_center_fraction)
    p.add_argument("--min-dynamic", type=float, default=d.min_dynamic)
    p.add_argument("--contact-sheets", type=int, default=50, help="At most this many PNGs of disagreements.")
    p.add_argument("--workers", type=int, default=4, help="Decoding threads (default 4).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    if not 0 < args.center_fraction <= 1:
        raise SystemExit("--center-fraction must be in (0, 1].")
    th = Thresholds(args.chroma_threshold, args.value_threshold, args.center_fraction, args.min_center_fraction,
                    args.min_dynamic)
    run_validation(sa.load_index(args.index), args.metadata, args.out_dir, sa.parse_strata(args.strata),
                   args.per_stratum, args.seed, args.max_videos, args.frames, th, args.contact_sheets, args.workers)
    print("[out]    review_sheet.csv (paths: local review only), contact_sheets/, validation_summary.md/.json.")


if __name__ == "__main__":
    main()
