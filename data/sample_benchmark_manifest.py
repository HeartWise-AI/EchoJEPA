# /data/sample_benchmark_manifest.py

"""Draw a representative video sample from the archive metadata for loader and training throughput
benchmarks, so the measurement reads many distinct files (cold storage, all years, scanners and
views) instead of cycling a small warm cache.

Allocation is proportional to the archive's (year, vendor, view family) strata, with a floor so small
strata (rare scanners) are still read. Only training patients of the archive's patient split
(`patient_splits.csv` of `data/build_manifests.py`) are sampled: a patient the split does not list, or
lists under val or test, is left out, so no held-out patient is used even for a throwaway run.

Example:
    python data/sample_benchmark_manifest.py --metadata <snapshot.parquet> --splits <patient_splits.csv> \\
        --n 24000 --out-dir <benchmark_dir>

Writes `sample.csv` (loader manifest, `<video_path> 0`), `sample_dicom_paths.txt` (for the header-only
acquisition scan), `sample_videos.csv` (stratum per video) and `sample_info.json` (aggregate counts).
"""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

from build_manifests import ensure_outside_repo, git_commit, normalize_id, sha256

TTE = "ECHO CARDIAQUE AVEC DOPPLER"
STRATA = ["year", "vendor", "view_family"]


def allocate(sizes, n, floor):
    """Videos per stratum: proportional to its size, at least `floor` (or all of it when smaller), summing to n."""
    base = np.minimum(sizes, floor)
    rest = n - base.sum()
    if rest < 0:
        raise ValueError(f"--n {n} is below the per-stratum floor total {base.sum()}; lower --floor.")
    room = sizes - base
    share = rest * room / room.sum()
    extra = np.floor(share).astype(int)
    # Hand out the rounding remainder to the largest fractional parts (ties: stratum order).
    left = int(rest - extra.sum())
    order = np.argsort(-(share - extra), kind="stable")[:left]
    extra[order] += 1
    return base + np.minimum(extra, room)


def sample(meta, n, floor, seed):
    meta = meta.sort_values("avi_path").reset_index(drop=True)
    sizes = meta.groupby(STRATA).size()
    counts = pd.Series(allocate(sizes.to_numpy(), n, floor), index=sizes.index)
    rng = np.random.default_rng(seed)
    parts = []
    for key, rows in meta.groupby(STRATA, sort=True):
        k = int(counts[key])
        if k:
            parts.append(rows.iloc[np.sort(rng.choice(len(rows), size=k, replace=False))])
    return pd.concat(parts).sort_values("avi_path").reset_index(drop=True), counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True, help="The archive's patient_splits.csv; only its train patients are sampled.")
    p.add_argument("--n", type=int, default=24000)
    p.add_argument("--floor", type=int, default=20, help="Minimum videos per (year, vendor, view family) stratum.")
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)

    cols = ["mrn", "study_type", "avi_status", "avi_path", "dcm_path", "predicted_class", "date", "brand"]
    meta = pd.read_parquet(args.metadata, columns=cols)
    meta = meta[(meta.study_type == TTE) & (meta.avi_status == "success") & meta.avi_path.notna()]
    splits = pd.read_csv(args.splits, dtype=str)
    if "in_archive" not in splits:
        raise SystemExit("--splits must be the archive's patient_splits.csv (a one-study studies.csv lists only part "
                         "of the archive's patients, and the others' splits are unknown).")
    train = set(normalize_id(splits.loc[splits.split == "train", "patient_id"]))
    meta = meta[normalize_id(meta.mrn).isin(train)]
    meta = meta.assign(year=meta.date.str[:4], vendor=meta.brand.fillna("<missing>"),
                       view_family=meta.predicted_class.fillna("<missing>").str.split("_").str[0])
    chosen, counts = sample(meta, args.n, args.floor, args.seed)

    chosen[["avi_path"]].assign(label=0).to_csv(os.path.join(args.out_dir, "sample.csv"), sep=" ",
                                                 header=False, index=False)
    chosen[["avi_path", "dcm_path", *STRATA, "predicted_class"]].to_csv(
        os.path.join(args.out_dir, "sample_videos.csv"), index=False)
    with open(os.path.join(args.out_dir, "sample_dicom_paths.txt"), "w") as f:
        f.write("\n".join(chosen.dcm_path) + "\n")
    info = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(), "seed": args.seed, "n": int(len(chosen)), "floor": args.floor,
        "patients": "train patients of --splits only", "eligible_videos": int(len(meta)),
        "inputs": {"metadata_sha256": sha256(args.metadata), "splits_sha256": sha256(args.splits)},
        "by_year": {k: int(v) for k, v in chosen.year.value_counts().sort_index().items()},
        "by_vendor": {k: int(v) for k, v in chosen.vendor.value_counts().items()},
        "by_view_family": {k: int(v) for k, v in chosen.view_family.value_counts().items()},
        "strata": int((counts > 0).sum()),
        "outputs": {f: sha256(os.path.join(args.out_dir, f)) for f in ("sample.csv", "sample_dicom_paths.txt")},
    }
    with open(os.path.join(args.out_dir, "sample_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[sample] {info['n']} of {info['eligible_videos']} eligible videos from {info['strata']} strata; "
          f"vendors {info['by_vendor']}.")


if __name__ == "__main__":
    main()
