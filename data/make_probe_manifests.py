# data/make_probe_manifests.py

"""Create frozen-probe manifests for LVEF regression.

Selects requested views from `videos.csv`, z-scores LVEF using training-split statistics, and
writes headerless train/val/test manifests as: `<video_path> <z-scored LVEF>`. Also writes
`probe_info.json` with normalization statistics and output metadata.

Example:
    python data/make_probe_manifests.py --manifests <issue_6_output_dir> --out-dir <probe_dir> --views A4C
"""

import argparse
import json
import os

import pandas as pd

from build_manifests import SPLITS, ensure_outside_repo, sha256


def parse_args(argv=None):
    """Defines the command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", required=True, help="Output directory of `build_manifests.py` (`videos.csv`).")
    p.add_argument("--out-dir", required=True, help="Where the probe manifests go; must be outside this repository.")
    p.add_argument("--views", nargs="+", default=["A4C"], help="Exact view labels to keep (default: A4C).")
    p.add_argument("--overwrite", action="store_true", help="Replace the files of a non-empty `--out-dir`.")
    return p.parse_args(argv)


def make(videos, views):
    """Select the videos of `views` and z-score their study's EF with the train split's statistics.

    Returns ({split: DataFrame[video_path, label, target]}, mean, std).
    """
    # Keep only the requested views.
    v = videos[videos.view.isin(views)]
    # Make sure every selected video has an EF.
    if v.label.isna().any():
        raise ValueError(f"{int(v.label.isna().sum())} videos of {views} have no EF label.")
    # Make sure paths contain no spaces.
    if (v.video_path.str.contains(r"\s", regex=True)).any():
        raise ValueError("Video paths with whitespace cannot be written to a space-separated manifest.")

    # Compute EF normalization using training data only.
    train = v[v.split == "train"]
    if train.empty:
        raise ValueError(f"No training video of {views}.")
    # Population std (ddof=0), as a standard scaler computes it.
    mean, std = float(train.label.mean()), float(train.label.std(ddof=0))
    # If std == 0, error: division by zero.
    if not std > 0:
        raise ValueError(f"The training EF values of {views} do not vary (std {std}).")
    splits = {}
    for split in SPLITS:
        rows = v[v.split == split].sort_values("video_path")
        splits[split] = rows.assign(target=(rows.label - mean) / std)
    return splits, mean, std


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
        raise SystemExit(f"--out-dir {args.out_dir} is not empty; pass --overwrite to replace its files.")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the source manifest.
    source = os.path.join(args.manifests, "videos.csv")
    # Read `patient_id` and `study_id` as strings.
    videos = pd.read_csv(source, dtype={"patient_id": str, "study_id": str})
    splits, mean, std = make(videos, args.views)

    # For reproducibility, create metadata describing exactly how the manifests were made.
    info = {
        "source": {"videos.csv": sha256(source)},  # records exactly which `videos.csv` was used.
        "views": args.views,
        "target_mean": mean,                       # used to turn z-scored (standardized) EF back into the original scale.
        "target_std": std,                         # used to turn z-scored (standardized) EF back into the original scale.
        "splits": {},
        "outputs": {},
    }
    for split, rows in splits.items():
        name = f"{split}.csv"
        path = os.path.join(args.out_dir, name)
        # Core output.
        rows[["video_path", "target"]].to_csv(path, sep=" ", header=False, index=False)
        # Info for each split.
        info["splits"][split] = {
            "videos": len(rows),
            "studies": int(rows.study_id.nunique()),
            "patients": int(rows.patient_id.nunique()),
            "ef_mean": float(rows.label.mean()) if len(rows) else None,
        }
        info["outputs"][name] = {"rows": len(rows), "sha256": sha256(path)}
        print(f"[probe]  {split:5s} {len(rows):7d} videos  {rows.study_id.nunique():5d} studies  -> {path}")
    with open(os.path.join(args.out_dir, "probe_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[probe]  views {args.views}; train EF mean {mean:.4f}, std {std:.4f} "
          "(data.target_mean / data.target_std of the eval config).")


if __name__ == "__main__":
    main()
