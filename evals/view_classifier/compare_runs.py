# evals/view_classifier/compare_runs.py

"""Two or more snapshots of the comparison side by side: the same labels and folds, another input (e.g. s1 vs s2).

Each snapshot is one config (its `paths.runs`). Every snapshot must score the same labelled clips with the same folds
(checked), so their out-of-fold predictions are paired clip by clip. Outputs, in `--out`:

- `summary_side_by_side.csv`: every snapshot's `report/summary.csv` rows (step 07 run for each config first), with a
  `snapshot` column (the name of its runs folder);
- `paired_across_snapshots.csv`: for each run (arm, pseudo) that two snapshots both have, the later snapshot minus the
  earlier one in accuracy and macro-F1, with paired bootstrap intervals, at the new granularity and collapsed to the
  current classifier's 20 classes;
- `compare.md`: the headline rows (all clips) and the paired differences.

Nothing is written into a snapshot's own runs folder except under `--out`, which may not lie inside the first
snapshot's folder. Nothing written names a patient.

    python -m evals.view_classifier.compare_runs --config config.yaml --config config_s2.yaml --out DIR
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import metrics as vm
from evals.view_classifier.io import read_folds

HEADLINE = ("accuracy", "macro_f1", "balanced_accuracy", "mean_auroc", "macro_f1_all")


def snapshot_name(cfg):
    return os.path.basename(os.path.normpath(cfg["paths"]["runs"]))


def oof_runs(runs_dir):
    """{(arm, pseudo): out-of-fold table} of every finished run of a snapshot."""
    found = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "arm_*", "pseudo_*", "oof_predictions.csv"))):
        parts = path.split(os.sep)
        found[(parts[-3], parts[-2].removeprefix("pseudo_"))] = pd.read_csv(path, dtype={"clip_id": str})
    return found


def same_labels(configs):
    """The labels table of the first config, after checking that every config has the same classes, clips, labels
    and folds."""
    first = None
    for cfg in configs:
        classes, parent = vc.from_config(cfg)
        labels = read_folds(cfg["paths"]["folds"], classes)[["clip_id", "y", "fold"]]
        labels = labels.sort_values("clip_id").reset_index(drop=True)
        key = (tuple(classes), tuple(parent[c] for c in classes))
        if first is None:
            first = (key, labels)
        elif key != first[0] or not labels.equals(first[1]):
            raise ValueError(f"{snapshot_name(cfg)} has other classes, labels or folds than "
                             f"{snapshot_name(configs[0])}: its predictions cannot be paired.")
    return first[1]


def predictions(labels, oof, classes, parent):
    """(new-class argmax, old-class argmax) per labelled clip, in the labels' order."""
    t = labels[["clip_id"]].merge(oof, on="clip_id", how="left", validate="one_to_one")
    probs = t[[f"p_{c}" for c in classes]].to_numpy(dtype=float)
    if np.isnan(probs).any():
        raise ValueError("A run does not predict every labelled clip.")
    old = vm.collapse_probabilities(probs, classes, parent, vc.OLD_CLASSES)
    return probs.argmax(axis=1), old.argmax(axis=1)


def paired_rows(configs, labels, samples, seed):
    classes, parent = vc.from_config(configs[0])
    y_new = labels.y.to_numpy()
    y_old = np.asarray([vc.OLD_CLASSES.index(parent[classes[i]]) for i in y_new])
    runs = [(snapshot_name(c), oof_runs(c["paths"]["runs"])) for c in configs]
    rows = []
    for i, (name_a, runs_a) in enumerate(runs):
        for name_b, runs_b in runs[i + 1:]:
            for key in sorted(set(runs_a) & set(runs_b)):
                new_a, old_a = predictions(labels, runs_a[key], classes, parent)
                new_b, old_b = predictions(labels, runs_b[key], classes, parent)
                for granularity, truth, a, b, n in (("new", y_new, new_a, new_b, len(classes)),
                                                    ("old", y_old, old_a, old_b, len(vc.OLD_CLASSES))):
                    d = vm.paired_difference(truth, b, a, n, samples, seed)     # later snapshot minus earlier
                    rows.append({"arm": key[0], "pseudo": key[1], "granularity": granularity,
                                 "later": name_b, "earlier": name_a,
                                 "accuracy_later": vm.accuracy(truth, b), "accuracy_earlier": vm.accuracy(truth, a),
                                 **d})
    return pd.DataFrame(rows)


def side_by_side(configs):
    frames = []
    for cfg in configs:
        path = os.path.join(cfg["paths"]["runs"], "report", "summary.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{snapshot_name(cfg)} has no report/summary.csv: run step 07 with its config.")
        frames.append(pd.read_csv(path).assign(snapshot=snapshot_name(cfg)))
    out = pd.concat(frames, ignore_index=True)
    return out[["snapshot"] + [c for c in out.columns if c != "snapshot"]]


def _ci(row, name):
    v, lo, hi = row.get(name), row.get(f"{name}_low"), row.get(f"{name}_high")
    if v is None or pd.isna(v):
        return ""
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if not (pd.isna(lo) or pd.isna(hi)) else f"{v:.3f}"


def markdown(summary, pairs):
    lines = ["# Snapshots side by side", "",
             "Same labels, folds and selection rule; only what the snapshot's config changes (e.g. the input). "
             "Pooled out-of-fold predictions; 95% bootstrap intervals over clips.", "",
             "## All clips", ""]
    head = summary[summary.subset == "all"].sort_values(["granularity", "arm", "pseudo", "snapshot"])
    cols = ["snapshot", "arm", "pseudo", "granularity", "n"] + [c for c in HEADLINE if c in head.columns]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in head.iterrows():
        cells = [str(r[c]) for c in cols[:5]] + [_ci(r, c) for c in cols[5:]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences (later snapshot minus earlier, same run)", ""]
    if pairs.empty:
        lines.append("No run is present in two snapshots yet.")
    else:
        cols = ["arm", "pseudo", "granularity", "later", "earlier", "n"]
        lines.append("| " + " | ".join(cols + ["accuracy diff", "macro-F1 diff"]) + " |")
        lines.append("|" + "---|" * (len(cols) + 2))
        for _, r in pairs.iterrows():
            cells = [str(r[c]) for c in cols] + [
                f"{r.accuracy_diff:+.3f} [{r.accuracy_diff_low:+.3f}, {r.accuracy_diff_high:+.3f}]",
                f"{r.macro_f1_diff:+.3f} [{r.macro_f1_diff_low:+.3f}, {r.macro_f1_diff_high:+.3f}]"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def compare(configs, out):
    if len(configs) < 2:
        raise ValueError("Name at least two configs.")
    names = [snapshot_name(c) for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"Two configs share a runs folder name: {names}.")
    first = os.path.realpath(configs[0]["paths"]["runs"])
    if os.path.commonpath([first, os.path.realpath(out)]) == first:
        raise ValueError("--out is inside the first snapshot's runs folder; write the comparison elsewhere.")
    labels = same_labels(configs)
    r = configs[0].get("report", {})
    summary = side_by_side(configs)
    pairs = paired_rows(configs, labels, int(r.get("bootstrap", 1000)), int(r.get("seed", 0)))
    os.makedirs(out, exist_ok=True)
    summary.to_csv(os.path.join(out, "summary_side_by_side.csv"), index=False)
    pairs.to_csv(os.path.join(out, "paired_across_snapshots.csv"), index=False)
    with open(os.path.join(out, "compare.md"), "w") as f:
        f.write(markdown(summary, pairs))
    return summary, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", action="append", required=True, help="one per snapshot, earliest first")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    configs = []
    for path in args.config:
        with open(path) as f:
            configs.append(yaml.safe_load(f))
    summary, pairs = compare(configs, args.out)
    print(f"{len(summary)} summary rows, {len(pairs)} paired differences -> {args.out}")


if __name__ == "__main__":
    main()
