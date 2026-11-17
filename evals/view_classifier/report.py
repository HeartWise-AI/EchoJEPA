# evals/view_classifier/report.py

"""The comparison report: both arms' pooled out-of-fold predictions, the current classifier as a baseline, and
throughput.

Rows of `summary.csv` (arm, pseudo, subset, granularity):
- arms A and B at the new granularity (the config's classes) and collapsed to the old 20 classes (`metrics.collapse_probabilities`
  on predictions, the parent of each human label), on all clips, B-mode clips and colour-flow clips;
- `old_classifier`: the current classifier's probabilities from the archive (no rerun) against the human label's
  parent, on all clips (`old`), and against the human label on the clips whose label is a class it has (`old_kept`).
  The review form was prefilled with this classifier's prediction, so its agreement may be inflated by anchoring.

Each with n, accuracy (every clip), macro-F1, balanced accuracy and mean AUROC over the evaluated classes (at least
`min_class_support` usable human labels in the run; AUROC also needs `min_positives` in the subset), macro-F1 over
every class, with bootstrap intervals, and the accuracy weighted to the archive's mix of predicted classes. Also: per-class tables and
confusion matrices, paired differences between runs, the current classifier's precision on the pool by confidence for
the pseudo-labelled classes, and the throughput with the archive's cost. `report.md` and `report.html` gather them.

    python -m evals.view_classifier.report --config CONFIG
"""

import argparse
import glob
import html
import json
import os

import numpy as np
import pandas as pd
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import metrics as vm
from evals.view_classifier.io import read_folds

SUBSETS = {"all": None, "bmode": "bmode", "color_flow": "color_flow"}
ARCHIVE_VIDEOS = 6_744_445


def _probs(table, names):
    return table[[f"p_{c}" for c in names]].to_numpy(dtype=float)


def find_runs(runs_dir):
    """{(arm, pseudo): oof table} for every finished run."""
    found = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "arm_*", "pseudo_*", "oof_predictions.csv"))):
        parts = path.split(os.sep)
        found[(parts[-3], parts[-2].removeprefix("pseudo_"))] = pd.read_csv(path, dtype={"clip_id": str})
    return found


def evaluate(truth, probs, names, weights, cfg_r, label, evaluated=None):
    """(summary row, per-class table, confusion frame) of one classifier on one subset."""
    s = vm.summary(truth, probs, len(names), weights, int(cfg_r["min_positives"]), int(cfg_r["bootstrap"]),
                   int(cfg_r.get("seed", 0)), evaluated)
    per = vm.per_class(truth, probs, names, int(cfg_r["min_positives"]), int(cfg_r["bootstrap"]),
                       int(cfg_r.get("seed", 0)), evaluated)
    conf = vm.confusion_frame(truth, probs.argmax(axis=1), names)
    return {**label, **s}, per.assign(**label), conf


def with_targets(labels, parent, archive_counts):
    """`labels` with `y_old` (the parent's index among the old classes) and `weight` (`metrics.archive_weights`)."""
    out = labels.copy()
    old_index = {c: i for i, c in enumerate(vc.OLD_CLASSES)}
    out["y_old"] = [old_index[parent[c]] for c in out.label]
    out["weight"] = vm.archive_weights(out.pool_view, archive_counts)
    return out


def evaluated_classes(labels, classes, parent, min_support):
    """{granularity: bool per class}: classes with at least `min_support` usable human labels, among the new classes
    and among the old ones (labels collapsed to their parents). Fixed for the run, whatever the subset scored."""
    new = labels.label.value_counts()
    old = labels.label.map(parent).value_counts()
    return {"new": np.array([new.get(c, 0) >= min_support for c in classes]),
            "old": np.array([old.get(c, 0) >= min_support for c in vc.OLD_CLASSES])}


class Collector:
    """Scores classifiers on every subset and gathers the rows, per-class tables and confusion matrices."""

    def __init__(self, cfg_r, evaluated):
        self.cfg_r, self.evaluated = cfg_r, evaluated
        self.rows, self.per_class, self.confusions = [], [], {}

    def add(self, name, pseudo, table, probs, names, truth, granularity):
        for subset, acq in SUBSETS.items():
            keep = np.ones(len(table), dtype=bool) if acq is None else (table.acq_category == acq).to_numpy()
            if not keep.any():
                continue
            label = {"arm": name, "pseudo": pseudo, "subset": subset, "granularity": granularity}
            row, per, conf = evaluate(truth[keep], probs[keep], names, table.weight.to_numpy()[keep], self.cfg_r,
                                      label, self.evaluated["new" if granularity == "new" else "old"])
            self.rows.append(row)
            self.per_class.append(per)
            self.confusions[(name, pseudo, subset, granularity)] = conf


def baseline(labels, old_probs, classes, parent, collector, cfg):
    """The current classifier (archive probabilities, `p_<old class>` by global_key) against the human labels:
    collapsed to the old classes (`old`), and on the clips whose label is an old class kept as is (`old_kept`).
    Returns (crosstab human x old prediction, pseudo-label precision table, counts)."""
    t = labels.merge(old_probs, on="global_key", how="inner")
    t = t[np.isfinite(_probs(t, vc.OLD_CLASSES)).all(axis=1)].reset_index(drop=True)
    counts = {"labelled": len(labels), "with_archive_probabilities": len(t)}
    probs = _probs(t, vc.OLD_CLASSES)
    collector.add("old_classifier", "-", t, probs, vc.OLD_CLASSES, t.y_old.to_numpy(), "old")
    kept = t.label.isin([c for c in classes if parent[c] == c]).to_numpy()
    tk = t[kept].reset_index(drop=True)
    collector.add("old_classifier", "-", tk, probs[kept], vc.OLD_CLASSES, tk.y_old.to_numpy(), "old_kept")
    old_vs_new = pd.crosstab(pd.Categorical(t.label, classes),
                             pd.Categorical(np.asarray(vc.OLD_CLASSES)[probs.argmax(axis=1)], vc.OLD_CLASSES),
                             dropna=False)
    old_vs_new.index.name, old_vs_new.columns.name = "human", "old_predicted"
    return old_vs_new, pseudo_precision(t, probs, cfg, classes, parent), counts


def collect(cfg):
    p, r = cfg["paths"], cfg["report"]
    classes, parent = vc.from_config(cfg)
    with open(os.path.join(p["inputs"], "archive_counts.json")) as f:
        archive = json.load(f)["all_splits"]
    labels = with_targets(read_folds(p["folds"], classes), parent, archive)
    evaluated = evaluated_classes(labels, classes, parent, int(r.get("min_class_support", 10)))
    col = Collector(r, evaluated)
    for (arm, pseudo), oof in find_runs(p["runs"]).items():
        t = labels.merge(oof[["clip_id"] + [f"p_{c}" for c in classes]], on="clip_id", how="inner",
                         validate="one_to_one")
        if len(t) != len(labels):
            raise ValueError(f"{arm} pseudo {pseudo} predicts {len(t)} of {len(labels)} labelled clips.")
        probs = _probs(t, classes)
        col.add(arm, pseudo, t, probs, classes, t.y.to_numpy(), "new")
        col.add(arm, pseudo, t, vm.collapse_probabilities(probs, classes, parent, vc.OLD_CLASSES), vc.OLD_CLASSES,
                t.y_old.to_numpy(), "old")
    old = pd.read_csv(os.path.join(p["inputs"], "pool_old_probs.csv"), dtype={"global_key": str})
    old_vs_new, precision, baseline_n = baseline(labels, old, classes, parent, col, cfg)
    return labels, col.rows, col.per_class, col.confusions, old_vs_new, precision, baseline_n, evaluated


def pseudo_precision(t, probs, cfg, classes, parent):
    """For each pseudo-labelled old class: of the pool clips the classifier gives it, how many the reviewers gave
    its new class, overall and at the pseudo-label threshold. The pool has about 22 clips per class: a rough check."""
    threshold = float(cfg["pseudo"]["threshold"])
    mapping = vc.pseudo_label_classes(new=classes, parent=parent, exclude=cfg["pseudo"].get("exclude_classes", ()))
    pred = np.asarray(vc.OLD_CLASSES)[probs.argmax(axis=1)]
    pmax = probs.max(axis=1)
    rows = []
    for old, new in mapping.items():
        mine = pred == old
        high = mine & (pmax >= threshold)
        right = t.label.to_numpy() == new
        rows.append({"old_class": old, "pseudo_label": new, "n_predicted": int(mine.sum()),
                     "precision": float(right[mine].mean()) if mine.any() else float("nan"),
                     "n_at_threshold": int(high.sum()),
                     "precision_at_threshold": float(right[high].mean()) if high.any() else float("nan")})
    return pd.DataFrame(rows)


def paired(labels, runs, classes, cfg):
    names = sorted(runs)
    rows = []
    ordered = labels[["clip_id", "y"]]
    preds = {k: ordered.merge(v, on="clip_id")[[f"p_{c}" for c in classes]].to_numpy().argmax(axis=1)
             for k, v in runs.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = vm.paired_difference(ordered.y.to_numpy(), preds[a], preds[b], len(classes),
                                     int(cfg["report"]["bootstrap"]), int(cfg["report"].get("seed", 0)))
            rows.append({"a": f"{a[0]} pseudo {a[1]}", "b": f"{b[0]} pseudo {b[1]}", **d})
    return pd.DataFrame(rows)


def throughput(cfg):
    p = cfg["paths"]
    rows = []
    bench = json.load(open(p["throughput"])) if os.path.exists(p["throughput"]) else None
    zs_path = os.path.join(p["runs"], "arm_b", "zeroshot", "zeroshot.json")
    zs = json.load(open(zs_path)) if os.path.exists(zs_path) else None
    decode = zs["decode_clips_per_s"] if zs else float("nan")
    if bench:
        for arm, key in (("arm_a", "encoder_plus_probe_clips_per_s"), ("arm_b", "bf16_clips_per_s"),
                         ("arm_b_fp32", "fp32_clips_per_s")):
            model = bench["arm_a" if arm == "arm_a" else "arm_b"][key]
            end_to_end = min(model, decode) if np.isfinite(decode) else model
            rows.append({"arm": arm, "device": bench["device"], "model_clips_per_s": model,
                         "decode_clips_per_s": decode, "decode_workers": zs["decode_workers"] if zs else None,
                         "end_to_end_clips_per_s": end_to_end,
                         "archive_gpu_hours": ARCHIVE_VIDEOS / end_to_end / 3600})
    return pd.DataFrame(rows), zs


def markdown(frame):
    """A Markdown table of a DataFrame (pandas' to_markdown needs a package the environment lacks)."""
    if not len(frame):
        return "(none)"

    def cell(v):
        if isinstance(v, float):
            return "-" if not np.isfinite(v) else f"{v:.3f}"
        return str(v).replace("|", "/")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def _fmt(row, name):
    v, lo, hi = row.get(name), row.get(f"{name}_low"), row.get(f"{name}_high")
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None and np.isfinite(lo) else f"{v:.3f}"


def _heat(frame):
    """An HTML table of a confusion matrix, cells shaded by the row's share."""
    share = frame.div(frame.sum(axis=1).replace(0, 1), axis=0)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = []
    for name, row in frame.iterrows():
        cells = "".join(
            f'<td style="background: rgba(37, 99, 235, {share.loc[name, c]:.2f})">{int(v) if v else ""}</td>'
            for c, v in row.items())
        body.append(f"<tr><th>{html.escape(str(name))}</th>{cells}</tr>")
    return f'<table class="cm"><tr><th>human \\ predicted</th>{head}</tr>{"".join(body)}</table>'


def write(cfg):
    p = cfg["paths"]
    out = os.path.join(p["runs"], "report")
    os.makedirs(out, exist_ok=True)
    classes, _ = vc.from_config(cfg)
    labels, rows, per_class, confusions, old_vs_new, precision, baseline_n, evaluated = collect(cfg)
    classes_all = vc.from_config(cfg)[0]
    below = [c for c, e in zip(classes_all, evaluated["new"]) if not e]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    pd.concat(per_class, ignore_index=True).to_csv(os.path.join(out, "per_class.csv"), index=False)
    for (arm, pseudo, subset, gran), conf in confusions.items():
        conf.to_csv(os.path.join(out, f"confusion_{arm}_pseudo-{pseudo}_{subset}_{gran}.csv"))
    old_vs_new.to_csv(os.path.join(out, "old_classifier_vs_human.csv"))
    precision.to_csv(os.path.join(out, "pseudo_label_precision.csv"), index=False)
    runs = find_runs(p["runs"])
    pairs = paired(labels, runs, classes, cfg) if len(runs) > 1 else pd.DataFrame()
    pairs.to_csv(os.path.join(out, "paired_differences.csv"), index=False)
    tp, zs = throughput(cfg)
    tp.to_csv(os.path.join(out, "throughput.csv"), index=False)
    drops_path = os.path.join(os.path.dirname(p["folds"]), "label_drops.csv")
    drops = pd.read_csv(drops_path) if os.path.exists(drops_path) else pd.DataFrame()
    selections = {}
    for path in sorted(glob.glob(os.path.join(p["runs"], "arm_*", "pseudo_*", "fold_*", "selection.json"))):
        s = json.load(open(path))
        selections.setdefault(f"{s['arm']} pseudo {path.split(os.sep)[-3].removeprefix('pseudo_')}", []).append(
            (s["fold"], s["chosen_hp"], round(s["seconds"] / 60, 1)))

    cols = ["arm", "pseudo", "subset", "granularity", "n"]
    metric_names = ["accuracy", "macro_f1", "balanced_accuracy", "mean_auroc", "macro_f1_all",
                    "archive_weighted_accuracy"]
    table = summary[cols].copy()
    for m in metric_names:
        table[m] = [_fmt(r, m) for r in summary.to_dict("records")]
    md = [f"# View classifier comparison: {cfg.get('name', '')}", "",
          f"Labelled clips used: {len(labels)}. Dropped: "
          + (", ".join(f"{r.reason} {r.n}" for r in drops.itertuples() if r.n) or "none") + ".",
          f"Baseline clips with archive probabilities: {baseline_n['with_archive_probabilities']} of "
          f"{baseline_n['labelled']}.",
          f"Classes below the minimum support of {int(cfg['report'].get('min_class_support', 10))} human labels "
          f"(trained and counted in accuracy and macro_f1_all, not scored one by one): "
          f"{', '.join(below) if below else 'none'}.", "",
          "Metrics on pooled out-of-fold predictions (5-fold, nested selection); intervals: percentile bootstrap "
          "over clips (one clip per patient). `old` granularity: 23 classes collapsed to the current 20. "
          "`old_kept`: clips whose human label is one of the current classes. The review form was prefilled with "
          "the current classifier's view: its agreement may be inflated.", "", markdown(table), ""]
    if len(pairs):
        md += ["## Paired differences (new granularity, all clips)", "", markdown(pairs), ""]
    md += ["## Current classifier: precision on the pool for the pseudo-labelled classes", "",
           markdown(precision), "", "## Throughput", "",
           markdown(tp) if len(tp) else "not measured yet", ""]
    if zs:
        md += ["Zero-shot reproduction (unchanged classifier on our decoding, against the archive's predictions): "
               + ", ".join(f"{o}: {v['agreement_with_archive']:.3f}" for o, v in zs["orders"].items()), ""]
    md += ["## Selected settings per fold (fold, setting, minutes)", ""]
    md += [f"- {k}: {v}" for k, v in selections.items()] + [""]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(md))

    parts = ["<h1>View classifier comparison</h1>", f"<p>{len(labels)} labelled clips.</p>",
             summary_html(table)]
    for (arm, pseudo, subset, gran), conf in confusions.items():
        if subset == "all":
            parts.append(f"<h2>{html.escape(arm)} pseudo {pseudo}, {gran} classes</h2>{_heat(conf)}")
    parts.append(f"<h2>Current classifier against the human labels</h2>{_heat(old_vs_new)}")
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>View classifier comparison</title><style>"
            "body{font-family:sans-serif;margin:16px;background:#fff;color:#111}"
            "table{border-collapse:collapse;font-size:12px;margin:8px 0}"
            "td,th{border:1px solid #ccc;padding:2px 5px;text-align:right}"
            "</style></head><body>" + "".join(parts) + "</body></html>")
    with open(os.path.join(out, "report.html"), "w") as f:
        f.write(page)
    print(f"report written to {out}")


def summary_html(table):
    return table.to_html(index=False, escape=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    write(cfg)


if __name__ == "__main__":
    main()
