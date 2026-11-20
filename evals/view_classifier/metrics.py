# evals/view_classifier/metrics.py

"""Metrics of a view classifier on pooled out-of-fold predictions.

- Confusion matrix: rows the human view, columns the predicted view (counts).
- Per class: support (human positives), predicted count, precision, recall, F1, one-vs-rest AUROC of the class's
  probability with a percentile bootstrap interval. A class is `evaluated` when it has at least the run's minimum
  support of human labels overall (set by the caller); the AUROC of a class that is not, or that has fewer than
  `min_positives` positives (or negatives) in the subset scored, is not reported (value kept for reference).
- Summary: accuracy (every clip, whatever its class); macro-F1, balanced accuracy (mean recall) and mean reported
  AUROC over the evaluated classes; `macro_f1_all` over every class that is a label or a prediction at least once
  (scikit-learn's rule: a class only ever predicted wrongly counts with F1 0); the accuracy weighted to the archive's
  mix of predicted classes (`archive_weights`). Each with a bootstrap interval over clips (one clip per patient).
- `collapse_probabilities` sums the probabilities of the classes sharing an old parent, to compare at the old
  classifier's granularity; `paired_difference` bootstraps the difference of two classifiers on the same clips.

AUROC is `evals.video_classification_frozen.metrics.auroc` (ties count one half).
"""

import numpy as np
import pandas as pd

from evals.video_classification_frozen.metrics import auroc


def confusion_matrix(truth, pred, n_classes):
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (np.asarray(truth, dtype=int), np.asarray(pred, dtype=int)), 1)
    return m


def confusion_frame(truth, pred, classes):
    """Confusion matrix as a DataFrame (index human view, columns predicted view), only rows and columns in use."""
    m = confusion_matrix(truth, pred, len(classes))
    used = (m.sum(axis=0) > 0) | (m.sum(axis=1) > 0)
    names = [c for c, u in zip(classes, used) if u]
    frame = pd.DataFrame(m[np.ix_(used, used)], index=names, columns=names)
    frame.index.name, frame.columns.name = "human", "predicted"
    return frame


def _prf(m):
    tp = np.diag(m).astype(float)
    support, predicted = m.sum(axis=1).astype(float), m.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(support > 0, np.where(precision + recall > 0,
                                            2 * precision * recall / (precision + recall), 0.0),
                      np.where(predicted > 0, 0.0, np.nan))
    return precision, recall, f1, support, predicted


def accuracy(truth, pred, weights=None):
    hit = (np.asarray(truth) == np.asarray(pred)).astype(float)
    return float(np.average(hit, weights=weights)) if len(hit) else float("nan")


def macro_f1(truth, pred, n_classes):
    return float(np.nanmean(_prf(confusion_matrix(truth, pred, n_classes))[2]))


def balanced_accuracy(truth, pred, n_classes):
    return float(np.nanmean(_prf(confusion_matrix(truth, pred, n_classes))[1]))


def class_aurocs(truth, probs, min_positives=5):
    """(AUROC per class, reportable per class): one-vs-rest on column c of `probs`."""
    truth, probs = np.asarray(truth, dtype=int), np.asarray(probs, dtype=float)
    values, ok = [], []
    for c in range(probs.shape[1]):
        pos = truth == c
        values.append(auroc(pos, probs[:, c]))
        ok.append(bool(pos.sum() >= min_positives and (~pos).sum() >= min_positives))
    return np.asarray(values), np.asarray(ok)


def mean_auroc(truth, probs, min_positives=5):
    values, ok = class_aurocs(truth, probs, min_positives)
    return float(np.nanmean(values[ok])) if ok.any() else float("nan")


def _interval(stat, n, samples, seed, level=0.95):
    """Percentile bootstrap of stat(index array) over n items; (low, high), NaN when samples is 0."""
    if samples <= 0 or n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.array([stat(rng.integers(0, n, n)) for _ in range(samples)], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(values, a)), float(np.quantile(values, 1 - a))


def per_class(truth, probs, classes, min_positives=5, samples=1000, seed=0, evaluated=None):
    """One row per class; `evaluated` False (too few human labels overall) leaves the class's AUROC unreported."""
    truth, probs = np.asarray(truth, dtype=int), np.asarray(probs, dtype=float)
    keep = np.ones(len(classes), dtype=bool) if evaluated is None else np.asarray(evaluated, dtype=bool)
    pred = probs.argmax(axis=1)
    precision, recall, f1, support, predicted = _prf(confusion_matrix(truth, pred, len(classes)))
    values, ok = class_aurocs(truth, probs, min_positives)
    reported = ok & keep
    rows = []
    for c, name in enumerate(classes):
        low = high = float("nan")
        if reported[c]:
            low, high = _interval(lambda i: auroc(truth[i] == c, probs[i, c]), len(truth), samples, seed + c)
        rows.append({"class": name, "support": int(support[c]), "predicted": int(predicted[c]),
                     "precision": float(precision[c]) if predicted[c] else float("nan"),
                     "recall": float(recall[c]), "f1": float(f1[c]), "auroc": float(values[c]),
                     "auroc_low": low, "auroc_high": high, "auroc_reported": bool(reported[c]),
                     "evaluated": bool(keep[c]),
                     "note": "" if reported[c] else ("below the minimum support" if not keep[c] else
                                                     f"fewer than {min_positives} positives"
                                                     if support[c] < min_positives else "too few negatives")})
    return pd.DataFrame(rows)


def summary(truth, probs, n_classes=None, weights=None, min_positives=5, samples=1000, seed=0, evaluated=None):
    """Dict of the summary metrics, each with `_low` and `_high` bootstrap bounds.

    `evaluated` (bool per class): the classes with enough human labels to be scored one by one. Accuracy counts every
    clip; macro-F1, balanced accuracy and mean AUROC average over the evaluated classes only, and `macro_f1_all` over
    every class (the scikit-learn rule)."""
    truth, probs = np.asarray(truth, dtype=int), np.asarray(probs, dtype=float)
    n_classes = probs.shape[1] if n_classes is None else n_classes
    pred = probs.argmax(axis=1)
    w = None if weights is None else np.asarray(weights, dtype=float)
    keep = np.ones(n_classes, dtype=bool) if evaluated is None else np.asarray(evaluated, dtype=bool)

    def averaged(i, which):
        values = _prf(confusion_matrix(truth[i], pred[i], n_classes))[which]
        return float(np.nanmean(values[keep])) if np.isfinite(values[keep]).any() else float("nan")

    def auroc_mean(i):
        values, ok = class_aurocs(truth[i], probs[i], min_positives)
        ok = ok & keep
        return float(np.nanmean(values[ok])) if ok.any() else float("nan")

    stats = {
        "accuracy": lambda i: accuracy(truth[i], pred[i]),
        "macro_f1": lambda i: averaged(i, 2),
        "balanced_accuracy": lambda i: averaged(i, 1),
        "mean_auroc": auroc_mean,
        "macro_f1_all": lambda i: macro_f1(truth[i], pred[i], n_classes),
    }
    if w is not None:
        stats["archive_weighted_accuracy"] = lambda i: accuracy(truth[i], pred[i], w[i])
    every = np.arange(len(truth))
    out = {"n": int(len(truth))}
    for k, (name, stat) in enumerate(stats.items()):
        out[name] = stat(every)
        out[f"{name}_low"], out[f"{name}_high"] = _interval(stat, len(truth), samples, seed + 101 * k)
    return out


def collapse_probabilities(probs, new_classes, parent, old_classes):
    """[N, len(old_classes)]: each old class's probability is the sum over the new classes descending from it."""
    probs = np.asarray(probs, dtype=float)
    out = np.zeros((probs.shape[0], len(old_classes)))
    for j, c in enumerate(new_classes):
        out[:, old_classes.index(parent[c])] += probs[:, j]
    return out


def paired_difference(truth, pred_a, pred_b, n_classes, samples=2000, seed=0):
    """Accuracy and macro-F1 of A minus B on the same clips, with bootstrap intervals."""
    truth, pred_a, pred_b = (np.asarray(x, dtype=int) for x in (truth, pred_a, pred_b))
    stats = {"accuracy": lambda i: accuracy(truth[i], pred_a[i]) - accuracy(truth[i], pred_b[i]),
             "macro_f1": lambda i: macro_f1(truth[i], pred_a[i], n_classes) - macro_f1(truth[i], pred_b[i], n_classes)}
    out = {"n": int(len(truth))}
    for k, (name, stat) in enumerate(stats.items()):
        out[f"{name}_diff"] = stat(np.arange(len(truth)))
        out[f"{name}_diff_low"], out[f"{name}_diff_high"] = _interval(stat, len(truth), samples, seed + k)
    return out


def archive_weights(pool_view, archive_counts):
    """Weight of each labelled clip so the labelled pool has the archive's mix of predicted classes: the archive's
    share of the clip's predicted class over the pool's share. `archive_counts`: {class: number of archive clips}."""
    pool_view = pd.Series(pool_view, dtype=str).reset_index(drop=True)
    pool_share = pool_view.map(pool_view.value_counts(normalize=True))
    total = float(sum(archive_counts.values()))
    archive_share = pool_view.map(lambda v: archive_counts.get(v, 0) / total)
    if (archive_share == 0).any():
        raise ValueError("A predicted class of the pool has no archive clips.")
    return (archive_share / pool_share).to_numpy()
