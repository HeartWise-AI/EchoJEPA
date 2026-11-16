# evals/radar_frozen/zero_shot_metrics.py

"""Metrics of zero-shot predictions, one task at a time, with bootstrap intervals. NumPy only.

Binary task (reference 0 or 1, predicted probability of the finding):

- `n`, `positives`, `prevalence`;
- discrimination: `auroc`, `auprc` (average precision);
- calibration: `brier` (mean squared error of the probability), `ece` (expected calibration error over equal-width
  probability bins: the mean over bins, weighted by their size, of |mean probability - observed rate|), and the
  logistic recalibration line `calibration_intercept`, `calibration_slope`: reference ~ sigmoid(a + b * logit(p)),
  fitted by maximum likelihood (a = 0 and b = 1 for a calibrated model; b < 1 says the probabilities are too
  extreme). The line has no value when one class is absent, when every probability is the same, or when the classes
  are perfectly separated (the fit has no finite maximum);
- at the operating `threshold` (the finding is called when probability >= threshold): `precision`, `recall`
  (sensitivity), `specificity`, `f1`.

A zero-shot probability is a softmax over two prompt sets at the trained temperature: nothing fitted it to the
prevalence of any population, so the calibration metrics describe that probability as it comes out, and the
discrimination metrics do not depend on it being calibrated.

Regression task: `n`, `mae`, `rmse`, `r2`, `pearson_r`, `bias` (the probe evaluation's `regression`).

Categorical task (reference grade, one probability per grade): `n`, `accuracy`, `macro_f1` (mean F1 of the grades
that have a reference or a predicted study) and `macro_auroc` (mean one-against-rest AUROC of the grades that have
both classes).

`probability_threshold` chooses a binary threshold by Youden's J. It is for a tuning split: a threshold chosen on the
split it is reported on makes precision, recall, specificity and F1 optimistic.

`bootstrap` gives percentile intervals by resampling studies with replacement, or whole groups (for example patients)
when studies of one group are not independent. A metric that has no value in a resample (one class absent, for
example) is left out of that metric's percentiles, and the number of resamples that had a value is reported beside
the interval. The operating threshold is held fixed across resamples: the interval describes the metric at that
threshold, not the choice of the threshold.
"""

import numpy as np

from evals.video_classification_frozen.metrics import auroc, average_precision, regression

BINARY_INTERVALS = ("prevalence", "auroc", "auprc", "brier", "ece", "calibration_intercept", "calibration_slope",
                    "precision", "recall", "specificity", "f1")
REGRESSION_INTERVALS = ("mae", "rmse", "r2", "pearson_r", "bias")
CATEGORICAL_INTERVALS = ("accuracy", "macro_f1", "macro_auroc")
NAN = float("nan")
regression_metrics = regression


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else NAN


def expected_calibration_error(labels, probabilities, bins=10):
    """Mean over `bins` equal-width probability bins, weighted by size, of |mean probability - observed rate|."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.size == 0:
        return NAN
    which = np.minimum((p * bins).astype(int), bins - 1)             # probability 1.0 belongs to the last bin
    count = np.bincount(which, minlength=bins)
    filled = count > 0
    mean_p = np.bincount(which, weights=p, minlength=bins)[filled] / count[filled]
    rate = np.bincount(which, weights=y, minlength=bins)[filled] / count[filled]
    return float((count[filled] / y.size * np.abs(mean_p - rate)).sum())


def calibration_line(labels, probabilities, eps=1e-6, max_iter=50, tol=1e-8):
    """(intercept, slope) of reference ~ sigmoid(intercept + slope * logit(probability)) by Newton's method, with
    probabilities clipped to [eps, 1 - eps]. (NaN, NaN) when the fit has no finite maximum (see the module
    docstring) or does not converge."""
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    x = np.log(p) - np.log1p(-p)
    if y.size < 2 or y.min() == y.max() or np.ptp(x) == 0:
        return NAN, NAN
    design = np.stack([np.ones_like(x), x], axis=1)
    beta = np.array([0.0, 1.0])
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30.0, 30.0)))
        weight = mu * (1.0 - mu)
        hessian = design.T @ (design * weight[:, None])
        if not np.isfinite(hessian).all() or np.linalg.cond(hessian) > 1e12:
            return NAN, NAN
        step = np.linalg.solve(hessian, design.T @ (y - mu))
        beta = beta + step
        if np.abs(beta).max() > 1e3:                                  # separated classes: the slope runs away
            return NAN, NAN
        if np.abs(step).max() < tol:
            return float(beta[0]), float(beta[1])
    return NAN, NAN


def probability_threshold(labels, probabilities):
    """The threshold t maximising Youden's J = sensitivity + specificity - 1 of the rule `probability >= t`, among
    the midpoints of consecutive distinct probabilities, 0 (every study called) and just above 1 (none called); of
    several maxima, the one closest to 0.5. None when the reference has one class."""
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=float)
    if not y.any() or y.all():
        return None
    u = np.unique(p)
    candidates = np.r_[0.0, (u[:-1] + u[1:]) / 2.0, np.nextafter(1.0, 2.0)]
    flagged = p[None, :] >= candidates[:, None]
    j = (flagged & y).sum(1) / y.sum() + (~flagged & ~y).sum(1) / (~y).sum() - 1.0
    ties = candidates[np.isclose(j, j.max())]
    return float(ties[np.argmin(np.abs(ties - 0.5))])


def binary_metrics(labels, probabilities, threshold=0.5):
    """The binary metrics of the module docstring. `labels` are 0 or 1."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    pos = y > 0.5
    flagged = p >= threshold
    tp, fp = int((flagged & pos).sum()), int((flagged & ~pos).sum())
    fn, tn = int((~flagged & pos).sum()), int((~flagged & ~pos).sum())
    intercept, slope = calibration_line(y, p)
    return {
        "n": int(y.size),
        "positives": int(pos.sum()),
        "prevalence": _ratio(pos.sum(), y.size),
        "auroc": auroc(pos, p),
        "auprc": average_precision(pos, p),
        "brier": float(((p - y) ** 2).mean()) if y.size else NAN,
        "ece": expected_calibration_error(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": float(threshold),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def categorical_metrics(labels, probabilities, grades):
    """The categorical metrics of the module docstring. `labels` [N] hold values of `grades`; `probabilities` [N, G]
    follow the order of `grades`; the predicted grade is the most probable one."""
    grades = np.asarray(grades, dtype=float)
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float).reshape(-1, grades.size)
    if y.size == 0:
        return {"n": 0, "accuracy": NAN, "macro_f1": NAN, "macro_auroc": NAN}
    if not np.isin(y, grades).all():
        raise ValueError("A reference grade is not one of the task's grades.")
    truth = np.argmax(y[:, None] == grades[None, :], axis=1)
    predicted = p.argmax(axis=1)
    f1, aucs = [], []
    for g in range(grades.size):
        tp = int(((predicted == g) & (truth == g)).sum())
        other = int(((predicted == g) != (truth == g)).sum())                 # false positives + false negatives
        if tp + other:
            f1.append(2 * tp / (2 * tp + other))
        aucs.append(auroc(truth == g, p[:, g]))
    aucs = [a for a in aucs if a == a]
    return {"n": int(y.size), "accuracy": float((predicted == truth).mean()),
            "macro_f1": float(np.mean(f1)) if f1 else NAN, "macro_auroc": float(np.mean(aucs)) if aucs else NAN}


def bootstrap(metric, columns, keys, groups=None, samples=1000, seed=0, level=0.95):
    """Percentile intervals of `metric(*columns)[key]` for each key: {key: {"low", "high", "resamples"}}.

    `columns` are arrays with one row per study. Rows are drawn with replacement; with `groups` (one value per row),
    whole groups are. `resamples` counts the resamples in which the metric had a value; `low` and `high` are None when
    there were none (or when there is nothing to resample)."""
    columns = [np.asarray(c) for c in columns]
    n = len(columns[0])
    empty = {key: {"low": None, "high": None, "resamples": 0} for key in keys}
    if n == 0 or samples <= 0:
        return empty
    rng = np.random.default_rng(seed)
    members = None
    if groups is not None:
        _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        order = np.argsort(inverse, kind="mergesort")
        members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    draws = {key: [] for key in keys}
    for _ in range(samples):
        if members is None:
            rows = rng.integers(0, n, n)
        else:
            rows = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        values = metric(*[c[rows] for c in columns])
        for key in keys:
            draws[key].append(values[key])
    tail = (1.0 - level) / 2.0 * 100.0
    out = {}
    for key in keys:
        finite = np.asarray(draws[key], dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            out[key] = empty[key]
            continue
        low, high = np.percentile(finite, [tail, 100.0 - tail])
        out[key] = {"low": float(low), "high": float(high), "resamples": int(finite.size)}
    return out

