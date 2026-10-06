# evals/video_classification_frozen/metrics.py

"""Metrics for evaluating the frozen Visual EF probe in EF percentage points.

Regression metrics are MAE, RMSE, R^2, Pearson correlation, and mean error (bias).
Reduced EF is defined from the reference EF using a configurable cutoff (40% by default)
and evaluated with AUROC and average precision (AUPRC). Sensitivity and specificity use
a predicted-EF threshold selected on validation data by maximizing Youden's J statistic.

Study-level evaluation averages predictions across videos from the same study.
"""

import numpy as np
import pandas as pd


def smooth_l1(labels, predictions):
    """The probe's training loss (`torch.nn.SmoothL1Loss(beta=1)`), averaged over `labels`.

    L(e) = 0.5 * e**2,  if |e| < 1
           |e| - 0.5,   otherwise

    Both are z-scored EF targets, so the returned value is directly comparable with the probe's
    training loss. Returns `NaN` for empty input.
    """
    err = np.abs(np.asarray(predictions, dtype=float) - np.asarray(labels, dtype=float))
    if err.size == 0:
        return float("nan")
    return float(np.where(err < 1.0, 0.5 * err**2, err - 0.5).mean())


def by_study(study_ids, labels, predictions):
    """One row per study: its reference EF and the mean prediction over its videos. All videos
    belonging to the same study must carry the same EF label; otherwise a `ValueError` is raised.

    Returns a DataFrame with columns [study_id, label, prediction, videos], sorted by `study_id`.
    """
    frame = pd.DataFrame({
        "study_id": np.asarray(study_ids).astype(str),
        "label": np.asarray(labels, dtype=float),
        "prediction": np.asarray(predictions, dtype=float),
    })
    grouped = frame.groupby("study_id", sort=True)

    # All videos from the same study are expected to carry the same study-level EF label; refuses to
    # silently average inconsistent labels.
    spread = grouped.label.max() - grouped.label.min()
    if (spread > 1e-6).any():
        raise ValueError(f"{int((spread > 1e-6).sum())} studies have videos with different EF labels.")
    
    # Post-hoc averaging across videos.
    return grouped.agg(
        label=("label", "first"), prediction=("prediction", "mean"), videos=("label", "size")
    ).reset_index()


def regression(labels, predictions):
    """Compute regression metrics for predicted vs. reference EF.

    Returns the sample count and:
        MAE  = mean(|prediction - reference|)
        RMSE = sqrt(mean((prediction - reference)**2))
        R^2   = 1 - sum((prediction - reference)**2) / sum((reference - mean(reference))**2)
        bias = mean(prediction - reference)
        `pearson_r` = the Pearson correlation between reference and predicted EF. 
        
    R^2 is `NaN` when the reference values have zero variance. Pearson r is `NaN` when there are
    fewer than two samples or when either reference or predicted values have zero variance. All
    metrics are `NaN` for empty input except `n`, which is zero.
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(predictions, dtype=float)
    out = {"n": int(y.size)}
    if y.size == 0:
        return {**out, **dict.fromkeys(("mae", "rmse", "r2", "pearson_r", "bias"), float("nan"))}
    err = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r = float(np.corrcoef(y, p)[0, 1]) if y.size > 1 and y.std() > 0 and p.std() > 0 else float("nan")
    return {
        **out,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),    # penalize large mistakes more strongly than MAE.
        "r2": float(1.0 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"), # how much variation in ref EF is explained by the predictions.
        "pearson_r": r,                             # whether high true EF tends to correspond to high predicted EF.
        "bias": float(err.mean()),
    }


def _average_ranks(x):
    """Return ranks 1..n of `x`, assigning tied values their mean rank (e.g., values occupying
    ranks 2 and 3 both receive rank 2.5)."""
    # Sorting make tie handling deterministic.
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(np.diff(xs)) + 1]
    ends = np.r_[starts[1:], xs.size]
    ranks = np.empty(xs.size)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def auroc(positive, score):
    """Compute AUROC from binary labels and continuous scores.
    
    AUROC: the probability that a randomly chosen positive example receives a higher score than a
    randomly chosen negative example, with tied scores contributing 0.5:
        AUROC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    
    Returns `NaN` if either the positive or negative class is absent.
    """
    pos = np.asarray(positive, dtype=bool)
    score = np.asarray(score, dtype=float)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    # If only positives or only negatives exist, return `nan` because ROC discrimination cannot be defined without both classes.
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _average_ranks(score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(positive, score):
    """Compute average precision from binary labels and continuous scores.

    Examples are ranked from highest to lowest score. At each distinct score, precision is
    weighted by the increase in recall:
        AP = sum_k (recall_k - recall_{k-1}) * precision_k
    This matches the non-interpolated average-precision definition used by
    scikit-learn. 
    
    Returns `NaN` when there are no positive examples.
    """
    pos = np.asarray(positive, dtype=bool)
    score = np.asarray(score, dtype=float)
    n1 = int(pos.sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], pos[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]  # last row of each distinct score
    tp = np.cumsum(t)[last]
    precision = tp / (last + 1.0)
    recall = tp / n1
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def sensitivity_specificity(positive, predictions, threshold):
    """Compute sensitivity and specificity at a predicted-EF threshold. A sample is predicted
    to have reduced EF when: predicted EF < threshold.

    With `positive` defining the reference reduced-EF class:
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

    Sensitivity is `NaN` if no positive examples are present, and specificity
    is `NaN` if no negative examples are present.
    """
    pos = np.asarray(positive, dtype=bool)
    flagged = np.asarray(predictions, dtype=float) < threshold
    sens = float((flagged & pos).sum() / pos.sum()) if pos.any() else float("nan")
    spec = float((~flagged & ~pos).sum() / (~pos).sum()) if (~pos).any() else float("nan")
    return sens, spec


def youden_threshold(labels, predictions, below=40.0):
    """Choose the predicted-EF threshold that maximizes Youden's J statistic. Reference cases with
    reference EF < `below` are treated as positive. For each candidate predicted-EF threshold `t`,
    predictions below `t` are classified as positive, and J(t) = sensitivity(t) + specificity(t) - 1.

    Candidate thresholds are the midpoints between consecutive distinct predictions, together with
    one threshold below the minimum prediction and one above the maximum. If multiple candidates
    achieve the same maximum J, the threshold closest to `below` is chosen.

    Returns `None` when the reference labels contain only one class.
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(predictions, dtype=float)
    pos = y < below
    if not pos.any() or pos.all():
        return None
    u = np.unique(p)
    # Candidate thresholds are midpoints: classification only changes when the threshold crosses a prediction.
    candidates = np.r_[u[0] - 1.0, (u[:-1] + u[1:]) / 2.0, u[-1] + 1.0]
    flagged = p[None, :] < candidates[:, None]
    j = (flagged & pos).sum(1) / pos.sum() + (~flagged & ~pos).sum(1) / (~pos).sum() - 1.0
    ties = candidates[np.isclose(j, j.max())]
    # Chooses closest to `below=40` when tied.
    return float(ties[np.argmin(np.abs(ties - below))])


def reduced_ef(labels, predictions, below=40.0, threshold=None):
    """Evaluate detection of reduced EF from continuous EF predictions. A reference case is positive
    when: reference EF < `below`.

    Returns the number and prevalence of positive cases, AUROC, and average precision (AUPRC). If `threshold` is
    provided, predictions satisfying predicted EF < `threshold` are classified as positive, and sensitivity and
    specificity are also returned. If `threshold` is `None`, those threshold-dependent metrics are omitted.
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(predictions, dtype=float)
    pos = y < below
    out = {
        "below": float(below),
        "positives": int(pos.sum()),
        "prevalence": float(pos.mean()) if y.size else float("nan"),
        # `-p`: because lower predicted EF means more likely to be positive (higher score, greater likelihood of reduced EF),
        # so flip the predicted EF to be used as the ranking score.
        "auroc": auroc(pos, -p),
        "auprc": average_precision(pos, -p),
        "threshold": None if threshold is None else float(threshold),
    }
    if threshold is not None:
        out["sensitivity"], out["specificity"] = sensitivity_specificity(pos, p, threshold)
    return out


def summarize(labels, predictions, below=40.0, threshold=None):
    """The top-level dictionary contains the metrics from `regression()` and a `reduced_ef` entry containing
    the output of `reduced_ef()`."""
    return {**regression(labels, predictions), "reduced_ef": reduced_ef(labels, predictions, below, threshold)}


def by_range(labels, predictions, edges=(30, 40, 50, 60)):
    """Summarize prediction error within reference-EF ranges. Reference EF is divided into left-closed,
    right-open intervals defined by `edges`. 
    
    With the default `edges`, the ranges are:
        (-inf, 30), [30, 40), [40, 50), [50, 60), [60, inf).

    For each range, returns its sample count together with MAE and bias = mean(prediction - reference).
    MAE and bias are `NaN` for an empty range.
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(predictions, dtype=float)
    bounds = [-np.inf, *edges, np.inf]
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        inside = (y >= lo) & (y < hi)
        name = f"< {hi:g}" if lo == -np.inf else (f">= {lo:g}" if hi == np.inf else f"{lo:g}-{hi:g}")
        err = p[inside] - y[inside]
        rows.append({
            "range": name,
            "n": int(inside.sum()),
            "mae": float(np.abs(err).mean()) if inside.any() else float("nan"),
            "bias": float(err.mean()) if inside.any() else float("nan"),
        })
    return rows
