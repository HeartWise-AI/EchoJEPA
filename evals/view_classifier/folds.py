# evals/view_classifier/folds.py

"""Stratified patient-level folds, and the inner validation split that selects hyper-parameters and epochs inside
the training folds.

`assign_folds` puts every clip of a patient in one fold (a patient's class is its clips' most frequent label) and
balances each class over the folds: classes are placed from the largest down, each patient going to the fold with
the fewest of that class so far, then the fewest patients, then the lowest number. The patients of a class are
shuffled with `seed` first, so the folds depend only on the table's content and the seed, not on its row order.

`inner_split` holds out about `fraction` of the training patients of each class for validation, never the last
`min_train` of a class, so a class with few clips stays in training.
"""

import numpy as np
import pandas as pd


def _patients(table, label, group):
    """One row per patient: group, its label (most frequent; ties to the smallest), its number of clips."""
    counts = table.groupby([group, label]).size().reset_index(name="n")
    counts = counts.sort_values([group, "n", label], ascending=[True, False, True])
    first = counts.drop_duplicates(group)
    return first[[group, label]].reset_index(drop=True)


def _shuffled(values, seed, salt):
    rng = np.random.default_rng([seed, salt])
    values = sorted(values)
    return [values[i] for i in rng.permutation(len(values))]


def assign_folds(table, k=5, label="label", group="patient_id", seed=0):
    """Series of fold numbers 0..k-1 aligned with `table`."""
    if k < 2:
        raise ValueError("Need at least two folds.")
    patients = _patients(table, label, group)
    sizes = patients[label].value_counts()
    classes = sorted(sizes.index, key=lambda c: (-sizes[c], str(c)))
    fold_of, totals = {}, np.zeros(k, dtype=int)
    for i, cls in enumerate(classes):
        members = patients.loc[patients[label] == cls, group].astype(str).tolist()
        counts = np.zeros(k, dtype=int)
        for p in _shuffled(members, seed, i):
            f = min(range(k), key=lambda j: (counts[j], totals[j], j))
            fold_of[p] = f
            counts[f] += 1
            totals[f] += 1
    return table[group].astype(str).map(fold_of).astype(int)


def check_folds(table, fold="fold", group="patient_id"):
    """Raise unless every patient is in exactly one fold."""
    spread = table.groupby(table[group].astype(str))[fold].nunique()
    if (spread > 1).any():
        raise ValueError(f"{int((spread > 1).sum())} patients are in more than one fold.")


def inner_split(table, label="label", group="patient_id", fraction=0.15, min_train=2, seed=0):
    """Boolean Series aligned with `table`: True for the inner validation clips."""
    patients = _patients(table, label, group)
    held = set()
    for i, cls in enumerate(sorted(patients[label].unique(), key=str)):
        members = _shuffled(patients.loc[patients[label] == cls, group].astype(str).tolist(), seed, 10_000 + i)
        n_val = min(int(round(fraction * len(members))), max(0, len(members) - min_train))
        held |= set(members[:n_val])
    return table[group].astype(str).isin(held)


def fold_table(labels, k=5, seed=0):
    """`labels` with a `fold` column; one clip per patient is expected but not required."""
    out = labels.copy()
    out["fold"] = assign_folds(out, k=k, seed=seed)
    check_folds(out)
    return out


def class_by_fold(table, label="label", fold="fold"):
    """Counts of each class in each fold (a table for the design note and the logs)."""
    return pd.crosstab(table[label], table[fold])
