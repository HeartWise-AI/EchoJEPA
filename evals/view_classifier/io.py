# evals/view_classifier/io.py

"""The run's tables as both arms read them.

`folds.csv` (written by `prepare folds`): one labelled clip per row with `clip_id` (the video path), `patient_id`,
`label`, `y` and `fold`. `pseudo_labels.csv` (written by `prepare pool`): `video_path`, `patient_id`, `label`. Both
stay in the experiment folder, outside the repository.
"""

import os

import pandas as pd

from evals.view_classifier import classes as vc


def read_folds(path, classes=vc.NEW_CLASSES):
    labels = pd.read_csv(path, dtype={"clip_id": str, "patient_id": str, "label": str, "global_key": str})
    for column in ("clip_id", "patient_id", "label", "y", "fold"):
        if column not in labels.columns:
            raise ValueError(f"{path} lacks the column {column}.")
    if labels.clip_id.duplicated().any():
        raise ValueError("A clip appears twice in the folds table.")
    if (labels.label.map({c: i for i, c in enumerate(classes)}) != labels.y).any():
        raise ValueError("The folds table's y does not match its labels under this class list.")
    if labels.groupby("patient_id").fold.nunique().max() > 1:
        raise ValueError("A patient is in more than one fold.")
    return labels


def read_pseudo(path, labels, classes=vc.NEW_CLASSES):
    pseudo = pd.read_csv(path, dtype={"video_path": str, "patient_id": str, "label": str})
    pseudo = pseudo.rename(columns={"video_path": "clip_id"})
    pseudo["y"] = pseudo.label.map({c: i for i, c in enumerate(classes)})
    if pseudo.y.isna().any():
        raise ValueError("A pseudo-label is not a class.")
    pseudo["y"] = pseudo.y.astype(int)
    if set(pseudo.patient_id) & set(labels.patient_id):
        raise ValueError("Pseudo-labelled clips share patients with the labelled clips.")
    if set(pseudo.clip_id) & set(labels.clip_id):
        raise ValueError("A clip is both labelled and pseudo-labelled.")
    return pseudo[["clip_id", "patient_id", "label", "y"]]


def read_run_inputs(cfg, use_pseudo):
    paths = cfg["paths"]
    classes, _ = vc.from_config(cfg)
    labels = read_folds(paths["folds"], classes)
    pseudo = read_pseudo(paths["pseudo_labels"], labels, classes) if use_pseudo else None
    return labels, pseudo


def run_dir(cfg, arm, pseudo):
    return os.path.join(cfg["paths"]["runs"], arm, f"pseudo_{pseudo}")
