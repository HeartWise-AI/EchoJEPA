# evals/view_classifier/io.py

"""The run's tables as both arms read them.

`folds.csv` (written by `prepare folds`): one labelled clip per row with `clip_id` (the video path), `patient_id`,
`label`, `y` and `fold`. `pseudo_labels.csv` (written by `prepare pool`): `video_path`, `patient_id`, `label`. Both
stay in the experiment folder, outside the repository.
"""

import os

import pandas as pd

from evals.view_classifier import classes as vc


def read_folds(path, classes=vc.NEW_CLASSES, merge=None):
    """The folds table. With `merge` ({label name: class}, `classes.label_merge`), a merged label becomes its class
    and `y` is recomputed under `classes` (the name as written is kept in `label_as_written`); the clips and folds
    are unchanged, so another snapshot's folds can be reused read-only with a merged class list."""
    labels = pd.read_csv(path, dtype={"clip_id": str, "patient_id": str, "label": str, "global_key": str})
    for column in ("clip_id", "patient_id", "label", "y", "fold"):
        if column not in labels.columns:
            raise ValueError(f"{path} lacks the column {column}.")
    if merge:
        labels["label_as_written"] = labels.label
        labels["label"] = vc.merge_labels(labels.label, merge)
        y = labels.label.map({c: i for i, c in enumerate(classes)})
        if y.isna().any():
            bad = sorted(set(labels.label[y.isna()]))
            raise ValueError(f"Labels of {path} that are not classes after the merge: {bad}.")
        labels["y"] = y.astype(int)
    if labels.clip_id.duplicated().any():
        raise ValueError("A clip appears twice in the folds table.")
    if (labels.label.map({c: i for i, c in enumerate(classes)}) != labels.y).any():
        raise ValueError("The folds table's y does not match its labels under this class list.")
    if labels.groupby("patient_id").fold.nunique().max() > 1:
        raise ValueError("A patient is in more than one fold.")
    return labels


def read_pseudo(path, labels, classes=vc.NEW_CLASSES, merge=None):
    pseudo = pd.read_csv(path, dtype={"video_path": str, "patient_id": str, "label": str})
    pseudo = pseudo.rename(columns={"video_path": "clip_id"})
    if merge:
        pseudo["label"] = vc.merge_labels(pseudo.label, merge)
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
    merge = vc.label_merge(cfg, classes)
    labels = read_folds(paths["folds"], classes, merge)
    pseudo = read_pseudo(paths["pseudo_labels"], labels, classes, merge) if use_pseudo else None
    return labels, pseudo


def colour_fix_on():
    """True when the legacy YBR colour fix is applied at decoding (`src.datasets.utils.color_order`; on unless
    ECHOJEPA_LEGACY_YBR_FIX=0)."""
    from src.datasets.utils import color_order
    return color_order._enabled()


def cache_colour_fix(meta):
    """Whether a token cache's clips were decoded with the colour fix (its provenance `legacy_ybr_fix`; a cache
    written before the fix existed has no such key: False)."""
    return bool(meta.get("provenance", {}).get("legacy_ybr_fix", False))


def check_colour_fix(cfg, cache_meta=None):
    """Refuse a run whose colour handling differs from its config's `legacy_ybr_fix` (true / false): the decoding
    now (ECHOJEPA_LEGACY_YBR_FIX) and, with `cache_meta`, the token cache it reads. A config without the key (made
    before the fix) is not checked. Returns the decoding's state."""
    on = colour_fix_on()
    want = cfg.get("legacy_ybr_fix") if isinstance(cfg, dict) else None
    if want is None:
        return on
    if bool(want) != on:
        raise SystemExit(f"The config says legacy_ybr_fix: {bool(want)}, but ECHOJEPA_LEGACY_YBR_FIX "
                         f"{'turns the fix on' if on else '=0 turns it off'}: set the variable to match "
                         f"({1 if want else 0}).")
    if cache_meta is not None and cache_colour_fix(cache_meta) != bool(want):
        raise SystemExit(f"The config says legacy_ybr_fix: {bool(want)}, but its token cache was written "
                         f"{'with' if cache_colour_fix(cache_meta) else 'without'} the colour fix.")
    return on


def run_dir(cfg, arm, pseudo):
    return os.path.join(cfg["paths"]["runs"], arm, f"pseudo_{pseudo}")
