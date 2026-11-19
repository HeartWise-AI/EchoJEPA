# evals/view_classifier/labels.py

"""The human view labels and the pseudo-labels.

Human labels come from the label export (`LABEL_COLUMNS`). A clip is used when its `label_status` is `done` and its
`human_view` is a class; `unsure`, an empty view and every other status are dropped and counted by reason
(`read_labels`). When the label project holds several upload batches, the export's `batch` column names each row's
batch and `batches` keeps only the batches listed (the others are counted as `other_batch`). A view that is neither a
class, an alias of one nor `unsure` stops the run, since it means the export and the class list disagree.
`check_against_pool` ties each label to the clip drawn for labelling (same patient, same video) and
`check_train_split` to the train split.

Pseudo-labels (`select_pseudo_labels`) are the current classifier's predictions on other train-split clips: only for
the old classes that map to exactly one new class, above a probability threshold, at most one clip per patient, never
a patient of the labelling pool, and only clips the pool could have held (B-mode or colour flow, a known acquisition
mode, more than a still's frames). They are training data only, never evaluated.
"""

import hashlib

import numpy as np
import pandas as pd

from evals.view_classifier import classes as vc

LABEL_COLUMNS = ("global_key", "patient_id", "study_id", "video_path", "acq_category", "predicted_view",
                 "human_view", "label_status", "n_labels", "labelled_at")
STATUSES = ("done", "unlabelled", "skipped", "in_rework", "conflict")
IMAGING_MODES = ("bmode", "color_flow")
STILL_MAX_FRAMES = 3     # as data/radar_pilot.py: this many frames or fewer is a still.


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_labels(table, new_classes=vc.NEW_CLASSES, aliases=None, batches=None):
    """(usable labels, drop counts). `table` is a DataFrame or a CSV path. Usable rows get `label` (the class name)
    and `y` (its index in `new_classes`). Drop counts: a DataFrame with `reason` and `n`, zero rows included.
    `batches` (a list of batch numbers, or None for every row): keep only rows whose `batch` is listed; the others
    are dropped first, as `other_batch`. The checks of statuses, keys and views still cover every row."""
    labels = pd.read_csv(table, dtype=str, keep_default_na=False) if isinstance(table, str) else table.copy()
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        raise ValueError(f"The label table lacks columns {missing}.")
    if labels.global_key.duplicated().any():
        raise ValueError("A global_key appears twice in the label table.")
    status = labels.label_status.astype(str).str.strip().str.lower()
    unknown_status = sorted(set(status) - set(STATUSES))
    if unknown_status:
        raise ValueError(f"Unknown label_status values {unknown_status}.")
    view = labels.human_view.map(lambda v: vc.canonical(v, aliases))
    done = status == "done"
    known = view.isin(new_classes) | view.isin(["", vc.UNSURE])
    bad = sorted(set(view[done & ~known]))
    if bad:
        raise ValueError(f"Labels that are not classes: {bad}. Add them to the class list or to ALIASES.")
    other = pd.Series(False, index=labels.index)
    if batches is not None:
        other = ~batch_numbers(labels).isin(parse_batches(batches))
    reasons = np.select([other, ~done, view == vc.UNSURE, view == ""],
                        ["other_batch", "status_" + status, "unsure", "empty_view"], default="")
    counts = pd.Series(reasons[reasons != ""], dtype=str).value_counts()
    every = ([] if batches is None else ["other_batch"]) \
        + [f"status_{s}" for s in STATUSES if s != "done"] + ["unsure", "empty_view"]
    drops = pd.DataFrame({"reason": every, "n": [int(counts.get(r, 0)) for r in every]})
    usable = labels[reasons == ""].copy()
    usable["label"] = view[reasons == ""]
    index = {c: i for i, c in enumerate(new_classes)}
    usable["y"] = usable.label.map(index).astype(int)
    return usable.reset_index(drop=True), drops


def parse_batches(batches):
    """The batch numbers to keep, from a config list (`label_batches`): positive integers, at least one."""
    if isinstance(batches, (str, int)) or not batches:
        raise ValueError("`label_batches` must be a non-empty list of batch numbers, e.g. [1].")
    out = []
    for b in batches:
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError(f"`label_batches` holds {b!r}: batch numbers are integers from 1.")
        out.append(b)
    return out


def batch_numbers(labels):
    """The `batch` column of a label export as integers; raise when it is missing or not a whole number."""
    if "batch" not in labels.columns:
        raise ValueError("`label_batches` is set but the label table has no `batch` column (export it with every "
                         "batch's picked list).")
    numbers = pd.to_numeric(labels.batch.astype(str).str.strip(), errors="coerce")
    if numbers.isna().any() or (numbers != numbers.round()).any():
        raise ValueError("The label table's `batch` column holds values that are not batch numbers.")
    return numbers.astype(int)


def check_against_pool(labels, pool):
    """Each label's clip as drawn for labelling (`picked_clips.csv`: global_key, patient_id, video_path, view):
    raise when a key is not in the pool or its patient or video differs. Returns `labels` with `pool_view`, the
    classifier's view at the draw."""
    pool = pool.drop_duplicates("global_key")
    merged = labels.merge(pool[["global_key", "patient_id", "video_path", "view"]].rename(
        columns={"patient_id": "pool_patient", "video_path": "pool_video", "view": "pool_view"}),
        on="global_key", how="left", validate="one_to_one")
    if merged.pool_video.isna().any():
        raise ValueError(f"{int(merged.pool_video.isna().sum())} labelled clips are not in the labelling pool.")
    same_patient = merged.patient_id.astype(str).str.strip() == merged.pool_patient.astype(str).str.strip()
    same_video = merged.video_path.astype(str) == merged.pool_video.astype(str)
    if not (same_patient & same_video).all():
        raise ValueError(f"{int((~(same_patient & same_video)).sum())} labels name another patient or video than "
                         "the pool's clip of the same key.")
    return merged.drop(columns=["pool_patient", "pool_video"])


def check_train_split(patient_ids, splits):
    """Raise unless every patient is in `splits` (patient_id, split) with split `train`."""
    split = dict(zip(splits.patient_id.astype(str).str.strip(), splits.split.astype(str)))
    wrong = [p for p in pd.Series(patient_ids).astype(str).str.strip() if split.get(p) != "train"]
    if wrong:
        raise ValueError(f"{len(wrong)} labelled patients are not in the train split.")


def eligible_for_pseudo(videos):
    """Clips the labelling pool could have held: train split, B-mode or colour flow read as such (a validated
    vendor's mode read from the header), more than a still's frames, a view."""
    validated = videos.get("acq_vendor_validated", pd.Series(False, index=videos.index)).fillna(False).astype(bool)
    confidence = videos.get("acq_confidence", pd.Series("header", index=videos.index)).astype(str)
    known = videos.acq_category.isin(IMAGING_MODES) & (~validated | (confidence == "header"))
    frames = pd.to_numeric(videos.n_frames, errors="coerce")
    return (videos.split.astype(str) == "train") & known & (frames > STILL_MAX_FRAMES).fillna(False) \
        & videos.view.notna()


def select_pseudo_labels(videos, probabilities, train_patients, excluded_patients, mapping=None, threshold=0.9,
                         per_class=150, seed=0, old_classes=vc.OLD_CLASSES):
    """Pseudo-labelled clips.

    videos: patient_id, study_id, video_path, view, split, acq_category, n_frames (+ acq_vendor_validated,
        acq_confidence), one row per clip.
    probabilities: video_path and one column per old class (the classifier's softmax).
    train_patients: patients of the train split (patient_splits.csv); excluded_patients: the labelling pool's.
    mapping: {old class: pseudo-label} (`classes.pseudo_label_classes`; by default that of the default classes).
    Keeps clips whose old class is in `mapping`, whose top probability is at least
    `threshold` and whose argmax is the manifest's view; then at most one clip per patient over all classes (the
    rarest class choosing first, as the pool's draw did) and `per_class` clips per class, at random with `seed`.
    Returns video_path, patient_id, study_id, acq_category, old_class, label, pmax."""
    mapping = vc.pseudo_label_classes(old_classes) if mapping is None else mapping
    v = videos.copy()
    v["patient_id"] = v.patient_id.astype(str).str.strip()
    v = v[eligible_for_pseudo(v) & v.view.isin(list(mapping))]
    v = v[v.patient_id.isin(set(map(str, train_patients))) & ~v.patient_id.isin(set(map(str, excluded_patients)))]
    p = probabilities.drop_duplicates("video_path")
    v = v.merge(p, on="video_path", how="inner")
    prob = v[list(old_classes)].to_numpy(dtype=float)
    ok = np.isfinite(prob).all(axis=1)
    v, prob = v[ok], prob[ok]
    v = v.assign(pmax=prob.max(axis=1), argmax=np.asarray(old_classes)[prob.argmax(axis=1)])
    v = v[(v.pmax >= threshold) & (v.argmax == v.view)]
    v = v.sample(frac=1.0, random_state=seed)
    taken, used = [], set()
    pools = v.groupby("view").size()
    for old in sorted(mapping, key=lambda c: (pools.get(c, 0), c)):
        pool = v[(v.view == old) & ~v.patient_id.isin(used)].drop_duplicates("patient_id").head(per_class)
        taken.append(pool)
        used |= set(pool.patient_id)
    out = pd.concat(taken, ignore_index=True) if taken else v.head(0)
    out = out.assign(old_class=out.view, label=out.view.map(mapping))
    return out[["video_path", "patient_id", "study_id", "acq_category", "old_class", "label", "pmax"]] \
        .sort_values(["label", "video_path"]).reset_index(drop=True)
