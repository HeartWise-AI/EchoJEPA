# evals/view_classifier/prepare.py

"""The run's input tables, written outside the repository (they hold paths and patient ids).

`pool` (no labels needed; run first, so the token cache can be written while labelling goes on):
- `pool_clips.csv`: the labelling pool (`picked_clips.csv`) with the classifier's view at the draw.
- `pool_old_probs.csv`: the classifier's 20 probabilities for each pool clip (master metadata, joined on the DICOM
  path), for the baseline's AUROC and the pseudo-label precision by confidence.
- `archive_counts.json`: clips per predicted view in the archive (all splits) and in the pseudo-label-eligible train
  clips, to weight the pool back to the archive's mix.
- `pseudo_candidates.csv`: pseudo-labelled clips (`labels.select_pseudo_labels`).
- `cache_clips.csv`: `clip_id` (video path) and `split` of every clip to encode: the pool and the candidates.

`check` (CPU, writes nothing): the label export's checks, the folds and the current classifier's agreement with the
labels, printed as counts and metrics only. Uses the `pool` tables when they exist, else reads the archive.

`folds` (needs the label export and the cache's `failed.csv`):
- `folds.csv`: the usable labels (`labels.read_labels`) checked against the pool and the train split, without clips
  the decoder cannot read, with `clip_id`, `y` and `fold` (`folds.fold_table`).
- `label_drops.csv`: dropped labels by reason; `class_by_fold.csv`; `pseudo_labels.csv`: the readable candidates;
  `folds.json`: inputs' sha256, counts, settings. Refuses to overwrite an existing `folds.csv`.

    python -m evals.view_classifier.prepare pool --config CONFIG
    python -m evals.view_classifier.prepare check --config CONFIG [--labels EXPORT]
    python -m evals.view_classifier.prepare folds --config CONFIG [--labels EXPORT]

`--labels` replaces `paths.labels` (a later export under another name). When `paths.form` names the label project's
form export, a view it offers that the config's `classes` lacks stops the run. `label_batches` (a list, e.g. [1])
keeps only the export rows of those upload batches (its `batch` column); the other rows are counted as `other_batch`.
"""

import argparse
import json
import os

import pandas as pd
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import folds as vf
from evals.view_classifier import labels as vl

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_COLUMNS = ["patient_id", "study_id", "split", "view", "video_path", "dicom_path", "acq_category",
                 "acq_confidence", "acq_vendor_validated", "n_frames"]


def outside_repo(path):
    root, target = os.path.realpath(REPO), os.path.realpath(path)
    if os.path.commonpath([root, target]) == root:
        raise ValueError(f"{path} is inside the code repository; these tables name patients, write them elsewhere.")


def read_master_probs(path, dicom_paths, old_classes=vc.OLD_CLASSES):
    """dicom_path and the classifier's probabilities (one column per old class) of the listed DICOMs."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["dcm_path", *old_classes])
    table = table.filter(pc.is_in(table["dcm_path"], value_set=pa.array(sorted(set(dicom_paths)))))
    out = table.to_pandas().rename(columns={"dcm_path": "dicom_path"})
    if out.dicom_path.duplicated().any():
        raise ValueError("The master metadata lists a DICOM twice.")
    return out


def pool(cfg):
    p, ps = cfg["paths"], cfg["pseudo"]
    out = p["inputs"]
    outside_repo(out)
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o700)
    vc.read_old_mapping(p["class_mapping"])
    picked = pd.read_csv(p["pool"], dtype=str)
    picked["patient_id"] = picked.patient_id.str.strip()
    if picked.patient_id.duplicated().any():
        print("note: the pool has patients with more than one clip; folds group them.")
    splits = pd.read_csv(p["patient_splits"], dtype=str)
    vl.check_train_split(picked.patient_id, splits)
    picked[["global_key", "patient_id", "study_id", "video_path", "dicom_path", "view", "acq_category"]] \
        .to_csv(os.path.join(out, "pool_clips.csv"), index=False)

    videos = pd.read_parquet(p["videos"], columns=VIDEO_COLUMNS)
    videos["patient_id"] = videos.patient_id.astype(str).str.strip()
    archive = {"all_splits": videos.view.value_counts().to_dict(),
               "pseudo_eligible_train": videos[vl.eligible_for_pseudo(videos)].view.value_counts().to_dict()}
    with open(os.path.join(out, "archive_counts.json"), "w") as f:
        json.dump({k: {str(a): int(b) for a, b in v.items()} for k, v in archive.items()}, f, indent=1)

    train_patients = set(splits.patient_id[splits.split == "train"].str.strip())
    classes, parent = vc.from_config(cfg)
    mapping = vc.pseudo_label_classes(new=classes, parent=parent, exclude=ps.get("exclude_classes", ()))
    candidates = videos[vl.eligible_for_pseudo(videos) & videos.view.isin(list(mapping))
                        & videos.patient_id.isin(train_patients) & ~videos.patient_id.isin(set(picked.patient_id))]
    probs = read_master_probs(p["master"], list(candidates.dicom_path) + list(picked.dicom_path))
    pool_probs = picked[["global_key", "dicom_path"]].merge(probs, on="dicom_path", how="left")
    if pool_probs[list(vc.OLD_CLASSES)].isna().any(axis=None):
        print(f"note: {int(pool_probs[list(vc.OLD_CLASSES)].isna().any(axis=1).sum())} pool clips have no "
              "probabilities in the master metadata.")
    pool_probs.drop(columns=["dicom_path"]).rename(columns={c: f"p_{c}" for c in vc.OLD_CLASSES}) \
        .to_csv(os.path.join(out, "pool_old_probs.csv"), index=False)
    by_video = candidates[["video_path", "dicom_path"]].merge(probs, on="dicom_path").drop(columns=["dicom_path"])
    pseudo = vl.select_pseudo_labels(candidates, by_video, train_patients, set(picked.patient_id), mapping,
                                     threshold=float(ps["threshold"]), per_class=int(ps["per_class"]),
                                     seed=int(ps.get("seed", 0)))
    pseudo.to_csv(os.path.join(out, "pseudo_candidates.csv"), index=False)
    clips = pd.DataFrame({"clip_id": list(picked.video_path) + list(pseudo.video_path), "split": "train"})
    if clips.clip_id.duplicated().any():
        raise ValueError("A pseudo-labelled clip is also in the pool.")
    clips.to_csv(os.path.join(out, "cache_clips.csv"), index=False)
    summary = {"pseudo_label_classes": mapping, "pool_clips": len(picked), "pool_patients": int(picked.patient_id.nunique()),
               "pseudo_clips": len(pseudo), "pseudo_per_class": pseudo.label.value_counts().to_dict(),
               "pseudo_settings": ps, "cache_clips": len(clips),
               "inputs_sha256": {k: vl.file_sha256(p[k]) for k in ("pool", "patient_splits", "class_mapping")}}
    with open(os.path.join(out, "pool.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in ("pool_clips", "pseudo_clips", "cache_clips")}))


def usable_labels(cfg):
    """(usable labels checked against the form, the pool and the split; drop counts)."""
    p = cfg["paths"]
    classes, _ = vc.from_config(cfg)
    if not os.path.exists(p["labels"]):
        raise FileNotFoundError("The label export is missing.")
    merge = vc.label_merge(cfg, classes)
    if p.get("form"):
        unused = vc.check_against_form(classes, p["form"], merge)
        if unused:
            print(f"note: classes the label form does not offer (never a label): {unused}")
    usable, drops = vl.read_labels(p["labels"], classes, batches=cfg.get("label_batches"), merge=merge)
    pool_path = os.path.join(p["inputs"], "pool_clips.csv")
    pool_clips = pd.read_csv(pool_path if os.path.exists(pool_path) else p["pool"], dtype=str)
    usable = vl.check_against_pool(usable, pool_clips)
    vl.check_train_split(usable.patient_id, pd.read_csv(p["patient_splits"], dtype=str))
    return usable, drops, pool_clips


def check(cfg):
    """Print what `folds` and the report's baseline would give on the current export; write nothing."""
    from evals.view_classifier import report as vr
    p, cv = cfg["paths"], cfg["cv"]
    classes, parent = vc.from_config(cfg)
    usable, drops, pool_clips = usable_labels(cfg)
    usable["patient_id"] = usable.patient_id.astype(str).str.strip()
    usable["clip_id"] = usable.video_path
    table = vf.fold_table(usable, k=int(cv["k"]), seed=int(cv.get("seed", 0)))
    stated = usable.predicted_view.astype(str).str.strip()
    print(json.dumps({"usable": len(table), "dropped": dict(zip(drops.reason, drops.n.astype(int))),
                      "patients_with_several_clips": int(usable.patient_id.duplicated().sum()),
                      "predicted_view_differs_from_pool": int(((stated != "") & (stated != usable.pool_view)).sum()),
                      "note": "unreadable videos are dropped by `folds` only (the token cache lists them)"}))
    print("\nlabels per class and fold:")
    print(vf.class_by_fold(table).reindex([c for c in classes if c in set(table.label)]).to_string())
    print("classes without labels:", [c for c in classes if c not in set(table.label)])
    counts_path = os.path.join(p["inputs"], "archive_counts.json")
    if os.path.exists(counts_path):
        archive = json.load(open(counts_path))["all_splits"]
    else:
        archive = pd.read_parquet(p["videos"], columns=["view"]).view.value_counts().to_dict()
    labels = vr.with_targets(table, parent, archive)
    min_support = int(cfg["report"].get("min_class_support", 10))
    evaluated = vr.evaluated_classes(labels, classes, parent, min_support)
    print(f"\nevaluated classes (>= {min_support} labels):", [c for c, e in zip(classes, evaluated["new"]) if e])
    print("below the minimum support:", [c for c, e in zip(classes, evaluated["new"]) if not e])
    probs_path = os.path.join(p["inputs"], "pool_old_probs.csv")
    if os.path.exists(probs_path):
        old = pd.read_csv(probs_path, dtype={"global_key": str})
    else:
        old = pool_clips[["global_key", "dicom_path"]].merge(read_master_probs(p["master"], pool_clips.dicom_path),
                                                             on="dicom_path", how="left")
        old = old.drop(columns=["dicom_path"]).rename(columns={c: f"p_{c}" for c in vc.OLD_CLASSES})
    col = vr.Collector(cfg["report"], evaluated)
    old_vs_new, precision, n = vr.baseline(labels, old, classes, parent, col, cfg)
    print(f"\ncurrent classifier against the labels ({n['with_archive_probabilities']} of {n['labelled']} clips "
          "with archive probabilities; prefilled form, so possibly inflated):")
    s = pd.DataFrame(col.rows)
    keep = ["subset", "granularity", "n", "accuracy", "accuracy_low", "accuracy_high", "macro_f1", "balanced_accuracy",
            "mean_auroc", "macro_f1_all", "archive_weighted_accuracy"]
    print(s[keep].round(3).to_string(index=False))
    print("\nhuman label x current prediction (counts; rows and columns in use):")
    used = old_vs_new.loc[old_vs_new.sum(axis=1) > 0, old_vs_new.sum(axis=0) > 0]
    print(used.to_string())
    print("\npseudo-label classes: precision of the current classifier on the pool")
    print(precision.round(3).to_string(index=False))


def make_folds(cfg):
    p, cv = cfg["paths"], cfg["cv"]
    out = os.path.dirname(p["folds"])
    outside_repo(out)
    if os.path.exists(p["folds"]):
        raise FileExistsError(f"{p['folds']} exists: folds are frozen once written; use a new SNAPSHOT folder.")
    classes, _ = vc.from_config(cfg)
    usable, drops, _ = usable_labels(cfg)
    os.makedirs(out, exist_ok=True)
    stated = usable.predicted_view.astype(str).str.strip()
    prediction_mismatch = int(((stated != "") & (stated != usable.pool_view.astype(str))).sum())
    failed_path = os.path.join(p["token_cache"], "failed.csv")
    if not os.path.exists(failed_path):
        raise FileNotFoundError("The token cache has no failed.csv: build the cache first.")
    unreadable = set(pd.read_csv(failed_path, dtype=str, keep_default_na=False).clip_id)
    gone = usable.video_path.isin(unreadable)
    drops = pd.concat([drops, pd.DataFrame({"reason": ["unreadable_video"], "n": [int(gone.sum())]})],
                      ignore_index=True)
    usable = usable[~gone].copy()
    usable["clip_id"] = usable.video_path
    usable["patient_id"] = usable.patient_id.astype(str).str.strip()
    if usable.patient_id.duplicated().any():
        print("note: some patients have several labelled clips; each patient stays in one fold.")
    table = vf.fold_table(usable, k=int(cv["k"]), seed=int(cv.get("seed", 0)))
    table.to_csv(p["folds"], index=False)
    drops.to_csv(os.path.join(out, "label_drops.csv"), index=False)
    vf.class_by_fold(table).to_csv(os.path.join(out, "class_by_fold.csv"))
    pseudo = pd.read_csv(os.path.join(p["inputs"], "pseudo_candidates.csv"), dtype=str)
    pseudo = pseudo[~pseudo.video_path.isin(unreadable)]
    pseudo.to_csv(p["pseudo_labels"], index=False)
    record = {"labels": p["labels"], "labels_sha256": vl.file_sha256(p["labels"]), "usable": len(table),
              "dropped": dict(zip(drops.reason, drops.n.astype(int))), "pseudo_labels": len(pseudo),
              "label_batches": cfg.get("label_batches"), "k": int(cv["k"]), "seed": int(cv.get("seed", 0)),
              "predicted_view_differs_from_pool": prediction_mismatch,
              "per_class": table.label.value_counts().to_dict(),
              "classes_without_labels": [c for c in classes if c not in set(table.label)]}
    with open(os.path.join(out, "folds.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: record[k] for k in ("usable", "dropped", "pseudo_labels", "classes_without_labels")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("pool", "check", "folds"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels", help="the label export to use instead of paths.labels")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.labels:
        cfg["paths"]["labels"] = args.labels
    {"pool": pool, "check": check, "folds": make_folds}[args.stage](cfg)


if __name__ == "__main__":
    main()
