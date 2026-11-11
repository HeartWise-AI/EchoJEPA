# evals/radar_frozen/zero_shot_eval.py

"""Zero-shot evaluation of a trained RADAR checkpoint against study-level labels, without a task head.

Reads a checkpoint written by `evals/radar_frozen/train.py` with the config that trained it, the clip table and token
cache of an evaluation split, a study-level label table (`data/make_study_targets.py`) and a prompt file
(`configs/radar/prompts_v0.yaml`, read by `src.models.radar.zero_shot`).

Predictions. Each study is embedded from `num_bags` evaluation bags, chosen by `src.datasets.study_bags` from
(eval_seed, bag, study) exactly as the matched baseline chooses its bags (`configs/sweeps/radar_matched_baseline.yaml`:
eval_seed 0, 3 bags), so with the same clip table, routing registry, selection and number of clips both methods read
the same clips. Each bag gives one embedding per trained structure; a task is scored on its structure's embedding
against its prompts, encoded by the checkpoint's caption encoder, at the checkpoint's temperature. The study's
prediction is the mean over the bags that support it: the probability of the finding (binary), the estimate
(regression), or the grade probabilities, whose most probable grade is the prediction (categorical).

Support. A bag supports a task when it holds a clip that the routing registry lets the task's structure read. A study
none of whose bags supports a task has `prediction_supported` false and no prediction: nothing is imputed, and the
study is left out of the task's metrics. `coverage` is the share of labelled studies that have a prediction, and the
positives among the unsupported ones are counted (binary), because a comparison with a model that predicts for every
study must be made on the same studies or must count these as misses.

Metrics (`zero_shot_metrics`), each with a 95 % percentile bootstrap interval over studies, or over the groups of
`--group_column` of the label table (for example the patient, when a patient has several studies):

- binary: prevalence, AUROC, AUPRC, Brier score, expected calibration error, calibration intercept and slope, and at
  the operating threshold precision, recall, specificity and F1;
- regression: MAE, RMSE, R2, Pearson r and bias;
- categorical: accuracy, macro F1 and macro one-against-rest AUROC.

Operating threshold. 0.5 (the finding's prompt set is the closer one) unless `--thresholds` names the
`thresholds.json` of another run of this script on a tuning split, with the same checkpoint, prompt file and pooling.
Every run writes its own `thresholds.json` (Youden's J per binary task on the split it read) for that use, and a run
refuses the thresholds of its own split: a threshold chosen on the reported split makes the metrics at that threshold
optimistic. `threshold_source` in the report says which was used.

Refusals. A config whose model or caption-encoder section or routing registry is not the checkpoint's; an evaluated
study that is in the training split of the config, or a training clip table that is not the one the checkpoint
trained on (`--allow_training_studies` skips both, for a sanity check on training studies); a label column that is
missing, a label that is not a number, a binary label that is not 0 or 1, a grade that is not one of the task's
`values` (a blank label is a missing label, and the study is left out of that task); a prompt longer than the caption
encoder reads; a task whose structure is not trained; an output folder inside the repository.

Leakage between patients: a clip table does not name the patient, so the split must have been made by patient. With
`--group_column` naming the patient in the label table, a patient who has both an evaluated study and a study of the
training split is refused as well; that covers only the training studies the label table holds, and
`training_studies_checked_by_group` in the report says how many those were (0: nothing was checked).

Outputs in `out_dir`, which must be outside the repository because `predictions.csv` holds study identifiers:

- `predictions.csv`: one row per (study, task): `study_id`, `task`, `mode`, `structure`, `label`, `prediction`,
  `prediction_supported`, `supported_bags`, and for a categorical task `probabilities` (JSON, in grade order);
- `metrics.json`: per task the coverage, the metrics and their intervals, and what the run read (checksums of the
  checkpoint, clip table, label table and prompt file, the bag record, the temperature, the pooling);
- `thresholds.json`.

The evaluation cache may be another cache than the training one (another split), written by the same frozen encoder:
the provenance keys on which the two caches differ are listed in `metrics.json` (`cache_provenance_differences`),
and should be only those naming the clips.

Weights & Biases (optional): `--wandb_project` with `--wandb_entity` logs every metric, interval bound and coverage
as scalars at the checkpoint's step, with the run's settings (file names instead of paths). No study identifier and
no prediction is sent. `held_out` prompts of a prompt file are not scored here.

Usage:
    python -m evals.radar_frozen.zero_shot_eval --config <training config> --checkpoint <run>/latest.pt \\
        --clips <clip table> --split val --labels <study_targets.csv> --out_dir <folder outside the repository>
"""

import argparse
import json
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import yaml

from evals.radar_frozen import zero_shot_metrics as zm
from evals.radar_frozen.train import AMP_MODES, REPO, WORKER_START_METHODS, _device, _to, build_model
from src.datasets.radar_stills import STILL_ITEM_KEYS, StillSource, same_encoder
from src.datasets.radar_study_dataset import RadarStudyDataset, file_sha256, read_table
from src.datasets.radar_token_cache import TokenCache
from src.models.radar.model import STILL_KEYS
from src.models.radar.routing import DEFAULT_ROUTING, load_routing
from src.models.radar.zero_shot import (
    POOLINGS,
    binary_probability,
    categorical_probabilities,
    encode_prompts,
    load_prompt_tasks,
    regression_estimate,
    regression_prompts,
)
from src.utils.wandb_logging import PRIVATE_WANDB_SETTINGS, finish_wandb, init_wandb, log_scalars

DEFAULT_PROMPTS = os.path.join(REPO, "configs", "radar", "prompts_v0.yaml")
BAG_KEYS = ("tokens", "clip_valid", "view_ids", "acquisition_ids", "eligible")
CAPTION_COLUMNS = ("study_id", "structure", "caption", "caption_id", "normality")
INTERVALS = {"binary": zm.BINARY_INTERVALS, "regression": zm.REGRESSION_INTERVALS,
             "categorical": zm.CATEGORICAL_INTERVALS}
DEFAULT_THRESHOLD = 0.5


def check_outside_repo(path, flag):
    """Refuse an output folder inside this repository: its files hold study identifiers."""
    root, target = os.path.realpath(REPO), os.path.realpath(path)
    if os.path.commonpath([root, target]) == root:
        raise ValueError(f"{flag} {path} is inside the code repository ({root}). predictions.csv holds study "
                         "identifiers; write it outside the repository.")


def load_model(config, checkpoint, routing, device):
    """(model in eval mode, tokenizer, checkpoint state), after checking that `config` builds the checkpoint's
    model."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained = state["fingerprint"]

    def acquisition(c):
        return list(c.get("data", {}).get("acquisition_classes", ("bmode", "color_flow")))

    problems = [f"the {section} section differs" for section in ("model", "text")
                if trained["config"].get(section) != config.get(section)]
    if acquisition(trained["config"]) != acquisition(config):
        problems.append("data.acquisition_classes differ")
    if trained["routing_sha256"] != routing.sha256:
        problems.append("another routing registry")
    if problems:
        raise ValueError(f"The config does not describe the model of {checkpoint}: {'; '.join(problems)}. "
                         "Evaluate a checkpoint with the config that trained it.")
    model, tokenizer = build_model(config, routing)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), tokenizer, state


def check_not_trained_on(config, state, study_ids):
    """Refuse when an evaluated study is in the training split, or when that cannot be told. Returns the training
    split's study ids: those of the clip table and, for a checkpoint trained with Doppler stills, those of its still
    table too (a study may have trained through its stills alone)."""
    data = config["data"]
    if not os.path.exists(data["clips"]) or file_sha256(data["clips"]) != state["fingerprint"]["clips_sha256"]:
        raise ValueError("data.clips of the config is not the clip table the checkpoint trained on, so the evaluated "
                         "studies cannot be checked against the training studies. Point the config at that table, "
                         "or pass allow_training_studies to skip the check.")
    train = read_table(data["clips"])
    if "split" in train.columns:
        train = train[train.split.astype(str) == data.get("split", "train")]
    trained = set(train.study_id.astype(str))
    if state["fingerprint"].get("stills"):
        path = (data.get("stills") or {}).get("table")
        if not path or not os.path.exists(path) or file_sha256(path) != state["fingerprint"]["stills"]["table_sha256"]:
            raise ValueError("data.stills.table of the config is not the still table the checkpoint trained on, so "
                             "the evaluated studies cannot be checked against the studies trained through their "
                             "stills. Point the config at that table, or pass allow_training_studies to skip the "
                             "check.")
        stills = pd.read_csv(path, dtype=str, usecols=lambda c: c in ("study_id", "split"))
        if "split" in stills.columns:
            stills = stills[stills.split.astype(str) == data.get("split", "train")]
        trained |= set(stills.study_id.astype(str))
    overlap = trained & set(study_ids)
    if overlap:
        raise ValueError(f"{len(overlap)} of the {len(study_ids)} evaluated studies are in the training split: "
                         "their zero-shot predictions are not held out. Evaluate another split, or pass "
                         "allow_training_studies for a sanity check on training studies.")
    return trained


def check_groups_not_trained_on(labels, group_column, study_ids, training_ids):
    """Refuse when a group (for example a patient) of an evaluated study also has a training study in the label
    table. Returns the number of training studies whose group the table gives: the check covers only those."""
    table = pd.read_csv(labels, dtype=str, keep_default_na=False)
    group = table[group_column].str.strip()
    known = group != ""
    trained = set(group[known & table.study_id.isin(training_ids)])
    shared = trained & set(group[known & table.study_id.isin(set(study_ids))])
    if shared:
        raise ValueError(f"{len(shared)} values of {group_column!r} have both an evaluated study and a training "
                         "study: the evaluation is not held out by group. Pass allow_training_studies to skip the "
                         "check.")
    return int((known & table.study_id.isin(training_ids)).sum())


def collate_bags(items):
    batch = {key: torch.stack([item[key] for item in items]) for key in BAG_KEYS}
    for key in STILL_ITEM_KEYS:         # the still stream, when the dataset has one
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    batch["study_id"] = [item["study_id"] for item in items]
    return batch


@torch.no_grad()
def embed_studies(model, dataset, device, amp="none", num_bags=3, batch_size=32, num_workers=0,
                  start_method="spawn"):
    """Structure embeddings z [bags, N, S, P] (float32) and has_clip [bags, N, S] of the dataset's studies, in the
    dataset's order, on the CPU. Bag b is the dataset's evaluation bag b."""
    if dataset.training:
        raise ValueError("Evaluation bags need a dataset built with training=False.")
    if start_method not in WORKER_START_METHODS:
        raise ValueError(f"worker_start_method must be one of {WORKER_START_METHODS}, not {start_method!r}.")
    workers = {"multiprocessing_context": start_method, "persistent_workers": False} if num_workers > 0 else {}
    z_bags, has_bags = [], []
    for bag in range(num_bags):
        dataset.eval_bag_index = bag            # workers are not persistent: each loader's workers see this bag
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             collate_fn=collate_bags, **workers)
        zs, has = [], []
        for batch in loader:
            batch = _to(batch, device)
            extra = {key: batch.get(key) for key in STILL_KEYS} if "still_tokens" in batch else {}
            if "demo" in batch and getattr(model.pooler, "demo_film", None) is not None:
                extra["demo"] = batch["demo"]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp == "bf16"):
                _, z, has_clip = model.pooler(batch["tokens"], batch["clip_valid"], batch["eligible"],
                                              batch["view_ids"], batch["acquisition_ids"], **extra)
            zs.append(z.float().cpu())
            has.append(has_clip.cpu())
        z_bags.append(torch.cat(zs))
        has_bags.append(torch.cat(has))
    return torch.stack(z_bags), torch.stack(has_bags)


def encode_task(task, text_encoder, tokenizer):
    """The task's prompt embeddings, on the CPU: {"values", "prompts"} (regression), {"positive", "negative"}
    (binary) or {"grades", "sets"} (categorical: one set per distinct value, in increasing order)."""
    def encode(prompts):
        prompts = list(prompts)
        too_long = sum(n > tokenizer.max_length for n in tokenizer.lengths(prompts))
        if too_long:
            raise ValueError(f"{task.name}: {too_long} prompts are longer than the {tokenizer.max_length} tokens the "
                             "caption encoder reads; a truncated prompt is another prompt.")
        return encode_prompts(text_encoder, tokenizer, prompts).cpu()

    if task.mode == "regression":
        pairs = regression_prompts(task.label_sources, *task.value_range)
        return {"values": [v for v, _ in pairs], "prompts": encode([p for _, p in pairs])}
    if task.mode == "binary":
        return {"positive": encode(task.label_sources), "negative": encode(task.negative_sources)}
    grades = sorted(set(task.values))
    return {"grades": grades,
            "sets": [encode([p for p, v in zip(task.label_sources, task.values) if v == g]) for g in grades]}


def score_task(task, z, encoded, temperature, pooling="mean_similarity", keep=0.2):
    """One bag's predictions for z [N, P]: [N] (binary probability, regression estimate) or [N, G] (grades)."""
    if task.mode == "regression":
        return regression_estimate(z, encoded["prompts"], encoded["values"], keep=keep)
    if task.mode == "binary":
        return binary_probability(z, encoded["positive"], encoded["negative"], temperature, pooling=pooling)
    return categorical_probabilities(z, encoded["sets"], temperature)


def pool_bags(per_bag, supported):
    """The mean over each study's supporting bags of `per_bag` [bags, N] or [bags, N, G], as float64 NumPy; NaN for
    a study without a supporting bag. Also the number of supporting bags [N]."""
    values = per_bag.double().numpy()
    weight = supported.numpy().astype(float)
    count = weight.sum(axis=0)
    weight = weight.reshape(weight.shape + (1,) * (values.ndim - 2))
    total = (np.where(weight > 0, values, 0.0) * weight).sum(axis=0)
    denominator = count.reshape(count.shape + (1,) * (values.ndim - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(denominator > 0, total / denominator, np.nan)
    return mean, count.astype(int)


def read_labels(path, tasks, study_ids, group_column=None):
    """The label table aligned to `study_ids`: one float column per task target (NaN where the label is blank or the
    study has no row), `group` (text, blank without a row) when `group_column` is given, and the number of label rows
    whose study is not evaluated."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "study_id" not in table.columns:
        raise ValueError("The label table needs a study_id column.")
    if table.study_id.duplicated().any():
        raise ValueError(f"{int(table.study_id.duplicated().sum())} label rows repeat a study: one row per study.")
    targets = sorted({task.target for task in tasks.values()})
    missing = [c for c in targets + ([group_column] if group_column else []) if c not in table.columns]
    if missing:
        raise ValueError(f"The label table has no column {missing}: a task's `target` names its label column.")
    unused = int((~table.study_id.isin(set(study_ids))).sum())
    table = table.set_index("study_id").reindex(study_ids)
    labels = pd.DataFrame(index=pd.Index(study_ids, name="study_id"))
    for column in targets:
        text = table[column].fillna("").str.strip()
        values = pd.to_numeric(text.where(text != ""), errors="coerce").to_numpy(dtype=float)
        bad = int(((text != "").to_numpy() & ~np.isfinite(values)).sum())
        if bad:
            raise ValueError(f"{bad} labels of column {column!r} are not numbers. A missing label is a blank cell.")
        labels[column] = values
    if group_column:
        labels["group"] = table[group_column].fillna("").str.strip().to_numpy()
    return labels, unused


def check_task_labels(task, values, grades=None):
    """Refuse labels (the non-missing ones) that the task cannot be scored against."""
    if task.mode == "binary" and not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"{task.name}: labels of column {task.target!r} must be 0 or 1.")
    if task.mode == "categorical" and not np.isin(values, np.asarray(grades, dtype=float)).all():
        raise ValueError(f"{task.name}: a label of column {task.target!r} is not one of the task's values {grades}.")


def read_thresholds(path, source, tasks):
    """{task: threshold} of another run's `thresholds.json`, refused when it was chosen on this run's split, for
    another checkpoint, prompt file or pooling, or when it lacks one of this run's binary tasks."""
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    chosen_on = saved["source"]
    if all(chosen_on.get(key) == source[key] for key in ("clips_sha256", "split")):
        raise ValueError(f"{path} was chosen on the split being reported. Choose thresholds on a tuning split.")
    other = [key for key in ("checkpoint_sha256", "prompts_sha256", "pooling") if chosen_on.get(key) != source[key]]
    if other:
        raise ValueError(f"{path} was chosen with another {', '.join(other)}: its thresholds do not apply here.")
    binary = [name for name, task in tasks.items() if task.mode == "binary"]
    absent = [name for name in binary if saved["thresholds"].get(name) is None]
    if absent:
        raise ValueError(f"{path} has no threshold for {absent}.")
    return {name: float(saved["thresholds"][name]) for name in binary}


def _json(value):
    """`value` with NaN as null and NumPy scalars as Python numbers: valid JSON."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def task_report(task, label, prediction, supported, groups, grades, threshold, samples, seed):
    """Coverage, metrics and intervals of one task from the studies' labels [N], pooled predictions ([N] or [N, G])
    and support [N]. Also Youden's threshold on these studies (binary, else None)."""
    labelled = np.isfinite(label)
    used = labelled & supported
    coverage = {"studies": int(label.size), "studies_supported": int(supported.sum()),
                "labelled": int(labelled.sum()), "labelled_supported": int(used.sum()),
                "coverage": float(used.sum() / labelled.sum()) if labelled.any() else None}
    y, p = label[used], prediction[used]
    check_task_labels(task, label[labelled], grades)
    if task.mode == "binary":
        coverage["unsupported_positives"] = int((label[labelled & ~supported] == 1).sum())
        metric = partial(zm.binary_metrics, threshold=threshold)
    elif task.mode == "regression":
        metric = zm.regression_metrics
    else:
        metric = partial(zm.categorical_metrics, grades=grades)
    intervals = zm.bootstrap(metric, (y, p), INTERVALS[task.mode], groups=None if groups is None else groups[used],
                             samples=samples, seed=seed)
    youden = zm.probability_threshold(y, p) if task.mode == "binary" else None
    return {"coverage": coverage, "metrics": metric(y, p), "intervals": intervals}, youden


def read_eval_stills(config, state, routing, model, path, split, cache_dir=None, demographics=None):
    """The still source of the evaluated studies, read as in training: the config's still settings, the training
    split's age statistics (from the checkpoint's fingerprint), no measurement target."""
    section, m = config["data"]["stills"], config["model"]
    table = pd.read_csv(path, dtype={"study_id": str, "still_id": str, "still_tag": str, "still_type": str})
    if split is not None:
        if "split" not in table.columns:
            raise ValueError("split was given but the still table has no split column.")
        table = table[table.split.astype(str) == split]
    demographics = demographics or section.get("demographics")
    if getattr(model.pooler, "demo_film", None) is not None and not demographics:
        raise ValueError("The checkpoint's queries read demographics (model.stills.demo_film): pass demographics.")
    demo = pd.read_csv(demographics, dtype={"study_id": str}) if demographics else None
    cache = TokenCache(cache_dir or section["cache_dir"], token_grid=list(m["stills"]["token_grid"]),
                       embed_dim=m["embed_dim"])
    return StillSource(table, routing, cache, [s.name for s in routing.trained],
                       num_stills=section.get("num_stills", 8), cond_columns=section.get("cond_columns", []),
                       heatmap_column=section.get("heatmap_column"), demographics=demo,
                       demo_stats=state["fingerprint"]["stills"].get("demo_stats"))


def evaluate(config, checkpoint, clips, labels, out_dir, prompts=DEFAULT_PROMPTS, split=None, cache_dir=None,
             num_bags=3, eval_seed=0, num_clips=None, selection=None, pooling="mean_similarity", keep=0.2,
             thresholds=None, group_column=None, samples=1000, seed=0, batch_size=32, num_workers=0,
             allow_training_studies=False, wandb=None, stills=None, still_cache_dir=None, demographics=None,
             clip_only=False):
    """Evaluate and write the outputs; returns the report written to `metrics.json`. `clips`, `labels`, `prompts`,
    `checkpoint`, `thresholds` are paths; `num_clips` and `selection` default to the training config's; `wandb` is
    {wandb_project, wandb_entity, wandb_run_name} or None. A checkpoint trained with Doppler stills is evaluated
    with them: `stills` is the evaluated studies' still table (`split` applies to it too), `still_cache_dir` and
    `demographics` default to the config's. `clip_only` evaluates such a checkpoint on clips alone, an ablation that
    the report names; without either the evaluation is refused, or the still stream would be dropped silently."""
    check_outside_repo(out_dir, "out_dir")
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}.")
    if num_bags < 1:
        raise ValueError("num_bags must be at least 1.")
    amp = config["optimization"].get("amp", "none")
    if amp not in AMP_MODES:
        raise ValueError(f"amp must be one of {AMP_MODES}.")
    data, m = config["data"], config["model"]
    device = _device(config)
    routing = load_routing(data.get("routing") or DEFAULT_ROUTING)
    model, tokenizer, state = load_model(config, checkpoint, routing, device)
    structures = [s.name for s in routing.trained]
    tasks = load_prompt_tasks(prompts)
    untrained = sorted({task.structure for task in tasks.values()} - set(structures))
    if untrained:
        raise ValueError(f"Prompt tasks score structures that are not trained: {untrained}.")

    trained_stills = state["fingerprint"].get("stills")
    if stills and (not trained_stills or clip_only):
        raise ValueError("stills were given, but the checkpoint trained without stills or clip_only is set.")
    if trained_stills and not stills and not clip_only:
        raise ValueError("The checkpoint trained with Doppler stills: pass stills (the evaluated studies' still "
                         "table), or clip_only for an evaluation on clips alone.")
    if clip_only and getattr(model.pooler, "demo_film", None) is not None:
        raise ValueError("clip_only cannot evaluate a checkpoint whose queries read demographics "
                         "(model.stills.demo_film): they come with the still source.")
    still_source = (read_eval_stills(config, state, routing, model, stills, split, still_cache_dir, demographics)
                    if stills else None)

    table = read_table(clips)
    if split is not None:
        if "split" not in table.columns:
            raise ValueError("split was given but the clip table has no split column.")
        table = table[table.split.astype(str) == split]
    if table.empty:
        raise ValueError("No clip to evaluate.")
    cache = TokenCache(cache_dir or data["cache_dir"], token_grid=m["token_grid"], embed_dim=m["embed_dim"])
    if still_source is not None:
        same_encoder(cache.meta, still_source.cache.meta)
    dataset = RadarStudyDataset(table, pd.DataFrame(columns=list(CAPTION_COLUMNS)), routing, cache,
                                num_clips=num_clips or data.get("num_clips", 12),
                                selection=selection or data.get("selection", "view_balanced"), training=False,
                                eval_seed=eval_seed,
                                acquisition_classes=data.get("acquisition_classes", ("bmode", "color_flow")),
                                stills=still_source)
    study_ids = [study for study, _ in dataset.studies]
    training_ids = None if allow_training_studies else check_not_trained_on(config, state, study_ids)
    label_table, unused_rows = read_labels(labels, tasks, study_ids, group_column)
    groups = label_table["group"].to_numpy() if group_column else None
    group_checked = None
    if group_column and training_ids is not None:
        group_checked = check_groups_not_trained_on(labels, group_column, study_ids, training_ids)

    source = {"checkpoint_sha256": file_sha256(checkpoint), "clips_sha256": file_sha256(clips),
              "labels_sha256": file_sha256(labels), "prompts_sha256": file_sha256(prompts), "split": split,
              "pooling": pooling}
    chosen = read_thresholds(thresholds, source, tasks) if thresholds else {}

    temperature = float(model.loss.effective_temperature().detach())
    z, has_clip = embed_studies(model, dataset, device, amp, num_bags, batch_size, num_workers,
                                config["optimization"].get("worker_start_method", "spawn"))
    trained_cache = state["fingerprint"]["cache_meta"]
    provenance, trained_provenance = cache.meta.get("provenance", {}), trained_cache.get("provenance", {})
    report = {
        **source,
        "checkpoint_fingerprint": state["fingerprint"]["sha256"], "checkpoint_epoch": state["epoch"],
        "checkpoint_step": state["step"], "temperature": temperature, "keep": keep, "num_bags": num_bags,
        "bags": dataset.bag_record(), "training_studies_allowed": bool(allow_training_studies),
        "studies": len(study_ids), "label_rows_not_evaluated": unused_rows,
        "training_studies_checked_by_group": group_checked,
        "same_cache_as_training": cache.meta == trained_cache,
        "clip_only": bool(clip_only),
        "stills": None if still_source is None else {
            "table_sha256": file_sha256(stills), "studies": len(still_source.studies),
            "unrouted": still_source.unrouted,
            "same_cache_as_training": still_source.cache.meta == trained_stills["cache_meta"]},
        "cache_provenance_differences": sorted(k for k in set(provenance) | set(trained_provenance)
                                               if provenance.get(k) != trained_provenance.get(k)),
        "bootstrap": {"samples": samples, "seed": seed, "level": 0.95,
                      "resampled": group_column if group_column else "study_id"},
        "tasks": {},
    }
    rows, youden = [], {}
    for name, task in tasks.items():
        s = structures.index(task.structure)
        encoded = encode_task(task, model.text, tokenizer)
        per_bag = torch.stack([score_task(task, z[b, :, s], encoded, temperature, pooling, keep)
                               for b in range(num_bags)])
        pooled, bags = pool_bags(per_bag, has_clip[:, :, s])
        supported = bags > 0
        label = label_table[task.target].to_numpy(dtype=float)
        if groups is not None and (groups[np.isfinite(label) & supported] == "").any():
            raise ValueError(f"{name}: a labelled study has no {group_column!r} to resample by.")
        grades = encoded.get("grades")
        threshold = chosen.get(name, DEFAULT_THRESHOLD)
        result, youden[name] = task_report(task, label, pooled, supported, groups, grades, threshold, samples, seed)
        if task.mode == "binary":
            result["threshold_source"] = "tuning_split" if name in chosen else "default_0.5"
        report["tasks"][name] = {"mode": task.mode, "structure": task.structure, "target": task.target, **result}
        prediction = pooled if task.mode != "categorical" else np.where(
            supported, np.asarray(grades, dtype=float)[np.nan_to_num(pooled).argmax(axis=1)], np.nan)
        frame = pd.DataFrame({"study_id": study_ids, "task": name, "mode": task.mode, "structure": task.structure,
                              "label": label, "prediction": prediction, "prediction_supported": supported,
                              "supported_bags": bags})
        frame["probabilities"] = ([json.dumps([round(float(x), 6) for x in row]) if ok else ""
                                   for row, ok in zip(pooled, supported)] if task.mode == "categorical" else "")
        rows.append(frame)

    os.makedirs(out_dir, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    report = _json(report)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"source": source, "thresholds": {k: v for k, v in youden.items()
                                                    if tasks[k].mode == "binary"}}, f, indent=2, sort_keys=True)
    log_report(report, wandb, out_dir)
    return report


def scalars(report):
    """The report's numbers as flat W&B scalars: `zero_shot/<task>/<metric>`, with `_low` and `_high` bounds."""
    out = {"zero_shot/temperature": report["temperature"], "zero_shot/studies": report["studies"]}
    for name, task in report["tasks"].items():
        prefix = f"zero_shot/{name}"
        for key, value in {**task["coverage"], **task["metrics"]}.items():
            if value is not None:
                out[f"{prefix}/{key}"] = value
        for key, interval in task["intervals"].items():
            if interval["low"] is not None:
                out[f"{prefix}/{key}_low"], out[f"{prefix}/{key}_high"] = interval["low"], interval["high"]
    return out


def log_report(report, wandb, out_dir):
    """Log the report to W&B when `wandb` names a project: scalars and settings only, no study identifier."""
    settings = {k: v for k, v in report.items() if k != "tasks"}
    run, _ = init_wandb(wandb or {}, settings, 0, out_dir, settings=PRIVATE_WANDB_SETTINGS)
    log_scalars(run, scalars(report), step=report["checkpoint_step"])
    finish_wandb(run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="the config that trained the checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clips", required=True, help="clip table of the evaluation split")
    parser.add_argument("--split", help="keep the rows of this split of the clip table")
    parser.add_argument("--cache_dir", help="token cache holding the evaluation clips (default: the config's)")
    parser.add_argument("--labels", required=True, help="study-level label table (data/make_study_targets.py)")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--out_dir", required=True, help="outside the repository: predictions hold study ids")
    parser.add_argument("--num_bags", type=int, default=3)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--num_clips", type=int, help="clips per bag (default: the config's)")
    parser.add_argument("--selection", help="bag selection (default: the config's)")
    parser.add_argument("--pooling", default="mean_similarity", choices=POOLINGS)
    parser.add_argument("--keep", type=float, default=0.2, help="regression: fraction of best-ranked values kept")
    parser.add_argument("--thresholds", help="thresholds.json of a run on a tuning split")
    parser.add_argument("--group_column", help="label-table column to resample by (for example the patient)")
    parser.add_argument("--samples", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--allow_training_studies", action="store_true",
                        help="evaluate studies of the training split (a sanity check, not a held-out result)")
    parser.add_argument("--stills", help="still table of the evaluated studies (a checkpoint trained with stills)")
    parser.add_argument("--still_cache_dir", help="token cache of the evaluated stills (default: the config's)")
    parser.add_argument("--demographics", help="demographics of the evaluated studies (default: the config's)")
    parser.add_argument("--clip_only", action="store_true",
                        help="evaluate a checkpoint trained with stills on clips alone (an ablation)")
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_run_name")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    wandb = {"wandb_project": args.wandb_project, "wandb_entity": args.wandb_entity,
             "wandb_run_name": args.wandb_run_name}
    report = evaluate(config, args.checkpoint, args.clips, args.labels, args.out_dir, prompts=args.prompts,
                      split=args.split, cache_dir=args.cache_dir, num_bags=args.num_bags, eval_seed=args.eval_seed,
                      num_clips=args.num_clips, selection=args.selection, pooling=args.pooling, keep=args.keep,
                      thresholds=args.thresholds, group_column=args.group_column, samples=args.samples,
                      seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
                      allow_training_studies=args.allow_training_studies, wandb=wandb, stills=args.stills,
                      still_cache_dir=args.still_cache_dir, demographics=args.demographics,
                      clip_only=args.clip_only)
    print(json.dumps({name: {**task["coverage"], **task["metrics"]} for name, task in report["tasks"].items()}))


if __name__ == "__main__":
    main()
