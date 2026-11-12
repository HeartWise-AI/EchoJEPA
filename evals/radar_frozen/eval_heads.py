# evals/radar_frozen/eval_heads.py

"""Evaluation of the per-still measurement heads of a RADAR checkpoint, with an outside model's prediction beside
them as a baseline.

Every still of the evaluated split that carries a target is read once (not through bags), with the training split's
target scaler and age statistics from the checkpoint's fingerprint, and the head's prediction is decoded to the
target's unit (m/s, cm, mmHg, ms). Per target: MAE, RMSE, R2, Pearson r, bias (`zero_shot_metrics.regression_metrics`)
and the median absolute error in percent of the measured value, each with a 95 % bootstrap interval over studies.

Three head experiments share this script; what a head read is in the checkpoint's config and is copied to the report
(`head_inputs`):

- tokens with the velocity scale only: how much the frozen tokens hold (the primary result);
- tokens with an outside prediction and demographics: a performance experiment, reported apart;
- side inputs alone (`heads.visual: false`): what is predictable without the image.

Outside baseline (`--baseline_column`, default `echonet_vmax_ms`): for a peak-velocity target (`y_<STRUCT>_VMAX`), on
the stills of that structure that have the outside prediction, the outside prediction is scored against the same
measured values, and the head is scored again on those same stills (`paired`), so the two are compared on equal
ground. An outside model trained on patients or studies that are evaluated here has seen them: the report carries
`baseline_overlap_checked: false` until that overlap is established by whoever holds that model's training list.

Refusals: a checkpoint without heads; the training split (unless `--allow_training_stills`, a sanity check); an
output folder inside the repository (`predictions.csv` holds study identifiers).

Usage:
    python -m evals.radar_frozen.eval_heads --config <training config> --checkpoint <run>/latest.pt \\
        --stills <stills.csv> --split val --out_dir <folder outside the repository>
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from evals.radar_frozen import zero_shot_metrics as zm
from evals.radar_frozen.train import _device
from evals.radar_frozen.zero_shot_eval import _json, check_outside_repo, load_model
from src.datasets.radar_stills import StillSource
from src.datasets.radar_study_dataset import file_sha256
from src.datasets.radar_token_cache import TokenCache
from src.models.radar.heads import decode_targets
from src.models.radar.routing import DEFAULT_ROUTING, load_routing

METRICS = (*zm.REGRESSION_INTERVALS, "median_ape")


def metrics(truth, prediction):
    """The regression metrics of values in the target's unit, with the median absolute percentage error."""
    out = zm.regression_metrics(truth, prediction)
    out["median_ape"] = float(np.median(np.abs(prediction - truth) / truth) * 100.0) if len(truth) else float("nan")
    return out


def scored(truth, prediction, groups, samples, seed):
    if len(truth) < 2:
        return {"n": int(len(truth)), "metrics": None, "intervals": None}
    return {"n": int(len(truth)), "metrics": metrics(truth, prediction),
            "intervals": zm.bootstrap(metrics, (truth, prediction), METRICS, groups=groups, samples=samples,
                                      seed=seed)}


@torch.no_grad()
def predict(model, source, device, batch_size=64):
    """One row per still of `source` with a target: still id, study, tag, z-scored targets, presence and predictions
    ([n, Q] arrays), in a fixed order."""
    rows = [(study, r) for study in sorted(source.by_study) for r in source.by_study[study]
            if r["present"] is not None and bool(r["present"].any())]
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        tokens = torch.stack([source.cache.load(r["still_id"]) for _, r in chunk])[:, None].to(device)
        cond = None
        if source.cond_columns:
            cond = torch.stack([r["cond"] for _, r in chunk])[:, None].to(device)
            if model.head_cond_index is not None:
                cond = cond[..., model.head_cond_index]
        demo = None
        if model.heads.film is not None:
            demo = torch.stack([source.demo.get(study, torch.tensor([0.0, 0.0, 0.0, 1.0])) for study, _ in chunk])
            demo = demo.to(device)
        out.append(model.heads(tokens, cond, demo)[:, 0].float().cpu())
    z = torch.cat(out) if out else torch.zeros(0, len(source.target_columns))
    return {"still_id": [r["still_id"] for _, r in rows], "study_id": [study for study, _ in rows],
            "tag": [r["tag"] for _, r in rows],
            "target_z": torch.stack([r["y"] for _, r in rows]) if rows else z,
            "present": torch.stack([r["present"] for _, r in rows]) if rows else z.bool(), "prediction_z": z}


def evaluate(config, checkpoint, stills, out_dir, split="val", still_cache_dir=None, demographics=None,
             baseline_column="echonet_vmax_ms", samples=1000, seed=0, allow_training_stills=False):
    check_outside_repo(out_dir, "out_dir")
    if not config.get("heads"):
        raise ValueError("The config has no heads section: nothing to evaluate.")
    if split == config["data"].get("split", "train") and not allow_training_stills:
        raise ValueError(f"{split!r} is the split the heads trained on: evaluate another split, or pass "
                         "allow_training_stills for a sanity check.")
    data, m = config["data"], config["model"]
    section = data["stills"]
    routing = load_routing(data.get("routing") or DEFAULT_ROUTING)
    device = _device(config)
    model, _, state = load_model(config, checkpoint, routing, device)
    trained = state["fingerprint"]["stills"]
    targets = list(trained["targets"])
    table = pd.read_csv(stills, dtype={"study_id": str, "still_id": str, "still_tag": str, "still_type": str})
    table = table[table.split.astype(str) == split]
    if table.empty:
        raise ValueError(f"No still of split {split!r}.")
    demographics = demographics or section.get("demographics")
    demo = pd.read_csv(demographics, dtype={"study_id": str}) if demographics else None
    cache = TokenCache(still_cache_dir or section["cache_dir"], token_grid=list(m["stills"]["token_grid"]),
                       embed_dim=m["embed_dim"])
    source = StillSource(table, routing, cache, [s.name for s in routing.trained], cond_columns=trained["cond"],
                         target_columns=targets, scaler=trained["scaler"], demographics=demo,
                         demo_stats=trained.get("demo_stats"))
    rows = predict(model, source, device)
    truth = decode_targets(rows["target_z"], trained["scaler"], targets).numpy().astype(float)
    prediction = decode_targets(rows["prediction_z"], trained["scaler"], targets).numpy().astype(float)
    present = rows["present"].numpy().astype(bool)
    studies, tags = np.asarray(rows["study_id"]), np.asarray(rows["tag"])
    outside = None
    if baseline_column and baseline_column in table.columns:
        by_still = dict(zip(table.still_id.astype(str), pd.to_numeric(table[baseline_column], errors="coerce")))
        outside = np.array([by_still.get(s, np.nan) for s in rows["still_id"]], dtype=float)
    heads = config["heads"]
    report = {"checkpoint_sha256": file_sha256(checkpoint), "stills_sha256": file_sha256(stills), "split": split,
              "checkpoint_epoch": state["epoch"], "stills_scored": len(rows["still_id"]),
              "token_grid": list(m["stills"]["token_grid"]),
              "head_inputs": {"visual": bool(heads.get("visual", True)),
                              "cond_columns": list(heads.get("cond_columns", trained["cond"])),
                              "demographics": bool(heads.get("demographics", True))},
              "training_stills_allowed": bool(allow_training_stills), "baseline_column": baseline_column,
              "baseline_overlap_checked": False,
              "bootstrap": {"samples": samples, "seed": seed, "level": 0.95, "resampled": "study_id"}, "targets": {}}
    frames = []
    for q, target in enumerate(targets):
        use = present[:, q]
        entry = {"head": scored(truth[use, q], prediction[use, q], studies[use], samples, seed)}
        structure, quantity = target.removeprefix("y_").rsplit("_", 1)
        if outside is not None and quantity == "VMAX":
            paired = use & np.isfinite(outside) & (outside > 0) & (tags == structure)
            entry["paired_stills"] = int(paired.sum())
            entry["baseline"] = scored(truth[paired, q], outside[paired], studies[paired], samples, seed)
            entry["head_on_paired"] = scored(truth[paired, q], prediction[paired, q], studies[paired], samples, seed)
        report["targets"][target] = entry
        frames.append(pd.DataFrame({"still_id": np.asarray(rows["still_id"])[use], "study_id": studies[use],
                                    "target": target, "measured": truth[use, q], "prediction": prediction[use, q],
                                    "baseline": outside[use] if outside is not None else np.nan}))
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(out_dir, "head_predictions.csv"), index=False)
    report = _json(report)
    with open(os.path.join(out_dir, "head_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="the config that trained the checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stills", required=True, help="the still table (data/build_still_manifest.py)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--still_cache_dir", help="token cache of the evaluated stills (default: the config's)")
    parser.add_argument("--demographics")
    parser.add_argument("--baseline_column", default="echonet_vmax_ms")
    parser.add_argument("--out_dir", required=True, help="outside the repository: predictions hold study ids")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_training_stills", action="store_true")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    report = evaluate(config, args.checkpoint, args.stills, args.out_dir, split=args.split,
                      still_cache_dir=args.still_cache_dir, demographics=args.demographics,
                      baseline_column=args.baseline_column, samples=args.samples, seed=args.seed,
                      allow_training_stills=args.allow_training_stills)
    print(json.dumps({t: {k: (v["metrics"] or {}).get("mae") if isinstance(v, dict) else v for k, v in e.items()}
                      for t, e in report["targets"].items()}))


if __name__ == "__main__":
    main()
