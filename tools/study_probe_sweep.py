"""Generate, launch and summarize controlled study-probe experiments.

Run from the repository root with `python -m tools.study_probe_sweep --help`.
Generation never launches training. Runs use the existing eval launcher and W&B integration.
"""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

import yaml


def override(config, values):
    for path, value in values.items():
        parts = path.split(".")
        node = config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown config section in {path!r}.")
            node = node[part]
        node[parts[-1]] = copy.deepcopy(value)


def expand(base, spec, design="ablation"):
    """One factor at a time + combined + controls, or the full binary factorial, per seed."""
    base = copy.deepcopy(base)
    override(base, spec.get("defaults", {}))
    if base.get("eval_name") != "study_multitask_frozen":
        raise ValueError("The base config must use study_multitask_frozen.")
    if base.get("test_only") or base.get("val_only"):
        raise ValueError("A sweep base must be a training config.")
    factors = spec["factors"]
    if not factors:
        raise ValueError("The sweep needs at least one factor.")
    names = list(factors)
    if design == "ablation":
        combinations = [()] + [(n,) for n in names] + [tuple(names)]
    elif design == "factorial":
        combinations = [tuple(n for n, on in zip(names, bits) if on)
                        for bits in itertools.product((False, True), repeat=len(names))]
    else:
        raise ValueError("design must be ablation or factorial.")
    variants = []
    for active in dict.fromkeys(combinations):
        values = {}
        for name in active:
            values.update(factors[name])
        name = "baseline" if not active else "all" if len(active) == len(names) else "+".join(active)
        variants.append((name, values, list(active)))
    variants += [(name, values, ["control"]) for name, values in spec.get("controls", {}).items()]
    if len({v[0] for v in variants}) != len(variants):
        raise ValueError("Sweep variant names must be unique.")
    seeds = spec.get("seeds", [0, 1, 2])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty unique training seeds.")
    runs = []
    for name, changes, active in variants:
        if not re.fullmatch(r"[A-Za-z0-9_+.-]+", name):
            raise ValueError(f"Unsafe variant name: {name!r}.")
        for seed in seeds:
            config = copy.deepcopy(base)
            override(config, changes)
            config.setdefault("meta", {})["seed"] = int(seed)
            tag = f"{spec.get('name', 'study-ablation')}-{name}-seed{seed}"
            if not re.fullmatch(r"[A-Za-z0-9_+.-]+", tag):
                raise ValueError("Sweep name and seed must form a safe run tag.")
            config["tag"] = tag
            config["meta"]["wandb_run_name"] = tag
            config["meta"]["study_sweep"] = {"name": spec.get("name"), "variant": name, "factors": active}
            config["resume_checkpoint"] = True
            runs.append((config, {"variant": name, "seed": int(seed), "factors": active}))
    # Bag budget, evaluation seed and selection rule must be identical across comparisons.
    protocols = [json.dumps({"evaluation": c["experiment"].get("evaluation", {}),
                             "eval_seed": c["experiment"]["data"].get("eval_seed", 0),
                             "num_videos": c["experiment"]["data"].get("num_videos"),
                             "encoder": c["model_kwargs"], "optimizer": c["experiment"]["optimization"],
                             "cohort": {k: c["experiment"]["data"].get(k) for k in
                                        ("data_filename", "dataset_train", "dataset_val", "split_column",
                                         "num_clips_per_video", "frames_per_clip", "resolution", "frame_step")},
                             "targets": {k: c["experiment"]["classifier"].get(k) for k in
                                         ("target_label", "head_task", "head_structure", "head_weights",
                                          "loss_structure", "labels_map")}}, sort_keys=True)
                 for c, _ in runs]
    if len(set(protocols)) != 1:
        raise ValueError("All variants must use the same selection rule, evaluation bags, seed, video budget, "
                         "encoder, optimizer and cohort/targets.")
    return runs


def generate(spec_path, output, design="ablation", base_path=None, checkpoint=None, data=None, folder=None):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    spec = yaml.safe_load(spec_path.read_text())
    base_path = Path(base_path).resolve() if base_path else (spec_path.parent / spec["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text())
    if checkpoint:
        base["model_kwargs"]["checkpoint"] = str(Path(checkpoint).resolve())
    if data:
        base["experiment"]["data"]["data_filename"] = str(Path(data).resolve())
    if folder:
        base["folder"] = str(Path(folder).resolve())
    generated = expand(base, spec, design)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (config, metadata) in enumerate(generated):
        path = output / f"{i:03d}-{config['tag']}.yaml"
        text = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != text:
            raise FileExistsError(f"Refusing to replace different sweep config {path}; use a new output directory.")
        entries.append({"index": i, **metadata, "config": str(path),
                        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "result_dir": str((Path(config["folder"]) / config["eval_name"] / config["tag"]).resolve())})
    manifest = {"version": 1, "spec": str(spec_path), "base": str(base_path), "design": design, "runs": entries}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise FileExistsError("A different sweep manifest exists; use a new output directory.")
    for entry, (config, _) in zip(entries, generated):
        Path(entry["config"]).write_text(yaml.safe_dump(config, sort_keys=False))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, len(entries)


def launch(manifest_path, devices, index=None, dry_run=False):
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest["runs"]
    if index is not None:
        entries = [r for r in entries if r["index"] == index]
        if not entries:
            raise ValueError(f"No run index {index} in the sweep.")
    for entry in entries:
        config_path = Path(entry["config"])
        content = config_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["config_sha256"]:
            raise ValueError(f"Sweep config changed: {config_path}. Regenerate in a new directory.")
        command = [sys.executable, "-m", "evals.main", "--fname", str(config_path), "--devices", *devices]
        print(json.dumps({"index": entry["index"], "command": command}), flush=True)
        if dry_run:
            continue
        config = yaml.safe_load(content)
        data = config["experiment"]["data"]
        required = [config["model_kwargs"]["checkpoint"]]
        required += ([data["data_filename"]] if data.get("data_filename")
                     else [data.get("dataset_train"), data.get("dataset_val")])
        if any(p is None or not Path(p).is_file() for p in required):
            raise FileNotFoundError("Set real checkpoint and train/validation manifest paths before launching.")
        result_dir = Path(entry["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        # Prevent two agents/jobs from mutating the same run. The OS releases the lock on exit.
        import fcntl
        with (result_dir / "sweep.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"index": entry["index"], "skipped": "locked by another job"}), flush=True)
                continue
            subprocess.run(command, check=True)


def summarize(manifest_path, output):
    manifest = json.loads(Path(manifest_path).read_text())
    rows, missing = [], []
    for entry in manifest["runs"]:
        path = Path(entry["result_dir"]) / "val_metrics.json"
        if not path.exists():
            missing.append(entry["index"])
            continue
        result = json.loads(path.read_text())
        if result.get("selected_on") != "validation":
            raise ValueError(f"Expected selected validation results at {path}.")
        row = {"index": entry["index"], "variant": entry["variant"], "seed": entry["seed"],
               "epoch": result["epoch"], "probe": result["probe"], "score": result["val"]["score"],
               "failed_studies": result["val"]["failed"]}
        for head, metrics in result["val"]["heads"].items():
            row.update({f"{head}/{metric}": value for metric, value in metrics.items()})
        row.update({f"{head}/bag_std": value
                    for head, value in result["val"].get("mean_bag_prediction_std", {}).items()})
        rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"missing_run_indices": missing, "variants": {}}
    baseline = {r["seed"]: r for r in rows if r["variant"] == "baseline"}
    metrics = [c for c in columns if c not in {"index", "variant", "seed", "epoch", "probe"}]
    for variant in dict.fromkeys(r["variant"] for r in rows):
        group = [r for r in rows if r["variant"] == variant]
        aggregates = {}
        for metric in metrics:
            values = [r[metric] for r in group if metric in r and math.isfinite(r[metric])]
            deltas = [r[metric] - baseline[r["seed"]][metric] for r in group
                      if r["seed"] in baseline and metric in r and metric in baseline[r["seed"]]
                      and math.isfinite(r[metric]) and math.isfinite(baseline[r["seed"]][metric])]
            aggregates[metric] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                                  "std": statistics.stdev(values) if len(values) > 1 else None,
                                  "paired_delta_n": len(deltas),
                                  "paired_delta_mean": statistics.mean(deltas) if deltas else None}
        summary["variants"][variant] = aggregates
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Write configs and a manifest; do not launch training")
    gen.add_argument("--spec", default="configs/sweeps/study_probe.yaml")
    gen.add_argument("--output", required=True)
    gen.add_argument("--design", choices=("ablation", "factorial"), default="ablation")
    gen.add_argument("--base")
    gen.add_argument("--checkpoint")
    gen.add_argument("--data")
    gen.add_argument("--folder")
    run = commands.add_parser("run", help="Launch/resume experiments, sequentially or one scheduler index")
    run.add_argument("--manifest", required=True)
    run.add_argument("--devices", nargs="+", default=["cuda:0"])
    run.add_argument("--index", type=int)
    run.add_argument("--dry-run", action="store_true")
    report = commands.add_parser("summarize", help="Summarize selected validation checkpoints, never test results")
    report.add_argument("--manifest", required=True)
    report.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        path, count = generate(args.spec, args.output, args.design, args.base, args.checkpoint, args.data, args.folder)
        print(f"Generated {count} runs: {path}")
    elif args.command == "run":
        launch(args.manifest, args.devices, args.index, args.dry_run)
    else:
        result = summarize(args.manifest, args.output)
        print(f"Summarized {len(result['variants'])} variants; {len(result['missing_run_indices'])} runs missing.")


if __name__ == "__main__":
    main()
