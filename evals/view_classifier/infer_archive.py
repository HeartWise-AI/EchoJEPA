# evals/view_classifier/infer_archive.py

"""Arm B over a list of clips (the whole archive): one row per clip with every class's probability and the top view.

The model is the fine-tuned classifier of one snapshot (its config), one pseudo-label setting, fold `final` (trained on
every label: the model to deploy), at `<runs>/arm_b/pseudo_<on|off>/fold_final/model.pt`. Each clip is read and
prepared exactly as that snapshot evaluated it: `mvit.ClipDecoder` with the config's `arm_b.data` (frame read,
preprocessing, size), then the arm's own `prepare` (channel order, normalisation, padding frames set to 0), bf16
autocast on a GPU, softmax in float32. The legacy YBR colour fix of the dataset loader applies (or not) as the config's
`legacy_ybr_fix` says; a run whose colour setting differs is refused (`io.check_colour_fix`).

Three stages:

- `plan`: the clip list (a manifest column, duplicates dropped, optional exclusions), sorted and cut into `--shards`
  contiguous shards, and a record of the model, config and inputs (`plan.json`, `clips.parquet`). Written once; a
  second plan into the same folder is refused.
- `run --stream j --streams N`: one process per stream, each with its own decoder workers; stream j does the shards
  s with s mod N = j that are not finished yet. A shard is written whole, to a temporary file then renamed
  (`shards/shard_<s>.parquet`), so a stopped stream loses at most the shards it was doing, and a rerun (with any N)
  skips the finished ones.
- `merge`: every shard finished, every planned clip exactly once; `predictions.parquet` (readable clips),
  `unreadable.csv` and `merge.json` (counts per top view, unreadable count).

A shard row: `video_path`, `readable`, `top_view`, `top_prob`, `view`, `view_prob`, then `p_<class>` (float32) for
every class of the config, in its order (empty for an unreadable clip). `view` / `view_prob` equal `top_view` /
`top_prob` unless the config's optional `inference.group_rule` is enabled (default off): with members M (e.g. the
PSAX levels), `output` O and thresholds `together_min` / `single_max`, a clip whose summed probability over M is at
least `together_min` while no single member reaches `single_max` gets `view` O with `view_prob` the summed probability.
The thresholds are parameters of the config, not tuned on the cross-validation folds.

Every file goes under `--out`, which must be outside the repository (the rows name clip paths). Nothing printed names
a clip.

    python -m evals.view_classifier.infer_archive plan --config CONFIG --pseudo off --manifest videos.parquet --out DIR
    CUDA_VISIBLE_DEVICES=2 python -m evals.view_classifier.infer_archive run --out DIR --stream 0 --streams 2 \\
        --device cuda:0 --workers 8
    python -m evals.view_classifier.infer_archive merge --out DIR
"""

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import mvit
from evals.view_classifier.io import check_colour_fix, run_dir
from evals.view_classifier.labels import file_sha256

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXCLUDE_COLUMNS = ("video_path", "avi_path", "clip_id")


def _commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def outside_repo(path):
    real, repo = os.path.realpath(path), os.path.realpath(REPO)
    if os.path.commonpath([real, repo]) == repo:
        raise ValueError("--out must be outside the repository: its files name clip paths.")


def group_rule(cfg, classes):
    """The config's `inference.group_rule`, checked, or None when absent or not enabled."""
    rule = (cfg.get("inference") or {}).get("group_rule") or {}
    if not rule.get("enabled", False):
        return None
    members = [str(m) for m in rule["members"]]
    missing = [m for m in members if m not in classes]
    if missing or len(members) < 2:
        raise ValueError(f"group_rule members must be at least two of the classes; not classes: {missing}.")
    output = str(rule["output"])
    if output in classes:
        raise ValueError(f"group_rule output {output!r} is already a class.")
    together, single = float(rule["together_min"]), float(rule["single_max"])
    if not (0 < together <= 1 and 0 < single <= 1):
        raise ValueError("group_rule thresholds must lie in (0, 1].")
    return {"output": output, "members": members, "together_min": together, "single_max": single}


def apply_group_rule(probs, classes, rule):
    """(view names, view probabilities) of float probabilities [N, C]: the argmax, or the rule's output where it
    applies (see the module docstring)."""
    top = probs.argmax(axis=1)
    view = np.asarray(classes, dtype=object)[top]
    view_prob = probs[np.arange(len(probs)), top]
    if rule is None or not len(probs):
        return view, view_prob
    idx = [classes.index(m) for m in rule["members"]]
    together = probs[:, idx].sum(axis=1)
    single = probs[:, idx].max(axis=1)
    hit = (together >= rule["together_min"]) & (single < rule["single_max"])
    view = np.where(hit, rule["output"], view)
    view_prob = np.where(hit, together, view_prob)
    return view, view_prob


def read_manifest(path, column="video_path", exclude=(), limit=None):
    """(sorted unique clip paths, number excluded, rows in the manifest)."""
    if path.endswith(".parquet"):
        table = pd.read_parquet(path, columns=[column])
    else:
        table = pd.read_csv(path, usecols=[column], dtype=str, keep_default_na=False)
    clips = table[column].astype(str)
    clips = clips[clips.str.strip() != ""]
    n_rows = len(table)
    excluded = set()
    for name in exclude:
        t = pd.read_csv(name, dtype=str, keep_default_na=False)
        col = next((c for c in EXCLUDE_COLUMNS if c in t.columns), None)
        if col is None:
            raise ValueError(f"An exclusion file needs one of the columns {EXCLUDE_COLUMNS}.")
        excluded |= set(t[col])
    unique = pd.Series(sorted(set(clips)))
    keep = ~unique.isin(excluded)
    out = unique[keep].reset_index(drop=True)
    if limit:
        out = out.iloc[:int(limit)]
    return out, int((~keep).sum()), n_rows


def model_path(cfg, pseudo):
    return os.path.join(run_dir(cfg, mvit.ARM, pseudo), "fold_final", "model.pt")


def plan(args):
    outside_repo(args.out)
    if os.path.exists(os.path.join(args.out, "plan.json")):
        raise FileExistsError(f"{args.out} already has a plan: resume it with `run`, or name another --out.")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    classes, _ = vc.from_config(cfg)
    rule = group_rule(cfg, classes)
    weights = model_path(cfg, args.pseudo)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"No final model at {weights}: train fold `final` of this run first.")
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    if list(saved["classes"]) != list(classes):
        raise ValueError("The final model's classes differ from the config's.")
    clips, n_excluded, n_rows = read_manifest(args.manifest, args.column, args.exclude, args.limit)
    if not len(clips):
        raise ValueError("No clip to score.")
    shards = max(1, min(int(args.shards), len(clips)))
    edges = np.linspace(0, len(clips), shards + 1).round().astype(int)
    shard = np.repeat(np.arange(shards), np.diff(edges))
    b = cfg[mvit.ARM]
    preprocessing, frame_read = mvit.data_settings(b.get("data"))
    record = {
        "config": os.path.abspath(args.config), "config_sha256": file_sha256(args.config),
        "snapshot": os.path.basename(os.path.normpath(cfg["paths"]["runs"])), "arm": mvit.ARM, "pseudo": args.pseudo,
        "model": os.path.abspath(weights), "model_sha256": file_sha256(weights), "hp": saved.get("hp"),
        "classes": list(classes), "channel_order": b.get("channel_order", "rgb"),
        "data": {"preprocessing": preprocessing, "frame_read": frame_read, **(b.get("data") or {})},
        "legacy_ybr_fix": check_colour_fix(cfg), "legacy_ybr_fix_in_config": cfg.get("legacy_ybr_fix"),
        "group_rule": rule,
        "manifest": os.path.abspath(args.manifest), "manifest_sha256": file_sha256(args.manifest),
        "manifest_rows": n_rows, "column": args.column,
        "exclude": [{"path": os.path.abspath(e), "sha256": file_sha256(e)} for e in args.exclude],
        "excluded": n_excluded, "limit": args.limit, "clips": len(clips), "shards": shards,
        "commit": _commit(), "planned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.join(args.out, "shards"), exist_ok=True)
    pd.DataFrame({"video_path": clips, "shard": shard}).to_parquet(os.path.join(args.out, "clips.parquet"),
                                                                   index=False)
    with open(os.path.join(args.out, "plan.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: record[k] for k in ("snapshot", "pseudo", "clips", "excluded", "shards",
                                             "legacy_ybr_fix")}))


def read_plan(out):
    with open(os.path.join(out, "plan.json")) as f:
        record = json.load(f)
    clips = pd.read_parquet(os.path.join(out, "clips.parquet"))
    if len(clips) != record["clips"]:
        raise ValueError("clips.parquet does not match plan.json.")
    return record, clips


def shard_path(out, s):
    return os.path.join(out, "shards", f"shard_{s:05d}.parquet")


def load_model(record, device):
    """(model on `device` in eval mode, prepare: uint8 [B, 3, T, S, S] -> model input) of the planned run."""
    with open(record["config"]) as f:
        cfg = yaml.safe_load(f)
    if file_sha256(record["config"]) != record["config_sha256"]:
        raise ValueError("The config changed since the plan was written.")
    if file_sha256(record["model"]) != record["model_sha256"]:
        raise ValueError("The final model changed since the plan was written.")
    if check_colour_fix(cfg) != record["legacy_ybr_fix"]:
        raise SystemExit("The colour fix (ECHOJEPA_LEGACY_YBR_FIX) differs from the plan's.")
    classes, parent = vc.from_config(cfg)
    b = cfg[mvit.ARM]
    arm = mvit.make_arm(None, None, b, cfg["paths"]["view_classifier"], classes, parent)
    saved = torch.load(record["model"], map_location="cpu", weights_only=True)
    model = arm.build(saved["hp"])
    model.load_state_dict(saved["state_dict"])
    model = model.to(device).eval()
    return model, lambda x: arm.prepare(x, False, None)


def shard_table(paths, probs, readable, classes, rule):
    """The rows of one shard: `probs` [N, C] float (rows of unreadable clips ignored)."""
    n = len(paths)
    t = pd.DataFrame({"video_path": paths, "readable": readable})
    view, view_prob = apply_group_rule(probs, list(classes), rule)
    top = probs.argmax(axis=1) if n else np.zeros(0, int)
    t["top_view"] = np.where(readable, np.asarray(classes, dtype=object)[top] if n else [], None)
    t["top_prob"] = np.where(readable, probs[np.arange(n), top] if n else [], np.nan).astype(np.float32)
    t["view"] = np.where(readable, view, None)
    t["view_prob"] = np.where(readable, view_prob, np.nan).astype(np.float32)
    p = probs.astype(np.float32).copy()
    p[~readable] = np.nan
    return pd.concat([t, pd.DataFrame(p, columns=[f"p_{c}" for c in classes])], axis=1)


def write_shard(out, s, table):
    final = shard_path(out, s)
    tmp = final + f".tmp{os.getpid()}"
    table.to_parquet(tmp, index=False)
    os.replace(tmp, final)


@torch.no_grad()
def run(args, model_loader=load_model):
    record, clips = read_plan(args.out)
    if not 0 <= args.stream < args.streams:
        raise ValueError("--stream must lie in [0, --streams).")
    mine = [s for s in range(record["shards"]) if s % args.streams == args.stream]
    todo = [s for s in mine if not os.path.exists(shard_path(args.out, s))]
    for s in todo:                     # half-written files of an earlier, stopped stream
        for stale in glob.glob(shard_path(args.out, s) + ".tmp*"):
            os.remove(stale)
    print(f"stream {args.stream}/{args.streams}: {len(mine)} shards, {len(mine) - len(todo)} already finished",
          flush=True)
    if not todo:
        return
    device = torch.device(args.device)
    model, prepare = model_loader(record, device)
    classes, rule, data = record["classes"], record["group_rule"], record["data"]
    by_shard = {s: clips.video_path[clips.shard == s].tolist() for s in todo}
    ids = [c for s in todo for c in by_shard[s]]
    dataset = mvit.ClipDecoder(ids, int(data.get("frames_per_clip", 16)), int(data.get("frame_step", 2)),
                               int(data.get("resolution", 224)), float(data.get("zoom", 0.1)),
                               data["preprocessing"], data["frame_read"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=args.workers,
                                         collate_fn=mvit._collate,
                                         multiprocessing_context="spawn" if args.workers else None)
    probs = np.zeros((len(ids), len(classes)), dtype=np.float32)
    readable = np.zeros(len(ids), dtype=bool)
    ends = np.cumsum([len(by_shard[s]) for s in todo])
    next_shard, start_of = 0, 0
    pending, done_upto, started = [], 0, time.time()

    def score(items):
        x = torch.stack([clip for _, clip in items]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(prepare(x))
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for (i, _), row in zip(items, p):
            probs[i], readable[i] = row, True

    for items in loader:
        for i, clip in items:
            if clip is not None:
                pending.append((i, clip))
            done_upto = max(done_upto, i + 1)
        if len(pending) >= args.batch:
            score(pending)
            pending = []
        while next_shard < len(todo) and done_upto >= ends[next_shard]:
            if pending:
                score(pending)
                pending = []
            end = ends[next_shard]
            write_shard(args.out, todo[next_shard],
                        shard_table(ids[start_of:end], probs[start_of:end], readable[start_of:end], classes, rule))
            rate = end / max(time.time() - started, 1e-9)
            print(f"stream {args.stream}: shard {todo[next_shard]} written ({next_shard + 1}/{len(todo)}), "
                  f"{end} clips, {rate:.1f} clips/s", flush=True)
            start_of, next_shard = end, next_shard + 1
    if next_shard != len(todo):
        raise RuntimeError("The decoder stopped before every shard was written.")


def merge(args):
    record, clips = read_plan(args.out)
    missing = [s for s in range(record["shards"]) if not os.path.exists(shard_path(args.out, s))]
    if missing:
        raise ValueError(f"{len(missing)} of {record['shards']} shards are not finished: run every stream first.")
    table = pd.concat([pd.read_parquet(shard_path(args.out, s)) for s in range(record["shards"])],
                      ignore_index=True)
    if table.video_path.duplicated().any():
        raise ValueError("A clip has more than one row.")
    if len(table) != len(clips) or set(table.video_path) != set(clips.video_path):
        raise ValueError("The shards do not hold exactly the planned clips.")
    expected = ["video_path", "readable", "top_view", "top_prob", "view", "view_prob"] + \
        [f"p_{c}" for c in record["classes"]]
    if list(table.columns) != expected:
        raise ValueError("A shard has other columns than the plan's classes.")
    bad = table[~table.readable]
    good = table[table.readable].reset_index(drop=True)
    if good[[f"p_{c}" for c in record["classes"]]].isna().any(axis=None):
        raise ValueError("A readable clip has missing probabilities.")
    good.to_parquet(os.path.join(args.out, "predictions.parquet"), index=False)
    bad[["video_path"]].to_csv(os.path.join(args.out, "unreadable.csv"), index=False)
    summary = {"clips": len(table), "readable": len(good), "unreadable": len(bad),
               "top_view": good.top_view.value_counts().to_dict(), "view": good.view.value_counts().to_dict(),
               "median_top_prob": float(good.top_prob.median()) if len(good) else None,
               "merged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(args.out, "merge.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("clips", "readable", "unreadable")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("plan", "run", "merge"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--pseudo", choices=("on", "off"), default="off")
    parser.add_argument("--manifest")
    parser.add_argument("--column", default="video_path")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips (a trial run)")
    parser.add_argument("--shards", type=int, default=1000)
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)
    if args.stage == "plan":
        if not (args.config and args.manifest):
            parser.error("plan needs --config and --manifest")
        return plan(args)
    return run(args) if args.stage == "run" else merge(args)


if __name__ == "__main__":
    main()
