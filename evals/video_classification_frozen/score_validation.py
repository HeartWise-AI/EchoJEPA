# evals/video_classification_frozen/score_validation.py

"""Score the validation split with a finished Visual EF probe's selected epoch and head, and nothing else.

The probe's test step (`--test_only`) writes per-video validation predictions only while it scores the
test split. A comparison that must stay on validation (the temporal-padding ablation compares two
encoders by study-level validation MAE with a paired bootstrap) needs those predictions without opening
the test split. This entry point rebuilds the probe from its config as `eval.main` does, checks it against
the fingerprint in `best.pt`, selects the head as `run_test` does, and scores each validation video once
with `score_videos`.

It never builds a test dataset and never reads `dataset_test`: the test manifest's checksum is the one
fingerprint entry it does not check (with the number of GPUs, as the test), because checking it would
mean reading that file.

    python -m evals.video_classification_frozen.score_validation --fname <probe config> --devices cuda:0

Writes `val_selected_predictions.csv` and `val_selected_metrics.json` to the probe's folder, or `--out`;
never inside the repository, since the predictions list video paths and study IDs.
"""

import argparse
import copy
import functools
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from evals.video_classification_frozen import eval as probe
from evals.video_classification_frozen import metrics as probe_metrics
from evals.video_classification_frozen import provenance
from src.models.attentive_pooler import AttentiveRegressor
from src.models.linear_pooler import LinearRegressor, MLPRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import close_distributed, init_distributed

logger = logging.getLogger()

EVAL_NAME = "video_classification_frozen"
PREDICTIONS = "val_selected_predictions.csv"
METRICS = "val_selected_metrics.json"
# Fingerprint entries that may differ from `best.pt`: the number of GPUs (as at the test) and the test
# manifest, which is never read here.
NOT_CHECKED = ("settings.world_size", "manifests_sha256.test")
# EF points. Each video's prediction is recomputed as validation computed it during training; on GPU,
# bfloat16 and another batch composition (another number of GPUs) move it slightly.
DEFAULT_TOLERANCE = 0.01


def refuse_overrides(environ=None):
    """`eval.main` lets `OVERRIDE_*` environment variables change the config; the score must come from
    the config file as written, so that it describes the probe it names."""
    found = sorted(k for k in (os.environ if environ is None else environ) if k.startswith("OVERRIDE_"))
    if found:
        raise ValueError(f"Unset {found}: validation scoring reads the probe's config file as written.")


def read_probe(args_eval):
    """The probe as `eval.main` reads it from its config, with the same defaults, and the settings its
    fingerprint holds. `settings(world_size)` must stay equal to `main()`'s `settings`: a difference is
    refused at the fingerprint check, and the tests compare it with probes `eval.main` trained."""
    eval_name = args_eval.get("eval_name", EVAL_NAME)
    if eval_name != EVAL_NAME:
        raise ValueError(f"This scores `{EVAL_NAME}` probes, not `{eval_name}`.")
    args_pretrain = args_eval.get("model_kwargs")
    args_exp = args_eval.get("experiment")
    args_classifier = args_exp.get("classifier")
    args_data = args_exp.get("data")
    args_opt = args_exp.get("optimization")
    args_evaluation = args_exp.get("evaluation") or {}

    task_type = args_classifier.get("task_type", "classification")
    video_index = args_data.get("video_index", None)
    targets = dict(args_evaluation.get("targets") or {})
    selection = args_evaluation.get("selection") or (
        "study" if video_index and task_type == "regression" else "video")
    tag = args_eval.get("tag", None)
    folder = os.path.join(args_eval.get("folder"), "video_classification_frozen/")
    p = SimpleNamespace(
        folder=os.path.join(folder, tag) if tag is not None else folder,
        num_workers=args_eval.get("num_workers", 12),
        checkpoint=args_pretrain.get("checkpoint"),
        module_name=args_pretrain.get("module_name"),
        args_model=args_pretrain.get("pretrain_kwargs"),
        args_wrapper=args_pretrain.get("wrapper_kwargs"),
        task_type=task_type,
        num_targets=args_classifier.get("num_targets", None),
        probe_type=args_classifier.get("probe_type", "attentive"),
        num_heads=args_classifier.get("num_heads", 16),
        num_probe_blocks=args_classifier.get("num_probe_blocks", 1),
        use_layernorm=args_classifier.get("use_layernorm", True),
        dropout=args_classifier.get("dropout", 0.0),
        dataset_type=args_data.get("dataset_type", "VideoDataset"),
        manifests={"train": args_data.get("dataset_train"), "val": args_data.get("dataset_val"),
                   "test": None, "video_index": video_index},  # the test manifest is never read.
        resolution=args_data.get("resolution", 224),
        num_segments=args_data.get("num_segments", 1),
        frames_per_clip=args_data.get("frames_per_clip", 16),
        frame_step=args_data.get("frame_step", 4),
        duration=args_data.get("clip_duration", None),
        num_views_per_segment=args_data.get("num_views_per_segment", 1),
        normalization=args_data.get("normalization", None),
        target_mean=args_data.get("target_mean", None),
        target_std=args_data.get("target_std", None),
        low_ef_below=float(args_evaluation.get("low_ef_below", 40.0)),
        ef_ranges=list(args_evaluation.get("ef_ranges", [30, 40, 50, 60])),
        targets=targets,
        selection=selection,
        seed=int((args_eval.get("meta") or {}).get("seed", probe._GLOBAL_SEED)),
        batch_size=args_opt.get("batch_size"),
        num_epochs=args_opt.get("num_epochs"),
        use_bfloat16=args_opt.get("use_bfloat16"),
        multihead_kwargs=args_opt.get("multihead_kwargs"),
    )
    # Copied before the encoder is built, as `main()` does: building it may change them.
    encoder_settings = copy.deepcopy(
        {"module_name": p.module_name, "pretrain_kwargs": p.args_model, "wrapper_kwargs": p.args_wrapper})
    use_focal_loss = args_opt.get("use_focal_loss", False)

    def settings(world_size):
        return {
            "encoder": encoder_settings,
            "classifier": dict(args_classifier),
            "clips": {"dataset_type": p.dataset_type, "resolution": p.resolution,
                      "frames_per_clip": p.frames_per_clip, "frame_step": p.frame_step,
                      "clip_duration": p.duration, "num_segments": p.num_segments,
                      "num_views_per_segment": p.num_views_per_segment, "normalization": p.normalization},
            "optimization": {"batch_size": p.batch_size, "num_epochs": p.num_epochs,
                             "use_bfloat16": p.use_bfloat16, "use_focal_loss": use_focal_loss,
                             "multihead_kwargs": p.multihead_kwargs},
            "evaluation": {"low_ef_below": p.low_ef_below, "ef_ranges": list(p.ef_ranges),
                           "targets": copy.deepcopy(targets)},
            "selection": selection,
            "protocol": copy.deepcopy(probe.PROTOCOL),
            "seed": p.seed,
            "world_size": world_size,
        }

    p.settings = settings
    # As `main()` names the heads (its `opt_kwargs` hold `lr` as `ref_lr`, `weight_decay` as `ref_wd`).
    p.head_names = [f"head{i}_lr{k.get('lr'):g}_wd{k.get('weight_decay'):g}"
                    for i, k in enumerate(p.multihead_kwargs)]
    return p


def select_head(checkpoint):
    """The head `run_test` scores: the lowest validation score `best.pt` records (regression)."""
    return int(np.asarray(checkpoint["val_acc_per_head"], dtype=float).argmin())


def make_head(p, embed_dim):
    """One regression head as `main()`'s `make_probe` builds it, to load the selected head's weights into."""
    if p.probe_type == "linear":
        return LinearRegressor(embed_dim=embed_dim, num_targets=p.num_targets, use_layernorm=p.use_layernorm,
                               dropout=p.dropout)
    if p.probe_type == "mlp":
        return MLPRegressor(embed_dim=embed_dim, num_targets=p.num_targets, use_layernorm=p.use_layernorm,
                            dropout=p.dropout)
    return AttentiveRegressor(embed_dim=embed_dim, num_heads=p.num_heads, depth=p.num_probe_blocks,
                              num_targets=p.num_targets, use_activation_checkpointing=True)


def ensure_outside_repo(path, flag):
    """`data/build_manifests.ensure_outside_repo`. data/ holds standalone scripts, not a package, so the
    module is loaded from its file."""
    module = sys.modules.get("build_manifests")
    if module is None or not hasattr(module, "ensure_outside_repo"):
        spec = importlib.util.spec_from_file_location(
            "build_manifests", os.path.join(provenance.REPO_ROOT, "data", "build_manifests.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules.setdefault("build_manifests", module)
    module.ensure_outside_repo(path, flag)


def check_stored_scores(checkpoint, head, recomputed, tolerance):
    """Compare the recomputed validation MAE of `head` (per study, and per video) with what `best.pt`
    recorded for it at the end of its epoch. Training scored the same videos with the same function, so
    only floating-point noise may separate them; more than `tolerance` EF points means these are not
    the predictions the probe was selected on, and nothing is written."""
    stored = dict(checkpoint.get("val_scores_per_head") or {})
    # `val_acc_per_head` holds the selection metric, also for a checkpoint without the other scores.
    stored.setdefault(checkpoint.get("selection_metric", "mae"), checkpoint["val_acc_per_head"])
    if "study_mae" not in stored:
        raise ValueError("`best.pt` records no per-study validation MAE to check the recomputed one against.")
    check = {}
    for key, value in recomputed.items():
        if key in stored:
            saved = float(stored[key][head])
            check[key] = {"stored": saved, "recomputed": float(value), "difference": float(value) - saved}
    within = all(abs(c["difference"]) <= tolerance for c in check.values())
    if not within:
        found = ", ".join(f"{k} {c['recomputed']:.6g} (stored {c['stored']:.6g})" for k, c in check.items())
        raise RuntimeError(
            f"The recomputed validation scores of the selected head differ from `best.pt` by more than "
            f"{tolerance} EF points: {found}. Check that no validation video failed to load now that loaded "
            "during training, and score with the number of GPUs the probe was trained with; pass a larger "
            "--tolerance only once the difference is understood.")
    return {**check, "tolerance": float(tolerance), "within_tolerance": within}


def write_atomically(path, write):
    tmp = f"{path}.tmp"
    write(tmp)
    os.replace(tmp, path)


def score_validation(args_eval, out_dir=None, overwrite=False, tolerance=DEFAULT_TOLERANCE):
    """Score the validation split with the probe's selected epoch and head; write the per-video
    predictions and the metrics (rank 0). Returns the metrics on rank 0, `None` on other ranks."""
    refuse_overrides()
    p = read_probe(args_eval)
    if p.task_type != "regression":
        raise ValueError("Validation scoring is for regression (EF) probes.")
    probe.check_normalization(p.target_mean, p.target_std, p.manifests["train"])
    if not p.manifests["video_index"]:
        raise ValueError("Study-level validation scores need `experiment.data.video_index`.")
    if not tolerance >= 0:  # NaN too
        raise ValueError(f"The tolerance is a non-negative number of EF points, not {tolerance}.")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    world_size, rank = init_distributed()

    # Refuse before any expensive step: an unfinished probe, a `best.pt` that is not the selected epoch,
    # outputs inside the repository or already written.
    best_path = os.path.join(p.folder, "best.pt")
    if not os.path.exists(best_path):
        raise RuntimeError(f"{p.folder} holds no `best.pt`: train the probe first.")
    try:
        probe.check_finished(p.folder, p.num_epochs)
    except RuntimeError as e:
        raise RuntimeError(f"Validation scoring needs a finished probe. {e}") from e
    out_dir = p.folder if out_dir is None else out_dir
    ensure_outside_repo(out_dir, "--out")
    outputs = {name: os.path.join(out_dir, name) for name in (PREDICTIONS, METRICS)}
    existing = [name for name, path in outputs.items() if os.path.exists(path)]
    if existing and not overwrite:
        raise RuntimeError(f"{existing} already exist in {out_dir}: pass --overwrite to score again.")

    video_index = provenance.load_video_index(p.manifests["video_index"])
    fingerprint = probe.shared_fingerprint(
        rank, world_size, device, p.checkpoint, p.manifests,
        {"target_mean": p.target_mean, "target_std": p.target_std}, p.settings(world_size))
    checkpoint = robust_checkpoint_loader(best_path, map_location=torch.device("cpu"), weights_only=True)
    # Every rank checks the same two fingerprints, so all ranks stop together.
    provenance.check_fingerprint(checkpoint.get("fingerprint"), fingerprint, best_path, ignore=NOT_CHECKED)
    head = select_head(checkpoint)
    logger.info(f"Scoring validation with {p.head_names[head]} from epoch {checkpoint['epoch']} of {best_path}")

    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)
    encoder = probe.init_module(
        module_name=p.module_name,
        frames_per_clip=p.frames_per_clip,
        resolution=p.resolution,
        checkpoint=p.checkpoint,
        model_kwargs=p.args_model,
        wrapper_kwargs=p.args_wrapper,
        device=device,
    )
    classifier = make_head(p, encoder.embed_dim).to(device)
    classifier.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["classifiers"][head].items()})
    classifier.eval()

    val_loader, _ = probe.make_dataloader(
        dataset_type=p.dataset_type,
        root_path=[p.manifests["val"]],
        img_size=p.resolution,
        frames_per_clip=p.frames_per_clip,
        frame_step=p.frame_step,
        num_segments=p.num_segments,
        eval_duration=p.duration,
        num_views_per_segment=p.num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=p.batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=p.num_workers,
        normalization=p.normalization,
    )
    dataset = val_loader.dataset
    if not hasattr(dataset, "get_item_video"):
        raise ValueError("Validation scoring needs a `VideoDataset` validation set, scored video by video.")
    probe.check_split(video_index, dataset, "val")

    indices, labels, predictions, failed = probe.score_videos(
        device, encoder, [classifier], dataset, p.batch_size, p.num_workers, p.use_bfloat16, "regression", rank,
        world_size,
    )
    if rank != 0:
        return None

    paths = [dataset.samples[i] for i in indices]
    t_mean, t_std = float(p.target_mean), float(p.target_std)
    y, pred = labels[:, 0] * t_std + t_mean, predictions[0][:, 0] * t_std + t_mean
    study_ids = provenance.study_ids(video_index, paths, "val")
    studies = probe_metrics.by_study(study_ids, y, pred)
    study_mae = float((studies.prediction - studies.label).abs().mean())
    video_mae = float(np.abs(pred - y).mean())
    check = check_stored_scores(checkpoint, head, {"study_mae": study_mae, "mae": video_mae}, tolerance)

    # The reduced-EF threshold as the test step chooses it on validation, so these metrics are the
    # validation block `test_metrics.json` would hold.
    level = p.targets.get("level") or "study"
    chosen_on = studies if level == "study" else pd.DataFrame({"label": y, "prediction": pred})
    threshold = probe_metrics.youden_threshold(chosen_on.label, chosen_on.prediction, p.low_ef_below)
    selected = checkpoint.get("selection_metric", "mae")
    commit, dirty = provenance.git_state()
    result = {
        "split": "val",
        "selected_on": "validation",
        "selection": checkpoint.get("selection", probe.SELECTION["video"]),
        "checkpoint": best_path,
        "checkpoint_sha256": provenance.sha256(best_path),
        "encoder_sha256": fingerprint["encoder_sha256"],
        "epoch": int(checkpoint["epoch"]),
        "head": head,
        "head_name": p.head_names[head],
        f"val_{selected}": float(np.asarray(checkpoint["val_acc_per_head"], dtype=float)[head]),
        "stored_score_check": check,
        "aggregation": probe.AGGREGATION,
        "low_ef_threshold": {"rule": probe.THRESHOLD_RULE, "below": p.low_ef_below, "level": level,
                             "value": threshold},
        "val": {
            "study": probe_metrics.summarize(studies.label, studies.prediction, p.low_ef_below, threshold),
            "video": probe_metrics.summarize(y, pred, p.low_ef_below, threshold),
            "studies": len(studies),
            "videos": len(indices),
            "failed_videos": len(failed),
        },
        "failed_rows": [int(i) for i in failed],  # rows of the validation manifest, not paths.
        "fingerprint_not_checked": list(NOT_CHECKED),
        "world_size": {"trained": checkpoint.get("world_size"), "scored": world_size},
        "git_commit": commit,
        "git_dirty": dirty,
    }

    os.makedirs(out_dir, exist_ok=True)
    table = pd.DataFrame({"video_path": paths, "study_id": study_ids, "label": y, "prediction": pred})
    write_atomically(outputs[PREDICTIONS], lambda path: table.to_csv(path, index=False))

    def dump(path):
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    write_atomically(outputs[METRICS], dump)  # last: its presence means the scoring finished.
    logger.info(f"Validation of {p.head_names[head]} (epoch {result['epoch']}): study MAE {study_mae:.4f} over "
                f"{len(studies)} studies, {len(failed)} videos failed; written to {out_dir}")
    return result


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--fname", type=str, required=True, help="the probe's config file")
parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"], help="one process per device")
parser.add_argument("--out", type=str, default=None,
                    help="folder for the outputs, outside the repository (default: the probe's folder)")
parser.add_argument("--overwrite", action="store_true", help="replace outputs of an earlier scoring")
parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="largest difference, in EF points, between the recomputed and the stored validation MAE")


def process_main(args, rank, fname, world_size, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    logging.basicConfig()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    with open(fname) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    init_distributed(rank_and_world_size=(rank, world_size))
    score_validation(params, out_dir=args.out, overwrite=args.overwrite, tolerance=args.tolerance)
    close_distributed()


def launch_ranks(args):
    """One process per device, as `evals.main`: once a rank fails the others are stopped, and the command
    exits non-zero."""
    from app.main import launch

    failed = launch(args.fname, args.devices, target=functools.partial(process_main, args))
    if failed:
        sys.exit(f"Validation scoring failed on rank(s) {failed}; the other ranks were stopped.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    launch_ranks(parser.parse_args())
