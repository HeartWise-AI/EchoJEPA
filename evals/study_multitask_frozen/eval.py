# evals/study_multitask_frozen/eval.py

"""Frozen study-level multi-task probe, DeepCORO_CLIP style.

A study is read as a bag of its videos (`experiment.data.num_videos` slots, a random subset in
random order at every training read, reproducible seeded bags in evaluation, optionally
balanced across view families); the frozen encoder embeds every clip of every video; one probe pools the study's
tokens and reads them with one head per target (`experiment.classifier.target_label`,
`head_task`, `head_structure`, `loss_structure`, `head_weights`, `head_dropout`). Several
probes train side by side on the same features, one per `optimization.multihead_kwargs`
setting. Validation scores every study once with every probe after each epoch; `best.pt` keeps
the epoch and probe with the best selection score. `--test_only` scores the test split once with
that probe. `--val_only` rescores only validation. Protocol and outputs: `docs/study_multitask_probe.md`.
"""

import copy
import json
import logging
import os
import pprint
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from evals.video_classification_frozen import metrics as probe_metrics
from evals.video_classification_frozen import provenance
from evals.video_classification_frozen.eval import (
    DEFAULT_NORMALIZATION,
    CosineWDSchedule,
    WarmupCosineLRSchedule,
    gather_rng_states,
    save_atomically,
    set_rng_state,
)
from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_study_dataset import SELECTION, VideoStudyDataset, read_split
from src.models.study_probe import POOLING_MODES, StudyProbe, head_loss, head_specs, multitask_loss
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduceSum, any_rank_failed, init_distributed
from src.utils.logging import AverageMeter, CSVLogger
from src.utils.wandb_logging import PRIVATE_WANDB_SETTINGS, finish_wandb, init_wandb, log_scalars, public_config

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
EVAL_NAME = "study_multitask_frozen"

# How the studies of each split are read (`SELECTION`) and batched. The fingerprint holds it, so a
# probe is continued and tested only under the protocol it was trained with. Raise the version
# whenever the code changes how the probe is trained or evaluated in a way these fields do not describe.
PROTOCOL = {
    "version": 2,
    "train": {"videos": "random when `shuffle_videos`, else fixed", "clip_positions": "random", "drop_last": True},
    "evaluation": {"videos": "seeded bags using sampling_mode", "clip_positions": "fixed", "drop_last": False},
}
# Metrics a selection rule `<head>/<metric>` may name, per head task (as `head_metrics` reports
# them), and the direction that is better.
TASK_METRICS = {
    "regression": {"loss", "mae", "rmse", "r2", "pearson_r"},
    "binary": {"loss", "auroc", "auprc", "acc"},
    "classification": {"loss", "acc", "auroc"},
}
LOWER_IS_BETTER = {"loss", "mae", "rmse"}


def main(args_eval, resume_preempt=False):
    test_only = bool(args_eval.get("test_only", False))
    val_only = bool(args_eval.get("val_only", False))
    if test_only and val_only:
        raise ValueError("Choose either val_only or test_only.")
    scoring_only = test_only or val_only
    pretrain_folder = args_eval.get("folder")
    tag = args_eval.get("tag")
    num_workers = int(args_eval.get("num_workers", 8))
    resume_checkpoint = bool(args_eval.get("resume_checkpoint", False) or resume_preempt)
    seed = int((args_eval.get("meta") or {}).get("seed", _GLOBAL_SEED))

    # -- ENCODER
    args_pretrain = args_eval["model_kwargs"]
    checkpoint = args_pretrain["checkpoint"]
    encoder_settings = copy.deepcopy({"module_name": args_pretrain.get("module_name"),
                                      "pretrain_kwargs": args_pretrain.get("pretrain_kwargs"),
                                      "wrapper_kwargs": args_pretrain.get("wrapper_kwargs")})

    args_exp = args_eval["experiment"]
    # -- DATA (DeepCORO_CLIP's dataset keys)
    args_data = args_exp["data"]
    split_column = args_data.get("split_column", "split") if args_data.get("data_filename") else None
    path_column = args_data.get("datapoint_loc_label", "video_path")
    group_column = args_data.get("groupby_column", "study_id")
    view_column = args_data.get("view_column")
    views = args_data.get("views")
    num_videos = int(args_data.get("num_videos", 8))
    shuffle_videos = bool(args_data.get("shuffle_videos", True))
    num_clips_per_video = int(args_data.get("num_clips_per_video", 1))
    frames_per_clip = int(args_data.get("frames_per_clip", 16))
    frame_step = args_data.get("frame_step", 2)
    duration = args_data.get("clip_duration")
    fps = args_data.get("fps")
    if duration is not None or fps is not None:
        frame_step = None
    resolution = int(args_data.get("resolution", 224))
    normalization = args_data.get("normalization")
    min_videos = int(args_data.get("min_videos", 1))
    view_labels_map = args_data.get("view_labels_map")

    # -- PROBE (DeepCORO_CLIP's head keys)
    args_classifier = args_exp["classifier"]
    specs = head_specs(args_classifier)
    pooling_mode = args_classifier.get("pooling_mode", "attentive")
    if pooling_mode not in POOLING_MODES:
        raise ValueError(f"`experiment.classifier.pooling_mode` is one of {POOLING_MODES}, not {pooling_mode!r}.")
    num_heads = int(args_classifier.get("num_heads", 16))
    num_probe_blocks = int(args_classifier.get("num_probe_blocks", 4))
    attention_hidden = int(args_classifier.get("attention_hidden", 256))
    probe_dropout = float(args_classifier.get("dropout", 0.0))
    labels_map = args_classifier.get("labels_map")
    use_view_embedding = bool(args_classifier.get("use_view_embedding", False))
    if use_view_embedding and not (view_labels_map and view_column):
        raise ValueError("`use_view_embedding` needs `experiment.data.view_column` and `view_labels_map`.")
    num_view_classes = len(set(int(v) for v in view_labels_map.values())) if use_view_embedding else 0
    if args_data.get("view_probability_columns") and not use_view_embedding:
        raise ValueError("view_probability_columns requires use_view_embedding.")
    probe_options = study_probe_options(args_classifier, args_data, args_pretrain)
    data_options = {k: args_data[k] for k in (
        "sampling_mode", "eval_seed", "view_family_map", "view_probability_columns",
        "acquisition_column", "acquisition_labels_map", "patient_column") if k in args_data}
    data_options.setdefault("eval_seed", 0)  # identical evaluation bags across training seeds

    # -- ENCODER FINE-TUNING: the top `trainable_blocks` transformer blocks (and the final norm)
    # train with their own learning rate; 0 keeps the encoder frozen.
    args_encoder = args_exp.get("encoder") or {}
    trainable_blocks = int(args_encoder.get("trainable_blocks", 0))
    if trainable_blocks < 0:
        raise ValueError("`experiment.encoder.trainable_blocks` must not be negative.")
    encoder_opt_kwargs = None
    if trainable_blocks:
        encoder_opt_kwargs = dict(
            ref_wd=float(args_encoder.get("weight_decay", 0.0)),
            final_wd=float(args_encoder.get("final_weight_decay", args_encoder.get("weight_decay", 0.0))),
            start_lr=float(args_encoder.get("start_lr", args_encoder["lr"])), ref_lr=float(args_encoder["lr"]),
            final_lr=float(args_encoder.get("final_lr", 0.0)), warmup=float(args_encoder.get("warmup", 0.0)))

    # -- EVALUATION
    args_evaluation = args_exp.get("evaluation") or {}
    selection = args_evaluation.get("selection", "weighted_loss")
    selection_rule = parse_selection(selection, specs)
    num_eval_bags = int(args_evaluation.get("num_bags", 1))
    if num_eval_bags < 1:
        raise ValueError("evaluation.num_bags must be positive.")

    # -- OPTIMIZATION
    args_opt = args_exp["optimization"]
    batch_size = int(args_opt["batch_size"])
    num_epochs = int(args_opt["num_epochs"])
    use_bfloat16 = bool(args_opt.get("use_bfloat16", True))
    opt_kwargs = [
        dict(ref_wd=k.get("weight_decay"), final_wd=k.get("final_weight_decay"), start_lr=k.get("start_lr"),
             ref_lr=k.get("lr"), final_lr=k.get("final_lr"), warmup=k.get("warmup", 0.0))
        for k in args_opt["multihead_kwargs"]
    ]
    probe_names = [f"probe{i}_lr{k['ref_lr']:g}_wd{k['ref_wd']:g}" for i, k in enumerate(opt_kwargs)]
    if trainable_blocks and len(opt_kwargs) != 1:
        raise ValueError("With `experiment.encoder.trainable_blocks` > 0 the probes would share the encoder's "
                         "gradients: give `multihead_kwargs` one setting.")

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    use_ddp = dist.is_available() and dist.is_initialized() and world_size > 1

    folder = os.path.join(pretrain_folder, EVAL_NAME)
    if tag:
        folder = os.path.join(folder, tag)
    os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -- DATA
    transform_kwargs = dict(random_horizontal_flip=False, random_resize_aspect_ratio=(0.75, 4 / 3),
                            random_resize_scale=(0.08, 1.0), reprob=0.25, auto_augment=True, motion_shift=False,
                            crop_size=resolution, num_views_per_clip=1,
                            normalize=normalization or DEFAULT_NORMALIZATION)

    def make_dataset(split, training):
        return VideoStudyDataset(
            read_split(args_data, split), path_column=path_column, group_column=group_column,
            target_columns=list(specs), view_column=view_column, views=views, labels_map=labels_map,
            num_videos=num_videos, shuffle_videos=shuffle_videos, training=training,
            frames_per_clip=frames_per_clip, frame_step=frame_step, duration=duration, fps=fps,
            num_clips_per_video=num_clips_per_video, transform=make_transforms(training=training, **transform_kwargs),
            min_videos=min_videos, view_labels_map=view_labels_map, **data_options,
        )

    train_dataset = make_dataset("train", training=True)
    val_dataset = make_dataset("val", training=False)
    check_disjoint({"train": train_dataset, "val": val_dataset})
    normalization_targets = target_normalization(specs, train_dataset.targets, args_classifier)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=PROTOCOL["train"]["drop_last"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=PROTOCOL["train"]["drop_last"],
        num_workers=num_workers, pin_memory=True, persistent_workers=False)
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")
    if ipe == 0 and not scoring_only:
        raise ValueError(f"The training split gives no full batch of {batch_size} studies on each of {world_size} "
                         "GPUs: nothing to train on.")

    # -- MODELS
    encoder = init_module(module_name=args_pretrain["module_name"], frames_per_clip=frames_per_clip,
                          resolution=resolution, checkpoint=checkpoint, model_kwargs=args_pretrain["pretrain_kwargs"],
                          wrapper_kwargs=args_pretrain.get("wrapper_kwargs"), device=device)
    probes = [
        StudyProbe(embed_dim=encoder.embed_dim, heads=specs, pooling_mode=pooling_mode, num_heads=num_heads,
                   depth=num_probe_blocks, attention_hidden=attention_hidden, dropout=probe_dropout,
                   num_view_classes=num_view_classes, **probe_options).to(device)
        for _ in opt_kwargs
    ]
    logger.info(f"Initialized {len(probes)} study probes ({pooling_mode}) with heads {list(specs)}"
                f"{f' and {num_view_classes} view classes' if num_view_classes else ''}")
    if use_ddp:
        probes = [DistributedDataParallel(p, static_graph=True) for p in probes]
    optimizers, schedulers, wd_schedulers = init_opt(probes, opt_kwargs, ipe, num_epochs)

    trainable = unfreeze_top_blocks(encoder, trainable_blocks)
    encoder_module = encoder
    if trainable:
        if use_ddp:
            encoder = DistributedDataParallel(encoder, static_graph=True)
        enc_opt, enc_sched, enc_wd = init_opt([encoder], [encoder_opt_kwargs], ipe, num_epochs)
        optimizers += enc_opt
        schedulers += enc_sched
        wd_schedulers += enc_wd
        logger.info(f"Training the top {trainable_blocks} encoder blocks ({sum(p.numel() for p in trainable)} "
                    f"parameters) at lr {encoder_opt_kwargs['ref_lr']:g}")
    else:
        if any(p.requires_grad for p in encoder.parameters()):
            raise RuntimeError("The encoder must be frozen: some of its parameters require gradients.")
        frozen = {id(p) for p in encoder.parameters()}
        if any(id(p) in frozen for o in optimizers for g in o.param_groups for p in g["params"]):
            raise RuntimeError("An optimizer holds encoder parameters: only the probe may be trained.")
    per_probe = sum(p.numel() for p in probes[0].parameters() if p.requires_grad)
    parameter_counts = {"encoder_total": sum(p.numel() for p in encoder_module.parameters()),
                        "encoder_trainable": sum(p.numel() for p in trainable), "encoder_trainable_blocks": trainable_blocks,
                        "probes": len(probes), "probe_trainable_each": per_probe,
                        "probe_trainable_total": per_probe * len(probes)}

    # -- FINGERPRINT: a probe is resumed and tested only with the encoder, data and settings it was trained with.
    settings = {
        "encoder": encoder_settings,
        "classifier": {"pooling_mode": pooling_mode, "num_heads": num_heads, "num_probe_blocks": num_probe_blocks,
                       "attention_hidden": attention_hidden, "dropout": probe_dropout, "heads": specs,
                       "labels_map": labels_map, "use_view_embedding": use_view_embedding, **probe_options},
        "encoder_training": {"trainable_blocks": trainable_blocks, "optimization": encoder_opt_kwargs},
        "data": {"split_column": split_column, "path_column": path_column, "group_column": group_column,
                 "view_column": view_column, "views": list(views) if views else None,
                 "view_labels_map": view_labels_map,
                 "num_videos": num_videos, "shuffle_videos": shuffle_videos,
                 "num_clips_per_video": num_clips_per_video, "frames_per_clip": frames_per_clip,
                 "frame_step": frame_step, "clip_duration": duration, "fps": fps, "resolution": resolution,
                 "normalization": normalization, "min_videos": min_videos, **data_options},
        "evaluation": {"num_bags": num_eval_bags},
        "optimization": {"batch_size": batch_size, "num_epochs": num_epochs, "use_bfloat16": use_bfloat16,
                         "multihead_kwargs": args_opt["multihead_kwargs"]},
        "selection": selection,
        "protocol": copy.deepcopy(PROTOCOL),
        "seed": seed,
        "world_size": world_size,
    }
    manifests = manifest_paths(args_data)
    fingerprint = shared(rank, world_size, device, lambda: provenance.fingerprint(
        checkpoint, manifests, {h: dict(v) for h, v in normalization_targets.items()}, settings))

    # -- RESUME
    start_epoch, history = 0, {}
    if resume_checkpoint and os.path.exists(latest_path) and not scoring_only:
        start_epoch, history = load_checkpoint(latest_path, probes, optimizers, encoder_module if trainable else None)
        provenance.check_fingerprint(history.get("fingerprint"), fingerprint, f"The probe in {folder}")
        for _ in range(start_epoch * ipe):
            [s.step() for s in schedulers]
            [w.step() for w in wd_schedulers]

    run_info = None
    if rank == 0:
        run_info = describe_run(args_eval, checkpoint, manifests, fingerprint, seed, parameter_counts, opt_kwargs,
                                batch_size, world_size, num_epochs, specs, normalization_targets, selection,
                                {"train": train_dataset, "val": val_dataset})
        if not scoring_only:
            save_run_record(folder, args_eval, run_info, fresh=start_epoch == 0)

    wandb_run, wandb_error = None, None
    try:
        wandb_run, _ = init_wandb(args_eval.get("meta") or {}, {**public_config(args_eval), "run_info": run_info},
                                  rank, folder, resuming_training=start_epoch > 0 or scoring_only,
                                  settings=PRIVATE_WANDB_SETTINGS)
    except Exception as e:
        wandb_error = e
    if any_rank_failed(wandb_error is not None, device=device):
        if wandb_error is not None:
            raise wandb_error
        raise RuntimeError("wandb initialization failed on another rank: aborting.")

    score_kwargs = dict(device=device, encoder=encoder_module, batch_size=batch_size, num_workers=num_workers,
                        use_bfloat16=use_bfloat16, specs=specs, normalization=normalization_targets, rank=rank,
                        world_size=world_size, num_bags=num_eval_bags)

    if scoring_only:
        test_dataset = make_dataset("test", training=False) if test_only else None
        if test_only:
            check_disjoint({"train": train_dataset, "val": val_dataset, "test": test_dataset})
        run_test(folder=folder, probes=probes, probe_names=probe_names, val_dataset=val_dataset, trainable=bool(trainable),
                 test_dataset=test_dataset, num_epochs=num_epochs, fingerprint=fingerprint,
                 selection_rule=selection_rule, run_info=run_info, wandb_run=wandb_run, step=num_epochs * ipe,
                 val_only=val_only, **score_kwargs)
        finish_wandb(wandb_run)
        return

    csv_logger = None
    if rank == 0:
        csv_logger = CSVLogger(os.path.join(folder, f"log_r{rank}.csv"), ("%d", "epoch"), ("%.5f", "train_loss"),
                               ("%.5f", "val_score"), ("%d", "val_probe"))
    best_score = float(history.get("best_score", np.inf))
    best_epoch = history.get("best_epoch")
    best_probe = history.get("best_probe")
    if start_epoch > 0 and history.get("rng_states"):
        set_rng_state(history["rng_states"][rank])

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")
        train_sampler.set_epoch(epoch)
        train_losses, train_head_losses = train_one_epoch(
            device=device, encoder=encoder, probes=probes, optimizers=optimizers, schedulers=schedulers,
            wd_schedulers=wd_schedulers, loader=train_loader, use_bfloat16=use_bfloat16, specs=specs,
            normalization=normalization_targets, encoder_trainable=bool(trainable), world_size=world_size)
        val = [evaluate(prediction, specs, normalization_targets, selection_rule)
               for prediction in score_studies(dataset=val_dataset, probes=probes, **score_kwargs)]
        scores = np.asarray([v["score"] for v in val], dtype=float)
        if not np.isfinite(scores).any():
            raise RuntimeError(f"No probe has a finite validation {selection} at epoch {epoch + 1}: the selection "
                               "metric is undefined on this validation split (e.g. one class only, or no target).")
        probe = int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))
        score = float(scores[probe])
        is_best = score < best_score
        if is_best:
            best_score, best_epoch, best_probe = score, epoch + 1, probe
        logger.info(f"[{epoch + 1:5d}] train loss {train_losses.min():.4f}  val {selection} {score:.4f} "
                    f"({probe_names[probe]})  best {best_score:.4f} at epoch {best_epoch}")
        if csv_logger is not None:
            csv_logger.log(epoch + 1, float(train_losses.min()), score, probe)

        payload = {"probe/epoch": epoch + 1, "probe/train_loss": float(train_losses.min()),
                   "probe/val_score": score, "probe/val_score_best": best_score, "probe/best_epoch": best_epoch,
                   **{f"probe/lr/{n}": float(o.param_groups[0]["lr"]) for n, o in zip(probe_names, optimizers)},
                   **{f"probe/train_loss/{n}": float(v) for n, v in zip(probe_names, train_losses)},
                   **{f"probe/val_score/{n}": float(v["score"]) for n, v in zip(probe_names, val)},
                   "probe/val_failed_studies": val[0]["failed"]}
        for head, metrics in val[probe]["heads"].items():
            payload.update({f"probe/val/{head}/{k}": float(v) for k, v in metrics.items()})
            payload[f"probe/train_loss/{head}"] = float(train_head_losses[probe][head])
        log_scalars(wandb_run, payload, step=(epoch + 1) * ipe)

        save_checkpoint(folder, probes, optimizers, epoch + 1, val, scores, best_score, best_epoch, best_probe,
                        normalization_targets, specs, fingerprint, opt_kwargs, selection, world_size, is_best, rank,
                        device, encoder=encoder_module if trainable else None)
    # Record the selected epoch explicitly; W&B's last-epoch summaries are not the selected result.
    run_test(folder=folder, probes=probes, probe_names=probe_names, val_dataset=val_dataset,
             test_dataset=None, num_epochs=num_epochs, fingerprint=fingerprint, selection_rule=selection_rule,
             run_info=run_info, wandb_run=wandb_run, step=num_epochs * ipe, trainable=bool(trainable),
             val_only=True, **score_kwargs)
    finish_wandb(wandb_run)


def unfreeze_top_blocks(encoder, trainable_blocks):
    """Let the encoder's top `trainable_blocks` transformer blocks and its final norm train;
    everything else stays frozen. Returns the trainable parameters (none when 0)."""
    for p in encoder.parameters():
        p.requires_grad = False
    if not trainable_blocks:
        return []
    vit = getattr(encoder, "model", encoder)
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        raise ValueError("The encoder has no `blocks` to unfreeze.")
    if trainable_blocks > len(blocks):
        raise ValueError(f"`trainable_blocks` is {trainable_blocks}, but the encoder has {len(blocks)} blocks.")
    modules = list(blocks[len(blocks) - trainable_blocks:])
    if getattr(vit, "norm", None) is not None:
        modules.append(vit.norm)
    trainable = [p for m in modules for p in m.parameters()]
    for p in trainable:
        p.requires_grad = True
    return trainable


# --------------------------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------------------------- #

def manifest_paths(args_data):
    """{name: path} of the manifests the config names (unused ones `None`), for the fingerprint."""
    if args_data.get("data_filename"):
        return {"data_filename": args_data["data_filename"]}
    # Training and validation-only runs must not open a separate test manifest, even to hash it.
    return {split: args_data.get(f"dataset_{split}") for split in ("train", "val")}


def study_probe_options(classifier, data, pretrained):
    """Resolve architecture options once for construction and checkpoint fingerprinting."""
    encoder = pretrained.get("pretrain_kwargs", {}).get("encoder", {})
    grid = None
    if classifier.get("pooling_mode") == "hierarchical":
        patch = int(encoder.get("patch_size", 16))
        tubelet = int(encoder.get("tubelet_size", 2))
        frames, resolution = int(data.get("frames_per_clip", 16)), int(data.get("resolution", 224))
        if frames % tubelet or resolution % patch:
            raise ValueError("Hierarchical pooling requires frames/resolution divisible by tubelet/patch size.")
        grid = (frames // tubelet, resolution // patch, resolution // patch)
    return dict(query_mode=classifier.get("query_mode", "shared"), task_groups=classifier.get("task_groups"),
                view_dropout=float(classifier.get("view_dropout", 0.0)),
                num_acquisition_classes=len(set((data.get("acquisition_labels_map") or {}).values())),
                local_depth=int(classifier.get("local_depth", 1)),
                local_num_queries=int(classifier.get("local_num_queries", 4)),
                local_use_rope=bool(classifier.get("local_use_rope", False)), token_grid=grid,
                num_clips_per_video=int(data.get("num_clips_per_video", 1)))


def parse_selection(selection, specs):
    """The selection rule: `weighted_loss` (the training objective on validation, lower is better)
    or `<head>/<metric>`, a metric `head_metrics` reports for that head's task (e.g. `lvef/mae`,
    `as_present/auroc`)."""
    if selection == "weighted_loss":
        return {"head": None, "metric": "weighted_loss", "lower_is_better": True}
    if "/" not in selection:
        raise ValueError(f"`experiment.evaluation.selection` is `weighted_loss` or `<head>/<metric>`, not {selection!r}.")
    head, metric = selection.split("/", 1)
    if head not in specs:
        raise ValueError(f"Selection head {head!r} is not in `target_label` {list(specs)}.")
    task = specs[head]["task"]
    if metric not in TASK_METRICS[task]:
        raise ValueError(f"Selection metric {metric!r} is not reported for the {task} head {head!r}; "
                         f"choose one of {sorted(TASK_METRICS[task])}.")
    return {"head": head, "metric": metric, "lower_is_better": metric in LOWER_IS_BETTER}


def check_disjoint(datasets):
    """Refuse shared studies, and shared patients when the manifest supplies patient ids."""
    names = list(datasets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(datasets[a].study_ids) & set(datasets[b].study_ids)
            if shared:
                raise ValueError(f"{len(shared)} studies are in both the {a} and the {b} split (e.g. "
                                 f"{sorted(map(str, shared))[:3]}): every study belongs to one split.")
            pa, pb = getattr(datasets[a], "patient_ids", None), getattr(datasets[b], "patient_ids", None)
            if pa is not None and pb is not None and pa & pb:
                raise ValueError(f"{len(pa & pb)} patients are in both the {a} and the {b} split.")


def target_normalization(specs, train_targets, args_classifier):
    """{head: {mean, std}} for the regression heads: `experiment.classifier.target_mean` and
    `target_std` ({head: value}) when given, else the train split's statistics (population std)."""
    means = args_classifier.get("target_mean") or {}
    stds = args_classifier.get("target_std") or {}
    if not isinstance(means, dict) or not isinstance(stds, dict):
        raise ValueError("`target_mean` and `target_std` are per head: {head: value}.")
    out = {}
    for i, (name, spec) in enumerate(specs.items()):
        if spec["task"] != "regression":
            continue
        values = train_targets[:, i]
        values = values[np.isfinite(values)]
        if name in means or name in stds:
            if name not in means or name not in stds:
                raise ValueError(f"Set both `target_mean.{name}` and `target_std.{name}`, or neither.")
            mean, std = float(means[name]), float(stds[name])
        else:
            if values.size == 0:
                raise ValueError(f"No training study has a value of `{name}`, so it cannot be normalized.")
            mean, std = float(values.mean()), float(values.std(ddof=0))
        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
            raise ValueError(f"`{name}` normalization must be finite with a positive std, not ({mean}, {std}).")
        out[name] = {"mean": mean, "std": std}
    return out


def normalize_targets(targets, specs, normalization):
    """z-score the regression columns of `targets` [B, heads] (a tensor), in place on a copy."""
    out = targets.clone()
    for i, name in enumerate(specs):
        if name in normalization:
            out[:, i] = (out[:, i] - normalization[name]["mean"]) / normalization[name]["std"]
    return out


def shared(rank, world_size, device, compute):
    """`compute()` on rank 0, sent to every rank; a failure on rank 0 stops every rank."""
    value, error = None, None
    if rank == 0:
        try:
            value = compute()
        except Exception as e:
            error = e
    if any_rank_failed(error is not None, device=device):
        if error is not None:
            raise error
        raise RuntimeError("Rank 0 failed while preparing the run.")
    holder = [value]
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.broadcast_object_list(holder, src=0)
    return holder[0]


def init_opt(probes, opt_kwargs, iterations_per_epoch, num_epochs):
    optimizers, schedulers, wd_schedulers = [], [], []
    for probe, kwargs in zip(probes, opt_kwargs):
        param_groups = [{
            "params": [p for p in probe.parameters() if p.requires_grad],
            "mc_warmup_steps": int(kwargs["warmup"] * iterations_per_epoch),
            "mc_start_lr": kwargs["start_lr"], "mc_ref_lr": kwargs["ref_lr"], "mc_final_lr": kwargs["final_lr"],
            "mc_ref_wd": kwargs["ref_wd"], "mc_final_wd": kwargs["final_wd"],
        }]
        optimizers.append(torch.optim.AdamW(param_groups))
        schedulers.append(WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
        wd_schedulers.append(CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
    return optimizers, schedulers, wd_schedulers


# --------------------------------------------------------------------------------------------- #
# Forward passes
# --------------------------------------------------------------------------------------------- #

def encode(encoder, clips, clip_indices, device, use_bfloat16, grad=False):
    """The encoder's tokens for a batch of studies: clips [B, slots*clips, C, T, H, W] ->
    [B, slots*clips*tokens, D], slot by slot. Without `grad` (a frozen encoder) no graph is kept."""
    clips = clips.to(device, non_blocking=True)
    clip_indices = clip_indices.to(device, non_blocking=True)
    x = [[clips[:, j]] for j in range(clips.shape[1])]
    indices = [clip_indices[:, j] for j in range(clip_indices.shape[1])]
    with torch.set_grad_enabled(grad), autocast(device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
        return encoder(x, indices)[0]


def train_one_epoch(device, encoder, probes, optimizers, schedulers, wd_schedulers, loader, use_bfloat16, specs,
                    normalization, encoder_trainable=False, world_size=1):
    """One pass over the training studies for every probe (and the encoder's trainable blocks,
    with the encoder in training mode). Each head's loss is the mean over the labeled studies of
    the global batch (the present targets are counted over every rank). Returns each probe's
    mean loss and each probe's mean loss per head, averaged over the studies of every rank."""
    for p in probes:
        p.train()
    encoder.train(mode=encoder_trainable)
    loss_meters = [AverageMeter() for _ in probes]
    head_meters = [{h: AverageMeter() for h in specs} for _ in probes]
    for itr, batch in enumerate(loader):
        clips, targets, video_mask, clip_indices, view_ids, *extra = batch
        metadata = {k: v.to(device, non_blocking=True) for k, v in (extra[0] if extra else {}).items()}
        features = encode(encoder, clips, clip_indices, device, use_bfloat16, grad=encoder_trainable)
        video_mask = video_mask.to(device, non_blocking=True)
        view_ids = view_ids.to(device, non_blocking=True)
        targets = normalize_targets(targets, specs, normalization).to(device, non_blocking=True)
        # Present targets per head over every rank, so each head's loss is normalized the same way
        # on every rank whatever the sampler gave each one.
        local_counts = torch.isfinite(targets).sum(dim=0).to(torch.float64)
        global_counts = AllReduceSum.apply(local_counts.clone()).cpu().tolist()
        counts = {name: int(c) for name, c in zip(specs, global_counts)}
        batch_losses, batch_parts = [], []
        for probe in probes:
            with autocast(device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = probe(features, video_mask, view_ids, **metadata)
            loss, parts = multitask_loss(specs, outputs, targets, counts=counts, world_size=world_size)
            batch_losses.append(loss)
            batch_parts.append(parts)
        local = any(not bool(torch.isfinite(loss.detach()).item()) for loss in batch_losses)
        if any_rank_failed(local, device=device):
            raise FloatingPointError("Non-finite probe loss; stopping all ranks before the optimizer update.")
        [s.step() for s in schedulers]
        [w.step() for w in wd_schedulers]
        for loss in batch_losses:
            loss.backward()
        for optimizer in optimizers:  # the probes', then the encoder's when it trains.
            optimizer.step()
            optimizer.zero_grad()

        # Sum over the ranks' studies, then divide by their count. Each rank's loss is already
        # scaled by `world_size` / global count, so the rank-sum of losses is the global mean.
        with torch.no_grad():
            count = float(targets.shape[0])
            values = [count] + [float(loss.detach()) for loss in batch_losses]
            for parts in batch_parts:
                for h in specs:
                    values += [float(parts[h][0].detach()), float(parts[h][1])]
            totals = AllReduceSum.apply(torch.tensor(values, device=device, dtype=torch.float64)).cpu().tolist()
            n = totals[0]
            for meter, total in zip(loss_meters, totals[1:1 + len(probes)]):
                meter.update(total / world_size, n=n)
            rest = totals[1 + len(probes):]
            for i in range(len(probes)):
                for j, h in enumerate(specs):
                    s, c = rest[2 * (i * len(specs) + j)], rest[2 * (i * len(specs) + j) + 1]
                    if c > 0:
                        head_meters[i][h].update(s / c, n=c)
        if itr % 10 == 0:
            losses = np.array([m.avg for m in loss_meters])
            mem = torch.cuda.max_memory_allocated() / 1024.0**2 if device.type == "cuda" else 0.0
            logger.info("[%5d] loss %.4f [mean %.4f] [mem: %.2e]" % (itr, losses.min(), losses.mean(), mem))
    encoder.eval()
    return (np.array([m.avg for m in loss_meters]),
            [{h: (m.avg if m.count else float("nan")) for h, m in meters.items()} for meters in head_meters])


class _EachStudyOnce(torch.utils.data.Dataset):
    """Some of a `VideoStudyDataset`'s studies, each read once and returned with its index; a
    study whose videos cannot be read gives `None` (`get_item_study`) instead of being replaced.
    Any other error (a transform or shape problem) propagates and stops the evaluation."""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        return index, self.dataset.get_item_study(index)


def _collate_loaded(batch):
    loaded = [(index, item) for index, item in batch if item is not None]
    failed = [index for index, item in batch if item is None]
    if not loaded:
        return [], None, failed
    return [index for index, _ in loaded], default_collate([item for _, item in loaded]), failed


def score_studies(device, encoder, probes, dataset, batch_size, num_workers, use_bfloat16, specs, normalization,
                  rank, world_size, num_bags=1):
    """Average deterministic bags in prediction space; keep variability for failure analysis."""
    results = []
    for bag in range(num_bags):
        bag_dataset = copy.copy(dataset)
        bag_dataset.eval_bag_index = bag
        results.append(_score_studies_once(device, encoder, probes, bag_dataset, batch_size, num_workers,
                                           use_bfloat16, specs, normalization, rank, world_size))
    if num_bags == 1:
        return results[0]
    return [aggregate_bags([r[p] for r in results], specs, normalization) for p in range(len(probes))]


def aggregate_bags(bags, specs, normalization):
    # Require all requested bags for a study: otherwise an ensemble would silently use
    # different bag counts across patients. Report any failed bag as a failed study.
    common = sorted(set.intersection(*(set(b["indices"]) for b in bags)))
    if not common:
        raise RuntimeError("No study loaded successfully in every evaluation bag.")
    positions = [{int(i): j for j, i in enumerate(b["indices"])} for b in bags]
    outputs, variability = {}, {}
    for name, spec in specs.items():
        raw = np.stack([b["outputs"][name][[pos[i] for i in common]] for b, pos in zip(bags, positions)])
        if spec["task"] == "binary":
            values = 1 / (1 + np.exp(-np.clip(raw, -700, 700)))
            mean = np.clip(values.mean(0), 1e-12, 1 - 1e-12)
            outputs[name] = np.log(mean) - np.log1p(-mean)
        elif spec["task"] == "classification":
            values = np.exp(raw - raw.max(-1, keepdims=True))
            values /= values.sum(-1, keepdims=True)
            outputs[name] = np.log(np.maximum(values.mean(0), 1e-12))
        else:
            values = raw * normalization[name]["std"] + normalization[name]["mean"]
            outputs[name] = raw.mean(0)
        outputs[name][~np.isfinite(raw).all(axis=(0, 2))] = np.nan
        variability[name] = values.std(0)
    return {"indices": np.asarray(common),
            "targets": bags[0]["targets"][[positions[0][i] for i in common]], "outputs": outputs,
            "failed": sorted(set.union(*(set(b["failed"]) for b in bags))),
            "prediction_std": variability, "num_bags": len(bags)}


def _score_studies_once(device, encoder, probes, dataset, batch_size, num_workers, use_bfloat16, specs, normalization,
                        rank, world_size):
    """Score every study of `dataset` exactly once with each probe (each rank every `world_size`-th
    study; results gathered on every rank). Returns, per probe, a dict with the studies' indices
    (dataset order), their raw `targets` [n, heads], `outputs` {head: [n, outputs]} (raw head
    outputs) and the `failed` study indices."""
    modules = [getattr(p, "module", p) for p in probes]
    for m in modules:
        m.eval()
    encoder.eval()
    loader = torch.utils.data.DataLoader(_EachStudyOnce(dataset, range(rank, len(dataset), world_size)),
                                         batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_loaded,
                                         pin_memory=True)
    indices, targets, failed = [], [], []
    outputs = [{h: [] for h in specs} for _ in modules]
    for batch_indices, data, batch_failed in loader:
        failed += batch_failed
        if data is None:
            continue
        clips, batch_targets, video_mask, clip_indices, view_ids, *extra = data
        metadata = {k: v.to(device, non_blocking=True) for k, v in (extra[0] if extra else {}).items()}
        features = encode(encoder, clips, clip_indices, device, use_bfloat16)
        video_mask = video_mask.to(device, non_blocking=True)
        view_ids = view_ids.to(device, non_blocking=True)
        with torch.no_grad(), autocast(device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            for m, out in zip(modules, outputs):
                for h, value in m(features, video_mask, view_ids, **metadata).items():
                    out[h] += value.float().cpu().tolist()
        indices += batch_indices
        targets += batch_targets.tolist()

    local = (indices, targets, outputs, failed)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    indices = [i for g in gathered for i in g[0]]
    failed = sorted(i for g in gathered for i in g[3])
    if len(set(indices)) != len(indices) or len(indices) + len(failed) != len(dataset):
        raise RuntimeError(f"Scored {len(indices)} and failed {len(failed)} of {len(dataset)} studies: every study "
                           "should be scored or failed exactly once.")
    if not indices:
        raise RuntimeError(f"None of the {len(dataset)} studies could be read.")
    order = np.argsort(indices)
    n = len(order)
    targets = np.asarray([t for g in gathered for t in g[1]], dtype=float).reshape(n, -1)[order]
    results = []
    for p in range(len(modules)):
        out = {h: np.asarray([v for g in gathered for v in g[2][p][h]], dtype=float).reshape(n, -1)[order]
               for h in specs}
        results.append({"indices": np.asarray(indices)[order], "targets": targets, "outputs": out, "failed": failed})
    return results


# --------------------------------------------------------------------------------------------- #
# Metrics, selection and predictions
# --------------------------------------------------------------------------------------------- #

def head_metrics(spec, target, output, normalization=None):
    """The metrics of one head over the studies whose target is present. `target`: [n] raw;
    `output`: [n, outputs] raw head outputs. Regression: MAE, RMSE, R^2, Pearson r and bias in
    the target's units; binary: AUROC, AUPRC and accuracy at 0.5; classification: accuracy and
    the one-vs-rest AUROC averaged over the classes present. Every head reports its loss (the
    training criterion) and `n`."""
    valid = np.isfinite(target)
    y, out = target[valid], output[valid]
    invalid = int((~np.isfinite(output).all(axis=1)).sum())
    if invalid:
        metrics = {k: float("nan") for k in TASK_METRICS[spec["task"]]}
        metrics.update(n=int(y.size), nonfinite_outputs=invalid)
        return metrics
    if spec["task"] == "regression":
        z = (y - normalization["mean"]) / normalization["std"]
        prediction = out[:, 0] * normalization["std"] + normalization["mean"]
        metrics = probe_metrics.regression(y, prediction)
    elif spec["task"] == "binary":
        z = y
        prob = 1.0 / (1.0 + np.exp(-out[:, 0]))
        metrics = {"n": int(y.size), "auroc": probe_metrics.auroc(y > 0.5, prob),
                   "auprc": probe_metrics.average_precision(y > 0.5, prob),
                   "acc": float(((prob > 0.5) == (y > 0.5)).mean()) if y.size else float("nan"),
                   "prevalence": float((y > 0.5).mean()) if y.size else float("nan")}
    else:
        z = y
        predicted = out.argmax(axis=1) if y.size else np.zeros(0)
        prob = np.exp(out - out.max(axis=1, keepdims=True)) if y.size else out
        prob = prob / prob.sum(axis=1, keepdims=True) if y.size else prob
        present = sorted(set(y.astype(int).tolist()))
        aurocs = [probe_metrics.auroc(y == c, prob[:, c]) for c in present]
        aurocs = [a for a in aurocs if np.isfinite(a)]
        metrics = {"n": int(y.size), "acc": float((predicted == y).mean()) if y.size else float("nan"),
                   "auroc": float(np.mean(aurocs)) if aurocs else float("nan")}
    loss, count = head_loss(spec, torch.as_tensor(out, dtype=torch.float32), torch.as_tensor(z, dtype=torch.float32))
    metrics["loss"] = float(loss) / count if count else float("nan")
    return metrics


def evaluate(scored, specs, normalization, selection_rule):
    """Per-head metrics of one probe's scores, and its selection score (lower is better)."""
    heads = {name: head_metrics(spec, scored["targets"][:, i], scored["outputs"][name], normalization.get(name))
             for i, (name, spec) in enumerate(specs.items())}
    if selection_rule["head"] is None:
        terms = [spec["weight"] * heads[name]["loss"] for name, spec in specs.items() if heads[name]["n"] > 0]
        score = float(sum(terms)) if terms else float("nan")
    else:
        value = heads[selection_rule["head"]][selection_rule["metric"]]
        score = float(value) if selection_rule["lower_is_better"] else -float(value)
    if any(m.get("nonfinite_outputs", 0) for m in heads.values()):
        score = float("nan")
    result = {"score": score, "heads": heads, "studies": int(len(scored["indices"])), "failed": len(scored["failed"])}
    if "prediction_std" in scored:
        result["num_bags"] = scored["num_bags"]
        result["mean_bag_prediction_std"] = {h: float(v.mean()) for h, v in scored["prediction_std"].items()}
    return result


def prediction_table(scored, dataset, specs, normalization):
    """One row per scored study: its id, and each head's label and prediction (regression: in
    the target's units; binary: the probability; classification: the class and its probabilities)."""
    table = {"study_id": [dataset.study_ids[i] for i in scored["indices"]]}
    for i, (name, spec) in enumerate(specs.items()):
        out = scored["outputs"][name]
        table[f"{name}_label"] = scored["targets"][:, i]
        if spec["task"] == "regression":
            table[f"{name}_prediction"] = out[:, 0] * normalization[name]["std"] + normalization[name]["mean"]
        elif spec["task"] == "binary":
            table[f"{name}_prediction"] = 1.0 / (1.0 + np.exp(-out[:, 0]))
        else:
            prob = np.exp(out - out.max(axis=1, keepdims=True))
            prob = prob / prob.sum(axis=1, keepdims=True)
            table[f"{name}_prediction"] = out.argmax(axis=1)
            for c in range(out.shape[1]):
                table[f"{name}_prob_{c}"] = prob[:, c]
        if "prediction_std" in scored:
            std = scored["prediction_std"][name]
            for c in range(std.shape[1]):
                suffix = f"_{c}" if spec["task"] == "classification" else ""
                table[f"{name}_bag_std{suffix}"] = std[:, c]
    return pd.DataFrame(table)


# --------------------------------------------------------------------------------------------- #
# Checkpoints, records and the test
# --------------------------------------------------------------------------------------------- #

def save_checkpoint(folder, probes, optimizers, epoch, val, scores, best_score, best_epoch, best_probe, normalization,
                    specs, fingerprint, opt_kwargs, selection, world_size, is_best, rank, device, encoder=None):
    """Save `latest.pt` (and `best.pt` when selected): the probes, every optimizer (the encoder's
    last when it trains), the fine-tuned encoder when it trains, the epoch's validation metrics,
    the selection state, the normalization, the fingerprint and each rank's random number state."""
    rng_states = gather_rng_states(world_size)  # on every rank, before rank 0 saves.
    save_dict = {
        "probes": [p.state_dict() for p in probes],
        "opt": [o.state_dict() for o in optimizers],
        "encoder": None if encoder is None else encoder.state_dict(),
        "epoch": epoch,
        "world_size": world_size,
        "val": val,                                   # every probe's validation metrics for this epoch.
        "val_score_per_probe": np.asarray(scores, dtype=float).tolist(),
        "best_score": float(best_score), "best_epoch": best_epoch, "best_probe": best_probe,
        "selection": selection,
        "normalization": normalization, "heads": specs,
        "fingerprint": fingerprint, "rng_states": rng_states, "opt_grid": opt_kwargs,
    }
    error = None
    if rank == 0:
        try:
            if is_best:
                save_atomically(save_dict, os.path.join(folder, "best.pt"))
                logger.info(f"Generated new best probe: {os.path.join(folder, 'best.pt')}")
            save_atomically(save_dict, os.path.join(folder, "latest.pt"))
        except Exception as exc:
            error = exc
    if any_rank_failed(error is not None, device=device):
        if error is not None:
            raise error
        raise RuntimeError("Rank 0 could not save the probe checkpoint; stopping every rank.")


def load_checkpoint(path, probes, optimizers, encoder=None):
    """Resume the probes, the optimizers and, when it trains, the encoder from `path`."""
    checkpoint = robust_checkpoint_loader(path, map_location=torch.device("cpu"))
    for p, state in zip(probes, checkpoint["probes"]):
        p.load_state_dict(state)
    if len(checkpoint["opt"]) != len(optimizers):
        raise ValueError(f"{path} holds {len(checkpoint['opt'])} optimizers, this run {len(optimizers)}: the probe "
                         "was trained with another encoder-training setting.")
    for o, state in zip(optimizers, checkpoint["opt"]):
        o.load_state_dict(state)
    if encoder is not None:
        if checkpoint.get("encoder") is None:
            raise ValueError(f"{path} holds no fine-tuned encoder, but this run trains the encoder.")
        encoder.load_state_dict(checkpoint["encoder"])
    epoch = int(checkpoint["epoch"])
    logger.info(f"Resumed the probes and optimizers from epoch {epoch} of {path}")
    keys = ("best_score", "best_epoch", "best_probe", "fingerprint", "rng_states")
    return epoch, {k: checkpoint[k] for k in keys if k in checkpoint}


def describe_run(args_eval, checkpoint, manifests, fingerprint, seed, parameter_counts, opt_kwargs, batch_size,
                 world_size, num_epochs, specs, normalization, selection, datasets):
    """The run's record without filesystem paths: code version, encoder and manifest identities,
    split sizes, heads, normalization, optimization, protocol and selection."""
    commit, dirty = provenance.git_state()
    files = {}
    for name, path in manifests.items():
        if path and os.path.exists(path):
            with open(path) as f:
                rows = sum(1 for line in f if line.strip())
            files[name] = {"file": os.path.basename(path), "sha256": fingerprint["manifests_sha256"][name], "rows": rows}
    meta = args_eval.get("meta") or {}
    return {
        "experiment": meta.get("wandb_run_name") or args_eval.get("tag"),
        "git_commit": commit, "git_dirty": dirty, "seed": seed,
        "encoder_checkpoint": {"file": os.path.basename(checkpoint), "sha256": fingerprint["encoder_sha256"]},
        "manifests": files,
        "splits": {split: {"studies": len(ds), "videos": sum(len(s[1]) for s in ds.studies),
                           "selection": SELECTION[ds.selection], "eval_seed": ds.eval_seed}
                   for split, ds in datasets.items()},
        "study_probe": fingerprint["settings"]["classifier"],
        "evaluation": fingerprint["settings"]["evaluation"],
        "heads": specs, "normalization": normalization, "parameters": parameter_counts,
        "optimization": {"optimizer": "AdamW", "lr_schedule": "linear warmup from start_lr, then cosine to final_lr",
                         "weight_decay_schedule": "cosine from weight_decay to final_weight_decay",
                         "epochs": num_epochs, "batch_size_per_gpu": batch_size, "world_size": world_size,
                         "global_batch_size": batch_size * world_size, "probes": opt_kwargs},
        "protocol": PROTOCOL, "selection": selection,
    }


def save_run_record(folder, args_eval, run_info, fresh):
    path = os.path.join(folder, "params-probe.yaml")
    text = yaml.safe_dump(args_eval, sort_keys=False)
    if not fresh and os.path.exists(path):
        with open(path) as f:
            if f.read() != text:
                logger.warning(f"The config differs from the one this probe started with; {path} keeps the original.")
        return
    with open(path, "w") as f:
        f.write(text)
    with open(os.path.join(folder, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)


def check_finished(folder, num_epochs):
    """Refuse to test a probe that has not finished, or whose `best.pt` is not the recorded best."""
    def saved(name):
        path = os.path.join(folder, name)
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True) if os.path.exists(path) else {}

    latest = saved("latest.pt")
    done = latest.get("epoch", 0)
    if done < num_epochs:
        raise RuntimeError(f"The probe has finished {done} of {num_epochs} epochs; test it once it is complete.")
    best = saved("best.pt")
    if (best.get("epoch"), best.get("best_score")) != (latest.get("best_epoch"), latest.get("best_score")):
        raise RuntimeError(f"`best.pt` holds epoch {best.get('epoch')}, but training selected epoch "
                           f"{latest.get('best_epoch')}: the folder does not hold the selected probe.")


def run_test(folder, probes, probe_names, val_dataset, test_dataset, num_epochs, fingerprint, selection_rule, run_info,
             wandb_run, step, device, encoder, batch_size, num_workers, use_bfloat16, specs, normalization, rank,
             world_size, trainable=False, val_only=False, num_bags=1):
    """Score validation and optionally the test with the probe and epoch chosen on validation (`best.pt`, with
    its fine-tuned encoder when the encoder trained), and write `test_metrics.json`,
    `test_predictions.csv` and `val_predictions.csv`. The prediction files list study ids, which
    can identify patients: keep them where the data is."""
    check_finished(folder, num_epochs)
    if not val_only and os.path.exists(os.path.join(folder, "test_metrics.json")):
        raise RuntimeError("This probe has already been tested (`test_metrics.json` exists); repeat testing only if "
                           "the previous evaluation was invalid.")
    checkpoint_path = os.path.join(folder, "best.pt")
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    provenance.check_fingerprint(checkpoint.get("fingerprint"), fingerprint, checkpoint_path,
                                 ignore=("settings.world_size",))
    probe = int(checkpoint["best_probe"])
    module = getattr(probes[probe], "module", probes[probe])
    module.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["probes"][probe].items()})
    module.eval()
    if trainable:
        if checkpoint.get("encoder") is None:
            raise ValueError(f"{checkpoint_path} holds no fine-tuned encoder, but this config trains the encoder.")
        encoder.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["encoder"].items()})
        logger.info("Loaded the fine-tuned encoder of the selected epoch.")
    logger.info(f"Testing {probe_names[probe]} from epoch {checkpoint['epoch']} of {checkpoint_path}")

    def score(dataset):
        return score_studies(device, encoder, [module], dataset, batch_size, num_workers, use_bfloat16, specs,
                             normalization, rank, world_size, num_bags=num_bags)[0]

    val_scored = score(val_dataset)
    test_scored = None if val_only else score(test_dataset)
    if rank != 0:
        return None
    val = evaluate(val_scored, specs, normalization, selection_rule)
    test = None if val_only else evaluate(test_scored, specs, normalization, selection_rule)
    prediction_table(val_scored, val_dataset, specs, normalization).to_csv(
        os.path.join(folder, "val_predictions.csv"), index=False)
    if not val_only:
        prediction_table(test_scored, test_dataset, specs, normalization).to_csv(
            os.path.join(folder, "test_predictions.csv"), index=False)
    result = {
        "selected_on": "validation", "selection": checkpoint.get("selection"),
        "checkpoint": checkpoint_path, "checkpoint_sha256": provenance.sha256(checkpoint_path),
        "epoch": int(checkpoint["epoch"]), "probe": probe, "probe_name": probe_names[probe],
        "val_score": float(checkpoint["best_score"]),
        "val": val, "normalization": normalization,
        "run_info": run_info, "failed_rows": (val_scored if val_only else test_scored)["failed"],
    }
    if not val_only:
        result["test"] = test
    prefix = "selected_val" if val_only else "test"
    filename = "val_metrics.json" if val_only else "test_metrics.json"
    tmp = os.path.join(folder, filename + ".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, os.path.join(folder, filename))

    metrics_result = val if val_only else test
    scalars = {f"probe/{prefix}_epoch": result["epoch"], f"probe/{prefix}_probe": probe,
               f"probe/{prefix}/score": metrics_result["score"],
               f"probe/{prefix}/failed_studies": metrics_result["failed"]}
    for head, metrics in metrics_result["heads"].items():
        scalars.update({f"probe/{prefix}/{head}/{k}": float(v) for k, v in metrics.items()})
    for head, value in metrics_result.get("mean_bag_prediction_std", {}).items():
        scalars[f"probe/{prefix}/{head}/bag_std"] = value
    log_scalars(wandb_run, scalars, step=step)
    logger.info(f"{prefix} of {probe_names[probe]} (epoch {result['epoch']}): {json.dumps(metrics_result)}")
    return result
