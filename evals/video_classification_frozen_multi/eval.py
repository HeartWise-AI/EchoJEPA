# evals/video_classification_frozen_multi/eval.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen multi-view probe for study-level RVSP regression (early fusion).

The unit is the study: a manifest row (`data/make_rvsp_probe_manifests.py`) holds one RVSP target
and the videos of the study's view slots. The frozen encoder embeds each present video; the tokens
of every present slot position, plus a learned slot (view) and clip embedding, are concatenated and
pooled by one attentive probe. Missing or unreadable videos are removed by the key padding mask,
never replaced by a stand-in video.

The protocol is the frozen Visual EF probe's (`evals/video_classification_frozen/eval.py`,
docs/visual_ef_probe.md): its optimizer grid and schedules, checkpoint writing, fingerprint,
resume, random-state, finished-probe and test-once utilities are imported from there, not copied.
What is specific here is the study-level loader and scoring. Protocol and outputs:
docs/rvsp_multiview_probe.md.
"""

import copy
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen import eval as ef
from evals.video_classification_frozen import metrics as probe_metrics
from evals.video_classification_frozen import provenance
from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from evals.video_classification_frozen_multi.models import FUSIONS, SlotEncoder, make_probe
from src.datasets.video_dataset import DEFAULT_LOADER_TIMEOUT, _worker_init_fn, dataloader_timeout
from src.datasets.video_group_dataset import VideoGroupDataset, collate_studies
from src.utils import wandb_logging
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduceSum, any_rank_failed, init_distributed
from src.utils.logging import AverageMeter, CSVLogger
from src.utils.wandb_logging import finish_wandb, init_wandb, log_reference, log_scalars

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

EVAL_NAME = "video_classification_frozen_multi"
SELECTION = ("Select the epoch and probe head with the lowest per-study validation MAE (mmHg); after each epoch, "
             "every validation study is scored once by every head.")
AGGREGATION = {
    "early": ("Study prediction: the frozen encoder embeds each of a present video's num_segments clips, whose tokens "
              "are concatenated along time; the tokens of every present slot position, plus a learned slot (view) "
              "and clip embedding, are concatenated and one attentive probe pools them into the prediction. "
              "Positions without a readable video are removed by the key padding mask."),
    "late": ("Development baseline: each present slot position's tokens, plus its slot and clip embedding, are pooled "
             "into a prediction of their own; the study's prediction is the mean over its present positions."),
}
# How each split's studies are drawn and batched; in the fingerprint, so a probe is continued and
# tested only under the protocol it was trained with. Raise the version when that changes.
PROTOCOL = {
    "version": 1,
    "unit": "study",
    "missing_videos": "masked",
    "train": {"clip_positions": "random", "drop_last": True},
    "evaluation": {"clip_positions": "fixed", "drop_last": False},
}
DEFAULT_RANGES = [30, 40, 50, 60]  # error by reference RVSP (mmHg): < 30, 30-40, ..., >= 60.
# Validation-only scoring (`score_validation.py`): its outputs, and how far (mmHg) the recomputed MAE of the
# selected head may sit from the one `best.pt` recorded (bfloat16 on GPU, another batch composition).
VAL_PREDICTIONS = "val_selected_predictions.csv"
VAL_METRICS = "val_selected_metrics.json"
DEFAULT_SCORE_TOLERANCE = 0.01


def main(args_eval, resume_preempt=False):
    if args_eval.get("val_only"):
        raise ValueError("The multi-view probe has no `val_only` mode: validation runs after every training epoch.")
    test_only = bool(args_eval.get("test_only", False))
    # Validation-only scoring of a finished probe (`score_validation.py`): {"overwrite": bool, "tolerance": mmHg}.
    score_val = args_eval.get("score_validation")
    if test_only and score_val:
        raise ValueError("`test_only` and `score_validation` are separate runs.")
    evaluating = test_only or bool(score_val)  # nothing is trained

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder")
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag")
    num_workers = args_eval.get("num_workers", 12)
    # Seconds without any batch before a loader raises (0: never); see src/datasets/video_dataset.py.
    loader_timeout = args_eval.get("loader_timeout", DEFAULT_LOADER_TIMEOUT)

    # -- ENCODER (as the EF probe builds it)
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")
    encoder_settings = copy.deepcopy(
        {"module_name": module_name, "pretrain_kwargs": args_model, "wrapper_kwargs": args_wrapper})

    args_exp = args_eval.get("experiment")

    # -- PROBE
    args_classifier = args_exp.get("classifier")
    if args_classifier.get("task_type", "regression") != "regression":
        raise ValueError("The multi-view probe is a regression probe (`task_type: regression`).")
    if args_classifier.get("probe_type", "attentive") != "attentive":
        raise ValueError("The multi-view probe is attentive (`probe_type: attentive`).")
    if args_classifier.get("num_targets", 1) != 1:
        raise ValueError("The multi-view probe predicts one target per study (`num_targets: 1`).")
    fusion = args_classifier.get("fusion", "early")
    if fusion not in FUSIONS:
        raise ValueError(f"`experiment.classifier.fusion` is one of {FUSIONS}, not {fusion!r}.")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    # -- DATA
    args_data = args_exp.get("data")
    if args_data.get("dataset_type", "VideoGroupDataset") != "VideoGroupDataset":
        raise ValueError("The multi-view probe reads per-study manifests (`dataset_type: VideoGroupDataset`).")
    train_path, val_path = args_data.get("dataset_train"), args_data.get("dataset_val")
    test_path = args_data.get("dataset_test")
    if test_only and not test_path:
        raise ValueError("`test_only` needs `experiment.data.dataset_test`.")
    video_index_path = args_data.get("video_index")
    if not video_index_path:
        raise ValueError("The multi-view probe needs `experiment.data.video_index` (the split check of every video).")
    if args_data.get("num_views_per_segment", 1) != 1:
        raise ValueError("The multi-view probe reads one spatial view per clip (`num_views_per_segment: 1`).")
    clip_settings = read_clip_settings(args_data)
    target_mean, target_std = args_data.get("target_mean"), args_data.get("target_std")
    ef.check_normalization(target_mean, target_std, train_path)
    target_mean, target_std = float(target_mean), float(target_std)
    info_path, info = load_probe_info(train_path)

    # -- EVALUATION
    args_evaluation = args_exp.get("evaluation") or {}
    if args_evaluation.get("selection", "study") != "study":
        raise ValueError("A manifest row is a study: `experiment.evaluation.selection` is `study`.")
    ranges = list(args_evaluation.get("rvsp_ranges", DEFAULT_RANGES))
    targets = dict(args_evaluation.get("targets") or {})
    if set(targets) - {"level", "mae_below"} or targets.get("level", "study") != "study":
        raise ValueError("`experiment.evaluation.targets` holds `mae_below` (mmHg), per study.")
    targets.pop("level", None)

    seed = int((args_eval.get("meta") or {}).get("seed", ef._GLOBAL_SEED))

    # -- OPTIMIZATION (the EF probe's grid and schedules)
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(ref_wd=k.get("weight_decay"), final_wd=k.get("final_weight_decay"), start_lr=k.get("start_lr"),
             ref_lr=k.get("lr"), final_lr=k.get("final_lr"), warmup=k.get("warmup"))
        for k in args_opt.get("multihead_kwargs")
    ]

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    folder = os.path.join(pretrain_folder, EVAL_NAME)
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, "latest.pt")
    if score_val:  # refuse before building anything
        try:
            ef.check_finished(folder, num_epochs)
        except RuntimeError as e:
            raise RuntimeError(f"Validation scoring needs a finished probe. {e}") from e
        existing = [n for n in (VAL_PREDICTIONS, VAL_METRICS) if os.path.exists(os.path.join(folder, n))]
        if existing and not score_val.get("overwrite", False):
            raise RuntimeError(f"{existing} already exist in {folder}: pass --overwrite to score again.")
    if rank == 0 and not evaluating:
        csv_logger = CSVLogger(os.path.join(folder, f"log_r{rank}.csv"), ("%d", "epoch"), ("%.5f", "train_mae"),
                               ("%.5f", "val_study_mae"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -- DATA: the train and validation manifests only; the test manifest is opened by `--test_only` alone.
    video_index = provenance.load_video_index(video_index_path)
    train_set = make_dataset(train_path, training=True, **clip_settings)
    val_set = make_dataset(val_path, training=False, **clip_settings)
    check_layout(info, train_set, val_set)
    check_manifest(video_index, val_set, "val")
    if not test_only:
        check_manifest(video_index, train_set, "train")
    train_loader, train_sampler = make_train_loader(train_set, batch_size, world_size, rank, num_workers,
                                                    loader_timeout=loader_timeout)
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")
    if ipe == 0 and not evaluating:
        raise ValueError(f"The training manifest gives no full batch of {batch_size} studies on each of {world_size} "
                         "GPUs: nothing to train on.")

    # -- MODELS
    encoder = SlotEncoder(init_module(
        module_name=module_name, frames_per_clip=clip_settings["frames_per_clip"],
        resolution=clip_settings["resolution"], checkpoint=checkpoint, model_kwargs=args_model,
        wrapper_kwargs=args_wrapper, device=device,
    ))
    classifiers = [
        make_probe(fusion, embed_dim=encoder.embed_dim, num_heads=num_heads, depth=num_probe_blocks,
                   layout=train_set.layout, num_targets=1).to(device)
        for _ in opt_kwargs
    ]
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    optimizer, scaler, scheduler, wd_scheduler = ef.init_opt(
        classifiers=classifiers, opt_kwargs=opt_kwargs, iterations_per_epoch=ipe, num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)

    # Train only the probe: the encoder is frozen and none of its parameters is in an optimizer.
    if any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("The encoder must be frozen: some of its parameters require gradients.")
    frozen = {id(p) for p in encoder.parameters()}
    if any(id(p) in frozen for o in optimizer for g in o.param_groups for p in g["params"]):
        raise RuntimeError("An optimizer holds encoder parameters: only the probe may be trained.")
    per_head = sum(p.numel() for p in classifiers[0].parameters() if p.requires_grad)
    parameter_counts = {"encoder_frozen": sum(p.numel() for p in encoder.parameters()),
                        "probe_heads": len(classifiers), "probe_trainable_per_head": per_head,
                        "probe_trainable_total": per_head * len(classifiers)}

    # -- FINGERPRINT: the encoder, the train and validation manifests, the video index, the
    # probe_info.json (which records the test manifest's checksum, so the test split is fixed
    # before training without being opened), the normalization and the settings.
    settings = {
        "encoder": encoder_settings,
        "classifier": dict(args_classifier),
        "slots": {"spec": info.get("slots"), "positions": list(train_set.position_names), "seed": info.get("seed")},
        "clips": {"dataset_type": "VideoGroupDataset", "num_views_per_segment": 1,
                  **clip_settings, "temporal_padding": clip_settings["temporal_padding"] or "black"},
        "optimization": {"batch_size": batch_size, "num_epochs": num_epochs, "use_bfloat16": use_bfloat16,
                         "multihead_kwargs": args_opt.get("multihead_kwargs")},
        "evaluation": {"rvsp_ranges": ranges, "targets": copy.deepcopy(targets)},
        "selection": "study",
        "protocol": copy.deepcopy(PROTOCOL),
        "seed": seed,
        "world_size": world_size,
    }
    manifests = {"train": train_path, "val": val_path, "video_index": video_index_path, "probe_info": info_path}
    fingerprint = ef.shared_fingerprint(rank, world_size, device, checkpoint, manifests,
                                        {"target_mean": target_mean, "target_std": target_std}, settings)

    start_epoch, history = 0, {}
    if resume_checkpoint and os.path.exists(latest_path) and not evaluating:
        classifiers, optimizer, scaler, start_epoch, history = ef.load_checkpoint(
            device=device, r_path=latest_path, classifiers=classifiers, opt=optimizer, scaler=scaler)
        provenance.check_fingerprint(history.get("fingerprint"), fingerprint, f"The probe in {folder}")
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    run_info = None
    if rank == 0:
        run_info = describe_run(args_eval, checkpoint, manifests, fingerprint, video_index, info, train_set, val_set,
                                seed, parameter_counts, opt_kwargs, batch_size, world_size, num_epochs, fusion,
                                ranges, targets)
        if not evaluating:
            ef.save_run_record(folder, args_eval, run_info, fresh=start_epoch == 0)

    wandb_run, wandb_init_error = None, None
    try:
        if not score_val:  # validation scoring sends nothing to wandb
            wandb_run, _ = init_wandb(args_eval.get("meta") or {},
                                      {**provenance.public_config(args_eval), "run_info": run_info}, rank, folder,
                                      resuming_training=start_epoch > 0 or test_only,
                                      settings=provenance.PRIVATE_WANDB_SETTINGS)
    except Exception as e:
        wandb_init_error = e
    if any_rank_failed(wandb_init_error is not None, device=device):
        if wandb_init_error is not None:
            raise wandb_init_error
        raise RuntimeError("wandb initialization failed on another rank: aborting.")
    head_names = [f"head{i}_lr{k['ref_lr']:g}_wd{k['ref_wd']:g}" for i, k in enumerate(opt_kwargs)]

    if score_val:
        return score_selected_validation(
            device=device, encoder=encoder, classifiers=classifiers, dataset=val_set, batch_size=batch_size,
            num_workers=num_workers, loader_timeout=loader_timeout, checkpoint_path=os.path.join(folder, "best.pt"),
            use_bfloat16=use_bfloat16, target_mean=target_mean, target_std=target_std, head_names=head_names,
            folder=folder, rank=rank, world_size=world_size, ranges=ranges, aggregation=AGGREGATION[fusion],
            fingerprint=fingerprint, tolerance=float(score_val.get("tolerance", DEFAULT_SCORE_TOLERANCE)))

    if test_only:
        ef.check_finished(folder, num_epochs)
        if os.path.exists(os.path.join(folder, "test_metrics.json")):
            raise RuntimeError("This probe has already been tested (`test_metrics.json` exists); repeat testing only "
                               "if the previous evaluation was invalid.")
        check_test_manifest(info, test_path)
        test_set = make_dataset(test_path, training=False, **clip_settings)
        check_layout(info, test_set)
        check_manifest(video_index, test_set, "test")
        run_test(device=device, encoder=encoder, classifiers=classifiers, dataset=test_set, val_dataset=val_set,
                 batch_size=batch_size, num_workers=num_workers, loader_timeout=loader_timeout,
                 checkpoint_path=os.path.join(folder, "best.pt"),
                 use_bfloat16=use_bfloat16, target_mean=target_mean, target_std=target_std, head_names=head_names,
                 folder=folder, rank=rank, world_size=world_size, ranges=ranges, targets=targets,
                 aggregation=AGGREGATION[fusion], run_info=run_info, wandb_run=wandb_run, step=num_epochs * ipe,
                 fingerprint=fingerprint)
        finish_wandb(wandb_run)
        return

    # A resumed probe updates `best.pt` only for an epoch better than every epoch before the interruption.
    selection = Selection(history if start_epoch > 0 else None, start_epoch)
    if start_epoch > 0 and rank == 0 and selection.best_epoch is not None:
        ef.restore_best(folder, selection.best_epoch)
    if start_epoch > 0:
        if history.get("rng_states"):
            ef.set_rng_state(history["rng_states"][rank])
        else:
            logger.warning("The checkpoint has no random number state: the resumed run draws other data "
                           "orders and augmentations than an uninterrupted one.")

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        train_mae, train_heads, train_losses = train_one_epoch(
            device=device, encoder=encoder, classifiers=classifiers, optimizer=optimizer, scheduler=scheduler,
            wd_scheduler=wd_scheduler, data_loader=train_loader, use_bfloat16=use_bfloat16, target_std=target_std)
        val = validate(device=device, encoder=encoder, classifiers=classifiers, dataset=val_set,
                       batch_size=batch_size, num_workers=num_workers, use_bfloat16=use_bfloat16,
                       target_mean=target_mean, target_std=target_std, rank=rank, world_size=world_size,
                       loader_timeout=loader_timeout)
        val_heads = val["study_mae"]
        is_best = selection.update(epoch + 1, val_heads)
        logger.info("[%5d] train MAE: %.3f  val study MAE (best head): %.3f (best: %.3f, epoch %s)" % (
            epoch + 1, train_mae, float(val_heads.min()), selection.best, selection.best_epoch))
        if rank == 0:
            csv_logger.log(epoch + 1, train_mae, float(val_heads.min()))

        payload = {
            "probe/epoch": epoch + 1,
            "probe/val_study_mae_best": selection.best,
            "probe/best_epoch": selection.best_epoch,
            **{f"probe/lr/{h}": float(o.param_groups[0]["lr"]) for h, o in zip(head_names, optimizer)},
            "probe/train_mae": train_mae,
            **{f"probe/train_mae/{h}": float(v) for h, v in zip(head_names, train_heads)},
            "probe/train_loss": float(np.min(train_losses)),
            **{f"probe/train_loss/{h}": float(v) for h, v in zip(head_names, train_losses)},
            "probe/val_failed_studies": val["failed_studies"],
            "probe/val_failed_videos": val["failed_videos"],
        }
        for key in ("study_mae", "loss"):
            payload[f"probe/val_{key}"] = float(val[key].min())
            payload.update({f"probe/val_{key}/{h}": float(v) for h, v in zip(head_names, val[key])})
        log_scalars(wandb_run, payload, step=(epoch + 1) * ipe)

        save_dict = {
            "classifiers": [c.state_dict() for c in classifiers],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": None if scaler is None else [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch + 1,
            "batch_size": batch_size,
            "world_size": world_size,
            **selection.record(val_heads),
            "selection": SELECTION,
            "selection_metric": "study_mae",
            "val_scores_per_head": {k: np.asarray(val[k], dtype=float).tolist() for k in ("study_mae", "loss")},
            "fingerprint": fingerprint,
            "rng_states": ef.gather_rng_states(world_size),  # on every rank, before rank 0 saves.
            "opt_grid": opt_kwargs,
        }
        ef._save_probe_checkpoints(save_dict, folder, is_best, rank, device)

    best_path = os.path.join(folder, "best.pt")
    if rank == 0 and os.path.exists(best_path):
        log_reference(wandb_run, "probe_checkpoint", "probe-checkpoint",
                      ef.checkpoint_reference(best_path, "regression", head_names, fingerprint))
    finish_wandb(wandb_run)


class Selection:
    """The validation record a probe checkpoint holds, as the EF probe keeps it (the fields
    `check_finished`, `restore_best` and `checkpoint_reference` read): per epoch, every head's
    per-study validation MAE; the epoch whose best head is lowest so far is `best.pt`. Lower is
    better; on a tie the earlier epoch is kept."""

    def __init__(self, history=None, epochs_done=0):
        self.best, self.best_epoch, self.count, self.total = float("inf"), None, 0, 0.0
        self.best_per_head = self.sum_per_head = self.min_per_head = self.best_epoch_per_head = None
        if history and history.get("best_epoch_per_head") is not None:
            self.best, self.best_epoch = float(history["best_val_acc"]), history.get("best_epoch")
            self.count, self.total = epochs_done, float(history["mean_val_acc"]) * epochs_done
            self.best_per_head = np.asarray(history["best_val_acc_per_head"], dtype=float)
            self.min_per_head = np.asarray(history["min_val_acc_per_head"], dtype=float)
            self.sum_per_head = np.asarray(history["mean_val_acc_per_head"], dtype=float) * epochs_done
            self.best_epoch_per_head = np.asarray(history["best_epoch_per_head"], dtype=int)

    def update(self, epoch, per_head):
        """Record an epoch's per-head MAE; True when it becomes the selected epoch."""
        per_head = np.asarray(per_head, dtype=float)
        scalar = float(per_head.min())
        self.count, self.total = self.count + 1, self.total + scalar
        is_best = scalar < self.best
        if is_best:
            self.best, self.best_epoch = scalar, epoch
        if self.best_per_head is None:
            self.best_per_head, self.sum_per_head = per_head.copy(), per_head.copy()
            self.min_per_head = per_head.copy()
            self.best_epoch_per_head = np.full(per_head.shape, epoch, dtype=int)
        else:
            self.best_epoch_per_head[per_head < self.best_per_head] = epoch
            self.best_per_head = np.minimum(self.best_per_head, per_head)
            self.sum_per_head = self.sum_per_head + per_head
            self.min_per_head = np.minimum(self.min_per_head, per_head)
        return is_best

    def record(self, per_head):
        return {
            "mean_val_acc": self.total / self.count,
            "best_val_acc": self.best,
            "val_acc_per_head": np.asarray(per_head, dtype=float).tolist(),
            "best_val_acc_per_head": self.best_per_head.tolist(),
            "mean_val_acc_per_head": (self.sum_per_head / self.count).tolist(),
            "min_val_acc_per_head": self.min_per_head.tolist(),
            "best_epoch_per_head": self.best_epoch_per_head.tolist(),
            "best_epoch": self.best_epoch,
        }


def read_clip_settings(args_data):
    """How each video is read, from `experiment.data`; also the fingerprint's `clips`. `fps` (it replaces
    `frame_step`) and `fractional_frame_step` (docs/frame_sampling.md) appear only when set, so the settings, and
    the fingerprint, of a probe run without them stay as they were."""
    settings = {
        "resolution": args_data.get("resolution", 224),
        "frames_per_clip": args_data.get("frames_per_clip", 16),
        "frame_step": args_data.get("frame_step", 4),
        "num_segments": args_data.get("num_segments", 1),
        "normalization": args_data.get("normalization", None),
        "temporal_padding": args_data.get("temporal_padding", None),
    }
    if args_data.get("fps") is not None:
        settings.update(fps=args_data["fps"], frame_step=None)
    if args_data.get("fractional_frame_step", False):
        settings["fractional_frame_step"] = True
    return settings


def make_dataset(path, training, resolution, frames_per_clip, frame_step, num_segments, normalization=None,
                 temporal_padding=None, fps=None, fractional_frame_step=False):
    """A split's studies, each video read and augmented as the EF probe's `make_dataloader` reads one:
    random clip positions and augmentation in training, fixed positions and no augmentation otherwise."""
    transform = make_transforms(
        training=training, num_views_per_clip=1, random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3), random_resize_scale=(0.08, 1.0), reprob=0.25, auto_augment=True,
        motion_shift=False, crop_size=resolution,
        normalize=normalization or ef.DEFAULT_NORMALIZATION,
    )
    sampling = PROTOCOL["train" if training else "evaluation"]
    return VideoGroupDataset(path, frames_per_clip=frames_per_clip, frame_step=frame_step, num_segments=num_segments,
                             transform=transform, random_clip_sampling=sampling["clip_positions"] == "random",
                             allow_clip_overlap=True, temporal_padding=temporal_padding, fps=fps,
                             fractional_frame_step=fractional_frame_step)


def make_train_loader(dataset, batch_size, world_size, rank, num_workers, loader_timeout=DEFAULT_LOADER_TIMEOUT):
    """Shuffled studies, each GPU's last incomplete batch dropped (the EF probe's training loader)."""
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                                              shuffle=True)
    kwargs = dict(dataset=dataset, sampler=sampler, batch_size=batch_size, drop_last=PROTOCOL["train"]["drop_last"],
                  collate_fn=collate_studies, pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                  worker_init_fn=_worker_init_fn, timeout=dataloader_timeout(loader_timeout, num_workers))
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(**kwargs), sampler


def load_probe_info(train_manifest):
    """The `probe_info.json` written with the manifests, next to the train manifest: it holds the
    normalization, the slot specification and the checksum of the test manifest."""
    path = os.path.join(os.path.dirname(train_manifest), "probe_info.json")
    if not os.path.exists(path):
        raise ValueError("The multi-view probe needs the probe_info.json written with the manifests next to the "
                         "train manifest: it fixes the slots and the test manifest before training.")
    with open(path) as f:
        return path, json.load(f)


def check_layout(info, *datasets):
    """Every split's manifest has the slot positions `probe_info.json` records, in its order."""
    for dataset in datasets:
        if list(dataset.position_names) != list(info.get("positions") or []):
            raise ValueError(f"The slot positions {dataset.position_names} differ from probe_info.json's "
                             f"{info.get('positions')}.")


def check_manifest(index, dataset, split):
    """Every video of a manifest is in the video index, under `split`, and belongs to its row's study."""
    rows = [(study, path) for study, paths in zip(dataset.studies, dataset.paths)
            for path in paths if path is not None]
    owners = provenance.study_ids(index, [path for _, path in rows], split)
    wrong = sum(str(study) != str(owner) for (study, _), owner in zip(rows, owners))
    if wrong:
        raise ValueError(f"{wrong} videos of the {split} manifest belong to another study than their row's.")


def check_test_manifest(info, test_manifest):
    """The test manifest must be the `test.csv` written with the training manifests."""
    expected = ((info.get("outputs") or {}).get("test.csv") or {}).get("sha256")
    if expected is None or provenance.sha256(test_manifest) != expected:
        raise ValueError("The test manifest is not the test.csv written with the training manifests: its SHA-256 "
                         "differs from the one probe_info.json records.")


def _to_device(batch, device):
    clips = [c.to(device, non_blocking=True) for c in batch["clips"]]
    clip_indices = [c.to(device, non_blocking=True) for c in batch["clip_indices"]]
    return clips, clip_indices, batch["present"].to(device, non_blocking=True)


def train_one_epoch(device, encoder, classifiers, optimizer, scheduler, wd_scheduler, data_loader, use_bfloat16,
                    target_std):
    """One training epoch of every head: the frozen tokens are computed once per batch and shared.
    Returns the best head's MAE (mmHg), every head's MAE and every head's mean loss, each weighted by
    the number of studies of each batch and summed over ranks."""
    from torch.amp import autocast

    for c in classifiers:
        c.train(True)
    criterion = torch.nn.SmoothL1Loss()  # Huber, on the z-scored target, as the EF probe.
    mae_meters = [AverageMeter() for _ in classifiers]
    loss_meters = [AverageMeter() for _ in classifiers]
    for itr, batch in enumerate(data_loader):
        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            clips, clip_indices, present = _to_device(batch, device)
            tokens = encoder(clips, clip_indices, present)
            outputs = [c(tokens, present) for c in classifiers]
        labels = batch["labels"].to(device).float()
        losses = [criterion(o.float(), labels) for o in outputs]
        ef._raise_if_non_finite_probe_loss([[loss] for loss in losses], device)
        [s.step() for s in scheduler]
        [wds.step() for wds in wd_scheduler]
        with torch.no_grad():
            count = float(labels.numel())
            scores = [(o.float() - labels).abs().sum() for o in outputs]
            loss_sums = [loss.detach().float() * count for loss in losses]
            totals = AllReduceSum.apply(torch.stack([torch.tensor(count, device=device), *scores, *loss_sums]))
            totals = totals.cpu().tolist()
            n, scores, loss_sums = totals[0], totals[1:1 + len(outputs)], totals[1 + len(outputs):]
            for meter, total in zip(mae_meters, scores):
                meter.update(total / n * target_std, n=n)
            for meter, total in zip(loss_meters, loss_sums):
                meter.update(total / n, n=n)
        for loss in losses:
            loss.backward()
        [o.step() for o in optimizer]
        [o.zero_grad() for o in optimizer]
        if itr % 10 == 0:
            logger.info("[%5d] train MAE %.3f (best head)" % (itr, min(m.avg for m in mae_meters)))
    maes = np.array([m.avg for m in mae_meters])
    return float(maes.min()), maes, np.array([m.avg for m in loss_meters])


class _EachStudyOnce(torch.utils.data.Dataset):
    """Some studies of a `VideoGroupDataset`, each read once and returned with its row; a study none
    of whose videos can be read is returned as `None`, to be counted rather than replaced."""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        try:
            return index, self.dataset.get_item_study(index)
        except Exception:
            return index, None


def _collate_scored(batch):
    loaded = [(index, item) for index, item in batch if item is not None]
    failed = [index for index, item in batch if item is None]
    if not loaded:
        return [], None, failed
    return [index for index, _ in loaded], collate_studies([item for _, item in loaded]), failed


def score_studies(device, encoder, heads, dataset, batch_size, num_workers, use_bfloat16, rank, world_size,
                  loader_timeout=DEFAULT_LOADER_TIMEOUT):
    """Score every study of `dataset` exactly once with each head, as the EF probe's `score_videos`
    scores videos: each rank takes every `world_size`-th study, and the results are gathered so every
    rank holds all of them, in dataset order.

    Returns (rows, labels [N, 1], predictions [heads, N, 1], videos read per study [N], failed rows,
    failed videos).
    """
    from torch.amp import autocast

    modules = [getattr(h, "module", h) for h in heads]
    for m in modules:
        m.eval()
    loader = torch.utils.data.DataLoader(
        _EachStudyOnce(dataset, range(rank, len(dataset), world_size)), batch_size=batch_size,
        num_workers=num_workers, collate_fn=_collate_scored, pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn, timeout=dataloader_timeout(loader_timeout, num_workers))
    indices, labels, predictions, videos, failed, failed_videos = [], [], [[] for _ in modules], [], [], 0
    for batch_indices, batch, batch_failed in loader:
        failed += batch_failed
        if batch is None:
            continue
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            clips, clip_indices, present = _to_device(batch, device)
            tokens = encoder(clips, clip_indices, present)
            outputs = [m(tokens, present) for m in modules]
        for head_predictions, output in zip(predictions, outputs):
            head_predictions += output.float().cpu().tolist()
        indices += batch_indices
        labels += batch["labels"].tolist()
        videos += batch["present"].sum(1).tolist()
        failed_videos += batch["failed_videos"]

    local = (indices, labels, predictions, videos, failed, failed_videos)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    indices = [i for g in gathered for i in g[0]]
    failed = sorted(i for g in gathered for i in g[4])
    if len(set(indices)) != len(indices) or len(indices) + len(failed) != len(dataset):
        raise RuntimeError(f"Scored {len(indices)} and failed {len(failed)} of {len(dataset)} studies: every study "
                           "should be scored or failed exactly once.")
    if not indices:
        raise RuntimeError(f"None of the {len(dataset)} studies could be read.")
    order = np.argsort(indices)
    n = len(order)
    labels = np.asarray([x for g in gathered for x in g[1]], dtype=float).reshape(n, -1)[order]
    predictions = np.stack([np.asarray([x for g in gathered for x in g[2][h]], dtype=float).reshape(n, -1)[order]
                            for h in range(len(modules))])
    videos = np.asarray([x for g in gathered for x in g[3]], dtype=int)[order]
    return np.asarray(indices)[order], labels, predictions, videos, failed, sum(g[5] for g in gathered)


def validate(device, encoder, classifiers, dataset, batch_size, num_workers, use_bfloat16, target_mean, target_std,
             rank, world_size, loader_timeout=DEFAULT_LOADER_TIMEOUT):
    """Every head's per-study validation MAE (mmHg; what selection uses) and loss (the training loss
    on the z-scored target), with the numbers of studies scored and failed and of failed videos."""
    indices, labels, predictions, _, failed, failed_videos = score_studies(
        device, encoder, classifiers, dataset, batch_size, num_workers, use_bfloat16, rank, world_size,
        loader_timeout=loader_timeout)
    y = labels * target_std + target_mean
    return {
        "study_mae": np.asarray([float(np.abs(p_z * target_std + target_mean - y).mean()) for p_z in predictions]),
        "loss": np.asarray([probe_metrics.smooth_l1(labels, p_z) for p_z in predictions]),
        "studies": len(indices),
        "failed_studies": len(failed),
        "failed_videos": failed_videos,
    }


def by_videos_present(labels, predictions, videos):
    """Error and bias by the number of videos read for a study."""
    frame = pd.DataFrame({"err": np.asarray(predictions) - np.asarray(labels), "videos": np.asarray(videos)})
    return [{"videos": int(k), "n": int(len(part)), "mae": float(part.err.abs().mean()),
             "bias": float(part.err.mean())} for k, part in frame.groupby("videos", sort=True)]


def load_selected_head(checkpoint_path, classifiers, fingerprint):
    """`best.pt`, checked against this run's fingerprint (the number of GPUs aside), and the head it selects:
    the lowest per-study validation MAE of its epoch, loaded into its classifier in eval mode.

    Returns (checkpoint, head index, classifier)."""
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    if fingerprint is not None:
        provenance.check_fingerprint(checkpoint.get("fingerprint"), fingerprint, checkpoint_path,
                                     ignore=("settings.world_size",))
    head = int(np.asarray(checkpoint["val_acc_per_head"], dtype=float).argmin())
    classifier = getattr(classifiers[head], "module", classifiers[head])
    classifier.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["classifiers"][head].items()})
    classifier.eval()
    return checkpoint, head, classifier


def score_split(device, encoder, classifier, dataset, batch_size, num_workers, use_bfloat16, target_mean, target_std,
                rank, world_size, loader_timeout=DEFAULT_LOADER_TIMEOUT):
    """Every study of `dataset` scored once by `classifier`: a table (study ID, reference and prediction in
    mmHg, absolute error, videos read), its regression metrics and counts, and the manifest rows that failed."""
    rows, labels, predictions, videos, failed, failed_videos = score_studies(
        device, encoder, [classifier], dataset, batch_size, num_workers, use_bfloat16, rank, world_size,
        loader_timeout=loader_timeout)
    y, p = labels[:, 0] * target_std + target_mean, predictions[0][:, 0] * target_std + target_mean
    table = pd.DataFrame({"study_id": [dataset.studies[i] for i in rows], "label": y, "prediction": p,
                          "abs_error": np.abs(p - y), "videos": videos})
    block = {**probe_metrics.regression(y, p), "studies": len(rows), "failed_studies": len(failed),
             "failed_videos": failed_videos}
    return table, block, failed


def score_selected_validation(device, encoder, classifiers, dataset, batch_size, num_workers, checkpoint_path,
                              use_bfloat16, target_mean, target_std, head_names, folder, rank, world_size, ranges,
                              aggregation, fingerprint=None, tolerance=DEFAULT_SCORE_TOLERANCE,
                              loader_timeout=DEFAULT_LOADER_TIMEOUT):
    """Score the validation split once with the epoch (`best.pt`) and head chosen on validation, and nothing
    else: what a comparison that must stay on validation needs (encoders compared by per-study validation MAE
    with a paired bootstrap), without the test split. The head is selected and scored exactly as `run_test`
    does. Training scored the same studies with the same function, so the recomputed MAE must equal the one
    `best.pt` recorded for that head within `tolerance` mmHg; otherwise nothing is written.

    Writes `val_selected_predictions.csv` (study IDs: it stays with the data) and, last, `val_selected_metrics.json`.
    Returns the metrics on rank 0 and `None` on the others.
    """
    if not tolerance >= 0:  # NaN too
        raise ValueError(f"The tolerance is a non-negative number of mmHg, not {tolerance}.")
    checkpoint, head, classifier = load_selected_head(checkpoint_path, classifiers, fingerprint)
    logger.info(f"Scoring validation with {head_names[head]} from epoch {checkpoint['epoch']}")
    table, block, failed = score_split(device, encoder, classifier, dataset, batch_size, num_workers, use_bfloat16,
                                       target_mean, target_std, rank, world_size, loader_timeout=loader_timeout)
    # Every rank holds the gathered scores, so all ranks stop together.
    stored = float(np.asarray(checkpoint["val_acc_per_head"], dtype=float)[head])
    difference = block["mae"] - stored
    if not abs(difference) <= tolerance:
        raise RuntimeError(
            f"The recomputed validation MAE of {head_names[head]} ({block['mae']:.6g} mmHg) differs from the "
            f"{stored:.6g} `best.pt` recorded by more than {tolerance} mmHg. Check that no validation video failed "
            "to load now that loaded during training, and score with the number of GPUs the probe was trained with; "
            "pass a larger --tolerance only once the difference is understood.")
    if rank != 0:
        return None

    commit, dirty = provenance.git_state()
    result = {
        "split": "val",
        "selected_on": "validation",
        "selection": checkpoint.get("selection", SELECTION),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": provenance.sha256(checkpoint_path),
        "encoder_sha256": None if fingerprint is None else fingerprint.get("encoder_sha256"),
        "epoch": int(checkpoint["epoch"]),
        "head": head,
        "head_name": head_names[head],
        "val_study_mae": stored,
        "unit": "mmHg",
        "aggregation": aggregation,
        "stored_score_check": {"stored": stored, "recomputed": float(block["mae"]), "difference": float(difference),
                               "tolerance": float(tolerance)},
        "val": block,
        "val_by_reference_range": {"level": "study", "rows": probe_metrics.by_range(table.label, table.prediction,
                                                                                      ranges)},
        "val_by_videos_present": by_videos_present(table.label, table.prediction, table.videos),
        "failed_rows": failed,  # rows of the validation manifest, not identifiers.
        "world_size": {"trained": checkpoint.get("world_size"), "scored": world_size},
        "git_commit": commit,
        "git_dirty": dirty,
    }
    def dump(path):
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    for name, write in ((VAL_PREDICTIONS, lambda path: table.to_csv(path, index=False)), (VAL_METRICS, dump)):
        tmp = os.path.join(folder, f"{name}.tmp")
        write(tmp)
        os.replace(tmp, os.path.join(folder, name))  # the metrics last: their presence means the scoring finished
    logger.info(f"Validation of {head_names[head]} (epoch {result['epoch']}): study MAE {block['mae']:.4f} mmHg over "
                f"{block['studies']} studies, {block['failed_studies']} failed; written to {folder}")
    return result


def run_test(device, encoder, classifiers, dataset, val_dataset, batch_size, num_workers, checkpoint_path,
             use_bfloat16, target_mean, target_std, head_names, folder, rank, world_size, ranges, targets, aggregation,
             run_info=None, wandb_run=None, step=0, fingerprint=None, loader_timeout=DEFAULT_LOADER_TIMEOUT):
    """Score the test split once with the epoch (`best.pt`) and head chosen on validation.

    `best.pt` must carry the fingerprint of this run, the number of GPUs aside. The validation
    studies are scored again with the selected head, for the record. Writes `val_predictions.csv`,
    `test_predictions.csv` (study IDs: they stay with the data) and, last and atomically,
    `test_metrics.json`. Returns the metrics on rank 0 and `None` on the others.
    """
    checkpoint, head, classifier = load_selected_head(checkpoint_path, classifiers, fingerprint)
    val_per_head = np.asarray(checkpoint["val_acc_per_head"], dtype=float)
    logger.info(f"Testing {head_names[head]} from epoch {checkpoint['epoch']}")

    def score(split_dataset):
        return score_split(device, encoder, classifier, split_dataset, batch_size, num_workers, use_bfloat16,
                           target_mean, target_std, rank, world_size, loader_timeout=loader_timeout)

    val_table, val_block, _ = score(val_dataset)
    test_table, test_block, failed = score(dataset)
    if rank != 0:
        return None

    by_range = probe_metrics.by_range(test_table.label, test_table.prediction, ranges)
    met = {"mae": bool(test_block["mae"] < targets["mae_below"])} if "mae_below" in targets else {}
    result = {
        "selected_on": "validation",
        "selection": checkpoint.get("selection", SELECTION),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": provenance.sha256(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "head": head,
        "head_name": head_names[head],
        "val_study_mae": float(val_per_head[head]),
        "unit": "mmHg",
        "aggregation": aggregation,
        "val": val_block,
        "test": test_block,
        "test_by_reference_range": {"level": "study", "rows": by_range},
        "test_by_videos_present": by_videos_present(test_table.label, test_table.prediction, test_table.videos),
        "targets": {**targets, "level": "study", "met": met},
        "run_info": run_info,
        "failed_rows": failed,  # rows of the test manifest, not identifiers.
    }
    val_table.to_csv(os.path.join(folder, "val_predictions.csv"), index=False)
    test_table.to_csv(os.path.join(folder, "test_predictions.csv"), index=False)
    tmp = os.path.join(folder, "test_metrics.json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, os.path.join(folder, "test_metrics.json"))

    scalars = {"probe/test_epoch": result["epoch"], "probe/test_head": head,
               "probe/test_val_study_mae": result["val_study_mae"], **ef._scalars("probe/test/study", test_block),
               **{f"probe/test/target_met_{k}": float(v) for k, v in met.items()}}
    log_scalars(wandb_run, scalars, step=step)
    log_test_distribution(wandb_run, test_table.label, test_table.prediction, by_range, step, "probe/test/study")
    log_reference(wandb_run, "probe_checkpoint", "probe-checkpoint",
                  ef.checkpoint_reference(checkpoint_path, "regression", head_names, fingerprint,
                                          sha=result["checkpoint_sha256"]))
    logger.info(f"Test of {head_names[head]} (epoch {result['epoch']}): {json.dumps(result['test'])}")
    return result


def log_test_distribution(run, labels, predictions, ranges, step, prefix):
    """The residual histogram and the error by reference RVSP range, without identifiers. The EF
    probe's scatter image is not logged: its axes are labelled in EF points."""
    wandb = wandb_logging.wandb
    if run is None or wandb is None:
        return
    try:
        payload = {f"{prefix}/residuals": wandb.Histogram(np.asarray(predictions) - np.asarray(labels)),
                   wandb_logging.STEP_METRIC: step}
        for row in ranges:
            for stat in ("n", "mae", "bias"):
                if np.isfinite(row[stat]):
                    payload[f"{prefix}/by_reference_range/{wandb_logging.range_key(row['range'])}/{stat}"] = row[stat]
        run.log(payload)
    except Exception as e:
        logger.warning(f"Failed to log the test results to wandb: {e}")


def describe_run(args_eval, checkpoint, manifests, fingerprint, video_index, info, train_set, val_set, seed,
                 parameter_counts, opt_kwargs, batch_size, world_size, num_epochs, fusion, ranges, targets):
    """The probe run without filesystem paths: code version, encoder and manifest identities, the
    test manifest's checksum fixed before training, slots, split sizes, settings and rules."""
    commit, dirty = provenance.git_state()
    rows = {"train": len(train_set), "val": len(val_set)}
    files = {name: {"file": os.path.basename(path), "sha256": fingerprint["manifests_sha256"][name],
                    **({"studies": rows[name]} if name in rows else {})}
             for name, path in manifests.items()}
    meta = args_eval.get("meta") or {}
    return {
        "experiment": meta.get("wandb_run_name") or args_eval.get("tag"),
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "encoder_checkpoint": {"file": os.path.basename(checkpoint), "sha256": fingerprint["encoder_sha256"]},
        "normalization": fingerprint["normalization"],
        "unit": "mmHg",
        "manifests": files,
        "test_manifest_sha256": ((info.get("outputs") or {}).get("test.csv") or {}).get("sha256"),
        "slots": info.get("slots"),
        "positions": list(train_set.position_names),
        "slot_presence": {split: block.get("slots") for split, block in (info.get("splits") or {}).items()},
        "splits": provenance.split_counts(video_index),
        "parameters": parameter_counts,
        "optimization": {
            "optimizer": "AdamW",
            "lr_schedule": "linear warmup from start_lr, then cosine to final_lr",
            "weight_decay_schedule": "cosine from weight_decay to final_weight_decay",
            "epochs": num_epochs,
            "batch_size_per_gpu": batch_size,
            "world_size": world_size,
            "global_batch_size": batch_size * world_size,
            "heads": opt_kwargs,
        },
        "fusion": fusion,
        "aggregation": AGGREGATION[fusion],
        "protocol": PROTOCOL,
        "selection": SELECTION,
        "rvsp_ranges": ranges,
        "targets": targets,
    }
