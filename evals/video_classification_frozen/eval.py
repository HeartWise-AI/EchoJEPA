# evals/video_classification_frozen/eval.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import json
import logging
import math
import pprint
import random
import shutil

import numpy as np
import pandas as pd
import torch
import yaml
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from evals.video_classification_frozen import metrics as probe_metrics
from evals.video_classification_frozen import provenance
from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.models.attentive_pooler import AttentiveRegressor
from src.models.linear_pooler import LinearClassifier, LinearRegressor
from src.models.linear_pooler import MLPClassifier, MLPRegressor

from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduceSum, any_rank_failed, init_distributed
from src.utils.logging import AverageMeter, CSVLogger
from src.utils.wandb_logging import finish_wandb, init_wandb, log_reference, log_regression_results, log_scalars

import os
import tempfile  # <-- ADD THIS

# Fix for "AF_UNIX path too long" error
short_tmp = "/tmp/vjepa_run"
os.makedirs(short_tmp, exist_ok=True)
tempfile.tempdir = short_tmp
os.environ["TMPDIR"] = short_tmp

# from evals.action_anticipation_frozen.losses import sigmoid_focal_loss

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

# Which probe is considered "best," how multiple predictions are averaged, and how a continuous EF prediction
# is turned into a reduced-EF classification.
# `experiment.evaluation.selection` sets the level `best.pt` is selected at.
SELECTION = {
    "study": ("Select the epoch and probe head with the lowest per-study validation MAE (a study's prediction "
              "is the mean of its videos'); after each epoch, every validation video is evaluated once by "
              "every head."),
    "video": ("Select the epoch and probe head with the lowest per-video validation MAE; "
              "after each epoch, every validation video is evaluated once by every head."),
}
AGGREGATION = ("Video prediction: the encoder embeds each of the video's num_segments clips, their tokens are "
               "concatenated along time, and the attentive probe pools them into one prediction (averaged over "
               "num_views_per_segment spatial views); study prediction: mean over all videos in the study.")
THRESHOLD_RULE = ("predicted EF below the threshold is classified as reduced EF; the threshold is "
                  "chosen on the validation set to maximize sensitivity + specificity - 1, using "
                  "study-level predictions for study-level targets and video-level predictions for "
                  "video-level targets; ties are resolved by choosing the threshold closest to the EF cutoff.")
# How the clips of each split are drawn and batched (`make_dataloader`). The fingerprint holds it, so a
# probe is continued and tested only under the protocol it was trained with. Raise the version whenever
# the code changes how the probe is trained or evaluated in a way these fields do not describe.
# Version 1, the code before this field, also drew random clip positions in evaluation.
PROTOCOL = {
    "version": 2,
    "train": {"clip_positions": "random", "drop_last": True},
    "evaluation": {"clip_positions": "fixed", "drop_last": False},
}

# --- INSERT THIS CLASS IN eval.py ---
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits
        # targets: [B] class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ------------------------------------

def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    import os
    
    # Helper to safe-set nested keys with type conversion
    def set_override(env_var, target_dict, key, type_func=str):
        val = os.environ.get(env_var)
        if val is not None:
            # Handle boolean explicitly
            if type_func == bool:
                val = val.lower() in ('true', '1', 't', 'yes')
            else:
                val = type_func(val)
            print(f"!!! MANUAL OVERRIDE: {key} -> {val}")
            target_dict[key] = val

    # 1. Top-level parameters
    set_override("OVERRIDE_TAG", args_eval, "tag")
    set_override("OVERRIDE_VAL_ONLY", args_eval, "val_only", bool)
    set_override("OVERRIDE_TEST_ONLY", args_eval, "test_only", bool)
    set_override("OVERRIDE_PRED_PATH", args_eval, "predictions_save_path")
    set_override("OVERRIDE_CKPT", args_eval, "probe_checkpoint")

    # Ensure nested dictionaries exist
    exp = args_eval.setdefault("experiment", {})
    clf = exp.setdefault("classifier", {})
    data = exp.setdefault("data", {})
    opt = exp.setdefault("optimization", {})

    # 2. Classifier parameters
    set_override("OVERRIDE_NUM_HEADS", clf, "num_heads", int)
    set_override("OVERRIDE_NUM_BLOCKS", clf, "num_probe_blocks", int)

    # 3. Data parameters
    set_override("OVERRIDE_TRAIN_DATA", data, "dataset_train")
    set_override("OVERRIDE_VAL_DATA", data, "dataset_val")
    set_override("OVERRIDE_TEST_DATA", data, "dataset_test")
    set_override("OVERRIDE_NUM_CLASSES", data, "num_classes", int)
    set_override("OVERRIDE_RES", data, "resolution", int)

    # --- NEW: Override Mean/Std for regression ---
    set_override("OVERRIDE_TARGET_MEAN", data, "target_mean", float)
    set_override("OVERRIDE_TARGET_STD", data, "target_std", float)

    # 4. Optimization parameters
    set_override("OVERRIDE_EPOCHS", opt, "num_epochs", int)
    set_override("OVERRIDE_FOCAL_LOSS", opt, "use_focal_loss", bool)
    set_override("OVERRIDE_BATCH", opt, "batch_size", int)  # <--- ADD THIS LINE
    # --- INSERT END ---

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")
    predictions_save_path = args_eval.get("predictions_save_path", None)

    # -- TEST ONLY: score `dataset_test` once with the epoch and head chosen on validation.
    test_only = args_eval.get("test_only", False)
    if test_only and val_only:
        raise ValueError("Choose one of `val_only` and `test_only`.")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")
    # Copy the encoder construction settings (checkpoint weights and options) before
    # model initialization can modify them; these settings form part of the probe fingerprint.
    encoder_settings = copy.deepcopy(
        {"module_name": module_name, "pretrain_kwargs": args_model, "wrapper_kwargs": args_wrapper})

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    probe_checkpoint = args_eval.get("probe_checkpoint", None)

    # -- REGRESSION
    task_type = args_classifier.get("task_type", "classification")  # "classification" or "regression"  
    num_targets = args_classifier.get("num_targets", None)  # Only for regression
    probe_type = args_classifier.get("probe_type", "attentive")  # "attentive", "linear", or "mlp"
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    test_data_path = args_data.get("dataset_test")
    if test_only and not test_data_path:
        raise ValueError("`test_only` needs `experiment.data.dataset_test`.")
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    
    # --- NEW: Get Mean/Std from config ---
    target_mean = args_data.get("target_mean", None)
    target_std = args_data.get("target_std", None)
    if task_type == "regression":
        check_normalization(target_mean, target_std, args_data.get("dataset_train"))
    # Maps each video to its study and patient (`data/make_probe_manifests.py`): needed for
    # study-level metrics and the patient-leakage check.
    video_index_path = args_data.get("video_index", None)

    # -- EVALUATION (regression): reduced EF is a reference EF below `low_ef_below`.
    args_evaluation = args_exp.get("evaluation") or {}
    low_ef_below = float(args_evaluation.get("low_ef_below", 40.0))
    ef_ranges = list(args_evaluation.get("ef_ranges", [30, 40, 50, 60]))
    targets = dict(args_evaluation.get("targets") or {})
    # Use study-level results and targets when `video_index` provides study IDs; otherwise use video-level ones.
    selection = args_evaluation.get("selection") or (
        "study" if video_index_path and task_type == "regression" else "video")
    if selection not in ("study", "video"):
        raise ValueError(f"`experiment.evaluation.selection` is `study` or `video`, not {selection!r}.")
    if selection == "study" and (task_type != "regression" or not video_index_path):
        raise ValueError("Per-study selection is for regression and needs `experiment.data.video_index`.")
    # Validation metric used to select `best.pt`, as reported in logs, wandb, and results.
    # `val_only` evaluates the batch loop's per-video metric and does not save a checkpoint.
    if task_type != "regression":
        selected = "acc"
    else:
        selected = "study_mae" if selection == "study" and not val_only else "mae"

    # -- SEED: probe initialization, data order and augmentation.
    seed = int((args_eval.get("meta") or {}).get("seed", _GLOBAL_SEED))

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    use_focal_loss = args_opt.get("use_focal_loss", False)

    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

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

    # -- log/checkpointing paths  
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")  
    if eval_tag is not None:  
        folder = os.path.join(folder, eval_tag)  
    if not os.path.exists(folder):  
        os.makedirs(folder, exist_ok=True)  
    log_file = os.path.join(folder, f"log_r{rank}.csv")  
      
    # Use custom probe checkpoint if specified, otherwise use default  
    if probe_checkpoint is not None:  
        latest_path = probe_checkpoint  
    else:  
        latest_path = os.path.join(folder, "latest.pt")

    # -- make `csv_logger` (a test trains nothing, so it leaves the probe's log alone).
    if rank == 0 and not test_only:
        if task_type == "regression":  
            csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train_mae"), ("%.5f", f"val_{selected}"))
        else:  # classification  
            csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train_acc"), ("%.5f", "val_acc"))

    # Seed all randomness before probe initialization, data ordering, and augmentation.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Study-level targets require the video index, so reject invalid configurations
    # before training rather than failing later during evaluation.
    if targets.get("level", "study") not in ("study", "video"):
        raise ValueError(f"`experiment.evaluation.targets.level` is `study` or `video`, not {targets['level']!r}.")
    if targets.get("level", "study") == "study" and targets and not video_index_path:
        raise ValueError("`experiment.evaluation.targets` are per study: set `experiment.data.video_index`.")
    video_index = provenance.load_video_index(video_index_path) if video_index_path else None

    # Initialize model

    # -- init models
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # -- init classifier based on probe_type
    def make_probe(task, probe, embed_dim, num_classes, num_targets, num_heads, depth, use_ln, dropout):
        if task == "regression":
            if probe == "linear":
                return LinearRegressor(
                    embed_dim=embed_dim,
                    num_targets=num_targets,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            elif probe == "mlp":
                return MLPRegressor(
                    embed_dim=embed_dim,
                    num_targets=num_targets,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            else:  # attentive (default)
                return AttentiveRegressor(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    depth=depth,
                    num_targets=num_targets,
                    use_activation_checkpointing=True,
                )
        else:  # classification
            if probe == "linear":
                return LinearClassifier(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            elif probe == "mlp":
                return MLPClassifier(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            else:  # attentive (default)
                return AttentiveClassifier(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    depth=depth,
                    num_classes=num_classes,
                    use_activation_checkpointing=True,
                )

    classifiers = [
        make_probe(
            task=task_type,
            probe=probe_type,
            embed_dim=encoder.embed_dim,
            num_classes=num_classes,
            num_targets=num_targets,
            num_heads=num_heads,
            depth=num_probe_blocks,
            use_ln=use_layernorm,
            dropout=probe_dropout,
        ).to(device)
        for _ in opt_kwargs
    ]
    
    logger.info(f"Initialized {len(classifiers)} {probe_type} probes for {task_type}")

    
    # classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers
    # Add distributed check to avoid DDP crash when running concurrent jobs  
    from torch import distributed as dist  
    use_ddp = dist.is_available() and dist.is_initialized() and world_size > 1  
    if use_ddp:  
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]  
    else:  
        logger.info(f"DDP disabled (world_size={world_size}); running single-process.")
        
    print(classifiers[0])

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")
    if ipe == 0 and not (test_only or val_only):
        raise ValueError(f"The training manifest gives no full batch of {batch_size} on each of {world_size} GPUs: "
                         "nothing to train on.")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # Train only the probe; keep the encoder frozen and exclude its parameters from the optimizers.
    if any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("The encoder must be frozen: some of its parameters require gradients.")
    frozen = {id(p) for p in encoder.parameters()}
    if any(id(p) in frozen for o in optimizer for g in o.param_groups for p in g["params"]):
        raise RuntimeError("An optimizer holds encoder parameters: only the probe may be trained.")
    per_head = sum(p.numel() for p in classifiers[0].parameters() if p.requires_grad)
    parameter_counts = {
        "encoder_frozen": sum(p.numel() for p in encoder.parameters()),
        "probe_heads": len(classifiers),
        "probe_trainable_per_head": per_head,
        "probe_trainable_total": per_head * len(classifiers),
    }

    # Each manifest must contain only videos from its assigned split to prevent data leakage between
    # training, validation, and test sets.
    if video_index is not None:
        check_split(video_index, val_loader.dataset, "val")
        if not test_only and not val_only:
            check_split(video_index, train_loader.dataset, "train")

    # Fingerprint the probe configuration (encoder checkpoint, manifests, normalization, and settings)
    # and require resumed or test runs to match it.
    settings = {
        "encoder": encoder_settings,
        "classifier": dict(args_classifier),
        "clips": {"dataset_type": dataset_type, "resolution": resolution, "frames_per_clip": frames_per_clip,
                  "frame_step": frame_step, "clip_duration": duration, "num_segments": num_segments,
                  "num_views_per_segment": num_views_per_segment, "normalization": normalization},
        "optimization": {"batch_size": batch_size, "num_epochs": num_epochs, "use_bfloat16": use_bfloat16,
                         "use_focal_loss": use_focal_loss, "multihead_kwargs": args_opt.get("multihead_kwargs")},
        "evaluation": {"low_ef_below": low_ef_below, "ef_ranges": list(ef_ranges),
                       "targets": copy.deepcopy(targets)},
        "selection": selection,
        "protocol": copy.deepcopy(PROTOCOL),
        "seed": seed,
        "world_size": world_size,
    }
    manifests = {"train": args_data.get("dataset_train"), "val": args_data.get("dataset_val"),
                 "test": test_data_path, "video_index": video_index_path}
    fingerprint = shared_fingerprint(rank, world_size, device, checkpoint, manifests,
                                     {"target_mean": target_mean, "target_std": target_std}, settings)

    # -- load training checkpoint
    start_epoch = 0
    history = {}
    if resume_checkpoint and os.path.exists(latest_path) and not test_only:
        classifiers, optimizer, scaler, start_epoch, history = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        if not val_only:
            provenance.check_fingerprint(history.get("fingerprint"), fingerprint, f"The probe in {folder}")
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    # wandb, as in pretraining: on when `meta.wandb_project` and `meta.wandb_entity` are set. A
    # resumed probe reattaches to its run through the `wandb_run_id.txt` kept in its folder.
    # Every rank votes, so a failure on one rank stops the whole job.
    # Record run provenance on rank 0, including the code, encoder, manifests, settings, and
    # encoder-checkpoint hash; save it with the probe and log it to wandb without local paths.
    run_info = None
    if rank == 0:
        run_info = describe_run(
            args_eval, checkpoint, manifests, fingerprint, video_index, seed, parameter_counts, opt_kwargs,
            batch_size, world_size, num_epochs, low_ef_below, targets, selection,
        )
        if not test_only:
            save_run_record(folder, args_eval, run_info, fresh=start_epoch == 0)

    wandb_run, wandb_init_error = None, None
    try:
        # Add test results to the wandb run of the probe being evaluated.
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
    #Name each probe head using its optimizer setting (learning rate, weight decay).
    head_names = [f"head{i}_lr{k['ref_lr']:g}_wd{k['ref_wd']:g}" for i, k in enumerate(opt_kwargs)]

    if test_only:
        # Only a finished probe is tested, with the epoch its training selected.
        check_finished(folder, num_epochs)
        # Score the test set only once to prevent test results from influencing model selection.
        if os.path.exists(os.path.join(folder, "test_metrics.json")):
            raise RuntimeError("This probe has already been tested (`test_metrics.json` exists); repeat testing only "
                               "if the previous evaluation was invalid.")
        test_loader, _ = make_dataloader(
            dataset_type=dataset_type,
            root_path=[test_data_path],
            img_size=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_segments=num_segments,
            eval_duration=duration,
            num_views_per_segment=num_views_per_segment,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            training=False,
            num_workers=num_workers,
            normalization=normalization,
        )
        if video_index is not None:
            check_split(video_index, test_loader.dataset, "test")
        run_test(
            device=device,
            encoder=encoder,
            classifiers=classifiers,
            dataset=test_loader.dataset,
            val_dataset=val_loader.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint_path=os.path.join(folder, "best.pt"),
            task_type=task_type,
            use_bfloat16=use_bfloat16,
            target_mean=target_mean,
            target_std=target_std,
            head_names=head_names,
            folder=folder,
            rank=rank,
            world_size=world_size,
            video_index=video_index,
            low_ef_below=low_ef_below,
            ef_ranges=ef_ranges,
            targets=targets,
            run_info=run_info,
            wandb_run=wandb_run,
            step=num_epochs * ipe,
            fingerprint=fingerprint,
        )
        finish_wandb(wandb_run)
        return

    # ---- per-head running stats ----
    best_per_head = None
    sum_per_head = None
    min_per_head = None
    best_epoch_per_head = None
    count_epochs = 0

    def save_checkpoint(epoch, mean_val_acc, best_val_acc,
                        val_heads, best_per_head, mean_per_head, min_per_head, best_epoch_per_head,
                        best_epoch, is_best=False, val_scores=None):
        
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]
        rng_states = gather_rng_states(world_size)  # on every rank, before rank 0 saves.

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if (scaler is None) else [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "mean_val_acc": float(mean_val_acc),
            "best_val_acc": float(best_val_acc),
            "val_acc_per_head": np.asarray(val_heads, dtype=float).tolist(),
            "best_val_acc_per_head": np.asarray(best_per_head, dtype=float).tolist(),
            "mean_val_acc_per_head": np.asarray(mean_per_head, dtype=float).tolist(),
            "min_val_acc_per_head": np.asarray(min_per_head, dtype=float).tolist(),
            "best_epoch_per_head": np.asarray(best_epoch_per_head, dtype=int).tolist(),
            "best_epoch": best_epoch,    # the epoch saved as `best.pt`.
            "selection": SELECTION[selection],
            "selection_metric": selected,       # what `val_acc_per_head` holds.
            "val_scores_per_head": val_scores,  # every validation metric of the epoch, per head.
            "fingerprint": fingerprint,  # data and settings used to train the probe heads.
            "rng_states": rng_states,  # each rank's random number state, for an exact resume.
            "opt_grid": opt_kwargs,
        }
        
        _save_probe_checkpoints(save_dict, folder, is_best, rank, device)

    # ---- per-head running stats ----
    best_per_head = None
    sum_per_head = None
    min_per_head = None
    best_epoch_per_head = None
    count_epochs = 0
    
    # [FIX 3] Initialize Best Scalar based on Task
    if task_type == "regression":
        best_val_acc_scalar = float('inf')
    else:
        best_val_acc_scalar = 0.0

    # TRAIN LOOP
    val_cnt = 0
    val_sum_scalar = 0.0
    best_epoch = None

    # For regression with a `VideoDataset`, `validate()` scores every validation video once with
    # every probe head; other datasets, classification, and `val_only` use the standard batch loop.
    val_dataset = getattr(val_loader, "dataset", None)
    exact_val = task_type == "regression" and not val_only and hasattr(val_dataset, "get_item_video")
    val_studies = None
    if exact_val and video_index is not None:
        val_studies = provenance.study_ids(video_index, val_dataset.samples, "val")
    elif selection == "study" and not val_only:
        raise ValueError("Per-study selection needs a `VideoDataset` validation set, scored video by video.")
    elif not exact_val and not val_only:
        logger.warning("Validation uses the standard batch loop, so its metric may include duplicate videos "
                       "introduced by distributed-sampler padding.")

    # Restore validation history when resuming, so `best.pt` is updated only if the current epoch
    # outperforms every epoch seen before the interruption.
    if start_epoch > 0 and history.get("best_epoch_per_head") is not None:
        best_val_acc_scalar = float(history["best_val_acc"])
        val_cnt = count_epochs = start_epoch
        val_sum_scalar = float(history["mean_val_acc"]) * start_epoch
        best_per_head = np.asarray(history["best_val_acc_per_head"], dtype=float)
        min_per_head = np.asarray(history["min_val_acc_per_head"], dtype=float)
        sum_per_head = np.asarray(history["mean_val_acc_per_head"], dtype=float) * start_epoch
        best_epoch_per_head = np.asarray(history["best_epoch_per_head"], dtype=int)
        best_epoch = history.get("best_epoch")
        if rank == 0 and best_epoch is not None:
            restore_best(folder, best_epoch)

    # Continue each rank's random number streams where the interrupted run saved them, so the
    # resumed probe draws the same data order, augmentation and clip positions as an
    # uninterrupted one (the loader's workers are re-created, and re-seeded, every epoch).
    if start_epoch > 0 and not val_only:
        if history.get("rng_states"):
            set_rng_state(history["rng_states"][rank])
        else:
            logger.warning("The checkpoint has no random number state: the resumed run draws other data "
                           "orders and augmentations than an uninterrupted one.")

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)

        if val_only:
            train_acc_scalar, train_heads, train_losses = -1.0, None, None
        else:
            train_acc_scalar, train_heads, train_losses = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                use_focal_loss=use_focal_loss,
                val_only=val_only,
                predictions_save_path=predictions_save_path,
                task_type=task_type,
                target_mean=target_mean,
                target_std=target_std,
            )

        val = None
        if exact_val:
            val = validate(
                device=device,
                encoder=encoder,
                classifiers=classifiers,
                dataset=val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                use_bfloat16=use_bfloat16,
                target_mean=target_mean,
                target_std=target_std,
                rank=rank,
                world_size=world_size,
                studies=val_studies,
            )
            val_heads = val[selected]
            val_acc_scalar = float(val_heads.min())
        else:
            val_acc_scalar, val_heads, _ = run_one_epoch(
                device=device,
                training=False,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=val_loader,
                use_bfloat16=use_bfloat16,
                use_focal_loss=use_focal_loss,
                val_only=val_only,
                predictions_save_path=predictions_save_path,
                task_type=task_type,
                target_mean=target_mean,
                target_std=target_std,
            )

        # ---- update scalar running stats ----
        val_cnt += 1
        val_sum_scalar += float(val_acc_scalar)
        mean_val_acc_scalar = val_sum_scalar / val_cnt
        
        # --- NEW: Logic for determining "Best" ---
        is_best = False
        if task_type == "regression":
            # For Regression: LOWER is better
            # Note: Initialize best_val_acc_scalar to float('inf') before loop!
            if float(val_acc_scalar) < best_val_acc_scalar:
                best_val_acc_scalar = float(val_acc_scalar)
                is_best = True
        else:
            # For Classification: HIGHER is better
            if float(val_acc_scalar) > best_val_acc_scalar:
                best_val_acc_scalar = float(val_acc_scalar)
                is_best = True
        if is_best:
            best_epoch = epoch + 1

        # ---- update per-head running stats ----
        count_epochs += 1
        if best_per_head is None:
            best_per_head = val_heads.copy()
            sum_per_head = val_heads.copy()
            min_per_head = val_heads.copy()
            best_epoch_per_head = np.full_like(val_heads, epoch + 1, dtype=int)
        else:
            # For per-head, we need similar conditional logic for "improved"
            if task_type == "regression":
                improved = val_heads < best_per_head # Lower is better
                best_per_head = np.minimum(best_per_head, val_heads)
            else:
                improved = val_heads > best_per_head # Higher is better
                best_per_head = np.maximum(best_per_head, val_heads)
                
            best_epoch_per_head[improved] = epoch + 1
            sum_per_head += val_heads
            min_per_head = np.minimum(min_per_head, val_heads)
            
        mean_per_head = sum_per_head / count_epochs

        # Log appropriate metric name
        metric_label = "MAE" if task_type == "regression" else "Acc"
        symbol = "" if task_type == "regression" else "%"
        
        val_label = "val(min-head)" if task_type == "regression" else "val(max-head)"
        logger.info("[%5d] train: %.3f%s  %s: %.3f%s (Best: %.3f%s)" % (
            epoch + 1, train_acc_scalar, symbol,
            val_label, val_acc_scalar, symbol,
            best_val_acc_scalar, symbol
        ))

        
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc_scalar, val_acc_scalar)

        # Log the best-performing head and each individual head to wandb, using the current
        # optimizer-step count as the logging step.
        metric = "mae" if task_type == "regression" else "acc"
        payload = {
            "probe/epoch": epoch + 1,
            f"probe/val_{selected}_best": float(best_val_acc_scalar),
            **({"probe/best_epoch": best_epoch} if best_epoch is not None else {}),
            # End-of-epoch learning rate for each probe head.
            **{f"probe/lr/{h}": float(o.param_groups[0]["lr"]) for h, o in zip(head_names, optimizer)},
        }
        if train_heads is not None:
            payload[f"probe/train_{metric}"] = float(train_acc_scalar)
            payload.update({f"probe/train_{metric}/{h}": float(v) for h, v in zip(head_names, train_heads)})
            payload["probe/train_loss"] = float(np.min(train_losses))
            payload.update({f"probe/train_loss/{h}": float(v) for h, v in zip(head_names, train_losses)})
        if val is None:  # the batch loop: one metric per head.
            val_scores = {metric: val_heads}
        else:
            val_scores = {key: val[key] for key in ("mae", "study_mae", "loss") if val[key] is not None}
            payload["probe/val_failed_videos"] = val["failed_videos"]
        for key, per_head in val_scores.items():
            payload[f"probe/val_{key}"] = float(per_head.min())
            payload.update({f"probe/val_{key}/{h}": float(v) for h, v in zip(head_names, per_head)})
        log_scalars(wandb_run, payload, step=(epoch + 1) * ipe)

        if val_only:
            finish_wandb(wandb_run)
            return

        save_checkpoint(
            epoch + 1,
            mean_val_acc_scalar,
            best_val_acc_scalar,
            val_heads,
            best_per_head,
            mean_per_head,
            min_per_head,
            best_epoch_per_head,
            best_epoch,
            is_best=is_best,
            val_scores={key: np.asarray(v, dtype=float).tolist() for key, v in val_scores.items()},
        )

    # Reference the trained probe in wandb by the best checkpoint's checksum, epoch, and head,
    # without uploading the local checkpoint file.
    best_path = os.path.join(folder, "best.pt")
    if rank == 0 and not val_only and os.path.exists(best_path):
        log_reference(wandb_run, "probe_checkpoint", "probe-checkpoint",
                      checkpoint_reference(best_path, task_type, head_names, fingerprint))

    # If the probe fails with an exception, wandb closes the run automatically when the process exits.
    finish_wandb(wandb_run)


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    task_type="classification",  # "classification" or "regression"
    use_focal_loss=False,
    val_only=False,
    predictions_save_path=None,
    target_mean=None,
    target_std=None,
):
    # --- NEW: Import tqdm for progress bar ---
    from tqdm import tqdm
    
    for c in classifiers:
        c.train(mode=training)

    # Choose loss function based on task type
    if task_type == "regression":
        criterion = torch.nn.SmoothL1Loss()  # Huber Loss
        metric_meters = [AverageMeter() for _ in classifiers]
    else:  # classification
        if use_focal_loss:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        top1_meters = [AverageMeter() for _ in classifiers]
    # Weight each metric by its number of labels so smaller batches contribute proportionally less than full batches.
    loss_meters = [AverageMeter() for _ in classifiers]
    
    all_predictions = []
    all_video_paths = []
    all_labels = []

    # --- NEW: Wrap loader in tqdm if val_only ---
    if val_only:
        iterator = tqdm(data_loader, desc="Inference", unit="batch", dynamic_ncols=True)
    else:
        iterator = data_loader

    from torch.amp import autocast
    for itr, data in enumerate(iterator):
        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        # with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)
            
            video_paths = data[3] if len(data) > 3 else [f"video_{itr}_{i}" for i in range(batch_size)]

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    outputs = [[c(o) for o in outputs] for c in classifiers]
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]

        # Compute loss with proper dtype handling
        if task_type == "regression":
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            losses = [[criterion(o.float(), labels) for o in coutputs] for coutputs in outputs]
        else:
            losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]

        if training:
            _raise_if_non_finite_probe_loss(losses, device)
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]
            
        # Compute metrics based on task type: sums over this batch on every rank, added up across
        # ranks in one all-reduce, then divided by the number of labels they cover.
        with torch.no_grad():
            if task_type == "regression":
                outputs = [sum([o for o in coutputs]) / len(coutputs) for coutputs in outputs]
                # Absolute error of the z-scored targets, summed over the batch.
                scores = [(o.float() - labels).abs().sum() for o in outputs]
            else:  # classification
                outputs = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
                # Correct predictions in the batch.
                scores = [coutputs.max(dim=1).indices.eq(labels).sum().float() for coutputs in outputs]
            count = float(labels.numel())
            # Scale each head's mean loss by the label count to obtain its batch loss sum.
            loss_sums = [sum(l.detach().float() for l in li) / len(li) * count for li in losses]
            totals = AllReduceSum.apply(torch.stack([torch.tensor(count, device=device), *scores, *loss_sums]))
            totals = totals.cpu().tolist()
            n, scores, loss_sums = totals[0], totals[1:1 + len(outputs)], totals[1 + len(outputs):]
            if task_type == "regression":
                # Convert normalized MAE to EF percentage points using the training-set target standard deviation.
                t_std = target_std if target_std is not None else 1.0
                for meter, total in zip(metric_meters, scores):
                    meter.update(total / n * t_std, n=n)
            else:
                for t1m, total in zip(top1_meters, scores):
                    t1m.update(100.0 * total / n, n=n)
            for meter, total in zip(loss_meters, loss_sums):
                meter.update(total / n, n=n)
                    
            if val_only and predictions_save_path is not None:
                for i, pred in enumerate(outputs[0]):
                    all_predictions.append(pred.float().cpu().numpy())  # Convert to float32 first
                    all_video_paths.append(video_paths[i])
                    all_labels.append(labels[i].float().cpu().numpy())  # Also convert labels

        if training:
            [[lij.backward() for lij in li] for li in losses]
            [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]


        # Aggregate metrics for logging
        if task_type == "regression":
            _agg_metrics = np.array([m.avg for m in metric_meters])
            metric_name = "MAE" 
            metric_symbol = ""
        else:  # classification
            _agg_metrics = np.array([t1m.avg for t1m in top1_meters])
            metric_name = "Acc"
            metric_symbol = "%"
            
        # Only log to text log periodically (keep tqdm clean)
        if itr % 10 == 0:
            if val_only:
                # Update description dynamically with metrics
                if task_type == "regression":
                    # [FIX 1] Changed Label to MAE
                    iterator.set_description(f"Inf MAE: {_agg_metrics.min():.4f}")
                else:
                    iterator.set_description(f"Inf Acc: {_agg_metrics.max():.2f}%")
            else:
                best_scalar = float(_agg_metrics.min()) if task_type == "regression" else float(_agg_metrics.max())
                msg = "[%5d] %.3f%s [mean %.3f%s] [mem: %.2e]" % (
                    itr, best_scalar, metric_symbol, _agg_metrics.mean(), metric_symbol,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
                logger.info(msg)


    # Save predictions (Un-normalized)
    if val_only and predictions_save_path is not None and len(all_predictions) > 0:
        import pandas as pd
        import os
        os.makedirs(os.path.dirname(predictions_save_path), exist_ok=True)
            
        if task_type == "regression":
            # For regression, save REAL values
            # 1. Get raw normalized predictions
            pred_values_norm = [pred[0] if len(pred.shape) > 0 else pred for pred in all_predictions]
            
            # 2. Un-normalize logic for CSV
            # --- CHANGE THIS BLOCK ---
            t_mean = target_mean if target_mean is not None else 0.0
            t_std = target_std if target_std is not None else 1.0
            
            # Convert arrays to scalars and un-normalize
            labels_real = []
            for l in all_labels:
                val = l[0] if isinstance(l, (np.ndarray, list)) else l
                labels_real.append((val * t_std) + t_mean)  # Use variables
                
            preds_real = []
            for p in pred_values_norm:
                val = p[0] if isinstance(p, (np.ndarray, list)) else p
                preds_real.append((val * t_std) + t_mean)   # Use variables

            df = pd.DataFrame({
                'video_path': all_video_paths,
                'label_real': labels_real,
                'pred_real': preds_real,
                'abs_error': [abs(a-b) for a,b in zip(labels_real, preds_real)]
            })
        else:  # classification
            pred_classes = [np.argmax(pred) for pred in all_predictions]
            pred_probs = [pred.max() for pred in all_predictions]
            df = pd.DataFrame({
                'video_path': all_video_paths,
                'true_label': all_labels,
                'predicted_class': pred_classes,
                'prediction_confidence': pred_probs
            })
            
        df.to_csv(predictions_save_path, index=False)
        logger.info(f"Saved {len(all_predictions)} predictions to {predictions_save_path}")

    _agg_metrics = np.array([m.avg for m in (metric_meters if task_type == "regression" else top1_meters)])
    scalar = float(_agg_metrics.min()) if task_type == "regression" else float(_agg_metrics.max())
    return scalar, _agg_metrics, np.array([m.avg for m in loss_meters])


def _raise_if_non_finite_probe_loss(losses, device):
    """Stop every rank before training state changes when any probe loss is non-finite."""
    local_failure = any(
        not bool(torch.isfinite(loss.detach()).all().item())
        for head_losses in losses
        for loss in head_losses
    )
    if any_rank_failed(local_failure, device=device):
        if local_failure:
            raise FloatingPointError("Non-finite probe loss; stopping all ranks before optimizer update.")
        raise FloatingPointError(
            "Another rank produced a non-finite probe loss; stopping all ranks before optimizer update."
        )


class _EachVideoOnce(torch.utils.data.Dataset):
    """Some of a `VideoDataset`'s videos, each read once and returned with its index.
    
    This is a small wrapper dataset (test-only evaluation path) around the existing `VideoDataset`.
    During final testing, every test video should be attempted exactly once. If a video cannot be loaded,
    record that failure instead of silently replacing it with another random video (as in `VideoDataset`).
    """

    def __init__(self, dataset, indices):
        # `self.dataset`: original `VideoDataset`, containing all test videos.
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        """Keep the original index so the code knows exactly which test row this prediction belongs to."""
        index = self.indices[i]
        try:
            return index, self.dataset.get_item_video(index)
        except Exception:
            return index, None


def _collate_loaded(batch):
    """(indices, batch, failed indices) for a batch of `_EachVideoOnce` items. Separates the successful and failed ones."""
    loaded = [(index, item) for index, item in batch if item]
    failed = [index for index, item in batch if not item]
    if not loaded:
        return [], None, failed
    return [index for index, _ in loaded], default_collate([item for _, item in loaded]), failed


def score_videos(device, encoder, heads, dataset, batch_size, num_workers, use_bfloat16, task_type, rank,
                 world_size):
    """Score every video in `dataset` exactly once with each probe head. Each rank processes every
    `world_size`-th video, avoiding duplicate samples from distributed-sampler padding. Videos that
    fail to load are recorded as failures rather than replaced. Each video's prediction is the mean
    over its clips; for classification, softmax probabilities are averaged.

    Results are gathered across all ranks so every rank uses the same complete set of predictions
    for subsequent decisions.

    Returns (indices, labels, predictions, failed): the dataset rows scored, in dataset order;
    their labels [N, targets]; the predictions [heads, N, outputs]; the rows that failed.
    """
    from torch.amp import autocast

    # After leaving DDP, ranks may process different numbers of videos, so probe heads
    # must not perform cross-rank synchronization.
    modules = [getattr(h, "module", h) for h in heads]
    for m in modules:
        m.eval()
    loader = torch.utils.data.DataLoader(
        _EachVideoOnce(dataset, range(rank, len(dataset), world_size)),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_loaded,
        pin_memory=True,
    )
    indices, labels, predictions, failed = [], [], [[] for _ in modules], []
    for batch_indices, data, batch_failed in loader:
        failed += batch_failed
        # If every item in this batch failed, there's nothing to infer.
        if data is None:
            continue
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            # One token sequence per spatial view, holding the tokens of all the video's clips
            # (`ClipAggregation`), reused by every probe head.
            features = encoder(clips, clip_indices)
            outputs = [[m(f) for f in features] for m in modules]
        for head_predictions, outs in zip(predictions, outputs):
            if task_type == "regression":
                output = sum(o.float() for o in outs) / len(outs)
            else:
                output = sum(F.softmax(o.float(), dim=1) for o in outs) / len(outs)
            head_predictions += output.cpu().tolist()
        indices += batch_indices
        labels += data[1].tolist()

    local = (indices, labels, predictions, failed)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    indices = [i for g in gathered for i in g[0]]
    failed = sorted(i for g in gathered for i in g[3])
    # Correctness check.
    if len(set(indices)) != len(indices) or len(indices) + len(failed) != len(dataset):
        raise RuntimeError(
            f"Scored {len(indices)} and failed {len(failed)} of {len(dataset)} videos: "
            "every video should be scored or failed exactly once."
        )
    if not indices:
        raise RuntimeError(f"None of the {len(dataset)} videos could be loaded.")
    # Ranks process videos interleaved: restore dataset order so each prediction remains aligned
    # with its corresponding video and label.
    order = np.argsort(indices)
    n = len(order)
    labels = np.asarray([x for g in gathered for x in g[1]], dtype=float).reshape(n, -1)[order]
    predictions = np.stack([
        np.asarray([x for g in gathered for x in g[2][h]], dtype=float).reshape(n, -1)[order]
        for h in range(len(modules))
    ])
    return np.asarray(indices)[order], labels, predictions, failed


def validate(device, encoder, classifiers, dataset, batch_size, num_workers, use_bfloat16, target_mean,
             target_std, rank, world_size, studies=None):
    """Validate every probe head for one epoch, scoring each video exactly once.

    For each head, compute the per-video MAE in EF percentage points, the L1 loss on z-scored
    targets, and, when study IDs are provided, the per-study MAE; `experiment.evaluation.selection`
    picks the one probe selection uses. Also report the numbers of successfully scored and failed
    videos.
    """
    indices, labels, predictions, failed = score_videos(
        device, encoder, classifiers, dataset, batch_size, num_workers, use_bfloat16, "regression", rank,
        world_size,
    )
    t_mean = target_mean if target_mean is not None else 0.0
    t_std = target_std if target_std is not None else 1.0
    y = labels * t_std + t_mean
    out = {"mae": [], "loss": [], "study_mae": [] if studies is not None else None,
           "videos": len(indices), "failed_videos": len(failed)}
    # Loops over probe heads.
    for p_z in predictions:
        # Converts the head's normalized EF predictions back into actual EF values.
        p = p_z * t_std + t_mean
        # Compute per-video MAE.
        out["mae"].append(float(np.abs(p - y).mean()))
        # Compute the L1 loss in normalized target space.
        out["loss"].append(probe_metrics.smooth_l1(labels, p_z))
        if studies is not None:
            # Compute study-level MAE: each study's prediction is the mean of its videos'.
            per_study = probe_metrics.by_study(np.asarray(studies)[indices], y[:, 0], p[:, 0])
            out["study_mae"].append(float((per_study.prediction - per_study.label).abs().mean()))
    for key in ("mae", "loss", "study_mae"):
        if out[key] is not None:
            out[key] = np.asarray(out[key])
    return out


def _scalars(prefix, block):
    """Extract numeric values from a metrics block and flatten them into wandb keys under `prefix`."""
    out = {}
    for key, value in block.items():
        if isinstance(value, dict):
            out.update(_scalars(prefix, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"{prefix}_{key}"] = value
    return out


def run_test(
    device,
    encoder,
    classifiers,
    dataset,
    batch_size,
    num_workers,
    checkpoint_path,
    task_type,
    use_bfloat16,
    target_mean,
    target_std,
    head_names,
    folder,
    rank,
    world_size,
    val_dataset=None,
    video_index=None,
    low_ef_below=40.0,
    ef_ranges=(30, 40, 50, 60),
    targets=None,
    run_info=None,
    wandb_run=None,
    step=0,
    fingerprint=None,
):
    """Score the test split once, with the epoch and head chosen on validation.

    `best.pt` is the epoch whose best head had the best validation score, and that head is
    the one scored. For regression, first rescore the validation split with the selected head
    to determine the reduced-EF threshold at the target level (study or video), then apply that
    fixed threshold to the test set. A video that fails to load is counted, not replaced.

    Write `test_metrics.json` and `test_predictions.csv` to `folder`; for regression, also write
    `val_predictions.csv`. The predictions list video paths, which can identify patients: keep
    them where the data is. Only de-identified metrics and plots are logged to wandb.
    
    Require the fingerprint stored in `best.pt` to match the encoder, data, and evaluation
    settings used for testing, except for the number of GPUs.
    
    Returns the metrics on rank 0 and `None` on the other ranks.
    """
    # First load the checkpoint onto CPU, then retrieve the validation score for every probe head
    # and choose the best head using validation only.
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    if fingerprint is not None:
        # Every rank checks the same two fingerprints, so all ranks stop together.
        provenance.check_fingerprint(checkpoint.get("fingerprint"), fingerprint, checkpoint_path,
                                     ignore=("settings.world_size",))
    val_per_head = np.asarray(checkpoint["val_acc_per_head"], dtype=float)
    head = int(val_per_head.argmin() if task_type == "regression" else val_per_head.argmax())
    # Outside DDP, each rank independently runs its assigned videos through the probe head.
    classifier = getattr(classifiers[head], "module", classifiers[head])
    # Load the selected head's weights from checkpoints.
    classifier.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["classifiers"][head].items()})
    # Put the probe into eval mode, so it turns off training-specific behavior such as dropout.
    classifier.eval()
    logger.info(f"Testing {head_names[head]} from epoch {checkpoint['epoch']} of {checkpoint_path}")

    def score(split_dataset):
        return score_videos(device, encoder, [classifier], split_dataset, batch_size, num_workers, use_bfloat16,
                            task_type, rank, world_size)

    regression = task_type == "regression"
    val_scored = score(val_dataset) if regression and val_dataset is not None else None
    indices, labels, predictions, failed = score(dataset)
    # Avoids all GPUs independently writing the same output files.
    if rank != 0:
        return None

    metric = "mae" if regression else "acc"
    selected = checkpoint.get("selection_metric", metric)
    paths = [dataset.samples[i] for i in indices]
    # Build the prediction table.
    table = {"video_path": paths}
    result = {
        "selected_on": "validation",
        "selection": checkpoint.get("selection", SELECTION["video"]),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": provenance.sha256(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "head": head,
        "head_name": head_names[head],
        f"val_{selected}": float(val_per_head[head]),
    }
    if regression:
        t_mean = target_mean if target_mean is not None else 0.0
        t_std = target_std if target_std is not None else 1.0
        y, p = labels[:, 0] * t_std + t_mean, predictions[0][:, 0] * t_std + t_mean
        # Use the same target level (study or video) for the EF-range table and plots.
        targets = dict(targets or {})
        level = targets.get("level") or ("study" if video_index is not None else "video")
        if level == "study" and video_index is None:
            raise ValueError("Study-level results need `experiment.data.video_index`.")
        test_studies = None
        if video_index is not None:
            table["study_id"] = provenance.study_ids(video_index, paths, "test")
            test_studies = probe_metrics.by_study(table["study_id"], y, p)

        # The reduced-EF threshold comes from validation, at the level the targets use.
        threshold, val_block = None, None
        if val_scored is not None:
            v_indices, v_labels, v_predictions, v_failed = val_scored
            v_paths = [val_dataset.samples[i] for i in v_indices]
            vy, vp = v_labels[:, 0] * t_std + t_mean, v_predictions[0][:, 0] * t_std + t_mean
            val_table = {"video_path": v_paths, "label": vy, "prediction": vp}
            val_studies, val_block = None, {}
            if video_index is not None:
                val_table["study_id"] = provenance.study_ids(video_index, v_paths, "val")
                val_studies = probe_metrics.by_study(val_table["study_id"], vy, vp)
            chosen_on = val_studies if level == "study" else pd.DataFrame({"label": vy, "prediction": vp})
            threshold = probe_metrics.youden_threshold(chosen_on.label, chosen_on.prediction, low_ef_below)
            if val_studies is not None:
                val_block["study"] = probe_metrics.summarize(val_studies.label, val_studies.prediction,
                                                             low_ef_below, threshold)
            val_block.update(video=probe_metrics.summarize(vy, vp, low_ef_below, threshold),
                             videos=len(v_indices), failed_videos=len(v_failed))
            pd.DataFrame(val_table).to_csv(os.path.join(folder, "val_predictions.csv"), index=False)

        test = {"video": probe_metrics.summarize(y, p, low_ef_below, threshold)}
        if test_studies is not None:
            test["study"] = probe_metrics.summarize(test_studies.label, test_studies.prediction, low_ef_below,
                                                    threshold)
        test.update(videos=len(indices), failed_videos=len(failed))
        if test_studies is not None:
            test["studies"] = int(len(test_studies))
        ranged = test_studies if level == "study" else pd.DataFrame({"label": y, "prediction": p})
        by_range = probe_metrics.by_range(ranged.label, ranged.prediction, ef_ranges)

        met = {}
        if "mae_below" in targets:
            met["mae"] = bool(test[level]["mae"] < targets["mae_below"])
        if "auroc_above" in targets:
            met["auroc"] = bool(test[level]["reduced_ef"]["auroc"] > targets["auroc_above"])
        result.update(
            aggregation=AGGREGATION,
            low_ef_threshold={"rule": THRESHOLD_RULE, "below": low_ef_below, "level": level, "value": threshold},
            val=val_block,
            test=test,
            test_by_reference_range={"level": level, "rows": by_range},
            targets={**targets, "level": level, "met": met},
        )
        table.update(label=y, prediction=p, abs_error=np.abs(p - y))
    else:   # classification
        predicted = predictions[0].argmax(axis=1)
        result["test"] = {"acc": float(100.0 * (predicted == labels[:, 0]).mean()),
                          "videos": len(indices), "failed_videos": len(failed)}
        table.update(label=labels[:, 0].astype(int), prediction=predicted, confidence=predictions[0].max(axis=1))
    result.update(run_info=run_info, failed_rows=failed)  # rows of the test list (dataset row indices), not paths.

    pd.DataFrame(table).to_csv(os.path.join(folder, "test_predictions.csv"), index=False)
    # Written last and atomically: its presence means the test finished.
    tmp = os.path.join(folder, "test_metrics.json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, os.path.join(folder, "test_metrics.json"))

    # Log metrics and identifier-free plots to wandb using the same target level (study or video).
    scalars = {"probe/test_epoch": result["epoch"], "probe/test_head": head,
               f"probe/test_val_{selected}": result[f"val_{selected}"]}
    if regression:
        for block_level in ("video", "study"):
            if block_level in result["test"]:
                scalars.update(_scalars(f"probe/test/{block_level}", result["test"][block_level]))
        if threshold is not None:
            scalars["probe/test/threshold"] = threshold
        scalars.update({f"probe/test/target_met_{k}": float(v) for k, v in met.items()})
    else:
        scalars["probe/test/video_acc"] = result["test"]["acc"]
    log_scalars(wandb_run, scalars, step=step)
    if regression:
        log_regression_results(wandb_run, ranged.label, ranged.prediction, by_range, step, f"probe/test/{level}")
    # Reference the tested checkpoint by checksum, matching the reference recorded at the end of training.
    log_reference(wandb_run, "probe_checkpoint", "probe-checkpoint",
                  checkpoint_reference(checkpoint_path, task_type, head_names, fingerprint,
                                       sha=result["checkpoint_sha256"]))
    logger.info(f"Test of {head_names[head]} (epoch {result['epoch']}): {json.dumps(result['test'])}")
    return result


def describe_run(args_eval, checkpoint, manifests, fingerprint, video_index, seed, parameter_counts, opt_kwargs,
                 batch_size, world_size, num_epochs, low_ef_below, targets, selection):
    """Describe the probe run without filesystem paths, including the code version, encoder checkpoint and
    manifest identities, split sizes, seed, optimization, aggregation, selection, and target settings."""
    # Gets the current Git commit and whether the repository has uncommitted changes.
    commit, dirty = provenance.git_state()
    files = {}
    # For each manifest file, records only the filename, its SHA-256 hash from the fingerprint, and the
    # number of non-empty rows.
    for name, path in manifests.items():
        if path and os.path.exists(path):
            with open(path) as f:
                rows = sum(1 for line in f if line.strip())
            files[name] = {"file": os.path.basename(path), "sha256": fingerprint["manifests_sha256"][name],
                           "rows": rows}
    meta = args_eval.get("meta") or {}
    
    # Returned dictionary: a run provenance record.
    return {
        "experiment": meta.get("wandb_run_name") or args_eval.get("tag"),
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "encoder_checkpoint": {"file": os.path.basename(checkpoint), "sha256": fingerprint["encoder_sha256"]},
        "normalization": fingerprint["normalization"],
        "manifests": files,
        "splits": provenance.split_counts(video_index) if video_index is not None else None,
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
        "aggregation": AGGREGATION,
        "protocol": PROTOCOL,
        "selection": SELECTION[selection],
        "low_ef_below": low_ef_below,
        "low_ef_threshold_rule": THRESHOLD_RULE,
        "targets": targets,
    }


def save_run_record(folder, args_eval, run_info, fresh):
    """Save the probe's original configuration and run metadata in its output folder.

    For a fresh run, write `params-probe.yaml` and `run_info.json`. For a resumed run, preserve the
    original files and warn if the current configuration differs in settings not covered by the
    fingerprint (e.g., `num_workers`)."""
    # `params-probe.yaml`: contains the full original config, including paths.
    path = os.path.join(folder, "params-probe.yaml")
    text = yaml.safe_dump(args_eval, sort_keys=False)
    # `fresh`: determines whether this is a new probe or a resumed one.
    if not fresh and os.path.exists(path):
        with open(path) as f:
            if f.read() != text:
                logger.warning(f"The config differs from the one this probe started with; {path} keeps "
                               "the original.")
        return
    with open(path, "w") as f:
        f.write(text)
    # `run_info.json`: contains the cleaner path-free summary created by `describe_run()`.
    with open(os.path.join(folder, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)


def check_normalization(mean, std, train_manifest):
    """Refuse a regression probe without a usable EF normalization: the probe manifests hold
    z-scored EF, so a missing mean or standard deviation would report z-scores as EF points. When
    the train manifest's folder holds the `probe_info.json` written with it, the two must agree."""
    try:
        mean, std = float(mean), float(std)
    except (TypeError, ValueError):
        raise ValueError("Set `experiment.data.target_mean` and `target_std` (from the manifests' "
                         "probe_info.json).") from None
    if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
        raise ValueError(f"`experiment.data.target_mean` and `target_std` must be finite and the standard "
                         f"deviation positive, not {mean} and {std}.")
    info_path = os.path.join(os.path.dirname(train_manifest), "probe_info.json") if train_manifest else None
    if info_path and os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if not (math.isclose(mean, info["target_mean"], rel_tol=1e-9)
                and math.isclose(std, info["target_std"], rel_tol=1e-9)):
            raise ValueError(f"`target_mean` / `target_std` ({mean}, {std}) differ from the probe_info.json next "
                             f"to the train manifest ({info['target_mean']}, {info['target_std']}).")


def gather_rng_states(world_size):
    """Every rank's random number state (Python, NumPy, Torch CPU and CUDA), in rank order, held in
    tensors and plain values so `torch.load(weights_only=True)` reads it back."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    state = {"python": random.getstate(),
             "numpy": {"kind": kind, "keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
                       "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
             "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        states = [None] * world_size
        dist.all_gather_object(states, state)
        return states
    return [state]


def set_rng_state(state):
    """Restore one rank's random number state saved by `gather_rng_states`."""
    random.setstate(tuple(tuple(v) if isinstance(v, list) else v for v in state["python"]))
    n = state["numpy"]
    np.random.set_state((n["kind"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"],
                         n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def check_split(index, dataset, split):
    """Validate that every manifest video exists in the video index and belongs to the expected split."""
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError(f"The {split} dataset lists no samples to check against the video index.")
    provenance.study_ids(index, samples, split)


def shared_fingerprint(rank, world_size, device, checkpoint, manifests, normalization, settings):
    """The probe's fingerprint (`provenance.fingerprint`), hashed on rank 0 and sent to every
    rank, so all ranks take the same decision from it. A failure on rank 0 stops every rank."""
    current, error = None, None
    # Only rank 0 computes the fingerprint.
    if rank == 0:
        try:
            current = provenance.fingerprint(checkpoint, manifests, normalization, settings)
        except Exception as e:
            error = e
    # Failure is communicated to all ranks.
    if any_rank_failed(error is not None, device=device):
        if error is not None:
            raise error
        raise RuntimeError("Computing the probe's fingerprint failed on rank 0.")
    shared = [current]
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        # Rank 0 sends the fingerprint to all other ranks.
        dist.broadcast_object_list(shared, src=0)
    return shared[0]


def save_atomically(obj, path):
    """`torch.save` to a temporary file renamed to `path`: `path` holds the old or the new
    checkpoint, never part of one."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _save_probe_checkpoints(save_dict, folder, is_best, rank, device):
    """Save on rank 0, then make every rank stop if any checkpoint write failed."""
    error = None
    if rank == 0:
        try:
            # The epoch snapshot, then `best.pt` when selected, and `latest.pt` last: an
            # interrupted epoch leaves `latest.pt` at the previous completed epoch.
            epoch = save_dict["epoch"]
            save_atomically(save_dict, os.path.join(folder, f"epoch_{epoch:03d}.pt"))
            if is_best:
                best_path = os.path.join(folder, "best.pt")
                save_atomically(save_dict, best_path)
                logger.info(f"Generated new best model: {best_path}")
            save_atomically(save_dict, os.path.join(folder, "latest.pt"))
        except Exception as exc:
            error = exc

    if any_rank_failed(error is not None, device=device):
        if error is not None:
            raise error
        raise RuntimeError("Rank 0 could not save the probe checkpoint; stopping every rank.")


def check_finished(folder, num_epochs):
    """Refuse to test a probe that has not finished training, or whose `best.pt` is not the epoch
    `latest.pt` records as the best (by epoch and validation score)."""
    def saved(name):
        path = os.path.join(folder, name)
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True) if os.path.exists(path) else {}

    latest = saved("latest.pt")
    done = latest.get("epoch", 0)
    if done < num_epochs:
        raise RuntimeError(f"The probe has finished {done} of {num_epochs} epochs; test it once it is complete.")
    best = saved("best.pt")
    recorded = (latest.get("best_epoch"), latest.get("best_val_acc"))
    if (best.get("epoch"), best.get("best_val_acc")) != recorded:
        raise RuntimeError(f"`best.pt` holds epoch {best.get('epoch')}, but training selected epoch {recorded[0]} "
                           f"(validation {recorded[1]}): the folder does not hold the selected probe.")


def restore_best(folder, best_epoch):
    """Before training resumes, make `best.pt` the best epoch `latest.pt` records. They differ when a
    run stopped after saving an epoch's `best.pt` and before its `latest.pt`: that epoch is trained
    again, and `best.pt` goes back to the recorded best epoch, copied from its snapshot."""
    best_path = os.path.join(folder, "best.pt")
    if (os.path.exists(best_path)
            and torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)["epoch"] == best_epoch):
        return
    snapshot = os.path.join(folder, f"epoch_{best_epoch:03d}.pt")
    if not os.path.exists(snapshot):
        logger.warning(f"`best.pt` is not epoch {best_epoch}, the best epoch the checkpoint records, and that "
                       "epoch's snapshot is not in the folder: `--test_only` will refuse this probe.")
        return
    shutil.copyfile(snapshot, f"{best_path}.tmp")
    os.replace(f"{best_path}.tmp", best_path)
    logger.warning(f"`best.pt` held an epoch the checkpoint does not record; restored epoch {best_epoch}.")


def checkpoint_reference(path, task_type, head_names, fingerprint, sha=None):
    """Describe a saved probe without its filesystem path using its file name, SHA-256,
    epoch, validation-selected head and score, and encoder checkpoint identity."""
    # Loads the checkpoint.
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    # Gets the validation score of every head.
    val = np.asarray(saved["val_acc_per_head"], dtype=float)
    # Chooses the best head.
    head = int(val.argmin() if task_type == "regression" else val.argmax())
    metric = "mae" if task_type == "regression" else "acc"
    return {
        "file": os.path.basename(path),
        "sha256": sha or provenance.sha256(path),
        "epoch": int(saved["epoch"]),
        "head": head,
        "head_name": head_names[head],
        f"val_{saved.get('selection_metric', metric)}": float(val[head]),
        "encoder_sha256": (fingerprint or {}).get("encoder_sha256"),
    }


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading classifier(s)
    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        # Log metrics if present (no change to return signature)
        if "best_val_acc" in checkpoint or "mean_val_acc" in checkpoint:
            logger.info(
                "loaded metrics: best_val_acc=%s mean_val_acc=%s",
                checkpoint.get("best_val_acc", "NA"),
                checkpoint.get("mean_val_acc", "NA"),
            )
        logger.info(f"loaded pretrained classifier (val_only) with msg: {msg}")
        return classifiers, opt, scaler, 0, {}

    epoch = int(checkpoint["epoch"])
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    # -- scaler (if used)
    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        for s, sd in zip(scaler, checkpoint["scaler"]):
            if s is None or sd is None:
                continue
            s.load_state_dict(sd)

    # Log metrics if present (keeps return arity identical)
    if "best_val_acc" in checkpoint or "mean_val_acc" in checkpoint:
        logger.info(
            "loaded metrics: best_val_acc=%s mean_val_acc=%s",
            checkpoint.get("best_val_acc", "NA"),
            checkpoint.get("mean_val_acc", "NA"),
        )

    logger.info(f"loaded optimizers from epoch {epoch}")
    # Keep the validation history so a resumed probe can continue from the previous run.
    history_keys = ("mean_val_acc", "best_val_acc", "best_val_acc_per_head", "mean_val_acc_per_head",
                    "min_val_acc_per_head", "best_epoch_per_head", "best_epoch", "fingerprint", "rng_states")
    history = {k: checkpoint[k] for k in history_keys if k in checkpoint}
    return classifiers, opt, scaler, epoch, history


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    # Training: a random clip position in each segment, and each GPU's last incomplete batch
    # dropped. Evaluation: the start of each segment, every time, and every video kept.
    sampling = PROTOCOL["train" if training else "evaluation"]
    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        random_clip_sampling=sampling["clip_positions"] == "random",
        drop_last=sampling["drop_last"],
        subset_file=subset_file,
    )
    return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        # scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
        scalers += [None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
