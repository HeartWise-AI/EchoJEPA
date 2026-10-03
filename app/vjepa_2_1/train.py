# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from app.vjepa.utils import grad_norm
from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.models.utils.modules import Lambda_LinearWarmupHold
from app.vjepa_2_1.transforms import make_transforms
from app.vjepa_2_1.utils import (
    clip_grads_with_norm_,
    feature_spread,
    init_opt,
    init_video_model,
    load_checkpoint,
    load_pretrained_weights,
    masked_l1_loss,
    normalize_nested,
    select_load_path,
    token_validity,
)
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import any_rank_failed, global_sample_weighted_means, init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.wandb_logging import finish_wandb, init_wandb, log_input_clips, log_scalars
from torch.nn.parallel import DistributedDataParallel


log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
MAX_REPEAT_COUNTS = 10

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# ImageNet mean/std, the transform's default; passed explicitly so logged clips are
# denormalized with the statistics that normalized them.
NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

logger = get_logger(__name__, force=True)


def _raise_if_non_finite_loss(loss, device):
    """Stop every rank when any rank produces a non-finite loss."""
    local_failure = not bool(torch.isfinite(loss.detach()).item())
    if any_rank_failed(local_failure, device=device):
        raise FloatingPointError(
            "Non-finite loss detected on at least one rank before backward; "
            "stopping all ranks."
        )


def _atomic_torch_save(save_dict, path):
    """Write a checkpoint beside its destination, then publish it atomically."""
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    prefix = f".{os.path.basename(destination)}."
    fd, temporary_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        with open(temporary_path, "wb") as temporary_file:
            torch.save(save_dict, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def _run_rank_zero_checkpoint_save(save, rank, device):
    """Run a checkpoint write on rank 0 and stop every rank if it fails."""
    error = None
    if rank == 0:
        try:
            save()
        except Exception as exc:
            error = exc

    if any_rank_failed(error is not None, device=device):
        message = "Checkpoint save failed on rank 0; stopping all ranks."
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)


def _log_data_loading_retry(error, retry_number, max_retries):
    """Log a retry without exposing paths or other details from the exception."""
    logger.warning(
        "Data loading failed with %s; retrying (%d/%d).",
        type(error).__name__,
        retry_number,
        max_retries,
    )


def _raise_data_loading_retries_exhausted(error, max_retries):
    """Raise a safe public error while preserving the original exception as its cause."""
    raise RuntimeError(
        f"Exceeded max retries ({max_retries}) when loading data."
    ) from error


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    # Initialize the model from pretrained weights for the encoder, target encoder,
    # and predictor, but start a brand-new optimizer and learning-rate schedule.
    # If this training run later has its own checkpoint, resume from that
    # checkpoint instead and ignore the original weights-only initialization.
    init_checkpoint = cfgs_meta.get("init_checkpoint", None)
    # `read_checkpoint` resumes the full training state, including model weights, optimizer, scaler,
    # epoch, and schedules. Do not use it when starting a new run from pretrained weights only.
    if init_checkpoint is not None and r_file is not None:
        raise ValueError(
            "Set `meta.init_checkpoint` (weights only, fresh optimizer and schedule) or "
            "`meta.read_checkpoint` (full resume of a run), not both."
        )
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    # wandb: as in `app/vjepa`. Enabled only when `wandb_project` (and `wandb_entity`) are set.
    # Log `num_log_videos` input videos at the first step of the run, then log `num_log_videos`
    # more every `log_video_freq` steps during that same epoch, if `log_video_freq > 0`.
    wandb_enabled = bool(cfgs_meta.get("wandb_project", None))
    log_video_freq = cfgs_meta.get("log_video_freq", 0)
    num_log_videos = cfgs_meta.get("num_log_videos", 2)
    logger.info(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD')}")
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    is_causal = cfgs_model.get("is_causal", False)
    pred_is_causal = cfgs_model.get("pred_is_causal", False)
    init_type = cfgs_model.get("init_type", "default")
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    n_registers = cfgs_model.get("n_registers", 0)
    has_cls_first = cfgs_model.get("has_cls_first", False)
    interpolate_rope = cfgs_model.get("interpolate_rope", False)
    lambda_value_img = cfgs_model.get("lambda_value_img", 0.0)
    lambda_value_vid = cfgs_model.get("lambda_value_vid", 0.0)
    n_registers_predictor = cfgs_model.get("n_registers_predictor", 0)
    lambda_progressive = cfgs_model.get("lambda_progressive", True)
    # Upstream uses a fixed 15k–30k-step ramp intended for ~300k-step training runs.
    # For much shorter runs, that ramp should be shortened proportionally;
    # otherwise the context-loss weighting/schedule spends most of training in
    # the ramp phase instead of its intended steady state.
    lambda_start_iter = cfgs_model.get("lambda_start_iter", 15000)
    lambda_end_iter = cfgs_model.get("lambda_end_iter", 30000)
    normalize_predictor = cfgs_model.get("normalize_predictor", False)
    modality_embedding = cfgs_model.get("modality_embedding", False)
    # Number of encoder layers used as predictor targets:
    # `4`: gives deep multi-layer self-supervision.
    # `1`: uses only the final layer.
    # Encoder, predictor, and target representations must use the same normalization,
    # so a single config key controls all three; upstream applied it only to the target side.
    n_output_distillation = cfgs_model.get("n_output_distillation", 4)
    # The encoder only defines its distillation layers for these two values.
    if n_output_distillation not in (1, 4):
        raise ValueError(f"model.n_output_distillation must be 1 or 4, got {n_output_distillation!r}.")
    levels_predictor = n_output_distillation
    if has_cls_first:
        raise NotImplementedError("Padding-mask alignment assumes the sequence contains no [CLS] token.")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    grid_size = crop_size // patch_size
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)

    # -- IMG DATA
    cfgs_img_data = args.get("img_data")
    img_rank_ratio = 0.25
    img_mask = None
    if cfgs_img_data is not None:
        img_dataset_type = cfgs_img_data.get("dataset_type", "imagenet")
        img_dataset_paths = cfgs_img_data.get("datasets", [])
        img_dataset_weights = cfgs_img_data.get("datasets_weights", [])
        img_dataset_fpcs = cfgs_img_data.get("dataset_fpcs")
        img_dataset_batch_size = cfgs_img_data.get("batch_size")
        img_rank_ratio = cfgs_img_data.get("rank_ratio", img_rank_ratio)
        img_num_workers = cfgs_img_data.get("num_workers", num_workers)

        img_mask = args.get("img_mask", img_mask)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)
    # Horizontal flipping is disabled by default because it creates anatomically
    # unrealistic mirrored views that the scanner does not produce.
    random_hflip = cfgs_data_aug.get("random_horizontal_flip", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    shift_by_n = cfgs_loss.get("shift_by_n")
    predict_all = cfgs_loss.get("predict_all", True)
    weight_distance_loss = cfgs_loss.get("weight_distance_loss", False)
    offset_context_loss = cfgs_loss.get("offset_context_loss", False)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (
        is_anneal and resume_preempt
    )
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    use_radamw = cfgs_opt.get("use_radamw", False)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    loss_reg_std_mult = cfgs_opt.get("loss_reg_std_mult", None)
    loss_reg_num_tracking_steps = cfgs_opt.get("loss_reg_num_tracking_steps", 300)
    loss_reg_min_epoch = cfgs_opt.get("loss_reg_min_epoch", 50)
    if loss_reg_std_mult is not None:
        logger.info("Loss regulation activated")
    # Maximum combined gradient norm for the encoder and predictor.
    # `null` disables gradient clipping (upstream); otherwise larger gradients are scaled down to this value.
    clip_grad = cfgs_opt.get("clip_grad", None)
    if clip_grad is not None and not float(clip_grad) > 0:
        raise ValueError(f"optimization.clip_grad must be a positive number or null, got {clip_grad!r}.")
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    data_world_size, data_rank = world_size, rank
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    img_world_size = 0

    # make adjustments to batch size for image data
    model_fpcs = dataset_fpcs
    model_cfgs_mask = cfgs_mask
    model_tubelet_size = tubelet_size
    if cfgs_img_data is not None:
        img_world_size = int(world_size * img_rank_ratio)
        num_video_ranks = world_size - img_world_size
        img_total_batch_size = img_dataset_batch_size * world_size
        video_total_batch_size = batch_size * world_size

        if img_total_batch_size % img_world_size != 0:
            raise ValueError(
                f"img_total_batch_size ({img_total_batch_size}) must be divisible by num_img_ranks ({img_world_size})"
            )
        if video_total_batch_size % num_video_ranks != 0:
            raise ValueError(
                f"video_total_batch_size ({video_total_batch_size}) must be divisible by num_video_ranks ({num_video_ranks})"
            )

        # img_dataset_batch_size = img_total_batch_size // img_world_size
        batch_size = video_total_batch_size // num_video_ranks

        if rank < int(world_size * img_rank_ratio):
            crop_size = cfgs_img_data.get("crop_size", 512)
            grid_size = crop_size // patch_size

        if rank < int(world_size * img_rank_ratio):
            logger.info(
                f"On rank {rank}, updating dataset with dataset type {img_dataset_type}"
            )
            if img_temporal_dim_size is not None:
                if img_dataset_fpcs[0] != 1:
                    raise NotImplementedError(
                        "Image loader only supports 1 frame per clip with img_temporal_dim_size=1"
                    )
                tubelet_size = 1
            else:
                tubelet_size = tubelet_size

            dataset_type = img_dataset_type
            dataset_paths = img_dataset_paths
            datasets_weights = img_dataset_weights
            dataset_fpcs = img_dataset_fpcs
            batch_size = img_dataset_batch_size
            num_workers = img_num_workers
            if img_mask is not None:
                logger.info("Using image mask")
                cfgs_mask = img_mask

            data_rank = rank
            data_world_size = img_world_size
            lambda_value = lambda_value_img  # We select a different lambda value depending on video vs. image
        else:
            data_rank = rank - img_world_size
            data_world_size = world_size - img_world_size
            lambda_value = lambda_value_vid  # We select a different lambda value depending on video vs. image

        logger.info(
            f"For rank {rank} with world size {world_size}, "
            f"we have total image batch size {img_total_batch_size}, total video batch size {video_total_batch_size}, "
            f"image ranks: {img_world_size}, video ranks: {num_video_ranks}, "
            f"using the following params: "
            f"dataset_type: {dataset_type}, "
            f"dataset_paths: {dataset_paths}, "
            f"datasets_weights: {datasets_weights}, "
            f"dataset_fpcs: {dataset_fpcs}, "
            f"batch_size: {batch_size}, "
            f"num_workers: {num_workers}, "
            f"data_rank: {data_rank}, "
            f"data_world_size: {data_world_size}"
            f"lambda_value for the context loss: {lambda_value}"
        )
    else:
        lambda_value = lambda_value_vid

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)

    # Upstream silently initialized a fresh model when the pretraining checkpoint was unavailable,
    # or when `meta.load_checkpoint` was `False`, so a cooldown could train from random weights
    # instead of the intended pretrained checkpoint. Cooldown runs now require the pretrained
    # checkpoint to load successfully; otherwise they fail.
    load_path, resume_anneal = select_load_path(
        latest_path, load_model, r_file, is_anneal, resume_anneal, anneal_ckpt
    )

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(model_cfgs_mask) * len(model_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=model_tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        n_output_distillation=n_output_distillation,
    )
    target_encoder = copy.deepcopy(encoder)
    embed_dim_encoder = encoder.backbone.embed_dim

    if init_checkpoint is not None:
        if is_anneal:
            raise ValueError(
                "`init_checkpoint` and `is_anneal` cannot both be set; cooldown runs must start from `anneal_ckpt`."
            )
        if load_path is None:
            load_pretrained_weights(init_checkpoint, encoder, predictor, target_encoder)
        else:
            logger.info(f"Resuming from {load_path}; init_checkpoint {init_checkpoint} not used.")

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    transform = make_transforms(
        random_horizontal_flip=random_hflip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=NORMALIZE,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        # clip_len=clip_len,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=data_rank,
        world_size=data_world_size,
        datasets_weights=datasets_weights,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        # Upstream never passed `data.persistent_workers` to the `DataLoader`,
        # so each rank recreated its workers on every dataset pass, adding
        # startup overhead each time.
        persistent_workers=cfgs_data.get("persistent_workers", False),
        in_order=cfgs_data.get("in_order", True),
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        try:
            _dlen = unsupervised_loader.num_batches
        except Exception:
            _dlen = -1
    if ipe is None:
        ipe = _dlen
    logger.info(f"Using batch size of {batch_size}, fpcs of {dataset_fpcs}")
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # zizi

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        use_radamw=use_radamw,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(
        predictor, static_graph=False, find_unused_parameters=True
    )
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    # Trainable encoder and predictor parameters used for gradient-norm measurement and clipping.
    trained_params = [p for m in (encoder, predictor) for p in m.parameters() if p.requires_grad]

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs) + 1)
    )
    lambda_sched = Lambda_LinearWarmupHold(
        lambda_value=lambda_value,
        start_iter=lambda_start_iter,
        end_iter=lambda_end_iter,
    )

    start_epoch = 0
    # Only a job that continues this run's own checkpoint reattaches to its wandb run;
    # a cooldown starting from `anneal_ckpt` is a new run and gets a new wandb run.
    resuming_training = False
    checkpoint_run_id = None
    # Set by `init_wandb` and saved in every checkpoint.
    effective_wandb_run_id = None
    # -- load training checkpoint
    print("Loadind checkpoint from: ", load_path)
    # Upstream entered the checkpoint-loading branch whenever `latest.pth.tar` existed, even if
    # `meta.load_checkpoint` was `False`, then crashed by attempting to load a `None` checkpoint path.
    if load_path is not None:
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
            checkpoint_run_id,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
        )
        resuming_training = not is_anneal or resume_anneal
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch, path):
        def save_on_rank_zero():
            save_dict = {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "epoch": epoch,
                "loss": loss_meter.avg,
                "batch_size": batch_size,
                "world_size": world_size,
                "lr": lr,
                "wandb_run_id": effective_wandb_run_id,
            }
            _atomic_torch_save(save_dict, path)

        _run_rank_zero_checkpoint_save(save_on_rank_zero, rank, device)

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # Open the wandb run once setup has succeeded. Every rank votes, so a failure on one rank
    # stops the whole job instead of leaving the other ranks waiting in a collective.
    wandb_run, wandb_init_error = None, None
    try:
        wandb_run, effective_wandb_run_id = init_wandb(
            cfgs_meta,
            args,
            rank,
            folder,
            resuming_training=resuming_training,
            checkpoint_run_id=checkpoint_run_id,
            # This trainer's checkpoints always hold the run id (None without wandb), so the
            # `wandb_run_id.txt` fallback for older checkpoints never applies.
            checkpoint_has_wandb_run_id=True,
        )
    except Exception as e:
        wandb_init_error = e
    if any_rank_failed(wandb_init_error is not None, device=device):
        if wandb_init_error is not None:
            raise wandb_init_error
        raise RuntimeError("wandb initialization failed on another rank: aborting.")

    trailing_losses = []
    step_count = 0
    # Since the last wandb log, track the largest gradient norm and the number of steps
    # where clipping occurred (how many steps `clip_grad` scaled down).
    grad_norm_max, clipped_steps = float("nan"), 0

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()
        valid_meter = AverageMeter()    # fraction of tokens not built from padding.
        padded_meter = AverageMeter()   # fraction of frames that are black padding.

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    if "airstore" in dataset_type.lower():
                        unsupervised_sampler.increase_epoch()
                    else:
                        unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        iter_retries += 1
                        _log_data_loading_retry(e, iter_retries, NUM_RETRIES)
                        time.sleep(5)
                    else:
                        _raise_data_loading_retries_exhausted(e, NUM_RETRIES)

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred, all_valid = [], [], [], []
                valid_frac, padded_frac = [], []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [
                        [m.to(device, non_blocking=True) for m in masks_enc]
                    ]
                    all_masks_pred += [
                        [m.to(device, non_blocking=True) for m in masks_pred]
                    ]
                    # `udata[-1]`: stores the loader's `clip_indices`; padded black frames are marked as `-1`.
                    # If no padding is present, all tokens are valid and the loss matches the upstream implementation.
                    v = token_validity(udata[-1][0], tubelet_size, grid_size)
                    # Measured on the CPU copy, so logging it never waits on the GPU.
                    valid_frac += [v.mean(dim=(1, 2))]
                    padded_frac += [(udata[-1][0] < 0).float().mean(dim=1)]
                    all_valid += [v.to(device, non_blocking=True)]
                valid_meter.update(float(torch.cat(valid_frac).mean()))
                padded_meter.update(float(torch.cat(padded_frac).mean()))
                return all_clips, all_masks_enc, all_masks_pred, all_valid

            clips, masks_enc, masks_pred, valid = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if wandb_run is not None and epoch == start_epoch:
                if itr == 0 or (log_video_freq > 0 and itr % log_video_freq == 0):
                    log_input_clips(
                        wandb_run,
                        clips,
                        step=epoch * ipe + itr,
                        normalize=NORMALIZE,
                        num_videos=num_log_videos,
                        # Report black padding frames (`clip_index == -1`) separately from repeated real frames.
                        padded=[fpc_sample[0][-1][0] == -1 for fpc_sample in sample],
                    )

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            # Every `log_freq` steps and at the end of each epoch: the same steps on every rank.
            log_step = wandb_enabled and ((itr % log_freq == 0) or (itr == ipe - 1))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(c, embed_dim=embed_dim_encoder):
                    with torch.no_grad():
                        h = target_encoder(c, gram_mode=False, training_mode=True)
                        new_h = []
                        for hi in h:
                            if levels_predictor > 1:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim], (embed_dim,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim : embed_dim * 2],
                                    (embed_dim,),
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, embed_dim * 2 : embed_dim * 3],
                                    (embed_dim,),
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim:], (embed_dim,))
                                hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2)
                                new_h.append(hi_norm)
                            else:
                                new_h.append(F.layer_norm(hi, (hi.size(-1),)))
                        return new_h

                def forward_context(clips, embed_dim=embed_dim_encoder):
                    modality = "video"
                    if img_temporal_dim_size is not None:
                        if clips[0].shape[2] == img_temporal_dim_size:
                            modality = "image"
                    z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
                    z_pred, z_context = predictor(
                        z, masks_enc, masks_pred, mod=modality
                    )
                    if normalize_predictor:
                        z_pred = normalize_nested(z_pred, embed_dim)

                        if predict_all:
                            z_context = normalize_nested(z_context, embed_dim)
                    return z_pred, z_context

                def loss_fn(z, h, masks_to_apply, cls_loss, d_weights, valid):
                    if cls_loss:
                        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
                        h = [
                            apply_masks(hi[:, 1:], mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]
                        loss, n = 0, 0
                        for zi, hi, hi_cls in zip(z, h, h_cls):
                            for zij, hij in zip(zi, hi):
                                h_term = torch.cat([hi_cls, hij], dim=1)
                                loss += (
                                    torch.mean(torch.abs(zij - h_term) ** loss_exp)
                                    / loss_exp
                                )
                                n += 1

                        loss /= n
                        return loss
                    else:
                        h = [
                            apply_masks(hi, mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]
                        # Gather padding weights with the same indices as the target
                        # tokens, so each target retains its corresponding padding weight.
                        v = [
                            apply_masks(vi, mi, concat=False)
                            for vi, mi in zip(valid, masks_to_apply)
                        ]
                        if d_weights is None:
                            d_weights = [[None] * len(zi) for zi in z]

                        loss, n = 0, 0
                        for zi, hi, vi, d_i in zip(z, h, v, d_weights):
                            for zij, hij, vij, d_ij in zip(zi, hi, vi, d_i):
                                loss += masked_l1_loss(
                                    zij, hij, loss_exp, valid=vij, d_weights=d_ij
                                )
                                n += 1
                        loss /= n
                        return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z_pred, z_context = forward_context(clips)
                    loss = 0
                    loss_pred = loss_fn(
                        z_pred,
                        h,
                        masks_pred,
                        cls_loss=has_cls_first,
                        d_weights=None,
                        valid=valid,
                    )
                    loss += loss_pred
                    loss_context, lambda_value_step = None, 0.0

                    # Context loss
                    if predict_all:
                        distance_weights = compute_mask_distance(
                            masks_pred, masks_enc, grid_size, offset_context_loss
                        )
                        if weight_distance_loss:
                            d_weights = distance_weights
                        else:
                            d_weights = None
                        loss_context = loss_fn(
                            z_context,
                            h,
                            masks_enc,
                            cls_loss=False,
                            d_weights=d_weights,
                            valid=valid,
                        )
                        if lambda_progressive:
                            lambda_value_step = lambda_sched.value(epoch * ipe + itr)
                        else:
                            lambda_value_step = lambda_value
                        loss += loss_context * lambda_value_step

                # Measure how different the target encoder’s clip representations are from one another,
                # but only compute this metric on steps that are being logged.
                spread = feature_spread(h, valid) if log_step else (float("nan"), float("nan"))

                _raise_if_non_finite_loss(loss, device)

                # Step 2. Backward & step
                run_step = True
                if loss_reg_std_mult is not None:
                    meanval = np.mean(trailing_losses)
                    stdval = np.std(trailing_losses)
                    max_bound = meanval + loss_reg_std_mult * stdval
                    if (
                        loss > max_bound
                        and epoch > loss_reg_min_epoch
                        and len(trailing_losses)
                        > int(0.5 * loss_reg_num_tracking_steps)
                    ):
                        run_step = False
                        loss.backward()
                        logger.info(
                            f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                        )

                _grad_norm = None
                if run_step:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    # Unscaled gradients, before the optimizer applies them.
                    _grad_norm = grad_norm(encoder, predictor)
                    if clip_grad is not None:
                        clip_grads_with_norm_(trained_params, clip_grad, _grad_norm)
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = min(next(momentum_scheduler), ema[1])
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(
                        encoder.parameters(), target_encoder.parameters()
                    ):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                # For the logs: reading `loss` already synchronizes the step, so reading the others adds no extra wait.
                return (
                    float(loss.detach()),
                    _new_lr,
                    _new_wd,
                    run_step,
                    float(loss_pred.detach()),
                    float("nan") if loss_context is None else float(loss_context.detach()),
                    float(lambda_value_step),
                    float("nan") if _grad_norm is None else float(_grad_norm),
                    m,
                    spread,
                )

            (
                loss,
                _new_lr,
                _new_wd,
                run_step,
                loss_pred,
                loss_context,
                lambda_value_step,
                _grad_norm,
                _ema_m,
                (clip_cosine, effective_rank),
            ), gpu_etime_ms = gpu_timer(train_step)
            if not np.isnan(_grad_norm):        # `NaN` when the step was skipped
                grad_norm_max = _grad_norm if np.isnan(grad_norm_max) else max(grad_norm_max, _grad_norm)
                clipped_steps += int(clip_grad is not None and _grad_norm > clip_grad)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            if loss_reg_std_mult is not None:
                if run_step:
                    trailing_losses.append(loss)
                    if len(trailing_losses) > loss_reg_num_tracking_steps:
                        trailing_losses = trailing_losses[1:]
                else:
                    step_count += 1
                    if step_count > MAX_REPEAT_COUNTS:
                        raise RuntimeError(
                            "Loss is above bound for too many tries. Exiting."
                        )

            # Log to wandb every `log_freq` steps and at each epoch end. All ranks join the reduction.
            # Losses and data fractions are averages over all ranks; the gradient norm is already the
            # same on every rank by DDP.
            if log_step:
                (
                    (global_loss, _),
                    (global_loss_avg, _),
                    (global_loss_pred, _),
                    (global_loss_context, _),
                    (global_valid, _),
                    (global_padded, _),
                    (global_clip_cosine, _),
                    (global_effective_rank, _),
                ) = global_sample_weighted_means(
                    [loss, loss_meter.avg, loss_pred, loss_context, valid_meter.val, padded_meter.val,
                     clip_cosine, effective_rank],
                    [1, loss_meter.count, 1, 1, 1, 1, 1, 1],
                    device=device,
                )
                log_scalars(
                    wandb_run,
                    {
                        "train/loss": global_loss,  # this step: prediction + weighted context loss.
                        "train/loss_avg": global_loss_avg,  # running mean over the epoch so far.
                        "train/loss_pred": global_loss_pred,
                        "train/loss_context": global_loss_context,
                        "train/lambda_context": lambda_value_step,  # weight of the context loss.
                        "train/lr": _new_lr,
                        "train/wd": _new_wd,
                        "train/epoch": epoch + 1,
                        "optimization/grad_norm": _grad_norm,  # encoder + predictor, unscaled.
                        "optimization/grad_norm_max": grad_norm_max,  # largest since the last log.
                        **({"optimization/clipped_steps": clipped_steps} if clip_grad is not None else {}),
                        "optimization/ema_momentum": _ema_m,
                        # Target-encoder features of this batch's clips, per GPU (mean over GPUs).
                        "features/clip_cosine": global_clip_cosine,  # towards 1: clips look alike.
                        "features/effective_rank": global_effective_rank,  # falling: fewer directions in use.
                        "data/valid_token_frac": global_valid,  # tokens counted in the loss.
                        "data/batch_padded_frame_frac": global_padded,  # black padding frames.
                        "time/iter_ms": iter_elapsed_time_ms,
                        "time/gpu_ms": gpu_etime_ms,
                        "time/data_ms": data_elapsed_time_ms,
                        "memory/max_allocated_mb": (
                            torch.cuda.max_memory_allocated() / 1024.0**2 if torch.cuda.is_available() else 0.0
                        ),
                    },
                    step=epoch * ipe + itr,
                )
                grad_norm_max, clipped_steps = float("nan"), 0

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if (
                    (itr % log_freq == 0)
                    or (itr == ipe - 1)
                    or np.isnan(loss)
                    or np.isinf(loss)
                ):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[valid: %.2f] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            "["
                            + ", ".join(
                                [
                                    f"{k}: " + "%.1f" % mask_meters[k].avg
                                    for k in mask_meters
                                ]
                            )
                            + "]",
                            _new_wd,
                            _new_lr,
                            valid_meter.avg,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                # Named after the epoch count stored inside; upstream used the 0-based index.
                save_every_file = f"e{epoch + 1}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

    # If training crashes, wandb closes the run automatically when the process exits.
    finish_wandb(wandb_run)
