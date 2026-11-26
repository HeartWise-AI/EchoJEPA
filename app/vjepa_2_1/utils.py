# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as video_vit
import torch
import torch.nn.functional as F
import yaml
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.datasets.utils import frame_sampling
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import (
    CosineWDSchedule,
    LinearDecaySchedule,
    WarmupCosineSchedule,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into chunks of size `embed_dim` along the last axis (one per
    distilled encoder layer), apply LayerNorm to each chunk, then concatenate back."""
    chunks = [
        F.layer_norm(chunk, (embed_dim,)) for chunk in tensor.split(embed_dim, dim=-1)
    ]
    return torch.cat(chunks, dim=-1)


def normalize_nested(nested, embed_dim):
    """Apply normalize_and_concat to every tensor of nested lists.

    Upstream always descended three list levels, but the predictor returns two
    ([clip length][mask] -> tensor), so it iterated over batch rows and failed.
    """
    if isinstance(nested, torch.Tensor):
        return normalize_and_concat(nested, embed_dim)
    return [normalize_nested(x, embed_dim) for x in nested]


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    import warnings

    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = (
                _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            )
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn(
                    "Configs for online evals must use same number of nodes for slurm-batch processing"
                )

            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu
            _args["pretrain"]["frames_per_clip"] = num_frames
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
    clip_sampling=None,
):
    """Resume training from a checkpoint of this trainer: model weights, optimizer and scaler
    state, and the epoch (0 when a cooldown starts from `anneal_ckpt`). Also returns the
    wandb run id the checkpoint was logged under (`None` if it has none).

    Everything must match the run exactly. Upstream skipped missing tensors, kept the current
    weights for mis-shaped ones and started a fresh optimizer when its groups did not match,
    which silently turned a resume into a partial re-initialisation.

    With `clip_sampling` (this run's `frame_sampling.sampling_record`), a resume of a checkpoint trained with
    other clip-sampling settings is refused (`frame_sampling.check_resume`). A cooldown starting from
    `anneal_ckpt` (`is_anneal`) is a new phase, not a resume, and is not checked.
    """
    logger.info(f"Loading {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    if clip_sampling is not None and not is_anneal:
        frame_sampling.check_resume(checkpoint, clip_sampling, r_path)
    needed = ["encoder", "predictor", "opt", "scaler"]
    needed += [] if target_encoder is None else ["target_encoder"]
    needed += [] if is_anneal else ["epoch"]
    missing = [k for k in needed if k not in checkpoint]
    if missing:
        raise KeyError(
            f"{r_path} is not a training checkpoint of this trainer: it has no {missing}. "
            "To start from another run's weights only, use `meta.init_checkpoint`."
        )
    epoch = 0 if is_anneal else checkpoint["epoch"]

    for name, model in (("encoder", encoder), ("predictor", predictor), ("target_encoder", target_encoder)):
        if model is None:
            continue
        try:
            model.load_state_dict(checkpoint[name], strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"{r_path}: `{name}` does not match the {name} of this run.") from e
        logger.info(f"Loaded {name} from epoch {checkpoint.get('epoch')}, strict.")

    try:
        opt.load_state_dict(checkpoint["opt"])
    except ValueError as e:
        raise ValueError(f"{r_path}: the optimizer state does not match this run's parameter groups.") from e
    if (scaler is None) != (checkpoint["scaler"] is None):
        raise ValueError(
            f"{r_path} was saved {'without' if checkpoint['scaler'] is None else 'with'} a gradient scaler "
            f"and this run uses {'none' if scaler is None else 'one'}; check `meta.dtype`."
        )
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"Loaded optimizer and scaler state; resuming at epoch {epoch}.")
    wandb_run_id = checkpoint.get("wandb_run_id")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
        wandb_run_id,
    )


def _load_matching_shapes(model, state, name):
    """Load the tensors whose name and shape match; keep the model's fresh initialization for the rest."""
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    fresh = sorted(k for k in own if k not in matched)
    unused = sorted(k for k in state if k not in matched)
    model.load_state_dict(matched, strict=False)
    logger.info(f"Loaded {len(matched)}/{len(own)} {name} tensors by name and shape; "
                f"freshly initialized: {fresh}; checkpoint tensors not used: {unused}.")
    if not matched:
        raise RuntimeError(f"No {name} tensor matched the checkpoint.")


def load_pretrained_weights(r_path, encoder, predictor, target_encoder, encoder_key="encoder",
                            partial_predictor=False, partial_encoder=False):
    """Initialize model weights from another run. Loads encoder, predictor, and
    target encoder weights strictly while keeping the optimizer and schedule fresh.
    Call before wrapping models in DDP.

    Args:
        r_path: path to the source checkpoint.
        encoder_key: checkpoint entry that seeds the online encoder (`encoder`, or the EMA
            entry so that student and teacher start from the same released weights).
        partial_predictor: load only the predictor tensors whose shapes match. Needed when the
            source predictor was trained for another target (e.g. Meta's ViT-L distilled from
            ViT-G predicts 1664-d ViT-G features, while deep self-supervision needs 4 x 1024).
    """
    logger.info(f"Initialising weights from {r_path}")
    # Load the checkpoint into CPU memory first rather than immediately putting everything
    # onto the GPU, which is safer (avoid suddenly allocating GPU memory while loading).
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    ema_key = "target_encoder" if "target_encoder" in checkpoint else "ema_encoder"
    if encoder_key == "ema":
        encoder_key = ema_key
    for name, key, model in (
        ("encoder", encoder_key, encoder),
        ("predictor", "predictor", predictor),
        ("target_encoder", ema_key, target_encoder),
    ):
        if key not in checkpoint:
            raise KeyError(
                f"{r_path} has no `{key}` weights for the {name}; "
                f"its entries are {sorted(checkpoint)}."
            )
        # Remove the `module.` prefix from DDP-wrapped model.
        state = {k.removeprefix("module."): v for k, v in checkpoint[key].items()}
        if name == "predictor" and partial_predictor:
            _load_matching_shapes(model, state, name)
            continue
        if name != "predictor" and partial_encoder:
            _load_encoder_from_v2(model, state, name)
            continue
        try:
            # `strict=True`: the source checkpoint and the destination model must match exactly.
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"{r_path}: `{key}` does not match the {name} built from this config; "
                "check `model_name`, `pred_depth`, `pred_embed_dim` and `n_output_distillation`."
            ) from e
        logger.info(f"Loaded {name} from `{key}` (source epoch {checkpoint.get('epoch')}), strict.")
    logger.info("Weights only: optimizer, scaler, schedules and epoch start fresh.")
    del checkpoint


def _load_encoder_from_v2(model, state, name):
    """Load a V-JEPA 2.0-recipe encoder (EchoJEPA) into a V-JEPA 2.1 encoder (see
    `load_pretrained_weights(partial_encoder=True)`)."""
    own = model.state_dict()
    levels = sorted({k.split(".")[2] for k in own if k.startswith("backbone.norms_block.")}, key=int)
    remap = {}
    for k, v in state.items():
        if k.startswith("backbone.norm.") and levels:
            remap[k.replace("backbone.norm.", f"backbone.norms_block.{levels[-1]}.")] = v
        else:
            remap[k] = v
    matched = {k: v for k, v in remap.items() if k in own and own[k].shape == v.shape}
    required = [k for k in own if k.startswith(("backbone.blocks.", "backbone.patch_embed."))]
    missing_required = [k for k in required if k not in matched]
    if missing_required:
        raise RuntimeError(f"{name}: 2.0 checkpoint is missing required tensors {missing_required[:10]}.")
    unused = sorted(k for k in remap if k not in matched)
    if unused:
        raise RuntimeError(f"{name}: checkpoint tensors with no destination {unused[:10]}.")
    model.load_state_dict(matched, strict=False)
    fresh = sorted(k for k in own if k not in matched)
    logger.info(f"Loaded {len(matched)}/{len(own)} {name} tensors from a 2.0-recipe encoder; fresh: {fresh}.")


def load_frozen_teacher(r_path, model, encoder_key="ema"):
    """Frozen distillation teacher built from its own model config (may differ from the online
    encoder, e.g. Meta's V-JEPA 2.1 teaching an EchoJEPA student); loaded strictly."""
    return load_frozen_anchor(r_path, model, encoder_key=encoder_key)


def load_frozen_anchor(r_path, model, encoder_key="ema"):
    """Load the released encoder into `model` and freeze it, as a fixed anchor for continued
    pretraining (the online encoder is pulled towards its features so it cannot drift into a
    collapsed solution). Loaded from `r_path` itself, so it is the same on a fresh start and on
    a resume (where the online encoder comes from `latest.pth.tar`)."""
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    if encoder_key == "ema":
        encoder_key = "target_encoder" if "target_encoder" in checkpoint else "ema_encoder"
    state = {k.removeprefix("module."): v for k, v in checkpoint[encoder_key].items()}
    model.load_state_dict(state, strict=True)
    del checkpoint
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"Frozen anchor encoder loaded from `{encoder_key}` of {r_path}, strict.")
    return model


class AnchorProjector(torch.nn.Module):
    """One Linear(embed_dim, embed_dim) per output level, mapping the student's layer-normed tokens
    [..., levels, embed_dim] into the frozen teacher's feature space. A single forward call per step,
    so it can be wrapped in DistributedDataParallel (multi-GPU gradient averaging)."""

    def __init__(self, levels, embed_dim):
        super().__init__()
        self.proj = torch.nn.ModuleList([torch.nn.Linear(embed_dim, embed_dim) for _ in range(levels)])

    def forward(self, x):
        if x.size(-2) != len(self.proj):
            raise ValueError(f"AnchorProjector has {len(self.proj)} levels, input has {x.size(-2)}.")
        return torch.stack([p(x[..., k, :]) for k, p in enumerate(self.proj)], dim=-2)


def anchor_loss(z, a, embed_dim, valid=None, proj=None):
    """Feature-distillation anchor: mean (1 - cosine) between the online encoder's context
    tokens `z` and the frozen anchor's tokens `a` for the same visible tokens, per output level
    (each level layer-normed, as the JEPA targets). Same nesting as the encoder output
    (list per clip length of list per mask of [B, N, levels * embed_dim]).

    `valid`: matching token-validity weights ([B, N, 1], same nesting) or None (every token counts).
    `proj`: optional `AnchorProjector` (or its DDP wrapper), applied to the student's layer-normed
    tokens [B, N, levels, embed_dim] before the cosine (needed when the teacher is another model).
    """
    loss, n = 0.0, 0
    for i, (zi, ai) in enumerate(zip(z, a)):
        for j, (zij, aij) in enumerate(zip(zi, ai)):
            levels = zij.size(-1) // embed_dim
            zl = F.layer_norm(zij.float().unflatten(-1, (levels, embed_dim)), (embed_dim,))
            if proj is not None:  # learned student->teacher map (teacher in another feature space)
                zl = proj(zl)
            al = F.layer_norm(aij.float().unflatten(-1, (levels, embed_dim)), (embed_dim,))
            dist = 1.0 - F.cosine_similarity(zl, al, dim=-1)  # [B, N, levels]
            if valid is None:
                loss = loss + dist.mean()
            else:
                w = valid[i][j].float()  # [B, N, 1]
                loss = loss + (dist * w).sum() / (w.sum() * levels).clamp(min=1)
            n += 1
    return loss / max(n, 1)


def anchor_weight(step, weight, final_weight, start_iter, total_iters):
    """Anchor weight: 0 before `start_iter` (encoder frozen, nothing to anchor), then linear
    decay from `weight` at `start_iter` to `final_weight` at `total_iters`, held after."""
    if weight <= 0 or step < start_iter:
        return 0.0
    t = min(1.0, (step - start_iter) / max(1, total_iters - start_iter))
    return float(weight + t * (final_weight - weight))


def select_load_path(latest_path, load_model, r_file, is_anneal, resume_anneal, anneal_ckpt):
    """Select the checkpoint to load. Annealing always loads a checkpoint; normal training may
    start fresh.

    Returns:
        (checkpoint_path, resumed_anneal)
    """
    # If it's a cooldown run, it must load something because a cooldown run is never allowed to
    # start from random weights.
    if is_anneal:
        # A cooldown loads `anneal_ckpt` or its own latest checkpoint; a named file would be ignored.
        if r_file is not None:
            raise ValueError(
                "A cooldown starts from `optimization.anneal_ckpt` or resumes its own latest.pth.tar; "
                "remove `meta.read_checkpoint`."
            )
        # If the cooldown started previously, continue the cooldown from where it stopped.
        if resume_anneal and os.path.exists(latest_path):
            return latest_path, True
        # Otherwise, start from checkpoint `anneal_ckpt` (starting point of cooldown training).
        if not os.path.exists(anneal_ckpt):
            raise FileNotFoundError(f"Cooldown needs `anneal_ckpt`, not found: {anneal_ckpt}.")
        return anneal_ckpt, False
    # Otherwise, this is a normal training run.
    # A named checkpoint is a resume, so it needs `load_checkpoint`; upstream silently ignored it.
    if r_file is not None and not load_model:
        raise ValueError(
            f"`meta.read_checkpoint` ({r_file}) needs `meta.load_checkpoint: true`; "
            "set it to resume from that file, or remove `meta.read_checkpoint`."
        )
    # Start using its normal fresh initialization if `load_model=False`.
    if not load_model:
        return None, resume_anneal
    # Load the checkpoint otherwise.
    if r_file is not None:
        if not os.path.exists(r_file):
            raise FileNotFoundError(f"`read_checkpoint` not found: {r_file}.")
        return r_file, resume_anneal
    return (latest_path if os.path.exists(latest_path) else None), resume_anneal


def token_validity(clip_indices, tubelet_size, grid_size):
    """Return 1.0 for tokens containing real frames and 0.0 for pure padding.

    Padding frames use clip index `-1`. A tubelet is valid if it contains at
    least one real frame.

    Args:
        clip_indices: frame indices [B, T], with `-1` for padding.

    Returns:
        Validity mask [B, T' * grid_size**2, 1].
    """
    # Change indices into Boolean mask: mask out padded frames (`-1`).
    real = clip_indices >= 0
    B, T = real.shape
    if T % tubelet_size:
        raise ValueError(
            f"Clips of {T} frames do not split into tubelets of {tubelet_size}; "
            "every `data.dataset_fpcs` entry must be a multiple of `data.tubelet_size`."
        )
    # Keep tubelets that contain at least one real frame.
    real = real.view(B, T // tubelet_size, tubelet_size).any(dim=2)
    return real.repeat_interleave(grid_size * grid_size, dim=1).unsqueeze(-1).float()


def masked_l1_loss(z, h, loss_exp, valid=None, d_weights=None):
    """Calculate the actual prediction loss between the predictor's prediction and the
    target encoder's target: mean |z - h|^p / p over valid tokens. Optionally ignores
    padded tokens and applies inverse distance weighting.

    Args:
        z, h: prediction and target embeddings [B, K, D].
        valid: optional token-validity mask [B, K, 1].
        d_weights: optional distance weights [B, K].

    Returns:
        Scalar prediction loss.
    """
    err = torch.abs(z - h) ** loss_exp
    if d_weights is not None:
        err = err * (1 / d_weights.unsqueeze(2))
    if valid is None:
        return torch.mean(err) / loss_exp
    n = (valid.sum() * err.size(-1)).clamp(min=1)
    return (err * valid).sum() / n / loss_exp


def feature_spread(features, valid=None):
    """Measure how different the target encoder's clips are from each other, to detect the
    representation collapse that a falling loss can hide (features that barely differ are
    easy to predict).

    Args:
        features: target features [B, N, D], or a list of them (one per clip length).
        valid: matching token-validity masks [B, N, 1], or None to pool every token.

    Returns:
        (clip_cosine, effective_rank) as floats; `NaN` when there are fewer than two clips.
    """
    # Normalize input into a list.
    if isinstance(features, torch.Tensor):
        features, valid = [features], [valid]
    if valid is None:
        valid = [None] * len(features)
    
    # Each clip is mean-pooled over its valid tokens.
    pooled = []
    with torch.no_grad():
        # Padded tokens are excluded.
        for h, v in zip(features, valid):
            h = h.detach().float()
            if v is None:
                pooled.append(h.mean(dim=1))
            else:
                v = v.detach().float()
                pooled.append((h * v).sum(dim=1) / v.sum(dim=1).clamp(min=1))
        # Values (clip cos & effective rank) depend on the number of clips, so compare
        # runs with the same batch size.
        pooled = torch.cat(pooled)
        n = pooled.size(0)
        if n < 2:
            return float("nan"), float("nan")
        # Clip cosine: mean cosine similarity between different clips; it rises towards `1`
        # as every clip maps to the same features.
        unit = F.normalize(pooled, dim=1)
        sim = unit @ unit.T
        clip_cosine = (sim.sum() - sim.diagonal().sum()) / (n * (n - 1))
        # Effective rank: measures the number of meaningful independent directions the
        # clip features span; exp of the entropy of the singular values of the centered
        # pooled features. Ranges from 0 (all clips identical) to at most `num_clips - 1`.
        s = torch.linalg.svdvals(pooled - pooled.mean(dim=0, keepdim=True))
        if s.sum() <= 1e-6 * pooled.norm():
            return float(clip_cosine), 0.0
        p = s / s.sum()
        p = p[p > 0]
        effective_rank = torch.exp(-(p * p.log()).sum())
    return float(clip_cosine), float(effective_rank)


def clip_grads_with_norm_(parameters, max_norm, total_norm):
    """Standard gradient clipping: scale gradients in place so their total norm does not
    exceed `max_norm` for one training step. Uses the precomputed `total_norm` to determine
    the scaling factor.
    """
    total_norm = torch.as_tensor(total_norm, dtype=torch.float32)
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    with torch.no_grad():
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef.to(p.grad.device))


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
    n_output_distillation=4,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        n_output_distillation=n_output_distillation,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=(
            encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads
        ),
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
        n_output_distillation=n_output_distillation,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    use_radamw=False,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    if use_radamw:
        from src.utils.adamw import AdamW as RAdamW

        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
