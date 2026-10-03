# app/vjepa_2_1/check_checkpoint.py

"""Validate a V-JEPA 2.1 checkpoint against a training config.

Builds the encoder, predictor, and target encoder exactly as `train.py` does, then
checks that the checkpoint matches all three models and can be loaded with the same
strict loader used by `meta.init_checkpoint`.

If `--sha256` is given, the checkpoint is loaded only when its checksum matches.
With `--forward`, also runs a deterministic masked training forward pass on CPU and
verifies that the resulting prediction/context loss is finite. No training or GPU is
required.

    python -m app.vjepa_2_1.check_checkpoint \\
        --fname configs/train_2_1/vitb16/pretrain-continued-224px-16f.yaml \\
        --checkpoint /path/to/vjepa2_1_vitb_mimic_pt169_c60.pt \\
        --sha256 40384a17d3d32375142e3bd3a43d2f5bfa2b174b44c6960958bffd731e9624c1 \\
        --forward

If `--checkpoint` is omitted, `meta.init_checkpoint` from the config is used.
Exits with 0 on success and 1 on failure.
"""

import argparse
import copy
import hashlib
import logging
import os
import sys

import torch
import yaml

from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.utils import (
    init_video_model,
    load_pretrained_weights,
    masked_l1_loss,
    normalize_and_concat,
    normalize_nested,
    token_validity,
)
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger()

# A training checkpoint stores both model weights and training state.
# `init_checkpoint` loads only the model weights and ignores the saved
# training state, so training starts as a new run.
RUN_STATE = ("epoch", "itr", "loss", "batch_size", "world_size", "lr")


def sha256sum(path, chunk_size=1 << 24):
    """Compute the checkpoint's SHA-256 hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_kwargs(cfg):
    """Build `init_video_model` arguments from the training config, following the same logic as `train.py`.
    Video-only configs without an `img_data` section are also supported, as in `train.py`'s model construction
    for them.
    """
    cfgs_model = cfg["model"]
    cfgs_data = cfg["data"]
    cfgs_loss = cfg["loss"]
    dataset_fpcs = cfgs_data["dataset_fpcs"]
    return dict(
        uniform_power=cfgs_model.get("uniform_power", False),
        use_mask_tokens=cfgs_model.get("use_mask_tokens", False),
        num_mask_tokens=int(len(cfg["mask"]) * len(dataset_fpcs)),  # decide number of mask-token sets the model needs.
        zero_init_mask_tokens=cfgs_model.get("zero_init_mask_tokens", True),
        patch_size=cfgs_data.get("patch_size"),
        max_num_frames=max(dataset_fpcs),               # model is built to support the largest configured clip length.
        tubelet_size=cfgs_data.get("tubelet_size"),
        model_name=cfgs_model.get("model_name"),
        crop_size=cfgs_data.get("crop_size", 224),
        pred_depth=cfgs_model.get("pred_depth"),
        pred_num_heads=cfgs_model.get("pred_num_heads", None),
        pred_embed_dim=cfgs_model.get("pred_embed_dim"),
        is_causal=cfgs_model.get("is_causal", False),
        pred_is_causal=cfgs_model.get("pred_is_causal", False),
        use_sdpa=cfg["meta"].get("use_sdpa", False),
        use_silu=cfgs_model.get("use_silu", False),
        use_pred_silu=cfgs_model.get("use_pred_silu", False),
        wide_silu=cfgs_model.get("wide_silu", True),
        use_rope=cfgs_model.get("use_rope", False),
        use_activation_checkpointing=cfgs_model.get("use_activation_checkpointing", False),
        return_all_tokens=cfgs_loss.get("predict_all", True),
        chop_last_n_tokens=cfgs_loss.get("shift_by_n"),
        init_type=cfgs_model.get("init_type", "default"),
        img_temporal_dim_size=cfgs_model.get("img_temporal_dim_size", None),
        n_registers=cfgs_model.get("n_registers", 0),
        n_registers_predictor=cfgs_model.get("n_registers_predictor", 0),
        has_cls_first=cfgs_model.get("has_cls_first", False),
        interpolate_rope=cfgs_model.get("interpolate_rope", False),
        modality_embedding=cfgs_model.get("modality_embedding", False),
        n_output_distillation=cfgs_model.get("n_output_distillation", 4),
    )


def build_models(cfg):
    """Build the encoder, predictor, and target encoder (initialized on CPU)."""
    encoder, predictor = init_video_model(torch.device("cpu"), **model_kwargs(cfg))
    return encoder, predictor, copy.deepcopy(encoder)


def compare_state(model, state):
    """Compare the model and checkpoint keys after removing the DDP `module.` prefix.
    Report missing keys, unexpected keys, and keys with mismatched shapes.
    """
    # Distributed training frequently wraps models with PyTorch DDP.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    expected = model.state_dict()
    return {
        "missing": sorted(set(expected) - set(state)),      # model expects but checkpoint doesn't contain.
        "unexpected": sorted(set(state) - set(expected)),   # checkpoint contains but model doesn't expect.
        "mismatched": sorted(                               # mis-shaped: both have it, but tensor dimensions differ.
            (k, tuple(state[k].shape), tuple(v.shape))
            for k, v in expected.items()
            if k in state and state[k].shape != v.shape
        ),
    }


def count_parameters(model):
    """Count all model parameters and the parameters used only by the image path.

    Note: image-only parameters are identified by `img` in their names.
    """
    named = list(model.named_parameters())
    return {
        "total": sum(p.numel() for _, p in named),
        "image_only": sum(p.numel() for n, p in named if "img" in n),
    }


def loss_fn(z, h, masks_to_apply, loss_exp, cls_loss, d_weights, valid):
    """Match `train.py`'s training loss, using the provided `loss_exp`. Keep this in sync with the trainer,
    since tests compare the two. `valid` identifies tokens that contain at least some real video data,
    rather than tokens made entirely from padded frames."""
    if cls_loss:
        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
        h = [apply_masks(hi[:, 1:], mi, concat=False) for hi, mi in zip(h, masks_to_apply)]
        loss, n = 0, 0
        for zi, hi, hi_cls in zip(z, h, h_cls):
            for zij, hij in zip(zi, hi):
                h_term = torch.cat([hi_cls, hij], dim=1)
                loss += torch.mean(torch.abs(zij - h_term) ** loss_exp) / loss_exp
                n += 1
        return loss / n
    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_to_apply)]
    v = [apply_masks(vi, mi, concat=False) for vi, mi in zip(valid, masks_to_apply)]
    if d_weights is None:
        d_weights = [[None] * len(zi) for zi in z]
    loss, n = 0, 0
    for zi, hi, vi, d_i in zip(z, h, v, d_weights):
        for zij, hij, vij, d_ij in zip(zi, hi, vi, d_i):
            loss += masked_l1_loss(zij, hij, loss_exp, valid=vij, d_weights=d_ij)
            n += 1
    return loss / n


def masked_forward(cfg, encoder, predictor, target_encoder, seed=0, batch_size=2):
    """Run one training forward pass on deterministic random clips in float32 on CPU (as in `train.py`).
    Reproduce `train.py`'s masks, target/context/predictor forward passes, prediction loss, and optional
    final-weight context loss.
    `seed` fixes the clips and the masks, so repeated runs report the same numbers. Use at least 2 clips
    to avoid mask-distance squeezing.
    """
    # Read relevant config values.
    data, cfgs_model, cfgs_loss = cfg["data"], cfg["model"], cfg["loss"]
    # Check the maximum configured video length.
    fpc, crop = max(data["dataset_fpcs"]), data.get("crop_size", 224)
    # L1 loss.
    loss_exp = cfgs_loss.get("loss_exp", 1.0)
    predict_all = cfgs_loss.get("predict_all", True)
    # Find the embedding dimension.
    embed_dim = encoder.backbone.embed_dim
    # Construct the exact mask generator used in training.
    collator = MaskCollator(
        cfgs_mask=cfg["mask"],
        dataset_fpcs=[fpc],
        crop_size=crop,
        patch_size=data["patch_size"],
        tubelet_size=data["tubelet_size"],
    )
    # Seed the global generator used for mask positions, while preserving the caller's random state.
    # Block sizes remain controlled by the collator's own step counter.
    with torch.random.fork_rng(devices=[]):  # CPU only: masks and clips are made on CPU
        torch.manual_seed(seed)
        # Generate mask pairs: (encoder, predictor).
        masks = [generator(batch_size) for generator in collator.mask_generators[fpc]]
        # Create fake training video.
        clips = [torch.randn(batch_size, 3, fpc, crop, crop)]
    masks_enc, masks_pred = [[m[0] for m in masks]], [[m[1] for m in masks]]
    # Every frame of these clips is real, so every token counts in the loss.
    valid = [token_validity(torch.arange(fpc).expand(batch_size, fpc), data["tubelet_size"], crop // data["patch_size"])]
    # Determine image vs. video mode.
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size")
    modality = "image" if img_temporal_dim_size is not None and fpc == img_temporal_dim_size else "video"
    # Put all models in eval mode to prevent training-mode behavior such as dropout.
    for model in (encoder, predictor, target_encoder):
        model.eval()
    with torch.no_grad():
        # As `train.py`'s forward_target: one LayerNorm per distilled layer.
        h = [normalize_and_concat(hi, embed_dim) for hi in target_encoder(clips, gram_mode=False, training_mode=True)]
        z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
        z_pred, z_context = predictor(z, masks_enc, masks_pred, mod=modality)
        if cfgs_model.get("normalize_predictor", False):
            z_pred = normalize_nested(z_pred, embed_dim)
            if predict_all:
                z_context = normalize_nested(z_context, embed_dim)
        loss_pred = loss_fn(z_pred, h, masks_pred, loss_exp, cls_loss=cfgs_model.get("has_cls_first", False), d_weights=None, valid=valid)
        loss = {"prediction": float(loss_pred), "context": None, "context_weight": 0.0, "distance_weighted": False}
        total = loss_pred
        # `if predict_all`: `total = loss_pred + weight * loss_context`.
        if predict_all:
            distance_weighted = cfgs_loss.get("weight_distance_loss", False)
            d_weights = None
            if distance_weighted:
                grid_size = crop // data["patch_size"]
                d_weights = compute_mask_distance(
                    masks_pred, masks_enc, grid_size, cfgs_loss.get("offset_context_loss", False)
                )
            loss_context = loss_fn(z_context, h, masks_enc, loss_exp, cls_loss=False, d_weights=d_weights, valid=valid)
            # Use the final pretraining ramp weight, which is also the constant cooldown weight.
            weight = cfgs_model.get("lambda_value_vid", 0.0)
            total = loss_pred + weight * loss_context
            loss.update(context=float(loss_context), context_weight=weight, distance_weighted=distance_weighted)
    loss["total"] = float(total)
    return {
        "input": tuple(clips[0].shape),
        "seed": seed,
        "tokens": [(m_enc.shape[1], m_pred.shape[1]) for m_enc, m_pred in zip(masks_enc[0], masks_pred[0])],
        "loss": loss,
        "finite": bool(torch.isfinite(total)),      # NaN in either loss makes the total NaN.
    }


def check(cfg, checkpoint_path, expected_sha256=None, forward=False):
    """Compare a checkpoint against the models built from the training config.

    Returns `ok=True` only if the checksum matches (when provided) and all models load successfully.
    """
    # If no `--sha256` given, checksum mismatch cannot make the test fail; otherwise, it must match exactly.
    report = {"checkpoint": checkpoint_path, "ok": False, "strict_load": None}
    report["sha256"] = sha256sum(checkpoint_path)
    report["sha256_expected"] = expected_sha256
    report["sha256_ok"] = expected_sha256 is None or report["sha256"] == expected_sha256.lower()
    if not report["sha256_ok"]:
        # Skip deserialization if this is not the expected file.
        # Older PyTorch versions may unpickle arbitrary objects when loading checkpoints.
        return report

    # Build models from the config.
    encoder, predictor, target_encoder = build_models(cfg)
    report["parameters"] = {"encoder": count_parameters(encoder), "predictor": count_parameters(predictor)}

    # Load the checkpoint; a checkpoint saved from GPU training can still be inspected without a GPU.
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    # Inspect the top-level checkpoint entries.
    report["entries"] = sorted(checkpoint)
    # Report training state.
    report["training_state"] = {k: checkpoint[k] for k in RUN_STATE if k in checkpoint}
    if isinstance(checkpoint.get("opt"), dict) and checkpoint["opt"].get("param_groups"):
        report["training_state"]["opt_lr"] = checkpoint["opt"]["param_groups"][0].get("lr")
    # Different checkpoint versions/naming conventions may store the EMA model as either
    # `target_encoder` or `ema_encoder`.
    ema_key = "target_encoder" if "target_encoder" in checkpoint else "ema_encoder"
    report["models"] = {}
    # Check the three model components separately.
    for name, key, model in (
        ("encoder", "encoder", encoder),
        ("predictor", "predictor", predictor),
        ("target_encoder", ema_key, target_encoder),
    ):
        if key not in checkpoint:
            report["models"][name] = {"source": None}
            continue
        report["models"][name] = {"source": key, **compare_state(model, checkpoint[key])}
    del checkpoint

    keys_match = all(
        m["source"] is not None and not (m["missing"] or m["unexpected"] or m["mismatched"])
        for m in report["models"].values()
    )
    report["strict_load"] = None
    # Test the real initialization path, not an approximate imitation of it.
    if keys_match:
        try:
            load_pretrained_weights(checkpoint_path, encoder, predictor, target_encoder)
            report["strict_load"] = "ok"
        except (KeyError, RuntimeError) as e:
            report["strict_load"] = f"failed: {e}"

    # If run `--forward`, make a fake video to test the backbone.
    if forward and report["strict_load"] == "ok":
        try:
            report["forward"] = masked_forward(cfg, encoder, predictor, target_encoder)
        except RuntimeError as e:  # e.g. predictions and targets of different shapes.
            report["forward"] = {"error": str(e), "finite": False}

    forward_ok = report.get("forward", {}).get("finite", True)
    report["ok"] = report["sha256_ok"] and report["strict_load"] == "ok" and forward_ok
    return report


def print_report(report):
    """Turn the report into readable output."""
    print(f"Checkpoint: {os.path.basename(report['checkpoint'])}")
    if report["sha256_expected"] is None:
        verdict = "no --sha256 given"
    else:
        verdict = "matches --sha256" if report["sha256_ok"] else f"DOES NOT MATCH --sha256 {report['sha256_expected']}"
    print(f"sha256:     {report['sha256']} ({verdict})")
    if "entries" not in report:
        print("Not loaded: the file's SHA-256 hash does not match the expected value.")
        print("RESULT: FAIL")
        return
    print(f"Entries:    {', '.join(report['entries'])}")
    state = ", ".join(f"{k}={v}" for k, v in report["training_state"].items())
    print(f"Training state in the file (never read by `init_checkpoint`): {state}.")
    for name, p in report["parameters"].items():
        print(f"{name} parameters: {p['total']:,} ({p['image_only']:,} used only for images)")
    for name, m in report["models"].items():
        if m["source"] is None:
            print(f"{name}: NO ENTRY in the checkpoint")
            continue
        print(f"{name} <- `{m['source']}`: {len(m['missing'])} missing, {len(m['unexpected'])} unexpected, "
              f"{len(m['mismatched'])} mis-shaped keys")
        for k in m["missing"]:
            print(f"    missing:    {k}")
        for k in m["unexpected"]:
            print(f"    unexpected: {k}")
        for k, got, want in m["mismatched"]:
            print(f"    mis-shaped: {k}: checkpoint {got}, model {want}")
    print(f"Strict load with the trainer's loader: {report['strict_load'] or 'not attempted (keys differ)'}.")
    if "forward" in report:
        f = report["forward"]
        if "error" in f:
            print(f"Masked forward: FAILED: {f['error']}")
        else:
            tokens = "; ".join(
                f"mask {i + 1}: {e} context / {p} predicted tokens" for i, (e, p) in enumerate(f["tokens"])
            )
            loss = f["loss"]
            print(f"Masked forward on random clips {f['input']}, seed {f['seed']}: {tokens}.")
            if loss["context"] is None:
                print(f"    prediction loss {loss['prediction']:.4f} (no context loss: predict_all is off)")
            else:
                weighting = "distance-weighted" if loss["distance_weighted"] else "not distance-weighted"
                print(f"    prediction loss {loss['prediction']:.4f} + {loss['context_weight']} x context loss "
                      f"{loss['context']:.4f} ({weighting}) = {loss['total']:.4f}")
            print(f"    finite={f['finite']}")
    print("RESULT: PASS" if report["ok"] else "RESULT: FAIL")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fname", required=True, help="V-JEPA 2.1 training config yaml.")
    p.add_argument("--checkpoint", default=None, help="Checkpoint to check (default: `meta.init_checkpoint`).")
    p.add_argument("--sha256", default=None, help="Expected SHA-256 of the checkpoint file.")
    p.add_argument("--forward", action="store_true", help="Also run one masked training forward pass on random clips.")
    args = p.parse_args(argv)

    with open(args.fname) as f:
        cfg = yaml.safe_load(f)
    path = args.checkpoint or cfg["meta"].get("init_checkpoint")
    if path is None:
        p.error("The config has no `meta.init_checkpoint`; provide the checkpoint explicitly with `--checkpoint`.")
    if not os.path.isfile(path):
        p.error(f"Checkpoint not found: {path}")

    previous = logging.root.manager.disable
    logging.disable(logging.INFO)   # Suppress the full model printouts logged by `init_video_model`.
    try:
        report = check(cfg, path, expected_sha256=args.sha256, forward=args.forward)
    finally:
        logging.disable(previous)
    print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
