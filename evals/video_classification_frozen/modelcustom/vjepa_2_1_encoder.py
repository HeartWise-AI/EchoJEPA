# evals/video_classification_frozen/modelcustom/vjepa_2_1_encoder.py

"""V-JEPA 2.1 encoder adapter for frozen-probe evaluation.

Constructs the V-JEPA 2.1 encoder from eval/config settings, strictly loads checkpoint weights after
removing training-wrapper prefixes, and returns the encoder wrapped with `ClipAggregation` for
multiclip evaluation. The encoder returns its last layer, normalized as in the training targets.
"""

import logging

import app.vjepa_2_1.models.vision_transformer as vit
from evals.video_classification_frozen.modelcustom.vit_encoder_multiclip import ClipAggregation
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger()

# The whitelist of configuration fields that are allowed to be passed into the vjepa-2.1 encoder:
# the code deliberately validates these options so a typo or unsupported field doesn't silently get ignored.
ENCODER_OPTIONS = (
    "patch_size",
    "tubelet_size",
    "uniform_power",
    "use_sdpa",
    "use_silu",
    "wide_silu",
    "is_causal",
    "use_rope",
    "init_type",
    "img_temporal_dim_size",
    "n_registers",
    "has_cls_first",
    "interpolate_rope",
    "modality_embedding",
    "n_output_distillation",
)


def init_module(resolution, frames_per_clip, checkpoint, model_kwargs, wrapper_kwargs):
    """Given an evaluation configuration and a checkpoint, construct the exact V-JEPA 2.1 encoder
    and return the wrapped encoder expected by the frozen-evaluation framework."""
    # Copy the encoder config.
    kwargs = dict(model_kwargs["encoder"])
    model_name = kwargs.pop("model_name")
    # Unless instructed otherwise, evaluate the learned EMA target encoder.
    checkpoint_key = kwargs.pop("checkpoint_key", "target_encoder")
    # Avoid accidental architecture mismatch.
    unknown = sorted(set(kwargs) - set(ENCODER_OPTIONS))
    if unknown:
        raise ValueError(f"Unknown V-JEPA 2.1 encoder options {unknown}; expected some of {ENCODER_OPTIONS}.")

    # Construct the vjepa-2.1 encoder.
    model = vit.__dict__[model_name](img_size=resolution, num_frames=frames_per_clip, **kwargs)
    # Load the checkpoint.
    logger.info(f"Loading `{checkpoint_key}` from {checkpoint}")
    saved = robust_checkpoint_loader(checkpoint, map_location="cpu")
    if checkpoint_key not in saved:
        raise KeyError(f"{checkpoint} has no `{checkpoint_key}`; its entries are {sorted(saved)}.")
    # Handle wrapper prefixes introduced during training: during DDP(MultiSeqWrapper(encoder)), the actual
    # encoder may have been wrapped (keys start with `module.backbone.`).
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in saved[checkpoint_key].items()}
    # Strict: a probe must never train on an encoder that only partly loaded.
    # Every expected model parameter must exist in the checkpoint, and every checkpoint parameter must correspond to the model.
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded `{checkpoint_key}` (epoch {saved.get('epoch', 'n/a')}), strict.")
    del saved

    return ClipAggregation(model, tubelet_size=model.tubelet_size, **(wrapper_kwargs or {}))
