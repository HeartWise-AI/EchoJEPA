# evals/video_classification_frozen/modelcustom/deepcoro_clip_echojepa_encoder.py

"""The EchoJEPA ViT out of a DeepCORO_CLIP checkpoint, for the probes.

DeepCORO_CLIP's `echojepa` backbone keeps the EchoJEPA encoder under `video_encoder` as
`model.encoder.<vit parameter>` (next to the CLIP aggregator and projection, which are not
used here). This adapter reads that checkpoint, keeps the encoder's weights, loads them
strictly into the ViT built from `pretrain_kwargs.encoder` (as `vit_encoder_multiclip` does) and
wraps it with `ClipAggregation`. Use it to probe or fine-tune the encoder that DeepCORO_CLIP's
contrastive stage produced, with the same manifests and settings as an SSL EchoJEPA encoder.

`pretrain_kwargs.encoder` also accepts `state_dict_key` (default `video_encoder`: the entry of
the checkpoint holding the state dict, `null` for a bare state dict) and `encoder_prefix`
(default `model.encoder.`: the prefix of the encoder's keys inside it).
"""

import logging

import src.models.vision_transformer as vit
from evals.video_classification_frozen.modelcustom.vit_encoder_multiclip import ClipAggregation
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger()


def encoder_state(checkpoint, state_dict_key="video_encoder", encoder_prefix="model.encoder."):
    """The encoder's parameters, their prefix removed, out of a loaded DeepCORO_CLIP checkpoint."""
    state = checkpoint
    if state_dict_key:
        if state_dict_key not in checkpoint:
            raise KeyError(f"The checkpoint has no `{state_dict_key}` entry; its entries are {sorted(checkpoint)[:20]}.")
        state = checkpoint[state_dict_key]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    encoder = {k[len(encoder_prefix):]: v for k, v in state.items() if k.startswith(encoder_prefix)}
    if not encoder:
        prefixes = sorted({".".join(k.split(".")[:2]) for k in state})
        raise KeyError(f"No parameter starts with `{encoder_prefix}`; the state dict's prefixes are {prefixes}.")
    others = sorted({".".join(k.split(".")[:1]) for k in state if not k.startswith(encoder_prefix)})
    logger.info(f"Kept {len(encoder)} encoder parameters under `{encoder_prefix}`; left out {others}.")
    return encoder


def init_module(resolution, frames_per_clip, checkpoint, model_kwargs, wrapper_kwargs):
    kwargs = dict(model_kwargs["encoder"])
    model_name = kwargs.pop("model_name")
    kwargs.pop("checkpoint_key", None)  # not used: the encoder is read from the DeepCORO_CLIP entry.
    state_dict_key = kwargs.pop("state_dict_key", "video_encoder")
    encoder_prefix = kwargs.pop("encoder_prefix", "model.encoder.")
    model = vit.__dict__[model_name](img_size=resolution, num_frames=frames_per_clip, **kwargs)

    logger.info(f"Loading the EchoJEPA encoder out of the DeepCORO_CLIP checkpoint {checkpoint}")
    saved = robust_checkpoint_loader(checkpoint, map_location="cpu")
    state = encoder_state(saved, state_dict_key, encoder_prefix)
    # Strict: a probe must never run on an encoder that only partly loaded.
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded the encoder (DeepCORO_CLIP epoch {saved.get('epoch', 'n/a')}), strict.")
    del saved
    return ClipAggregation(model, tubelet_size=model.tubelet_size, **(wrapper_kwargs or {}))
