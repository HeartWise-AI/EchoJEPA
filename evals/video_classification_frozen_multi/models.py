# evals/video_classification_frozen_multi/models.py

"""The frozen encoder and the probe heads of the multi-view probe.

`SlotEncoder` runs the frozen single-view encoder (`ClipAggregation` around the backbone, as the EF
probe builds it) on the videos a batch holds and places each video's tokens at its slot position;
positions without a video stay empty and are masked out downstream.

`EarlyFusionRegressor` (the primary probe) adds a learned slot (view) embedding and clip embedding
to each position's tokens, concatenates the tokens of every position and pools them with one
attentive probe whose key padding mask removes the positions without a video.
`LateAveragingRegressor` (a development baseline) pools each present position on its own and
averages the predictions of a study's present positions.
"""

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler
from src.utils.tensors import trunc_normal_

FUSIONS = ("early", "late")


class SlotEncoder(nn.Module):
    """The frozen encoder applied to the videos of a batch of studies.

    `forward(clips, clip_indices, present)` takes the stacked clips of the read videos (row-major
    over studies and positions, `collate_studies`) and returns tokens [studies, positions, N, D],
    zero where `present` is False. The encoder runs without gradients and stays in eval mode.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim

    def train(self, mode=True):
        # A frozen encoder: dropout and other training behaviour stay off whatever its caller sets.
        return super().train(False)

    @torch.no_grad()
    def forward(self, clips, clip_indices, present):
        if int(present.sum()) != clips[0].shape[0]:
            raise ValueError(f"{clips[0].shape[0]} videos for {int(present.sum())} present slot positions.")
        # One spatial view per clip; the video's clips are concatenated along time (`ClipAggregation`).
        tokens = self.encoder([[c] for c in clips], clip_indices)[0]
        out = tokens.new_zeros(*present.shape, *tokens.shape[1:])
        out[present] = tokens
        return out


class _SlotProbe(nn.Module):
    """Shared parts of the heads: a slot (view) embedding and a clip embedding per position, an
    attentive pooler (one query, `depth` blocks) and a linear regressor."""

    def __init__(self, embed_dim, num_heads, depth, layout, num_targets=1, use_activation_checkpointing=True,
                 init_std=0.02):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"The probe's {num_heads} attention heads must divide the embedding size {embed_dim}.")
        slots = list(dict.fromkeys(slot for slot, _ in layout))
        self.layout = [tuple(p) for p in layout]
        self.register_buffer("position_slot", torch.tensor([slots.index(s) for s, _ in layout]), persistent=False)
        self.register_buffer("position_clip", torch.tensor([k for _, k in layout]), persistent=False)
        self.slot_embed = nn.Parameter(torch.zeros(len(slots), embed_dim))
        self.clip_embed = nn.Parameter(torch.zeros(max(k for _, k in layout) + 1, embed_dim))
        # Small, like the pooler's query: the frozen tokens are layer-normalized, so unit-variance
        # embeddings would swamp them.
        trunc_normal_(self.slot_embed, std=init_std)
        trunc_normal_(self.clip_embed, std=init_std)
        self.pooler = AttentivePooler(num_queries=1, embed_dim=embed_dim, num_heads=num_heads, depth=depth,
                                      use_activation_checkpointing=use_activation_checkpointing)
        self.regressor = nn.Linear(embed_dim, num_targets, bias=True)

    def embed(self, tokens):
        """`tokens` [B, P, N, D] plus each position's slot and clip embedding."""
        if tokens.shape[1] != len(self.layout):
            raise ValueError(f"{tokens.shape[1]} slot positions; the probe was built for {len(self.layout)}.")
        position = self.slot_embed[self.position_slot] + self.clip_embed[self.position_clip]  # [P, D]
        return tokens + position[None, :, None, :].to(tokens.dtype)


class EarlyFusionRegressor(_SlotProbe):
    """One attentive probe over the tokens of all of a study's present positions."""

    def forward(self, tokens, present):
        x = self.embed(tokens).flatten(1, 2)  # [B, P * N, D]
        ignore = (~present).repeat_interleave(tokens.shape[2], dim=1)  # True = masked out
        return self.regressor(self.pooler(x, key_padding_mask=ignore).squeeze(1))


class LateAveragingRegressor(_SlotProbe):
    """Each present position pooled on its own; a study's prediction is the mean of its present
    positions' predictions. A development baseline, not the primary probe."""

    def forward(self, tokens, present):
        x = self.embed(tokens)[present]  # [present positions, N, D]
        each = self.regressor(self.pooler(x).squeeze(1))  # [present positions, targets]
        out = each.new_zeros(*present.shape, each.shape[-1])
        out[present] = each
        return out.sum(1) / present.sum(1, keepdim=True).to(out.dtype)


def make_probe(fusion, **kwargs):
    if fusion == "early":
        return EarlyFusionRegressor(**kwargs)
    if fusion == "late":
        return LateAveragingRegressor(**kwargs)
    raise ValueError(f"`experiment.classifier.fusion` is one of {FUSIONS}, not {fusion!r}.")
