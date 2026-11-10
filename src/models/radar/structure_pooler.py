# src/models/radar/structure_pooler.py

"""Structure queries over the frozen patch tokens of a study's clips (RADAR's organ queries, for echo).

One learned query per trained structure cross-attends to the tokens of the study's clips whose view may show that
structure (`eligible`, from the routing registry) and to nothing else: there is no token self-attention, so a
structure's embedding never depends on the clips it may not use. The cross-attention weights are shared across
structures; each structure has its own query and its own output projection (RADAR has one vision projection per
organ).

A structure with no usable clip (no clip both eligible and real) would have a softmax over no keys, which is NaN, so
that row alone attends to a fixed zero null token. A row with a usable clip cannot attend to it: its embedding comes
from its clips only. A row on the null token is finite but not zero (LayerNorm offsets, the value bias, the query and
the residual blocks remain) and depends on no token of the study; it is flagged by `has_clip` and must be left out of
losses and predictions.

Padded slots are filled with zeros before anything else, whatever they hold (a NaN in a padded slot would
otherwise reach every query through the key and value projections); a non-finite token in a real clip is an error.

Tokens get, before attention: a fixed 3D sin-cos position within their clip (T, H, W), a view embedding and an
acquisition embedding (the last row of each embedding stands for an unknown value or a padded slot).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.models.utils.modules import MLP
from src.models.utils.pos_embs import get_3d_sincos_pos_embed


class MaskedCrossAttention(nn.Module):
    """Cross-attention of S queries over N keys with a per-query key mask.

    The key projection has no bias: a bias shared by every key shifts each query's logits by a constant, which the
    softmax removes, so its gradient is zero and an adaptive optimizer would only move it with rounding noise."""

    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim)

    def forward(self, q, x, allowed):
        """q: [B, S, D]; x: [B, N, D]; allowed: [B, S, N] bool, True where query s may attend key n."""
        B, S, D = q.shape
        N = x.shape[1]
        h = self.num_heads
        q = self.q(q).reshape(B, S, h, D // h).transpose(1, 2)
        k = self.k(x).reshape(B, N, h, D // h).transpose(1, 2)
        v = self.v(x).reshape(B, N, h, D // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
        return out.transpose(1, 2).reshape(B, S, D)


class StructureBlock(nn.Module):
    """V-JEPA's cross-attention block (pre-norm on the keys, residual query update, MLP) with per-query masks."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.xattn = MaskedCrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, q, x, allowed):
        q = q + self.xattn(q, self.norm1(x), allowed)
        return q + self.mlp(self.norm2(q))


class StructurePooler(nn.Module):
    """One embedding per (study, trained structure) from the eligible clips' frozen tokens."""

    def __init__(self, embed_dim, num_structures, token_grid, num_view_classes=0, num_acquisition_classes=0,
                 depth=2, num_heads=12, mlp_ratio=4.0, proj_dim=256, use_activation_checkpointing=False):
        super().__init__()
        T, H, W = token_grid
        if H != W:
            raise ValueError("The sin-cos position assumes a square token grid (H == W).")
        if num_structures < 1 or depth < 1:
            raise ValueError("num_structures and depth must be positive.")
        self.embed_dim = embed_dim
        self.num_structures = num_structures
        self.tokens_per_clip = T * H * W
        self.use_activation_checkpointing = use_activation_checkpointing
        pos = torch.from_numpy(get_3d_sincos_pos_embed(embed_dim, H, T)).float()
        self.register_buffer("pos_embed", pos, persistent=False)
        self.register_buffer("null_token", torch.zeros(1, 1, embed_dim), persistent=False)
        self.queries = nn.Parameter(torch.zeros(num_structures, embed_dim))
        self.view_embed = nn.Embedding(num_view_classes + 1, embed_dim) if num_view_classes else None
        self.acquisition_embed = (nn.Embedding(num_acquisition_classes + 1, embed_dim)
                                  if num_acquisition_classes else None)
        self.blocks = nn.ModuleList([StructureBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_weight = nn.Parameter(torch.empty(num_structures, embed_dim, proj_dim))
        self.proj_bias = nn.Parameter(torch.zeros(num_structures, proj_dim))
        for p in (self.queries, self.proj_weight):
            nn.init.trunc_normal_(p, std=0.02)
        for e in (self.view_embed, self.acquisition_embed):
            if e is not None:
                nn.init.trunc_normal_(e.weight, std=0.02)

    @staticmethod
    def _embed(layer, ids):
        """Embedding of class ids, with -1 (unknown or padded) mapped to the last row."""
        return layer(torch.where(ids < 0, torch.full_like(ids, layer.num_embeddings - 1), ids))

    def forward(self, tokens, clip_valid, eligible, view_ids=None, acquisition_ids=None):
        """
        tokens: [B, C, L, D] frozen tokens of C clip slots, L = T * H * W in (t, h, w) order.
        clip_valid: [B, C] bool, False for padded slots.
        eligible: [B, S, C] bool, structure s may use clip c (combined with clip_valid here).
        view_ids, acquisition_ids: [B, C] class ids, -1 for unknown or padded.
        Returns (features [B, S, D] before the projection, embeddings [B, S, P] L2-normalised,
        has_clip [B, S] bool: at least one usable clip).
        """
        B, C, L, D = tokens.shape
        S = self.num_structures
        if L != self.tokens_per_clip or D != self.embed_dim:
            raise ValueError(f"Expected {self.tokens_per_clip} tokens of dim {self.embed_dim} per clip.")
        if eligible.shape != (B, S, C) or clip_valid.shape != (B, C):
            raise ValueError("eligible must be [B, S, C] and clip_valid [B, C].")
        zero = torch.zeros((), dtype=tokens.dtype, device=tokens.device)
        tokens = torch.where(clip_valid[:, :, None, None], tokens, zero)
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("Non-finite tokens in a real clip.")
        x = tokens + self.pos_embed.to(tokens.dtype)
        if self.view_embed is not None:
            if view_ids is None:
                raise ValueError("view_ids are required with view embeddings.")
            x = x + self._embed(self.view_embed, view_ids)[:, :, None].to(x.dtype)
        if self.acquisition_embed is not None:
            if acquisition_ids is None:
                raise ValueError("acquisition_ids are required with acquisition embeddings.")
            x = x + self._embed(self.acquisition_embed, acquisition_ids)[:, :, None].to(x.dtype)
        x = torch.cat([self.null_token.expand(B, 1, D).to(x.dtype), x.reshape(B, C * L, D)], dim=1)
        usable = eligible & clip_valid[:, None, :]
        has_clip = usable.any(dim=-1)
        allowed = torch.cat([~has_clip[..., None], usable.repeat_interleave(L, dim=-1)], dim=-1)
        q = self.queries.expand(B, S, D).to(x.dtype)
        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                q = torch.utils.checkpoint.checkpoint(block, q, x, allowed, use_reentrant=False)
            else:
                q = block(q, x, allowed)
        features = self.norm(q)
        embeddings = torch.einsum("bsd,sdp->bsp", features, self.proj_weight.to(features.dtype)) + self.proj_bias
        return features, F.normalize(embeddings.float(), dim=-1), has_clip
