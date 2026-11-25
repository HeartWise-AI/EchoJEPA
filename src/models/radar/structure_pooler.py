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

Doppler stills (optional, off unless `still_token_grid` is given): a second stream of keys, the frozen tokens of a
study's stills (one tubelet, `[1, H, W]`), routed to structures by their measurement tag (`Routing.still_eligibility`)
instead of their view. Still tokens get their own sin-cos position, a still marker, a still-type embedding and, with
`still_cond_dim`, a projection of the still's scalar conditioning (velocity scale of the crop, mode, EchoNet's
predicted peak and confidence): its last layer starts at zero, so the stream starts as plain tokens. With
`still_heatmap_bias`, a per-token log heat-map (EchoNet's peak heat-map pooled to the token grid) is added to the
attention logits of the still's keys, times a learned scale that starts at zero, and normalised per still so that
the still's total prior weight stays that of unbiased tokens. With `demo_dim`, the queries are
modulated by the study's demographics (FiLM, starting at the identity); off by default because age and sex can stand
in for the image in the contrastive loss. A structure with neither a usable clip nor a usable still falls back on
the null token, as before. Without the still stream, the module, its parameters and its outputs are those of v0.
"""

import math

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

    def forward(self, q, x, allowed, key_bias=None):
        """q: [B, S, D]; x: [B, N, D]; allowed: [B, S, N] bool, True where query s may attend key n; key_bias: None or
        [B, N] added to the logits of key n for every query (where allowed)."""
        B, S, D = q.shape
        N = x.shape[1]
        h = self.num_heads
        q = self.q(q).reshape(B, S, h, D // h).transpose(1, 2)
        k = self.k(x).reshape(B, N, h, D // h).transpose(1, 2)
        v = self.v(x).reshape(B, N, h, D // h).transpose(1, 2)
        mask = allowed[:, None]
        if key_bias is not None:
            bias = key_bias[:, None, None, :].to(q.dtype)
            mask = torch.where(mask, bias, torch.full((), float("-inf"), dtype=q.dtype, device=q.device))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(B, S, D)


class StructureBlock(nn.Module):
    """V-JEPA's cross-attention block (pre-norm on the keys, residual query update, MLP) with per-query masks."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.xattn = MaskedCrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, q, x, allowed, key_bias=None):
        q = q + self.xattn(q, self.norm1(x), allowed, key_bias)
        return q + self.mlp(self.norm2(q))


class StructurePooler(nn.Module):
    """One embedding per (study, trained structure) from the eligible clips' frozen tokens."""

    def __init__(self, embed_dim, num_structures, token_grid, num_view_classes=0, num_acquisition_classes=0,
                 depth=2, num_heads=12, mlp_ratio=4.0, proj_dim=256, use_activation_checkpointing=False,
                 still_token_grid=None, num_still_types=0, still_cond_dim=0, still_heatmap_bias=False, demo_dim=0):
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
        self.has_stills = still_token_grid is not None
        if not self.has_stills and (num_still_types or still_cond_dim or still_heatmap_bias):
            raise ValueError("Still options need still_token_grid.")
        if self.has_stills:
            Ts, Hs, Ws = still_token_grid
            if Hs != Ws:
                raise ValueError("The sin-cos position assumes a square still token grid (H == W).")
            self.tokens_per_still = Ts * Hs * Ws
            self.register_buffer("still_pos_embed", torch.from_numpy(get_3d_sincos_pos_embed(embed_dim, Hs, Ts))
                                 .float(), persistent=False)
            self.still_marker = nn.Parameter(torch.zeros(embed_dim))
            nn.init.trunc_normal_(self.still_marker, std=0.02)
            self.still_type_embed = nn.Embedding(num_still_types + 1, embed_dim)
            nn.init.trunc_normal_(self.still_type_embed.weight, std=0.02)
            self.still_cond = MLP(still_cond_dim, embed_dim, embed_dim) if still_cond_dim else None
            if self.still_cond is not None:
                nn.init.zeros_(self.still_cond.fc2.weight)
                nn.init.zeros_(self.still_cond.fc2.bias)
            self.heatmap_scale = nn.Parameter(torch.zeros(())) if still_heatmap_bias else None
        self.demo_film = nn.Linear(demo_dim, 2 * embed_dim) if demo_dim else None
        if self.demo_film is not None:
            nn.init.zeros_(self.demo_film.weight)
            nn.init.zeros_(self.demo_film.bias)

    @staticmethod
    def _embed(layer, ids):
        """Embedding of class ids, with -1 (unknown or padded) mapped to the last row."""
        return layer(torch.where(ids < 0, torch.full_like(ids, layer.num_embeddings - 1), ids))

    def _still_keys(self, still_tokens, still_valid, still_eligible, still_type_ids, still_cond, still_bias, B, S):
        """(keys [B, K * Ls, D], usable [B, S, K * Ls], key bias [B, K * Ls] or None) of the still stream."""
        if not self.has_stills:
            raise ValueError("Still tokens were given to a pooler built without a still stream.")
        _, K, Ls, D = still_tokens.shape
        if Ls != self.tokens_per_still or D != self.embed_dim:
            raise ValueError(f"Expected {self.tokens_per_still} tokens of dim {self.embed_dim} per still.")
        if still_eligible.shape != (B, S, K) or still_valid.shape != (B, K):
            raise ValueError("still_eligible must be [B, S, K] and still_valid [B, K].")
        zero = torch.zeros((), dtype=still_tokens.dtype, device=still_tokens.device)
        x = torch.where(still_valid[:, :, None, None], still_tokens, zero)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Non-finite tokens in a real still.")
        type_ids = still_type_ids if still_type_ids is not None else torch.full((B, K), -1, device=x.device)
        x = (x + self.still_pos_embed.to(x.dtype) + self.still_marker.to(x.dtype)
             + self._embed(self.still_type_embed, type_ids)[:, :, None].to(x.dtype))
        if self.still_cond is not None:
            if still_cond is None:
                raise ValueError("still_cond is required with still_cond_dim.")
            x = x + self.still_cond(still_cond.to(x.dtype))[:, :, None]
        usable = (still_eligible & still_valid[:, None, :]).repeat_interleave(Ls, dim=-1)
        bias = None
        if self.heatmap_scale is not None:
            if still_bias is None:
                raise ValueError("still_bias is required with still_heatmap_bias.")
            bias = self.heatmap_scale.float() * torch.where(still_valid[:, :, None], still_bias, 0.0).float()
            # The bias moves attention within a still, not between keys: each still keeps the total weight of Ls
            # unbiased tokens, so a still with a heat-map does not lose to one without, or to the clips.
            bias = bias - (torch.logsumexp(bias, dim=-1, keepdim=True) - math.log(Ls))
            bias = bias.reshape(B, K * Ls)
        return x.reshape(B, K * Ls, D), usable, bias

    def forward(self, tokens, clip_valid, eligible, view_ids=None, acquisition_ids=None, still_tokens=None,
                still_valid=None, still_eligible=None, still_type_ids=None, still_cond=None, still_bias=None,
                demo=None):
        """
        tokens: [B, C, L, D] frozen tokens of C clip slots, L = T * H * W in (t, h, w) order.
        clip_valid: [B, C] bool, False for padded slots.
        eligible: [B, S, C] bool, structure s may use clip c (combined with clip_valid here).
        view_ids, acquisition_ids: [B, C] class ids, -1 for unknown or padded.
        Still stream (with `still_token_grid`; all None skips it): still_tokens [B, K, Ls, D], still_valid [B, K],
        still_eligible [B, S, K], still_type_ids [B, K] (-1 unknown), still_cond [B, K, c], still_bias [B, K, Ls].
        demo: [B, demo_dim] (with `demo_dim`).
        Returns (features [B, S, D] before the projection, embeddings [B, S, P] L2-normalised,
        has_clip [B, S] bool: at least one usable clip, or still when the still stream is used).
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
        keys = [self.null_token.expand(B, 1, D).to(x.dtype), x.reshape(B, C * L, D)]
        usable = [(eligible & clip_valid[:, None, :]).repeat_interleave(L, dim=-1)]
        key_bias = None
        if still_tokens is not None:
            still_x, still_usable, still_key_bias = self._still_keys(still_tokens.to(x.dtype), still_valid,
                                                                     still_eligible, still_type_ids, still_cond,
                                                                     still_bias, B, S)
            keys.append(still_x)
            usable.append(still_usable)
            if still_key_bias is not None:
                key_bias = torch.cat([still_key_bias.new_zeros(B, 1 + C * L), still_key_bias], dim=1)
        x = torch.cat(keys, dim=1)
        usable = torch.cat(usable, dim=-1)
        has_clip = usable.any(dim=-1)
        allowed = torch.cat([~has_clip[..., None], usable], dim=-1)
        q = self.queries.expand(B, S, D).to(x.dtype)
        if self.demo_film is not None:
            if demo is None:
                raise ValueError("demo is required with demo_dim.")
            gamma, beta = self.demo_film(demo.to(q.dtype)).chunk(2, dim=-1)
            q = q * (1 + gamma[:, None]) + beta[:, None]
        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                q = torch.utils.checkpoint.checkpoint(block, q, x, allowed, key_bias, use_reentrant=False)
            else:
                q = block(q, x, allowed, key_bias)
        features = self.norm(q)
        embeddings = torch.einsum("bsd,sdp->bsp", features, self.proj_weight.to(features.dtype)) + self.proj_bias
        return features, F.normalize(embeddings.float(), dim=-1), has_clip
