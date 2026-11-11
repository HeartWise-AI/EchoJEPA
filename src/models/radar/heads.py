# src/models/radar/heads.py

"""Per-still measurement heads for RADAR's Doppler stills (optional; off unless the config has a `heads` section).

Each still carries the sonographer's measurements read from its burned-in box (`data/build_still_manifest.py`): AV
Vmax, LVOT VTI, TR Vmax... `StillMeasurementHeads` predicts them from the still's own frozen tokens: one learned
query attends over the still's tokens (the still's scalar conditioning, such as the velocity scale of its crop, is
added to the pooled feature first), the study's demographics modulate the result (FiLM, starting at the identity),
and one linear output per target follows. Targets are regressed as z-scores of their logarithm, with the training
split's mean and standard deviation (`fit_log_scaler`), so a 10 % error weighs the same at 1 m/s and at 4 m/s.

The measured values are targets only, never inputs: the manifest refuses conditioning columns that are targets,
and quantities that determine each other (MaxPG = 4 Vmax^2, AVA from the VTIs) must not be split between input and
target. The burned-in box and the calipers also show the value in the pixels: the stills are masked before encoding
(`src/datasets/still_preprocess.py`) and the leak audit in `docs/radar/RADAR_stills_integration.md` applies.

`measurement_loss` is the weighted sum over targets of the Huber loss averaged over the present targets of the global
batch: each rank divides its sum by the global count (all-reduced) times the world size, so after DDP averages the
gradients they equal the single-process gradient. A target absent from the whole batch contributes zero but stays in
the graph (DDP).
"""

import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import MLP


def fit_log_scaler(table, columns):
    """{column: [mean, std]} of log(value) over the positive finite values of `table[column]` (the training split's
    stills). A column with fewer than two such values is refused: its head would have nothing to learn from."""
    scaler = {}
    for column in columns:
        values = np.asarray(table[column], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size < 2:
            raise ValueError(f"Target {column!r} has fewer than two positive values in the training split.")
        logs = np.log(values)
        scaler[column] = [float(logs.mean()), float(max(logs.std(), 1e-6))]
    return scaler


def encode_targets(values, scaler, columns):
    """(z [N, Q] float32, present [N, Q] bool) from raw values [N, Q]; non-positive or missing values are absent."""
    values = np.asarray(values, dtype=float).reshape(-1, len(columns))
    present = np.isfinite(values) & (values > 0)
    z = np.zeros_like(values, dtype=np.float32)
    for q, column in enumerate(columns):
        mean, std = scaler[column]
        z[:, q] = np.where(present[:, q], (np.log(np.where(present[:, q], values[:, q], 1.0)) - mean) / std, 0.0)
    return torch.from_numpy(z), torch.from_numpy(present)


def decode_targets(z, scaler, columns):
    """Values in the targets' units from z-scores [..., Q]."""
    mean = torch.tensor([scaler[c][0] for c in columns], dtype=z.dtype, device=z.device)
    std = torch.tensor([scaler[c][1] for c in columns], dtype=z.dtype, device=z.device)
    return torch.exp(z * std + mean)


class DemographicFiLM(nn.Module):
    """x * (1 + gamma) + beta with (gamma, beta) a linear map of the demographics; starts at the identity."""

    def __init__(self, demo_dim, dim):
        super().__init__()
        self.linear = nn.Linear(demo_dim, 2 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, demo):
        gamma, beta = self.linear(demo.to(x.dtype)).chunk(2, dim=-1)
        shape = (demo.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
        return x * (1 + gamma.reshape(shape)) + beta.reshape(shape)


class StillMeasurementHeads(nn.Module):
    """One prediction per (still, target) from the still's frozen tokens [B, K, Ls, D]."""

    def __init__(self, embed_dim, targets, still_cond_dim=0, demo_dim=0, hidden_dim=256):
        super().__init__()
        if not targets:
            raise ValueError("StillMeasurementHeads needs at least one target.")
        self.targets = list(targets)
        self.norm = nn.LayerNorm(embed_dim)
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.key = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.value = nn.Linear(embed_dim, hidden_dim)
        self.cond = MLP(still_cond_dim, hidden_dim, hidden_dim) if still_cond_dim else None
        self.film = DemographicFiLM(demo_dim, hidden_dim) if demo_dim else None
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, len(self.targets))

    def forward(self, still_tokens, still_cond=None, demo=None):
        x = self.norm(still_tokens)
        logits = self.key(x) @ self.query / math.sqrt(self.query.numel())        # [B, K, Ls]
        pooled = (logits.softmax(dim=-1)[..., None] * self.value(x)).sum(dim=-2)  # [B, K, H]
        if self.cond is not None:
            if still_cond is None:
                raise ValueError("still_cond is required with still_cond_dim.")
            pooled = pooled + self.cond(still_cond.to(pooled.dtype))
        if self.film is not None:
            if demo is None:
                raise ValueError("demo is required with demo_dim.")
            pooled = self.film(pooled, demo)
        return self.out(self.out_norm(pooled)).float()


def _world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def measurement_loss(pred, target, present, weights=None, delta=1.0):
    """(loss, stats) over every (still, target) present in the global batch; see the module docstring.
    pred, target: [B, K, Q] (z-scores); present: [B, K, Q] bool (valid still and measured value); weights: [Q] or
    None. stats[q] = {"n": global count, "loss": this rank's mean Huber loss, or None when the global count is 0}."""
    Q = pred.shape[-1]
    weights = torch.ones(Q) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred, target = pred.float(), target.float()
        per = F.huber_loss(pred, target, reduction="none", delta=delta) * present
        sums = per.reshape(-1, Q).sum(dim=0)
        counts = present.reshape(-1, Q).sum(dim=0).to(torch.float32)
        world = _world_size()
        if world > 1:
            counts = counts.clone()
            dist.all_reduce(counts)
        loss = 0.0 * pred.sum()
        stats = {}
        for q in range(Q):
            n = int(counts[q].item())
            stats[q] = {"n": n, "loss": None}
            if n == 0:
                continue
            term = sums[q] * world / n
            stats[q]["loss"] = float(sums[q].detach()) / max(1, int(present[..., q].sum()))
            loss = loss + weights[q].to(pred.device) * term
    return loss, stats
