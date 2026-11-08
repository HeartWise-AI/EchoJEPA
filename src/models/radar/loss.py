# src/models/radar/loss.py

"""RADAR's structure-wise adaptive contrastive loss.

For each trained structure s, over the valid rows of the global batch (studies with an eligible clip and a training
caption for s), the targets follow the released RADAR implementation (damo-radar, `radar_pretrain.py`):

    H_ij = (normal_i AND normal_j) OR same_caption(i, j)        Boolean union, not a sum
    A_ij = abnormal_i AND abnormal_j                            (unknown is neither normal nor abnormal)
    P_ij = softmax_j(teacher_i . teacher_j / tau)               momentum text features, no gradient
    R_ij = 1 if i == j, else H_ij + A_ij * P_ij
    Q    = R / R.sum(-1)

    loss_i2t = -(Q * log_softmax(z @ t.T / tau, dim=-1)).sum(-1).mean()
    loss_t2i = -(Q * log_softmax(t @ z.T / tau, dim=-1)).sum(-1).mean()
    L_s = (loss_i2t + loss_t2i) / 2, and the loss is the sum of L_s over the structures used in the step.

The same Q serves both directions, as released. Variants: `use_exact` (E), `use_normal` (N),
`use_abnormal_teacher` (W_abn); with all three off the targets are the identity (CLIP InfoNCE).

Similarities, targets and cross-entropies are computed in float32 with autocast disabled: under bfloat16 the
logits at a small temperature lose most of their precision.

The temperature is a learnable parameter used through a functional clamp to `temperature_bounds`, so several
forwards may share one backward (micro-batches, several bags); `clamp_temperature_()` puts the stored value back
inside the bounds and is called after each optimizer step (the release clamps it in place before each forward).

A structure is skipped in a step when it has fewer than `min_pairs` valid rows or its targets are uniform. With
`release_gate`, a structure with no abnormal row is skipped too, as the release does; that gate is off by default
because captions of unknown normality still carry identity and exact-caption targets.

Distributed training: every rank computes the same global loss from all-gathered rows. Features are gathered with
`src.utils.distributed.AllGather`, whose backward sums the gradients of every rank, so after DDP averages the
parameter gradients they equal the single-process gradient of the global loss; the temperature, used directly, gets
its full gradient on every rank. (The release's detached gather with the rank's own slice put back gives the
features only 1 / world_size of their gradient under DDP.) Rows are padded to the largest local batch, and the
skip decisions are made on gathered counts, so every rank takes the same decisions.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.utils.distributed import AllGather


def caption_targets(caption_ids, normal=None, abnormal=None, teacher=None, tau=None, use_exact=True,
                    use_normal=False):
    """Q [n, n] for one structure (see the module docstring), in float32. `teacher` (with `tau`) enables W_abn."""
    n = caption_ids.shape[0]
    off = torch.zeros(n, n, dtype=torch.bool, device=caption_ids.device)
    if use_exact:
        off |= caption_ids[:, None] == caption_ids[None, :]
    if use_normal:
        off |= normal[:, None] & normal[None, :]
    with torch.autocast(device_type=caption_ids.device.type, enabled=False):
        R = off.float()
        if teacher is not None:
            teacher = teacher.float()
            weights = F.softmax(teacher @ teacher.T / torch.as_tensor(tau, dtype=torch.float32), dim=1)
            R = R + (abnormal[:, None] & abnormal[None, :]).float() * weights
        R.fill_diagonal_(1.0)
        return R / R.sum(dim=1, keepdim=True)


def contrastive_loss(z, t, Q, tau):
    """Soft-target contrastive loss of one structure, both directions with the same targets, in float32."""
    with torch.autocast(device_type=z.device.type, enabled=False):
        tau = tau.float() if torch.is_tensor(tau) else float(tau)
        logits = z.float() @ t.float().T / tau
        Q = Q.float()
        i2t = -(Q * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        t2i = -(Q * F.log_softmax(logits.T, dim=-1)).sum(-1).mean()
        return (i2t + t2i) / 2


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather_rows(x, rows, differentiable):
    """All-gather a [B, ...] tensor padded to `rows` rows per rank. With `differentiable`, the gradient of every
    rank's use of the gathered tensor flows back to the local rows (AllGather sums it across ranks)."""
    if x.shape[0] < rows:
        x = torch.cat([x, x.new_zeros((rows - x.shape[0],) + tuple(x.shape[1:]))], dim=0)
    if not _distributed():
        return x
    if differentiable:
        return AllGather.apply(x)
    with torch.no_grad():
        return AllGather.apply(x.contiguous())


class StructureContrastiveLoss(nn.Module):
    """Sum over structures of RADAR's adaptive contrastive loss, with a learnable temperature."""

    def __init__(self, structures, use_exact=True, use_normal=False, use_abnormal_teacher=False,
                 release_gate=False, min_pairs=2, init_temperature=0.07, temperature_bounds=(0.001, 0.5)):
        super().__init__()
        self.structures = list(structures)
        self.use_exact = use_exact
        self.use_normal = use_normal
        self.use_abnormal_teacher = use_abnormal_teacher
        self.release_gate = release_gate
        if min_pairs < 2:
            raise ValueError("min_pairs must be at least 2: one row has no contrast.")
        self.min_pairs = min_pairs
        self.temperature_bounds = tuple(temperature_bounds)
        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))

    def effective_temperature(self):
        """The temperature used in the loss: the parameter clamped to the bounds, without changing it in place."""
        return self.temperature.clamp(*self.temperature_bounds)

    @torch.no_grad()
    def clamp_temperature_(self):
        """Put the stored temperature back inside the bounds; call after each optimizer step."""
        self.temperature.clamp_(*self.temperature_bounds)

    def forward(self, z, t, valid, caption_ids, normal=None, abnormal=None, teacher=None):
        """
        z, t: [B, S, P] L2-normalised image and caption embeddings (any value where invalid, but finite).
        valid: [B, S] bool, the row takes part for structure s. caption_ids: [B, S] int64 canonical caption ids.
        normal, abnormal: [B, S] bool (needed with use_normal / release_gate / use_abnormal_teacher).
        teacher: [B, S, P] momentum caption embeddings (needed with use_abnormal_teacher).
        Returns (loss, stats): stats[name] = {"rows": int, "loss": float or None, "skipped": reason or None,
        "exact_pairs": int}. `exact_pairs` counts the unordered pairs of valid rows with the same caption id, in every
        arm: E makes them positives (A1-A3), the identity alone keeps them negatives (A0). It shows how far A1's
        targets differ from A0's in the batches actually trained on.
        """
        B, S, _ = z.shape
        if S != len(self.structures):
            raise ValueError(f"Expected {len(self.structures)} structures, got {S}.")
        needs_flags = self.use_normal or self.release_gate or self.use_abnormal_teacher
        if needs_flags and (normal is None or abnormal is None):
            raise ValueError("normal and abnormal flags are required by N, W_abn and the release gate.")
        if self.use_abnormal_teacher and teacher is None:
            raise ValueError("teacher embeddings are required by W_abn.")
        tau = self.effective_temperature()

        rows = B
        if _distributed():
            size = torch.tensor([B], device=z.device)
            dist.all_reduce(size, op=dist.ReduceOp.MAX)
            rows = int(size.item())
        z_all = gather_rows(z, rows, differentiable=True)
        t_all = gather_rows(t, rows, differentiable=True)
        valid_all = gather_rows(valid.to(torch.uint8), rows, differentiable=False).bool()
        ids_all = gather_rows(caption_ids.to(torch.int64), rows, differentiable=False)
        flags = None
        if normal is not None:
            flags = gather_rows(torch.stack([normal, abnormal], dim=-1).to(torch.uint8), rows,
                                differentiable=False).bool()
        teacher_all = gather_rows(teacher.detach(), rows, differentiable=False) if self.use_abnormal_teacher else None

        # Zero-weighted terms keep every parameter in the graph when structures are skipped (DDP static graph).
        loss = 0.0 * (z.sum() + t.sum() + tau)
        stats = {}
        for s, name in enumerate(self.structures):
            idx = valid_all[:, s].nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            ids_s = ids_all[idx, s]
            same = int((ids_s[:, None] == ids_s[None, :]).sum()) - n
            stat = {"rows": n, "loss": None, "skipped": None, "exact_pairs": same // 2}
            stats[name] = stat
            if n < self.min_pairs:
                stat["skipped"] = "too_few_rows"
                continue
            normal_s = flags[idx, s, 0] if flags is not None else None
            abnormal_s = flags[idx, s, 1] if flags is not None else None
            if self.release_gate and not bool(abnormal_s.any()):
                stat["skipped"] = "no_abnormal_row"
                continue
            Q = caption_targets(ids_s, normal_s, abnormal_s,
                                teacher_all[idx, s] if teacher_all is not None else None, tau.detach(),
                                use_exact=self.use_exact, use_normal=self.use_normal)
            if torch.allclose(Q, torch.full_like(Q, 1.0 / n)):
                stat["skipped"] = "uniform_targets"
                continue
            loss_s = contrastive_loss(z_all[idx, s], t_all[idx, s], Q, tau)
            stat["loss"] = float(loss_s.detach())
            loss = loss + loss_s
        return loss, stats
