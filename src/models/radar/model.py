# src/models/radar/model.py

"""RADAR structure alignment as one module (the one DDP wraps): structure pooler, caption encoder and the
structure loss with its temperature. The momentum teacher stays outside, because it is never trained.

A (study, structure) row takes part when the study's bag has an eligible clip for the structure and the structure
has a training caption. Only those captions are encoded; when a rank has none, one padding caption is encoded with
zero weight. The dense caption tensor that the loss all-gathers always depends on the encoded captions (with zero
weight where no row is scattered), so on every rank, with or without captions, its gather takes part in backward
(the gather's backward is a collective) and backward reaches the same parameters (DDP needs it).

Doppler stills (optional): when the batch holds `still_tokens`, the pooler also reads the still stream, and with
measurement heads the loss adds `measurement_weight` times their loss (`src/models/radar/heads.py`). The structure
stats keep their v0 shape; the heads' stats are in `last_measurement` ({target: {"n", "loss"}}, or None).
"""

import torch
import torch.nn as nn

from src.models.radar.heads import measurement_loss

STILL_KEYS = ("still_tokens", "still_valid", "still_eligible", "still_type_ids", "still_cond", "still_bias")


class RadarModel(nn.Module):

    def __init__(self, pooler, text_encoder, loss, heads=None, measurement_weight=1.0, target_weights=None):
        super().__init__()
        self.pooler = pooler
        self.text = text_encoder
        self.loss = loss
        self.heads = heads
        self.measurement_weight = float(measurement_weight)
        self.target_weights = target_weights
        self.last_measurement = None

    def forward(self, batch, teacher=None):
        """Returns (loss, per-structure stats, valid [B, S])."""
        extra = {k: batch.get(k) for k in STILL_KEYS} if "still_tokens" in batch else {}
        if "demo" in batch and getattr(self.pooler, "demo_film", None) is not None:
            extra["demo"] = batch["demo"]
        _, z, has_clip = self.pooler(batch["tokens"], batch["clip_valid"], batch["eligible"], batch["view_ids"],
                                     batch["acquisition_ids"], **extra)
        valid = has_clip & batch["has_caption"]
        b, s = valid.nonzero(as_tuple=True)
        ids, mask = batch["text_ids"], batch["text_mask"]
        if b.numel():
            rows_ids, rows_mask = ids[b, s], mask[b, s]
        else:
            rows_ids, rows_mask = ids[:1, 0], mask[:1, 0]
        t_rows = self.text(rows_ids, rows_mask)
        # Zero-weighted dependency inside the gathered tensor: without it a rank with no caption would gather a
        # constant, skip the gather's backward collective, and stall or break the other ranks' backward.
        t = t_rows.new_zeros(z.shape) + 0.0 * t_rows.sum()
        if b.numel():
            t = t.index_put((b, s), t_rows)
        teacher_dense = None
        if self.loss.use_abnormal_teacher:
            if teacher is None:
                raise ValueError("The loss uses abnormal teacher weights: pass the momentum teacher.")
            teacher_dense = torch.zeros(z.shape, dtype=torch.float32, device=z.device)
            if b.numel():
                teacher_dense[b, s] = teacher(rows_ids, rows_mask).float()
        loss, stats = self.loss(z, t, valid, batch["caption_ids"], batch["normal"], batch["abnormal"], teacher_dense)
        self.last_measurement = None
        if self.heads is not None:
            pred = self.heads(batch["still_tokens"], batch.get("still_cond"),
                              batch.get("demo") if self.heads.film is not None else None)
            present = batch["still_y_mask"] & batch["still_valid"][..., None]
            m_loss, m_stats = measurement_loss(pred, batch["still_y"], present, self.target_weights)
            loss = loss + self.measurement_weight * m_loss
            self.last_measurement = {self.heads.targets[q]: v for q, v in m_stats.items()}
        return loss, stats, valid
