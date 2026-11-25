"""Multi-positive symmetric InfoNCE with an explicit positive / negative / ignore mask.

A batch of echo clips is full of pairs that are not true negatives: two clips of the same study, two clips whose
captions are identical ("Coupe apicale 4 cavités (A4C), mode B." is common), or captions that assert the same
concepts in other words. Standard CLIP loss would push all of them apart. Here every pair (i, j) is classified as

  POSITIVE  the pair itself, identical normalized captions, and (optionally) identical concept sets;
  IGNORE    excluded from both numerator and denominator: same-study pairs and equal concept sets by default;
  NEGATIVE  everything else.

Same-study and semantically equivalent pairs are never negatives unless a caller builds the mask differently.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Hashable, Sequence

import torch
import torch.nn as nn

POSITIVE, NEGATIVE, IGNORE = 1, 0, -1
SAME_STUDY_POLICIES = ("ignore", "positive")
EQUAL_CONCEPTS_POLICIES = ("ignore", "positive", "off")


def normalize_caption(text: str) -> str:
    """Captions that differ only in case, Unicode form, spacing or final punctuation are the same caption."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" .;")


def build_pair_mask(
    caption_keys: Sequence[Hashable],
    study_keys: Sequence[Hashable],
    concept_keys: Sequence[Hashable] | None = None,
    *,
    same_study: str = "ignore",
    equal_concepts: str = "ignore",
) -> torch.Tensor:
    """A [B, B] int8 mask of POSITIVE / NEGATIVE / IGNORE. Positive wins over ignore; the diagonal is positive.

    `study_keys` only group pairs here; they never reach the model.
    """
    if same_study not in SAME_STUDY_POLICIES:
        raise ValueError(f"same_study must be one of {SAME_STUDY_POLICIES}, not {same_study!r}.")
    if equal_concepts not in EQUAL_CONCEPTS_POLICIES:
        raise ValueError(f"equal_concepts must be one of {EQUAL_CONCEPTS_POLICIES}, not {equal_concepts!r}.")
    n = len(caption_keys)
    if len(study_keys) != n or (concept_keys is not None and len(concept_keys) != n):
        raise ValueError("caption, study and concept keys must have one entry per batch element.")

    def same(keys):
        index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        ids = torch.tensor([index[k] for k in keys])
        return ids[:, None] == ids[None, :]

    positive = same(caption_keys) | torch.eye(n, dtype=torch.bool)
    ignore = same(study_keys)
    if same_study == "positive":
        positive |= ignore
    if concept_keys is not None and equal_concepts != "off":
        equal = same(concept_keys)
        if equal_concepts == "positive":
            positive |= equal
        else:
            ignore |= equal
    mask = torch.full((n, n), NEGATIVE, dtype=torch.int8)
    mask[ignore] = IGNORE
    mask[positive] = POSITIVE
    return mask


def multi_positive_infonce(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over rows of -mean_{j in P_i} log( exp(l_ij) / sum_{k in P_i u N_i} exp(l_ik) ).

    Ignored entries appear in neither the numerator nor the denominator.
    """
    if logits.shape != mask.shape or logits.dim() != 2:
        raise ValueError("logits and mask must be the same [B, B] shape.")
    positive = mask == POSITIVE
    if not bool(positive.any(dim=1).all()):
        raise ValueError("Every row needs at least one positive (the diagonal).")
    logits = logits.float()
    valid = mask != IGNORE
    log_denominator = logits.masked_fill(~valid, float("-inf")).logsumexp(dim=1, keepdim=True)
    log_prob = (logits - log_denominator).masked_fill(~positive, 0.0)
    return (-log_prob.sum(dim=1) / positive.sum(dim=1)).mean()


class ContrastiveLoss(nn.Module):
    """Symmetric video->text and text->video multi-positive InfoNCE with a learnable temperature."""

    def __init__(self, init_temperature: float = 0.07, learnable_temperature: bool = True,
                 max_logit_scale: float = 100.0, same_study: str = "ignore", equal_concepts: str = "ignore"):
        super().__init__()
        if not 0 < init_temperature < 1:
            raise ValueError("init_temperature must be in (0, 1).")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)),
                                        requires_grad=learnable_temperature)
        self.max_logit_scale = max_logit_scale
        self.same_study = same_study
        self.equal_concepts = equal_concepts
        build_pair_mask(["a"], ["s"], same_study=same_study, equal_concepts=equal_concepts)  # validate policies

    @property
    def temperature(self) -> float:
        return float(1.0 / self.scale().detach())

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)

    def mask(self, caption_keys, study_keys, concept_keys=None) -> torch.Tensor:
        return build_pair_mask(caption_keys, study_keys, concept_keys, same_study=self.same_study,
                               equal_concepts=self.equal_concepts)

    def forward(self, video: torch.Tensor, text: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """`video` and `text` are L2-normalized [B, D] embeddings; `mask` from `build_pair_mask`."""
        mask = mask.to(video.device)
        logits = self.scale() * video.float() @ text.float().t()
        v2t = multi_positive_infonce(logits, mask)
        t2v = multi_positive_infonce(logits.t(), mask.t())
        loss = (v2t + t2v) / 2
        off_diagonal = ~torch.eye(len(mask), dtype=torch.bool, device=mask.device)
        stats = {
            "loss_v2t": float(v2t.detach()),
            "loss_t2v": float(t2v.detach()),
            "temperature": self.temperature,
            "extra_positives": int(((mask == POSITIVE) & off_diagonal).sum()),
            "ignored": int((mask == IGNORE).sum()),
            "negatives": int((mask == NEGATIVE).sum()),
        }
        return loss, stats
