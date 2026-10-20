# src/models/study_probe.py

"""Study-level multi-task probe on frozen video tokens, DeepCORO_CLIP style.

The encoder embeds every clip of every video of a study; the probe pools the tokens of the
study's video slots into one embedding and reads it with one linear head per target. Padded
slots (a study with fewer videos than slots) are masked out of the pooling.

Pooling modes (`pooling_mode`):

- `attentive`: the V-JEPA attentive probe (`AttentivePooler`) over the tokens of every clip of
  every video, with the padded slots' tokens masked out of the attention;
- `mean`: each video's tokens are averaged, then the videos are averaged;
- `max`: each video's tokens are averaged, then the videos are max-pooled per feature;
- `attention`: each video's tokens are averaged, then a gated attention (Ilse et al., 2018,
  as in DeepCORO_CLIP) weighs the videos.

With `num_view_classes`, a learned view embedding is added to the tokens of every video slot
(DeepCORO_CLIP's view embeddings), so the pooling knows which view each video shows.

Each head is `regression` (one output, Huber/MSE/L1 loss on z-scored targets), `binary` (one
logit, BCE) or `classification` (`num_classes` logits, cross-entropy). A missing target (NaN)
takes the study out of that head's loss and metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attentive_pooler import AttentivePooler

POOLING_MODES = ("attentive", "mean", "max", "attention")
TASKS = ("regression", "binary", "classification")
LOSSES = {"regression": ("huber", "mse", "l1"), "binary": ("bce",), "classification": ("ce",)}
DEFAULT_LOSS = {"regression": "huber", "binary": "bce", "classification": "ce"}


def head_specs(classifier):
    """The heads of the `experiment.classifier` block, DeepCORO_CLIP style: `target_label` lists
    the target columns; `head_task`, `head_structure` (outputs), `loss_structure`, `head_weights`
    and `head_dropout` are per head and optional except `head_task`.

    Returns {name: {task, outputs, loss, weight, dropout}} in `target_label` order.
    """
    names = list(classifier.get("target_label") or [])
    if not names:
        raise ValueError("`experiment.classifier.target_label` lists the target columns to probe.")
    if len(set(names)) != len(names):
        raise ValueError("`target_label` repeats a column.")
    tasks = classifier.get("head_task") or {}
    structure = classifier.get("head_structure") or {}
    losses = classifier.get("loss_structure") or {}
    weights = classifier.get("head_weights") or {}
    dropouts = classifier.get("head_dropout") or {}
    for block, values in (("head_task", tasks), ("head_structure", structure), ("loss_structure", losses),
                          ("head_weights", weights), ("head_dropout", dropouts)):
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise ValueError(f"`{block}` names heads {unknown} that `target_label` does not list.")
    specs = {}
    for name in names:
        task = tasks.get(name)
        if task not in TASKS:
            raise ValueError(f"`head_task.{name}` is one of {TASKS}, not {task!r}.")
        outputs = int(structure.get(name, 1))
        if task in ("regression", "binary") and outputs != 1:
            raise ValueError(f"A {task} head has one output; `head_structure.{name}` is {outputs}.")
        if task == "classification" and outputs < 2:
            raise ValueError(f"`head_structure.{name}` must give the number of classes (at least 2).")
        loss = losses.get(name, DEFAULT_LOSS[task])
        if loss not in LOSSES[task]:
            raise ValueError(f"`loss_structure.{name}` is one of {LOSSES[task]} for a {task} head, not {loss!r}.")
        weight = float(weights.get(name, 1.0))
        if weight < 0:
            raise ValueError(f"`head_weights.{name}` must not be negative.")
        specs[name] = {"task": task, "outputs": outputs, "loss": loss, "weight": weight,
                       "dropout": float(dropouts.get(name, 0.0))}
    return specs


class GatedAttention(nn.Module):
    """Gated attention over a bag of embeddings (Ilse et al., 2018): w = softmax(W (tanh(V h) * sigmoid(U h)))."""

    def __init__(self, embed_dim, hidden=256, dropout=0.0):
        super().__init__()
        self.V = nn.Linear(embed_dim, hidden)
        self.U = nn.Linear(embed_dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, mask):
        """h: [B, N, D]; mask: [B, N] (True = real). Returns [B, D] and the weights [B, N]."""
        scores = self.w(self.dropout(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bn,bnd->bd", weights, h), weights


class StudyProbe(nn.Module):
    """Pools the tokens of a study's video slots and reads them with one head per target."""

    def __init__(self, embed_dim, heads, pooling_mode="attentive", num_heads=16, depth=4, attention_hidden=256,
                 dropout=0.0, use_activation_checkpointing=True, num_view_classes=0):
        super().__init__()
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"`pooling_mode` is one of {POOLING_MODES}, not {pooling_mode!r}.")
        self.pooling_mode = pooling_mode
        self.embed_dim = embed_dim
        self.specs = dict(heads)
        # One learned embedding per view class, added to every token of a video of that view, plus
        # one for a padded slot (index `num_view_classes`), as DeepCORO_CLIP's view embeddings.
        self.num_view_classes = int(num_view_classes)
        self.view_embed = nn.Embedding(self.num_view_classes + 1, embed_dim) if self.num_view_classes else None
        if self.view_embed is not None:
            nn.init.trunc_normal_(self.view_embed.weight, std=0.02)
        if pooling_mode == "attentive":
            self.pooler = AttentivePooler(num_queries=1, embed_dim=embed_dim, num_heads=num_heads, depth=depth,
                                          use_activation_checkpointing=use_activation_checkpointing)
            self.norm = nn.Identity()
        else:
            self.pooler = GatedAttention(embed_dim, attention_hidden, dropout) if pooling_mode == "attention" else None
            self.norm = nn.LayerNorm(embed_dim)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(spec["dropout"]), nn.Linear(embed_dim, spec["outputs"]))
            for name, spec in self.specs.items()
        })

    def pool(self, tokens, video_mask, view_ids=None):
        """tokens: [B, L, D], the tokens of every clip of every video slot, slot by slot;
        video_mask: [B, N] (True = real video); view_ids: [B, N] view class per slot (-1 for a
        padded slot), needed with a view embedding. Returns the study embedding [B, D]."""
        B, L, D = tokens.shape
        N = video_mask.shape[1]
        if L % N:
            raise ValueError(f"{L} tokens do not split into {N} video slots.")
        if not bool(video_mask.any(dim=1).all()):
            raise ValueError("Every study needs at least one real video.")
        per_video = L // N
        if self.view_embed is not None:
            if view_ids is None:
                raise ValueError("The probe has a view embedding: pass each slot's `view_ids`.")
            if bool((view_ids[video_mask] < 0).any()) or bool((view_ids >= self.num_view_classes).any()):
                raise ValueError(f"View classes must be in [0, {self.num_view_classes}) for every real video.")
            classes = view_ids.clamp(min=0).masked_fill(~video_mask, self.num_view_classes)
            tokens = tokens + self.view_embed(classes).to(tokens.dtype).repeat_interleave(per_video, dim=1)
        if self.pooling_mode == "attentive":
            key_padding_mask = (~video_mask).repeat_interleave(per_video, dim=1)  # True = ignore
            return self.pooler(tokens, key_padding_mask=key_padding_mask).squeeze(1)
        videos = tokens.reshape(B, N, per_video, D).mean(dim=2)  # [B, N, D]
        mask = video_mask.to(videos.dtype).unsqueeze(-1)
        if self.pooling_mode == "mean":
            return (videos * mask).sum(dim=1) / mask.sum(dim=1)
        if self.pooling_mode == "max":
            return videos.masked_fill(~video_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        pooled, _ = self.pooler(videos, video_mask)
        return pooled

    def forward(self, tokens, video_mask, view_ids=None):
        """{head: [B, outputs]} from the tokens of a batch of studies."""
        embedding = self.norm(self.pool(tokens, video_mask, view_ids))
        return {name: head(embedding) for name, head in self.heads.items()}


def head_loss(spec, output, target):
    """The loss of one head over the studies whose target is present, as a (loss sum, count) pair.
    `output`: [B, outputs]; `target`: [B] (z-scored for regression, 0/1 for binary, class index
    for classification; NaN where missing). The sum is 0 when no target is present."""
    valid = torch.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return output.sum() * 0.0, 0
    output, target = output[valid].float(), target[valid].float()
    task, loss = spec["task"], spec["loss"]
    if task == "regression":
        if loss == "huber":
            value = F.smooth_l1_loss(output.squeeze(-1), target, reduction="sum")
        elif loss == "mse":
            value = F.mse_loss(output.squeeze(-1), target, reduction="sum")
        else:
            value = F.l1_loss(output.squeeze(-1), target, reduction="sum")
    elif task == "binary":
        value = F.binary_cross_entropy_with_logits(output.squeeze(-1), target, reduction="sum")
    else:
        value = F.cross_entropy(output, target.long(), reduction="sum")
    return value, count


def multitask_loss(specs, outputs, targets):
    """The weighted sum of the heads' mean losses over a batch, and each head's (loss sum, count).

    `targets`: [B, heads] in `specs` order. A head without a present target in the batch adds
    nothing. Returns (loss, {head: (sum, count)}).
    """
    total, parts = None, {}
    for i, (name, spec) in enumerate(specs.items()):
        value, count = head_loss(spec, outputs[name], targets[:, i])
        parts[name] = (value, count)
        if count:
            term = spec["weight"] * value / count
            total = term if total is None else total + term
    if total is None:
        total = sum(v for v, _ in parts.values())  # zero, attached to the graph.
    return total, parts
