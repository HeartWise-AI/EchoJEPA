# src/models/study_probe.py

"""Study-level multi-task probe on frozen video tokens, DeepCORO_CLIP style.

The encoder embeds every clip of every video of a study; the probe pools the tokens of the
study's video slots into one embedding and reads it with one linear head per target. Padded
slots (a study with fewer videos than slots) are masked out of the pooling.

Pooling modes (`pooling_mode`):

- `attentive`: the V-JEPA attentive probe (`AttentivePooler`) over the tokens of every clip of
  every video, with the padded slots' tokens masked out of the attention;
- `hierarchical`: attention within each clip, optionally with 3D RoPE, followed by learned
  clip summaries and study attention without positional encoding across clips;
- `mean`: each video's tokens are averaged, then the videos are averaged;
- `max`: each video's tokens are averaged, then the videos are max-pooled per feature;
- `attention`: each video's tokens are averaged, then a gated attention (Ilse et al., 2018,
  as in DeepCORO_CLIP) weighs the videos.

With `num_view_classes`, a learned view embedding is added to the tokens of every video slot
(DeepCORO_CLIP's view embeddings), so the pooling knows which view each video shows.
Soft view probabilities, whole-video view dropout and acquisition-type embeddings are optional.
Attention readouts can be shared, per target, or shared within explicit anatomical groups.

Each head is `regression` (one output, Huber/MSE/L1 loss on z-scored targets), `binary` (one
logit, BCE) or `classification` (`num_classes` logits, cross-entropy). A missing target (NaN)
takes the study out of that head's loss and metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attentive_pooler import AttentivePooler

POOLING_MODES = ("attentive", "hierarchical", "mean", "max", "attention")
QUERY_MODES = ("shared", "task", "group")
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
                 dropout=0.0, use_activation_checkpointing=True, num_view_classes=0,
                 query_mode="shared", task_groups=None, view_dropout=0.0, num_acquisition_classes=0,
                 local_depth=1, local_num_queries=4, local_use_rope=False, token_grid=None,
                 num_clips_per_video=1):
        super().__init__()
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"`pooling_mode` is one of {POOLING_MODES}, not {pooling_mode!r}.")
        self.pooling_mode = pooling_mode
        self.embed_dim = embed_dim
        self.specs = dict(heads)
        if query_mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}.")
        if query_mode != "shared" and pooling_mode not in ("attentive", "hierarchical"):
            raise ValueError("Task/group queries require attentive or hierarchical pooling.")
        if task_groups and query_mode != "group":
            raise ValueError("task_groups requires query_mode: group.")
        if query_mode == "group" and (not task_groups or set(task_groups) != set(heads)):
            raise ValueError("task_groups must map every target head, and only those heads, to a group.")
        if query_mode == "group" and any(not isinstance(g, str) or not g for g in task_groups.values()):
            raise ValueError("Every task group must have a nonempty string name.")
        groups = {h: "shared" if query_mode == "shared" else h if query_mode == "task" else task_groups[h]
                  for h in heads}
        self.query_names = list(dict.fromkeys(groups.values()))
        self.head_query = {h: self.query_names.index(g) for h, g in groups.items()}
        self.query_mode = query_mode
        if not 0 <= view_dropout <= 1 or (view_dropout and not num_view_classes):
            raise ValueError("view_dropout must be in [0, 1] and requires view embeddings.")
        self.view_dropout = float(view_dropout)
        self.num_clips_per_video = int(num_clips_per_video)
        if self.num_clips_per_video < 1:
            raise ValueError("num_clips_per_video must be positive.")
        if local_use_rope and pooling_mode != "hierarchical":
            raise ValueError("local_use_rope requires hierarchical pooling; study slots have no spatial coordinates.")
        # One learned embedding per view class, added to every token of a video of that view, plus
        # one for a padded slot (index `num_view_classes`), as DeepCORO_CLIP's view embeddings.
        self.num_view_classes = int(num_view_classes)
        self.view_embed = nn.Embedding(self.num_view_classes + 1, embed_dim) if self.num_view_classes else None
        if self.view_embed is not None:
            nn.init.trunc_normal_(self.view_embed.weight, std=0.02)
        self.num_acquisition_classes = int(num_acquisition_classes)
        self.acquisition_embed = (nn.Embedding(self.num_acquisition_classes + 1, embed_dim)
                                  if self.num_acquisition_classes else None)
        if self.acquisition_embed is not None:
            nn.init.trunc_normal_(self.acquisition_embed.weight, std=0.02)
        self.clip_pooler = None
        if pooling_mode == "hierarchical":
            if local_depth < 0 or local_num_queries < 1:
                raise ValueError("local_depth must be nonnegative and local_num_queries positive.")
            self.clip_pooler = AttentivePooler(
                num_queries=local_num_queries, embed_dim=embed_dim, num_heads=num_heads, depth=local_depth + 1,
                use_activation_checkpointing=use_activation_checkpointing, use_rope=local_use_rope,
                token_grid=token_grid,
            )
        if pooling_mode in ("attentive", "hierarchical"):
            self.pooler = AttentivePooler(num_queries=len(self.query_names), embed_dim=embed_dim,
                                          num_heads=num_heads, depth=depth,
                                          use_activation_checkpointing=use_activation_checkpointing)
            self.norm = nn.Identity()
        else:
            self.pooler = GatedAttention(embed_dim, attention_hidden, dropout) if pooling_mode == "attention" else None
            self.norm = nn.LayerNorm(embed_dim)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(spec["dropout"]), nn.Linear(embed_dim, spec["outputs"]))
            for name, spec in self.specs.items()
        })

    def pool(self, tokens, video_mask, view_ids=None, view_probs=None, acquisition_ids=None):
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
        slot_embedding = tokens.new_zeros(B, N, D)
        if self.view_embed is not None:
            if view_probs is not None:
                if view_probs.shape != (B, N, self.num_view_classes):
                    raise ValueError("view_probs must have shape [batch, videos, num_view_classes].")
                real = view_probs[video_mask]
                if (not bool(torch.isfinite(real).all()) or bool((real < 0).any())
                        or not torch.allclose(real.sum(-1), torch.ones_like(real[:, 0]), atol=1e-4)):
                    raise ValueError("Real videos need finite nonnegative view probabilities summing to one.")
                probs = view_probs.masked_fill(~video_mask.unsqueeze(-1), 0)
                embedding = probs.to(tokens.dtype) @ self.view_embed.weight[:self.num_view_classes].to(tokens.dtype)
            else:
                embedding = self._categorical_embedding(self.view_embed, view_ids, video_mask, "view_ids")
            # Drop the entire video's view signal, shared by all its clips/tokens. No rescaling.
            if self.training and self.view_dropout:
                keep = torch.rand(B, N, 1, device=tokens.device) >= self.view_dropout
                embedding = embedding * keep
            slot_embedding = slot_embedding + embedding.to(tokens.dtype)
        elif view_probs is not None:
            raise ValueError("view_probs requires view embeddings.")
        if self.acquisition_embed is not None:
            slot_embedding = slot_embedding + self._categorical_embedding(
                self.acquisition_embed, acquisition_ids, video_mask, "acquisition_ids").to(tokens.dtype)
        if self.clip_pooler is not None:
            C = self.num_clips_per_video
            if per_video % C:
                raise ValueError("Tokens must split evenly into clips within each video.")
            # Each clip has its own local coordinate system. Flatten only into the batch axis;
            # no local attention or rotary position ever spans clips.
            local = tokens.reshape(B, N, C, per_video // C, D)
            local = local.masked_fill(~video_mask[:, :, None, None, None], 0)
            local = self.clip_pooler(local.reshape(B * N * C, per_video // C, D))
            tokens = local.reshape(B, N * C * local.shape[1], D)
            per_video = tokens.shape[1] // N
        # Broadcast the per-slot embedding instead of materializing a [B, L, D] copy of it.
        tokens = (tokens.reshape(B, N, per_video, D) + slot_embedding[:, :, None]).masked_fill(
            ~video_mask[:, :, None, None], 0).reshape(B, N * per_video, D)
        if self.pooling_mode in ("attentive", "hierarchical"):
            key_padding_mask = (~video_mask).repeat_interleave(per_video, dim=1)  # True = ignore
            pooled = self.pooler(tokens, key_padding_mask=key_padding_mask)
            return pooled.squeeze(1) if self.query_mode == "shared" else pooled
        videos = tokens.reshape(B, N, per_video, D).mean(dim=2)  # [B, N, D]
        mask = video_mask.to(videos.dtype).unsqueeze(-1)
        if self.pooling_mode == "mean":
            return (videos * mask).sum(dim=1) / mask.sum(dim=1)
        if self.pooling_mode == "max":
            return videos.masked_fill(~video_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        pooled, _ = self.pooler(videos, video_mask)
        return pooled

    @staticmethod
    def _categorical_embedding(layer, ids, mask, name):
        if ids is None or ids.shape != mask.shape:
            raise ValueError(f"Pass each slot's `{name}` with shape [batch, videos].")
        count = layer.num_embeddings - 1
        if bool(((ids[mask] < 0) | (ids[mask] >= count)).any()):
            raise ValueError(f"View classes / acquisition classes must be in [0, {count}) for real videos.")
        return layer(ids.masked_fill(~mask, count))

    def forward(self, tokens, video_mask, view_ids=None, view_probs=None, acquisition_ids=None):
        """{head: [B, outputs]} from the tokens of a batch of studies."""
        embedding = self.norm(self.pool(tokens, video_mask, view_ids, view_probs, acquisition_ids))
        return {name: head(embedding if self.query_mode == "shared" else embedding[:, self.head_query[name]])
                for name, head in self.heads.items()}


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
    if task == "binary" and not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("A binary head's targets must be 0 or 1 (or missing).")
    if task == "classification" and not bool(((target == target.round()) & (target >= 0)
                                              & (target < output.shape[1])).all()):
        raise ValueError(f"A classification head's targets must be whole numbers in [0, {output.shape[1]}) "
                         "(or missing).")
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


def multitask_loss(specs, outputs, targets, counts=None, world_size=1):
    """The weighted sum of the heads' mean losses over a batch, and each head's (loss sum, count).

    `targets`: [B, heads] in `specs` order. Every head's term stays in the graph, so the set of
    parameters that take part in the backward pass is the same whatever the batch holds (as
    DDP's static graph needs); a head without a present target contributes zero.

    Under DDP the gradients are averaged over the ranks, so each rank divides a head's loss sum
    by the number of present targets over every rank (`counts`, {head: global count}) and
    multiplies by `world_size`: the mean over all the labeled studies of the global batch, not a
    mean of per-rank means. Without `counts`, the local counts are used.
    Returns (loss, {head: (sum, count)}).
    """
    total, parts = None, {}
    for i, (name, spec) in enumerate(specs.items()):
        value, count = head_loss(spec, outputs[name], targets[:, i])
        parts[name] = (value, count)
        denominator = count if counts is None else counts[name]
        term = spec["weight"] * value * (world_size / denominator if denominator else 0.0)
        total = term if total is None else total + term
    return total, parts
