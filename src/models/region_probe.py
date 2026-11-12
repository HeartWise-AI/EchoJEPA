# src/models/region_probe.py

"""Two-pass study probe that reads the regions the model segments itself (`pooling_mode: region`).

One study, one encoder, two encoder calls:

1. Pass 1 embeds every view of the study at `resolution`. A frozen `RegionMaskDecoder` reads
   the final tokens and the outputs of a few intermediate encoder blocks and labels every pixel
   with one of `num_classes` region classes.
2. For every region group (a named set of classes), the views where the group is seen are
   ranked by its mask area, and a square box follows the group's mask over the frames of the
   best `crops_per_group` views (`tracked_boxes`). The boxes are cut from the
   `source_resolution` frames and resized to `crop_resolution` (`crop_clip`).
3. Pass 2 embeds all the crops with the same encoder.
4. For each group, the crop tokens (weighted by a learned query and the group's mask share of
   each token) and the view tokens (weighted by the group's mask share) are pooled by the gated
   attention of `study_probe.GatedAttention` into one feature per group.
5. `readouts` turn the group features into the heads of `target_label`, so the losses, metrics,
   selection and checkpoints of the study probe apply unchanged.

Nothing here knows the anatomy: classes, groups, resolutions, tracking and readouts come from
`experiment.classifier.region` (see docs/study_multitask_probe.md). The coronary configuration is
configs/eval/vitl/study_multitask_deepcoro_jepa.yaml (DeepCORO-JEPA).
"""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.study_probe import GatedAttention

READOUT_KINDS = ("group_mlp", "group_linear", "groups_mlp")
REGION_KEYS = ("num_classes", "groups", "decoder", "probe_checkpoint", "source_resolution", "crop_resolution",
               "crops_per_group", "tracking", "readouts", "study_pooler", "output_range")
DECODER_KEYS = ("blocks", "width", "checkpoint")
TRACKING_KEYS = ("min_pixels", "min_frames", "margin", "min_side", "line_filter")
LINE_FILTER_KEYS = ("linearity", "length", "min_pixels")
READOUT_KEYS = ("kind", "heads", "hidden", "bias", "study_head", "groups")
STUDY_POOLER_KEYS = ("width", "queries", "num_heads", "bias")


def _block(where, value, allowed, required=()):
    """`value` as a config mapping with only `allowed` keys and every `required` key set."""
    if not isinstance(value, dict):
        raise ValueError(f"`{where}` is a mapping with the keys {list(allowed)}.")
    unknown = sorted(set(value) - set(allowed), key=str)
    if unknown:
        raise ValueError(f"`{where}` has unknown keys {unknown}; it takes {list(allowed)}.")
    missing = [k for k in required if value.get(k) is None]
    if missing:
        raise ValueError(f"`{where}` needs {missing}.")
    return value


def _positive(where, value, kind=int):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or kind(value) != value or not value > 0 \
            or not math.isfinite(value):
        raise ValueError(f"`{where}` is a positive {'integer' if kind is int else 'number'}, not {value!r}.")
    return kind(value)


def region_options(classifier, data, pretrained, specs):
    """Validate `experiment.classifier.region` against the heads (`specs`) and resolve it to the
    plain values that build `RegionStudyProbe` and enter the checkpoint fingerprint. An unknown
    key, a head no readout reads and a head read twice are errors."""
    where = "experiment.classifier.region"
    region = _block(where, classifier.get("region"), REGION_KEYS,
                    required=("num_classes", "groups", "decoder", "source_resolution", "crop_resolution",
                              "crops_per_group", "tracking", "readouts"))
    if classifier.get("query_mode", "shared") != "group":
        raise ValueError("Region pooling reads each head from its group: set `query_mode: group` and `task_groups`.")
    if classifier.get("use_view_embedding") or float(classifier.get("view_dropout", 0.0)):
        raise ValueError("Region pooling has no view embedding: leave `use_view_embedding` and `view_dropout` unset.")
    encoder = pretrained.get("pretrain_kwargs", {}).get("encoder", {})
    patch, tubelet = int(encoder.get("patch_size", 16)), int(encoder.get("tubelet_size", 2))
    frames, resolution = int(data.get("frames_per_clip", 16)), int(data.get("resolution", 224))
    source_resolution = _positive(f"{where}.source_resolution", region["source_resolution"])
    crop_resolution = _positive(f"{where}.crop_resolution", region["crop_resolution"])
    if frames % tubelet or resolution % patch or crop_resolution % patch:
        raise ValueError("Region pooling requires frames/resolution/crop_resolution divisible by tubelet/patch size.")
    if int(data.get("num_clips_per_video", 1)) != 1:
        raise ValueError("Region pooling reads one clip per video: set `num_clips_per_video: 1`.")
    crops_per_group = region["crops_per_group"]
    if isinstance(crops_per_group, bool) or not isinstance(crops_per_group, int) or crops_per_group < 0:
        raise ValueError(f"`{where}.crops_per_group` is a non-negative integer, not {crops_per_group!r}.")

    num_classes = _positive(f"{where}.num_classes", region["num_classes"])
    if not isinstance(region["groups"], dict) or not region["groups"]:
        raise ValueError(f"`{where}.groups` maps each group name to its region classes.")
    groups = {}
    for name, classes in region["groups"].items():
        if not isinstance(classes, (list, tuple)) or not classes or len(set(classes)) != len(classes) or any(
                isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < num_classes for c in classes):
            raise ValueError(f"`{where}.groups.{name}` lists distinct classes in [0, {num_classes}), not {classes!r}.")
        groups[str(name)] = [int(c) for c in classes]

    decoder = _block(f"{where}.decoder", region["decoder"], DECODER_KEYS, required=DECODER_KEYS)
    blocks = decoder["blocks"]
    if not isinstance(blocks, (list, tuple)) or len(set(blocks)) != len(blocks) or any(
            isinstance(b, bool) or not isinstance(b, int) or b < 0 for b in blocks):
        raise ValueError(f"`{where}.decoder.blocks` lists distinct encoder block indices (0-based), not {blocks!r}.")
    width = _positive(f"{where}.decoder.width", decoder["width"])
    if width % 8:
        raise ValueError(f"`{where}.decoder.width` is a multiple of 8 (group normalization), not {width}.")

    tracking = _block(f"{where}.tracking", region["tracking"], TRACKING_KEYS, required=TRACKING_KEYS[:4])
    line_filter = tracking.get("line_filter")
    if line_filter is not None:
        line_filter = _block(f"{where}.tracking.line_filter", line_filter, LINE_FILTER_KEYS, required=LINE_FILTER_KEYS)
        line_filter = {k: _positive(f"{where}.tracking.line_filter.{k}", line_filter[k], float)
                       for k in LINE_FILTER_KEYS}
    tracking = {"min_pixels": _positive(f"{where}.tracking.min_pixels", tracking["min_pixels"], float),
                "min_frames": _positive(f"{where}.tracking.min_frames", tracking["min_frames"]),
                "margin": float(tracking["margin"]),
                "min_side": _positive(f"{where}.tracking.min_side", tracking["min_side"], float),
                "line_filter": line_filter}
    if not 0 <= tracking["margin"] < 1 or tracking["min_side"] > 1:
        raise ValueError(f"`{where}.tracking.margin` and `min_side` are fractions of the frame's short side.")

    task_groups = classifier.get("task_groups") or {}
    if not isinstance(region["readouts"], dict) or not region["readouts"]:
        raise ValueError(f"`{where}.readouts` maps each readout name to its kind and heads.")
    readouts, read, grouped = {}, [], {}
    for name, readout in region["readouts"].items():
        here = f"{where}.readouts.{name}"
        readout = _block(here, readout, READOUT_KEYS, required=("kind", "heads"))
        kind = readout["kind"]
        if kind not in READOUT_KINDS:
            raise ValueError(f"`{here}.kind` is one of {READOUT_KINDS}, not {kind!r}.")
        heads = list(readout["heads"]) if isinstance(readout["heads"], (list, tuple)) else []
        if not heads:
            raise ValueError(f"`{here}.heads` lists the heads this readout reads.")
        unused = {"group_mlp": ("bias", "study_head", "groups"), "group_linear": ("hidden", "groups"),
                  "groups_mlp": ("bias", "study_head")}[kind]
        given = [k for k in unused if readout.get(k) is not None]
        if given:
            raise ValueError(f"`{here}` is a {kind} readout, which takes no {given}.")
        resolved = {"kind": kind, "heads": heads}
        if kind == "group_linear":
            bias = float(readout.get("bias", 0.0))
            if not math.isfinite(bias):
                raise ValueError(f"`{here}.bias` is a finite number.")
            resolved.update(bias=bias, study_head=readout.get("study_head"))
        else:
            if readout.get("hidden") is None:
                raise ValueError(f"`{here}` needs ['hidden'].")
            resolved["hidden"] = _positive(f"{here}.hidden", readout["hidden"])
        if kind == "groups_mlp":
            listed = readout.get("groups")
            if not isinstance(listed, (list, tuple)) or not listed or len(set(listed)) != len(listed) or any(
                    g not in groups for g in listed):
                raise ValueError(f"`{here}.groups` lists distinct groups of `{where}.groups`, not {listed!r}.")
            resolved["groups"] = list(listed)
        else:
            for head in heads:
                if task_groups.get(head) not in groups:
                    raise ValueError(f"`task_groups.{head}` must name a group of `{where}.groups` "
                                     f"(readout {name!r}), not {task_groups.get(head)!r}.")
                grouped[head] = task_groups[head]
        study_head = resolved.get("study_head")
        if study_head is not None:
            if study_head not in specs or specs[study_head]["task"] != "binary":
                raise ValueError(f"`{here}.study_head` names a binary head of `target_label`, not {study_head!r}.")
            wide = [h for h in heads if h in specs and specs[h]["outputs"] != 1]
            if wide:
                raise ValueError(f"`{here}` has a `study_head`, so its heads have one output; {wide} do not.")
            read.append(study_head)
        read += heads
        readouts[str(name)] = resolved
    unknown = sorted(set(read) - set(specs))
    if unknown:
        raise ValueError(f"`{where}.readouts` names heads {unknown} that `target_label` does not list.")
    twice = sorted({h for h in read if read.count(h) > 1})
    if twice:
        raise ValueError(f"`{where}.readouts` reads {twice} more than once.")
    unread = [h for h in specs if h not in read]
    if unread:
        raise ValueError(f"No readout of `{where}.readouts` reads the heads {unread}.")
    stray = sorted(set(task_groups) - set(grouped))
    if stray:
        raise ValueError(f"`task_groups` names heads {stray} that no group readout reads.")

    pooled = any(r.get("study_head") is not None for r in readouts.values())
    study_pooler = region.get("study_pooler")
    if pooled != (study_pooler is not None):
        raise ValueError(f"`{where}.study_pooler` is set exactly when a readout has a `study_head`.")
    if study_pooler is not None:
        study_pooler = _block(f"{where}.study_pooler", study_pooler, STUDY_POOLER_KEYS, required=STUDY_POOLER_KEYS[:3])
        bias = float(study_pooler.get("bias", 0.0))
        study_pooler = {k: _positive(f"{where}.study_pooler.{k}", study_pooler[k]) for k in STUDY_POOLER_KEYS[:3]}
        if study_pooler["width"] % study_pooler["num_heads"] or not math.isfinite(bias):
            raise ValueError(f"`{where}.study_pooler.width` is a multiple of `num_heads`, and `bias` is finite.")
        study_pooler["bias"] = bias

    output_range = {}
    for head, bounds in (region.get("output_range") or {}).items():
        if head not in specs or specs[head]["task"] != "regression":
            raise ValueError(f"`{where}.output_range.{head}` does not name a regression head of `target_label`.")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not all(
                isinstance(b, (int, float)) and math.isfinite(b) for b in bounds) or not bounds[0] < bounds[1]:
            raise ValueError(f"`{where}.output_range.{head}` is [low, high] with low < high, not {bounds!r}.")
        output_range[head] = [float(bounds[0]), float(bounds[1])]

    return dict(num_classes=num_classes, groups=groups, decoder={"blocks": [int(b) for b in blocks], "width": width},
                task_groups=grouped, resolution=resolution, source_resolution=source_resolution,
                crop_resolution=crop_resolution, crops_per_group=int(crops_per_group), tracking=tracking,
                readouts=readouts, study_pooler=study_pooler, output_range=output_range,
                frames_per_clip=frames, patch_size=patch, tubelet_size=tubelet)


def validate_blocks(blocks, depth):
    """Refuse decoder taps the encoder does not have (`blocks` are 0-based indices of its blocks)."""
    beyond = [b for b in blocks if not 0 <= b < depth]
    if beyond:
        raise ValueError(f"`region.decoder.blocks` {beyond} are outside the encoder's {depth} blocks.")


def token_fractions(prob, grid):
    """The share of each token's pixels that each class holds: `prob` [B, T, K, H, W] (class
    probabilities per pixel) -> [B, tokens, K] for a (T', H', W') token grid, in token order."""
    b, t, k, h, w = prob.shape
    tt, gh, gw = grid
    if t % tt:
        raise ValueError(f"{t} frames do not divide into {tt} token frames.")
    p = F.adaptive_avg_pool2d(prob.reshape(b * t, k, h, w), (gh, gw))
    p = p.reshape(b, tt, t // tt, k, gh, gw).mean(2)
    return p.permute(0, 1, 3, 4, 2).reshape(b, tt * gh * gw, k)


def tracked_boxes(mask, source_hw, min_pixels, min_frames, margin, min_side, line_filter=None):
    """One square box per frame that follows a region through a clip.

    mask: [T, h, w] bool, the region at the mask's resolution; source_hw: the frame size the
    boxes are for. A connected component of the mask's union over the frames is dropped when it
    is seen in fewer than `min_frames` frames (or all of them if the clip is shorter), or when
    `line_filter` reads it as a straight line: more than `min_pixels` pixels, a share above
    `linearity` of its spread along one axis, and a length (four standard deviations along that
    axis) above `length` of the mask height. The centre is the mask's centroid per frame,
    interpolated over the frames without mask and smoothed over three frames; the side is the
    90th percentile of the mask's per-frame extent plus `margin` on each side, at least
    `min_side` (both fractions of the frame's short side), at most the frame.

    Returns [T, 4] float32 (top, left, height, width) in source pixels, or None when fewer than
    `min_pixels` source pixels of mask remain.
    """
    t, rh, rw = mask.shape
    h, w = source_hw
    union = mask.any(0)
    n, components = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(union)
    for k in range(1, n):
        part = components == k
        if mask[:, part].any(1).sum() < min(min_frames, t):
            continue
        yy, xx = np.nonzero(part)
        if line_filter is not None and len(yy) > line_filter["min_pixels"]:
            spread = np.linalg.eigvalsh(np.cov(np.stack((yy, xx))))
            if spread[-1] / max(spread.sum(), 1e-9) > line_filter["linearity"] \
                    and 4 * np.sqrt(spread[-1]) > rh * line_filter["length"]:
                continue
        keep |= part
    mask = mask & keep[None]
    if mask.sum() * (h * w / (rh * rw)) < min_pixels:
        return None
    centre = np.full((t, 2), np.nan)
    extents = []
    for f in range(t):
        yy, xx = np.nonzero(mask[f])
        if len(yy):
            centre[f] = (yy.mean() * h / rh, xx.mean() * w / rw)
            extents.append(max(np.ptp(yy) * h / rh, np.ptp(xx) * w / rw))
    valid = np.isfinite(centre[:, 0])
    if not valid.any():
        return None
    for j in range(2):
        centre[:, j] = np.interp(np.arange(t), np.flatnonzero(valid), centre[valid, j])
        centre[:, j] = np.convolve(np.pad(centre[:, j], 1, mode="edge"), np.ones(3) / 3, mode="valid")
    side = min(max(float(np.percentile(extents, 90)) + 2 * margin * min(h, w), min_side * min(h, w)), h, w)
    top = np.clip(centre[:, 0] - side / 2, 0, h - side)
    left = np.clip(centre[:, 1] - side / 2, 0, w - side)
    return np.stack((top, left, np.full(t, side), np.full(t, side)), axis=1).astype(np.float32)


def crop_clip(clip, boxes, resolution):
    """Cut a tracked box out of every frame: `clip` [T, C, H, W], `boxes` [T, 4] (top, left,
    height, width in the clip's pixels) -> [T, C, resolution, resolution], bilinear."""
    t, _, h, w = clip.shape
    v = (torch.arange(resolution, device=clip.device, dtype=torch.float32) + 0.5) / resolution
    yy = boxes[:, 0, None, None] + boxes[:, 2, None, None] * v[None, :, None]
    xx = boxes[:, 1, None, None] + boxes[:, 3, None, None] * v[None, None, :]
    yy = (yy / h * 2 - 1).expand(t, resolution, resolution)
    xx = (xx / w * 2 - 1).expand(t, resolution, resolution)
    grid = torch.stack((xx, yy), dim=-1)
    return F.grid_sample(clip.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


def encode_with_taps(encoder, clips, clip_indices=None, blocks=()):
    """One call of the probe's encoder on `clips` [N, C, T, H, W]: the final tokens [N, tokens, D]
    and the raw output of each encoder block of `blocks`, read by forward hooks that are removed
    before returning."""
    captured, handles = {}, []

    def tap(index):
        def hook(module, inputs, output):
            if index in captured:
                raise RuntimeError(f"Encoder block {index} ran more than once in one encoder call.")
            captured[index] = output
        return hook

    try:
        for index in blocks:
            handles.append(getattr(encoder, "model", encoder).blocks[index].register_forward_hook(tap(index)))
        tokens = encoder([[clips]], None if clip_indices is None else [clip_indices])[0]
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(blocks) or any(captured[i].shape != tokens.shape for i in blocks):
        raise RuntimeError(f"The encoder blocks {list(blocks)} did not each give tokens shaped as the final ones.")
    return tokens, [captured[i] for i in blocks]


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(8, c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.GroupNorm(8, c_out), nn.GELU(),
        )


class RegionMaskDecoder(nn.Module):
    """Pixel-wise region classes from the encoder's tokens.

    The final tokens are projected to `width` channels; the output of each tapped block of
    `blocks` is projected too and added through a gate, tanh(gate) with the gate starting at 0.
    The token map is then upsampled in two steps, each fused with features of the grey frame
    itself (strides 4 and 2), to class logits per pixel.
    """

    def __init__(self, embed_dim, num_classes, width=128, blocks=()):
        super().__init__()
        self.embed_dim, self.num_classes, self.width, self.blocks = embed_dim, num_classes, width, tuple(blocks)
        self.token_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, width), nn.GELU())
        self.intermediate_proj = nn.ModuleDict({str(i): nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, width), nn.GELU()) for i in self.blocks})
        self.intermediate_gate = nn.Parameter(torch.zeros(len(self.blocks)))
        self.detail1 = ConvBlock(1, 32, stride=2)
        self.detail2 = ConvBlock(32, 48, stride=2)
        self.fuse2 = ConvBlock(width + 48, width)
        self.fuse1 = ConvBlock(width + 32, 64)
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, tokens, taps, frames, grid):
        """tokens: [B, tokens, D]; taps: one [B, tokens, D] per tapped block; frames: [B, T, H, W]
        grey in [0, 1]; grid: the (T', H', W') token grid. Returns logits [B, T, K, H, W]."""
        b, t, h, w = frames.shape
        tt, gh, gw = grid
        if tokens.shape != (b, tt * gh * gw, self.embed_dim) or len(taps) != len(self.blocks):
            raise ValueError(f"The region decoder expects {tt * gh * gw} tokens per clip and {len(self.blocks)} "
                             f"tapped blocks, not {tuple(tokens.shape)} and {len(taps)}: check patch_size and "
                             "tubelet_size in pretrain_kwargs.encoder.")
        if t % tt:
            raise ValueError(f"{t} frames do not divide into {tt} token frames.")
        z = self.token_proj(tokens)
        for j, (index, feature) in enumerate(zip(self.blocks, taps)):
            residual = self.intermediate_proj[str(index)](feature).to(z.dtype)
            z = z + torch.tanh(self.intermediate_gate[j]).to(z.dtype) * residual
        z = z.reshape(b, tt, gh, gw, self.width).permute(0, 1, 4, 2, 3)
        z = z.repeat_interleave(t // tt, dim=1).reshape(b * t, self.width, gh, gw)
        d1 = self.detail1(frames.reshape(b * t, 1, h, w))
        d2 = self.detail2(d1)
        z = F.interpolate(z, d2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.fuse2(torch.cat((z, d2), dim=1))
        z = F.interpolate(z, d1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.fuse1(torch.cat((z, d1), dim=1))
        z = F.interpolate(self.out(z), (h, w), mode="bilinear", align_corners=False)
        return z.reshape(b, t, self.num_classes, h, w)


class StudyTokenPooler(nn.Module):
    """Study-level findings from the pass-1 tokens of every view, with no further encoder call.

    `queries` learned queries per finding attend over all the tokens of the study. Each finding's
    pooled vector gives one study logit (`any`) and one offset per region group (`segment`,
    zero at the start), which the caller adds to that finding's per-group logits.
    """

    def __init__(self, embed_dim, num_groups, num_findings, width=256, queries=4, num_heads=8, bias=0.0):
        super().__init__()
        self.num_findings = num_findings
        self.proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, width))
        self.query = nn.Parameter(torch.randn(num_findings * queries, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.norm = nn.LayerNorm(queries * width)
        self.segment = nn.Parameter(torch.zeros(num_findings, num_groups, queries * width))
        self.any = nn.Parameter(torch.randn(num_findings, queries * width) * 0.002)
        self.any_bias = nn.Parameter(torch.full((num_findings,), float(bias)))

    def forward(self, tokens):
        """tokens: [views, tokens, D]. Returns the offsets [findings, groups] and the study logits [findings]."""
        x = self.proj(tokens.reshape(1, -1, tokens.shape[-1]).to(self.query.dtype))
        z = self.attn(self.query[None], x, x, need_weights=False)[0][0]
        z = self.norm((z + self.mlp(z)).reshape(self.num_findings, -1))
        return (z[:, None] * self.segment).sum(-1), (z * self.any).sum(-1) + self.any_bias


class RegionStudyProbe(nn.Module):
    """The two-pass probe of one study; the arguments are `region_options`' (see the module docstring).

    Readouts (`readouts`: {name: {kind, heads, ...}}), on the pooled feature of each group:

    - `group_mlp`: each head reads the group `task_groups` gives it, through LayerNorm, a linear
      layer to `hidden` and GELU, shared by the readout's heads of that group;
    - `group_linear`: LayerNorm (one per readout), then one linear layer per head, zero at the
      start with the initial `bias`. With `study_head`, the study pooler adds an offset per group
      to the readout's heads, and the head `study_head` is the pooler's study logit;
    - `groups_mlp`: the features of `groups` side by side, through LayerNorm, a linear layer to
      `hidden` and GELU, shared by the readout's heads.

    Each head ends in the head's own dropout (`head_dropout`) and linear layer. A regression head
    in `output_range` is squashed into [low, high] and expressed with the head's target
    normalization (`normalization`: {head: {mean, std}}), as the loss and the metrics expect.

    The region decoder is a frozen submodule (saved with the probe); the encoder is given at
    each call, so one encoder serves both passes and trains under the caller's optimizer.
    """

    def __init__(self, embed_dim, heads, num_classes, groups, decoder, task_groups, resolution, source_resolution,
                 crop_resolution, crops_per_group, tracking, readouts, study_pooler=None, output_range=None,
                 frames_per_clip=16, patch_size=16, tubelet_size=2, attention_hidden=256, dropout=0.0,
                 normalization=None, pixel_normalization=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))):
        super().__init__()
        self.embed_dim = embed_dim
        self.specs = dict(heads)
        self.groups = {name: list(classes) for name, classes in groups.items()}
        self.task_groups = dict(task_groups)
        self.resolution, self.source_resolution, self.crop_resolution = resolution, source_resolution, crop_resolution
        self.crops_per_group, self.tracking = crops_per_group, dict(tracking)
        self.frames_per_clip, self.patch_size, self.tubelet_size = frames_per_clip, patch_size, tubelet_size
        self.readouts = {name: dict(readout) for name, readout in readouts.items()}
        self.output_range = {}
        for head, (low, high) in (output_range or {}).items():
            scale = (normalization or {}).get(head)
            if scale is None:
                raise ValueError(f"`output_range.{head}` needs the target normalization of {head!r}.")
            self.output_range[head] = (float(low), float(high), float(scale["mean"]), float(scale["std"]))
        mean, std = pixel_normalization
        self.register_buffer("pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1, 1), persistent=False)

        self.region_decoder = RegionMaskDecoder(embed_dim, num_classes, decoder["width"], decoder["blocks"])
        self.region_decoder.requires_grad_(False)
        num_groups = len(self.groups)
        self.crop_query = nn.Parameter(torch.zeros(num_groups, embed_dim))
        self.mask_weight = nn.Parameter(torch.ones(num_groups))
        self.view_proj = nn.Sequential(nn.LayerNorm(2 * embed_dim + 2), nn.Linear(2 * embed_dim + 2, embed_dim),
                                       nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.type_embedding = nn.Parameter(torch.zeros(2, embed_dim))  # 0: a crop, 1: a whole view.
        self.group_embedding = nn.Parameter(torch.zeros(num_groups, embed_dim))
        self.attention = GatedAttention(embed_dim, hidden=attention_hidden, dropout=dropout)

        def head(name, features):
            return nn.Sequential(nn.Dropout(self.specs[name]["dropout"]), nn.Linear(features, self.specs[name]["outputs"]))

        def trunk(features, hidden):
            return nn.Sequential(nn.LayerNorm(features), nn.Linear(features, hidden), nn.GELU())

        self.trunks, self.heads = nn.ModuleDict(), nn.ModuleDict()
        for name, readout in self.readouts.items():
            if readout["kind"] == "group_mlp":
                used = dict.fromkeys(self.task_groups[h] for h in readout["heads"])
                self.trunks[name] = nn.ModuleDict({g: trunk(embed_dim, readout["hidden"]) for g in used})
            elif readout["kind"] == "groups_mlp":
                self.trunks[name] = trunk(len(readout["groups"]) * embed_dim, readout["hidden"])
            else:
                self.trunks[name] = nn.LayerNorm(embed_dim)
            for h in readout["heads"]:
                self.heads[h] = head(h, embed_dim if readout["kind"] == "group_linear" else readout["hidden"])
                if readout["kind"] == "group_linear":
                    nn.init.zeros_(self.heads[h][1].weight)
                    nn.init.constant_(self.heads[h][1].bias, readout["bias"])
        self.findings = [name for name, readout in self.readouts.items() if readout.get("study_head") is not None]
        self.study_pooler = None
        if self.findings:
            self.study_pooler = StudyTokenPooler(embed_dim, num_groups, len(self.findings), width=study_pooler["width"],
                                                 queries=study_pooler["queries"], num_heads=study_pooler["num_heads"],
                                                 bias=study_pooler["bias"])

    def token_grid(self, resolution):
        return (self.frames_per_clip // self.tubelet_size, resolution // self.patch_size, resolution // self.patch_size)

    def encode(self, encoder, clips, clip_indices=None, blocks=()):
        """Normalize `clips` [N, C, T, H, W] (in [0, 1]) and embed them in one encoder call."""
        clips = (clips - self.pixel_mean) / self.pixel_std
        tokens, taps = encode_with_taps(encoder, clips, clip_indices, blocks)
        expected = math.prod(self.token_grid(clips.shape[-1]))
        if tokens.shape[1] != expected:
            raise ValueError(f"Region pooling assumes {expected} tokens per {clips.shape[-1]} px clip, but the encoder "
                             f"gives {tokens.shape[1]}: set patch_size and tubelet_size in pretrain_kwargs.encoder "
                             "to the encoder's.")
        return tokens, taps

    def find_crops(self, labels, source_hw):
        """The crops of pass 2, from the per-pixel classes of pass 1 (`labels` [views, T, h, w],
        numpy): for each group, the views ranked by the group's mask area that give a tracked
        box, at most `crops_per_group`. Returns [(view, group index, boxes [T, 4] numpy)]."""
        crops = []
        for gi, classes in enumerate(self.groups.values()):
            found = []
            for vi in range(len(labels)):
                mask = np.isin(labels[vi], classes)
                boxes = tracked_boxes(mask, source_hw, **self.tracking) if self.crops_per_group else None
                if boxes is not None:
                    found.append((float(mask.sum()), vi, boxes))
            found.sort(key=lambda c: -c[0])  # stable: ties keep the view order.
            crops.extend((vi, gi, boxes) for _, vi, boxes in found[:self.crops_per_group])
        return crops

    def pool(self, tokens, fractions, crops, crop_tokens, crop_fractions):
        """One feature per group [groups, D] from the view tokens and the group's crop tokens."""
        dtype = self.crop_query.dtype
        view_mean = tokens.mean(1)
        features = []
        for gi, classes in enumerate(self.groups.values()):
            share = fractions[:, :, classes].sum(-1)
            mass = share.sum(1)
            pooled = (share[:, :, None].to(tokens.dtype) * tokens).sum(1) / mass[:, None].clamp_min(1e-6)
            view = torch.cat((pooled, view_mean, torch.log1p(mass)[:, None], torch.ones_like(mass[:, None])), dim=-1)
            instances = self.view_proj(view.to(dtype)) + self.group_embedding[gi] + self.type_embedding[1]
            chosen = [i for i, (_, group, _) in enumerate(crops) if group == gi]
            if chosen:
                ct, cf = crop_tokens[chosen], crop_fractions[chosen]
                logits = (ct * self.crop_query[gi][None, None]).sum(-1) / ct.shape[-1] ** 0.5
                logits = logits + self.mask_weight[gi] * (cf + 1e-3).log()
                crop = (logits.softmax(-1)[:, :, None].to(ct.dtype) * ct).sum(1) + self.type_embedding[0]
                instances = torch.cat((crop.to(instances.dtype), instances), dim=0)
            mask = torch.ones(1, len(instances), dtype=torch.bool, device=instances.device)
            features.append(self.attention(instances[None], mask)[0][0])
        return torch.stack(features)

    def read(self, features, tokens):
        """The heads' outputs {head: [1, outputs]} from the group features and the pass-1 tokens."""
        dtype = self.crop_query.dtype
        index = {name: i for i, name in enumerate(self.groups)}
        outputs = {}
        if self.study_pooler is not None:
            offsets, study_logits = self.study_pooler(tokens)
        for name, readout in self.readouts.items():
            if readout["kind"] == "group_mlp":
                hidden = {g: module(features[index[g]].to(dtype)) for g, module in self.trunks[name].items()}
                for h in readout["heads"]:
                    outputs[h] = self.heads[h](hidden[self.task_groups[h]])
            elif readout["kind"] == "groups_mlp":
                hidden = self.trunks[name](features[[index[g] for g in readout["groups"]]].flatten().to(dtype))
                for h in readout["heads"]:
                    outputs[h] = self.heads[h](hidden)
            else:
                normed = self.trunks[name](features.to(dtype))
                finding = self.findings.index(name) if readout.get("study_head") is not None else None
                for h in readout["heads"]:
                    gi = index[self.task_groups[h]]
                    outputs[h] = self.heads[h](normed[gi])
                    if finding is not None:
                        outputs[h] = outputs[h] + offsets[finding, gi].to(outputs[h].dtype)
                if finding is not None:
                    outputs[readout["study_head"]] = study_logits[finding].reshape(1)
        for h, (low, high, mean, std) in self.output_range.items():
            outputs[h] = (low + (high - low) * torch.sigmoid(outputs[h].float()) - mean) / std
        return {h: outputs[h].float()[None] for h in self.specs}

    def forward_study(self, encoder, clips, video_mask, clip_indices=None):
        """One study through both passes.

        clips: [slots, C, T, S, S] in [0, 1], whole frames at `source_resolution`; video_mask:
        [slots] (True = a real video); clip_indices: [slots, T] or None. Returns the heads'
        outputs {head: [1, outputs]} and what the model looked at: `mask_logits`
        [views, T, K, resolution, resolution], `crops` [(view, group name, boxes [T, 4])] and
        `encoder_passes` (2, or 1 when no group gave a crop).
        """
        clips = clips[video_mask]
        indices = None if clip_indices is None else clip_indices[video_mask]
        views, channels, frames, height, width = clips.shape
        if not views:
            raise ValueError("A study needs at least one video.")
        if frames != self.frames_per_clip or (height, width) != (self.source_resolution, self.source_resolution):
            raise ValueError(f"Region pooling expects {self.frames_per_clip} frames of {self.source_resolution} px "
                             f"(`region.source_resolution`), not {frames} of {height}x{width}.")
        res = self.resolution
        full = F.interpolate(clips.transpose(1, 2).flatten(0, 1), (res, res), mode="bilinear", align_corners=False)
        full = full.reshape(views, frames, channels, res, res).transpose(1, 2)
        tokens, taps = self.encode(encoder, full, indices, self.region_decoder.blocks)             # pass 1
        mask_logits = self.region_decoder(tokens, taps, full.mean(1), self.token_grid(res))
        prob = mask_logits.float().softmax(2)
        fractions = token_fractions(prob, self.token_grid(res))
        crops = self.find_crops(prob.argmax(2).cpu().numpy(), (height, width))
        crop_tokens = crop_fractions = None
        if crops:
            classes = list(self.groups.values())
            scale = clips.new_tensor((res / height, res / width, res / height, res / width))
            boxes = [torch.from_numpy(b).to(clips.device) for _, _, b in crops]
            pixels = torch.stack([crop_clip(clips[vi].transpose(0, 1), b, self.crop_resolution).transpose(0, 1)
                                  for (vi, _, _), b in zip(crops, boxes)])
            crop_grid = self.token_grid(self.crop_resolution)
            crop_fractions = torch.stack([
                token_fractions(crop_clip(prob[vi][:, classes[gi]].sum(1, keepdim=True), b * scale,
                                          self.crop_resolution)[None], crop_grid)[0, :, 0]
                for (vi, gi, _), b in zip(crops, boxes)])
            crop_indices = None if indices is None else indices[[vi for vi, _, _ in crops]]
            crop_tokens, _ = self.encode(encoder, pixels, crop_indices)                            # pass 2
        features = self.pool(tokens, fractions, crops, crop_tokens, crop_fractions)
        names = list(self.groups)
        regions = {"mask_logits": mask_logits, "crops": [(vi, names[gi], b) for vi, gi, b in crops],
                   "encoder_passes": 2 if crops else 1, "features": features}
        return self.read(features, tokens), regions

    def forward(self, encoder, clips, video_mask, clip_indices=None):
        return self.forward_study(encoder, clips, video_mask, clip_indices)[0]
