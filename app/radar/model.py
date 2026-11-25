"""RADAR video-text model: a V-JEPA 2.1 video encoder and a pluggable text encoder projected to a shared space.

The video encoder is built and loaded exactly like the frozen EF probe's (`vjepa_2_1_encoder.init_module`): the same
option whitelist, the same `module.` / `backbone.` prefix stripping and a strict load. Starting from the published
EchoJEPA checkpoint or an ICM checkpoint is a change of `visual.checkpoint` (and `visual.model_name` if the size
differs); widths, depths, patch and tubelet sizes are checked before loading so a mismatch fails with a clear message.

Text encoders:
  - "huggingface": any local Hugging Face encoder (`local_files_only` by default; nothing is downloaded). ICM
    reports are French, so choose a French or multilingual model; none is hard-coded.
  - "tiny": a small random-init transformer with a hashing word tokenizer, for tests and smoke runs only.
"""

from __future__ import annotations

import hashlib
import os
import re
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import app.vjepa_2_1.models.vision_transformer as vit
from app.radar.ontology import normalize_text
from app.vjepa_2_1.utils import token_validity
from evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder import ENCODER_OPTIONS
from src.models.attentive_pooler import AttentivePooler
from src.utils.checkpoint_loader import robust_checkpoint_loader

POOLING = ("attentive", "mean")


# --------------------------------------------------------------------------- video


def build_video_backbone(model_name: str, resolution: int, frames_per_clip: int, options: dict) -> nn.Module:
    unknown = sorted(set(options) - set(ENCODER_OPTIONS))
    if unknown:
        raise ValueError(f"Unknown V-JEPA 2.1 encoder options {unknown}; expected some of {ENCODER_OPTIONS}.")
    if model_name not in vit.__dict__ or not model_name.startswith("vit_"):
        raise ValueError(f"Unknown V-JEPA 2.1 model {model_name!r}.")
    return vit.__dict__[model_name](img_size=resolution, num_frames=frames_per_clip, **options)


def check_encoder_dimensions(backbone: nn.Module, state: dict, what: str = "The encoder checkpoint") -> None:
    """Compare the checkpoint's shapes with the configured model before a strict load, with an actionable message."""
    weight = state.get("patch_embed.proj.weight")
    if weight is None or weight.dim() != 5:
        raise ValueError(f"{what} has no V-JEPA 2.1 video patch embedding (patch_embed.proj.weight).")
    width, _, tubelet, patch, _ = weight.shape
    depth = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    found = {"embed_dim": width, "depth": depth, "tubelet_size": tubelet, "patch_size": patch}
    built = {"embed_dim": backbone.embed_dim, "depth": len(backbone.blocks), "tubelet_size": backbone.tubelet_size,
             "patch_size": backbone.patch_size}
    differ = {k: (found[k], built[k]) for k in found if found[k] != built[k]}
    if differ:
        detail = ", ".join(f"{k}: checkpoint {a} vs config {b}" for k, (a, b) in differ.items())
        raise ValueError(f"{what} does not match visual.model_name/visual.encoder ({detail}). An ICM checkpoint of "
                         "another size needs the matching visual.model_name (e.g. vit_base = 768 wide, vit_large = "
                         "1024) and visual.embed_dim.")


def load_video_weights(backbone: nn.Module, checkpoint: str, checkpoint_key: str = "target_encoder") -> dict:
    """Strictly load an encoder from a pretraining checkpoint, as the frozen probe does. Returns safe metadata."""
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("visual.checkpoint does not exist.")
    saved = robust_checkpoint_loader(checkpoint, map_location="cpu")
    if checkpoint_key not in saved:
        raise KeyError(f"The visual checkpoint has no `{checkpoint_key}`; its entries are {sorted(saved)}.")
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in saved[checkpoint_key].items()}
    check_encoder_dimensions(backbone, state)
    try:
        backbone.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError("The visual checkpoint does not load strictly into the configured encoder; check "
                           "visual.encoder against the pretraining config.") from e
    meta = {"checkpoint_key": checkpoint_key, "epoch": saved.get("epoch")}
    del saved
    return meta


class VideoEncoder(nn.Module):
    """Backbone tokens pooled to one clip embedding; padding-only tubelets are excluded from the pool."""

    def __init__(self, backbone: nn.Module, pooling: str = "attentive", pool_heads: int = 12, freeze: bool = True):
        super().__init__()
        if pooling not in POOLING:
            raise ValueError(f"visual.pooling must be one of {POOLING}.")
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.pooling = pooling
        self.frozen = freeze
        self.pool = AttentivePooler(num_queries=1, embed_dim=self.embed_dim, num_heads=pool_heads, depth=1) \
            if pooling == "attentive" else None
        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # a frozen encoder keeps its evaluation behaviour
        return self

    def forward(self, video: torch.Tensor, clip_indices: torch.Tensor | None = None) -> torch.Tensor:
        b, _, t, h, w = video.shape
        if h != w:
            raise ValueError("RADAR clips are square crops.")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            tokens = self.backbone(video)
        if clip_indices is None:
            valid = torch.ones(b, tokens.shape[1], device=tokens.device, dtype=torch.bool)
        else:
            grid = h // self.backbone.patch_size
            valid = token_validity(clip_indices.to(tokens.device), self.backbone.tubelet_size, grid)[..., 0] > 0
        if valid.shape[1] != tokens.shape[1]:
            raise ValueError("clip_indices do not match the encoder's token layout.")
        if self.pool is None:
            weights = valid.to(tokens.dtype).unsqueeze(-1)
            return (tokens * weights).sum(1) / weights.sum(1).clamp(min=1)
        return self.pool(tokens, key_padding_mask=~valid).squeeze(1)


# --------------------------------------------------------------------------- text


class HashingTokenizer:
    """A vocabulary-free word tokenizer (stable CRC32 hashing) for tests and smoke runs; not for real training."""

    PAD, CLS = 0, 1

    def __init__(self, vocab_size: int = 4096, max_length: int = 64):
        if vocab_size < 16 or max_length < 2:
            raise ValueError("vocab_size >= 16 and max_length >= 2 are required.")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def ids(self, text: str) -> list[int]:
        words = re.findall(r"\w+|[^\w\s]", normalize_text(text))
        return [self.CLS] + [2 + zlib.crc32(w.encode()) % (self.vocab_size - 2) for w in words][: self.max_length - 1]

    def __call__(self, texts: list[str]) -> dict:
        rows = [self.ids(t) for t in texts]
        length = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), length), self.PAD, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
        return {"input_ids": input_ids, "attention_mask": (input_ids != self.PAD).long()}


class TinyTextEncoder(nn.Module):
    """A small transformer over hashed words, returning the first (CLS) token. Deterministic, no dropout."""

    def __init__(self, vocab_size: int = 4096, dim: int = 64, depth: int = 2, heads: int = 4, max_length: int = 64):
        super().__init__()
        self.tokenizer = HashingTokenizer(vocab_size, max_length)
        self.embed_dim = dim
        self.tokens = nn.Embedding(vocab_size, dim, padding_idx=HashingTokenizer.PAD)
        self.positions = nn.Parameter(torch.zeros(1, max_length, dim))
        nn.init.normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4 * dim, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def identity(self) -> dict:
        return {"type": "tiny", "vocab_size": self.tokenizer.vocab_size, "dim": self.embed_dim}

    def files_sha256(self) -> None:
        return None  # no files: the weights are trained, so every checkpoint holds them

    def tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(texts)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.tokens(input_ids) + self.positions[:, : input_ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        return self.norm(x[:, 0])


class HuggingFaceTextEncoder(nn.Module):
    """A local Hugging Face encoder, mean- or CLS-pooled. Loaded with `local_files_only` unless told otherwise."""

    def __init__(self, name: str, local_files_only: bool = True, max_length: int = 128, pooling: str = "mean",
                 freeze: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        if pooling not in ("mean", "cls"):
            raise ValueError("text.pooling must be mean or cls.")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(name, local_files_only=local_files_only)
        except OSError as e:
            raise OSError("The text encoder is not available locally. Put an approved French or multilingual "
                          "encoder snapshot at text.name, or set text.local_files_only: false deliberately.") from e
        self.name = name
        self.max_length = max_length
        self.pooling = pooling
        self.embed_dim = self.model.config.hidden_size
        self.frozen = freeze
        self.local_files_only = local_files_only
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()  # a frozen encoder keeps its evaluation behaviour (no dropout), as the video one does
        return self

    def snapshot_dir(self) -> str:
        """The local directory the model and tokenizer were loaded from."""
        if os.path.isdir(self.name):
            return self.name
        from huggingface_hub import snapshot_download

        return snapshot_download(self.name, local_files_only=True)

    def files_sha256(self) -> str:
        """SHA-256 over every file of the snapshot (relative path and content): weights, configuration and tokenizer.
        Hidden files and folders (download metadata) are skipped."""
        root = self.snapshot_dir()
        h = hashlib.sha256()
        for folder, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(f for f in files if not f.startswith(".")):
                path = os.path.join(folder, name)
                h.update(os.path.relpath(path, root).encode() + b"\0")
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                h.update(b"\0")
        return h.hexdigest()

    def identity(self) -> dict:
        return {"type": "huggingface", "name": os.path.basename(os.path.normpath(self.name)),
                "revision": getattr(self.model.config, "_commit_hash", None),
                "model_type": self.model.config.model_type, "frozen": self.frozen}

    def tokenize(self, texts: list[str]) -> dict:
        return dict(self._tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors="pt"))

    def forward(self, **tokens) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            hidden = self.model(**tokens).last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


# --------------------------------------------------------------------------- joint model


def _projection(in_dim: int, out_dim: int, hidden: int | None) -> nn.Module:
    if not hidden:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class RadarModel(nn.Module):
    def __init__(self, video: VideoEncoder, text: nn.Module, proj_dim: int, proj_hidden: int | None = None):
        super().__init__()
        self.video = video
        self.text = text
        self.video_proj = _projection(video.embed_dim, proj_dim, proj_hidden)
        self.text_proj = _projection(text.embed_dim, proj_dim, proj_hidden)

    def encode_video(self, video: torch.Tensor, clip_indices: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(self.video_proj(self.video(video, clip_indices)), dim=-1)

    def encode_text(self, tokens: dict) -> torch.Tensor:
        return F.normalize(self.text_proj(self.text(**tokens)), dim=-1)

    def forward(self, video, clip_indices, tokens) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_video(video, clip_indices), self.encode_text(tokens)

    def frozen_prefixes(self) -> tuple[str, ...]:
        """State-dict prefixes not trained here; checkpoints omit them and record the source checkpoint instead."""
        prefixes = []
        if self.video.frozen:
            prefixes.append("video.backbone.")
        if getattr(self.text, "frozen", False):
            prefixes.append("text.model.")
        return tuple(prefixes)


def build_model(cfg: dict, resolution: int, frames_per_clip: int) -> tuple[RadarModel, dict]:
    """The model from the `visual`, `text` and `projection` config blocks, plus safe load metadata."""
    v = cfg["visual"]
    backbone = build_video_backbone(v["model_name"], resolution, frames_per_clip, dict(v.get("encoder") or {}))
    expected = v.get("embed_dim")
    if expected is not None and expected != backbone.embed_dim:
        raise ValueError(f"visual.embed_dim is {expected} but {v['model_name']} is {backbone.embed_dim} wide.")
    if v.get("checkpoint"):
        meta = load_video_weights(backbone, v["checkpoint"], v.get("checkpoint_key", "target_encoder"))
    elif v.get("allow_random_init"):
        meta = {"checkpoint_key": None, "epoch": None, "random_init": True}
    else:
        raise ValueError("visual.checkpoint is required; random initialisation needs visual.allow_random_init.")
    video = VideoEncoder(backbone, pooling=v.get("pooling", "attentive"), pool_heads=v.get("pool_heads", 12),
                         freeze=v.get("freeze", True))
    t = cfg["text"]
    if t["type"] == "tiny":
        text = TinyTextEncoder(vocab_size=t.get("vocab_size", 4096), dim=t.get("dim", 64), depth=t.get("depth", 2),
                               heads=t.get("heads", 4), max_length=t.get("max_length", 64))
    elif t["type"] == "huggingface":
        text = HuggingFaceTextEncoder(t["name"], local_files_only=t.get("local_files_only", True),
                                      max_length=t.get("max_length", 128), pooling=t.get("pooling", "mean"),
                                      freeze=t.get("freeze", False))
    else:
        raise ValueError("text.type must be huggingface or tiny.")
    p = cfg["projection"]
    return RadarModel(video, text, p["dim"], p.get("hidden")), meta
