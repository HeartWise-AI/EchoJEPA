# src/models/radar/text_encoder.py

"""Caption encoder for RADAR structure alignment, its momentum teacher, and caption tokenizers.

`TextEncoder` wraps a BERT-like backbone (anything returning `last_hidden_state` and holding `encoder.layer`): the
[CLS] state goes through a linear projection and is L2-normalised in float32. Backbones, by `build_text_encoder`:

- `hf_pretrained`: a pinned Hugging Face checkpoint (`name` and `revision`), safetensors weights only, no code from
  the model repository;
- `hf_random`: a randomly initialised Hugging Face architecture from a config (smoke runs);
- `tiny`: `TinyTextBackbone`, a small pure-PyTorch encoder that needs neither downloads nor transformers' torch
  support (tests, CPU smoke runs).

`MomentumTeacher` is RADAR's momentum text encoder: an exponential moving average of the student (backbone and
projection), without gradients, updated once per optimizer step. Its features weight the abnormal-abnormal targets.

Tokenizers return `(input_ids, attention_mask)` for a list of captions, the first position always a [CLS]-like
token: `HFTokenizer` wraps the pinned model's tokenizer, `HashTokenizer` hashes words (pairs with `tiny`). Both
truncate at `max_length`, and `lengths` says how many tokens each caption needs before that: training keeps only the
captions that fit, because flags and caption ids describe the whole text (`fit_captions` in
`evals/radar_frozen/train.py`).
"""

import copy
import hashlib
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_BACKBONES = ("hf_pretrained", "hf_random", "tiny")


class TextEncoder(nn.Module):

    def __init__(self, backbone, hidden_size, proj_dim=256, trainable_layers=None):
        """`trainable_layers`: None trains the whole backbone; k >= 0 freezes the embeddings and all but the top k
        encoder layers (the projection is always trained)."""
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(hidden_size, proj_dim)
        if trainable_layers is not None:
            layers = getattr(getattr(backbone, "encoder", None), "layer", None)
            if layers is None:
                raise ValueError("trainable_layers needs a backbone with `encoder.layer` (BERT-like).")
            if not 0 <= trainable_layers <= len(layers):
                raise ValueError(f"trainable_layers must be between 0 and {len(layers)}.")
            for p in backbone.parameters():
                p.requires_grad_(False)
            for layer in list(layers)[len(layers) - trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

    @classmethod
    def from_pretrained(cls, name, revision, proj_dim=256, trainable_layers=None):
        """A pinned Hugging Face checkpoint, safetensors weights only, no remote code."""
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(name, revision=revision, use_safetensors=True, trust_remote_code=False,
                                             add_pooling_layer=False)
        return cls(backbone, backbone.config.hidden_size, proj_dim, trainable_layers)

    @classmethod
    def from_config(cls, config, proj_dim=256, trainable_layers=None):
        """A randomly initialised Hugging Face backbone (smoke runs)."""
        from transformers import AutoModel
        backbone = AutoModel.from_config(config, add_pooling_layer=False)
        return cls(backbone, config.hidden_size, proj_dim, trainable_layers)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(self.proj(hidden[:, 0]).float(), dim=-1)


class TinyTextBackbone(nn.Module):
    """A small BERT-like encoder in plain PyTorch (token and position embeddings, post-norm transformer layers in
    `encoder.layer`), for tests and CPU smoke runs. Its layers have no biases, so no parameter has the exactly-zero
    gradient of an attention key bias, which an adaptive optimizer would move by rounding noise alone (this keeps
    one-process and multi-process runs comparable to float precision)."""

    def __init__(self, vocab_size=1000, hidden_size=32, num_layers=2, num_heads=2, max_length=64, dropout=0.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(max_length, hidden_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=2 * hidden_size, dropout=dropout,
                                       batch_first=True, bias=False)
            for _ in range(num_layers)
        ])
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embed(input_ids) + self.position_embed(positions)[None]
        padding = ~attention_mask.bool()
        for layer in self.encoder.layer:
            x = layer(x, src_key_padding_mask=padding)
        return SimpleNamespace(last_hidden_state=x)


class MomentumTeacher(nn.Module):
    """EMA copy of a student module, never trained by gradients and always in eval mode."""

    def __init__(self, student, momentum=0.995):
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1].")
        self.momentum = momentum
        self.model = copy.deepcopy(student)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def update(self, student):
        """One EMA step: teacher = m * teacher + (1 - m) * student; buffers are copied."""
        for t, s in zip(self.model.parameters(), student.parameters(), strict=True):
            t.mul_(self.momentum).add_(s.detach(), alpha=1.0 - self.momentum)
        for t, s in zip(self.model.buffers(), student.buffers(), strict=True):
            t.copy_(s)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class HashTokenizer:
    """Word-level tokenizer by hashing (BLAKE2b), id 1 = [CLS], 0 = padding: deterministic, no vocabulary file."""

    def __init__(self, vocab_size=1000, max_length=64):
        if vocab_size < 3 or max_length < 2:
            raise ValueError("vocab_size must be at least 3 and max_length at least 2.")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _word(self, word):
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        return 2 + int.from_bytes(digest, "big") % (self.vocab_size - 2)

    def __call__(self, texts):
        rows = [[1] + [self._word(w) for w in text.lower().split()][: self.max_length - 1] for text in texts]
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
        return ids, (ids > 0).long()

    def lengths(self, texts):
        """The tokens each text needs, before truncation."""
        return [1 + len(text.split()) for text in texts]


class HFTokenizer:
    """The pinned Hugging Face tokenizer, truncating at `max_length`."""

    def __init__(self, name, revision, max_length=256):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        self.max_length = max_length

    def __call__(self, texts):
        out = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        return out["input_ids"], out["attention_mask"]

    def lengths(self, texts):
        """The tokens each text needs, special tokens included, before truncation."""
        return [len(ids) for ids in self.tokenizer(list(texts), truncation=False)["input_ids"]]


def _hf_config(cfg):
    from transformers import AutoConfig
    return AutoConfig.for_model(cfg.get("model_type", "bert"), **cfg.get("config", {}))


def build_tokenizer(cfg):
    """The tokenizer of a text config, without its encoder."""
    kind = cfg.get("backbone")
    if kind == "hf_pretrained":
        return HFTokenizer(cfg["name"], cfg["revision"], cfg.get("max_length", 256))
    if kind == "hf_random":
        return HashTokenizer(_hf_config(cfg).vocab_size, cfg.get("max_length", 64))
    if kind == "tiny":
        return HashTokenizer(cfg.get("vocab_size", 1000), cfg.get("max_length", 64))
    raise ValueError(f"text backbone must be one of {TEXT_BACKBONES}, not {kind!r}.")


def build_text_encoder(cfg, proj_dim):
    """(TextEncoder, tokenizer) from a config: {backbone: hf_pretrained | hf_random | tiny, ...}."""
    kind = cfg.get("backbone")
    trainable = cfg.get("trainable_layers")
    tokenizer = build_tokenizer(cfg)
    if kind == "hf_pretrained":
        return TextEncoder.from_pretrained(cfg["name"], cfg["revision"], proj_dim, trainable), tokenizer
    if kind == "hf_random":
        return TextEncoder.from_config(_hf_config(cfg), proj_dim, trainable), tokenizer
    backbone = TinyTextBackbone(tokenizer.vocab_size, cfg.get("hidden_size", 32), cfg.get("num_layers", 2),
                                cfg.get("num_heads", 2), tokenizer.max_length, cfg.get("dropout", 0.0))
    return TextEncoder(backbone, backbone.hidden_size, proj_dim, trainable), tokenizer
