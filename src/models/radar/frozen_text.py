# src/models/radar/frozen_text.py

"""Frozen caption encoder with cached caption features (`text.frozen: true`).

The backbone is never trained: each distinct caption is encoded once, before training, into one pooled feature
vector, and only a small head on those features is trained (with the structure pooler). The language model never
runs inside a training step. Pieces:

- `FrozenBackbone` loads and runs the frozen model: a pinned Hugging Face checkpoint (`hf_pretrained`: a BERT-like
  encoder, the language model of a Gemma 3 checkpoint such as MedGemma, or the text tower of a SigLIP checkpoint such
  as MedSigLIP; safetensors weights only, no code from the model repository), a random Hugging Face architecture
  (`hf_random`) or `TinyTextBackbone` (`tiny`), both seeded by `init_seed` (tests, smoke runs). It is outside every
  `nn.Module` of the RADAR model, so it has no parameter in the optimizer, the checkpoint or DDP, and it runs in eval
  mode under `torch.no_grad`.
- Pooling (`pooling`, at hidden layer `layer`, default -1 = the last): `mean` over the caption's tokens (special
  tokens such as [CLS], [SEP] or <bos> left out), `last` (the last caption token: for a decoder-only model, the only
  one that has read the whole caption), `cls` (the first position), `pooler` (the model's own pooled output, e.g.
  MedSigLIP's text embedding; the caption is then padded to `max_length` as that model was trained). `template` (one
  `{}` slot) wraps each caption before tokenization, e.g. a prompt ending the text for `last` pooling. Captions are
  padded on the right, so with causal attention a caption's features do not depend on the rest of its batch.
- `CaptionFeatureStore`: distinct text -> feature row. With a cache folder (`data.text_cache_dir`), rows are kept on
  disk in `text_features_<key>.pt`, keyed by `feature_key`: the backbone's identity (hub name and revision, or the
  sha256 of a local folder's files, or of a random backbone's weights), pooling, layer, template, max_length, dtype
  and `FEATURE_VERSION`. The file stores sha256 digests of the texts, never the texts. At load the stored key must
  equal the expected one and the features must be finite with the right width; otherwise the file is rebuilt, never
  reused. A write keeps the file's rows that this store does not hold (another run's captions or prompts). The
  folder digest is remembered in `weights_digest.json` beside the features, by file names, sizes and modification
  times. Training fills the store with every caption before the first step, then seals it: a caption missing then
  is an error, not an encoding inside the step. Unseen texts are encoded on the device of the head.
- `CachedTokenizer` is what the loader and the zero-shot code call as the tokenizer: it maps texts to store rows
  (ids [N, 1], mask of ones) and encodes texts it has not seen when the store is not sealed (zero-shot prompts go
  through the same features as captions). `lengths` and `max_length` are those of the backbone's tokenizer, so
  `fit_captions` measures captions against the frozen model's limit (64 tokens for MedSigLIP).
- `FrozenTextEncoder` is the trainable part: optional LayerNorm on the features, then a linear layer or a two-layer
  MLP (`head`: mlp by default, `head_hidden_dim` 1024) to `proj_dim`, L2-normalised in float32. It holds the feature
  key as a buffer: loading a state dict whose key differs (a head trained on other features) is refused.
"""

import hashlib
import json
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.radar.text_encoder import HashTokenizer, TinyTextBackbone, _hf_config

FEATURE_VERSION = 1
POOLINGS = ("mean", "last", "cls", "pooler")
HEADS = ("linear", "mlp")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
SETTINGS = {"pooling": "mean", "layer": -1, "template": "{}", "dtype": "float32", "head": "mlp",
            "head_hidden_dim": 1024, "input_norm": True, "init_seed": 0}
SKIPPED_FILES = ("README.md",)
_DIGESTS = {}          # in-process memo of folder digests, keyed by the files' (name, size, mtime_ns)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frozen_settings(cfg):
    """The frozen-encoder settings of a text config, with defaults; refuses what cannot be built as written."""
    if cfg.get("trainable_layers") is not None:
        raise ValueError("text.frozen trains no backbone layer: leave text.trainable_layers unset.")
    s = {k: cfg.get(k, v) for k, v in SETTINGS.items()}
    if s["pooling"] not in POOLINGS:
        raise ValueError(f"text.pooling must be one of {POOLINGS}, not {s['pooling']!r}.")
    if s["head"] not in HEADS:
        raise ValueError(f"text.head must be one of {HEADS}, not {s['head']!r}.")
    if s["dtype"] not in DTYPES:
        raise ValueError(f"text.dtype must be one of {tuple(DTYPES)}, not {s['dtype']!r}.")
    if not isinstance(s["template"], str) or s["template"].count("{}") != 1:
        raise ValueError("text.template needs exactly one {} slot for the caption.")
    if s["pooling"] == "pooler" and s["layer"] != -1:
        raise ValueError("pooling: pooler reads the model's own pooled output; leave text.layer at -1.")
    if isinstance(s["layer"], bool) or not isinstance(s["layer"], int):
        raise ValueError("text.layer must be an integer index into the hidden states (-1 = last).")
    return s


def folder_digest(path, memo_file=None):
    """sha256 over the regular files of a model folder (names and contents; README left out). Memoised by the files'
    names, sizes and modification times, in this process and in `memo_file` when given."""
    path = os.path.realpath(path)
    names = sorted(n for n in os.listdir(path)
                   if os.path.isfile(os.path.join(path, n)) and n not in SKIPPED_FILES and not n.startswith("."))
    stats = [[n, os.stat(os.path.join(path, n)).st_size, os.stat(os.path.join(path, n)).st_mtime_ns] for n in names]
    stamp = json.dumps([path, stats])
    if stamp in _DIGESTS:
        return _DIGESTS[stamp]
    memo = {}
    if memo_file and os.path.exists(memo_file):
        try:
            with open(memo_file) as f:
                memo = json.load(f)
        except (OSError, ValueError):
            memo = {}
    if stamp not in memo:
        h = hashlib.sha256()
        for name in names:
            h.update(name.encode() + b"\0")
            with open(os.path.join(path, name), "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            h.update(b"\0")
        memo[stamp] = h.hexdigest()
        if memo_file:
            os.makedirs(os.path.dirname(memo_file), exist_ok=True)
            tmp = memo_file + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(memo, f)
            os.replace(tmp, memo_file)
    _DIGESTS[stamp] = memo[stamp]
    return memo[stamp]


def state_digest(module):
    """sha256 over a module's state dict (names, shapes, float32 values)."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode() + str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenBackbone:
    """Loads, runs and identifies the frozen text model of a text config (see the module docstring)."""

    def __init__(self, cfg, memo_dir=None):
        self.cfg = dict(cfg)
        self.kind = cfg.get("backbone")
        if self.kind not in ("hf_pretrained", "hf_random", "tiny"):
            raise ValueError(f"text backbone must be hf_pretrained, hf_random or tiny, not {self.kind!r}.")
        self.settings = frozen_settings(cfg)
        self.max_length = int(cfg.get("max_length", 256 if self.kind == "hf_pretrained" else 64))
        self.memo_dir = memo_dir
        self._tokenizer, self._model, self._hf_config, self._random, self._identity = None, None, None, None, None

    # ---- light parts: config, tokenizer, identity --------------------------------------------------------------

    def hf_config(self):
        if self._hf_config is None:
            if self.kind == "hf_pretrained":
                from transformers import AutoConfig
                self._hf_config = AutoConfig.from_pretrained(self.cfg["name"], revision=self.cfg["revision"],
                                                             trust_remote_code=False)
            elif self.kind == "hf_random":
                self._hf_config = _hf_config(self.cfg)
        return self._hf_config

    def text_config(self):
        config = self.hf_config()
        return config.get_text_config() if hasattr(config, "get_text_config") else config

    @property
    def hidden_size(self):
        if self.kind == "tiny":
            return int(self.cfg.get("hidden_size", 32))
        return int(self.text_config().hidden_size)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.kind == "hf_pretrained":
                from transformers import AutoTokenizer
                limit = getattr(self.text_config(), "max_position_embeddings", None)
                if limit is not None and self.max_length > limit:
                    raise ValueError(f"text.max_length {self.max_length} exceeds the {limit} positions of "
                                     f"{self.cfg['name']}.")
                tok = AutoTokenizer.from_pretrained(self.cfg["name"], revision=self.cfg["revision"],
                                                    trust_remote_code=False)
                tok.padding_side = "right"
                if tok.pad_token is None:
                    raise ValueError(f"The tokenizer of {self.cfg['name']} has no padding token.")
                self._tokenizer = tok
            else:
                vocab = self.hf_config().vocab_size if self.kind == "hf_random" else self.cfg.get("vocab_size", 1000)
                self._tokenizer = HashTokenizer(vocab, self.max_length)
        return self._tokenizer

    def wrap(self, texts):
        return [self.settings["template"].replace("{}", t) for t in texts]

    def lengths(self, texts):
        """The tokens each text needs, template and special tokens included, before truncation."""
        if self.kind != "hf_pretrained":
            return self.tokenizer.lengths(self.wrap(texts))
        return [len(ids) for ids in self.tokenizer(self.wrap(texts), truncation=False)["input_ids"]]

    def identity(self):
        """What fixes the backbone's weights and tokenizer, without loading the model when it can."""
        if self._identity is None:
            if self.kind == "hf_pretrained" and os.path.isdir(self.cfg["name"]):
                memo = os.path.join(self.memo_dir, "weights_digest.json") if self.memo_dir else None
                self._identity = {"backbone": self.kind, "folder_sha256": folder_digest(self.cfg["name"], memo)}
            elif self.kind == "hf_pretrained":
                self._identity = {"backbone": self.kind, "name": self.cfg["name"], "revision": self.cfg["revision"]}
            else:
                keys = (("vocab_size", "hidden_size", "num_layers", "num_heads") if self.kind == "tiny"
                        else ("model_type", "config"))
                config = {k: self.cfg.get(k) for k in keys}
                self._identity = {"backbone": self.kind, "config": config,
                                  "weights_sha256": state_digest(self._random_model())}
        return self._identity

    def key(self):
        s = self.settings
        return {"version": FEATURE_VERSION, **self.identity(), "pooling": s["pooling"], "layer": s["layer"],
                "template": s["template"], "max_length": self.max_length, "dtype": s["dtype"]}

    # ---- the model ----------------------------------------------------------------------------------------------

    def _random_model(self):
        """The seeded random backbone (tiny, hf_random), built without touching the global RNG streams."""
        if self._random is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.settings["init_seed"]))
                if self.kind == "tiny":
                    c = self.cfg
                    model = TinyTextBackbone(c.get("vocab_size", 1000), c.get("hidden_size", 32),
                                             c.get("num_layers", 2), c.get("num_heads", 2), self.max_length, 0.0)
                else:
                    from transformers import AutoModel
                    model = AutoModel.from_config(self.hf_config())
            self._random = model.eval().requires_grad_(False)
        return self._random

    def load(self, device):
        """The frozen model on `device`, in eval mode, without gradients."""
        if self._model is None:
            if self.kind == "hf_pretrained":
                from transformers import AutoModel
                model = AutoModel.from_pretrained(self.cfg["name"], revision=self.cfg["revision"],
                                                  use_safetensors=True, trust_remote_code=False,
                                                  dtype=DTYPES[self.settings["dtype"]])
                # the text part only: Gemma 3 keeps its language model in `language_model`, SigLIP its text
                # tower in `text_model`; their vision towers are dropped
                for part in ("language_model", "text_model"):
                    if getattr(model, part, None) is not None:
                        model = getattr(model, part)
                        break
            else:
                model = self._random_model().to(DTYPES[self.settings["dtype"]])
            self._model = model.eval().requires_grad_(False)
        self._model.to(device)
        return self._model

    def release(self):
        """Drop the model (its features stay in the store)."""
        self._model, self._random = None, None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _tokenize(self, texts):
        """(ids, mask, special) [N, L] of wrapped texts; `special` marks [CLS]-like and other special tokens."""
        texts = self.wrap(texts)
        if self.kind != "hf_pretrained":
            ids, mask = self.tokenizer(texts)
            special = torch.zeros_like(mask, dtype=torch.bool)
            special[:, 0] = True
            return ids, mask, special
        padding = "max_length" if self.settings["pooling"] == "pooler" else True
        out = self.tokenizer(texts, padding=padding, truncation=True, max_length=self.max_length,
                             return_special_tokens_mask=True, return_attention_mask=True, return_tensors="pt")
        return out["input_ids"], out["attention_mask"], out["special_tokens_mask"].bool()

    @torch.no_grad()
    def encode(self, texts, device="cpu", batch_size=64):
        """Pooled features [N, H] (float32, CPU) of `texts`, in order."""
        model = self.load(device)
        s, out = self.settings, []
        decoder = bool(getattr(self.text_config(), "is_decoder", False)) or "gemma" in str(
            getattr(self.text_config(), "model_type", ""))
        for start in range(0, len(texts), batch_size):
            ids, mask, special = self._tokenize(texts[start:start + batch_size])
            ids, mask, special = ids.to(device), mask.to(device), special.to(device)
            if self.kind == "tiny":
                result = model(ids, mask)
            else:
                kwargs = {"output_hidden_states": s["pooling"] != "pooler"}
                if decoder:
                    kwargs["use_cache"] = False
                if s["pooling"] != "pooler" or "attention_mask" in self.tokenizer.model_input_names:
                    kwargs["attention_mask"] = mask
                result = model(input_ids=ids, **kwargs)
            out.append(pool(result, mask, special, s["pooling"], s["layer"]).float().cpu())
        return torch.cat(out) if out else torch.zeros(0, self.hidden_size)


def pool(result, mask, special, pooling, layer=-1):
    """One vector per text from a model output (see the module docstring)."""
    if pooling == "pooler":
        pooled = getattr(result, "pooler_output", None)
        if pooled is None:
            raise ValueError("pooling: pooler needs a model with its own pooled output (e.g. a SigLIP text tower).")
        return pooled
    states = getattr(result, "hidden_states", None)
    if states is None and layer != -1:
        raise ValueError("This backbone returns no hidden states: only layer -1 can be read.")
    hidden = states[layer] if states is not None else result.last_hidden_state
    hidden = hidden.float()
    if pooling == "cls":
        return hidden[:, 0]
    keep = mask.bool() & ~special
    keep = torch.where(keep.any(dim=1, keepdim=True), keep, mask.bool())      # an empty text: its special tokens
    if pooling == "mean":
        w = keep.float().unsqueeze(-1)
        return (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    positions = torch.arange(keep.shape[1], device=keep.device).expand_as(keep)
    last = torch.where(keep, positions, -1).max(dim=1).values.clamp_min(0)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class CaptionFeatureStore:
    """Distinct text -> pooled feature row, optionally backed by a validated file (see the module docstring)."""

    def __init__(self, backbone, cache_dir=None):
        self.backbone = backbone
        self.cache_dir = cache_dir
        self.index = {}
        self.features = torch.zeros(0, backbone.hidden_size)
        self.sealed = False
        self.version = 0
        self._key = None
        self._on_device = None
        self._disk_read = False
        self._stale = False              # the cache file did not validate: rewrite it at the next encoding
        self.default_device = torch.device("cpu")     # where unseen texts are encoded: the head's device
        self._disk_rows, self._disk_features = {}, None

    def __deepcopy__(self, memo):      # the momentum teacher shares the student's store
        return self

    def __getstate__(self):
        """What a loader worker receives: the row index only, sealed (workers never encode)."""
        return {"index": dict(self.index), "sealed": True, "max_length": self.backbone.max_length}

    def __setstate__(self, state):
        self.backbone, self.cache_dir, self.features, self._on_device = None, None, None, None
        self.index, self.sealed, self.version, self._key, self._disk_read = state["index"], True, 0, None, True
        self.worker_max_length = state["max_length"]

    @property
    def key(self):
        if self._key is None:
            self._key = self.backbone.key()
        return self._key

    @property
    def key_sha256(self):
        return hashlib.sha256(json.dumps(self.key, sort_keys=True).encode()).hexdigest()

    @property
    def path(self):
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"text_features_{self.key_sha256[:16]}.pt")

    def _read_disk(self):
        """Merge the cache file's rows, after validating it; a file that does not validate is rebuilt later."""
        self._disk_read = True
        if not self.path or not os.path.exists(self.path):
            return
        try:
            saved = torch.load(self.path, map_location="cpu", weights_only=True)
            ok = (saved.get("key") == self.key and saved.get("key_sha256") == self.key_sha256
                  and saved["features"].ndim == 2 and saved["features"].shape[1] == self.backbone.hidden_size
                  and saved["features"].shape[0] == len(saved["hashes"])
                  and bool(torch.isfinite(saved["features"]).all()))
        except Exception as error:          # an unreadable file is rebuilt, never trusted
            ok, saved = False, None
            warnings.warn(f"{self.path} cannot be read ({error}); it will be rebuilt.")
        if not ok:
            warnings.warn(f"{self.path} does not match the text feature key; it will be rebuilt.")
            self._stale = True
            return
        self._disk_rows = {h: i for i, h in enumerate(saved["hashes"])}
        self._disk_features = saved["features"].float()

    def _write_disk(self):
        """Write this store's rows, keeping the valid file's other rows (another run's captions or prompts)."""
        if not self.path:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        if not self._stale:
            self._read_disk()                  # rows another process may have added since this store read the file
        by_row = sorted(self.index.items(), key=lambda item: item[1])
        hashes = [text_sha256(t) for t, _ in by_row]
        held = set(hashes)
        others = [(h, i) for h, i in self._disk_rows.items() if h not in held]
        features = self.features.clone()
        if others:
            hashes += [h for h, _ in others]
            features = torch.cat([features, self._disk_features[[i for _, i in others]]])
        state = {"key": self.key, "key_sha256": self.key_sha256, "hashes": hashes, "features": features}
        tmp = self.path + f".tmp{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, self.path)

    def prepare(self, texts, device=None, batch_size=64):
        """Make every text a row: from the cache file when it holds it, else encoded now. Returns how many were
        encoded."""
        if not self._disk_read:
            self._read_disk()
        missing = [t for t in dict.fromkeys(texts) if t not in self.index]
        if not missing:
            return 0
        if self.sealed:
            raise ValueError(f"{len(missing)} texts have no cached feature, and the caption feature store is sealed "
                             "(training never runs the text model inside a step).")
        disk_rows = self._disk_rows
        from_disk = [t for t in missing if text_sha256(t) in disk_rows]
        to_encode = [t for t in missing if text_sha256(t) not in disk_rows]
        new = []
        if from_disk:
            new.append(self._disk_features[[disk_rows[text_sha256(t)] for t in from_disk]])
        if to_encode:
            device = device or self.default_device
            new.append(self.backbone.encode(to_encode, device, batch_size))
        for t in from_disk + to_encode:
            self.index[t] = len(self.index)
        self.features = torch.cat([self.features] + new)
        self.version += 1
        if to_encode or self._stale:
            self._write_disk()
            self._stale = False
        return len(to_encode)

    def rows(self, texts):
        """Row indices [N] of `texts`, encoding unseen texts first unless sealed."""
        if not self.sealed:
            self.prepare(texts)
        try:
            return torch.tensor([self.index[t] for t in texts], dtype=torch.long)
        except KeyError:
            raise ValueError("A text has no cached feature, and the caption feature store is sealed (training never "
                             "runs the text model inside a step).") from None

    def features_on(self, device):
        if self.features is None:
            raise ValueError("This copy of the store holds the row index only (a loader worker).")
        if self._on_device is None or self._on_device[0] != (device, self.version):
            self._on_device = ((device, self.version), self.features.to(device))
        return self._on_device[1]

    def seal(self):
        self.sealed = True

    def release_backbone(self):
        self.backbone.release()


class CachedTokenizer:
    """The tokenizer interface over a feature store: text -> (row ids [N, 1], mask [N, 1])."""

    def __init__(self, store):
        self.store = store

    @property
    def max_length(self):
        if self.store.backbone is None:
            return self.store.worker_max_length
        return self.store.backbone.max_length

    def __call__(self, texts):
        rows = self.store.rows(list(texts))[:, None]
        return rows, torch.ones_like(rows)

    def lengths(self, texts):
        return self.store.backbone.lengths(list(texts))

    def prepare(self, texts, device=None):
        return self.store.prepare(list(texts), device)

    def seal(self):
        self.store.seal()

    def release_backbone(self):
        self.store.release_backbone()


class FrozenTextEncoder(nn.Module):
    """The trainable head on cached frozen features (see the module docstring)."""

    def __init__(self, store, proj_dim=256, head="mlp", head_hidden_dim=1024, input_norm=True):
        super().__init__()
        self.store = store
        hidden = store.backbone.hidden_size
        layers = [nn.LayerNorm(hidden)] if input_norm else []
        if head == "linear":
            layers.append(nn.Linear(hidden, proj_dim))
        else:
            layers += [nn.Linear(hidden, head_hidden_dim), nn.GELU(), nn.Linear(head_hidden_dim, proj_dim)]
        self.head = nn.Sequential(*layers)
        self.register_buffer("feature_key", torch.tensor(list(bytes.fromhex(store.key_sha256)), dtype=torch.uint8))

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self.store.default_device = self.feature_key.device      # unseen texts are encoded where the head is
        return out

    def forward(self, input_ids, attention_mask=None):
        features = self.store.features_on(input_ids.device)[input_ids[:, 0]]
        return F.normalize(self.head(features).float(), dim=-1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = state_dict.get(prefix + "feature_key")
        if key is not None and not torch.equal(key.cpu(), self.feature_key.cpu()):
            raise ValueError("This head was trained on other caption features (text feature key differs): the "
                             "frozen model, pooling, layer, template, max_length or dtype changed.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def build_frozen(cfg, proj_dim, cache_dir=None):
    """(FrozenTextEncoder, CachedTokenizer) sharing one store."""
    store = CaptionFeatureStore(FrozenBackbone(cfg, memo_dir=cache_dir), cache_dir)
    s = store.backbone.settings
    encoder = FrozenTextEncoder(store, proj_dim, s["head"], int(s["head_hidden_dim"]), bool(s["input_norm"]))
    return encoder, CachedTokenizer(store)


def build_frozen_tokenizer(cfg):
    """The tokenizer of a frozen text config, for caption lengths (it encodes nothing until called)."""
    return CachedTokenizer(CaptionFeatureStore(FrozenBackbone(cfg)))


def frozen_feature_key(cfg, cache_dir=None):
    """The feature key of a frozen text config (for the run's fingerprint), None for a trained encoder."""
    if not cfg.get("frozen"):
        return None
    return CaptionFeatureStore(FrozenBackbone(cfg, memo_dir=cache_dir), cache_dir).key
