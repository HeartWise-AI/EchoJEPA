"""RADAR contrastive training skeleton.

    python -m app.radar.train --fname configs/radar/skeleton-vitb16.yaml [--device cuda:0]

Single process. Starts from the encoder checkpoint in `visual.checkpoint` (published EchoJEPA by default), trains the
pooler, projections, text encoder (unless frozen) and temperature with AdamW, and writes to `meta.output_dir`
(outside the repository):

  latest.pt         atomic checkpoint of the trained parameters, optimizer, step and RNG state;
  provenance.json   config hash, git state, encoder-checkpoint, records and patient-split SHA-256, ontology, text
                    encoder, caption sources, split uses and development flags;
  log.jsonl         per-step loss and temperature (no identifiers).

`data.patient_splits` is the archive's patient_splits.csv the records were built from: every record must carry its
SHA-256 (`split_source`), and validation needs it. A rerun with `meta.resume: true` continues from latest.pt only if
the config, encoder checkpoint, text-model files, records and patient split are byte-identical to the run that wrote
it. This skeleton is not a validated training recipe.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile

import numpy as np
import torch
import yaml

from app.radar.data import RadarClipDataset, collate, select_records
from app.radar.loss import ContrastiveLoss
from app.radar.model import build_model
from app.radar.ontology import DEFAULT_ONTOLOGY_PATH, REPO_ROOT, load_ontology
from app.radar.records import ensure_outside_repo, read_records, utc_now
from src.datasets.video_dataset import DEFAULT_LOADER_TIMEOUT, dataloader_timeout
from evals.video_classification_frozen.provenance import git_state, sha256

SECTIONS = ("meta", "data", "visual", "text", "projection", "loss", "optimization")
PLACEHOLDER = "/your_"


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return check_config(cfg)


def check_config(cfg: dict) -> dict:
    missing = [s for s in SECTIONS if s not in cfg]
    unknown = sorted(set(cfg) - set(SECTIONS))
    if missing or unknown:
        raise ValueError(f"RADAR config sections: missing {missing}, unknown {unknown}.")
    placeholders = []

    def walk(obj, where):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{where}.{k}" if where else k)
        elif isinstance(obj, str) and obj.startswith(PLACEHOLDER):
            placeholders.append(where)

    walk(cfg, "")
    if placeholders:
        raise ValueError(f"Replace the placeholder paths before training: {placeholders}.")
    return cfg


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def lr_at(step: int, total: int, opt: dict) -> float:
    """Linear warmup to `lr`, then cosine decay to `final_lr` at `total` steps. Stateless, so resume is exact."""
    lr, final, warmup = opt["lr"], opt.get("final_lr", 0.0), opt.get("warmup_steps", 0)
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + 0.5 * (lr - final) * (1 + math.cos(math.pi * progress))


def _atomic_write(path: str, write) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_json(path: str, obj: dict) -> None:
    _atomic_write(path, lambda f: f.write((json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode()))


def trainable_state(model, prefixes: tuple) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith(prefixes)}


def _optimizer(model, loss_fn, opt: dict) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for module in (model, loss_fn):
        for name, p in module.named_parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": opt.get("weight_decay", 0.0)}, {"params": no_decay, "weight_decay": 0.0}],
        lr=opt["lr"], betas=tuple(opt.get("betas", (0.9, 0.999))),
    )


def _loader(dataset, batch_size: int, seed: int, epoch: int, num_workers: int, skip: int = 0, timeout: float = 0):
    """The epoch's full batches in RandomSampler order, without the first `skip` (already trained on before an
    interruption). Draws nothing from the global RNGs, so dropout after a resume sees the same random stream.
    `timeout` (seconds, `data.loader_timeout`) turns a worker stalled on a read into an error; PyTorch enforces it
    only with worker processes."""
    dataset.set_epoch(epoch)
    key = seed * 1_000_003 + epoch
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(key))  # RandomSampler's draw
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=order[skip * batch_size:].tolist(), drop_last=True,
        num_workers=num_workers, collate_fn=collate, persistent_workers=False,
        generator=torch.Generator().manual_seed(key + 2**62),  # worker base seeds
        timeout=dataloader_timeout(timeout, num_workers),
    )


def _rng_state(device: torch.device) -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _set_rng_state(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _step_loss(model, loss_fn, batch, device):
    tokens = {k: v.to(device) for k, v in model.text.tokenize(batch["caption"]).items()}
    video, text = model(batch["video"].to(device), batch["clip_indices"].to(device), tokens)
    mask = loss_fn.mask(batch["caption_key"], batch["study_key"], batch["concept_key"])
    return loss_fn(video, text, mask)


def run(cfg: dict, device: str = "cpu", stop_after_steps: int | None = None) -> dict:
    """Train (or resume) per `cfg`; returns a summary. `stop_after_steps` simulates an interruption (tests)."""
    cfg = check_config(copy.deepcopy(cfg))
    device = torch.device(device)
    meta, data, opt = cfg["meta"], cfg["data"], cfg["optimization"]
    out = meta["output_dir"]
    ensure_outside_repo(out, "meta.output_dir")
    os.makedirs(out, exist_ok=True)
    latest = os.path.join(out, "latest.pt")
    if os.path.exists(latest) and not meta.get("resume", True):
        raise FileExistsError("meta.output_dir already holds latest.pt and meta.resume is false; use a new folder.")

    seed = int(meta.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ontology_path = data.get("ontology") or DEFAULT_ONTOLOGY_PATH
    if not os.path.isabs(ontology_path):
        ontology_path = os.path.join(REPO_ROOT, ontology_path)
    ontology = load_ontology(ontology_path)
    records = read_records(data["records"], ontology)
    dev = bool(meta.get("dev_allow_unreviewed_candidates", False))
    split_source = sha256(data["patient_splits"]) if data.get("patient_splits") else None
    if data.get("validation", False) and split_source is None:
        raise ValueError("data.validation needs data.patient_splits, the archive patient split the records were built "
                         "from, so validation patients are the shared split's.")
    train_items, train_use = select_records(records, split="train", purpose="training",
                                            allow_unreviewed_candidates=dev, split_source=split_source)
    split_uses = [train_use]
    val_items = []
    if data.get("validation", False):
        val_items, val_use = select_records(records, split="val", purpose="validation_loss",
                                            allow_unreviewed_candidates=dev, split_source=split_source)
        split_uses.append(val_use)
    batch_size = int(data["batch_size"])
    # Training and validation load in `num_workers` worker processes, so a stalled read ends the run (resumable from
    # latest.pt) after `loader_timeout` seconds instead of hanging it; without workers there is no timeout.
    num_workers = int(data.get("num_workers", 0))
    loader_timeout = dataloader_timeout(data.get("loader_timeout", DEFAULT_LOADER_TIMEOUT), num_workers)
    if len(train_items) < max(2, batch_size):
        raise ValueError(f"{len(train_items)} captioned training clips cannot fill one batch of {batch_size} "
                         "(approved captions only unless meta.dev_allow_unreviewed_candidates).")

    model, visual_meta = build_model(cfg, data["resolution"], data["frames_per_clip"])
    model.to(device)
    loss_fn = ContrastiveLoss(**cfg["loss"]).to(device)
    optimizer = _optimizer(model, loss_fn, opt)
    frozen = model.frozen_prefixes()

    def dataset(items):
        return RadarClipDataset(items, frames_per_clip=data["frames_per_clip"], resolution=data["resolution"],
                                fps=data.get("fps"), frame_step=data.get("frame_step"),
                                temporal_padding=data.get("temporal_padding"),
                                random_clip_sampling=data.get("random_clip_sampling", True), seed=seed,
                                video_dataset_kwargs=data.get("video_dataset_kwargs"))

    train_set = dataset(train_items)
    val_set = dataset(val_items) if len(val_items) >= 2 else None
    steps_per_epoch = len(train_set) // batch_size
    epochs = int(opt["epochs"])
    total_steps = steps_per_epoch * epochs

    fingerprint = {
        "config_sha256": config_sha256(cfg),
        "visual_checkpoint_sha256": sha256(cfg["visual"]["checkpoint"]) if cfg["visual"].get("checkpoint") else None,
        "records_sha256": sha256(data["records"]),
        "patient_splits_sha256": split_source,
        # The text model's files (weights, tokenizer): frozen text weights are not in latest.pt.
        "text_files_sha256": model.text.files_sha256(),
        "ontology": {"tag": ontology.tag, "sha256": ontology.sha256},
    }
    step = 0
    resumed_from = rng = None
    if os.path.exists(latest):
        state = torch.load(latest, map_location="cpu", weights_only=False)  # our own checkpoint, written below
        diffs = [k for k in fingerprint if state["fingerprint"].get(k) != fingerprint[k]]
        if diffs:
            raise ValueError(f"latest.pt was written with another {', '.join(diffs)}; resume only the same run, or "
                             "start a new meta.output_dir.")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if unexpected or any(not k.startswith(frozen) for k in missing):
            raise RuntimeError("latest.pt does not match this model's trainable parameters.")
        loss_fn.load_state_dict(state["loss"])
        optimizer.load_state_dict(state["optimizer"])
        step = resumed_from = int(state["step"])
        rng = state["rng"]  # restored just before training, after everything that may draw from it

    git_commit, git_dirty = git_state()
    provenance = {
        **fingerprint,
        "config": cfg,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "visual": {**visual_meta, "model_name": cfg["visual"]["model_name"], "frozen": model.video.frozen,
                   "pooling": model.video.pooling},
        "text_encoder": model.text.identity(),
        "loader": {"num_workers": num_workers, "timeout_s": loader_timeout},
        "dev_flags": {"allow_unreviewed_candidates": dev,
                      "visual_random_init": bool(visual_meta.get("random_init"))},
        "split_uses": split_uses,
        "caption_sources": {"approved": train_use["approved_captions"],
                            "unvalidated_candidate": train_use["unvalidated_candidate_captions"]},
        "torch": torch.__version__,
        "started_utc": utc_now(),
        "resumed_from_step": resumed_from,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    _write_json(os.path.join(out, "provenance.json"), provenance)

    def save():
        state = {
            "model": trainable_state(model, frozen), "loss": loss_fn.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "fingerprint": fingerprint,
            "rng": _rng_state(device), "saved_utc": utc_now(),
        }
        _atomic_write(latest, lambda f: torch.save(state, f))
        provenance.update(last_checkpoint_step=step, last_checkpoint_utc=state["saved_utc"])
        _write_json(os.path.join(out, "provenance.json"), provenance)

    def validate(epoch):
        model.eval()
        with torch.no_grad():
            losses = [float(_step_loss(model, loss_fn, b, device)[0])
                      for b in _loader(val_set, min(batch_size, len(val_set)), seed, epoch, num_workers,
                                       timeout=loader_timeout)]
        log.write(json.dumps({"step": step, "epoch": epoch, "val_loss": float(np.mean(losses))}) + "\n")
        log.flush()

    log = open(os.path.join(out, "log.jsonl"), "a")
    every = int(opt.get("checkpoint_every", max(1, steps_per_epoch)))
    loss_value = None
    if rng is not None:
        _set_rng_state(rng, device)
    try:
        while step < total_steps:
            epoch, skip = divmod(step, steps_per_epoch)
            model.train()
            for batch in _loader(train_set, batch_size, seed, epoch, num_workers, skip, loader_timeout):
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total_steps, opt)
                loss, stats = _step_loss(model, loss_fn, batch, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite RADAR loss at step {step}.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if opt.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"]
                                                   + optimizer.param_groups[1]["params"], opt["grad_clip"])
                optimizer.step()
                step += 1
                loss_value = float(loss.detach())
                log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_value,
                                      "lr": optimizer.param_groups[0]["lr"], **stats}) + "\n")
                log.flush()
                if val_set is not None and step % steps_per_epoch == 0:
                    validate(epoch)  # before the epoch-end checkpoint, so a resume from it does not skip it
                if step % every == 0 or step == total_steps or step == stop_after_steps:
                    save()
                if step == stop_after_steps:
                    return {"step": step, "loss": loss_value, "stopped": True}
    finally:
        log.close()
    return {"step": step, "loss": loss_value, "stopped": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RADAR contrastive training skeleton.")
    parser.add_argument("--fname", required=True, help="RADAR config YAML")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    summary = run(load_config(args.fname), device=args.device)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
