# evals/view_classifier/train.py

"""The cross-validation loop both arms share.

For an outer fold f: the labelled clips of the other folds are split into an inner training set and an inner
validation set (`folds.inner_split`); pseudo-labelled clips, when used, join the inner training set only. Every
hyper-parameter setting of the grid trains on the inner training set; after each epoch the mean cross-entropy on the
inner validation set is measured, and the setting and epoch with the lowest one are kept (ties: the earlier setting,
then the earlier epoch). That model predicts fold f, which it has never seen. Fold `final` does the same on every
labelled clip and predicts nothing: the model to deploy.

An arm (`Arm`) supplies the data (a tensor in memory and the row of each clip), a model builder, its parameter groups,
and `prepare`, which turns a batch on the device into model input (with augmentation when training). Losses are
weighted per clip: pseudo-labels by `pseudo_weight`, and classes by `class_weighting` (`none`, or `inv_sqrt`: one over
the square root of the class's count in the inner training set, scaled to mean one).

A finished fold (its `done.json`) is skipped on a rerun. Nothing written names a patient: rows are clip ids, which the
caller keeps outside the repository.
"""

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from evals.view_classifier import folds as vf

FINAL = "final"


@dataclass
class Arm:
    name: str
    data: torch.Tensor                       # [N, ...] inputs in memory (CPU)
    row: dict                                # clip_id -> row of `data`
    build: Callable                          # hp -> nn.Module
    param_groups: Callable                   # (model, hp) -> optimizer parameter groups
    prepare: Callable                        # (batch on device, training, torch.Generator) -> model input


def class_weights(y, n_classes, mode):
    if mode == "none":
        return np.ones(n_classes)
    if mode != "inv_sqrt":
        raise ValueError(f"Unknown class_weighting {mode!r}.")
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    present = counts > 0
    return w / w[present].mean()


def learning_rate(base, step, total, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _batches(n, batch_size, generator=None):
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


@torch.no_grad()
def predict(arm, model, rows, device, batch_size=32):
    """Softmax probabilities [len(rows), C] of the clips at `rows` of arm.data."""
    model.eval()
    rows = torch.tensor(np.asarray(rows, dtype=np.int64))
    out = []
    for idx in _batches(len(rows), batch_size):
        x = arm.prepare(arm.data[rows[idx]].to(device, non_blocking=True), False, None)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        out.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


def mean_nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), np.asarray(y, dtype=int)], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit(arm, train_rows, train_y, train_w, val_rows, val_y, hp, device, seed=0):
    """Train one setting; returns (state of the best epoch on CPU, best epoch, history)."""
    if not len(val_rows):
        raise ValueError("The inner validation set is empty: too few labelled clips to select a setting.")
    torch.manual_seed(seed)
    model = arm.build(hp).to(device)
    optimizer = torch.optim.AdamW(arm.param_groups(model, hp), betas=(0.9, 0.999))
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]
    epochs, batch_size = int(hp["epochs"]), int(hp["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_rows) / batch_size))
    total, warmup = epochs * steps_per_epoch, int(float(hp.get("warmup_epochs", 1)) * steps_per_epoch)
    generator = torch.Generator().manual_seed(seed)
    rows = torch.tensor(np.asarray(train_rows, dtype=np.int64))
    y_all = torch.tensor(np.asarray(train_y, dtype=np.int64))
    w_all = torch.tensor(np.asarray(train_w, dtype=np.float32))
    best, best_epoch, best_loss, history, step = None, -1, float("inf"), [], 0
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for idx in _batches(len(rows), batch_size, generator):
            for g in optimizer.param_groups:
                g["lr"] = learning_rate(g["base_lr"], step, total, warmup)
            x = arm.prepare(arm.data[rows[idx]].to(device, non_blocking=True), True, generator)
            y, w = y_all[idx].to(device), w_all[idx].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = (F.cross_entropy(logits.float(), y, reduction="none") * w).sum() / w.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if hp.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp["grad_clip"]))
            optimizer.step()
            running, seen, step = running + float(loss.detach()) * len(idx), seen + len(idx), step + 1
        probs = predict(arm, model, val_rows, device, batch_size)
        val_loss = mean_nll(probs, val_y)
        val_acc = float((probs.argmax(axis=1) == np.asarray(val_y)).mean())
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, seen), "val_loss": val_loss,
                        "val_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    del model, optimizer
    return best, best_epoch, history


def fold_sets(labels, pseudo, fold, inner_fraction, seed):
    """(inner train, inner validation, outer test) tables of `labels` (clip_id, y, fold, patient_id, label) for
    `fold` (an int or FINAL); pseudo-labelled rows (clip_id, y) join the inner training set."""
    outer = labels if fold == FINAL else labels[labels.fold != int(fold)]
    test = labels.head(0) if fold == FINAL else labels[labels.fold == int(fold)]
    held = vf.inner_split(outer, fraction=inner_fraction, seed=seed)
    train = outer[~held].assign(source="human")
    if pseudo is not None and len(pseudo):
        train = pd.concat([train, pseudo.assign(source="pseudo")], ignore_index=True)
    return train, outer[held], test


def run_fold(arm, labels, pseudo, fold, grid, cfg, classes, out_dir, device):
    """Train, select and (for an outer fold) predict one fold; writes `<out_dir>/fold_<fold>/`."""
    folder = os.path.join(out_dir, f"fold_{fold}")
    if os.path.exists(os.path.join(folder, "done.json")):
        print(f"{arm.name} fold {fold}: already finished", flush=True)
        return
    os.makedirs(folder, exist_ok=True)
    seed = int(cfg.get("seed", 0))
    train, val, test = fold_sets(labels, pseudo, fold, float(cfg.get("inner_fraction", 0.15)), seed)
    missing = [c for c in pd.concat([train, val, test]).clip_id if c not in arm.row]
    if missing:
        raise ValueError(f"{len(missing)} clips of fold {fold} have no input (not cached or unreadable).")
    cw = class_weights(train.y, len(classes), cfg.get("class_weighting", "none"))
    weights = cw[train.y.to_numpy()] * np.where(train.source == "pseudo", float(cfg.get("pseudo_weight", 1.0)), 1.0)
    rows = lambda t: [arm.row[c] for c in t.clip_id]
    started, results, best = time.time(), [], None
    for h, hp in enumerate(grid):
        t0 = time.time()
        state, epoch, history = fit(arm, rows(train), train.y.to_numpy(), weights, rows(val), val.y.to_numpy(), hp,
                                    device, seed=seed * 1000 + h)
        loss = min(r["val_loss"] for r in history)
        results.append({"setting": h, "hp": hp, "best_epoch": epoch, "best_val_loss": loss, "history": history,
                        "seconds": time.time() - t0})
        print(f"{arm.name} fold {fold} setting {h}: val loss {loss:.4f} at epoch {epoch}", flush=True)
        if best is None or loss < best[0]:
            best = (loss, h, state)
    chosen = best[1]
    torch.save({"state_dict": best[2], "hp": grid[chosen], "classes": list(classes)},
               os.path.join(folder, "model.pt"))
    if len(test):
        model = arm.build(grid[chosen]).to(device)
        model.load_state_dict(best[2])
        probs = predict(arm, model, rows(test), device, int(grid[chosen]["batch_size"]))
        out = pd.DataFrame({"clip_id": test.clip_id.to_numpy(), "fold": int(fold), "y": test.y.to_numpy()})
        out = pd.concat([out, pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])], axis=1)
        out.to_csv(os.path.join(folder, "predictions.csv"), index=False)
    record = {"arm": arm.name, "fold": fold, "chosen_setting": chosen, "chosen_hp": grid[chosen],
              "n_train_human": int((train.source == "human").sum()), "n_train_pseudo": int((train.source == "pseudo").sum()),
              "n_val": len(val), "n_test": len(test), "seconds": time.time() - started, "settings": results,
              "device": str(device)}
    with open(os.path.join(folder, "selection.json"), "w") as f:
        json.dump(record, f, indent=1)
    with open(os.path.join(folder, "done.json"), "w") as f:
        json.dump({"seconds": record["seconds"]}, f)


def merge_folds(out_dir, labels, k, classes):
    """`<out_dir>/oof_predictions.csv`: every labelled clip predicted once, by the fold that held it out."""
    parts = []
    for f in range(k):
        path = os.path.join(out_dir, f"fold_{f}", "predictions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"fold {f} has no predictions: train it first.")
        parts.append(pd.read_csv(path, dtype={"clip_id": str}))
    oof = pd.concat(parts, ignore_index=True)
    if oof.clip_id.duplicated().any() or set(oof.clip_id) != set(labels.clip_id):
        raise ValueError("The folds' predictions do not cover each labelled clip exactly once.")
    probs = oof[[f"p_{c}" for c in classes]].to_numpy()
    oof["predicted"] = np.asarray(classes)[probs.argmax(axis=1)]
    oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    return oof


def grid_from_config(cfg):
    """The settings: every combination of the `grid` lists, each merged over the `fixed` values."""
    import itertools
    fixed, grid = dict(cfg.get("fixed", {})), dict(cfg.get("grid", {}))
    keys = sorted(grid)
    return [{**fixed, **dict(zip(keys, values))} for values in itertools.product(*(grid[k] for k in keys))] \
        if keys else [fixed]
