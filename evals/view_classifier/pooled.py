# evals/view_classifier/pooled.py

"""Low-capacity readouts of the frozen EchoJEPA tokens (arm A-linear): how much view information a pooled vector of
the cached tokens holds, without the attentive probe.

Pooling (`POOLINGS`) of a clip's [T*H*W, D] tokens, in the encoder's order (time, then rows, then columns):
- `mean`: the mean over all tokens [D];
- `mean_std`: the mean and the standard deviation over all tokens [2D];
- `time`: the mean over space at each time step [T*D];
- `space2`, `space4`: the mean over time, then over a 2x2 or 4x4 grid of spatial cells (adaptive average pooling,
  as `torch.nn.functional.adaptive_avg_pool2d`) [4D], [16D].
The cache is read one clip at a time (`pooled_features`), so only the pooled vectors are held in memory.

Readouts (`READOUTS`). Each trains on the inner training set of a fold (`train.fold_sets`, pseudo-labelled clips
included when used) and keeps the setting with the lowest mean cross-entropy on the inner validation set, as the
attentive probe does; ties go to the earlier setting. Features are standardised with the inner training set's mean and
standard deviation.
- `logistic`: multinomial logistic regression, full batch L-BFGS, minimising sum_i w_i CE_i / sum_i w_i
  + |W|^2 / (2 C sum_i w_i) (scikit-learn's C; the bias is not penalised). The clip weights w are the run's class
  weights times the pseudo-label weight (`train.class_weights`, `cv.class_weighting`, `cv.pseudo_weight`).
- `ncm`: nearest class mean. Features are L2-normalised after standardising; a class's mean is the normalised mean of
  its clips; probabilities are softmax(scale * cosine similarity).
- `knn`: k nearest neighbours by cosine similarity; a class's probability is its share of the neighbours' summed
  weights (the same clip weights), smoothed by `alpha`.
A class without training clips gets probability 0 (so it is never predicted).

The fold folders have the attentive probe's layout (`selection.json`, `predictions.csv`, `model.pt`, `done.json`), so
`train.merge_folds` and the report read them unchanged.

    python -m evals.view_classifier.pooled --config CONFIG --arm arm_a_linear --pseudo off --folds 0 1 2 3 4 final
    python -m evals.view_classifier.pooled --config CONFIG --arm arm_a_linear --pseudo off --merge
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import train as vt
from evals.view_classifier.io import check_colour_fix, read_run_inputs, run_dir

POOLINGS = ("mean", "mean_std", "time", "space2", "space4")
READOUTS = ("logistic", "ncm", "knn")


def pool_tokens(tokens, token_grid, modes=POOLINGS):
    """{mode: float32 vector} of one clip's tokens [T*H*W, D] (any float dtype)."""
    t, h, w = (int(v) for v in token_grid)
    x = torch.as_tensor(tokens).float()
    if x.dim() != 2 or x.shape[0] != t * h * w:
        raise ValueError(f"Expected [{t * h * w}, D] tokens, got {tuple(x.shape)}.")
    d = x.shape[1]
    grid = x.reshape(t, h, w, d)
    out = {}
    for mode in modes:
        if mode == "mean":
            v = x.mean(0)
        elif mode == "mean_std":
            v = torch.cat([x.mean(0), x.std(0, unbiased=False)])
        elif mode == "time":
            v = grid.mean(dim=(1, 2)).reshape(-1)
        elif mode in ("space2", "space4"):
            cells = int(mode[-1])
            image = grid.mean(0).permute(2, 0, 1).unsqueeze(0)              # [1, D, H, W]
            v = F.adaptive_avg_pool2d(image, cells)[0].permute(1, 2, 0).reshape(-1)   # cell-major, [cells^2 * D]
        else:
            raise ValueError(f"Unknown pooling {mode!r}; one of {POOLINGS}.")
        out[mode] = v.numpy().astype(np.float32)
    return out


def pooled_features(cache_root, clip_ids, modes=POOLINGS, token_grid=(8, 14, 14), embed_dim=768,
                    expected_sha256=None, progress_every=0):
    """({mode: float32 array [N, F]}, [clip ids]) of `clip_ids` (duplicates dropped, order kept), read one clip at a
    time from a finished token cache."""
    from src.datasets.radar_token_cache import TokenCache
    cache = TokenCache(cache_root, token_grid=token_grid, embed_dim=embed_dim)
    found = cache.meta.get("provenance", {}).get("encoder_sha256")
    if expected_sha256 and found != expected_sha256:
        raise ValueError(f"The token cache was written with encoder sha256 {found}, not {expected_sha256}.")
    ids = list(dict.fromkeys(str(c) for c in clip_ids))
    missing = [c for c in ids if c not in cache]
    if missing:
        raise ValueError(f"{len(missing)} clips are not in the token cache (unreadable, or the cache predates them).")
    out = None
    for i, c in enumerate(ids):
        pooled = pool_tokens(cache.load(c), token_grid, modes)
        if out is None:
            out = {m: np.empty((len(ids), len(v)), dtype=np.float32) for m, v in pooled.items()}
        for m, v in pooled.items():
            out[m][i] = v
        if progress_every and (i + 1) % progress_every == 0:
            print(f"pooled {i + 1} of {len(ids)} clips", flush=True)
    return out, ids


def save_features(path, feats, ids):
    np.savez(path, clip_ids=np.asarray(ids, dtype=object), **feats)


def load_features(path):
    with np.load(path, allow_pickle=True) as z:
        ids = [str(c) for c in z["clip_ids"]]
        feats = {k: z[k] for k in z.files if k != "clip_ids"}
    return feats, ids


# ---------------------------------------------------------------- readouts

def standardiser(x):
    mean = x.mean(0)
    std = x.std(0)
    return mean, np.where(std > 1e-6, std, 1.0)


def _normalise(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _expand(probs_present, present, n_classes):
    out = np.zeros((probs_present.shape[0], n_classes))
    out[:, present] = probs_present
    return out


def fit_logistic(x, y, w, n_classes, C, max_iter=500, threads=None):
    """Multinomial logistic regression on standardised features; returns the model dict (`predict` reads it)."""
    if threads:
        torch.set_num_threads(int(threads))
    y = np.asarray(y, dtype=int)
    present = np.flatnonzero(np.bincount(y, minlength=n_classes) > 0)
    remap = {c: i for i, c in enumerate(present)}
    mean, std = standardiser(x)
    X = torch.as_tensor((x - mean) / std, dtype=torch.float64)
    Y = torch.as_tensor([remap[v] for v in y])
    W8 = torch.as_tensor(np.asarray(w, dtype=float))
    total = float(W8.sum())
    weight = torch.zeros(X.shape[1], len(present), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(len(present), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=int(max_iter), history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ce = F.cross_entropy(X @ weight + bias, Y, reduction="none")
        loss = (ce * W8).sum() / total + 0.5 * (weight ** 2).sum() / (float(C) * total)
        loss.backward()
        return loss

    opt.step(closure)
    return {"readout": "logistic", "mean": mean, "std": std, "present": present, "n_classes": n_classes,
            "weight": weight.detach().numpy(), "bias": bias.detach().numpy(), "C": float(C)}


def fit_ncm(x, y, w, n_classes, scale):
    y = np.asarray(y, dtype=int)
    present = np.flatnonzero(np.bincount(y, minlength=n_classes) > 0)
    mean, std = standardiser(x)
    z = _normalise((x - mean) / std)
    centres = _normalise(np.stack([z[y == c].mean(0) for c in present]))
    return {"readout": "ncm", "mean": mean, "std": std, "present": present, "n_classes": n_classes,
            "centres": centres, "scale": float(scale)}


def fit_knn(x, y, w, n_classes, k, alpha=0.01):
    mean, std = standardiser(x)
    return {"readout": "knn", "mean": mean, "std": std, "n_classes": n_classes,
            "bank": _normalise((x - mean) / std).astype(np.float32), "y": np.asarray(y, dtype=int),
            "w": np.asarray(w, dtype=float), "k": int(k), "alpha": float(alpha),
            "present": np.flatnonzero(np.bincount(np.asarray(y, dtype=int), minlength=n_classes) > 0)}


def predict(model, x):
    """Probabilities [N, n_classes] (rows sum to 1; classes absent from training get 0)."""
    z = (x - model["mean"]) / model["std"]
    n_classes, present = model["n_classes"], model["present"]
    if model["readout"] == "logistic":
        logits = z @ model["weight"] + model["bias"]
        return _expand(torch.softmax(torch.as_tensor(logits), -1).numpy(), present, n_classes)
    z = _normalise(z)
    if model["readout"] == "ncm":
        return _expand(torch.softmax(torch.as_tensor(model["scale"] * z @ model["centres"].T), -1).numpy(),
                       present, n_classes)
    if model["readout"] == "knn":
        sim = z.astype(np.float32) @ model["bank"].T
        k = min(model["k"], sim.shape[1])
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        votes = np.zeros((len(z), n_classes))
        for j in range(k):
            np.add.at(votes, (np.arange(len(z)), model["y"][top[:, j]]), model["w"][top[:, j]])
        votes[:, present] += model["alpha"] * votes.sum(1, keepdims=True) / len(present)
        return votes / np.maximum(votes.sum(1, keepdims=True), 1e-12)
    raise ValueError(f"Unknown readout {model['readout']!r}.")


def fit(readout, x, y, w, n_classes, hp, threads=None):
    if readout == "logistic":
        return fit_logistic(x, y, w, n_classes, hp["C"], int(hp.get("max_iter", 500)), threads)
    if readout == "ncm":
        return fit_ncm(x, y, w, n_classes, hp["scale"])
    if readout == "knn":
        return fit_knn(x, y, w, n_classes, hp["k"], float(hp.get("alpha", 0.01)))
    raise ValueError(f"Unknown readout {readout!r}; one of {READOUTS}.")


def grid_from_config(a):
    """Every combination of `grid` (lists; `pooling` among them), merged over `fixed`."""
    fixed, grid = dict(a.get("fixed", {})), dict(a.get("grid", {}))
    keys = sorted(grid)
    settings = [{**fixed, **dict(zip(keys, v))} for v in itertools.product(*(grid[k] for k in keys))] if keys \
        else [fixed]
    for s in settings:
        if s.get("pooling", "mean") not in POOLINGS:
            raise ValueError(f"Unknown pooling {s.get('pooling')!r}; one of {POOLINGS}.")
    return settings


def run_fold(name, readout, feats, row, labels, pseudo, fold, grid, cv, classes, out_dir, threads=None):
    """One fold: fit every setting on the inner training set, keep the lowest inner-validation cross-entropy, predict
    the outer fold; writes `<out_dir>/fold_<fold>/` as `train.run_fold` does."""
    folder = os.path.join(out_dir, f"fold_{fold}")
    if os.path.exists(os.path.join(folder, "done.json")):
        print(f"{name} fold {fold}: already finished", flush=True)
        return
    os.makedirs(folder, exist_ok=True)
    seed = int(cv.get("seed", 0))
    train, val, test = vt.fold_sets(labels, pseudo, fold, float(cv.get("inner_fraction", 0.15)), seed)
    missing = [c for c in pd.concat([train, val, test]).clip_id if c not in row]
    if missing:
        raise ValueError(f"{len(missing)} clips of fold {fold} have no features.")
    weights = vt.clip_weights(train, len(classes), cv)
    rows = lambda t: np.asarray([row[c] for c in t.clip_id], dtype=int)
    started, results, best = time.time(), [], None
    for h, hp in enumerate(grid):
        t0 = time.time()
        x = feats[hp.get("pooling", "mean")]
        model = fit(readout, x[rows(train)], train.y.to_numpy(), weights, len(classes), hp, threads)
        probs = predict(model, x[rows(val)])
        loss = vt.mean_nll(probs, val.y.to_numpy())
        acc = float((probs.argmax(1) == val.y.to_numpy()).mean())
        results.append({"setting": h, "hp": hp, "best_epoch": 0, "best_val_loss": loss, "val_accuracy": acc,
                        "history": [], "seconds": time.time() - t0})
        if best is None or loss < best[0]:
            best = (loss, h, model)
    chosen, model = best[1], best[2]
    print(f"{name} fold {fold}: setting {chosen} {grid[chosen]} val loss {best[0]:.4f}", flush=True)
    torch.save({"model": {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in model.items()},
                "hp": grid[chosen], "classes": list(classes)}, os.path.join(folder, "model.pt"))
    if len(test):
        probs = predict(model, feats[grid[chosen].get("pooling", "mean")][rows(test)])
        out = pd.DataFrame({"clip_id": test.clip_id.to_numpy(), "fold": int(fold), "y": test.y.to_numpy()})
        out = pd.concat([out, pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])], axis=1)
        out.to_csv(os.path.join(folder, "predictions.csv"), index=False)
    record = {"arm": name, "fold": fold, "readout": readout, "chosen_setting": chosen, "chosen_hp": grid[chosen],
              "n_train_human": int((train.source == "human").sum()),
              "n_train_pseudo": int((train.source == "pseudo").sum()), "n_val": len(val), "n_test": len(test),
              "seconds": time.time() - started, "settings": results, "device": "cpu",
              "weighting": vt.weighting_record(cv)}
    with open(os.path.join(folder, "selection.json"), "w") as f:
        json.dump(record, f, indent=1)
    with open(os.path.join(folder, "done.json"), "w") as f:
        json.dump({"seconds": record["seconds"]}, f)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", default="arm_a_linear", help="the config section and the output folder")
    parser.add_argument("--pseudo", choices=("on", "off"), required=True)
    parser.add_argument("--folds", nargs="*", default=[], help="fold numbers and/or `final`")
    parser.add_argument("--merge", action="store_true", help="write oof_predictions.csv from the finished folds")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.arm in ("arm_a", "arm_b") or not args.arm.startswith("arm_") or "readout" not in cfg.get(args.arm, {}):
        raise ValueError(f"{args.arm!r} is not a pooled-readout arm: its config section needs `readout`, and the "
                         "first runs' folders (arm_a, arm_b) are never written.")
    torch.set_num_threads(args.threads)
    labels, pseudo = read_run_inputs(cfg, args.pseudo == "on")
    out = run_dir(cfg, args.arm, args.pseudo)
    classes, _ = vc.from_config(cfg)
    if args.merge:
        oof = vt.merge_folds(out, labels, int(cfg["cv"]["k"]), classes)
        print(f"{args.arm} pseudo {args.pseudo}: {len(oof)} out-of-fold predictions, accuracy "
              f"{float((oof.y == np.asarray(oof[[f'p_{c}' for c in classes]]).argmax(1)).mean()):.3f}")
        return
    a = cfg[args.arm]
    with open(os.path.join(cfg["paths"]["token_cache"], "meta.json")) as f:
        check_colour_fix(cfg, json.load(f))
    readout = a.get("readout", "logistic")
    grid = grid_from_config(a)
    modes = sorted({s.get("pooling", "mean") for s in grid})
    ids = list(labels.clip_id) + ([] if pseudo is None else list(pseudo.clip_id))
    os.makedirs(out, exist_ok=True)
    feats, kept = pooled_features(cfg["paths"]["token_cache"], ids, modes, tuple(a.get("token_grid", (8, 14, 14))),
                                  int(a.get("embed_dim", 768)), a.get("encoder_sha256"), progress_every=500)
    row = {c: i for i, c in enumerate(kept)}
    cv = vt.arm_cv(cfg["cv"], a.get("cv"))
    for fold in args.folds:
        run_fold(args.arm, readout, feats, row, labels, pseudo, fold if fold == vt.FINAL else int(fold), grid,
                 cv, classes, out, args.threads)


if __name__ == "__main__":
    main()
