# evals/radar_frozen/linear_probe.py

"""Linear probes on RADAR's pooled study embeddings: is a label linearly present in the embedding?

Zero-shot evaluation (`zero_shot_eval.py`) reads a label out of a structure embedding through prompts. A weak
zero-shot score can mean that the label is not in the embedding, or that it is there and the prompt read-out misses
it. This script separates the two: it fits linear models on the embeddings of training studies and scores them on
held-out studies, beside a baseline that needs no RADAR training at all.

Embeddings (`--stage embed`). Every study of the training and of the evaluation clip table is embedded from
`num_bags` evaluation bags (eval_seed, bag, study), built as `zero_shot_eval.py` builds them: the training config's
number of clips, selection, acquisition classes and routing registry, `training=False`, so the evaluation studies
read exactly the clips their zero-shot predictions read. Training studies get evaluation bags too (their training
bags change every epoch). Per bag and trained structure the pooler gives the feature before the output projection
(`pre`, the pooler's LayerNorm output) and the L2-normalised embedding after it (`post`); a study's value is the mean
over the bags that support the structure (a usable clip, `has_clip`), and for a study with no supporting bag the
null-token output, which depends on no token (it marks the structure as absent). Several checkpoints that share the
data settings are embedded in one pass over the token cache. Baselines from the same bags, no RADAR weight involved:
`tokens_routed`, per structure the mean of the frozen tokens of the bag's clips that the routing registry lets the
structure read (zero without one), and `tokens_study`, the mean of the frozen tokens of all the bag's clips.

Probes (`--stage probe`), per run and target of the label table (`targets_info.json` gives each target's task;
categorical targets are skipped):

- regression: ridge regression on standardised inputs, closed form (an eigendecomposition gives every penalty at
  once); metrics MAE, RMSE, R2, Pearson r and bias on the evaluation studies, the standardised MAE (MAE / standard
  deviation of the training labels) and the constant guess (the training median, the MAE-optimal constant);
- binary: L2-penalised logistic regression on standardised inputs (L-BFGS, every fold and penalty solved together);
  AUROC and AUPRC.

The penalty is chosen by `folds`-fold cross-validation on the training studies alone (lowest mean MAE, highest mean
AUROC), then the model is refitted on all training studies and scored once on the evaluation studies. The
standardisation statistics come from all training studies of the target (no evaluation study). Inputs compared:
`routed_post`, `routed_pre` (the embedding of the structure that `ROUTED_STRUCTURE` assigns the target),
`concat_post`, `concat_pre` (every trained structure), and the run-independent `tokens_routed`, `tokens_study`,
`tokens_concat`. For a target with a routed structure, every input is scored on the same studies: those with a label
and a supporting bag for that structure, in each split; other targets use every labelled study.

Refusals: an output folder inside the repository; a clip table whose `split` column names another split than the
one given; an evaluation study that is also a training study; a checkpoint whose fingerprint names another training
clip table than `--train_clips` or another routing registry; runs that do not share the data settings.

Outputs in `out_dir` (outside the repository, files hold study identifiers): `embeddings/<split>/` (`study_ids.txt`,
`<run>.pre.npy` [N, S, D], `<run>.post.npy` [N, S, P], `<run>.support.npy` [N, S] supporting bags, `tokens_routed.npy`,
`tokens_study.npy`, `tokens_support.npy`, `meta.json`), and `probe/<run>/metrics.json` (per input and target: rows,
chosen penalty and its place in the grid, cross-validation curve, metrics) and `probe/<run>/predictions.csv`, one
folder per run (`tokens` for the baselines), so runs can be probed in parallel (`--probe_runs`).

Usage:
    python -m evals.radar_frozen.linear_probe --run a1_seed0 <config> <run>/latest.pt [--run ...] \\
        --train_clips <train clip table> --train_cache_dir <cache> --val_clips <val clip table> \\
        --val_cache_dir <cache> --labels <study_targets.csv> --targets_info <targets_info.json> --out_dir <folder>
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from evals.radar_frozen.train import AMP_MODES, REPO, _device, _to, build_model
from evals.video_classification_frozen.metrics import auroc, average_precision, regression
from src.datasets.radar_study_dataset import RadarStudyDataset, file_sha256, read_table
from src.datasets.radar_token_cache import TokenCache
from src.models.radar.routing import DEFAULT_ROUTING, load_routing

BAG_KEYS = ("tokens", "clip_valid", "view_ids", "acquisition_ids", "eligible")
CAPTION_COLUMNS = ("study_id", "structure", "caption", "caption_id", "normality")
DATA_KEYS = ("num_clips", "selection", "acquisition_classes", "routing")
RUN_INPUTS = ("routed_post", "routed_pre", "concat_post", "concat_pre")
TOKEN_INPUTS = ("tokens_routed", "tokens_study", "tokens_concat")
TOKEN_RUN = "tokens"
# the structure whose embedding a target is read from (the zero-shot prompt files' choice where they have one)
ROUTED_STRUCTURE = {
    "lvef": "lv", "lvidd": "lv", "lvids": "lv", "ivsd": "lv", "lvpwd": "lv", "e_over_e_prime": "lv",
    "reduced_ef_40": "lv", "lvef_le_45": "lv", "lvh_any": "lv", "lvh_modsev": "lv", "lvot_obstruction": "lv",
    "rvsp": "rv", "rv_basal_diameter": "rv", "tapse": "rv", "rv_s_prime": "rv", "rv_dysfunction": "rv",
    "rv_dilated": "rv", "la_dimension": "la", "lavi": "la", "lvot_diameter": "aortic", "av_peak_velocity": "aortic",
    "av_mean_gradient": "aortic", "av_area": "aortic", "as_present": "aortic", "as_severe": "aortic",
    "ar_modsev": "aortic", "mr_modsev": "mitral", "mv_stenosis": "mitral", "tr_peak_velocity": "tricuspid",
    "tr_modsev": "tricuspid", "pr_modsev": "pulmonic", "aortic_root": "aorta", "ivc_diameter": "ivc",
    "rap_elevated": "ivc", "pericardial_effusion": "pericardium",
}
RIDGE_PENALTIES = tuple(10.0 ** np.linspace(-3, 4, 15))      # times the number of training rows
LOGISTIC_PENALTIES = tuple(10.0 ** np.linspace(-5, 2, 8))    # on the mean log-loss


def check_outside_repo(path):
    root, target = os.path.realpath(REPO), os.path.realpath(path)
    if os.path.commonpath([root, target]) == root:
        raise ValueError(f"out_dir {path} is inside the code repository ({root}): its files hold study identifiers.")


# ---------------------------------------------------------------------------------------------------------- fitting

def fold_ids(n, folds, seed=0, labels=None):
    """[n] fold index of each row, from a seeded permutation; with binary `labels`, stratified (each class is dealt
    round robin over the folds in its own permuted order)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.int64)
    groups = [np.arange(n)] if labels is None else [np.flatnonzero(labels > 0.5), np.flatnonzero(labels <= 0.5)]
    start = 0
    for rows in groups:
        rows = rows[rng.permutation(rows.size)]
        out[rows] = (np.arange(rows.size) + start) % folds
        start += rows.size
    return out


def standardise(train, *others):
    """(train, *others) standardised with the training rows' mean and standard deviation (1 where constant)."""
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    return tuple((x - mean) / std for x in (train,) + others)


def ridge_predictions(x_train, y_train, x_test, penalties):
    """[len(test), len(penalties)] ridge predictions (with an intercept, not penalised) for every penalty, from one
    eigendecomposition: of X'X when there are fewer features than rows, else of XX' (the dual form)."""
    mx, my = x_train.mean(0, keepdim=True), y_train.mean()
    xc, yc = x_train - mx, y_train - my
    lam = torch.as_tensor(penalties, dtype=xc.dtype, device=xc.device)
    n, d = xc.shape
    if d <= n:
        s, v = torch.linalg.eigh(xc.T @ xc)
        w = v @ ((v.T @ (xc.T @ yc))[:, None] / (s.clamp(min=0)[:, None] + lam[None, :]))
        return (x_test - mx) @ w + my
    s, u = torch.linalg.eigh(xc @ xc.T)
    a = u @ ((u.T @ yc)[:, None] / (s.clamp(min=0)[:, None] + lam[None, :]))
    return ((x_test - mx) @ xc.T) @ a + my


def fit_ridge(x_train, y_train, x_test, folds=5, seed=0, penalties=RIDGE_PENALTIES):
    """(test predictions, report): the penalty (times the number of rows) with the lowest mean cross-validated MAE
    on the training rows, refitted on all of them."""
    n = x_train.shape[0]
    fold = torch.as_tensor(fold_ids(n, folds, seed), device=x_train.device)
    cv = torch.zeros(len(penalties), dtype=x_train.dtype, device=x_train.device)
    for f in range(folds):
        tr, te = fold != f, fold == f
        pred = ridge_predictions(x_train[tr], y_train[tr], x_train[te], [p * int(tr.sum()) for p in penalties])
        cv += (pred - y_train[te][:, None]).abs().mean(0) / folds
    best = int(torch.argmin(cv))
    pred = ridge_predictions(x_train, y_train, x_test, [penalties[best] * n])[:, 0]
    return pred, {"penalty": float(penalties[best]), "penalty_index": best, "penalties": len(penalties),
                  "cv_mae": [float(v) for v in cv]}


def logistic_fit(x, y, masks, penalties, max_iter=500):
    """(W [F, L, d], b [F, L]) of L2-penalised logistic regressions, one per (row mask, penalty): F masks [F, n] over
    the rows of x [n, d], L penalties on the mean log-loss. The problems are independent; their sum is minimised in
    one L-BFGS run."""
    F_, L = masks.shape[0], len(penalties)
    dtype, device = x.dtype, x.device
    weight = masks.to(dtype) / masks.sum(1, keepdim=True).to(dtype)               # [F, n]
    prevalence = (weight * y[None]).sum(1).clamp(1e-4, 1 - 1e-4)
    W = torch.zeros(F_, L, x.shape[1], dtype=dtype, device=device, requires_grad=True)
    b = torch.logit(prevalence)[:, None].repeat(1, L).clone().requires_grad_(True)
    lam = torch.as_tensor(penalties, dtype=dtype, device=device)
    optimizer = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = torch.einsum("nd,fld->fln", x, W) + b[..., None]
        nll = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[None, None].expand_as(logits),
                                                                   reduction="none")
        loss = (nll * weight[:, None]).sum() + 0.5 * (lam[None, :] * (W ** 2).sum(-1)).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return W.detach(), b.detach()


def fit_logistic(x_train, y_train, x_test, folds=5, seed=0, penalties=LOGISTIC_PENALTIES):
    """(test probabilities, report): the penalty with the highest mean cross-validated AUROC on the training rows,
    refitted on all of them (the refit is solved together with the folds)."""
    n = x_train.shape[0]
    fold = torch.as_tensor(fold_ids(n, folds, seed, y_train.cpu().numpy()), device=x_train.device)
    masks = torch.stack([fold != f for f in range(folds)] + [torch.ones(n, dtype=torch.bool, device=fold.device)])
    W, b = logistic_fit(x_train, y_train, masks, penalties)
    cv = np.zeros(len(penalties))
    for f in range(folds):
        te = (fold == f).cpu().numpy()
        scores = (torch.einsum("nd,ld->nl", x_train[fold == f], W[f]) + b[f]).cpu().numpy()
        cv += np.array([auroc(y_train.cpu().numpy()[te] > 0.5, scores[:, j]) for j in range(len(penalties))]) / folds
    best = int(np.nanargmax(cv))
    prob = torch.sigmoid(x_test @ W[folds, best] + b[folds, best])
    return prob, {"penalty": float(penalties[best]), "penalty_index": best, "penalties": len(penalties),
                  "cv_auroc": [float(v) for v in cv]}


# ------------------------------------------------------------------------------------------------------- embeddings

def pool_bags(values, support, fallback):
    """values [bags, N, S, ...] and support [bags, N, S] -> the mean over the supporting bags [N, S, ...], and
    `fallback` [N, S, ...] where no bag supports; also the number of supporting bags [N, S]."""
    w = support.to(values.dtype)
    count = w.sum(0)
    w = w.reshape(w.shape + (1,) * (values.dim() - w.dim()))
    total = (values * w).sum(0)
    denominator = count.reshape(count.shape + (1,) * (values.dim() - count.dim() - 1))
    pooled = torch.where(denominator > 0, total / denominator.clamp(min=1), fallback)
    return pooled, count.to(torch.int64)


def collate_bags(items):
    batch = {key: torch.stack([item[key] for item in items]) for key in BAG_KEYS}
    batch["study_id"] = [item["study_id"] for item in items]
    return batch


def token_means(tokens, clip_valid, eligible):
    """Baselines of one batch of bags: (per-structure mean of the routed clips' mean token [B, S, D], its support
    [B, S], mean of all clips' mean token [B, D])."""
    clip_mean = tokens.float().mean(dim=2)                                     # [B, C, D]
    usable = (eligible & clip_valid[:, None, :]).float()                       # [B, S, C]
    count = usable.sum(-1)
    routed = torch.einsum("bsc,bcd->bsd", usable, clip_mean) / count.clamp(min=1)[..., None]
    valid = clip_valid.float()
    study = (valid[..., None] * clip_mean).sum(1) / valid.sum(1).clamp(min=1)[:, None]
    return routed, count > 0, study


@torch.no_grad()
def embed_split(models, dataset, device, amp, num_bags, batch_size, num_workers):
    """{run: {"pre", "post", "support"}} and the token baselines {"tokens_routed", "tokens_study", "tokens_support"}
    of the dataset's studies (its order), as float32 / int64 CPU tensors."""
    workers = {"multiprocessing_context": "spawn", "persistent_workers": False} if num_workers > 0 else {}
    per_bag = {name: {"pre": [], "post": [], "has": []} for name in models}
    routed_bags, routed_has, study_bags = [], [], []
    for bag in range(num_bags):
        dataset.eval_bag_index = bag
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             collate_fn=collate_bags, **workers)
        chunks = {name: {"pre": [], "post": [], "has": []} for name in models}
        routed_c, has_c, study_c = [], [], []
        for batch in loader:
            batch = _to(batch, device)
            routed, has, study = token_means(batch["tokens"], batch["clip_valid"], batch["eligible"])
            routed_c.append(routed.cpu())
            has_c.append(has.cpu())
            study_c.append(study.cpu())
            for name, model in models.items():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp == "bf16"):
                    pre, post, has_clip = model.pooler(batch["tokens"], batch["clip_valid"], batch["eligible"],
                                                       batch["view_ids"], batch["acquisition_ids"])
                chunks[name]["pre"].append(pre.float().cpu())
                chunks[name]["post"].append(post.float().cpu())
                chunks[name]["has"].append(has_clip.cpu())
        for name in models:
            for key in ("pre", "post", "has"):
                per_bag[name][key].append(torch.cat(chunks[name][key]))
        routed_bags.append(torch.cat(routed_c))
        routed_has.append(torch.cat(has_c))
        study_bags.append(torch.cat(study_c))
        print(f"bag {bag + 1}/{num_bags}: {len(dataset)} studies", flush=True)
    out = {}
    for name in models:
        has = torch.stack(per_bag[name]["has"])
        result = {"support": None}
        for key in ("pre", "post"):
            values = torch.stack(per_bag[name][key])
            # a structure without a usable clip in any bag keeps its null-token output (the same in every bag)
            result[key], result["support"] = pool_bags(values, has, values[0])
        out[name] = result
    routed_has_t = torch.stack(routed_has)
    routed_v = torch.stack(routed_bags)
    out[TOKEN_RUN] = {}
    out[TOKEN_RUN]["tokens_routed"], out[TOKEN_RUN]["tokens_support"] = pool_bags(routed_v, routed_has_t,
                                                                                  torch.zeros_like(routed_v[0]))
    out[TOKEN_RUN]["tokens_study"] = torch.stack(study_bags).mean(0)
    return out


def read_split_clips(path, split):
    table = read_table(path)
    if "split" in table.columns:
        other = set(table.split.astype(str)) - {split}
        if other:
            raise ValueError(f"{path} holds rows of {sorted(other)} besides {split!r}: pass a table of one split.")
    if table.empty:
        raise ValueError(f"{path} has no clip.")
    return table


def load_runs(runs, train_clips, device):
    """{name: model} in eval mode, after checking that every run trained on `train_clips` with the same routing
    registry and data settings; returns (models, the shared config's data section, routing, amp)."""
    models, data_ref, routing, amp = {}, None, None, None
    train_sha = file_sha256(train_clips)
    for name, config_path, checkpoint in runs:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        data = {key: config["data"].get(key) for key in DATA_KEYS}
        if data_ref is None:
            data_ref, amp = data, config["optimization"].get("amp", "none")
            routing = load_routing(data.get("routing") or DEFAULT_ROUTING)
        elif data != data_ref:
            raise ValueError(f"Run {name} has other data settings than the first run: embed it separately.")
        if amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fingerprint = state["fingerprint"]
        if fingerprint["clips_sha256"] != train_sha:
            raise ValueError(f"{checkpoint} trained on another clip table than {train_clips}: its training studies "
                             "would not be the probe's training studies.")
        if fingerprint["routing_sha256"] != routing.sha256:
            raise ValueError(f"{checkpoint} trained with another routing registry.")
        model, _ = build_model(config, routing)
        model.load_state_dict(state["model"])
        models[name] = model.to(device).eval()
    return models, data_ref, routing, amp


def embed(args):
    check_outside_repo(args.out_dir)
    device = _device({"device": args.device})
    models, data, routing, amp = load_runs(args.run, args.train_clips, device)
    with open(args.run[0][1]) as f:
        m = yaml.safe_load(f)["model"]
    tables = {"train": read_split_clips(args.train_clips, "train"),
              args.val_split: read_split_clips(args.val_clips, args.val_split)}
    overlap = set(tables["train"].study_id) & set(tables[args.val_split].study_id)
    if overlap:
        raise ValueError(f"{len(overlap)} evaluation studies are also training studies.")
    caches = {"train": args.train_cache_dir, args.val_split: args.val_cache_dir}
    for split, table in tables.items():
        cache = TokenCache(caches[split], token_grid=m["token_grid"], embed_dim=m["embed_dim"])
        dataset = RadarStudyDataset(table, pd.DataFrame(columns=list(CAPTION_COLUMNS)), routing, cache,
                                    num_clips=data["num_clips"] or 12, selection=data["selection"] or "view_balanced",
                                    training=False, eval_seed=args.eval_seed,
                                    acquisition_classes=data["acquisition_classes"] or ("bmode", "color_flow"))
        out = embed_split(models, dataset, device, amp, args.num_bags, args.batch_size, args.num_workers)
        folder = os.path.join(args.out_dir, "embeddings", split)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "study_ids.txt"), "w") as f:
            f.write("\n".join(study for study, _ in dataset.studies) + "\n")
        for run, arrays in out.items():
            for key, value in arrays.items():
                stem = key if run == TOKEN_RUN else f"{run}.{key}"
                np.save(os.path.join(folder, f"{stem}.npy"), value.numpy())
        meta = {"split": split, "studies": len(dataset), "structures": [s.name for s in routing.trained],
                "num_bags": args.num_bags, "eval_seed": args.eval_seed, "bag_record": dataset.bag_record(),
                "clips_sha256": file_sha256(args.train_clips if split == "train" else args.val_clips),
                "runs": {name: {"config": config, "checkpoint": checkpoint}
                         for name, config, checkpoint in args.run}}
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)


# ------------------------------------------------------------------------------------------------------------ probe

def read_labels(path, study_ids):
    """{column: float array aligned to study_ids} of the label table (NaN where blank or absent). Only the rows of
    `study_ids` are kept."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table = table[table.study_id.isin(set(study_ids))]
    if table.study_id.duplicated().any():
        raise ValueError("The label table repeats a study.")
    table = table.set_index("study_id").reindex(study_ids)
    out = {}
    for column in table.columns:
        text = table[column].fillna("").str.strip()
        out[column] = pd.to_numeric(text.where(text != ""), errors="coerce").to_numpy(dtype=float)
    return out


def load_split(folder):
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(folder, "study_ids.txt")) as f:
        ids = [line.strip() for line in f if line.strip()]
    return meta, ids


def run_inputs(folder, run, structure_index):
    """{input name: function(structure) -> [N, d] array} for one run (or the token baselines)."""
    def load(stem):
        return np.load(os.path.join(folder, f"{stem}.npy"), mmap_mode="r")

    if run == TOKEN_RUN:
        routed, study = load("tokens_routed"), load("tokens_study")
        return {"tokens_routed": lambda s: routed[:, structure_index[s]],
                "tokens_study": lambda s: study,
                "tokens_concat": lambda s: routed.reshape(routed.shape[0], -1)}, load("tokens_support")
    pre, post = load(f"{run}.pre"), load(f"{run}.post")
    return {"routed_post": lambda s: post[:, structure_index[s]], "routed_pre": lambda s: pre[:, structure_index[s]],
            "concat_post": lambda s: post.reshape(post.shape[0], -1),
            "concat_pre": lambda s: pre.reshape(pre.shape[0], -1)}, load(f"{run}.support")


def probe(args):
    check_outside_repo(args.out_dir)
    device = _device({"device": args.device})
    root = os.path.join(args.out_dir, "embeddings")
    meta_tr, ids_tr = load_split(os.path.join(root, "train"))
    meta_ev, ids_ev = load_split(os.path.join(root, args.val_split))
    if set(ids_tr) & set(ids_ev):
        raise ValueError("Evaluation studies are also training studies.")
    structures = meta_tr["structures"]
    index = {s: i for i, s in enumerate(structures)}
    with open(args.targets_info) as f:
        kinds = json.load(f)["tasks"]
    labels_tr, labels_ev = read_labels(args.labels, ids_tr), read_labels(args.labels, ids_ev)
    runs = [TOKEN_RUN] + list(meta_tr["runs"])
    if args.probe_runs:
        unknown = sorted(set(args.probe_runs) - set(runs))
        if unknown:
            raise ValueError(f"No embeddings of {unknown}; embedded: {runs}.")
        runs = [run for run in runs if run in args.probe_runs]
    settings = {"min_train": args.min_train, "min_eval": args.min_eval, "min_positives": args.min_positives,
                "folds": args.folds, "seed": args.seed}
    for run in runs:
        report, predictions = {**settings, "runs": {}}, []
        inputs_tr, support_tr = run_inputs(os.path.join(root, "train"), run, index)
        inputs_ev, support_ev = run_inputs(os.path.join(root, args.val_split), run, index)
        report["runs"][run] = {}
        for target, kind in kinds.items():
            if kind not in ("regression", "binary") or target not in labels_tr:
                continue
            structure = ROUTED_STRUCTURE.get(target)
            y_tr, y_ev = labels_tr[target], labels_ev[target]
            rows_tr, rows_ev = np.isfinite(y_tr), np.isfinite(y_ev)
            if structure is not None:
                rows_tr &= np.asarray(support_tr[:, index[structure]]) > 0
                rows_ev &= np.asarray(support_ev[:, index[structure]]) > 0
            n_tr, n_ev = int(rows_tr.sum()), int(rows_ev.sum())
            if n_tr < args.min_train or n_ev < args.min_eval:
                continue
            if kind == "binary" and min(y_tr[rows_tr].sum(), (1 - y_tr[rows_tr]).sum(), y_ev[rows_ev].sum(),
                                        (1 - y_ev[rows_ev]).sum()) < args.min_positives:
                continue
            entry = {"kind": kind, "structure": structure, "train_rows": n_tr, "eval_rows": n_ev, "inputs": {}}
            if kind == "regression":
                yt = y_tr[rows_tr]
                const = float(np.median(yt))
                entry["train_sd"] = float(yt.std(ddof=1))
                entry["constant"] = {"value": const, **regression(y_ev[rows_ev], np.full(n_ev, const))}
                entry["constant"]["standardised_mae"] = entry["constant"]["mae"] / entry["train_sd"]
            for name, get in inputs_tr.items():
                if structure is None and name.startswith(("routed", "tokens_routed")):
                    continue
                x_tr = torch.as_tensor(np.asarray(get(structure))[rows_tr], dtype=torch.float64, device=device)
                x_ev = torch.as_tensor(np.asarray(inputs_ev[name](structure))[rows_ev], dtype=torch.float64,
                                       device=device)
                x_tr, x_ev = standardise(x_tr, x_ev)
                t_tr = torch.as_tensor(y_tr[rows_tr], dtype=torch.float64, device=device)
                if kind == "regression":
                    pred, fit = fit_ridge(x_tr, t_tr, x_ev, args.folds, args.seed)
                    pred = pred.cpu().numpy()
                    metrics = regression(y_ev[rows_ev], pred)
                    metrics["standardised_mae"] = metrics["mae"] / entry["train_sd"]
                else:
                    pred, fit = fit_logistic(x_tr, t_tr, x_ev, args.folds, args.seed)
                    pred = pred.cpu().numpy()
                    positive = y_ev[rows_ev] > 0.5
                    metrics = {"n": n_ev, "positives": int(positive.sum()), "auroc": auroc(positive, pred),
                               "auprc": average_precision(positive, pred)}
                entry["inputs"][name] = {"dim": int(x_tr.shape[1]), "fit": fit, "metrics": metrics}
                predictions.append(pd.DataFrame({"run": run, "input": name, "target": target,
                                                 "study_id": np.asarray(ids_ev)[rows_ev], "label": y_ev[rows_ev],
                                                 "prediction": pred}))
            report["runs"][run][target] = entry
            shown = {k: round(v["metrics"].get("mae", v["metrics"].get("auroc")), 4)
                     for k, v in entry["inputs"].items()}
            print(f"{run} {target} ({kind}, {structure}, {n_tr}/{n_ev}): {shown}", flush=True)
        folder = os.path.join(args.out_dir, "probe", run)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "metrics.json"), "w") as f:
            json.dump(report, f, indent=1, default=float)
        pd.concat(predictions).to_csv(os.path.join(folder, "predictions.csv"), index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", nargs=3, action="append", metavar=("NAME", "CONFIG", "CHECKPOINT"),
                        help="a finished run: name, the config that trained it, its checkpoint; repeatable")
    parser.add_argument("--stage", choices=("embed", "probe", "all"), default="all")
    parser.add_argument("--train_clips", help="clip table of the training split (the runs' training table)")
    parser.add_argument("--train_cache_dir", help="token cache of the training clips")
    parser.add_argument("--val_clips", help="clip table of the evaluation split")
    parser.add_argument("--val_cache_dir", help="token cache of the evaluation clips")
    parser.add_argument("--val_split", default="val", help="name of the evaluation split")
    parser.add_argument("--labels", help="study-level label table (data/make_study_targets.py)")
    parser.add_argument("--targets_info", help="targets_info.json beside the label table (each target's task)")
    parser.add_argument("--out_dir", required=True, help="outside the repository: files hold study ids")
    parser.add_argument("--num_bags", type=int, default=3)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe_runs", nargs="+", help="probe only these embedded runs (`tokens`: the baselines)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="cross-validation fold seed")
    parser.add_argument("--min_train", type=int, default=200, help="fewest labelled training studies of a target")
    parser.add_argument("--min_eval", type=int, default=50, help="fewest labelled evaluation studies of a target")
    parser.add_argument("--min_positives", type=int, default=10,
                        help="binary: fewest positives (and negatives) in each split")
    args = parser.parse_args(argv)
    if args.val_split == "train":
        raise ValueError("val_split must name a held-out split.")
    if args.stage in ("embed", "all"):
        missing = [flag for flag in ("run", "train_clips", "train_cache_dir", "val_clips", "val_cache_dir")
                   if not getattr(args, flag)]
        if missing:
            raise ValueError(f"The embed stage needs {missing}.")
        embed(args)
    if args.stage in ("probe", "all"):
        if not args.labels or not args.targets_info:
            raise ValueError("The probe stage needs labels and targets_info.")
        probe(args)


if __name__ == "__main__":
    main()
