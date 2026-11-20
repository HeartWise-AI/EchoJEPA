# evals/view_classifier/probe.py

"""Arm A: an attentive probe on the frozen EchoJEPA encoder's cached tokens.

The tokens are written once by `evals.radar_frozen.build_token_cache` (one clip: the first 16 frames at step 2,
224 px, ImageNet normalisation, as the study probe reads a clip; [1568, 768] tokens for the ViT-B). All clips of the
run are loaded into memory (float16, about 2.4 MB a clip), and every fold and setting trains on them: the encoder never
runs during training. The probe is `src.models.attentive_pooler.AttentiveClassifier`, the probe of the frozen
classification evaluation (`evals/video_classification_frozen`), with the depth and heads of the config.

`--arm` names the config section and the output folder (default `arm_a`, the first runs). Another arm (e.g.
`arm_a_v2`) may set, under `probe`, `head_dropout` (dropout on the pooled vector before the linear layer, training
only; the state dict keys stay those of `AttentiveClassifier`) and `token_drop` (each training clip keeps a random
subset of its tokens, a different one at every step; evaluation uses all tokens), and under `cv` its loss weighting
(`train.arm_cv`). With none of these set, the probe trains exactly as before.

    python -m evals.view_classifier.probe --config CONFIG --pseudo off --folds 0 1 2 --device cuda:0
    python -m evals.view_classifier.probe --config CONFIG --pseudo off --merge
    python -m evals.view_classifier.probe --config CONFIG --arm arm_a_v2 --pseudo off --folds 0 --device cuda:0
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import train as vt
from evals.view_classifier.io import check_colour_fix, read_run_inputs, run_dir

ARM = "arm_a"


def build_probe(embed_dim, num_heads, depth, num_classes, head_dropout=0.0):
    from src.models.attentive_pooler import AttentiveClassifier
    model = AttentiveClassifier(embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_classes=num_classes)
    if head_dropout:
        p = float(head_dropout)
        if not 0.0 < p < 1.0:
            raise ValueError("head_dropout must be in [0, 1).")
        # A pre-hook, not a new module: the state dict is unchanged, so a saved probe loads into AttentiveClassifier.
        model.linear.register_forward_pre_hook(
            lambda module, args: (torch.nn.functional.dropout(args[0], p, module.training),) + tuple(args[1:]))
    return model


def drop_tokens(x, fraction, generator):
    """x [B, L, D] -> [B, K, D]: a random subset of K = round((1 - fraction) L) tokens per clip, in their order."""
    if not fraction:
        return x
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("token_drop must be in [0, 1).")
    b, n = x.shape[:2]
    keep = max(1, int(round(n * (1.0 - float(fraction)))))
    noise = torch.rand(b, n, generator=generator)
    idx = noise.argsort(dim=1)[:, :keep].sort(dim=1).values.to(x.device)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[2]))


def param_groups(model, hp):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.endswith("query_tokens") else decay).append(p)
    return [{"params": decay, "lr": float(hp["lr"]), "weight_decay": float(hp["weight_decay"])},
            {"params": no_decay, "lr": float(hp["lr"]), "weight_decay": 0.0}]


def load_tokens(cache_root, clip_ids, expected_sha256=None, token_grid=(8, 14, 14), embed_dim=768):
    """(float16 tensor [N, L, D], {clip_id: row}) of `clip_ids` from a finished token cache."""
    from src.datasets.radar_token_cache import TokenCache
    cache = TokenCache(cache_root, token_grid=token_grid, embed_dim=embed_dim)
    found = cache.meta.get("provenance", {}).get("encoder_sha256")
    if expected_sha256 and found != expected_sha256:
        raise ValueError(f"The token cache was written with encoder sha256 {found}, not {expected_sha256}.")
    ids = list(dict.fromkeys(str(c) for c in clip_ids))
    missing = [c for c in ids if c not in cache]
    if missing:
        raise ValueError(f"{len(missing)} clips are not in the token cache (unreadable, or the cache predates them).")
    data = torch.empty((len(ids), *cache.shape), dtype=torch.float16)
    for i, c in enumerate(ids):
        data[i] = cache.load(c).to(torch.float16)
    return data, {c: i for i, c in enumerate(ids)}


def make_arm(data, row, probe_cfg, classes=vc.NEW_CLASSES, name=ARM):
    token_drop = float(probe_cfg.get("token_drop", 0.0))
    head_dropout = float(probe_cfg.get("head_dropout", 0.0))

    def build(hp):
        return build_probe(data.shape[-1], int(probe_cfg.get("num_heads", 16)), int(probe_cfg.get("depth", 4)),
                           len(classes), head_dropout)

    def prepare(x, training, generator):
        x = x.float()
        return drop_tokens(x, token_drop, generator) if training and token_drop else x

    return vt.Arm(name=name, data=data, row=row, build=build, param_groups=param_groups, prepare=prepare)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo", choices=("on", "off"), required=True)
    parser.add_argument("--folds", nargs="*", default=[], help="fold numbers and/or `final`")
    parser.add_argument("--merge", action="store_true", help="write oof_predictions.csv from the finished folds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", default=ARM, help="the config section and the output folder (default arm_a)")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    name = args.arm
    if name == "arm_b" or not name.startswith("arm_") or name not in cfg or "readout" in cfg[name]:
        raise ValueError(f"{name!r} is not an attentive-probe arm of the config.")
    labels, pseudo = read_run_inputs(cfg, args.pseudo == "on")
    out = run_dir(cfg, name, args.pseudo)
    classes, _ = vc.from_config(cfg)
    if args.merge:
        oof = vt.merge_folds(out, labels, int(cfg["cv"]["k"]), classes)
        print(f"{name} pseudo {args.pseudo}: {len(oof)} out-of-fold predictions, accuracy "
              f"{float((oof.y == np.asarray(oof[[f'p_{c}' for c in classes]]).argmax(1)).mean()):.3f}")
        return
    a = cfg[name]
    with open(os.path.join(cfg["paths"]["token_cache"], "meta.json")) as f:
        check_colour_fix(cfg, json.load(f))
    ids = list(labels.clip_id) + ([] if pseudo is None else list(pseudo.clip_id))
    data, row = load_tokens(cfg["paths"]["token_cache"], ids, a.get("encoder_sha256"),
                            tuple(a.get("token_grid", (8, 14, 14))), int(a.get("embed_dim", 768)))
    arm = make_arm(data, row, a.get("probe", {}), classes, name)
    device = torch.device(args.device)
    os.makedirs(out, exist_ok=True)
    cv = vt.arm_cv(cfg["cv"], a.get("cv"))
    for fold in args.folds:
        vt.run_fold(arm, labels, pseudo, fold if fold == vt.FINAL else int(fold), vt.grid_from_config(a), cv,
                    classes, out, device)


if __name__ == "__main__":
    main()
