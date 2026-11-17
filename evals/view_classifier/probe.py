# evals/view_classifier/probe.py

"""Arm A: an attentive probe on the frozen EchoJEPA encoder's cached tokens.

The tokens are written once by `evals.radar_frozen.build_token_cache` (one clip: the first 16 frames at step 2,
224 px, ImageNet normalisation, as the study probe reads a clip; [1568, 768] tokens for the ViT-B). All clips of the
run are loaded into memory (float16, about 2.4 MB a clip), and every fold and setting trains on them: the encoder never
runs during training. The probe is `src.models.attentive_pooler.AttentiveClassifier`, the probe of the frozen
classification evaluation (`evals/video_classification_frozen`), with the depth and heads of the config.

    python -m evals.view_classifier.probe --config CONFIG --pseudo off --folds 0 1 2 --device cuda:0
    python -m evals.view_classifier.probe --config CONFIG --pseudo off --merge
"""

import argparse
import os

import numpy as np
import torch
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import train as vt
from evals.view_classifier.io import read_run_inputs, run_dir

ARM = "arm_a"


def build_probe(embed_dim, num_heads, depth, num_classes):
    from src.models.attentive_pooler import AttentiveClassifier
    return AttentiveClassifier(embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_classes=num_classes)


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


def make_arm(data, row, probe_cfg, classes=vc.NEW_CLASSES):
    def build(hp):
        return build_probe(data.shape[-1], int(probe_cfg.get("num_heads", 16)), int(probe_cfg.get("depth", 4)),
                           len(classes))

    def prepare(x, training, generator):
        return x.float()

    return vt.Arm(name=ARM, data=data, row=row, build=build, param_groups=param_groups, prepare=prepare)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo", choices=("on", "off"), required=True)
    parser.add_argument("--folds", nargs="*", default=[], help="fold numbers and/or `final`")
    parser.add_argument("--merge", action="store_true", help="write oof_predictions.csv from the finished folds")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    labels, pseudo = read_run_inputs(cfg, args.pseudo == "on")
    out = run_dir(cfg, ARM, args.pseudo)
    classes, _ = vc.from_config(cfg)
    if args.merge:
        oof = vt.merge_folds(out, labels, int(cfg["cv"]["k"]), classes)
        print(f"{ARM} pseudo {args.pseudo}: {len(oof)} out-of-fold predictions, accuracy "
              f"{float((oof.y == np.asarray(oof[[f'p_{c}' for c in classes]]).argmax(1)).mean()):.3f}")
        return
    a = cfg[ARM]
    ids = list(labels.clip_id) + ([] if pseudo is None else list(pseudo.clip_id))
    data, row = load_tokens(cfg["paths"]["token_cache"], ids, a.get("encoder_sha256"),
                            tuple(a.get("token_grid", (8, 14, 14))), int(a.get("embed_dim", 768)))
    arm = make_arm(data, row, a.get("probe", {}), classes)
    device = torch.device(args.device)
    os.makedirs(out, exist_ok=True)
    for fold in args.folds:
        vt.run_fold(arm, labels, pseudo, fold if fold == vt.FINAL else int(fold), vt.grid_from_config(a), cfg["cv"],
                    classes, out, device)


if __name__ == "__main__":
    main()
