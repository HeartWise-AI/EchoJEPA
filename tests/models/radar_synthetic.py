# tests/models/radar_synthetic.py

"""A small synthetic RADAR training set: a token cache, a clip table and a caption table on disk.

Sixteen studies with three to six clips of a few views. Six studies have captions; every captioned study has an A4C
clip and a distinct LV caption (an appended EF of unknown normality), so any global batch holding two of them
contributes through the LV. Pericardium captions alternate normal and abnormal; aortic captions exist for every
third captioned study and need a PLAX or PSAX-AV clip, so structure availability changes from batch to batch. With
a global batch of 4, every epoch has a batch with at most one captioned study (no structure can contribute
anywhere: a skipped step) and a batch with at least two (a contributing step), whatever the permutation.
"""

import json
import os
import random
import sys

import pandas as pd
import torch
import yaml

from src.datasets.radar_study_dataset import file_sha256
from src.datasets.radar_token_cache import TokenCacheWriter
from src.models.radar.routing import load_routing

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import radar_captions as rc  # noqa: E402
import radar_normality as rn  # noqa: E402

GRID, DIM = (2, 2, 2), 16
OTHER_VIEWS = ["PLAX", "PSAX_AV", "SUBCOSTAL", "A2C", "A3C"]
N_STUDIES, CAPTIONED = 16, 6


def write_approved_policy(path):
    """A copy of the shipped policy, approved for the current rules. Returns its digest."""
    with open(rn.DEFAULT_POLICY, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    digest = rn.policy_digest(raw)
    raw["approval"] = {"by": "synthetic", "on": "synthetic", "audit_report": "synthetic", "digest": digest}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)
    return digest


def seal_captions(table, folder, clips, split="train"):
    """Write `table` as the caption builder does, with `build_info.json` beside it; return its path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "captions.csv")
    table.to_csv(path, index=False)
    info = {"split": split, "routing_sha256": load_routing().sha256, "clips_sha256": file_sha256(clips),
            "captions_sha256": file_sha256(path), "policy_digest": "synthetic"}
    with open(os.path.join(folder, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)
    return path


def make_synthetic(root, seed=0):
    """Write the synthetic set under `root`; return the paths."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    writer = TokenCacheWriter(os.path.join(root, "cache"), GRID, DIM, provenance={"synthetic": True})
    clips, captions = [], []
    for i in range(N_STUDIES):
        study = f"S{i:02d}"
        views = ["A4C"] + [rng.choice(OTHER_VIEWS) for _ in range(2 + i % 4)]
        for j, view in enumerate(views):
            clip = f"{study}-c{j}"
            writer.add(clip, torch.randn(GRID[0] * GRID[1] * GRID[2], DIM, generator=generator))
            clips.append({"study_id": study, "clip_id": clip, "view": view, "acquisition": "bmode",
                          "split": "train"})
        if i >= CAPTIONED:
            continue
        lv, _ = rc.lv_caption("VG non dilaté.", 40 + 3 * i)
        captions.append({"study_id": study, "structure": "lv", "caption": lv, "caption_id": rc.caption_id(lv),
                         "normality": "unknown"})
        peri = "Pas d'épanchement péricardique." if i % 2 == 0 else "Épanchement péricardique modéré."
        captions.append({"study_id": study, "structure": "pericardium", "caption": peri,
                         "caption_id": rc.caption_id(peri), "normality": "normal" if i % 2 == 0 else "abnormal"})
        if i % 3 == 0:
            aortic = "Valve aortique calcifiée." if i % 2 else "Valve aortique de structure normale."
            captions.append({"study_id": study, "structure": "aortic", "caption": aortic,
                             "caption_id": rc.caption_id(aortic), "normality": "abnormal" if i % 2 else "normal"})
    writer.close()
    pd.DataFrame(clips).to_csv(os.path.join(root, "clips.csv"), index=False)
    policy = os.path.join(root, "policy.yaml")
    digest = write_approved_policy(policy)
    seal_captions(pd.DataFrame(captions).assign(normality_status="approved", policy_digest=digest), root,
                  os.path.join(root, "clips.csv"))
    return {"cache_dir": os.path.join(root, "cache"), "clips": os.path.join(root, "clips.csv"),
            "captions": os.path.join(root, "captions.csv"), "normality_policy": policy}


def synthetic_config(paths, text=None, device="cpu", **optimization):
    """A training config for the synthetic set; `text` updates the caption encoder section (for example dropout),
    keyword arguments update the optimization section."""
    opt = {"epochs": 2, "global_batch": 4, "lr": 1e-3, "text_lr": 1e-3, "min_lr": 1e-5, "weight_decay": 0.05,
           "warmup_fraction": 0.25, "seed": 0, "amp": "none", "num_workers": 0}
    opt.update(optimization)
    text_config = {"backbone": "tiny", "vocab_size": 200, "hidden_size": 16, "num_layers": 2, "num_heads": 2,
                   "max_length": 32, "trainable_layers": 1}
    text_config.update(text or {})
    return {
        "device": device,
        "data": {**paths, "split": "train", "num_clips": 4, "selection": "view_balanced",
                 "acquisition_classes": ["bmode", "color_flow"]},
        "model": {"token_grid": list(GRID), "embed_dim": DIM, "depth": 1, "num_heads": 2, "proj_dim": 8},
        "text": text_config,
        "loss": {"use_exact": True, "use_normal": True, "use_abnormal_teacher": True, "teacher_momentum": 0.9},
        "optimization": opt,
    }
