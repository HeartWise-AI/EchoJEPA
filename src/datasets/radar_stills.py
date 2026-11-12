# src/datasets/radar_stills.py

"""The Doppler still stream of a RADAR study item (optional; `RadarStudyDataset(stills=StillSource(...))`).

Inputs:

- `stills`: the still table of one split (`data/build_still_manifest.py`): `study_id`, `still_id`, `still_tag` (a tag
  of the registry's `still_structs`, empty for none), `still_type`, the conditioning columns, the target columns and
  optionally a heat-map column (a `.npy` path, already in crop coordinates, empty for none);
- a `TokenCache` of the stills' frozen tokens (one tubelet, e.g. grid [1, 14, 14]), keyed by `still_id`;
- optionally `demographics`: `study_id`, `age_years`, `sex` (M, F or empty).

A still whose tag the registry does not route (or without a tag) is left out, unless it carries a measurement
target: then it stays for the heads, feeds no structure, and is counted (`unrouted_with_targets`). Per study, at
most `num_stills` stills are read, balanced over still tags as clips are over view families
(`study_bags.select_clips`), drawn from the study's RNG after its clips, so the clip bag is the v0 bag. Each item
gets: `still_tokens` [K, Ls, D], `still_valid` [K], `still_tag_ids`, `still_type_ids` [K] (-1 padded or unknown),
`still_eligible` [S, K] (`Routing.still_eligibility`), `still_cond` [K, c] (missing values as 0), `still_bias`
[K, Ls] (zero without a heat-map; the spatial map repeated over the cache's temporal positions, so a still encoded
from 16 repeated frames, grid [8, H, W], reads the same map at each), `still_y` [K, Q] and `still_y_mask` [K, Q]
(z-scored log targets) and `demo` [4] (z-scored age, male, female, age missing).

Leakage guard: a conditioning column may not be a target, nor start with `y_` (a measured value is never an input
here), and it must be one of `INFERENCE_INPUTS`: the columns known before anyone measures the still. A name check
alone lets through a column that holds the measurement under another name (a gradient computed from the peak
velocity, the number of measurement rows), so every other column is refused, whatever it is called. The targets'
scaler and the age statistics are fitted on the source's own stills and studies (the training split's, whatever
else the demographics table holds) and passed to the others.

Preprocessing guard (`known_preprocess`): a still cache that records where it came from must also record a
preprocessing recipe of `PREPROCESS_ACCEPTED`. Tokens of stills that were not masked, or masked by a recipe nobody
listed, are refused. This shows which recipe wrote the tokens; it does not show that the recipe removes every
drawn measurement (see docs/radar/RADAR_stills_integration.md, leakage controls).
"""

import hashlib

import numpy as np
import pandas as pd
import torch

from src.datasets.still_preprocess import PREPROCESS_VERSION, heatmap_bias
from src.datasets.study_bags import select_clips
from src.models.radar.heads import encode_targets, fit_log_scaler

DEMO_DIM = 4
ENCODER_KEYS = ("encoder_sha256", "encoder")     # of a token cache's provenance (the cache writer's record)
# What a still's conditioning may hold: the velocity scale of the crop (DICOM calibration) and, when a run asks for
# them, EchoNet's own prediction on the still. Nothing that the measurement table or the report gives.
INFERENCE_INPUTS = ("v_top_ms", "v_bot_ms", "echonet_present", "echonet_vmax_ms", "echonet_conf")
PREPROCESS_ACCEPTED = (PREPROCESS_VERSION,)      # recipes of `still_preprocess.prepare_still` a still cache may hold


def known_preprocess(still_meta):
    """Refuse a still cache whose tokens were not written from stills prepared by an accepted recipe. A cache
    written by `build_still_cache.py` records `preprocess`; one that records an encoder or anything else about its
    origin but no accepted recipe is refused. Returns the recipe, or None for a cache without any record or one
    marked `synthetic` (the tests' caches: no image was ever read)."""
    provenance = still_meta.get("provenance") or {}
    if not provenance or provenance.get("synthetic"):
        return None
    recipe = provenance.get("preprocess")
    if recipe not in PREPROCESS_ACCEPTED:
        raise ValueError(f"The still token cache records the preprocessing recipe {recipe!r}; accepted: "
                         f"{list(PREPROCESS_ACCEPTED)}. Tokens of stills that were not masked by an accepted recipe "
                         "may show the measured value: write the cache again with evals/radar_frozen/"
                         "build_still_cache.py.")
    return recipe


def same_encoder(clip_meta, still_meta):
    """Refuse a still cache that another encoder wrote than the clip cache: their tokens would not share a space,
    and one pooler reads both. Compares the embedding size and the encoder record of the caches' provenance (the
    checkpoint's sha256 and how it was built). Returns True when both records are there and equal, None when
    neither cache has one (nothing to compare: synthetic caches); one cache with a record and one without is
    refused, since it cannot be shown."""
    if clip_meta.get("embed_dim") != still_meta.get("embed_dim"):
        raise ValueError(f"The still token cache has embed_dim {still_meta.get('embed_dim')}, the clip cache "
                         f"{clip_meta.get('embed_dim')}.")
    clip, still = ({k: (meta.get("provenance") or {}).get(k) for k in ENCODER_KEYS} for meta in (clip_meta, still_meta))
    known = [any(v is not None for v in record.values()) for record in (clip, still)]
    if not any(known):
        return None
    if not all(known):
        raise ValueError("Only one of the clip and still token caches records its encoder: they cannot be shown "
                         "to come from the same one.")
    differing = [k for k in ENCODER_KEYS if clip[k] != still[k]]
    if differing:
        raise ValueError(f"The clip and still token caches come from different encoders ({differing} differ): "
                         "write the still cache with the clip cache's encoder checkpoint and settings.")
    return True


class StillSource:

    def __init__(self, stills, routing, cache, structures, num_stills=8, cond_columns=(), target_columns=(),
                 scaler=None, heatmap_column=None, demographics=None, demo_stats=None):
        for column in ("study_id", "still_id", "still_tag", "still_type", *cond_columns, *target_columns):
            if column not in stills.columns:
                raise ValueError(f"The still table needs a {column!r} column.")
        leaking = [c for c in cond_columns if c in target_columns or c.startswith("y_")]
        if leaking:
            raise ValueError(f"Conditioning columns {leaking} are measured values: a target is never an input.")
        unlisted = [c for c in cond_columns if c not in INFERENCE_INPUTS]
        if unlisted:
            raise ValueError(f"Conditioning columns {unlisted} are not among the inputs known before a measurement "
                             f"({list(INFERENCE_INPUTS)}): a column that follows from the measurement is refused "
                             "whatever its name.")
        self.preprocess = known_preprocess(cache.meta)
        if stills.still_id.astype(str).duplicated().any():
            raise ValueError("A still appears twice in the still table.")
        missing = [s for s in stills.still_id.astype(str) if s not in cache]
        if missing:
            raise ValueError(f"{len(missing)} stills are not in the still token cache.")
        self.routing, self.cache, self.structures = routing, cache, list(structures)
        self.grid = tuple(int(n) for n in cache.meta["token_grid"])     # [T, H, W]: T = 1 for a 2-frame still
        if heatmap_column and self.grid[1] != self.grid[2]:
            raise ValueError("The heat-map bias needs a square still token grid (H == W).")
        self.num_stills = int(num_stills)
        self.cond_columns, self.target_columns = list(cond_columns), list(target_columns)
        self.heatmap_column = heatmap_column
        self.scaler = scaler if scaler is not None or not target_columns else fit_log_scaler(stills, target_columns)
        stills = stills.assign(study_id=stills.study_id.astype(str), still_id=stills.still_id.astype(str),
                               still_tag=stills.still_tag.fillna("").astype(str),
                               still_type=stills.still_type.fillna("").astype(str))
        # A still whose tag the registry does not route feeds no structure: left out, or it would take a bag slot
        # from one that does. One that carries a measurement target stays, for the heads: it is counted apart.
        routed = stills.still_tag.map(lambda tag: routing.still_tag_index(tag) >= 0).to_numpy(dtype=bool)
        values = (stills[self.target_columns].to_numpy(dtype=float) if self.target_columns
                  else np.zeros((len(stills), 0)))
        measured = (np.isfinite(values) & (values > 0)).any(axis=1)
        self.unrouted = int((~routed & ~measured).sum())
        self.unrouted_with_targets = int((~routed & measured).sum())
        stills = stills[routed | measured]
        # One tensor per table, of which each still holds a row (a view): a tensor per still would be a shared
        # memory segment per still when the loader's workers start, more than the system maps.
        cond = (torch.from_numpy(np.nan_to_num(stills[self.cond_columns].to_numpy(dtype=np.float32)))
                if self.cond_columns else None)
        if target_columns:
            y, present = encode_targets(stills[self.target_columns].to_numpy(dtype=float), self.scaler,
                                        self.target_columns)
        self.by_study = {}
        for i, row in enumerate(stills.itertuples(index=False)):
            heat = getattr(row, heatmap_column) if heatmap_column else ""
            self.by_study.setdefault(row.study_id, []).append({
                "still_id": row.still_id, "tag": row.still_tag, "type": row.still_type,
                "cond": cond[i] if cond is not None else None,
                "y": y[i] if target_columns else None, "present": present[i] if target_columns else None,
                "heatmap": heat if isinstance(heat, str) else ""})
        for rows in self.by_study.values():
            rows.sort(key=lambda r: r["still_id"])
        self.demo = {}
        if demographics is not None:
            demo = demographics.assign(study_id=demographics.study_id.astype(str))
            ages = pd.to_numeric(demo.age_years, errors="coerce")
            if demo_stats is None:      # fitted on this source's studies only: the table may hold every split
                own = ages[demo.study_id.isin(self.by_study).to_numpy()]
                demo_stats = [float(own.mean()), float(max(own.std(ddof=0), 1e-6))] if own.notna().any() \
                    else [0.0, 1.0]
            sexes = demo.sex.fillna("").astype(str)
            known = np.isfinite(ages.to_numpy(dtype=float))
            table = torch.from_numpy(np.stack([
                np.where(known, (ages.to_numpy(dtype=float) - demo_stats[0]) / demo_stats[1], 0.0),
                (sexes == "M").to_numpy(dtype=float), (sexes == "F").to_numpy(dtype=float),
                (~known).astype(float)], axis=1).astype(np.float32))
            self.demo = {study: table[i] for i, study in enumerate(demo.study_id)}
        self.demo_stats = demo_stats

    @property
    def studies(self):
        return set(self.by_study)

    def record(self):
        """What fixes this source's stills (part of the run's fingerprint)."""
        digest, heat = hashlib.sha256(), hashlib.sha256()
        for study in sorted(self.by_study):
            for r in self.by_study[study]:
                digest.update(f"{study}\0{r['still_id']}\0{r['tag']}\0{r['type']}\n".encode("utf-8"))
                if r["heatmap"]:        # the file's contents: the table only pins its path
                    with open(r["heatmap"], "rb") as f:
                        heat.update(f"{r['still_id']}\0{hashlib.sha256(f.read()).hexdigest()}\n".encode("utf-8"))
        return {"stills_sha256": digest.hexdigest(), "num_stills": self.num_stills, "cond": self.cond_columns,
                "targets": self.target_columns, "scaler": self.scaler, "demo_stats": self.demo_stats,
                "heatmap": bool(self.heatmap_column), "unrouted": self.unrouted,
                "unrouted_with_targets": self.unrouted_with_targets,
                "heatmaps_sha256": heat.hexdigest() if self.heatmap_column else None}

    def item(self, study_id, rng):
        rows = self.by_study.get(study_id, [])
        chosen = [rows[i] for i in select_clips([(r["still_id"], r["tag"]) for r in rows], self.num_stills,
                                                "view_balanced", rng)]
        K, (Ls, D) = self.num_stills, self.cache.shape
        out = {"still_tokens": torch.zeros(K, Ls, D), "still_valid": torch.zeros(K, dtype=torch.bool),
               "still_tag_ids": torch.full((K,), -1, dtype=torch.long),
               "still_type_ids": torch.full((K,), -1, dtype=torch.long),
               "still_cond": torch.zeros(K, max(1, len(self.cond_columns))),
               "still_bias": torch.zeros(K, Ls),
               "still_y": torch.zeros(K, max(1, len(self.target_columns))),
               "still_y_mask": torch.zeros(K, max(1, len(self.target_columns)), dtype=torch.bool),
               "demo": self.demo.get(study_id, torch.tensor([0.0, 0.0, 0.0, 1.0]))}
        for slot, r in enumerate(chosen):
            out["still_tokens"][slot] = self.cache.load(r["still_id"])
            out["still_valid"][slot] = True
            out["still_tag_ids"][slot] = self.routing.still_tag_index(r["tag"]) if r["tag"] else -1
            out["still_type_ids"][slot] = self.routing.still_type_index(r["type"]) if r["type"] else -1
            if r["cond"] is not None:
                out["still_cond"][slot] = r["cond"]
            if r["y"] is not None:
                out["still_y"][slot], out["still_y_mask"][slot] = r["y"], r["present"]
            if r["heatmap"]:        # one spatial map, the same at every temporal position ((t, h, w) order)
                out["still_bias"][slot] = heatmap_bias(np.load(r["heatmap"]), self.grid[1]).repeat(self.grid[0])
        out["still_eligible"] = self.routing.still_eligibility(out["still_tag_ids"][None], out["still_valid"][None],
                                                               self.structures)[0]
        return out


STILL_ITEM_KEYS = ("still_tokens", "still_valid", "still_tag_ids", "still_type_ids", "still_cond", "still_bias",
                   "still_y", "still_y_mask", "demo", "still_eligible")
