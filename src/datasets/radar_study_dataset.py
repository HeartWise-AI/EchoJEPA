# src/datasets/radar_study_dataset.py

"""Study items for RADAR structure alignment, read from cached frozen tokens.

Inputs:

- `clips`: one row per clip of the final eligible manifest of one split, with `study_id`, `clip_id`, `view` (a view
  class name) and `acquisition` (for example `bmode`). Identifiers are text.
- `captions`: one row per (study, structure) that has a training caption: `study_id`, `structure`, `caption`,
  `caption_id` (signed 64-bit, from the caption builder) and `normality` (normal, abnormal or unknown). A missing
  row means no caption; rows of untrained structures are ignored. The builder's provenance columns are not read here.
- `routing` (`src.models.radar.routing`) and a `TokenCache` holding every clip.

Each item is one study: at most `num_clips` of its clips, chosen by `src.datasets.study_bags`, which the matched
baseline shares: view-balanced (round robin over the study's view families in a random order, a random clip of each
family per round) or at random. A view's family comes from the routing registry's `view_families`, the only place that
map lives, and is the view class itself when the map does not name it: A4C_LV and A4C_ZOOM count as A4C, so a small
budget covers more of the heart, while PSAX_AV stays apart from PSAX as the pulmonic valve's only view. Families only
steer the choice: routing eligibility uses each clip's own view class.
The choice depends only on (seed, epoch, study) in training and on (eval_seed, eval_bag_index, study) at evaluation:
never on the rank, the worker, or the clip table's row order (a study's clips are sorted by clip id first), so a global
batch holds the same clips however many processes read it. Slots beyond the study's clips are padded with zeros.

`StudyBatchSampler` cuts one seeded permutation of the studies into global batches (dropping the last partial one)
and gives each rank every `world_size`-th study of each global batch: the union over ranks of step k's batches is
the single-process batch k. `collate` stacks items and tokenizes every (study, structure) caption slot, a missing
caption as the empty string.

Doppler stills (optional, `stills`: a `src.datasets.radar_stills.StillSource`): each item also holds the study's
still stream, a study with stills but no clip becomes an item of its own (its B-mode structures have no input), and a
caption table with a `scope` column (`build_radar_captions.py --doppler-captions`) gives a structure its `full`
caption (B-mode and Doppler statements) when the study's bag holds a still of it, its `bmode` caption otherwise.
Without `stills`, `full` rows are never read and items are those of v0.
"""

import hashlib
import json
import os
import random

import pandas as pd
import torch

from src.datasets.radar_stills import STILL_ITEM_KEYS
from src.datasets.study_bags import bag_record, check_study_ids, choose_bag, evaluation_rng, training_rng

NORMALITY = ("normal", "abnormal", "unknown")


class RadarStudyDataset(torch.utils.data.Dataset):

    def __init__(self, clips, captions, routing, cache, num_clips=12, selection="view_balanced", training=True,
                 seed=0, eval_seed=0, eval_bag_index=0, acquisition_classes=("bmode", "color_flow"),
                 normal_fraction=1.0, stills=None):
        if not 0.0 < normal_fraction <= 1.0:
            raise ValueError(f"normal_fraction must be in (0, 1], got {normal_fraction}.")
        self.normal_fraction = float(normal_fraction)
        self.routing = routing
        self.structures = [s.name for s in routing.trained]
        self.cache = cache
        self.num_clips = int(num_clips)
        self.selection = selection
        self.training = training
        self.seed = seed
        self.eval_seed = eval_seed
        self.eval_bag_index = eval_bag_index
        self.epoch = 0
        self.acquisition_classes = tuple(acquisition_classes)
        for column in ("study_id", "clip_id", "view", "acquisition"):
            if column not in clips.columns:
                raise ValueError(f"clips needs a {column!r} column.")
        check_views(clips, routing)
        check_study_ids(clips.study_id)
        missing = [c for c in clips.clip_id.astype(str) if c not in cache]
        if missing:
            raise ValueError(f"{len(missing)} clips are not in the token cache.")
        if clips.clip_id.astype(str).duplicated().any():
            raise ValueError("A clip appears twice in the clip table.")
        self.stills = stills
        by_study = {study_id: sorted(zip(rows.clip_id.astype(str), rows.view.astype(str),
                                         rows.acquisition.astype(str)))
                    for study_id, rows in clips.groupby(clips.study_id.astype(str), sort=True)}
        if stills is not None:
            for study_id in stills.studies - set(by_study):
                by_study[study_id] = []
        self.studies = sorted(by_study.items())
        known = {s for s, _ in self.studies}
        captions = captions[captions.structure.isin(self.structures)]
        if "scope" not in captions.columns:
            captions = captions.assign(scope="bmode")
        bad_scope = set(captions.scope) - {"bmode", "full"}
        if bad_scope:
            raise ValueError(f"scope must be bmode or full, got {sorted(bad_scope)}.")
        if stills is None:
            captions = captions[captions.scope == "bmode"]
        if captions.duplicated(["study_id", "structure", "scope"]).any():
            raise ValueError("A (study, structure) has more than one caption.")
        unknown = set(captions.study_id.astype(str)) - known
        if unknown:
            raise ValueError(f"{len(unknown)} captioned studies have no clip in this split.")
        bad = set(captions.normality) - set(NORMALITY)
        if bad:
            raise ValueError(f"normality must be one of {NORMALITY}, got {sorted(bad)}.")
        if captions.caption.map(lambda v: not isinstance(v, str) or not v.strip()).any():
            raise ValueError("A caption row is empty; leave the row out instead.")
        self.captions = {(str(r.study_id), r.structure, r.scope): (r.caption, int(r.caption_id), r.normality)
                         for r in captions.itertuples()}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _left_out(self, study_id, structure, normality):
        """Whether a normal caption sits out this epoch (training only). The draw depends on (seed, epoch, study,
        structure) alone, so every rank and every worker makes the same choice; abnormal and unknown captions are
        never left out. Only the caption sits out: the study's clips and its other structures are unchanged."""
        if not self.training or self.normal_fraction >= 1.0 or normality != "normal":
            return False
        return random.Random(f"normal:{self.seed}:{self.epoch}:{study_id}:{structure}").random() >= self.normal_fraction

    def __len__(self):
        return len(self.studies)

    def _rng(self, study_id):
        if self.training:
            return training_rng(self.seed, self.epoch, study_id)
        return evaluation_rng(self.eval_seed, self.eval_bag_index, study_id)

    def bag_record(self):
        """What fixes this split's bags (`study_bags.bag_record`)."""
        candidates = [(study, clip, view) for study, clips in self.studies for clip, view, _ in clips]
        return bag_record(candidates, self.routing.sha256, self.selection, self.num_clips,
                          self.seed if self.training else self.eval_seed)

    def __getitem__(self, index):
        study_id, clips = self.studies[index]
        by_id = {clip[0]: clip for clip in clips}
        rng = self._rng(study_id)
        chosen = [by_id[c] for c in choose_bag([(c, v) for c, v, _ in clips], self.num_clips, self.selection,
                                               rng, self.routing.view_families)]
        C, S = self.num_clips, len(self.structures)
        L, D = self.cache.shape
        tokens = torch.zeros(C, L, D)
        clip_valid = torch.zeros(C, dtype=torch.bool)
        view_ids = torch.full((C,), -1, dtype=torch.long)
        acquisition_ids = torch.full((C,), -1, dtype=torch.long)
        for slot, (clip_id, view, acquisition) in enumerate(chosen):
            tokens[slot] = self.cache.load(clip_id)
            clip_valid[slot] = True
            view_ids[slot] = self.routing.view_index(view)
            acquisition_ids[slot] = (self.acquisition_classes.index(acquisition)
                                     if acquisition in self.acquisition_classes else -1)
        eligible = self.routing.clip_eligibility(view_ids[None], clip_valid[None], self.structures)[0]
        still = self.stills.item(study_id, rng) if self.stills is not None else {}
        has_still = still["still_eligible"].any(dim=-1) if still else torch.zeros(S, dtype=torch.bool)
        texts, ids = [""] * S, torch.zeros(S, dtype=torch.long)
        has_caption = torch.zeros(S, dtype=torch.bool)
        normal, abnormal = torch.zeros(S, dtype=torch.bool), torch.zeros(S, dtype=torch.bool)
        for s, name in enumerate(self.structures):
            entry = self.captions.get((study_id, name, "full")) if has_still[s] else None
            entry = entry or self.captions.get((study_id, name, "bmode"))
            if entry is not None and self._left_out(study_id, name, entry[2]):
                entry = None
            if entry is not None:
                texts[s], ids[s] = entry[0], entry[1]
                has_caption[s] = True
                normal[s], abnormal[s] = entry[2] == "normal", entry[2] == "abnormal"
        return {"study_id": study_id, "clip_ids": [c for c, _, _ in chosen], "tokens": tokens,
                "clip_valid": clip_valid, "view_ids": view_ids, "acquisition_ids": acquisition_ids,
                "eligible": eligible, "captions": texts, "caption_ids": ids, "has_caption": has_caption,
                "normal": normal, "abnormal": abnormal, **still}


def collate(items, tokenizer):
    """Stack study items; tokenize every caption slot into `text_ids` / `text_mask` [B, S, length]."""
    batch = {key: torch.stack([item[key] for item in items])
             for key in ("tokens", "clip_valid", "view_ids", "acquisition_ids", "eligible", "caption_ids",
                         "has_caption", "normal", "abnormal")}
    for key in STILL_ITEM_KEYS:
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    batch["study_id"] = [item["study_id"] for item in items]
    batch["clip_ids"] = [item["clip_ids"] for item in items]
    B, S = batch["caption_ids"].shape
    ids, mask = tokenizer([text for item in items for text in item["captions"]])
    batch["text_ids"] = ids.reshape(B, S, -1)
    batch["text_mask"] = mask.reshape(B, S, -1)
    return batch


class StudyBatchSampler(torch.utils.data.Sampler):
    """Global batches of studies from one seeded permutation; each rank reads its share of every global batch."""

    def __init__(self, n_studies, global_batch, rank=0, world_size=1, seed=0, shuffle=True):
        if global_batch % world_size:
            raise ValueError("global_batch must be a multiple of world_size.")
        if n_studies < global_batch:
            raise ValueError(f"{n_studies} studies do not fill one global batch of {global_batch}.")
        self.n_studies, self.global_batch = n_studies, global_batch
        self.rank, self.world_size = rank, world_size
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_studies // self.global_batch

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n_studies, generator=torch.Generator().manual_seed(self.seed + self.epoch))
            order = order.tolist()
        else:
            order = list(range(self.n_studies))
        for k in range(len(self)):
            yield order[k * self.global_batch:(k + 1) * self.global_batch][self.rank::self.world_size]


def read_table(path, id_columns=("study_id", "clip_id")):
    """A CSV with its identifier columns as text, and every cell as written: `NA` in a caption stays those letters."""
    return pd.read_csv(path, dtype={c: str for c in id_columns}, keep_default_na=False)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_views(clips, routing):
    """Refuse a clip table with a blank view or a view the registry does not know: such a clip is eligible for no
    structure, so a misspelled view would silently take clips from every structure it should reach."""
    unknown = sorted(set(clips.view.fillna("").astype(str)) - set(routing.view_classes))
    if unknown:
        raise ValueError(f"The clip table has views the routing registry does not know: {unknown}.")


def check_caption_build(captions_path, routing, clips_path, split):
    """`build_info.json` of the caption table, after checking that the table is the file the builder wrote, for this
    registry, clip table and split. Readers need this: nothing in the table itself says what it was built from."""
    info_path = os.path.join(os.path.dirname(os.path.abspath(captions_path)), "build_info.json")
    if not os.path.exists(info_path):
        raise ValueError(f"No build_info.json beside {captions_path}: rebuild the captions.")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    problems = [what for what, same in (
        ("the file has changed since it was built", info.get("captions_sha256") == file_sha256(captions_path)),
        ("another routing registry", info.get("routing_sha256") == routing.sha256),
        ("another clip table", info.get("clips_sha256") == file_sha256(clips_path)),
        (f"another split than {split!r}", info.get("split") == split)) if not same]
    if problems:
        raise ValueError(f"The caption table does not belong to this run: {'; '.join(problems)}. Rebuild it.")
    return info
