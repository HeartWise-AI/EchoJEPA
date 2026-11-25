"""RADAR training data: clip records -> (sampled clip, caption, grouping metadata).

Caption supervision is selected explicitly:
  - by default only `approved_caption` (derived from returned review data) is used;
  - unreviewed candidate captions are used only with the development flag `allow_unreviewed_candidates`, and every
    item then records `caption_source = "unvalidated_candidate"` so provenance can count them.

Split guard: a records file used for training must not contain test-split records at all (final RADAR test patients
must not reach prompt tuning, training or model selection); the whole file must keep each patient and study in one
split from one patient split table (`records.split_problems`); validation needs records bound to the shared archive
patient split (`split_source`); and validation-split use is returned for logging.

Clips are sampled by `VideoDataset` itself (same frame-rate/stride logic, padding policy and clip indices as JEPA
pretraining) through a temporary manifest; its retry-on-failure behaviour is bypassed because a substitute video
would silently pair the wrong clip with a caption.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import torch

from app.radar.loss import normalize_caption
from app.radar.records import ClipRecord, caption_concept_key, split_problems
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset

CAPTION_APPROVED = "approved"
CAPTION_UNVALIDATED = "unvalidated_candidate"


class SplitGuardError(ValueError):
    """Records from a split that this use may not read."""


@dataclass(frozen=True)
class CaptionedClip:
    record_id: str
    video_path: str
    caption: str
    caption_source: str
    study_key: str  # grouping only (loss mask); never a model input
    caption_key: str
    concept_key: tuple
    view: str
    acquisition: str


def select_records(records: list[ClipRecord], *, split: str, purpose: str, allow_unreviewed_candidates: bool = False,
                   split_source: str | None = None) -> tuple[list[CaptionedClip], dict]:
    """The captioned clips of one split for one purpose, and counts for provenance.

    `split_source` is the SHA-256 of the archive patient split table the run is bound to; every record must carry it.
    Validation (`split="val"`) needs bound records, so its patients are the shared split's validation patients."""
    if split == "test":
        raise SplitGuardError("RADAR training and model selection never read test-split records.")
    if split not in ("train", "val"):
        raise ValueError(f"Unknown split {split!r}.")
    n_test = sum(r.split == "test" for r in records)
    if n_test:
        raise SplitGuardError(f"The records file holds {n_test} test-split records; build the training records "
                              "without them so test reports and captions cannot reach training or selection.")
    problems = split_problems(records)
    if problems:
        raise SplitGuardError("The records do not separate patients by split: " + "; ".join(problems) + ".")
    sources = {r.split_source for r in records}
    if split_source is not None and sources != {split_source}:
        raise SplitGuardError("The records were not split with the expected patient split table (split_source).")
    if split == "val" and None in sources:
        raise SplitGuardError("Validation needs records bound to the shared archive patient split (split_source): "
                              "rebuild them with the archive's patient_splits.csv.")
    items, approved, candidates, skipped = [], 0, 0, 0
    for r in records:
        if r.split != split:
            continue
        if r.approved_caption is not None:
            text, source = r.approved_caption.text, CAPTION_APPROVED
            approved += 1
        elif allow_unreviewed_candidates and r.candidate_caption is not None:
            text, source = r.candidate_caption.text, CAPTION_UNVALIDATED
            candidates += 1
        else:
            skipped += 1
            continue
        items.append(CaptionedClip(
            record_id=r.record_id, video_path=r.video_path, caption=text, caption_source=source,
            study_key=r.study_key, caption_key=normalize_caption(text), concept_key=caption_concept_key(r),
            view=r.view, acquisition=r.acquisition.category,
        ))
    stats = {
        "split": split, "purpose": purpose, "split_source": next(iter(sources)) if records else None,
        "records_in_split": approved + candidates + skipped, "used": len(items),
        "approved_captions": approved, "unvalidated_candidate_captions": candidates,
        "skipped_without_approved_caption": skipped, "allow_unreviewed_candidates": bool(allow_unreviewed_candidates),
    }
    return items, stats


class RadarClipDataset(torch.utils.data.Dataset):
    """One sampled clip per captioned record, with its caption and grouping keys."""

    def __init__(self, items: list[CaptionedClip], *, frames_per_clip: int, resolution: int, fps: float | None = None,
                 frame_step: int | None = None, temporal_padding: str | None = None, random_clip_sampling: bool = True,
                 seed: int = 0, video_dataset_kwargs: dict | None = None):
        if not items:
            raise ValueError("No captioned clips to load.")
        if any(any(c.isspace() for c in item.video_path) for item in items):
            raise ValueError("Video paths with whitespace cannot go through the VideoDataset manifest format.")
        self.items = list(items)
        self.seed = seed
        self.epoch = 0
        kwargs = dict(video_dataset_kwargs or {})
        if temporal_padding is not None:
            if "temporal_padding" not in inspect.signature(VideoDataset).parameters:
                raise ValueError("This VideoDataset has no temporal_padding option.")
            kwargs["temporal_padding"] = temporal_padding
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "radar_clips.csv")
            with open(manifest, "w") as f:
                f.writelines(f"{item.video_path} 0\n" for item in self.items)
            self.videos = VideoDataset(
                data_paths=[manifest], frames_per_clip=frames_per_clip, fps=fps,
                frame_step=None if fps is not None else frame_step, num_clips=1,
                transform=make_transforms(training=False, crop_size=resolution),
                random_clip_sampling=random_clip_sampling, **kwargs,
            )
        if len(self.videos) != len(self.items):
            raise RuntimeError("The clip manifest did not round-trip.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        # Clip sampling draws from numpy's global generator; seed it per (epoch, item) so a resumed run samples the
        # same clips as an uninterrupted one, whatever worker serves the item.
        np.random.seed((self.seed * 1_000_003 + self.epoch * 10_007 + index) % 2**32)
        loaded = self.videos.get_item_video(index)
        if not loaded:
            raise RuntimeError(f"Clip record {item.record_id} could not be decoded.")
        buffer, _, clip_indices = loaded
        video = buffer[0][0] if isinstance(buffer[0], (list, tuple)) else buffer[0]
        return {
            "video": video,
            "clip_indices": torch.as_tensor(np.asarray(clip_indices[0]), dtype=torch.long),
            "caption": item.caption,
            "caption_source": item.caption_source,
            "record_id": item.record_id,
            "study_key": item.study_key,
            "caption_key": item.caption_key,
            "concept_key": item.concept_key,
        }


def collate(batch: list[dict]) -> dict:
    out = {"video": torch.stack([b["video"] for b in batch]),
           "clip_indices": torch.stack([b["clip_indices"] for b in batch])}
    for key in ("caption", "caption_source", "record_id", "study_key", "caption_key", "concept_key"):
        out[key] = [b[key] for b in batch]
    return out
