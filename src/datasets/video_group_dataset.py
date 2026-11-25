# src/datasets/video_group_dataset.py

"""One sample per study for the multi-view frozen probe: the videos of a study's view slots.

A manifest (`data/make_rvsp_probe_manifests.py`) is a CSV with a header: `study_id`, one column per
slot position named `<slot>[<k>]` (the k-th video of the slot) holding a video path or `MISS`, and
`label`. A missing position is never loaded and never replaced by a placeholder video: the sample
says which positions hold a video (`present`), and the probe masks the others out. A listed video
that cannot be read is treated as missing and counted.

Each video is read by `VideoDataset`'s loader with its sampling and padding (`num_segments` clips of
`frames_per_clip` frames, black padding of short videos at clip index -1 by default), so a video is
read exactly as the single-view probe reads it.

This replaces the upstream `VideoGroupDataset` (headerless rows, integer labels, black dummy clips
standing in for missing views).
"""

import re
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch

from src.datasets.video_dataset import DEFAULT_MAX_REPLACEMENTS, VideoDataset, validate_max_replacements

logger = getLogger()

MISSING_TOKEN = "MISS"
POSITION = re.compile(r"^(?P<slot>[A-Za-z0-9_]+)\[(?P<clip>\d+)\]$")


def read_group_manifest(path):
    """Read a per-study manifest. Returns (frame, layout): the rows, and the slot positions in column
    order as (slot name, clip index within the slot) pairs.

    Refuses a manifest without `study_id` or `label`, a column that is not a slot position, a slot
    whose clip indices are not 0..K-1, a study listed twice, a label that is not a finite number, a
    row without any video, and a video listed twice in a row.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = list(frame.columns)
    if columns[:1] != ["study_id"] or columns[-1:] != ["label"] or len(columns) < 3:
        raise ValueError("A group manifest's columns are study_id, the slot positions `<slot>[<k>]`, then label.")
    layout = []
    for name in columns[1:-1]:
        match = POSITION.match(name)
        if not match:
            raise ValueError(f"{name!r} is not a slot position `<slot>[<k>]`.")
        layout.append((match["slot"], int(match["clip"])))
    slots = list(dict.fromkeys(slot for slot, _ in layout))
    for slot in slots:
        clips = [k for s, k in layout if s == slot]
        if clips != list(range(len(clips))):
            raise ValueError(f"Slot {slot!r} has the clip positions {clips}, not 0..{len(clips) - 1} in order.")
    names = [slot for slot, _ in layout]
    if names != [slot for slot in slots for _ in range(names.count(slot))]:
        raise ValueError("A slot's positions must be adjacent columns.")
    if (frame.study_id.str.strip() == "").any() or frame.study_id.duplicated().any():
        raise ValueError("A group manifest lists a study twice or without an ID.")
    labels = pd.to_numeric(frame.label, errors="coerce")
    if not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise ValueError(f"{int((~np.isfinite(labels.to_numpy(dtype=float))).sum())} labels are not finite numbers.")
    paths = frame[columns[1:-1]].apply(lambda c: c.str.strip())
    empty = (paths == "").to_numpy()
    if empty.any():
        raise ValueError(f"{int(empty.sum())} slot positions are empty: write `{MISSING_TOKEN}` for a missing video.")
    listed = (paths != MISSING_TOKEN).to_numpy()
    if not listed.any(axis=1).all():
        raise ValueError(f"{int((~listed.any(axis=1)).sum())} studies have no video in any slot.")
    for row, keep in zip(paths.to_numpy(), listed):
        if len(set(row[keep])) != int(keep.sum()):
            raise ValueError("A study lists the same video in two slot positions.")
    frame = frame.assign(label=labels.astype(float))
    frame[columns[1:-1]] = paths
    return frame, layout


class VideoGroupDataset(torch.utils.data.Dataset):
    """The studies of a per-study manifest. `get_item_study(i)` returns a dict:

    - `videos`: one entry per slot position, `None` when the position has no video (missing or
      unreadable), else (clips, clip_indices): `num_segments` clips [C, T, H, W] of the video and their
      source frame indices.
    - `present`: bool tensor [positions], True where a video was read.
    - `listed`: bool tensor [positions], True where the manifest lists a video.
    - `label`: the study's (z-scored) label. `index`: the row.

    It returns `None` when none of the study's videos can be read. `__getitem__`, used in training,
    then draws another study, as `VideoDataset` does (the substitute carries its own label and row, so
    videos and label still match), and raises `RuntimeError` once `max_replacements` substitutes in a
    row could not be read either. Evaluation calls `get_item_study` directly so a failed study is
    counted, not replaced.
    """

    def __init__(
        self,
        manifest,
        frames_per_clip=16,
        frame_step=4,
        num_segments=1,
        transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        temporal_padding=None,
        max_replacements=DEFAULT_MAX_REPLACEMENTS,
        fps=None,
        fractional_frame_step=False,
    ):
        self.max_replacements = validate_max_replacements(max_replacements)
        self.manifest = manifest
        self.frame, self.layout = read_group_manifest(manifest)
        self.position_names = list(self.frame.columns[1:-1])
        self.studies = self.frame.study_id.tolist()
        self.labels = self.frame.label.tolist()
        self.paths = [[None if p == MISSING_TOKEN else p for p in row]
                      for row in self.frame[self.position_names].to_numpy()]
        self.frames_per_clip = frames_per_clip
        self.num_segments = num_segments
        self.transform = transform
        # The single-view loader reads every video, so sampling and padding match the EF probe's.
        options = {} if temporal_padding is None else {"temporal_padding": temporal_padding}
        if fractional_frame_step:       # exact-rate clips (src/datasets/utils/frame_sampling.py); off by default
            options["fractional_frame_step"] = True
        self.reader = VideoDataset(
            data_paths=[],
            frames_per_clip=frames_per_clip,
            frame_step=None if fps is not None else frame_step,     # `fps` replaces `frame_step`
            fps=fps,
            num_clips=num_segments,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            **options,
        )
        logger.info(f"Loaded {len(self.studies)} studies with {len(self.layout)} slot positions")

    @property
    def samples(self):
        """Every video the manifest lists, row by row (the split check against the video index)."""
        return [p for row in self.paths for p in row if p is not None]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        # At most `max_replacements` substitutes, so a broken storage raises instead of spinning forever.
        requested, replacements = index, 0
        while True:
            item = self.get_item_study(index)
            if item is not None:
                return item
            if replacements == self.max_replacements:
                tried = f"study row {requested}"
                if replacements:
                    tried += f" and {replacements} random substitute studies (the last: row {index})"
                raise RuntimeError(f"No video of {tried} could be read; max_replacements={self.max_replacements} "
                                   "reached, so the videos are probably unreachable or the manifest is wrong.")
            replacements += 1
            warnings.warn(f"Retrying with another study: no video of study row {index} could be read.")
            index = np.random.randint(len(self))

    def load_video(self, path):
        """(clips, clip_indices) of one video, as `VideoDataset.get_item_video` reads it, or `None`."""
        buffer, clip_indices = self.reader.loadvideo_decord(path, self.frames_per_clip)
        if buffer is None or len(buffer) == 0:
            return None
        fpc = self.frames_per_clip
        clips = [buffer[i * fpc : (i + 1) * fpc] for i in range(self.num_segments)]
        if self.transform is not None:
            # One spatial view per clip: the transform returns a list of views.
            clips = [self.transform(clip)[0] for clip in clips]
        else:
            clips = [torch.from_numpy(np.ascontiguousarray(clip)) for clip in clips]
        return clips, [torch.as_tensor(np.asarray(c), dtype=torch.long) for c in clip_indices]

    def get_item_study(self, index):
        videos, listed = [], []
        for path in self.paths[index]:
            listed.append(path is not None)
            if path is None:
                videos.append(None)
                continue
            try:
                videos.append(self.load_video(path))
            except Exception as e:  # an unreadable video counts as missing; its path is not logged.
                logger.warning(f"Failed to read a video of study row {index}: {type(e).__name__}.")
                videos.append(None)
        present = [v is not None for v in videos]
        if not any(present):
            return None
        return {
            "videos": videos,
            "present": torch.tensor(present, dtype=torch.bool),
            "listed": torch.tensor(listed, dtype=torch.bool),
            "label": float(self.labels[index]),
            "index": int(index),
        }


def collate_studies(items):
    """Batch studies of `VideoGroupDataset`. The read videos of the batch are stacked in row-major
    (study, position) order, the order of `present.nonzero()`:

    - `clips`: `num_segments` tensors [videos, C, T, H, W]; `clip_indices`: `num_segments` [videos, T].
    - `present`: bool [studies, positions]; `labels`: float [studies, 1]; `index`: the rows.
    - `failed_videos`: listed videos that could not be read.
    """
    videos = [v for item in items for v in item["videos"] if v is not None]
    segments = len(videos[0][0])
    present = torch.stack([item["present"] for item in items])
    listed = torch.stack([item["listed"] for item in items])
    return {
        "clips": [torch.stack([v[0][s] for v in videos]) for s in range(segments)],
        "clip_indices": [torch.stack([v[1][s] for v in videos]) for s in range(segments)],
        "present": present,
        "labels": torch.tensor([[item["label"]] for item in items], dtype=torch.float32),
        "index": torch.tensor([item["index"] for item in items], dtype=torch.long),
        "failed_videos": int((listed & ~present).sum()),
    }


def make_videogroupdataset(**_):
    """The upstream entry point of `init_data(data="VideoGroupDataset")`, kept so that path fails
    with a clear message: the multi-view probe builds its own loaders from per-study manifests."""
    raise ValueError("The upstream headerless VideoGroupDataset format is no longer supported. Build per-study "
                     "manifests with data/make_rvsp_probe_manifests.py and run the multi-view probe "
                     "(eval_name: video_classification_frozen_multi).")
