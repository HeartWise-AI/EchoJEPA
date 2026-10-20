# src/datasets/video_study_dataset.py

"""One item per study: a bag of its videos, DeepCORO_CLIP style.

The manifest is a long CSV with one row per video (header required): a video path column, a
group column (the study), the target columns, and optionally a view column and a split column.
Each item returns a fixed number of video slots (`num_videos`), each holding
`num_clips_per_video` clips of `frames_per_clip` frames:

- training with `shuffle_videos`: a random subset of the study's videos, in random order, drawn
  again every time the study is read, so an epoch sees another bag of the same study;
- evaluation (or `shuffle_videos: false`): the first `num_videos` videos in path order, every time.

A study with fewer videos than slots is padded with black clips; `video_mask` says which slots are
real. A video that fails to load is skipped and its slot padded; a study whose selected videos all
fail returns `None`. Targets are study-level: every row of a study must agree (missing values are
allowed and masked in the loss). With a `view_column`, `views` keeps only some views and
`view_labels_map` gives each slot its view class (for the probe's view embedding).
"""

import logging
import random

import numpy as np
import pandas as pd
import torch

from src.datasets.video_dataset import VideoDataset

logger = logging.getLogger()

# How the videos of a study are chosen for one item.
SELECTION = {
    "random": "a random subset of the study's videos, in random order, redrawn at every read",
    "fixed": "the first `num_videos` videos of the study in path order, every time",
}


def read_split(args_data, split):
    """The rows of `split` from the manifests `args_data` names: `data_filename` (one CSV with a
    `split_column`) or `dataset_<split>` (one CSV per split)."""
    path = args_data.get("data_filename")
    if path:
        table = pd.read_csv(path)
        column = args_data.get("split_column", "split")
        if column not in table.columns:
            raise ValueError(f"`data_filename` has no `{column}` column; set `experiment.data.split_column`.")
        rows = table[table[column].astype(str) == split]
        if rows.empty:
            raise ValueError(f"`data_filename` has no row with {column} == {split!r}.")
        return rows.reset_index(drop=True)
    path = args_data.get(f"dataset_{split}")
    if not path:
        raise ValueError(f"Set `experiment.data.data_filename` (with a split column) or `experiment.data.dataset_{split}`.")
    return pd.read_csv(path)


def make_clip_loader(frames_per_clip, frame_step=None, duration=None, fps=None, num_clips_per_video=1,
                     random_clip_sampling=True, filter_long_videos=int(10**9)):
    """A `VideoDataset` without samples, used for its clip sampler only: `loadvideo_decord(path, fpc)`
    reads `num_clips_per_video` clips, pads short videos with black frames (clip index -1) and hides
    decoder output that could name the file."""
    return VideoDataset(
        data_paths=[],
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        duration=duration,
        fps=fps,
        num_clips=num_clips_per_video,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=True,
        filter_long_videos=filter_long_videos,
    )


def view_ids(values, view_labels_map):
    """The view class of each video: `view_labels_map` ({view: int}) applied to the view column;
    a view the map lacks takes its `Other` entry, or stops the run. Returns a list of ints."""
    mapping = {str(k): int(v) for k, v in view_labels_map.items()}
    if sorted(set(mapping.values())) != list(range(len(set(mapping.values())))):
        raise ValueError("`view_labels_map` must number the view classes 0, 1, ..., without gaps.")
    unknown = sorted({str(v) for v in values} - set(mapping))
    if unknown and "Other" not in mapping:
        raise ValueError(f"Views {unknown} are missing from `view_labels_map` (add them, or an `Other` class).")
    return [mapping.get(str(v), mapping.get("Other")) for v in values]


def group_studies(table, path_column, group_column, target_columns, view_column=None, views=None,
                  labels_map=None, min_videos=1, view_labels_map=None):
    """Group the manifest's rows into studies.

    Returns a list of (study_id, [video paths in path order], targets[float, len(target_columns)],
    [view class per video, or -1]). `views` keeps only the rows whose `view_column` is listed.
    `labels_map` maps string labels of a target column to integers ({column: {label: int}});
    `view_labels_map` maps the view column to view classes (`view_ids`). A study whose rows disagree
    on a target stops the run; a study with fewer than `min_videos` videos is left out and counted.
    """
    for column in (path_column, group_column, *target_columns):
        if column not in table.columns:
            raise ValueError(f"The manifest has no `{column}` column; its columns are {list(table.columns)}.")
    rows = table
    if (views or view_labels_map) and not (view_column and view_column in rows.columns):
        raise ValueError("`views` and `view_labels_map` need `experiment.data.view_column`, a column of the manifest.")
    if views:
        rows = rows[rows[view_column].astype(str).isin([str(v) for v in views])]
    if rows.empty:
        raise ValueError("The manifest keeps no video after the view filter.")
    classes = pd.Series(view_ids(rows[view_column].tolist(), view_labels_map) if view_labels_map
                        else [-1] * len(rows), index=rows.index)

    targets = rows[list(target_columns)].copy()
    for column in target_columns:
        mapping = (labels_map or {}).get(column)
        values = targets[column]
        if values.dtype == object:
            if not mapping:
                raise ValueError(f"Target `{column}` holds strings: map them to integers with `labels_map.{column}`.")
            unknown = sorted(set(values.dropna().astype(str)) - set(map(str, mapping)))
            if unknown:
                raise ValueError(f"Target `{column}` has labels {unknown} missing from `labels_map.{column}`.")
            values = values.map(lambda v: np.nan if pd.isna(v) else float(mapping[str(v)]))
        targets[column] = pd.to_numeric(values, errors="raise").astype(float)

    studies, dropped = [], 0
    for study_id, index in rows.groupby(group_column, sort=True).groups.items():
        videos = sorted(zip((str(p) for p in rows.loc[index, path_column]), classes.loc[index].tolist()))
        paths = [p for p, _ in videos]
        if len(paths) != len(set(paths)):
            raise ValueError(f"A study lists the same video more than once (study {study_id!r}).")
        values = []
        for column in target_columns:
            distinct = targets.loc[index, column].dropna().unique()
            if len(distinct) > 1:
                raise ValueError(f"Study {study_id!r} has {len(distinct)} different values of `{column}`; "
                                 "targets are per study.")
            values.append(float(distinct[0]) if len(distinct) else np.nan)
        if len(paths) < min_videos:
            dropped += 1
            continue
        studies.append((study_id, paths, np.asarray(values, dtype=np.float64), [int(c) for _, c in videos]))
    if dropped:
        logger.info(f"Left out {dropped} studies with fewer than {min_videos} videos.")
    if not studies:
        raise ValueError("No study has enough videos.")
    return studies


class VideoStudyDataset(torch.utils.data.Dataset):
    """A split of studies, each read as `num_videos` video slots of `num_clips_per_video` clips."""

    def __init__(
        self,
        table,
        path_column="video_path",
        group_column="study_id",
        target_columns=(),
        view_column=None,
        views=None,
        labels_map=None,
        num_videos=8,
        shuffle_videos=True,
        training=False,
        frames_per_clip=16,
        frame_step=2,
        duration=None,
        fps=None,
        num_clips_per_video=1,
        transform=None,
        min_videos=1,
        filter_long_videos=int(10**9),
        view_labels_map=None,
    ):
        if num_videos < 1:
            raise ValueError("`num_videos` must be at least 1.")
        self.studies = group_studies(table, path_column, group_column, list(target_columns), view_column, views,
                                     labels_map, min_videos, view_labels_map)
        self.target_columns = list(target_columns)
        self.num_videos = int(num_videos)
        self.num_clips_per_video = int(num_clips_per_video)
        self.frames_per_clip = int(frames_per_clip)
        self.training = bool(training)
        self.selection = "random" if self.training and shuffle_videos else "fixed"
        self.transform = transform
        self.loader = make_clip_loader(frames_per_clip, frame_step, duration, fps, num_clips_per_video,
                                       random_clip_sampling=self.training, filter_long_videos=filter_long_videos)
        logger.info(f"Loaded {len(self.studies)} studies ({sum(len(s[1]) for s in self.studies)} videos); "
                    f"{self.num_videos} slots x {self.num_clips_per_video} clips, selection: {self.selection}.")

    def __len__(self):
        return len(self.studies)

    @property
    def study_ids(self):
        return [s[0] for s in self.studies]

    @property
    def targets(self):
        """[studies, targets], NaN where missing."""
        return np.stack([s[2] for s in self.studies])

    def select(self, paths):
        """The positions of the videos read for one item, as `SELECTION[self.selection]` says."""
        if self.selection == "random":
            chosen = random.sample(range(len(paths)), min(self.num_videos, len(paths)))
            random.shuffle(chosen)
            return chosen
        return list(range(min(self.num_videos, len(paths))))

    def __getitem__(self, index):
        # Training: a study that cannot be read is replaced by another, as `VideoDataset` does.
        while True:
            item = self.get_item_study(index)
            if item is not None:
                return item
            if not self.training:
                raise RuntimeError(f"Study {index} could not be read.")
            logger.warning(f"Retrying with another study: none of the selected videos of study {index} loaded.")
            index = np.random.randint(len(self))

    def get_item_study(self, index):
        """(clips [slots*clips, C, T, H, W], targets [targets], video_mask [slots], clip_indices
        [slots*clips, T], view_ids [slots]) for the study at `index`, or `None` when none of its
        selected videos loads. `view_ids` is -1 for a padded slot, and for every slot without a
        `view_labels_map`."""
        study_id, paths, targets, classes = self.studies[index]
        clips, indices, mask, views = [], [], [], []
        for position in self.select(paths):
            loaded = self.load_video(paths[position])
            if loaded is None:
                logger.warning(f"Skipping a video of study {index} that failed to load; its slot is padded.")
                continue
            video_clips, video_indices = loaded
            clips += video_clips
            indices += video_indices
            mask.append(True)
            views.append(classes[position])
        if not mask:
            return None
        pad = torch.zeros_like(clips[0])
        while len(mask) < self.num_videos:
            clips += [pad.clone() for _ in range(self.num_clips_per_video)]
            indices += [np.full(self.frames_per_clip, -1, dtype=np.int64) for _ in range(self.num_clips_per_video)]
            mask.append(False)
            views.append(-1)
        return (
            torch.stack(clips),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.from_numpy(np.stack(indices)),
            torch.tensor(views, dtype=torch.long),
        )

    def load_video(self, path):
        """The video's `num_clips_per_video` transformed clips and their frame indices, or `None`."""
        buffer, clip_indices = self.loader.loadvideo_decord(path, self.frames_per_clip)
        if buffer is None or len(buffer) == 0:
            return None
        fpc = self.frames_per_clip
        clips = [buffer[i * fpc: (i + 1) * fpc] for i in range(self.num_clips_per_video)]
        if self.transform is not None:
            # The probe transforms return one spatial view per clip.
            clips = [self.transform(clip)[0] for clip in clips]
        else:
            clips = [torch.from_numpy(np.ascontiguousarray(clip)).permute(3, 0, 1, 2).float() / 255.0 for clip in clips]
        return clips, [np.asarray(ci, dtype=np.int64) for ci in clip_indices]
