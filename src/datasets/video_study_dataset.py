# src/datasets/video_study_dataset.py

"""One item per study: a bag of its videos, DeepCORO_CLIP style.

The manifest is a long CSV with one row per video (header required): a video path column, a
group column (the study), the target columns, and optionally a view column and a split column.
Each item returns a fixed number of video slots (`num_videos`), each holding
`num_clips_per_video` clips of `frames_per_clip` frames:

- training with `shuffle_videos`: a random subset of the study's videos, in random order, drawn
  again every time the study is read, so an epoch sees another bag of the same study;
- evaluation: deterministic random bags, seeded by study id, eval_seed and bag index;
- `sampling_mode: view_balanced`: round-robin draws across the available view families;
- `sampling_mode: fixed` (or `shuffle_videos: false` without a mode): legacy path-order bags.

A study with fewer videos than slots is padded with black clips; `video_mask` says which slots are
real. A video that fails to load is skipped and its slot padded; a study whose selected videos all
fail returns `None`. Targets are study-level: every row of a study must agree (missing values are
allowed and masked in the loss). With a `view_column`, `views` keeps only some views and
`view_labels_map` gives each slot its view class (for the probe's view embedding).
"""

import hashlib
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
    "seeded_random": "a random bag seeded by eval_seed, bag index and study identity",
    "view_balanced": "round-robin sampling across available view families; seeded during evaluation",
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
    if (views or view_labels_map) and not (view_column and view_column in table.columns):
        raise ValueError("`views` and `view_labels_map` need `experiment.data.view_column`, a column of the manifest.")

    # Study-level targets, checked over every row of the study (the view filter comes after, so a
    # conflicting value on a filtered-out video is still refused).
    targets = table[list(target_columns)].copy()
    for column in target_columns:
        mapping = (labels_map or {}).get(column)
        values = targets[column]
        if not pd.api.types.is_numeric_dtype(values):  # object (pandas 2) or str (pandas 3) labels
            if not mapping:
                raise ValueError(f"Target `{column}` holds strings: map them to integers with `labels_map.{column}`.")
            unknown = sorted(set(values.dropna().astype(str)) - set(map(str, mapping)))
            if unknown:
                raise ValueError(f"Target `{column}` has labels {unknown} missing from `labels_map.{column}`.")
            values = values.map(lambda v: np.nan if pd.isna(v) else float(mapping[str(v)]))
        targets[column] = pd.to_numeric(values, errors="raise").astype(float)
    study_targets = {}
    for study_id, index in table.groupby(group_column, sort=True).groups.items():
        values = []
        for column in target_columns:
            distinct = targets.loc[index, column].dropna().unique()
            if len(distinct) > 1:
                raise ValueError(f"Study {study_id!r} has {len(distinct)} different values of `{column}`; "
                                 "targets are per study.")
            values.append(float(distinct[0]) if len(distinct) else np.nan)
        study_targets[study_id] = np.asarray(values, dtype=np.float64)

    rows = table
    if views:
        rows = rows[rows[view_column].astype(str).isin([str(v) for v in views])]
    if rows.empty:
        raise ValueError("The manifest keeps no video after the view filter.")
    classes = pd.Series(view_ids(rows[view_column].tolist(), view_labels_map) if view_labels_map
                        else [-1] * len(rows), index=rows.index)

    studies, dropped = [], 0
    for study_id, index in rows.groupby(group_column, sort=True).groups.items():
        videos = sorted(zip((str(p) for p in rows.loc[index, path_column]), classes.loc[index].tolist()))
        paths = [p for p, _ in videos]
        if len(paths) != len(set(paths)):
            raise ValueError(f"A study lists the same video more than once (study {study_id!r}).")
        values = study_targets[study_id]
        if len(paths) < min_videos:
            dropped += 1
            continue
        studies.append((study_id, paths, values, [int(c) for _, c in videos]))
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
        sampling_mode=None,
        eval_seed=0,
        eval_bag_index=0,
        view_family_map=None,
        view_probability_columns=None,
        acquisition_column=None,
        acquisition_labels_map=None,
        patient_column="patient_id",
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
        mode = sampling_mode or ("random" if shuffle_videos else "fixed")
        if mode not in ("random", "view_balanced", "fixed"):
            raise ValueError("sampling_mode must be random, view_balanced or fixed.")
        self.selection = "seeded_random" if mode == "random" and not self.training else mode
        self.eval_seed, self.eval_bag_index = int(eval_seed), int(eval_bag_index)
        if mode == "view_balanced" and (not view_column or view_column not in table):
            raise ValueError("view_balanced sampling requires view_column in the manifest.")
        self.patient_ids = None
        if patient_column and patient_column in table:
            patients = table.groupby(group_column)[patient_column]
            if table[patient_column].isna().any() or (patients.nunique() != 1).any():
                raise ValueError("Each study must have exactly one nonmissing patient id.")
            self.patient_ids = set(patients.first().loc[self.study_ids].astype(str))
        # Metadata follows the sorted video paths, independently of sampled slot order.
        self.video_metadata = {}
        self.view_probability_columns = view_probability_columns
        self.acquisition_column = acquisition_column
        probability_columns = None
        if view_probability_columns:
            if not view_labels_map:
                raise ValueError("view_probability_columns needs view_labels_map.")
            by_class = {}
            for label, column in view_probability_columns.items():
                if label not in view_labels_map or view_labels_map[label] in by_class:
                    raise ValueError("view_probability_columns must name one label per unique view class.")
                by_class[view_labels_map[label]] = column
            if set(by_class) != set(view_labels_map.values()):
                raise ValueError("view_probability_columns must cover every view class.")
            probability_columns = [by_class[c] for c in range(len(by_class))]
            if any(c not in table for c in probability_columns):
                raise ValueError("A view probability column is missing from the manifest.")
        if bool(acquisition_column) != bool(acquisition_labels_map):
            raise ValueError("Set both acquisition_column and acquisition_labels_map.")
        if acquisition_column and acquisition_column not in table:
            raise ValueError("acquisition_column is missing from the manifest.")
        acquisition = (view_ids(table[acquisition_column].tolist(), acquisition_labels_map)
                       if acquisition_column else [-1] * len(table))
        for (_, row), acq in zip(table.iterrows(), acquisition):
            label = str(row[view_column]) if view_column and view_column in table else "Other"
            metadata = {"family": (view_family_map or {}).get(label, label), "acquisition_ids": acq}
            if probability_columns:
                probabilities = row[probability_columns].to_numpy(dtype=float)
                if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                        or not np.isclose(probabilities.sum(), 1, atol=1e-4)):
                    raise ValueError("View probabilities must be finite, nonnegative and sum to one on every row.")
                metadata["view_probs"] = probabilities.astype(np.float32)
            self.video_metadata[(row[group_column], str(row[path_column]))] = metadata
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

    def select(self, paths, families=None, study_id=None):
        """The positions of the videos read for one item, as `SELECTION[self.selection]` says."""
        rng = random
        if not self.training:
            identity = str(study_id) if study_id is not None else "\0".join(paths)
            digest = hashlib.sha256(f"{self.eval_seed}:{self.eval_bag_index}:{identity}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
        if self.selection in ("random", "seeded_random"):
            chosen = rng.sample(range(len(paths)), min(self.num_videos, len(paths)))
            return chosen
        if self.selection == "view_balanced":
            if families is None or len(families) != len(paths):
                raise ValueError("view_balanced sampling needs the family of every candidate video.")
            buckets = {}
            for index, family in enumerate(families):
                buckets.setdefault(family, []).append(index)
            for bucket in buckets.values():
                rng.shuffle(bucket)
            chosen = []
            while len(chosen) < min(self.num_videos, len(paths)):
                active = [family for family, bucket in buckets.items() if bucket]
                rng.shuffle(active)
                for family in active:
                    chosen.append(buckets[family].pop())
                    if len(chosen) == min(self.num_videos, len(paths)):
                        break
            rng.shuffle(chosen)
            return chosen
        return list(range(min(self.num_videos, len(paths))))

    MAX_RETRIES = 20  # studies tried in training before an unreadable split stops the run.

    def __getitem__(self, index):
        # Training: a study that cannot be read is replaced by another, as `VideoDataset` does,
        # up to `MAX_RETRIES` times, so an unreadable split stops the run instead of hanging it.
        for attempt in range(self.MAX_RETRIES + 1):
            item = self.get_item_study(index)
            if item is not None:
                return item
            if not self.training:
                raise RuntimeError(f"Study {index} could not be read.")
            logger.warning(f"Retrying with another study: none of the selected videos of study {index} loaded.")
            index = np.random.randint(len(self))
        raise RuntimeError(f"None of {self.MAX_RETRIES + 1} studies drawn could be read: check the video paths.")

    def get_item_study(self, index):
        """(clips [slots*clips, C, T, H, W], targets [targets], video_mask [slots], clip_indices
        [slots*clips, T], view_ids [slots], optional metadata dict) for the study at `index`, or `None` when none of its
        selected videos loads. `view_ids` is -1 for a padded slot, and for every slot without a
        `view_labels_map`."""
        study_id, paths, targets, classes = self.studies[index]
        clips, indices, mask, views = [], [], [], []
        metadata = [self.video_metadata[(study_id, path)] for path in paths]
        probabilities, acquisition = [], []
        for position in self.select(paths, [m["family"] for m in metadata], study_id):
            loaded = self.load_video(paths[position])
            if loaded is None:
                logger.warning(f"Skipping a video of study {index} that failed to load; its slot is padded.")
                continue
            video_clips, video_indices = loaded
            clips += video_clips
            indices += video_indices
            mask.append(True)
            views.append(classes[position])
            if self.view_probability_columns:
                probabilities.append(metadata[position]["view_probs"])
            acquisition.append(metadata[position]["acquisition_ids"])
        if not mask:
            return None
        pad = torch.zeros_like(clips[0])
        while len(mask) < self.num_videos:
            clips += [pad.clone() for _ in range(self.num_clips_per_video)]
            indices += [np.full(self.frames_per_clip, -1, dtype=np.int64) for _ in range(self.num_clips_per_video)]
            mask.append(False)
            views.append(-1)
            acquisition.append(-1)
            if self.view_probability_columns:
                probabilities.append(np.zeros_like(probabilities[0]))
        item = (
            torch.stack(clips),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.from_numpy(np.stack(indices)),
            torch.tensor(views, dtype=torch.long),
        )
        extra = {}
        if self.view_probability_columns:
            extra["view_probs"] = torch.from_numpy(np.stack(probabilities))
        if self.acquisition_column:
            extra["acquisition_ids"] = torch.tensor(acquisition, dtype=torch.long)
        return (*item, extra) if extra else item

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
