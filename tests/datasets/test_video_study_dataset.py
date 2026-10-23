# tests/datasets/test_video_study_dataset.py

"""`VideoStudyDataset`: a long manifest grouped into studies, a random bag of a study's videos at
every training read and a fixed one in evaluation, padded slots masked, failed videos padded,
and study-level targets checked."""

import os
import random
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.datasets.video_study_dataset import VideoStudyDataset, group_studies, read_split, seeded_bag

FRAMES, SIZE = 4, 32


def write_video(path, frames=12, fps=8):
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
    for i in range(frames):
        writer.write(np.full((SIZE, SIZE, 3), (40 + 10 * i) % 256, np.uint8))
    writer.release()


class TestVideoStudyDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = tmp.name
        rows = []
        # Study s0: 4 videos (2 A4C, 2 A2C); s1: 1 video; s2: 2 videos, one unreadable.
        plan = {"s0": [("A4C", True)] * 2 + [("A2C", True)] * 2, "s1": [("A4C", True)],
                "s2": [("A4C", True), ("A4C", False)]}
        targets = {"s0": (60.0, "normal"), "s1": (35.0, "abnormal"), "s2": (np.nan, "normal")}
        for study, videos in plan.items():
            for v, (view, readable) in enumerate(videos):
                path = os.path.join(cls.tmp, study, f"{v}.mp4")
                if readable:
                    write_video(path)
                else:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(b"not a video")
                rows.append({"video_path": path, "study_id": study, "view": view, "split": "train",
                             "lvef": targets[study][0], "lv_dysfunction": targets[study][1]})
        cls.table = pd.DataFrame(rows)

    def dataset(self, training, **kwargs):
        options = dict(target_columns=["lvef", "lv_dysfunction"], labels_map={"lv_dysfunction": {"normal": 0, "abnormal": 1}},
                       num_videos=3, shuffle_videos=True, training=training, frames_per_clip=FRAMES, frame_step=1,
                       num_clips_per_video=2)
        options.update(kwargs)
        return VideoStudyDataset(self.table, **options)

    def test_studies_are_grouped_with_study_level_targets(self):
        ds = self.dataset(training=False)
        self.assertEqual(ds.study_ids, ["s0", "s1", "s2"])
        np.testing.assert_array_equal(ds.targets[:, 1], [0.0, 1.0, 0.0])
        self.assertEqual(ds.targets[0, 0], 60.0)
        self.assertTrue(np.isnan(ds.targets[2, 0]))

    def test_item_shapes_and_padding(self):
        ds = self.dataset(training=False)
        clips, targets, mask, indices, views = ds[1]  # s1 has one video for three slots.
        self.assertEqual(tuple(clips.shape), (3 * 2, 3, FRAMES, SIZE, SIZE))
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertEqual(tuple(indices.shape), (6, FRAMES))
        self.assertTrue((indices[2:] == -1).all())
        self.assertTrue((clips[2:] == 0).all())
        self.assertEqual(targets.tolist(), [35.0, 1.0])
        self.assertEqual(views.tolist(), [-1, -1, -1])

    def test_evaluation_reads_the_first_videos_in_path_order_every_time(self):
        ds = self.dataset(training=False, num_videos=2)
        self.assertEqual(ds.selection, "fixed")
        for _ in range(3):
            self.assertEqual(ds.select(ds.studies[0][1]), [0, 1])
        a, b = ds[0], ds[0]
        torch.testing.assert_close(a[0], b[0])

    def test_seeded_evaluation_bags_are_random_subsets_drawn_once(self):
        ds = self.dataset(training=False, num_videos=2, eval_videos="seeded", eval_seed=0)
        self.assertEqual(ds.selection, "seeded")
        paths = ds.studies[0][1]
        bag = ds.select(paths, "s0")
        self.assertEqual((len(bag), len(set(bag))), (2, 2))
        self.assertTrue(set(bag) <= {0, 1, 2, 3})
        # The same bag at every read, in another dataset object, whatever the global random state.
        random.seed(5)
        np.random.seed(5)
        again = self.dataset(training=False, num_videos=2, eval_videos="seeded", eval_seed=0)
        self.assertEqual(again.select(paths, "s0"), bag)
        torch.testing.assert_close(ds[0][0], again[0][0])
        # The study id and the evaluation seed change the bag, and every video can be drawn.
        bags = {tuple(ds.select(paths, f"study{i}")) for i in range(50)}
        bags |= {tuple(seeded_bag("s0", eval_seed, 4, 2)) for eval_seed in range(20)}
        self.assertGreater(len(bags), 1)
        self.assertEqual({p for b in bags for p in b}, {0, 1, 2, 3})
        self.assertEqual(ds.select(ds.studies[1][1], "s1"), [0])  # a study with one video keeps it.
        # Training still redraws its bags; an unknown option is refused.
        self.assertEqual(self.dataset(training=True, eval_videos="seeded").selection, "random")
        with self.assertRaisesRegex(ValueError, "eval_videos"):
            self.dataset(training=False, eval_videos="all")

    def test_the_split_keeps_its_patients_when_the_manifest_names_them(self):
        self.assertIsNone(self.dataset(training=False).patient_ids)  # no patient column
        table = self.table.assign(patient_id=self.table.study_id.map({"s0": "p0", "s1": "p1", "s2": "p0"}))
        ds = VideoStudyDataset(table, target_columns=["lvef"], num_videos=1, frames_per_clip=FRAMES, frame_step=1,
                               view_column="view", views=["A2C"])
        self.assertEqual(ds.study_ids, ["s0"])
        self.assertEqual(ds.patient_ids, {"p0", "p1"})  # every row of the split, before the view filter
        gap = table.copy()
        gap.loc[gap.study_id == "s1", "patient_id"] = None
        with self.assertRaisesRegex(ValueError, "1 rows have no `patient_id`"):
            VideoStudyDataset(gap, target_columns=["lvef"], num_videos=1, frames_per_clip=FRAMES, frame_step=1)

    def test_training_draws_another_bag_of_the_study(self):
        ds = self.dataset(training=True, num_videos=2)
        self.assertEqual(ds.selection, "random")
        random.seed(0)
        bags = {tuple(ds.select(ds.studies[0][1])) for _ in range(30)}
        self.assertGreater(len(bags), 1)
        self.assertTrue(all(len(b) == 2 and len(set(b)) == 2 and set(b) <= {0, 1, 2, 3} for b in bags))
        ds_fixed = self.dataset(training=True, num_videos=2, shuffle_videos=False)
        self.assertEqual(ds_fixed.selection, "fixed")

    def test_a_failed_video_pads_its_slot_and_a_failed_study_is_none(self):
        ds = self.dataset(training=False)
        clips, _, mask, _, _ = ds[2]
        self.assertEqual(mask.tolist(), [True, False, False])
        broken = self.table[self.table.video_path.str.endswith("s2/1.mp4")]
        with self.assertRaisesRegex(ValueError, "enough videos"):
            VideoStudyDataset(broken, target_columns=["lvef"], num_videos=1, frames_per_clip=FRAMES, frame_step=1,
                              min_videos=2)
        ds_broken = VideoStudyDataset(broken, target_columns=["lvef"], num_videos=1, frames_per_clip=FRAMES,
                                      frame_step=1)
        self.assertIsNone(ds_broken.get_item_study(0))
        with self.assertRaisesRegex(RuntimeError, "could not be read"):
            ds_broken[0]

    def test_view_filter_and_min_videos(self):
        ds = self.dataset(training=False, view_column="view", views=["A2C"])
        self.assertEqual(ds.study_ids, ["s0"])
        self.assertEqual(len(ds.studies[0][1]), 2)
        ds = self.dataset(training=False, min_videos=2)
        self.assertEqual(ds.study_ids, ["s0", "s2"])

    def test_disagreeing_study_targets_and_unmapped_labels_are_refused(self):
        table = self.table.copy()
        table.loc[table.study_id == "s0", "lvef"] = [60.0, 61.0, 60.0, 60.0]
        with self.assertRaisesRegex(ValueError, "different values"):
            group_studies(table, "video_path", "study_id", ["lvef"])
        with self.assertRaisesRegex(ValueError, "labels_map"):
            group_studies(self.table, "video_path", "study_id", ["lv_dysfunction"])
        with self.assertRaisesRegex(ValueError, "missing from"):
            group_studies(self.table, "video_path", "study_id", ["lv_dysfunction"],
                          labels_map={"lv_dysfunction": {"normal": 0}})
        with self.assertRaisesRegex(ValueError, "no `missing` column"):
            group_studies(self.table, "video_path", "study_id", ["missing"])

    def test_a_conflict_on_a_filtered_out_video_is_still_refused(self):
        table = self.table.copy()
        a2c = (table.study_id == "s0") & (table.view == "A2C")
        table.loc[a2c, "lvef"] = 61.0
        with self.assertRaisesRegex(ValueError, "different values"):
            group_studies(table, "video_path", "study_id", ["lvef"], view_column="view", views=["A4C"])

    def test_training_stops_after_bounded_retries(self):
        broken = self.table[self.table.video_path.str.endswith("s2/1.mp4")]
        ds = VideoStudyDataset(broken, target_columns=["lvef"], num_videos=1, frames_per_clip=FRAMES, frame_step=1,
                               training=True)
        with self.assertRaisesRegex(RuntimeError, "could be read"):
            ds[0]

    def test_read_split(self):
        path = os.path.join(self.tmp, "videos.csv")
        table = self.table.copy()
        table.loc[table.study_id == "s2", "split"] = "val"
        table.to_csv(path, index=False)
        self.assertEqual(sorted(read_split({"data_filename": path}, "val").study_id.unique()), ["s2"])
        with self.assertRaisesRegex(ValueError, "no row"):
            read_split({"data_filename": path}, "test")
        self.assertEqual(len(read_split({"dataset_train": path}, "train")), len(table))
        with self.assertRaisesRegex(ValueError, "dataset_test"):
            read_split({}, "test")
        # Identifier columns are read as text: a gap in a numeric patient column does not turn 123 into 123.0.
        ids = os.path.join(self.tmp, "ids.csv")
        with open(ids, "w") as f:
            f.write("video_path,study_id,patient_id,split\na.mp4,1,123,val\nb.mp4,2,,val\n")
        rows = read_split({"data_filename": ids}, "val")
        self.assertEqual(rows.patient_id.iloc[0], "123")
        self.assertEqual(rows.study_id.tolist(), ["1", "2"])

    def test_view_classes_follow_the_selected_videos(self):
        view_map = {"A4C": 0, "A2C": 1}
        ds = self.dataset(training=False, view_column="view", view_labels_map=view_map)
        self.assertEqual(ds.studies[0][3], [0, 0, 1, 1])  # s0's videos in path order: 0, 1 (A4C), 2, 3 (A2C)
        _, _, mask, _, views = ds[0]
        self.assertEqual((mask.tolist(), views.tolist()), ([True, True, True], [0, 0, 1]))
        _, _, mask, _, views = ds[1]
        self.assertEqual((mask.tolist(), views.tolist()), ([True, False, False], [0, -1, -1]))
        ds = self.dataset(training=True, view_column="view", view_labels_map=view_map, num_videos=2)
        random.seed(3)
        for _ in range(10):
            _, _, _, _, views = ds[0]
            self.assertTrue(set(views.tolist()) <= {0, 1})
        # Unknown views need an `Other` class; classes are numbered without gaps.
        with self.assertRaisesRegex(ValueError, "Other"):
            self.dataset(training=False, view_column="view", view_labels_map={"A4C": 0})
        ds = self.dataset(training=False, view_column="view", view_labels_map={"A4C": 0, "Other": 1})
        self.assertEqual(ds.studies[0][3], [0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "without gaps"):
            self.dataset(training=False, view_column="view", view_labels_map={"A4C": 0, "A2C": 2})
        with self.assertRaisesRegex(ValueError, "view_column"):
            self.dataset(training=False, view_labels_map=view_map)
