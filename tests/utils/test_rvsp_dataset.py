# tests/utils/test_rvsp_dataset.py

"""The per-study loader of the multi-view probe (`src/datasets/video_group_dataset.py`), on tiny
synthetic videos: the slot layout of a manifest, missing positions that are never read, each video
read exactly as the EF probe's `VideoDataset` reads it (short videos padded the same way), an
unreadable video masked and counted, and batches whose videos follow the presence mask's order."""

import os
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import torch

from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset
from src.datasets.video_group_dataset import (MISSING_TOKEN, VideoGroupDataset, collate_studies,
                                              make_videogroupdataset, read_group_manifest)
from tests.utils.test_vjepa_2_1_utils import QuietLogs

FRAMES, CROP = 4, 32


def write_video(path, brightness, frames=24, fps=8):
    """A video whose frame i has the grey level brightness + 3 i (so frames are told apart)."""
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CROP, CROP))
    for i in range(frames):
        writer.write(np.full((CROP, CROP, 3), (brightness + 3 * i) % 256, np.uint8))
    writer.release()
    return path


def write_manifest(path, rows, columns):
    pd.DataFrame(rows, columns=["study_id", *columns, "label"]).to_csv(path, index=False)
    return path


class TestReadGroupManifest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "train.csv")

    def test_the_layout_comes_from_the_header(self):
        write_manifest(self.path, [("s1", "/a.mp4", MISSING_TOKEN, "/b.mp4", 0.5)], ["A4C[0]", "A4C[1]", "RVINF[0]"])
        frame, layout = read_group_manifest(self.path)
        self.assertEqual(layout, [("A4C", 0), ("A4C", 1), ("RVINF", 0)])
        self.assertEqual(frame.label.tolist(), [0.5])

    def test_malformed_manifests_are_refused(self):
        cases = [
            (["A4C[0]"], [("s1", "/a.mp4", "x")], "finite"),
            (["A4C[0]"], [("s1", MISSING_TOKEN, 0.1)], "no video in any slot"),
            (["A4C[0]", "RVINF[0]"], [("s1", "/a.mp4", "/a.mp4", 0.1)], "same video"),
            (["A4C[0]"], [("s1", "/a.mp4", 0.1), ("s1", "/b.mp4", 0.2)], "study twice"),
            (["A4C[0]", "A4C[2]"], [("s1", "/a.mp4", "/b.mp4", 0.1)], "clip positions"),
            (["A4C[0]", "RVINF[0]", "A4C[1]"], [("s1", "/a.mp4", "/b.mp4", "/c.mp4", 0.1)], "adjacent"),
            (["A4C"], [("s1", "/a.mp4", 0.1)], "not a slot position"),
        ]
        for columns, rows, message in cases:
            with self.subTest(message):
                write_manifest(self.path, rows, columns)
                with self.assertRaisesRegex(ValueError, message):
                    read_group_manifest(self.path)
        pd.DataFrame([("/a.mp4", 0.1)], columns=["A4C[0]", "label"]).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "study_id"):
            read_group_manifest(self.path)
        with open(self.path, "w") as f:
            f.write("study_id,A4C[0],label\ns1,,0.1\n")
        with self.assertRaisesRegex(ValueError, "write `MISS`"):
            read_group_manifest(self.path)

    def test_the_upstream_loader_entry_point_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "make_rvsp_probe_manifests"):
            make_videogroupdataset(data_paths=[self.path], batch_size=1, group_size=2)


class TestVideoGroupDataset(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = tmp.name
        cls.long = write_video(os.path.join(cls.tmp, "long.mp4"), 10)
        cls.other = write_video(os.path.join(cls.tmp, "other.mp4"), 120)
        cls.short = write_video(os.path.join(cls.tmp, "short.mp4"), 60, frames=5)  # shorter than 4 frames x 2
        cls.absent = os.path.join(cls.tmp, "absent.mp4")
        cls.manifest = write_manifest(os.path.join(cls.tmp, "val.csv"), [
            ("s1", cls.long, MISSING_TOKEN, cls.short, 1.0),
            ("s2", MISSING_TOKEN, cls.other, MISSING_TOKEN, -1.0),
            ("s3", cls.absent, cls.other, cls.long, 0.0),
            ("s4", cls.absent, MISSING_TOKEN, MISSING_TOKEN, 0.0),
        ], ["A4C[0]", "A4C[1]", "RVINF[0]"])

    def dataset(self, training=False, transform=True):
        transform = make_transforms(training=training, crop_size=CROP) if transform else None
        return VideoGroupDataset(self.manifest, frames_per_clip=FRAMES, frame_step=2, num_segments=2,
                                 transform=transform, random_clip_sampling=training, allow_clip_overlap=True)

    def test_missing_positions_are_never_read(self):
        dataset = self.dataset()
        with mock.patch.object(dataset.reader, "loadvideo_decord", wraps=dataset.reader.loadvideo_decord) as read:
            item = dataset.get_item_study(0)
        self.assertEqual([c.args[0] for c in read.call_args_list], [self.long, self.short])
        self.assertEqual(item["present"].tolist(), [True, False, True])
        self.assertEqual(item["listed"].tolist(), [True, False, True])
        self.assertIsNone(item["videos"][1])  # no stand-in video for the probe to attend to
        self.assertEqual((item["label"], item["index"]), (1.0, 0))
        self.assertEqual(dataset.samples, [self.long, self.short, self.other, self.absent, self.other, self.long,
                                           self.absent])

    def test_each_video_is_read_as_the_ef_probe_reads_it(self):
        dataset = self.dataset()
        for path in (self.long, self.short):
            manifest = os.path.join(self.tmp, "single.csv")
            with open(manifest, "w") as f:
                f.write(f"{path} 0\n")
            single = VideoDataset(manifest, frames_per_clip=FRAMES, frame_step=2, num_clips=2,
                                  transform=make_transforms(training=False, crop_size=CROP),
                                  random_clip_sampling=False, allow_clip_overlap=True)
            clips, _, indices = single.get_item_video(0)
            ours, our_indices = dataset.load_video(path)
            self.assertEqual([i.tolist() for i in our_indices], [i.tolist() for i in indices])
            for a, b in zip(ours, clips):
                self.assertTrue(torch.equal(a, b[0]))
        _, short_indices = dataset.load_video(self.short)
        self.assertIn(-1, short_indices[0].tolist())  # black padding, marked as in the EF probe

    def test_an_unreadable_video_is_masked_and_counted(self):
        dataset = self.dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            item = dataset.get_item_study(2)
            self.assertIsNone(dataset.get_item_study(3))  # no readable video: the study fails
        self.assertEqual(item["present"].tolist(), [False, True, True])
        self.assertEqual(item["listed"].tolist(), [True, True, True])
        self.assertEqual(collate_studies([item])["failed_videos"], 1)
        np.random.seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replaced = dataset[3]  # training draws another study instead
        self.assertNotEqual(replaced["index"], 3)

    def test_a_batch_stacks_its_videos_in_the_order_of_the_presence_mask(self):
        dataset = self.dataset(transform=False)
        items = [dataset.get_item_study(0), dataset.get_item_study(1)]
        batch = collate_studies(items)
        self.assertEqual(batch["present"].tolist(), [[True, False, True], [False, True, False]])
        expected = [self.long, self.short, self.other]  # row-major over (study, position)
        self.assertEqual(len(batch["clips"]), 2)  # num_segments
        for segment in range(2):
            self.assertEqual(tuple(batch["clips"][segment].shape), (3, FRAMES, CROP, CROP, 3))
            for j, path in enumerate(expected):
                clips, indices = dataset.load_video(path)
                self.assertTrue(torch.equal(batch["clips"][segment][j], clips[segment]))
                self.assertTrue(torch.equal(batch["clip_indices"][segment][j], indices[segment]))
        self.assertEqual(batch["labels"].tolist(), [[1.0], [-1.0]])
        self.assertEqual((batch["index"].tolist(), batch["failed_videos"]), ([0, 1], 0))

    def test_evaluation_reads_the_same_frames_and_training_draws_positions(self):
        val = self.dataset()
        first = [i.tolist() for i in val.get_item_study(0)["videos"][0][1]]
        self.assertEqual([i.tolist() for i in val.get_item_study(0)["videos"][0][1]], first)
        self.assertEqual([i[0] for i in first], [0, 12])  # the start of each half
        train = self.dataset(training=True)
        draws = {tuple(int(i[0]) for i in train.get_item_study(0)["videos"][0][1]) for _ in range(20)}
        self.assertGreater(len(draws), 1)


if __name__ == "__main__":
    unittest.main()
