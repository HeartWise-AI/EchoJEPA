# tests/data/test_make_probe_manifests.py

"""make_probe_manifests.py on synthetic issue #6 outputs: view selection, z-scoring with the
train split's statistics, the manifest format the loader reads, and the output-folder rules."""

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import make_probe_manifests as mpm
from build_manifests import REPO_ROOT


def videos_csv(path):
    """videos.csv of a tiny cohort: 3 train studies, 1 val, 1 test; A4C, A2C and OTHER views."""
    rows = []
    studies = [("p1", "s1", "train", 50.0), ("p2", "s2", "train", 60.0), ("p3", "s3", "train", 40.0),
               ("p4", "s4", "val", 55.0), ("p5", "s5", "test", 35.0)]
    for patient, study, split, ef in studies:
        for i, view in enumerate(("A4C", "A4C", "A2C", "OTHER")):
            rows.append((patient, study, "TTE", view, f"/synthetic/{patient}/{study}/{i:04d}.mp4",
                         "success", "20240101", ef, split, 32, 30.0, False))
    rows.pop(4 * 2)  # study s3 keeps a single A4C video
    pd.DataFrame(rows, columns=["patient_id", "study_id", "exam_type", "view", "video_path", "avi_status",
                                "study_date", "label", "split", "n_frames", "fps", "needs_padding"]).to_csv(path, index=False)


class TestMakeProbeManifests(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.manifests = os.path.join(tmp.name, "manifests")
        os.makedirs(self.manifests)
        videos_csv(os.path.join(self.manifests, "videos.csv"))
        self.out = os.path.join(tmp.name, "probe")

    def run_main(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            mpm.main(["--manifests", self.manifests, "--out-dir", self.out, *extra])

    def read(self, split):
        return pd.read_csv(os.path.join(self.out, f"{split}.csv"), sep=" ", header=None, names=["path", "target"])

    def test_one_row_per_video_of_the_views_with_train_z_scores(self):
        self.run_main()
        train = self.read("train")
        self.assertEqual(len(train), 5)  # 2 + 2 + 1 A4C videos
        self.assertTrue(train.path.str.endswith((".mp4",)).all())
        efs = np.array([50, 50, 60, 60, 40], dtype=float)  # rows sorted by path: s1, s1, s2, s2, s3
        mean, std = efs.mean(), efs.std()
        np.testing.assert_allclose(train.target, (efs - mean) / std)
        with open(os.path.join(self.out, "probe_info.json")) as f:
            info = json.load(f)
        self.assertAlmostEqual(info["target_mean"], mean)
        self.assertAlmostEqual(info["target_std"], std)
        # Val and test use the train statistics, not their own.
        np.testing.assert_allclose(self.read("val").target, [(55 - mean) / std] * 2)
        np.testing.assert_allclose(self.read("test").target, [(35 - mean) / std] * 2)
        self.assertEqual(info["splits"]["train"], {"videos": 5, "studies": 3, "patients": 3, "ef_mean": efs.mean()})

    def test_several_views(self):
        self.run_main("--views", "A4C", "A2C")
        self.assertEqual(len(self.read("train")), 8)  # 3 + 3 + 2
        self.assertEqual(len(self.read("val")), 3)

    def test_rows_are_what_the_loader_parses(self):
        from src.datasets.video_dataset import VideoDataset
        self.run_main()
        dataset = VideoDataset(data_paths=[os.path.join(self.out, "val.csv")], frames_per_clip=16)
        self.assertEqual(len(dataset.samples), 2)
        self.assertTrue(all(isinstance(label, float) for label in dataset.labels))

    def edit_videos(self, edit):
        path = os.path.join(self.manifests, "videos.csv")
        videos = pd.read_csv(path)
        edit(videos)
        videos.to_csv(path, index=False)

    def test_videos_without_a_valid_ef_are_left_out_and_counted(self):
        def edit(videos):
            videos.loc[0, "label"] = np.nan  # s1, A4C, train
            videos.loc[videos.study_id == "s4", "label"] = 120.0  # val: 2 A4C videos
        self.edit_videos(edit)
        self.run_main()
        self.assertEqual(len(self.read("train")), 4)
        self.assertEqual(len(self.read("val")), 0)
        with open(os.path.join(self.out, "probe_info.json")) as f:
            info = json.load(f)
        self.assertEqual(info["excluded"]["train"], {"missing_label": 1, "invalid_label": 0})
        self.assertEqual(info["excluded"]["val"], {"missing_label": 0, "invalid_label": 2})
        self.assertEqual(info["excluded"]["test"], {"missing_label": 0, "invalid_label": 0})

    def test_a_patient_in_two_splits_is_refused(self):
        def edit(videos):
            videos.loc[videos.study_id == "s4", "patient_id"] = "p1"  # p1 is a train patient
        self.edit_videos(edit)
        with self.assertRaisesRegex(ValueError, "1 patients appear in more than one split"):
            self.run_main()

    def test_the_video_index_gives_each_video_its_split_study_and_patient(self):
        self.run_main()
        index = pd.read_csv(os.path.join(self.out, "video_index.csv"), dtype=str)
        self.assertEqual(list(index.columns), ["split", "video_path", "study_id", "patient_id"])
        for split in ("train", "val", "test"):
            self.assertEqual(index[index.split == split].video_path.tolist(), self.read(split).path.tolist())
        self.assertEqual(index.set_index("video_path").loc["/synthetic/p4/s4/0000.mp4"].tolist(), ["val", "s4", "p4"])
        with open(os.path.join(self.out, "probe_info.json")) as f:
            info = json.load(f)
        self.assertEqual(info["outputs"]["video_index.csv"]["rows"], 9)  # 5 + 2 + 2 A4C videos
        self.assertEqual(info["patients_in_several_splits"], 0)

    def test_output_folder_rules(self):
        with self.assertRaisesRegex(SystemExit, "inside the code repository"):
            mpm.main(["--manifests", self.manifests, "--out-dir", os.path.join(REPO_ROOT, "probe_out")])
        self.run_main()
        with self.assertRaisesRegex(SystemExit, "not empty"):
            self.run_main()
        self.run_main("--overwrite")


if __name__ == "__main__":
    unittest.main()
