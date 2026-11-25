# tests/data/test_screen_validation.py

"""screen_validation.py on a synthetic manifest: the groups, the patient-aware two-stage sample and its
probabilities, reproducibility, the reference clips and the outputs' privacy."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import screen_validation as sv

MAIN, OTHER_VENDOR = "GE Vingmed Ultrasound", "Philips Medical Systems"


def manifest():
    """Patients p0-p19 (GE): p0 has 60 clips, the others 3; studies of p10-p19 have a low EF. Every patient's first
    clip has 8 frames. p20-p24 are Philips; p25 is a validation-split patient; one clip is undecodable."""
    rows = []
    for i in range(26):
        n = 60 if i == 0 else 3
        for j in range(n):
            rows.append({"patient_id": f"p{i}", "study_id": f"s{i}", "study_date": "20230101",
                         "split": "val" if i == 25 else "train", "view": "A4C",
                         "video_path": f"/synthetic/p{i}/{j}.mp4", "dicom_path": f"/synthetic/p{i}/{j}.dcm",
                         "vendor": OTHER_VENDOR if 20 <= i < 25 else MAIN, "model": "m",
                         "acq_category": "bmode" if j % 2 else "color_flow", "n_frames": 8 if j == 0 else 40,
                         "fps": 30.0, "reason": None})
    rows[1]["reason"] = "undecodable"
    ef = pd.DataFrame({"study_id": [f"s{i}" for i in range(26)],
                       "ef": [30 if 10 <= i < 20 else 60 for i in range(26)]})
    return pd.DataFrame(rows), ef


class TestGroups(unittest.TestCase):
    def test_each_clip_falls_in_the_first_group_that_applies(self):
        clips = pd.DataFrame({"vendor": [MAIN] * 4 + [OTHER_VENDOR] * 2, "n_frames": [8, 40, 40, 40, 8, 40],
                              "fps": [30.0] * 6, "ef": [30, 30, 60, None, 30, 30]})
        self.assertEqual(sv.assign_groups(clips, 10, 0, 35).tolist(),
                         ["frozen_suspect", "low_ef", "general", "general", "frozen_suspect",
                          f"vendor:{OTHER_VENDOR}"])
        clips.loc[2, "fps"] = 4.0
        self.assertEqual(sv.assign_groups(clips, 10, 5, 35).iloc[2], "frozen_suspect")
        # --frozen-max-fps 0: the frame rate is not used, so a manifest without it works (a rate of 0 is no rule).
        clips.loc[3, "fps"] = 0.0
        self.assertEqual(sv.assign_groups(clips, 10, 0, 35).iloc[3], "general")
        self.assertEqual(sv.assign_groups(clips.drop(columns="fps"), 10, 0, 35).tolist(),
                         ["frozen_suspect", "low_ef", "general", "general", "frozen_suspect",
                          f"vendor:{OTHER_VENDOR}"])


class TestTwoStageSample(unittest.TestCase):
    def members(self):
        rows = [("big", f"/b/{j}") for j in range(100)] + [(f"p{i}", f"/p/{i}") for i in range(4)]
        return pd.DataFrame(rows, columns=["patient_id", "video_path"])

    def test_one_clip_per_patient_and_its_probability(self):
        drawn = sv.two_stage_sample(self.members(), 3, 1, np.random.default_rng(0))
        self.assertEqual(len(drawn), 3)
        self.assertFalse(drawn.patient_id.duplicated().any())  # the 100-clip patient fills at most one place.
        expected = {"big": 3 / 5 * 1 / 100, **{f"p{i}": 3 / 5 for i in range(4)}}
        self.assertTrue(np.allclose(drawn.sample_prob, drawn.patient_id.map(expected)))

    def test_clips_per_patient_and_a_whole_group(self):
        drawn = sv.two_stage_sample(self.members(), 4, 2, np.random.default_rng(0))
        self.assertEqual(drawn.patient_id.nunique(), 2)
        self.assertTrue((drawn.groupby("patient_id").size() <= 2).all())
        # Asked for more than the group holds (10 patients of up to 100 clips): every patient and every clip, each
        # with probability 1, so the weights add up to the group's size exactly.
        whole = sv.two_stage_sample(self.members(), 1000, 100, np.random.default_rng(0))
        self.assertEqual((len(whole), (1 / whole.sample_prob).sum()), (104, 104.0))

    def test_the_draw_depends_on_the_seed_not_the_row_order(self):
        members = self.members()
        a = sv.two_stage_sample(members, 3, 1, np.random.default_rng(7))
        b = sv.two_stage_sample(members.iloc[::-1], 3, 1, np.random.default_rng(7))
        self.assertEqual(sorted(a.video_path), sorted(b.video_path))


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        videos, ef = manifest()
        videos.to_parquet(os.path.join(self.tmp, "videos.parquet"))
        ef.to_parquet(os.path.join(self.tmp, "ef.parquet"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def build(self, name, *options):
        out = os.path.join(self.tmp, name)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sv.main(["--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"), "--out-dir", out,
                     "--n-frozen", "4", "--n-low-ef", "3", "--n-per-vendor", "2", "--n-general", "5", *options])
        with open(os.path.join(out, "validation.json")) as f:
            return (pd.read_csv(os.path.join(out, "validation_clips.csv")),
                    pd.read_csv(os.path.join(out, "reference_clips.csv")), json.load(f), out)

    def test_a_stratified_patient_aware_sample_with_its_record(self):
        sample, reference, info, out = self.build("a")
        groups = info["groups"]
        self.assertEqual(sorted(groups), ["frozen_suspect", "general", "low_ef", f"vendor:{OTHER_VENDOR}"])
        # The first clip of each of the 25 training patients, Philips ones included (the first group that applies);
        # p25 is a validation patient.
        self.assertEqual((groups["frozen_suspect"]["clips"], groups["frozen_suspect"]["patients"]), (25, 25))
        self.assertEqual(groups["low_ef"]["clips"], 20)
        self.assertEqual(sample.groupby("sample_stratum").size().to_dict(),
                         {"frozen_suspect": 4, "general": 5, "low_ef": 3, f"vendor:{OTHER_VENDOR}": 2})
        self.assertFalse(sample.duplicated(["sample_stratum", "patient_id"]).any())
        self.assertNotIn("p25", set(sample.patient_id))
        self.assertNotIn("/synthetic/p0/1.mp4", set(sample.video_path))
        self.assertTrue(np.allclose(sample.sample_prob, sample.patient_prob * sample.clip_prob))
        self.assertTrue(sample.global_key.str.fullmatch(r"radar-pilot-[0-9a-f]{24}").all())
        self.assertTrue({"video_path", "dicom_path", "global_key", "vendor", "model", "acq_category", "ef",
                         "stratum_clips", "stratum_patients", "patient_clips"} <= set(sample))
        # Reference clips: the sample's layouts, patients not drawn.
        self.assertFalse(set(reference.patient_id) & set(sample.patient_id))
        layouts = ["vendor", "model", "acq_category"]
        self.assertTrue(set(map(tuple, reference[layouts].to_numpy())) <= set(map(tuple, sample[layouts].to_numpy())))
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o700)
        self.assertEqual({os.stat(os.path.join(out, n)).st_mode & 0o777 for n in sv.OUTPUTS}, {0o600})
        self.assertEqual(set(info["outputs"]), {os.path.join(out, n) for n in sv.OUTPUTS[:2]})

    def test_every_group_of_the_population_gets_a_clip(self):
        # A group with clips but none drawn would drop out of the estimates unnoticed: refused, nothing written.
        for option in ("--n-frozen", "--n-per-vendor"):
            with self.subTest(option), self.assertRaises(SystemExit):
                self.build("empty_group", option, "0")
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "empty_group", "validation.json")))
        sample, _, info, _ = self.build("a")
        self.assertEqual(sum(g["clips"] for g in info["groups"].values()), info["population"]["clips"])
        self.assertTrue((sample.population_clips == info["population"]["clips"]).all())

    def test_a_sample_that_leaves_no_reference_clip_is_refused(self):
        # The defaults draw every patient of this small manifest: no reference clip would be left, and flag would
        # exclude every clip as incomplete. Refused before anything is written.
        out = os.path.join(self.tmp, "all_drawn")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
            sv.main(["--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"), "--out-dir", out])
        self.assertIn("No reference clips", str(stop.exception))
        self.assertEqual(os.listdir(out), [])
        # A list with reference clips counts the layouts with too few for a baseline, and warns of them.
        warned = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(warned):
            sv.main(["--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"), "--out-dir", os.path.join(self.tmp, "a"),
                     "--n-frozen", "4", "--n-low-ef", "3", "--n-per-vendor", "2", "--n-general", "5"])
        with open(os.path.join(self.tmp, "a", "validation.json")) as f:
            layouts = json.load(f)["reference_layouts"]
        self.assertEqual(layouts["min_layout_clips"], sv.clip_screen.MIN_LAYOUT_CLIPS)
        self.assertEqual(layouts["below_min_layout_clips"], layouts["sample_layouts"])  # a small manifest.
        n = layouts["sample_layouts"]
        self.assertIn(f"{n} of the sample's {n} layouts have fewer than", warned.getvalue())

    def test_a_check_list_leaves_out_the_patients_the_screen_was_tuned_on(self):
        tuned, _, _, tuned_dir = self.build("tuned")
        check, reference, info, _ = self.build("check", "--n-frozen", "2", "--exclude-patients",
                                               os.path.join(tuned_dir, "validation_clips.csv"))
        self.assertFalse(set(check.patient_id) & set(tuned.patient_id))
        self.assertFalse(set(reference.patient_id) & set(tuned.patient_id))
        self.assertEqual(info["population"]["excluded_patients"]["patients"], tuned.patient_id.nunique())
        self.assertEqual(list(info["population"]["excluded_patients"]["lists"]), ["validation_clips.csv"])

    def test_an_input_replaced_while_the_list_is_drawn_completes_nothing(self):
        ef_path = os.path.join(self.tmp, "ef.parquet")
        ef = pd.read_parquet(ef_path)
        real = sv.assign_groups

        def meanwhile(*args, **kwargs):  # the EF labels rewritten after they were read: other low-EF studies.
            ef.assign(ef=10.0).to_parquet(ef_path)
            return real(*args, **kwargs)

        with mock.patch.object(sv, "assign_groups", meanwhile), self.assertRaises(SystemExit) as stop:
            self.build("replaced")
        self.assertIn("ef.parquet changed while this step ran", str(stop.exception))
        out = os.path.join(self.tmp, "replaced")
        self.assertFalse(set(os.listdir(out)) & set(sv.OUTPUTS))

    def test_a_manifest_without_the_acquisition_columns_is_refused(self):
        videos = os.path.join(self.tmp, "videos.parquet")
        pd.read_parquet(videos).drop(columns=["vendor"]).to_parquet(videos)
        with self.assertRaises(SystemExit) as stop:
            self.build("no_vendor")
        self.assertIn("['vendor']", str(stop.exception))
        self.assertIn("commit 57204d1", str(stop.exception))

    def test_a_manifest_without_frame_rates_works_unless_the_rate_rule_is_on(self):
        videos = pd.read_parquet(os.path.join(self.tmp, "videos.parquet")).drop(columns="fps")
        videos.to_parquet(os.path.join(self.tmp, "videos.parquet"))
        sample, _, info, _ = self.build("no_fps")
        self.assertEqual(info["groups"]["frozen_suspect"]["clips"], 25)  # by the number of frames alone.
        self.assertNotIn("fps", sample)
        with self.assertRaises(SystemExit) as stop:
            self.build("fps_rule", "--frozen-max-fps", "5")
        self.assertIn("fps column", str(stop.exception))

    def test_the_seed_fixes_the_sample(self):
        first, _, _, _ = self.build("a")
        again, _, _, _ = self.build("b")
        other, _, _, _ = self.build("c", "--seed", "1")
        self.assertEqual(first.video_path.tolist(), again.video_path.tolist())
        self.assertNotEqual(first.video_path.tolist(), other.video_path.tolist())

    def test_a_shared_folder_is_refused(self):
        shared = os.path.join(self.tmp, "shared")
        os.mkdir(shared)
        os.chmod(shared, 0o755)
        with self.assertRaises(SystemExit):
            self.build("shared")
        self.assertEqual(os.listdir(shared), [])


if __name__ == "__main__":
    unittest.main()
