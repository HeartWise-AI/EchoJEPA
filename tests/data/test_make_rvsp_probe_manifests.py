# tests/data/test_make_rvsp_probe_manifests.py

"""make_rvsp_probe_manifests.py on a synthetic archive index: one row per study with fixed slot
positions, per-slot view and acquisition filters, a seeded and order-independent video choice,
explicit missing positions, z-scores from the probe train split only, counted exclusions, and the
refusals (split disagreement, patient leakage, ambiguous labels, outputs inside the repository)."""

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import make_rvsp_probe_manifests as mrm
from build_manifests import REPO_ROOT
from evals.video_classification_frozen import provenance

# (study, patient, split, RVSP); s6 has no RVSP, s7 an implausible one (the default range is 5-150 mmHg),
# s8 only an A2C video.
STUDIES = [("s1", "p1", "train", 30.0), ("s2", "p2", "train", 45.0), ("s3", "p3", "train", 60.0),
           ("s4", "p4", "val", 40.0), ("s5", "p5", "test", 50.0), ("s6", "p6", "train", None),
           ("s7", "p7", "val", 250.0), ("s8", "p8", "test", 35.0)]
# (study, view, acquisition, reason): the videos of each study.
VIDEOS = [
    ("s1", "A4C", "bmode", None), ("s1", "A4C", "bmode", None), ("s1", "A4C", "bmode", None),
    ("s1", "A4C", "bmode", "decode_failed"), ("s1", "A4C", "color_flow", None), ("s1", "RVINF", "unknown", None),
    ("s1", "PSAX", "color_flow", None), ("s1", "PSAX", "bmode", None), ("s1", "PSAX_AV", "mmode", None),
    ("s2", "A4C", "bmode", None),
    ("s3", "RVINF", "bmode", None), ("s3", "PSAX_AV", "color_flow", None), ("s3", "SUBCOSTAL", None, None),
    ("s4", "A4C", "bmode", None), ("s4", "A4C", "bmode", None),
    ("s5", "SUBCOSTAL", "bmode", None),
    ("s6", "A4C", "bmode", None),
    ("s7", "A4C", "bmode", None),
    ("s8", "A2C", "bmode", None),
    ("s9", "A4C", "bmode", None),  # patient p1's other study, not in the probe
    # s5: header B-mode from a vendor outside acquisition.bmode_vendors (0020, 0022) and a B-mode the
    # header does not confirm (0021): unresolved, so no B-mode slot takes them; `any` slots still do.
    ("s5", "A4C", "bmode", None), ("s5", "PSAX", "bmode", None), ("s5", "RVINF", "bmode", None),
]
UNVALIDATED_VENDOR = {20, 22}
UNCONFIRMED = {21}


def index_frame():
    split = {s: (p, sp) for s, p, sp, _ in STUDIES}
    split["s9"] = ("p1", "train")
    rows = [(split[s][0], s, split[s][1], view, f"/synthetic/{split[s][0]}/{s}/{i:04d}.avi", reason, acq,
             None if acq is None else "unknown" if acq == "unknown" or i in UNCONFIRMED else "header",
             i not in UNVALIDATED_VENDOR)
            for i, (s, view, acq, reason) in enumerate(VIDEOS)]
    return pd.DataFrame(rows, columns=list(mrm.INDEX_COLUMNS))


def labels_frame():
    return pd.DataFrame([(s, rvsp) for s, _, _, rvsp in STUDIES] + [("s9", 33.0)], columns=["study_id", "rvsp"])


def probe_frame():
    return pd.DataFrame([(s, sp) for s, _, sp, _ in STUDIES], columns=["study_id", "split"])


class TestMakeRvspProbeManifests(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.write(index_frame(), labels_frame(), probe_frame())

    def write(self, index, labels, probe):
        self.index, self.labels, self.probe = (os.path.join(self.tmp, n) for n in
                                               ("videos.parquet", "rvsp.parquet", "probe_studies.csv"))
        index.to_parquet(self.index, index=False)
        labels.to_parquet(self.labels, index=False)
        probe.to_csv(self.probe, index=False)

    def run_main(self, out="probe", *extra):
        out = os.path.join(self.tmp, out)
        with contextlib.redirect_stdout(io.StringIO()):
            mrm.main(["--video-index", self.index, "--labels", self.labels, "--probe-splits", self.probe,
                      "--out-dir", out, *extra])
        return out

    def read(self, out, split):
        return pd.read_csv(os.path.join(out, f"{split}.csv"), dtype=str)

    def info(self, out):
        with open(os.path.join(out, "probe_info.json")) as f:
            return json.load(f)

    def test_one_row_per_study_with_a_video_or_the_missing_token_in_each_position(self):
        out = self.run_main()
        train = self.read(out, "train")
        positions = mrm.positions(mrm.check_slots(mrm.DEFAULT_SLOTS))
        self.assertEqual(list(train.columns), ["study_id", *positions, "label"])
        self.assertEqual(list(train.study_id), ["s1", "s2", "s3"])
        s1, s2, s3 = (train.iloc[i] for i in range(3))
        self.assertEqual(s1["A4C_color[0]"], "/synthetic/p1/s1/0004.avi")
        self.assertEqual(s1["RVINF[0]"], "/synthetic/p1/s1/0005.avi")         # `any` admits an unknown mode
        self.assertEqual(s1["PSAX_bmode[0]"], "/synthetic/p1/s1/0007.avi")     # not the colour PSAX
        self.assertEqual(s1["PSAX_AV[0]"], "/synthetic/p1/s1/0008.avi")        # `any` admits M-mode
        self.assertEqual(s1["SUBCOSTAL[0]"], mrm.MISSING_TOKEN)
        self.assertEqual([c for c in positions if s2[c] != mrm.MISSING_TOKEN], ["A4C_bmode[0]"])
        self.assertEqual(s3["SUBCOSTAL[0]"], "/synthetic/p3/s3/0012.avi")      # a null category is `any`
        self.assertEqual(s3["A4C_bmode[0]"], mrm.MISSING_TOKEN)
        self.assertEqual(list(self.read(out, "val").study_id), ["s4"])
        self.assertEqual(list(self.read(out, "test").study_id), ["s5"])

    def test_slot_filters_by_view_acquisition_and_verification(self):
        out = self.run_main()
        s1 = self.read(out, "train").iloc[0]
        bmode = {f"/synthetic/p1/s1/{i:04d}.avi" for i in (0, 1, 2)}  # 0003 is not verified
        self.assertIn(s1["A4C_bmode[0]"], bmode)
        chosen = pd.read_csv(os.path.join(out, "video_index.csv"))
        self.assertNotIn("/synthetic/p1/s1/0003.avi", set(chosen.video_path))
        self.assertNotIn("/synthetic/p1/s1/0006.avi", set(chosen.video_path))  # colour PSAX: no slot takes it

    def test_unresolved_bmode_fills_no_bmode_slot(self):
        out = self.run_main()
        s5 = self.read(out, "test").iloc[0]
        self.assertEqual(s5["A4C_bmode[0]"], mrm.MISSING_TOKEN)              # vendor not validated
        self.assertEqual(s5["PSAX_bmode[0]"], mrm.MISSING_TOKEN)             # header does not confirm it
        self.assertEqual(s5["RVINF[0]"], "/synthetic/p5/s5/0022.avi")        # `any` admits it
        self.assertEqual(s5["SUBCOSTAL[0]"], "/synthetic/p5/s5/0015.avi")
        # Reported under the category the slots used, as in pretraining.
        self.assertEqual(self.info(out)["splits"]["test"]["acquisition"], {"bmode": 1, "unknown": 1})

    def test_the_vendor_flag_reads_from_csv_and_is_checked(self):
        index = index_frame()
        index.to_csv(os.path.join(self.tmp, "videos.csv"), index=False)
        self.index = os.path.join(self.tmp, "videos.csv")
        s5 = self.read(self.run_main("from_csv"), "test").iloc[0]
        self.assertEqual(s5["A4C_bmode[0]"], mrm.MISSING_TOKEN)
        index["acq_vendor_validated"] = index.acq_vendor_validated.astype(str).replace({"True": "yes"})
        index.to_csv(self.index, index=False)
        with self.assertRaisesRegex(ValueError, "not true or false"):
            self.run_main("bad_flag")

    def test_the_choice_is_the_seeded_hash_minimum(self):
        out = self.run_main()
        candidates = [f"/synthetic/p1/s1/{i:04d}.avi" for i in (0, 1, 2)]
        expected = min(candidates, key=lambda p: mrm.rank_key(0, "A4C_bmode", p))
        self.assertEqual(self.read(out, "train").iloc[0]["A4C_bmode[0]"], expected)
        picks = {min(candidates, key=lambda p: mrm.rank_key(seed, "A4C_bmode", p)) for seed in range(10)}
        self.assertGreater(len(picks), 1)  # the seed matters; every pick is reproducible

    def test_the_choice_depends_on_neither_row_order_nor_other_studies(self):
        out = self.run_main("first")
        index = index_frame().sample(frac=1.0, random_state=3).reset_index(drop=True)
        self.write(index, labels_frame().iloc[::-1], probe_frame().iloc[::-1])
        shuffled = self.run_main("shuffled")
        for split in ("train", "val", "test", "video_index"):
            with open(os.path.join(out, f"{split}.csv"), "rb") as a, \
                    open(os.path.join(shuffled, f"{split}.csv"), "rb") as b:
                self.assertEqual(a.read(), b.read(), split)
        self.write(index_frame(), labels_frame(), probe_frame()[probe_frame().study_id != "s2"])
        fewer = self.read(self.run_main("fewer"), "train")
        pd.testing.assert_series_equal(fewer.iloc[0].drop("label"), self.read(out, "train").iloc[0].drop("label"),
                                       check_names=False)

    def test_several_clips_per_slot_and_a_video_in_one_position_only(self):
        spec = os.path.join(self.tmp, "slots.yaml")
        with open(spec, "w") as f:
            f.write("slots:\n"
                    "  - {name: A4C_any, views: [A4C], acq: any, clips: 2}\n"
                    "  - {name: A4C_bmode, views: [A4C], acq: [bmode], clips: 2}\n")
        out = self.run_main("probe", "--slots", spec)
        train = self.read(out, "train")
        self.assertEqual(list(train.columns), ["study_id", "A4C_any[0]", "A4C_any[1]", "A4C_bmode[0]",
                                               "A4C_bmode[1]", "label"])
        s1 = [p for p in train.iloc[0][["A4C_any[0]", "A4C_any[1]", "A4C_bmode[0]", "A4C_bmode[1]"]]
              if p != mrm.MISSING_TOKEN]
        self.assertEqual(len(s1), len(set(s1)))  # no video used twice
        bmode = {f"/synthetic/p1/s1/{i:04d}.avi" for i in (0, 1, 2)}
        self.assertLessEqual(set(s1[2:]), bmode)
        self.assertEqual(len(s1), 4 if "/synthetic/p1/s1/0004.avi" in s1[:2] else 3)
        s2 = train.iloc[1]
        self.assertEqual((s2["A4C_any[1]"], s2["A4C_bmode[0]"]), (mrm.MISSING_TOKEN, mrm.MISSING_TOKEN))
        self.assertEqual(list(self.read(out, "train").study_id), ["s1", "s2"])  # s3 has no A4C

    def test_z_scores_use_the_probe_train_split_only(self):
        out = self.run_main()
        info = self.info(out)
        train_rvsp = np.array([30.0, 45.0, 60.0])
        mean, std = train_rvsp.mean(), train_rvsp.std()
        self.assertAlmostEqual(info["target_mean"], mean)
        self.assertAlmostEqual(info["target_std"], std)
        np.testing.assert_allclose(self.read(out, "train").label.astype(float), (train_rvsp - mean) / std)
        np.testing.assert_allclose(self.read(out, "val").label.astype(float), [(40.0 - mean) / std])
        # Other val and test RVSP values leave the training targets and the normalization as they were.
        labels = labels_frame()
        labels.loc[labels.study_id.isin(["s4", "s5"]), "rvsp"] = [90.0, 20.0]
        self.write(index_frame(), labels, probe_frame())
        other = self.run_main("other")
        self.assertEqual(self.info(other)["target_mean"], info["target_mean"])
        with open(os.path.join(out, "train.csv"), "rb") as a, open(os.path.join(other, "train.csv"), "rb") as b:
            self.assertEqual(a.read(), b.read())
        for key in ("splits", "counts"):  # no val or test label statistics
            self.assertNotIn("rvsp", json.dumps(info[key]))
            self.assertNotIn("mean", json.dumps(info[key]))

    def test_exclusions_and_presence_are_counted(self):
        info = self.info(self.run_main())
        counts = info["counts"]
        self.assertEqual(counts["train"]["missing_label"], 1)  # s6
        self.assertEqual(counts["val"]["invalid_label"], 1)    # s7: 250 mmHg
        labels = labels_frame()
        labels.loc[labels.study_id == "s2", "rvsp"] = 4.0           # below 5 mmHg
        _, _, _, low = mrm.make(index_frame(), labels, probe_frame(), mrm.DEFAULT_SLOTS)
        self.assertEqual(low["train"]["invalid_label"], 1)
        self.assertEqual(counts["test"]["no_slot_filled"], 1)  # s8: only A2C
        self.assertEqual(counts["train"]["unverified_videos"], 1)
        self.assertEqual(counts["train"]["listed"], 4)
        train = info["splits"]["train"]
        self.assertEqual((train["studies"], train["patients"], train["videos"]), (3, 3, 9))
        self.assertEqual(train["slots"]["A4C_bmode"], {"present": 2, "missing": 1})
        self.assertEqual(train["positions"]["SUBCOSTAL[0]"], {"present": 1, "missing": 2})
        # Effective categories: a missing one counts as unknown.
        self.assertEqual(train["acquisition"], {"bmode": 4, "color_flow": 2, "mmode": 1, "unknown": 2})
        self.assertEqual(info["outputs"]["test.csv"]["sha256"],
                         provenance.sha256(os.path.join(self.tmp, "probe", "test.csv")))

    def test_the_video_index_loads_for_the_probe(self):
        out = self.run_main()
        index = provenance.load_video_index(os.path.join(out, "video_index.csv"))
        train = self.read(out, "train")
        listed = [p for p in train.drop(columns=["study_id", "label"]).to_numpy().ravel() if p != mrm.MISSING_TOKEN]
        self.assertEqual(list(provenance.study_ids(index, listed, "train")),
                         [s for s, row in zip(train.study_id, train.drop(columns=["study_id", "label"]).to_numpy())
                          for p in row if p != mrm.MISSING_TOKEN])

    def test_a_split_that_disagrees_with_the_index_is_refused(self):
        probe = probe_frame()
        probe.loc[probe.study_id == "s4", "split"] = "train"
        self.write(index_frame(), labels_frame(), probe)
        with self.assertRaisesRegex(ValueError, "another split"):
            self.run_main()

    def test_a_patient_in_two_probe_splits_is_refused(self):
        index = index_frame()
        index.loc[index.study_id == "s4", "patient_id"] = "p1"  # p1 also has s1 in train
        self.write(index, labels_frame(), probe_frame())
        with self.assertRaisesRegex(ValueError, "patients have probe studies in more than one split"):
            self.run_main()

    def test_ambiguous_inputs_are_refused(self):
        cases = {
            "not in the video index": (index_frame(), labels_frame(), pd.concat(
                [probe_frame(), pd.DataFrame([("s99", "test")], columns=["study_id", "split"])])),
            "one row per study": (index_frame(), pd.concat([labels_frame(), labels_frame().iloc[:1]]), probe_frame()),
            "probe splits list a study twice": (index_frame(), labels_frame(),
                                                pd.concat([probe_frame(), probe_frame().iloc[:1]])),
            "empty `reason`": (index_frame().assign(reason=lambda f: f.reason.where(f.study_id != "s2", "")),
                               labels_frame(), probe_frame()),
            "videos twice": (pd.concat([index_frame(), index_frame().iloc[:1]]), labels_frame(), probe_frame()),
        }
        for i, (message, frames) in enumerate(cases.items()):
            with self.subTest(message):
                self.write(*frames)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_main(f"case{i}")
        labels = labels_frame().astype({"rvsp": object})
        labels.loc[0, "rvsp"] = "high"
        with self.assertRaisesRegex(ValueError, "not numbers"):
            mrm.make(index_frame(), labels, probe_frame(), mrm.DEFAULT_SLOTS)

    def test_slot_specifications_are_checked(self):
        for slots, message in [
            ([], "non-empty"),
            ([{"name": "A4C", "views": ["A4C"], "acq": ["doppler"]}], "`acq`"),
            ([{"name": "A4C", "views": ["A4C"]}, {"name": "A4C", "views": ["A2C"]}], "twice"),
            ([{"name": "A4C", "views": ["A4C"], "clips": 0}], "positive integer"),
            ([{"name": "A4C view", "views": ["A4C"]}], "letters"),
            ([{"name": "A4C", "views": ["A4C"], "mode": "bmode"}], "Unknown slot keys"),
        ]:
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                mrm.check_slots(slots)
        with self.assertRaisesRegex(ValueError, "view labels \\['A5C'\\]"):  # a typo, not an empty slot
            mrm.make(index_frame(), labels_frame(), probe_frame(), [{"name": "A5C", "views": ["A5C"]}])

    def test_outputs_go_outside_the_repository_into_an_empty_folder(self):
        with self.assertRaises(SystemExit):
            mrm.main(["--video-index", self.index, "--labels", self.labels, "--probe-splits", self.probe,
                      "--out-dir", os.path.join(REPO_ROOT, "rvsp_probe")])
        self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "rvsp_probe")))
        out = self.run_main()
        with self.assertRaisesRegex(SystemExit, "not empty"):
            self.run_main()
        self.run_main("probe", "--overwrite")
        self.assertTrue(os.path.exists(os.path.join(out, "probe_info.json")))


if __name__ == "__main__":
    unittest.main()
