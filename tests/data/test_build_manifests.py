# tests/data/test_build_manifests.py

"""Tests for the reproducible EchoJEPA manifest pipeline (issue #6).

Synthetic metadata only: identifiers, dates and paths are made up, and the
videos used for verification are tiny clips written to a temporary directory.
"""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import yaml

import build_manifests as bm
import link_ef_labels as lel

TTE = "ECHO CARDIAQUE AVEC DOPPLER"
TEE = "ETO SOP ANESTHESIE"
STRESS = "ECHO CARDIAQUE A L'EFFORT"
VIEWS = ["A4C", "A2C", "PLAX", "PSAX", "OTHER"]

CFG = {
    "columns": {
        "patient_id": "mrn", "study_id": "study_uid", "exam_type": "study_type", "view": "predicted_class",
        "video_path": "avi_path", "status": "avi_status", "study_date": "date",
    },
    "labels": {"study_id": "study_id", "value": "ef", "low_threshold": 40},
    "eligibility": {"exam_types": [TTE], "status_ok": ["success"], "require_label": True, "years": None},
    "selection": {"n_patients": 100, "seed": 7},
    "split": {"train": 0.7, "val": 0.1, "test": 0.2},
}
META_COLUMNS = ["mrn", "study_uid", "study_type", "predicted_class", "avi_path", "avi_status", "date"]


def config(**selection):
    cfg = copy.deepcopy(CFG)
    cfg["selection"].update(selection)
    return cfg


def accept_all(path):
    """Stand-in verifier for tests about selection rather than files."""
    return None


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def synthetic(n_patients, seed=0, root="/synthetic/videos"):
    """Metadata and labels shaped like the real export, with made-up values.

    Patients have 1-3 studies of mixed exam types, 2-6 videos each over all
    views including OTHER; a few videos failed conversion and some studies
    have no label. Study ids carry literal quotes, as the real export does.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for p in range(n_patients):
        pid = f"{p:07d}"
        for s in range(rng.integers(1, 4)):
            sid = f"1.2.999.{p}.{s}"
            exam = rng.choice([TTE] * 8 + [TEE, STRESS])
            date = f"20{rng.integers(10, 25):02d}0101"
            for v in range(rng.integers(2, 7)):
                status = "success" if rng.random() < 0.95 else "missing video"
                rows.append((pid, f"'{sid}'", exam, rng.choice(VIEWS), f"{root}/{pid}/{sid}/{v:04d}.mp4", status, date))
            if rng.random() < 0.85:
                labels.append((sid, float(np.clip(rng.normal(55, 10), 10, 80))))
    return pd.DataFrame(rows, columns=META_COLUMNS), pd.DataFrame(labels, columns=["study_id", "ef"])


def write_video(path, frames=12):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), 10 * i, np.uint8))
    writer.release()


def write_partly_corrupt_video(path):
    """A video whose container and first frames are intact but later frames are not."""
    import cv2

    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
    for _ in range(40):
        writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    writer.release()
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        start, stop = int(len(data) * 0.6), int(len(data) * 0.8)
        data[start:stop] = rng.integers(0, 256, stop - start, dtype=np.uint8).tobytes()
        f.seek(0)
        f.write(data)


class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.videos, cls.labels = synthetic(400, seed=1)
        cls.result = quiet(bm.build, cls.videos, cls.labels, config(), validate=accept_all)

    def test_exactly_ten_thousand_studies_from_more_video_rows(self):
        videos, labels = synthetic(12_500, seed=2)
        result = quiet(bm.build, videos, labels, config(n_patients=10_000), validate=accept_all)
        self.assertEqual(result["studies"].study_id.nunique(), 10_000)
        self.assertEqual(result["studies"].patient_id.nunique(), 10_000)
        self.assertGreater(len(result["videos"]), 10_000)
        self.assertEqual(result["studies"].split.value_counts().to_dict(), {"train": 7000, "test": 2000, "val": 1000})

    def test_same_seed_and_shuffled_rows_give_identical_cohort(self):
        shuffled = self.videos.sample(frac=1, random_state=3)
        labels = self.labels.sample(frac=1, random_state=4)
        again = quiet(bm.build, shuffled, labels, config(), validate=accept_all)
        for key in ("videos", "studies"):
            pd.testing.assert_frame_equal(self.result[key], again[key])

    def test_streamed_files_give_identical_cohort_at_any_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet, csv = os.path.join(tmp, "meta.parquet"), os.path.join(tmp, "meta.csv")
            self.videos.to_parquet(parquet, index=False)
            self.videos.to_csv(csv, index=False)
            for source, batch_size in ((self.videos, 37), (parquet, 97), (csv, 250)):
                again = quiet(bm.build, source, self.labels, config(), validate=accept_all, batch_size=batch_size)
                for key in ("videos", "studies"):
                    pd.testing.assert_frame_equal(self.result[key], again[key])
                self.assertEqual(self.result["counts"], again["counts"])

    def test_only_candidate_patients_videos_are_loaded(self):
        requested = []

        def spy(source, cfg, labels, study_ids, batch_size):
            requested.extend(study_ids)
            return fetch(source, cfg, labels, study_ids, batch_size=batch_size)

        fetch = bm.fetch_videos
        with mock.patch.object(bm, "fetch_videos", spy):
            result = quiet(bm.build, self.videos, self.labels, config(n_patients=20), validate=accept_all)
        self.assertLessEqual(set(result["studies"].study_id), set(requested))
        self.assertLess(len(set(requested)), result["counts"]["eligible_studies"] / 4)

    def test_different_seed_changes_cohort(self):
        other = quiet(bm.build, self.videos, self.labels, config(seed=8), validate=accept_all)
        self.assertNotEqual(set(self.result["studies"].study_id), set(other["studies"].study_id))

    def test_patients_never_cross_splits_and_have_one_study(self):
        studies = self.result["studies"]
        self.assertFalse(studies.patient_id.duplicated().any())
        per_patient = self.result["videos"].groupby("patient_id").split.nunique()
        self.assertTrue((per_patient == 1).all())
        self.assertEqual(studies.split.value_counts().to_dict(), {"train": 70, "test": 20, "val": 10})

    def test_eligibility_rules(self):
        kept = self.result["videos"]
        self.assertTrue((kept.exam_type == TTE).all())
        self.assertTrue(kept.label.notna().all())
        self.assertIn("OTHER", set(kept.view))
        self.assertFalse(kept.study_id.str.contains("'").any(), "quoted ids were not normalized")
        failed = set(self.videos.loc[self.videos.avi_status != "success", "avi_path"])
        self.assertFalse(failed & set(kept.video_path))
        unlabelled = set(self.videos.study_uid.str.strip("'")) - set(self.labels.study_id)
        self.assertFalse(unlabelled & set(kept.study_id))

    def test_manifest_rows_are_every_verified_video_of_the_selected_studies(self):
        chosen = set(self.result["studies"].study_id)
        meta = self.videos.assign(study_uid=self.videos.study_uid.str.strip("'"))
        expected = meta[meta.study_uid.isin(chosen) & (meta.avi_status == "success")]
        self.assertEqual(set(expected.avi_path), set(self.result["videos"].video_path))

    def test_too_few_eligible_patients_raises(self):
        with self.assertRaises(ValueError):
            quiet(bm.build, self.videos, self.labels, config(n_patients=10_000), validate=accept_all)


class TestConsistencyChecks(unittest.TestCase):
    def setUp(self):
        self.videos, self.labels = synthetic(60, seed=5)
        self.cfg = config(n_patients=10)

    def eligible_rows(self, videos):
        """Rows that survive eligibility, so a planted defect reaches the checks."""
        labelled = videos.study_uid.str.strip("'").isin(set(self.labels.study_id))
        return videos.index[(videos.study_type == TTE) & (videos.avi_status == "success") & labelled]

    def test_same_path_under_two_studies_raises_across_batches(self):
        videos = self.videos.copy()
        rows = self.eligible_rows(videos)
        videos.loc[rows[-1], "avi_path"] = videos.loc[rows[0], "avi_path"]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            quiet(bm.build, videos, self.labels, self.cfg, validate=accept_all, batch_size=50)

    def test_exact_duplicate_rows_are_ignored(self):
        doubled = pd.concat([self.videos, self.videos])
        once = quiet(bm.build, self.videos, self.labels, self.cfg, validate=accept_all)
        twice = quiet(bm.build, doubled, self.labels, self.cfg, validate=accept_all, batch_size=50)
        pd.testing.assert_frame_equal(once["videos"], twice["videos"])
        self.assertEqual(once["counts"]["eligible_videos"], twice["counts"]["eligible_videos"])

    def test_study_with_two_patients_raises(self):
        videos = self.videos.copy()
        rows = self.eligible_rows(videos)
        study = videos.loc[rows, "study_uid"].value_counts().index[0]  # has >= 2 eligible videos
        videos.loc[rows[videos.loc[rows, "study_uid"] == study][0], "mrn"] = "9999999"
        with self.assertRaisesRegex(ValueError, "more than one patient"):
            quiet(bm.build, videos, self.labels, self.cfg, validate=accept_all, batch_size=50)

    def test_conflicting_labels_raise(self):
        labels = pd.concat([self.labels, self.labels.head(1).assign(ef=lambda d: d.ef + 5)])
        with self.assertRaises(ValueError):
            quiet(bm.build, self.videos, labels, self.cfg, validate=accept_all)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            quiet(bm.build, self.videos.drop(columns=["predicted_class"]), self.labels, self.cfg, validate=accept_all)

    def test_label_required_without_label_table_raises(self):
        with self.assertRaises(ValueError):
            quiet(bm.build, self.videos, None, self.cfg, validate=accept_all)


class TestVideoFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.dir, name)

    def valid(self, name):
        write_video(self.path(name))
        return self.path(name)

    def test_verify_video_reasons(self):
        empty = self.path("empty.mp4")
        open(empty, "wb").close()
        corrupt = self.path("corrupt.mp4")
        with open(corrupt, "wb") as f:
            f.write(os.urandom(2048))
        spaced = self.valid("has space.mp4")
        self.assertIsNone(bm.verify_video(self.valid("ok.mp4")))
        self.assertEqual(bm.verify_video(empty), "empty")
        self.assertEqual(bm.verify_video(corrupt), "undecodable")
        self.assertEqual(bm.verify_video(self.path("absent.mp4")), "missing")
        self.assertEqual(bm.verify_video(spaced), "whitespace_in_path")

    def test_verify_video_decodes_past_the_first_frame(self):
        from decord import VideoReader, cpu

        path = self.path("tail.mp4")
        write_partly_corrupt_video(path)
        VideoReader(path, ctx=cpu(0), num_threads=1)[0]  # a first-frame check would pass it
        self.assertEqual(bm.verify_video(path), "corrupt_frames")

    def rows(self, patient, study, paths):
        return [(patient, study, TTE, "A4C", p, "success", "20240101") for p in paths]

    def test_bad_videos_dropped_and_patient_falls_back_to_a_valid_study(self):
        empty = self.path("empty.mp4")
        open(empty, "wb").close()
        corrupt = self.path("corrupt.mp4")
        with open(corrupt, "wb") as f:
            f.write(os.urandom(2048))
        rows = (
            self.rows("0000001", "1.2.999.1", [self.valid("a.mp4"), empty, corrupt])
            + self.rows("0000002", "1.2.999.2", [self.path("gone.mp4")])
            + self.rows("0000002", "1.2.999.3", [self.valid("b.mp4")])
            + self.rows("0000003", "1.2.999.4", [self.path("gone2.mp4")])
        )
        videos = pd.DataFrame(rows, columns=META_COLUMNS)
        labels = pd.DataFrame({"study_id": ["1.2.999.1", "1.2.999.2", "1.2.999.3", "1.2.999.4"], "ef": 55.0})

        result = quiet(bm.build, videos, labels, config(n_patients=2), workers=2)
        self.assertEqual(dict(zip(result["studies"].patient_id, result["studies"].study_id)),
                         {"0000001": "1.2.999.1", "0000002": "1.2.999.3"})
        self.assertEqual(set(result["videos"].video_path), {self.path("a.mp4"), self.path("b.mp4")})
        self.assertEqual(sorted(result["excluded"].reason), ["empty", "undecodable"])
        with self.assertRaises(ValueError):
            quiet(bm.build, videos, labels, config(n_patients=3))

    def test_outputs_parse_with_the_echojepa_parser(self):
        rows = []
        for p in range(10):
            rows += self.rows(f"{p:07d}", f"1.2.999.{p}", [self.valid(f"p{p}_{v}.mp4") for v in range(2)])
        videos = pd.DataFrame(rows, columns=META_COLUMNS)
        labels = pd.DataFrame({"study_id": [f"1.2.999.{p}" for p in range(10)], "ef": 50.0})
        cfg = config(n_patients=10)
        result = quiet(bm.build, videos, labels, cfg)
        out = self.path("manifests")
        quiet(bm.write_outputs, result, out, cfg, inputs={}, argv=["test"])

        for split, n_patients in (("train", 7), ("val", 1), ("test", 2)):
            with open(os.path.join(out, f"{split}.csv")) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2 * n_patients)
            for line in lines:
                path, label = line.split(" ")
                self.assertEqual(label, "0")
                self.assertTrue(os.path.isfile(path))
        quiet(bm.smoke_test, out)

        with open(os.path.join(out, "manifest_info.json")) as f:
            info = json.load(f)
        self.assertEqual(info["seed"], 7)
        self.assertEqual(info["counts"]["selected_patients"], 10)
        self.assertEqual(info["outputs"]["train.csv"]["sha256"], bm.sha256(os.path.join(out, "train.csv")))

    def test_cli_seed_override_is_used_and_recorded(self):
        rows = []
        for p in range(10):
            rows += self.rows(f"{p:07d}", f"1.2.999.{p}", [self.valid(f"p{p}.mp4")])
        metadata, labels, config_path, out = (self.path(n) for n in ("meta.csv", "labels.csv", "cfg.yaml", "out"))
        pd.DataFrame(rows, columns=META_COLUMNS).to_csv(metadata, index=False)
        pd.DataFrame({"study_id": [f"1.2.999.{p}" for p in range(10)], "ef": 50.0}).to_csv(labels, index=False)
        with open(config_path, "w") as f:
            yaml.safe_dump(config(n_patients=10), f)

        argv = ["build_manifests.py", "--config", config_path, "--metadata", metadata, "--labels", labels,
                "--out-dir", out, "--seed", "123", "--workers", "1", "--no-smoke-test"]
        with mock.patch.object(sys, "argv", argv):
            quiet(bm.main)

        with open(os.path.join(out, "manifest_info.json")) as f:
            info = json.load(f)
        self.assertEqual(info["seed"], 123)
        self.assertEqual(info["config"]["selection"]["seed"], 123)
        expected = quiet(bm.build, pd.read_csv(metadata, dtype=str), pd.read_csv(labels), config(n_patients=10, seed=123))
        written = pd.read_csv(os.path.join(out, "studies.csv"), dtype={"patient_id": str})
        self.assertEqual(dict(zip(written.patient_id, written.split)),
                         dict(zip(expected["studies"].patient_id, expected["studies"].split)))


class TestLinkEfLabels(unittest.TestCase):
    def tables(self):
        reports = pd.DataFrame(
            [("0000001", "A1", 55.0), ("0000002", "A2", 60.0), ("0000003", "A3", 40.0),
             ("0000003", "A4", 45.0), ("0000004", "A5", 50.0), ("0000004", "A5", 65.0),
             ("0000006", "A6", 50.0), ("0000007", "A6", 50.0)],
            columns=["report_patient", "accession", "label"],
        )
        pacs = pd.DataFrame(
            [("A1", "'1.2.999.1'", "0000001", "20200101"), ("A2", "1.2.999.2", "0000099", "20200101"),
             ("A3", "1.2.999.3", "0000003", "20200101"), ("A4", "1.2.999.3", "0000003", "20200101"),
             ("A5", "1.2.999.5", "0000004", "20200101"), ("A6", "1.2.999.6", "0000006", "20200101")],
            columns=["accession", "study_id", "pacs_patient", "pacs_date"],
        )
        videos = pd.DataFrame(
            [("0000001", "1.2.999.1", "20200101", "/synthetic/0000001/1.2.999.1/0001.mp4"),
             ("0000003", "1.2.999.3", "20200101", "/synthetic/0000003/1.2.999.3/0001.mp4"),
             ("0000004", "1.2.999.5", "20200101", "/synthetic/0000004/1.2.999.5/0001.mp4"),
             ("0000006", "1.2.999.6", "20200101", "/synthetic/0000006/1.2.999.6/0001.mp4")],
            columns=["patient_id", "study_id", "study_date", "video_path"],
        )
        return reports, pacs, videos

    def test_parse_labels_keeps_numbers_in_range_and_counts_every_drop(self):
        raw = pd.Series(["55", " 60.5 ", "100", "0.5", "", "  ", None, "60%", "abc",
                         "0", "0.0", "-5", "100.01", "555"])
        values, dropped = lel.parse_labels(raw)
        self.assertEqual(values.dropna().tolist(), [55.0, 60.5, 100.0, 0.5])
        self.assertEqual(dropped, {"empty": 3, "invalid": 2, "zero": 2, "negative": 1, "above_100": 2})

        values, dropped = lel.parse_labels(pd.Series([55.0, np.nan, 0.0, 120.0]))
        self.assertEqual(values.dropna().tolist(), [55.0])
        self.assertEqual(dropped, {"empty": 1, "invalid": 0, "zero": 1, "negative": 0, "above_100": 1})

    def test_link_keeps_only_unambiguous_verified_studies(self):
        reports, pacs, videos = self.tables()
        out = quiet(lel.link, reports, pacs, videos, min_id_agreement=0.5, check_path_layout=True)
        # A2: patient disagrees between report and PACS. A3/A4: one study, two
        # labels. A5: one accession, two labels. A6: one accession, same label,
        # two patients.
        self.assertEqual(out.study_id.tolist(), ["1.2.999.1"])
        self.assertEqual(out.label.tolist(), [55.0])

    def test_link_fails_when_metadata_disagrees_with_pacs(self):
        reports, pacs, videos = self.tables()
        videos.loc[0, "patient_id"] = "0000077"
        with self.assertRaises(ValueError):
            quiet(lel.link, reports, pacs, videos, min_id_agreement=0.5)

    def test_link_fails_when_id_agreement_is_too_low(self):
        reports, pacs, videos = self.tables()
        with self.assertRaises(ValueError):
            quiet(lel.link, reports, pacs, videos, min_id_agreement=0.999)

    def test_stream_videos_reads_only_the_given_studies_from_any_source(self):
        _, _, videos = self.tables()
        # Source column names, in an order unlike the roles, with quoted study ids
        # and a column stage 1 does not read.
        source = pd.DataFrame({"avi_path": videos.video_path, "predicted_class": "A4C", "date": videos.study_date,
                               "study_uid": "'" + videos.study_id + "'", "mrn": videos.patient_id})
        wanted = {"1.2.999.1", "1.2.999.6"}
        expected = videos[videos.study_id.isin(wanted)].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmp:
            parquet, csv = os.path.join(tmp, "meta.parquet"), os.path.join(tmp, "meta.csv")
            source.to_parquet(parquet, index=False)
            source.to_csv(csv, index=False)
            for src in (source, parquet, csv):
                got = lel.stream_videos(src, CFG["columns"], wanted, batch_size=1)
                pd.testing.assert_frame_equal(got.astype(object), expected.astype(object))


if __name__ == "__main__":
    unittest.main()
