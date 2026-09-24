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
import re
import subprocess
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

META_COLUMNS = ["mrn", "study_uid", "study_type", "predicted_class", "avi_path", "avi_status", "date"]

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIPPED_CONFIG = os.path.join(REPO, "configs", "data", "manifests_tte_10k_ef.yaml")

# The tests run on the shipped configuration, so a change to its split, eligibility rules,
# column mapping or clip setting is tested too. Only the cohort size and the seed are
# scaled to the synthetic data.
CFG = bm.load_config(SHIPPED_CONFIG)
CFG["selection"] = {"n_patients": 100, "seed": 7}


def config(**selection):
    cfg = copy.deepcopy(CFG)
    cfg["selection"].update(selection)
    return cfg


def accept_all(path):
    """Stand-in verifier for tests about selection rather than files."""
    return bm.Check(None)


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


def eligible(videos, labels):
    """The eligible rows of synthetic metadata, worked out independently of the code under
    test, with a `sid` column holding the unquoted study id."""
    videos = videos.assign(sid=videos.study_uid.str.strip("'"))
    ok = (videos.study_type == TTE) & (videos.avi_status == "success") & videos.sid.isin(set(labels.study_id))
    return videos[ok]


def write_video(path, frames=12, fps=30):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), 10 * i % 256, np.uint8))
    writer.release()


def cli_inputs(root, n_patients=10, **selection):
    """Metadata, label and config files for a command-line run over real tiny videos."""
    rows = []
    for p in range(n_patients):
        path = os.path.join(root, f"p{p}.mp4")
        write_video(path)
        rows.append((f"{p:07d}", f"1.2.999.{p}", TTE, "A4C", path, "success", "20240101"))
    metadata, labels, config_path = (os.path.join(root, n) for n in ("meta.csv", "labels.csv", "cfg.yaml"))
    pd.DataFrame(rows, columns=META_COLUMNS).to_csv(metadata, index=False)
    pd.DataFrame({"study_id": [r[1] for r in rows], "ef": 50.0}).to_csv(labels, index=False)
    with open(config_path, "w") as f:
        yaml.safe_dump(config(n_patients=n_patients, **selection), f)
    return metadata, labels, config_path, rows


def run_cli(module, argv):
    with mock.patch.object(sys, "argv", argv):
        quiet(module.main)


def summary_without_timestamp(out_dir):
    with open(os.path.join(out_dir, "summary.md")) as f:
        return [line for line in f if not line.startswith("- Created")]


def new_files_in_repo():
    """Untracked and ignored files of the repository, compiled Python aside; None without git."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--ignored", "--untracked-files=all"],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line[3:] for line in out.splitlines() if line[:2] in ("??", "!!") and "__pycache__" not in line}


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


class TestShippedConfig(unittest.TestCase):
    """The defaults documented in issue #6 match the values set in the shipped YAML configuration."""

    def test_documented_defaults(self):
        cfg = bm.load_config(SHIPPED_CONFIG)
        self.assertEqual(cfg["split"], {"train": 0.7, "val": 0.1, "test": 0.2})
        self.assertEqual(cfg["selection"], {"n_patients": 10_000, "seed": 42})
        self.assertEqual(cfg["eligibility"]["exam_types"], [TTE])
        self.assertEqual(cfg["eligibility"]["status_ok"], ["success"])
        self.assertTrue(cfg["eligibility"]["require_label"])
        self.assertIsNone(cfg["eligibility"]["years"])
        self.assertEqual(cfg["clip"], {"frames_per_clip": 16, "fps": 8})
        self.assertEqual(cfg["columns"], dict(zip(
            ("patient_id", "study_id", "exam_type", "view", "video_path", "avi_status", "study_date"), META_COLUMNS)))


class TestConfigValidation(unittest.TestCase):
    """A malformed configuration fails when it is loaded, with a message naming the problem."""

    CASES = [
        (lambda c: c["split"].update(train=0.8, val=-0.1, test=0.3), r"fraction in \[0, 1\]"),
        (lambda c: c["split"].update(train=1.5, val=-0.25, test=-0.25), r"fraction in \[0, 1\]"),
        (lambda c: c["split"].update(train=0.5), "sum to"),
        (lambda c: c["selection"].update(n_patients=0), "n_patients must be a positive integer"),
        (lambda c: c["selection"].update(n_patients=2.5), "n_patients must be a positive integer"),
        (lambda c: c["selection"].update(seed=-1), "seed must be a non-negative integer"),
        (lambda c: c["selection"].update(n_patients=5), r"split \['val'\] would get no patient"),
        (lambda c: c["eligibility"].update(exam_types=[]), "exam_types must be a non-empty list"),
        (lambda c: c["eligibility"].update(status_ok=[]), "status_ok must be a non-empty list"),
        (lambda c: c["clip"].update(fps=0), "clip.fps must be a positive number"),
        (lambda c: c["clip"].update(frames_per_clip=0), "frames_per_clip must be a positive integer"),
        (lambda c: c["clip"].update(frames_per_clip=16.5), "frames_per_clip must be a positive integer"),
        (lambda c: c["columns"].pop("avi_status"), "no mapping for"),
        (lambda c: c["columns"].update(view=""), "non-empty column name"),
        (lambda c: c["columns"].update(view="mrn"), r"several roles to the same column \['mrn'\]"),
        (lambda c: c["eligibility"].update(require_label="yes"), "require_label must be true or false"),
        (lambda c: c["eligibility"].update(years="2020"), "years must be null"),
        (lambda c: c["eligibility"].update(years=[]), "years must be null"),
        (lambda c: (c["columns"].pop("study_date"), c["eligibility"].update(years=[2020])),
         "years needs columns.study_date"),
        (lambda c: c.update(split=[0.7, 0.1, 0.2]), "`split` must be a mapping"),
        (lambda c: c["labels"].update(low_threshold="40%"), "low_threshold must be null or a number"),
        (lambda c: c["labels"].update(low_threshold=0), "low_threshold must be null or a number"),
    ]

    def test_a_configuration_that_is_not_a_mapping_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.yaml")
            with open(path, "w") as f:
                f.write("- columns\n- split\n")
            with self.assertRaisesRegex(ValueError, "must be a mapping, got list"):
                bm.load_config(path)

    def test_malformed_configurations_are_refused_when_loaded(self):
        shipped = bm.load_config(SHIPPED_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.yaml")
            for change, message in self.CASES:
                with self.subTest(message):
                    cfg = copy.deepcopy(shipped)
                    change(cfg)
                    with open(path, "w") as f:
                        yaml.safe_dump(cfg, f)
                    with self.assertRaisesRegex(ValueError, message):
                        bm.load_config(path)


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
        # Thousands of the chosen patients had several eligible studies and still got one.
        studies_per_patient = eligible(videos, labels).groupby("mrn").sid.nunique()
        self.assertGreater((result["studies"].patient_id.map(studies_per_patient) > 1).sum(), 3000)

    def test_patients_with_several_studies_give_one_study_and_all_its_videos(self):
        rows = eligible(self.videos, self.labels)
        studies = self.result["studies"]
        several = studies[studies.patient_id.map(rows.groupby("mrn").sid.nunique()) > 1]
        self.assertGreater(len(several), 20)  # the multi-study case is really exercised
        self.assertFalse(studies.patient_id.duplicated().any())

        # The kept study is drawn among the patient's studies, not always the same one.
        first = several.patient_id.map(rows.groupby("mrn").sid.min())
        self.assertTrue((several.study_id == first).any() and (several.study_id != first).any())

        # Patients are split exactly; each study keeps all its videos, so video counts follow
        # the studies and only approximate the ratios instead of being forced to them.
        self.assertEqual(studies.split.value_counts().to_dict(), {"train": 70, "val": 10, "test": 20})
        self.assertTrue((studies.n_videos == studies.study_id.map(rows.groupby("sid").size())).all())
        videos = self.result["videos"].split.value_counts().to_dict()
        self.assertEqual(videos, studies.groupby("split").n_videos.sum().to_dict())
        share = {s: n / sum(videos.values()) for s, n in videos.items()}
        self.assertNotEqual(share, {"train": 0.7, "val": 0.1, "test": 0.2})

    def test_same_seed_and_shuffled_rows_give_identical_cohort(self):
        shuffled = self.videos.sample(frac=1, random_state=3)
        labels = self.labels.sample(frac=1, random_state=4)
        again = quiet(bm.build, shuffled, labels, config(), validate=accept_all)
        for key in ("videos", "studies"):
            pd.testing.assert_frame_equal(self.result[key], again[key])

    def test_same_seed_writes_identical_files(self):
        again = quiet(bm.build, self.videos.sample(frac=1, random_state=5), self.labels, config(), validate=accept_all)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            quiet(bm.write_outputs, self.result, a, config(), inputs={}, argv=["test"])
            quiet(bm.write_outputs, again, b, config(), inputs={}, argv=["test"])
            for name in ("train.csv", "val.csv", "test.csv", "videos.csv", "studies.csv", "excluded_videos.csv"):
                self.assertEqual(bm.sha256(os.path.join(a, name)), bm.sha256(os.path.join(b, name)), name)
            self.assertEqual(summary_without_timestamp(a), summary_without_timestamp(b))

    def test_summary_shows_the_low_label_share_only_with_a_threshold(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            quiet(bm.write_outputs, self.result, a, cfg, inputs={}, argv=["test"])
            cfg["labels"]["low_threshold"] = None
            quiet(bm.write_outputs, self.result, b, cfg, inputs={}, argv=["test"])
            self.assertIn("| label < 40 |", "".join(summary_without_timestamp(a)))
            self.assertNotIn("label <", "".join(summary_without_timestamp(b)))

    def test_counts_match_the_data_and_the_written_files(self):
        rows = eligible(self.videos, self.labels)
        with tempfile.TemporaryDirectory() as out:
            quiet(bm.write_outputs, self.result, out, config(), inputs={}, argv=["test"])
            with open(os.path.join(out, "manifest_info.json")) as f:
                info = json.load(f)
            studies = pd.read_csv(os.path.join(out, "studies.csv"), dtype=str)
            lines = {}
            for s in bm.SPLITS:
                with open(os.path.join(out, f"{s}.csv")) as f:
                    lines[s] = len(f.read().splitlines())

        counts = info["counts"]
        self.assertEqual(counts["eligible_videos"], len(rows))
        self.assertEqual(counts["eligible_studies"], rows.sid.nunique())
        self.assertEqual(counts["eligible_patients"], rows.mrn.nunique())
        self.assertEqual(counts["eligibility_funnel"]["metadata_videos"], len(self.videos))
        by_exam = counts["eligibility_by_exam_type"]
        self.assertEqual({k: v["metadata_videos"] for k, v in by_exam.items()}, self.videos.study_type.value_counts().to_dict())
        self.assertEqual((by_exam[TEE]["in_years"], by_exam[STRESS]["eligible_studies"]), (0, 0))
        self.assertEqual(by_exam[TTE]["eligible_studies"], rows.sid.nunique())
        by_view = {k: v["in_years"] for k, v in counts["eligibility_by_view"].items() if v["in_years"]}
        self.assertEqual(by_view, rows.predicted_class.value_counts().to_dict())

        for s, n_patients in (("train", 70), ("val", 10), ("test", 20)):
            self.assertEqual(info["splits"][s]["patients"], n_patients)
            self.assertEqual((studies.split == s).sum(), n_patients)
            self.assertEqual(info["splits"][s]["videos"], lines[s])
            self.assertEqual(info["outputs"][f"{s}.csv"]["rows"], lines[s])
        self.assertEqual(sum(lines.values()), counts["selected_videos"])

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

    def test_a_split_left_without_patients_is_refused_before_any_work(self):
        # 5 patients at 70/10/20 give val round(0.5) = 0 patients.
        validate = mock.Mock(side_effect=accept_all)
        with self.assertRaisesRegex(ValueError, r"split \['val'\] would get no patient"):
            quiet(bm.build, self.videos, self.labels, config(n_patients=5), validate=validate)
        validate.assert_not_called()


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

    def test_repeated_rows_change_no_output_and_no_count(self):
        # Exact copies, and copies that differ only in how the study id is quoted.
        unquoted = self.videos.assign(study_uid=self.videos.study_uid.str.strip("'"))
        repeated = pd.concat([self.videos, self.videos, unquoted])
        once = quiet(bm.build, self.videos, self.labels, self.cfg, validate=accept_all)
        again = quiet(bm.build, repeated, self.labels, self.cfg, validate=accept_all, batch_size=50)
        for key in ("videos", "studies", "excluded"):
            pd.testing.assert_frame_equal(once[key], again[key])
        self.assertEqual(once["counts"], again["counts"])
        self.assertEqual(once["counts"]["eligible_videos"], once["counts"]["eligibility_funnel"]["in_years"])
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            quiet(bm.write_outputs, once, a, self.cfg, inputs={}, argv=["test"])
            quiet(bm.write_outputs, again, b, self.cfg, inputs={}, argv=["test"])
            self.assertEqual(summary_without_timestamp(a), summary_without_timestamp(b))

    def test_rows_without_a_path_are_each_counted(self):
        # Videos that were never converted have no path, so identical rows of them cannot be
        # told apart and are different videos, not repeats.
        unconverted = self.videos.head(1).assign(avi_path=None, avi_status="missing video")
        videos = pd.concat([self.videos, unconverted, unconverted])
        result = quiet(bm.build, videos, self.labels, self.cfg, validate=accept_all, batch_size=50)
        self.assertEqual(result["counts"]["eligibility_funnel"]["metadata_videos"], len(videos))

    def test_label_table_is_held_to_the_stage_1_range(self):
        # A label table from elsewhere gets the check stage 1 applies: an EF in (0, 100].
        labels = self.labels.copy()
        labels.loc[:3, "ef"] = [-5.0, 0.0, 100.5, np.nan]
        out = quiet(bm.load_labels, labels, self.cfg)
        expected = labels.iloc[4:].set_index("study_id").ef.to_dict()
        self.assertEqual(out.set_index("study_id").label.to_dict(), expected)  # kept values exactly

        text = labels.astype({"ef": str})
        text.loc[4, "ef"] = "60%"
        out = quiet(bm.load_labels, text, self.cfg)
        self.assertEqual(set(out.study_id), set(labels.study_id[5:]))

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
        self.assertEqual(bm.verify_video(self.valid("ok.mp4")), bm.Check(None, 12, 30.0))
        self.assertEqual(bm.verify_video(empty).reason, "empty")
        self.assertEqual(bm.verify_video(corrupt).reason, "undecodable")
        self.assertEqual(bm.verify_video(self.path("absent.mp4")).reason, "missing")
        self.assertEqual(bm.verify_video(spaced).reason, "whitespace_in_path")

    def test_verify_video_decodes_past_the_first_frame(self):
        from decord import VideoReader, cpu

        path = self.path("tail.mp4")
        write_partly_corrupt_video(path)
        VideoReader(path, ctx=cpu(0), num_threads=1)[0]  # a first-frame check would pass it
        self.assertEqual(bm.verify_video(path).reason, "corrupt_frames")

    def test_verify_video_keeps_a_video_without_a_usable_fps(self):
        # A decodable video is valid whatever frame rate its decoder reports; an unusable
        # rate is recorded as unknown.
        path = self.valid("ok.mp4")

        class Reader:
            def __init__(self, fps):
                self.fps = fps

            def __len__(self):
                return 12

            def get_avg_fps(self):
                if self.fps is None:
                    raise RuntimeError("no frame rate")
                return self.fps

            def get_batch(self, idx):
                return np.zeros((len(idx), 2, 2, 3), np.uint8)

        for fps in (float("nan"), float("inf"), 0.0, -30.0, None):
            with mock.patch("decord.VideoReader", return_value=Reader(fps)):
                self.assertEqual(bm.verify_video(path), bm.Check(None, 12, None), fps)

    def test_padding_flag_without_a_usable_fps_assumes_every_frame(self):
        # The fallback frame step is 1, so one clip is 16 frames.
        flags = bm.needs_padding(pd.Series([15, 16, 40, None], dtype="Int64"),
                                 pd.Series([np.nan, np.nan, 30.0, np.nan]), {"frames_per_clip": 16, "fps": 8})
        self.assertEqual(flags.tolist(), [True, False, True, pd.NA])

    def test_frame_count_fps_and_padding_flag(self):
        # One clip of 16 frames at 8 fps spans 16 * max(1, ceil(video fps) // 8) frames.
        cases = {"a": (30, 12, True), "b": (25, 47, True), "c": (25, 48, False),
                 "d": (8, 15, True), "e": (8, 16, False), "f": (5, 12, True)}  # name: (fps, frames, needs padding)
        rows = []
        for i, (name, (fps, frames, _)) in enumerate(cases.items()):
            write_video(self.path(f"{name}.mp4"), frames=frames, fps=fps)
            rows += self.rows(f"{i:07d}", f"1.2.999.{i}", [self.path(f"{name}.mp4")])
        labels = pd.DataFrame({"study_id": [r[1] for r in rows], "ef": 50.0})
        result = quiet(bm.build, pd.DataFrame(rows, columns=META_COLUMNS), labels, config(n_patients=len(cases)))

        got = result["videos"].set_index("video_path")
        for name, (fps, frames, padding) in cases.items():
            row = got.loc[self.path(f"{name}.mp4")]
            self.assertEqual((row.n_frames, row.fps, row.needs_padding), (frames, fps, padding), name)
        self.assertTrue((result["videos"].avi_status == "success").all())

    def rows(self, patient, study, paths):
        return [(patient, study, TTE, "A4C", p, "success", "20240101") for p in paths]

    def test_bad_videos_dropped_and_patient_falls_back_to_a_valid_study(self):
        empty = self.path("empty.mp4")
        open(empty, "wb").close()
        corrupt = self.path("corrupt.mp4")
        with open(corrupt, "wb") as f:
            f.write(os.urandom(2048))
        # Patients 4-7 have one valid video each, so every split gets a patient.
        fillers = [self.valid(f"f{p}.mp4") for p in range(4, 8)]
        rows = (
            self.rows("0000001", "1.2.999.1", [self.valid("a.mp4"), empty, corrupt])
            + self.rows("0000002", "1.2.999.2", [self.path("gone.mp4")])
            + self.rows("0000002", "1.2.999.3", [self.valid("b.mp4")])
            + self.rows("0000003", "1.2.999.4", [self.path("gone2.mp4")])
            + [r for p, path in zip(range(4, 8), fillers) for r in self.rows(f"{p:07d}", f"1.2.999.{p + 1}", [path])]
        )
        videos = pd.DataFrame(rows, columns=META_COLUMNS)
        labels = pd.DataFrame({"study_id": sorted(set(r[1] for r in rows)), "ef": 55.0})

        result = quiet(bm.build, videos, labels, config(n_patients=6), workers=2)
        chosen = dict(zip(result["studies"].patient_id, result["studies"].study_id))
        self.assertEqual({p: chosen[p] for p in ("0000001", "0000002")}, {"0000001": "1.2.999.1", "0000002": "1.2.999.3"})
        self.assertNotIn("0000003", chosen)
        self.assertEqual(set(result["videos"].video_path), {self.path("a.mp4"), self.path("b.mp4"), *fillers})
        self.assertEqual(sorted(result["excluded"].reason), ["empty", "undecodable"])
        self.assertEqual(result["counts"]["excluded_by_exam_type"], {TTE: {"empty": 1, "undecodable": 1}})
        self.assertEqual(result["counts"]["excluded_by_view"], {"A4C": {"empty": 1, "undecodable": 1}})
        out = self.path("out")
        quiet(bm.write_outputs, result, out, config(n_patients=6), inputs={}, argv=["test"])
        with open(os.path.join(out, "manifest_info.json")) as f:
            self.assertEqual(json.load(f)["counts"]["excluded_by_exam_type"], {TTE: {"empty": 1, "undecodable": 1}})
        with open(os.path.join(out, "summary.md")) as f:
            summary = f.read()
        self.assertIn("### Excluded videos of the selected studies, by exam type", summary)
        self.assertIn("- Valid video: file exists and is non-empty, every frame decodes", summary)
        self.assertIn(f"| {TTE} | 1 | 1 |", summary)
        self.assertEqual(pd.read_csv(os.path.join(out, "excluded_videos.csv")).exam_type.tolist(), [TTE, TTE])
        verification = result["counts"]["verification"]
        self.assertEqual(verification["patients_without_valid_study"], 1)  # patient 3
        self.assertGreaterEqual(verification["studies_without_valid_video"], 1)
        self.assertEqual({k: verification["failed_by_reason"][k] for k in ("empty", "undecodable")},
                         {"empty": 1, "undecodable": 1})
        self.assertGreaterEqual(verification["failed_by_reason"]["missing"], 1)
        with self.assertRaisesRegex(ValueError, "Only 6 patients"):
            quiet(bm.build, videos, labels, config(n_patients=7))

    def test_videos_without_a_path_are_excluded_as_no_path(self):
        # Null and empty paths reach verification instead of stopping the build; two null
        # paths in one study are two videos, not a path conflict.
        rows = self.rows("0000001", "1.2.999.1", [self.valid("a.mp4"), None, None, ""])
        for p in range(2, 7):
            rows += self.rows(f"{p:07d}", f"1.2.999.{p}", [self.valid(f"p{p}.mp4")])
        # Patient 7's only study has no path at all: it is tried, rejected and reported.
        rows += self.rows("0000007", "1.2.999.7", [None, None])
        labels = pd.DataFrame({"study_id": [f"1.2.999.{p}" for p in range(1, 8)], "ef": 50.0})
        result = quiet(bm.build, pd.DataFrame(rows, columns=META_COLUMNS), labels, config(n_patients=6))

        excluded = result["excluded"]
        self.assertEqual((excluded.study_id.tolist(), excluded.reason.tolist()), (["1.2.999.1"] * 3, ["no_path"] * 3))
        self.assertEqual(len(result["videos"]), 6)
        self.assertEqual(result["counts"]["eligible_videos"], 11)
        self.assertEqual(result["counts"]["eligibility_funnel"]["in_years"], 11)
        self.assertEqual(result["counts"]["verification"], {
            "videos_verified": 11, "failed_by_reason": {"no_path": 5},
            "studies_without_valid_video": 1, "patients_without_valid_study": 1})

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
        quiet(bm.smoke_test, out, bm.SPLITS, cfg["clip"])

        with open(os.path.join(out, "manifest_info.json")) as f:
            info = json.load(f)
        self.assertEqual(info["seed"], 7)
        self.assertEqual(info["counts"]["selected_patients"], 10)
        self.assertEqual(info["outputs"]["train.csv"]["sha256"], bm.sha256(os.path.join(out, "train.csv")))

        # The richer fields live in videos.csv only; the EchoJEPA inputs stay '<path> 0'.
        written = pd.read_csv(os.path.join(out, "videos.csv"))
        for column in ("patient_id", "study_id", "exam_type", "view", "avi_status", "n_frames", "needs_padding"):
            self.assertIn(column, written.columns)
        self.assertTrue(os.path.isfile(os.path.join(out, "summary.md")))

        # An empty manifest shows nothing about whether it can be read, so it fails the smoke test.
        open(os.path.join(out, "val.csv"), "w").close()
        with self.assertRaisesRegex(RuntimeError, "val.csv is empty"):
            quiet(bm.smoke_test, out, bm.SPLITS, cfg["clip"])

    def test_smoke_test_loads_each_special_case_with_the_pretraining_sampling(self):
        from src.datasets.video_dataset import VideoDataset

        clip = CFG["clip"]
        paths = {name: self.path(f"{name}.mp4") for name in ("first", "slow", "unknown", "short", "plain")}
        write_video(paths["first"], frames=60), write_video(paths["plain"], frames=60)
        write_video(paths["slow"], frames=20, fps=5), write_video(paths["unknown"], frames=60)
        write_video(paths["short"], frames=12)
        out = self.path("manifests")
        os.makedirs(out)
        # videos.csv as build_manifests writes it; the smoke test picks its cases from it.
        cases = [("first", 30.0, False), ("slow", 5.0, False), ("unknown", np.nan, False), ("short", 30.0, True),
                 ("plain", 30.0, False)]
        pd.DataFrame([(paths[n], "train", fps, pad) for n, fps, pad in cases],
                     columns=["video_path", "split", "fps", "needs_padding"]).to_csv(os.path.join(out, "videos.csv"), index=False)
        with open(os.path.join(out, "train.csv"), "w") as f:
            f.writelines(f"{paths[n]} 0\n" for n, *_ in cases)

        loaded = []

        def record(ds, index):
            loaded.append((ds.samples[index], ds.fps, ds.frame_step, ds.dataset_fpcs))
            return [np.zeros((clip["frames_per_clip"], 2, 2, 3))], 0, [np.arange(clip["frames_per_clip"])]

        with mock.patch.object(VideoDataset, "get_item_video", autospec=True, side_effect=record):
            quiet(bm.smoke_test, out, ["train"], clip)
        self.assertEqual({p for p, *_ in loaded}, {paths[n] for n in ("first", "slow", "unknown", "short")})
        self.assertEqual({tuple(x[1:3]) + (tuple(x[3]),) for x in loaded}, {(clip["fps"], None, (clip["frames_per_clip"],))})

        def fail_below_clip_fps(ds, index):
            if ds.samples[index] == paths["slow"]:
                raise AssertionError("frame step 0")
            return record(ds, index)

        with mock.patch.object(VideoDataset, "get_item_video", autospec=True, side_effect=fail_below_clip_fps):
            with self.assertRaisesRegex(RuntimeError, r"a frame rate below 8 fps \(1 video\), sampling 16 frames at 8 fps: .*slow\.mp4"):
                quiet(bm.smoke_test, out, ["train"], clip)

    def test_cli_seed_override_is_used_and_recorded(self):
        metadata, labels, config_path, _ = cli_inputs(self.dir)
        out = self.path("out")
        run_cli(bm, ["build_manifests.py", "--config", config_path, "--metadata", metadata, "--labels", labels,
                     "--out-dir", out, "--seed", "123", "--workers", "1", "--no-smoke-test"])

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

    def test_link_fails_when_the_metadata_lacks_the_patient_or_date(self):
        # A missing value is a mismatch, not a comparison to skip.
        for column in ("patient_id", "study_date"):
            reports, pacs, videos = self.tables()
            videos = videos.astype({"patient_id": "string", "study_date": "string"})  # as stream_videos returns them
            videos.loc[0, column] = pd.NA
            with self.assertRaisesRegex(ValueError, "1 studies fail", msg=column):
                quiet(lel.link, reports, pacs, videos, min_id_agreement=0.5)

    def test_stage_1_needs_the_study_date_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata, _, config_path, _ = cli_inputs(tmp, n_patients=6)
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            del cfg["columns"]["study_date"]
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f)
            # Fails before reading any export: these files are not even valid inputs.
            with self.assertRaisesRegex(ValueError, "stage 1 needs columns.study_date"):
                run_cli(lel, ["link_ef_labels.py", "--config", config_path, "--reports", metadata, "--pacs", metadata,
                              "--metadata", metadata, "--out", os.path.join(tmp, "study_labels.csv")])

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


class TestRepositoryHygiene(unittest.TestCase):
    """No real patient data or paths reach the repository."""

    # The files of this pipeline that are committed.
    PIPELINE_FILES = ("configs/data/manifests_tte_10k_ef.yaml", "data/build_manifests.py", "data/link_ef_labels.py",
                      "data/README.md", "tests/data/__init__.py", "tests/data/test_build_manifests.py")
    # Absolute paths under the roots real data lives on, and object-store URLs.
    REAL_PATH = re.compile(r"(?<![\w.])/(?:media|volume|mnt|home|root|Users|nfs|net)/|\bs3:/{2}")
    # DICOM UIDs have many components; the synthetic ones here have at most five.
    REAL_UID = re.compile(r"\b\d+(?:\.\d+){5,}\b")

    def test_pipeline_files_hold_no_real_paths_or_study_ids(self):
        for name in self.PIPELINE_FILES:
            with open(os.path.join(REPO, name)) as f:
                text = f.read()
            self.assertIsNone(self.REAL_PATH.search(text), f"{name}: absolute data path")
            self.assertIsNone(self.REAL_UID.search(text), f"{name}: study UID")

    def test_patient_tables_copied_into_the_repository_are_ignored_by_git(self):
        if new_files_in_repo() is None:
            self.skipTest("not a git checkout")
        for name in ("report_export.csv", "data/pacs_index.csv", "tests/data/metadata.parquet", "manifests/train.csv"):
            ignored = subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO).returncode == 0
            self.assertTrue(ignored, f"{name} is not ignored by .gitignore")

    def test_outputs_inside_the_repository_are_refused(self):
        inside = os.path.join(REPO, "data", "manifests_must_not_exist")
        with tempfile.TemporaryDirectory() as tmp:
            metadata, labels, config_path, _ = cli_inputs(tmp, n_patients=2)
            with self.assertRaisesRegex(SystemExit, "inside the code repository"):
                run_cli(bm, ["build_manifests.py", "--config", config_path, "--metadata", metadata,
                             "--labels", labels, "--out-dir", inside, "--workers", "1"])
            with self.assertRaisesRegex(SystemExit, "inside the code repository"):
                run_cli(lel, ["link_ef_labels.py", "--config", config_path, "--reports", metadata, "--pacs", metadata,
                              "--metadata", metadata, "--out", os.path.join(inside, "study_labels.csv")])
        self.assertFalse(os.path.exists(inside))

    def test_cli_run_creates_no_files_in_the_repository(self):
        before = new_files_in_repo()
        if before is None:
            self.skipTest("not a git checkout")
        with tempfile.TemporaryDirectory() as tmp:
            metadata, labels, config_path, _ = cli_inputs(tmp)
            run_cli(bm, ["build_manifests.py", "--config", config_path, "--metadata", metadata, "--labels", labels,
                         "--out-dir", os.path.join(tmp, "out"), "--workers", "1", "--no-smoke-test"])
        self.assertEqual(new_files_in_repo(), before)

    def test_summary_holds_no_paths_or_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata, labels, config_path, rows = cli_inputs(tmp)
            out = os.path.join(tmp, "out")
            run_cli(bm, ["build_manifests.py", "--config", config_path, "--metadata", metadata, "--labels", labels,
                         "--out-dir", out, "--workers", "1", "--no-smoke-test"])
            with open(os.path.join(out, "summary.md")) as f:
                summary = f.read()
        self.assertIn("## Splits", summary)
        for secret in (tmp, "meta.csv", "labels.csv", "cfg.yaml", *(r[0] for r in rows), *(r[1] for r in rows)):
            self.assertNotIn(secret, summary)


if __name__ == "__main__":
    unittest.main()
