# tests/data/test_build_archive_manifests.py

"""Tests for the archive mode of build_manifests.py (`selection.mode: all_studies`): every eligible
study of every patient, the patient split anchored on an earlier build, the resumable verification
cache, the master video index and the inclusive and B-mode training arms.

Synthetic metadata only: identifiers, dates and paths are made up, and the videos that are really
decoded are tiny clips written to a temporary directory.
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
from scan_acquisition import SCANNER_VERSION

TTE = "ECHO CARDIAQUE AVEC DOPPLER"
TEE = "ETO SOP ANESTHESIE"
VIEWS = ["A4C", "A2C", "PLAX", "PSAX", "OTHER"]
SCANNERS = [("GE Vingmed Ultrasound", "Vivid E95"), ("Philips Medical Systems", "EPIQ 7C"),
            ("SIEMENS", "Acuson SC2000")]
META_COLUMNS = [
    "mrn",
    "study_uid",
    "study_type",
    "predicted_class",
    "avi_path",
    "avi_status",
    "date",
    "dcm_path",
    "brand",
    "model",
]

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVE_CONFIG = os.path.join(REPO, "configs", "data", "manifests_tte_archive.yaml")
CFG = bm.load_config(ARCHIVE_CONFIG)


def config(**selection):
    cfg = copy.deepcopy(CFG)
    cfg["selection"].update(selection)
    return cfg


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def run_cli(argv):
    with mock.patch.object(sys, "argv", ["build_manifests.py", *argv]):
        quiet(bm.main)


def accept_all(path):
    return bm.Check(None, 40, 30.0)


def metadata(patients, seed=0, root="/synthetic/videos", exam_mix=0.85):
    """Metadata shaped like the real export, with made-up values. Each patient's rows come from a
    random stream of its own, so adding patients leaves the others' rows unchanged.

    Patients have 1-4 studies (longitudinal), mostly TTE; 1-5 videos each over all views, a few
    with a failed conversion. Study ids carry literal quotes, as the real export does.
    """
    rows = []
    for p in patients:
        rng = np.random.default_rng([seed, p])
        pid = f"{p:07d}"
        for s in range(rng.integers(1, 5)):
            sid = f"1.2.999.{p}.{s}"
            exam = TTE if rng.random() < exam_mix else TEE
            date = f"20{rng.integers(10, 26):02d}{rng.integers(1, 13):02d}01"
            brand, model = SCANNERS[rng.integers(len(SCANNERS))]
            for v in range(rng.integers(1, 6)):
                status = "success" if rng.random() < 0.95 else "missing video"
                rows.append(
                    (
                        pid,
                        f"'{sid}'",
                        exam,
                        VIEWS[rng.integers(len(VIEWS))],
                        f"{root}/{pid}/{sid}/{v:04d}.avi",
                        status,
                        date,
                        f"/synthetic/dicom/{pid}/{sid}/{v:04d}.dcm",
                        brand,
                        model,
                    )
                )
    return pd.DataFrame(rows, columns=META_COLUMNS)


def eligible(meta):
    """The eligible rows, worked out independently of the code under test."""
    ok = (meta.study_type == TTE) & (meta.avi_status == "success")
    return meta[ok].assign(sid=meta.study_uid.str.strip("'"))


def legacy_anchor(meta, n=100, absent=(), seed=3):
    """A 10k-style studies.csv: one study for each of the first `n` eligible patients, split 70/10/20,
    plus patients in `absent` that have no eligible study."""
    rows = eligible(meta)
    patients = sorted(rows.mrn.unique())[:n] + list(absent)
    split = np.random.default_rng(seed).permutation(
        ["train"] * round(len(patients) * 0.7)
        + ["val"] * round(len(patients) * 0.1)
        + ["test"] * (len(patients) - round(len(patients) * 0.7) - round(len(patients) * 0.1))
    )
    study = rows.groupby("mrn").sid.min()
    return pd.DataFrame(
        {
            "study_id": [study.get(p, f"1.2.888.{i}") for i, p in enumerate(patients)],
            "split": split,
            "patient_id": patients,
            "exam_type": TTE,
            "label": 55.0,
        }
    )


def as_read(df):
    """A table as `read_anchor` returns it from CSV: every column as text."""
    return df.astype(str)


def assign(meta, anchor, cfg=None, **kwargs):
    return quiet(bm.build_archive, meta, None, cfg or config(), as_read(anchor), assign_only=True, **kwargs)


def acquisition_index(meta, unscanned_every=9):
    """An acquisition index for the metadata's DICOMs, cycling through the categories; every
    `unscanned_every`-th DICOM is left out of the index (never scanned)."""
    kinds = [
        ("bmode", "header", True, False, False, False, False),
        ("color_flow", "header", True, True, False, False, False),
        ("bmode", "header", True, False, False, False, False),
        ("spectral_doppler", "header", True, False, True, False, False),
        ("other", "header", None, None, None, None, None),
        ("unknown", "unknown", None, None, None, None, None),
        ("bmode", "unknown", True, False, False, False, False),
        ("mmode", "header", True, False, False, False, True),
    ]
    rows = []
    for i, path in enumerate(sorted(meta.dcm_path.dropna().unique())):
        if i % unscanned_every == unscanned_every - 1:
            continue
        cat, conf, tissue, color, pw, cw, mm = kinds[i % len(kinds)]
        rows.append(
            {
                "dcm_path": path,
                "source_size": 1000,
                "source_mtime_ns": 1,
                "scanner_version": SCANNER_VERSION,
                "read_status": "ok",
                "failure_reason": None,
                "acq_category": cat,
                "acq_confidence": conf,
                "acq_rule": f"rule_{cat}",
                "has_tissue_2d": tissue,
                "has_color_flow": color,
                "has_pw_doppler": pw,
                "has_cw_doppler": cw,
                "has_mmode": mm,
            }
        )
    out = pd.DataFrame(rows)
    return out.astype({f: "boolean" for f in bm.ACQ_FLAGS})


def new_files_in_repo():
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--ignored", "--untracked-files=all"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line[3:] for line in out.splitlines() if line[:2] in ("??", "!!") and "__pycache__" not in line}


def write_video(path, frames=40, fps=30):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), 10 * i % 256, np.uint8))
    writer.release()


class TestArchiveConfig(unittest.TestCase):
    def test_shipped_archive_config(self):
        cfg = bm.load_config(ARCHIVE_CONFIG)
        self.assertEqual(cfg["selection"], {"mode": "all_studies", "seed": 42})
        self.assertEqual(cfg["split"], {"train": 0.7, "val": 0.1, "test": 0.2})
        self.assertFalse(cfg["eligibility"]["require_label"])
        self.assertIsNone(cfg["eligibility"]["years"])
        self.assertEqual(cfg["eligibility"]["exam_types"], [TTE])
        self.assertEqual(
            {k: cfg["columns"][k] for k in ("dicom_path", "vendor", "model")},
            {"dicom_path": "dcm_path", "vendor": "brand", "model": "model"},
        )
        # The cohort definition is the 10k one apart from the mode, labels and the extra roles.
        tenk = bm.load_config(os.path.join(REPO, "configs", "data", "manifests_tte_10k_ef.yaml"))
        self.assertEqual(cfg["clip"], tenk["clip"])
        self.assertEqual({k: v for k, v in cfg["columns"].items() if k in tenk["columns"]}, tenk["columns"])

    def test_mode_specific_keys_are_checked(self):
        cases = [
            (lambda c: c["selection"].update(n_patients=10), "n_patients must be absent or null"),
            (lambda c: c["selection"].update(mode="every_study"), "selection.mode must be one of"),
            (lambda c: c["selection"].pop("seed"), "seed must be a non-negative integer"),
            (lambda c: c["columns"].update(vendor="mrn"), "several roles to the same column"),
            (lambda c: c["columns"].update(dicom_path=""), "non-empty column name"),
        ]
        for change, message in cases:
            with self.subTest(message):
                cfg = copy.deepcopy(CFG)
                change(cfg)
                with self.assertRaisesRegex(ValueError, message):
                    bm.check_config(cfg)
        cfg = copy.deepcopy(CFG)
        cfg["selection"]["n_patients"] = None  # explicitly null is the same as absent
        bm.check_config(cfg)

    def test_one_study_mode_is_the_default_and_still_needs_n_patients(self):
        cfg = copy.deepcopy(CFG)
        del cfg["selection"]["mode"]
        with self.assertRaisesRegex(ValueError, "n_patients must be a positive integer"):
            bm.check_config(cfg)
        cfg["selection"]["mode"] = "one_study_per_patient"
        cfg["selection"]["n_patients"] = 100
        bm.check_config(cfg)


class TestAnchor(unittest.TestCase):
    def test_studies_csv_anchor_keeps_leading_zeros_and_gets_origin_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "studies.csv")
            pd.DataFrame(
                {
                    "study_id": ["'1.2.1'", "1.2.2", "1.2.3"],
                    "split": ["train", "val", "train"],
                    "patient_id": ["0000123", "NA", "0000123"],
                }
            ).to_csv(path, index=False)
            anchor = bm.parse_anchor(bm.read_anchor(path))
        self.assertEqual(anchor.values.tolist(), [["0000123", "train", "legacy"], ["NA", "val", "legacy"]])

    def test_patient_splits_anchor_keeps_its_origins(self):
        table = pd.DataFrame(
            {
                "patient_id": ["'001'", "002"],
                "split": ["test", "train"],
                "origin": ["new", "legacy"],
                "in_archive": ["True", "False"],
            }
        )
        self.assertEqual(bm.parse_anchor(table).values.tolist(), [["001", "test", "new"], ["002", "train", "legacy"]])

    def test_ambiguous_or_malformed_anchors_raise(self):
        base = {"study_id": ["1", "2"], "split": ["train", "val"], "patient_id": ["0000001", "0000001"]}
        with self.assertRaisesRegex(ValueError, "more than one split"):
            bm.parse_anchor(pd.DataFrame(base))
        with self.assertRaisesRegex(ValueError, "more than one origin"):
            bm.parse_anchor(
                pd.DataFrame({"patient_id": ["1", "1"], "split": ["val", "val"], "origin": ["new", "legacy"]})
            )
        with self.assertRaisesRegex(ValueError, "split outside"):
            bm.parse_anchor(pd.DataFrame({**base, "split": ["train", "holdout"], "patient_id": ["1", "2"]}))
        with self.assertRaisesRegex(ValueError, "no patient_id"):
            bm.parse_anchor(pd.DataFrame({**base, "patient_id": ["1", None]}))
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            bm.parse_anchor(pd.DataFrame({"patient_id": ["1"], "split": ["train"]}))


class TestAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = metadata(range(400), seed=1)
        cls.anchor = legacy_anchor(cls.meta, n=100, absent=["9000001", "9000002", "9000003"])
        cls.result = assign(cls.meta, cls.anchor)

    def splits(self, result=None):
        ps = (result or self.result)["patient_splits"]
        return dict(zip(ps.patient_id, ps.split))

    def test_every_eligible_study_of_every_patient_is_kept_without_labels(self):
        rows = eligible(self.meta)
        studies = self.result["studies"]
        self.assertEqual(set(studies.study_id), set(rows.sid))
        self.assertGreater((rows.groupby("mrn").sid.nunique() > 1).sum(), 100)  # longitudinal patients exist
        self.assertEqual(studies.groupby("patient_id").size().to_dict(), rows.groupby("mrn").sid.nunique().to_dict())
        self.assertEqual(
            studies.set_index("study_id").n_videos_metadata.to_dict(), rows.groupby("sid").size().to_dict()
        )
        self.assertNotIn("label", studies)
        # Every study of a patient follows the patient's split.
        self.assertTrue((studies.groupby("patient_id").split.nunique() == 1).all())
        self.assertTrue((studies.split == studies.patient_id.map(self.splits())).all())

    def test_require_label_false_keeps_unlabelled_studies_and_labels_are_attached(self):
        rows = eligible(self.meta)
        labelled = sorted(rows.sid.unique())[::2]
        labels = pd.DataFrame({"study_id": labelled, "ef": 55.0})
        result = quiet(bm.build_archive, self.meta, labels, config(), as_read(self.anchor), assign_only=True)
        studies = result["studies"]
        self.assertEqual(len(studies), rows.sid.nunique())
        self.assertEqual(set(studies.study_id[studies.label.notna()]), set(labelled))
        self.assertEqual(sum(result["counts"]["assignment"]["labelled_studies"].values()), len(labelled))
        pd.testing.assert_frame_equal(result["patient_splits"], self.result["patient_splits"])

    def test_exact_patient_targets_over_all_eligible_patients(self):
        n = eligible(self.meta).mrn.nunique()
        counts = self.result["counts"]["assignment"]
        expected = dict(zip(bm.SPLITS, bm.split_sizes(n, [0.7, 0.1, 0.2])))
        self.assertEqual(counts["patients"], expected)
        self.assertEqual(counts["targets"], expected)
        ps = self.result["patient_splits"]
        self.assertEqual(ps[ps.in_archive].split.value_counts().to_dict(), expected)
        new = ps[ps.origin == "new"].split.value_counts().to_dict()
        legacy = ps[(ps.origin == "legacy") & ps.in_archive].split.value_counts().to_dict()
        self.assertEqual({s: new.get(s, 0) + legacy.get(s, 0) for s in bm.SPLITS}, expected)

    def test_legacy_assignments_are_preserved_and_absent_patients_counted(self):
        anchor = dict(zip(self.anchor.patient_id, self.anchor.split))
        splits = self.splits()
        self.assertEqual({p: splits[p] for p in anchor}, anchor)
        ps = self.result["patient_splits"].set_index("patient_id")
        self.assertTrue((ps.loc[list(anchor), "origin"] == "legacy").all())
        absent = self.result["anchor_absent"]
        self.assertEqual(sorted(absent.patient_id), ["9000001", "9000002", "9000003"])
        self.assertFalse(ps.loc[["9000001", "9000002", "9000003"], "in_archive"].any())
        counts = self.result["counts"]["assignment"]
        self.assertEqual(sum(counts["anchored_absent"]["legacy"].values()), 3)
        self.assertEqual(sum(counts["anchored_present"]["legacy"].values()), 100)
        self.assertEqual(counts["anchor_patients"], 103)

    def test_identical_mapping_under_metadata_row_shuffles(self):
        shuffled = self.meta.sample(frac=1, random_state=5)
        again = assign(shuffled, self.anchor.sample(frac=1, random_state=6), batch_size=97)
        pd.testing.assert_frame_equal(self.result["patient_splits"], again["patient_splits"])
        pd.testing.assert_frame_equal(self.result["studies"], again["studies"])

    def test_expansion_keeps_every_earlier_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "patient_splits.csv")
            self.result["patient_splits"].to_csv(path, index=False)
            previous = bm.read_anchor(path)
        # A later export adds patients (and a legacy patient who was absent comes back).
        bigger = pd.concat(
            [metadata(range(400, 520), seed=1), self.meta, metadata([9000001], seed=1).assign(mrn="9000001")],
            ignore_index=True,
        )
        later = quiet(bm.build_archive, bigger, None, config(), previous, assign_only=True)
        before, after = self.splits(), self.splits(later)
        self.assertEqual({p: after[p] for p in before}, before)
        ps = later["patient_splits"].set_index("patient_id")
        self.assertEqual(ps.loc["9000001"].tolist(), [self.splits()["9000001"], "legacy", True])
        origins = self.result["patient_splits"].set_index("patient_id").origin
        self.assertTrue((ps.loc[origins.index, "origin"] == origins).all())
        counts = later["counts"]["assignment"]
        self.assertEqual(
            counts["patients"], dict(zip(bm.SPLITS, bm.split_sizes(eligible(bigger).mrn.nunique(), [0.7, 0.1, 0.2])))
        )
        self.assertGreater(sum(counts["anchored_present"]["new"].values()), 0)

    def test_new_patients_use_their_own_random_stream(self):
        # A different seed reorders only the new patients; anchored patients never move.
        other = assign(self.meta, self.anchor, cfg=config(seed=43))
        a, b = self.splits(), self.splits(other)
        self.assertEqual({p: b[p] for p in self.anchor.patient_id}, {p: a[p] for p in self.anchor.patient_id})
        self.assertNotEqual(a, b)
        # The stream is not one of the one-study stages.
        rng = np.random.default_rng(np.random.SeedSequence(42, spawn_key=(bm.NEW_PATIENT_STREAM,)))
        self.assertNotIn(bm.NEW_PATIENT_STREAM, range(len(bm.STAGES)))
        self.assertNotEqual(rng.random(), bm.stage_rngs(42)["patients"].random())

    def test_an_anchor_that_overfills_a_split_raises(self):
        anchor = self.anchor.assign(split="val")
        with self.assertRaisesRegex(ValueError, r"exceed the target of split \['val'\]"):
            assign(self.meta, anchor)

    def test_assign_only_passes_the_assignment_checks(self):
        ps = self.result["patient_splits"].copy()
        ps.loc[ps.patient_id == self.anchor.patient_id.iloc[0], "split"] = (
            "test" if self.anchor.split.iloc[0] != "test" else "val"
        )
        studies = self.result["studies"].assign(split=lambda d: d.patient_id.map(ps.set_index("patient_id").split))
        with self.assertRaisesRegex(AssertionError, "lost its anchor split"):
            bm.check_archive(studies, ps, bm.parse_anchor(as_read(self.anchor)))


class TestAssignOnlyCli(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.meta = metadata(range(60), seed=2)
        self.metadata = os.path.join(self.dir, "meta.parquet")
        self.meta.to_parquet(self.metadata, index=False)
        self.anchor = os.path.join(self.dir, "studies.csv")
        legacy_anchor(self.meta, n=20, absent=["0999999"]).to_csv(self.anchor, index=False)
        self.out = os.path.join(self.dir, "out")

    def args(self, *extra):
        return [
            "--config",
            ARCHIVE_CONFIG,
            "--metadata",
            self.metadata,
            "--anchor-splits",
            self.anchor,
            "--out-dir",
            self.out,
            *extra,
        ]

    def test_assign_only_writes_the_mapping_studies_and_reports(self):
        run_cli(self.args("--assign-only"))
        self.assertEqual(
            sorted(os.listdir(self.out)),
            ["BUILD_COMPLETE.json", "anchor_absent_patients.csv", "manifest_info.json", "outputs.sha256",
             "patient_splits.csv", "studies.csv", "summary.md"],
        )
        ps = pd.read_csv(os.path.join(self.out, "patient_splits.csv"), dtype=str)
        self.assertEqual(list(ps.columns), ["patient_id", "split", "origin", "in_archive"])
        self.assertTrue(ps.patient_id.str.fullmatch(r"\d{7}").all())  # leading zeros kept
        self.assertIn("0999999", set(ps.patient_id[ps.in_archive == "False"]))
        studies = pd.read_csv(os.path.join(self.out, "studies.csv"), dtype=str)
        self.assertEqual(
            list(studies.columns),
            ["study_id", "patient_id", "split", "origin", "study_date", "exam_type", "n_videos_metadata"],
        )
        self.assertEqual(set(studies.study_id), set(eligible(self.meta).sid))
        absent = pd.read_csv(os.path.join(self.out, "anchor_absent_patients.csv"), dtype=str)
        self.assertEqual(absent.patient_id.tolist(), ["0999999"])
        with open(os.path.join(self.out, "manifest_info.json")) as f:
            info = json.load(f)
        self.assertTrue(info["assign_only"])
        self.assertEqual(info["inputs"]["anchor_splits"]["sha256"], bm.sha256(self.anchor))
        self.assertEqual(
            info["outputs"]["patient_splits.csv"]["sha256"], bm.sha256(os.path.join(self.out, "patient_splits.csv"))
        )
        with open(os.path.join(self.out, "summary.md")) as f:
            summary = f.read()
        self.assertIn("## Patient split", summary)
        for secret in (self.dir, *self.meta.mrn, *self.meta.study_uid.str.strip("'"), "0999999"):
            self.assertNotIn(secret, summary)

    def test_assign_only_is_reproducible_and_matches_the_full_build(self):
        run_cli(self.args("--assign-only"))
        again = os.path.join(self.dir, "again")
        run_cli(self.args("--assign-only")[:-2] + [again, "--assign-only"])
        for name in ("patient_splits.csv", "studies.csv", "anchor_absent_patients.csv"):
            self.assertEqual(bm.sha256(os.path.join(self.out, name)), bm.sha256(os.path.join(again, name)), name)
        full = quiet(bm.build_archive, self.meta, None, config(), bm.read_anchor(self.anchor), validate=accept_all)
        pd.testing.assert_frame_equal(
            pd.read_csv(os.path.join(self.out, "patient_splits.csv"), dtype=str), full["patient_splits"].astype(str)
        )

    def test_dry_run_prints_split_counts_and_writes_nothing(self):
        out = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["build_manifests.py", *self.args("--dry-run")]),
            contextlib.redirect_stdout(out),
        ):
            bm.main()
        self.assertIn("[split]", out.getvalue())
        self.assertIn("anchored legacy", out.getvalue())
        self.assertFalse(os.path.exists(self.out))

    def test_refused_option_combinations(self):
        cases = [
            (
                ["--config", ARCHIVE_CONFIG, "--metadata", self.metadata, "--out-dir", self.out],
                "--anchor-splits is required",
            ),
            (
                self.args("--assign-only", "--verify-cache", os.path.join(self.dir, "cache")),
                "--assign-only verifies nothing",
            ),
            (self.args("--dry-run", "--verify-limit", "3"), "--dry-run verifies and writes nothing"),
            (self.args("--allow-partial-acquisition"), "--allow-partial-acquisition and .* need --acquisition-index"),
            (self.args("--accept-acquisition-read-errors", "3"), "--accept-acquisition-read-errors need"),
            (self.args("--verify-limit", "-1"), "--verify-limit must be 0 or more"),
            (self.args("--exclude-videos", "black.csv"), "--exclude-videos and --exclude-reason go together"),
            (self.args("--exclude-reason", "deid_black"), "--exclude-videos and --exclude-reason go together"),
            (self.args("--exclude-videos", "black.csv", "--exclude-reason", "Deid-Black"), "lower-case letters"),
            (
                self.args("--verify-cache", os.path.join(REPO, "data", "cache_must_not_exist")),
                "inside the code repository",
            ),
            (
                [
                    "--config",
                    ARCHIVE_CONFIG,
                    "--metadata",
                    self.metadata,
                    "--anchor-splits",
                    self.anchor,
                    "--out-dir",
                    os.path.dirname(self.anchor),
                    "--overwrite",
                    "--assign-only",
                ],
                "lies inside --out-dir",
            ),
        ]
        for argv, message in cases:
            with self.subTest(message), self.assertRaisesRegex(SystemExit, message):
                run_cli(argv)
        self.assertFalse(os.path.exists(os.path.join(REPO, "data", "cache_must_not_exist")))

    def test_non_empty_out_dir_needs_overwrite(self):
        run_cli(self.args("--assign-only"))
        with self.assertRaisesRegex(SystemExit, "not empty"):
            run_cli(self.args("--assign-only"))
        run_cli(self.args("--assign-only", "--overwrite"))

    def test_one_study_mode_refuses_archive_options(self):
        tenk = os.path.join(REPO, "configs", "data", "manifests_tte_10k_ef.yaml")
        with self.assertRaisesRegex(SystemExit, "only apply to selection.mode all_studies"):
            run_cli(
                ["--config", tenk, "--metadata", self.metadata, "--out-dir", self.out, "--anchor-splits", self.anchor]
            )


class TestVerifyCache(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.cache = os.path.join(self.dir, "cache")
        self.paths = []
        for i in range(10):
            path = os.path.join(self.dir, f"v{i}.avi")
            with open(path, "wb") as f:
                f.write(b"x" * (i + 1))
            self.paths.append(path)
        self.calls = []

    def validate(self, path):
        self.calls.append(path)
        return bm.Check(None, 12, 30.0) if os.path.exists(path) else bm.Check("missing")

    def verify(self, paths=None, **kwargs):
        self.calls = []
        kwargs.setdefault("chunk", 3)
        return quiet(bm.verify_archive, sorted(paths or self.paths), self.validate, self.cache, **kwargs)

    def test_cached_results_are_reused(self):
        first, stats = self.verify()
        self.assertEqual(sorted(self.calls), sorted(self.paths))
        self.assertEqual((stats["verified_this_run"], stats["reused_from_cache"], stats["pending"]), (10, 0, 0))
        self.assertEqual(len([f for f in os.listdir(self.cache) if f.endswith(".parquet")]), 4)  # chunks of 3
        second, stats = self.verify()
        self.assertEqual(self.calls, [])
        self.assertEqual((stats["verified_this_run"], stats["reused_from_cache"]), (0, 10))
        pd.testing.assert_frame_equal(first.sort_index(), second.sort_index())
        row = second.loc[self.paths[0]]
        self.assertTrue(pd.isna(row.reason))
        self.assertEqual((row.n_frames, row.fps), (12, 30.0))

    def test_only_cached_videos_are_stat_ed_up_front(self):
        # A cold stat of millions of network paths takes hours: pending videos are stat'ed by their worker,
        # right before decoding, and only cached entries are checked up front.
        stat_calls = []
        real = bm.stat_paths

        def recording(paths, *args, **kwargs):
            stat_calls.append(len(paths))
            return real(paths, *args, **kwargs)

        with mock.patch.object(bm, "stat_paths", recording):
            _, stats = self.verify(limit=4)
            self.assertEqual(stat_calls, [0])
            self.assertEqual(stats["verified_this_run"], 4)
            stat_calls.clear()
            _, stats = self.verify()
        self.assertEqual(stat_calls, [4])  # the four cached entries, nothing else
        self.assertEqual((stats["reused_from_cache"], stats["verified_this_run"]), (4, 6))
        # The fingerprint stored for a video verified by its worker is the file's own.
        cache = bm.read_verify_cache(self.cache).set_index("video_path")
        for path in self.paths:
            st = os.stat(path)
            self.assertEqual((cache.loc[path, "size"], cache.loc[path, "mtime_ns"]), (st.st_size, st.st_mtime_ns))

    def test_a_changed_file_or_verifier_version_is_verified_again(self):
        self.verify()
        with open(self.paths[0], "ab") as f:
            f.write(b"more")  # size changes
        os.utime(self.paths[1], ns=(1, 1))  # only the modification time changes
        _, stats = self.verify()
        self.assertEqual(sorted(self.calls), sorted(self.paths[:2]))
        self.assertEqual(stats["stale_cache_entries"], 2)
        with mock.patch.object(bm, "VERIFY_VERSION", "verify_video/test-bump"):
            _, stats = self.verify()
        self.assertEqual(sorted(self.calls), sorted(self.paths))
        self.assertEqual(stats["reused_from_cache"], 0)

    def test_a_missing_file_is_reused_while_missing_and_verified_once_it_appears(self):
        gone = os.path.join(self.dir, "gone.avi")
        verdicts, _ = self.verify(self.paths + [gone])
        self.assertEqual(verdicts.loc[gone, "reason"], "missing")
        self.verify(self.paths + [gone])
        self.assertEqual(self.calls, [])
        with open(gone, "wb") as f:
            f.write(b"now here")
        verdicts, _ = self.verify(self.paths + [gone])
        self.assertEqual(self.calls, [gone])
        self.assertTrue(pd.isna(verdicts.loc[gone, "reason"]))

    def test_an_interrupted_run_resumes_after_its_last_complete_chunk(self):
        class Interrupted(Exception):
            pass

        def flaky(path):
            if len(self.calls) == 7:
                raise Interrupted
            return self.validate(path)

        with self.assertRaises(Interrupted):
            quiet(bm.verify_archive, sorted(self.paths), flaky, self.cache, chunk=3)
        done = bm.read_verify_cache(self.cache)
        self.assertEqual(len(done), 6)  # two complete chunks were written
        self.assertFalse([f for f in os.listdir(self.cache) if not f.startswith("verify-")])
        _, stats = self.verify()
        self.assertEqual((stats["reused_from_cache"], stats["verified_this_run"]), (6, 4))
        self.assertEqual(set(self.calls), set(self.paths) - set(done.video_path))

    def test_verify_limit_stops_after_n_pending_videos(self):
        verdicts, stats = self.verify(limit=4)
        self.assertEqual((len(self.calls), stats["pending"], len(verdicts)), (4, 6, 4))
        _, stats = self.verify(limit=0)
        self.assertEqual((self.calls, stats["reused_from_cache"], stats["pending"]), ([], 4, 6))
        _, stats = self.verify()
        self.assertEqual((len(self.calls), stats["pending"]), (6, 0))

    def test_stat_paths_reports_unreadable_paths(self):
        size, mtime = bm.stat_paths([self.paths[2], os.path.join(self.dir, "absent.avi")], threads=2, block=1)
        self.assertEqual(size.tolist(), [3, -1])
        self.assertEqual(mtime[0], os.stat(self.paths[2]).st_mtime_ns)


class TestArchiveBuild(unittest.TestCase):
    """The master index and the two arms, with a stand-in verifier and a synthetic acquisition index that
    leaves every 9th DICOM unscanned (a development build: `allow_partial_acquisition`)."""

    @classmethod
    def setUpClass(cls):
        cls.meta = metadata(range(200), seed=4)
        rows = eligible(cls.meta)
        cls.anchor = legacy_anchor(cls.meta, n=60)
        # One study of a legacy test patient and one of a train patient lose every video.
        test_patient = cls.anchor.patient_id[cls.anchor.split == "test"].iloc[0]
        cls.lost_studies = {
            rows.sid[rows.mrn == test_patient].iloc[0],
            rows.sid[rows.mrn == cls.anchor.patient_id[cls.anchor.split == "train"].iloc[0]].iloc[0],
        }
        cls.failing = set(rows.avi_path[rows.sid.isin(cls.lost_studies)]) | set(rows.avi_path.iloc[::17])
        cls.acq = acquisition_index(cls.meta)
        cls.result = cls.build()

    @classmethod
    def build(cls, meta=None, acq=..., **kwargs):
        failing = cls.failing
        kwargs.setdefault("allow_partial_acquisition", True)

        def validate(path):
            # Video 3 of a study is shorter than one clip; the others span more than one.
            return (
                bm.Check("undecodable")
                if path in failing
                else bm.Check(None, 8 if path.endswith("3.avi") else 60, 30.0)
            )

        return quiet(
            bm.build_archive,
            cls.meta if meta is None else meta,
            None,
            config(),
            as_read(cls.anchor),
            validate=validate,
            acquisition=cls.acq if acq is ... else acq,
            **kwargs,
        )

    def test_master_index_holds_every_eligible_video_including_failed_ones(self):
        index, rows = self.result["videos"], eligible(self.meta)
        self.assertEqual(sorted(index.video_path), sorted(rows.avi_path))
        self.assertEqual(set(index.video_path[index.reason == "undecodable"]), self.failing)
        for column in (
            "patient_id",
            "study_id",
            "study_date",
            "split",
            "origin",
            "view",
            "video_path",
            "dicom_path",
            "vendor",
            "model",
            "reason",
            "n_frames",
            "fps",
            "needs_padding",
            "acq_category",
            "acq_confidence",
            "acq_scanned",
            *bm.ACQ_FLAGS,
            "in_inclusive",
            "in_bmode",
        ):
            self.assertIn(column, index.columns)
        # One 16-frame clip at 8 fps spans 48 frames of a 30 fps video.
        self.assertTrue(index.needs_padding[index.n_frames == 8].all())
        self.assertFalse(index.needs_padding[index.n_frames == 60].any())
        self.assertTrue(index.needs_padding[index.reason.notna()].isna().all())

    def test_inclusive_arm_is_every_valid_train_video(self):
        index = self.result["videos"]
        train_ok = index[(index.split == "train") & index.reason.isna()]
        self.assertEqual(sorted(self.result["manifests"]["train_inclusive"]), sorted(train_ok.video_path))
        self.assertIn("OTHER", set(train_ok.view))
        self.assertEqual(set(train_ok.acq_category), set(bm.ACQ_CATEGORIES))  # every mode, unknown included
        for split in ("val", "test"):
            expected = index.video_path[(index.split == split) & index.reason.isna()]
            self.assertEqual(sorted(self.result["manifests"][split]), sorted(expected))

    def test_bmode_arm_is_a_header_confirmed_subset(self):
        index, manifests = self.result["videos"], self.result["manifests"]
        bmode, inclusive = set(manifests["train_bmode"]), set(manifests["train_inclusive"])
        self.assertTrue(bmode and bmode < inclusive)
        rows = index.set_index("video_path").loc[sorted(bmode)]
        self.assertTrue((rows.acq_category == "bmode").all() and (rows.acq_confidence == "header").all())
        self.assertTrue(rows.acq_scanned.all())
        # Only vendors whose headers were validated against decoded videos (acquisition.bmode_vendors).
        self.assertTrue(rows.vendor.isin(CFG["acquisition"]["bmode_vendors"]).all())
        self.assertTrue(rows.acq_vendor_validated.all())
        left_out = index[index.video_path.isin(inclusive - bmode)]
        self.assertEqual(
            set(left_out.acq_category), {"color_flow", "spectral_doppler", "other", "unknown", "mmode", "bmode"}
        )
        left_bmode = left_out[left_out.acq_category == "bmode"]
        unconfirmed = left_bmode.acq_confidence == "unknown"
        self.assertTrue(unconfirmed.any() and (~unconfirmed).any())
        # A header B-mode left out comes from a vendor outside the validated list.
        self.assertFalse(left_bmode.acq_vendor_validated[~unconfirmed].any())
        self.assertFalse(left_bmode.vendor[~unconfirmed].isin(CFG["acquisition"]["bmode_vendors"]).any())

    def test_without_a_vendor_list_every_vendor_header_counts(self):
        cfg = config()
        cfg["acquisition"]["bmode_vendors"] = None
        result = quiet(bm.build_archive, self.meta, None, cfg, as_read(self.anchor),
                       validate=lambda path: bm.Check(None, 60, 30.0), acquisition=self.acq,
                       allow_partial_acquisition=True)
        index = result["videos"]
        header_bmode = index[(index.split == "train") & (index.acq_category == "bmode")
                             & (index.acq_confidence == "header")]
        self.assertGreater(header_bmode.vendor.nunique(), 1)
        self.assertTrue(header_bmode.in_bmode.all())

    def test_bmode_vendors_must_be_a_list_of_names(self):
        for bad in ([], "GE Vingmed Ultrasound", [""], [3]):
            cfg = config()
            cfg["acquisition"]["bmode_vendors"] = bad
            with self.assertRaisesRegex(ValueError, "bmode_vendors"):
                bm.check_config(cfg)

    def test_videos_missing_from_the_acquisition_index_are_unknown_and_not_scanned(self):
        index = self.result["videos"]
        unscanned = index[~index.dicom_path.isin(set(self.acq.dcm_path))]
        self.assertGreater(len(unscanned), 10)
        self.assertTrue((unscanned.acq_category == "unknown").all() and (~unscanned.acq_scanned).all())
        self.assertTrue((unscanned.acq_rule == "not_scanned").all())
        self.assertFalse(unscanned.in_bmode.any())
        counts = self.result["counts"]
        self.assertEqual(counts["acquisition"]["videos_not_scanned"], len(unscanned))
        self.assertEqual(counts["videos_by"]["acquisition"]["not_scanned"]["eligible"], len(unscanned))
        self.assertEqual(counts["videos_by"]["acquisition"]["not_scanned"]["train_bmode"], 0)

    def test_a_partial_acquisition_index_is_refused_before_any_video_is_verified(self):
        decoded = []

        def validate(path):
            decoded.append(path)
            return bm.Check(None, 60, 30.0)

        with self.assertRaisesRegex(ValueError, r"lacks \d+ eligible videos .*--allow-partial-acquisition"):
            quiet(bm.build_archive, self.meta, None, config(), as_read(self.anchor), validate=validate,
                  acquisition=self.acq)
        self.assertEqual(decoded, [])
        # Unscanned videos of a split that is not verified do not count.
        rows = self.result["videos"]
        train_dicoms = set(rows.dicom_path[rows.split == "train"])
        complete = acquisition_index(self.meta, unscanned_every=10**6)
        partial = complete[~complete.dcm_path.isin(train_dicoms)]
        with self.assertRaisesRegex(ValueError, r"'train': \d+"):
            self.build(acq=partial, verify_splits=["train"], allow_partial_acquisition=False)
        result = self.build(acq=partial, verify_splits=["val", "test"], allow_partial_acquisition=False)
        self.assertTrue(result["counts"]["acquisition"]["coverage_complete"])

    def test_a_complete_index_builds_and_scanned_unknown_videos_stay_out_of_the_bmode_arm(self):
        complete = acquisition_index(self.meta, unscanned_every=10**6)
        result = self.build(acq=complete, allow_partial_acquisition=False)
        counts, index = result["counts"]["acquisition"], result["videos"]
        self.assertTrue(counts["coverage_complete"] and counts["ready"])
        self.assertEqual(sum(counts["eligible_not_in_index"].values()), 0)
        self.assertTrue(index.acq_scanned.all())
        unknown = index[index.acq_category == "unknown"]
        self.assertGreater(len(unknown), 0)
        self.assertFalse(unknown.in_bmode.any())
        self.assertNotIn("acq:not_scanned", set(result["excluded"].reason))

    def test_a_development_build_records_the_partial_index(self):
        counts = self.result["counts"]["acquisition"]
        self.assertFalse(counts["coverage_complete"] or counts["ready"])
        self.assertTrue(counts["allow_partial_acquisition"])
        self.assertEqual(sum(counts["eligible_not_in_index"].values()), int((~self.result["videos"].acq_scanned).sum()))

    def test_effective_category_leaves_unconfirmed_bmode_unresolved(self):
        videos = pd.DataFrame({
            "acq_category": ["bmode", "bmode", "bmode", "color_flow", "color_flow", None],
            "acq_confidence": ["header", "header", "unknown", "header", "header", None],
            "acq_vendor_validated": pd.array([True, False, True, False, True, True], dtype="boolean"),
        })
        self.assertEqual(bm.effective_acq_category(videos).tolist(),
                         ["bmode", "unknown", "unknown", "color_flow", "color_flow", "unknown"])
        # The B-mode arm and the effective category agree on every video of the build.
        index = self.result["videos"]
        effective = bm.effective_acq_category(index)
        self.assertTrue(((effective == "bmode").to_numpy() == bm.confirmed_bmode(index)).all())

    def test_exclusion_table_reasons(self):
        excluded, index = self.result["excluded"], self.result["videos"]
        self.assertEqual(
            list(excluded.columns), ["split", "patient_id", "study_id", "view", "video_path", "arm", "reason"]
        )
        inclusive = excluded[excluded.arm == "inclusive"]
        self.assertEqual(set(inclusive.video_path), self.failing)
        bmode = excluded[excluded.arm == "bmode"]
        self.assertTrue((bmode.split == "train").all())
        train = index[index.split == "train"]
        self.assertEqual(len(bmode), int((~train.in_bmode).sum()))
        self.assertEqual(
            set(bmode.reason),
            {
                "undecodable",
                "acq:color_flow",
                "acq:spectral_doppler",
                "acq:other",
                "acq:unknown",
                "acq:not_scanned",
                "acq:mmode",
                "acq:bmode_unconfirmed",
                "acq:bmode_vendor_unvalidated",
            },
        )

    def test_a_study_losing_every_clip_keeps_its_split_and_is_counted(self):
        studies = self.result["studies"].set_index("study_id")
        splits = dict(zip(self.result["patient_splits"].patient_id, self.result["patient_splits"].split))
        anchor = dict(zip(self.anchor.patient_id, self.anchor.split))
        for sid in self.lost_studies:
            row = studies.loc[sid]
            self.assertEqual((row.n_ok, row.n_bmode), (0, 0))
            self.assertEqual(row.split, splits[row.patient_id])
            self.assertEqual(row.split, anchor[row.patient_id])
        # Same mapping as the assignment alone: verification never redraws.
        assigned = assign(self.meta, self.anchor)
        pd.testing.assert_frame_equal(assigned["patient_splits"], self.result["patient_splits"])
        lost = sum(e["studies_without_verified_video"] for e in self.result["counts"]["splits"].values())
        self.assertGreaterEqual(lost, len(self.lost_studies))

    def test_listed_videos_are_excluded_without_being_decoded(self):
        ok = self.result["videos"][self.result["videos"].reason.isna()]
        listed = set(ok.video_path[ok.split == "train"].iloc[:3]) | set(ok.video_path[ok.split == "val"].iloc[:1])
        decoded = []

        def validate(path):
            decoded.append(path)
            return bm.Check(None, 60, 30.0)

        result = quiet(bm.build_archive, self.meta, None, config(), as_read(self.anchor), validate=validate,
                       acquisition=self.acq, allow_partial_acquisition=True,
                       exclude={"reason": "deid_black", "paths": listed | {"/not/an/eligible/video.mp4"}})
        self.assertTrue(decoded)
        self.assertFalse(listed & set(decoded))
        index = result["videos"].set_index("video_path")
        self.assertTrue((index.reason[sorted(listed)] == "excluded:deid_black").all())
        self.assertFalse(listed & set().union(*(set(m) for m in result["manifests"].values())))
        excluded = result["excluded"]
        self.assertEqual(set(excluded.video_path[excluded.reason == "excluded:deid_black"]), listed)
        self.assertEqual(result["counts"]["exclusion_list"],
                         {"reason": "excluded:deid_black", "listed": 5, "eligible": {"train": 3, "val": 1, "test": 0},
                          "excluded": 4, "not_eligible": 1})
        self.assertEqual(result["counts"]["verification"]["videos_by_reason"]["excluded:deid_black"], 4)
        # The split is the one assigned without a list.
        pd.testing.assert_frame_equal(result["patient_splits"], self.result["patient_splits"])

    def test_unverified_splits_are_excluded_as_not_verified(self):
        result = self.build(verify_splits=["train"])
        index = result["videos"]
        self.assertEqual(set(result["manifests"]), {"train_inclusive", "train_bmode"})
        held = index[index.split != "train"]
        self.assertTrue((held.reason == "not_verified").all())
        self.assertEqual(result["counts"]["verification"]["videos_to_verify"], int((index.split == "train").sum()))
        self.assertIn("not_verified", result["counts"]["verification"]["videos_by_reason"])
        self.assertEqual(set(result["excluded"].reason[result["excluded"].split != "train"]), {"not_verified"})

    def test_without_an_acquisition_index_there_is_no_bmode_manifest(self):
        result = self.build(acq=None)
        self.assertNotIn("train_bmode", result["manifests"])
        self.assertFalse(result["videos"].in_bmode.any())
        self.assertFalse(result["counts"]["acquisition"]["index_given"])
        self.assertEqual(set(result["excluded"].arm), {"inclusive"})

    def test_duplicate_dicom_path_in_the_acquisition_index_raises(self):
        acq = pd.concat([self.acq, self.acq.head(1)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "repeat a dcm_path"):
            self.build(acq=acq)
        bad = self.acq.copy()
        bad.loc[0, "acq_category"] = "doppler"
        with self.assertRaisesRegex(ValueError, "acq_category values outside"):
            self.build(acq=bad)

    def test_conflicting_paths_raise(self):
        meta = self.meta.copy()
        ok = eligible(meta).index
        meta.loc[ok[1], "avi_path"] = meta.loc[ok[0], "avi_path"]  # one video path under two rows
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.build(meta=meta, batch_size=50)
        meta = self.meta.copy()
        meta.loc[ok[1], "dcm_path"] = meta.loc[ok[0], "dcm_path"]  # one DICOM behind two videos
        with self.assertRaisesRegex(ValueError, "DICOM paths appear under more than one"):
            self.build(meta=meta)

    def test_identical_index_under_metadata_row_shuffles(self):
        again = self.build(meta=self.meta.sample(frac=1, random_state=9), batch_size=113)
        pd.testing.assert_frame_equal(self.result["videos"], again["videos"])
        pd.testing.assert_frame_equal(self.result["studies"], again["studies"])
        pd.testing.assert_frame_equal(self.result["excluded"], again["excluded"])

    def test_reports_cover_splits_dimensions_and_concentration(self):
        counts = self.result["counts"]
        index = self.result["videos"]
        self.assertEqual(
            counts["manifests"]["train_inclusive"]["videos"], len(self.result["manifests"]["train_inclusive"])
        )
        self.assertEqual(counts["manifests"]["train_bmode"]["patients"], index.patient_id[index.in_bmode].nunique())
        for key in ("year", "view", "vendor", "model", "vendor_model", "acquisition"):
            self.assertEqual(sum(v["eligible"] for v in counts["videos_by"][key].values()), len(index), key)
        conc = counts["concentration"]["train_inclusive"]["videos_per_patient"]
        per_patient = index[index.in_inclusive].patient_id.value_counts()
        self.assertEqual(
            (conc["units"], conc["total"], conc["max"]), (len(per_patient), per_patient.sum(), per_patient.max())
        )
        top = int(np.ceil(len(per_patient) * 0.1))
        self.assertAlmostEqual(
            conc["share_top_10pct"], per_patient.sort_values(ascending=False)[:top].sum() / per_patient.sum(), 3
        )
        self.assertEqual(sum(sum(v.values()) for v in counts["verification_reasons_by_split"].values()), len(index))

    def test_check_archive_catches_each_injected_violation(self):
        index, ps = self.result["videos"], self.result["patient_splits"]
        anchor = bm.parse_anchor(as_read(self.anchor))
        bm.check_archive(index, ps, anchor)  # the build itself passes
        held = set(anchor.patient_id[anchor.split == "val"])
        val_row = index.index[index.patient_id.isin(held) & index.reason.isna()][0]
        failed_train = index.index[(index.split == "train") & index.reason.notna()][0]
        color_train = index.index[index.in_inclusive & (index.acq_category == "color_flow")][0]
        multi = index.groupby("study_id").patient_id.transform("size") > 1
        study_row = index.index[multi & (index.split == "train")][0]
        other_train_patient = index.patient_id[
            (index.split == "train") & (index.patient_id != index.patient_id[study_row])
        ].iloc[0]
        anchored_patient = anchor.patient_id[anchor.split == "train"].iloc[0]

        def moved(df):  # the anchored train patient moved to test, consistently everywhere but the anchor
            return df.assign(split=np.where(df.patient_id == anchored_patient, "test", df.split))

        cases = [
            (
                "not in its patient's split",
                lambda v: v.assign(split=np.where(v.index == val_row, "train", v.split)),
                None,
            ),
            (
                "anchored val/test patient has a video in a training",
                lambda v: v.assign(in_inclusive=v.index == val_row),
                None,
            ),
            (
                "failed verification is in train_inclusive",
                lambda v: v.assign(in_inclusive=v.in_inclusive | (v.index == failed_train)),
                None,
            ),
            (
                "not a subset of train_inclusive",
                lambda v: v.assign(in_bmode=v.in_bmode | (v.index == failed_train)),
                None,
            ),
            ("does not confirm B-mode", lambda v: v.assign(in_bmode=v.in_bmode | (v.index == color_train)), None),
            (
                "video_path appears under more than one",
                lambda v: v.assign(video_path=np.where(v.index == study_row, v.video_path[color_train], v.video_path)),
                None,
            ),
            (
                "dicom_path appears under more than one",
                lambda v: v.assign(dicom_path=np.where(v.index == study_row, v.dicom_path[color_train], v.dicom_path)),
                None,
            ),
            (
                "whitespace",
                lambda v: v.assign(video_path=np.where(v.index == color_train, "/synthetic/a b.avi", v.video_path)),
                None,
            ),
            (
                "study belongs to more than one patient",
                lambda v: v.assign(patient_id=np.where(v.index == study_row, other_train_patient, v.patient_id)),
                None,
            ),
            ("more than one split in patient_splits", None, lambda p: pd.concat([p, p.head(1).assign(split="val")])),
            ("in_archive", None, lambda p: p.assign(in_archive=p.patient_id != p.patient_id.iloc[0])),
            ("lost its anchor split", moved, moved),
        ]
        for message, videos, splits in cases:
            with self.subTest(message), self.assertRaisesRegex(AssertionError, message):
                bm.check_archive((videos or (lambda v: v))(index.copy()), (splits or (lambda p: p))(ps.copy()), anchor)


class TestArchiveCli(unittest.TestCase):
    """End to end on real tiny videos: verification, cache reuse, the arms and the smoke test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        rows = []
        for p in range(12):
            for s in range(2 if p % 3 == 0 else 1):
                for v in range(2):
                    path = os.path.join(self.dir, f"p{p}_s{s}_v{v}.mp4")
                    write_video(path, frames=12 if (p, s, v) == (1, 0, 0) else 40, fps=5 if p == 2 else 30)
                    rows.append(
                        (
                            f"{p:07d}",
                            f"'1.2.999.{p}.{s}'",
                            TTE,
                            VIEWS[v],
                            path,
                            "success",
                            f"20{18 + s}0101",
                            f"/synthetic/dicom/{p}/{s}/{v}.dcm",
                            "GE Vingmed Ultrasound",
                            "Vivid E95",
                        )
                    )
        empty = os.path.join(self.dir, "empty.mp4")
        open(empty, "wb").close()
        rows.append(
            (
                "0000003",
                "'1.2.999.3.0'",
                TTE,
                "A4C",
                empty,
                "success",
                "20180101",
                "/synthetic/dicom/3/0/9.dcm",
                "GE Vingmed Ultrasound",
                "Vivid E95",
            )
        )
        self.meta = pd.DataFrame(rows, columns=META_COLUMNS)
        self.metadata = os.path.join(self.dir, "meta.csv")
        self.meta.to_csv(self.metadata, index=False)
        anchor = pd.DataFrame(
            {
                "study_id": [f"1.2.999.{p}.0" for p in range(10)],
                "split": ["train"] * 7 + ["val", "test", "test"],
                "patient_id": [f"{p:07d}" for p in range(10)],
            }
        )
        self.anchor = os.path.join(self.dir, "studies.csv")
        anchor.to_csv(self.anchor, index=False)
        acq = acquisition_index(self.meta, unscanned_every=1000)
        acq["acq_category"] = np.where(acq.dcm_path.str.endswith("/1.dcm"), "color_flow", "bmode")
        acq["acq_confidence"] = "header"
        self.acq = os.path.join(self.dir, "acq")
        os.makedirs(self.acq)
        acq.iloc[: len(acq) // 2].to_parquet(os.path.join(self.acq, "part-0.parquet"), index=False)
        acq.iloc[len(acq) // 2 :].to_parquet(os.path.join(self.acq, "part-1.parquet"), index=False)
        self.cache = os.path.join(self.dir, "cache")
        self.out = os.path.join(self.dir, "out")

    def args(self, *extra, out=None):
        return [
            "--config",
            ARCHIVE_CONFIG,
            "--metadata",
            self.metadata,
            "--anchor-splits",
            self.anchor,
            "--acquisition-index",
            self.acq,
            "--verify-cache",
            self.cache,
            "--out-dir",
            out or self.out,
            "--workers",
            "1",
            *extra,
        ]

    def info(self, out=None):
        with open(os.path.join(out or self.out, "manifest_info.json")) as f:
            return json.load(f)

    def test_full_build_smoke_tests_and_reuses_the_cache(self):
        run_cli(self.args())  # includes the smoke test of the four manifests
        names = set(os.listdir(self.out))
        self.assertLessEqual(
            {
                "train_inclusive.csv",
                "train_bmode.csv",
                "val.csv",
                "test.csv",
                "videos.parquet",
                "studies.csv",
                "patient_splits.csv",
                "excluded_videos.csv",
                "anchor_absent_patients.csv",
                "manifest_info.json",
                "summary.md",
            },
            names,
        )
        index = pd.read_parquet(os.path.join(self.out, "videos.parquet"))
        self.assertEqual(index.reason.dropna().tolist(), ["empty"])
        with open(os.path.join(self.out, "train_inclusive.csv")) as f:
            inclusive = [line.split(" ") for line in f.read().splitlines()]
        self.assertTrue(all(label == "0" for _, label in inclusive))
        with open(os.path.join(self.out, "train_bmode.csv")) as f:
            bmode = [line.split(" ")[0] for line in f.read().splitlines()]
        self.assertTrue(bmode and set(bmode) < {p for p, _ in inclusive})
        self.assertTrue(all(p.endswith("_v0.mp4") for p in bmode))
        info = self.info()
        self.assertEqual(info["counts"]["verification"]["verified_this_run"], len(self.meta))
        self.assertEqual(info["inputs"]["acquisition_index"]["sha256"], bm.sha256_path(self.acq)[0])
        with open(os.path.join(self.out, "summary.md")) as f:
            summary = f.read()
        for heading in (
            "## Verification",
            "## Manifests",
            "### Videos by acquisition mode",
            "### Videos by study year",
            "## Concentration of training videos",
            "clips) uniformly",
        ):
            self.assertIn(heading, summary)
        for secret in (self.dir, *self.meta.mrn, *self.meta.study_uid.str.strip("'")):
            self.assertNotIn(secret, summary)

        # A rebuild reads every verification from the cache and writes identical manifests.
        again = os.path.join(self.dir, "again")
        run_cli(self.args("--no-smoke-test", out=again))
        verification = self.info(again)["counts"]["verification"]
        self.assertEqual((verification["verified_this_run"], verification["reused_from_cache"]), (0, len(self.meta)))
        for name in (
            "train_inclusive.csv",
            "train_bmode.csv",
            "val.csv",
            "test.csv",
            "patient_splits.csv",
            "studies.csv",
            "excluded_videos.csv",
        ):
            self.assertEqual(bm.sha256(os.path.join(self.out, name)), bm.sha256(os.path.join(again, name)), name)

    def test_an_exclusion_list_is_recorded_with_the_inputs(self):
        listed = self.meta.avi_path.iloc[[0, 3]]
        path = os.path.join(self.dir, "black.csv")
        pd.DataFrame({"avi_path": listed, "made_black_by_fix": True}).to_csv(path, index=False)
        run_cli(self.args("--exclude-videos", path, "--exclude-reason", "deid_black", "--no-smoke-test"))
        info = self.info()
        self.assertEqual(info["inputs"]["exclude_videos"]["sha256"], bm.sha256_path(path)[0])
        self.assertEqual(info["counts"]["exclusion_list"]["excluded"], 2)
        excluded = pd.read_csv(os.path.join(self.out, "excluded_videos.csv"))
        self.assertEqual(set(excluded.video_path[excluded.reason == "excluded:deid_black"]), set(listed))
        for name in ("train_inclusive.csv", "train_bmode.csv", "val.csv", "test.csv"):
            with open(os.path.join(self.out, name)) as f:
                self.assertFalse(set(listed) & {line.split(" ")[0] for line in f.read().splitlines()}, name)
        with open(os.path.join(self.out, "summary.md")) as f:
            self.assertIn("Excluded by --exclude-videos, not decoded: 2 videos as `excluded:deid_black`", f.read())

    def test_a_partial_index_stops_the_build_unless_explicitly_allowed(self):
        acq = pd.read_parquet(self.acq)
        sample = os.path.join(self.dir, "acq_sample.parquet")
        acq.iloc[::2].to_parquet(sample, index=False)
        argv = self.args()
        argv[argv.index("--acquisition-index") + 1] = sample
        with self.assertRaisesRegex(ValueError, "lacks .* eligible videos"):
            run_cli(argv)
        self.assertFalse(os.path.exists(self.out))
        self.assertFalse(os.path.exists(self.cache))  # stopped before verifying anything

        run_cli([*argv, "--allow-partial-acquisition", "--no-smoke-test"])
        counts = self.info()["counts"]["acquisition"]
        self.assertEqual((counts["coverage_complete"], counts["ready"], counts["allow_partial_acquisition"]),
                         (False, False, True))
        self.assertGreater(sum(counts["eligible_not_in_index"].values()), 0)
        with open(os.path.join(self.out, "summary.md")) as f:
            self.assertIn("Development build: the acquisition index is not ready", f.read())

    def test_verify_limit_caches_a_sample_and_writes_no_manifest(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "keep.txt"), "w") as f:
            f.write("an earlier build")  # a benchmark run needs no empty --out-dir
        out = io.StringIO()
        with (
            mock.patch.object(
                sys, "argv", ["build_manifests.py", *self.args("--verify-limit", "5", "--workers", "2")]
            ),
            contextlib.redirect_stdout(out),
        ):
            bm.main()
        self.assertIn("--verify-limit 5: verified 5 videos", out.getvalue())
        self.assertEqual(os.listdir(self.out), ["keep.txt"])
        self.assertEqual(len(bm.read_verify_cache(self.cache)), 5)

    def test_cli_run_creates_no_files_in_the_repository(self):
        before = new_files_in_repo()
        if before is None:
            self.skipTest("not a git checkout")
        run_cli(self.args("--no-smoke-test"))
        self.assertEqual(new_files_in_repo(), before)


class TestRepositoryHygiene(unittest.TestCase):
    FILES = (
        "configs/data/manifests_tte_archive.yaml",
        "tests/data/test_build_archive_manifests.py",
        "data/build_manifests.py",
        "data/README.md",
    )
    REAL_PATH = re.compile(r"(?<![\w.])/(?:media|volume|mnt|home|root|Users|nfs|net)/|\bs3:/{2}")
    REAL_UID = re.compile(r"\b\d+(?:\.\d+){5,}\b")

    def test_archive_files_hold_no_real_paths_or_study_ids(self):
        for name in self.FILES:
            with open(os.path.join(REPO, name)) as f:
                text = f.read()
            self.assertIsNone(self.REAL_PATH.search(text), f"{name}: absolute data path")
            self.assertIsNone(self.REAL_UID.search(text), f"{name}: study UID")


if __name__ == "__main__":
    unittest.main()
