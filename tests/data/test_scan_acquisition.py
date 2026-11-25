# tests/data/test_scan_acquisition.py

"""Tests for the header-only acquisition-mode index (data/scan_acquisition.py).

Synthetic DICOMs only: tiny files written by pydicom to a temporary directory, with made-up
identifiers and a few bytes of pixel data.
"""

import contextlib
import errno
import glob
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import pyarrow.parquet as pq
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, generate_uid

import scan_acquisition as sa

TTE = sa.TTE
US_MULTIFRAME = "1.2.840.10008.5.1.4.1.1.3.1"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NO_SEQUENCE = object()


def make_dicom(path, regions=((1, 1),), photometric="MONOCHROME2", pixel=b"\x00" * 48, implicit=False,
               manufacturer="ACME", model="Model-1", software=("1.0", "2"), frames=4):
    """A minimal US multi-frame DICOM. `regions` is a list of (RegionDataType, RegionSpatialFormat);
    None leaves that attribute out, a list makes it multi-valued; NO_SEQUENCE omits the sequence.
    """
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = US_MULTIFRAME
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = US_MULTIFRAME
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID = "SYNTHETIC-0001"
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = model
    ds.SoftwareVersions = list(software)
    ds.NumberOfFrames = frames
    ds.PhotometricInterpretation = photometric
    ds.Rows, ds.Columns = 2, 2
    ds.SamplesPerPixel = 3 if photometric == "RGB" else 1
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    if regions is not NO_SEQUENCE:
        items = []
        for t, f in regions:
            it = Dataset()
            if f is not None:
                it.add_new(0x00186012, "US", f)
            if t is not None:
                it.add_new(0x00186014, "US", t)
            it.RegionFlags = 0
            it.RegionLocationMinX0, it.RegionLocationMinY0, it.RegionLocationMaxX1, it.RegionLocationMaxY1 = 0, 0, 1, 1
            items.append(it)
        ds.SequenceOfUltrasoundRegions = Sequence(items)
    ds.PixelData = pixel
    ds.save_as(path, enforce_file_format=True)
    return path


def silent(*args, **kwargs):
    pass


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def dicom(self, name, *args, **kwargs):
        return make_dicom(os.path.join(self.tmp, name), *args, **kwargs)


class TestCodeTables(unittest.TestCase):
    def test_tables_match_ps3_3(self):
        self.assertEqual(sa.REGION_SPATIAL_FORMATS[3], "spectral")
        self.assertEqual([sa.REGION_DATA_TYPES[i] for i in range(5)],
                         ["none", "tissue", "color_flow", "pw_spectral_doppler", "cw_spectral_doppler"])
        # The standard lists hex codes: 000AH ECG, 000DH gray bar, 000EH color bar, 0012H other physiological.
        self.assertEqual([sa.REGION_DATA_TYPES[c] for c in (0x0A, 0x0D, 0x0E, 0x12)],
                         ["ecg_trace", "gray_bar", "color_bar", "other_physiological_input"])
        self.assertEqual(set(sa.REGION_DATA_TYPES), set(range(19)))

    def test_every_rule_names_a_category(self):
        self.assertEqual(set(sa.HEADER_RULES.values()) | {"unknown"}, set(sa.CATEGORIES))


class TestClassification(TempDirTest):
    def scan(self, regions, **kwargs):
        return sa.scan_file(self.dicom("x.dcm", regions, **kwargs))

    def assertFlags(self, row, **expected):
        for flag in sa.FLAGS:
            self.assertIs(bool(row[flag]), expected.get(flag, False), flag)

    def assertUnknown(self, row, rule):
        self.assertEqual((row["acq_category"], row["acq_confidence"], row["acq_rule"]), ("unknown", "unknown", rule))
        for flag in sa.FLAGS:
            self.assertIs(row[flag], pd.NA, flag)

    def test_tissue_only_is_bmode(self):
        row = self.scan([(1, 1)])
        self.assertEqual((row["read_status"], row["acq_category"], row["acq_confidence"]), ("ok", "bmode", "header"))
        self.assertEqual(row["acq_rule"], "tissue_2d_with_allowed_regions")
        self.assertFlags(row, has_tissue_2d=True)
        self.assertEqual((row["n_regions"], row["region_data_types"], row["region_spatial_formats"]), (1, "1", "1"))

    def test_every_region_is_read_not_just_the_first(self):
        # Color flow third, after the tissue region and a color bar.
        row = self.scan([(1, 1), (14, 5), (2, 1)])
        self.assertEqual(row["acq_category"], "color_flow")
        self.assertFlags(row, has_tissue_2d=True, has_color_flow=True)
        self.assertEqual((row["n_regions"], row["region_data_types"], row["region_spatial_formats"]),
                         (3, "1,2,14", "1,5"))
        self.assertEqual(row["region_pairs"], "1/1,2/1,14/5")

    def test_pw_doppler(self):
        row = self.scan([(1, 1), (3, 3)])
        self.assertEqual((row["acq_category"], row["acq_rule"]), ("spectral_doppler", "pw_or_cw_region"))
        self.assertFlags(row, has_tissue_2d=True, has_pw_doppler=True)

    def test_cw_doppler_with_trace(self):
        row = self.scan([(4, 3), (7, 3), (10, 4)])
        self.assertEqual(row["acq_category"], "spectral_doppler")
        self.assertFlags(row, has_cw_doppler=True)

    def test_spectral_wins_over_color(self):
        row = self.scan([(2, 1), (1, 1), (3, 3)])
        self.assertEqual(row["acq_category"], "spectral_doppler")
        self.assertFlags(row, has_tissue_2d=True, has_color_flow=True, has_pw_doppler=True)

    def test_mmode(self):
        row = self.scan([(1, 1), (1, 2)])
        self.assertEqual((row["acq_category"], row["acq_rule"]), ("mmode", "tissue_mmode_region"))
        self.assertFlags(row, has_tissue_2d=True, has_mmode=True)

    def test_color_mmode_is_color_flow(self):
        row = self.scan([(1, 1), (2, 2)])
        self.assertEqual(row["acq_category"], "color_flow")
        self.assertFlags(row, has_tissue_2d=True, has_color_flow=True, has_mmode=True)

    def test_ecg_and_gray_bar_with_tissue_is_bmode(self):
        row = self.scan([(10, 4), (1, 1), (13, 5), (0, 0)])
        self.assertEqual(row["acq_category"], "bmode")
        self.assertFlags(row, has_tissue_2d=True)

    def test_dual_tissue_2d_is_bmode(self):
        self.assertEqual(self.scan([(1, 1), (1, 1)])["acq_category"], "bmode")

    def test_color_bar_without_color_is_other(self):
        row = self.scan([(1, 1), (14, 5)])
        self.assertEqual((row["acq_category"], row["acq_confidence"]), ("other", "header"))
        self.assertEqual(row["acq_rule"], "region_outside_bmode_allowlist")
        # Observed negatives, not unknown.
        self.assertFlags(row, has_tissue_2d=True)

    def test_no_tissue_is_other(self):
        row = self.scan([(10, 4)])
        self.assertEqual((row["acq_category"], row["acq_rule"]), ("other", "no_tissue_2d_region"))

    def test_missing_sequence_is_unknown(self):
        row = self.scan(NO_SEQUENCE)
        self.assertEqual(row["read_status"], "ok")
        self.assertIs(row["has_region_sequence"], False)
        self.assertEqual(row["n_regions"], 0)
        self.assertUnknown(row, "no_region_sequence")

    def test_empty_sequence_is_unknown(self):
        row = self.scan([])
        self.assertIs(row["has_region_sequence"], True)
        self.assertUnknown(row, "empty_region_sequence")

    def test_region_without_data_type_is_unknown(self):
        # The tissue region alone would be B-mode; the second region could be anything.
        row = self.scan([(1, 1), (None, 1)])
        self.assertUnknown(row, "region_missing_type_or_format")
        self.assertEqual(row["region_pairs"], "1/1,?/1")

    def test_region_without_format_is_unknown(self):
        self.assertUnknown(self.scan([(1, 1), (2, None)]), "region_missing_type_or_format")

    def test_contradictory_pairs_are_unknown(self):
        for regions in ([(1, 1), (3, 1)], [(1, 1), (2, 3)], [(4, 1)], [(1, 4)], [(1, 1), (0, 1)], [(1, 3), (3, 3)]):
            with self.subTest(regions=regions):
                self.assertUnknown(self.scan(regions), "contradictory_type_format")

    def test_multivalued_code_is_unknown(self):
        self.assertUnknown(self.scan([(1, 1), ([1, 2], 1)]), "invalid_region_value")

    def test_undefined_code_is_unknown(self):
        self.assertUnknown(self.scan([(1, 1), (99, 1)]), "undefined_region_code")
        self.assertUnknown(self.scan([(1, 7)]), "undefined_region_code")

    def test_doppler_trace_without_pw_or_cw_is_unknown(self):
        self.assertUnknown(self.scan([(1, 1), (5, 4)]), "doppler_kind_undetermined")

    def test_photometric_never_decides(self):
        gray = self.scan([(1, 1), (10, 4)])
        rgb = sa.scan_file(self.dicom("rgb.dcm", [(1, 1), (10, 4)], photometric="RGB", pixel=b"\x00" * 96))
        self.assertEqual(rgb["photometric_interpretation"], "RGB")
        self.assertEqual({k: rgb[k] for k in sa.FLAGS + ("acq_category",)},
                         {k: gray[k] for k in sa.FLAGS + ("acq_category",)})

    def test_header_metadata_recorded(self):
        path = self.dicom("m.dcm", [(1, 1)], implicit=True)
        row = sa.scan_file(path)
        st = os.stat(path)
        self.assertEqual((row["manufacturer"], row["model_name"], row["software_versions"]),
                         ("ACME", "Model-1", "1.0\\2"))
        self.assertEqual((row["n_frames_header"], row["sop_class_uid"]), (4, US_MULTIFRAME))
        self.assertEqual((row["source_size"], row["source_mtime_ns"]), (st.st_size, st.st_mtime_ns))
        self.assertEqual((row["scanner_version"], row["dcm_path"]), (sa.SCANNER_VERSION, path))

    def test_corrupt_pixel_data_still_reads(self):
        # Wrong length and garbage pixels: the header scan never decodes them.
        row = self.scan([(1, 1), (2, 1)], photometric="RGB", pixel=b"\xff\x00\x13")
        self.assertEqual((row["read_status"], row["acq_category"]), ("ok", "color_flow"))
        # Truncated inside the pixel data: still fine.
        path = self.dicom("cut.dcm", [(1, 1)], pixel=b"\x01" * 400)
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - 300)
        self.assertEqual(sa.scan_file(path)["acq_category"], "bmode")


class TestReadErrors(TempDirTest):
    def assertError(self, row, reason):
        self.assertEqual((row["read_status"], row["failure_reason"]), ("error", reason))
        self.assertEqual((row["acq_category"], row["acq_confidence"], row["acq_rule"]),
                         ("unknown", "unknown", "read_error"))
        for flag in sa.FLAGS + ("has_region_sequence",):
            self.assertIsNone(row[flag]) if flag == "has_region_sequence" else self.assertIs(row[flag], pd.NA)

    def test_missing_file(self):
        row = sa.scan_file(os.path.join(self.tmp, "nope.dcm"))
        self.assertError(row, "missing")
        self.assertIsNone(row["source_size"])

    def test_empty_file(self):
        path = os.path.join(self.tmp, "empty.dcm")
        open(path, "wb").close()
        self.assertError(sa.scan_file(path), "empty_file")

    def test_not_dicom(self):
        path = os.path.join(self.tmp, "junk.dcm")
        with open(path, "wb") as f:
            f.write(os.urandom(2000))
        self.assertError(sa.scan_file(path), "not_dicom")

    def test_truncated_headers_are_errors_not_negative(self):
        path = self.dicom("full.dcm", [(1, 1), (14, 5), (2, 1)])
        with open(path, "rb") as f:
            data = f.read()
        pixel_tag = data.rindex(b"\xe0\x7f\x10\x00")
        reasons = set()
        for cut in range(140, pixel_tag, 7):
            with open(path, "wb") as f:
                f.write(data[:cut])
            row = sa.scan_file(path)
            with self.subTest(cut=cut):
                self.assertEqual(row["read_status"], "error")
                self.assertEqual(row["acq_category"], "unknown")
                self.assertIs(row["has_color_flow"], pd.NA)
            reasons.add(row["failure_reason"])
        self.assertIn("ended_before_pixel_data", reasons)
        self.assertLessEqual(reasons, {"ended_before_pixel_data", "unparseable_header"})

    def test_transient_errors_are_retried(self):
        path = self.dicom("ok.dcm", [(1, 1)])
        real_open, calls, sleeps = open, [], []

        def flaky(p, *a, **k):
            calls.append(p)
            if len(calls) <= 2:
                raise OSError(errno.EIO, "I/O error")
            return real_open(p, *a, **k)

        with mock.patch("builtins.open", flaky):
            row = sa.scan_file(path, max_retries=3, backoff=0.01, sleep=sleeps.append)
        self.assertEqual((row["read_status"], row["acq_category"]), ("ok", "bmode"))
        self.assertEqual((len(calls), sleeps), (3, [0.01, 0.02]))

    def test_retries_are_bounded(self):
        sleeps = []
        with mock.patch("builtins.open", side_effect=OSError(errno.ESTALE, "stale")) as m:
            row = sa.scan_file("/synthetic/a.dcm", max_retries=2, backoff=5, sleep=sleeps.append)
        self.assertEqual((row["failure_reason"], m.call_count), ("io_error", 3))
        self.assertEqual(sleeps, [5, sa.MAX_BACKOFF_S])

    def test_permanent_errors_are_not_retried(self):
        with mock.patch("builtins.open", side_effect=PermissionError(errno.EACCES, "denied")) as m:
            row = sa.scan_file("/synthetic/a.dcm", sleep=self.fail)
        self.assertEqual((row["failure_reason"], m.call_count), ("permission_denied", 1))

    def test_pydicom_oserror_without_errno_is_a_parse_error(self):
        path = self.dicom("p.dcm", [(1, 1)])
        with mock.patch("pydicom.dcmread", side_effect=OSError("No tag to read at file position 2BC")) as m:
            row = sa.scan_file(path, sleep=self.fail)
        self.assertEqual((row["failure_reason"], m.call_count), ("unparseable_header", 1))


class TestCache(TempDirTest):
    def setUp(self):
        super().setUp()
        self.cache = os.path.join(self.tmp, "cache")
        self.paths = [self.dicom(f"v{i}.dcm", [(1, 1)] if i % 2 else [(1, 1), (2, 1)]) for i in range(5)]

    def scan(self, paths=None, **kwargs):
        return sa.run_scan(self.paths if paths is None else paths, self.cache, workers=2, log=silent, **kwargs)

    def shards(self):
        return sa.shard_files(self.cache)

    def test_resume_scans_nothing_the_second_time(self):
        first = self.scan()
        self.assertEqual((first["status"], first["files_scanned"], first["plan"]["new"]), ("complete", 5, 5))
        n_shards = len(self.shards())
        second = self.scan()
        self.assertEqual((second["files_scanned"], second["plan"]["reused"]), (0, 5))
        self.assertEqual(len(self.shards()), n_shards)
        with open(os.path.join(self.cache, "progress.json")) as f:
            progress = json.load(f)
        self.assertEqual((progress["status"], progress["files_scanned"]), ("complete", 0))
        self.assertEqual(len(glob.glob(os.path.join(self.cache, "runs", "run-*.json"))), 2)
        self.assertFalse(os.path.exists(os.path.join(self.cache, "scan.lock")))

    def test_new_paths_only_are_scanned(self):
        self.scan(self.paths[:3])
        self.assertEqual(self.scan()["files_scanned"], 2)

    def test_check_changed_rescans_a_touched_file(self):
        self.scan()
        st = os.stat(self.paths[1])
        os.utime(self.paths[1], ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        self.assertEqual(self.scan()["files_scanned"], 0)
        again = self.scan(check_changed=True)
        self.assertEqual((again["files_scanned"], again["plan"]["changed"]), (1, 1))
        self.assertEqual(self.scan(check_changed=True)["files_scanned"], 0)

    def test_check_changed_rescans_a_deleted_file(self):
        self.scan()
        os.remove(self.paths[0])
        self.assertEqual(self.scan(check_changed=True)["files_scanned"], 1)
        index = sa.load_index(self.cache).set_index("dcm_path")
        self.assertEqual(index.loc[self.paths[0], "failure_reason"], "missing")
        # Still missing: unchanged, not rescanned again.
        self.assertEqual(self.scan(check_changed=True)["files_scanned"], 0)

    def test_scanner_version_bump_rescans_everything(self):
        self.scan()
        with mock.patch.object(sa, "SCANNER_VERSION", "test-next"):
            again = self.scan()
            self.assertEqual((again["files_scanned"], again["plan"]["stale_version"]), (5, 5))
            self.assertEqual(set(sa.load_index(self.cache).scanner_version), {"test-next"})
            self.assertEqual(self.scan()["files_scanned"], 0)

    def test_retry_errors(self):
        missing = os.path.join(self.tmp, "gone.dcm")
        self.scan(self.paths + [missing])
        self.assertEqual(self.scan(self.paths + [missing])["files_scanned"], 0)
        again = self.scan(self.paths + [missing], retry_errors=True)
        self.assertEqual((again["files_scanned"], again["plan"]["retried_errors"]), (1, 1))

    def test_limit_chunks_then_resume(self):
        part = self.scan(chunk_size=2, limit_chunks=1)
        self.assertEqual((part["status"], part["files_scanned"], part["chunks_total"]), ("partial", 2, 3))
        rest = self.scan(chunk_size=2)
        self.assertEqual((rest["status"], rest["files_scanned"]), ("complete", 3))
        self.assertEqual(len(sa.load_index(self.cache)), 5)

    def test_shards_are_atomic_and_typed(self):
        self.scan(chunk_size=2)
        names = os.listdir(sa.shard_dir(self.cache))
        self.assertEqual(len(names), 3)
        self.assertFalse([n for n in names if n.startswith(".tmp")])
        for path in self.shards():
            self.assertEqual(pq.read_schema(path).remove_metadata(), sa.INDEX_SCHEMA)
        # A file left over by a scan killed mid-write is never read.
        with open(os.path.join(sa.shard_dir(self.cache), ".tmp-shard-x.parquet"), "wb") as f:
            f.write(b"partial")
        self.assertEqual(len(sa.load_index(self.cache)), 5)

    def test_consolidate_keeps_latest_row(self):
        self.scan()
        target = self.paths[1]  # B-mode at first
        make_dicom(target, [(1, 1), (14, 5), (2, 1)])
        st = os.stat(target)
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        self.scan(check_changed=True)
        index, info = sa.consolidate(self.cache)
        self.assertEqual(len(index), 5)
        self.assertFalse(index.dcm_path.duplicated().any())
        self.assertEqual(index.set_index("dcm_path").loc[target, "acq_category"], "color_flow")
        self.assertEqual((info["rows"], info["rows_in_shards"]), (5, 6))
        out = os.path.join(self.cache, "acquisition_index.parquet")
        self.assertEqual(pq.read_schema(out).remove_metadata(), sa.INDEX_SCHEMA)
        self.assertEqual(info["output"]["sha256"], sa.sha256(out))
        reread = sa.load_index(out)
        self.assertEqual(str(reread.has_color_flow.dtype), "boolean")
        self.assertEqual(str(reread.source_mtime_ns.dtype), "Int64")
        self.assertEqual(int(reread.set_index("dcm_path").loc[target, "source_mtime_ns"]), os.stat(target).st_mtime_ns)
        pd.testing.assert_frame_equal(reread, sa.load_index(self.cache).sort_values("dcm_path").reset_index(drop=True))

    def test_conflicting_rows_in_one_run_are_refused(self):
        os.makedirs(sa.shard_dir(self.cache))
        rows = sa.rows_frame([sa.scan_file(self.paths[0]), sa.scan_file(self.paths[1])])
        run = "20260101T000000000000Z"
        sa.write_parquet_atomic(rows, os.path.join(sa.shard_dir(self.cache), f"shard-{run}-000000.parquet"))
        # The same row again in the same run is harmless.
        sa.write_parquet_atomic(rows.iloc[:1], os.path.join(sa.shard_dir(self.cache), f"shard-{run}-000001.parquet"))
        self.assertEqual(len(sa.consolidate(self.cache)[0]), 2)
        conflict = rows.iloc[:1].assign(acq_category="other")
        sa.write_parquet_atomic(conflict, os.path.join(sa.shard_dir(self.cache), f"shard-{run}-000002.parquet"))
        with self.assertRaisesRegex(ValueError, "conflicting rows within one scan run"):
            sa.consolidate(self.cache)
        # Across runs the later row simply wins.
        os.rename(os.path.join(sa.shard_dir(self.cache), f"shard-{run}-000002.parquet"),
                  os.path.join(sa.shard_dir(self.cache), "shard-20260102T000000000000Z-000000.parquet"))
        latest = sa.consolidate(self.cache)[0].set_index("dcm_path")
        self.assertEqual(latest.loc[self.paths[0], "acq_category"], "other")

    def test_a_running_scan_locks_the_cache(self):
        os.makedirs(self.cache)
        open(os.path.join(self.cache, "scan.lock"), "w").close()
        with self.assertRaises(SystemExit):
            self.scan()


def metadata_frame():
    """Metadata shaped like the real export, with made-up values."""
    rows = []
    for i in range(60):
        rows.append({
            "dcm_path": f"/synthetic/dcm/{i:04d}.dcm",
            "avi_path": f"/synthetic/avi/{i:04d}.avi" if i % 10 else None,
            "avi_status": "success" if i % 10 else "missing video",
            "study_type": TTE if i % 7 else "ETO SOP ANESTHESIE",
            "date": f"{2015 + i % 4}0101",
            "brand": ["GE", "Philips", "Siemens"][i % 3] if i % 11 else None,
            "model": "M",
            "predicted_class": ["A4C", "PLAX", "OTHER"][i % 3],
            "file_id": str(i // 2),
        })
    return pd.DataFrame(rows)


class TestSelection(unittest.TestCase):
    def test_usable_and_excluded_tte(self):
        meta = metadata_frame()
        meta.loc[3, "dcm_path"] = None
        usable, info = sa.select_rows(meta, "usable-tte")
        tte = meta.study_type == TTE
        ok = meta.avi_status == "success"
        self.assertEqual(set(usable.dcm_path), set(meta.dcm_path[tte & ok & meta.dcm_path.notna()]))
        excluded, _ = sa.select_rows(meta, "excluded-tte")
        self.assertEqual(set(excluded.dcm_path), set(meta.dcm_path[tte & ~ok & meta.dcm_path.notna()]))
        self.assertFalse(set(usable.dcm_path) & set(excluded.dcm_path))
        self.assertEqual(info["tte_rows_without_dcm_path"], 1)

    def test_file_id_uniqueness_is_reported_not_assumed(self):
        _, info = sa.select_rows(metadata_frame(), "usable-tte")
        self.assertFalse(info["file_id"]["unique"])
        self.assertGreater(info["file_id"]["duplicated_values"], 0)
        meta = metadata_frame().assign(file_id=lambda d: d.index.astype(str))
        self.assertTrue(sa.select_rows(meta, "usable-tte")[1]["file_id"]["unique"])

    def test_duplicate_dcm_path_raises(self):
        meta = metadata_frame()
        meta.loc[2, "dcm_path"] = meta.loc[1, "dcm_path"]
        with self.assertRaisesRegex(ValueError, "keyed by dcm_path"):
            sa.select_rows(meta, "usable-tte")
        # Even an exact repeat of a selected row is refused.
        meta = pd.concat([metadata_frame(), metadata_frame().iloc[[1]]], ignore_index=True)
        with self.assertRaises(ValueError):
            sa.select_rows(meta, "usable-tte")

    def test_duplicate_outside_the_selection_is_ignored(self):
        meta = metadata_frame()
        meta.loc[7, "dcm_path"] = meta.loc[14, "dcm_path"]  # both non-TTE rows
        self.assertEqual(len(sa.select_rows(meta, "usable-tte")[0]), 46)

    def test_paths_file_must_be_in_the_metadata(self):
        meta = metadata_frame()
        sel, _ = sa.select_rows(meta, None, ["/synthetic/dcm/0001.dcm", "/synthetic/dcm/0002.dcm"], ["year"])
        self.assertEqual(list(sel.year), ["2016", "2017"])
        with self.assertRaisesRegex(ValueError, "not in the metadata"):
            sa.select_rows(meta, None, ["/synthetic/dcm/0001.dcm", "/synthetic/dcm/9999.dcm"])

    def test_parquet_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.parquet")
            metadata_frame().to_parquet(path, index=False)
            sel, _ = sa.select_rows(path, "usable-tte", columns=["year", "brand"], batch_size=7)
        self.assertEqual(len(sel), 46)
        self.assertEqual(list(sel.columns), ["dcm_path", "brand", "year"])

    def test_stratified_sample_is_deterministic_and_order_independent(self):
        sel, _ = sa.select_rows(metadata_frame(), "usable-tte", columns=["year", "brand"])
        a = sa.stratified_sample(sel, ["year", "brand"], seed=3, n=20)
        b = sa.stratified_sample(sel.sample(frac=1, random_state=1), ["year", "brand"], seed=3, n=20)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 20)
        self.assertFalse(a.equals(sa.stratified_sample(sel, ["year", "brand"], seed=4, n=20)))

    def test_allocation_is_equal_ish(self):
        self.assertEqual(sa.allocate([1, 100, 100], 10), [1, 4, 5])
        self.assertEqual(sa.allocate([3, 3], 100), [3, 3])
        self.assertEqual(sum(sa.allocate([7, 2, 9, 4], 15)), 15)
        sel, _ = sa.select_rows(metadata_frame(), "usable-tte", columns=["year", "brand"])
        sample = sa.stratified_sample(sel, ["year", "brand"], seed=0, n=16)
        sizes = sel.fillna(sa.MISSING).groupby(["year", "brand"]).size()
        got = sample.fillna(sa.MISSING).groupby(["year", "brand"]).size().reindex(sizes.index, fill_value=0)
        # Every stratum gets its share or all of its rows.
        self.assertTrue(((got == sizes) | (got >= 16 // len(sizes))).all())

    def test_per_stratum_cap(self):
        sel, _ = sa.select_rows(metadata_frame(), "usable-tte", columns=["brand"])
        sample = sa.stratified_sample(sel, ["brand"], seed=0, per_stratum=2)
        self.assertEqual(sample.brand.fillna(sa.MISSING).value_counts().to_dict(),
                         {"GE": 2, "Philips": 2, "Siemens": 2, sa.MISSING: 2})


class TestCli(TempDirTest):
    def test_scan_consolidate_report(self):
        paths = [self.dicom(f"c{i}.dcm", regions) for i, regions in
                 enumerate([[(1, 1)], [(1, 1), (2, 1)], [(3, 3), (1, 1)], NO_SEQUENCE])]
        paths.append(os.path.join(self.tmp, "absent.dcm"))
        meta = pd.DataFrame({"dcm_path": paths, "study_type": TTE, "avi_status": "success",
                             "date": ["20190101", "20190102", "20200101", "20200101", "20210101"],
                             "brand": "GE", "model": "M", "predicted_class": "A4C", "file_id": "1"})
        meta_path = os.path.join(self.tmp, "meta.parquet")
        meta.to_parquet(meta_path, index=False)
        paths_file = os.path.join(self.tmp, "paths.txt")
        with open(paths_file, "w") as f:
            f.write("\n".join(paths + [paths[0], ""]) + "\n")
        cache, out = os.path.join(self.tmp, "cache"), os.path.join(self.tmp, "report")

        quiet(sa.main, ["scan", "--paths-file", paths_file, "--cache-dir", cache, "--workers", "2"])
        quiet(sa.main, ["scan", "--metadata", meta_path, "--select", "usable-tte", "--cache-dir", cache,
                        "--sample", "3", "--seed", "1", "--no-input-checksum"])
        with open(os.path.join(cache, "progress.json")) as f:
            progress = json.load(f)
        self.assertEqual((progress["files_scanned"], progress["selection"]["rows_sampled"]), (0, 3))
        self.assertIs(progress["selection"]["file_id"]["unique"], False)
        quiet(sa.main, ["consolidate", "--cache-dir", cache])
        quiet(sa.main, ["report", "--index", os.path.join(cache, "acquisition_index.parquet"),
                        "--metadata", meta_path, "--select", "usable-tte", "--out-dir", out])

        with open(os.path.join(out, "acquisition_report.json")) as f:
            report = json.load(f)
        self.assertEqual(report["acq_category"],
                         {"unknown": 2, "bmode": 1, "color_flow": 1, "spectral_doppler": 1})
        self.assertEqual(report["flags"]["has_color_flow"], {"true": 1, "false": 2, "unknown": 2})
        self.assertEqual(report["failure_reason"], {"missing": 1})
        self.assertEqual(report["coverage"]["indexed"], 5)
        self.assertEqual(report["tables"]["year (metadata)"]["2020"]["total"], 2)
        for name in ("acquisition_report.md", "acquisition_report.json"):
            with open(os.path.join(out, name)) as f:
                text = f.read()
            self.assertNotIn(self.tmp, text)
            self.assertNotIn(".dcm", text)

    def test_report_tables_stay_well_formed(self):
        # A "|" in a header value, and an empty region sequence, must not break any Markdown table.
        rows = [sa.scan_file(self.dicom("p.dcm", [(1, 1), (2, 1)], model="Probe | X")),
                sa.scan_file(self.dicom("q.dcm", []))]
        text = sa.report_markdown(sa.build_report(sa.rows_frame(rows)))
        self.assertIn("types 1,2 / formats 1", text)
        self.assertIn("types - / formats -", text)
        self.assertIn("Probe \\| X", text)
        table = []
        for line in text.splitlines() + [""]:
            if line.startswith("|"):
                table.append(line.replace("\\|", "").count("|"))
            elif table:
                self.assertEqual(len(set(table)), 1, table)
                table = []

    def test_outputs_inside_the_repository_are_refused(self):
        with self.assertRaises(SystemExit):
            quiet(sa.main, ["scan", "--paths-file", "x", "--cache-dir", os.path.join(REPO, "acq_cache")])
        with self.assertRaises(SystemExit):
            quiet(sa.main, ["report", "--index", self.tmp, "--out-dir", os.path.join(REPO, "docs", "acq")])
        self.assertFalse(os.path.exists(os.path.join(REPO, "acq_cache")))


if __name__ == "__main__":
    unittest.main()
