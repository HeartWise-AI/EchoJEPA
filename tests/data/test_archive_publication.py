"""Archive builds are published whole or not at all, and only from an acquisition index fit for the B-mode arm.

Publication: a build is written and smoke-tested in a staging folder, gets its checksums and then its completion
receipt, and only then replaces `--out-dir` by a rename, so a failure at any step leaves the earlier build intact.
Acquisition: every eligible video's row must come from the current scanner version and have read its header;
failed reads are admitted only in a reviewed number. Synthetic data and tiny real videos only."""

import contextlib
import hashlib
import io
import json
import os
import threading
import time
import unittest
from unittest import mock

import pandas as pd

from tests.data import test_build_archive_manifests as tb

bm = tb.bm


def checksums_hold(folder):
    """Every file of `outputs.sha256` matches, and the receipt matches `outputs.sha256`."""
    with open(os.path.join(folder, bm.OUTPUT_CHECKSUMS)) as f:
        listed = [line.split("  ", 1) for line in f.read().splitlines()]
    with open(os.path.join(folder, bm.BUILD_RECEIPT)) as f:
        receipt = json.load(f)
    files_ok = all(bm.sha256(os.path.join(folder, name)) == digest for digest, name in listed)
    return files_ok and receipt["outputs_sha256"] == bm.sha256(os.path.join(folder, bm.OUTPUT_CHECKSUMS))


def manifest(folder, name):
    with open(os.path.join(folder, name)) as f:
        return {line.split(" ")[0] for line in f.read().splitlines()}


class TestPublication(unittest.TestCase):
    setUp = tb.TestArchiveCli.setUp
    args = tb.TestArchiveCli.args

    def acq_with_more_bmode(self):
        acq = pd.read_parquet(self.acq)
        acq["acq_category"] = "bmode"
        path = os.path.join(self.dir, "acq_all_bmode.parquet")
        acq.to_parquet(path, index=False)
        return path

    def siblings(self, tag):
        parent, base = os.path.split(self.out)
        return [n for n in os.listdir(parent) if n.startswith(f"{base}.{tag}-")]

    def test_a_published_build_carries_checksums_and_a_receipt(self):
        tb.run_cli(self.args())
        self.assertTrue(checksums_hold(self.out))
        with open(os.path.join(self.out, bm.BUILD_RECEIPT)) as f:
            receipt = json.load(f)
        self.assertEqual((receipt["smoke_tested"], receipt["acquisition_ready"]), (True, True))
        self.assertIn("train_bmode.csv", receipt["files"])
        self.assertEqual(self.siblings("staging"), [])

    def test_a_failed_smoke_test_publishes_nothing(self):
        acq = pd.read_parquet(self.acq).assign(acq_category="color_flow")  # train_bmode.csv comes out empty
        argv = self.args()
        argv[argv.index("--acquisition-index") + 1] = os.path.join(self.dir, "acq_no_bmode.parquet")
        acq.to_parquet(argv[argv.index("--acquisition-index") + 1], index=False)
        with self.assertRaisesRegex(SystemExit, r"SmokeTestFailed: .*train_bmode.csv is empty(.|\n)*Nothing was "
                                                r"published"):
            tb.run_cli(argv)
        self.assertFalse(os.path.exists(self.out))
        staged = self.siblings("staging")
        self.assertEqual(len(staged), 1)  # kept for inspection, without a receipt
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.out), staged[0], bm.BUILD_RECEIPT)))

    def test_a_stalled_smoke_read_fails_after_its_deadline(self):
        tb.run_cli(self.args("--no-smoke-test"))
        with open(os.path.join(self.out, "val.csv")) as f:
            first = f.readline().split(" ")[0]
        os.replace(first, first + ".moved")
        os.mkfifo(first)  # opening it for reading blocks in the decoder, like a stalled network read
        outcome = {}

        def smoke():
            try:
                bm.supervised_smoke_test(self.out, ["val"], {"frames_per_clip": 16, "fps": 8}, timeout=3, retries=1)
            except Exception as e:
                outcome["error"] = e

        start = time.monotonic()
        worker = threading.Thread(target=smoke, daemon=True)  # a regression must fail, not hang the suite
        worker.start()
        worker.join(120)
        self.assertFalse(worker.is_alive(), "the smoke test did not end")
        self.assertIsInstance(outcome.get("error"), bm.SmokeTestFailed)
        self.assertIn("val: no result (timeout after 2 attempts", str(outcome["error"]))
        self.assertLess(time.monotonic() - start, 90)

    def test_a_failed_final_rename_puts_the_earlier_build_back(self):
        tb.run_cli(self.args())
        before = {n: bm.sha256(os.path.join(self.out, n)) for n in os.listdir(self.out)}
        real = os.rename

        def rename(src, dst, *a, **k):
            if ".staging-" in os.path.basename(src) and dst == os.path.abspath(self.out):
                raise OSError("injected failure of the final rename")
            return real(src, dst, *a, **k)

        with mock.patch.object(bm.os, "rename", rename), self.assertRaisesRegex(OSError, "final rename"):
            tb.run_cli(self.args("--overwrite"))
        self.assertEqual({n: bm.sha256(os.path.join(self.out, n)) for n in os.listdir(self.out)}, before)
        self.assertTrue(checksums_hold(self.out))
        self.assertEqual(self.siblings("replaced"), [])
        self.assertEqual(len(self.siblings("staging")), 1)  # the new build, unpublished

    def test_an_interrupted_replacement_is_reported_on_the_next_run(self):
        tb.run_cli(self.args())
        # As if the process died between moving the earlier build aside and publishing the new one.
        aside = f"{os.path.abspath(self.out)}.replaced-20261007T000000Z-1"
        os.rename(self.out, aside)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(bm.report_interrupted_publication(self.out), aside)
        self.assertIn("a replacement was interrupted", out.getvalue())
        tb.run_cli(self.args())  # a new build can still be published into the missing folder
        self.assertTrue(checksums_hold(self.out))

    def test_an_interrupted_overwrite_leaves_the_earlier_build_whole(self):
        tb.run_cli(self.args())
        before = {n: bm.sha256(os.path.join(self.out, n)) for n in os.listdir(self.out)}
        argv = self.args("--overwrite")
        argv[argv.index("--acquisition-index") + 1] = self.acq_with_more_bmode()
        real = pd.DataFrame.to_csv

        def interrupted(df, path=None, *a, **k):  # dies while writing the B-mode list of the new build
            if isinstance(path, str) and path.endswith("train_bmode.csv"):
                raise OSError("interrupted")
            return real(df, path, *a, **k)

        with mock.patch.object(pd.DataFrame, "to_csv", interrupted), self.assertRaises(OSError):
            tb.run_cli(argv)
        self.assertEqual({n: bm.sha256(os.path.join(self.out, n)) for n in os.listdir(self.out)}, before)
        self.assertTrue(checksums_hold(self.out))
        self.assertLessEqual(manifest(self.out, "train_bmode.csv"), manifest(self.out, "train_inclusive.csv"))

    def test_an_overwrite_replaces_the_whole_build_and_keeps_the_earlier_one(self):
        tb.run_cli(self.args())
        old_bmode = manifest(self.out, "train_bmode.csv")
        argv = self.args("--overwrite")
        argv[argv.index("--acquisition-index") + 1] = self.acq_with_more_bmode()
        tb.run_cli(argv)
        self.assertTrue(checksums_hold(self.out))
        new_bmode = manifest(self.out, "train_bmode.csv")
        self.assertGreater(len(new_bmode), len(old_bmode))
        self.assertLessEqual(new_bmode, manifest(self.out, "train_inclusive.csv"))
        replaced = self.siblings("replaced")
        self.assertEqual(len(replaced), 1)
        self.assertEqual(manifest(os.path.join(os.path.dirname(self.out), replaced[0]), "train_bmode.csv"), old_bmode)
        tb.run_cli(self.args("--overwrite", "--no-smoke-test"))  # within the same second: names must not collide
        self.assertEqual(len(self.siblings("replaced")), 2)


class TestAcquisitionReadiness(unittest.TestCase):
    """Coverage alone is not enough: stale scanner versions and failed header reads stop a full build."""

    setUp = tb.TestArchiveCli.setUp
    args = tb.TestArchiveCli.args
    info = tb.TestArchiveCli.info

    def with_index(self, change):
        acq = pd.read_parquet(self.acq)
        acq = change(acq)
        path = os.path.join(self.dir, f"acq-{hashlib.sha256(acq.to_json().encode()).hexdigest()[:8]}.parquet")
        acq.to_parquet(path, index=False)
        argv = self.args()
        argv[argv.index("--acquisition-index") + 1] = path
        return argv

    def test_rows_from_another_scanner_version_are_refused(self):
        argv = self.with_index(lambda a: a.assign(scanner_version="0"))
        with self.assertRaisesRegex(ValueError, r"from another scanner version \(versions: \{'0': \d+\}\)"):
            tb.run_cli(argv)
        self.assertFalse(os.path.exists(self.out))
        # A development build may use them, and says it is not ready.
        tb.run_cli([*argv, "--allow-partial-acquisition"])
        counts = self.info()["counts"]["acquisition"]
        self.assertEqual((counts["coverage_complete"], counts["ready"]), (True, False))

    def failed_reads(self, acq, n=3):
        failed = acq.index[:n]
        acq.loc[failed, ["read_status", "acq_category", "acq_confidence"]] = ["error", "unknown", "unknown"]
        return acq

    def test_failed_header_reads_need_a_reviewed_acceptance(self):
        argv = self.with_index(self.failed_reads)
        with self.assertRaisesRegex(ValueError, r"3 eligible videos have a failed header read"):
            tb.run_cli(argv)
        with self.assertRaisesRegex(ValueError, "--accept-acquisition-read-errors 3"):
            tb.run_cli([*argv, "--accept-acquisition-read-errors", "2"])
        tb.run_cli([*argv, "--accept-acquisition-read-errors", "3"])
        counts = self.info()["counts"]["acquisition"]
        self.assertEqual((sum(counts["read_errors"].values()), counts["accepted_read_errors"], counts["ready"]),
                         (3, 3, True))
        failed = set(pd.read_parquet(self.acq).dcm_path[:3])
        index = pd.read_parquet(os.path.join(self.out, "videos.parquet"))
        rows = index[index.dicom_path.isin(failed) & (index.split == "train")]
        self.assertGreater(len(rows), 0)
        self.assertFalse(rows.in_bmode.any())
        self.assertTrue(rows.in_inclusive[rows.reason.isna()].all())

    def test_a_successful_read_with_an_unknown_mode_is_a_result_not_an_error(self):
        def unknown_half(acq):
            acq.loc[acq.index[::2], ["acq_category", "acq_confidence"]] = ["unknown", "unknown"]
            return acq

        argv = self.with_index(unknown_half)
        tb.run_cli(argv)
        counts = self.info()["counts"]["acquisition"]
        self.assertEqual((sum(counts["read_errors"].values()), counts["ready"]), (0, True))
        index = pd.read_parquet(os.path.join(self.out, "videos.parquet"))
        unknown = index[(index.acq_category == "unknown") & (index.split == "train")]
        self.assertGreater(len(unknown), 0)
        self.assertFalse(unknown.in_bmode.any())


if __name__ == "__main__":
    unittest.main()
