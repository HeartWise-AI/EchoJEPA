# tests/data/test_supervision.py

"""Worker supervision of the long archive jobs (data/supervised_pool.py): video verification in
build_manifests.py and the DICOM header scan in scan_acquisition.py.

Faults are injected into worker processes by tests/data/supervision_faults.py, which every scenario
runs in a subprocess under a hard deadline: a regression fails the test instead of hanging the suite.
Synthetic files only (tiny videos and DICOMs written to a temporary directory).
"""

import contextlib
import glob
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import pandas as pd

import build_manifests as bm
import scan_acquisition as sa
import supervised_pool as sp

from .supervision_faults import verdicts_json
from .test_scan_acquisition import make_dicom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FAULTS = os.path.join(HERE, "supervision_faults.py")
ARCHIVE_CONFIG = os.path.join(REPO, "configs", "data", "manifests_tte_archive.yaml")
TTE = "ECHO CARDIAQUE AVEC DOPPLER"
DEADLINE = 120  # seconds for one scenario, worker start-up included; each takes a few


def run_faults(spec, folder, deadline=DEADLINE):
    """Run one fault scenario in its own process group: (exit code, stdout, stderr). Fails, killing
    the group, when the scenario is not done within `deadline` seconds."""
    path = os.path.join(folder, f"spec-{time.time_ns()}.json")
    with open(path, "w") as f:
        json.dump(spec, f)
    proc = subprocess.Popen([sys.executable, FAULTS, path], cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=deadline)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise AssertionError(f"the scenario did not finish within {deadline} s: the supervision hangs") from None
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)  # no worker outlives its scenario
    return proc.returncode, out, err


def result(out):
    lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
    return json.loads(lines[-1][len("RESULT "):])


def attempts(path):
    return os.path.getsize(path + ".attempts") if os.path.exists(path + ".attempts") else 0


def write_video(path, frames=24, fps=25):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), 10 * i % 256, np.uint8))
    writer.release()
    return path


class TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.cache = os.path.join(self.dir, "cache")

    def files(self, *names):
        paths = []
        for name in names:
            path = os.path.join(self.dir, name)
            with open(path, "wb") as f:
                f.write(b"synthetic " + name.encode())
            paths.append(path)
        return paths

    def cached(self):
        return bm.read_verify_cache(self.cache).set_index("video_path") if os.path.isdir(self.cache) else None


class TestVerifySupervision(TempDir):
    """Verification in supervised worker processes, with the stand-in verifier of supervision_faults."""

    def verify(self, paths, validate=None, cache=True, **kwargs):
        kwargs.setdefault("workers", 2)
        kwargs.setdefault("chunk", 2)
        spec = {"run": "verify", "paths": sorted(paths), "cache": self.cache if cache else None, "kwargs": kwargs,
                "validate": validate}
        code, out, err = run_faults(spec, self.dir)
        self.assertEqual(code, 0, err)
        return result(out)

    def test_a_crashed_stalled_or_flaky_attempt_is_retried_and_only_its_result_cached(self):
        ok = self.files(*[f"ok{i}.avi" for i in range(6)])
        faulty = self.files("crash1_a.avi", "stall1_b.avi", "flaky1_c.avi")
        start = time.monotonic()
        out = self.verify(ok + faulty, timeout=3, retries=2)
        self.assertNotIn("error", out, out.get("message"))
        self.assertLess(time.monotonic() - start, 60)
        self.assertEqual({p: v[0] for p, v in out["verdicts"].items()}, dict.fromkeys(ok + faulty))
        stats = out["stats"]
        self.assertEqual(stats["failed_attempts"], {"timeout": 1, "crash": 1, "error": 0})
        self.assertEqual((stats["decode_failures_retried"], stats["recovered_on_retry"]), (1, 3))
        self.assertEqual([attempts(p) for p in faulty], [2, 2, 2])
        self.assertEqual([attempts(p) for p in ok], [1] * 6)
        cache = self.cached()
        self.assertEqual(sorted(cache.index), sorted(ok + faulty))
        self.assertTrue(cache.reason.isna().all())

    def test_unresolved_videos_stop_the_run_uncached_and_a_rerun_retries_them(self):
        ok = self.files(*[f"ok{i}.avi" for i in range(5)])
        crash, stall = self.files("crash_x.avi", "stall_y.avi")
        out = self.verify(ok + [crash, stall], timeout=2, retries=1)
        self.assertEqual(out["error"], "VerificationIncomplete")
        self.assertIn("2 videos are unresolved after 2 attempts each", out["message"])
        for path in ok + [crash, stall]:
            self.assertNotIn(path, out["message"])  # counts only: paths hold patient identifiers
        self.assertEqual((attempts(crash), attempts(stall)), (2, 2))
        # Every completed result is cached; the unresolved videos are not, and are listed for inspection.
        self.assertEqual(sorted(self.cached().index), sorted(ok))
        listed = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(self.cache, "unresolved-*.csv")))
        self.assertEqual(dict(zip(listed.video_path, listed.failure)), {crash: "crash", stall: "timeout"})

        # A rerun tries them again (the cached results are reused); one accepted failure is too few.
        out = self.verify(ok + [crash, stall], timeout=2, retries=1, accept_failures=1)
        self.assertEqual(out["error"], "VerificationIncomplete")
        self.assertIn("more than --accept-verify-failures 1", out["message"])
        self.assertEqual((attempts(crash), attempts(stall)), (4, 4))
        self.assertEqual([attempts(p) for p in ok], [1] * 5)

        out = self.verify(ok + [crash, stall], timeout=2, retries=1, accept_failures=2)
        self.assertNotIn("error", out)
        reasons = {p: v[0] for p, v in out["verdicts"].items()}
        self.assertEqual((reasons[crash], reasons[stall]), ("verify_crash", "verify_timeout"))
        self.assertEqual(out["stats"]["accepted_unresolved"], {"verify_timeout": 1, "verify_crash": 1})
        self.assertEqual(out["stats"]["pending"], 0)
        self.assertEqual(sorted(self.cached().index), sorted(ok))  # an accepted failure is never cached

    def test_attempts_failing_in_a_row_stop_the_pool(self):
        out = self.verify(self.files(*[f"crash_{i}.avi" for i in range(4)]), timeout=30, retries=2)
        self.assertEqual(out["error"], "PoolBroken")
        self.assertIn("failed in a row", out["message"])

    def test_without_faults_the_results_match_in_process_verification(self):
        paths = [write_video(os.path.join(self.dir, f"v{i}.mp4"), frames=10 + i) for i in range(6)]
        empty, junk = self.files("empty.mp4", "junk.mp4")
        open(empty, "wb").close()
        paths += [empty, junk, os.path.join(self.dir, "absent.mp4")]
        supervised = self.verify(paths, validate="real", cache=False)
        with contextlib.redirect_stdout(io.StringIO()):
            inline, _ = bm.verify_archive(sorted(paths), bm.verify_video, None, workers=1)
        self.assertEqual(supervised["verdicts"], json.loads(json.dumps(verdicts_json(inline))))
        reasons = {p: v[0] for p, v in supervised["verdicts"].items()}
        self.assertEqual((reasons[empty], reasons[junk], reasons[paths[-1]]), ("empty", "undecodable", "missing"))
        self.assertEqual(supervised["verdicts"][paths[3]][1:], [13, 25.0])
        # --verify-limit draws the same videos, whichever way they are verified.
        limited = self.verify(paths, validate="real", cache=False, limit=4)
        with contextlib.redirect_stdout(io.StringIO()):
            inline, _ = bm.verify_archive(sorted(paths), bm.verify_video, None, workers=1, limit=4)
        self.assertEqual(sorted(limited["verdicts"]), sorted(inline.index))


class TestVerifyRetries(TempDir):
    """In-run retries of decode failures and `--retry-failed`, verified in this process."""

    def setUp(self):
        super().setUp()
        self.paths = self.files(*[f"v{i}.avi" for i in range(5)])
        self.calls, self.broken = [], set()

    def validate(self, path):
        self.calls.append(path)
        if not os.path.exists(path):
            return bm.Check("missing")
        return bm.Check("undecodable") if path in self.broken else bm.Check(None, 12, 30.0)

    def verify(self, paths=None, **kwargs):
        self.calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            return bm.verify_archive(sorted(paths or self.paths), self.validate, self.cache, **kwargs)

    def test_a_decode_failure_is_retried_within_the_run_before_its_verdict_is_cached(self):
        first_only, always = self.paths[0], self.paths[1]
        self.broken = {first_only, always}

        gone = os.path.join(self.dir, "gone.avi")

        def validate(path):
            out = self.validate(path)
            if path == first_only:
                self.broken.discard(first_only)  # the storage recovers after the first read
            return out

        self.calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            verdicts, stats = bm.verify_archive(sorted(self.paths + [gone]), validate, self.cache)
        self.assertEqual((self.calls.count(first_only), self.calls.count(always)), (2, 1 + bm.VERIFY_RETRIES))
        self.assertEqual(self.calls.count(gone), 1)  # `missing` is not retried
        self.assertTrue(pd.isna(verdicts.loc[first_only, "reason"]))
        self.assertEqual(verdicts.loc[always, "reason"], "undecodable")
        self.assertEqual((stats["decode_failures_retried"], stats["recovered_on_retry"]), (1 + bm.VERIFY_RETRIES, 1))
        self.assertEqual(self.cached().loc[always, "reason"], "undecodable")
        # Without retries the first failure is the verdict.
        self.calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            bm.verify_archive(sorted(self.paths), self.validate, None, retries=0)
        self.assertEqual(self.calls.count(always), 1)

    def test_retry_failed_verifies_cached_failures_again_and_reuses_successes(self):
        gone = os.path.join(self.dir, "gone.avi")
        self.broken = {self.paths[2]}
        self.verify(self.paths + [gone])
        self.broken = set()  # the storage recovered; the file is unchanged
        verdicts, stats = self.verify(self.paths + [gone])
        self.assertEqual(self.calls, [])
        self.assertEqual(verdicts.loc[self.paths[2], "reason"], "undecodable")  # reused without the flag
        verdicts, stats = self.verify(self.paths + [gone], retry_failed=True)
        self.assertEqual(self.calls, [self.paths[2]])  # neither the successes nor `missing` are redone
        self.assertTrue(pd.isna(verdicts.loc[self.paths[2], "reason"]))
        self.assertEqual(verdicts.loc[gone, "reason"], "missing")
        self.assertEqual((stats["cached_failures_verified_again"], stats["reused_from_cache"]), (1, 5))
        self.assertTrue(pd.isna(self.cached().loc[self.paths[2], "reason"]))

    def test_a_stalled_stat_raises_instead_of_hanging(self):
        release, real = threading.Event(), os.stat
        self.addCleanup(release.set)

        def stat(path, *args, **kwargs):
            if path == self.paths[3]:
                release.wait(60)
            return real(path, *args, **kwargs)

        with mock.patch.object(bm.os, "stat", stat), self.assertRaisesRegex(sp.PoolBroken, "no progress for 0.5 s"):
            bm.stat_paths(self.paths, threads=2, block=1, timeout=0.5)

    def test_thread_map_keeps_order_and_passes_errors_on(self):
        self.assertEqual(sp.thread_map(lambda x: x * x, range(50), 4, 5), [x * x for x in range(50)])
        with self.assertRaises(ZeroDivisionError):
            sp.thread_map(lambda x: 1 / x, range(-3, 3), 2, 5)

    def test_refused_options(self):
        base = ["--config", ARCHIVE_CONFIG, "--metadata", "meta.csv", "--anchor-splits", "studies.csv",
                "--out-dir", os.path.join(self.dir, "out")]
        cases = [
            (["--retry-failed"], "--retry-failed .* needs --verify-cache"),
            (["--verify-timeout", "0"], "--verify-timeout must be positive"),
            (["--verify-retries", "-1"], "--verify-retries must be 0 or more"),
            (["--accept-verify-failures", "-2"], "--accept-verify-failures must be 0 or more"),
            (["--assign-only", "--verify-retries", "1"], "--assign-only verifies nothing"),
        ]
        for extra, message in cases:
            with self.subTest(message), mock.patch.object(sys, "argv", ["build_manifests.py", *base, *extra]), \
                    self.assertRaisesRegex(SystemExit, message):
                bm.main()


class TestBuildCliSupervision(TempDir):
    """The archive build end to end: an unresolved video stops it with nothing published, a rerun tries
    the video again, and --accept-verify-failures lets the build finish without caching that verdict."""

    def setUp(self):
        super().setUp()
        rows, self.videos = [], []
        for p in range(6):
            for v in range(2):
                name = "stall_p3.mp4" if (p, v) == (3, 1) else f"p{p}_v{v}.mp4"
                path = write_video(os.path.join(self.dir, name))
                self.videos.append(path)
                rows.append((f"{p:07d}", f"'1.2.999.{p}.0'", TTE, "A4C", path, "success", "20190101",
                             f"/synthetic/dicom/{p}/{v}.dcm", "GE Vingmed Ultrasound", "Vivid E95"))
        self.stall = os.path.join(self.dir, "stall_p3.mp4")
        meta = pd.DataFrame(rows, columns=["mrn", "study_uid", "study_type", "predicted_class", "avi_path",
                                           "avi_status", "date", "dcm_path", "brand", "model"])
        meta.to_csv(os.path.join(self.dir, "meta.csv"), index=False)
        pd.DataFrame({"study_id": [f"1.2.999.{p}.0" for p in range(6)], "patient_id": [f"{p:07d}" for p in range(6)],
                      "split": ["train"] * 4 + ["val", "test"]}).to_csv(os.path.join(self.dir, "studies.csv"),
                                                                        index=False)
        self.out = os.path.join(self.dir, "out")

    def build(self, *extra):
        argv = ["--config", ARCHIVE_CONFIG, "--metadata", os.path.join(self.dir, "meta.csv"), "--anchor-splits",
                os.path.join(self.dir, "studies.csv"), "--verify-cache", self.cache, "--out-dir", self.out,
                "--workers", "2", "--verify-timeout", "2", "--verify-retries", "1", "--verify-chunk", "4",
                "--no-smoke-test", *extra]
        return run_faults({"run": "build", "argv": argv}, self.dir)

    def test_an_unresolved_video_stops_the_build_until_it_is_accepted(self):
        code, out, err = self.build()
        self.assertNotEqual(code, 0)
        self.assertIn("VerificationIncomplete: 1 videos are unresolved", err)
        self.assertFalse(os.path.isdir(self.out) and os.listdir(self.out))  # nothing published
        self.assertEqual(sorted(self.cached().index), sorted(set(self.videos) - {self.stall}))
        self.assertEqual(attempts(self.stall), 2)

        code, out, err = self.build()  # the same command: the cache is reused, the stalled video retried
        self.assertNotEqual(code, 0)
        self.assertIn(f"{len(self.videos) - 1} reused from the cache", out)
        self.assertEqual(attempts(self.stall), 4)
        self.assertFalse(os.path.isdir(self.out) and os.listdir(self.out))

        code, out, err = self.build("--accept-verify-failures", "1")
        self.assertEqual(code, 0, err)
        with open(os.path.join(self.out, "manifest_info.json")) as f:
            verification = json.load(f)["counts"]["verification"]
        self.assertEqual(verification["accepted_unresolved"], {"verify_timeout": 1})
        self.assertEqual(verification["videos_by_reason"].get("verify_timeout"), 1)
        excluded = pd.read_csv(os.path.join(self.out, "excluded_videos.csv"))
        self.assertEqual(excluded.reason[excluded.video_path == self.stall].tolist(), ["verify_timeout"])
        with open(os.path.join(self.out, "train_inclusive.csv")) as f:
            self.assertNotIn(self.stall, f.read())
        self.assertNotIn(self.stall, self.cached().index)  # never cached: a later build tries it again


class TestScanSupervision(TempDir):
    """The header scan in supervised worker processes, through its CLI."""

    def setUp(self):
        super().setUp()
        self.ok = [make_dicom(os.path.join(self.dir, f"ok{i}.dcm"), [(1, 1)]) for i in range(4)]

    def scan(self, paths, *extra):
        listing = os.path.join(self.dir, "paths.txt")
        with open(listing, "w") as f:
            f.write("\n".join(paths) + "\n")
        argv = ["scan", "--paths-file", listing, "--cache-dir", self.cache, "--workers", "2", "--timeout", "2",
                "--worker-retries", "1", "--chunk-size", "3", *extra]
        return run_faults({"run": "scan", "argv": argv}, self.dir)

    def progress(self):
        with open(os.path.join(self.cache, "progress.json")) as f:
            return json.load(f)

    def test_a_crashed_or_stalled_read_is_retried(self):
        faulty = [make_dicom(os.path.join(self.dir, name), [(1, 1), (2, 1)])
                  for name in ("crash1_a.dcm", "stall1_b.dcm")]
        code, out, err = self.scan(self.ok + faulty)
        self.assertEqual(code, 0, err)
        index = sa.load_index(self.cache).set_index("dcm_path")
        self.assertEqual(sorted(index.index), sorted(self.ok + faulty))
        self.assertEqual(set(index.read_status), {"ok"})
        self.assertEqual(set(index.loc[faulty, "acq_category"]), {"color_flow"})
        progress = self.progress()
        self.assertEqual((progress["status"], progress["failed_attempts"]),
                         ("complete", {"timeout": 1, "crash": 1, "error": 0}))
        self.assertEqual([attempts(p) for p in faulty], [2, 2])

    def test_unresolved_files_get_no_row_until_accepted_and_retry_errors_scans_them_again(self):
        stall, crash = (make_dicom(os.path.join(self.dir, name), [(1, 1)]) for name in ("stall_c.dcm", "crash_d.dcm"))
        paths = self.ok + [stall, crash]
        code, out, err = self.scan(paths)
        self.assertNotEqual(code, 0)
        self.assertIn("2 DICOMs are unresolved after 2 attempts each", err)
        self.assertEqual(sorted(sa.load_index(self.cache).dcm_path), sorted(self.ok))
        progress = self.progress()
        self.assertEqual((progress["status"], progress["unresolved"]), ("incomplete", {"timeout": 1, "crash": 1}))
        self.assertEqual(len(glob.glob(os.path.join(self.cache, "runs", "unresolved-*.csv"))), 1)

        code, out, err = self.scan(paths)  # no row, so the same command reads them again
        self.assertNotEqual(code, 0)
        self.assertEqual((attempts(stall), attempts(crash)), (4, 4))
        self.assertEqual([attempts(p) for p in self.ok], [1] * 4)

        code, out, err = self.scan(paths, "--accept-scan-failures", "2")
        self.assertEqual(code, 0, err)
        index = sa.load_index(self.cache).set_index("dcm_path")
        self.assertEqual(index.loc[[stall, crash], "failure_reason"].tolist(), ["scan_timeout", "scan_crash"])
        self.assertEqual(set(index.loc[[stall, crash], "read_status"]), {"error"})
        self.assertEqual(set(index.loc[[stall, crash], "acq_category"]), {"unknown"})
        self.assertEqual(self.progress()["status"], "complete")

        code, out, err = self.scan(paths, "--retry-errors", "--accept-scan-failures", "2")
        self.assertEqual(code, 0, err)
        self.assertEqual((attempts(stall), attempts(crash)), (8, 8))  # accepted rows are read errors: retried

    def test_a_stalled_stat_of_check_changed_raises_instead_of_hanging(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def stat(path):
            release.wait(60)
            return sa.stat_identity(path)

        cached = sa.rows_frame([sa.empty_row(p) for p in self.ok])
        with self.assertRaisesRegex(sp.PoolBroken, "no progress"):
            sa.plan_scan(self.ok, cached, check_changed=True, stat=stat, workers=2, timeout=0.5)


if __name__ == "__main__":
    unittest.main()
