# tests/datasets/test_video_dataset_logs.py

"""Test that `VideoDataset` never exposes video or manifest paths, manifest names,or patient-like
identifiers. Failed samples are reported only by manifest position in `data_paths` and line number.
Use tiny synthetic manifests and corrupt videos in patient-named temporary paths and assert none
of those identifiers appear in logs, warnings, stdout, stderr, or native decord output.
"""

import contextlib
import io
import logging
import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

import src.datasets.video_dataset as video_dataset
from src.datasets.video_dataset import VideoDataset

IDENTIFIER = "MRN0012345"


def write_video(path, frames=24, fps=8):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), 40 + 5 * i, np.uint8))
    writer.release()


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def native_stderr(lines):
    """Appends to `lines` what is written to file descriptor 2, which C++ code writes to
    directly, bypassing sys.stderr."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as f:
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            f.seek(0)
            lines.extend(f.read().decode(errors="replace").splitlines())


@contextlib.contextmanager
def everything_emitted():
    """Collects log records, warnings, stdout and stderr into one list of strings."""
    handler, out, err = Capture(), io.StringIO(), []
    logger = video_dataset.logger
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out), native_stderr(err):
            warnings.simplefilter("always")
            emitted = []
            yield emitted
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    emitted.extend(handler.messages + [str(w.message) for w in caught] + out.getvalue().splitlines() + err)


class TestNoPathsInLogs(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = os.path.join(tmp.name, IDENTIFIER)
        os.makedirs(self.folder)
        self.real = os.path.join(self.folder, "real.mp4")
        write_video(self.real)

    def manifest(self, name, *paths):
        path = os.path.join(self.folder, f"{IDENTIFIER}_{name}")
        with open(path, "w") as f:
            f.writelines(f"{p} 0\n" for p in paths)
        return path

    def dataset(self, *manifests):
        return VideoDataset(data_paths=list(manifests), dataset_fpcs=[16] * len(manifests), fps=8, frame_step=None)

    def assertSaid(self, emitted, text):
        """Some message says `text`, whatever its capitalisation or closing punctuation."""
        self.assertTrue(any(text.lower() in m.lower() for m in emitted), f"{text!r} not in {emitted}")

    def assertNoPath(self, emitted):
        self.assertTrue(emitted, "the loader emitted nothing")
        for message in emitted:
            self.assertNotIn(IDENTIFIER, message)

    def test_building_the_dataset_logs_counts_only(self):
        with everything_emitted() as emitted:
            self.dataset(self.manifest("train.csv", self.real, os.path.join(self.folder, "other.mp4")))
        self.assertNoPath(emitted)
        self.assertSaid(emitted, "Loaded 2 samples")

    def test_a_missing_video_is_reported_by_its_manifest_line(self):
        missing = os.path.join(self.folder, "missing.mp4")
        dataset = self.dataset(self.manifest("a.csv", self.real), self.manifest("b.csv", self.real, missing))
        with everything_emitted() as emitted, mock.patch.object(video_dataset.np.random, "randint", return_value=0):
            buffer, _, _ = dataset[2]  # line 2 of the second manifest; the retry falls back to sample 0
        self.assertEqual(len(buffer[0]), 16)
        self.assertNoPath(emitted)
        self.assertSaid(emitted, "video file not found")
        self.assertSaid(emitted, "Retrying with new sample, failed to load sample 2 (manifest 1, line 2)")

    def test_an_unreadable_video_is_reported_by_its_manifest_line(self):
        # decord's C++ log prints what it cannot open, by path or by the start of its bytes.
        broken = os.path.join(self.folder, "broken.mp4")
        with open(broken, "wb") as f:
            f.write(f"{IDENTIFIER} is not a video".encode())
        dataset = self.dataset(self.manifest("train.csv", broken, self.real))
        with everything_emitted() as emitted, mock.patch.object(video_dataset.np.random, "randint", return_value=1):
            dataset[0]
            os.write(2, b"stderr works again\n")
        self.assertNoPath(emitted)
        self.assertSaid(emitted, "stderr works again")
        self.assertSaid(emitted, "Failed to load video: RuntimeError")
        self.assertSaid(emitted, "Retrying with new sample, failed to load sample 0 (manifest 0, line 1)")

    def test_a_video_that_vanishes_after_the_existence_check_is_skipped(self):
        # Between exists() and getsize() the file can vanish or lose its permissions; the
        # error names the path. The sample is skipped like any unreadable video, not raised.
        vanished = os.path.join(self.folder, "vanished.mp4")
        write_video(vanished)
        getsize = os.path.getsize
        dataset = self.dataset(self.manifest("train.csv", vanished, self.real))
        for error in (FileNotFoundError, PermissionError):
            def racing_getsize(path):
                if path == vanished:
                    raise error(f"the file changed: {vanished}")
                return getsize(path)

            with self.subTest(error=error.__name__):
                with everything_emitted() as emitted, \
                        mock.patch.object(video_dataset.os.path, "getsize", racing_getsize), \
                        mock.patch.object(video_dataset.np.random, "randint", return_value=1):
                    buffer, _, _ = dataset[0]
                self.assertEqual(len(buffer[0]), 16)
                self.assertNoPath(emitted)
                self.assertSaid(emitted, f"Failed to load video: {error.__name__}")
                self.assertSaid(emitted, "Retrying with new sample, failed to load sample 0 (manifest 0, line 1)")

    def test_late_decoder_failures_are_sanitized_and_skipped(self):
        # A damaged stream can open successfully and fail only when Decord seeks or decodes frames.
        # Both its native output and Python exception may contain the video path.
        late_failure = os.path.join(self.folder, "late_failure.mp4")
        write_video(late_failure)
        actual_video_reader = video_dataset.VideoReader

        class LateFailureReader:
            def __init__(self, operation):
                self.operation = operation

            def __len__(self):
                return 24

            def get_avg_fps(self):
                return 8

            def seek(self, _):
                if self.operation == "seek":
                    os.write(2, f"native decoder error for {late_failure}\n".encode())
                    raise RuntimeError(f"could not seek in {late_failure}")

            def get_batch(self, _):
                os.write(2, f"native decoder error for {late_failure}\n".encode())
                raise RuntimeError(f"could not decode {late_failure}")

        for operation in ("seek", "get_batch"):
            with self.subTest(operation=operation):
                dataset = self.dataset(self.manifest(f"{operation}.csv", late_failure, self.real))

                def video_reader(path, *args, **kwargs):
                    if path == late_failure:
                        return LateFailureReader(operation)
                    return actual_video_reader(path, *args, **kwargs)

                with everything_emitted() as emitted, \
                        mock.patch.object(video_dataset, "VideoReader", side_effect=video_reader), \
                        mock.patch.object(video_dataset.np.random, "randint", return_value=1):
                    buffer, _, _ = dataset[0]

                self.assertEqual(len(buffer[0]), 16)
                self.assertNoPath(emitted)
                self.assertSaid(emitted, "Failed to decode video: RuntimeError")
                self.assertSaid(emitted, "Retrying with new sample, failed to load sample 0 (manifest 0, line 1)")

    def test_an_unreadable_image_is_reported_by_its_manifest_line(self):
        image = os.path.join(self.folder, "frame.png")
        dataset = self.dataset(self.manifest("train.csv", image, self.real))
        with everything_emitted() as emitted, mock.patch.object(video_dataset.np.random, "randint", return_value=1):
            dataset[0]
        self.assertNoPath(emitted)
        self.assertSaid(emitted, "Failed to load image sample 0 (manifest 0, line 1): ")

    def test_s3_failures_give_the_reason_without_the_key(self):
        uri = f"s3://bucket/{IDENTIFIER}/video.mp4"
        dataset = self.dataset(self.manifest("train.csv", uri))
        s3 = mock.Mock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3.exceptions.ClientError = type("ClientError", (Exception,), {})
        dataset.s3_client = s3
        cases = [
            ("head_object", s3.exceptions.NoSuchKey(uri), "video object not found on S3"),
            ("head_object", s3.exceptions.ClientError(f"AccessDenied: {uri}"), "S3 access error: ClientError"),
            ("get_object", RuntimeError(f"connection reset reading {uri}"), "Failed to load video: RuntimeError"),
            ("get_object", {"Body": io.BytesIO(f"{IDENTIFIER} is not a video".encode())},
             "Failed to load video: RuntimeError"),
        ]
        for method, error, expected in cases:
            with self.subTest(expected=expected):
                s3.reset_mock(side_effect=True, return_value=True)
                s3.head_object.return_value = {"ContentLength": 10}
                if isinstance(error, dict):  # a corrupt object
                    getattr(s3, method).return_value = error
                else:
                    getattr(s3, method).side_effect = error
                with everything_emitted() as emitted:
                    self.assertEqual(dataset.loadvideo_decord(uri, 16), ([], None))
                self.assertNoPath(emitted)
                self.assertSaid(emitted, expected)
        with self.subTest(expected="Empty S3 object"):
            s3.reset_mock(side_effect=True, return_value=True)
            s3.head_object.return_value = {"ContentLength": 10}
            s3.get_object.return_value = {"Body": io.BytesIO(b"")}
            with everything_emitted() as emitted:
                self.assertEqual(dataset.loadvideo_decord(uri, 16), ([], None))
            self.assertNoPath(emitted)
            self.assertSaid(emitted, "Empty S3 object")


if __name__ == "__main__":
    unittest.main()
