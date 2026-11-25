# tests/datasets/test_loading_bounds.py

"""A sample that cannot be loaded is replaced by a random other one, but only a bounded number of
times: `VideoDataset` and `VideoGroupDataset` raise once `max_replacements` replacements in a row
have failed too, instead of spinning forever over unreadable storage. Below the bound the
replacements are drawn exactly as upstream draws them. And the loaders that `init_data` builds carry
the configured timeout, so a hung worker surfaces as an error.

The tests that could loop forever under a regression run in a forked child with a deadline.
"""

import multiprocessing
import os
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

import src.datasets.video_dataset as video_dataset
from src.datasets.data_manager import init_data
from src.datasets.video_dataset import (DEFAULT_LOADER_TIMEOUT, DEFAULT_MAX_REPLACEMENTS, VideoDataset,
                                        dataloader_timeout)
from src.datasets.video_group_dataset import VideoGroupDataset

IDENTIFIER = "MRN0012345"  # in every path: no message may show it
DEADLINE = 120  # seconds; a bounded loop over a few tiny files takes well under one


def write_video(path, frames=24, fps=8):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), 40 + 5 * i, np.uint8))
    writer.release()
    return path


def call_with_deadline(fn, seconds=DEADLINE):
    """`fn()` run in a forked child, which is killed after `seconds`: an unbounded retry loop fails
    the test instead of hanging the suite. `fn` returns a picklable value and handles the errors it
    expects; any other error fails the test."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child():
        try:
            result = ("returned", fn())
        except BaseException as e:  # reported to the parent, which fails the test
            result = ("raised", f"{type(e).__name__}: {e}")
        sender.send(result)

    process = context.Process(target=child, daemon=True)
    with warnings.catch_warnings():
        # Forking a multi-threaded process, as DataLoader workers do: the child only reads videos.
        warnings.filterwarnings("ignore", "This process .* is multi-threaded", DeprecationWarning)
        process.start()
    sender.close()
    try:
        if not receiver.poll(seconds):
            raise AssertionError(f"Still loading after {seconds} s: the replacement loop is unbounded.")
        outcome, value = receiver.recv()
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        receiver.close()
    if outcome != "returned":
        raise AssertionError(f"Unexpected error in the child: {value}")
    return value


def upstream_getitem(dataset, index):
    """`VideoDataset.__getitem__` before the bound: retries until a sample loads."""
    while True:
        sample_path = dataset.samples[index]
        if isinstance(sample_path, str):
            is_image = sample_path.split(".")[-1].lower() in ("jpg", "jpeg", "png")
            loaded = dataset.get_item_image(index) if is_image else dataset.get_item_video(index)
            if loaded:
                return loaded
        warnings.warn(f"Retrying with new sample, failed to load {dataset.describe(index)}.")
        index = np.random.randint(len(dataset))


class Files(unittest.TestCase):
    """Readable videos and three kinds of unreadable ones, in a folder named like a patient."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = os.path.join(tmp.name, IDENTIFIER)
        os.makedirs(self.folder)
        self.real = [write_video(os.path.join(self.folder, f"real{i}.mp4")) for i in range(4)]
        self.empty = os.path.join(self.folder, "empty.mp4")
        open(self.empty, "wb").close()
        self.garbage = os.path.join(self.folder, "garbage.mp4")
        with open(self.garbage, "wb") as f:
            f.write(f"{IDENTIFIER} is not a video".encode() * 64)
        self.missing = os.path.join(self.folder, "missing.mp4")
        self.broken = [self.empty, self.garbage, self.missing]


class TestVideoDatasetBound(Files):

    def manifest(self, *paths):
        """A manifest whose labels are the line numbers, so a returned label names its sample."""
        path = os.path.join(self.folder, f"{IDENTIFIER}_manifest.csv")
        with open(path, "w") as f:
            f.writelines(f"{p} {i}\n" for i, p in enumerate(paths))
        return path

    def dataset(self, *paths, **options):
        return VideoDataset(data_paths=[self.manifest(*paths)], frames_per_clip=16, fps=8, frame_step=None,
                            **options)

    def test_an_unreadable_manifest_raises_after_the_bound(self):
        dataset = self.dataset(*self.broken, max_replacements=4)

        def load():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    dataset[1]
                except RuntimeError as e:
                    retries = sum("Retrying with new sample" in str(w.message) for w in caught)
                    return str(e), retries
            return None

        outcome = call_with_deadline(load)
        self.assertIsNotNone(outcome, "an unreadable manifest returned a sample")
        message, retries = outcome
        self.assertEqual(retries, 4)  # the sample, then 4 replacements: 5 failed loads
        self.assertIn("sample 1 (manifest 0, line 2) and its 4 random replacements", message)
        self.assertIn("max_replacements=4", message)
        self.assertNotIn(IDENTIFIER, message)
        self.assertNotIn(self.folder, message)

    def test_the_default_bound_is_ten_replacements(self):
        self.assertEqual(DEFAULT_MAX_REPLACEMENTS, 10)
        dataset = self.dataset(*self.broken)
        self.assertEqual(dataset.max_replacements, 10)

        def load():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    dataset[0]
                except RuntimeError:
                    return sum("Retrying with new sample" in str(w.message) for w in caught)
            return None

        self.assertEqual(call_with_deadline(load), 10)

    def test_no_replacement_raises_at_once_without_drawing(self):
        dataset = self.dataset(self.garbage, *self.real, max_replacements=0)

        def load():
            with mock.patch.object(video_dataset.np.random, "randint") as randint, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    dataset[0]
                except RuntimeError as e:
                    return str(e), randint.call_count
            return None

        message, draws = call_with_deadline(load)
        self.assertEqual(draws, 0)
        self.assertIn("Failed to load sample 0 (manifest 0, line 1); max_replacements=0", message)

    def test_a_mixed_manifest_still_returns_a_replacement(self):
        dataset = self.dataset(self.garbage, self.real[0], max_replacements=1)
        with mock.patch.object(video_dataset.np.random, "randint", return_value=1), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buffer, label, clip_indices = dataset[0]
        self.assertEqual(label, 1)  # the replacement's own label
        self.assertEqual(len(buffer[0]), 16)

    def test_bad_bounds_are_refused(self):
        for bad in (-1, 1.5, None, True, "10"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "max_replacements"):
                self.dataset(*self.real, max_replacements=bad)

    def test_replacements_below_the_bound_draw_as_upstream(self):
        # Half the samples unreadable, so some items take several replacements; long videos draw a
        # random clip window, so the comparison covers the success path's draws too.
        paths = [p for pair in zip(self.real, self.broken + [self.garbage]) for p in pair]
        dataset = self.dataset(*paths)
        most_replacements = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for seed in range(12):
                for index in range(len(paths)):
                    np.random.seed(seed)
                    del caught[:]
                    expected = upstream_getitem(dataset, index)
                    expected_state = np.random.get_state()
                    replacements = sum("Retrying with new sample" in str(w.message) for w in caught)
                    if replacements > dataset.max_replacements:
                        continue  # upstream would keep going; the bounded loader raises
                    most_replacements = max(most_replacements, replacements)
                    np.random.seed(seed)
                    buffer, label, clip_indices = dataset[index]
                    state = np.random.get_state()
                    with self.subTest(seed=seed, index=index):
                        self.assertEqual(label, expected[1])
                        np.testing.assert_array_equal(np.asarray(buffer), np.asarray(expected[0]))
                        np.testing.assert_array_equal(np.asarray(clip_indices), np.asarray(expected[2]))
                        np.testing.assert_array_equal(state[1], expected_state[1])
                        self.assertEqual(state[2:], expected_state[2:])
        self.assertGreaterEqual(most_replacements, 2, "no item needed several replacements: the test is too weak")


class TestVideoGroupDatasetBound(Files):

    def manifest(self, rows):
        path = os.path.join(self.folder, f"{IDENTIFIER}_studies.csv")
        pd.DataFrame(rows, columns=["study_id", "A4C[0]", "RVINF[0]", "label"]).to_csv(path, index=False)
        return path

    def unreadable_studies(self, n):
        """`n` studies none of whose videos can be read (distinct files: a row never lists one twice)."""
        rows = []
        for i in range(n):
            a = os.path.join(self.folder, f"garbage{i}.mp4")
            with open(a, "wb") as f:
                f.write(b"not a video" * 32)
            rows.append((f"{IDENTIFIER}-{i}", a, os.path.join(self.folder, f"missing{i}.mp4"), float(i)))
        return rows

    def dataset(self, rows, **options):
        return VideoGroupDataset(self.manifest(rows), frames_per_clip=4, frame_step=2, **options)

    def test_unreadable_studies_raise_after_the_bound(self):
        dataset = self.dataset(self.unreadable_studies(3), max_replacements=2)

        def load():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    dataset[0]
                except RuntimeError as e:
                    return str(e), sum("Retrying with another study" in str(w.message) for w in caught)
            return None

        outcome = call_with_deadline(load)
        self.assertIsNotNone(outcome, "a study was returned although no video can be read")
        message, retries = outcome
        self.assertEqual(retries, 2)
        self.assertIn("No video of study row 0 and 2 random substitute studies", message)
        self.assertIn("max_replacements=2", message)
        self.assertNotIn(IDENTIFIER, message)

    def test_the_default_bound_is_ten_substitutes(self):
        self.assertEqual(self.dataset(self.unreadable_studies(1)).max_replacements, 10)

    def test_a_substitute_study_keeps_its_own_label(self):
        rows = self.unreadable_studies(1) + [("readable", self.real[0], self.real[1], 7.5)]
        dataset = self.dataset(rows, max_replacements=1)
        with mock.patch("src.datasets.video_group_dataset.np.random.randint", return_value=1), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            item = dataset[0]
        self.assertEqual((item["index"], item["label"]), (1, 7.5))  # videos and label of the same study
        self.assertEqual(item["present"].tolist(), [True, True])

    def test_evaluation_still_counts_an_unreadable_study_instead_of_replacing_it(self):
        dataset = self.dataset(self.unreadable_studies(2), max_replacements=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(dataset.get_item_study(0))


class TestLoaderTimeout(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.manifest = os.path.join(tmp.name, "train.csv")
        with open(self.manifest, "w") as f:
            f.write("/synthetic/videos/0001.mp4 0\n")  # never read: no batch is drawn

    def loader(self, data="VideoDataset", **options):
        loader, _ = init_data(data=data, root_path=[self.manifest], batch_size=1, clip_len=16, frame_sample_rate=4,
                              **options)
        return loader

    def test_the_configured_timeout_reaches_a_loader_with_workers(self):
        self.assertEqual(self.loader(num_workers=2, loader_timeout=900).timeout, 900)
        self.assertEqual(self.loader(num_workers=2, loader_timeout=12.5).timeout, 12.5)

    def test_no_timeout_unless_configured(self):
        self.assertEqual(self.loader(num_workers=2).timeout, 0)

    def test_without_workers_the_timeout_is_dropped(self):
        # PyTorch refuses a timeout for loading in the main process (at the first batch).
        loader = self.loader(num_workers=0, loader_timeout=900)
        self.assertEqual(loader.timeout, 0)

    def test_bad_timeouts_are_refused(self):
        for bad in (-1, None, True, float("nan"), float("inf"), "900"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "loader_timeout"):
                self.loader(num_workers=2, loader_timeout=bad)
        with self.assertRaisesRegex(ValueError, "loader_timeout must be"):
            dataloader_timeout(-5, 0)  # validated even when no worker would use it

    def test_another_dataset_type_refuses_a_timeout(self):
        with self.assertRaisesRegex(ValueError, "loader_timeout=900 is only supported by VideoDataset"):
            self.loader(data="ImageNet", loader_timeout=900)

    def test_the_default_waits_long_enough_for_a_slow_read(self):
        self.assertGreaterEqual(DEFAULT_LOADER_TIMEOUT, 600)
        self.assertLessEqual(DEFAULT_LOADER_TIMEOUT, 1800)


if __name__ == "__main__":
    unittest.main()
