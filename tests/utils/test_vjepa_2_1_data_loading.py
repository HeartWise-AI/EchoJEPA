# tests/utils/test_vjepa_2_1_data_loading.py

"""The V-JEPA 2.1 trainer's guard around `next(loader)` (`_next_batch`): a loading error, such as a
`DataLoader` timeout or a sample whose replacements all failed, is retried a few times, then stops
the run instead of being swallowed; a timed-out batch that is only slow still arrives on a retry;
and `data.loader_timeout` reaches the loader."""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from app.vjepa_2_1 import train
from src.datasets.video_dataset import DEFAULT_LOADER_TIMEOUT
from tests.utils.test_vjepa_2_1_baseline import QuietLogs, Stop, trainer_on_cpu
from tests.utils.test_vjepa_2_1_wandb import config, fake_init_data

TIMED_OUT = "DataLoader timed out after 900 seconds"


class Raising:
    """An iterator that raises `errors` in turn, then yields `batches`."""

    def __init__(self, errors=(), batches=()):
        self.errors, self.batches, self.calls = list(errors), list(batches), 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.batches:
            return self.batches.pop(0)
        raise StopIteration


class TestNextBatch(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(train.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)
        self.refresh = mock.Mock(side_effect=AssertionError("refreshed"))

    def test_a_loader_that_keeps_failing_raises_after_the_retries(self):
        loader = Raising(errors=[RuntimeError(TIMED_OUT) for _ in range(100)])
        with self.assertLogs(train.logger, "WARNING"), \
                self.assertRaisesRegex(RuntimeError, r"Exceeded max retries \(5\) when loading data") as raised:
            train._next_batch(loader, self.refresh)
        self.assertEqual(loader.calls, 6)  # the first try and 5 retries
        self.assertEqual(self.sleep.call_args_list, [mock.call(5)] * 5)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(str(raised.exception.__cause__), TIMED_OUT)
        self.refresh.assert_not_called()

    def test_a_transient_error_is_retried(self):
        loader = Raising(errors=[RuntimeError(TIMED_OUT), OSError("a stale handle")], batches=["batch"])
        with self.assertLogs(train.logger, "WARNING") as logs:
            self.assertEqual(train._next_batch(loader, self.refresh), ("batch", loader))
        self.assertEqual(len(logs.output), 2)
        self.assertNotIn("stale handle", "\n".join(logs.output))  # only the error type is logged

    def test_an_exhausted_pass_is_refreshed(self):
        fresh = Raising(batches=["next pass"])
        refresh = mock.Mock(return_value=fresh)
        self.assertEqual(train._next_batch(Raising(), refresh), ("next pass", fresh))
        refresh.assert_called_once_with()

    def test_a_loader_without_any_batch_raises_instead_of_refreshing_forever(self):
        refresh = mock.Mock(side_effect=[Raising(), Raising(), AssertionError("refreshed again")])
        with self.assertRaisesRegex(RuntimeError, "yields no batch"):
            train._next_batch(Raising(), refresh)
        refresh.assert_called_once_with()

    def test_a_failing_refresh_stops_the_run(self):
        # With persistent workers, starting a new pass waits for every worker; a hung one times out.
        refresh = mock.Mock(side_effect=RuntimeError(TIMED_OUT))
        with self.assertRaisesRegex(RuntimeError, TIMED_OUT):
            train._next_batch(Raising(), refresh)


class Sleepy(torch.utils.data.Dataset):
    """Each item takes `seconds` to load, as a hung or a slow read does."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __len__(self):
        return 1

    def __getitem__(self, i):
        time.sleep(self.seconds)
        return torch.zeros(1)


class TestLoaderTimeout(unittest.TestCase):
    """A real `DataLoader` with one worker and a 1 s timeout."""

    def next_batch(self, seconds, max_retries):
        loader = torch.utils.data.DataLoader(Sleepy(seconds), num_workers=1, timeout=1)
        iterator = iter(loader)
        try:
            return train._next_batch(iterator, mock.Mock(side_effect=AssertionError("refreshed")),
                                     max_retries=max_retries, delay=0)
        finally:
            iterator._shutdown_workers()  # stops the sleeping worker now (join, then terminate)

    def test_a_hung_worker_times_out_and_stops_the_run(self):
        start = time.monotonic()
        with self.assertLogs(train.logger, "WARNING"), \
                self.assertRaisesRegex(RuntimeError, r"Exceeded max retries \(2\)") as raised:
            self.next_batch(seconds=600, max_retries=2)
        self.assertIn("DataLoader timed out after 1 seconds", str(raised.exception.__cause__))
        self.assertLess(time.monotonic() - start, 60)

    def test_a_slow_batch_still_arrives_after_a_timeout(self):
        # The retry waits again on the same iterator, for the same batch.
        with self.assertLogs(train.logger, "WARNING") as logs:
            batch, _ = self.next_batch(seconds=2.5, max_retries=5)
        self.assertEqual(batch.tolist(), [[0.0]])
        self.assertTrue(all("RuntimeError" in line for line in logs.output))


class TimingOut:
    """A loader whose every batch times out, as one with a hung worker does."""

    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 2

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        raise RuntimeError(TIMED_OUT)


class TestTrainerLoading(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.store = tempfile.TemporaryDirectory()
        dist.init_process_group("gloo", init_method=f"file://{cls.store.name}/store", rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.store.cleanup()
        super().tearDownClass()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = os.path.join(tmp.name, "pretrain")
        os.makedirs(self.folder)

    def run_trainer(self, cfg, init_data):
        with trainer_on_cpu(), mock.patch.object(train, "init_data", side_effect=init_data), \
                mock.patch.object(train.time, "sleep") as sleep, \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0):
            train.main(cfg)
        return sleep

    def loader_timeout_of(self, cfg):
        options = {}

        def capture(**kwargs):
            options.update(kwargs)
            raise Stop

        with self.assertRaises(Stop):
            self.run_trainer(cfg, capture)
        return options["loader_timeout"]

    def test_the_shipped_configs_set_the_default_timeout(self):
        self.assertEqual(config(self.folder, wandb=False)["data"]["loader_timeout"], DEFAULT_LOADER_TIMEOUT)

    def test_the_configured_timeout_reaches_the_loader(self):
        cfg = config(self.folder, wandb=False)
        cfg["data"]["loader_timeout"] = 123
        self.assertEqual(self.loader_timeout_of(cfg), 123)
        del cfg["data"]["loader_timeout"]
        self.assertEqual(self.loader_timeout_of(cfg), DEFAULT_LOADER_TIMEOUT)

    def test_a_loader_that_keeps_timing_out_stops_the_trainer(self):
        loader = TimingOut()
        with self.assertRaisesRegex(RuntimeError, r"Exceeded max retries \(5\)") as raised:
            self.run_trainer(config(self.folder, wandb=False),
                             lambda **_: (loader, SimpleNamespace(set_epoch=lambda epoch: None)))
        self.assertEqual(str(raised.exception.__cause__), TIMED_OUT)
        self.assertEqual(loader.calls, 6)

    def test_a_working_loader_trains(self):
        self.run_trainer(config(self.folder, wandb=False), fake_init_data)
        self.assertTrue(os.path.exists(os.path.join(self.folder, "latest.pth.tar")))


if __name__ == "__main__":
    unittest.main()
