# tests/datasets/test_data_loader_options.py

"""make_videodataset passes only the DataLoader options the installed PyTorch supports."""

import inspect
import os
import tempfile
import unittest
from unittest import mock

import torch

from src.datasets.video_dataset import make_videodataset


def dataloader_without_in_order(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None,
                                num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None, multiprocessing_context=None, generator=None, *,
                                prefetch_factor=None, persistent_workers=False, pin_memory_device=""):
    """The DataLoader signature of PyTorch 2.3, which has no `in_order`."""
    return "loader"


class TestInOrder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.manifest = os.path.join(tmp.name, "manifest.csv")
        with open(self.manifest, "w") as f:
            f.write("/synthetic/videos/0001.mp4 0\n")  # never read: no batch is drawn

    def loader(self, **options):
        _, loader, _ = make_videodataset(data_paths=[self.manifest], batch_size=1, num_workers=1,
                                         persistent_workers=False, **options)
        return loader

    def test_default_builds_on_a_pytorch_without_in_order(self):
        with mock.patch.object(torch.utils.data, "DataLoader", dataloader_without_in_order):
            self.assertEqual(self.loader(), "loader")

    def test_out_of_order_on_a_pytorch_without_in_order_warns_and_stays_in_order(self):
        with mock.patch.object(torch.utils.data, "DataLoader", dataloader_without_in_order):
            with self.assertLogs(level="WARNING") as logs:
                self.assertEqual(self.loader(in_order=False), "loader")
        self.assertIn("`in_order=False` ignored", "\n".join(logs.output))

    def test_out_of_order_reaches_a_pytorch_that_supports_it(self):
        if "in_order" not in inspect.signature(torch.utils.data.DataLoader).parameters:
            self.skipTest("this PyTorch has no DataLoader in_order")
        self.assertFalse(self.loader(in_order=False).in_order)
        self.assertTrue(self.loader().in_order)


if __name__ == "__main__":
    unittest.main()
