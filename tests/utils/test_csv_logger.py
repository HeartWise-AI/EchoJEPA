# tests/utils/test_csv_logger.py

"""`src.utils.logging.CSVLogger` across a resume: one header, rows appended under it."""

import os
import tempfile
import unittest

from src.utils.logging import CSVLogger

COLUMNS = (("%d", "epoch"), ("%.1f", "val_mae"))


class TestCSVLogger(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "log_r0.csv")

    def read(self):
        with open(self.path) as f:
            return f.read().splitlines()

    def test_a_resumed_run_keeps_one_header(self):
        CSVLogger(self.path, *COLUMNS).log(1, 5.0)
        CSVLogger(self.path, *COLUMNS).log(2, 4.5)  # the restarted job
        self.assertEqual(self.read(), ["epoch,val_mae", "1,5.0", "2,4.5"])

    def test_an_empty_file_gets_its_header(self):
        open(self.path, "w").close()
        CSVLogger(self.path, *COLUMNS).log(1, 5.0)
        self.assertEqual(self.read(), ["epoch,val_mae", "1,5.0"])

    def test_write_mode_starts_over(self):
        CSVLogger(self.path, *COLUMNS).log(1, 5.0)
        CSVLogger(self.path, *COLUMNS, mode="w").log(1, 4.0)
        self.assertEqual(self.read(), ["epoch,val_mae", "1,4.0"])


if __name__ == "__main__":
    unittest.main()
