# tests/data/test_still_overlay_audit.py

"""`data/still_overlay_audit.py` on synthetic arrays: the features respond to a thin white mark and to a filled box,
the AUROC, and the caliper row found at the value's row."""

import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import still_overlay_audit as audit  # noqa: E402


def spectrum(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 120, size=(200, 300)).astype(np.uint8)


class TestOverlayAudit(unittest.TestCase):

    def test_features_respond_to_a_mark_and_to_a_filled_box(self):
        clean = spectrum()
        marked = clean.copy()
        marked[50, 100:140] = 255                               # a thin white caliper line
        boxed = clean.copy()
        boxed[20:50, 200:280] = 77                              # a filled label box
        self.assertEqual(audit.thin_bright_mask(clean).sum(), 0)
        self.assertGreaterEqual(audit.thin_bright_mask(marked).sum(), 40)
        self.assertEqual(audit.flat_share(clean), 0.0)
        self.assertGreater(audit.flat_share(boxed), 0.01)
        self.assertEqual(audit.flat_share(np.zeros((50, 50), dtype=np.uint8)), 0.0)     # black is not a footprint

    def test_auroc(self):
        self.assertEqual(audit.auroc([2, 3], [0, 1]), 1.0)
        self.assertEqual(audit.auroc([0, 1], [2, 3]), 0.0)
        self.assertEqual(audit.auroc([1, 1], [1, 1]), 0.5)
        self.assertTrue(np.isnan(audit.auroc([], [1])))

    def test_marker_row_at_the_value(self):
        image = spectrum(1)
        top, bottom = 1.0, -5.0                                 # baseline at row 199 / 6
        row = int(round(audit.value_rows(4.0, top, bottom, 200)[1]))      # -4 m/s, below the baseline
        image[row, 120:150] = 255
        image[int(round(audit.value_rows(0.0, top, bottom, 200)[0])), :] = 255        # the baseline is ignored
        found = audit.marker_row(image, top, bottom)
        self.assertEqual(found, row)
        self.assertTrue(audit.at_value(found, 4.0, top, bottom, 200))
        self.assertFalse(audit.at_value(found, 2.0, top, bottom, 200))
        self.assertIsNone(audit.marker_row(spectrum(2), top, bottom))
        self.assertFalse(audit.at_value(None, 4.0, top, bottom, 200))


if __name__ == "__main__":
    unittest.main()
