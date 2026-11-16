# tests/data/test_still_leak_audit.py

"""`data/still_leak_audit.py`: reading numbers from OCR lines, matching them to measured values across units, and
the suppression of small counts. (The OCR step itself needs the OCR environment and is not run here.)"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import still_leak_audit as audit  # noqa: E402


class TestLeakAudit(unittest.TestCase):

    def test_numbers_and_matches(self):
        read = audit.numbers_in(["AV Vmax 412 cm/s", "2,5", "-1.0"])
        self.assertEqual(read, [412.0, 2.5, 1.0])
        self.assertTrue(audit.matches(read, [4.12]))            # m/s stored, cm/s on the screen
        self.assertTrue(audit.matches(read, [-2.5]))            # the sign is not compared
        self.assertTrue(audit.matches([231.0], [0.231]))        # s stored, ms on the screen
        self.assertFalse(audit.matches(read, [3.3]))
        self.assertFalse(audit.matches([0.0], [0.0004]))        # too small to compare
        self.assertFalse(audit.matches([], [4.12]))

    def test_small_counts_are_suppressed(self):
        self.assertEqual([audit.suppressed(n) for n in (0, 1, 4, 5)], [0, "<5", "<5", 5])

    def test_folders_inside_the_repository_are_refused(self):
        with self.assertRaises(SystemExit):
            audit.outside_repo(os.path.join(REPO, "x"), "--work")


if __name__ == "__main__":
    unittest.main()
