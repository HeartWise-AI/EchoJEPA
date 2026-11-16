"""Audit artifacts must form a complete run and never overwrite existing outputs."""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools import radar_still_leakage_audit as audit


class TestAuditArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.panels = [("synthetic", np.zeros((audit.SIDE, audit.SIDE), dtype=np.uint8))]

    def test_existing_symlink_targets_are_untouched_and_runs_are_distinct(self):
        victim = self.root / "external.txt"
        victim.write_text("preserve me")
        out = self.root / "out"
        out.mkdir()
        for name in ("synthetic_metrics.json", "synthetic_panel.png"):
            (out / name).symlink_to(victim)
        first = audit.publish_evidence(out, {"run": 1}, self.panels)
        second = audit.publish_evidence(out, {"run": 2}, self.panels)
        self.assertNotEqual(first, second)
        self.assertEqual(victim.read_text(), "preserve me")
        for directory, number in ((first, 1), (second, 2)):
            self.assertEqual(json.loads((directory / "synthetic_metrics.json").read_text()), {"run": number})
            self.assertTrue((directory / "synthetic_panel.png").read_bytes().startswith(b"\x89PNG"))

    def test_panel_failure_publishes_nothing_and_preserves_previous_run(self):
        out = self.root / "out"
        previous = audit.publish_evidence(out, {"run": 1}, self.panels)
        with mock.patch.object(audit, "write_panel", side_effect=OSError("simulated write failure")):
            with self.assertRaisesRegex(OSError, "simulated"):
                audit.publish_evidence(out, {"run": 2}, self.panels)
        self.assertEqual(list(out.iterdir()), [previous])
        self.assertEqual(json.loads((previous / "synthetic_metrics.json").read_text()), {"run": 1})

    def test_provenance_records_environment_and_binds_output_bytes(self):
        run = audit.publish_evidence(self.root / "out", {"fixture": True}, self.panels)
        provenance = json.loads((run / "provenance.json").read_text())
        self.assertIn("torch", provenance["packages"])
        self.assertIsNone(provenance["test_results"])
        self.assertEqual(set(provenance["source_sha256"]), {"audit_tool", "preprocessing"})
        for name, digest in provenance["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), digest)
