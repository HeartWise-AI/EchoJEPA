# tests/data/test_still_support_audit.py

"""The finding-support audit: an LVOT-only study that reads an aortic stenosis statement is counted as unsupported,
an AV still supports it, a study without a feeding still is not counted, and small counts are suppressed."""

import os
import sys
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import still_support_audit as ssa  # noqa: E402

from src.models.radar.routing import DEFAULT_ROUTING, load_routing  # noqa: E402

V1 = load_routing(os.path.join(os.path.dirname(DEFAULT_ROUTING), "radar_routing_v1.yaml"))


class TestAudit(unittest.TestCase):

    def test_lvot_only_evidence_is_counted(self):
        captions, stills = [], []
        for n in range(12):
            study = f"S{n:02d}"
            text = "Sténose aortique modérée." if n < 10 else "Insuffisance aortique légère."
            captions.append({"study_id": study, "structure": "aortic", "caption": text, "scope": "full"})
            captions.append({"study_id": study, "structure": "aortic", "caption": text, "scope": "doppler"})
            captions.append({"study_id": study, "structure": "aortic", "caption": "Valve calcifiée.",
                             "scope": "bmode"})
            tags = ["LVOT"] if n < 6 else ["LVOT", "AV"] if n < 9 else ["MR"] if n == 9 else ["AR"]
            stills += [{"study_id": study, "still_tag": tag} for tag in tags]
        report = ssa.audit(pd.DataFrame(captions), pd.DataFrame(stills), V1)["aortic"]
        self.assertEqual(report["captions_with_feeding_stills"], 11)        # S09 has no still feeding the valve
        stenosis = report["stenosis"]
        self.assertEqual((stenosis["stated"], stenosis["supported"], stenosis["unsupported"]), (9, "<5", 6))
        self.assertEqual(stenosis["unsupported_share"], round(6 / 9, 3))
        self.assertEqual(stenosis["unsupported_by_tags"], {"LVOT": 6})
        self.assertEqual(report["regurgitation"]["stated"], "<5")
        self.assertIsNone(report["regurgitation"]["unsupported_share"])

    def test_negated_statement_and_accents(self):
        self.assertRegex(ssa.plain("Pas de STÉNOSE aortique"), ssa.STENOSIS)
        self.assertRegex(ssa.plain("Rétrécissement aortique serré"), ssa.STENOSIS)
        self.assertNotRegex(ssa.plain("Valve aortique calcifiée"), ssa.STENOSIS)


if __name__ == "__main__":
    unittest.main()
