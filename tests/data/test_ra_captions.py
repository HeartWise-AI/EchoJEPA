# tests/data/test_ra_captions.py

"""The right atrium's captions (`build_radar_captions_merged.ra_sentences`, routing registry v2) on synthetic report
cells, and the merged builder with registry v2: an RA row from a study with an A4C clip, and a pulmonic colour-flow
row from a colour clip of generic PSAX (`--regurgitation-grade-valves`). Uses this repository's column review module
unless `COLUMN_REVIEW_ROOT` names another checkout."""

import json
import os
import sys
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import build_radar_captions_merged as merged  # noqa: E402
import compare_column_prefill as cmp  # noqa: E402
import radar_normality as rn  # noqa: E402

from src.models.radar.routing import load_routing  # noqa: E402

ROOT = os.environ.get("COLUMN_REVIEW_ROOT") or REPO
CHECKOUT = os.path.isfile(os.path.join(ROOT, "tools", "labelbox", "column_review.py"))
RA, CONCLUSION = "Right Atrial findings", merged.CONCLUSION_COLUMN
AREA, AXIS = merged.RA_LIMITS["area"]["column"], merged.RA_LIMITS["axis"]["column"]
V2 = os.path.join(REPO, "configs", "radar", "radar_routing_v2.yaml")


@unittest.skipUnless(CHECKOUT, "no column review module")
class TestRaSentences(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cr, cls.spec = cmp.load_column_review(ROOT)

    def caption(self, **cells):
        report = {RA: cells.get("cell"), CONCLUSION: cells.get("conclusion"), AREA: cells.get("area"),
                  AXIS: cells.get("axis")}
        kept, flag, reason = merged.ra_sentences(report, RA, self.cr)
        text = " ".join(t for c in sorted(kept, key=lambda c: c != merged.RA_COLUMN) for t in kept[c][0])
        return text, flag, reason

    def test_the_cell_with_a_grade(self):
        self.assertEqual(self.caption(cell="OD légèrement dilatée."),
                         ("Right atrium is mildly dilated.", "abnormal", "written"))
        self.assertEqual(self.caption(cell="Oreillette droite sévèrement dilatée."),
                         ("Right atrium is severely dilated.", "abnormal", "written"))
        self.assertEqual(self.caption(cell="OD non dilatée.")[:2], ("Right atrium is not dilated.", "normal"))

    def test_a_grade_of_something_else_is_not_the_atriums(self):
        self.assertEqual(self.caption(cell="OD dilatée, IT importante.")[0], "Right atrium is dilated.")

    def test_the_area(self):
        self.assertEqual(self.caption(cell="OD dilatée.", area=24.4),
                         ("Right atrium is dilated. Right atrial area is 24 cm2.", "abnormal", "written"))
        self.assertEqual(self.caption(area=23.0)[0], "Right atrium is dilated. Right atrial area is 23 cm2.")
        self.assertEqual(self.caption(area=12.5)[:2],
                         ("Right atrium is not dilated. Right atrial area is 13 cm2.", "normal"))
        self.assertEqual(self.caption(area=18.0)[:2], ("Right atrial area is 18 cm2.", "unknown"))   # grey zone
        self.assertEqual(self.caption(area=500.0)[2], "no_ra_statement")                             # not valid

    def test_the_axis_sizes_only_without_text_or_area(self):
        self.assertEqual(self.caption(axis=6.2)[0], "Right atrium is dilated.")
        self.assertEqual(self.caption(axis=5.3)[2], "no_ra_statement")
        self.assertEqual(self.caption(cell="OD non dilatée.", axis=6.2)[0], "Right atrium is not dilated.")

    def test_a_measurement_that_contradicts_the_text_writes_nothing(self):
        self.assertEqual(self.caption(cell="OD non dilatée.", area=26.0), ("", "unknown", "text_area_conflict"))
        self.assertEqual(self.caption(cell="OD dilatée.", area=12.0)[2], "text_area_conflict")

    def test_the_conclusion(self):
        self.assertEqual(self.caption(conclusion="Dilatation de l'OD.")[0], "Right atrium is dilated.")
        self.assertEqual(self.caption(conclusion="VG dilaté et OD normale.")[2], "no_ra_statement")
        self.assertEqual(self.caption(cell="OD non dilatée.", conclusion="OD dilatée.")[0],
                         "Right atrium is not dilated.")         # the cell first

    def test_hedged_and_two_values(self):
        self.assertEqual(self.caption(cell="OD possiblement dilatée.")[2], "no_ra_statement")
        self.assertEqual(self.caption(cell="OD dilatée. OD non dilatée.")[2], "no_ra_statement")


@unittest.skipUnless(CHECKOUT, "no column review module")
class TestBuildWithRegistryV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cr, cls.spec = cmp.load_column_review(ROOT)
        cmp.index_columns(cls.spec)

    def test_ra_row_and_a_colour_row_from_generic_psax(self):
        routing = load_routing(V2)
        report = {"study_id": "S1", RA: "OD dilatée.", AREA: 25.0, CONCLUSION: "",
                  "Pulmonary valve findings": "Valve pulmonaire normale.", "Referring PR Grade": "1",
                  "StudyDate_Year": 2022}
        linked = pd.DataFrame([report])
        clips = pd.DataFrame([{"study_id": "S1", "view": "A4C"}, {"study_id": "S1", "view": "PSAX"}])
        color = pd.DataFrame([{"study_id": "S1", "view": "PSAX"}])
        reasons = {}
        table, _ = merged.build_tables(linked, clips, routing, rn.load_policy(), self.spec, self.cr,
                                       cmp.load_limits(), grade_valves=("pulmonic",), color_clips=color,
                                       grade_reasons=reasons)
        ra = table[table.structure == "ra"]
        self.assertEqual(ra.caption.tolist(), ["Right atrium is dilated. Right atrial area is 25 cm2."])
        self.assertEqual(ra.normality.tolist(), ["abnormal"])
        self.assertEqual([e[2] for e in json.loads(ra.sources.iloc[0])], ["ra_text", "ra_area"])
        pulmonic = table[table.structure == "pulmonic"].set_index("scope").caption.to_dict()
        self.assertIn("Mild pulmonic regurgitation.", pulmonic["color_flow"])
        self.assertNotIn("regurgitation", pulmonic.get("bmode", ""))


if __name__ == "__main__":
    unittest.main()
