# tests/data/test_build_radar_captions_merged.py

"""`data/build_radar_captions_merged.py` on synthetic reports, through the column review's own code (needs a column
review checkout in `COLUMN_REVIEW_ROOT`): prefill sentences, findings dropped by the checks, parser and EF fills, the
wall segment summary, the flag, and a table the RADAR dataset accepts."""

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

POLICY = rn.load_policy()
ROUTING = load_routing()
LIMITS = cmp.load_limits()
COLUMN_REVIEW_ROOT = os.environ.get("COLUMN_REVIEW_ROOT", "")
CHECKOUT = bool(COLUMN_REVIEW_ROOT) and os.path.isfile(
    os.path.join(COLUMN_REVIEW_ROOT, "tools", "labelbox", "column_review.py"))
EF, LVIDD = POLICY.ef["column"], "LVID Diastole (2D)"
LV, PERI = "Left ventricule findings", "Pericardium findings"


@unittest.skipUnless(CHECKOUT, "COLUMN_REVIEW_ROOT does not name a column review checkout")
class TestMergedCaptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cr, cls.spec = cmp.load_column_review(COLUMN_REVIEW_ROOT)
        cmp.index_columns(cls.spec)

    def row(self, structure, report, views=("A4C", "PLAX")):
        entry = next(s for s in ROUTING.trained if s.name == structure)
        return merged.merged_row("S1", entry, report, set(views) & set(entry.views), POLICY, self.spec, self.cr,
                                 LIMITS)

    def sources(self, row):
        return {column: (value, source) for column, value, source in json.loads(row["sources"])}

    def test_prefill_sentences_in_english(self):
        row, dropped = self.row("lv", {LV: "Ventricule gauche non dilaté. Fonction systolique normale.",
                                       LVIDD: 4.8, EF: 60.0})
        self.assertEqual(dropped, [])
        self.assertIn("LV cavity is not dilated.", row["caption"])
        self.assertIn("LV systolic function appears normal.", row["caption"])
        self.assertEqual(self.sources(row)["lv_size"], ("normal", "prefill_text"))
        self.assertNotRegex(row["caption"], r"\d")                 # no EF number, no measurement
        self.assertEqual(row["normality_status"], "provisional")

    def test_a_contradicted_finding_is_dropped_and_listed(self):
        row, dropped = self.row("lv", {LV: "Ventricule gauche non dilaté. Fonction systolique normale.",
                                       LVIDD: 6.2, EF: 60.0})
        self.assertNotIn("not dilated", row["caption"])
        self.assertEqual([(d["column"], d["reason"]) for d in dropped], [("lv_size", "contradiction")])
        self.assertEqual(row["normality"], "unknown")              # the parser's own flag is not normal here

    def test_effusion_wording_and_side(self):
        row, _ = self.row("pericardium", {PERI: "Pas d'épanchement péricardique."})
        self.assertEqual(row["caption"], "No pericardial effusion.")
        self.assertEqual(row["normality"], "normal")
        row, _ = self.row("pericardium", {PERI: "Épanchement péricardique léger sans signe de tamponnade."})
        self.assertEqual(row["caption"], "Pericardial effusion is present.")
        self.assertEqual(row["normality"], "abnormal")

    def test_ef_band_fills_a_silent_lv_function(self):
        row, _ = self.row("lv", {LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 35.0})
        self.assertEqual(self.sources(row)["lv_function"], ("moderately_reduced", "ef_band"))
        self.assertIn("LV systolic function appears moderately reduced.", row["caption"])
        self.assertEqual(row["normality"], "abnormal")
        row, _ = self.row("lv", {LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 53.0})     # a grey zone
        self.assertNotIn("lv_function", self.sources(row))

    def test_nothing_to_say_gives_no_row(self):
        row, dropped = self.row("pericardium", {PERI: ""})
        self.assertIsNone(row)
        self.assertEqual(dropped, [])

    def test_wall_segments_are_summarised(self):
        report = {LV: "Ventricule gauche non dilaté.", LVIDD: 4.8}
        segments = [c for c in self.spec["fields"] if c.startswith("wall_")]
        self.assertEqual(len(segments), 16)
        normal = {**report, **{c: 1.0 for c in cmp.report_columns(ROUTING, POLICY, self.spec, LIMITS)
                               if c.startswith("WMS ")}}
        row, _ = self.row("lv", normal)
        self.assertEqual(row["caption"].count("segment"), 0)
        self.assertIn(merged.WALLS_NORMAL, row["caption"])
        one = {**normal, "WMS 4C Apical septum": 2.0}
        row, _ = self.row("lv", one)
        self.assertEqual(row["caption"].count("segment shows"), 1)
        self.assertNotIn(merged.WALLS_NORMAL, row["caption"])

    def test_hypertrophy_mass_and_remodelling_are_written_from_the_parser(self):
        base = "Ventricule gauche non dilaté. Fonction systolique normale. "
        row, _ = self.row("lv", {LV: base + "Hypertrophie ventriculaire gauche. Masse augmentée.", LVIDD: 4.8})
        self.assertIn("LV hypertrophy is present.", row["caption"])
        self.assertIn("LV mass is increased.", row["caption"])
        self.assertEqual(self.sources(row)["lv_hypertrophy"], ("abnormal", "parser"))
        self.assertEqual(row["normality"], "abnormal")
        row, _ = self.row("lv", {LV: base + "Pas d'hypertrophie. Masse normale.", LVIDD: 4.8})
        self.assertIn("No LV hypertrophy.", row["caption"])
        self.assertIn("LV mass is normal.", row["caption"])
        self.assertNotIn("is present", row["caption"])
        row, _ = self.row("lv", {LV: base + "Remodelage concentrique.", LVIDD: 4.8})
        self.assertIn("LV remodelling is present.", row["caption"])
        row, _ = self.row("lv", {LV: base, LVIDD: 4.8, "IVS Diastole Thickness (2D)": 1.6,
                                 "LVIW Diastolic Thickness (2D)": 1.6})
        self.assertNotIn("hypertrophy", row["caption"])            # a measurement alone writes no sentence

    def test_disagreeing_claims_write_no_extra_sentence(self):
        facts = [{"source": "text", "attribute": "lv_mass", "finding": "state", "value": "normal"},
                 {"source": "text", "attribute": "lv_mass", "finding": "state", "value": "abnormal"},
                 {"source": "text", "attribute": "lv_wall_thickness", "finding": "hypertrophy", "value": "uncertain"},
                 {"source": "structured", "attribute": "lv_geometry", "finding": "remodelling", "value": "abnormal"}]
        self.assertEqual(merged.extra_sentences(facts, "lv"), {})
        self.assertEqual(merged.extra_sentences(facts, "rv"), {})

    def test_an_enlarged_left_atrium_is_written_as_dilated(self):
        row, _ = self.row("la", {"Left atrial findings": "Volume indexé légèrement augmenté."})
        if row is not None and "la_size" in self.sources(row):
            self.assertEqual(row["caption"], "Left atrium is dilated.")
            self.assertEqual(row["normality"], "abnormal")
        else:
            self.fail("an increased left atrial volume wrote no size sentence")

    def test_table_is_read_by_the_radar_dataset_columns(self):
        reports = pd.DataFrame([{"study_id": "S1", PERI: "Pas d'épanchement péricardique.",
                                 LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 60.0},
                                {"study_id": "S2", PERI: "", LV: ""}])
        clips = pd.DataFrame({"study_id": ["S1", "S1", "S2"], "view": ["A4C", "PLAX", "A4C"]})
        table, dropped = merged.build_tables(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS)
        self.assertEqual(list(table.columns), list(merged.COLUMNS))
        self.assertEqual(set(table.study_id), {"S1"})
        self.assertFalse(table.duplicated(["study_id", "structure"]).any())
        self.assertTrue(set(table.normality) <= {"normal", "abnormal", "unknown"})
        self.assertTrue((table.caption.str.len() > 0).all())
        self.assertEqual(table.caption_id.dtype.kind, "i")


if __name__ == "__main__":
    unittest.main()
