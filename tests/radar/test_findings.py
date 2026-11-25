"""Report-finding extraction: structured columns, unreliable grade encodings, phrase hints with negation."""

import math
import unittest

from app.radar.findings import extract_findings, find_mentions, parse_number
from tests.radar.synthetic import ONTOLOGY


def extract(row):
    findings, issues = extract_findings(row, ONTOLOGY)
    return {f.concept_id: f for f in findings}, issues


class TestStructuredColumns(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(parse_number("42,5"), 42.5)
        self.assertEqual(parse_number("55 %"), 55.0)
        for value in ("55-60", "<20", "normale", "", None, True, math.nan):
            self.assertIsNone(parse_number(value))
        found, _ = extract({"Visually Estimated EF": "35", "RV Systolic Pressure": "42,5"})
        self.assertEqual(found["lv_ejection_fraction"].value, 35.0)
        self.assertEqual(found["rv_systolic_pressure"].value, 42.5)
        self.assertEqual(found["rv_systolic_pressure"].provenance.method, "structured_column")
        self.assertEqual(found["rv_systolic_pressure"].provenance.source_field, "RV Systolic Pressure")

    def test_zero_is_never_a_measurement(self):
        found, issues = extract({"RV Systolic Pressure": 0, "Visually Estimated EF": "0"})
        self.assertEqual(found, {})
        self.assertEqual({i.reason for i in issues}, {"out_of_range"})

    def test_mixed_grade_encodings_are_unknown_not_none(self):
        # The ICM grade columns mix 0 / 0.5 / 1-3, words and date-like strings: only words map.
        for raw in (0, 0.5, 1, 2.0, "3", "2+"):
            with self.subTest(raw=raw):
                found, issues = extract({"Referring MR Grade": raw})
                self.assertNotIn("mitral_regurgitation", found)
                self.assertEqual([(i.reason, i.raw_value) for i in issues], [("unmapped_value", str(raw))])
        found, issues = extract({"Referring MR Grade": "2022-01-04 00:00:00"})
        self.assertNotIn("mitral_regurgitation", found)
        self.assertEqual([(i.reason, i.raw_value) for i in issues], [("date_like_value", None)])
        for raw, value in (("Trace", "trace"), ("Légère", "mild"), ("modérée à sévère", "moderate_to_severe")):
            self.assertEqual(extract({"Referring MR Grade": raw})[0]["mitral_regurgitation"].value, value)

    def test_missing_values_are_skipped_silently(self):
        for raw in (None, math.nan, "", "  ", "nan", "None", "NaT"):
            with self.subTest(raw=raw):
                self.assertEqual(extract({"Referring TR Grade": raw}), ({}, []))

    def test_wall_motion_takes_the_worst_segment(self):
        found, _ = extract({"WMS 1": 1, "WMS 7": 3.0, "WMS 12": "2"})
        self.assertEqual(found["lv_regional_wall_motion"].value, "akinetic")
        self.assertEqual(found["lv_regional_wall_motion"].provenance.source_field, "WMS 7")

    def test_structured_column_beats_free_text(self):
        found, _ = extract({"Referring MR Grade": "Sévère", "Mitral valve findings": "Insuffisance mitrale légère."})
        self.assertEqual(found["mitral_regurgitation"].value, "severe")


class TestFreeText(unittest.TestCase):
    def test_severity_negation_and_span(self):
        text = "Valve aortique fine. Pas de sténose aortique. Insuffisance aortique modérée."
        found, _ = extract({"Aortic valve findings": text})
        self.assertEqual(found["aortic_stenosis"].value, "none")
        self.assertEqual(found["aortic_regurgitation"].value, "moderate")
        prov = found["aortic_regurgitation"].provenance
        self.assertEqual(prov.method, "phrase_hint")
        self.assertEqual(text[prov.span[0]:prov.span[1]].lower(), "insuffisance aortique")

    def test_nearest_severity_in_the_clause(self):
        found, _ = extract({"Report conclusions findings": "Insuffisance mitrale légère, insuffisance tricuspide "
                                                           "modérée. Épanchement péricardique minime."})
        self.assertEqual(found["mitral_regurgitation"].value, "mild")
        self.assertEqual(found["tricuspid_regurgitation"].value, "moderate")
        self.assertEqual(found["pericardial_effusion"].value, "trivial")

    def test_booleans_and_negated_booleans(self):
        found, _ = extract({"Mitral valve findings": "Calcification de l’anneau mitral.",
                            "Aortic valve findings": "Pas de bicuspidie."})
        self.assertIs(found["mitral_annular_calcification"].value, True)
        self.assertIs(found["bicuspid_aortic_valve"].value, False)

    def test_ambiguity_is_reported_not_guessed(self):
        found, issues = extract({"Report conclusions findings": "Dilatation du ventricule droit."})
        self.assertNotIn("rv_size", found)
        self.assertIn(("rv_size", "unresolved_mention"), [(i.concept_id, i.reason) for i in issues])
        found, issues = extract({"Mitral valve findings": "Insuffisance mitrale légère. Insuffisance mitrale sévère."})
        self.assertNotIn("mitral_regurgitation", found)
        self.assertIn("conflict", [i.reason for i in issues])
        # Free-text issues never carry text.
        self.assertTrue(all(i.raw_value is None for i in issues))

    def test_acronyms_are_case_sensitive_whole_words(self):
        self.assertEqual(len(find_mentions("HVG concentrique", ONTOLOGY, "lv_wall_thickness")), 1)
        self.assertEqual(find_mentions("hvgx", ONTOLOGY, "lv_wall_thickness"), [])


if __name__ == "__main__":
    unittest.main()
