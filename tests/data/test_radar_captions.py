# tests/data/test_radar_captions.py

"""`data/radar_captions.py`: missing cells never become text, canonical captions and stable ids, EF statements
(points, intervals, bounds, hedges, clause boundaries; not any percentage), the appended LV value and its conflicts,
and the EF rule on whole ranges. `test_radar_normality.py` covers caption flags. The phrases are synthetic."""

import math
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import radar_captions as rc  # noqa: E402


class TestMissingAndIds(unittest.TestCase):

    def test_missing_cells_are_empty_never_nan(self):
        try:
            import pandas as pd
            na = [pd.NA]
        except ImportError:
            na = []
        for value in [None, float("nan"), math.nan, 3.0, "   ", *na]:
            with self.subTest(value=value):
                self.assertEqual(rc.caption_text(value), "")
        self.assertEqual(rc.lv_caption(float("nan"), 60.0), ("FEVG estimée visuellement : 60 %.", "appended"))
        self.assertEqual(rc.lv_caption(None, float("nan")), ("", "none"))

    def test_canonical_form(self):
        self.assertEqual(rc.canonical_caption("  Pas d’épanchement   péricardique. "),
                         "pas d'épanchement péricardique")
        self.assertEqual(rc.canonical_caption("FEVG 52,5 %."), "fevg 52,5 %")
        with self.assertRaises(TypeError):
            rc.canonical_caption(float("nan"))

    def test_ids_are_stable_across_processes_and_refuse_empty(self):
        text = "Valve aortique tricuspide. // Pas d'insuffisance aortique"
        here = rc.caption_id(text)
        self.assertEqual(here, rc.caption_id(text.upper()))
        code = ("import sys; sys.path.append(sys.argv[1]); import radar_captions as rc; "
                "print(rc.caption_id(sys.argv[2]))")
        there = subprocess.run([sys.executable, "-c", code, os.path.join(REPO, "data"), text],
                               capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": "1"})
        self.assertEqual(int(there.stdout), here)
        for empty in ["", " . "]:
            with self.assertRaises(ValueError):
                rc.caption_id(empty)


class TestEfStatements(unittest.TestCase):

    def test_statements(self):
        S = rc.EfStatement
        cases = {
            "FEVG estimée à 55 %.": [S(55, 55)],
            "Fraction d'éjection du ventricule gauche 60%": [S(60, 60)],
            "FE VG 55-60 %": [S(55, 60)],
            "FEVG estimée entre 50 et 60 %": [S(50, 60)],
            "FE à 52,5 %": [S(52.5, 52.5)],
            "FEVG < 55 %": [S(5, 55, high_open=True)],
            "FEVG ≤ 40 %": [S(5, 40)],
            "FEVG supérieure à 55 %": [S(55, 90, low_open=True)],
            "FEVG environ 55 %": [S(55, 55, certain=False)],
            "FEVG probablement 35 %": [S(35, 35, certain=False)],
            "Probablement FEVG 60 %": [S(60, 60, certain=False)],                # hedge before the label
            "FEVG 60 % environ": [S(60, 60, certain=False)],                      # after the percentage
            "FEVG 60 % ?": [S(60, 60, certain=False)],
            "FEVG 60 % +/- 5": [S(60, 60, certain=False)],
            "FEVG 55 % ou moins": [S(55, 55, certain=False)],
            "FEVG 55 % (limite inférieure de la normale)": [S(55, 55, certain=False)],
            "FEVG 60 % au Simpson biplan": [S(60, 60)],                           # unqualified rest of clause
            "Probablement dilaté. FEVG 60 %": [S(60, 60)],                        # a hedge in another clause
            "FEVG < 55 %, FE 50 %": [S(5, 55, high_open=True), S(50, 50)],        # other EF wording is not a hedge
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(rc.find_ef(text), expected)

    def test_not_ef(self):
        for text in ["FEVG non mesurée. Volume normal 60 %", "FEVG normale. Strain longitudinal global -18 %",
                     "FE (strain) 18 %", "Hypertrophie septale 15 %", "FEVG 150 %", "FEVG normale, VD 60 %"]:
            with self.subTest(text):
                self.assertEqual(rc.find_ef(text), [])

    def test_lv_caption(self):
        self.assertEqual(rc.lv_caption("VG non dilaté.", 60.0),
                         ("VG non dilaté. FEVG estimée visuellement : 60 %.", "appended"))
        self.assertEqual(rc.lv_caption("", 52.5), ("FEVG estimée visuellement : 52,5 %.", "appended"))
        self.assertEqual(rc.lv_caption("FEVG à 55 %.", 57.0), ("FEVG à 55 %.", "stated"))
        self.assertEqual(rc.lv_caption("FEVG à 55 %.", 40.0), ("FEVG à 55 %.", "conflict"))
        self.assertEqual(rc.lv_caption("FEVG entre 50 et 60 %", 63.0), ("FEVG entre 50 et 60 %", "stated"))
        self.assertEqual(rc.lv_caption("FEVG < 40 %", 60.0), ("FEVG < 40 %", "conflict"))
        self.assertEqual(rc.lv_caption("FEVG non mesurée. Volume normal 60 %", 58.0)[1], "appended")
        self.assertEqual(rc.lv_caption("VG non dilaté.", float("nan")), ("VG non dilaté.", "none"))


class TestReportCells(unittest.TestCase):

    def test_a_cell_that_only_says_it_is_empty_is_missing(self):
        for cell in ("nan", "NaN", "None", "NULL", " N/A ", "NA", "<NA>", "#N/A"):
            self.assertEqual(rc.caption_text(cell), "", cell)
        self.assertEqual(rc.caption_text("Nanisme"), "Nanisme")

    def test_a_character_that_would_stop_a_reader_or_the_id_is_gone(self):
        self.assertEqual(rc.caption_text("VG\x00 normal"), "VG  normal")
        self.assertTrue(rc.caption_id(rc.caption_text("VG \ud800 normal")))
        self.assertEqual(rc.text_clauses("a\x0bb\rc_x000D_d"), ["a", "b", "c", "d"])


class TestEfValues(unittest.TestCase):

    def test_a_structured_value_out_of_range_counts_as_missing(self):
        for value in (-1, 0, 100, 0.6):
            self.assertEqual(rc.lv_caption("FEVG 60 %", value)[1], "stated")
            self.assertEqual(rc.lv_caption("VG normal", value), ("VG normal", "none"))

    def test_an_ef_the_expression_cannot_read_gets_no_second_one(self):
        self.assertEqual(rc.lv_caption("FE à 5 mètres 60 %", 30), ("FE à 5 mètres 60 %", "none"))

    def test_the_method_and_a_repeated_percent_sign_are_read(self):
        self.assertEqual([(s.low, s.high) for s in rc.find_ef("FEVG 2D 60 %")], [(60, 60)])
        self.assertEqual([(s.low, s.high) for s in rc.find_ef("FEVG 40 % à 60 %")], [(40, 60)])


class TestClausesAndEfRule(unittest.TestCase):

    def test_split_keeps_decimals(self):
        self.assertEqual(rc.split_clauses("VG non dilaté. FEVG 52.5 %; trace // fin"),
                         ["vg non dilaté", "fevg 52.5 %", "trace", "fin"])
        self.assertEqual(rc.text_clauses("VG non Dilaté. FEVG 52.5 %"), ["VG non Dilaté", "FEVG 52.5 %"])
        self.assertEqual(rc.split_clauses(float("nan")), [])

    def test_ef_rule_needs_a_certain_whole_range(self):
        cases = {
            "FEVG à 58 %": "normal",
            "FEVG à 45 %": "abnormal",
            "FEVG < 55 %": "unknown",           # allows 54.9 and 30
            "FEVG < 54 %": "abnormal",
            "FEVG entre 50 et 60 %": "unknown",
            "FEVG entre 55 et 60 %": "normal",
            "FEVG > 55 %": "normal",
            "FEVG environ 60 %": "unknown",     # hedged
            "Probablement FEVG à 60 %": "unknown",
            "FEVG à 60 % environ": "unknown",
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(rc.ef_rule_flag(rc.find_ef(text), 54), expected)
        self.assertEqual(rc.ef_rule_flag(rc.find_ef("FEVG à 58 %"), None), "unknown")   # no threshold, no normal

    def test_ef_value_without_wording(self):
        self.assertEqual(rc.ef_value("55", "60"), rc.EfStatement(55, 60))
        self.assertEqual(rc.ef_value("52,5"), rc.EfStatement(52.5, 52.5))
        self.assertIsNone(rc.ef_value("150"))

    def test_ef_expressions_leave_the_rest_of_the_clause_alone(self):
        (match, statement), = rc.ef_expressions("probablement FEVG 60 %")
        self.assertTrue(statement.certain)            # its own wording; `radar_normality` judges the hedge around it
        self.assertEqual(match.group("first"), "60")
        self.assertEqual(list(rc.ef_expressions("FE (strain) 18 %")), [])
        self.assertEqual(list(rc.ef_expressions("FEVG 150 %")), [])

    def test_a_negative_percentage_is_no_ef(self):
        self.assertEqual(list(rc.ef_expressions("FEVG : -20 %")), [])
        self.assertEqual(rc.lv_caption("FEVG : -20 %", 55), ("FEVG : -20 %", "none"))
        self.assertEqual([st.low for _, st in rc.ef_expressions("FEVG 60-65 %")], [60.0])
        self.assertEqual([st.low for _, st in rc.ef_expressions("FEVG - 60 %")], [60.0])

    def test_a_decimal_ef_the_expression_cannot_read_gets_no_second_ef(self):
        for text in ("FEVG 52.5 ± 5 %", "FEVG 52,5 ± 5 %"):
            with self.subTest(text):
                self.assertEqual(rc.lv_caption(text, 55), (text, "none"))


if __name__ == "__main__":
    unittest.main()
