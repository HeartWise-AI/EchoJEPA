# tests/data/test_compare_column_prefill.py

"""`data/compare_column_prefill.py` on synthetic text: the parser comparison (side, named finding against a generic
state, whole-structure statements resolved first, claims that do not speak for a column, aorta part), the
structured checks against the calibrated limits (EF grades with grey zones, the one-grade tolerance and
`--strict-normal`, sizes, LA index before diameter, walls one by one, code findings against the text, the RV code), the
limits file itself, the wall segment summary and count
suppression. With a column review checkout in `COLUMN_REVIEW_ROOT`, also the whole comparison on synthetic reports,
through the column review's own code."""

import json
import os
import sys
import tempfile
import unittest

import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import build_radar_captions as builder  # noqa: E402
import compare_column_prefill as cmp  # noqa: E402
import radar_normality as rn  # noqa: E402

from src.models.radar.routing import load_routing  # noqa: E402

POLICY = rn.load_policy()
ROUTING = load_routing()
COLUMN_REVIEW_ROOT = os.environ.get("COLUMN_REVIEW_ROOT", "")
CHECKOUT = bool(COLUMN_REVIEW_ROOT) and os.path.isfile(
    os.path.join(COLUMN_REVIEW_ROOT, "tools", "labelbox", "column_review.py"))
LVIDD, IVS, LVIW = "LVID Diastole (2D)", "IVS Diastole Thickness (2D)", "LVIW Diastolic Thickness (2D)"
LAVI, LAD = "LA Volume Index (BP MOD) DuBois", "LA Dimension"
LIMITS = cmp.load_limits()


def reading(structure, text, covered=()):
    """The parser's reading of a synthetic report cell, through the caption builder."""
    entry = next(s for s in ROUTING.trained if s.name == structure)
    row = builder.caption_row("S1", entry, {entry.report_column: text}, POLICY)
    return cmp.parser_reading(json.loads(row["facts"]), covered)


class TestOutcome(unittest.TestCase):

    def test_sides(self):
        self.assertEqual(cmp.outcome("dilated", "abnormal"), "agree")
        self.assertEqual(cmp.outcome("not_demonstrated", "abnormal"), "disagree")
        self.assertEqual(cmp.outcome("bioprosthesis", "device"), "agree")
        self.assertEqual(cmp.outcome("normal", "device"), "disagree")

    def test_findings(self):
        self.assertEqual(cmp.outcome("calcified", "abnormal", {"calcified", "thickened"}), "agree")
        self.assertEqual(cmp.outcome("calcified", "abnormal", {"sclerotic"}), "finding_mismatch")
        self.assertEqual(cmp.outcome("tethered", "abnormal", {"restricted"}), "agree")       # side only
        self.assertEqual(cmp.outcome("mildly_reduced", "abnormal", {"dysfunction"}), "agree")  # graded: side only
        self.assertEqual(cmp.outcome("calcified", "abnormal", ()), "agree")    # an abnormal state, no finding

    def test_one_side_silent_or_unsure(self):
        self.assertEqual(cmp.outcome("normal", "not_stated"), "prefill_only")
        self.assertEqual(cmp.outcome("normal", "removed"), "prefill_only")
        self.assertEqual(cmp.outcome("present", "no_attribute"), "prefill_only")
        self.assertEqual(cmp.outcome("calcified", "uncertain"), "parser_unsure")
        self.assertEqual(cmp.outcome(None, "normal"), "parser_only")
        self.assertIsNone(cmp.outcome(None, "uncertain"))

    def test_other_values(self):
        self.assertEqual(cmp.outcome("hyperdynamic", "abnormal"), "prefill_other")

    def test_an_unknown_value_is_an_error(self):
        with self.assertRaises(ValueError):
            cmp.outcome("enlarged", "abnormal")


class TestParserReading(unittest.TestCase):

    def test_attributes_and_findings(self):
        statuses, own, findings = reading("aortic", "Valve aortique calcifiée et épaissie.")
        self.assertEqual(statuses["aortic_appearance"], "abnormal")
        self.assertEqual(findings["aortic_appearance"], {"calcified", "thickened"})
        self.assertEqual(own, statuses)

    def test_a_denied_finding_beside_an_effusion(self):
        statuses, _, _ = reading("pericardium", "Épanchement péricardique léger sans signe de tamponnade.")
        self.assertEqual(statuses["pericardial_effusion"], "abnormal")

    def test_a_whole_structure_normal(self):
        statuses, own, _ = reading("lv", "Ventricule gauche normal.",
                                   covered={"lv_size", "lv_function", "lv_wall_motion"})
        self.assertEqual(statuses["lv_size"], "normal")
        self.assertNotIn("lv_wall_motion", statuses)     # a normal LV says nothing segment by segment
        self.assertNotIn("lv_size", own)

    def test_a_generic_state_is_no_finding(self):
        statuses, _, findings = reading("lv", "VG nettement dilaté. Dimensions augmentées.")
        self.assertEqual(statuses["lv_size"], "abnormal")
        self.assertNotIn("lv_size", findings)       # `augmentées` gives the side, it names no other finding
        self.assertEqual(cmp.outcome("dilated", statuses["lv_size"], findings.get("lv_size", ())), "agree")
        _, _, findings = reading("lv", "Anomalie de la fonction systolique.")
        self.assertEqual(findings, {})

    def test_a_named_finding_still_mismatches(self):
        statuses, _, findings = reading("aortic", "Valve aortique scléreuse.")
        self.assertEqual(cmp.outcome("calcified", statuses["aortic_appearance"], findings["aortic_appearance"]),
                         "finding_mismatch")

    def test_conflicting_whole_structure_statements(self):
        covered = {"rv_size", "rv_function"}
        statuses, own, _ = reading("rv", "Ventricule droit normal. Ventricule droit anormal.", covered)
        self.assertEqual(statuses, {"rv_size": "conflict", "rv_function": "conflict"})
        self.assertEqual(own, {})
        self.assertEqual(cmp.outcome("moderately_reduced", statuses["rv_function"]), "parser_unsure")
        statuses, own, _ = reading("rv", "Ventricule droit normal. Ventricule droit anormal. "
                                         "Fonction systolique normale.", covered)
        self.assertEqual(statuses, {"rv_size": "conflict", "rv_function": "normal"})   # its own statement stands
        self.assertEqual(own, {"rv_function": "normal"})

    def test_a_whole_structure_abnormal_says_nothing_of_one_attribute(self):
        statuses, _, _ = reading("rv", "Ventricule droit anormal.", {"rv_size", "rv_function"})
        self.assertEqual(statuses, {})

    def test_only_concentric_hypertrophy_speaks_for_the_walls(self):
        for text in ("Pas d'hypertrophie ventriculaire gauche.",
                     "Augmentation sévère de la masse du VG (HVG excentrique)."):
            with self.subTest(text):
                statuses, _, _ = reading("lv", text)
                self.assertNotIn("lv_wall_thickness", statuses)
        statuses, _, _ = reading("lv", "Hypertrophie ventriculaire gauche concentrique légère.")
        self.assertEqual(statuses["lv_wall_thickness"], "abnormal")
        statuses, _, _ = reading("lv", "Parois d'épaisseur normale.")
        self.assertEqual(statuses["lv_wall_thickness"], "normal")

    def test_a_denied_valve_finding_is_no_normal_valve(self):
        statuses, _, _ = reading("mitral", "Pas de prolapsus de la valve mitrale.")
        self.assertNotIn("mitral_motion", statuses)
        statuses, _, findings = reading("mitral", "Prolapsus de la valve mitrale.")
        self.assertEqual((statuses["mitral_motion"], findings["mitral_motion"]), ("abnormal", {"prolapse"}))

    def test_no_cusp_count_or_device_from_a_normal_valve(self):
        statuses, _, _ = reading("aortic", "Valve aortique normale.", covered={"aortic_cusps", "aortic_device"})
        self.assertNotIn("aortic_cusps", statuses)
        self.assertNotIn("aortic_device", statuses)

    def test_a_device(self):
        statuses, _, _ = reading("aortic", "Bioprothèse aortique en bonne position.")
        self.assertEqual(statuses["aortic_device"], "device")

    def test_aorta_parts(self):
        statuses, _, _ = reading("aorta", "Racine aortique dilatée. Aorte ascendante normale.")
        self.assertEqual(cmp.parser_status(statuses, "aorta_size:root", "not_stated"), "abnormal")
        self.assertEqual(cmp.parser_status(statuses, "aorta_size:ascending", "not_stated"), "normal")
        self.assertEqual(cmp.parser_status(statuses, "aorta_size:arch", "not_stated"), "not_stated")

    def test_a_whole_aorta_stands_in_for_a_part(self):
        statuses, _, _ = reading("aorta", "Aorte non dilatée.", covered={"aorta_size"})
        self.assertEqual(cmp.parser_status(statuses, "aorta_size:arch", "not_stated"), "normal")

    def test_structured_and_appended_facts_are_left_out(self):
        facts = [{"attribute": "lv_size", "value": "abnormal", "source": "structured", "finding": ""},
                 {"attribute": "lv_function", "value": "normal", "source": "appended", "finding": ""}]
        self.assertEqual(cmp.parser_reading(facts), ({}, {}, {}))


class TestLimits(unittest.TestCase):

    def test_the_file_loads_and_is_ordered(self):
        self.assertEqual(LIMITS["version"], 0)
        for measure in LIMITS["measures"].values():
            for source in measure["sources"]:
                self.assertLess(source["abnormal_up_to"], source["normal_from"])

    def test_a_bad_file_is_refused(self):
        bad = {**LIMITS, "walls": {"septum": {**LIMITS["walls"]["septum"], "normal_from": 0.5},
                                   "posterior": LIMITS["walls"]["posterior"]}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(bad, f)
        try:
            with self.assertRaises(SystemExit):
                cmp.load_limits(f.name)
        finally:
            os.unlink(f.name)


class TestEf(unittest.TestCase):

    def test_grades_an_ef_may_carry(self):
        efs = (60, 55, 54, 53, 52, 51, 45, 44, 41, 40, 35, 34, 30, 29)
        self.assertEqual([cmp.ef_grades(ef, LIMITS) for ef in efs],
                         [{0}, {0}, {0, 1}, {0, 1}, {0, 1}, {1}, {1}, {1, 2}, {1, 2}, {2}, {2}, {2, 3}, {2, 3}, {3}])

    def test_check(self):
        self.assertEqual(cmp.check_ef("normal", 60, LIMITS), "consistent")
        self.assertEqual(cmp.check_ef("normal", 53, LIMITS), "consistent")          # grey zone
        self.assertEqual(cmp.check_ef("mildly_reduced", 53, LIMITS), "consistent")
        self.assertEqual(cmp.check_ef("hyperdynamic", 70, LIMITS), "consistent")
        self.assertEqual(cmp.check_ef("moderately_reduced", 47, LIMITS), "grade_off_by_one")
        self.assertEqual(cmp.check_ef("mildly_reduced", 25, LIMITS), "contradiction")
        self.assertEqual(cmp.check_ef("normal", 38, LIMITS), "contradiction")
        self.assertEqual(cmp.check_ef("normal", None, LIMITS), "no_measurement")

    def test_across_normal(self):
        self.assertEqual(cmp.check_ef("normal", 50, LIMITS), "off_by_one_across_normal")
        self.assertEqual(cmp.check_ef("mildly_reduced", 60, LIMITS), "off_by_one_across_normal")
        self.assertEqual(cmp.check_ef("normal", 50, LIMITS, strict_normal=True), "contradiction")
        self.assertEqual(cmp.check_ef("hyperdynamic", 50, LIMITS, strict_normal=True), "contradiction")
        self.assertEqual(cmp.check_ef("normal", 53, LIMITS, strict_normal=True), "consistent")

    def test_ef_value(self):
        column = LIMITS["ef"]["column"]
        self.assertEqual(cmp.ef_value({column: 55.0}, LIMITS), 55.0)
        self.assertIsNone(cmp.ef_value({column: 150.0}, LIMITS))
        self.assertIsNone(cmp.ef_value({}, LIMITS))


class TestMeasures(unittest.TestCase):

    def test_sizes(self):
        self.assertEqual(cmp.check_measure("lv_size", "normal", {LVIDD: 4.8}, LIMITS), "consistent")
        self.assertEqual(cmp.check_measure("lv_size", "normal", {LVIDD: 5.9}, LIMITS), "grey")    # lab: not yet
        self.assertEqual(cmp.check_measure("lv_size", "normal", {LVIDD: 6.2}, LIMITS), "contradiction")
        self.assertEqual(cmp.check_measure("lv_size", "dilated", {LVIDD: 5.2}, LIMITS), "contradiction")
        self.assertEqual(cmp.check_measure("lv_size", "dilated", {LVIDD: 5.5}, LIMITS), "grey")
        self.assertEqual(cmp.check_measure("lv_size", "dilated", {LVIDD: 6.0}, LIMITS), "consistent")
        self.assertEqual(cmp.check_measure("lv_size", "normal", {}, LIMITS), "no_measurement")
        self.assertEqual(cmp.check_measure("lv_size", "normal", {LVIDD: 40.0}, LIMITS), "no_measurement")

    def test_a_small_rv_diameter_does_not_contradict_a_dilated_rv(self):
        rvb = "RV Basal Diastolic Dimension"
        self.assertEqual(cmp.check_measure("rv_size", "dilated", {rvb: 3.6}, LIMITS), "consistent")
        self.assertEqual(cmp.check_measure("rv_size", "normal", {rvb: 5.2}, LIMITS), "contradiction")
        self.assertEqual(cmp.check_measure("rv_size", "normal", {rvb: 4.5}, LIMITS), "grey")

    def test_la_index_first_then_diameter(self):
        self.assertEqual(cmp.check_measure("la_size", "normal", {LAVI: 0.040, LAD: 3.0}, LIMITS), "contradiction")
        self.assertEqual(cmp.check_measure("la_size", "normal", {LAD: 4.5}, LIMITS), "grey")
        self.assertEqual(cmp.check_measure("la_size", "normal", {LAD: 5.1}, LIMITS), "contradiction")

    def test_walls_one_by_one(self):
        # A regional value speaks of its own wall only (form v5): a thick posterior wall does not contradict it.
        self.assertEqual(cmp.check_walls("septal_thickening", {IVS: 1.3, LVIW: 1.2}, LIMITS), "consistent")
        self.assertEqual(cmp.check_walls("septal_thickening", {IVS: 1.0, LVIW: 0.8}, LIMITS), "grey")    # lab: not yet
        self.assertEqual(cmp.check_walls("septal_thickening", {IVS: 0.85}, LIMITS), "contradiction")
        self.assertEqual(cmp.check_walls("posterior_thickening", {IVS: 0.8}, LIMITS), "no_measurement")
        self.assertEqual(cmp.check_walls("increased", {IVS: 0.8, LVIW: 1.3}, LIMITS), "contradiction")
        self.assertEqual(cmp.check_walls("not_increased", {IVS: 0.8, LVIW: 1.05}, LIMITS), "grey")
        self.assertEqual(cmp.check_walls("not_increased", {}, LIMITS), "no_measurement")


class TestStructuredCheck(unittest.TestCase):

    def test_text_findings(self):
        ef = LIMITS["ef"]["column"]
        self.assertEqual(cmp.structured_check("lv_size", "normal", "text", {LVIDD: 6.2}, {}, LIMITS), "contradiction")
        self.assertEqual(cmp.structured_check("lv_function", "normal", "text", {ef: 60.0}, {}, LIMITS), "consistent")
        self.assertEqual(cmp.structured_check("mitral_appearance", "calcified", "text", {}, {}, LIMITS),
                         "not_checkable")
        self.assertEqual(cmp.structured_check("ivc_size", "dilated", "text", {}, {}, LIMITS),
                         "not_checkable")      # IVC diameter is not checked

    def test_code_findings_against_the_text(self):
        own = {"lv_wall_motion": "abnormal"}
        self.assertEqual(cmp.structured_check(cmp.WALL_COLUMN, "normal", "code", {}, own, LIMITS), "contradiction")
        self.assertEqual(cmp.structured_check(cmp.WALL_COLUMN, "abnormal_segment", "code", {}, own, LIMITS),
                         "consistent")
        self.assertEqual(cmp.structured_check(cmp.WALL_COLUMN, "normal", "code", {}, {}, LIMITS), "no_measurement")
        self.assertEqual(cmp.structured_check(cmp.WALL_COLUMN, "hyperkinetic", "code", {}, own, LIMITS),
                         "not_checkable")
        self.assertEqual(cmp.structured_check("rv_function", "normal", "code", {}, {"rv_function": "abnormal"},
                                              LIMITS), "contradiction")

    def test_rv_code(self):
        self.assertEqual(cmp.rv_code({cmp.RV_CODE: 2.0}), 2)
        self.assertIsNone(cmp.rv_code({cmp.RV_CODE: -1.0}))
        self.assertIsNone(cmp.rv_code({}))
        self.assertEqual(cmp.structured_check("rv_function", "normal", "text", {cmp.RV_CODE: 0.0}, {}, LIMITS),
                         "consistent")
        self.assertEqual(cmp.structured_check("rv_function", "normal", "text", {}, {}, LIMITS), "no_measurement")

    def test_value_source(self):
        self.assertEqual(cmp.value_source("rv_function", cmp.RV_CODE), "code")
        self.assertEqual(cmp.value_source("rv_function", "Right ventricule findings"), "text")
        self.assertEqual(cmp.value_source(cmp.WALL_COLUMN, None), "code")


class TestSmallParts(unittest.TestCase):

    def test_parser_attribute(self):
        self.assertEqual(cmp.parser_attribute(cmp.WALL_COLUMN), "lv_wall_motion")
        self.assertEqual(cmp.parser_attribute("aortic_cusp_count"), "aortic_cusps")
        self.assertEqual(cmp.parser_attribute("aorta_root_size"), "aorta_size:root")
        self.assertEqual(cmp.parser_attribute("aorta_ascending_size"), "aorta_size:ascending")
        self.assertIsNone(cmp.parser_attribute("septal_bulge"))
        self.assertIsNone(cmp.parser_attribute("mitral_regurgitant_jet"))

    def test_wall_summary(self):
        self.assertEqual(cmp.wall_summary({"normal", "hypokinetic"}), "abnormal_segment")
        self.assertEqual(cmp.wall_summary({"normal", "hyperkinetic"}), "hyperkinetic")
        self.assertEqual(cmp.wall_summary({"normal", "normal_or_hyperkinetic"}), "normal")

    def test_suppression(self):
        self.assertEqual([cmp.suppressed(n) for n in (0, 1, 4, 5, 12)], [0, "<5", "<5", 5, 12])
        frame = cmp.suppress_frame(pd.DataFrame({"name": ["a", "b"], "count": [3, 7]}))
        self.assertEqual(frame["count"].tolist(), ["<5", 7])


@unittest.skipUnless(CHECKOUT, "COLUMN_REVIEW_ROOT does not name a column review checkout")
class TestWithColumnReview(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cr, cls.spec = cmp.load_column_review(COLUMN_REVIEW_ROOT)
        cmp.index_columns(cls.spec)

    def test_every_value_has_a_polarity_and_v5(self):
        values = {v for field in self.spec["fields"].values() for v in field["values"]}
        self.assertLessEqual(values, set(cmp.POLARITY))
        self.assertEqual(self.spec["version"], cmp.SPEC_VERSION)

    def test_the_checkout_leaves_the_path(self):
        self.assertNotIn(os.path.abspath(COLUMN_REVIEW_ROOT), sys.path)

    def test_subcostal_level_follows_the_structure(self):
        self.assertEqual(cmp.level_for("SUBCOSTAL", "ivc", self.spec, self.cr), "ivc")
        self.assertEqual(cmp.level_for("SUBCOSTAL", "lv", self.spec, self.cr), "four_chamber")
        self.assertEqual(cmp.level_for("A4C", "lv", self.spec, self.cr), "not_applicable")

    def test_report_columns_cover_both_sides(self):
        columns = cmp.report_columns(ROUTING, POLICY, self.spec, LIMITS)
        for column in ("Pericardium findings", cmp.RV_CODE, "WMS 2C Basal Inferior", POLICY.ef["column"]):
            self.assertIn(column, columns)

    def test_whole_comparison_on_synthetic_reports(self):
        ef = POLICY.ef["column"]
        reports = pd.DataFrame([
            {"study_id": "S1", "Pericardium findings": "Épanchement péricardique léger sans signe de tamponnade.",
             "Left ventricule findings": "Ventricule gauche non dilaté. Fonction systolique normale.",
             LVIDD: 6.2, ef: 50.0, "WMS 4C Apical septum": 2.0},
            {"study_id": "S2", "Pericardium findings": "Pas d'épanchement péricardique.",
             "Left ventricule findings": "Ventricule gauche non dilaté.", LVIDD: 4.8, ef: 35.0},
        ])
        clips = pd.DataFrame({"study_id": ["S1", "S1", "S2"], "view": ["A4C", "PLAX", "A4C"]})
        records, flagged, fill_in, ef_fill, coverage = cmp.compare(reports, clips, ROUTING, POLICY, self.spec,
                                                                   self.cr, LIMITS)
        got = {(r[0], r[1], r[2], r[4], r[6], r[7]) for r in records if r[0] in ("lv", "pericardium")}
        self.assertIn(("lv", "lv_size", "text", "normal", "agree", "contradiction"), got)       # S1: 6.2 cm
        self.assertIn(("lv", "lv_size", "text", "normal", "agree", "consistent"), got)          # S2: 4.8 cm
        self.assertIn(("lv", "lv_function", "text", "normal", "agree", "off_by_one_across_normal"), got)
        self.assertIn(("lv", cmp.WALL_COLUMN, "code", "abnormal_segment", "prefill_only", "no_measurement"), got)
        # The prefill reads `sans signe de tamponnade` as about the tamponade, not the effusion (since 4aa3a71).
        self.assertIn(("pericardium", "pericardial_effusion", "text", "present", "agree", "not_checkable"), got)
        self.assertIn(("pericardium", "pericardial_effusion", "text", "not_demonstrated", "agree", "not_checkable"),
                      got)
        details = sorted((f["study_id"], f["column"], f["detail"]) for f in flagged)
        self.assertEqual(details, [("S1", "lv_function", "off_by_one_across_normal"),
                                   ("S1", "lv_size", "contradiction")])
        self.assertEqual(ef_fill[("moderately_reduced", "filled")], 1)       # S2: no function in the text
        frames, flags = cmp.tables(records, flagged, fill_in, ef_fill, coverage)
        survival = frames["survival"].set_index(["structure", "column"])
        self.assertEqual(int(survival.loc[("lv", "lv_size"), "prefill"]), 2)
        self.assertEqual(int(survival.loc[("lv", "lv_size"), "kept_structured"]), 1)
        self.assertEqual(int(survival.loc[("pericardium", "pericardial_effusion"), "kept_both"]), 2)
        strict = cmp.compare(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS, strict_normal=True)[1]
        self.assertIn(("S1", "lv_function", "contradiction"), [(f["study_id"], f["column"], f["detail"])
                                                               for f in strict])

    def test_generic_states_and_whole_conflicts_through_the_comparison(self):
        reports = pd.DataFrame([
            {"study_id": "S1", "Left ventricule findings": "VG nettement dilaté. Dimensions augmentées.",
             LVIDD: 6.5},
            {"study_id": "S2", "Right ventricule findings": "Ventricule droit normal. Ventricule droit anormal.",
             cmp.RV_CODE: 2},
        ])
        clips = pd.DataFrame({"study_id": ["S1", "S2"], "view": ["A4C", "A4C"]})
        records, flagged, *_ = cmp.compare(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS)
        got = {(r[0], r[1], r[4], r[5], r[6], r[7]) for r in records}
        self.assertIn(("lv", "lv_size", "dilated", "abnormal", "agree", "consistent"), got)
        self.assertIn(("rv", "rv_function", "moderately_reduced", "conflict", "parser_unsure", "no_measurement"), got)
        self.assertNotIn(("S2", "rv_function"), [(f["study_id"], f["column"]) for f in flagged])
        self.assertNotIn(("S1", "lv_size"), [(f["study_id"], f["column"]) for f in flagged])


if __name__ == "__main__":
    unittest.main()
