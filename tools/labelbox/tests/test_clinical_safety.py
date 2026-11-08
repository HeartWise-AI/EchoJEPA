"""Clinical counterexamples and export-path regressions using synthetic reports only."""
import copy
import tempfile
import unittest
from pathlib import Path

from app.radar.ontology import load_ontology
from tools.labelbox import column_review as cr
from tools.labelbox.pilot_upload import summarize_export
from tools.labelbox.tests.test_pilot_upload import (
    A4C, COLUMN_REVIEW, column_review_label, exported, make_folder, read_plan, radio,
)


def candidate(report, view="PLAX", modality="bmode"):
    return cr.build_candidate(report, view, modality, global_key="synthetic-safety",
                              input_provenance_verified=True)


class TestClinicalAssertions(unittest.TestCase):
    def test_posterior_measurements_never_establish_normal_elsewhere(self):
        for measurement in (None, 0, 0.95, 1.6, 100, float("nan"), float("inf")):
            with self.subTest(measurement=measurement):
                c = candidate({"Ventricular septum findings": "Augmentation légère de l'épaisseur du septum",
                               "LVIW Diastolic Thickness (2D)": measurement})
                self.assertEqual([(x["concept"], x["value"]) for x in c["columns"]],
                                 [("lv_wall_thickness", "septal_thickening")])
                self.assertEqual(c["candidate_caption"], "Septal thickening is present.")

    def test_diffuse_septal_thickening_stays_regional(self):
        report = {"Ventricular septum findings": "Augmentation sévère et diffuse de l'épaisseur du septum"}
        self.assertEqual(candidate(report)["columns"][0]["value"], "septal_thickening")
        self.assertEqual(candidate(report, "A2C")["columns"], [])
        generalized = {"Left ventricule findings": "Augmentation diffuse de l'épaisseur des parois"}
        self.assertEqual(candidate(generalized)["columns"][0]["value"], "increased")
        for text in ("Augmentation diffuse de l'épaisseur des parois septales",
                     "Augmentation diffuse de l'épaisseur pariétale septale"):
            self.assertEqual(candidate({"Ventricular septum findings": text}, "A2C")["columns"], [])

    def test_negation_uncertainty_and_adjacent_severity(self):
        cases = [
            ("Mitral valve findings", "Aucune évidence d'insuffisance mitrale", "mitral_regurgitant_jet", "not_demonstrated", "A4C", "color_flow"),
            ("Mitral valve findings", "Insuffisance mitrale modérée sans prolapsus", "mitral_regurgitant_jet", "demonstrated", "A4C", "color_flow"),
            ("Ventricular septum findings", "Pas de signe de bourrelet septal", "septal_bulge", None, "PLAX", "bmode"),
            ("Ventricular septum findings", "Un bourrelet septal ne peut être exclu", "septal_bulge", None, "PLAX", "bmode"),
            ("Ventricular septum findings", "Basal septal bulge cannot be excluded", "septal_bulge", None, "PLAX", "bmode"),
            ("Ventricular septum findings", "Basal septal bulge", "septal_bulge", "present", "PLAX", "bmode"),
            ("Pericardium findings", "Un épanchement péricardique ne peut être exclu", "pericardial_effusion", None, "PLAX", "bmode"),
            ("Pericardium findings", "Épanchement péricardique minime sans tamponnade", "pericardial_effusion", "present", "PLAX", "bmode"),
            ("Aortic valve findings", "Pas de valve bicuspide", "aortic_cusp_count", None, "PSAX_AV", "bmode"),
            ("Mitral valve findings", "Absence de prothèse mécanique", "mitral_device", None, "PLAX", "bmode"),
            ("Left ventricule findings", "Dysfonction systolique légère, dilatation sévère", "lv_function", "mildly_reduced", "A4C", "bmode"),
            ("Left ventricule findings", "Dysfonction diastolique légère", "lv_function", None, "A4C", "bmode"),
            ("Aortic valve findings", "Valve is not normal", "aortic_appearance", None, "PLAX", "bmode"),
            ("Aortic valve findings", "Valve aortique non normale", "aortic_appearance", None, "PLAX", "bmode"),
            ("Aortic valve findings", "Leaflet motion is not normal", "aortic_motion", None, "PLAX", "bmode"),
            ("Aortic valve findings", "Opening is not normal", "aortic_motion", None, "PLAX", "bmode"),
        ]
        for field, text, concept, expected, view, mode in cases:
            with self.subTest(text=text):
                values = {x["concept"]: x["value"] for x in candidate({field: text}, view, mode)["columns"]}
                self.assertEqual(values.get(concept), expected)

    def test_wall_form_maps_to_distribution_never_lvh_grade(self):
        field = cr.load_spec()["fields"]["lv_wall_thickness"]
        concept = load_ontology().concept(field["canonical_concept"])
        self.assertEqual(concept.id, "lv_wall_thickening_distribution")
        self.assertEqual(set(field["canonical_values"]), set(field["values"]))
        self.assertEqual(set(field["canonical_values"].values()), set(concept.values))

    def test_negated_normality_does_not_become_normal_or_abnormal(self):
        for text, key in (("No normal wall thickness", "lv_wall_thickness"),
                          ("Pas d'épaisseur pariétale normale", "lv_wall_thickness"),
                          ("No normal size", "lv_size"), ("Pas de dimension normale", "lv_size")):
            with self.subTest(text=text):
                values = {x["concept"]: x["value"] for x in candidate({"Left ventricule findings": text})["columns"]}
                self.assertNotIn(key, values)

    def test_wall_distribution_relevance_covers_form_routes(self):
        spec, ontology = cr.load_spec(), load_ontology()
        concept = spec["fields"]["lv_wall_thickness"]["canonical_concept"]
        for view, route in spec["views"].items():
            for level in cr.level_choices(view, spec):
                for mode in ("bmode", "color_flow"):
                    if "lv_wall_thickness" in cr.allowed_columns(view, mode, level=level, spec=spec):
                        with self.subTest(view=view, level=level, mode=mode):
                            self.assertEqual(ontology.relevance(concept, route.get("refines", view), mode), "relevant")
        # A broad concept prior never makes a septal claim valid in a two-chamber view.
        self.assertFalse(cr.value_allowed("lv_wall_thickness", "septal_thickening", "A2C", spec))


class TestReviewSafety(unittest.TestCase):
    def test_septal_claims_require_actual_septal_visibility(self):
        report = {"Ventricular septum findings": "Augmentation de l'épaisseur du septum"}
        without = cr.build_candidate(report, "PLAX", "bmode", global_key="synthetic", targets=["lv"])
        self.assertEqual(without["columns"], [])
        c = candidate(report)
        a = {**cr.decode_annotations(cr.prelabels(c)), "review_completed": "yes", "visible_targets": ["lv"]}
        score = cr.score_review(c, a, submitted=True)
        self.assertIn("lv_wall_thickness__structure_unticked", score["pending"])
        self.assertIn("Interventricular septum", score["warnings"][0])
        self.assertFalse(score["ready_for_validation"])
        for changes in ({"lv_wall_thickness__status": "no", "lv_wall_thickness__error": "wrong_value",
                         "lv_wall_thickness__correction": "septal_thickening"},
                        {"missing_information": "yes", "missing_columns": ["lv_wall_thickness"],
                         "lv_wall_thickness__missing_value": "septal_thickening", "lv_wall_thickness__missing_visible": "yes"}):
            if "missing_information" in changes:
                empty = candidate({})
                answers = {**cr.decode_annotations(cr.prelabels(empty)), "review_completed": "yes", "visible_targets": ["lv"], **changes}
                self.assertFalse(cr.score_review(empty, answers, submitted=True)["ready_for_validation"])
            else:
                self.assertFalse(cr.score_review(c, {**a, **changes}, submitted=True)["ready_for_validation"])

    def test_psax_mid_has_no_valve_claims_and_refinement_does_not_imply_visibility(self):
        c = candidate({}, "PSAX")
        a = {**cr.decode_annotations(cr.prelabels(c)), "view": "PSAX_BASAL", "review_completed": "yes",
             "missing_information": "yes", "missing_columns": ["mitral_appearance"],
             "mitral_appearance__missing_value": "normal", "mitral_appearance__missing_visible": "yes"}
        self.assertIn("mitral_appearance__not_in_view", cr.score_review(c, a, submitted=True)["pending"])
        a["visible_targets"].append("mitral")
        self.assertTrue(cr.score_review(c, a, submitted=True)["ready_for_validation"])
        a["view"] = "PSAX_MID"
        score = cr.score_review(c, a, submitted=True)
        self.assertIn("mitral_appearance__not_in_view", score["pending"])
        self.assertFalse(score["ready_for_validation"])

    def test_uninterpretable_psax_needs_no_view_or_clinical_answers(self):
        c = candidate({}, "PSAX")
        a = {"uninterpretable": "yes", "uninterpretable_reason": "poor_quality", "review_completed": "yes"}
        records = cr.encode_answers(a, cr.build_form())
        self.assertEqual(cr.review_problems(c, records), [])
        score = cr.score_review(c, a, submitted=True)
        self.assertEqual(score["caption"], "")
        self.assertIsNone(score["view_correct"])
        self.assertFalse(score["ready_for_validation"])
        # A view placed at top level bypasses conditional nesting and must be rejected.
        records.append({"name": "view", "answer": {"name": "PSAX"}})
        self.assertTrue(cr.review_problems(c, records))

    def test_old_candidates_are_rejected(self):
        c = candidate({})
        c["version"] = 4
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            cr.prelabels(c)

    def test_export_rejects_incomplete_duplicate_misplaced_and_invalid_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
            original = column_review_label(plan.candidates[A4C])
            variants = []
            variants.append([x for x in original if x["value"] != "review_completed"])
            duplicate = copy.deepcopy(original)
            duplicate.append(copy.deepcopy(duplicate[-1]))
            variants.append(duplicate)
            misplaced = copy.deepcopy(original)
            children = misplaced[0]["radio_answer"]["classifications"]
            misplaced.append(children.pop(0))
            variants.append(misplaced)
            invalid = copy.deepcopy(original)
            invalid[-1] = radio("review_completed", "no")
            variants.append(invalid)
            for records in variants:
                with self.subTest(records=records):
                    item = summarize_export({COLUMN_REVIEW: [exported(A4C, records)]}, plan)["items"][0]
                    self.assertEqual(item["status"], "invalid")
                    self.assertTrue(item["problems"])
                    self.assertFalse(item.get("score", {}).get("ready_for_validation", False))
            valid = summarize_export({COLUMN_REVIEW: [exported(A4C, original)]}, plan)["items"][0]
            self.assertEqual(valid["status"], "valid")
            self.assertTrue(valid["score"]["submitted"])
