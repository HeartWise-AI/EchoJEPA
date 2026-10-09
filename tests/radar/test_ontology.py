"""The versioned concept ontology: shipped file validity, the validator's refusals and the relevance prior."""

import copy
import os
import unittest

import yaml

from app.radar.ontology import (
    ACQUISITION_UNKNOWN,
    DEFAULT_ONTOLOGY_PATH,
    MEASUREMENT_REQUIRED,
    MODALITY_ABSENT,
    NOT_IN_VIEW,
    RELEVANT,
    VIEWS,
    OntologyError,
    load_ontology,
    normalize_text,
    validate_ontology,
)


def raw():
    with open(DEFAULT_ONTOLOGY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestShippedOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = load_ontology()

    def test_version_size_and_kinds(self):
        self.assertEqual(self.o.tag, "radar-caption-ontology-v1")
        self.assertEqual(os.path.basename(DEFAULT_ONTOLOGY_PATH), f"{self.o.tag}.yaml")
        self.assertTrue(40 <= len(self.o.concepts) <= 50, len(self.o.concepts))
        kinds = {c.kind for c in self.o.concepts.values()}
        self.assertEqual(kinds, {"structure_visibility", "finding"})
        self.assertEqual(len(self.o.sha256), 64)

    def test_visibility_is_separate_from_disease(self):
        self.assertEqual(self.o.concept("mitral_valve_visible").kind, "structure_visibility")
        self.assertEqual(self.o.concept("mitral_regurgitation").kind, "finding")
        self.assertIn("severe", self.o.allowed_values("mitral_regurgitation"))
        self.assertEqual(self.o.allowed_values("mitral_valve_visible"), (True, False))

    def test_view_taxonomy(self):
        self.assertEqual(tuple(self.o.views), VIEWS)
        self.assertEqual(self.o.views["A4C_ZOOM"].variant, "zoomed")
        self.assertEqual(self.o.views["RVINF"].window, "parasternal")
        self.assertEqual(self.o.view_group_of("A4C_LV"), ("apical_4c",))
        self.assertEqual(self.o.view_group_of("OTHER"), ())
        self.assertEqual(self.o.view_labels["fr"]["A4C"], "Coupe apicale 4 cavités")
        for c in self.o.concepts.values():
            self.assertNotIn("OTHER", c.prior_views, c.id)  # the unclassified view carries no prior

    def test_statement_concepts(self):
        statements = self.o.statement_concepts
        self.assertIn("mitral_valve_visible", statements)
        self.assertIn("mitral_regurgitation", statements)
        self.assertNotIn("tapse", statements)  # measurements are never stated
        self.assertEqual(len(statements), sum(c.kind == "structure_visibility" or c.evidence == "visual"
                                              for c in self.o.concepts.values()))
        self.assertEqual([q.id for q in self.o.review.questions][:5],
                         ["uninterpretable_clip", "view_correct", "caption_supported", "missing_information",
                          "unsupported_information"])

    def test_structure_taxonomy(self):
        used = {s for c in self.o.concepts.values() for s in c.structures}
        self.assertEqual(used, set(self.o.structures))  # every structure is used, every use is declared
        self.assertEqual(self.o.visibility_concept("mitral_valve"), "mitral_valve_visible")
        self.assertIsNone(self.o.visibility_concept("left_ventricular_outflow_tract"))
        self.assertEqual(self.o.concept("pacing_lead").structures, ("right_atrium", "right_ventricle"))
        self.assertIn("mitral_regurgitation", self.o.concepts_of_structure("mitral_valve"))

    def test_report_columns_cover_the_icm_export_fields(self):
        columns = {spec.get("name") for c in self.o.concepts.values() for spec in c.report_sources.get("columns", [])}
        fields = {f for c in self.o.concepts.values() for f in c.report_sources.get("text_fields", [])}
        for name in ("Visually Estimated EF", "MHI VD fonction systolique", "RV Systolic Pressure", "TR Peak Velocity",
                     "RA Pressure", "Referring MR Grade", "Referring AR Grade", "Referring TR Grade",
                     "Referring PR Grade", "TR ERO (PISA)"):
            self.assertIn(name, columns)
        for name in ("Left ventricule findings", "Left atrial findings", "Aortic valve findings",
                     "Mitral valve findings", "Tricuspid valve findings", "Aorta findings",
                     "Inferior vena cava findings", "Report conclusions findings"):
            self.assertIn(name, fields)

    def test_relevance_outcomes(self):
        r = self.o.relevance
        self.assertEqual(r("mitral_regurgitation", "A4C", "color_flow"), RELEVANT)
        self.assertEqual(r("mitral_regurgitation", "A4C", "bmode"), MODALITY_ABSENT)
        self.assertEqual(r("mitral_regurgitation", "SUBCOSTAL", "color_flow"), NOT_IN_VIEW)
        self.assertEqual(r("rv_systolic_pressure", "A4C", "color_flow"), MODALITY_ABSENT)
        self.assertEqual(r("tricuspid_regurgitation_ero", "A4C", "color_flow"), MEASUREMENT_REQUIRED)
        self.assertEqual(r("lv_size", "A4C", "unknown"), ACQUISITION_UNKNOWN)
        self.assertEqual(r("lv_size", "A4C", "other"), ACQUISITION_UNKNOWN)
        self.assertEqual(r("lv_size", "OTHER", "bmode"), NOT_IN_VIEW)
        with self.assertRaises(ValueError):
            r("lv_size", "A6C", "bmode")
        with self.assertRaises(KeyError):
            r("lv_mood", "A4C", "bmode")

    def test_spectral_findings_are_never_relevant_to_a_video_modality(self):
        for cid, c in self.o.concepts.items():
            if c.required_modality == ("spectral_doppler",):
                for view in VIEWS:
                    for acq in ("bmode", "color_flow", "mmode"):
                        self.assertNotEqual(self.o.relevance(cid, view, acq), RELEVANT, (cid, view, acq))

    def test_values_and_labels(self):
        self.assertIsNone(self.o.value_error("lv_ejection_fraction", 35))
        self.assertIsNotNone(self.o.value_error("lv_ejection_fraction", 0))  # a 0 is not an EF
        self.assertIsNotNone(self.o.value_error("rv_systolic_pressure", 0))
        self.assertIsNotNone(self.o.value_error("lv_ejection_fraction", True))
        self.assertIsNotNone(self.o.value_error("mitral_regurgitation", "massive"))
        self.assertEqual(self.o.value_label("mitral_regurgitation", "moderate", "fr"), "modérée")
        self.assertEqual(self.o.value_label("lv_ejection_fraction", 35.0, "en"), "35%")

    def test_an_unranked_value_is_no_severity(self):
        self.assertIn("indeterminate", self.o.allowed_values("lv_diastolic_function"))
        self.assertEqual(self.o.severity_values("lv_diastolic_function"), ("normal", "grade_1", "grade_2", "grade_3"))
        self.assertEqual(self.o.severity_values("mitral_regurgitation"), self.o.allowed_values("mitral_regurgitation"))
        self.assertIsNone(self.o.severity_values("lv_ejection_fraction"))

    def test_normalize_keeps_offsets(self):
        text = "Insuffisance mitrale modérée, l’oreillette"
        self.assertEqual(len(normalize_text(text)), len(text))
        self.assertEqual(normalize_text(text), "insuffisance mitrale moderee, l'oreillette")


class TestValidator(unittest.TestCase):
    def refused(self, mutate, message):
        data = copy.deepcopy(raw())
        mutate(data)
        with self.assertRaises(OntologyError) as ctx:
            validate_ontology(data)
        self.assertIn(message, str(ctx.exception))

    def test_shipped_file_passes(self):
        validate_ontology(raw())

    def test_refusals(self):
        self.refused(lambda d: d.update(version="v1"), "positive integer")
        self.refused(lambda d: d.update(version="1.0.0"), "positive integer")
        self.refused(lambda d: d.update(version=True), "positive integer")
        self.refused(lambda d: d.pop("version"), "positive integer")
        self.refused(lambda d: d.pop("assessment_limitations"), "assessment_limitations")
        self.refused(lambda d: d["assessment_limitations"]["crop"].pop("fr"), "needs ['fr', 'en'] labels")
        self.refused(lambda d: d["concepts"].append(copy.deepcopy(d["concepts"][0])), "duplicate id")
        self.refused(lambda d: d["concepts"][0]["potentially_visible_in"][0]["views"].append("A6C"),
                     "neither a view nor a view group")
        self.refused(lambda d: d["concepts"][0]["potentially_visible_in"][0]["acquisitions"].append("doppler"),
                     "unknown acquisition")
        self.refused(lambda d: d["views"].pop("RVINF"), "views must be exactly")
        self.refused(lambda d: d.update(views=dict(reversed(list(d["views"].items())))), "in that order")
        self.refused(lambda d: d["views"]["A4C"].update(window="apex"), "window 'apex'")
        self.refused(lambda d: d["views"]["A4C"]["description"].pop("fr"), "needs a fr label and description")
        self.refused(lambda d: d["view_groups"]["apical_2c"].append("A4C"), "exactly one view group, not 2")
        self.refused(lambda d: d["view_groups"]["apical_2c"].append("OTHER"), "belongs to no view group")
        self.refused(lambda d: d["structures"]["aortic_root"].update(category="vessel"), "not in structure_categories")

        def concept(d, cid):
            return next(c for c in d["concepts"] if c["id"] == cid)

        self.refused(lambda d: concept(d, "lv_size").update(structures=["left_heart"]), "unknown structures")
        self.refused(lambda d: concept(d, "lv_size").update(structures=[]), "non-empty list")
        self.refused(lambda d: concept(d, "lv_visible").update(structures=["left_ventricle", "left_atrium"]),
                     "names exactly one structure")
        self.refused(lambda d: concept(d, "la_visible").update(structures=["left_ventricle"]),
                     "more than one visibility concept")
        self.refused(lambda d: concept(d, "lv_size")["potentially_visible_in"][0]["views"].append("OTHER"),
                     "OTHER is the unclassified view and carries no prior")

    def test_review_section_refusals(self):
        def question(d, qid):
            return next(q for q in d["review"]["questions"] if q["id"] == qid)

        self.refused(lambda d: question(d, "view_correct")["answers"][0].update(id=True),
                     "must be a unique snake_case string (quote yes and no)")
        self.refused(lambda d: question(d, "view_correct")["answers"][2].update(detail="correct_view"),
                     "detail 'correct_view' must be used exactly once, not 2 times")
        self.refused(lambda d: question(d, "missing_information")["answers"][1].update(detail="notes"),
                     "detail 'notes' not in")
        self.refused(lambda d: question(d, "unconfirmed_statements").pop("detail"), "a checklist question")
        self.refused(lambda d: question(d, "comments").update(required="no"), "required must be true or false")
        self.refused(lambda d: d["review"]["detail_labels"].pop("unsupported_kind"), "must label exactly")
        self.refused(lambda d: d["review"]["unconfirmed_reasons"].update(crop={"fr": "x", "en": "y"}),
                     "repeat assessment limitations ['crop']")
        self.refused(lambda d: d["concepts"][0].update(id="acquisition"), "reserved for the acquisition statement")
        self.refused(lambda d: d.pop("review"), "the review form section is missing")

        def spectral_prior_in_color(d):
            rvsp = next(c for c in d["concepts"] if c["id"] == "rv_systolic_pressure")
            rvsp["potentially_visible_in"][0]["acquisitions"].append("color_flow")

        self.refused(spectral_prior_in_color, "not in required_modality")

        def bad_scale(d):
            mr = next(c for c in d["concepts"] if c["id"] == "mitral_regurgitation")
            mr["value"]["scale"] = "nope"

        self.refused(bad_scale, "unknown scale")
        self.refused(lambda d: d["scales"]["stenosis"]["aliases"]["mild"].append("moderee"), "maps to both")
        self.refused(lambda d: d["scales"]["stenosis"].update(unranked=["unknown"]), "unranked values must be values")
        self.refused(lambda d: d["scales"]["stenosis"].update(unranked=["none"]), "other than negated_value")
        self.refused(lambda d: d["concepts"][0].update(kind="finding_maybe"), "kind")


if __name__ == "__main__":
    unittest.main()
