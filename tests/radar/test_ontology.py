"""The versioned concept ontology: shipped file validity, the validator's refusals and the relevance prior."""

import copy
import json
import os
import unittest

import pandas as pd
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELCUSTOM = os.path.join(ROOT, "evals", "video_classification_frozen", "modelcustom")


def raw():
    with open(DEFAULT_ONTOLOGY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestShippedOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = load_ontology()

    def test_version_size_and_kinds(self):
        self.assertRegex(self.o.version, r"^\d+\.\d+\.\d+$")
        self.assertEqual(self.o.tag, f"concepts_v0@{self.o.version}")
        self.assertTrue(40 <= len(self.o.concepts) <= 50, len(self.o.concepts))
        kinds = {c.kind for c in self.o.concepts.values()}
        self.assertEqual(kinds, {"structure_visibility", "finding"})
        self.assertEqual(len(self.o.sha256), 64)

    def test_visibility_is_separate_from_disease(self):
        self.assertEqual(self.o.concept("mitral_valve_visible").kind, "structure_visibility")
        self.assertEqual(self.o.concept("mitral_regurgitation").kind, "finding")
        self.assertIn("severe", self.o.allowed_values("mitral_regurgitation"))
        self.assertEqual(self.o.allowed_values("mitral_valve_visible"), (True, False))

    def test_related_tasks_exist_in_panecho_and_echoprime(self):
        panecho = set(pd.read_pickle(os.path.join(MODELCUSTOM, "PanEcho", "content", "tasks.pkl")))
        with open(os.path.join(MODELCUSTOM, "EchoPrime", "assets", "per_section.json")) as f:
            echoprime = set(json.load(f))
        named = {"panecho": set(), "echoprime": set()}
        for c in self.o.concepts.values():
            for source, tasks in c.related_tasks.items():
                named[source].update(tasks)
        self.assertLessEqual(named["panecho"], panecho)
        self.assertLessEqual(named["echoprime"], echoprime)
        # Every EchoPrime finding task is represented by some concept.
        self.assertEqual(named["echoprime"], echoprime)

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
        self.refused(lambda d: d.update(version="v0"), "MAJOR.MINOR.PATCH")
        self.refused(lambda d: d.pop("version"), "MAJOR.MINOR.PATCH")
        self.refused(lambda d: d["concepts"].append(copy.deepcopy(d["concepts"][0])), "duplicate id")
        self.refused(lambda d: d["concepts"][0]["potentially_visible_in"][0]["views"].append("A6C"),
                     "neither a view nor a view group")
        self.refused(lambda d: d["concepts"][0]["potentially_visible_in"][0]["acquisitions"].append("doppler"),
                     "unknown acquisition")
        self.refused(lambda d: d.update(views=d["views"][:-1]), "views must be exactly")

        def spectral_prior_in_color(d):
            rvsp = next(c for c in d["concepts"] if c["id"] == "rv_systolic_pressure")
            rvsp["potentially_visible_in"][0]["acquisitions"].append("color_flow")

        self.refused(spectral_prior_in_color, "not in required_modality")

        def bad_scale(d):
            mr = next(c for c in d["concepts"] if c["id"] == "mitral_regurgitation")
            mr["value"]["scale"] = "nope"

        self.refused(bad_scale, "unknown scale")
        self.refused(lambda d: d["scales"]["stenosis"]["aliases"]["mild"].append("moderee"), "maps to both")
        self.refused(lambda d: d["concepts"][0].update(kind="finding_maybe"), "kind")


if __name__ == "__main__":
    unittest.main()
