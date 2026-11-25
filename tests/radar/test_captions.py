"""Candidate captions: view/acquisition filtering, explicit gaps, no report text, and the strict LLM contract."""

import json
import unittest

from app.radar.captions import (
    LLM_OUTPUT_SCHEMA_VERSION,
    PROMPT_VERSION,
    RULE_GENERATOR_VERSION,
    LLMOutputError,
    build_llm_prompt,
    parse_llm_output,
    plan_clip,
    rule_based_caption,
)
from app.radar.demo import DEMO_EXPECTED, DEMO_LLM_OUTPUTS, DEMO_RECORDS, DEMO_TIME, expected_outputs, llm_outcomes
from app.radar.findings import extract_findings
from app.radar.records import read_records, record_problems
from tests.radar.synthetic import ONTOLOGY, finding, record, replace

STUDY = [
    finding("lv_systolic_function", "moderately_reduced"),
    finding("lv_ejection_fraction", 35),
    finding("mitral_regurgitation", "moderate"),
    finding("rv_systolic_pressure", 42),
    finding("tricuspid_regurgitation_ero", 0.2),
    finding("ivc_dilation", True),
]


def caption(view, acq, findings=STUDY, language="fr"):
    return rule_based_caption(record(1, view=view, acq=acq, findings=findings), ONTOLOGY, language=language,
                              created_utc=DEMO_TIME)


class TestRuleBased(unittest.TestCase):
    def test_demo_outputs_match_the_reviewed_expectation(self):
        records = read_records(DEMO_RECORDS, ONTOLOGY)
        with open(DEMO_EXPECTED, encoding="utf-8") as f:
            self.assertEqual(expected_outputs(ONTOLOGY, records), json.load(f))

    def test_bmode_clip_states_bmode_findings_only(self):
        cap = caption("A4C", "bmode")
        self.assertEqual([c.concept_id for c in cap.claims], ["lv_systolic_function", "lv_ejection_fraction"])
        self.assertTrue(cap.text.startswith("Coupe apicale 4 cavités (A4C), mode B."))
        self.assertIn("modérément diminuée", cap.text)
        self.assertNotIn("Insuffisance mitrale", cap.text)
        gaps = {g.concept_id: g.reason for g in cap.gaps}
        self.assertEqual(gaps, {"mitral_regurgitation": "modality_absent", "rv_systolic_pressure": "modality_absent",
                                "tricuspid_regurgitation_ero": "modality_absent"})
        self.assertEqual(cap.out_of_view, ["ivc_dilation"])

    def test_color_loop_never_carries_spectral_or_measured_values(self):
        cap = caption("A4C", "color_flow")
        self.assertEqual([c.concept_id for c in cap.claims], ["mitral_regurgitation"])
        self.assertNotIn("42", cap.text)
        self.assertNotIn("0.2", cap.text)
        gaps = {g.concept_id: g.reason for g in cap.gaps}
        self.assertEqual(gaps["rv_systolic_pressure"], "modality_absent")
        self.assertEqual(gaps["tricuspid_regurgitation_ero"], "measurement_required")
        self.assertEqual(gaps["lv_ejection_fraction"], "modality_absent")
        self.assertEqual(next(g for g in cap.gaps if g.concept_id == "rv_systolic_pressure").required_modality,
                         ["spectral_doppler"])

    def test_unknown_acquisition_and_unclassified_view_state_nothing(self):
        cap = caption("A4C", "unknown")
        self.assertEqual(cap.claims, [])
        self.assertTrue(all(g.reason == "acquisition_unknown" for g in cap.gaps if g.concept_id != "ivc_dilation"))
        cap = caption("OTHER", "bmode")
        self.assertEqual((cap.claims, cap.gaps), ([], []))
        self.assertEqual(cap.text, "Coupe non classée (OTHER), mode B.")

    def test_versions_and_status_are_stamped(self):
        cap = caption("SUBCOSTAL", "bmode", language="en")
        self.assertEqual((cap.generator, cap.generator_version, cap.ontology_version, cap.status),
                         ("rule_based", RULE_GENERATOR_VERSION, ONTOLOGY.version, "unvalidated"))
        self.assertEqual(cap.text, "Subcostal view (SUBCOSTAL), B-mode. Left ventricular systolic function: "
                                   "moderately reduced. Inferior vena cava dilation.")
        self.assertIn("ivc_visible", cap.expected_structures)

    def test_report_text_is_never_copied(self):
        report = {
            "Mitral valve findings": "Insuffisance mitrale modérée, jet excentrique dirigé vers la paroi postérieure, "
                                     "prolapsus du feuillet postérieur P2 à confirmer en ETO.",
            "Referring TR Grade": "Légère",
        }
        findings, _ = extract_findings(report, ONTOLOGY)
        rec = record(3, acq="color_flow", findings=findings)
        cap = rule_based_caption(rec, ONTOLOGY)
        self.assertEqual([c.concept_id for c in cap.claims], ["mitral_regurgitation", "tricuspid_regurgitation"])
        # Ontology labels may coincide with report words ("Insuffisance mitrale"); no longer run of report text may.
        text = report["Mitral valve findings"]
        for start in range(0, len(text) - 25):
            self.assertNotIn(text[start:start + 25], cap.text)
        self.assertNotIn("excentrique", cap.text)
        self.assertEqual(record_problems(replace(rec, candidate_caption=cap), ONTOLOGY), [])

    def test_boolean_absence_wording(self):
        cap = caption("PSAX_AV", "bmode", findings=[finding("bicuspid_aortic_valve", False)])
        self.assertIn("Absence de bicuspidie aortique.", cap.text)
        cap = caption("A4C", "bmode", findings=[finding("atrial_septal_hypertrophy", False)], language="en")
        self.assertIn("No lipomatous atrial septal hypertrophy.", cap.text)


class TestLLMContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_records(DEMO_RECORDS, ONTOLOGY)

    def test_demo_cases(self):
        with open(DEMO_LLM_OUTPUTS, encoding="utf-8") as f:
            cases = {c["name"]: c for c in json.load(f)["cases"]}
        for result in llm_outcomes(ONTOLOGY, self.records):
            case = cases[result["name"]]
            with self.subTest(case=case["name"]):
                self.assertEqual(result["outcome"], case["expect"])
                if case["expect"] == "rejected":
                    self.assertIn(case["error"], result["error"])

    def test_accepted_output_becomes_an_unvalidated_candidate(self):
        with open(DEMO_LLM_OUTPUTS, encoding="utf-8") as f:
            case = json.load(f)["cases"][0]
        rec = self.records[case["record"]]
        cap = parse_llm_output(case["response"], rec, ONTOLOGY, model_id="endpoint-x", created_utc=DEMO_TIME)
        self.assertEqual((cap.generator, cap.generator_version, cap.prompt_version, cap.status),
                         ("llm", "endpoint-x", PROMPT_VERSION, "unvalidated"))
        self.assertEqual([c.concept_id for c in cap.claims], ["mitral_regurgitation", "tricuspid_regurgitation"])
        self.assertEqual(cap.gaps, list(plan_clip(rec, ONTOLOGY).gaps))  # gaps come from the plan, not the model
        self.assertEqual(record_problems(replace(rec, candidate_caption=cap), ONTOLOGY), [])

    def test_invalid_outputs(self):
        rec = self.records[1]
        good = {"schema_version": LLM_OUTPUT_SCHEMA_VERSION, "language": "fr", "caption": "Coupe A4C.",
                "claims": [], "omitted": []}
        bad = [
            "not json",
            "[]",
            json.dumps({**good, "claims": [{"concept_id": "mitral_regurgitation"}]}),
            json.dumps({**good, "claims": [{"concept_id": "mitral_regurgitation", "value": "moderate"}] * 2}),
            json.dumps({**good, "omitted": [{"concept_id": "lv_size", "reason": "lazy"}]}),
            json.dumps({**good, "caption": ""}),
            json.dumps({**good, "caption": "Accession 1234567."}),
            json.dumps({**good, "schema_version": "radar-llm-caption/9"}),
        ]
        for raw in bad:
            with self.subTest(raw=raw[:60]), self.assertRaises(LLMOutputError):
                parse_llm_output(raw, rec, ONTOLOGY, model_id="m")

    def test_prompt_is_structured_and_identifier_free(self):
        rec = self.records[1]
        prompt = build_llm_prompt(rec, ONTOLOGY)
        self.assertEqual(prompt["prompt_version"], PROMPT_VERSION)
        payload = json.loads(prompt["user"])
        self.assertEqual([f["concept_id"] for f in payload["allowed_findings"]],
                         ["mitral_regurgitation", "tricuspid_regurgitation"])
        self.assertIn("rv_systolic_pressure", [g["concept_id"] for g in payload["gaps"]])
        everything = prompt["system"] + prompt["user"]
        for value in (rec.study_key, rec.patient_key, rec.video_path, rec.record_id):
            self.assertNotIn(value, everything)

    def test_prompts_refuse_test_split_records(self):
        rec = replace(self.records[1], split="test")
        with self.assertRaisesRegex(ValueError, "test split"):
            build_llm_prompt(rec, ONTOLOGY)
        build_llm_prompt(rec, ONTOLOGY, allow_test_split=True)


if __name__ == "__main__":
    unittest.main()
