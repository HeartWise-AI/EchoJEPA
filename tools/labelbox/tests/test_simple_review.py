import json
import importlib.util
import unittest

from app.radar.ontology import VIEWS
from tools.labelbox.simple_review import FORM, SPEC, build_form, score_review


class TestSimpleRouting(unittest.TestCase):
    def test_twenty_views_and_a4c_does_not_route_aortic_findings(self):
        spec = json.loads(SPEC.read_text())
        self.assertEqual(set(spec["views"]), set(VIEWS))
        a4c = spec["views"]["A4C"]
        fields = a4c["bmode_priority"] + a4c["color_flow_priority"]
        self.assertNotIn("aortic_morphology", fields)
        self.assertNotIn("aortic_regurgitant_jet", fields)
        self.assertEqual(spec["views"]["OTHER"]["bmode_priority"], [])
        for route in spec["views"].values():
            for field in route["bmode_priority"]:
                self.assertNotEqual(spec["fields"][field].get("required_modality"), "color_flow")
            for field in route["color_flow_priority"]:
                self.assertEqual(spec["fields"][field]["required_modality"], "color_flow")
        self.assertIsNone(spec["fields"]["rv_function"]["verified_codes"]["-1"])


class TestSimpleSubmission(unittest.TestCase):
    def setUp(self):
        self.candidate = {"claims": [{"concept": "lv_function", "value": "mildly_reduced"},
                                     {"concept": "lv_size", "value": "normal"}],
                          "view": "A4C", "modality": "bmode",
                          "input_provenance_verified": True, "report_context": {"EF": 48}}
        self.answers = {"view_ok": "yes", "modality_ok": "yes", "claim_1_ok": "yes",
                        "claim_2_ok": "yes", "claim_3_ok": "no_claim", "claim_4_ok": "no_claim",
                        "claim_5_ok": "no_claim", "important_missing": "no"}

    def test_cannot_assess_is_not_supported_or_a_negative(self):
        self.answers["claim_2_ok"] = "cannot_assess"
        result = score_review(self.candidate, self.answers)
        self.assertEqual(result["supported"], self.candidate["claims"][:1])
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["resolved_count"], 1)
        self.assertFalse(result["all_supported"])
        self.assertNotIn("EF", str(result["supported"]))

    def test_rejected_claims_are_not_inverted(self):
        self.answers["claim_2_ok"] = "no"
        result = score_review(self.candidate, self.answers)
        self.assertEqual(result["rejected"], self.candidate["claims"][1:])
        self.assertEqual(result["supported"], self.candidate["claims"][:1])

    def test_wrong_routing_unverified_input_and_correction_block_release(self):
        for change in ({"view_ok": "no"}, {"modality_ok": "cannot_assess"},
                       {"optional_correction": "LV size is dilated"}):
            result = score_review(self.candidate, {**self.answers, **change})
            self.assertFalse(result["partial_caption_ready_for_validation"])
        result = score_review({**self.candidate, "input_provenance_verified": False}, self.answers)
        self.assertFalse(result["partial_caption_ready_for_validation"])

    def test_unsubmitted_or_inconsistent_answers_are_refused(self):
        for change in ({"view_ok": None}, {"claim_1_ok": "no_claim"}, {"claim_3_ok": "yes"},
                       {"important_missing": "yes"}):
            with self.assertRaises(ValueError):
                score_review(self.candidate, {**self.answers, **change})

    def test_committed_form(self):
        self.assertEqual(json.loads(FORM.read_text()), build_form())

    def test_missing_source_grade_and_wrong_modality_are_not_released(self):
        with self.assertRaises(ValueError):
            score_review({**self.candidate, "claims": [{"concept": "rv_function", "value": -1}]}, self.answers)
        jet = {"concept": "mitral_regurgitant_jet", "value": "demonstrated"}
        result = score_review({**self.candidate, "claims": [jet, self.candidate["claims"][1]]}, self.answers)
        self.assertFalse(result["partial_caption_ready_for_validation"])
        result = score_review({**self.candidate, "view": "OTHER"}, self.answers)
        self.assertFalse(result["partial_caption_ready_for_validation"])
        av = {"concept": "aortic_morphology", "value": "three_cusp_morphology"}
        result = score_review({**self.candidate, "claims": [av, self.candidate["claims"][1]]}, self.answers)
        self.assertFalse(result["partial_caption_ready_for_validation"])
        result = score_review({**self.candidate, "view": "A4C_ZOOM"}, self.answers)
        self.assertFalse(result["partial_caption_ready_for_validation"])

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK not installed")
    def test_sdk_import(self):
        from labelbox import OntologyBuilder

        form = build_form()
        self.assertEqual(OntologyBuilder.from_dict(form).asdict(), form)


if __name__ == "__main__":
    unittest.main()
