"""The view and type review for `OTHER` items: its committed JSON and doc, the validity rules and the routing back
into the column review. The SDK round trip is skipped where the Labelbox SDK is not installed."""

import importlib.util
import json
import unittest

from app.radar.ontology import VIEWS
from tools.labelbox import view_type_review as vt
from tools.labelbox.column_review import build_candidate, decode_annotations, load_spec, validate_annotations

VALID = {"input_type": "cine_loop", "view": "A4C", "modality": "bmode", "identifier_visible": "no"}


def answers(**changes):
    return {**VALID, **changes}


class TestForm(unittest.TestCase):
    def test_committed_json_and_doc_are_current(self):
        with open(vt.FORM_PATH, encoding="utf-8") as f:
            self.assertEqual(json.load(f), vt.build_form())
        self.assertEqual(vt.main(["--check"]), 0)

    def test_choices_share_ids_with_the_caption_reviews(self):
        form = {q["name"]: q for q in vt.build_form()["classifications"]}
        values = {name: [o["value"] for o in q["options"]] for name, q in form.items()}
        self.assertEqual(values["view"], list(VIEWS) + [vt.CANNOT_TELL, vt.NO_VIEW])
        self.assertEqual(values["modality"], list(load_spec()["modalities"]) + [vt.NO_ULTRASOUND])
        self.assertEqual(values["input_type"], list(vt.INPUT_TYPES))

    def test_every_question_is_global_and_only_comments_are_optional(self):
        for q in vt.build_form()["classifications"]:
            self.assertEqual(q["scope"], "global")
            self.assertEqual(q["required"], q["name"] != "comments", q["name"])

    def test_an_exported_review_decodes_and_validates(self):
        records = [{"name": k, "answer": {"name": v}} for k, v in VALID.items()]
        records.append({"name": "comments", "answer": "measurement frame"})
        validate_annotations(records, vt.build_form(), require_complete=True)
        self.assertEqual(vt.review_problems(decode_annotations(records)), [])

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "the Labelbox SDK is not installed")
    def test_sdk_round_trip(self):
        from labelbox import OntologyBuilder

        self.assertEqual(OntologyBuilder.from_dict(vt.build_form()).asdict(), vt.build_form())


class TestReviewProblems(unittest.TestCase):
    def test_valid_reviews(self):
        for review in [VALID,
                       answers(input_type="measurement_still", view="cannot_tell", modality="spectral"),
                       answers(input_type="text_page", view="no_view", modality="not_applicable"),
                       answers(input_type="analysis_screen", view="no_view", modality="not_applicable"),
                       answers(input_type="analysis_screen", view="A4C", modality="mixed", comments="strain")]:
            self.assertEqual(vt.review_problems(review), [], review)

    def test_missing_or_unknown_answers(self):
        self.assertIn("view is required", vt.review_problems({k: v for k, v in VALID.items() if k != "view"}))
        self.assertIn("view has no answer 'A6C'", vt.review_problems(answers(view="A6C")))
        self.assertIn("identifier_visible has no answer 'maybe'",
                      vt.review_problems(answers(identifier_visible="maybe")))

    def test_inconsistent_answers(self):
        for review in [answers(input_type="text_page", modality="not_applicable"),
                       answers(input_type="still_image", view="no_view", modality="not_applicable"),
                       answers(view="no_view"),
                       answers(modality="not_applicable"),
                       answers(input_type="other", view="no_view")]:
            self.assertTrue(vt.review_problems(review), review)


class TestRoute(unittest.TestCase):
    def test_cine_loops_and_plain_stills_go_to_the_column_review(self):
        self.assertEqual(vt.route(VALID), {"view": "A4C", "modality": "bmode", "input_kind": "video"})
        still = vt.route(answers(input_type="still_image", view="PLAX", modality="color_flow"))
        self.assertEqual(still, {"view": "PLAX", "modality": "color_flow", "input_kind": "image"})
        report = {"Left ventricule findings": "Fonction systolique normale du ventricule gauche"}
        candidate = build_candidate(report, global_key="synthetic-row", **still)
        self.assertFalse([c for c in candidate["columns"] if c["concept"] == "lv_function"])

    def test_everything_else_stays_out(self):
        for review in [answers(input_type="measurement_still"),
                       answers(input_type="analysis_screen"),
                       answers(view="OTHER"),
                       answers(view="cannot_tell"),
                       answers(modality="spectral"),
                       answers(modality="m_mode"),
                       answers(identifier_visible="yes"),
                       answers(view="no_view")]:
            self.assertIsNone(vt.route(review), review)


if __name__ == "__main__":
    unittest.main()
