"""The view and type review for `OTHER` items: its committed JSON, the validity rules and the routing back
into the column review. The SDK round trip is skipped where the Labelbox SDK is not installed."""

import importlib.util
import json
import unittest

from app.radar.ontology import VIEWS
from tools.labelbox import view_type_review as vt
from tools.labelbox.column_review import (
    build_candidate,
    decode_annotations,
    encode_answers,
    load_spec,
    validate_annotations,
)

VALID = {"input_type": "cine_loop", "view": "A4C", "modality": "bmode", "identifier_visible": "no"}
NOT_A_LOOP = {"input_type": "analysis_screen", "identifier_visible": "no"}


def answers(**changes):
    return {**VALID, **changes}


class TestForm(unittest.TestCase):
    def test_committed_json_is_current(self):
        with open(vt.FORM_PATH, encoding="utf-8") as f:
            self.assertEqual(json.load(f), vt.build_form())
        self.assertEqual(vt.main(["--check"]), 0)

    def test_view_and_mode_are_asked_only_after_a_cine_loop(self):
        form = {q["name"]: q for q in vt.build_form()["classifications"]}
        self.assertEqual(list(form), ["input_type", "identifier_visible", "comments"])
        options = {o["value"]: o for o in form["input_type"]["options"]}
        self.assertEqual(list(options), list(vt.INPUT_TYPES))
        nested = {q["name"]: q for q in options["cine_loop"]["options"]}
        self.assertEqual(list(nested), ["view", "modality"])
        self.assertTrue(all(not o["options"] for value, o in options.items() if value != "cine_loop"))
        self.assertEqual([o["value"] for o in nested["view"]["options"]], list(VIEWS) + [vt.CANNOT_TELL])
        self.assertEqual([o["value"] for o in nested["modality"]["options"]], list(load_spec()["modalities"]))
        self.assertTrue(all(q["required"] and "scope" not in q for q in nested.values()))

    def test_top_level_questions_are_global_and_only_comments_are_optional(self):
        for q in vt.build_form()["classifications"]:
            self.assertEqual(q["scope"], "global")
            self.assertEqual(q["required"], q["name"] != "comments", q["name"])

    def test_an_exported_review_decodes_and_validates(self):
        form = vt.build_form()
        for review in (VALID, NOT_A_LOOP, {**NOT_A_LOOP, "input_type": "no_echo_image", "comments": "text page"}):
            records = encode_answers(review, form)
            validate_annotations(records, form, require_complete=True)
            self.assertEqual(decode_annotations(records), review)
            self.assertEqual(vt.review_problems(decode_annotations(records)), [])
        with self.assertRaises(ValueError):
            encode_answers({**NOT_A_LOOP, "view": "A4C"}, form)  # Labelbox never shows it

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "the Labelbox SDK is not installed")
    def test_sdk_round_trip(self):
        from labelbox import OntologyBuilder

        self.assertEqual(OntologyBuilder.from_dict(vt.build_form()).asdict(), vt.build_form())


class TestReviewProblems(unittest.TestCase):
    def test_valid_reviews(self):
        for review in [VALID, NOT_A_LOOP,
                       answers(view="cannot_tell", modality="spectral"),
                       answers(view="OTHER", modality="mixed", comments="strain"),
                       {**NOT_A_LOOP, "input_type": "no_echo_image", "identifier_visible": "yes"}]:
            self.assertEqual(vt.review_problems(review), [], review)

    def test_missing_unknown_or_unasked_answers(self):
        self.assertIn("view is required", vt.review_problems({k: v for k, v in VALID.items() if k != "view"}))
        self.assertIn("view has no answer 'no_view'", vt.review_problems(answers(view="no_view")))
        self.assertIn("modality has no answer 'not_applicable'",
                      vt.review_problems(answers(modality="not_applicable")))
        self.assertIn("identifier_visible has no answer 'maybe'",
                      vt.review_problems(answers(identifier_visible="maybe")))
        self.assertIn("input_type has no answer 'still_image'", vt.review_problems(answers(input_type="still_image")))
        self.assertIn("input_type has no answer 'should_be_removed'",
                      vt.review_problems({**NOT_A_LOOP, "input_type": "should_be_removed"}))
        self.assertEqual(vt.review_problems({**NOT_A_LOOP, "modality": "bmode"}),
                         ["modality is asked only when input_type is 'cine_loop'"])

    def test_unasked_questions_are_recorded_as_not_applicable(self):
        self.assertEqual(vt.with_not_applicable(NOT_A_LOOP),
                         {**NOT_A_LOOP, "view": "not_applicable", "modality": "not_applicable"})
        self.assertEqual(vt.with_not_applicable(VALID), VALID)


class TestRoute(unittest.TestCase):
    def test_cine_loops_go_to_the_column_review(self):
        self.assertEqual(vt.route(VALID), {"view": "A4C", "modality": "bmode", "input_kind": "video"})
        loop = vt.route(answers(view="PLAX", modality="color_flow"))
        self.assertEqual(loop, {"view": "PLAX", "modality": "color_flow", "input_kind": "video"})
        report = {"Left ventricule findings": "Fonction systolique normale du ventricule gauche"}
        self.assertTrue(build_candidate(report, global_key="synthetic-row", **loop)["columns"])

    def test_everything_else_stays_out(self):
        for review in [NOT_A_LOOP,
                       {**NOT_A_LOOP, "input_type": "no_echo_image"},
                       answers(view="OTHER"),
                       answers(view="cannot_tell"),
                       answers(modality="spectral"),
                       answers(modality="m_mode"),
                       answers(identifier_visible="yes"),
                       answers(view="no_view")]:
            self.assertIsNone(vt.route(review), review)


if __name__ == "__main__":
    unittest.main()
