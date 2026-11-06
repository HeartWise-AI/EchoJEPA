"""The one-question view review: its committed form, its choices, its prefill and what counts as a valid answer."""

import importlib.util
import json
import unittest

from tools.labelbox import view_review as vr
from tools.labelbox.column_review import load_spec

KEY = "radar-view-" + "a" * 24


class TestForm(unittest.TestCase):
    def test_committed_json_is_current(self):
        with open(vr.FORM_PATH, encoding="utf-8") as f:
            self.assertEqual(json.load(f), vr.build_form())
        self.assertEqual(vr.main(["--check"]), 0)

    def test_the_view_is_the_only_required_question(self):
        form = {q["name"]: q for q in vr.build_form()["classifications"]}
        self.assertEqual(list(form), ["view", "comments"])
        self.assertTrue(form["view"]["required"])
        self.assertFalse(form["comments"]["required"])

    def test_every_view_can_be_answered_but_psax_without_a_level(self):
        choices = vr.view_choices()
        refined = [view for view, route in load_spec()["views"].items() if route.get("refines")]
        self.assertTrue(set(refined) <= set(choices))
        self.assertTrue({"PLAX_ZOOM_AV", "PSAX_BASAL", "PSAX_MID", "PSAX_APICAL", "PLAX_ZOOM", "OTHER", "unsure"}
                        <= set(choices))
        self.assertNotIn("PSAX", choices)
        self.assertNotIn("cannot_tell", choices)
        self.assertEqual(choices["unsure"], "unsure (not clear / poor quality)")

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "the Labelbox SDK is not installed")
    def test_sdk_round_trip(self):
        from labelbox import OntologyBuilder

        self.assertEqual(OntologyBuilder.from_dict(vr.build_form()).asdict(), vr.build_form())


class TestPrefill(unittest.TestCase):
    def test_a_clip_opens_with_its_predicted_view(self):
        record, = vr.prelabels({"global_key": KEY, "view": "PLAX_ZOOM"})
        self.assertEqual(record["name"], "view")
        self.assertEqual(record["answer"], {"name": "PLAX_ZOOM"})
        self.assertEqual(record["dataRow"], {"globalKey": KEY})

    def test_a_psax_clip_opens_with_no_view(self):
        self.assertEqual(vr.prelabels({"global_key": KEY, "view": "PSAX"}), [])

    def test_a_prediction_the_form_does_not_offer_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a choice"):
            vr.prelabels({"global_key": KEY, "view": "A9C"})


class TestAnswers(unittest.TestCase):
    def test_a_view_or_unsure_is_valid(self):
        self.assertEqual(vr.review_problems({"view": "PSAX_MID"}), [])
        self.assertEqual(vr.review_problems({"view": "unsure"}), [])

    def test_no_view_or_a_view_the_form_does_not_offer_is_not(self):
        self.assertTrue(vr.review_problems({}))
        self.assertTrue(vr.review_problems({"view": "PSAX"}))


if __name__ == "__main__":
    unittest.main()
