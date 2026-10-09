"""The RADAR caption review form: the v1 questions and answers, the committed JSON and its Labelbox SDK round trip.

The SDK test is skipped where the Labelbox SDK is not installed; run it at least once, not skipped, before approval:

    UV_PROJECT_ENVIRONMENT=$HOME/.venvs/echojepa-labelbox uv run --frozen --only-group labelbox \\
        python -m unittest tools.labelbox.tests.test_review_ontology -v
"""

import importlib.util
import json
import unittest

from app.radar.ontology import ACQUISITION_STATEMENT, VIEWS, load_ontology
from tools.labelbox.review_ontology import (
    build_review_ontology,
    default_json_path,
    feature_names,
    review_version,
    serialize,
    statement_ids,
    walk,
)

ONTOLOGY = load_ontology()
HAS_SDK = importlib.util.find_spec("labelbox") is not None

# The questions and answers the pilot metrics (docs/radar/pilot_metrics_v1.md) are defined on.
CONTRACT = {
    "uninterpretable_clip": ("radio", True, ("no", "yes")),
    "view_correct": ("radio", True, ("yes", "no", "unsure", "not_applicable")),
    "caption_supported": ("radio", True, ("yes", "no", "unsure", "not_applicable")),
    "missing_information": ("radio", True, ("no", "yes", "not_applicable")),
    "unsupported_information": ("radio", True, ("no", "yes", "not_applicable")),
    "unconfirmed_statements": ("checklist", False, None),
    "corrected_caption": ("text", True, None),
    "comments": ("text", False, None),
}


def committed():
    with open(default_json_path(ONTOLOGY), encoding="utf-8") as f:
        return json.load(f)


def by_name(classifications):
    return {c["name"]: c for c in classifications}


def nested(classification, answer):
    return next(o for o in classification["options"] if o["value"] == answer)["options"]


class TestReviewForm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.form = build_review_ontology(ONTOLOGY)
        cls.q = by_name(cls.form["classifications"])

    def test_questions_answers_and_required(self):
        self.assertEqual(list(self.q), list(CONTRACT))
        for name, (kind, required, answers) in CONTRACT.items():
            c = self.q[name]
            self.assertEqual((c["type"], c["required"], c["scope"]), (kind, required, "global"), name)
            if answers:
                self.assertEqual(tuple(o["value"] for o in c["options"]), answers, name)
        self.assertEqual(self.form["review_version"], review_version(ONTOLOGY))
        self.assertEqual(self.form["review_version"], "radar-caption-ontology-v1")
        self.assertEqual(self.form["concept_ontology"]["sha256"], ONTOLOGY.sha256)

    def test_conditional_details(self):
        reason = nested(self.q["uninterpretable_clip"], "yes")[0]
        self.assertEqual([o["value"] for o in reason["options"]], list(ONTOLOGY.limitations))
        view = nested(self.q["view_correct"], "no")[0]
        self.assertEqual((view["name"], view["required"]), ("correct_view", True))
        self.assertEqual([o["value"] for o in view["options"]], list(VIEWS))
        missing = by_name(nested(self.q["missing_information"], "yes"))
        self.assertEqual(set(missing), {"missing_concepts", "missing_other"})
        self.assertEqual(missing["missing_other"]["type"], "text")
        for name in ("view_correct", "caption_supported", "missing_information", "unsupported_information"):
            self.assertEqual(nested(self.q[name], "not_applicable"), [], name)

    def test_statements_cover_visibility_and_acquisition_but_not_measurements(self):
        unsupported = nested(self.q["unsupported_information"], "yes")[0]
        self.assertTrue(unsupported["required"])
        values = [o["value"] for o in unsupported["options"]]
        self.assertEqual(values, list(statement_ids(ONTOLOGY)))
        self.assertEqual(values[0], ACQUISITION_STATEMENT)
        self.assertIn("mitral_valve_visible", values)  # a visibility claim can be flagged
        self.assertIn("mitral_regurgitation", values)
        self.assertNotIn("tapse", values)  # measurement findings are never stated
        kind = unsupported["options"][1]["options"][0]
        self.assertEqual([o["value"] for o in kind["options"]], ["not_supported", "wrong_value"])
        unconfirmed = self.q["unconfirmed_statements"]
        self.assertEqual([o["value"] for o in unconfirmed["options"]], values)
        reason = unconfirmed["options"][0]["options"][0]
        self.assertEqual([o["value"] for o in reason["options"]], ["uncertain", *ONTOLOGY.limitations])

    def test_names_are_unique_and_nesting_is_shallow(self):
        names = [c["name"] for c, _ in walk(self.form["classifications"])]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), feature_names(self.form))
        self.assertLessEqual(max(depth for _, depth in walk(self.form["classifications"])), 3)
        for c, _ in walk(self.form["classifications"]):
            values = [o["value"] for o in c["options"]]
            self.assertEqual(len(values), len(set(values)), c["name"])
            self.assertTrue(c["instructions"] and all(o["label"] for o in c["options"]), c["name"])

    def test_committed_json_is_up_to_date(self):
        with open(default_json_path(ONTOLOGY), encoding="utf-8") as f:
            self.assertEqual(f.read(), serialize(self.form),
                             "the review form JSON is out of date: run python -m tools.labelbox.review_ontology")


@unittest.skipUnless(HAS_SDK, "the Labelbox SDK is not installed")
class TestLabelboxSDK(unittest.TestCase):
    def test_round_trip_changes_nothing(self):
        from labelbox import OntologyBuilder

        form = committed()
        normalized = OntologyBuilder.from_dict(form).asdict()
        self.assertEqual(normalized, {"tools": form["tools"], "classifications": form["classifications"]})


if __name__ == "__main__":
    unittest.main()
