"""The reviewer instructions: their generated parts are current, every worked example is a valid review of its form,
an invalid example is refused, and the English and French versions have the same structure."""

import copy
import os
import re
import tempfile
import unittest

import yaml

from tools.labelbox import column_review as cr
from tools.labelbox import instructions as ins
from tools.labelbox import view_type_review as vt


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestInstructions(unittest.TestCase):
    def test_generated_parts_are_current(self):
        self.assertEqual(ins.main(["--check"]), 0)

    def test_both_languages_have_the_same_structure_and_every_question(self):
        en, fr = (read(ins.INSTRUCTIONS_PATHS[lang]) for lang in ("en", "fr"))
        for level in ("## ", "### ", "#### "):
            self.assertEqual(en.count("\n" + level), fr.count("\n" + level), level)
        questions = [q["name"] for q in cr.build_form()["classifications"]] + [q.name for q in vt.questions()]
        for name in questions:
            self.assertIn(f"`{name}`", en)
            self.assertIn(f"`{name}`", fr)

    def test_the_examples_cover_every_kind_of_answer(self):
        examples = ins.load_examples()
        scores = [e["score"] for e in examples["column_review"]]
        self.assertTrue(any(s["supported"] and not (s["rejected"] or s["cannot_assess"] or s["omissions"])
                            for s in scores))
        for key in ("rejected", "corrections", "cannot_assess", "omissions"):
            self.assertTrue(any(s[key] for s in scores), key)
        self.assertTrue(any(s["uninterpretable"] for s in scores))
        self.assertTrue(any(not s["routing"] and not s["uninterpretable"] for s in scores))  # a corrected view
        routes = [e["route"] for e in examples["view_type_review"]]
        self.assertTrue(any(routes) and not all(routes))
        self.assertIn("yes", {e["answers"]["identifier_visible"] for e in examples["view_type_review"]})
        self.assertEqual({e["answers"]["input_type"] for e in examples["view_type_review"]}, set(vt.INPUT_TYPES) - {
            "no_echo_image"})

    def test_invalid_examples_are_refused(self):
        with open(ins.EXAMPLES_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        wrong = next(e for e in raw["column_review"] if e["id"] == "wrong_severity")

        def refused(change):
            broken = copy.deepcopy(raw)
            change(broken)
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "examples.yaml")
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(broken, f, allow_unicode=True)
                with self.assertRaises(ValueError):
                    ins.load_examples(path)

        def first(broken):
            return next(e for e in broken["column_review"] if e["id"] == wrong["id"])

        refused(lambda b: first(b)["changes"].pop("lv_function__correction"))  # an error without its correction
        refused(lambda b: first(b)["columns"].update(aortic_regurgitant_jet="demonstrated"))  # not carried by A4C
        refused(lambda b: first(b)["changes"].update(view_ok="no"))  # a wrong view left as proposed
        refused(lambda b: first(b)["title"].pop("fr"))
        refused(lambda b: b["view_type_review"][0]["answers"].update(view="no_view"))
        refused(lambda b: next(e for e in b["view_type_review"] if e["id"] == "analysis_screen")["answers"].update(
            view="A4C"))  # a view the form does not ask for an analysis screen
        refused(lambda b: b["view_type_review"].append(copy.deepcopy(b["view_type_review"][0])))  # a repeated id

    def test_rebuilt_captions_come_from_the_scorer(self):
        en = read(ins.INSTRUCTIONS_PATHS["en"])
        for e in ins.load_examples()["column_review"]:
            if e["score"]["caption"]:
                self.assertIn(f"- **Rebuilt caption:** \"{e['score']['caption']}\"", en)
        self.assertFalse(re.search(r"\bdraft\b", en, re.I))


if __name__ == "__main__":
    unittest.main()
