"""Review consistency rules, and the worked examples of the reviewer instructions checked against the form, the claims
contract and those rules. Synthetic data only."""

import dataclasses
import itertools
import unittest

from app.radar.ontology import ACQUISITION_CATEGORIES, ACQUISITION_STATEMENT, RELEVANT, VIEWS, load_ontology
from app.radar.review_rules import (
    CLIP_QUESTIONS,
    Review,
    answer_problems,
    consistency_problems,
    example_review,
    load_examples,
    review_problems,
    unclaimed_flags,
)
from tools.labelbox.review_ontology import statement_ids, unconfirmed_reasons

ONTOLOGY = load_ontology()
FINE = {"uninterpretable_clip": "no", "view_correct": "yes", "caption_supported": "yes", "missing_information": "no",
        "unsupported_information": "no"}
VIEW = "A4C"  # the view the candidate caption of the synthetic reviews states
STATEMENTS = [ACQUISITION_STATEMENT, "mitral_regurgitation"]  # and the statements it makes
NA = dict.fromkeys(CLIP_QUESTIONS, "not_applicable")


def review(**changes):
    answers = dict(FINE, **changes.pop("answers", {}))
    changes.setdefault("corrected_caption", "Coupe apicale 4 cavités (A4C), Doppler couleur.")
    return Review(answers=answers, **changes)


def header(view, acquisition):
    """A caption's first sentence, in French: the view (None when the reviewer cannot tell it) and the acquisition."""
    shown = f"{ONTOLOGY.views[view].labels['fr']} ({view})" if view else ONTOLOGY.review.undetermined_view["fr"]
    return f"{shown}, {ONTOLOGY.acquisition_labels['fr'][acquisition]}."


def caption(view, acquisition, statements):
    """The caption a candidate with these statements reads (French, view first), as the claims contract requires."""
    parts = [header(view, acquisition)]
    for cid, value in statements.items():
        c = ONTOLOGY.concept(cid)
        if c.kind == "structure_visibility":
            parts.append(f"{c.labels['fr']}.")
        else:
            parts.append(f"{c.labels['fr']} : {ONTOLOGY.value_label(cid, value, 'fr')}.")
    return " ".join(parts)


class TestConsistencyRules(unittest.TestCase):
    def assertBreaks(self, r, message):
        problems = consistency_problems(r, VIEW)
        self.assertTrue(any(message in p for p in problems), problems)

    def test_a_consistent_review_passes(self):
        self.assertEqual(consistency_problems(review(), VIEW), [])

    def test_rules(self):
        self.assertBreaks(review(answers={"uninterpretable_clip": "yes"}, limitation="quality"), "all four")
        self.assertBreaks(review(answers=NA), "An interpretable clip answers every question")
        self.assertBreaks(review(answers={"uninterpretable_clip": "yes", **NA}), "needs a reason")
        self.assertEqual(consistency_problems(review(answers={"uninterpretable_clip": "yes", **NA},
                                                     limitation="quality"), VIEW), [])
        self.assertBreaks(review(answers={"view_correct": "no"}), "needs the correct view")
        self.assertBreaks(review(limitation="quality"), "A reason is given only for an uninterpretable clip")
        self.assertBreaks(review(correct_view="A5C"), "A correct view is given only for a wrong view")
        self.assertBreaks(review(unconfirmed={"mitral_regurgitation": "uncertain"}), "A supported caption has no")
        self.assertBreaks(review(answers={"caption_supported": "no"}), "goes with")
        self.assertBreaks(review(answers={"caption_supported": "no", "unsupported_information": "yes"}),
                          "at least one unsupported statement")
        self.assertBreaks(review(unsupported={"acquisition": "not_supported"}), "listed only under")
        self.assertBreaks(review(answers={"caption_supported": "unsure"}), "at least one unconfirmed")
        self.assertBreaks(review(answers={"missing_information": "yes"}), "a missing concept or a free-text item")
        self.assertBreaks(review(missing=("tricuspid_regurgitation",)), "listed only under missing information")
        self.assertBreaks(review(answers={"caption_supported": "no", "unsupported_information": "yes"},
                                 unsupported={"mitral_regurgitation": "wrong_value"},
                                 unconfirmed={"mitral_regurgitation": "uncertain"}),
                          "both unsupported and unconfirmed")

    def test_a_wrong_view_is_corrected_to_another_class(self):
        wrong = next(e for e in load_examples(ONTOLOGY) if e["id"] == "wrong_view")
        r = example_review(wrong)
        self.assertEqual(consistency_problems(r, wrong["view"]), [])
        self.assertIn(f"A wrong view is corrected to another class than the stated view {wrong['view']}.",
                      consistency_problems(dataclasses.replace(r, correct_view=wrong["view"]), wrong["view"]))

    def test_not_applicable_is_all_or_none(self):
        for n in range(1, len(CLIP_QUESTIONS) + 1):
            for subset in itertools.combinations(CLIP_QUESTIONS, n):
                partial = dict.fromkeys(subset, "not_applicable")
                self.assertBreaks(review(answers=partial), "An interpretable clip answers every question")
                if n < len(CLIP_QUESTIONS):
                    self.assertBreaks(review(answers={"uninterpretable_clip": "yes", **partial}, limitation="crop"),
                                      "all four")

    def test_unclaimed_flags(self):
        r = review(answers={"caption_supported": "no", "unsupported_information": "yes", "missing_information": "yes"},
                   unsupported={"pericardial_effusion": "not_supported"}, missing=("mitral_regurgitation",))
        flags = unclaimed_flags(r, [ACQUISITION_STATEMENT, "mitral_regurgitation"])
        self.assertEqual(flags, ["pericardial_effusion is flagged but not stated",
                                 "mitral_regurgitation is listed as missing but is stated"])


class TestValidity(unittest.TestCase):
    """A review counts only with the required answers, from the form's choices, consistent and without unclaimed
    flags; `consistency_problems` alone is not enough."""

    def test_refusals(self):
        def problems(r):
            return "\n".join(review_problems(r, ONTOLOGY, VIEW, STATEMENTS))

        self.assertEqual(problems(review()), "")
        self.assertEqual(consistency_problems(Review(answers={}), VIEW), [])
        self.assertIn("uninterpretable_clip is required", problems(Review(answers={})))
        partial = Review(answers={q: a for q, a in FINE.items() if q != "uninterpretable_clip"}, corrected_caption="x")
        self.assertIn("uninterpretable_clip is required", problems(partial))
        self.assertIn("view_correct has no answer 'maybe'", problems(review(answers={"view_correct": "maybe"})))
        self.assertIn("questions the form does not ask: ['overall']", problems(review(answers={"overall": "yes"})))
        self.assertIn("'A6C' is not a view class",
                      problems(review(answers={"view_correct": "no"}, correct_view="A6C")))
        self.assertIn("'blurry' is not an assessment limitation",
                      problems(review(answers={"uninterpretable_clip": "yes", **NA}, limitation="blurry")))
        self.assertIn("Missing concepts the form does not offer: ['acquisition']",
                      problems(review(answers={"missing_information": "yes"}, missing=(ACQUISITION_STATEMENT,))))
        wrong = {"caption_supported": "no", "unsupported_information": "yes"}
        self.assertIn("unsupported statements the form does not offer: ['made_up']",
                      problems(review(answers=wrong, unsupported={"made_up": "not_supported"})))
        self.assertIn("unsupported statements without an allowed kind: ['mitral_regurgitation']",
                      problems(review(answers=wrong, unsupported={"mitral_regurgitation": "maybe"})))
        self.assertIn("unconfirmed statements without an allowed reason: ['mitral_regurgitation']",
                      problems(review(answers={"caption_supported": "unsure"},
                                      unconfirmed={"mitral_regurgitation": None})))
        self.assertIn("`corrected_caption` is required", problems(review(corrected_caption=" ")))

    def test_null_answers_are_reported_not_raised(self):
        """An export gives null for an empty answer."""
        r = Review(answers=FINE, missing=None, missing_other=None, unsupported=None, unconfirmed=None,
                   corrected_caption=None)
        self.assertEqual(review_problems(r, ONTOLOGY, VIEW, STATEMENTS), ["`corrected_caption` is required."])
        self.assertIn("uninterpretable_clip is required.",
                      review_problems(Review(answers=None), ONTOLOGY, VIEW, STATEMENTS))

    def test_an_unclaimed_flag_keeps_a_review_out(self):
        """Consistent answers that flag a statement the caption does not make: the review is not valid, so it never
        counts as a clip with unsupported information."""
        r = review(answers={"caption_supported": "no", "unsupported_information": "yes"},
                   unsupported={"pericardial_effusion": "not_supported"})
        self.assertEqual(consistency_problems(r, VIEW), [])
        self.assertEqual(review_problems(r, ONTOLOGY, VIEW, [ACQUISITION_STATEMENT]),
                         ["pericardial_effusion is flagged but not stated"])

    def test_every_choice_of_the_form_is_accepted(self):
        statements = statement_ids(ONTOLOGY)
        flagged = [review(answers={"caption_supported": "unsure"},
                          unconfirmed=dict(zip(statements, itertools.cycle(unconfirmed_reasons(ONTOLOGY))))),
                   review(answers={"caption_supported": "no", "unsupported_information": "yes"},
                          unsupported=dict(zip(statements, itertools.cycle(ONTOLOGY.review.unsupported_kinds)))),
                   review(answers={"missing_information": "yes"}, missing=ONTOLOGY.statement_concepts)]
        flagged += [review(answers={"view_correct": "no"}, correct_view=v) for v in VIEWS]
        flagged += [review(answers={"uninterpretable_clip": "yes", **NA}, limitation=r) for r in ONTOLOGY.limitations]
        for r in flagged:
            self.assertEqual(answer_problems(r, ONTOLOGY), [], r)


class TestWorkedExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = load_examples(ONTOLOGY)

    def test_examples_are_valid_reviews(self):
        for e in self.examples:
            statements = [ACQUISITION_STATEMENT, *e["statements"]]
            self.assertEqual(review_problems(example_review(e), ONTOLOGY, e["view"], statements), [], e["id"])

    def test_examples_follow_the_claims_contract(self):
        for e in self.examples:
            for cid, value in e["statements"].items():
                self.assertEqual(ONTOLOGY.relevance(cid, e["view"], e["acquisition"]), RELEVANT, (e["id"], cid))
                self.assertIsNone(ONTOLOGY.value_error(cid, value), (e["id"], cid))
                if ONTOLOGY.concept(cid).kind == "structure_visibility":
                    self.assertIs(value, True, (e["id"], cid))
            self.assertEqual(e["caption"], caption(e["view"], e["acquisition"], e["statements"]), e["id"])
            self.assertNotEqual(e["corrected_caption"], e["caption"], f"{e['id']}: an unchanged caption is null")

    def test_corrected_captions_keep_the_caption_form(self):
        """The corrected header follows the answers: the corrected class after a wrong view, the undetermined view
        after "unsure", the unknown acquisition when the mode cannot be confirmed and another mode when it is wrong."""
        for e in self.examples:
            r = example_review(e)
            if r.answers["uninterpretable_clip"] == "yes":
                continue
            view = {"yes": e["view"], "no": r.correct_view, "unsure": None}[r.answers["view_correct"]]
            if ACQUISITION_STATEMENT in r.unconfirmed:
                acquisitions = ["unknown"]
            elif ACQUISITION_STATEMENT in r.unsupported:
                acquisitions = [a for a in ACQUISITION_CATEGORIES if a != e["acquisition"]]
            else:
                acquisitions = [e["acquisition"]]
            corrected = e["corrected_caption"] or e["caption"]
            self.assertTrue(any(corrected.startswith(header(view, a) + " ") or corrected == header(view, a)
                                for a in acquisitions), e["id"])

    def test_examples_cover_the_rules(self):
        reviews = [example_review(e) for e in self.examples]
        self.assertEqual({r.answers["caption_supported"] for r in reviews},
                         {"yes", "no", "unsure", "not_applicable"})
        self.assertEqual({k for r in reviews for k in r.unsupported.values()}, {"not_supported", "wrong_value"})
        reasons = {k for r in reviews for k in r.unconfirmed.values()}
        self.assertIn("uncertain", reasons)
        self.assertTrue(reasons & set(ONTOLOGY.limitations))
        self.assertEqual({r.answers["view_correct"] for r in reviews}, {"yes", "no", "unsure", "not_applicable"})
        self.assertIn(ACQUISITION_STATEMENT, {s for r in reviews for s in r.unconfirmed})
        self.assertIn("yes", {r.answers["missing_information"] for r in reviews})
        self.assertIn(ACQUISITION_STATEMENT, {s for r in reviews for s in r.unsupported})
        self.assertTrue(any(ONTOLOGY.concept(s).kind == "structure_visibility"
                            for r in reviews for s in r.unsupported if s != ACQUISITION_STATEMENT))
        self.assertTrue(any(v in ("none", False) for e in self.examples for v in e["statements"].values()))


if __name__ == "__main__":
    unittest.main()
