"""Clip records: strict decoding, semantic validation, approval derived only from returned review data, JSONL I/O."""

import json
import os
import tempfile
import unittest

from app.radar.demo import DEMO_RECORDS
from app.radar.records import (
    ApprovedCaption,
    Claim,
    ConceptReason,
    RecordError,
    WrongValue,
    derive_approved_caption,
    make_record_id,
    read_records,
    record_from_dict,
    record_problems,
    validate_record,
    write_records,
)
from tests.radar.synthetic import ONTOLOGY, T1, candidate, finding, record, replace, reviewed, with_review

MR = finding("mitral_regurgitation", "moderate", "Referring MR Grade")


def color_record(**changes):
    rec = record(1, acq="color_flow", findings=[MR],
                 caption=candidate("Coupe A4C, Doppler couleur. Insuffisance mitrale : modérée.",
                                   [("mitral_regurgitation", "moderate")]))
    return replace(rec, **changes)


class TestRecordIds(unittest.TestCase):
    def test_pseudonymous_and_stable(self):
        a = make_record_id("secret", "/media/x/study-123/video.avi")
        self.assertEqual(a, make_record_id("secret", "/media/x/study-123/video.avi"))
        self.assertNotEqual(a, make_record_id("other", "/media/x/study-123/video.avi"))
        self.assertRegex(a, r"^rc-[0-9a-f]{24}$")
        self.assertNotIn("123", a[:3])
        with self.assertRaises(ValueError):
            make_record_id("", "x")


class TestValidation(unittest.TestCase):
    def problems(self, rec):
        return "\n".join(record_problems(rec, ONTOLOGY))

    def test_demo_records_are_valid(self):
        self.assertEqual(len(read_records(DEMO_RECORDS, ONTOLOGY)), 9)

    def test_valid_candidate(self):
        validate_record(color_record(), ONTOLOGY)

    def test_field_errors(self):
        cases = [
            (dict(record_id="study-a-1"), "record_id"),
            (dict(view="A6C"), "classifier view"),
            (dict(split="holdout"), "split"),
            (dict(view_confidence=1.5), "view_confidence"),
            (dict(report_findings=[finding("mitral_regurgitation", "massive")]), "is not one of"),
            (dict(report_findings=[MR, MR]), "repeats"),
            (dict(report_findings=[finding("mitral_valve_visible", True)]), "visibility concept"),
        ]
        for changes, message in cases:
            with self.subTest(changes=list(changes)):
                self.assertIn(message, self.problems(color_record(**changes)))

    def test_candidate_cannot_claim_what_the_report_or_clip_does_not_support(self):
        not_in_report = candidate("x", [("tricuspid_regurgitation", "mild")])
        self.assertIn("not a report finding", self.problems(color_record(candidate_caption=not_in_report)))
        wrong_value = candidate("x", [("mitral_regurgitation", "severe")])
        self.assertIn("not a report finding", self.problems(color_record(candidate_caption=wrong_value)))
        bmode = color_record()
        bmode.acquisition.category = "bmode"
        self.assertIn("modality_absent", self.problems(bmode))
        rvsp = color_record(report_findings=[MR, finding("rv_systolic_pressure", 42)],
                            candidate_caption=candidate("x", [("rv_systolic_pressure", 42)]))
        self.assertIn("cannot be stated", self.problems(rvsp))

    def test_candidate_status_and_versions(self):
        cap = candidate("x", [("mitral_regurgitation", "moderate")])
        cap.status = "approved"
        self.assertIn("must stay 'unvalidated'", self.problems(color_record(candidate_caption=cap)))
        cap = candidate("x", [("mitral_regurgitation", "moderate")])
        cap.ontology_version = "0.0.9"
        self.assertIn("built with ontology", self.problems(color_record(candidate_caption=cap)))
        cap = candidate("x", generator="llm")
        cap.prompt_version = None
        self.assertIn("prompt_version", self.problems(color_record(candidate_caption=cap)))

    def test_identifiers_never_in_caption_text(self):
        rec = color_record()
        rec.candidate_caption.text = "Étude study-a : insuffisance mitrale."
        self.assertIn("local identifier", self.problems(rec))
        rec = color_record()
        rec.candidate_caption.text = "video-0001"
        self.assertIn("local identifier", self.problems(rec))

    def test_review_rules(self):
        self.assertIn("carries review data", self.problems(color_record(review=reviewed(status="unreviewed"))))
        self.assertIn("reviewer_id is required", self.problems(color_record(review=reviewed(reviewer_id=None))))
        self.assertIn("not an e-mail", self.problems(color_record(review=reviewed(reviewer_id="a@b.org"))))
        self.assertIn("needs a candidate", self.problems(color_record(candidate_caption=None, review=reviewed())))
        self.assertIn("UTC", self.problems(color_record(review=reviewed(completed_utc="2026-01-02 10:00"))))

    def test_approved_caption_only_from_review(self):
        rec = color_record()
        rec.approved_caption = ApprovedCaption("x", "fr", "candidate_accepted", "v", "r", "l", T1, ONTOLOGY.version)
        self.assertIn("without a completed review", self.problems(rec))
        rec = with_review(color_record(), reviewed(satisfied=False, corrected_caption="Insuffisance mitrale sévère."))
        rec.approved_caption.text = "something the reviewer did not write"
        self.assertIn("does not follow from the review", self.problems(rec))

    def test_approved_concept_values_must_follow_from_the_review(self):
        rec = with_review(color_record(), reviewed())
        self.assertEqual(self.problems(rec), "")
        rec.approved_caption.concepts[0].value = "severe"  # the review still approves "moderate"
        self.assertIn("does not follow from the review", self.problems(rec))
        rec = with_review(color_record(), reviewed())
        rec.approved_caption.concepts.append(Claim("mitral_valve_visible", None))
        self.assertIn("does not follow from the review", self.problems(rec))


class TestApproval(unittest.TestCase):
    def test_satisfied_review_approves_the_candidate(self):
        rec = with_review(color_record(), reviewed())
        self.assertEqual(rec.approved_caption.derived_from, "candidate_accepted")
        self.assertEqual(rec.approved_caption.text, rec.candidate_caption.text)
        self.assertEqual([(c.concept_id, c.value) for c in rec.approved_caption.concepts],
                         [("mitral_regurgitation", "moderate")])

    def test_correction_wins_and_concepts_follow_the_review(self):
        review = reviewed(satisfied=False, corrected_caption="Coupe A4C, Doppler couleur. IM sévère.",
                          wrong_values=[WrongValue("mitral_regurgitation", "moderate", "severe")],
                          missing_concepts=["mitral_valve_visible"])
        rec = with_review(color_record(), review)
        self.assertEqual(rec.approved_caption.derived_from, "corrected_caption")
        self.assertEqual([(c.concept_id, c.value) for c in rec.approved_caption.concepts],
                         [("mitral_regurgitation", "severe"), ("mitral_valve_visible", None)])

    def test_flags_without_correction_approve_nothing(self):
        for review in (
            reviewed(satisfied=False),
            reviewed(unsupported_concepts=["mitral_regurgitation"]),
            reviewed(clip_unassessable_reason="quality", corrected_caption="x y z"),
        ):
            with self.subTest(review=review):
                self.assertIsNone(with_review(color_record(), review).approved_caption)

    def test_uncertain_or_unassessable_flags_block_an_unchanged_caption(self):
        mr = "mitral_regurgitation"
        for flag in (dict(uncertain=[ConceptReason(mr, "quality")]), dict(unassessable=[ConceptReason(mr, "crop")])):
            unchanged = color_record().candidate_caption.text
            for review in (reviewed(**flag), reviewed(**flag, corrected_caption=unchanged),
                           reviewed(**flag, corrected_caption=f"  {unchanged}\n")):  # the form's preloaded text
                with self.subTest(flag=list(flag), corrected=review.corrected_caption):
                    self.assertIsNone(with_review(color_record(), review).approved_caption)

    def test_a_correction_resolves_the_flags_and_drops_their_concepts(self):
        review = reviewed(satisfied=False, corrected_caption="Coupe A4C, Doppler couleur.",
                          uncertain=[ConceptReason("mitral_regurgitation", "quality")],
                          missing_concepts=["mitral_valve_visible"])
        approved = with_review(color_record(), review).approved_caption
        self.assertEqual(approved.derived_from, "corrected_caption")
        self.assertEqual([(c.concept_id, c.value) for c in approved.concepts], [("mitral_valve_visible", None)])
        # Flagged as missing and as uncertain: not asserted either.
        review.uncertain.append(ConceptReason("mitral_valve_visible", "crop"))
        self.assertEqual(with_review(color_record(), review).approved_caption.concepts, [])

    def test_in_review_approves_nothing(self):
        rec = color_record(review=reviewed(status="in_review", completed_utc=None, satisfied=None))
        self.assertIsNone(derive_approved_caption(rec, ONTOLOGY))

    def test_unassessable_concepts_leave_the_concept_set(self):
        review = reviewed(satisfied=False, corrected_caption="Coupe A4C, Doppler couleur.",
                          unassessable=[ConceptReason("mitral_regurgitation", "crop")])
        self.assertEqual(with_review(color_record(), review).approved_caption.concepts, [])


class TestJsonl(unittest.TestCase):
    def test_round_trip_and_line_numbered_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "records.jsonl")
            recs = [color_record(), with_review(record(2, findings=[MR], caption=candidate("Coupe A4C.")), reviewed())]
            write_records(path, recs, ONTOLOGY)
            back = read_records(path, ONTOLOGY)
            self.assertEqual([r.to_dict() for r in back], [r.to_dict() for r in recs])

            with open(path) as f:
                lines = f.read().splitlines()
            bad = json.loads(lines[1])
            bad["view"] = "A6C"
            extra = json.loads(lines[0])
            extra["patient_name"] = "nobody"
            with open(path, "w") as f:
                f.write("\n".join([json.dumps(extra), json.dumps(bad), lines[1], "{not json"]) + "\n")
            with self.assertRaises(RecordError) as ctx:
                read_records(path, ONTOLOGY)
            message = str(ctx.exception)
            for expected in ("line 1: RecordError: record has unknown fields ['patient_name']", "line 2: view",
                             "line 3: duplicate record_id", "line 4: JSONDecodeError"):
                self.assertIn(expected, message)
            self.assertNotIn("/synthetic/videos", message)
            self.assertNotIn(tmp, message)

    def test_strict_decoding(self):
        data = color_record().to_dict()
        del data["acquisition"]
        with self.assertRaisesRegex(RecordError, "acquisition is required"):
            record_from_dict(data)

    def test_write_refuses_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RecordError):
            write_records(os.path.join(tmp, "r.jsonl"), [color_record(), color_record()], ONTOLOGY)


if __name__ == "__main__":
    unittest.main()
