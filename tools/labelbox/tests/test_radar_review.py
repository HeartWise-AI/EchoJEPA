"""RADAR caption review: the review form, guarded local payloads and export parsing. No Labelbox access is used."""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from app.radar.captions import rule_based_caption
from app.radar.demo import DEMO_RECORDS, DEMO_TIME
from app.radar.ontology import load_ontology
from app.radar.records import read_records, write_records
from tools.labelbox import radar_review
from tools.labelbox.payloads import (
    PLACEHOLDER_URL_TEMPLATE,
    UnsafePayloadError,
    assert_payload_safe,
    build_data_rows,
    build_prelabels,
    payload_problems,
    validate_url_template,
)
from tools.labelbox.review_export import ExportError, apply_export
from tools.labelbox.review_ontology import build_review_ontology, feature_names, review_version

ONTOLOGY = load_ontology()
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def captioned_demo():
    records = read_records(DEMO_RECORDS, ONTOLOGY)
    for r in records:
        r.candidate_caption = rule_based_caption(r, ONTOLOGY, created_utc=DEMO_TIME)
    return records


def export_row(record, classifications, status="DONE", labels=None, account="cardio@example.org"):
    label = {"id": "label-1", "label_details": {"created_by": account, "created_at": "2026-10-07T10:00:00Z",
                                                "updated_at": "2026-10-07T10:05:00.123Z"},
             "annotations": {"frames": {}, "segments": {}, "classifications": classifications}}
    return {"data_row": {"id": "dr-1", "global_key": record.record_id},
            "projects": {"project-1": {"project_details": {"workflow_status": status},
                                       "labels": [label] if labels is None else labels}}}


def radio(name, value):
    return {"name": name.title(), "value": name, "radio_answer": {"name": value.title(), "value": value}}


def checklist(name, options):
    return {"value": name, "checklist_answers": options}


def text(name, content):
    return {"value": name, "text_answer": {"content": content}}


class TestReviewForm(unittest.TestCase):
    def test_questions_and_options(self):
        form = build_review_ontology(ONTOLOGY)
        by_name = {c["name"]: c for c in form["classifications"]}
        self.assertEqual(list(by_name), ["caption_satisfactory", "clip_assessability", "missing_visible_concepts",
                                         "unsupported_concepts", "wrong_value_concepts", "uncertain_concepts",
                                         "unassessable_concepts", "corrected_caption", "comments"])
        self.assertTrue(by_name["caption_satisfactory"]["required"])
        self.assertEqual({c["scope"] for c in form["classifications"]}, {"global"})
        missing = {o["value"] for o in by_name["missing_visible_concepts"]["options"]}
        self.assertIn("mitral_valve_visible", missing)  # visibility and findings are separate options
        self.assertIn("mitral_regurgitation", missing)
        unsupported = {o["value"] for o in by_name["unsupported_concepts"]["options"]}
        self.assertNotIn("mitral_valve_visible", unsupported)  # captions claim findings, not visibility
        wrong = by_name["wrong_value_concepts"]["options"][0]
        self.assertEqual(wrong["options"][0]["type"], "text")
        reason = by_name["unassessable_concepts"]["options"][0]["options"][0]
        self.assertEqual([o["value"] for o in reason["options"]], ["quality", "crop", "modality", "other"])
        self.assertEqual(form["review_version"], f"radar-review/0.2+{ONTOLOGY.tag}")
        names = []

        def walk(cs):
            for c in cs:
                names.append(c["name"])
                for o in c["options"]:
                    walk(o["options"])

        walk(form["classifications"])
        self.assertEqual(len(names), len(set(names)))  # nested names are unique
        self.assertEqual(set(names), feature_names(form))


class TestPayloads(unittest.TestCase):
    def setUp(self):
        self.records = captioned_demo()

    def test_data_rows_carry_only_pseudonymous_keys_and_captions(self):
        rows = build_data_rows(self.records, ONTOLOGY)
        self.assertEqual(len(rows), 9)
        row = rows[1]
        self.assertEqual(row["global_key"], self.records[1].record_id)
        self.assertEqual(row["row_data"], PLACEHOLDER_URL_TEMPLATE.format(global_key=row["global_key"]))
        self.assertEqual(row["media_type"], "VIDEO")
        attachments = {a["name"]: a["value"] for a in row["attachments"]}
        self.assertEqual(attachments["candidate_caption"], self.records[1].candidate_caption.text)
        self.assertIn("Pression systolique ventriculaire droite (modality_absent)",
                      attachments["not_groundable_in_this_clip"])
        self.assertNotIn("42", attachments["not_groundable_in_this_clip"])  # gap values stay local
        dumped = json.dumps(rows)
        for r in self.records:
            for value in (r.study_key, r.patient_key, r.video_path, os.path.basename(r.video_path)):
                self.assertNotIn(value, dumped)

    def test_prelabels_preload_the_caption(self):
        pre = build_prelabels(self.records)
        self.assertEqual(pre[0]["name"], "corrected_caption")
        self.assertEqual(pre[0]["answer"], self.records[0].candidate_caption.text)
        self.assertEqual(pre[0]["dataRow"], {"globalKey": self.records[0].record_id})
        self.assertEqual(pre, build_prelabels(self.records))  # stable ids

    def test_guard_refuses_identifiers_and_paths(self):
        cases = [
            ({"video_path": "x"}, "identifier field name"),
            ({"MRN": "x"}, "identifier field name"),
            ({"AccessionNumber": "x"}, "identifier field name"),
            ({"caption": "/media/data1/datasets/a.avi"}, "local path"),
            ({"caption": "see ~/clips"}, "local path"),
            ({"caption": "C:\\data\\x"}, "local path"),
            ({"caption": "file:///tmp/a"}, "local path"),
            ({"caption": "clip IM_0001.dcm"}, "file name"),
            ({"caption": "1.2.840.113619.2.55.3"}, "DICOM UID"),
            ({"caption": "examen du 2019-03-04"}, "date"),
            ({"caption": "examen 20190304"}, "date"),
            ({"caption": "dossier 0012345"}, "long digit run"),
            ({"caption": "par doc@hospital.org"}, "e-mail address"),
            ({"global_key": "study-123"}, "not a pseudonymous record id"),
        ]
        for payload, rule in cases:
            with self.subTest(payload=payload):
                problems = payload_problems([payload])
                self.assertTrue(any(rule in p for p in problems), problems)
        with self.assertRaises(UnsafePayloadError) as ctx:
            assert_payload_safe([{"caption": "Insuffisance mitrale, demo-study-1"}], ["demo-study-1"])
        self.assertIn("$[0].caption: local identifier or path", str(ctx.exception))
        self.assertNotIn("demo-study-1", str(ctx.exception))  # errors never echo the value
        assert_payload_safe([{"caption": "Coupe A4C, Doppler couleur. FEVG 35 %. concepts_v0@0.1.0"}])

    def test_builder_refuses_unsafe_records(self):
        leaky = copy.deepcopy(self.records)
        leaky[0].candidate_caption.text += " demo-video-001"
        with self.assertRaises(UnsafePayloadError):
            build_data_rows(leaky, ONTOLOGY)
        test_split = copy.deepcopy(self.records)
        test_split[0].split = "test"
        with self.assertRaisesRegex(ValueError, "test split"):
            build_data_rows(test_split, ONTOLOGY)
        uncaptioned = read_records(DEMO_RECORDS, ONTOLOGY)
        with self.assertRaisesRegex(ValueError, "no candidate caption"):
            build_prelabels(uncaptioned)

    def test_pseudonymous_ids_are_checked_by_format_not_by_substring(self):
        # An HMAC id can contain a short local identifier by chance; that reveals nothing.
        key = "rc-0a1234567bcdef0123456789"
        url = PLACEHOLDER_URL_TEMPLATE.format(global_key=key)
        row = {"global_key": key, "row_data": url}
        self.assertEqual(payload_problems([row], ["1234567"], PLACEHOLDER_URL_TEMPLATE), [])
        # Anything else in the URL or key slot is still refused.
        for bad_key in ("rc-xyz", "1234567", "/media/x.mp4"):
            with self.subTest(key=bad_key):
                bad = {"global_key": bad_key, "row_data": PLACEHOLDER_URL_TEMPLATE.format(global_key=bad_key)}
                problems = payload_problems([bad], ["1234567"], PLACEHOLDER_URL_TEMPLATE)
                self.assertTrue(any("global_key" in p for p in problems))
                self.assertTrue(any("row_data" in p for p in problems))
        uid = "1234567a-0000-4000-8000-000000000000"
        self.assertEqual(payload_problems([{"uuid": uid}], ["1234567"]), [])
        self.assertTrue(payload_problems([{"uuid": "1234567"}], ["1234567"]))
        # The same identifier elsewhere in a payload is still caught.
        problems = payload_problems([{"global_key": key, "caption": "dossier 1234567"}], ["1234567"])
        self.assertTrue(any("caption: local identifier" in p for p in problems))

    def test_url_templates(self):
        validate_url_template("https://videos.approved.example/radar/{global_key}.mp4")
        for bad in ("http://x/{global_key}", "/media/videos/{global_key}.mp4", "https://x/radar.mp4",
                    "https://x/{global_key}/{other}", "https://x/2019-03-04/{global_key}"):
            with self.subTest(template=bad), self.assertRaises(ValueError):
                validate_url_template(bad)


class TestExport(unittest.TestCase):
    def setUp(self):
        self.records = captioned_demo()
        self.version = review_version(ONTOLOGY)

    def apply(self, rows, **kw):
        return apply_export(self.records, rows, ONTOLOGY, expected_review_version=self.version, **kw)

    def test_full_review_maps_onto_the_record(self):
        rec = self.records[1]  # A4C color: MR moderate, TR mild
        row = export_row(rec, [
            radio("caption_satisfactory", "no"),
            radio("clip_assessability", "assessable"),
            checklist("missing_visible_concepts", [{"value": "mitral_valve_visible"}]),
            checklist("unsupported_concepts", [{"value": "tricuspid_regurgitation"}]),
            checklist("wrong_value_concepts", [{"value": "mitral_regurgitation", "classifications": [
                text("mitral_regurgitation__corrected_value", "Sévère")]}]),
            checklist("uncertain_concepts", [{"value": "la_visible", "classifications": [
                radio("la_visible__uncertain_reason", "crop")]}]),
            text("corrected_caption",
                 "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : sévère."),
            text("comments", "Jet excentrique."),
        ])
        updated, counts = self.apply([row])
        self.assertEqual(counts, {"records": 9, "reviews": 1, "reviewed": 1, "in_review": 0, "approved_captions": 1})
        review = updated[1].review
        self.assertEqual((review.status, review.source, review.satisfied), ("reviewed", "labelbox_export", False))
        self.assertEqual(review.review_version, self.version)
        self.assertRegex(review.reviewer_id, r"^reviewer-[0-9a-f]{12}$")
        self.assertNotIn("cardio@example.org", json.dumps(updated[1].to_dict()))
        self.assertEqual(review.completed_utc, "2026-10-07T10:05:00+00:00")
        self.assertEqual(review.missing_concepts, ["mitral_valve_visible"])
        self.assertEqual(review.unsupported_concepts, ["tricuspid_regurgitation"])
        self.assertEqual([(w.concept_id, w.claimed_value, w.corrected_value) for w in review.wrong_values],
                         [("mitral_regurgitation", "moderate", "severe")])
        self.assertEqual([(u.concept_id, u.reason) for u in review.uncertain], [("la_visible", "crop")])
        approved = updated[1].approved_caption
        self.assertEqual(approved.derived_from, "corrected_caption")
        self.assertEqual([(c.concept_id, c.value) for c in approved.concepts],
                         [("mitral_regurgitation", "severe"), ("mitral_valve_visible", None)])
        self.assertIsNone(updated[0].approved_caption)  # untouched records stay unreviewed

    def test_status_reviewer_map_and_unassessable_clip(self):
        rec = self.records[0]
        row = export_row(rec, [radio("caption_satisfactory", "yes")], status="IN_REVIEW")
        updated, counts = self.apply([row], reviewer_map={"cardio@example.org": "cardiologist-a"})
        self.assertEqual((updated[0].review.status, updated[0].review.reviewer_id), ("in_review", "cardiologist-a"))
        self.assertIsNone(updated[0].approved_caption)
        with self.assertRaisesRegex(ExportError, "missing from the reviewer map"):
            self.apply([row], reviewer_map={})
        row = export_row(rec, [radio("caption_satisfactory", "yes"), radio("clip_assessability", "limited_quality")])
        updated, _ = self.apply([row])
        self.assertEqual(updated[0].review.clip_unassessable_reason, "quality")
        self.assertIsNone(updated[0].approved_caption)
        row = export_row(rec, [radio("caption_satisfactory", "yes"), radio("clip_assessability", "assessable")])
        updated, _ = self.apply([row])
        self.assertEqual(updated[0].approved_caption.derived_from, "candidate_accepted")

    def test_refusals(self):
        rec = self.records[0]
        ok = [radio("caption_satisfactory", "yes"), radio("clip_assessability", "assessable")]
        cases = [
            ([export_row(rec, ok + [radio("free_opinion", "x")])], "does not define"),
            ([export_row(rec, ok, labels=[{}, {}])], "2 labels"),
            ([export_row(rec, [radio("caption_satisfactory", "yes")])], "without the required answers"),
            ([export_row(rec, ok), export_row(rec, ok)], "appears twice"),
            ([export_row(rec, ok + [checklist("unsupported_concepts", [{"value": "lv_mood"}])])], "outside"),
            ([{"data_row": {"global_key": "rc-" + "0" * 24}, "projects": {}}], "not in the records"),
        ]
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ExportError, message):
                self.apply(rows)
        with self.assertRaisesRegex(ExportError, "review form"):
            apply_export(self.records, [], ONTOLOGY, expected_review_version="radar-review/0.0")


class TestCommandLine(unittest.TestCase):
    def test_dry_run_writes_nothing_and_outputs_stay_outside_the_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = os.path.join(tmp, "records.jsonl")
            write_records(records, captioned_demo(), ONTOLOGY)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(radar_review.main(["payload", "--records", records]), 0)
            self.assertIn("Dry run: nothing written", out.getvalue())
            self.assertEqual(sorted(os.listdir(tmp)), ["records.jsonl"])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(radar_review.main(["payload", "--records", records, "--out-dir",
                                                    os.path.join(tmp, "lb")]), 0)
            self.assertEqual(sorted(os.listdir(os.path.join(tmp, "lb"))), ["data_rows.ndjson", "prelabels.ndjson"])
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                radar_review.main(["payload", "--records", records, "--out-dir", os.path.join(REPO, "lb_out")])
            self.assertFalse(os.path.exists(os.path.join(REPO, "lb_out")))
            form = os.path.join(tmp, "form.json")
            with redirect_stdout(io.StringIO()):
                radar_review.main(["ontology", "--out", form])
            with open(form) as f:
                self.assertEqual(json.load(f)["review_version"], review_version(ONTOLOGY))


if __name__ == "__main__":
    unittest.main()
