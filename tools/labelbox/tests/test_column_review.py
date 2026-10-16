import copy
import hashlib
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from app.radar.ontology import VIEWS
from tools.labelbox.column_review import (
    FORM, allowed_columns, build_candidate, build_form, context_attachment,
    decode_annotations, encode_answers, finding_sentence, load_spec, prelabels, review_problems, score_review,
    validate_annotations,
)
from tools.labelbox.import_column_review import (
    ImportStopped, apply_bundle, comparable_shape, import_outcome, schema_shape, with_schema_ids,
)


def as_stored(form, name):
    """A form as Labelbox returns it: its ontology's name at the top, server ids, no empty option lists."""
    def strip(v, parent=""):
        if isinstance(v, list):
            return [strip(x, parent) for x in v]
        if isinstance(v, dict):
            key = v.get("name") or v.get("value")
            here = f"{parent}/{key}" if key else parent
            out = {k: strip(x, here) for k, x in v.items() if not (k == "options" and x == [])}
            if "schemaNodeId" in out:
                out["schemaNodeId"] = out["featureSchemaId"] = "id:" + here
            return out
        return v
    return {"name": name, **strip(form)}


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


class TestColumnPrefills(unittest.TestCase):
    def setUp(self):
        self.report = {"Left ventricule findings": "Dysfonction légère du ventricule gauche. // Ventricule gauche non-dilaté",
                       "Tricuspid valve findings": "Valve tricuspide normale",
                       "Visually Estimated EF": 48, "WMS Basal Inferoseptal": 2}

    def candidate(self, report=None, view="A4C", modality="bmode", **kwargs):
        return build_candidate(report if report is not None else self.report, view, modality,
                               global_key="synthetic-row", **kwargs)

    def test_twenty_views_and_sixteen_segments(self):
        spec = load_spec()
        self.assertEqual(set(spec["views"]), set(VIEWS))
        segments = [k for k in spec["fields"] if k.startswith("wall_")]
        self.assertEqual(len(segments), 16)
        self.assertFalse(any("cap" in k for k in segments))
        self.assertEqual(json.loads(FORM.read_text()), build_form())

    def test_explicit_normal_tricuspid_prefills_value_and_caption(self):
        for text in ["Valve tricuspide normale", "Normal tricuspid valve", "Tricuspid valve: normal", "Normal"]:
            c = self.candidate({**self.report, "Tricuspid valve findings": text})
            self.assertIn({"concept": "tricuspid_appearance", "value": "normal"},
                          [{k: col[k] for k in ("concept", "value")} for col in c["columns"]])
            self.assertIn("The visible tricuspid valve has a normal appearance.", c["candidate_caption"])
            self.assertFalse(any(x["concept"] == "tricuspid_motion" for x in c["columns"]))

    def test_missing_and_regurgitation_text_never_infer_normal_leaflets(self):
        for text in ["", "Insuffisance tricuspidienne légère", "Pas de régurgitation tricuspidienne", "Possible normal tricuspid valve"]:
            c = self.candidate({"Tricuspid valve findings": text})
            self.assertEqual(c["columns"], [])
            self.assertEqual(c["candidate_caption"], "")

    def test_normal_motion_does_not_establish_normal_appearance(self):
        for text in ["Tricuspid valve has normal motion", "Normal tricuspid valve motion", "Valve tricuspide: mobilité normale"]:
            c = self.candidate({"Tricuspid valve findings": text})
            self.assertFalse(any(x["concept"] == "tricuspid_appearance" for x in c["columns"]))
            self.assertEqual([x["concept"] for x in c["columns"]], ["tricuspid_motion"])

    def test_numbers_do_not_become_clinical_candidates(self):
        c = self.candidate({"Visually Estimated EF": 48, "Referring TR Grade": 1, "MHI VD fonction systolique": -1})
        self.assertEqual(c["columns"], [])
        self.assertNotIn("48", c["candidate_caption"])
        self.assertEqual(c["report_context"]["Visually Estimated EF"], 48)

    def test_text_negation_uncertainty_and_conflicts(self):
        for text in ["Pas de dysfonction modérée du ventricule gauche", "Possible mild dysfunction", "Dysfonction légère. // Dysfonction sévère."]:
            self.assertFalse(any(c["concept"] == "lv_function" for c in self.candidate({"Left ventricule findings": text})["columns"]))
        c = self.candidate({"Mitral valve findings": "Pas de sténose et insuffisance mitrale légère"}, modality="color_flow")
        self.assertFalse(any(x["concept"] == "mitral_regurgitant_jet" for x in c["columns"]))

    def test_actual_rv_codebook_and_conflict(self):
        self.assertEqual(self.candidate({"MHI VD fonction systolique": 0})["columns"][0]["value"], "normal")
        c = self.candidate({"MHI VD fonction systolique": 0, "Right ventricule findings": "Dysfonction sévère du ventricule droit"})
        self.assertFalse(any(x["concept"] == "rv_function" for x in c["columns"]))

    def test_view_modality_and_focused_target_routing(self):
        spec = load_spec()
        self.assertNotIn("aortic_appearance", allowed_columns("A4C", "bmode"))
        self.assertNotIn("tricuspid_regurgitant_jet", allowed_columns("A4C", "bmode"))
        self.assertIn("tricuspid_regurgitant_jet", allowed_columns("A4C", "color_flow"))
        self.assertEqual(self.candidate(view="OTHER")["columns"], [])
        self.assertFalse(any(c["concept"].startswith("tricuspid_") for c in self.candidate(view="A4C_LV")["columns"]))
        self.assertEqual(self.candidate(view="A4C_ZOOM")["columns"], [])
        c = self.candidate(view="A4C_ZOOM", targets=["tricuspid"])
        self.assertEqual([x["concept"] for x in c["columns"]], ["tricuspid_appearance"])

    def test_cusp_count_and_prolapse_are_view_restricted(self):
        report = {"Aortic valve findings": "Valve aortique tricuspide", "Mitral valve findings": "Prolapsus mitral"}
        self.assertEqual(self.candidate(report)["columns"], [])
        plax = self.candidate(report, view="PLAX")
        self.assertNotIn("aortic_cusp_count", [c["concept"] for c in plax["columns"]])
        self.assertIn("mitral_motion", [c["concept"] for c in plax["columns"]])
        av = self.candidate(report, view="PSAX_AV")
        self.assertEqual(av["columns"][0]["value"], "three")

    def test_wms_ambiguity_and_psax_level(self):
        report = {"WMS Basal Inferoseptal": 3, "WMS 4C Apical Lateral": 1, "WMS SAX Inferior": 2}
        c = self.candidate(report)
        self.assertEqual([col["value"] for col in c["columns"]], ["akinetic_or_severely_hypokinetic", "normal_or_hyperkinetic"])
        self.assertEqual(self.candidate(report, view="PSAX")["columns"], [])
        c = self.candidate(report, view="PSAX", level="mid")
        self.assertEqual(c["columns"][0]["concept"], "wall_mid_inferior")
        self.assertEqual(self.candidate(report, view="PSAX", level="basal")["columns"], [])

    def test_static_images_have_no_temporal_prefills(self):
        c = self.candidate(input_kind="image")
        self.assertNotIn("lv_function", [x["concept"] for x in c["columns"]])
        self.assertFalse(any(x["concept"].startswith("wall_") for x in c["columns"]))
        self.assertIn("tricuspid_appearance", [x["concept"] for x in c["columns"]])

    def test_subcostal_subtypes_do_not_mix_ivc_and_chambers(self):
        report = {"Inferior vena cava findings": "Dilatation de la veine cave inférieure", **self.report}
        self.assertEqual(self.candidate(report, view="SUBCOSTAL")["columns"], [])
        c = self.candidate(report, view="SUBCOSTAL", level="ivc", targets=["ivc"])
        self.assertEqual([x["concept"] for x in c["columns"]], ["ivc_size"])
        self.assertFalse(any(c["concept"] == "ivc_size" for c in self.candidate(report, view="SUBCOSTAL", level="four_chamber", targets=["lv"])["columns"]))


class TestColumnSubmissions(unittest.TestCase):
    def setUp(self):
        self.c = build_candidate({"Left ventricule findings": "Dysfonction légère du ventricule gauche",
                                  "Tricuspid valve findings": "Valve tricuspide normale"},
                                 "A4C", "bmode", global_key="synthetic-submission", input_provenance_verified=True)
        self.a = decode_annotations(prelabels(self.c))

    def score(self, **kwargs):
        return score_review(self.c, self.a, submitted=True, **kwargs)

    def test_imported_drafts_cannot_count_as_reviews(self):
        self.assertFalse(self.score()["submitted"])
        self.assertEqual(self.score()["caption"], "")
        self.a["review_completed"] = "yes"
        self.assertFalse(score_review(self.c, self.a)["submitted"])
        self.assertTrue(self.score()["ready_for_validation"])

    def test_rejection_drops_claim_without_inverting_it(self):
        self.a.update(review_completed="yes", tricuspid_appearance__status="no",
                      tricuspid_appearance__error="not_visible", tricuspid_appearance__correction="remove")
        score = self.score()
        self.assertEqual(len(score["rejected"]), 1)
        self.assertNotIn("tricuspid", score["caption"])
        self.assertNotIn("abnormal", score["caption"])

    def test_wrong_value_correction_and_original_score_are_separate(self):
        self.a.update(review_completed="yes", tricuspid_appearance__status="no",
                      tricuspid_appearance__error="wrong_value", tricuspid_appearance__correction="thickened")
        score = self.score()
        self.assertEqual(score["supported"], self.c["columns"][:1])
        self.assertEqual(score["corrections"][0]["value"], "thickened")
        self.assertIn("appears thickened", score["caption"])
        self.assertEqual(score["error_rate"], 0.5)

    def test_cannot_assess_is_neither_error_nor_negative(self):
        self.a.update(review_completed="yes", tricuspid_appearance__status="cannot_assess",
                      tricuspid_appearance__limitation="poor_quality")
        score = self.score()
        self.assertEqual(score["rejected"], [])
        self.assertEqual(score["errors"], [])
        self.assertNotIn("tricuspid", score["caption"])
        self.assertEqual(score["resolved_count"], 1)

    def test_missing_field_requires_typed_visible_value(self):
        self.a.update(review_completed="yes", missing_information="yes", missing_columns=["lv_size"],
                      lv_size__missing_value="dilated", lv_size__missing_visible="cannot_assess")
        self.assertEqual(self.score()["caption"], "")
        self.a["lv_size__missing_visible"] = "yes"
        score = self.score()
        self.assertEqual(score["omissions"][0]["value"], "dilated")
        self.assertIn("LV cavity is dilated", score["caption"])
        self.assertEqual(score["original_count"], 2)

    def test_unknown_or_existing_omission_and_mutated_original_are_refused(self):
        self.a.update(review_completed="yes", missing_information="yes", missing_columns=["lv_function"],
                      lv_function__missing_value="normal", lv_function__missing_visible="yes")
        with self.assertRaises(ValueError):
            self.score()
        self.a = decode_annotations(prelabels(self.c))
        self.a["lv_function__value"] = "normal"
        with self.assertRaises(ValueError):
            self.score()

    def test_view_modality_target_and_input_provenance_gates(self):
        self.a["review_completed"] = "yes"
        for change in [{"view_ok": "no"}, {"modality_ok": "cannot_assess"}, {"view": "A2C"}, {"visible_targets": ["lv"]}]:
            self.assertEqual(score_review(self.c, {**self.a, **change}, submitted=True)["caption"], "")
        self.assertFalse(score_review({**self.c, "input_provenance_verified": False}, self.a, submitted=True)["ready_for_validation"])

    def test_additional_unsupported_numeric_statement_is_scored_separately(self):
        self.a.update(review_completed="yes", unsupported_information="yes", unsupported_columns=["exact_ef"],
                      exact_ef__unsupported_kind="study_only", exact_ef__statement="EF is 48% in this clip")
        score = self.score()
        self.assertEqual(score["errors"], [{"concept": "exact_ef", "kind": "study_only"}])
        self.assertNotIn("48", score["caption"])

    def test_pending_details_block_caption(self):
        self.a.update(review_completed="yes", tricuspid_appearance__status="no",
                      tricuspid_appearance__correction="remove")
        self.assertTrue(self.score()["pending"])
        self.assertEqual(self.score()["caption"], "")

    def test_an_uninterpretable_clip_counts_in_no_rate(self):
        self.assertEqual(self.a["uninterpretable"], "no")
        self.a.update(review_completed="yes", uninterpretable="yes", uninterpretable_reason="poor_quality",
                      tricuspid_appearance__status="no")  # the other answers are not read
        score = self.score()
        self.assertTrue(score["uninterpretable"])
        self.assertEqual((score["pending"], score["caption"], score["support_rate"], score["error_rate"]),
                         ([], "", None, None))
        self.assertEqual((score["rejected"], score["errors"]), ([], []))
        self.a["uninterpretable_reason"] = "other"
        self.assertEqual(self.score()["pending"], ["uninterpretable_reason"])  # "other" needs a comment
        self.a["comments"] = "abdominal scan filed under the echo"
        self.assertEqual(self.score()["pending"], [])
        del self.a["uninterpretable_reason"]
        self.assertEqual(self.score()["pending"], ["uninterpretable_reason"])
        self.a = decode_annotations(prelabels(self.c))
        self.a.update(review_completed="yes", uninterpretable=None)
        self.assertIn("uninterpretable", self.score()["pending"])

    def test_review_problems_define_a_valid_review(self):
        done = prelabels(self.c) + [{"name": "review_completed", "answer": {"name": "yes"}}]

        def changed(records, **answers):
            out = copy.deepcopy(records)
            for name, answer in answers.items():
                next(r for r in out if r["name"] == name)["answer"] = answer
            return out

        self.assertEqual(review_problems(self.c, done), [])
        self.assertTrue(review_problems(self.c, prelabels(self.c)))  # not completed
        self.assertEqual(review_problems(self.c, changed(done, view_ok={"name": "no"})),
                         ["view_ok is no exactly when view is changed from the proposal"])
        self.assertEqual(review_problems(self.c, changed(done, view_ok={"name": "no"}, view={"name": "A5C"})), [])
        self.assertTrue(review_problems(self.c, changed(done, view={"name": "A5C"})))
        self.assertTrue(review_problems(self.c, changed(done, modality_ok={"name": "no"})))
        self.assertTrue(review_problems(self.c, changed(done, uninterpretable={"name": "yes"})))  # no reason
        reason = [{"name": "uninterpretable_reason", "answer": {"name": "not_cardiac"}}]
        uninterpretable = {"name": "yes", "classifications": reason}
        unread = changed(done, uninterpretable=uninterpretable, view_ok={"name": "no"})  # the view is not read
        self.assertEqual(review_problems(self.c, unread), [])
        self.assertEqual(review_problems(self.c, changed(done, review_columns=[])),
                         ["Original columns must be reviewed; additions go in missing_columns"])

    def test_encode_answers_inverts_decoding(self):
        records = prelabels(self.c) + [{"name": "review_completed", "answer": {"name": "yes"}}]
        answers = decode_annotations(records)
        self.assertEqual(decode_annotations(encode_answers(answers, build_form())), answers)
        answers.update(tricuspid_appearance__status="no", tricuspid_appearance__error="wrong_value",
                       tricuspid_appearance__correction="thickened", missing_information="yes",
                       missing_columns=["lv_size"], lv_size__missing_value="dilated", lv_size__missing_visible="yes")
        encoded = encode_answers(answers, build_form())
        validate_annotations(encoded, build_form(), require_complete=True)
        self.assertEqual(decode_annotations(encoded), answers)
        with self.assertRaises(ValueError):  # nested under an answer that is not selected
            encode_answers({**answers, "uninterpretable_reason": "poor_quality"}, build_form())
        with self.assertRaises(ValueError):
            encode_answers({**answers, "view_ok": "maybe"}, build_form())

    def test_nested_ndjson_path_validation_and_required_confirmation(self):
        p = prelabels(self.c)
        validate_annotations(p, build_form())
        with self.assertRaises(ValueError):
            validate_annotations(p, build_form(), require_complete=True)
        p.append({"name": "review_completed", "answer": {"name": "yes"}})
        validate_annotations(p, build_form(), require_complete=True)
        validate_annotations(p + [{"name": "comments", "answer": "jet partly outside the color box"}], build_form(),
                             require_complete=True)
        self.assertNotIn("comments", decode_annotations(prelabels(self.c)))
        wrong = copy.deepcopy(p)
        next(r for r in wrong if r["name"] == "review_columns")["answer"][0]["classifications"][0]["name"] = "view_ok"
        with self.assertRaises(ValueError):
            validate_annotations(wrong, build_form())

    def test_prefills_and_context_are_native_and_stable(self):
        self.assertEqual(prelabels(self.c), prelabels(self.c))
        self.assertNotIn("review_completed", decode_annotations(prelabels(self.c)))
        attachment = context_attachment(self.c)
        self.assertEqual(attachment["attachment_type"], "RAW_TEXT")
        self.assertRegex(attachment["attachment_name"], r"^[A-Za-z0-9 ]+$")  # Labelbox refused an em dash.
        self.assertIn("tricuspid_appearance", attachment["attachment_value"])
        with self.assertRaises(ValueError):
            context_attachment({**self.c, "report_context": {"MRN": "not-allowed"}})

    def test_attachment_uploads_no_report_text(self):
        report = {"Left ventricule findings":
                  "Dysfonction légère du ventricule gauche. // Ventricule gauche non-dilaté",
                  "Tricuspid valve findings": "Valve tricuspide normale", "Mitral valve findings": "Feuillets fins",
                  "Visually Estimated EF": 48.0, "LVID Diastole (2D)": "5.1", "WMS Basal Inferoseptal": 2}
        c = build_candidate(report, "A4C", "bmode", global_key="synthetic-text")
        self.assertTrue(any(col.get("source_text") for col in c["columns"]))  # kept in the local candidate only.
        value = context_attachment(c)["attachment_value"]
        for text in ("Dysfonction", "non-dilaté", "Valve tricuspide", "Feuillets"):
            self.assertNotIn(text, value)
        self.assertIn("tricuspid_appearance | Normal | Tricuspid valve findings", value)
        self.assertIn("Visually Estimated EF: 48", value)
        self.assertIn("LVID Diastole (2D): 5.1", value)
        self.assertIn("WMS Basal Inferoseptal: 2", value)

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
    def test_sdk_form_roundtrip(self):
        from labelbox import OntologyBuilder
        self.assertEqual(OntologyBuilder.from_dict(build_form()).asdict(), build_form())


@unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
class TestV2Import(unittest.TestCase):
    def setUp(self):
        import labelbox as lb
        self.candidates = [build_candidate({"Tricuspid valve findings": "Normal"}, "A4C", "bmode", global_key="synthetic-import")]
        self.ontology = SimpleNamespace(name="DeepECHO column review v2", uid="ontology",
                                        media_type=lb.MediaType.Video,
                                        normalized=as_stored(build_form(), "DeepECHO column review v2"))
        self.row = SimpleNamespace(media_attributes={"frameCount": 24}, _attachments=[])
        self.row.attachments = lambda: self.row._attachments
        self.row.create_attachment = lambda **kw: self.row._attachments.append(SimpleNamespace(**kw))
        self.project = SimpleNamespace(media_type=lb.MediaType.Video, _ontology=None, _jobs=[], _batches=[], data_row_count=0)
        self.project.ontology = lambda: self.project._ontology
        self.project.connect_ontology = lambda ontology: setattr(self.project, "_ontology", ontology)
        self.project.get_mal_prediction_imports = lambda: self.project._jobs
        self.project.batches = lambda: self.project._batches
        self.project.create_batch = lambda **kw: self.project._batches.append(SimpleNamespace(**kw))
        self.client = SimpleNamespace(get_project=lambda _: self.project, get_data_row_by_global_key=lambda _: self.row,
                                      get_ontologies=lambda _: [], create_ontology=lambda *a, **k: self.ontology)
        def create_job(**kw):
            job = SimpleNamespace(name=kw["name"], uid="job", errors=[], wait_till_done=lambda: None,
                                  predictions=kw["predictions"])
            self.project._jobs.append(job)
            return job
        self.mal = SimpleNamespace(create_from_objects=create_job)

    def test_import_connects_ontology_attaches_context_queues_rows_and_reuses_job(self):
        first = apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        second = apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        self.assertEqual(first, second)
        self.assertEqual(len(self.project._jobs), 1)
        self.assertEqual(len(self.project._batches), 1)
        self.assertEqual(len(self.row._attachments), 1)

    def earlier_job(self, name, errors, state="FINISHED"):
        """An import job of an earlier run; `errors` None when they can no longer be read (after 24 hours)."""
        def read_errors(job):
            if errors is None:
                raise RuntimeError("error file expired")
            return errors

        job = type("Job", (), {"errors": property(read_errors)})()
        job.name, job.uid, job.state, job.wait_till_done = name, "earlier", SimpleNamespace(name=state), lambda: None
        self.project._jobs.append(job)
        return job

    def batch_name(self):
        payloads = [p for c in self.candidates for p in prelabels(c)]
        return "deepecho-v2-" + hashlib.sha256(json.dumps(payloads, sort_keys=True).encode()).hexdigest()[:16]

    def test_a_failed_import_of_named_predictions_is_superseded(self):
        # The run before schema ids: its batch and context are in, its job (same prelabels, by name) failed.
        self.project._batches.append(SimpleNamespace(name=self.batch_name()))
        self.earlier_job(self.batch_name(), errors=[{"message": "unknown feature"}])
        result = apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        self.assertEqual(result["job_id"], "job")
        self.assertEqual([j.name for j in self.project._jobs][1:], [self.batch_name() + "-f2"])
        self.assertEqual(len(self.project._batches), 1)
        self.assertFalse([node for node in walk(self.project._jobs[1].predictions) if "name" in node])

    def test_a_failed_import_of_these_predictions_is_retried_only_when_asked(self):
        self.earlier_job(self.batch_name() + "-f2", errors=None, state="FAILED")
        with self.assertRaises(ImportStopped):
            apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        self.assertEqual((len(self.project._jobs), self.row._attachments, self.project._ontology), (1, [], None))
        apply_bundle(self.client, "project", self.candidates, mal_import=self.mal, retry_failed=True)
        self.assertEqual(self.project._jobs[-1].name, self.batch_name() + "-f2-r2")

    def test_an_import_whose_errors_can_no_longer_be_read_stops(self):
        # Had it succeeded, its prelabels would be in Labelbox already: a person checks before any new import.
        self.earlier_job(self.batch_name(), errors=None)
        with self.assertRaises(ImportStopped):
            apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        self.assertEqual(len(self.project._jobs), 1)

    def test_a_clean_import_of_an_earlier_form_is_not_repeated(self):
        self.earlier_job(self.batch_name(), errors=[])
        self.assertEqual(apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)["job_id"],
                         "earlier")
        self.assertEqual(len(self.project._jobs), 1)

    def test_prelabels_are_sent_with_schema_ids(self):
        apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        (job,) = self.project._jobs
        self.assertFalse([node for node in walk(job.predictions) if "name" in node])
        ids = {node["schemaId"] for node in walk(job.predictions) if "schemaId" in node}
        self.assertIn("id:/uninterpretable", ids)
        self.assertIn("id:/review_columns/tricuspid_appearance/tricuspid_appearance__status/yes", ids)
        self.assertNotIn("id:/review_completed", ids)
        self.assertEqual({p["dataRow"]["globalKey"] for p in job.predictions}, {"synthetic-import"})

    def test_names_the_ontology_lacks_are_refused(self):
        classifications = as_stored(build_form(), "x")["classifications"]
        records = prelabels(self.candidates[0])
        self.assertEqual(with_schema_ids([dict(records[0], name="visible_targets", answer=[])], classifications), [])
        with self.assertRaises(ValueError):
            with_schema_ids([dict(records[0], name="not_a_question")], classifications)
        with self.assertRaises(ValueError):
            with_schema_ids([dict(records[0], name="view_ok", answer={"name": "maybe"})], classifications)

    def test_conflicting_schema_or_nonvideo_row_stops_before_mutation(self):
        self.project._ontology = SimpleNamespace(normalized={"tools": [], "classifications": [{"name": "old"}]})
        with self.assertRaises(ValueError):
            apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        self.assertEqual(self.row._attachments, [])
        self.project._ontology = None
        self.row.media_attributes = {"width": 640, "height": 480}
        with self.assertRaises(ValueError):
            apply_bundle(self.client, "project", self.candidates, mal_import=self.mal)
        self.assertIsNone(self.project._ontology)

    def test_server_schema_ids_do_not_change_semantics(self):
        modified = copy.deepcopy(build_form())
        modified["classifications"][0]["schemaNodeId"] = "server-id"
        self.assertEqual(schema_shape(modified), schema_shape(build_form()))
        stored = as_stored(build_form(), "DeepECHO column review v2")
        self.assertEqual(comparable_shape(stored), comparable_shape(build_form()))
        stored["classifications"][0]["options"][0]["label"] = "Changed"
        self.assertNotEqual(comparable_shape(stored), comparable_shape(build_form()))


@unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
class TestV2ImportWithTheSdk(unittest.TestCase):
    """The SDK's own classes, offline: its checks before an import is sent and the outcome of its import jobs."""

    def test_the_sdk_takes_the_predictions(self):
        import labelbox as lb
        report = {"Tricuspid valve findings": "Normal", "Mitral valve findings": "Mild regurgitation"}
        candidate = build_candidate(report, "A4C", "color_flow", global_key="synthetic-import")
        predictions = with_schema_ids(prelabels(candidate), as_stored(build_form(), "v2")["classifications"])
        sent = lb.MALPredictionImport._get_ndjson_from_objects(predictions, "annotations")
        self.assertEqual(len(sent.splitlines()), len(predictions))

    def test_each_outcome_of_an_sdk_import_job(self):
        import labelbox as lb
        import requests
        from labelbox.schema import annotation_import

        failure = json.dumps({"uuid": "u", "dataRow": {"globalKey": "g"}, "status": "FAILURE",
                              "errors": [{"name": "x", "message": "m"}]})
        cases = {"clean": ("FINISHED", 200, "", "done"), "errors": ("FINISHED", 200, failure, "failed"),
                 "failed": ("FAILED", 200, "", "failed"), "expired": ("FINISHED", 403, "", "unreadable")}
        for uid, (state, status, text, expected) in cases.items():
            # Each job its own id: the SDK caches a job's error file per job.
            job = lb.MALPredictionImport(None, {"id": uid, "name": "deepecho-v2-0-f2", "state": state,
                                                "progress": "100%", "inputFileUrl": None, "statusFileUrl": None,
                                                "errorFileUrl": f"https://example.invalid/{uid}",
                                                "project": {"id": "project"}})
            response = mock.Mock(status_code=status, text=text)
            response.raise_for_status.side_effect = requests.HTTPError(str(status)) if status >= 400 else None
            with self.subTest(uid), mock.patch.object(annotation_import.requests, "get", return_value=response):
                self.assertEqual(import_outcome(job), expected)


if __name__ == "__main__":
    unittest.main()
