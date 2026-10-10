import copy
import importlib.util
import json
import unittest
from types import SimpleNamespace

from app.radar.ontology import VIEWS
from tools.labelbox.column_review import (
    FORM, allowed_columns, build_candidate, build_form, context_attachment,
    decode_annotations, finding_sentence, load_spec, prelabels, score_review, validate_annotations,
)
from tools.labelbox.import_column_review import apply_bundle, schema_shape


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

    def test_nested_ndjson_path_validation_and_required_confirmation(self):
        p = prelabels(self.c)
        validate_annotations(p, build_form())
        with self.assertRaises(ValueError):
            validate_annotations(p, build_form(), require_complete=True)
        p.append({"name": "review_completed", "answer": {"name": "yes"}})
        validate_annotations(p, build_form(), require_complete=True)
        wrong = copy.deepcopy(p)
        wrong[6]["answer"][0]["classifications"][0]["name"] = "view_ok"
        with self.assertRaises(ValueError):
            validate_annotations(wrong, build_form())

    def test_prefills_and_context_are_native_and_stable(self):
        self.assertEqual(prelabels(self.c), prelabels(self.c))
        self.assertNotIn("review_completed", decode_annotations(prelabels(self.c)))
        attachment = context_attachment(self.c)
        self.assertEqual(attachment["attachment_type"], "RAW_TEXT")
        self.assertIn("tricuspid_appearance", attachment["attachment_value"])
        with self.assertRaises(ValueError):
            context_attachment({**self.c, "report_context": {"MRN": "not-allowed"}})

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
    def test_sdk_form_roundtrip(self):
        from labelbox import OntologyBuilder
        self.assertEqual(OntologyBuilder.from_dict(build_form()).asdict(), build_form())


@unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
class TestV2Import(unittest.TestCase):
    def setUp(self):
        import labelbox as lb
        self.candidates = [build_candidate({"Tricuspid valve findings": "Normal"}, "A4C", "bmode", global_key="synthetic-import")]
        self.ontology = SimpleNamespace(name="DeepECHO column review v2", normalized=build_form(), uid="ontology", media_type=lb.MediaType.Video)
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
            job = SimpleNamespace(name=kw["name"], uid="job", errors=[], wait_until_done=lambda: None)
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


if __name__ == "__main__":
    unittest.main()
