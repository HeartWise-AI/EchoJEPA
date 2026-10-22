import copy
import importlib.util
import json
import unittest
from types import SimpleNamespace

from app.radar.ontology import VIEWS, load_ontology
from tools.labelbox.column_review import (
    FORM, ROOT, SPEC, allowed_columns, build_candidate, build_form, context_attachment,
    decode_annotations, encode_answers, finding_sentence, load_spec, prelabels, review_problems, score_review,
    validate_annotations, view_targets,
)
from tools.labelbox.import_column_review import ONTOLOGY_NAME, apply_bundle, schema_shape


def concepts(candidate):
    return [c["concept"] for c in candidate["columns"]]


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

    def test_review_selection_names_sources_and_generated_artifact(self):
        selection = json.loads((ROOT / "configs/radar/labelbox/current_review.json").read_text())
        self.assertEqual(selection["schema"], "radar-review-selection")
        self.assertEqual(selection["version"], 1)
        self.assertEqual(
            selection["clinical_ontology"],
            "configs/radar/radar-caption-ontology-v1.yaml",
        )
        self.assertEqual(selection["source_specification"], SPEC.relative_to(ROOT).as_posix())
        self.assertEqual(
            selection["generated_labelbox_ontology"], FORM.relative_to(ROOT).as_posix()
        )
        self.assertEqual(
            selection["instructions"], "docs/radar/labeling_instructions_v1.md"
        )
        self.assertNotIn("ontology", selection)
        self.assertNotIn("spec", selection)

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

    def test_view_modality_and_structure_routing(self):
        self.assertNotIn("aortic_appearance", allowed_columns("A4C", "bmode"))
        self.assertNotIn("tricuspid_regurgitant_jet", allowed_columns("A4C", "bmode"))
        self.assertIn("tricuspid_regurgitant_jet", allowed_columns("A4C", "color_flow"))
        self.assertEqual(self.candidate(view="OTHER")["columns"], [])
        self.assertIn("lv_size", concepts(self.candidate()))
        # LV-focused: the LV and the view's valves, with LV function and wall motion but no chamber size.
        lv_focused = self.candidate(view="A4C_LV")
        self.assertEqual(lv_focused["suggested_targets"], ["lv", "mitral", "tricuspid"])
        self.assertEqual(concepts(lv_focused), ["lv_function", "tricuspid_appearance", "wall_basal_inferoseptal"])
        # Zoomed: the base view's structures are prefilled, and every finding but chamber size can be.
        zoomed = self.candidate(view="A4C_ZOOM")
        self.assertEqual(zoomed["suggested_targets"], self.candidate()["suggested_targets"])
        self.assertEqual(concepts(zoomed), ["lv_function", "tricuspid_appearance", "wall_basal_inferoseptal"])
        self.assertEqual(concepts(self.candidate(view="A4C_ZOOM", targets=["tricuspid"])), ["tricuspid_appearance"])

    def test_each_view_prefills_the_structures_of_the_audit(self):
        spec = load_spec()
        a4c = ["lv", "rv", "la", "ra", "ivs", "ias", "mitral", "tricuspid", "pericardium"]
        plax = ["lv", "rv", "la", "ivs", "mitral", "aortic", "lvot", "aorta_root", "aorta_ascending", "pericardium"]
        self.assertEqual(spec["views"]["A4C"]["targets"], a4c)
        self.assertEqual(spec["views"]["A5C"]["targets"], a4c + ["aortic", "lvot", "aorta_root"])
        self.assertEqual(spec["views"]["PLAX"]["targets"], plax)
        self.assertEqual(spec["views"]["PLAX_DEEP"]["targets"], plax + ["aorta_descending"])
        self.assertEqual(spec["views"]["PLAX_DEEP"]["columns"], ["pericardial_effusion"])
        self.assertEqual(set(spec["views"]["PSAX_AV"]["targets"]),
                         {"aortic", "rv", "rvot", "ra", "la", "tricuspid", "pulmonic", "pulmonary_artery", "ias"})
        self.assertEqual(spec["views"]["A3C"]["targets"][-3:], ["lvot", "aorta_root", "pericardium"])
        self.assertEqual(spec["views"]["A2C"]["targets"], ["lv", "la", "mitral", "pericardium"])
        for view, route in spec["views"].items():
            levels = route.get("levels", {})
            reachable = set(route["targets"]) | {t for level in levels.values() for t in level["targets"]}
            self.assertLessEqual(reachable, set(spec["targets"]), view)
            for key in route["columns"]:  # every finding of a view is about one of its structures
                self.assertIn(spec["fields"][key]["target"], reachable, f"{view} {key}")
            if view.endswith(("_ZOOM", "_LV")):  # the whole chamber may not be in the image
                self.assertFalse({"lv_size", "rv_size", "la_size", "ra_size"} & set(route["columns"]), view)
                self.assertIn("lv_function", route["columns"], view)
            if view.endswith("_ZOOM"):
                self.assertEqual(route["targets"], spec["views"][view.removesuffix("_ZOOM")]["targets"], view)

    def test_the_structures_are_the_ontology_s(self):
        """One structure list: each form structure is an ontology structure whose visibility prior names exactly the
        views that prefill it, the PSAX and subcostal levels included."""
        spec, ontology = load_spec(), load_ontology()
        links = spec["target_structures"]
        self.assertEqual(list(links), list(spec["targets"]))
        self.assertEqual(sorted(links.values()), sorted(ontology.structures))
        for target, structure in links.items():
            views = {view for view, route in spec["views"].items()
                     if target in route["targets"] + [t for level in route.get("levels", {}).values()
                                                      for t in level["targets"]]}
            concept = ontology.concept(ontology.visibility_concept(structure))
            self.assertEqual(set(concept.prior_views), views, target)
            self.assertLessEqual({(view, "bmode") for view in views} | {(view, "color_flow") for view in views},
                                 set(concept.visible_in), target)

    def test_findings_of_structures_the_audit_added(self):
        for view, key in [("PSAX_AV", "rv_size"), ("PSAX", "rv_size"), ("PLAX", "rv_size"), ("PSAX_AV", "la_size"),
                          ("PSAX_AV", "ra_size"), ("A3C", "aorta_root_size"), ("A5C", "aorta_root_size"),
                          ("A5C", "tricuspid_appearance")]:
            self.assertIn(key, allowed_columns(view, "bmode"), f"{view} {key}")
        for view in ("A4C", "PSAX_AV", "PLAX"):  # the new structures have no findings yet
            self.assertTrue(set(view_targets(view, None)) - {load_spec()["fields"][k]["target"]
                                                             for k in allowed_columns(view, "bmode")})
        self.assertEqual(finding_sentence("aortic_regurgitant_jet", "demonstrated"),
                         "An aortic regurgitant jet is demonstrated in this color acquisition.")
        self.assertEqual(finding_sentence("mitral_regurgitant_jet", "demonstrated"),
                         "A mitral regurgitant jet is demonstrated in this color acquisition.")

    def test_cusp_count_and_prolapse_are_view_restricted(self):
        report = {"Aortic valve findings": "Valve aortique tricuspide", "Mitral valve findings": "Prolapsus mitral"}
        self.assertEqual(self.candidate(report)["columns"], [])
        plax = self.candidate(report, view="PLAX")
        self.assertNotIn("aortic_cusp_count", [c["concept"] for c in plax["columns"]])
        self.assertIn("mitral_motion", [c["concept"] for c in plax["columns"]])
        av = self.candidate(report, view="PSAX_AV")
        self.assertEqual(av["columns"][0]["value"], "three")

    def test_wms_ambiguity_and_psax_level(self):
        report = {"WMS Basal Inferoseptal": 3, "WMS 4C Apical Lateral": 1, "WMS SAX Inferior": 2,
                  "Mitral valve findings": "Valve mitrale calcifiée"}
        walls = [col["value"] for col in self.candidate(report)["columns"] if col["concept"].startswith("wall_")]
        self.assertEqual(walls, ["akinetic_or_severely_hypokinetic", "normal_or_hyperkinetic"])
        # The view classifier gives no PSAX level: only the structures of every level, and the reviewer states it.
        psax = self.candidate(report, view="PSAX")
        self.assertIsNone(psax["level"])
        self.assertEqual(psax["suggested_targets"], ["lv", "rv", "ivs", "pericardium"])
        self.assertEqual(psax["columns"], [])
        self.assertNotIn("psax_level", decode_annotations(prelabels(psax)))
        self.assertEqual(concepts(self.candidate(report, view="PSAX", level="mid")),
                         ["mitral_appearance", "wall_mid_inferior"])
        self.assertEqual(concepts(self.candidate(report, view="PSAX", level="basal_mitral")), ["mitral_appearance"])
        self.assertEqual(self.candidate(report, view="PSAX", level="apical")["columns"], [])
        self.assertEqual(self.candidate(report, view="PSAX", level="mid")["suggested_targets"][-2:],
                         ["papillary_muscles", "chordae"])
        for view, level in [("PSAX", "basal"), ("PSAX", "not_applicable"), ("A4C", "mid")]:
            with self.assertRaises(ValueError):
                self.candidate(report, view=view, level=level)

    def test_static_images_have_no_temporal_prefills(self):
        c = self.candidate(input_kind="image")
        self.assertNotIn("lv_function", [x["concept"] for x in c["columns"]])
        self.assertFalse(any(x["concept"].startswith("wall_") for x in c["columns"]))
        self.assertIn("tricuspid_appearance", [x["concept"] for x in c["columns"]])

    def test_subcostal_subtypes_do_not_mix_ivc_and_chambers(self):
        report = {"Inferior vena cava findings": "Dilatation de la veine cave inférieure", **self.report}
        four_chamber = self.candidate(report, view="SUBCOSTAL")  # the default subtype
        self.assertEqual(four_chamber["level"], "four_chamber")
        self.assertEqual(concepts(four_chamber), ["lv_function", "lv_size", "tricuspid_appearance"])
        ivc = self.candidate(report, view="SUBCOSTAL", level="ivc")
        self.assertEqual(ivc["suggested_targets"], ["ivc", "hepatic_veins"])
        self.assertEqual(concepts(ivc), ["ivc_size"])
        self.assertEqual(self.candidate(report, view="SUBCOSTAL", level="unresolved")["columns"], [])


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
        for change in [{"view": "A2C"}, {"modality": "color_flow"}, {"visible_targets": ["lv"]}]:
            self.assertEqual(score_review(self.c, {**self.a, **change}, submitted=True)["caption"], "")
        corrected = score_review(self.c, {**self.a, "view": "A5C"}, submitted=True)
        self.assertFalse(corrected["view_correct"])
        self.assertTrue(corrected["modality_correct"])
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
        self.assertEqual(review_problems(self.c, changed(done, view={"name": "A5C"})), [])
        self.assertEqual(review_problems(self.c, changed(done, modality={"name": "color_flow"})), [])
        self.assertTrue(review_problems(self.c, changed(done, uninterpretable={"name": "yes"})))  # no reason
        reason = [{"name": "uninterpretable_reason", "answer": {"name": "not_cardiac"}}]
        uninterpretable = {"name": "yes", "classifications": reason}
        unread = changed(done, uninterpretable=uninterpretable)  # the other answers are not read
        self.assertEqual(review_problems(self.c, unread), [])
        self.assertEqual(review_problems(self.c, changed(done, review_columns=[])),
                         ["A prefilled finding was unticked: mark it Incorrect and remove it instead"])
        self.assertEqual(review_problems(self.c, changed(done, psax_level={"name": "mid"})), ["psax_level"])

    def test_a_ticked_finding_that_was_not_prefilled_is_an_addition(self):
        self.a.update(review_completed="yes", review_columns=[*self.a["review_columns"], "lv_size"],
                      lv_size__value="dilated", lv_size__status="yes")
        score = self.score()
        self.assertEqual(score["omissions"],
                         [{"concept": "lv_size", "value": "dilated", "origin": "reviewer_addition"}])
        self.assertEqual((score["pending"], score["original_count"]), ([], 2))
        self.assertIn("The visible LV cavity is dilated.", score["caption"])
        self.a["lv_size__status"] = "cannot_assess"
        self.assertEqual(self.score()["pending"], ["lv_size__status"])
        self.a.update(lv_size__status="yes", missing_information="yes", missing_columns=["lv_size"],
                      lv_size__missing_value="dilated", lv_size__missing_visible="yes")
        self.assertIn("missing_columns", self.score()["pending"])  # added twice

    def test_the_level_fits_the_confirmed_view(self):
        psax = build_candidate({"Left ventricule findings": "Dysfonction légère du ventricule gauche"},
                               "PSAX", "bmode", global_key="synthetic-psax")
        answers = {**decode_annotations(prelabels(psax)), "review_completed": "yes"}
        self.assertEqual(score_review(psax, answers, submitted=True)["pending"], ["psax_level"])
        for level, pending in [("not_applicable", ["psax_level"]), ("unresolved", []), ("mid", [])]:
            self.assertEqual(score_review(psax, {**answers, "psax_level": level}, submitted=True)["pending"], pending)
        # The mitral valve is claimed at the mid level too; a level's structures come with it, unticked.
        mid = {**answers, "psax_level": "mid", "missing_information": "yes",
               "missing_columns": ["mitral_motion", "wall_mid_anterior"],
               "mitral_motion__missing_value": "normal", "mitral_motion__missing_visible": "yes",
               "wall_mid_anterior__missing_value": "normal", "wall_mid_anterior__missing_visible": "yes"}
        score = score_review(psax, mid, submitted=True)
        self.assertTrue(score["routing"])
        self.assertIn("mitral leaflets show normal motion", score["caption"])
        self.assertFalse(score_review(psax, {**mid, "psax_level": "apical"}, submitted=True)["routing"])
        self.assertNotIn("mitral", mid["visible_targets"])
        # A view corrected away from PSAX has no level.
        corrected = {**answers, "view": "PLAX", "psax_level": "apical"}
        self.assertEqual(score_review(psax, corrected, submitted=True)["pending"], ["psax_level"])

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
            encode_answers({**answers, "view": "maybe"}, build_form())

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
        self.assertNotIn("view_ok", decode_annotations(prelabels(self.c)))
        self.assertNotIn("modality_ok", decode_annotations(prelabels(self.c)))
        labels = {q["name"]: q["instructions"] for q in build_form()["classifications"]}
        self.assertEqual(labels["view"], "View (prefilled - correct if needed)")
        self.assertEqual(labels["modality"], "Acquisition mode (prefilled - correct if needed)")
        wrong = copy.deepcopy(p)
        next(r for r in wrong if r["name"] == "review_columns")["answer"][0]["classifications"][0]["name"] = "view"
        with self.assertRaises(ValueError):
            validate_annotations(wrong, build_form())

    def test_prefills_and_context_are_native_and_stable(self):
        self.assertEqual(prelabels(self.c), prelabels(self.c))
        revised_spec = copy.deepcopy(load_spec())
        revised_spec["modalities"]["bmode"] = "B-mode revised"
        self.assertNotEqual(
            {record["uuid"] for record in prelabels(self.c)},
            {record["uuid"] for record in prelabels(self.c, revised_spec)},
        )
        self.assertNotIn("review_completed", decode_annotations(prelabels(self.c)))
        attachment = context_attachment(self.c)
        self.assertEqual(attachment["attachment_type"], "RAW_TEXT")
        self.assertEqual(attachment["attachment_name"], "DeepECHO review v3 - original proposals")
        self.assertTrue(attachment["attachment_name"].isascii())
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

    def test_an_ivc_sweep_brings_its_structures_and_prefilled_ones_need_their_tick(self):
        report = {"Inferior vena cava findings": "Dilatation de la veine cave inférieure",
                  "Right ventricule findings": "Ventricule droit non dilaté"}
        subcostal = build_candidate(report, "SUBCOSTAL", "bmode", global_key="synthetic-subcostal")
        self.assertEqual(concepts(subcostal), ["rv_size"])
        answers = {**decode_annotations(prelabels(subcostal)), "review_completed": "yes"}
        self.assertTrue(score_review(subcostal, answers, submitted=True)["caption"])
        unticked = {**answers, "visible_targets": [t for t in answers["visible_targets"] if t != "rv"]}
        self.assertFalse(score_review(subcostal, unticked, submitted=True)["routing"])
        ivc = {**answers, "psax_level": "ivc", "rv_size__status": "no", "rv_size__error": "wrong_view",
               "rv_size__correction": "remove", "missing_information": "yes", "missing_columns": ["ivc_size"],
               "ivc_size__missing_value": "dilated", "ivc_size__missing_visible": "yes"}
        score = score_review(subcostal, ivc, submitted=True)
        self.assertEqual((score["pending"], score["routing"]), ([], True))
        self.assertEqual(score["caption"], "IVC size: dilated in this input.")

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
    def test_sdk_form_roundtrip(self):
        from labelbox import OntologyBuilder
        self.assertEqual(OntologyBuilder.from_dict(build_form()).asdict(), build_form())


@unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
class TestImport(unittest.TestCase):
    def setUp(self):
        import labelbox as lb
        self.candidates = [build_candidate({"Tricuspid valve findings": "Normal"}, "A4C", "bmode", global_key="synthetic-import")]
        self.assertEqual(ONTOLOGY_NAME, "DeepECHO column review v3")
        self.ontology = SimpleNamespace(name=ONTOLOGY_NAME, normalized=build_form(), uid="ontology",
                                        media_type=lb.MediaType.Video)
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
