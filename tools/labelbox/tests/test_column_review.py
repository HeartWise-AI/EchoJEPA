import copy
import hashlib
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from app.radar.ontology import VIEWS, load_ontology
from tools.labelbox.column_review import (
    FORM, ROOT, SPEC, _normalize, _text_value, allowed_columns, build_candidate, build_form, context_attachment,
    decode_annotations, encode_answers, finding_sentence, load_spec, prelabels, review_problems, score_review,
    validate_annotations, view_targets,
)
from tools.labelbox.import_column_review import (
    ONTOLOGY_NAME, ImportStopped, apply_bundle, comparable_shape, import_outcome, schema_shape, with_schema_ids,
)


def concepts(candidate):
    return [c["concept"] for c in candidate["columns"]]


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

    def test_the_classes_their_refinements_and_sixteen_segments(self):
        spec = load_spec()
        refinements = {view: route["refines"] for view, route in spec["views"].items() if "refines" in route}
        self.assertEqual(refinements, {"PSAX_BASAL": "PSAX", "PSAX_MID": "PSAX", "PSAX_APICAL": "PSAX",
                                       "PLAX_ZOOM_AV": "PLAX_ZOOM"})
        self.assertEqual(set(spec["views"]) - set(refinements), set(VIEWS))
        psax = spec["views"]["PSAX"]  # the class prefills only what every level claims.
        for view in ("PSAX_BASAL", "PSAX_MID", "PSAX_APICAL"):
            self.assertLessEqual(set(psax["targets"]), set(spec["views"][view]["targets"]), view)
            self.assertLessEqual(set(psax["columns"]), set(spec["views"][view]["columns"]), view)
        zoom, on_the_valve = spec["views"]["PLAX_ZOOM"], spec["views"]["PLAX_ZOOM_AV"]
        self.assertLessEqual(set(on_the_valve["targets"]), set(zoom["targets"]))
        self.assertLessEqual(set(on_the_valve["columns"]), set(zoom["columns"]))
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
            "configs/radar/radar-caption-ontology-v2.yaml",
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
            self.assertIn("Tricuspid valve has a normal appearance.", c["candidate_caption"])
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

    def value_of(self, key, column, text, modality="bmode", view="A4C"):
        found = [c["value"] for c in self.candidate({column: text}, view=view, modality=modality)["columns"] if c["concept"] == key]
        return found[0] if found else None

    def test_diastolic_dysfunction_is_not_a_reduced_systolic_function(self):
        for text, value in [("Dysfonction diastolique légère", None), ("Mild diastolic dysfunction", None),
                            ("Dysfonction diastolique de grade 1, fonction systolique normale", "normal"),
                            ("Dysfonction diastolique légère et dysfonction systolique modérée", "moderately_reduced"),
                            ("Fonction diastolique normale", None), ("Dysfonction légère du ventricule gauche", "mildly_reduced")]:
            self.assertEqual(self.value_of("lv_function", "Left ventricule findings", text), value, text)

    def test_a_negation_negates_only_the_finding_it_names(self):
        for text, value in [("Sans dilatation ou HVG", "normal"), ("Sans hypertrophie ni dilatation", "normal"),
                            ("Pas de HVG, VG dilaté", "dilated"), ("VG non dilaté mais dilatation apicale", None)]:
            self.assertEqual(self.value_of("lv_size", "Left ventricule findings", text), value, text)
        for text, value in [("Insuffisance mitrale modérée sans prolapsus", "demonstrated"),
                            ("Insuffisance mitrale légère et aucune sténose", "demonstrated"),
                            ("Pas d'insuffisance mitrale", "not_demonstrated"), ("No significant mitral regurgitation", "not_demonstrated"),
                            ("Mitral regurgitation not seen", "not_demonstrated"), ("Insuffisance mitrale : aucune", "not_demonstrated")]:
            self.assertEqual(self.value_of("mitral_regurgitant_jet", "Mitral valve findings", text, modality="color_flow"), value, text)
        self.assertIsNone(self.value_of("mitral_regurgitant_jet", "Mitral valve findings", "Insuffisance mitrale modérée sans prolapsus"))
        for text, value in [("Épanchement péricardique minime sans tamponnade", "present"), ("Pas d'épanchement péricardique", "not_demonstrated")]:
            self.assertEqual(_text_value("pericardial_effusion", _normalize(text)), value, text)

    def test_each_aortic_segment_takes_its_own_part_of_the_clause(self):
        text = _normalize("Racine aortique dilatée, aorte ascendante normale")
        self.assertEqual(_text_value("aorta_root_size", text), "dilated")
        self.assertEqual(_text_value("aorta_ascending_size", text), "normal")
        self.assertEqual(_text_value("aorta_root_size", _normalize("Racine aortique et aorte ascendante dilatées")), "dilated")

    def test_abnormal_is_not_normal(self):
        for key, text in [("aorta_root_size", "Racine aortique anormale"), ("aorta_ascending_size", "Abnormal ascending aorta"),
                          ("mitral_appearance", "Feuillets anormaux"), ("mitral_appearance", "Leaflets abnormal"),
                          ("mitral_appearance", "Abnormal leaflets"), ("aortic_appearance", "Cusps abnormal"),
                          ("mitral_motion", "Mobilité anormale"), ("mitral_motion", "Abnormal leaflet motion"),
                          ("mitral_motion", "Opening abnormal"), ("ivc_collapse", "Collapsus respiratoire anormal"),
                          ("lv_function", "Systolic function abnormal"), ("lv_size", "Abnormal size")]:
            self.assertIsNone(_text_value(key, _normalize(text)), text)
        self.assertEqual(_text_value("aorta_root_size", _normalize("Racine aortique normale")), "normal")
        self.assertEqual(_text_value("mitral_motion", _normalize("Mobilité normale")), "normal")

    def test_normal_wordings_of_function_and_size(self):
        for text, function, size in [("FEVG normale", "normal", None), ("FEVG estimée normale à 60 %", "normal", None),
                                     ("De dimension, de masse et de fonction normales", "normal", "normal"),
                                     ("Normal size and function", "normal", "normal"), ("VG de taille normale", None, "normal"),
                                     ("FEVG 45 % (normale > 55 %)", None, None)]:
            self.assertEqual(self.value_of("lv_function", "Left ventricule findings", text), function, text)
            self.assertEqual(self.value_of("lv_size", "Left ventricule findings", text), size, text)
        self.assertIsNone(_text_value("rv_function", _normalize("FEVG normale")))

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
        self.assertEqual(lv_focused["suggested_targets"], ["lv", "mitral", "tricuspid", "ivs"])
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
        self.assertEqual(spec["views"]["PLAX_DEEP"]["columns"], spec["views"]["PLAX"]["columns"])
        self.assertEqual(set(spec["views"]["PSAX_AV"]["targets"]),
                         {"aortic", "rv", "rvot", "ra", "la", "tricuspid", "pulmonic", "pulmonary_artery", "ias"})
        self.assertEqual(spec["views"]["A3C"]["targets"][-4:], ["lvot", "aorta_root", "pericardium", "ivs"])
        self.assertEqual(spec["views"]["A2C"]["targets"], ["lv", "la", "mitral", "pericardium"])
        for view, route in spec["views"].items():
            levels = route.get("levels", {})
            reachable = set(route["targets"]) | {t for level in levels.values() for t in level["targets"]}
            self.assertLessEqual(reachable, set(spec["targets"]), view)
            for key in route["columns"]:  # every finding of a view is about one of its structures
                self.assertIn(spec["fields"][key]["target"], reachable, f"{view} {key}")
            if view.endswith(("_ZOOM", "_LV")):  # the whole chamber may not be in the image
                self.assertFalse({"lv_size", "rv_size", "la_size", "ra_size"} & set(route["columns"]), view)
                if view != "PLAX_ZOOM":  # no LV there: the medical review's rule.
                    self.assertIn("lv_function", route["columns"], view)
            if view.endswith("_ZOOM"):
                not_judged = {"lv", "rv", "pericardium"} if view == "PLAX_ZOOM" else set()
                base = spec["views"][view.removesuffix("_ZOOM")]["targets"]
                self.assertEqual(route["targets"], [t for t in base if t not in not_judged], view)

    def test_the_structures_are_the_ontology_s(self):
        """One structure list: each form structure is an ontology structure whose visibility prior names exactly the
        views that prefill it, a refinement counted as its class and the subcostal levels included."""
        spec, ontology = load_spec(), load_ontology()
        links = spec["target_structures"]
        self.assertEqual(list(links), list(spec["targets"]))
        self.assertEqual(sorted(links.values()), sorted(ontology.structures))
        for target, structure in links.items():
            views = {route.get("refines", view) for view, route in spec["views"].items()
                     if target in route["targets"] + [t for level in route.get("levels", {}).values()
                                                      for t in level["targets"]]}
            concept = ontology.concept(ontology.visibility_concept(structure))
            self.assertEqual(set(concept.prior_views), views, target)
            self.assertLessEqual({(view, "bmode") for view in views} | {(view, "color_flow") for view in views},
                                 set(concept.visible_in), target)

    def test_findings_of_structures_the_audit_added(self):
        for view, key in [("PSAX_AV", "rv_size"), ("PLAX", "rv_size"), ("PSAX_AV", "la_size"),
                          ("PSAX_AV", "ra_size"), ("A3C", "aorta_root_size"), ("A5C", "aorta_root_size"),
                          ("A5C", "tricuspid_appearance")]:
            self.assertIn(key, allowed_columns(view, "bmode"), f"{view} {key}")
        for view in ("A4C", "PSAX_AV", "PLAX"):  # the new structures have no findings yet
            self.assertTrue(set(view_targets(view, None)) - {load_spec()["fields"][k]["target"]
                                                             for k in allowed_columns(view, "bmode")})
        self.assertEqual(finding_sentence("aortic_regurgitant_jet", "demonstrated"),
                         "Aortic regurgitant jet is visible on color Doppler.")
        self.assertEqual(finding_sentence("mitral_regurgitant_jet", "demonstrated"),
                         "Mitral regurgitant jet is visible on color Doppler.")

    def test_the_medical_review_s_view_rules(self):
        spec = load_spec()
        family = ("PSAX", "PSAX_BASAL", "PSAX_MID", "PSAX_APICAL")
        for view in family:  # no chamber size in a short-axis clip.
            self.assertFalse({"lv_size", "rv_size", "rv_function"} & set(spec["views"][view]["columns"]), view)
            self.assertNotIn("rv", spec["views"][view]["targets"], view)
        for view in ("PSAX", "PSAX_MID", "PSAX_APICAL"):
            self.assertNotIn("lv_wall_thickness", spec["views"][view]["columns"], view)
        self.assertIn("lv_wall_thickness", spec["views"]["PSAX_BASAL"]["columns"])
        self.assertLessEqual({"rv", "rvot"}, set(spec["views"]["PSAX_AV"]["targets"]))  # this level keeps them.
        self.assertIn("rv_size", spec["views"]["PSAX_AV"]["columns"])
        for view in ("PLAX_ZOOM", "PLAX_ZOOM_AV"):
            route = spec["views"][view]
            self.assertFalse({"lv", "rv", "pericardium"} & set(route["targets"]), view)
            judged = {spec["fields"][k]["target"] for k in route["columns"]}
            self.assertFalse(judged & {"lv", "rv", "pericardium"}, view)
        self.assertIn("lv_size", spec["views"]["PLAX"]["columns"])  # the unzoomed view keeps them.
        self.assertIn("lv_function", spec["views"]["A4C_ZOOM"]["columns"])  # and so do the other zooms.

    def test_wall_thickness_tells_the_septum_from_the_whole_wall(self):
        septum = "Augmentation légère de l'épaisseur du septum"
        posterior = "Augmentation légère de l'épaisseur de la paroi postérieure"
        measured = {"LVIW Diastolic Thickness (2D)": 0.95}

        cases = [(septum, measured, "septal_thickening"), (f"{septum} // {posterior}", measured, None),
                 (posterior, measured, None), ("Épaississement des parois du VG", {}, "increased"),
                 (septum, {}, "septal_thickening"), (septum, {"LVIW Diastolic Thickness (2D)": float("nan")}, "septal_thickening"),
                 (septum, {"LVIW Diastolic Thickness (2D)": 0}, "septal_thickening"),
                 ("Épaisseur pariétale normale", {}, "not_increased"),
                 ("Pas d'épaississement des parois", {}, None),
                 (f"Épaisseur pariétale normale // {septum}", measured, None),
                 ("Présence d'un bourrelet septal mesurant 16 mm d'épaisseur, avec masse VG normale", {}, None),
                 ("Épaississement des parois apicales", {}, None)]
        for text, numbers, expected in cases:
            report = {"Ventricular septum findings": text, **numbers}
            got = {c["concept"]: c["value"] for c in self.candidate(report)["columns"]}.get("lv_wall_thickness")
            self.assertEqual(got, expected, text)
        report = {"Ventricular septum findings": septum, **measured}
        for view, expected in [("PLAX", "septal_thickening"), ("A4C", "septal_thickening"), ("A5C_ZOOM", "septal_thickening"),
                               ("PSAX_BASAL", "septal_thickening"), ("A2C", None), ("A2C_LV", None), ("A3C", "septal_thickening"),
                               ("A3C_LV", "septal_thickening"), ("A4C_LV", "septal_thickening"), ("SUBCOSTAL", None)]:  # only where the septum is.
            got = {c["concept"]: c["value"] for c in self.candidate(report, view=view)["columns"]}
            self.assertEqual(got.get("lv_wall_thickness"), expected, view)
        back = {"Ventricular septum findings": posterior}  # says nothing of the other walls: no measurement needed.
        for view, expected in [("PLAX", "posterior_thickening"), ("PLAX_DEEP", "posterior_thickening"), ("A3C", "posterior_thickening"),
                               ("A3C_LV", "posterior_thickening"), ("PSAX_BASAL", "posterior_thickening"), ("A4C", None),
                               ("A2C", None), ("A5C", None)]:
            got = {c["concept"]: c["value"] for c in self.candidate(back, view=view)["columns"]}
            self.assertEqual(got.get("lv_wall_thickness"), expected, view)
        both = {"Ventricular septum findings": f"{septum} // {posterior}", **measured}
        self.assertEqual(concepts(self.candidate(both, view="A2C")), [])  # two named walls do not establish all walls.

    def wall(self, text, measured=0.95, **columns):
        report = {"Ventricular septum findings": text, "LVIW Diastolic Thickness (2D)": measured, **columns}
        columns = self.candidate(report, view="PLAX")["columns"]
        return {c["concept"]: c["value"] for c in columns}.get("lv_wall_thickness")

    def test_normal_wall_thickness_takes_every_wall(self):
        for text in ("Épaisseur pariétale normale", "Parois d'épaisseur normale", "Normal wall thickness",
                     "Parois du VG d'épaisseur normale", "Épaisseur normale des parois ventriculaires gauches"):
            self.assertEqual(self.wall(text), "not_increased", text)
        for text in ("Épaisseur du septum à 12 mm avec fraction d'éjection normale",  # another "normale".
                     "Épaisseur maximale de 15 mm et augmentation sévère de la masse, fonction normale",
                     "Épaisseur des parois à la limite supérieure de la normale",  # borderline.
                     "La paroi postérieure est d'épaisseur normale", "Septum d'épaisseur normale",  # one wall.
                     "Parois d'épaisseur normale à la base mais 13 mm en moyen",
                     "Autres parois d'épaisseur normale",
                     "Hypertrophie concentrique légère avec parois d'épaisseur normale",
                     "Paroi septale d'épaisseur normale, mais léger épaississement de la paroi latérale"):
            self.assertIsNone(self.wall(text), text)

    def test_a_later_normal_word_does_not_undo_a_thick_septum(self):
        for text in ("Augmentation légère de l'épaisseur du septum, masse normale",
                     "Augmentation modérée de l'épaisseur du septum, fraction d'éjection normale"):
            self.assertEqual(self.wall(text), "septal_thickening", text)

    def test_a_whole_wall_thickest_at_the_septum_is_not_the_septum_alone(self):
        symmetric = "Augmentation symétrique de l'épaisseur pariétale jusqu'à 14 mm en antéro-septal"
        self.assertEqual(self.wall(symmetric), "increased")
        self.assertEqual(self.wall("Augmentation sévère et diffuse de l'épaisseur du septum"), "septal_thickening")
        for text in ("Augmentation de l'épaisseur pariétale jusqu'à 14 mm en antéro-septal basal",
                     "Augmentation asymétrique de l'épaisseur pariétale en antéro-septal basal",
                     "Augmentation sévère de la masse du ventricule gauche avec épaisseur septale de 15 mm"):
            self.assertIsNone(self.wall(text), text)
        self.assertEqual(self.wall("Épaississement de la paroi septale"), "septal_thickening")  # one wall, named.

    def test_a_negated_wall_statement_says_nothing(self):
        for text in ("Sans augmentation de l'épaisseur des parois", "Pas d'augmentation de l'épaisseur du septum",
                     "Absence d'épaississement des parois", "No increased wall thickness"):
            self.assertIsNone(self.wall(text), text)

    def test_concentric_hypertrophy_rules_out_normal_elsewhere(self):
        septum = "Augmentation légère de l'épaisseur du septum"
        for lv, expected in [("Hypertrophie ventriculaire gauche concentrique légère", "septal_thickening"),
                             ("Augmentation légère de la masse du ventricule gauche (HVG concentrique)", "septal_thickening"),
                             ("Pas d'hypertrophie concentrique", "septal_thickening"),
                             ("Remodelage concentrique du ventricule gauche", "septal_thickening")]:
            self.assertEqual(self.wall(septum, **{"Left ventricule findings": lv}), expected, lv)
        hypertrophy = {"Left ventricule findings": "Hypertrophie concentrique"}
        self.assertIsNone(self.wall("Épaisseur pariétale normale", **hypertrophy))
        self.assertEqual(self.wall(f"{septum} // Augmentation légère de l'épaisseur de la paroi postérieure",
                                   **hypertrophy), None)  # no whole-wall inference from two named walls.
        self.assertEqual(self.wall("Augmentation légère de l'épaisseur de la paroi postérieure", **hypertrophy),
                         "posterior_thickening")  # claims nothing of the other walls.

    def test_each_valve_offers_the_devices_it_can_carry(self):
        spec = load_spec()
        values = {key: set(spec["fields"][key]["values"]) for key in spec["fields"] if key.endswith("_device")}
        both = {"bioprosthesis", "mechanical_prosthesis"}
        self.assertEqual(values["mitral_device"], both | {"edge_to_edge_clip", "annuloplasty_ring"})
        self.assertEqual(values["tricuspid_device"], both | {"edge_to_edge_clip", "annuloplasty_ring"})
        self.assertEqual(values["aortic_device"], both | {"annuloplasty_ring"})  # Yacoub, Ross: in real reports.
        self.assertEqual(values["pulmonic_device"], both)
        clip = {"Aortic valve findings": "MitraClip sur la valve aortique"}
        self.assertEqual(self.candidate(clip, view="PLAX")["columns"], [])  # refused, not captioned.
        self.assertTrue(self.candidate({"Mitral valve findings": "MitraClip en place"}, view="PLAX")["columns"])

    def test_a_septal_bulge_is_its_own_finding(self):
        def found(text, view="PLAX", **numbers):
            columns = self.candidate({"Left ventricule findings": text, **numbers}, view=view)["columns"]
            return {c["concept"]: c["value"] for c in columns}

        bulge = "Présence d'un bourrelet septal basal non-obstructif mesurant 14 mm d'épaisseur"
        self.assertEqual(found(bulge), {"septal_bulge": "present"})  # and no wall thickness.
        for view in ("A4C", "A4C_LV", "A3C", "A3C_LV", "A3C_ZOOM", "A5C", "PLAX_DEEP", "PSAX_BASAL"):
            self.assertEqual(found(bulge, view), {"septal_bulge": "present"}, view)
        for view in ("A2C", "PSAX_MID", "PLAX_ZOOM"):
            self.assertEqual(found(bulge, view), {}, view)
        self.assertEqual(found("Bourrelet septal de 13 mm"), {"septal_bulge": "present"})
        self.assertEqual(found("Pas d'hypertrophie ventriculaire gauche (bourrelet septal 13mm)"),
                         {"septal_bulge": "present"})
        for text in ("Pas de bourrelet septal", "Absence de bourrelet septal", "Bourrelet sous aortique de 14mm",
                     "Pas d'hypertrophie ventriculaire gauche ni de bourrelet septal",
                     "Sans hypertrophie ni bourrelet septal", "Il n'y a plus de bourrelet septal"):
            self.assertEqual(found(text), {}, text)
        both = found(f"{bulge} // Augmentation légère de l'épaisseur du septum",
                     **{"LVIW Diastolic Thickness (2D)": 0.95})
        self.assertEqual(both, {"lv_wall_thickness": "septal_thickening", "septal_bulge": "present"})

    def test_every_finding_has_its_own_sentence(self):
        spec = load_spec()
        for key, field in spec["fields"].items():
            for value in field["values"]:
                sentence = finding_sentence(key, value, spec)
                self.assertRegex(sentence, r"^[A-Z][^:]*\.$", key)  # no "label: value", the form's wording.
                self.assertNotIn(field["label"].lower() + ":", sentence.lower(), key)
                self.assertNotIn("_", sentence, key)
                self.assertNotRegex(sentence, r"(?i)\bthe visible\b", key)
        # The reviewer's own words.
        self.assertEqual(finding_sentence("aortic_cusp_count", "three"), "Aortic valve has three cusps.")
        self.assertEqual(finding_sentence("lv_wall_thickness", "increased"), "The visualized LV walls are thickened.")
        self.assertEqual(finding_sentence("lv_wall_thickness", "septal_thickening"),
                         "Septal thickening is present.")
        self.assertEqual(finding_sentence("lv_wall_thickness", "posterior_thickening"), "Thickened LV posterior wall.")
        self.assertEqual(finding_sentence("septal_bulge", "present"), "Focal basal septal thickening (septal bulge).")
        self.assertEqual(finding_sentence("aorta_root_size", "normal"), "Aortic root size is not dilated.")
        self.assertEqual(finding_sentence("wall_basal_inferolateral", "hypokinetic"),
                         "Basal inferolateral segment shows hypokinesis.")
        self.assertEqual(finding_sentence("pericardial_effusion", "not_demonstrated"), "No pericardial effusion is visualized in this clip.")
        self.assertEqual(finding_sentence("lv_function", "normal"), "LV systolic function appears normal.")
        self.assertEqual(finding_sentence("lv_wall_thickness", "not_increased"), "The visualized LV walls are not thickened.")
        self.assertEqual(finding_sentence("pericardial_effusion", "present"), "Pericardial effusion is visualized in this clip.")
        # A jet is what one color clip shows: its absence there is not the absence of regurgitation.
        self.assertEqual(finding_sentence("mitral_regurgitant_jet", "not_demonstrated"),
                         "No mitral regurgitant jet is seen on color Doppler.")
        self.assertEqual(finding_sentence("aortic_device", "mechanical_prosthesis"),
                         "Mechanical aortic valve prosthesis is present.")
        self.assertEqual(finding_sentence("mitral_device", "bioprosthesis"), "Bioprosthetic mitral valve is present.")
        self.assertEqual(finding_sentence("mitral_device", "edge_to_edge_clip"),
                         "Mitral edge-to-edge clip is present.")
        self.assertEqual(finding_sentence("mitral_motion", "prolapse"), "Mitral valve prolapse is present.")
        self.assertEqual(finding_sentence("tricuspid_motion", "flail"), "Flail tricuspid leaflet is present.")
        self.assertEqual(finding_sentence("mitral_motion", "tethered"), "Mitral leaflets show tethered motion.")
        spec["fields"]["new_finding"] = {"label": "New finding", "values": {"yes": "Yes"}}
        with self.assertRaises(ValueError):  # a column without a sentence, not a caption from its form labels.
            finding_sentence("new_finding", "yes", spec)

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
        # The classifier gives no PSAX level: only the structures of every level, and the reviewer states the view.
        psax = self.candidate(report, view="PSAX")
        self.assertEqual(psax["level"], "not_applicable")
        self.assertEqual(psax["suggested_targets"], ["lv", "ivs", "pericardium"])
        self.assertEqual(psax["columns"], [])
        self.assertNotIn("view", decode_annotations(prelabels(psax)))
        self.assertEqual(decode_annotations(prelabels(self.candidate(report, view="PSAX_MID")))["view"], "PSAX_MID")
        self.assertEqual(concepts(self.candidate(report, view="PSAX_MID")), ["wall_mid_inferior"])
        self.assertEqual(concepts(self.candidate(report, view="PSAX_BASAL")), ["mitral_appearance"])
        self.assertEqual(self.candidate(report, view="PSAX_APICAL")["columns"], [])
        self.assertEqual(self.candidate(report, view="PSAX_MID")["suggested_targets"][-2:],
                         ["papillary_muscles", "chordae"])
        for view, level in [("PSAX", "mid"), ("PSAX_MID", "ivc"), ("A4C", "four_chamber")]:
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
                next(r for r in walk(out) if r.get("name") == name and "answer" in r)["answer"] = answer
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
        self.assertEqual(review_problems(self.c, changed(done, subcostal_level={"name": "ivc"})), ["subcostal_level"])

    def test_a_ticked_finding_that_was_not_prefilled_is_an_addition(self):
        self.a.update(review_completed="yes", review_columns=[*self.a["review_columns"], "lv_size"],
                      lv_size__value="dilated", lv_size__status="yes")
        score = self.score()
        self.assertEqual(score["omissions"],
                         [{"concept": "lv_size", "value": "dilated", "origin": "reviewer_addition"}])
        self.assertEqual((score["pending"], score["original_count"]), ([], 2))
        self.assertIn("LV cavity is dilated.", score["caption"])
        self.a["lv_size__status"] = "cannot_assess"
        self.assertEqual(self.score()["pending"], ["lv_size__status"])
        self.a.update(lv_size__status="yes", missing_information="yes", missing_columns=["lv_size"],
                      lv_size__missing_value="dilated", lv_size__missing_visible="yes")
        self.assertIn("missing_columns", self.score()["pending"])  # added twice

    def test_a_psax_clip_s_view_states_its_level(self):
        psax = build_candidate({"Left ventricule findings": "Dysfonction légère du ventricule gauche"},
                               "PSAX", "bmode", global_key="synthetic-psax")
        answers = {**decode_annotations(prelabels(psax)), "review_completed": "yes"}
        self.assertNotIn("view", answers)
        records = prelabels(psax) + [{"name": "review_completed", "answer": {"name": "yes"}}]
        self.assertEqual(review_problems(psax, records), ["Required review classification missing"])
        for view in ("PSAX", "PSAX_BASAL", "PSAX_MID", "PSAX_APICAL"):  # PSAX: the level cannot be told.
            score = score_review(psax, {**answers, "view": view}, submitted=True)
            self.assertEqual((score["pending"], score["view_correct"], score["routing"]), ([], True, True), view)
            self.assertEqual((score["out_of_view"], score["support_rate"]), ([], 1.0), view)
        at_the_valve = score_review(psax, {**answers, "view": "PSAX_AV"}, submitted=True)
        self.assertEqual((at_the_valve["view_correct"], at_the_valve["routing"]), (False, False))  # no refinement.
        # A fixed papillary-level clip cannot support mitral leaflet findings.
        mid = {**answers, "view": "PSAX_MID", "missing_information": "yes",
               "missing_columns": ["mitral_motion", "wall_mid_anterior"],
               "mitral_motion__missing_value": "normal", "mitral_motion__missing_visible": "yes",
               "wall_mid_anterior__missing_value": "normal", "wall_mid_anterior__missing_visible": "yes"}
        score = score_review(psax, mid, submitted=True)
        self.assertFalse(score["routing"])
        self.assertIn("mitral_motion__not_in_view", score["pending"])
        self.assertEqual(score["caption"], "")
        apical = score_review(psax, {**mid, "view": "PSAX_APICAL"}, submitted=True)
        self.assertEqual((apical["pending"], apical["routing"], apical["caption"]),
                         (["mitral_motion__not_in_view", "wall_mid_anterior__not_in_view"], False, ""))
        self.assertNotIn("mitral", mid["visible_targets"])
        # A value is judged in the view the reviewer chose, not the class the clip came with.
        basal = {**answers, "view": "PSAX_BASAL", "missing_information": "yes",
                 "missing_columns": ["lv_wall_thickness"], "lv_wall_thickness__missing_visible": "yes"}
        for value, sentence in [("posterior_thickening", "Thickened LV posterior wall."),
                                ("septal_thickening", "Septal thickening is present.")]:
            score = score_review(psax, {**basal, "lv_wall_thickness__missing_value": value}, submitted=True)
            self.assertEqual((score["pending"], score["routing"]), ([], True), value)
            self.assertIn(sentence, score["caption"])
        # A view that is not subcostal has no subtype.
        corrected = {**answers, "view": "PLAX", "subcostal_level": "ivc"}
        self.assertEqual(score_review(psax, corrected, submitted=True)["pending"], ["subcostal_level"])

    def test_an_added_finding_the_view_does_not_carry_is_sent_back(self):
        psax = build_candidate({"Left ventricule findings": "Dysfonction légère du ventricule gauche"},
                               "PSAX", "bmode", global_key="synthetic-psax")
        answers = {**decode_annotations(prelabels(psax)), "review_completed": "yes"}
        wall = {"lv_wall_thickness__missing_value": "increased", "lv_wall_thickness__missing_visible": "yes",
                "missing_information": "yes", "missing_columns": ["lv_wall_thickness"]}
        carried = score_review(psax, {**answers, **wall, "view": "PSAX_BASAL"}, submitted=True)
        self.assertEqual(carried["pending"], [])
        self.assertIn("The visualized LV walls are thickened.", carried["caption"])
        ticked = {"review_columns": [*answers["review_columns"], "lv_wall_thickness"],
                  "lv_wall_thickness__value": "increased", "lv_wall_thickness__status": "yes"}
        structure = {**wall, "view": "PSAX_BASAL", "visible_targets": ["ivs", "pericardium"]}  # the LV is unticked.
        for sent_back in ({**wall, "view": "PSAX_APICAL"}, {**ticked, "view": "PSAX_APICAL"}, structure):
            score = score_review(psax, {**answers, **sent_back}, submitted=True)
            self.assertIn("lv_wall_thickness__not_in_view", score["pending"])
            self.assertEqual(score["caption"], "")
        self.assertEqual(score_review(psax, {**answers, "view": "PSAX_APICAL"}, submitted=True)["pending"], [])

    def test_a_claim_on_an_unticked_structure_goes_back_to_the_reviewer(self):
        clip = build_candidate({"Left ventricule findings": "Dysfonction légère du ventricule gauche",
                                "Pericardium findings": "Pas d'épanchement péricardique"},
                               "A4C", "bmode", global_key="synthetic-row")
        self.assertEqual(concepts(clip), ["lv_function", "pericardial_effusion"])
        answers = {**decode_annotations(prelabels(clip)), "review_completed": "yes"}
        no_lv = {"visible_targets": [t for t in answers["visible_targets"] if t != "lv"]}
        incorrect = {"lv_function__status": "no", "lv_function__error": "not_visible"}
        removed = {**incorrect, "lv_function__correction": "remove"}

        def score(*changes):
            merged = dict(answers)
            for change in changes:
                merged.update(change)
            return score_review(clip, merged, submitted=True, spec=load_spec())

        both = "LV systolic function appears mildly reduced. No pericardial effusion is visualized in this clip."
        kept = score()  # ticked, supported.
        self.assertEqual((kept["pending"], kept["caption"], kept["warnings"]), ([], both, []))
        self.assertEqual((score(removed)["pending"], score(removed)["caption"]), ([], "No pericardial effusion is visualized in this clip."))
        left = score(no_lv)  # unticked, supported: neither answer is changed for the reviewer.
        self.assertEqual((left["pending"], left["caption"]), (["lv_function__structure_unticked"], ""))
        self.assertIn("lv_function", [c["concept"] for c in left["supported"]])
        warning = ('LV is marked as not visible, but "LV function" is still claimed. Mark the finding Incorrect and '
                   'remove the claim, or tick the structure again.')
        self.assertEqual(left["warnings"], [warning])
        resolved = score(no_lv, removed)  # unticked, incorrect.
        self.assertEqual((resolved["pending"], resolved["warnings"]), ([], []))
        self.assertEqual(resolved["caption"], "No pericardial effusion is visualized in this clip.")
        # A corrected value needs the structure too.
        value = score(no_lv, incorrect, {"lv_function__correction": "moderately_reduced"})
        self.assertEqual(value["pending"], ["lv_function__structure_unticked"])
        unsure = score(no_lv, {"lv_function__status": "cannot_assess", "lv_function__limitation": "target_outside"})
        self.assertEqual(unsure["pending"], [])
        corrected = score({"view": "PLAX_ZOOM"})  # nothing unticked: a corrected view is not this conflict.
        self.assertEqual((corrected["pending"], corrected["caption"]), ([], ""))

    def test_narrowing_a_zoom_to_the_aortic_valve_leaves_out_what_it_does_not_claim(self):
        report = {"Left ventricule findings": "Dysfonction modérée du ventricule gauche",  # no LV in a zoomed PLAX.
                  "Mitral valve findings": "Valve mitrale calcifiée",
                  "Aortic valve findings": "Valve aortique calcifiée",
                  "Aorta findings": "Pas de dilatation de l'aorte au niveau des sinus de Valsalva"}
        zoom = build_candidate(report, "PLAX_ZOOM", "bmode", global_key="synthetic-zoom")
        self.assertEqual(concepts(zoom), ["mitral_appearance", "aortic_appearance", "aorta_root_size"])
        answers = {**decode_annotations(prelabels(zoom)), "review_completed": "yes"}
        self.assertEqual(answers["view"], "PLAX_ZOOM")
        self.assertEqual(score_review(zoom, answers, submitted=True)["out_of_view"], [])
        narrowed = {**answers, "view": "PLAX_ZOOM_AV"}
        score = score_review(zoom, narrowed, submitted=True)
        self.assertEqual((score["pending"], score["view_correct"], score["routing"]), ([], True, True))
        self.assertEqual([c["concept"] for c in score["out_of_view"]], ["mitral_appearance"])
        self.assertEqual((score["original_count"], score["resolved_count"], score["support_rate"]), (2, 2, 1.0))
        self.assertEqual(score["caption"], "Aortic valve appears calcified. Aortic root size is not dilated.")
        # Unticking a finding the view leaves out changes nothing; unticking one it claims is still refused.
        kept = [key for key in narrowed["review_columns"] if key != "mitral_appearance"]
        self.assertEqual(score_review(zoom, {**narrowed, "review_columns": kept}, submitted=True)["caption"],
                         score["caption"])
        with self.assertRaises(ValueError):
            score_review(zoom, {**narrowed, "review_columns": ["mitral_appearance"]}, submitted=True)
        # A view the class does not refine is a wrong view, whatever it claims.
        wrong = score_review(zoom, {**answers, "view": "PSAX_AV"}, submitted=True)
        self.assertEqual((wrong["out_of_view"], wrong["view_correct"], wrong["routing"]), ([], False, False))

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
        labels = {q["name"]: q["instructions"] for q in walk(build_form()) if "instructions" in q}
        self.assertEqual(labels["view"], "View (prefilled - correct if needed)")
        self.assertEqual(labels["modality"], "Acquisition mode (prefilled - correct if needed)")
        wrong = copy.deepcopy(p)
        next(r for r in walk(wrong) if r.get("name") == "review_columns")["answer"][0]["classifications"][0]["name"] = "view"
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
        self.assertEqual(attachment["attachment_name"], "DeepECHO review v5 - original proposals")
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
        ivc = {**answers, "subcostal_level": "ivc", "rv_size__status": "no", "rv_size__error": "wrong_view",
               "rv_size__correction": "remove", "missing_information": "yes", "missing_columns": ["ivc_size"],
               "ivc_size__missing_value": "dilated", "ivc_size__missing_visible": "yes",
               "visible_targets": ["ivc", "hepatic_veins"]}
        score = score_review(subcostal, ivc, submitted=True)
        self.assertEqual((score["pending"], score["routing"]), ([], True))
        self.assertEqual(score["caption"], "IVC is dilated.")

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
    def test_sdk_form_roundtrip(self):
        from labelbox import OntologyBuilder
        self.assertEqual(OntologyBuilder.from_dict(build_form()).asdict(), build_form())


@unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK optional")
class TestImport(unittest.TestCase):
    def setUp(self):
        import labelbox as lb
        self.candidates = [build_candidate({"Tricuspid valve findings": "Normal"}, "A4C", "bmode", global_key="synthetic-import")]
        self.assertEqual(ONTOLOGY_NAME, "DeepECHO column review v5")
        self.ontology = SimpleNamespace(name=ONTOLOGY_NAME, uid="ontology", media_type=lb.MediaType.Video,
                                        normalized=as_stored(build_form(), ONTOLOGY_NAME))
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
        return "deepecho-v5-" + hashlib.sha256(json.dumps(payloads, sort_keys=True).encode()).hexdigest()[:16]

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
        self.assertIn("id:/uninterpretable/no/review_columns/tricuspid_appearance/tricuspid_appearance__status/yes", ids)
        self.assertNotIn("id:/review_completed", ids)
        self.assertEqual({p["dataRow"]["globalKey"] for p in job.predictions}, {"synthetic-import"})

    def test_names_the_ontology_lacks_are_refused(self):
        classifications = as_stored(build_form(), "x")["classifications"]
        records = prelabels(self.candidates[0])
        nested = classifications[0]["options"][0]["options"]
        self.assertEqual(with_schema_ids([dict(records[0], name="visible_targets", answer=[])], nested), [])
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
