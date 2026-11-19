# tests/data/test_radar_normality.py

"""`data/radar_normality.py` on synthetic phrases: how the rules read a caption, not how accurate they are on real
reports (that takes the manually checked sample of `radar_normality_audit.py`).

- The review's counterexamples: negation and mixed assertions, and EF wording hidden inside the EF expression.
- Normal needs every part understood, every claim normal and the required attributes stated; a confirmed abnormal
  finding wins over the unknown rest; missing or unrecognized evidence stays unknown.
- Conflicts about one attribute are recorded, never settled for one side, in the text or against a measurement.
- Measurements follow the policy's definitions, not the benchmark labels: each wall on its own, LAVI before diameter,
  units and valid ranges first, a grey zone, and they never set a flag without the caption.
- The B-mode scope applies to captions: what only Doppler can ground leaves the caption before it is judged.
- The policy: invalid ones are refused, and an approval does not survive a change to the rules or the policy.
"""

import json
import os
import random
import re
import shutil
import sys
import tempfile
import unittest

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import radar_captions as rc  # noqa: E402
import radar_normality as rn  # noqa: E402

POLICY = rn.load_policy()
LV_NORMAL = "VG non dilaté. Fonction systolique normale."


def flag(structure, caption, row=None, ef_status=None):
    return assess(structure, caption, row, ef_status).flag


def assess(structure, caption, row=None, ef_status=None):
    return rn.assess(structure, caption, POLICY, rn.structured_facts(structure, row or {}, POLICY), ef_status)


def lv_row(**values):
    columns = {"lvidd": "LVID Diastole (2D)", "ivs": "IVS Diastole Thickness (2D)",
               "lvpw": "LVIW Diastolic Thickness (2D)", "ef": "Visually Estimated EF",
               "mass_index": "LV Mass Index (2D Cubed) DuBois"}
    return {columns[k]: v for k, v in values.items()}


class TestReviewCounterexamples(unittest.TestCase):

    def test_negation_and_mixed_assertions(self):
        self.assertEqual(flag("lv", "Fonction systolique non normale"), "abnormal")
        effusion = assess("pericardium", "Épanchement péricardique présent sans tamponnade")
        self.assertEqual(effusion.flag, "abnormal")
        values = {f.attribute: f.value for f in effusion.facts}
        self.assertEqual(values, {"pericardial_effusion": "abnormal", "pericardial_tamponade": "normal"})
        retained = rn.retain_caption("Valve mitrale calcifiée sans sténose")[0]
        self.assertEqual(flag("mitral", retained), "abnormal")
        mixed = assess("lv", "VG non dilaté mais dilatation apicale")
        self.assertEqual(mixed.flag, "unknown")
        self.assertEqual(mixed.conflicts[0]["attribute"], "lv_size")
        self.assertEqual(mixed.conflicts[0]["kind"], "text_text")

    def test_words_inside_an_ef_expression_are_judged(self):
        self.assertEqual(len(rc.find_ef("FEVG avec hypokinésie globale 55 %")), 1)     # the expression spans them
        hidden = assess("lv", "FEVG avec hypokinésie globale 55 %")
        self.assertEqual(hidden.flag, "abnormal")
        self.assertIn("lv_wall_motion", {f.attribute for f in hidden.facts if f.value == "abnormal"})
        reduced = assess("lv", "VG non dilaté. FEVG diminuée à 55 %")
        self.assertEqual(reduced.flag, "unknown")                                      # words say low, the number not
        self.assertEqual([c["attribute"] for c in reduced.conflicts], ["lv_function"])
        self.assertEqual(flag("lv", "VG non dilaté. FEVG diminuée à 45 %"), "abnormal")
        self.assertEqual(flag("lv", "VG non dilaté. FEVG conservée à 60 %"), "normal")
        self.assertEqual(flag("lv", "VG non dilaté. FEVG normale à 45 %"), "unknown")  # the number does not win
        self.assertEqual(flag("lv", "VG non dilaté. FEVG après traitement 60 %"), "unknown")

    def test_ef_statements_keep_their_whole_range_rule_and_scope(self):
        cases = {"FEVG 60 %": "normal", "FEVG 45 %": "abnormal", "FEVG 55-60 %": "normal", "FEVG < 54 %": "abnormal",
                 "FEVG < 55 %": "unknown", "FEVG entre 50 et 60 %": "unknown", "FEVG environ 60 %": "unknown",
                 "Probablement FEVG 60 %": "unknown", "FEVG 60 % environ": "unknown", "FEVG 60 % ?": "unknown",
                 "FEVG 60 % +/- 5": "unknown", "FEVG 55 % ou moins": "unknown",
                 "FEVG 60 % au Simpson biplan": "normal", "FEVG 55 % (limite inférieure de la normale)": "unknown"}
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(flag("lv", f"VG non dilaté. {text}"), expected)
        self.assertEqual(flag("lv", "VG non dilaté, FEVG 60 %"), "normal")   # the negation stays in its own segment


class TestReading(unittest.TestCase):

    def test_plain_normal_captions(self):
        cases = [("lv", LV_NORMAL), ("lv", "VG normal"), ("lv", "VG de taille et de fonction normales"),
                 ("lv", "VG normal en taille et en fonction"), ("lv", "Taille normale, bonne fonction systolique"),
                 ("rv", "VD de taille et fonction normales"), ("rv", "Ventricule droit normal"),
                 ("la", "Oreillette gauche non dilatée."), ("la", "OG de taille normale"),
                 ("mitral", "Valve mitrale normale"), ("mitral", "Feuillets fins"),
                 ("aortic", "Valve aortique de structure normale."), ("tricuspid", "Valve tricuspide sans anomalie"),
                 ("pulmonic", "Valve pulmonaire normale"), ("aorta", "Aorte non dilatée"),
                 ("aorta", "Aorte de calibre normal"), ("aorta", "Racine aortique normale"),
                 ("ivc", "VCI non dilatée"), ("pericardium", "Pas d'épanchement péricardique."),
                 ("pericardium", "Absence d'épanchement"), ("pericardium", "Épanchement absent"),
                 ("pericardium", "Péricarde sec"), ("pericardium", "Pas d'épanchement ni tamponnade")]
        for structure, caption in cases:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "normal", assess(structure, caption).reasons)

    def test_confirmed_abnormal_captions(self):
        cases = [("lv", "VG dilaté"), ("lv", "Légère dilatation du VG"), ("lv", "Hypertrophie septale"),
                 ("lv", "Dysfonction systolique sévère"), ("lv", "Akinésie apicale"), ("lv", "VG dilaté (65 mm)"),
                 ("rv", "VD dilaté"), ("la", "OG modérément dilatée"), ("mitral", "Prolapsus de la valve mitrale"),
                 ("aortic", "Valve aortique calcifiée"), ("aortic", "Valve aortique bicuspide"),
                 ("aortic", "Bioprothèse aortique"), ("aorta", "Racine aortique normale, aorte ascendante dilatée"),
                 ("ivc", "VCI dilatée"), ("pericardium", "Épanchement péricardique modéré."),
                 ("pericardium", "Épanchement de faible abondance")]
        for structure, caption in cases:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "abnormal", assess(structure, caption).reasons)

    def test_a_confirmed_finding_wins_over_the_unknown_rest(self):
        result = assess("lv", "VG dilaté. Phrase jamais vue. Fonction systolique probablement normale")
        self.assertEqual(result.flag, "abnormal")
        self.assertIn("unrecognized_text", result.reasons)

    def test_anything_not_understood_cannot_be_normal(self):
        cases = ["Clause jamais vue", f"{LV_NORMAL} Clause jamais vue", "VG normal sauf hypokinésie apicale",
                 "VG normal hormis la paroi inférieure", "Fonction systolique normale au repos",
                 "VG non dilaté, 62 mm. Fonction systolique normale", "VG non dilaté à 62 mm. Fonction normale",
                 "Dilatation non significative",
                 "VG dilaté en rapport avec une cardiopathie", "Antécédent de dilatation", "RV normal", ""]
        for caption in cases:
            with self.subTest(caption):
                self.assertEqual(flag("lv", caption), "unknown")
        self.assertEqual(assess("lv", "Clause jamais vue").unrecognized, ("clause jamais vue",))

    def test_another_structure_in_a_caption_is_not_understood(self):
        self.assertEqual(flag("lv", f"{LV_NORMAL} VD normal"), "unknown")

    def test_hedges_and_not_assessed(self):
        for caption in ["VG probablement non dilaté. Fonction systolique normale",
                        "VG non dilaté. Fonction systolique limite", "VG non dilaté. Fonction systolique normale ?",
                        "VG non dilaté. Fonction systolique normale basse", "VG non dilaté. FEVG non mesurée",
                        "VG non dilaté. Fonction systolique difficile à évaluer", "VG mal vu"]:
            with self.subTest(caption):
                self.assertEqual(flag("lv", caption), "unknown")
        self.assertEqual(flag("lv", "Dilatation probable"), "unknown")    # not confirmed, so not abnormal either
        self.assertEqual(flag("pericardium", "Épanchement non exclu"), "unknown")
        self.assertEqual(flag("pericardium", "Trace d'épanchement"), "unknown")
        self.assertIn("not_assessed:overall", assess("pulmonic", "Valve pulmonaire non visualisée").reasons)

    def test_a_negated_or_normal_claim_with_a_modifier_is_not_a_plain_normal(self):
        for structure, caption in [("pericardium", "Pas d'épanchement important"),
                                   ("pericardium", "Pas d'épanchement circonférentiel"),
                                   ("lv", "VG non dilaté. Pas de dysfonction sévère"),
                                   ("lv", "VG non dilaté. Fonction septale normale")]:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "unknown")

    def test_negation_scope(self):
        self.assertEqual(flag("lv", "VG non dilaté et non hypertrophié. Fonction systolique normale"), "normal")
        self.assertEqual(flag("lv", "Pas de dilatation ni d'hypertrophie. Fonction systolique normale"), "normal")
        self.assertEqual(flag("lv", "Pas de dilatation ou d'hypertrophie. Fonction systolique normale"), "normal")
        ambiguous = assess("lv", "Pas de dilatation et d'hypertrophie. Fonction systolique normale")
        self.assertEqual(ambiguous.flag, "unknown")
        self.assertIn("uncertain:lv_wall_thickness", ambiguous.reasons)
        listed = assess("lv", "Pas de dilatation, d'hypertrophie. Fonction systolique normale")
        self.assertEqual(listed.flag, "unknown")
        self.assertEqual(flag("lv", "Pas de pas de dilatation"), "unknown")          # a double negation is not read
        self.assertEqual(flag("lv", "VG dilaté et non hypertrophié"), "abnormal")   # the first finding stands
        self.assertEqual(flag("mitral", "Valve mitrale non calcifiée"), "normal")
        self.assertEqual(flag("lv", "Fonction systolique n'est pas normale"), "abnormal")

    def test_required_attributes(self):
        only_ef = assess("lv", "FEVG 60 %")
        self.assertEqual(only_ef.flag, "unknown")                        # a normal EF says nothing of the LV's size
        self.assertEqual(only_ef.reasons, ("missing_required:lv_size",))
        self.assertEqual(flag("lv", "VG non dilaté"), "unknown")
        self.assertEqual(flag("aortic", "Valve aortique tricuspide"), "unknown")
        self.assertEqual(assess("lv", "").reasons[-1], "no_understood_claim")

    def test_devices_follow_the_policy(self):
        self.assertEqual(flag("mitral", "Pas de prothèse. Valve mitrale normale"), "normal")
        device = assess("mitral", "MitraClip")
        self.assertEqual(device.flag, "abnormal")
        self.assertEqual(device.facts[0].attribute, "mitral_device")

    def test_facts_keep_their_provenance(self):
        result = assess("lv", "VG dilaté. FEVG 45 %", lv_row(lvidd=6.5, ef=44))
        text = next(f for f in result.facts if f.attribute == "lv_size" and f.source == "text")
        self.assertEqual((text.value, text.evidence, text.finding, text.negated),
                         ("abnormal", "dilated", "dilated", False))
        self.assertEqual(text.rule_version, POLICY.rule_version)
        ef = next(f for f in result.facts if f.evidence == rn.EF_EVIDENCE)
        self.assertEqual((ef.low, ef.high, ef.unit, ef.source), (45.0, 45.0, "%", "text"))
        measured = next(f for f in result.facts if f.attribute == "lv_size" and f.source == "structured")
        self.assertEqual((measured.value, measured.low, measured.unit, measured.evidence),
                         ("abnormal", 6.5, "cm", "LVID Diastole (2D)"))
        self.assertEqual(result.evidence, "structured+text")
        negated = assess("pericardium", "Pas d'épanchement").facts[0]
        self.assertEqual((negated.value, negated.finding, negated.negated), ("normal", "effusion", True))


class TestReportTemplates(unittest.TestCase):
    """Synthetic sentences in the style of the structured report templates."""

    def test_normal_templates(self):
        cases = [("lv", "Ventricule gauche de dimension et de masse normales. Fraction d'éjection du ventricule "
                        "gauche normale."),
                 ("lv", "Ventricule gauche non-dilaté. Fonction systolique normale."),
                 ("lv", "VG non dilaté. Fonction systolique normale. Pas d'anomalie segmentaire."),
                 ("la", "Volume indexé de l'oreillette gauche normal"),
                 ("aorta", "Pas de dilatation de l'aorte ascendante proximale. Pas de dilatation de l'aorte au niveau "
                           "des sinus de Valsalva. Pas d'évidence de coarctation de l'aorte."),
                 ("mitral", "Valve mitrale de structure et fonction normales"),
                 ("tricuspid", "Valve tricuspide de structure et fonction normales"),
                 ("aortic", "Valve aortique de structure et de fonction normales"),
                 ("pulmonic", "Valve pulmonaire de structure et de fonction normales"),
                 ("ivc", "Veine cave inférieure non dilatée. Tension veineuse centrale normale.")]
        for structure, caption in cases:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "normal", assess(structure, caption).reasons)

    def test_abnormal_templates(self):
        cases = [("la", "Augmentation légère du volume indexé de l'oreillette gauche", "la_size"),
                 ("lv", "Augmentation légère de la masse du VG", "lv_mass"),
                 ("lv", "Augmentation sévère de la masse du ventricule gauche", "lv_mass"),
                 ("lv", "Présence d'un remodelage concentrique", "lv_geometry"),
                 ("lv", "Présence d'anomalie de la contractilité régionale", "lv_wall_motion"),
                 ("aorta", "Dilatation légère de l'aorte ascendante proximale", "aorta_size:ascending"),
                 ("aorta", "Dilatation légère de l'aorte au niveau des sinus de Valsalva", "aorta_size:root"),
                 ("ivc", "Augmentation modérée de la tension veineuse centrale", "ivc_pressure"),
                 ("ivc", "Tension veineuse centrale légèrement augmentée", "ivc_pressure"),
                 ("aortic", "Bioprothèse aortique Inspiris 23 mm en position aortique", "aortic_device"),
                 ("aortic", "Réduction de la mobilité des feuillets", "aortic_motion")]
        for structure, caption, attribute in cases:
            with self.subTest(caption):
                result = assess(structure, caption)
                self.assertEqual(result.flag, "abnormal", result.reasons)
                self.assertIn(attribute, {f.attribute for f in result.facts if f.value == "abnormal"})

    def test_segments_of_the_aorta_do_not_conflict(self):
        result = assess("aorta", "Dilatation légère de l'aorte ascendante proximale. Pas de dilatation de l'aorte au "
                                 "niveau des sinus de Valsalva.")
        self.assertEqual((result.flag, result.conflicts), ("abnormal", ()))

    def test_lv_mass_is_not_a_mass_lesion(self):
        normal = assess("lv", "Ventricule gauche de dimension et de masse normales")
        self.assertEqual({(f.attribute, f.value) for f in normal.facts},
                         {("lv_size", "normal"), ("lv_mass", "normal")})
        self.assertEqual({(f.attribute, f.value) for f in assess("lv", "Masse ventriculaire gauche normale").facts},
                         {("lv_mass", "normal")})
        for caption in ["Masse apicale", "Présence d'une masse"]:           # read as nothing: neither way
            with self.subTest(caption):
                self.assertEqual(flag("lv", caption), "unknown")
                self.assertEqual(assess("lv", caption).facts, ())

    def test_findings_kept_out_of_both_groups(self):
        lead = "Une électrode de cardiostimulateur est présente dans le ventricule droit"
        self.assertEqual(flag("rv", lead), "unknown")
        self.assertEqual(flag("rv", f"VD de taille et fonction normales. {lead}."), "unknown")
        self.assertIn("uncertain:rv_device", assess("rv", lead).reasons)
        self.assertEqual(flag("lv", f"{LV_NORMAL} Présence d'un bourrelet septal"), "unknown")
        self.assertEqual(flag("lv", "Présence d'un bourrelet septal"), "unknown")

    def test_a_long_ef_expression_is_still_read(self):
        text = "Fraction d'éjection du ventricule gauche évaluée visuellement à 60 %"
        self.assertEqual(rc.find_ef(text), [rc.EfStatement(60, 60)])
        self.assertEqual(flag("lv", f"VG non dilaté. {text}"), "normal")
        self.assertEqual(rc.lv_caption(text, 60.0)[1], "stated")           # no second EF sentence


class TestLvTemplates(unittest.TestCase):
    """LV wording of the report templates, one case per rule. The phrases are synthetic."""

    def assert_normal(self, caption):
        result = assess("lv", caption)
        self.assertEqual((result.flag, result.unrecognized), ("normal", ()), result.reasons)

    def assert_held_back(self, caption, reason):
        """Read in full, but in neither group."""
        result = assess("lv", f"{LV_NORMAL} {caption}")
        self.assertEqual((result.flag, result.unrecognized), ("unknown", ()), result.reasons)
        self.assertIn(reason, result.reasons)

    def ef_values(self, caption):
        return [(f.low, f.high) for f in assess("lv", caption).facts if f.evidence == rn.EF_EVIDENCE]

    def test_a_plural_mark_does_not_cut_its_word(self):
        result = assess("lv", "Présence d'anomalie(s) segmentaire(s) de contractilité du ventricule gauche "
                              "(voir le diagramme)")
        self.assertEqual((result.flag, result.unrecognized), ("abnormal", ()))
        self.assertIn(("lv_wall_motion", "abnormal"), {(f.attribute, f.value) for f in result.facts})
        self.assert_normal(f"{LV_NORMAL} Pas d'anomalie(s) régionale(s) de la contractilité")

    def test_a_reference_to_a_figure_is_not_a_claim(self):
        for note in ["voir le diagramme", "cf schéma", "Voir schéma", "cf diagramme"]:
            with self.subTest(note):
                self.assert_normal(f"{LV_NORMAL} {note}")
        for caption in ["cf schéma pour les anomalies contractiles régionales", "cf schéma pour les ARC"]:
            with self.subTest(caption):
                self.assertEqual(flag("lv", caption), "abnormal")

    def test_an_ef_cut_from_its_label(self):
        self.assertEqual(self.ef_values("FEVG 60 % au Simpson biplan, 58 % par WMS"), [(60, 60), (58, 58)])
        self.assertEqual(self.ef_values("FEVG 60 % par Simpson et 55-60 % visuellement"), [(60, 60), (55, 60)])
        self.assertEqual(self.ef_values("Fonction systolique normale, évaluée à 60 %"), [(60, 60)])
        self.assertEqual(self.ef_values("Fraction d'éjection du ventricule gauche normale, évaluée à 60 %"),
                         [(60, 60)])
        self.assertEqual(self.ef_values("Fonction systolique normale. Elle est calculée à 60 % par la méthode de "
                                        "Simpson"), [(60, 60)])
        self.assertEqual(self.ef_values("Fraction d'éjection ventriculaire gauche évaluée par méthode Simpson's "
                                        "biplan à 60 %"), [(60, 60)])
        self.assertEqual(self.ef_values("FEVG évaluée par plusieurs méthodes à 60 %"), [(60, 60)])
        self.assert_normal("VG non dilaté. FEVG 60 % au Simpson biplan, 58 % par WMS")
        self.assert_normal("VG non dilaté. FEVG 60 % par Simpson et WMS")

    def test_an_ef_cut_from_its_label_is_judged_like_any_other(self):
        split = assess("lv", "VG non dilaté. FEVG 60 % par Simpson et 40 % par WMS")
        self.assertEqual(split.flag, "unknown")
        self.assertIn("lv_function", {c["attribute"] for c in split.conflicts})
        self.assertEqual(flag("lv", "VG non dilaté. Fonction systolique normale, estimée à 40 % visuellement"),
                         "unknown")
        self.assertEqual(flag("lv", "VG non dilaté. FEVG 40 % par WMS"), "abnormal")
        hedged = assess("lv", "VG non dilaté. Fonction systolique normale, environ 60 % visuellement")
        self.assertEqual(hedged.flag, "unknown")

    def test_only_an_ef_method_or_subject_makes_a_percentage_an_ef(self):
        for caption in ["VG non dilaté, 60 %", "Volume évalué à 60 %", "Fonction systolique normale, 150 % par WMS",
                        "VG non dilaté et 60 %"]:
            with self.subTest(caption):
                result = assess("lv", caption)
                self.assertEqual(self.ef_values(caption), [])
                self.assertTrue(result.unrecognized)

    def test_lv_size(self):
        for caption in ["Augmentation légère du diamètre du VG en diastole",
                        "Augmentation sévère du volume télédiastolique du ventricule gauche"]:
            with self.subTest(caption):
                result = assess("lv", caption)
                self.assertEqual((result.flag, result.unrecognized), ("abnormal", ()))
                self.assertIn(("lv_size", "abnormal"), {(f.attribute, f.value) for f in result.facts})
        self.assert_normal("VG non dilaté en diastole. Fonction systolique normale.")

    def test_a_small_lv_and_the_outflow_tract_are_held_back(self):
        result = assess("lv", "Ventricule gauche de petite dimension. Fonction systolique normale.")
        self.assertEqual((result.flag, result.unrecognized), ("unknown", ()))
        self.assertIn("uncertain:lv_size", result.reasons)
        self.assert_held_back("Chambre de chasse du ventricule gauche de grande dimension", "uncertain:lvot_size")
        wide = assess("lv", f"{LV_NORMAL} Chambre de chasse du ventricule gauche de grande dimension")
        self.assertEqual({f.value for f in wide.facts if f.attribute == "lv_size"}, {"normal"})

    def test_denied_findings(self):
        for phrase in ["Pas de communication inter-ventriculaire", "Pas d'évidence de shunt à l'étage ventriculaire",
                       "Pas de CIV", "Pas de SAM", "Pas de mouvement systolique antérieur mitral",
                       "Pas de thrombus visualisé", "Pas d'évidence de thrombus intra-cavitaire",
                       "Pas d'évidence de cardiomyopathie hypertrophique",
                       "Pas de critère d'asynchronisme inter-ventriculaire",
                       "Absence d'hypertrophie des parois ventriculaires gauches",
                       "Pas d'anomalie segmentaire ou diffuse de la contractilité"]:
            with self.subTest(phrase):
                self.assert_normal(f"{LV_NORMAL} {phrase}")

    def test_stated_findings(self):
        cases = [("Anomalie de la contractilité dans le territoire de l'artère coronaire droite", "lv_wall_motion"),
                 ("Mouvement septal anormal post-opératoire", "lv_septal_motion"),
                 ("Mouvement septal anormal secondaire à un bloc de branche gauche", "lv_septal_motion"),
                 ("Mouvement systolique antérieur", "lv_sam"),
                 ("Ventricule gauche en D", "lv_geometry"),
                 ("Cardiomyopathie hypertrophique apicale", "lv_cardiomyopathy"),
                 ("Communication inter-ventriculaire", "lv_vsd"),
                 ("Augmentation légère de l'épaisseur du septum", "lv_wall_thickness")]
        for caption, attribute in cases:
            with self.subTest(caption):
                result = assess("lv", caption)
                self.assertEqual((result.flag, result.unrecognized), ("abnormal", ()))
                self.assertIn(attribute, {f.attribute for f in result.facts if f.value == "abnormal"})

    def test_a_diagnosis_from_the_history_is_held_back(self):
        self.assert_held_back("Cardiopathie ischémique", "uncertain:lv_cardiomyopathy")
        self.assertTrue(assess("lv", "Cardiopathie").unrecognized)     # alone, the word says nothing

    def test_a_territory_limits_a_negation(self):
        self.assert_held_back("Pas d'anomalie de la contractilité dans le territoire de l'artère circonflexe gauche",
                              "uncertain:lv_wall_motion")

    def test_septal_motion_does_not_contradict_regional_wall_motion(self):
        result = assess("lv", "Pas d'anomalie segmentaire. Mouvement septal anormal post-opératoire.")
        self.assertEqual((result.flag, result.conflicts), ("abnormal", ()))

    def test_a_repaired_defect_is_held_back(self):
        self.assert_held_back("Statut post-fermeture de communication inter-ventriculaire par voie chirurgicale, "
                              "sans shunt résiduel", "uncertain:lv_vsd")

    def test_a_septal_bulge_is_held_back_and_non_obstructive_denies_nothing(self):
        for caption in ["Présence d'un bourrelet septal basal non-obstructif mesurant 13 mm d'épaisseur",
                        "Bourrelet septal de 14 mm non obstructif", "Bourrelet septal à 12 mm"]:
            with self.subTest(caption):
                self.assert_held_back(caption, "uncertain:lv_morphology")
                bulge = next(f for f in assess("lv", caption).facts if f.finding == "septal_bulge")
                self.assertFalse(bulge.negated)

    def test_strain(self):
        self.assert_normal(f"{LV_NORMAL} Strain (déformation) myocardique longitudinal global normal à -20 %")
        reduced = assess("lv", "Strain (déformation) myocardique longitudinal global légèrement diminué à -15 %")
        self.assertEqual(reduced.flag, "abnormal")
        self.assertEqual({f.attribute for f in reduced.facts}, {"lv_strain"})     # never a claim on the whole LV
        alone = assess("lv", f"{LV_NORMAL} Strain (déformation) myocardique longitudinal global -20 %")
        self.assertEqual(alone.flag, "unknown")                                   # no cutoff: a value says nothing
        self.assertTrue(alone.unrecognized)

    def test_how_the_images_were_taken_is_not_a_finding(self):
        for note in ["Sans contraste", "Avec Definity"]:
            with self.subTest(note):
                self.assert_normal(f"{LV_NORMAL} {note}")

    def test_a_spreadsheet_escape_is_a_space(self):
        self.assert_normal("VG non dilaté. _x000B_Fraction d'éjection du ventricule gauche normale")


class TestHardWording(unittest.TestCase):
    """Phrases built to fool the rules. Wording that merely looks unusual should end up unknown."""

    NOT_NORMAL = {
        "lv": ["VG non dilaté, fonction systolique normale, sauf akinésie apicale",
               "VG non dilaté. Fonction systolique normale. Hypertrophie",
               "VG non dilaté. Fonction systolique normale. Pas de thrombus visible",
               "VG non dilaté. Fonction systolique conservée malgré des troubles de la cinétique",
               "VG de taille normale. Fonction systolique normale avec hypokinésie inférieure",
               "VG de taille normale, fonction systolique moins bonne",
               "VG de taille normale, fonction systolique bonne mais diminuée",
               "VG non dilaté. Fonction systolique plus normale",
               "VG non dilaté. Fonction systolique normale ou légèrement diminuée",
               "VG non dilaté. Fonction systolique normale à légèrement diminuée",
               "VG non dilaté. Fonction systolique normale et diminuée",
               "VG non dilaté. Fonction non normale", "VG non dilaté. Pas de fonction normale",
               "VG non dilaté. Fonction systolique normale pour l'âge",
               "VG non dilaté. Fonction systolique légèrement normale", "VG pas tout à fait normal",
               "VG presque normal", "VG à peu près normal", "VG normal à l'exception de la paroi inférieure",
               "VG normal, excepté l'apex", "VG non dilaté mais fonction systolique diminuée",
               "Pas de dilatation mais hypertrophie concentrique. Fonction systolique normale",
               "Pas de dilatation. Fonction systolique non conservée",
               "Fonction systolique normale. VG non encore dilaté", "VG plus dilaté. Fonction systolique normale",
               "VG non dilaté. FEVG normale à 40 %", "VG non dilaté. FE 45 %", "VG non dilaté. FEVG < 50 %",
               "VG non dilaté. Fonction systolique normale, FEVG 30 %",
               "VG non dilaté, fonction systolique normale, FEVG 60 % avant, 30 % ce jour",
               "VG non dilaté. Fonction systolique normale avant chimiothérapie",
               "VG non dilaté. Fonction normale, cardiomyopathie"],
        "pericardium": ["Pas d'épanchement circonférentiel mais épanchement postérieur",
                        "Pas d'épanchement sauf en regard de l'OD", "Pas d'épanchement, sinon lame postérieure",
                        "Pas d'épanchement. Péricarde épaissi", "Pas de tamponnade", "Sans tamponnade",
                        "Pas d'épanchement significatif", "Pas d'épanchement à ce jour", "Épanchement résorbé",
                        "Disparition de l'épanchement", "Épanchement non significatif",
                        "Péricarde sec avec calcifications", "Péricarde normal, épanchement minime",
                        "Lame d'épanchement"],
        "mitral": ["Valve mitrale normale sauf calcification de l'anneau",
                   "Valve mitrale normale, calcification de l'anneau", "Feuillets fins, prolapsus",
                   "Feuillets fins mais calcifiés", "Prothèse mitrale normale", "Bioprothèse normofonctionnelle",
                   "Valve mitrale d'aspect normal avec prolapsus",
                   "Valve normale, pas de prolapsus franc", "Pas de calcification franche", "Valve mitrale myxoïde",
                   "Morphologie non normale"],
        "aortic": ["Valve aortique tricuspide calcifiée", "Valve aortique normale, légèrement calcifiée",
                   "Valve aortique normale pour l'âge", "Sigmoïdes fines mais remaniées",
                   "Valve aortique de structure normale, bicuspide"],
        "aorta": ["Aorte ascendante non dilatée, racine dilatée", "Aorte non dilatée sauf la racine",
                  "Aorte non dilatée mais anévrisme de la crosse", "Racine aortique normale. Aorte ascendante 45 mm",
                  "Aorte ascendante limite à 41 mm"],
        "ivc": ["VCI non dilatée, collapsus absent", "VCI non dilatée sans collapsus",
                "VCI non dilatée mais peu compliante", "VCI normale, collapsus diminué"],
        "rv": ["VD non dilaté, fonction systolique normale, hypertrophie", "VD normal mais sonde de pacemaker",
               "VD de taille normale. Dysfonction", "VD de taille normale. Fonction systolique diminuée"],
        "la": ["OG non dilatée mais thrombus de l'auricule", "OG non dilatée. Contraste spontané", "OG normale, FA"],
    }
    NOT_ABNORMAL = [("lv", "Dysfonction exclue"), ("lv", "Dilatation non retrouvée"), ("lv", "Hypertrophie écartée"),
                    ("lv", "Absence de dilatation"), ("lv", "Dilatation peu probable"),
                    ("lv", "Aucun signe de dilatation"), ("pericardium", "Épanchement résorbé"),
                    ("pericardium", "Disparition de l'épanchement"), ("pericardium", "Pas de récidive d'épanchement"),
                    ("lv", "Ancienne hypertrophie"), ("lv", "Régression de l'hypertrophie"),
                    ("lv", "Amélioration de la dilatation"), ("lv", "Pas d'évolution de la dilatation"),
                    ("mitral", "Antécédent de prothèse"), ("aortic", "Remplacement valvulaire par prothèse"),
                    ("lv", "Risque de dilatation"), ("lv", "À surveiller pour dilatation"),
                    ("lv", "Dilatation à confirmer"), ("rv", "Pas de signe de dysfonction"),
                    ("la", "Dilatation non significative")]

    def test_no_false_normal(self):
        for structure, captions in self.NOT_NORMAL.items():
            for caption in captions:
                with self.subTest(caption):
                    self.assertNotEqual(flag(structure, caption), "normal")

    def test_no_false_abnormal(self):
        for structure, caption in self.NOT_ABNORMAL:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "unknown")


class TestAgainstMeasurements(unittest.TestCase):

    def test_agreement_and_conflict(self):
        self.assertEqual(flag("lv", LV_NORMAL, lv_row(lvidd=4.8, ef=60, ivs=0.8, lvpw=0.8)), "normal")
        conflict = assess("lv", LV_NORMAL, lv_row(lvidd=6.5))
        self.assertEqual(conflict.flag, "unknown")                     # neither side wins
        self.assertEqual(conflict.conflicts, ({"attribute": "lv_size", "kind": "text_structured", "text": ["normal"],
                                               "structured": "abnormal"},))
        self.assertEqual(flag("lv", "VG dilaté", lv_row(lvidd=4.8)), "unknown")        # not a confirmed finding
        self.assertEqual(flag("lv", "VG normal", lv_row(lvidd=6.5)), "unknown")        # a whole-structure statement
        self.assertEqual(flag("lv", "VG dilaté. Hypertrophie", lv_row(lvidd=4.8)), "abnormal")   # another one stands

    def test_a_measurement_never_sets_the_flag_alone(self):
        silent = assess("lv", LV_NORMAL, lv_row(ivs=1.4, lvpw=1.4))
        self.assertEqual(silent.flag, "unknown")
        self.assertIn("structured_abnormal_not_in_caption:lv_wall_thickness", silent.reasons)
        self.assertEqual(flag("lv", "FEVG 60 %", lv_row(lvidd=4.8)), "unknown")   # nor fills in the missing size
        self.assertEqual(flag("lv", "Clause jamais vue", lv_row(lvidd=6.5)), "unknown")

    def test_each_wall_counts_on_its_own(self):
        one_thick = lv_row(ivs=1.4, lvpw=0.9)
        self.assertEqual(flag("lv", "Pas d'hypertrophie", one_thick), "unknown")
        self.assertEqual(assess("lv", "Pas d'hypertrophie", one_thick).conflicts[0]["attribute"], "lv_wall_thickness")
        self.assertIn("structured_abnormal_not_in_caption:lv_wall_thickness",
                      assess("lv", LV_NORMAL, one_thick).reasons)
        missing_one = rn._measured("lv_wall_thickness", rn.structured_facts("lv", lv_row(ivs=0.8), POLICY), POLICY)
        self.assertIsNone(missing_one)                                 # one normal wall does not make the walls normal

    def test_a_mild_wall_needs_normal_mass_shown(self):
        wall = {"ivs": 1.15, "lvpw": 0.9, "lvidd": 4.8}
        cases = {
            "stated mass": ("Ventricule gauche de dimension et de masse normales. Fonction systolique normale.", {}),
            "measured mass": (LV_NORMAL, {"mass_index": 0.083}),
        }
        for name, (caption, extra) in cases.items():
            with self.subTest(name):
                result = assess("lv", caption, lv_row(**wall, **extra))
                self.assertEqual((result.flag, result.conflicts), ("normal", ()), result.reasons)
                self.assertTrue(result.boundary)              # an audit stratum
        for name, extra in {"no evidence": {}, "mass in the grey zone": {"mass_index": 0.105}}.items():
            with self.subTest(name):
                result = assess("lv", LV_NORMAL, lv_row(**wall, **extra))
                self.assertEqual(result.flag, "unknown")
                self.assertIn("structured_abnormal_not_in_caption:lv_wall_thickness", result.reasons)

    def test_only_mass_settles_a_mild_wall(self):
        wall = lv_row(ivs=1.15, lvpw=0.9, lvidd=4.8)
        for caption in [f"{LV_NORMAL} Pas d'hypertrophie.", "VG normal."]:
            with self.subTest(caption):
                result = assess("lv", caption, wall)
                self.assertEqual(result.flag, "unknown")
                self.assertEqual(result.conflicts[0]["attribute"], "lv_wall_thickness")

    def test_a_mild_wall_needs_normal_geometry_too(self):
        stated = "Ventricule gauche de dimension et de masse normales. Fonction systolique normale."
        for name, row in {"remodelling": {"ivs": 1.15, "lvpw": 1.1, "lvidd": 4.2},        # 2 x 1.1 / 4.2 = 0.52
                          "no cavity": {"ivs": 1.15, "lvpw": 0.9},
                          "no posterior wall": {"ivs": 1.15, "lvidd": 4.8}}.items():
            with self.subTest(name):
                result = assess("lv", stated, lv_row(**row, mass_index=0.083))
                self.assertEqual(result.flag, "unknown")
                self.assertIn("structured_abnormal_not_in_caption:lv_wall_thickness", result.reasons)

    def test_the_geometry_limit_is_not_rounded(self):
        stated = "Ventricule gauche de dimension et de masse normales. Fonction systolique normale."
        for cavity, expected in {5.05: "normal", 5.0: "normal", 4.95: "unknown", 4.99: "unknown"}.items():
            with self.subTest(ratio=2 * 1.05 / cavity):
                self.assertEqual(flag("lv", stated, lv_row(ivs=1.15, lvpw=1.05, lvidd=cavity)), expected)

    def test_geometry_alone_blocks_nothing(self):
        result = assess("lv", LV_NORMAL, lv_row(ivs=0.9, lvpw=0.9, lvidd=4.0))             # 0.45, walls in range
        self.assertEqual(result.flag, "normal", result.reasons)

    def test_a_wall_beyond_the_mild_band_counts_in_full(self):
        wall = {"ivs": 1.3, "lvpw": 0.9, "lvidd": 4.8}
        stated = "Ventricule gauche de dimension et de masse normales. Fonction systolique normale."
        for caption, extra in [(stated, {}), (LV_NORMAL, {"mass_index": 0.083})]:
            with self.subTest(caption):
                result = assess("lv", caption, lv_row(**wall, **extra))
                self.assertEqual(result.flag, "unknown")
                self.assertIn("structured_abnormal_not_in_caption:lv_wall_thickness", result.reasons)
        denied = assess("lv", f"{LV_NORMAL} Pas d'hypertrophie.", lv_row(**wall))
        self.assertEqual(denied.conflicts[0]["attribute"], "lv_wall_thickness")

    def test_an_abnormal_claim_is_not_settled_by_normal_mass(self):
        result = assess("lv", "Hypertrophie septale. Masse ventriculaire gauche normale.",
                        lv_row(ivs=1.15, lvpw=0.9, lvidd=4.8))
        self.assertEqual(result.flag, "abnormal")

    def test_lv_mass_index_against_the_caption(self):
        mass = "Masse ventriculaire gauche normale."
        conflict = assess("lv", f"{LV_NORMAL} {mass}", lv_row(mass_index=0.130))
        self.assertEqual((conflict.flag, conflict.conflicts[0]["attribute"]), ("unknown", "lv_mass"))
        hidden = assess("lv", LV_NORMAL, lv_row(mass_index=0.130))
        self.assertEqual(hidden.flag, "unknown")
        self.assertIn("structured_abnormal_not_in_caption:lv_mass", hidden.reasons)
        self.assertEqual(flag("lv", LV_NORMAL, lv_row(mass_index=0.083)), "normal")
        self.assertEqual(flag("lv", "Augmentation légère de la masse du VG", lv_row(mass_index=0.083)), "unknown")
        self.assertEqual(flag("lv", "Augmentation légère de la masse du VG", lv_row(mass_index=0.130)), "abnormal")

    def test_lv_mass_index_is_read_in_its_units(self):
        fact = next(f for f in rn.structured_facts("lv", lv_row(mass_index=0.083), POLICY) if f.attribute == "lv_mass")
        self.assertAlmostEqual(fact.low, 83)
        self.assertEqual(fact.unit, "g/m2")
        wrong = next(f for f in rn.structured_facts("lv", lv_row(mass_index=83), POLICY) if f.attribute == "lv_mass")
        self.assertEqual(wrong.value, "invalid")          # already in g/m2: refused, not read as enormous

    def test_la_volume_index_comes_before_the_diameter(self):
        lavi, diameter = "LA Volume Index (BP MOD) DuBois", "LA Dimension"
        caption = "OG non dilatée"
        self.assertEqual(flag("la", caption, {lavi: 0.045, diameter: 3.8}), "unknown")    # enlarged by volume
        self.assertEqual(flag("la", caption, {lavi: 0.030, diameter: 4.5}), "normal")     # the diameter is not read
        self.assertEqual(flag("la", caption, {diameter: 4.5}), "unknown")
        self.assertEqual(flag("la", caption, {diameter: 3.5}), "normal")
        facts = rn.structured_facts("la", {lavi: 0.045, diameter: 3.8}, POLICY)
        self.assertEqual([(f.evidence, f.low, f.unit) for f in facts], [(lavi, 45.0, "mL/m2")])

    def test_units_valid_ranges_and_exact_cutoffs(self):
        lavi = "LA Volume Index (BP MOD) DuBois"
        self.assertEqual(rn.structured_facts("la", {lavi: 34.0}, POLICY)[0].value, "invalid")   # already in mL/m2
        self.assertEqual(rn.structured_facts("la", {lavi: 0.034}, POLICY)[0].value, "normal")
        self.assertEqual(rn.structured_facts("la", {lavi: 0.035}, POLICY)[0].value, "abnormal")
        self.assertEqual(flag("la", "OG non dilatée", {lavi: 34.0}), "normal")      # invalid counts as missing
        self.assertEqual(rn.structured_facts("lv", lv_row(lvidd=48), POLICY)[0].value, "invalid")   # mm, not cm
        code = "MHI VD fonction systolique"
        self.assertEqual(rn.structured_facts("rv", {code: -1}, POLICY)[0].value, "invalid")
        self.assertEqual(rn.structured_facts("rv", {code: 0}, POLICY)[0].value, "normal")
        self.assertEqual(rn.structured_facts("rv", {code: 1}, POLICY)[0].value, "abnormal")
        self.assertEqual(rn.structured_facts("rv", {code: float("nan")}, POLICY), ())

    def test_grey_zone_neither_supports_nor_contradicts(self):
        grey = assess("lv", LV_NORMAL, lv_row(lvidd=5.5))
        self.assertEqual(grey.flag, "normal")
        self.assertTrue(grey.boundary)
        self.assertEqual(flag("lv", "VG dilaté", lv_row(lvidd=5.5)), "abnormal")

    def test_ef_near_the_cutoff_is_marked(self):
        self.assertTrue(assess("lv", "VG non dilaté. FEVG 55 %").boundary)
        self.assertFalse(assess("lv", "VG non dilaté. FEVG 65 %").boundary)

    def test_ef_against_the_structured_value(self):
        caption, status = rc.lv_caption("VG non dilaté. FEVG 60 %", 40)
        self.assertEqual(status, "conflict")
        self.assertEqual(flag("lv", caption, lv_row(ef=40), status), "unknown")
        caption, status = rc.lv_caption("VG non dilaté. FEVG 60 %", 74, tolerance=5)   # both normal, far apart
        self.assertEqual(status, "conflict")
        self.assertEqual(flag("lv", caption, lv_row(ef=74), status), "unknown")
        caption, status = rc.lv_caption("VG non dilaté. FEVG 55 %", 52)                # close, on either side
        self.assertEqual(status, "stated")
        self.assertEqual(flag("lv", caption, lv_row(ef=52), status), "unknown")

    def test_the_appended_ef_is_the_captions_own_claim(self):
        caption, status = rc.lv_caption("VG non dilaté.", 60.0)
        appended = assess("lv", caption, lv_row(ef=60.0), status)
        self.assertEqual(appended.flag, "normal")
        self.assertEqual(appended.evidence, "appended+structured+text")
        low, status = rc.lv_caption("VG non dilaté.", 45.0)
        self.assertEqual(flag("lv", low, lv_row(ef=45.0), status), "abnormal")
        contradiction, status = rc.lv_caption("VG normal", 45.0)
        result = assess("lv", contradiction, lv_row(ef=45.0), status)
        self.assertEqual(result.flag, "unknown")
        self.assertEqual(result.conflicts[0]["kind"], "text_structured")


class TestScope(unittest.TestCase):

    def test_doppler_clauses_leave_the_caption(self):
        cases = {
            "Valve mitrale calcifiée. Insuffisance mitrale modérée.": "Valve mitrale calcifiée.",
            "Valve mitrale calcifiée sans sténose": "Valve mitrale calcifiée.",
            "Gradient moyen 41 mmHg. Vmax 3,2 m/s. Pas de calcification": "Pas de calcification.",
            "Valve calcifiée (sténose modérée)": "Valve calcifiée.",
            "Pas de calcification ni de sténose": "Pas de calcification.",
            "Dysfonction diastolique grade 1. VG non dilaté": "VG non dilaté.",
            "FEVG entre 50 et 60 %, pas de dysfonction diastolique": "FEVG entre 50 et 60 %.",
            "L'aire valvulaire aortique est à 1,2 cm². Valve calcifiée": "Valve calcifiée.",
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(rn.retain_caption(text)[0], expected)

    def test_what_remains_must_stand_alone(self):
        self.assertEqual(rn.retain_caption("Pas de sténose ni de calcification"), ("", 1, 1))
        self.assertEqual(rn.retain_caption("Pas de sténose et de calcification. VG normal")[0], "VG normal.")
        self.assertEqual(rn.retain_caption("Insuffisance mitrale modérée"), ("", 1, 0))

    def test_what_only_described_a_removed_segment_goes_with_it(self):
        cases = {
            "Insuffisance tricuspidienne légère (1-2/4)": "",
            "Insuffisance tricuspidienne modérée (2+) centrale": "",
            "Gradient de 9 mmHg (11-2)": "",
            "Gradient de 38 mmHg avec volume 36 mL/m2. Valve tricuspide": "Valve tricuspide.",
            "Prolapsus discret. Le jet est toutefois central": "Prolapsus discret.",
            "Épanchement minime. Variations au Doppler (34 % en mitral, 60 % en tricuspide)": "Épanchement minime.",
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(rn.retain_caption(text)[0], expected)

    def test_a_statement_after_a_removed_segment_stays(self):
        cases = {
            "Insuffisance mitrale légère, calcification de l'anneau": "calcification de l'anneau.",
            "Insuffisance mitrale légère, hypokinésie apicale": "hypokinésie apicale.",
            "Insuffisance mitrale légère, fonction normale": "fonction normale.",
            "Valve calcifiée (sténose modérée), prolapsus discret": "Valve calcifiée, prolapsus discret.",
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(rn.retain_caption(text), (expected, 1, 0))

    def test_what_came_before_a_removed_part_outlives_a_tail_that_continues_it(self):
        text = "Léger mouvement de SAM avec gradient de 8 mmHg et 39 mmHg au repos"
        self.assertEqual(rn.retain_caption(text), ("Léger mouvement de SAM.", 2, 0))
        text = "Léger mouvement de SAM avec gradient de 8 mmHg et de la paroi"
        self.assertEqual(rn.retain_caption(text), ("Léger mouvement de SAM.", 1, 1))

    def test_a_finding_that_continued_a_removed_part_keeps_the_caption_from_normal(self):
        caption, _, stranded = rn.retain_caption("Valve mitrale de structure normale. Sténose et calcification")
        self.assertEqual((caption, stranded), ("Valve mitrale de structure normale.", 1))
        self.assertEqual(rn.assess("mitral", caption, POLICY, stranded=stranded).flag, "unknown")

    def test_a_closing_parenthesis_in_a_removed_part_stays(self):
        text = "Ventricule droit dilaté (aire 24 cm2, pression 30 mmHg). Fonction normale"
        self.assertEqual(rn.retain_caption(text)[0], "Ventricule droit dilaté (aire 24 cm2). Fonction normale.")

    def test_a_state_with_no_subject_does_not_stand_alone(self):
        self.assertEqual(rn.retain_caption("Prolapsus discret. Insuffisance mitrale (normale)"),
                         ("Prolapsus discret.", 1, 1))

    def test_a_stranded_part_is_counted_once_and_punctuation_is_not_one(self):
        self.assertEqual(rn.retain_caption("Insuffisance légère (1-2/4) centrale")[1:], (1, 2))
        self.assertEqual(rn.retain_caption("Insuffisance légère ()")[1:], (1, 0))
        self.assertEqual(rn.retain_caption("Insuffisance légère (1-2/4), fonction normale, centrale")[1:], (1, 1))

    def test_a_caption_that_lost_unread_text_is_never_normal(self):
        kept = "Ventricule droit de dimension et fonction normales"
        caption, _, stranded = rn.retain_caption(f"{kept}. Pression à 42 mmHg (signe nouveau)")
        self.assertEqual((caption, stranded), (f"{kept}.", 1))
        self.assertEqual(rn.assess("rv", caption, POLICY).flag, "normal")
        result = rn.assess("rv", caption, POLICY, stranded=stranded)
        self.assertEqual(result.flag, "unknown")
        self.assertIn("stranded_text", result.reasons)
        abnormal = rn.assess("rv", "Ventricule droit dilaté", POLICY, stranded=1)
        self.assertEqual(abnormal.flag, "abnormal")

    def test_a_stranded_part_leaves_its_closing_parenthesis(self):
        text = "Dysfonction systolique légère (FAC 35 %, pression à 30 mmHg, valeur 12)"
        caption = rn.retain_caption(text)[0]
        self.assertEqual((caption.count("("), caption.count(")")), (1, 1))
        self.assertEqual(flag("rv", caption), "abnormal")

    def test_a_list_removed_from_its_parentheses_leaves_no_parentheses(self):
        text = "Fonction systolique normale (S' 13 cm/s, indice 37 %, valeur 15 %). VG non dilaté"
        self.assertEqual(rn.retain_caption(text), ("Fonction systolique normale. VG non dilaté.", 1, 2))

    def test_no_space_or_comma_is_left_in_front_of_a_clause(self):
        caption = rn.retain_caption("Prolapsus discret. Insuffisance sévère (4/4) sur flail de P2")[0]
        self.assertEqual(caption, "Prolapsus discret. sur flail de P2.")
        caption = rn.retain_caption("Prolapsus discret. Insuffisance sévère (2+), calcification de l'anneau")[0]
        self.assertEqual(caption, "Prolapsus discret. calcification de l'anneau.")

    def test_a_clause_left_with_no_word_goes(self):
        self.assertEqual(rn.retain_caption("\xa0(sténose modérée). Valve calcifiée")[0], "Valve calcifiée.")
        self.assertEqual(rn.retain_caption("_x000b_ (sténose modérée). Valve calcifiée")[0], "Valve calcifiée.")

    def test_a_doppler_valve_area_leaves_the_caption(self):
        text = "Prothèse en bonne position. AVA: 1.2 cm²"
        self.assertEqual(rn.retain_caption(text)[0], "Prothèse en bonne position.")

    def test_a_shared_state_word_stays_with_its_subjects(self):
        text = "Ventricule droit de dimension, fonction et pression systolique normales."
        retained, dropped, stranded = rn.retain_caption(text)
        self.assertEqual((retained, dropped, stranded), ("Ventricule droit de dimension, fonction normales.", 1, 0))
        self.assertEqual(flag("rv", retained), "normal")
        self.assertNotIn("pression", retained)

    def test_a_state_moves_only_when_it_surely_covers_the_list(self):
        for state in ["non normales", "probablement normales", "globalement normales", "normales à 30 mmHg",
                      "légèrement augmentées", "augmentée", "pas normales"]:
            with self.subTest(state):
                text = f"Ventricule droit de dimension, fonction et pression systolique {state}."
                caption, dropped, stranded = rn.retain_caption(text)
                self.assertEqual((caption, dropped), ("", 1))
                self.assertGreater(stranded, 0)                # the clause is gone, and what it said is unknown
        for state in ["non normales", "probablement normales"]:
            with self.subTest(state):
                text = f"VG non dilaté. Fonction et pression systolique {state}."
                self.assertEqual(rn.retain_caption(text)[0], "VG non dilaté.")
        text = "Ventricule droit de dimension, fonction et pression systolique anormales"
        self.assertEqual(rn.retain_caption(text)[0], "Ventricule droit de dimension, fonction anormales.")

    def test_a_singular_normale_covers_subjects_that_have_no_other_state(self):
        tail = "de dimension, fonction et pression systolique normale"
        self.assertEqual(rn.retain_caption(f"Ventricule droit {tail}"),
                         ("Ventricule droit de dimension, fonction normale.", 1, 0))
        self.assertEqual(rn.retain_caption("Fonction et pression systolique normale du ventricule droit"),
                         ("Fonction normale du ventricule droit.", 1, 0))
        self.assertEqual(rn.retain_caption("Ventricule droit dilaté et pression systolique normale")[0],
                         "Ventricule droit dilaté.")

    def test_a_removed_diastolic_function_leaves_its_state_to_the_systolic_one(self):
        cell = "Ventricule gauche de dimensions, de masse et de fonction systolique et diastolique normales."
        self.assertEqual(rn.retain_caption(cell), (
            "Ventricule gauche de dimensions, de masse et de fonction systolique normales.", 1, 0))
        self.assertEqual(rn.retain_caption("Fonction systolique et diastolique augmentée")[0], "")

    def test_subjects_left_without_a_state_go(self):
        self.assertEqual(rn.retain_caption("Ventricule droit de dimension, fonction et pression à 30 mmHg")[0], "")
        kept = "Ventricule droit de dimension, fonction et pression à 30 mmHg. VD non dilaté"
        self.assertEqual(rn.retain_caption(kept)[0], "VD non dilaté.")

    def test_a_state_word_is_not_carried_onto_a_finding(self):
        self.assertEqual(rn.retain_caption("Valve mitrale calcifiée et gradient normal")[0],
                         "Valve mitrale calcifiée.")

    def test_a_caption_without_doppler_is_untouched(self):
        for text in ["Diamètre télédiastolique normal", "Oreillette gauche dilatée"]:
            self.assertEqual(rn.retain_caption(text), (f"{text}.", 0, 0))
        self.assertEqual(rn.retain_caption("VG non dilaté.  // FEVG à 60 %"), ("VG non dilaté. FEVG à 60 %.", 0, 0))
        self.assertEqual(rn.retain_caption(float("nan")), ("", 0, 0))

    def test_doppler_left_in_a_caption_is_not_understood(self):
        self.assertEqual(flag("mitral", "Valve mitrale normale. Insuffisance mitrale"), "unknown")


class TestFromTheAuditSheet(unittest.TestCase):
    """Wording that reading the audit sheet showed was not handled. The phrases are synthetic."""

    def test_valve_devices_are_read(self):
        cases = [("aortic", "Bioprothèse valvulaire aortique."),
                 ("aortic", "Bioprothèse valvulaire aortique Inspiris #27."),
                 ("mitral", "Prothèse valvulaire mitrale mécanique. Fonction normale de la prothèse."),
                 ("pulmonic", "Homogreffe pulmonaire."), ("pulmonic", "Statut post valvuloplastie pulmonaire."),
                 ("tricuspid", "Statut post remplacement valvulaire tricuspidien biologique.")]
        for structure, caption in cases:
            with self.subTest(caption):
                result = assess(structure, caption)
                self.assertEqual((result.flag, result.unrecognized), ("abnormal", ()), result.reasons)

    def test_a_modifier_for_devices_makes_nothing_normal(self):
        self.assertEqual(flag("mitral", "Structure valvulaire normale"), "unknown")

    def test_ivc_compliance_in_words(self):
        for caption in ["Veine cave inférieure non-dilatée et souple",
                        "Pas de dilatation de la veine cave inférieure, souple"]:
            with self.subTest(caption):
                result = assess("ivc", caption)
                self.assertEqual((result.flag, result.unrecognized), ("normal", ()), result.reasons)
        for caption in ["Veine cave inférieure dilatée et souple", "Veine cave inférieure dilatée et peu souple",
                        "Veine cave inférieure dilatée et fixe"]:
            with self.subTest(caption):
                self.assertEqual(flag("ivc", caption), "abnormal")
        stiff = assess("ivc", "Veine cave inférieure non dilatée et peu souple")
        self.assertEqual(stiff.flag, "unknown")              # `et` after a negation: an abnormal claim stays unsure

    def test_an_earlier_study_is_not_read(self):
        result = assess("lv", f"{LV_NORMAL} Amélioration vs examen précédent (alors avec FEVG à 40 % et hypokinésie).")
        self.assertEqual(result.flag, "unknown")
        self.assertTrue(result.unrecognized)
        self.assertNotIn("lv_wall_motion", {f.attribute for f in result.facts})

    def test_a_current_finding_stands_beside_an_earlier_study(self):
        self.assertEqual(flag("lv", "Hypokinésie apicale. Amélioration vs examen précédent."), "abnormal")

    def test_pleural_effusion_is_not_pericardial(self):
        result = assess("pericardium", "Pas d'épanchement péricardique. Présence d'épanchement pleural gauche.")
        self.assertEqual((result.flag, result.unrecognized), ("normal", ()), result.reasons)
        self.assertEqual(assess("pericardium", "Épanchement pleural droit.").flag, "unknown")
        self.assertEqual(flag("pericardium", "Épanchement péricardique léger. Épanchement pleural."), "abnormal")

    def test_doppler_wording_leaves_the_caption(self):
        for caption in ["L'ERO est calculé à 0,30 cm2", "Inversion du flot systolique de la veine sus-hépatique",
                        "Flot veineux pulmonaire normal", "Sans reflux au niveau des veines pulmonaires",
                        "Renversement systolique occasionnel"]:
            with self.subTest(caption):
                self.assertEqual(rn.retain_caption(caption)[0], "")
        kept = rn.retain_caption("Valve tricuspide de structure normale. L'ERO est calculé à 0,30 cm2.")[0]
        self.assertEqual(kept, "Valve tricuspide de structure normale.")


class TestPericardiumTemplates(unittest.TestCase):

    def test_epicardial_fat_is_neutral(self):
        result = assess("pericardium", "Pas d'épanchement péricardique. Graisse épicardique.")
        self.assertEqual((result.flag, result.unrecognized), ("normal", ()), result.reasons)
        self.assertEqual(flag("pericardium", "Épanchement péricardique. Graisse épicardique."), "abnormal")

    def test_fat_says_nothing_about_an_effusion(self):
        result = assess("pericardium", "Graisse épicardique.")
        self.assertEqual(result.flag, "unknown")
        self.assertIn("missing_required:pericardial_effusion", result.reasons)

    def test_denying_fat_is_no_finding(self):
        result = assess("pericardium", "Pas d'épanchement péricardique. Pas de graisse épicardique.")
        self.assertEqual(result.flag, "normal", result.reasons)

    def test_hedged_fat_is_held_back(self):
        result = assess("pericardium", "Pas d'épanchement péricardique. Probable graisse épicardique.")
        self.assertEqual(result.flag, "unknown")


class TestLvScope(unittest.TestCase):

    def test_what_only_doppler_shows_of_the_lv_leaves_the_caption(self):
        for caption in ["Profil pseudo-normal", "Profil restrictif", "Fusion des ondes E et A",
                        "Le débit cardiaque indexé est calculé à 3 L/min/m2", "Index cardiaque 3 L/min/m2",
                        "Sans obstruction sous-aortique", "Pas d'obstruction au repos ni avec manoeuvre de Valsalva"]:
            with self.subTest(caption):
                self.assertEqual(rn.retain_caption(caption)[0], "")

    def test_a_statement_stays_when_only_what_it_leads_to_goes(self):
        for cell, kept in {
            "Prothèse biologique Inspiris 23: gradient moyen de 11 mmHg": "Prothèse biologique Inspiris 23.",
            "Sclérose aortique / sténose légère": "Sclérose aortique.",
            "Léger SAM entraînant un gradient de 20 mmHg": "Léger SAM.",
            "Bourrelet septal de 13 mm qui ne cause aucune obstruction": "Bourrelet septal de 13 mm.",
            "Bourrelet septal de 12 mm ne causant pas d'obstruction": "Bourrelet septal de 12 mm.",
            "Hypertrophie en partie due à une sténose": "Hypertrophie.",
            "Hypertrophie responsable d'une obstruction": "Hypertrophie.",
            "Fonction systolique normale : TAPSE 14": "Fonction systolique normale.",
            "Fonction systolique normale suggérant une pression élevée": "Fonction systolique normale.",
            "Masse intracavitaire responsable d'un gradient": "Masse intracavitaire.",
            "Prothèse : bien fixée : gradient de 8 mmHg": "Prothèse : bien fixée.",
            "Calcification responsable de sténose: gradient de 30 mmHg": "Calcification.",
        }.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), (kept, 1, 0))

    def test_a_kept_statement_does_not_excuse_what_left_after_it(self):
        cell = "Septum aplati en systole suggérant une surcharge de pression"
        self.assertEqual(rn.retain_caption(cell), ("Septum aplati en systole.", 1, 1))

    def test_a_state_or_a_chamber_is_only_explained_by_its_pressure(self):
        for cell in ["Ventricule droit normal au Doppler", "Augmentation du ventricule droit au Doppler",
                     "Oreillette gauche au Doppler", "Normale au Doppler", "Aorte descendante de 45 mm au Doppler",
                     "Pression systolique probablement normale", "Pression systolique basse",
                     "Limite au Doppler"]:
            with self.subTest(cell):
                self.assertGreater(rn.retain_caption(cell)[2], 0)
        for cell in ["Pression systolique du ventricule droit normale", "Pression de remplissage du VG élevée",
                     "Pression systolique ventriculaire droite légèrement augmentée",
                     "Élévation modérée de la pression systolique du ventricule droit",
                     "Pression systolique du ventricule droit limite", "Pressions pulmonaires normales"]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), ("", 1, 0))

    def test_a_word_that_says_something_is_never_neutral(self):
        self.assertEqual([word for word in rn._NEUTRAL if rn._stands_alone(word)], [])
        states = (rn._NORMAL_STATE, rn._ABNORMAL_STATE)
        self.assertEqual([word for word in rn._NEUTRAL if any(p.search(word) for p in states)], [])
        subjects = [pattern for grammar in rn.GRAMMARS.values() for _, pattern in grammar.subjects]
        self.assertEqual([word for word in rn._NEUTRAL if any(p.search(word) for p in subjects)], [])
        for word in ("normal", "normale", "normales", "anormale", "augmentation", "augmentee", "elevation", "elevee",
                     "diminuee", "limite", "basse", "ventricule", "droit", "droite", "gauche", "oreillette", "aorte",
                     "descendante", "ascendante", "cavite", "paroi", "septum", "masse", "tumeur", "thrombus", "kyste"):
            with self.subTest(word):
                self.assertEqual(rn.retain_caption(f"Gradient {word}"), ("", 1, 1))

    def test_a_value_is_explained_and_a_word_with_digits_is_not(self):
        for cell in ["Gradient de 32+8 mmHg", "Gradient à 24mmHg", "Gradient de 1,5 cm", "Gradient 10-12 mm",
                     "Jet 1-2/4"]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), ("", 1, 0))
        self.assertEqual(rn.retain_caption("Gradient sur P2")[2], 1)

    def test_a_pronoun_is_no_statement_to_keep(self):
        for cell in ["Ceci entraîne un gradient de 30 mmHg", "Cela cause une insuffisance modérée"]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), ("", 1, 0))

    def test_what_follows_a_kept_statement_does_not_run_into_it(self):
        caption, _, _ = rn.retain_caption("Prothèse Edwards 20: gradient moyen de 39 mmHg (feuillets calcifiés)")
        self.assertEqual(caption, "Prothèse Edwards 20, feuillets calcifiés.")

    def test_a_segment_is_only_cut_after_a_statement_free_of_doppler(self):
        for cell, stranded in {"Valve aortique: gradient moyen de 5 mmHg": 0,      # a bare label says nothing
                               "Gradient moyen: 5 mmHg": 0,
                               "Sténose: valve calcifiée": 1,                     # the finding comes after the Doppler
                               "Gradient de 5 mmHg entraînant une calcification": 1,
                               "Diamètre diastolique augmenté au Doppler": 1}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), ("", 1, stranded))

    def test_the_sinuses_of_valsalva_stay(self):
        caption = "Pas de dilatation de l'aorte au niveau des sinus de Valsalva"
        self.assertEqual(rn.retain_caption(caption), (f"{caption}.", 0, 0))


class TestNegationScope(unittest.TestCase):
    """A negation denies one claim. What else it could reach is uncertain, never denied along with it."""

    def test_what_follows_a_denied_claim_is_not_denied(self):
        for structure, caption in [("mitral", "Valve mitrale non calcifiée à ouverture réduite."),
                                   ("lv", "VG non dilaté à fonction systolique altérée."),
                                   ("lv", "VG non dilaté : fonction systolique altérée."),
                                   ("lv", "VG non dilaté à parois épaissies. Fonction systolique normale."),
                                   ("lv", "VG non dilaté qui est hypertrophié. Fonction systolique normale."),
                                   ("rv", "VD non dilaté de fonction diminuée."),
                                   ("lv", "VG dilaté : thrombus absent. Fonction systolique normale."),
                                   ("lv", "VG non dilaté à fonction systolique conservée."),
                                   ("ivc", "VCI non dilatée souple")]:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "unknown")

    def test_one_claim_is_still_denied(self):
        for structure, caption in [("pericardium", "Pas d'épanchement péricardique."), ("lv", f"{LV_NORMAL}"),
                                   ("mitral", "Valve mitrale non calcifiée."),
                                   ("lv", "VG non dilaté, sans hypertrophie. Fonction systolique normale.")]:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "normal")

    def test_a_denial_after_two_claims_denies_the_nearer_one(self):
        for caption, expected in [("Dilatation hypertrophie absente.",
                                   {"lv_size": "uncertain", "lv_wall_thickness": "normal"}),
                                  ("VG dilaté thrombus absent.", {"lv_size": "uncertain", "lv_thrombus": "normal"})]:
            with self.subTest(caption):
                facts = rn.read_caption("lv", caption, POLICY)[0]
                self.assertEqual({fact.attribute: fact.value for fact in facts}, expected)

    def test_a_segment_without_a_finding_closes_the_list_before_it(self):
        facts = rn.read_caption("lv", "Pas de thrombus, VG 50 mm, hypertrophie.", POLICY)[0]
        self.assertEqual({fact.attribute: fact.value for fact in facts},
                         {"lv_thrombus": "normal", "lv_wall_thickness": "abnormal"})

    def test_a_statement_of_the_ef_is_not_a_member_of_the_list_before_it(self):
        facts = rn.read_caption("lv", "Pas de thrombus, dilatation FEVG 60 %.", POLICY)[0]
        self.assertEqual({fact.attribute: fact.value for fact in facts},
                         {"lv_thrombus": "normal", "lv_size": "abnormal", "lv_function": "normal"})

    def test_a_denial_after_its_finding_does_not_stand_for_a_confirmed_finding(self):
        for caption in ["Épanchement péricardique : aucun.", "Épanchement péricardique : non.",
                        "Épanchement péricardique (-).", "Épanchement péricardique : 0.",
                        "Épanchement péricardique 0 mm.", "Épanchement et tamponnade absents."]:
            with self.subTest(caption):
                self.assertEqual(flag("pericardium", caption), "unknown")

    def test_a_number_grades_a_finding_whatever_its_digits(self):
        for value in ("2", "22", "12 mm", "2,5 cm", "0,2 cm", "10 %"):
            with self.subTest(value):
                self.assertEqual(flag("pericardium", f"Épanchement péricardique {value}."), "abnormal")

    def test_a_list_after_a_denial_and_a_doubt_over_et_are_not_confirmed(self):
        for caption in ["Dilatation et hypertrophie absentes.", "Sans dilatation, hypertrophie ou thrombus.",
                        "Suspicion de thrombus et d'anévrisme."]:
            with self.subTest(caption):
                self.assertEqual(flag("lv", caption), "unknown")

    def test_an_ef_after_a_denial_is_its_own_claim(self):
        self.assertEqual(flag("lv", "VG non dilaté, FEVG 60 %"), "normal")

    def test_a_denied_adjective_opens_no_list(self):
        self.assertEqual(flag("lv", "VG non dilaté, HVG légère. Fonction systolique normale."), "abnormal")
        self.assertEqual(flag("lv", "VG non dilaté, hypokinésie apicale."), "abnormal")
        self.assertEqual(flag("lv", "Ventricule gauche non-dilaté, masse et fonction systolique normale"), "normal")


class TestDevices(unittest.TestCase):

    def test_a_denial_of_something_else_keeps_the_device(self):
        for structure, caption in [("mitral", "Pas de végétation sur la prothèse mitrale."),
                                   ("aortic", "Pas d'anomalie de la bioprothèse aortique."),
                                   ("aortic", "Pas de calcification du TAVI."),
                                   ("tricuspid", "Pas de végétation sur le TriClip."),
                                   ("aortic", "Bioprothèse aortique sans anomalie.")]:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "abnormal")

    def test_a_denial_of_the_device_itself_stays_one(self):
        self.assertNotEqual(flag("mitral", "Valve mitrale normale, pas de prothèse."), "abnormal")


class TestWhatAClaimCovers(unittest.TestCase):

    def test_a_claim_on_a_part_does_not_state_the_rest(self):
        for structure, caption in [("lv", "Pas d'anomalie de la fonction systolique."),
                                   ("lv", "Pas d'anomalie de la paroi."), ("mitral", "Pas d'anomalie de la mobilité."),
                                   ("lv", "WMSI normal."), ("lv", "Diastole normale."),
                                   ("mitral", "sans végétation."), ("aortic", "Pas de calcification."),
                                   ("lv", "Diastole altérée.")]:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), "unknown")

    def test_a_denial_of_a_named_valve_states_it(self):
        self.assertEqual(flag("mitral", "Valve mitrale non calcifiée."), "normal")
        self.assertEqual(flag("mitral", "Valve mitrale normale. Pas de végétation."), "normal")

    def test_a_state_is_not_carried_onto_a_reference_to_a_diagram(self):
        caption = "VG non dilaté. Fonction systolique normale et cinétique segmentaire : voir schéma."
        self.assertEqual(flag("lv", caption), "unknown")

    def test_a_low_strain_cannot_stand_beside_normal_wording(self):
        self.assertEqual(flag("lv", f"{LV_NORMAL} Strain normal à -8 %."), "unknown")
        self.assertEqual(flag("lv", f"{LV_NORMAL} Strain normal à -22 %."), "normal")

    def test_spellings(self):
        for structure, caption, expected in [("mitral", "Feuillets normaux.", "normal"),
                                             ("lv", "Apex akinétique.", "abnormal"),
                                             ("lv", "Parois hypokinétiques.", "abnormal"),
                                             ("lv", "Anévrysme apical.", "abnormal")]:
            with self.subTest(caption):
                self.assertEqual(flag(structure, caption), expected)

    def test_text_that_is_not_a_report_does_not_stop_the_build(self):
        for caption in ["efvalue0", "VG efvalue3 normal"]:
            self.assertEqual(flag("lv", caption), "unknown")
            self.assertEqual(flag("rv", caption), "unknown")


class TestBoundaries(unittest.TestCase):

    def retained(self, text):
        return rn.retain_caption(text)[0]

    def test_a_line_break_ends_a_clause_however_it_is_written(self):
        for mark in ("_x000D_", "_x000B_", "_x000a_", "\x0b", "\r", "\n", "\u2028"):
            with self.subTest(mark):
                caption = self.retained(f"VG non dilaté{mark}Fonction systolique diminuée")
                self.assertEqual(caption, "VG non dilaté. Fonction systolique diminuée.")
                self.assertEqual(flag("lv", caption), "abnormal")

    def test_an_emptied_parenthesis_leaves_its_sides_apart(self):
        self.assertEqual(self.retained("VG non dilaté (gradient 5 mmHg) fonction altérée."),
                         "VG non dilaté, fonction altérée.")
        self.assertEqual(flag("lv", "VG non dilaté, fonction altérée."), "abnormal")

    def test_every_caption_comes_back_in_one_form(self):
        self.assertEqual(self.retained("Aorte normale.. // VCI  normale ;c.f. schéma"),
                         "Aorte normale. VCI normale. cf schéma.")
        self.assertEqual(self.retained("VG dilaté. _x000B_Fraction normale"), "VG dilaté. Fraction normale.")
        self.assertEqual(self.retained("VG dilaté. nan"), "VG dilaté.")

    def test_a_clause_without_a_word_says_nothing_it_can_be_read_for(self):
        self.assertEqual(rn.retain_caption("Valve normale. 1-2/4"), ("Valve normale.", 1, 1))
        self.assertEqual(rn.retain_caption("1-2/4"), ("", 1, 1))

    def test_dates_and_names_leave_with_their_segment(self):
        kept = "VG non dilaté."
        for clause in ["Stable depuis 2019", "Comparé au 12/05/2020", "Examen du 27 février", "Avis du Dr. Tremblay",
                       "Discuté de la Dre Roy", "Avis du docteur"]:
            with self.subTest(clause):
                self.assertEqual(rn.retain_caption(f"{kept} {clause}."), (kept, 1, 1))
        self.assertEqual(rn.retain_caption("Jet 1-2/4")[2], 0)        # a grade is no date

    def test_what_a_date_sits_beside_stays(self):
        self.assertEqual(rn.retain_caption("Bioprothèse valvulaire aortique (#21, 2018)"),
                         ("Bioprothèse valvulaire aortique (#21).", 1, 0))
        caption, _, stranded = rn.retain_caption("Dilatation modérée de l'aorte (48 mm versus 47 mm en 2019)")
        self.assertEqual(caption, "Dilatation modérée de l'aorte.")
        self.assertEqual(flag("aorta", caption), "abnormal")
        self.assertEqual(stranded, 1)

    def test_a_dangling_connector_goes(self):
        self.assertEqual(self.retained("Hyperdynamisme ventriculaire gauche, mais gradient de 5 mmHg"),
                         "Hyperdynamisme ventriculaire gauche.")


class TestCoordinationAndDenial(unittest.TestCase):
    """What survives removal must mean what it meant beside the removed part."""

    def test_what_a_removed_part_leaves_behind_is_tidied(self):
        for cell, kept in [("Valve calcifiée et (fuite légère)", "Valve calcifiée."),
                           ("Valve calcifiée avec (fuite légère)", "Valve calcifiée."),
                           ("Valve calcifiée (fuite) , sans thrombus", "Valve calcifiée, sans thrombus."),
                           ("Valve calcifiée , sans thrombus, fuite légère", "Valve calcifiée, sans thrombus.")]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)

    def test_a_pressure_in_any_spelling_of_its_unit_is_doppler(self):
        for part in ("à 24 mmHg", "à 24mmHg", "à 24 mm Hg", "à 24mm Hg"):
            with self.subTest(part):
                self.assertTrue(rn._names_doppler(part))
        self.assertFalse(rn._names_doppler("à 24 mmol"))

    def test_a_pressure_quoted_with_the_central_venous_pressure_leaves_whole(self):
        caption, dropped, stranded = rn.retain_caption(
            "VD non dilaté. Pression systolique du ventricule droit évaluée à 30 mmHg + TVC")
        self.assertEqual((caption, dropped, stranded), ("VD non dilaté.", 1, 0))

    def test_a_subject_waiting_for_its_state_goes_even_when_the_state_is_singular(self):
        self.assertEqual(rn.retain_caption("Dimension et pression systolique augmentée, fonction normale")[0],
                         "fonction normale.")

    def test_a_qualifier_of_the_removed_part_does_not_reach_a_member_already_judged(self):
        for cell, kept in [("Valve normale et fuite absentes.", "Valve normale."),
                           ("Fonction altérée et fuite à confirmer.", "Fonction altérée.")]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)

    def test_a_doubt_at_the_end_does_not_leave_the_finding_before_it_confirmed(self):
        for cell in ["Calcification et sténose probables.", "Calcification et sténose probable.",
                     "Prolapsus et fuite possibles.", "Hypertrophie et obstruction suspectes.",
                     "Thrombus et fuite à éliminer.", "Thrombus ou fuite à exclure.",
                     "Végétation et fuite à préciser.",
                     "Végétation et fuite douteuses.", "Thrombus et fuite à discuter."]:
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual(caption, "")
                self.assertGreater(stranded, 0)

    def test_a_doubt_on_the_removed_finding_alone_stays_with_it(self):
        caption, _, _ = rn.retain_caption("Calcification et probable sténose.")
        self.assertEqual(caption, "Calcification.")
        caption, _, _ = rn.retain_caption("Probable calcification et sténose.")
        self.assertEqual(caption, "Probable calcification.")

    def test_a_member_that_needed_the_removed_one_goes_with_it(self):
        for cell in ["Dimension et pression systolique augmentées, fonction normale",
                     "Ventricule droit de dimension et pression systolique augmentées, fonction normale"]:
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual(caption, "fonction normale.")
                self.assertGreater(stranded, 0)
        for cell in ["Calcification et sténose absentes", "Thrombus et obstruction non retrouvés",
                     "Végétation et fuite à confirmer"]:
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual(caption, "")
                self.assertGreater(stranded, 0)

    def test_a_member_with_its_own_state_stays(self):
        for cell, kept in {"Fonction normale et pression normale": "Fonction normale.",
                           "Valve mitrale normale et fuite minime": "Valve mitrale normale.",
                           "Feuillets fins et fuite minime": "Feuillets fins.",
                           "Valve calcifiée et fuite légère": "Valve calcifiée."}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)

    def test_a_list_keeps_the_denial_that_left(self):
        for cell in ["Pas de sténose, calcification ou prolapsus", "Sans obstruction, thrombus ni anévrisme"]:
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual(caption, "")
                self.assertGreater(stranded, 0)
        for cell, kept in {"Pas de sténose, mais calcification": "mais calcification.",
                           "Pas de sténose, pas de calcification": "pas de calcification.",
                           "Pas de sténose, fonction normale": "fonction normale."}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)
        self.assertEqual(rn.retain_caption("Valve non sténosante, calcifiée")[0], "calcifiée.")    # `non`: no list

    def test_a_denial_narrowed_by_doppler_wording_is_not_widened(self):
        for cell in ["Pas de calcification responsable de sténose", "Absence de calcification entraînant une sténose"]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), ("", 1, 1))
        self.assertEqual(rn.retain_caption("Pas de calcification: gradient de 5 mmHg")[0], "Pas de calcification.")

    def test_central_venous_pressure_stays_as_the_policy_says(self):
        for cell in ["VCI non dilatée. Tension veineuse centrale estimée à 15 mmHg",
                     "VCI non dilatée. TVC 15 mmHg"]:
            with self.subTest(cell):
                caption, dropped, _ = rn.retain_caption(cell)
                self.assertEqual((caption, dropped), (f"{cell}.", 0))
                self.assertEqual(flag("ivc", caption), "unknown")
        self.assertGreater(rn.retain_caption("VCI non dilatée. Pas de variations respiratoires du flux")[2], 0)

    def test_what_only_sounds_like_doppler_stays(self):
        for cell in ["Images de qualité insuffisante pour le strain", "HTA connue"]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), (f"{cell}.", 0, 0))

    def test_a_pressure_value_with_no_subject_is_not_accounted_for(self):
        self.assertGreater(rn.retain_caption("VCI non dilatée, 15 mmHg")[2], 0)
        self.assertEqual(rn.retain_caption("Pression systolique normale (30 mmHg)")[2], 0)

    def test_doppler_written_other_ways_leaves_the_caption(self):
        for cell in ["IAo minime", "RAo serré", "RM serré", "Valve mitrale fuyante", "Valve aortique insuffisante",
                     "Pas d'HTAP", "Pas d'HTP", "Hypertension artérielle pulmonaire sévère", "PAPm 30", "TRIV 80 ms",
                     "E/Ea 12",
                     "Gradient à 24mmHg", "PAPS 40 mm Hg", "S' 12", "Vélocité maximale 3 m/s"]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")


class TestRecoveredFindings(unittest.TestCase):
    """What a Doppler statement says of the valve in 2D stays; its claim, its grade and its date do not."""

    def test_what_a_statement_rests_on_stays_as_written(self):
        for cell, kept in {
                "Sténose modérée sur valve aortique tricuspide calcifiée": "sur valve aortique tricuspide calcifiée.",
                "Insuffisance mitrale secondaire à léger prolapsus de A2": "secondaire à léger prolapsus de A2.",
                "Insuffisance tricuspidienne sévère par dilatation de l'anneau.": "par dilatation de l'anneau.",
                "Fuite via restriction du feuillet par la sonde": "via restriction du feuillet par la sonde.",
                "Insuffisance mitrale associée au SAM": "associée au SAM.",
                "Accélération du à un SAM de cordage": "du à un SAM de cordage.",
                "Sténose au niveau de la bioprothèse mitrale": "au niveau de la bioprothèse mitrale.",
                "Fuite sur défaut de coaptation": "sur défaut de coaptation."}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), (kept, 1, 0))

    def test_a_doubt_on_what_it_rests_on_stays_with_it(self):
        for cell, kept in {
                "Insuffisance mitrale sur possible prolapsus": "sur possible prolapsus.",
                "Fuite probablement secondaire à l'électrode": "probablement secondaire à l'électrode."}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)
        self.assertEqual(rn.assess("mitral", "sur possible prolapsus.", POLICY).flag, "unknown")

    def test_a_tail_goes_on_past_a_colon_only_as_far_as_the_finding(self):
        cell = "Sténose aortique modérée sur une valve aortique calcifiée: gradient moyen de 30 mmHg"
        self.assertEqual(rn.retain_caption(cell)[0], "sur une valve aortique calcifiée.")

    def test_no_normal_claim_comes_out_of_a_statement(self):
        for cell in ("Insuffisance mitrale sur valve normale", "Insuffisance mitrale sur valve non calcifiée",
                     "Pas d'insuffisance sur valve calcifiée",
                     "Pas de sténose sur valve calcifiée", "Insuffisance mitrale sur feuillet normal",
                     "Non sténosante sur valve calcifiée", "Insuffisance mitrale sur bioprothèse normale",
                     "Insuffisance mitrale sur anneau normal"):
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual(caption, "")
                self.assertGreater(stranded, 0)

    def test_what_a_statement_rests_on_must_be_the_valves(self):
        for cell in ("Insuffisance mitrale légère sur dilatation de l'oreillette gauche",
                     "Insuffisance tricuspidienne sévère secondaire à la dilatation du diamètre basal du VD"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")

    def test_a_finding_glued_to_the_doppler_noun_takes_the_valve_for_noun(self):
        for cell, kept in {
                "Sténose aortique calcifiée modérée.": "Valve aortique calcifiée.",
                "Sténose valvulaire aortique calcifiée sévère": "Valve aortique calcifiée.",
                "Sténose aortique valvulaire calcifiée légère à modérée": "Valve aortique calcifiée.",
                "Sténose aortique sévère calcifiée": "Valve aortique calcifiée.",
                "Rétrécissement aortique calcifié": "",
                "Sténose mitrale rhumatismale modérée à sévère": "Valve mitrale rhumatismale.",
                "Sténose mitrale dégénérative d'allure modérée": "Valve mitrale dégénérative.",
                "Sténose aortique modérée dégénérative": "Valve aortique dégénérative.",
                "Sténose valvulaire aortique calcifiée modérée au Doppler": "Valve aortique calcifiée.",
                "Sténose aortique calcifiée modérée valve aortique tricuspide": (
                    "Valve aortique calcifiée, valve aortique tricuspide.")}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)

    def test_the_valve_takes_the_nouns_place_only_in_a_plain_statement(self):
        self.assertEqual(rn.retain_caption("VG dilaté avec sténose aortique sévère calcifiée")[0],
                         "VG dilaté avec valve aortique calcifiée.")
        for cell in ("Sténose aortique calcifiée probable.", "Pas de sténose aortique calcifiée.",
                     "Sténose aortique calcifiée à confirmer", "Sténose tricuspide calcifiée"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")

    def test_a_glued_finding_counts_as_a_part_that_left(self):
        self.assertEqual(rn.retain_caption("Valve tricuspide. Sténose aortique calcifiée sévère"),
                         ("Valve tricuspide. Valve aortique calcifiée.", 1, 0))

    def test_a_procedure_loses_only_its_date(self):
        for cell, kept in {
                "Plastie mitrale du 26 août 2019. Valve fine.": "Plastie mitrale. Valve fine.",
                "Statut post-plastie mitrale chirurgicale 2018": "Statut post-plastie mitrale chirurgicale.",
                "Post plastie mitrale chirurgicale du 12/03/2018": "Post plastie mitrale chirurgicale.",
                "RVA CM25 en 2012": "RVA CM25.", "Bioprothèse depuis 2015": "Bioprothèse.",
                "Myomectomie septale en mars 2014": "Myomectomie septale.",
                "Post-greffe cardiaque en 2017": "Post-greffe cardiaque.",
                "Valvuloplastie 2011": "Valvuloplastie."}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)

    def test_any_word_after_a_date_takes_the_segment_in_any_spelling(self):
        for procedure in ("Bioprothèse mitrale", "Plastie mitrale", "RVA CM25"):
            for word in ("absente", "non implantée", "annulée", "laval", "par martin", "à laval", "au chum",
                         "explantée", "à reprendre", "retirée", "reportée", "non réalisée", "refusée"):
                for joint in (" ", ", ", " (", ": "):
                    cell = f"{procedure} en 2019{joint}{word}"
                    for spelling in (cell, f"{procedure} en 2019{joint}{word.title()}", cell.upper(), cell.title()):
                        with self.subTest(spelling):
                            caption, _, stranded = rn.retain_caption(spelling)
                            self.assertEqual(caption, "")
                            self.assertGreater(stranded, 0)

    def test_a_date_alone_leaves_the_procedure_in_any_spelling(self):
        for cell in ("Plastie mitrale en 2019", "Plastie mitrale du 12/03/2019.", "Plastie mitrale en 2019 ,"):
            for spelling in (cell, cell.upper(), cell.title()):
                with self.subTest(spelling):
                    caption, _, stranded = rn.retain_caption(spelling)
                    self.assertEqual((caption.lower(), stranded), ("plastie mitrale.", 0))

    def test_a_date_before_its_procedure_takes_the_segment(self):
        for cell in ("En 2019 plastie mitrale", "Examen du 12/03/2019 prothèse mitrale"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")

    def test_a_date_on_anything_else_still_takes_its_segment(self):
        for cell in ("FEVG 30 % avant remplacement en 2015", "FE stable depuis plastie en 2018",
                     "Hypokinésie apicale notée à l'examen du 12/03/2019",
                     "Plastie mitrale du docteur Martin en 2019"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")

    def test_a_procedure_with_a_doppler_statement_keeps_its_rule(self):
        self.assertEqual(rn.retain_caption("Fuite sur bioprothèse en 2018")[0], "")

    def test_a_state_moves_only_through_an_explicit_et(self):
        for subjects in ("Ventricule droit de dimension et fonction", "Dimensions et fonction systolique"):
            for joint in (" (pression systolique normale)", ", pression systolique normale",
                          ": pression systolique normale", " - pression systolique normale",
                          " / pression systolique normale",
                          " (pressions systoliques normales)", ", pressions systoliques normales"):
                with self.subTest(subjects + joint):
                    caption, _, stranded = rn.retain_caption(subjects + joint)
                    self.assertEqual(caption, "")
                    self.assertGreater(stranded, 0)
        for cell in ("Fonction systolique (diastolique normale)", "Fonction systolique, diastolique normale",
                     "Ventricule droit de dimension et fonction (et pression systolique normale)",
                     "Ventricule droit de dimension et fonction: et pression systolique normale",
                     "Ventricule droit de dimension et fonction - et pression systolique normale",
                     "Dimensions et fonction systolique: et pression pulmonaire normale"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")
        caption = rn.retain_caption("Ventricule droit (dimension et fonction et pression systolique normale)")[0]
        self.assertNotIn("normale", caption)

    def test_a_state_moves_only_onto_bare_subjects(self):
        for cell in ("Pas d'évaluation de la fonction et pression systolique normale",
                     "Impossible d'évaluer la fonction et pression systolique normale",
                     "Difficile d'évaluer la fonction et la pression systolique normale",
                     "Diminution de la fonction et pression systolique normale",
                     "Diminution de la fonction et pressions systoliques normales",
                     "Mauvaise fonction et pression systolique normale",
                     "Altération de la fonction systolique et diastolique normale",
                     "Ventricule droit dilaté de dimension et fonction et pression systolique normale"):
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual(caption, "")
                self.assertGreater(stranded, 0)

    def test_a_genuine_list_still_shares_its_state(self):
        for cell, kept in [
            ("Ventricule droit de dimension, fonction et pression systolique normale",
             "Ventricule droit de dimension, fonction normale."),
            ("Ventricule droit de dimension (aire 20 cm2), fonction et pression systolique normale",
             "Ventricule droit de dimension (aire 20 cm2), fonction normale."),
            ("Ventricule droit de dimension augmentée, fonction et pression systolique normale",
             "Ventricule droit de dimension augmentée, fonction normale."),
            ("Ventricule droit de dimension (aire 20 cm2) et fonction et pression systolique normale",
             "Ventricule droit de dimension (aire 20 cm2) et fonction normale."),
            ("Fonction et pression systolique normale du ventricule droit", "Fonction normale du ventricule droit."),
        ]:
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), (kept, 1, 0))

    def test_a_procedure_that_may_not_have_happened_leaves_whole(self):
        for procedure in ("Bioprothèse mitrale", "Plastie mitrale", "RVA CM25"):
            for date in (" en 2019", " du 12/03/2019"):
                for rest in (" non implantée", ", non implantée", " (non implantée)", " mais non réalisée",
                             ": non réalisée", " annulée", ", annulée", " Annulée", " prévue", " à discuter", " ?",
                             " probable",
                             " refusée par la patiente", " pour prolapsus", " Non réalisée", " Non Réalisée",
                             ", non réalisée à cause de la sténose"):
                    cell = procedure + date + rest
                    with self.subTest(cell):
                        caption, _, stranded = rn.retain_caption(cell)
                        self.assertEqual(caption, "")
                        self.assertGreater(stranded, 0)

    def test_what_stands_by_itself_after_a_date_lets_the_procedure_stay(self):
        for cell in ("Plastie mitrale en 2019, pas de fuite", "Plastie mitrale en 2019 sans fuite"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "Plastie mitrale.")
        self.assertEqual(rn.retain_caption("Plastie mitrale en 2019, valve calcifiée")[0],
                         "Plastie mitrale, valve calcifiée.")

    def test_a_procedure_keeps_a_qualifier_that_comes_before_its_date(self):
        for cell, kept in {"Plastie mitrale non réalisée en 2019": "Plastie mitrale non réalisée.",
                           "Plastie mitrale prévue en 2019": "Plastie mitrale prévue."}.items():
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], kept)

    def test_only_a_plain_statement_leaves_what_it_rests_on_standing(self):
        for tail in ("sur prolapsus", "par dilatation de l'anneau", "secondaire à la sonde", "sur valve calcifiée"):
            for head in ("À confirmer insuffisance mitrale", "Probable insuffisance mitrale",
                         "Probablement insuffisance mitrale", "Suspicion d'insuffisance mitrale",
                         "Insuffisance mitrale probable", "Insuffisance mitrale ?", "Insuffisance mitrale à confirmer",
                         "Si insuffisance mitrale", "Antécédent d'insuffisance mitrale",
                         "Rechercher une insuffisance mitrale",
                         "Pour éliminer une insuffisance mitrale", "Semble présenter une insuffisance mitrale",
                         "Non sténosante", "Pas d'insuffisance mitrale"):
                cell = f"{head} {tail}"
                with self.subTest(cell):
                    caption, _, stranded = rn.retain_caption(cell)
                    self.assertEqual(caption, "")
                    self.assertGreater(stranded, 0)

    def test_a_doubt_reaches_across_a_comma_to_what_rests_on_it(self):
        caption, _, stranded = rn.retain_caption("Insuffisance mitrale possible, sur prolapsus")
        self.assertEqual((caption, stranded > 0), ("", True))

    def test_a_doubt_on_a_statement_leaves_a_finding_stated_elsewhere(self):
        for cell, kept in {"Probable insuffisance mitrale. Prolapsus du feuillet postérieur.":
                           "Prolapsus du feuillet postérieur.",
                           "Probable insuffisance mitrale sur prolapsus. Insuffisance mitrale sur prolapsus.":
                           "sur prolapsus."}.items():
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual((caption, stranded), (kept, 1))

    def test_a_definite_statement_still_leaves_what_it_rests_on(self):
        for cell in ("Insuffisance mitrale sur prolapsus", "Insuffisance mitrale légère sur prolapsus",
                     "Insuffisance mitrale excentrique sur prolapsus"):
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual((caption, stranded), ("sur prolapsus.", 0))
                self.assertEqual(rn.assess("mitral", caption, POLICY).flag, "abnormal")

    def test_a_segment_that_is_only_a_doubt_puts_its_clause_in_doubt(self):
        for structure, caption in [("mitral", "Prolapsus, à confirmer."), ("mitral", "À confirmer, prolapsus."),
                                   ("mitral", "Prolapsus, probable."),
                                   ("aortic", "Valve aortique calcifiée, à confirmer."),
                                   ("aortic", "Valve aortique calcifiée, probable."), ("mitral", "sur prolapsus, ?")]:
            with self.subTest(caption):
                self.assertEqual(rn.assess(structure, caption, POLICY).flag, "unknown")
        for caption in ("Prolapsus. Calcification à confirmer.", "Prolapsus, calcification à confirmer."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("mitral", caption, POLICY).flag, "abnormal")

    def test_the_whole_parse_is_cached_for_repeated_captions(self):
        self.assertTrue(hasattr(rn.read_caption, "cache_info"))
        self.assertFalse(hasattr(rn._only_doubt, "cache_info"))
        rn.read_caption.cache_clear()
        first = rn.read_caption("mitral", "Valve mitrale fine.", POLICY)
        self.assertIs(rn.read_caption("mitral", "Valve mitrale fine.", POLICY), first)
        self.assertEqual(rn.read_caption.cache_info().hits, 1)

    def test_a_state_on_a_note_says_nothing_of_the_structure(self):
        for caption in ("Graisse épicardique normale.", "Graisse péricardique normale.", "Épanchement pleural normal.",
                        "Graisse épicardique, normale."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("pericardium", caption, POLICY).flag, "unknown")
        for caption in ("Péricarde normal, graisse épicardique.", "Pas d'épanchement. Graisse épicardique."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("pericardium", caption, POLICY).flag, "normal")

    def test_a_subject_sent_elsewhere_takes_no_state(self):
        for caption in ("Dimensions : voir schéma, fonction systolique normale.",
                        "Fonction systolique : voir schéma, dimensions normales.",
                        "Dimensions : cf schéma, fonction systolique normale."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("lv", caption, POLICY).flag, "unknown")

    def test_a_waiting_subject_takes_a_state_only_along_its_list(self):
        for caption in ("Dimensions, pas de thrombus, fonction normale.", "Dimensions mais fonction normale.",
                        "Dimensions avec fonction normale.", "Dimensions cependant fonction normale.",
                        "Fonction normale, FEVG 60 % et taille.", "Fonction normale, FEVG 60 %, et taille.",
                        # Everything required is stated: only the abandoned subject can hold these back.
                        "Épaisseur mais dimensions et fonction normales.",
                        "Épaisseur avec dimensions et fonction normales.",
                        "Épaisseur, pas de thrombus, dimensions et fonction normales.",
                        "Dimensions et fonction normales, épaisseur mais pas de thrombus.",
                        "Dimensions mais fonction normale, taille normale.",
                        "Dimensions, pas de thrombus, fonction normale, taille normale."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("lv", caption, POLICY).flag, "unknown")
        facts, _ = rn.read_caption("lv", "Dimensions, thrombus, fonction normale.", POLICY)
        self.assertFalse([f for f in facts if f.attribute == "lv_size"])
        for caption in ("Dimensions, épaisseur et fonction normales.", "Dimensions, fonction normale.",
                        "Dimensions et fonction normales.", "Dimension ainsi que fonction normale.",
                        "Ventricule gauche de dimension et de fonction normales."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("lv", caption, POLICY).flag, "normal")

    def test_a_device_stays_only_if_nothing_beside_it_was_lost(self):
        for device in ("Bioprothèse mitrale", "Prothèse mitrale", "Plastie mitrale", "Sonde", "Mitraclip"):
            for rest in (", non implantée en 2019", ", explantée en 2019", " (2019 non implantée)",
                         ", retirée en 2019", " - en 2019 absente", ", en 2019, non implantée",
                         ", non réalisée à cause de la sténose"):
                for spelling in (device + rest, (device + rest).upper()):
                    with self.subTest(spelling):
                        caption, _, stranded = rn.retain_caption(spelling)
                        self.assertEqual(caption, "")
                        self.assertGreater(stranded, 0)
        self.assertEqual(rn.retain_caption("Bioprothèse mitrale, gradient moyen 5 mmHg")[:2],
                         ("Bioprothèse mitrale.", 1))
        self.assertEqual(rn.retain_caption("Bioprothèse mitrale. Mot inconnu sténose")[0], "Bioprothèse mitrale.")

    def test_a_denied_list_takes_its_members_with_it(self):
        for cell in ("Calcification, sténose absentes", "Calcification, prolapsus et sténose absentes",
                     "Calcification et prolapsus et sténose absentes", "Végétation, prolapsus et fuite à confirmer"):
            with self.subTest(cell):
                caption, _, stranded = rn.retain_caption(cell)
                self.assertEqual(caption, "")
                self.assertGreater(stranded, 0)
        self.assertEqual(rn.retain_caption("Calcification, sténose absente")[0], "Calcification.")
        self.assertEqual(rn.retain_caption("Valve normale, sténose absentes")[0], "Valve normale.")
        self.assertEqual(rn.retain_caption("Valve fine et normale, prolapsus et sténose absentes")[0],
                         "Valve fine et normale.")

    def test_a_plural_denial_puts_the_list_before_it_in_doubt(self):
        for caption in ("Dilatation, hypertrophie absentes.", "Dilatation, thrombus et hypertrophie absents."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("lv", caption, POLICY).flag, "unknown")

    def test_every_doubt_word_reads_as_a_doubt_behind_any_connector(self):
        for doubt in ("à confirmer", "à vérifier", "à éliminer", "à exclure", "à écarter", "à préciser", "à discuter"):
            for joint in (", ", ", mais ", " et ", ", est "):
                caption = f"Valve aortique calcifiée{joint}{doubt}."
                with self.subTest(caption):
                    self.assertEqual(rn.assess("aortic", caption, POLICY).flag, "unknown")

    def test_a_following_negation_put_first_is_no_finding(self):
        for caption in ("Exclue dilatation.", "Absente : dilatation."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("lv", caption, POLICY).flag, "unknown")

    def test_a_sign_at_the_end_of_a_clause_is_no_finding(self):
        for caption in ("Épanchement -.", "Épanchement : -.", "Épanchement (--).", "Épanchement (+).",
                        "Épanchement (-).", "Thrombus -."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("pericardium", caption, POLICY).flag, "unknown")
        self.assertEqual(rn.assess("lv", "Dimension normale - fonction normale.", POLICY).flag, "normal")
        self.assertEqual(rn.assess("pericardium", "Épanchement (+), pas de tamponnade.", POLICY).flag, "unknown")

    def test_a_mixed_state_with_no_subject_is_no_finding(self):
        for caption in ("Fonction systolique normale ou légèrement diminuée.",
                        "Fonction systolique normale et diminuée.", "Fonction systolique bonne mais diminuée."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("lv", caption, POLICY).flag, "unknown")
        self.assertEqual(rn.assess("lv", "Fonction normale, dimension augmentée.", POLICY).flag, "abnormal")

    def test_a_carried_negation_denies_one_claim(self):
        facts, _ = rn.read_caption("lv", "Pas de thrombus ou dilatation hypertrophie.", POLICY)
        self.assertFalse([f for f in facts if f.attribute in ("lv_size", "lv_wall_thickness") and f.value == "normal"])

    def test_a_state_is_not_carried_onto_an_ef(self):
        self.assertEqual(rn.assess("lv", "Taille normale et FEVG 40 %.", POLICY).flag, "abnormal")
        facts, _ = rn.read_caption("lv", "Taille augmentée et FEVG 60 %.", POLICY)
        self.assertEqual({f.value for f in facts if f.attribute == "lv_function"}, {"normal"})

    def test_a_narrowed_or_poorly_assessed_ef_is_no_normal(self):
        for caption in ("VG non dilaté. FEVG apicale 60 %.", "VG non dilaté. FEVG mal évaluée à 60 %."):
            with self.subTest(caption):
                self.assertEqual(rn.assess("lv", caption, POLICY).flag, "unknown")

    def test_a_named_colleague_leaves_in_any_spelling(self):
        for cell in ("Avis du dr tremblay", "Avis du Dr.Tremblay", "Avis de M. Tremblay", "Avis du Dr Tremblay",
                     "Avis de Mme Tremblay"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")

    def test_a_short_date_leaves_and_a_measure_stays(self):
        for cell in ("Examen du 12/05", "Examen du 1er mars", "Examen le 3/11"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell)[0], "")
        self.assertEqual(rn.retain_caption("Dimension 47/38 mm")[0], "Dimension 47/38 mm.")

    def test_a_date_counts_the_same_however_it_is_written(self):
        for cell in ("Plastie mitrale du 12/03/2019", "Plastie mitrale du 12 mars 2019",
                     "Plastie mitrale en mars 2019", "Plastie mitrale en 2019", "Plastie mitrale depuis 2019"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), ("Plastie mitrale.", 1, 0))

    def test_a_curly_apostrophe_reads_like_a_straight_one(self):
        for straight in ("Hypertrophie n'entraînant pas de gradient", "Sténose mitrale dégénérative d'allure modérée"):
            with self.subTest(straight):
                self.assertEqual(rn.retain_caption(straight.replace("'", "’"))[0], rn.retain_caption(straight)[0])

    def test_a_septum_flat_in_diastole_is_seen_in_2d(self):
        for cell in ("D-shape proto-diastolique du septum IV", "D shape télé-diastolique du septum",
                     "Aplatissement diastolique du septum"):
            with self.subTest(cell):
                self.assertEqual(rn.retain_caption(cell), (f"{cell}.", 0, 0))
        self.assertEqual(rn.retain_caption("Remplissage diastolique normal")[0], "")

    def test_what_a_structural_finding_may_suggest_goes(self):
        cell = "Léger aplatissement systolique du septum pouvant suggérer une surcharge en pression du VD"
        self.assertEqual(rn.retain_caption(cell)[0], "Léger aplatissement systolique du septum.")

    def test_a_device_loses_its_doppler_adjective_and_a_cause_of_its_implant_goes(self):
        self.assertEqual(rn.retain_caption("Bioprothèse valvulaire pulmonaire légèrement sténosante")[0],
                         "Bioprothèse valvulaire pulmonaire.")
        self.assertEqual(rn.retain_caption("Remplacement valvulaire aortique pour sténose aortique")[0],
                         "Remplacement valvulaire aortique.")
        self.assertEqual(rn.retain_caption("Valve aortique calcifiée non sténosante")[0], "Valve aortique calcifiée.")
        for adverb in ("légèrement", "modérément", "sévèrement"):
            with self.subTest(adverb):
                self.assertEqual(rn.retain_caption(f"Valve remaniée {adverb} sténosante")[0], "Valve remaniée.")
        self.assertEqual(rn.retain_caption("Valve remaniée qui apparaît visuellement modérément sténosante")[0],
                         "Valve remaniée.")

    def test_recovered_text_never_reads_normal(self):
        for structure, caption in [("mitral", "sur dilatation de l'anneau mitral."), ("mitral", "Plastie mitrale."),
                                   ("tricuspid", "via restriction du feuillet septal par la sonde."),
                                   ("aortic", "Valve aortique calcifiée."), ("mitral", "Valve mitrale rhumatismale."),
                                   ("aortic", "sur valve aortique tricuspide calcifiée."),
                                   ("rv", "D-shape proto-diastolique du septum.")]:
            with self.subTest(caption):
                self.assertNotEqual(rn.assess(structure, caption, POLICY).flag, "normal")


class TestNothingLeavesQuietly(unittest.TestCase):
    """Whatever path a word takes through `retain_caption`: if the caption says nothing was lost, every word no rule
    explains is still in it. Random sentences of Doppler, B-mode and unknown wording, joined by every separator."""
    PIECES = ("Gradient moyen de 10 mmHg", "Insuffisance mitrale légère", "Pas de sténose aortique", "sténose",
              "fuite", "Pression systolique du ventricule droit normale", "pression systolique normales",
              "dysfonction diastolique", "VG non dilaté", "Fonction systolique normale", "Valve aortique calcifiée",
              "dimension", "fonction", "Sténose aortique calcifiée modérée", "Sténose mitrale rhumatismale",
              "prolapsus du feuillet", "Plastie mitrale en 2019", "Bioprothèse en 2018", "valve aortique calcifiée",
              "Hypertrophie", "Masse intracavitaire", "zorglub", "tumeur", "probablement normales", "non normales",
              "normale", "1-2/4", "42 mmHg", "ventricule droit", "aire diastolique 30 cm2")
    JOINTS = (". ", ", ", " et ", " ou ", " ni ", " avec ", " sans ", " mais ", " (", ") ", ": ", " / ", " - ",
              " sur ", " par ", " via ", " responsable d'un ", " secondaire à une ", " entraînant une ", " non ",
              " de ")

    def lost_quietly(self, source):
        caption, _, stranded = rn.retain_caption(source)
        kept = set(re.split(r"[\s/.,;:()]+", rn._plain(caption)))
        unexplained = {word for clause in rc.text_clauses(source) for word in rn._unexplained(clause)}
        return set() if stranded else unexplained - kept

    def test_a_removed_part_never_joins_its_neighbours(self):
        kept = rn.retain_caption("Valve calcifiée) fuite sans thrombus")[0]
        self.assertEqual(kept, "Valve calcifiée sans thrombus.")
        caption = rn.retain_caption("Valve calcifiée (fuite, 1-2/4)sans thrombus")[0]
        self.assertEqual(caption, "Valve calcifiée, sans thrombus.")

    def test_random_sentences(self):
        rng = random.Random(0)
        for _ in range(20000):
            pieces = rng.choices(self.PIECES, k=rng.randint(2, 6))
            source = "".join(p + rng.choice(self.JOINTS) for p in pieces[:-1]) + pieces[-1]
            self.assertEqual(self.lost_quietly(source), set(), source)


class TestPolicy(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        with open(rn.DEFAULT_POLICY, encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, raw):
        path = os.path.join(self.root, "policy.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
        return path

    def test_the_shipped_policy_is_provisional_and_covers_every_trained_structure(self):
        self.assertEqual(POLICY.status, "provisional")
        self.assertIsNone(POLICY.approval)
        self.assertEqual(POLICY.scope, "bmode")
        self.assertEqual(set(POLICY.required), set(rn.GRAMMARS))

    def test_invalid_policies_are_refused(self):
        changes = {"scope": lambda r: r.update(scope="colour"),
                   "normal_max": lambda r: r["measures"]["lv_size"]["sources"][0].update(normal_max=9.0),
                   "combine": lambda r: r["measures"]["la_size"].update(combine="any"),
                   "mild needs up_to": lambda r: r["measures"]["lv_wall_thickness"]["mild"].pop("up_to"),
                   "settled by 'lv_strain'": lambda r: r["measures"]["lv_wall_thickness"]["mild"].update(
                       settled_by="lv_strain"),
                   "settled by 'rv_size'": lambda r: r["measures"]["lv_wall_thickness"]["mild"].update(
                       settled_by="rv_size"),
                   "up_to is below abnormal_above": lambda r: r["measures"]["lv_wall_thickness"]["mild"].update(
                       up_to=0.95),
                   "geometry needs wall": lambda r: r["measures"]["lv_wall_thickness"]["mild"]["geometry"].pop("wall"),
                   "no measure of this structure reads": lambda r: r["measures"]["lv_wall_thickness"]["mild"][
                       "geometry"].update(cavity="LA Dimension"),
                   "unknown structure": lambda r: r["measures"]["lv_size"].update(structure="spleen"),
                   "required_for_normal must": lambda r: r["required_for_normal"].pop("ivc"),
                   "no wording rule states": lambda r: r["required_for_normal"].update(lv=["lv_sise"]),
                   "no caption could agree": lambda r: r["measures"].update(rv_sise=r["measures"]["rv_size"]),
                   "device_present": lambda r: r.update(device_present="normal"),
                   "approval needs": lambda r: r.update(approval={"by": "someone"})}
        for message, change in changes.items():
            with self.subTest(message):
                raw = json.loads(json.dumps(self.raw))
                change(raw)
                with self.assertRaisesRegex(ValueError, message):
                    rn.load_policy(self.write(raw))

    def test_approval_covers_exactly_these_rules_and_this_policy(self):
        digest = rn.load_policy(self.write(self.raw)).digest
        self.assertEqual(digest, POLICY.digest)
        approval = {"by": "reviewers", "on": "2026-11-30", "audit_report": "audit_report.json", "digest": digest}
        approved = rn.load_policy(self.write({**self.raw, "approval": approval}))
        self.assertEqual((approved.status, approved.digest), ("approved", digest))   # approving changes no digest
        changed = json.loads(json.dumps(self.raw))
        changed["ef"]["normal_min"] = 50
        with self.assertRaisesRegex(ValueError, "approve again"):
            rn.load_policy(self.write({**changed, "approval": approval}))
        self.assertNotEqual(rn.load_policy(self.write(changed)).digest, digest)


if __name__ == "__main__":
    unittest.main()
