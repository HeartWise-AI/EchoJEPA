# tests/data/test_build_radar_captions_merged.py

"""`data/build_radar_captions_merged.py` on synthetic reports, through the column review's own code (needs a column
review checkout in `COLUMN_REVIEW_ROOT`): prefill sentences, findings dropped by the checks, parser and EF fills, the
wall segment summary, the flag, and a table the RADAR dataset accepts. The EF sentence's reading of the text and its
choice of source need no checkout."""

import dataclasses
import json
import os
import sys
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import build_radar_captions_merged as merged  # noqa: E402
import compare_column_prefill as cmp  # noqa: E402
import radar_normality as rn  # noqa: E402

from src.models.radar.loss import caption_soft_values  # noqa: E402
from src.models.radar.routing import load_routing  # noqa: E402

POLICY = rn.load_policy()
ROUTING = load_routing()
LIMITS = cmp.load_limits()
COLUMN_REVIEW_ROOT = os.environ.get("COLUMN_REVIEW_ROOT", "")
CHECKOUT = bool(COLUMN_REVIEW_ROOT) and os.path.isfile(
    os.path.join(COLUMN_REVIEW_ROOT, "tools", "labelbox", "column_review.py"))
EF, LVIDD = POLICY.ef["column"], "LVID Diastole (2D)"
LV, PERI = "Left ventricule findings", "Pericardium findings"
TRICUSPID, TR, PR = "Tricuspid valve findings", "Referring TR Grade", "Referring PR Grade"
CONCLUSION = merged.CONCLUSION_COLUMN


class TestEfSentence(unittest.TestCase):
    """`stated_ef` and `ef_entry` (`--ef-sentence`) on synthetic phrases; no column review needed."""

    def test_stated_values(self):
        for text, expected in (("FEVG 55 %.", (55.0, "value")), ("FEVG 50-60 %.", (55.0, "value")),
                               ("FEVG estimée visuellement à 45 %. FE 48 %.", (46.5, "value")),
                               ("FEVG > 55 %.", (None, "bound")), ("FEVG environ 40 %.", (None, "uncertain")),
                               ("FEVG 30 %. FE 60 %.", (None, "disagree")),
                               ("FEVG calculée à 45 %.", (None, "other_method")),
                               ("FEVG calculée par méthode Simpson à 45 %. FEVG estimée à 55 %.", (55.0, "value")),
                               ("Fraction d'éjection normale, estimée à 60 %.", (None, "unread")),
                               ("Strain longitudinal global -18 %.", (None, "none")), ("", (None, "none"))):
            value, status, _ = merged.stated_ef(text)
            self.assertEqual((value, status), expected, text)

    def test_values_round_half_up(self):
        self.assertEqual([merged.ef_integer(v) for v in (57.5, 52.5, 64.99999761581421, 30.2)], [58, 53, 65, 30])

    def entry(self, lv_text, ef=None, conclusion=""):
        return merged.ef_entry({EF: ef, CONCLUSION: conclusion}, lv_text, POLICY)

    def test_the_text_comes_first_then_the_conclusion_then_the_structured_value(self):
        self.assertEqual(self.entry("FEVG 45 %.", 60.0),
                         (("LV ejection fraction is 45%.", merged.EF_SIDE, 45, "ef_text"), "ef_text"))
        self.assertEqual(self.entry("Ventricule gauche non dilaté.", 60.0, "FEVG 35 %.")[0][2:], (35, "ef_conclusion"))
        entry, reason = self.entry("Ventricule gauche non dilaté.", 57.5)
        self.assertEqual(entry, ("LV ejection fraction is 58%.", merged.EF_SIDE, 58, "ef_structured"))
        self.assertEqual(reason, "ef_structured:lv_cell=none,conclusion=none")

    def test_text_without_a_value_lets_a_compatible_structured_value_through(self):
        self.assertEqual(self.entry("FEVG environ 55 %.", 58.0)[0][2:], (58, "ef_structured"))
        self.assertEqual(self.entry("FEVG > 55 %.", 62.0)[0][2:], (62, "ef_structured"))
        self.assertEqual(self.entry("FEVG environ 55 %.", 40.0), (None, "text_conflict"))
        self.assertEqual(self.entry("FEVG environ 55 %.", None), (None, "text_conflict"))
        self.assertEqual(self.entry("FEVG environ 55 %.", 30.0, "FEVG 30 %."), (None, "text_conflict"))

    def test_other_methods_and_unread_wording_fall_through(self):
        entry, reason = self.entry("FEVG calculée à 45 %.", 60.0)
        self.assertEqual(entry[2:], (60, "ef_structured"))
        self.assertEqual(reason, "ef_structured:lv_cell=other_method,conclusion=none")
        entry, reason = self.entry("Fraction d'éjection normale, estimée à 60 %.", 60.0)
        self.assertEqual(entry[2:], (60, "ef_structured"))
        self.assertTrue(reason.startswith("ef_structured:lv_cell=unread"))

    def test_no_ef_or_an_invalid_structured_value_writes_nothing(self):
        self.assertEqual(self.entry("Ventricule gauche non dilaté.", None), (None, "no_ef"))
        self.assertEqual(self.entry("", 95.0), (None, "no_ef"))      # outside the policy's valid range
        self.assertEqual(self.entry("", float("nan")), (None, "no_ef"))

    def test_the_soft_ef_targets_read_the_sentence_back(self):
        captions = [f"LV systolic function appears normal. {merged.EF_SENTENCE.format(v)} LV mass is normal."
                    for v in (12, 45, 75)]
        table = pd.DataFrame({"structure": "lv", "caption": captions, "caption_id": [1, 2, 3]})
        values, coverage = caption_soft_values(table, {"source": "caption_number"})
        self.assertEqual(values, {1: 12.0, 2: 45.0, 3: 75.0})
        self.assertEqual(coverage["share_with_value"], 1.0)


@unittest.skipUnless(CHECKOUT, "COLUMN_REVIEW_ROOT does not name a column review checkout")
class TestMergedCaptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cr, cls.spec = cmp.load_column_review(COLUMN_REVIEW_ROOT)
        cmp.index_columns(cls.spec)

    def row(self, structure, report, views=("A4C", "PLAX"), policy=POLICY, **options):
        entry = next(s for s in ROUTING.trained if s.name == structure)
        return merged.merged_row("S1", entry, report, set(views) & set(entry.views), policy, self.spec, self.cr,
                                 LIMITS, **options)

    def sources(self, row):
        return {column: (value, source) for column, value, source in json.loads(row["sources"])}

    def test_prefill_sentences_in_english(self):
        row, dropped = self.row("lv", {LV: "Ventricule gauche non dilaté. Fonction systolique normale.",
                                       LVIDD: 4.8, EF: 60.0})
        self.assertEqual(dropped, [])
        self.assertIn("LV cavity is not dilated.", row["caption"])
        self.assertIn("LV systolic function appears normal.", row["caption"])
        self.assertEqual(self.sources(row)["lv_size"], ("normal", "prefill_text"))
        self.assertNotRegex(row["caption"], r"\d")                 # no EF number, no measurement
        self.assertEqual(row["normality_status"], POLICY.status)

    def test_rows_carry_the_policy_status(self):
        report = {LV: "Ventricule gauche non dilaté. Fonction systolique normale.", LVIDD: 4.8, EF: 60.0}
        approved = dataclasses.replace(POLICY, status="approved",
                                       approval={"by": "test", "on": "2026-01-01", "digest": POLICY.digest})
        provisional = dataclasses.replace(POLICY, status="provisional", approval=None)
        for policy in (approved, provisional):
            with self.subTest(status=policy.status):
                row, _ = self.row("lv", report, policy=policy)
                self.assertEqual(row["normality_status"], policy.status)
                self.assertEqual(row["policy_digest"], POLICY.digest)

    def test_a_contradicted_finding_is_dropped_and_listed(self):
        row, dropped = self.row("lv", {LV: "Ventricule gauche non dilaté. Fonction systolique normale.",
                                       LVIDD: 6.2, EF: 60.0})
        self.assertNotIn("not dilated", row["caption"])
        self.assertEqual([(d["column"], d["reason"]) for d in dropped], [("lv_size", "contradiction")])
        self.assertEqual(row["normality"], "unknown")              # the parser's own flag is not normal here

    def test_effusion_wording_and_side(self):
        row, _ = self.row("pericardium", {PERI: "Pas d'épanchement péricardique."})
        self.assertEqual(row["caption"], "No pericardial effusion.")
        self.assertEqual(row["normality"], "normal")
        row, _ = self.row("pericardium", {PERI: "Épanchement péricardique léger sans signe de tamponnade."})
        self.assertEqual(row["caption"], "Pericardial effusion is present.")
        self.assertEqual(row["normality"], "abnormal")

    def test_ef_band_fills_a_silent_lv_function(self):
        row, _ = self.row("lv", {LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 35.0})
        self.assertEqual(self.sources(row)["lv_function"], ("moderately_reduced", "ef_band"))
        self.assertIn("LV systolic function appears moderately reduced.", row["caption"])
        self.assertEqual(row["normality"], "abnormal")
        row, _ = self.row("lv", {LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 53.0})     # a grey zone
        self.assertNotIn("lv_function", self.sources(row))

    def test_ef_sentence_follows_the_lv_function_sentence_and_leaves_the_flag(self):
        for report in ({LV: "Ventricule gauche non dilaté. Fonction systolique normale. FEVG 60 %.", LVIDD: 4.8,
                        EF: 60.0},
                       {LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 35.0},
                       {LV: "Ventricule gauche non dilaté. Fonction systolique normale.", LVIDD: 4.8, EF: 48.0},
                       {LV: "Ventricule gauche dilaté. FEVG estimée à 45 %.", LVIDD: 6.4, EF: 45.0}):
            off, _ = self.row("lv", report)
            on, _ = self.row("lv", report, ef_sentence=True)
            sentences = [s + "." for s in on["caption"].split(". ") if s]
            value = merged.ef_integer(report[EF])
            ef = merged.EF_SENTENCE.format(value)
            self.assertIn(ef, on["caption"])
            self.assertEqual(on["caption"].replace(" " + ef, "").replace(ef + " ", ""), off["caption"])
            function = [s for s in sentences if s.startswith("LV systolic function")]
            if function:
                self.assertEqual(on["caption"].index(ef), on["caption"].index(function[0]) + len(function[0]) + 1)
            self.assertEqual(on["normality"], off["normality"])
            self.assertEqual(on["sentences"], off["sentences"] + 1)
            self.assertEqual([on[k] for k in ("from_prefill", "from_parser", "from_ef")],
                             [off[k] for k in ("from_prefill", "from_parser", "from_ef")])
            self.assertEqual(self.sources(on)["lv_ef"][0], value)
            self.assertNotIn("lv_ef", self.sources(off))

    def test_ef_sentence_alone_makes_an_unknown_row_and_only_on_the_lv(self):
        self.assertIsNone(self.row("lv", {LV: "", EF: 53.0})[0])          # a grey zone: no band sentence
        row, _ = self.row("lv", {LV: "", EF: 53.0}, ef_sentence=True)
        self.assertEqual((row["caption"], row["normality"]), ("LV ejection fraction is 53%.", "unknown"))
        row, _ = self.row("pericardium", {PERI: "Pas d'épanchement péricardique.", EF: 50.0}, ef_sentence=True)
        self.assertEqual(row["caption"], "No pericardial effusion.")

    def test_ef_sentence_reasons_are_counted_per_lv_study(self):
        from collections import Counter
        reports = pd.DataFrame([{"study_id": "S1", LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 60.0},
                                {"study_id": "S2", LV: "FEVG 45 %.", EF: 50.0},
                                {"study_id": "S3", LV: "Ventricule gauche non dilaté.", EF: None}])
        clips = pd.DataFrame({"study_id": ["S1", "S2", "S3"], "view": ["A4C", "A4C", "A4C"]})
        reasons = Counter()
        table, _ = merged.build_tables(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS,
                                       ef_sentence=True, ef_reasons=reasons)
        self.assertEqual(dict(reasons), {"ef_structured:lv_cell=none,conclusion=none": 1, "ef_text": 1, "no_ef": 1})
        self.assertEqual(list(table.columns), list(merged.COLUMNS))
        off, _ = merged.build_tables(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS)
        self.assertFalse(off.caption.str.contains("ejection fraction").any())

    def test_nothing_to_say_gives_no_row(self):
        row, dropped = self.row("pericardium", {PERI: ""})
        self.assertIsNone(row)
        self.assertEqual(dropped, [])

    def test_wall_segments_are_summarised(self):
        report = {LV: "Ventricule gauche non dilaté.", LVIDD: 4.8}
        segments = [c for c in self.spec["fields"] if c.startswith("wall_")]
        self.assertEqual(len(segments), 16)
        normal = {**report, **{c: 1.0 for c in cmp.report_columns(ROUTING, POLICY, self.spec, LIMITS)
                               if c.startswith("WMS ")}}
        row, _ = self.row("lv", normal)
        self.assertEqual(row["caption"].count("segment"), 0)
        self.assertIn(merged.WALLS_NORMAL, row["caption"])
        one = {**normal, "WMS 4C Apical septum": 2.0}
        row, _ = self.row("lv", one)
        self.assertEqual(row["caption"].count("segment shows"), 1)
        self.assertNotIn(merged.WALLS_NORMAL, row["caption"])

    def test_hypertrophy_mass_and_remodelling_are_written_from_the_parser(self):
        base = "Ventricule gauche non dilaté. Fonction systolique normale. "
        row, _ = self.row("lv", {LV: base + "Hypertrophie ventriculaire gauche. Masse augmentée.", LVIDD: 4.8})
        self.assertIn("LV hypertrophy is present.", row["caption"])
        self.assertIn("LV mass is increased.", row["caption"])
        self.assertEqual(self.sources(row)["lv_hypertrophy"], ("abnormal", "parser"))
        self.assertEqual(row["normality"], "abnormal")
        row, _ = self.row("lv", {LV: base + "Pas d'hypertrophie. Masse normale.", LVIDD: 4.8})
        self.assertIn("No LV hypertrophy.", row["caption"])
        self.assertIn("LV mass is normal.", row["caption"])
        self.assertNotIn("is present", row["caption"])
        row, _ = self.row("lv", {LV: base + "Remodelage concentrique.", LVIDD: 4.8})
        self.assertIn("LV remodelling is present.", row["caption"])
        row, _ = self.row("lv", {LV: base, LVIDD: 4.8, "IVS Diastole Thickness (2D)": 1.6,
                                 "LVIW Diastolic Thickness (2D)": 1.6})
        self.assertNotIn("hypertrophy", row["caption"])            # a measurement alone writes no sentence

    def test_disagreeing_claims_write_no_extra_sentence(self):
        facts = [{"source": "text", "attribute": "lv_mass", "finding": "state", "value": "normal"},
                 {"source": "text", "attribute": "lv_mass", "finding": "state", "value": "abnormal"},
                 {"source": "text", "attribute": "lv_wall_thickness", "finding": "hypertrophy", "value": "uncertain"},
                 {"source": "structured", "attribute": "lv_geometry", "finding": "remodelling", "value": "abnormal"}]
        self.assertEqual(merged.extra_sentences(facts, "lv"), {})
        self.assertEqual(merged.extra_sentences(facts, "rv"), {})

    def test_the_right_atrial_pressure_is_written_from_the_parser(self):
        claim = {"source": "text", "attribute": "ivc_pressure", "finding": "state"}
        self.assertEqual(merged.extra_sentences([{**claim, "value": "abnormal"}], "ivc"),
                         {"ra_pressure": ("Estimated right atrial pressure is elevated.", "abnormal")})
        self.assertEqual(merged.extra_sentences([{**claim, "value": "normal"}], "ivc"),
                         {"ra_pressure": ("Estimated right atrial pressure is normal.", "normal")})
        self.assertEqual(merged.extra_sentences([{**claim, "value": "normal"}, {**claim, "value": "abnormal"}],
                                                "ivc"), {})
        self.assertEqual(merged.extra_sentences([{**claim, "value": "abnormal"}], "lv"), {})

    def test_valve_procedures_are_written(self):
        pulmonic, tricuspid = "Pulmonary valve findings", "Tricuspid valve findings"
        row, _ = self.row("pulmonic", {pulmonic: "Homogreffe pulmonaire (procédure de Ross)."}, views=("PSAX_AV",))
        self.assertEqual(row["caption"], "Pulmonic valve homograft is present.")
        self.assertEqual(self.sources(row)["pulmonic_homograft"], ("present", "parser"))
        self.assertEqual(row["normality"], "abnormal")
        row, _ = self.row("tricuspid", {tricuspid: "Statut post-plastie tricuspidienne chirurgicale. "
                                                   "Trace d'insuffisance tricuspidienne."})
        self.assertEqual(row["caption"], "Status post surgical tricuspid valve repair.")    # the jet is Doppler
        self.assertEqual(row["normality"], "abnormal")
        row, _ = self.row("tricuspid", {tricuspid: "Valve tricuspide de structure normale. "
                                                   "Statut post-plastie tricuspidienne chirurgicale en 2015."})
        self.assertIn("Status post surgical tricuspid valve repair.", row["caption"])

    def test_a_denied_doubtful_or_other_valve_procedure_writes_nothing(self):
        pulmonic, tricuspid = "Pulmonary valve findings", "Tricuspid valve findings"
        row, _ = self.row("pulmonic", {pulmonic: "Pas d'homogreffe pulmonaire."}, views=("PSAX_AV",))
        self.assertTrue(row is None or "homograft" not in row["caption"])
        for text in ("Statut post-plastie tricuspidienne chirurgicale?",
                     "Statut post-plastie mitrale chirurgicale.",
                     "Statut post-plastie tricuspidienne chirurgicale probable."):
            row, _ = self.row("tricuspid", {tricuspid: text})
            self.assertTrue(row is None or "repair" not in row["caption"], text)
        self.assertEqual(merged.procedure_sentences("Homogreffe pulmonaire.", {}, "pulmonic"), {})   # no device
        self.assertEqual(merged.procedure_sentences("Homogreffe.", {"lv_device": "device"}, "lv"), {})

    def default_row(self, structure, report, views=("A4C", "PSAX_AV"), year=2020):
        entry = next(s for s in ROUTING.trained if s.name == structure)
        return merged.merged_row("S1", entry, report, set(views) & set(entry.views), POLICY, self.spec, self.cr,
                                 LIMITS, regurgitation_default_from=year)

    def test_regurgitation_default_is_off_unless_asked(self):
        report = {merged.YEAR_COLUMN: 2022, "Referring TR Grade": "0", "Tricuspid valve findings": ""}
        self.assertIsNone(self.row("tricuspid", report)[0])
        row, dropped = self.default_row("tricuspid", report)
        self.assertEqual(dropped, [])
        self.assertEqual(row["caption"], "No tricuspid regurgitation.")
        self.assertEqual(self.sources(row), {"tricuspid_regurgitation": ("none", "default")})
        self.assertEqual(row["normality"], "unknown")             # a default, not a reading of the text
        self.assertEqual((row["from_prefill"], row["from_parser"], row["from_ef"]), (0, 0, 0))
        row, _ = self.default_row("pulmonic", {merged.YEAR_COLUMN: "2021", "Referring PR Grade": None,
                                               merged.CONCLUSION_COLUMN: "Valve aortique tricuspide."})
        self.assertEqual(row["caption"], "No pulmonic regurgitation.")
        row, _ = self.default_row("tricuspid", {merged.YEAR_COLUMN: 2021.0, "Referring TR Grade": float("nan"),
                                                merged.CONCLUSION_COLUMN: "Valve aortique tricuspide."})
        self.assertEqual(row["caption"], "No tricuspid regurgitation.")   # three aortic cusps name no tricuspid

    def test_regurgitation_default_needs_every_source_silent(self):
        tricuspid = next(s for s in ROUTING.trained if s.name == "tricuspid")
        base = {merged.YEAR_COLUMN: 2022, "Referring TR Grade": "0.0", "Tricuspid valve findings": ""}
        cases = {"before_year": {merged.YEAR_COLUMN: 2019}, "no_year": {merged.YEAR_COLUMN: None},
                 "grade_trace": {"Referring TR Grade": "0.5"}, "grade_regurgitation": {"Referring TR Grade": "2"},
                 "grade_unreadable": {"Referring TR Grade": "n/a"},
                 "cell_filled": {"Tricuspid valve findings": "Insuffisance tricuspidienne légère."},
                 "named_in_conclusion": {merged.CONCLUSION_COLUMN: "IT modérée."}}
        self.assertIsNone(merged.regurgitation_default(base, tricuspid, 2020))
        for reason, change in cases.items():
            report = {**base, **change}
            self.assertEqual(merged.regurgitation_default(report, tricuspid, 2020), reason)
            row, _ = self.default_row("tricuspid", report)
            self.assertIsNone(row, reason)        # the Doppler-only cell leaves no caption either
        mitral = next(s for s in ROUTING.trained if s.name == "mitral")
        self.assertEqual(merged.regurgitation_default(base, mitral, 2020), "not_a_default_valve")
        self.assertIsNone(self.default_row("mitral", {**base, "Mitral valve findings": ""})[0])

    def test_regurgitation_default_never_changes_a_caption(self):
        text = "Valve tricuspide de structure normale."
        report = {merged.YEAR_COLUMN: 2022, "Referring TR Grade": "0", "Tricuspid valve findings": text}
        self.assertEqual(self.default_row("tricuspid", report)[0], self.row("tricuspid", report)[0])

    def test_regurgitation_default_reasons_are_counted(self):
        reports = pd.DataFrame([
            {"study_id": "S1", merged.YEAR_COLUMN: 2022, "Referring TR Grade": "0", "Referring PR Grade": "0.5"},
            {"study_id": "S2", merged.YEAR_COLUMN: 2015, "Referring TR Grade": "0", "Referring PR Grade": "0"}])
        clips = pd.DataFrame({"study_id": ["S1", "S2"], "view": ["PSAX_AV", "PSAX_AV"]})
        table, _ = merged.build_tables(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS)
        self.assertFalse(table.structure.isin(["tricuspid", "pulmonic"]).any())
        reasons = {}
        table, _ = merged.build_tables(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS,
                                       regurgitation_default_from=2020, default_reasons=reasons)
        valves = table[table.structure.isin(["tricuspid", "pulmonic"])]
        self.assertEqual(list(zip(valves.study_id, valves.caption)), [("S1", "No tricuspid regurgitation.")])
        self.assertEqual(dict(reasons["tricuspid"]), {"written": 1, "before_year": 1})
        self.assertEqual(dict(reasons["pulmonic"]), {"grade_trace": 1, "before_year": 1})

    def test_an_enlarged_left_atrium_is_written_as_dilated(self):
        row, _ = self.row("la", {"Left atrial findings": "Volume indexé légèrement augmenté."})
        if row is not None and "la_size" in self.sources(row):
            self.assertEqual(row["caption"], "Left atrium is dilated.")
            self.assertEqual(row["normality"], "abnormal")
        else:
            self.fail("an increased left atrial volume wrote no size sentence")

    def test_table_is_read_by_the_radar_dataset_columns(self):
        reports = pd.DataFrame([{"study_id": "S1", PERI: "Pas d'épanchement péricardique.",
                                 LV: "Ventricule gauche non dilaté.", LVIDD: 4.8, EF: 60.0},
                                {"study_id": "S2", PERI: "", LV: ""}])
        clips = pd.DataFrame({"study_id": ["S1", "S1", "S2"], "view": ["A4C", "PLAX", "A4C"]})
        table, dropped = merged.build_tables(reports, clips, ROUTING, POLICY, self.spec, self.cr, LIMITS)
        self.assertEqual(list(table.columns), list(merged.COLUMNS))
        self.assertEqual(set(table.study_id), {"S1"})
        self.assertFalse(table.duplicated(["study_id", "structure"]).any())
        self.assertTrue(set(table.normality) <= {"normal", "abnormal", "unknown"})
        self.assertTrue((table.caption.str.len() > 0).all())
        self.assertEqual(table.caption_id.dtype.kind, "i")

    def graded_tables(self, reports, bmode, color, valves=("tricuspid", "pulmonic"), default_from=None):
        clips = pd.DataFrame(bmode, columns=["study_id", "view"])
        color = pd.DataFrame(color, columns=["study_id", "view"])
        reasons = {}
        table, _ = merged.build_tables(pd.DataFrame(reports), clips, ROUTING, POLICY, self.spec, self.cr, LIMITS,
                                       regurgitation_default_from=default_from, default_reasons={},
                                       grade_valves=valves, color_clips=color, grade_reasons=reasons)
        return table, {name: dict(counter) for name, counter in reasons.items()}

    def test_graded_rows_are_off_by_default_and_leave_bmode_rows_unchanged(self):
        reports = [{"study_id": "S1", TRICUSPID: "Valve tricuspide de structure normale.", TR: "2",
                    merged.CONCLUSION_COLUMN: "", PERI: "Pas d'épanchement péricardique."}]
        bmode = [("S1", "A4C"), ("S1", "PLAX")]
        clips = pd.DataFrame(bmode, columns=["study_id", "view"])
        off, _ = merged.build_tables(pd.DataFrame(reports), clips, ROUTING, POLICY, self.spec, self.cr, LIMITS)
        self.assertEqual(list(off.columns), list(merged.COLUMNS))       # no scope column when off
        on, reasons = self.graded_tables(reports, bmode, [("S1", "A4C")])
        self.assertEqual(list(on.columns), [*merged.COLUMNS, "scope"])
        bmode_rows = on[on.scope == "bmode"].drop(columns="scope").reset_index(drop=True)
        pd.testing.assert_frame_equal(bmode_rows, off.reset_index(drop=True))
        graded = on[on.scope == merged.COLOR_FLOW]
        self.assertEqual(list(graded.structure), ["tricuspid"])
        row = graded.iloc[0]
        bmode_caption = off.set_index("structure").caption["tricuspid"]
        self.assertEqual(row.caption, bmode_caption + " Moderate tricuspid regurgitation.")
        self.assertEqual(self.sources(row)["tricuspid_regurgitation"], ("moderate", "grade"))
        self.assertEqual(row.normality, "abnormal")
        self.assertEqual(reasons, {"tricuspid": {"written": 1}})        # no PSAX_AV clip: pulmonic has no row

    def test_no_colour_clip_writes_no_graded_row(self):
        reports = [{"study_id": "S1", TRICUSPID: "", TR: "2", PR: "1", merged.CONCLUSION_COLUMN: ""}]
        table, reasons = self.graded_tables(reports, [("S1", "A4C"), ("S1", "PSAX_AV")], [("S1", "PLAX")])
        self.assertFalse((table.scope == merged.COLOR_FLOW).any())
        self.assertEqual(reasons, {"tricuspid": {"no_color_clip": 1}, "pulmonic": {"no_color_clip": 1}})

    def test_a_colour_only_valve_gets_the_graded_sentence_alone(self):
        reports = [{"study_id": "S1", "Pulmonary valve findings": "", PR: "1", merged.CONCLUSION_COLUMN: ""}]
        table, reasons = self.graded_tables(reports, [("S1", "A4C")], [("S1", "PSAX_AV")], valves=("pulmonic",))
        rows = table[table.structure == "pulmonic"]
        self.assertEqual(list(zip(rows.scope, rows.caption)), [(merged.COLOR_FLOW, "Mild pulmonic regurgitation.")])
        self.assertEqual(rows.iloc[0].normality, "abnormal")
        self.assertEqual(reasons, {"pulmonic": {"written": 1}})

    def test_the_valve_list_is_a_parameter(self):
        reports = [{"study_id": "S1", TRICUSPID: "", TR: "1", PR: "1", "Referring MR Grade": "0",
                    "Mitral valve findings": "", merged.CONCLUSION_COLUMN: ""}]
        color = [("S1", "A4C"), ("S1", "PSAX_AV")]
        table, reasons = self.graded_tables(reports, [("S1", "A4C")], color, valves=("pulmonic",))
        self.assertEqual(list(table[table.scope == merged.COLOR_FLOW].structure), ["pulmonic"])
        self.assertEqual(set(reasons), {"pulmonic"})
        table, _ = self.graded_tables(reports, [("S1", "A4C")], color, valves=("mitral",))
        self.assertEqual(list(table[table.scope == merged.COLOR_FLOW].caption), ["No mitral regurgitation."])
        self.assertEqual(merged.parse_args(["--column-review-root", "x", "--reports", "r", "--pacs", "p", "--clips",
                                            "c", "--splits", "s", "--out-dir", "o"]).regurgitation_grade_valves, None)

    def test_the_graded_sentence_takes_the_place_of_the_default(self):
        reports = [{"study_id": "S1", merged.YEAR_COLUMN: 2022, TRICUSPID: "", TR: "0",
                    merged.CONCLUSION_COLUMN: ""}]
        table, _ = self.graded_tables(reports, [("S1", "A4C")], [("S1", "A4C")], valves=("tricuspid",),
                                      default_from=2020)
        rows = table[table.structure == "tricuspid"].set_index("scope").caption
        self.assertEqual(rows["bmode"], "No tricuspid regurgitation.")
        self.assertEqual(rows[merged.COLOR_FLOW], "No tricuspid regurgitation.")     # once, from the grade
        graded = table[(table.structure == "tricuspid") & (table.scope == merged.COLOR_FLOW)].iloc[0]
        self.assertEqual(self.sources(graded), {"tricuspid_regurgitation": ("none", "grade")})


class TestRegurgitationGrade(unittest.TestCase):
    """The graded regurgitation sentence's rule (`regurgitation_grade`) on synthetic report cells."""

    tricuspid = next(s for s in ROUTING.trained if s.name == "tricuspid")
    pulmonic = next(s for s in ROUTING.trained if s.name == "pulmonic")

    def grade(self, cell, code, conclusion="", structure=None, has_color=True):
        structure = structure or self.tricuspid
        column = merged.GRADE_COLUMNS[structure.name]
        report = {structure.report_column: cell, column: code, merged.CONCLUSION_COLUMN: conclusion}
        return merged.regurgitation_grade(report, structure, has_color)

    def test_the_words_are_the_ontology_s(self):
        import yaml
        with open(os.path.join(REPO, "configs", "radar", "radar-caption-ontology-v2.yaml"), encoding="utf-8") as f:
            scale = yaml.safe_load(f)["scales"]["regurgitation"]
        self.assertEqual(list(merged.GRADE_LABELS), scale["values"])
        self.assertEqual(merged.GRADE_LABELS, scale["labels"]["en"])
        self.assertTrue(set(merged.GRADE_CODES.values()) <= set(scale["values"]))
        self.assertEqual(merged.grade_sentence("tricuspid", "moderate_to_severe"),
                         "Moderate to severe tricuspid regurgitation.")
        self.assertEqual(merged.grade_sentence("pulmonic", "none"), "No pulmonic regurgitation.")

    def test_each_blocking_reason(self):
        cases = {
            "no_color_clip": dict(cell="", code="2", has_color=False),
            "grade_missing": dict(cell="", code=None),
            "grade_unreadable": dict(cell="", code="1.5"),
            "text_uncertain": dict(cell="Insuffisance tricuspidienne probablement modérée.", code="2"),
            "text_conflict": dict(cell="Insuffisance tricuspidienne sévère.", code="2"),
            "trace_not_in_text": dict(cell="", code="0.5"),
        }
        for reason, case in cases.items():
            self.assertEqual(self.grade(**case), (reason, None), reason)
        self.assertEqual(self.grade("", float("nan")), ("grade_missing", None))
        self.assertEqual(self.grade("", "Trace"), ("grade_unreadable", None))       # words are not read
        self.assertEqual(self.grade("", "2"),
                         ("written", ("Moderate tricuspid regurgitation.", "abnormal", "moderate")))

    def test_trace_needs_the_text_to_agree(self):
        written = ("written", ("Trace tricuspid regurgitation.", "normal", "trace"))
        self.assertEqual(self.grade("Trace d'insuffisance tricuspidienne.", "0.5"), written)
        self.assertEqual(self.grade("Valve fine.", "0.5", conclusion="IT minime."), written)
        self.assertEqual(self.grade("Insuffisance tricuspidienne légère.", "0.5"), ("text_conflict", None))
        self.assertEqual(self.grade("Insuffisance tricuspidienne.", "0.5"), ("trace_not_in_text", None))
        self.assertEqual(self.grade("Trace d'insuffisance tricuspidienne.", "1"), ("text_conflict", None))

    def test_agreement_is_the_same_ontology_grade(self):
        self.assertEqual(self.grade("Insuffisance tricuspidienne modérée à sévère.", "3")[0], "written")
        self.assertEqual(self.grade("Insuffisance tricuspidienne légère à modérée.", "1")[0], "text_conflict")
        self.assertEqual(self.grade("Insuffisance tricuspidienne légère à modérée.", "2")[0], "text_conflict")
        self.assertEqual(self.grade("Pas d'insuffisance tricuspidienne.", "0")[0], "written")
        self.assertEqual(self.grade("Pas d'insuffisance tricuspidienne.", "1")[0], "text_conflict")
        self.assertEqual(self.grade("Insuffisance tricuspidienne.", "0")[0], "text_conflict")
        self.assertEqual(self.grade("Insuffisance tricuspidienne.", "1")[0], "written")   # no grade, no conflict
        self.assertEqual(self.grade("IT légère. IT modérée.", "1")[0], "text_conflict")   # two grades
        self.assertEqual(self.grade("Valve tricuspide normale.", "4")[0], "written")       # text silent

    def test_other_valves_and_other_words_are_not_read(self):
        self.assertEqual(self.grade("", "2", conclusion="IM sévère, IT modérée.")[0], "written")
        self.assertEqual(self.grade("", "2", conclusion="Insuffisance aortique sévère.")[0], "written")
        self.assertEqual(self.grade("", "2", conclusion="Insuffisance sévère.")[0], "written")   # no valve named
        self.assertEqual(self.grade("IT légère, HTAP sévère.", "1")[0], "written")
        self.assertEqual(self.grade("Pas de sténose, insuffisance légère.", "1")[0], "written")
        self.assertEqual(self.grade("", "1", conclusion="IM et IT sévères.")[0], "text_conflict")
        self.assertEqual(self.grade("", "1", conclusion="Insuffisance mitrale et tricuspidienne sévères.")[0],
                         "text_conflict")
        self.assertEqual(self.grade("Insuffisance pulmonaire légère.", "1", structure=self.pulmonic)[0], "written")
        self.assertEqual(self.grade("", "1", conclusion="IP modérée.", structure=self.pulmonic)[0], "text_conflict")

    def test_unmapped_hedged_or_earlier_statements_are_uncertain(self):
        for cell in ("Insuffisance tricuspidienne discrète.", "IT 2/4.", "IT légère, voire modérée.",
                     "IT légère ou modérée.", "IT modérée? ", "IT non évaluée.",
                     "IT légère, modérée à l'examen précédent.", "IT légère, FEVG possiblement réduite."):
            self.assertEqual(self.grade(cell, "1")[0], "text_uncertain", cell)


if __name__ == "__main__":
    unittest.main()
