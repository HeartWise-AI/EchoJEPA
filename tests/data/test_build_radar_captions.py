# tests/data/test_build_radar_captions.py

"""`data/build_radar_captions.py` on synthetic reports: only the split's studies and the trained structures, the
caption in the table is the exact text that was judged, the B-mode scope reaches the caption without changing what it
says, the structured EF is written in, facts and flags survive the round trip through the CSV, a missing report column
is an error, and a table must be patient-disjoint from the held-out splits."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import build_radar_captions as builder  # noqa: E402
import radar_captions as rc  # noqa: E402
import radar_normality as rn  # noqa: E402
import radar_normality_audit as audit  # noqa: E402
from make_study_targets import PLACEHOLDER_IDS, id_text, patient_key, usable_ids  # noqa: E402

from src.datasets.radar_study_dataset import RadarStudyDataset, file_sha256, read_table  # noqa: E402
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter  # noqa: E402
from src.models.radar.routing import load_routing  # noqa: E402
from tests.models.radar_synthetic import DIM, GRID  # noqa: E402

POLICY = rn.load_policy()
ROUTING = load_routing()
NO_ONE = ("", "   ", "''", '""', "' '", "0", "000", "0.0", "-", "_", "UNKNOWN", "unknown", "Inconnu")
SOMEONE = ("ALPHA", "ABCD", "S00", "A0", "A10", "0A", "UNKNOWN2")    # letters or digits: an id of someone
NORMAL = {"lv": "VG non dilaté. Fonction systolique normale.", "mitral": "Valve mitrale normale.",
          "rv": "Ventricule droit de dimension et fonction normales.", "aortic": "Valve aortique normale."}
DOPPLER_ONLY = (
    ("lv", "Gradient moyen de 10 mmHg."), ("lv", "Gradient maximal de 20 mmHg."),
    ("lv", "Pas de critère de dysfonction diastolique."),
    ("lv", "Présence de critères de dysfonction diastolique légère."), ("lv", "Anomalie de relaxation."),
    ("lv", "Impossibilité d'évaluer adéquatement la fonction diastolique en raison de la fibrillation auriculaire."),
    ("lv", "Profil pseudo-normal."), ("lv", "Pression de remplissage du ventricule gauche élevée."),
    ("lv", "Fonction diastolique normale pour l'âge."), ("lv", "Dysfonction diastolique de type restrictif."),
    ("lv", "Dysfonction diastolique de type pseudo-normal."),
    ("rv", "Pression systolique du ventricule droit normale."),
    ("rv", "Augmentation légère de la pression systolique ventriculaire droite."),
    ("rv", "Élévation de la pression systolique du ventricule droit mesurant 45 mmHg."),
    ("rv", "Examen ne permettant pas d'évaluer adéquatement la pression systolique du ventricule droit."),
    ("rv", "Pression systolique du ventricule droit normale sous réserve d'un profil Doppler tricuspidien imparfait."),
    ("mitral", "Trace d'insuffisance mitrale."), ("mitral", "Insuffisance mitrale modérée dirigée postérieurement."),
    ("mitral", "L'aire valvulaire mitrale est calculée à 2.71 cm2."), ("aortic", "Pas de sténose aortique."),
    ("aortic", "Le gradient moyen aortique est de 12 mmHg."), ("aortic", "Insuffisance aortique tout au plus légère."),
)


def report(study_id, **cells):
    """One synthetic report with every column the builder needs."""
    row = {s.report_column: None for s in ROUTING.trained}
    for structure in ROUTING.trained:
        row.update({c: None for c in rn.report_columns(structure.name, POLICY)})
    row["study_id"] = study_id
    names = {s.name: s.report_column for s in ROUTING.trained}
    row.update({names.get(k, k): v for k, v in cells.items()})
    return row


class TestBuildTable(unittest.TestCase):

    def table(self, *reports):
        return self.tables(*reports)[0].set_index(["study_id", "structure"])

    def tables(self, *reports):
        return builder.build_tables(pd.DataFrame(list(reports)), ROUTING, POLICY)

    def test_rows_captions_and_flags(self):
        table = self.table(report("S1", lv="VG non dilaté. Fonction systolique normale.",
                                  pericardium="Pas d'épanchement péricardique.",
                                  mitral="Insuffisance mitrale modérée.", **{"Visually Estimated EF": 60.0}))
        self.assertEqual(sorted(s for _, s in table.index), ["lv", "pericardium"])    # mitral: nothing B-mode is left
        lv = table.loc[("S1", "lv")]
        self.assertEqual(lv.caption, "VG non dilaté. Fonction systolique normale. FEVG estimée visuellement : 60 %.")
        self.assertEqual((lv.normality, lv.ef_status, lv.normality_status), ("normal", "appended", "provisional"))
        self.assertEqual(lv.caption_id, rc.caption_id(lv.caption))
        self.assertEqual(lv.policy_digest, POLICY.digest)
        self.assertEqual(table.loc[("S1", "pericardium")].normality, "normal")

    def test_the_scope_reaches_the_caption_that_is_trained_on(self):
        table = self.table(report("S1", mitral="Valve mitrale calcifiée. Insuffisance mitrale modérée."))
        row = table.loc[("S1", "mitral")]
        self.assertEqual((row.caption, row.dropped_parts, row.normality), ("Valve mitrale calcifiée.", 1, "abnormal"))

    def test_a_caption_that_lost_unread_text_says_so_and_is_not_normal(self):
        kept = "Ventricule droit de dimension et fonction normales."
        table = self.table(report("S1", rv=f"{kept} Pression à 42 mmHg (signe nouveau)"))
        row = table.loc[("S1", "rv")]
        self.assertEqual((row.caption, row.stranded_parts, row.normality), (kept, 1, "unknown"))
        self.assertEqual(audit.table_problems(table.reset_index(), POLICY), {})

    def test_a_caption_of_nothing_but_stranded_text_has_no_row(self):
        table = self.table(report("S1", tricuspid="Insuffisance tricuspidienne légère (1-2/4)"))
        self.assertNotIn(("S1", "tricuspid"), table.index)

    def test_a_cell_removed_in_full_goes_to_the_ledger_and_an_empty_cell_does_not(self):
        table, ledger = self.tables(report("S1", aortic="Sténose aortique calcifiée probable.",
                                           mitral="Insuffisance mitrale légère.", tricuspid="Valve fine.", lv=""))
        self.assertEqual(sorted(table.structure), ["tricuspid"])
        self.assertEqual(list(ledger.columns), list(builder.EXCLUSION_COLUMNS))
        rows = ledger.set_index("structure")
        self.assertEqual(sorted(rows.index), ["aortic", "mitral"])
        self.assertEqual(rows.reason.to_dict(), {"aortic": "stranded_text", "mitral": "doppler_only"})
        self.assertEqual(rows.source["aortic"], "Sténose aortique calcifiée probable.")
        self.assertEqual((rows.dropped_parts["aortic"], rows.stranded_parts["aortic"]), (1, 1))

    def test_a_cell_with_no_word_in_it_is_said_so(self):
        self.assertEqual(self.tables(report("S1", mitral="..."))[1].reason.tolist(), ["no_words"])

    def test_an_lv_with_nothing_left_but_its_structured_ef_is_captioned_not_excluded(self):
        table, ledger = self.tables(report("S1", lv="Anomalie de relaxation.", **{POLICY.ef["column"]: 60}))
        self.assertEqual(table.structure.tolist(), ["lv"])
        self.assertTrue(ledger.empty)

    def test_a_finding_in_a_doppler_segment_stays_or_stops_a_normal_flag(self):
        lv = "VG non dilaté. Fonction systolique normale."
        valve = "Valve aortique normale."
        cases = {
            ("lv", f"{lv} Hypertrophie secondaire à une sténose aortique."): (f"{lv} Hypertrophie.", 0, "abnormal"),
            ("lv", f"{lv} Hypertrophie non secondaire à une sténose."): (f"{lv} Hypertrophie.", 0, "abnormal"),
            ("lv", f"{lv} Hypertrophie probablement due à une sténose."): (f"{lv} Hypertrophie.", 0, "abnormal"),
            ("aortic", f"{valve} Calcification responsable de sténose."): (f"{valve} Calcification.", 0, "unknown"),
            ("aortic", "Valve aortique calcifiée non sténosante."): ("Valve aortique calcifiée.", 0, "abnormal"),
            ("aortic", f"{valve} Sténose aortique calcifiée modérée."):
                (f"{valve} Valve aortique calcifiée.", 0, "unknown"),
            ("aortic", f"{valve} Sténose secondaire à une calcification."):
                (f"{valve} secondaire à une calcification.", 0, "unknown"),
            ("mitral", "Valve mitrale normale. Insuffisance mitrale sur prolapsus de P2."):
                ("Valve mitrale normale. sur prolapsus de P2.", 0, "unknown"),
            ("aortic", f"{valve} Sténose aortique calcifiée probable."): (valve, 1, "unknown"),
        }
        for (structure, cell), expected in cases.items():
            with self.subTest(cell):
                table = self.table(report("S1", **{structure: cell}))
                row = table.loc[("S1", structure)]
                self.assertEqual((row.caption, row.stranded_parts, row.normality), expected)
                self.assertEqual(audit.table_problems(table.reset_index(), POLICY), {})

    def test_doppler_around_an_unknown_never_makes_it_normal(self):
        lv = NORMAL["lv"]
        plain = self.table(report("S1", lv=f"{lv} Masse intracavitaire.")).loc[("S1", "lv")]
        self.assertEqual(plain.normality, "unknown")
        for extra, kept in {"Masse intracavitaire responsable d'un gradient.": f"{lv} Masse intracavitaire.",
                            "Tumeur responsable d'un gradient.": f"{lv} Tumeur.",
                            "Gradient sur masse intracavitaire.": lv, "Gradient lié à une tumeur.": lv}.items():
            with self.subTest(extra):
                table = self.table(report("S1", lv=f"{lv} {extra}"))
                row = table.loc[("S1", "lv")]
                self.assertEqual((row.caption, row.normality), (kept, "unknown"))
                self.assertEqual(audit.table_problems(table.reset_index(), POLICY), {})

    def test_only_what_is_established_as_doppler_leaves_quietly(self):
        for structure, statement in DOPPLER_ONLY:
            kept = NORMAL[structure]
            with self.subTest(statement):
                row = self.table(report("S1", **{structure: f"{kept} {statement}"})).loc[("S1", structure)]
                self.assertEqual((row.caption, row.normality, row.stranded_parts), (kept, "normal", 0))
            for word in ("zorglub", "masse", "anormale", "ventricule"):
                with self.subTest(statement, word=word):
                    cell = f"{kept} {statement.rstrip('.')} {word}."
                    row = self.table(report("S1", **{structure: cell})).loc[("S1", structure)]
                    self.assertGreater(row.stranded_parts, 0)
                    self.assertNotEqual(row.normality, "normal")

    def test_a_clause_lost_for_want_of_a_state_stops_a_normal_flag(self):
        kept = "Ventricule droit non dilaté. Fonction systolique normale."
        for state in ("non normales", "probablement normales", "augmentée"):
            with self.subTest(state):
                cell = f"{kept} Dimension et pression systolique {state}."
                row = self.table(report("S1", rv=cell)).loc[("S1", "rv")]
                self.assertEqual((row.caption, row.normality), (kept, "unknown"))
                self.assertIn("stranded_text", row.reasons)

    def test_a_doppler_term_that_looks_like_a_finding_costs_no_normal_flag(self):
        kept = "VG non dilaté. Fonction systolique normale."
        for term in ["Dysfonction diastolique légère.", "Présence de critères de dysfonction diastolique modérée.",
                     "Anomalie de relaxation.", "Anomalies de la relaxation.", "Anomalie du remplissage.",
                     "Fonction diastolique normale."]:
            with self.subTest(term):
                row = self.table(report("S1", lv=f"{kept} {term}")).loc[("S1", "lv")]
                self.assertEqual((row.caption, row.stranded_parts, row.normality), (kept, 0, "normal"))
        for where in ("centro-prothétique", "péri-prothétique", "intra prothétique"):
            with self.subTest(where):
                cell = f"Valve aortique normale. Trace d'insuffisance aortique {where}."
                self.assertEqual(self.table(report("S1", aortic=cell)).loc[("S1", "aortic")].stranded_parts, 0)

    def test_a_size_at_end_diastole_is_b_mode_and_is_read(self):
        function = "Fonction systolique normale."
        for size, flag in {"Augmentation légère du diamètre diastolique.": "abnormal",
                           "Ventricule gauche de dimension diastolique augmentée.": "abnormal",
                           "Ventricule gauche de dimension télé-diastolique augmentée.": "abnormal",
                           "Ventricule gauche de dimension télé- diastolique augmentée.": "abnormal",
                           "Ventricule gauche de dimension diastolique normale.": "normal",
                           "Ventricule gauche de volume télé diastolique normal.": "normal"}.items():
            with self.subTest(size):
                row = self.table(report("S1", lv=f"{size} {function}")).loc[("S1", "lv")]
                self.assertEqual((row.caption, row.dropped_parts, row.normality), (f"{size} {function}", 0, flag))

    def test_removing_doppler_never_changes_what_a_caption_says(self):
        tail = "de dimension, fonction et pression systolique"
        kept = {f"Ventricule droit {tail} normales.": ("Ventricule droit de dimension, fonction normales.", "normal"),
                f"Ventricule droit {tail} anormales.":
                    ("Ventricule droit de dimension, fonction anormales.", "abnormal"),
                "Ventricule droit dilaté, fonction et pression systolique normales.":
                    ("Ventricule droit dilaté, fonction normales.", "abnormal")}
        for cell, expected in kept.items():
            with self.subTest(cell):
                row = self.table(report("S1", rv=cell)).loc[("S1", "rv")]
                self.assertEqual((row.caption, row.normality), expected)
        for state in ["non normales", "probablement normales", "globalement normales", "augmentée"]:
            with self.subTest(state):
                self.assertEqual(len(self.table(report("S1", rv=f"Ventricule droit {tail} {state}."))), 0)
        cell = "VD non dilaté. Fonction et pression systolique non normales."
        row = self.table(report("S1", rv=cell)).loc[("S1", "rv")]
        self.assertEqual((row.caption, row.normality), ("VD non dilaté.", "unknown"))

    def test_untrained_structures_get_no_row(self):
        untrained = next(s for s in ROUTING.structures if not s.train and s.report_column)
        table = self.table({**report("S1", lv="VG normal"), untrained.report_column: "Normale"})
        self.assertEqual([s for _, s in table.index], ["lv"])

    def test_missing_cells_are_no_caption_never_nan(self):
        table = self.table(report("S1", lv=float("nan"), pericardium=float("nan")))
        self.assertEqual(len(table), 0)
        table = self.table(report("S1", lv=float("nan"), **{"Visually Estimated EF": 45.0}))
        self.assertEqual(table.loc[("S1", "lv")].caption, "FEVG estimée visuellement : 45 %.")

    def test_conflicting_evidence_is_kept_in_the_row(self):
        table = self.table(report("S1", lv="VG non dilaté. Fonction systolique normale.",
                                  **{"LVID Diastole (2D)": 6.5, "Visually Estimated EF": 60.0}))
        row = table.loc[("S1", "lv")]
        self.assertEqual(row.normality, "unknown")
        self.assertEqual(json.loads(row.conflicts)[0]["attribute"], "lv_size")
        self.assertIn("conflict:lv_size", json.loads(row.reasons))
        facts = json.loads(row.facts)
        self.assertIn({"attribute": "lv_size", "value": "abnormal", "source": "structured",
                       "evidence": "LVID Diastole (2D)", "rule_version": POLICY.rule_version, "finding": "",
                       "negated": False, "low": 6.5, "high": 6.5, "unit": "cm", "narrow": False}, facts)

    def test_a_missing_report_column_is_an_error(self):
        frame = pd.DataFrame([report("S1", lv="VG normal")]).drop(columns=["LVID Diastole (2D)"])
        with self.assertRaisesRegex(SystemExit, "LVID Diastole"):
            builder.build_tables(frame, ROUTING, POLICY)


class TestRecoveredText(unittest.TestCase):
    """What recovery must not do, through the builder: a row that says more than the report goes to the ledger."""

    def test_a_cell_recovery_cannot_trust_goes_to_the_ledger(self):
        for structure, cell in [("rv", "Ventricule droit de dimension et fonction (pression systolique normale)."),
                                ("lv", "Dimensions et fonction systolique (pression pulmonaire normale)."),
                                ("mitral", "Bioprothèse mitrale en 2019 non implantée."),
                                ("mitral", "Plastie mitrale en 2019 non réalisée."),
                                ("mitral", "À confirmer insuffisance mitrale sur prolapsus."),
                                ("mitral", "Probable insuffisance mitrale sur prolapsus.")]:
            with self.subTest(cell):
                reports = pd.DataFrame([report("s1", **{structure: cell})])
                table, ledger = builder.build_tables(reports, ROUTING, POLICY)
                self.assertEqual(table[table.structure == structure].normality.tolist(), [])
                self.assertEqual(ledger[ledger.structure == structure].reason.tolist(), ["stranded_text"])

    def test_definite_statements_still_come_through(self):
        for structure, cell, caption, flag in [
                ("mitral", "Insuffisance mitrale sur prolapsus.", "sur prolapsus.", "abnormal"),
                ("mitral", "Insuffisance mitrale sur possible prolapsus.", "sur possible prolapsus.", "unknown")]:
            with self.subTest(cell):
                table, _ = builder.build_tables(pd.DataFrame([report("s1", **{structure: cell})]), ROUTING, POLICY)
                row = table[table.structure == structure].iloc[0]
                self.assertEqual((row.caption, row.normality), (caption, flag))


class TestIdentifiers(unittest.TestCase):

    def test_a_float_tail_goes_and_zeros_at_the_front_stay(self):
        self.assertEqual(id_text(pd.Series([10.0, 20.0, None])).tolist()[:2], ["10", "20"])
        self.assertEqual(id_text(pd.Series(["10.0", "001", "'002'", "1.2.0"])).tolist(), ["10", "001", "002", "1.2.0"])

    def test_a_patient_key_is_the_id_without_its_zeros_and_missing_for_no_one(self):
        self.assertEqual(patient_key(pd.Series(["00010", "10", 10.0, "0A"])).tolist(), ["10", "10", "10", "A"])
        self.assertTrue(patient_key(pd.Series(list(NO_ONE))).isna().all())

    def test_the_join_and_the_patient_check_read_an_id_alike(self):
        self.assertEqual(usable_ids(pd.Series([10.0])).tolist(), id_text(pd.Series([10.0])).tolist())


class TestWriting(unittest.TestCase):

    def test_a_failed_write_leaves_the_old_file_whole(self):
        def fails(temporary):
            with open(temporary, "w") as f:
                f.write("half")
            raise OSError("disk full")

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "captions.csv")
            with open(path, "w") as f:
                f.write("whole")
            with self.assertRaises(OSError):
                builder.write_atomic(path, fails)
            with open(path) as f:
                self.assertEqual(f.read(), "whole")


class TestCommandLine(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        reports = [report("S1", lv="VG normal"), report("S2", lv="VG dilaté"), report("S3", lv="VG non dilaté")]
        frame = pd.DataFrame(reports).assign(AccessionNumber=["1", "2", "3"], Dossier=["10", "20", "30"])
        self.paths = {name: os.path.join(self.root, f"{name}.csv") for name in ("reports", "pacs", "clips", "splits")}
        frame.drop(columns=["study_id"]).to_csv(self.paths["reports"], index=False)
        self.write_pacs(["S1", "S2", "S3"], ["10", "20", "30"])
        self.write_splits(["S1", "S2", "S3"], ["train", "train", "val"])
        pd.DataFrame({"study_id": ["S1", "S2", "S3"], "clip_id": ["c1", "c2", "c3"], "view": ["A4C"] * 3,
                      "split": ["train", "train", "val"]}).to_csv(self.paths["clips"], index=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_pacs(self, studies, patients):
        accessions = [str(i + 1) for i in range(len(studies))]
        pd.DataFrame({"AccessionNumber": accessions, "StudyInstanceUID": studies,
                      "PatientID": patients}).to_csv(self.paths["pacs"], index=False)

    def write_splits(self, studies, splits, **columns):
        pd.DataFrame({"study_id": studies, "split": splits, **columns}).to_csv(self.paths["splits"], index=False)

    def run_builder(self, *extra):
        out = os.path.join(self.root, "out")
        with contextlib.redirect_stdout(io.StringIO()):
            builder.main(["--reports", self.paths["reports"], "--pacs", self.paths["pacs"], "--clips",
                          self.paths["clips"], "--splits", self.paths["splits"], "--out-dir", out, *extra])
        return out

    def test_a_table_holds_one_split_and_agrees_with_its_rules(self):
        out = self.run_builder()
        table = audit.read_captions(os.path.join(out, "captions.csv"))
        self.assertEqual(sorted(table.study_id), ["S1", "S2"])            # S3 is in val
        self.assertEqual(audit.table_problems(table, POLICY), {})
        with open(os.path.join(out, "build_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        self.assertEqual((info["split"], info["rows"], info["policy_status"]), ("train", 2, "provisional"))
        self.assertEqual(sorted(info), ["captions_sha256", "clips_sha256", "excluded", "exclusions_sha256", "flags",
                                        "pacs_sha256", "patients", "policy_digest", "policy_status", "reports_sha256",
                                        "routing_sha256", "rows", "rule_version", "split", "splits_sha256",
                                        "studies_in_split", "studies_linked"])
        self.assertEqual(info["patients"], {"in_table": 2, "held_out": 1, "held_out_studies": 1})

    def test_a_patient_with_a_held_out_study_stops_the_build(self):
        self.write_pacs(["S1", "S2", "S3"], ["10", "20", "10"])
        with self.assertRaisesRegex(SystemExit, "1 patients of split 'train' also have a held-out study"):
            self.run_builder()
        self.assertFalse(os.path.exists(os.path.join(self.root, "out", "captions.csv")))

    def test_the_patient_id_is_compared_without_its_leading_zeros(self):
        self.write_pacs(["S1", "S2", "S3"], ["0010", "20", "10"])
        with self.assertRaisesRegex(SystemExit, "held-out study"):
            self.run_builder()

    def test_a_held_out_study_removed_by_screening_still_counts(self):
        self.write_pacs(["S1", "S2", "S3", "S4"], ["10", "20", "30", "10"])
        self.write_splits(["S1", "S2", "S3", "S4"], ["train", "train", "val", "test"])    # S4 has no clip
        with self.assertRaisesRegex(SystemExit, "held-out study"):
            self.run_builder()

    def test_the_split_tables_patient_column_counts_too(self):
        self.write_splits(["S1", "S2", "S3", "S4"], ["train", "train", "val", "test"],
                          patient_id=["10", "20", "30", "20"])                             # S4 is not in PACS
        with self.assertRaisesRegex(SystemExit, "held-out study"):
            self.run_builder()

    def test_a_study_whose_patient_is_unknown_stops_the_build(self):
        self.write_splits(["S1", "S2", "S3", "S4"], ["train", "train", "val", "test"])    # S4 nowhere in PACS
        with self.assertRaisesRegex(SystemExit, "no usable patient id"):
            self.run_builder()

    def test_an_id_that_identifies_no_one_is_a_missing_patient(self):
        for value in NO_ONE:
            for patients in (["10", "20", value], [value, "20", "30"]):    # a held-out study, a study of the table
                with self.subTest(patients=patients):
                    self.write_pacs(["S1", "S2", "S3"], patients)
                    with self.assertRaisesRegex(SystemExit, "1 studies have no usable patient id"):
                        self.run_builder()

    def test_the_split_tables_ids_follow_the_same_rule(self):
        self.write_pacs(["S1", "S2"], ["10", "20"])                                        # S3 is not in PACS
        for value in NO_ONE:
            with self.subTest(value=value):
                self.write_splits(["S1", "S2", "S3"], ["train", "train", "val"], patient_id=["10", "20", value])
                with self.assertRaisesRegex(SystemExit, "1 studies have no usable patient id"):
                    self.run_builder()

    def test_either_source_can_give_the_patient(self):
        self.write_pacs(["S1", "S2", "S3"], ["10", "20", "   "])
        self.write_splits(["S1", "S2", "S3"], ["train", "train", "val"], patient_id=["10", "20", "10"])
        with self.assertRaisesRegex(SystemExit, "held-out study"):
            self.run_builder()
        self.write_splits(["S1", "S2", "S3"], ["train", "train", "val"], patient_id=["10", "20", "30"])
        with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["patients"], {"in_table": 2, "held_out": 1, "held_out_studies": 1})

    def test_a_held_out_study_of_the_clip_table_counts_even_if_the_split_table_forgets_it(self):
        self.write_pacs(["S1", "S2", "S3"], ["10", "20", "10"])
        self.write_splits(["S1", "S2", "S0"], ["train", "train", "val"], patient_id=["10", "20", "99"])
        with self.assertRaisesRegex(SystemExit, "held-out study"):
            self.run_builder()

    def test_nothing_to_compare_stops_the_build(self):
        self.write_splits(["S1", "S2"], ["train", "train"])
        pd.read_csv(self.paths["clips"], dtype=str).iloc[:2].to_csv(self.paths["clips"], index=False)
        with self.assertRaisesRegex(SystemExit, "nothing to check the table against"):
            self.run_builder()

    def test_an_id_written_as_a_float_is_the_same_patient(self):
        self.write_pacs(["S1", "S2"], ["10", "20"])
        self.write_splits(["S1", "S2", "S3"], ["train", "train", "val"], patient_id=["10", "20", "10.0"])
        with self.assertRaisesRegex(SystemExit, "held-out study"):
            self.run_builder()

    def test_too_few_linked_studies_stop_the_build(self):
        self.write_pacs(["S1", "OTHER", "S3"], ["10", "20", "30"])
        self.write_splits(["S1", "S2", "S3"], ["train", "train", "val"], patient_id=["10", "20", "30"])
        with self.assertRaisesRegex(SystemExit, "Only 1 of the split's 2 studies"):
            self.run_builder()
        with open(os.path.join(self.run_builder("--min-linked", "0.5"), "build_info.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["studies_linked"], 1)

    def test_a_clip_study_id_that_is_not_plain_stops_the_build(self):
        for study in ("'S1'", '"S1"', " S1", "S1 ", ""):
            with self.subTest(study):
                pd.read_csv(self.paths["clips"], dtype=str).assign(study_id=[study, "S2", "S3"]).to_csv(
                    self.paths["clips"], index=False)
                with self.assertRaisesRegex(SystemExit, "1 clips have a study id"):
                    self.run_builder()

    def write_patients(self, report, pacs):
        """S1's patient id in its report and in PACS; the split table still knows every patient."""
        frame = pd.read_csv(self.paths["reports"], dtype=str, keep_default_na=False)
        frame.assign(Dossier=[report, "20", "30"]).to_csv(self.paths["reports"], index=False)
        self.write_pacs(["S1", "S2", "S3"], [pacs, "20", "30"])
        self.write_splits(["S1", "S2", "S3"], ["train", "train", "val"], patient_id=["10", "20", "30"])

    def test_two_ids_that_name_no_one_do_not_agree(self):
        for patient in ("   ", "000", "UNKNOWN", "unknown", "-", "''", "N/A"):
            with self.subTest(patient):
                self.write_patients(patient, patient)
                with self.assertRaisesRegex(SystemExit, "Only 1 of the split's 2 studies"):
                    self.run_builder()

    def test_a_report_links_only_with_a_usable_id_on_both_sides(self):
        for report_id, pacs_id in (("000", "10"), ("10", "000"), ("UNKNOWN", "10"), ("10", "UNKNOWN")):
            with self.subTest(report=report_id, pacs=pacs_id):
                self.write_patients(report_id, pacs_id)
                with open(os.path.join(self.run_builder("--min-linked", "0.5"), "build_info.json"),
                          encoding="utf-8") as f:
                    self.assertEqual(json.load(f)["studies_linked"], 1)

    def test_a_study_named_by_no_real_id_has_no_patient(self):
        self.write_pacs(["000", "S2", "S3"], ["10", "20", "30"])
        self.write_splits(["000", "S2", "S3"], ["train", "train", "val"])
        pd.read_csv(self.paths["clips"], dtype=str).assign(study_id=["000", "S2", "S3"]).to_csv(self.paths["clips"],
                                                                                                index=False)
        with self.assertRaisesRegex(SystemExit, "no usable patient id"):
            self.run_builder()

    def test_zeros_at_the_front_of_a_patient_id_do_not_count(self):
        self.write_patients("00010", "10")
        with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["studies_linked"], 2)

    def test_a_missing_held_out_patient_id_leaves_the_others_as_written(self):
        frame = pd.read_csv(self.paths["reports"], dtype=str)
        frame.assign(Dossier=["10", "20", ""]).to_csv(self.paths["reports"], index=False)
        with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["studies_linked"], 2)

    def test_zeros_at_the_front_of_an_accession_are_part_of_it(self):
        frame = pd.read_csv(self.paths["reports"], dtype=str)
        frame.assign(AccessionNumber=["001", "002", "003"]).to_csv(self.paths["reports"], index=False)
        pacs = pd.read_csv(self.paths["pacs"], dtype=str)
        pacs.assign(AccessionNumber=["001", "002", "003"]).to_csv(self.paths["pacs"], index=False)
        with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["studies_linked"], 2)

    def test_patient_ids_in_a_float_column_of_a_parquet_export_link(self):
        frame = pd.read_csv(self.paths["reports"], dtype=str).assign(Dossier=[10.0, 20.0, None])
        path = os.path.join(self.root, "reports.parquet")
        frame.to_parquet(path)
        self.paths["reports"] = path
        with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["studies_linked"], 2)

    def test_accessions_in_a_float_column_of_a_parquet_export_link(self):
        frame = pd.read_csv(self.paths["reports"], dtype=str).assign(AccessionNumber=[1.0, 2.0, 3.0])
        path = os.path.join(self.root, "reports.parquet")
        frame.to_parquet(path)
        self.paths["reports"] = path
        with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["studies_linked"], 2)

    def test_what_the_builder_writes_the_dataset_reads(self):
        clips = pd.read_csv(self.paths["clips"], dtype=str).assign(acquisition="bmode")
        clips.to_csv(self.paths["clips"], index=False)
        out = self.run_builder()
        writer = TokenCacheWriter(os.path.join(self.root, "cache"), GRID, DIM, provenance={"synthetic": True})
        for clip in clips.clip_id:
            writer.add(clip, torch.zeros(GRID[0] * GRID[1] * GRID[2], DIM))
        writer.close()
        cache = TokenCache(os.path.join(self.root, "cache"), token_grid=GRID, embed_dim=DIM)
        table = read_table(self.paths["clips"])
        dataset = RadarStudyDataset(table[table.split == "train"], read_table(os.path.join(out, "captions.csv"),
                                                                              id_columns=("study_id",)),
                                    ROUTING, cache, num_clips=1)
        self.assertEqual(sorted(study for study, _ in dataset.studies), ["S1", "S2"])
        self.assertEqual(sorted(key[0] for key in dataset.captions), ["S1", "S2"])

    def test_the_ledger_is_written_and_sealed(self):
        frame = pd.read_csv(self.paths["reports"], dtype=str)
        aortic = next(s.report_column for s in ROUTING.trained if s.name == "aortic")
        frame[aortic] = "Sténose aortique calcifiée probable."
        frame.to_csv(self.paths["reports"], index=False)
        out = self.run_builder()
        ledger = os.path.join(out, "exclusions.csv")
        table = pd.read_csv(ledger, dtype=str)
        self.assertEqual(sorted(table.study_id), ["S1", "S2"])
        self.assertEqual(set(table.reason), {"stranded_text"})
        with open(os.path.join(out, "build_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        self.assertEqual(info["excluded"], {"stranded_text": 2})
        self.assertEqual(info["exclusions_sha256"], file_sha256(ledger))
        self.assertNotIn("aortic", set(pd.read_csv(os.path.join(out, "captions.csv")).structure))

    def test_a_failed_build_leaves_the_old_table_whole(self):
        out = self.run_builder()
        path = os.path.join(out, "captions.csv")
        with open(path, "rb") as f:
            before = f.read()
        with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run_builder()
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_a_blank_accession_links_nothing(self):
        frame = pd.read_csv(self.paths["reports"], dtype=str)
        frame = frame.assign(AccessionNumber=["", "2", "3"], Dossier=["10"] * 3)
        pacs = pd.DataFrame({"AccessionNumber": ["", "2", "3"], "StudyInstanceUID": ["S1", "S2", "S3"],
                             "PatientID": ["10"] * 3})
        args = builder.parse_args(["--reports", "r", "--pacs", "p", "--clips", "c", "--splits", "s", "--out-dir", "o"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sorted(builder.link_studies(frame, pacs, args).study_id), ["S2", "S3"])

    def test_a_report_without_a_patient_is_counted_as_dropped(self):
        frame = pd.read_csv(self.paths["reports"], dtype=str).assign(Dossier=[None, "20", "30"])
        pacs = pd.DataFrame({"AccessionNumber": ["1", "2", "3"], "StudyInstanceUID": ["S1", "S2", "S3"],
                             "PatientID": ["10", "20", "30"]})
        args = builder.parse_args(["--reports", "r", "--pacs", "p", "--clips", "c", "--splits", "s", "--out-dir", "o"])
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            builder.link_studies(frame, pacs, args)
        self.assertIn("1 with another or no patient id dropped", printed.getvalue())

    def test_any_other_id_is_a_patient(self):
        for value in SOMEONE:
            with self.subTest(value=value):
                self.write_pacs(["S1", "S2", "S3"], ["10", "20", value])
                pd.read_csv(self.paths["reports"], dtype=str).assign(Dossier=["10", "20", value]).to_csv(
                    self.paths["reports"], index=False)
                with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
                    self.assertEqual(json.load(f)["patients"], {"in_table": 2, "held_out": 1, "held_out_studies": 1})
                self.write_pacs(["S1", "S2", "S3"], [value, "20", value])
                with self.assertRaisesRegex(SystemExit, "held-out study"):
                    self.run_builder()

    def test_any_other_id_is_a_study(self):
        for value in SOMEONE:
            with self.subTest(value=value):
                self.write_pacs(["S1", "S2", value], ["10", "20", "30"])
                self.write_splits(["S1", "S2", value], ["train", "train", "val"])
                pd.read_csv(self.paths["clips"], dtype=str).assign(study_id=["S1", "S2", value]).to_csv(
                    self.paths["clips"], index=False)
                with open(os.path.join(self.run_builder(), "build_info.json"), encoding="utf-8") as f:
                    self.assertEqual(json.load(f)["patients"]["held_out_studies"], 1)

    def test_what_names_no_one_and_what_does(self):
        words = sorted(PLACEHOLDER_IDS) + [w.lower() for w in PLACEHOLDER_IDS]
        missing = pd.Series([*NO_ONE, *words, None, float("nan")], dtype="object")
        self.assertTrue(usable_ids(missing).isna().all())
        self.assertEqual(usable_ids(pd.Series(SOMEONE)).tolist(), list(SOMEONE))

    def test_a_blank_study_id_is_no_study(self):
        studies, patients, splits = ["S1", "S2", "S3"], ["10", "20", "30"], ["train", "train", "train"]
        self.write_pacs([*studies, "S4"], [*patients, "40"])
        self.write_splits([*studies, "S4"], [*splits, "val"])
        self.run_builder()                 # the same table with a real id builds, so the blank is what stops it
        for value in ("", "   "):
            with self.subTest(value=value):
                self.write_pacs([*studies, value], [*patients, "40"])
                self.write_splits([*studies, value], [*splits, "val"])
                with self.assertRaisesRegex(SystemExit, "1 studies have no usable patient id"):
                    self.run_builder()

    def test_a_study_outside_its_split_stops_the_build(self):
        self.write_splits(["S1", "S2", "S3"], ["train", "val", "val"])
        with self.assertRaisesRegex(SystemExit, "not in split 'train'"):
            self.run_builder()

    def test_the_split_table_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            builder.main(["--reports", self.paths["reports"], "--pacs", self.paths["pacs"], "--clips",
                          self.paths["clips"], "--out-dir", os.path.join(self.root, "out")])

    def test_output_inside_the_repository_is_refused(self):
        with self.assertRaises(SystemExit):
            builder.main(["--reports", self.paths["reports"], "--pacs", self.paths["pacs"], "--clips",
                          self.paths["clips"], "--splits", self.paths["splits"], "--out-dir",
                          os.path.join(REPO, "captions_here")])

    def test_an_empty_split_is_an_error(self):
        with self.assertRaisesRegex(SystemExit, "No study"):
            self.run_builder("--split", "test")


if __name__ == "__main__":
    unittest.main()
