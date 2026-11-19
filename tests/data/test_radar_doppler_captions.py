# tests/data/test_radar_doppler_captions.py

"""Doppler captions for structures fed by stills: the Doppler clauses `retain_caption` removes come back as written
(a dated clause still leaves), the full caption is the B-mode caption followed by them, Hemo's templated caption and
provisional flag, and the caption builder's `--doppler-captions` rows (behind the flag only; the table without it is
the v0 table)."""

import math
import os
import tempfile
import shutil
import sys
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import build_radar_captions as builder  # noqa: E402
import radar_doppler_captions as rdc  # noqa: E402
import radar_normality as rn  # noqa: E402

from src.models.radar.routing import DEFAULT_ROUTING, load_routing  # noqa: E402

POLICY = rn.load_policy()
V0 = load_routing()
V1 = load_routing(os.path.join(os.path.dirname(DEFAULT_ROUTING), "radar_routing_v1_hemo.yaml"))
AORTIC = ("Valve aortique calcifiée. Sténose aortique modérée, gradient moyen 28 mmHg. "
          "Insuffisance aortique légère. Comparé à l'examen du 12/03/2021, sténose aortique stable.")
HEMO = {"RV Systolic Pressure": 42.0, "RA Pressure": 3.0, "MV E/e' (Lateral)": 10.0, "MV E/e' (Septal)": 14.0}


def report(study_id, routing, **cells):
    row = {s.report_column: None for s in routing.trained if s.report_column}
    for structure in routing.trained:
        row.update({c: None for c in rn.report_columns(structure.name, POLICY)})
    row.update({c: None for c in rdc.hemo_columns()})
    row["study_id"] = study_id
    names = {s.name: s.report_column for s in routing.trained}
    row.update({names.get(k) or k: v for k, v in cells.items()})
    return row


class TestRules(unittest.TestCase):

    def test_doppler_clauses(self):
        self.assertEqual(rdc.doppler_clauses(AORTIC),
                         ["Sténose aortique modérée, gradient moyen 28 mmHg", "Insuffisance aortique légère"])
        self.assertEqual(rdc.doppler_clauses("Valve aortique normale."), [])
        self.assertEqual(rdc.doppler_clauses(None), [])

    def test_full_caption(self):
        bmode = rn.retain_caption(AORTIC)[0]
        self.assertEqual(bmode, "Valve aortique calcifiée.")
        self.assertEqual(rdc.full_caption(bmode, AORTIC), "Valve aortique calcifiée. Sténose aortique modérée, "
                                                          "gradient moyen 28 mmHg. Insuffisance aortique légère.")
        self.assertEqual(rdc.full_caption("", "Pas de sténose aortique."), "Pas de sténose aortique.")
        self.assertEqual(rdc.full_caption(bmode, "Valve aortique calcifiée."), "")

    def test_full_caption_says_nothing_twice_and_nothing_of_an_earlier_examination(self):
        for source, full in (("Valve mitrale calcifiée sans sténose", "Valve mitrale calcifiée sans sténose."),
                             ("Feuillets épaissis. Valve calcifiée sans sténose",
                              "Feuillets épaissis. Valve calcifiée sans sténose.")):
            with self.subTest(source=source):
                self.assertEqual(rdc.full_caption(rn.retain_caption(source)[0], source), full)
        prior = "Sténose aortique sévère. Gradient moyen 45 mmHg, comparé à 30 mmHg à l'examen précédent"
        self.assertEqual(rdc.doppler_clauses(prior), ["Sténose aortique sévère"])

    def test_hemo(self):
        self.assertEqual(rdc.hemo_caption(HEMO), "Pression systolique ventriculaire droite estimée à 42 mmHg. "
                                                 "Pression auriculaire droite estimée à 3 mmHg. "
                                                 "Rapport E/e' moyen 12,0.")
        self.assertEqual(rdc.hemo_flag(HEMO), "abnormal")
        normal = dict(HEMO, **{"RV Systolic Pressure": 28.0})
        self.assertEqual(rdc.hemo_flag(normal), "normal")
        self.assertEqual(rdc.hemo_caption({"RV Systolic Pressure": 900.0}), "")       # implausible: left out
        self.assertEqual(rdc.hemo_caption({"MV E/e' (Lateral)": 10.0, "MV E/e' (Septal)": math.nan}), "")
        self.assertEqual(rdc.hemo_flag({}), "unknown")


class TestBuilder(unittest.TestCase):

    def test_off_by_default(self):
        reports = pd.DataFrame([report("S1", V0, aortic=AORTIC)])
        table, _ = builder.build_tables(reports, V0, POLICY)
        self.assertEqual(list(table.columns), list(builder.COLUMNS))
        table1, _ = builder.build_tables(pd.DataFrame([report("S1", V1, aortic=AORTIC, **HEMO)]), V1, POLICY)
        self.assertNotIn("hemo", set(table1.structure))        # v1 without the flag: no templated caption
        self.assertTrue(table.drop(columns="policy_digest").equals(table1.drop(columns="policy_digest")))
        with self.assertRaisesRegex(SystemExit, "still_structs"):
            builder.build_tables(reports, V0, POLICY, doppler=True)

    def test_default_registry_has_no_hemo_caption(self):
        plain = load_routing(os.path.join(os.path.dirname(DEFAULT_ROUTING), "radar_routing_v1.yaml"))
        self.assertFalse(plain.structure("hemo").train)
        self.assertEqual(plain.still_structs, V1.still_structs)          # the stills are routed alike
        reports = pd.DataFrame([report("S1", plain, aortic=AORTIC, **HEMO)])
        table, _ = builder.build_tables(reports, plain, POLICY, doppler=True)
        self.assertNotIn("hemo", set(table.structure))                   # regression only: no contrastive caption
        self.assertIn(("aortic", "full:regurgitation+stenosis"), set(zip(table.structure, table.scope)))
        self.assertNotIn(("aortic", "full"), set(zip(table.structure, table.scope)))

    def test_hemo_states_only_the_supported_fields(self):
        self.assertEqual(rdc.hemo_caption(HEMO, ["e_over_e_prime"]), "Rapport E/e' moyen 12,0.")
        self.assertEqual(rdc.hemo_caption(HEMO, []), "")
        self.assertEqual(rdc.hemo_flag(HEMO, ["e_over_e_prime"]), "normal")
        hemo = V1.structure("hemo")
        self.assertEqual(hemo.supported_fields({"TR"}), ("rvsp",))
        self.assertEqual(hemo.supported_fields({"MV_INFLOW"}), ())       # E without e': E/e' is not supported
        self.assertEqual(hemo.supported_fields({"MV_INFLOW", "TDI", "TR", "AV"}), ("rvsp", "e_over_e_prime"))

    def test_full_rows(self):
        reports = pd.DataFrame([report("S1", V1, aortic=AORTIC, lv="VG non dilaté.", **HEMO),
                                report("S2", V1, aortic="Pas de sténose aortique.")])
        table, ledger = builder.build_tables(reports, V1, POLICY, doppler=True)
        rows = table.set_index(["study_id", "structure", "scope"])
        self.assertEqual(rows.loc[("S1", "aortic", "bmode"), "caption"], "Valve aortique calcifiée.")
        full = rows.loc[("S1", "aortic", "full:regurgitation+stenosis")]
        self.assertIn("gradient moyen 28 mmHg", full.caption)
        self.assertEqual((full.normality, full.normality_status), ("unknown", "provisional"))
        self.assertNotIn(("S1", "lv", "full"), rows.index)     # LV is fed by no still
        self.assertEqual(rows.loc[("S1", "hemo", "ev:rvsp"), "normality"], "abnormal")
        self.assertEqual(rows.loc[("S1", "hemo", "ev:rvsp"), "caption"],
                         "Pression systolique ventriculaire droite estimée à 42 mmHg.")
        self.assertEqual(rows.loc[("S1", "hemo", "ev:e_over_e_prime"), "normality"], "normal")       # E/e' 12 alone
        self.assertIn("E/e'", rows.loc[("S1", "hemo", "ev:e_over_e_prime+rvsp"), "caption"])
        hemo = table[table.structure == "hemo"]
        self.assertEqual(len(hemo), 3)                         # one row per supportable set, never a fixed sentence
        self.assertFalse(hemo.caption.str.contains("auriculaire").any())     # RA pressure: no still shows the IVC
        self.assertNotIn(("S1", "hemo", "full"), rows.index)
        self.assertEqual(rows.loc[("S2", "aortic", "full:stenosis"), "caption"], "Pas de sténose aortique.")
        self.assertNotIn(("S2", "aortic", "full:regurgitation"), rows.index)      # the report states none
        self.assertNotIn(("S2", "aortic", "bmode"), rows.index)
        self.assertEqual(ledger.set_index(["study_id", "structure"]).loc[("S2", "aortic"), "reason"],
                         "doppler_only")                       # the B-mode ledger is unchanged
        self.assertEqual(rows.loc[("S1", "aortic", "doppler:regurgitation+stenosis"), "caption"],
                         "Sténose aortique modérée, gradient moyen 28 mmHg. Insuffisance aortique légère.")
        # a bag whose stills show one finding reads that finding's clauses only (an AV still: no regurgitation)
        self.assertEqual(rows.loc[("S1", "aortic", "doppler:stenosis"), "caption"],
                         "Sténose aortique modérée, gradient moyen 28 mmHg.")
        self.assertEqual(rows.loc[("S1", "aortic", "full:regurgitation"), "caption"],
                         "Valve aortique calcifiée. Insuffisance aortique légère.")
        self.assertNotIn(("S1", "aortic", "full"), rows.index)
        aortic = V1.structure("aortic")
        self.assertEqual(aortic.supported_findings({"LVOT"}), ())           # an LVOT still shows neither
        self.assertEqual(aortic.supported_findings({"LVOT", "AV"}), ("stenosis",))
        self.assertEqual(aortic.supported_findings({"AVA", "AR"}), ("stenosis", "regurgitation"))
        self.assertEqual(rdc.clause_findings("Pas de sténose ni d'insuffisance aortique"),
                         {"stenosis", "regurgitation"})
        self.assertEqual(rdc.doppler_caption("Gradient moyen 28 mmHg.", ["stenosis"]), "")   # names no finding
        self.assertNotIn(("S1", "hemo", "doppler"), rows.index)

    def test_a_study_with_stills_and_no_clip_gets_doppler_statements_only(self):
        reports = pd.DataFrame([report("S1", V1, aortic=AORTIC, lv="VG non dilaté.", **HEMO)])
        table, ledger = builder.build_tables(reports, V1, POLICY, doppler=True, still_only={"S1"})
        self.assertEqual(set(zip(table.structure, table.scope)),
                         {("aortic", "doppler:stenosis"), ("aortic", "doppler:regurgitation"),
                          ("aortic", "doppler:regurgitation+stenosis"), ("hemo", "ev:rvsp"),
                          ("hemo", "ev:e_over_e_prime"), ("hemo", "ev:e_over_e_prime+rvsp")})
        self.assertFalse(table[table.structure == "aortic"].caption.str.contains("calcifiée").any())
        self.assertTrue(ledger.empty)

    def test_still_studies_join_the_split_and_its_patient_check(self):
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, "stills.csv")
            pd.DataFrame({"study_id": ["C1", "C1", "T1", "V1"], "split": ["train", "train", "train", "val"],
                          "still_id": list("abcd")}).to_csv(path, index=False)
            splits = pd.DataFrame({"study_id": ["C1", "C2"], "split": ["train", "val"]})
            extended, still_only, elsewhere = builder.add_still_studies(splits, path, "train", {"C1", "C2"})
            self.assertEqual((still_only, elsewhere), ({"T1"}, {"V1"}))
            self.assertEqual(dict(zip(extended.study_id, extended.split)),
                             {"C1": "train", "C2": "val", "T1": "train", "V1": "val"})
            tagged = pd.read_csv(path).assign(still_tag=["AV", "", "", "TR"])       # T1's only still has no tag
            tagged.to_csv(path, index=False)
            _, still_only, elsewhere = builder.add_still_studies(splits, path, "train", {"C1", "C2"},
                                                                 routed_tags={"AV", "TR"})
            self.assertEqual((still_only, elsewhere), (set(), {"V1"}))      # nothing of T1 feeds a structure
            moved = pd.DataFrame({"study_id": ["C1", "C2"], "split": ["val", "val"]})
            with self.assertRaisesRegex(SystemExit, "another split"):
                builder.add_still_studies(moved, path, "train", {"C1", "C2"})
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
