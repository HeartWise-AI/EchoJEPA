# tests/data/test_radar_doppler_captions.py

"""Doppler captions for structures fed by stills: the Doppler clauses `retain_caption` removes come back as written
(a dated clause still leaves), the full caption is the B-mode caption followed by them, Hemo's templated caption and
provisional flag, and the caption builder's `--doppler-captions` rows (behind the flag only; the table without it is
the v0 table)."""

import math
import os
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
V1 = load_routing(os.path.join(os.path.dirname(DEFAULT_ROUTING), "radar_routing_v1.yaml"))
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

    def test_full_rows(self):
        reports = pd.DataFrame([report("S1", V1, aortic=AORTIC, lv="VG non dilaté.", **HEMO),
                                report("S2", V1, aortic="Pas de sténose aortique.")])
        table, ledger = builder.build_tables(reports, V1, POLICY, doppler=True)
        rows = table.set_index(["study_id", "structure", "scope"])
        self.assertEqual(rows.loc[("S1", "aortic", "bmode"), "caption"], "Valve aortique calcifiée.")
        full = rows.loc[("S1", "aortic", "full")]
        self.assertIn("gradient moyen 28 mmHg", full.caption)
        self.assertEqual((full.normality, full.normality_status), ("unknown", "provisional"))
        self.assertNotIn(("S1", "lv", "full"), rows.index)     # LV is fed by no still
        self.assertEqual(rows.loc[("S1", "hemo", "full"), "normality"], "abnormal")
        self.assertEqual(rows.loc[("S2", "aortic", "full"), "caption"], "Pas de sténose aortique.")
        self.assertNotIn(("S2", "aortic", "bmode"), rows.index)
        self.assertEqual(ledger.set_index(["study_id", "structure"]).loc[("S2", "aortic"), "reason"],
                         "doppler_only")                       # the B-mode ledger is unchanged


if __name__ == "__main__":
    unittest.main()
