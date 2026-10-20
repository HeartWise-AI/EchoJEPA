# tests/data/test_make_study_targets.py

"""`data/make_study_targets.py`: the target rules on a few hand-made report rows, the PACS link,
the join onto videos.csv and the per-split report."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import make_study_targets as mst  # noqa: E402

REPORT_COLUMNS = [
    "Visually Estimated EF", "IVS Diastole Thickness (2D)", "LVIW Diastolic Thickness (2D)", "LVID Diastole (2D)",
    "LVID Systole (2D)", "RV Basal Diastolic Dimension", "LA Volume Index (BP MOD) DuBois", "LA Dimension",
    "AV Peak Velocity", "AV Mean Gradient", "AV Area (Cont Eq VTI)", "TR Peak Velocity",
    "[MeasurementsDoppler]_[LVOTPeakVelocity]", "Referring MR Grade", "Referring TR Grade", "Referring AR Grade",
    "Referring PR Grade", "MHI VD fonction systolique", "RV Systolic Pressure", "LVOT Diameter", "TAPSE",
    "TV Lateral Ann s' Velocity", "MV E/e' (Lateral)", "MV E/e' (Septal)", "Sinus of Valsalva Diameter",
    "IVC Diameter (Exp 2D)", "RA Pressure", "MV Area (Cont Eq VTI)", "MV Mean Gradient", "MV PHT",
    "Pericardium findings", "MHI VG Fonction diastolique",
]


def report(**values):
    row = {c: np.nan for c in REPORT_COLUMNS}
    row["MHI VD fonction systolique"] = -1
    row.update(values)
    return row


class TestRules(unittest.TestCase):

    def test_grades_and_classes(self):
        g = mst.grade(pd.Series(["0.5", "Trace", "2", "4.0", "sévère", "2022-01-04 00:00:00", "7", None]))
        self.assertEqual([x if not np.isnan(x) else None for x in g], [0.5, 0.5, 2.0, 4.0, 4.0, None, None, None])
        classes = mst.regurg_class(g)
        self.assertEqual(classes[:5].tolist(), [2, 2, 1, 1, 1])
        self.assertTrue(np.isnan(classes[5]))

    def test_targets_on_a_sick_and_a_normal_study(self):
        rows = pd.DataFrame([
            report(**{"Visually Estimated EF": 30, "IVS Diastole Thickness (2D)": 1.4, "LVIW Diastolic Thickness (2D)": 1.3,
                      "LVID Diastole (2D)": 6.5, "RV Basal Diastolic Dimension": 5.1,
                      "LA Volume Index (BP MOD) DuBois": 0.045, "AV Peak Velocity": 420, "AV Mean Gradient": 45,
                      "AV Area (Cont Eq VTI)": 0.8, "TR Peak Velocity": 320, "Referring MR Grade": "3",
                      "Referring TR Grade": "0.5", "Referring AR Grade": "Trace", "Referring PR Grade": "1",
                      "MHI VD fonction systolique": 2, "RV Systolic Pressure": 55, "RA Pressure": 15,
                      "MV Area (Cont Eq VTI)": 1.2, "MV PHT": 0.2, "Pericardium findings": "Épanchement péricardique modéré",
                      "MHI VG Fonction diastolique": "Grade 3", "[MeasurementsDoppler]_[LVOTPeakVelocity]": 250}),
            report(**{"Visually Estimated EF": 62, "IVS Diastole Thickness (2D)": 0.9, "LVIW Diastolic Thickness (2D)": 0.9,
                      "LVID Diastole (2D)": 4.6, "RV Basal Diastolic Dimension": 3.4, "LA Dimension": 3.8,
                      "AV Peak Velocity": 130, "Referring MR Grade": "0.5", "Referring TR Grade": "1",
                      "Referring AR Grade": "0", "Referring PR Grade": "0.5", "MHI VD fonction systolique": 0,
                      "RV Systolic Pressure": 25, "RA Pressure": 3, "Pericardium findings": "Pas d'épanchement péricardique",
                      "MHI VG Fonction diastolique": "Grade 0", "[MeasurementsDoppler]_[LVOTPeakVelocity]": 95}),
            report(),
        ])
        t = mst.derive(rows)
        sick, normal, empty = t.iloc[0], t.iloc[1], t.iloc[2]
        self.assertEqual(sick[["reduced_ef_40", "lvef_le_45", "lvh_any", "lvh_modsev", "rv_dysfunction", "rv_dilated",
                               "rap_elevated", "lvot_obstruction", "mv_stenosis", "as_present", "as_severe", "mr_modsev",
                               "tr_modsev", "ar_modsev", "pr_modsev", "pericardial_effusion", "shd_composite"]].tolist(),
                         [1] * 17 if False else [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
        self.assertEqual(sick[["lv_systolic_function", "lv_size", "rv_size", "la_size", "lv_diastolic_function",
                               "av_stenosis", "mr_grade", "tr_grade", "ar_grade"]].tolist(),
                         [mst.THREE_WAY["modsev"], mst.THREE_WAY["modsev"], mst.THREE_WAY["modsev"],
                          mst.THREE_WAY["modsev"], mst.DIASTOLIC["modsev"], mst.AV_STENOSIS["severe"],
                          mst.REGURG["modsev"], mst.REGURG["none_trace"], mst.REGURG["none_trace"]])
        self.assertAlmostEqual(sick["lavi"], 45.0)
        self.assertAlmostEqual(sick["av_peak_velocity"], 4.2)
        self.assertEqual(normal[["reduced_ef_40", "lvh_any", "rv_dysfunction", "rv_dilated", "rap_elevated",
                                 "lvot_obstruction", "as_present", "mr_modsev", "pericardial_effusion",
                                 "shd_composite"]].tolist(), [0] * 10)
        self.assertTrue(np.isnan(normal["mv_stenosis"]))  # no mitral measurement: undecidable.
        self.assertEqual(normal[["lv_systolic_function", "lv_size", "rv_size", "la_size", "lv_diastolic_function",
                                 "av_stenosis", "mr_grade", "tr_grade"]].tolist(),
                         [mst.THREE_WAY["normal"], mst.THREE_WAY["normal"], mst.THREE_WAY["normal"],
                          mst.THREE_WAY["normal"], mst.DIASTOLIC["normal"], mst.AV_STENOSIS["none"],
                          mst.REGURG["none_trace"], mst.REGURG["mild"]])
        self.assertTrue(empty[list(mst.TASKS)].isna().all())
        self.assertEqual(set(mst.TASKS), set(t.columns))


class TestEndToEnd(unittest.TestCase):

    def test_link_join_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = pd.DataFrame([
                {"AccessionNumber": "RA1", "Dossier": "123", **report(**{"Visually Estimated EF": 35})},
                {"AccessionNumber": "RA2", "Dossier": "0456", **report(**{"Visually Estimated EF": 60})},
                {"AccessionNumber": "RA3", "Dossier": "789", **report(**{"Visually Estimated EF": 50})},  # other patient
                {"AccessionNumber": "RA4", "Dossier": "111", **report(**{"Visually Estimated EF": 55})},  # not in PACS
            ])
            pacs = pd.DataFrame([
                {"AccessionNumber": "RA1", "StudyInstanceUID": "1.2.1", "PatientID": "123"},
                {"AccessionNumber": "RA2", "StudyInstanceUID": "1.2.2", "PatientID": "456"},
                {"AccessionNumber": "RA3", "StudyInstanceUID": "1.2.3", "PatientID": "999"},
            ])
            videos = pd.DataFrame([
                {"video_path": "/v/a.mp4", "study_id": "1.2.1", "patient_id": "123", "split": "train", "view": "A4C"},
                {"video_path": "/v/b.mp4", "study_id": "1.2.1", "patient_id": "123", "split": "train", "view": "PLAX"},
                {"video_path": "/v/c.mp4", "study_id": "'1.2.2'", "patient_id": "456", "split": "val", "view": "A4C"},
                {"video_path": "/v/d.mp4", "study_id": "1.2.9", "patient_id": "777", "split": "test", "view": "A2C"},
            ])
            r, p, v = (os.path.join(tmp, n) for n in ("reports.csv", "pacs.csv", "videos.csv"))
            reports.to_csv(r, index=False)
            pacs.to_csv(p, index=False)
            videos.to_csv(v, index=False)
            out = os.path.join(tmp, "out")
            mst.main(["--reports", r, "--pacs", p, "--videos", v, "--out-dir", out])
            studies = pd.read_csv(os.path.join(out, "study_targets.csv"), dtype={"study_id": str})
            self.assertEqual(sorted(studies.study_id), ["1.2.1", "1.2.2"])
            merged = pd.read_csv(os.path.join(out, "videos_with_targets.csv"))
            self.assertEqual(merged.lvef.tolist()[:3], [35.0, 35.0, 60.0])
            self.assertTrue(np.isnan(merged.lvef.iloc[3]))
            with open(os.path.join(out, "targets_info.json")) as f:
                info = json.load(f)
            self.assertEqual(info["splits"]["train"]["reduced_ef_40"]["prevalence"], 1.0)
            self.assertEqual(info["splits"]["val"]["lvef"]["n"], 1)
            self.assertEqual(info["splits"]["test"]["lvef"]["n"], 0)
