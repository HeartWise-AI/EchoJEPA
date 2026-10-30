# tests/data/test_make_coronary_manifest.py

"""`data/make_coronary_manifest.py`: the DeepCORO_CLIP report CSV becomes the probe's long
manifest, with per-vessel findings, negative segments as missing, and conflicting study values
set to missing."""

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
import make_coronary_manifest as mcm  # noqa: E402


def row(video, study, patient, split, structure="Left Coronary", **segments):
    values = {s: 0.0 for s in mcm.SEGMENTS}
    values.update(segments)
    return {"FileName": video, "StudyInstanceUID": study, "patient_id_anon": patient, "Split": split,
            "main_structure_name": structure, "dominance_name": "right_dominant", "other": "x", **values}


class TestCoronaryManifest(unittest.TestCase):

    def test_targets_and_consistency(self):
        rows = [
            row("/v/1.avi", "s1", "p1", "train", prox_rca_stenosis=80, left_main_stenosis=50),
            row("/v/2.avi", "s1", "p1", "train", structure="Right Coronary", prox_rca_stenosis=80, left_main_stenosis=50),
            row("/v/3.avi", "s2", "p2", "val", mid_lad_stenosis=40, lvp_stenosis=-1),
            row("/v/4.avi", "s3", "p3", "test", pda_stenosis=70),
            row("/v/5.avi", "s3", "p3", "test", pda_stenosis=20),      # disagrees with the row above.
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reports.csv")
            pd.DataFrame(rows).to_csv(path, index=False, sep="α")
            out = os.path.join(tmp, "out")
            mcm.main(["--reports", path, "--out-dir", out])
            m = pd.read_csv(os.path.join(out, "coronary_videos_with_targets.csv"))
            self.assertEqual(list(m.columns[:5]), ["video_path", "study_id", "patient_id", "split", "structure"])
            s1 = m[m.study_id == "s1"].iloc[0]
            self.assertEqual((s1.rca_ge70, s1.lad_ge70, s1.left_main_ge50, s1.any_ge70), (1, 0, 1, 1))
            s2 = m[m.study_id == "s2"].iloc[0]
            self.assertTrue(np.isnan(s2.lvp_stenosis))
            self.assertEqual((s2.lad_ge70, s2.any_ge70), (0, 0))
            self.assertEqual(s2.lcx_ge70, 0)  # the other LCx segments are 0, so the vessel is known.
            s3 = m[m.study_id == "s3"]
            # Conflicting rows: the segment, and the findings derived from it, are missing for the whole study.
            self.assertTrue(s3.pda_stenosis.isna().all() and s3.rca_ge70.isna().all() and s3.any_ge70.isna().all())
            self.assertTrue((s3.lad_ge70 == 0).all())       # the other vessels are still decided.
            with open(os.path.join(out, "coronary_targets_info.json")) as f:
                info = json.load(f)
            self.assertEqual(info["conflicts"], {"pda_stenosis": 1, "rca_ge70": 1, "any_ge70": 1})
            self.assertEqual(info["splits"]["train"]["studies"], 1)
            self.assertIsNone(info["splits"]["test"]["targets"]["any_ge70"]["prevalence"])
            self.assertEqual(info["splits"]["test"]["targets"]["lad_ge70"]["prevalence"], 0.0)

    def test_a_study_in_two_splits_is_refused(self):
        rows = [row("/v/1.avi", "s1", "p1", "train"), row("/v/2.avi", "s1", "p1", "val")]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reports.csv")
            pd.DataFrame(rows).to_csv(path, index=False, sep="α")
            with self.assertRaisesRegex(ValueError, "more than one split"):
                mcm.main(["--reports", path, "--out-dir", os.path.join(tmp, "out")])
