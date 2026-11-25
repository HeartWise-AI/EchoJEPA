"""Study-level coverage ledger: candidate clips, reviewed support, multi-view needs and explicit gaps."""

import json
import unittest

from app.radar.captions import rule_based_caption
from app.radar.coverage import CANDIDATE, GAP, NOT_SUPPORTED, SUPPORTED, build_ledger, summarize
from app.radar.records import ConceptReason
from tests.radar.synthetic import ONTOLOGY, finding, record, replace, reviewed, with_review

FINDINGS = [
    finding("lv_ejection_fraction", 35),
    finding("lv_regional_wall_motion", "hypokinetic"),
    finding("mitral_regurgitation", "moderate"),
    finding("rv_systolic_pressure", 42),
    finding("tricuspid_regurgitation_ero", 0.2),
    finding("ivc_dilation", True),
    finding("aortic_regurgitation", "mild"),
]


def clip(n, view, acq, study="study-a", findings=FINDINGS):
    rec = record(n, view=view, acq=acq, study=study, findings=findings)
    return replace(rec, candidate_caption=rule_based_caption(rec, ONTOLOGY))


def by_concept(study):
    return {f.concept_id: f for f in study.findings}


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.clips = [clip(1, "A4C", "bmode"), clip(2, "A4C", "color_flow"), clip(3, "A2C", "bmode"),
                      clip(4, "PLAX", "color_flow")]

    def test_candidates_gaps_and_views(self):
        (study,) = build_ledger(self.clips, ONTOLOGY)
        f = by_concept(study)
        ids = [c.record_id for c in self.clips]
        self.assertEqual(f["mitral_regurgitation"].candidate_clips, [ids[1], ids[3]])
        self.assertEqual(f["mitral_regurgitation"].status, CANDIDATE)
        self.assertEqual(f["aortic_regurgitation"].candidate_clips, [ids[3]])
        # RVSP needs spectral Doppler, which the video corpus never has: an explicit gap, never a caption.
        self.assertEqual((f["rv_systolic_pressure"].status, f["rv_systolic_pressure"].gap_reason),
                         (GAP, "modality_absent"))
        self.assertEqual(f["rv_systolic_pressure"].required_modality, ["spectral_doppler"])
        self.assertEqual(f["tricuspid_regurgitation_ero"].gap_reason, "measurement_required")
        self.assertEqual(f["ivc_dilation"].gap_reason, "no_relevant_view")
        # Multi-view findings: EF has its apical 4- and 2-chamber views; wall motion still lacks a 3-chamber view.
        self.assertEqual(f["lv_ejection_fraction"].view_groups_missing, [])
        self.assertEqual(f["lv_regional_wall_motion"].view_groups_missing, ["apical_3c"])
        for clip_record in self.clips:
            for claim in clip_record.candidate_caption.claims:
                self.assertNotIn(claim.concept_id, ("rv_systolic_pressure", "tricuspid_regurgitation_ero"))

    def test_review_outcomes(self):
        self.clips[1] = with_review(self.clips[1], reviewed())
        self.clips[3] = with_review(self.clips[3], reviewed(
            satisfied=False, unsupported_concepts=["aortic_regurgitation"],
            uncertain=[ConceptReason("mitral_regurgitation", "quality")]))
        self.clips[0] = with_review(self.clips[0], reviewed(missing_concepts=["lv_regional_wall_motion"]))
        f = by_concept(build_ledger(self.clips, ONTOLOGY)[0])
        self.assertEqual(f["mitral_regurgitation"].status, SUPPORTED)
        self.assertEqual(f["mitral_regurgitation"].supported_by, [self.clips[1].record_id])
        self.assertEqual(f["mitral_regurgitation"].uncertain_in, [self.clips[3].record_id])
        self.assertEqual(f["aortic_regurgitation"].status, NOT_SUPPORTED)
        self.assertEqual(f["lv_ejection_fraction"].status, SUPPORTED)  # claimed in the accepted A4C caption

    def test_inconsistent_values_across_clips_are_flagged(self):
        other = clip(5, "A4C", "color_flow", findings=[finding("mitral_regurgitation", "severe")])
        study = build_ledger(self.clips + [other], ONTOLOGY)[0]
        self.assertEqual(study.inconsistent_findings, ["mitral_regurgitation"])
        self.assertNotIn("mitral_regurgitation", by_concept(study))

    def test_one_split_per_study(self):
        with self.assertRaises(ValueError):
            build_ledger([self.clips[0], replace(self.clips[1], split="val")], ONTOLOGY)

    def test_summary_is_aggregate(self):
        ledger = build_ledger(self.clips + [clip(9, "SUBCOSTAL", "bmode", study="study-b")], ONTOLOGY)
        summary = summarize(ledger)
        self.assertEqual((summary["studies"], summary["clips"]), (2, 5))
        self.assertEqual(summary["by_status"][GAP] + summary["by_status"][CANDIDATE], summary["findings"])
        self.assertEqual(summary["multi_view"]["needs_several_views"], 4)
        text = json.dumps(summary)
        for key in ("study-a", "study-b", self.clips[0].record_id):
            self.assertNotIn(key, text)


if __name__ == "__main__":
    unittest.main()
