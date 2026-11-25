# tests/data/test_radar_pilot.py

"""radar_pilot.py on synthetic tables: report categories, clip rules, study criteria, the shortlist and its global
keys, the selection after the screen (flagged clips removed, the best-ranked qualifying study taken) and
the column review candidates."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import radar_pilot as rp
from build_manifests import REPO_ROOT

FAMILY_VIEWS = ("A4C", "A2C", "PLAX", "PSAX", "A3C", "A5C")


class TestReportCategories(unittest.TestCase):
    def test_a_grade_counts_only_in_the_part_naming_the_regurgitation(self):
        cases = {
            "Insuffisance mitrale modérée. // Feuillets fins": "significant",
            "Severe aortic regurgitation": "significant",
            "Pas d'insuffisance aortique, insuffisance mitrale modérée": "significant",
            "Insuffisance mitrale légère à modérée": "significant",
            "Insuffisance mitrale légère, sténose aortique sévère": "possible",
            "Sténose aortique sévère et insuffisance aortique légère": "possible",
            "Pas d'insuffisance mitrale modérée": "possible",
            "Mitral regurgitation is not moderate or severe": "possible",
            "Insuffisance mitrale non sévère": "possible",
            "L'insuffisance aortique n'est pas sévère": "possible",
            "Possible moderate mitral regurgitation": "possible",
            "Cannot exclude severe aortic regurgitation": "possible",
            "Insuffisance mitrale possiblement modérée": "possible",
            "Insuffisance aortique sévère ne peut être exclue": "possible",
            "Insuffisance mitrale modérée ?": "possible",
            "Insuffisance mitrale légère": "none",
            "Pas d'insuffisance mitrale significative": "none",
            None: "none",
            "": "none",
            # Other wordings and abbreviations of MR and AR, and other ways to grade.
            "Régurgitation mitrale modérée": "significant",
            "Moderate MR": "significant",
            "Fuite aortique modérée": "significant",
            "Aortic valve insufficiency is severe": "significant",
            "IM grade 3/4": "significant",
            "IA 2+": "significant",
            "IM II/IV": "significant",
            "Insuffisance aortique grade III": "significant",
            "IM gr. 3. Feuillets fins": "significant",
            "IM 2 sur 4": "significant",
            "IM I/IV": "none",
            "IM 1/4": "none",
            "Insuffisance mitrale significative": "significant",
            # Any other moderate or worse statement is possible, so that wording the rule does not recognize never
            # lets a study count as normal.
            "Sténose aortique sévère": "possible",
            "Insuffisance tricuspide modérée": "possible",
            "Insuffisance modérée": "possible",
            # A grade that may describe another finding of the part is not the regurgitation's.
            "Insuffisance mitrale légère sur valve sévèrement calcifiée": "possible",
            "Insuffisance mitrale minime, valve mitrale modérément épaissie": "possible",
            "Mild mitral regurgitation on a severely thickened valve": "possible",
            "Insuffisance mitrale sévère sur prolapsus": "possible",
            "IM sévère sur sténose mitrale serrée": "possible",
            "Insuffisance mitrale modérée sur valve calcifiée": "possible",
            "Insuffisance aortique modérée, aorte ascendante dilatée": "significant",  # another part.
        }
        for text, grade in cases.items():
            self.assertEqual(rp.regurgitation_grade(text), grade, text)
        # In the mitral or aortic valve's own field, a regurgitation word needs no valve name.
        self.assertEqual(rp.regurgitation_grade("Insuffisance modérée", own_valve=True), "significant")
        self.assertEqual(rp.regurgitation_grade("Fuite légère", own_valve=True), "none")

    def test_categories(self):
        s = pd.DataFrame({"ef": [60, 30, 60, np.nan, 60, 60, 60], "av_mean_gradient": [10, 10, 45, 10, 10, 10, 10],
                          "rvsp": [30] * 7,
                          "Mitral valve findings": ["", "", "", "Insuffisance mitrale sévère",
                                                    "Insuffisance mitrale légère, sténose mitrale sévère",
                                                    "IM II/IV", ""],
                          "Aortic valve findings": [""] * 7,
                          "Tricuspid valve findings": [""] * 6 + ["Insuffisance modérée"]})
        cat = rp.categorize(s)
        # The fifth study only possibly has significant regurgitation, the last a moderate regurgitation of another
        # valve: neither normal nor valve disease. "IM II/IV" is moderate MR.
        self.assertEqual(cat.normal.tolist(), [True, False, False, False, False, False, False])
        self.assertEqual(cat.lv_dysfunction.tolist(), [False, True, False, False, False, False, False])
        self.assertEqual(cat.valve_disease.tolist(), [False, False, True, True, False, True, False])
        self.assertEqual(list(cat), list(rp.CATEGORIES))


class TestClipRules(unittest.TestCase):
    def test_a_funnel_step_counts_each_clip_under_its_first_reason_and_every_reason_in_a_note(self):
        reasons = pd.Series(["other_view;color_flow", "color_flow", "", "other_view", "acquisition", None])
        self.assertEqual(rp.reason_counts(reasons), "other_view=2; acquisition=1; color_flow=1")
        self.assertEqual(rp.each_reason_counts(reasons),
                         "every reason counted: color_flow=2; other_view=2; acquisition=1")
        self.assertEqual(rp.each_reason_counts(pd.Series(["min_clips", "", "view_families"])), "")

    def test_rules(self):
        clips = pd.DataFrame({"view": ["A4C", "OTHER", None, "A4C", "A4C", "A4C", "OTHER", "A4C", "A4C"],
                              "acq_category": ["bmode", "bmode", "bmode", "unknown", "color_flow", "bmode", "unknown",
                                               "bmode", "bmode"],
                              "acq_confidence": ["header"] * 5 + ["unknown"] * 2 + ["header"] * 2,
                              "acq_vendor_validated": [True] * 9,
                              "n_frames": pd.array([30, 30, 30, 30, 30, 30, 30, 3, 4], dtype="Int64")})
        self.assertEqual(rp.review_form(clips).tolist(), ["column_review", "view_type_review", "view_type_review",
                                                          "column_review", "column_review", "column_review",
                                                          "view_type_review", "column_review", "column_review"])
        # V1 captions B-mode clips of classified views: OTHER and unclassified views, a mode not read, GE color flow
        # and a still of 3 frames are removed, every reason named; 4 frames pass.
        self.assertEqual(rp.clip_rule(clips).tolist(), ["", "other_view", "other_view", "acquisition", "color_flow",
                                                        "acquisition", "other_view;acquisition", "few_frames", ""])
        # The manifest's frame counts are nullable: a clip without one is left to the screen.
        clips.loc[0, "n_frames"] = pd.NA
        self.assertEqual(rp.clip_rule(clips.iloc[[0]]).tolist(), [""])

    def test_another_vendors_mode_is_left_to_the_pixels(self):
        # A Philips header declares tissue imaging over color loops too: B-mode or color flow by its regions passes,
        # the mode then read from the pixels; a header declaring spectral Doppler or no region fails as before.
        clips = pd.DataFrame({"view": ["A4C", "A4C", "A4C", "A4C", "OTHER", "A4C"],
                              "acq_category": ["bmode", "color_flow", "spectral_doppler", "unknown", "unknown", "bmode"],
                              "acq_confidence": ["header"] * 5 + ["unknown"],
                              "acq_vendor_validated": [False] * 5 + [True], "n_frames": [30] * 6})
        self.assertEqual(rp.clip_rule(clips).tolist(), ["", "", "acquisition", "acquisition", "other_view;acquisition",
                                                        "acquisition"])
        self.assertEqual(rp.mode_source(clips).tolist(), ["pixels"] * 5 + ["header"])

    def test_select_leaves_out_color_read_from_the_pixels_and_3d_or_multipanel_displays(self):
        vingmed, gems, philips = "GE Vingmed Ultrasound", "GEMS Ultrasound", "Philips Medical Systems"
        clips = pd.DataFrame({
            "vendor": [vingmed] * 5 + [gems, philips, philips],
            "acq_category": ["bmode"] * 6 + ["color_flow", None],
            "dicom_image_content": ["0001", "0041", "0051", "0021", "0041", "0041", None, None],
            "decision": ["keep", "uncertain", "uncertain", "uncertain", "exclude", "uncertain", "keep", "exclude"],
            "decision_reasons": ["", "unusual_image_content", "unusual_image_content;measurement_text_unavailable",
                                 "unusual_image_content", "unusual_image_content;measurement_header",
                                 "unusual_image_content", "", "unreadable"]})
        out = rp.with_pilot_rule(clips)
        # The labelled codes of GE Vingmed 3D and multi-panel displays are excluded under that name, in place of the
        # screen's unusual_image_content; a code never labelled, or another vendor's, stays uncertain, not called 3D.
        self.assertEqual(list(zip(out.decision, out.decision_reasons)),
                         [("keep", ""), ("exclude", "3d_or_multipanel"),
                          ("exclude", "measurement_text_unavailable;3d_or_multipanel"),
                          ("uncertain", "unusual_image_content"), ("exclude", "measurement_header;3d_or_multipanel"),
                          ("uncertain", "unusual_image_content"), ("exclude", "color_flow"),
                          ("exclude", "unreadable;acquisition")])
        self.assertEqual(out.exclude.tolist(), [False] + [True] * 7)

    def test_vendor_slots_are_named_once_each_with_a_number_within_the_vendors_drawn_from(self):
        # A slot of any vendor, the categories' own among them: a study of each vendor is screened.
        self.assertEqual(rp.vendor_slots(["Philips Medical Systems=1", "GEMS Ultrasound=2", "GE Vingmed Ultrasound=1"],
                                         None),
                         {"Philips Medical Systems": 1, "GEMS Ultrasound": 2, "GE Vingmed Ultrasound": 1})
        for specs in (["Philips Medical Systems"], ["Philips Medical Systems=0"], ["GEMS Ultrasound=1"] * 2):
            with self.assertRaises(SystemExit):
                rp.vendor_slots(specs, None)
        # --vendor GE is GE's three vendor names: their slots stand, another vendor's is refused.
        ge = rp.VENDOR_GROUPS["GE"]
        self.assertEqual(set(ge), {"GE Vingmed Ultrasound", "GE Healthcare Ultrasound", "GEMS Ultrasound"})
        self.assertEqual(rp.vendor_slots(["GE Healthcare Ultrasound=1", "GEMS Ultrasound=1"], ge),
                         {"GE Healthcare Ultrasound": 1, "GEMS Ultrasound": 1})
        with self.assertRaises(SystemExit) as stop:
            rp.vendor_slots(["GEMS Ultrasound=1", "Philips Medical Systems=1"], ge)
        self.assertIn("'Philips Medical Systems', outside --vendor's vendors", str(stop.exception))
        # --vendor 'GE Vingmed Ultrasound' is that vendor alone: another GE name's slot is refused too.
        vingmed = rp.VENDOR_GROUPS["GE Vingmed Ultrasound"]
        self.assertEqual(rp.vendor_slots(["GE Vingmed Ultrasound=1"], vingmed), {"GE Vingmed Ultrasound": 1})
        with self.assertRaises(SystemExit) as stop:
            rp.vendor_slots(["GE Healthcare Ultrasound=1"], vingmed)
        self.assertIn("'GE Healthcare Ultrasound', outside --vendor's vendors", str(stop.exception))


def write_flags(flags, path, out, calibration=None, decisions=True):
    """Flags as completed clip_screen.py features and flag runs leave them for the shortlist in `out`: the features
    of its clip and reference lists, each with its record naming the list's SHA-256, then the flags with theirs
    naming those files."""
    sources = {}
    for role, listed in (("features", "shortlist_clips.csv"), ("reference", "reference_clips.csv")):
        table = os.path.join(os.path.dirname(path), f"{role}.parquet")
        pd.DataFrame({"video_path": ["/synthetic/x.mp4"]}).to_parquet(table)
        rp.clip_screen._write_info(table, {"clips": rp.sha256(os.path.join(out, listed))})
        sources[role] = rp.clip_screen._source(table)
    media = {"error": None, "video_sha256": "synthetic", "dicom_status": "ok", "dicom_regions": "[]", "height": 1,
             "width": 1, "dicom_image_content": "0001", "dicom_measure_flag": False, "dicom_source_image": True,
             "dicom_ecg": True, "dicom_sop_class": "1.2.840.10008.5.1.4.1.1.3.1", "pixel_mode": "bmode"}
    if decisions and "decision" not in flags:  # the screen's decision, from the exclusion given (no uncertain clip).
        flags = flags.assign(decision=np.where(flags.exclude, "exclude", "keep"),
                             decision_reasons=flags.exclude_reason)
    flags.assign(**{k: v for k, v in media.items() if k not in flags}).to_parquet(path)
    rp.clip_screen._write_info(path, {**sources, "thresholds": rp.clip_screen.THRESHOLDS,
                                      "calibration": calibration})


def left_out_total(text):
    """The total of a funnel step's `left_out` ('reason=count; ...')."""
    return sum(int(part.rsplit("=", 1)[1]) for part in text.split("; ")) if text else 0


def study_clips(study, patient, n_per_view=4, views=FAMILY_VIEWS):
    rows = []
    for view in views:
        for i in range(n_per_view):
            rows.append({"patient_id": patient, "study_id": study, "view": view,
                         "video_path": f"/synthetic/{study}/{view}_{i}.mp4",
                         "dicom_path": f"/synthetic/{study}/{view}_{i}.dcm", "vendor": "GE Vingmed Ultrasound",
                         "model": "m", "acq_category": "bmode" if i % 2 else "color_flow", "n_frames": 30,
                         "split": "train", "study_date": "20230101",
                         "acq_confidence": "header", "acq_vendor_validated": True})
    return rows


class TestChooseStudies(unittest.TestCase):
    """Per category in priority order, the best-ranked study that still meets the criteria once its flagged clips
    are removed, one study per patient."""

    def choose(self, ranked, flagged_psax=()):
        """ranked: {category: [(study, patient), ...]} in rank order; the screen flags the PSAX clips of
        `flagged_psax`."""
        short, rows = [], []
        for category, studies in ranked.items():
            for rank, (study, patient) in enumerate(studies):
                short.append({"category": category, "rank": str(rank), "study_id": study, "patient_id": patient,
                              "accession": f"A-{study}"})
                rows += study_clips(study, patient)
        # The clips as select has them: B-mode ones of classified views, those of another mode left out before.
        clips = pd.DataFrame(rows).drop_duplicates("video_path").query("acq_category == 'bmode'").assign(exclude=False)
        clips.loc[clips.study_id.isin(flagged_psax) & clips.view.eq("PSAX"), "exclude"] = True
        chosen = rp.choose_studies(pd.DataFrame(short), clips, 12, rp.REQUIRED_VIEW_FAMILIES, 3)
        return dict(zip(chosen.category, chosen.study_id))

    def test_the_best_ranked_study_is_taken(self):
        ranked = {"normal": [("n1", "p1"), ("n2", "p2")], "lv_dysfunction": [("l1", "p3")],
                  "valve_disease": [("v1", "p4")]}
        self.assertEqual(self.choose(ranked), {"normal": "n1", "lv_dysfunction": "l1", "valve_disease": "v1"})

    def test_a_study_short_of_a_view_after_the_screen_is_passed_over(self):
        ranked = {"normal": [("n1", "p1"), ("n2", "p2")], "lv_dysfunction": [("l1", "p3")],
                  "valve_disease": [("v1", "p4")]}
        self.assertEqual(self.choose(ranked, flagged_psax=("n1",))["normal"], "n2")

    def test_one_study_per_patient(self):
        ranked = {"normal": [("n1", "p1")], "lv_dysfunction": [("l1", "p1"), ("l2", "p2")],
                  "valve_disease": [("v1", "p3")]}
        self.assertEqual(self.choose(ranked), {"normal": "n1", "lv_dysfunction": "l2", "valve_disease": "v1"})

    def test_the_combinations_are_searched_not_each_category_in_turn(self):
        # The best LV study's patient is the only valve patient: taking it first would leave no valve study, while
        # the second LV study leaves three patients.
        ranked = {"normal": [("n1", "p1")], "lv_dysfunction": [("l1", "p2"), ("l2", "p3")],
                  "valve_disease": [("v1", "p2")]}
        self.assertEqual(self.choose(ranked), {"normal": "n1", "lv_dysfunction": "l2", "valve_disease": "v1"})
        # When the greedy choice works, it is the one taken.
        ranked["valve_disease"].insert(0, ("v0", "p4"))
        self.assertEqual(self.choose(ranked), {"normal": "n1", "lv_dysfunction": "l1", "valve_disease": "v0"})
        # No combination of different patients: no pilot.
        self.assertEqual(self.choose({"normal": [("n1", "p1")], "lv_dysfunction": [("l1", "p1")],
                                      "valve_disease": [("v1", "p2")]}), {})


class TestReportLink(unittest.TestCase):
    STUDIES = pd.DataFrame({"study_id": [f"s{i}" for i in range(1, 8)],
                            "patient_id": [f"0010{i}" for i in range(1, 8)], "study_date": ["20230101"] * 7})

    @staticmethod
    def pacs(*rows):
        return pd.DataFrame(rows, columns=["accession", "study_id", "pacs_patient", "pacs_date"])

    def test_a_report_is_linked_through_the_pacs_index_when_patients_and_dates_agree(self):
        day = "20230101"
        pacs = self.pacs(("A1", "s1", "101", day),
                         ("A2", "s2", "102", day),  # the report names another patient
                         ("A3", "s3", "103", "20220101"),  # PACS and manifest dates differ
                         ("A5", "s5", "105", day),  # no report
                         ("A6", "s6", "106", day), ("A7", "s6", "106", day),  # two reports
                         ("A8", "s7", "107", day), ("A8", "s1", "101", day))  # an accession of two studies
        reports = pd.DataFrame({"AccessionNumber": ["A1", "A2", "A3", "A6", "A7", "A8"],
                                "Dossier": [101, 999, 103, 106, 106, 107],
                                "Mitral valve findings": ["Valve mitrale normale"] * 6})
        linked, reason = rp.link_reports(reports, pacs, self.STUDIES)
        # The report's patient number, the PACS id and the manifest's agree without their zero padding; s4 has no
        # PACS accession, and s7's names two studies.
        self.assertEqual(linked[["study_id", "accession"]].values.tolist(), [["s1", "A1"]])
        self.assertEqual(sorted(linked.columns), ["Mitral valve findings", "accession", "study_id"])
        self.assertEqual(reason.to_dict(), {"s1": "", "s2": "patient_or_date_disagree",
                                            "s3": "patient_or_date_disagree", "s4": "no_pacs_accession",
                                            "s5": "no_report", "s6": "several_reports", "s7": "no_pacs_accession"})

    def test_rows_repeated_exactly_are_one_report_and_rows_that_differ_leave_the_study_out(self):
        pacs = self.pacs(("A1", "s1", "101", "20230101"), ("A2", "s2", "102", "20230101"))
        reports = pd.DataFrame({"AccessionNumber": ["A1", "A1", "A2", "A2"], "Dossier": [101, 101, 102, 102],
                                "Mitral valve findings": ["Valve mitrale normale"] * 3
                                + ["Insuffisance mitrale sévère"]})
        linked, reason = rp.link_reports(reports, pacs, self.STUDIES.iloc[:2])
        self.assertEqual(linked.study_id.tolist(), ["s1"])
        self.assertEqual(reason.to_dict(), {"s1": "", "s2": "report_rows_differ"})

    def test_a_second_conflicting_report_cannot_be_hidden_by_a_valid_accession(self):
        pacs = self.pacs(("A1", "s1", "101", "20230101"), ("A2", "s1", "101", "20230101"),
                         ("A3", "s2", "102", "20230101"))
        reports = pd.DataFrame({"AccessionNumber": ["A1", "A2", "A2", "A3"],
                                "Dossier": [101, 101, 101, 102],
                                "Mitral valve findings": ["Normal", "Moderate MR", "Severe MR", "Normal"]})
        for seed in (0, 1):
            with self.subTest(seed=seed):
                linked, reason = rp.link_reports(reports.sample(frac=1, random_state=seed), pacs,
                                                 self.STUDIES.iloc[:2])
                self.assertEqual(linked.study_id.tolist(), ["s2"])
                self.assertEqual(reason.to_dict(), {"s1": "report_rows_differ", "s2": ""})

    def test_multiple_reports_are_ambiguous_before_identity_filtering(self):
        pacs = self.pacs(("A1", "s1", "101", "20230101"), ("A2", "s1", "101", "20230101"))
        reports = pd.DataFrame({"AccessionNumber": ["A1", "A2"], "Dossier": [101, 999],
                                "Mitral valve findings": ["Normal", "Severe MR"]})
        linked, reason = rp.link_reports(reports, pacs, self.STUDIES.iloc[:1])
        self.assertTrue(linked.empty)
        self.assertEqual(reason.to_dict(), {"s1": "several_reports"})


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # The synthetic clips have no media files: the media check is tested on its own below.
        self.media = mock.patch.object(rp.clip_screen, "changed_media", return_value=[])
        self.media.start()
        self.addCleanup(self.media.stop)
        self.out = os.path.join(self.tmp, "pilot")
        rows = []
        # normal: s1, s2; lv_dysfunction: s3; valve_disease: s4. s5 is a validation patient; s6 lacks PSAX.
        for study, patient in (("s1", "p1"), ("s2", "p2"), ("s3", "p3"), ("s4", "p4")):
            rows += study_clips(study, patient)
        rows += [dict(r, split="val") for r in study_clips("s5", "p5")]
        rows += study_clips("s6", "p6", views=("A4C", "A2C", "PLAX", "A3C", "A5C"))
        for study, patient in (("s1", "p1"), ("s3", "p3")):
            rows += [dict(r, view="OTHER", video_path=r["video_path"].replace(".mp4", "_o.mp4"))
                     for r in study_clips(study, patient, n_per_view=1)]
        self.videos = pd.DataFrame(rows)
        self.videos.to_parquet(os.path.join(self.tmp, "videos.parquet"))
        pd.DataFrame({"study_id": [f"s{i}" for i in range(1, 7)], "accession": [f"A{i}" for i in range(1, 7)],
                      "ef": [60, 62, 30, 58, 60, 60]}).to_parquet(os.path.join(self.tmp, "ef.parquet"))
        pd.DataFrame({"study_id": [f"s{i}" for i in range(1, 7)], "accession": [f"A{i}" for i in range(1, 7)],
                      "rvsp": [30] * 6}).to_parquet(os.path.join(self.tmp, "rvsp.parquet"))
        pd.DataFrame({"AccessionNumber": [f"A{i}" for i in range(1, 7)],
                      "StudyInstanceUID": [f"s{i}" for i in range(1, 7)],
                      "PatientID": [f"p{i}" for i in range(1, 7)], "StudyDate": ["20230101"] * 6}).to_csv(
            os.path.join(self.tmp, "pacs.csv"), index=False)
        pd.DataFrame({"AccessionNumber": [f"A{i}" for i in range(1, 7)], "Dossier": [f"p{i}" for i in range(1, 7)],
                      "Mitral valve findings": (["Valve mitrale normale"] * 3 + ["Insuffisance mitrale sévère"]
                                                + [""] * 2),
                      "Aortic valve findings": [""] * 6, "Tricuspid valve findings": ["Valve tricuspide normale"] * 6,
                      "AV Mean Gradient": [5.0] * 6,
                      "Left ventricule findings": ["Ventricule gauche non-dilaté"] * 6}).to_parquet(
            os.path.join(self.tmp, "reports.parquet"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_cmd(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rp.main(list(argv))

    def shortlist(self, *options):
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--pacs", os.path.join(self.tmp, "pacs.csv"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "10", *options)

    def test_manifest_identity_conflicts_stop_before_any_clip_or_patient_filter(self):
        listed = os.path.join(self.tmp, "excluded.csv")
        pd.DataFrame({"patient_id": ["p999"]}).to_csv(listed, index=False)
        for column, value in (("patient_id", "p999"), ("study_date", "20150101")):
            for suffix in ("A4C_0.mp4", "A4C_1.mp4"):
                with self.subTest(column=column, suffix=suffix):
                    self.videos.to_parquet(os.path.join(self.tmp, "videos.parquet"))
                    self.shortlist()  # A failed rerun must also clear the previous completed shortlist.
                    rows = self.videos.copy()
                    rows.loc[(rows.study_id == "s3") & rows.video_path.str.endswith(suffix), column] = value
                    rows.to_parquet(os.path.join(self.tmp, "videos.parquet"))
                    with self.assertRaises(SystemExit) as stop:
                        self.shortlist("--exclude-patients", listed)
                    self.assertIn("conflicting patient IDs or study dates", str(stop.exception))
                    self.assertNotIn("p999", str(stop.exception))
                    self.assertNotIn("s3", str(stop.exception))
                    self.assertFalse(set(os.listdir(self.out)) & set(rp.SHORTLIST_OUTPUTS))

    def test_clean_backup_clips_are_recorded_as_not_chosen_and_reconcile_with_the_pilot(self):
        _, flags_path = self.flagged_shortlist()
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        removed = pd.read_csv(os.path.join(self.out, "removed_clips.csv"))
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))
        omitted = pd.read_csv(os.path.join(self.out, "not_kept_clips.csv"))
        backups = set(clips.video_path) - set(pilot.video_path)
        self.assertEqual(len(backups), 12)
        backup_rows = omitted[omitted.video_path.isin(backups)]
        self.assertEqual(set(backup_rows.video_path), backups)
        self.assertEqual(set(zip(backup_rows.step, backup_rows.decision, backup_rows.reasons)),
                         {("selection", "not_chosen", "study_not_chosen")})
        self.assertFalse(backup_rows.in_pilot.any())
        self.assertFalse(omitted.video_path.duplicated().any())
        self.assertFalse(set(pilot.video_path) & set(omitted.video_path))
        self.assertEqual(set(clips.video_path) | set(removed.video_path),
                         set(pilot.video_path) | set(omitted.video_path))
        funnel = pd.read_csv(os.path.join(self.out, "funnel.csv"), keep_default_na=False)
        self.assertEqual(funnel.iloc[-2].videos - funnel.iloc[-1].videos, len(backup_rows))

    def test_artifacts_without_the_identity_and_report_checks_cannot_be_reused(self):
        _, flags_path = self.flagged_shortlist()
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        path = os.path.join(self.out, "selection.json")
        with open(path) as f:
            selection = json.load(f)
        selection.pop("link_policy")
        with open(path, "w") as f:
            json.dump(selection, f)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("candidates", "--out-dir", self.out,
                         "--reports", os.path.join(self.tmp, "reports.parquet"), "--provisional-screen")
        self.assertIn("predates the study identity and report ambiguity checks", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))
        path = os.path.join(self.out, "shortlist.json")
        with open(path) as f:
            record = json.load(f)
        record["criteria"].pop("link_policy")
        with open(path, "w") as f:
            json.dump(record, f)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertIn("predates the study identity and report ambiguity checks", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))

    def test_shortlist_keeps_training_studies_and_removes_unreviewable_clips(self):
        self.shortlist()
        short = pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str)
        self.assertEqual(sorted(short.study_id), ["s1", "s2", "s3", "s4"])
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        # V1: the B-mode clips of classified views alone, all for the column review; the color clips and the OTHER
        # ones (color too, in this fixture) are removed and listed.
        self.assertEqual((len(clips), set(clips.form), set(clips.acq_category)),
                         (4 * len(FAMILY_VIEWS) * 2, {"column_review"}, {"bmode"}))
        removed = pd.read_csv(os.path.join(self.out, "removed_clips.csv"))
        self.assertEqual(removed.reasons.value_counts().to_dict(),
                         {"color_flow": 4 * len(FAMILY_VIEWS) * 2, "other_view;color_flow": 12})
        # Every clip has its opaque key before the screen, so sheets and Labelbox name it the same way.
        self.assertFalse(clips.global_key.duplicated().any())
        self.assertTrue(clips.global_key.str.fullmatch(r"radar-pilot-[0-9a-f]{24}").all())
        self.assertTrue(clips.dicom_path.str.endswith(".dcm").all())
        self.assertEqual({os.stat(os.path.join(self.out, n)).st_mode & 0o777 for n in rp.SHORTLIST_OUTPUTS}, {0o600})
        # select uses only the outputs of a completed shortlist, unchanged since.
        clips.iloc[:-1].to_csv(os.path.join(self.out, "shortlist_clips.csv"), index=False)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", os.path.join(self.tmp, "flags.parquet"))
        self.assertIn("shortlist_clips.csv changed", str(stop.exception))

    def test_a_report_needs_no_ef_label_and_an_ef_counts_only_for_its_own_report(self):
        path = os.path.join(self.tmp, "ef.parquet")
        ef = pd.read_parquet(path)
        # s2 (normal) and s4 (valve disease) have no EF label; s1's is that of another report.
        ef = ef[~ef.study_id.isin(["s2", "s4"])]
        ef.assign(accession=ef.accession.where(ef.study_id != "s1", "A9")).to_parquet(path)
        self.shortlist()
        short = pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str)
        # Severe MR makes s4 valve disease without an EF; a normal study needs one.
        self.assertEqual(dict(zip(short.study_id, short.category)), {"s3": "lv_dysfunction", "s4": "valve_disease"})
        with open(os.path.join(self.out, "shortlist.json")) as f:
            record = json.load(f)
        # Every training study (s5 is in the validation split) has its one report.
        self.assertEqual(record["report_link"], {"linked": 5, "no_pacs_accession": 0, "report_rows_differ": 0,
                                                 "no_report": 0, "patient_or_date_disagree": 0, "several_reports": 0})
        self.assertEqual(record["inputs"]["pacs"], rp.sha256(os.path.join(self.tmp, "pacs.csv")))

    def test_a_report_row_repeated_exactly_is_one_report_up_to_the_candidates(self):
        path = os.path.join(self.tmp, "reports.parquet")
        reports = pd.read_parquet(path)
        pd.concat([reports, reports], ignore_index=True).to_parquet(path)
        _, flags_path = self.flagged_shortlist()
        self.assertEqual(sorted(pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str).study_id),
                         ["s1", "s2", "s3", "s4"])
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.run_cmd("candidates", "--out-dir", self.out, "--reports", path, "--provisional-screen")
        with open(os.path.join(self.out, "candidates_summary.json")) as f:
            self.assertEqual(json.load(f)["candidates"], len(pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))))

    def test_the_funnel_accounts_for_every_step_from_the_manifest_to_the_pilot(self):
        _, flags_path = self.flagged_shortlist()
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        funnel = pd.read_csv(os.path.join(self.out, "funnel.csv"), keep_default_na=False)
        self.assertEqual(funnel.stage.tolist(), ["shortlist"] * 11 + ["select"] * 3)
        self.assertEqual(funnel.iloc[0][["videos", "studies", "patients"]].tolist(), [len(self.videos), 6, 6])
        # A step leaves what the one before it left, less what it left out, counted in clips or studies; a step
        # that leaves nothing out leaves the same videos.
        for before, row in zip(funnel.iloc[:-1].itertuples(), funnel.iloc[1:].itertuples()):
            unit = "studies" if row.left_out_unit == "studies" else "videos"
            self.assertEqual(getattr(before, unit) - getattr(row, unit), left_out_total(row.left_out), row.step)
        # s5's clips are in the validation split; s6, with no PSAX, misses a view family.
        self.assertEqual((funnel.left_out[1], funnel.note[1], funnel.left_out[7]),
                         ("non_train=24", "splits: val=24", "view_families=1"))
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))
        self.assertEqual(funnel.iloc[-1][["videos", "studies", "left_out"]].tolist(), [len(pilot), 3, "not_chosen=1"])
        with open(os.path.join(self.out, "shortlist.json")) as f:
            self.assertEqual(len(json.load(f)["funnel"]), 11)
        with open(os.path.join(self.out, "summary.md")) as f:
            self.assertIn("| training split | 128 |", f.read())

    def test_the_view_families_a_study_needs_are_an_option_select_keeps(self):
        # s6, a normal study, has no PSAX: eligible once PSAX is not asked for.
        self.shortlist("--view-families", "A4C", "A2C", "PLAX")
        self.assertIn("s6", set(pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str).study_id))
        with open(os.path.join(self.out, "shortlist.json")) as f:
            self.assertEqual(json.load(f)["criteria"]["required_view_families"], ["A4C", "A2C", "PLAX"])
        # select asks for the shortlist's families: with every PSAX clip flagged, each study still qualifies.
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        psax = clips.view.eq("PSAX")
        write_flags(clips[["video_path"]].assign(exclude=psax, exclude_reason=np.where(psax, "measurement", "")),
                    flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        studies = pd.read_csv(os.path.join(self.out, "pilot_studies.csv"), dtype=str)
        self.assertEqual(studies.category.tolist(), ["normal", "lv_dysfunction", "valve_disease"])
        # Given without a family, none is needed; a family no classified view has is refused.
        self.shortlist("--view-families")
        with open(os.path.join(self.out, "shortlist.json")) as f:
            self.assertEqual(json.load(f)["criteria"]["required_view_families"], [])
        with self.assertRaises(SystemExit) as stop:
            self.shortlist("--view-families", "A4C", "PSAXMID")
        self.assertIn("['PSAXMID'] name no classified view", str(stop.exception))

    def test_the_pilot_folder_must_be_private(self):
        os.makedirs(self.out)
        os.chmod(self.out, 0o755)
        with self.assertRaises(SystemExit):
            self.shortlist()
        self.assertEqual(os.listdir(self.out), [])

    def test_a_manifest_without_the_acquisition_columns_is_refused(self):
        # This branch's build_manifests.py drops them: the message names the producer that keeps them.
        videos = os.path.join(self.tmp, "videos.parquet")
        pd.read_parquet(videos).drop(columns=["dicom_path", "acq_category"]).to_parquet(videos)
        with self.assertRaises(SystemExit) as stop:
            self.shortlist()
        self.assertIn("['acq_category', 'dicom_path']", str(stop.exception))
        self.assertIn("commit 57204d1", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SHORTLIST_OUTPUTS))

    def test_reference_clips_come_from_other_studies_of_the_same_layouts(self):
        self.shortlist()
        reference = pd.read_csv(os.path.join(self.out, "reference_clips.csv"))
        # s6 (train, no PSAX) is the only other training study, its 10 B-mode clips the only eligible ones; s5 is a
        # validation patient.
        self.assertEqual((set(reference.study_id), set(reference.acq_category)), ({"s6"}, {"bmode"}))
        self.assertEqual(len(reference), 10)
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--pacs", os.path.join(self.tmp, "pacs.csv"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "10", "--reference-per-layout", "3")
        reference = pd.read_csv(os.path.join(self.out, "reference_clips.csv"))
        self.assertEqual(reference.groupby(["vendor", "model", "acq_category"]).size().tolist(), [3])
        self.assertTrue(reference.global_key.str.startswith("radar-pilot-").all())
        with open(os.path.join(self.out, "shortlist.json")) as f:
            self.assertEqual(json.load(f)["clips_removed_by_metadata_rule"], {"color_flow": 60, "other_view": 12})

    def add_philips(self, study="s7", patient="p7", date="20130101", ef=45):
        """A Philips study, by default of 2013 (outside --years) with a linked report, its EF in no category; its
        headers read every clip as B-mode, as the archive's do."""
        rows = [dict(r, vendor="Philips Medical Systems", model="iE33", acq_category="bmode",
                     acq_vendor_validated=False, study_date=date) for r in study_clips(study, patient)]
        pd.concat([self.videos, pd.DataFrame(rows)]).to_parquet(os.path.join(self.tmp, "videos.parquet"))
        for name, row in (("ef.parquet", {"study_id": study, "accession": "A7", "ef": ef}),
                          ("rvsp.parquet", {"study_id": study, "accession": "A7", "rvsp": 30}),
                          ("reports.parquet", {"AccessionNumber": "A7", "Dossier": patient,
                                               "Mitral valve findings": "Valve mitrale normale",
                                               "Aortic valve findings": "", "Tricuspid valve findings": "",
                                               "AV Mean Gradient": 5.0, "Left ventricule findings": ""})):
            path = os.path.join(self.tmp, name)
            pd.concat([pd.read_parquet(path), pd.DataFrame([row])]).to_parquet(path)
        path = os.path.join(self.tmp, "pacs.csv")
        pd.concat([pd.read_csv(path, dtype=str), pd.DataFrame([{"AccessionNumber": "A7", "StudyInstanceUID": study,
                                                                "PatientID": patient, "StudyDate": date}])]
                  ).to_csv(path, index=False)

    def test_the_categories_draw_from_any_vendor_unless_vendor_ge_keeps_them_to_ge(self):
        # s7, a Philips study of 2023 with an EF of 30, is LV dysfunction as s3 is.
        self.add_philips(date="20230101", ef=30)
        self.shortlist()
        short = pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str)
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"), dtype=str)
        self.assertEqual(sorted(short.study_id[short.category == "lv_dysfunction"]), ["s3", "s7"])
        # Its clips are screened with their mode read from the pixels, the GE ones from the header.
        self.assertEqual(clips.groupby("vendor").mode_source.agg(set).to_dict(),
                         {"GE Vingmed Ultrasound": {"header"}, "Philips Medical Systems": {"pixels"}})
        with open(os.path.join(self.out, "shortlist.json")) as f:
            record = json.load(f)
        self.assertEqual((record["criteria"]["vendor"], record["criteria"]["vendors"]), ("any", "any"))
        self.assertEqual(record["funnel"][4]["step"], "vendor: any (no --vendor)")
        # --vendor GE leaves it out, counted in the funnel; a Philips slot is refused there.
        self.shortlist("--vendor", "GE")
        short = pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str)
        self.assertEqual(short.study_id[short.category == "lv_dysfunction"].tolist(), ["s3"])
        with open(os.path.join(self.out, "shortlist.json")) as f:
            record = json.load(f)
        self.assertEqual(record["criteria"]["vendors"], list(rp.VENDOR_GROUPS["GE"]))
        self.assertEqual(record["funnel"][4]["left_out"], f"other_vendor={len(FAMILY_VIEWS) * 4}")
        with self.assertRaises(SystemExit) as stop:
            self.shortlist("--vendor", "GE", "--vendor-studies", "Philips Medical Systems=1")
        self.assertIn("outside --vendor's vendors", str(stop.exception))
        # GE Vingmed alone: the funnel names the vendor once.
        self.shortlist("--vendor", "GE Vingmed Ultrasound")
        with open(os.path.join(self.out, "shortlist.json")) as f:
            record = json.load(f)
        self.assertEqual((record["criteria"]["vendors"], record["funnel"][4]["step"]),
                         (["GE Vingmed Ultrasound"], "--vendor GE Vingmed Ultrasound"))

    def test_a_vendor_slot_adds_a_study_of_another_vendor_and_leaves_the_categories_alone(self):
        self.shortlist()
        before = {n: pd.read_csv(os.path.join(self.out, n), dtype=str) for n in rp.SHORTLIST_OUTPUTS[:3]}
        self.add_philips()
        self.shortlist()  # the Philips study is in no category and outside --years: nothing changes without a slot.
        for name, table in before.items():
            pd.testing.assert_frame_equal(pd.read_csv(os.path.join(self.out, name), dtype=str), table)
        slot = ("--vendor-studies", "Philips Medical Systems=1")
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--pacs", os.path.join(self.tmp, "pacs.csv"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "10", *slot)
        short = pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str)
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"), dtype=str)
        self.assertEqual(short[short.category == "vendor:Philips Medical Systems"].study_id.tolist(), ["s7"])
        # The category studies, their clips and their reference clips are those of the shortlist without the slot.
        pd.testing.assert_frame_equal(short.iloc[:len(before["shortlist.csv"])], before["shortlist.csv"])
        pd.testing.assert_frame_equal(clips[clips.study_id != "s7"].reset_index(drop=True),
                                      before["shortlist_clips.csv"])
        reference = pd.read_csv(os.path.join(self.out, "reference_clips.csv"), dtype=str)
        pd.testing.assert_frame_equal(reference[reference.vendor != "Philips Medical Systems"].reset_index(drop=True),
                                      before["reference_clips.csv"])
        self.assertEqual(clips.groupby("vendor").mode_source.agg(set).to_dict(),
                         {"GE Vingmed Ultrasound": {"header"}, "Philips Medical Systems": {"pixels"}})
        with open(os.path.join(self.out, "shortlist.json")) as f:
            record = json.load(f)
        self.assertEqual((record["criteria"]["vendor_studies"], record["pool_per_vendor_slot"]),
                         ({"Philips Medical Systems": 1}, {"Philips Medical Systems": 1}))
        # The slot's funnel row counts the shortlist so far, its own clips in the note: the clips to screen.
        draw, slot_row, screened = (next(r for r in record["funnel"] if r["step"].startswith(step))
                                    for step in ("seeded draw", "vendor slot:", "shortlist: clips to screen"))
        added = int((clips.study_id == "s7").sum())

        def counts(row):
            return [row[k] for k in ("videos", "studies", "patients")]

        self.assertEqual(counts(slot_row), [draw["videos"] + added, draw["studies"] + 1, draw["patients"] + 1])
        self.assertEqual(slot_row["note"], f"adds videos={added}, studies=1, patients=1; pool=1")
        self.assertEqual(counts(screened), [len(clips), *counts(slot_row)[1:]])
        # The patients of a validation list are drawn nowhere: without s7's patient no Philips study is left.
        listed = os.path.join(self.tmp, "validation_clips.csv")
        pd.DataFrame({"patient_id": ["p7"]}).to_csv(listed, index=False)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                         "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                         "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                         "--pacs", os.path.join(self.tmp, "pacs.csv"),
                         "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                         "--min-clips", "10", *slot, "--exclude-patients", listed)
        self.assertIn("No training study of Philips Medical Systems", str(stop.exception))
        pd.DataFrame({"patient_id": ["p1"]}).to_csv(listed, index=False)
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--pacs", os.path.join(self.tmp, "pacs.csv"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "10", *slot, "--exclude-patients", listed)
        short = pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str)
        self.assertNotIn("p1", set(short.patient_id))
        self.assertNotIn("p1", set(pd.read_csv(os.path.join(self.out, "reference_clips.csv"), dtype=str).patient_id))

    def test_select_takes_a_vendor_slot_study_at_the_mode_its_pixels_show(self):
        self.add_philips()
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--pacs", os.path.join(self.tmp, "pacs.csv"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "10", "--vendor-studies", "Philips Medical Systems=1")
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"), dtype=str)
        flags_path = os.path.join(self.tmp, "flags.parquet")
        philips = clips.vendor == "Philips Medical Systems"
        # Read from the pixels, every clip is color flow: the GE clips keep their header's B-mode, while the Philips
        # study has no B-mode clip, so no slot study.
        flags = clips[["video_path"]].assign(exclude=False, exclude_reason="", pixel_mode="color_flow")
        write_flags(flags, flags_path, self.out)
        fill = ("--vendor-slots", "Philips Medical Systems")
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path, *fill)
        self.assertIn("none for vendor:Philips Medical Systems", str(stop.exception))
        # Half of the Philips clips B-mode, as their pixels show it: the study qualifies on those, and its color
        # clips are left out, listed with their reason.
        flags.loc[philips & ~clips.video_path.str.contains(r"_[02]\.mp4$"), "pixel_mode"] = "bmode"
        write_flags(flags, flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path, *fill)
        studies = pd.read_csv(os.path.join(self.out, "pilot_studies.csv"), dtype=str)
        self.assertEqual(studies.set_index("category").study_id.get("vendor:Philips Medical Systems"), "s7")
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"), dtype=str)
        s7 = pilot[pilot.study_id == "s7"]
        self.assertEqual((len(s7), set(s7.header_acq_category), set(s7.acq_category), set(s7.mode_source)),
                         (len(FAMILY_VIEWS) * 2, {"bmode"}, {"bmode"}, {"pixels"}))
        ge = pilot[pilot.study_id != "s7"]  # a header mode stays the header's, whatever the pixels show.
        self.assertEqual((len(ge), set(ge.acq_category)), (3 * len(FAMILY_VIEWS) * 2, {"bmode"}))
        not_kept = pd.read_csv(os.path.join(self.out, "not_kept_clips.csv"), dtype=str)
        left_out = not_kept[not_kept.study_id == "s7"]
        self.assertEqual((len(left_out), set(zip(left_out.step, left_out.decision, left_out.reasons,
                                                 left_out.acq_category, left_out.header_acq_category))),
                         (len(FAMILY_VIEWS) * 2, {("screen", "exclude", "color_flow", "color_flow", "bmode")}))
        with open(os.path.join(self.out, "selection.json")) as f:
            selection = json.load(f)
        self.assertEqual((selection["studies"], selection["vendor_slots"]), (4, ["vendor:Philips Medical Systems"]))
        self.run_cmd("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                     "--provisional-screen")
        with open(os.path.join(self.out, "candidates.json"), encoding="utf-8") as f:
            candidates = json.load(f)
        keys = clips.set_index("global_key")
        self.assertEqual({c["modality"] for c in candidates}, {"bmode"})
        self.assertEqual(sum(keys.loc[c["global_key"], "study_id"] == "s7" for c in candidates), len(s7))
        with open(os.path.join(self.out, "candidates_summary.json")) as f:
            self.assertEqual(json.load(f)["candidates_by_mode_source"]["pixels"], len(s7))

    def test_select_fills_the_vendor_slots_it_is_given(self):
        self.add_philips()
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--pacs", os.path.join(self.tmp, "pacs.csv"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "10", "--vendor-studies", "Philips Medical Systems=1")
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"), dtype=str)
        flags_path = os.path.join(self.tmp, "flags.parquet")
        flags = clips[["video_path"]].assign(exclude=False, exclude_reason="", pixel_mode="bmode")
        color = (clips.vendor == "Philips Medical Systems") & clips.video_path.str.contains(r"_[02]\.mp4$")
        flags.loc[color, "pixel_mode"] = "color_flow"
        write_flags(flags, flags_path, self.out)
        # Without --vendor-slots, or with no vendor, the slot's study is screened but not taken.
        for given, slots in (((), []), (("--vendor-slots",), []),
                             (("--vendor-slots", "Philips Medical Systems"), ["vendor:Philips Medical Systems"])):
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path, *given)
            with open(os.path.join(self.out, "selection.json")) as f:
                selection = json.load(f)
            self.assertEqual((selection["studies"], selection["vendor_slots"]), (3 + len(slots), slots))
            studies = pd.read_csv(os.path.join(self.out, "pilot_studies.csv"), dtype=str)
            self.assertEqual(studies.category.tolist(), [*rp.CATEGORIES[:3], *slots])
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path, "--vendor-slots", "Siemens")
        self.assertIn("'Siemens'", str(stop.exception))

    def test_uncertain_clips_are_left_out_listed_and_never_count_for_a_study(self):
        self.shortlist()
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        # Every clip uncertain: no study qualifies on them, the stop says why, and nothing is written.
        uncertain = clips[["video_path"]].assign(decision="uncertain", decision_reasons="unusual_image_content",
                                                 exclude=False, exclude_reason="")
        write_flags(uncertain, flags_path, self.out)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertIn("'uncertain': ", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))
        # One clip uncertain: left out of the pilot, listed with its reasons, counted apart from the excluded ones.
        flags = uncertain.assign(decision="keep", decision_reasons="")
        flags.loc[0, ["decision", "decision_reasons"]] = ["uncertain", "unusual_image_content"]
        write_flags(flags, flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))
        listed = pd.read_csv(os.path.join(self.out, "uncertain_clips.csv"))
        self.assertEqual(listed.global_key.tolist(), [clips.global_key.iloc[0]])
        self.assertEqual(listed.decision_reasons.tolist(), ["unusual_image_content"])
        self.assertNotIn(clips.global_key.iloc[0], set(pilot.global_key))
        with open(os.path.join(self.out, "selection.json")) as f:
            selection = json.load(f)
        self.assertEqual(selection["uncertain_clips_shortlisted"], 1)
        # Flags of a screen that records no decisions (an older flag run) are refused.
        write_flags(clips[["video_path"]].assign(exclude=False, exclude_reason=""), flags_path, self.out,
                    decisions=False)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertIn("the screen's decisions", str(stop.exception))

    def test_every_clip_left_out_is_listed_with_its_reasons(self):
        # s3 and s4 (always chosen) get clips the shortlist removes under --vendor GE: no acquisition mode in the
        # header (one of them also of another vendor), another vendor's B-mode and a still of 3 frames.
        # A B-mode clip of GE Healthcare, one of GE's vendor names, stays (--vendor 'GE Vingmed
        # Ultrasound' would remove it).
        philips = {"vendor": "Philips Medical Systems", "acq_vendor_validated": False}
        extra = [dict(study_clips("s3", "p3", n_per_view=1, views=("A4C",))[0], acq_category="unknown",
                      acq_confidence="unknown", video_path=f"/synthetic/s3/A4C_x{i}.mp4", **kwargs)
                 for i, kwargs in enumerate(({}, philips))]
        extra += [dict(study_clips("s4", "p4", n_per_view=1, views=("A2C",))[0], acq_category="bmode",
                       video_path=f"/synthetic/s4/A2C_x{i}.mp4", **kwargs)
                  for i, kwargs in enumerate((philips, {"n_frames": 3}))]
        healthcare = dict(study_clips("s4", "p4", n_per_view=1, views=("A2C",))[0], acq_category="bmode",
                          vendor="GE Healthcare Ultrasound", video_path="/synthetic/s4/A2C_x3.mp4")
        pd.concat([self.videos, pd.DataFrame(extra + [healthcare])]).to_parquet(
            os.path.join(self.tmp, "videos.parquet"))
        self.shortlist("--vendor", "GE")
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        self.assertIn(healthcare["video_path"], set(clips.video_path))
        removed = pd.read_csv(os.path.join(self.out, "removed_clips.csv"))
        reasons = dict(zip(removed.video_path, removed.reasons))
        self.assertEqual({p: reasons.pop(p) for p in [r["video_path"] for r in extra]},
                         {"/synthetic/s3/A4C_x0.mp4": "acquisition", "/synthetic/s3/A4C_x1.mp4": "acquisition;vendor",
                          "/synthetic/s4/A2C_x0.mp4": "vendor", "/synthetic/s4/A2C_x1.mp4": "few_frames"})
        # The fixture's color clips, OTHER ones among them.
        self.assertEqual(pd.Series(reasons).value_counts().to_dict(), {"color_flow": 48, "other_view;color_flow": 12})
        self.assertFalse(set(removed.video_path) & set(clips.video_path))
        self.assertTrue(removed.global_key.str.fullmatch(r"radar-pilot-[0-9a-f]{24}").all())
        with open(os.path.join(self.out, "shortlist.json")) as f:
            record = json.load(f)
        self.assertEqual((record["clips_removed"], record["clips_removed_by_metadata_rule"]),
                         (64, {"color_flow": 60, "other_view": 12, "acquisition": 2, "vendor": 2,
                               "few_frames": 1}))

        # The screen excludes an A3C clip of each normal study (one of them not chosen) and is uncertain about two
        # clips of s4 for their GE image code: the labelled code of 3D and multi-panel displays is excluded, a code
        # never labelled stays uncertain. With the removed clips, all are listed, each with its step, decision,
        # reasons and pilot study.
        flags_path = os.path.join(self.tmp, "flags.parquet")
        flags = clips[["video_path"]].assign(decision="keep", decision_reasons="", exclude=False, exclude_reason="",
                                             dicom_image_content="0001")
        for path, decision, reasons, code in (
                ("/synthetic/s1/A3C_1.mp4", "exclude", "still", "0001"),
                ("/synthetic/s2/A3C_1.mp4", "exclude", "analysis_screen;measurement_text", "0000"),
                ("/synthetic/s4/A4C_1.mp4", "uncertain", "unusual_image_content", "0041"),
                ("/synthetic/s4/A5C_1.mp4", "uncertain", "unusual_image_content", "0021")):
            flags.loc[flags.video_path == path, ["decision", "decision_reasons", "dicom_image_content"]] = [
                decision, reasons, code]
        write_flags(flags, flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        chosen = set(pd.read_csv(os.path.join(self.out, "pilot_studies.csv"), dtype=str).study_id)
        listed = pd.read_csv(os.path.join(self.out, "not_kept_clips.csv"), dtype={"study_id": str})
        backups = listed[listed.step == "selection"]
        self.assertEqual((len(backups), set(backups.reasons), set(backups.decision)),
                         (11, {"study_not_chosen"}, {"not_chosen"}))
        self.assertFalse(backups.in_pilot.any())
        own = listed[~listed.reasons.isin(["color_flow", "other_view;color_flow", "study_not_chosen"])]
        self.assertEqual(sorted(zip(own.video_path, own.step, own.decision, own.reasons)),
                         [("/synthetic/s1/A3C_1.mp4", "screen", "exclude", "still"),
                          ("/synthetic/s2/A3C_1.mp4", "screen", "exclude", "analysis_screen;measurement_text"),
                          ("/synthetic/s3/A4C_x0.mp4", "shortlist", "removed", "acquisition"),
                          ("/synthetic/s3/A4C_x1.mp4", "shortlist", "removed", "acquisition;vendor"),
                          ("/synthetic/s4/A2C_x0.mp4", "shortlist", "removed", "vendor"),
                          ("/synthetic/s4/A2C_x1.mp4", "shortlist", "removed", "few_frames"),
                          ("/synthetic/s4/A4C_1.mp4", "screen", "exclude", "3d_or_multipanel"),
                          ("/synthetic/s4/A5C_1.mp4", "screen", "uncertain", "unusual_image_content")])
        self.assertEqual(len(listed) - len(own) - len(backups), 60)
        self.assertEqual(listed.in_pilot.tolist(), listed.study_id.isin(chosen).tolist())
        self.assertEqual(set(listed.study_id[~listed.in_pilot]), {"s1", "s2"} - chosen)  # the normal study not chosen.
        self.assertEqual(os.stat(os.path.join(self.out, "not_kept_clips.csv")).st_mode & 0o777, 0o600)
        uncertain = pd.read_csv(os.path.join(self.out, "uncertain_clips.csv"))
        self.assertEqual(uncertain.video_path.tolist(), ["/synthetic/s4/A5C_1.mp4"])
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))
        self.assertFalse({"/synthetic/s4/A4C_1.mp4", "/synthetic/s4/A5C_1.mp4"} & set(pilot.video_path))
        with open(os.path.join(self.out, "selection.json")) as f:
            selection = json.load(f)
        in_pilot = removed[removed.study_id.isin(chosen)]
        self.assertEqual((selection["clips_removed_before_screen"], selection["removed_reasons_in_pilot_studies"],
                          selection["not_kept_clips_shortlisted"]),
                         (len(in_pilot), in_pilot.reasons.str.split(";").explode().value_counts().to_dict(),
                          {"removed": 64, "exclude": 3, "uncertain": 1, "not_chosen": 11}))
        self.assertEqual(selection["pilot_rule"]["3d_or_multipanel"],
                         {"vendor": "GE Vingmed Ultrasound", "image_codes": ["0041", "0051"]})
        with open(os.path.join(self.out, "summary.md")) as f:
            self.assertIn(f"removed before the screen: {len(in_pilot)}", f.read())

        # A selection made before V1's rule is refused by candidates.
        path = os.path.join(self.out, "selection.json")
        with open(path, "w") as f:
            json.dump({k: v for k, v in selection.items() if k != "pilot_rule"}, f)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                         "--provisional-screen")
        self.assertIn("before V1's rule", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))

        # A shortlist made before removed_clips.csv existed, or before V1's rule, is refused, and nothing is selected.
        for older in ({**record, "outputs": {k: v for k, v in record["outputs"].items()
                                             if not k.endswith("removed_clips.csv")}},
                      {**record, "criteria": {k: v for k, v in record["criteria"].items() if k != "captioned"}}):
            with open(os.path.join(self.out, "shortlist.json"), "w") as f:
                json.dump(older, f)
            with self.assertRaises(SystemExit) as stop:
                self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
            self.assertIn("an older shortlist", str(stop.exception))
            self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))

    def test_a_pilot_short_of_its_studies_is_not_a_selection(self):
        self.shortlist()
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        trial = ("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                 "--provisional-screen")
        # Every clip excluded: no study is left, and nothing is written for candidates to take.
        write_flags(clips[["video_path"]].assign(exclude=True, exclude_reason="incomplete"), flags_path, self.out)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertIn("none for normal, lv_dysfunction, valve_disease", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))
        with self.assertRaises(SystemExit):
            self.run_cmd(*trial)
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))
        # A selection recording fewer studies than asked for (an older select warned and wrote it) is refused too.
        write_flags(clips[["video_path"]].assign(exclude=False, exclude_reason=""), flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        path = os.path.join(self.out, "selection.json")
        with open(path) as f:
            selection = json.load(f)
        with open(path, "w") as f:
            json.dump({**selection, "studies": 0}, f)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd(*trial)
        self.assertIn("fewer studies than it was asked for", str(stop.exception))

    def test_inputs_replaced_while_a_step_runs_complete_nothing(self):
        reports_path = os.path.join(self.tmp, "reports.parquet")
        reports = pd.read_parquet(reports_path)
        severe = reports.assign(**{"Mitral valve findings": "Insuffisance mitrale sévère"})

        def replaced_during(name, write):  # radar_pilot's `name`, run after `write` rewrote an input.
            real = getattr(rp, name)

            def meanwhile(*args, **kwargs):
                write()
                return real(*args, **kwargs)

            return mock.patch.object(rp, name, meanwhile)

        # shortlist: the reports replaced after they were read (every study severe MR, none normal).
        with replaced_during("categorize", lambda: severe.to_parquet(reports_path)), \
                self.assertRaises(SystemExit) as stop:
            self.shortlist()
        self.assertIn("reports.parquet changed while this step ran", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SHORTLIST_OUTPUTS))
        reports.to_parquet(reports_path)
        self.shortlist()
        # select: the flags replaced, with their record, while the studies are chosen.
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        kept = clips[["video_path"]].assign(exclude=False, exclude_reason="")
        write_flags(kept, flags_path, self.out)
        dropped = kept.assign(exclude=True, exclude_reason="measurement")
        with replaced_during("choose_studies", lambda: write_flags(dropped, flags_path, self.out)), \
                self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertIn("changed while this step ran", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))
        write_flags(kept, flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        # candidates: the reports replaced while the prefills are built.
        with replaced_during("report_record", lambda: severe.to_parquet(reports_path)), \
                self.assertRaises(SystemExit) as stop:
            self.run_cmd("candidates", "--out-dir", self.out, "--reports", reports_path, "--provisional-screen")
        self.assertIn("reports.parquet changed while this step ran", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))

    def test_candidates_build_every_prefill_from_the_spec_read_once(self):
        self.shortlist()
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        write_flags(clips[["video_path"]].assign(exclude=False, exclude_reason=""), flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        sys.path.insert(0, REPO_ROOT)
        from tools.labelbox import column_review

        # A spec reloaded per clip could change version midway: the builder must never load it itself.
        with mock.patch.object(column_review, "load_spec", side_effect=AssertionError("spec reloaded")):
            self.run_cmd("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                         "--provisional-screen")
        with open(os.path.join(self.out, "candidates_summary.json")) as f:
            summary = json.load(f)
        spec = os.path.relpath(column_review.SPEC, REPO_ROOT)  # the review spec, whatever its file is named.
        self.assertEqual(summary["spec"], {spec: rp.sha256(os.path.join(REPO_ROOT, spec))})

    def test_clips_replaced_in_place_since_the_screen_stop_select_and_candidates(self):
        self.shortlist()
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        write_flags(clips[["video_path"]].assign(exclude=False, exclude_reason=""), flags_path, self.out)
        trial = ("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                 "--provisional-screen")
        with mock.patch.object(rp.clip_screen, "changed_media", return_value=["k"]) as media, \
                self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertIn("1 clips of the chosen studies no longer hold", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))
        self.assertTrue(media.call_args.args[0].study_id.isin(set(clips.study_id)).all())
        self.assertEqual(media.call_args.args[1], os.path.join(self.out, "media.log"))  # a private folder.
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))
        self.assertTrue((pilot.video_sha256 == "synthetic").all())
        with mock.patch.object(rp.clip_screen, "changed_media", return_value=["k"]) as media, \
                self.assertRaises(SystemExit) as stop:
            self.run_cmd(*trial)
        self.assertIn("1 pilot clips no longer hold", str(stop.exception))
        self.assertEqual(media.call_args.kwargs, {"dicom": False})  # what reviewers see: the video.
        self.assertEqual(media.call_args.args[1], os.path.join(self.out, "media.log"))
        self.run_cmd(*trial)
        keys = pd.read_csv(os.path.join(self.out, "global_keys.csv"))
        self.assertTrue((keys.video_sha256 == "synthetic").all())  # for the upload to check.

    def test_select_after_the_screen_and_candidates(self):
        self.shortlist()
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        flags = clips[["video_path"]].assign(exclude=False, exclude_reason="")
        write_flags(flags, flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        first = pd.read_csv(os.path.join(self.out, "pilot_studies.csv"), dtype=str).set_index("category").study_id
        other_normal = ({"s1", "s2"} - {first["normal"]}).pop()
        # The screen flags the first normal study's PSAX clips, which are removed: that study no longer has every
        # view family, so the other normal study is taken. A flagged clip of a study that still qualifies is
        # removed from it.
        psax = clips.video_path.str.contains(f"/{first['normal']}/PSAX")
        flags.loc[psax, ["exclude", "exclude_reason"]] = [True, "measurement"]
        still = clips.video_path.str.endswith("/s3/A4C_1.mp4")
        flags.loc[still, ["exclude", "exclude_reason"]] = [True, "still"]
        write_flags(flags, flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        studies = pd.read_csv(os.path.join(self.out, "pilot_studies.csv"), dtype=str)
        self.assertEqual(dict(zip(studies.category, studies.study_id)),
                         {"normal": other_normal, "lv_dysfunction": "s3", "valve_disease": "s4"})
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))
        self.assertNotIn(clips.global_key[still].iloc[0], set(pilot.global_key))
        with open(os.path.join(self.out, "selection.json")) as f:
            selection = json.load(f)
        self.assertEqual((selection["clips_excluded"], selection["screen_reasons_in_pilot_studies"]["still"]), (1, 1))
        self.assertEqual(selection["inputs"][os.path.abspath(flags_path)], rp.sha256(flags_path))
        self.assertEqual(set(selection["code_sha256"]),
                         {os.path.join("data", "radar_pilot.py"), os.path.join("data", "clip_screen.py")})

        self.run_cmd("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                     "--provisional-screen")
        with open(os.path.join(self.out, "candidates.json"), encoding="utf-8") as f:
            candidates = json.load(f)
        # The B-mode clips of the three studies, less the still clip.
        self.assertEqual(len(candidates), 3 * len(FAMILY_VIEWS) * 2 - 1)
        self.assertEqual(({c["view"] for c in candidates} - set(FAMILY_VIEWS), {c["modality"] for c in candidates}),
                         (set(), {"bmode"}))
        # The OTHER videos (of s3, and s1 when chosen) were removed by the shortlist: none goes to the view and type
        # review.
        items = pd.read_csv(os.path.join(self.out, "view_type_items.csv"))
        keys_map = pd.read_csv(os.path.join(self.out, "global_keys.csv"))
        self.assertEqual((len(items), set(keys_map.form)), (0, {"column_review"}))
        shortlist_key = clips.set_index("video_path").global_key
        self.assertEqual(keys_map.global_key.tolist(), shortlist_key[keys_map.video_path].tolist())
        keys = [c["global_key"] for c in candidates]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertTrue(all(k.startswith("radar-pilot-") and other_normal not in k for k in keys))
        a4c = next(c for c in candidates if c["view"] == "A4C" and c["modality"] == "bmode")
        self.assertIn({"concept": "tricuspid_appearance", "value": "normal"},
                      [{k: col[k] for k in ("concept", "value")} for col in a4c["columns"]])
        self.assertEqual(os.stat(os.path.join(self.out, "candidates.json")).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(os.path.join(self.out, rp.SECRET_NAME)).st_mode & 0o777, 0o600)
        again = os.path.join(self.tmp, "again.json")
        os.rename(os.path.join(self.out, "candidates.json"), again)
        self.run_cmd("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                     "--provisional-screen")
        with open(os.path.join(self.out, "candidates.json"), encoding="utf-8") as f:
            self.assertEqual([c["global_key"] for c in json.load(f)], keys)
        with open(os.path.join(self.out, "candidates_summary.json")) as f:
            self.assertEqual(json.load(f)["reports"], rp.sha256(os.path.join(self.tmp, "reports.parquet")))
        # Report text changed after shortlist would no longer match the categories the studies were chosen in.
        reports = pd.read_parquet(os.path.join(self.tmp, "reports.parquet"))
        reports.loc[reports.AccessionNumber == "A1", "Mitral valve findings"] = "Insuffisance mitrale sévère"
        reports.to_parquet(os.path.join(self.tmp, "changed_reports.parquet"))
        with self.assertRaises(SystemExit):
            self.run_cmd("candidates", "--out-dir", self.out,
                         "--reports", os.path.join(self.tmp, "changed_reports.parquet"), "--provisional-screen")

        # New flags make the selection stale: candidates refuses it.
        valve = clips.video_path.str.endswith("/s4/A4C_1.mp4")
        flags.loc[valve, ["exclude", "exclude_reason"]] = [True, "still"]
        write_flags(flags, flags_path, self.out)
        with self.assertRaises(SystemExit):
            self.run_cmd("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                     "--provisional-screen")
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))  # no earlier candidates left.
        # A select that stops (a shortlisted clip unscreened) leaves neither a selection nor candidates behind.
        write_flags(flags.iloc[:-1], flags_path, self.out)
        with self.assertRaises(SystemExit):
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS + rp.CANDIDATE_OUTPUTS))
        with self.assertRaises(SystemExit):
            self.run_cmd("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                     "--provisional-screen")

    def flagged_shortlist(self, **kwargs):
        """A shortlist and flags (none excluding) screened from it; returns the flags table and path."""
        self.shortlist()
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        flags = clips[["video_path"]].assign(exclude=False, exclude_reason="")
        write_flags(flags, flags_path, self.out, **kwargs)
        return flags, flags_path

    def test_flags_must_be_screened_from_this_shortlist(self):
        flags, flags_path = self.flagged_shortlist()
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        # New DICOM paths and a new shortlist: the same videos, but the flags were screened from the old clip list.
        moved = self.videos.assign(dicom_path=self.videos.dicom_path.str.replace(".dcm", "_new.dcm"))
        moved.to_parquet(os.path.join(self.tmp, "videos.parquet"))
        self.shortlist()
        with self.assertRaises(SystemExit):
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))
        write_flags(flags, flags_path, self.out)  # screened again from the new lists.
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        # A features file rerun after `flag` read it breaks the chain too, as do flags recording no features file.
        features = os.path.join(self.tmp, "features.parquet")
        pd.DataFrame({"video_path": ["/synthetic/y.mp4"]}).to_parquet(features)
        rp.clip_screen._write_info(features, {"clips": rp.sha256(os.path.join(self.out, "shortlist_clips.csv"))})
        with self.assertRaises(SystemExit):
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        rp.clip_screen._write_info(flags_path, {"thresholds": rp.clip_screen.THRESHOLDS})
        with self.assertRaises(SystemExit):
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)

    def test_candidates_check_the_screening_chain_again(self):
        trial = ("candidates", "--out-dir", self.out, "--reports", os.path.join(self.tmp, "reports.parquet"),
                 "--provisional-screen")
        _, flags_path = self.flagged_shortlist()
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.run_cmd(*trial)
        # The reference list changed after select: the flags' baselines no longer describe it.
        reference = os.path.join(self.out, "reference_clips.csv")
        with open(reference) as f:
            saved = f.read()
        with open(reference, "a") as f:
            f.write("\n")
        with self.assertRaises(SystemExit):
            self.run_cmd(*trial)
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))
        with open(reference, "w") as f:
            f.write(saved)
        self.run_cmd(*trial)
        # The features file behind the flags, and its record of a completed run, deleted after select.
        for name in ("features.parquet", "features.parquet.info.json"):
            os.remove(os.path.join(self.tmp, name))
        with self.assertRaises(SystemExit):
            self.run_cmd(*trial)
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))

    def test_candidates_need_a_calibrated_screen_or_a_trial_run(self):
        reports = os.path.join(self.tmp, "reports.parquet")
        _, flags_path = self.flagged_shortlist()
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        with self.assertRaises(SystemExit):
            self.run_cmd("candidates", "--out-dir", self.out, "--reports", reports)
        self.assertFalse(set(os.listdir(self.out)) & set(rp.CANDIDATE_OUTPUTS))
        self.run_cmd("candidates", "--out-dir", self.out, "--reports", reports, "--provisional-screen")
        with open(os.path.join(self.out, "candidates_summary.json")) as f:
            self.assertTrue(json.load(f)["provisional_screen"])
        _, flags_path = self.flagged_shortlist(calibration="calibration report of 2026-11-02, sha256 0123abcd")
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.run_cmd("candidates", "--out-dir", self.out, "--reports", reports)
        with open(os.path.join(self.out, "candidates_summary.json")) as f:
            self.assertFalse(json.load(f)["provisional_screen"])
        with open(os.path.join(self.out, "summary.md")) as f:
            self.assertIn("calibrated (calibration report of 2026-11-02", f.read())

    def test_outputs_inside_the_repository_are_refused(self):
        with self.assertRaises(SystemExit):
            rp.main(["select", "--out-dir", os.path.join(REPO_ROOT, "pilot"), "--flags", "x.parquet"])


if __name__ == "__main__":
    unittest.main()

