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


class TestClipRules(unittest.TestCase):
    def test_rules(self):
        clips = pd.DataFrame({"view": ["A4C", "OTHER", None, "A4C", "A4C", "A4C", "OTHER", "OTHER"],
                              "acq_category": ["bmode", "bmode", "bmode", "unknown", "color_flow", "bmode", "unknown",
                                               "bmode"],
                              "acq_confidence": ["header"] * 5 + ["unknown"] * 2 + ["header"],
                              "acq_vendor_validated": [True] * 8, "n_frames": [30, 30, 30, 30, 4, 30, 30, 4]})
        self.assertEqual(rp.review_form(clips).tolist(), ["column_review", "view_type_review", "view_type_review",
                                                          "column_review", "column_review", "column_review",
                                                          "view_type_review", "view_type_review"])
        # An OTHER clip needs no known acquisition mode: its reviewer gives it. Short clips (4 frames) stay in
        # both forms.
        self.assertEqual(rp.clip_rule(clips).tolist(), ["", "", "", "acquisition", "", "acquisition", "", ""])


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
             "dicom_ecg": True}
    if decisions and "decision" not in flags:  # the screen's decision, from the exclusion given (no uncertain clip).
        flags = flags.assign(decision=np.where(flags.exclude, "exclude", "keep"),
                             decision_reasons=flags.exclude_reason)
    flags.assign(**{k: v for k, v in media.items() if k not in flags}).to_parquet(path)
    rp.clip_screen._write_info(path, {**sources, "thresholds": rp.clip_screen.THRESHOLDS,
                                      "calibration": calibration})


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
        clips = pd.DataFrame(rows).drop_duplicates("video_path").assign(exclude=False)
        clips.loc[clips.study_id.isin(flagged_psax) & clips.view.eq("PSAX"), "exclude"] = True
        chosen = rp.choose_studies(pd.DataFrame(short), clips, 12, 3)
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
        pd.DataFrame({"study_id": [f"s{i}" for i in range(1, 7)], "rvsp": [30] * 6}).to_parquet(
            os.path.join(self.tmp, "rvsp.parquet"))
        pd.DataFrame({"AccessionNumber": [f"A{i}" for i in range(1, 7)],
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

    def shortlist(self):
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "12")

    def test_shortlist_keeps_training_studies_and_removes_unreviewable_clips(self):
        self.shortlist()
        short = pd.read_csv(os.path.join(self.out, "shortlist.csv"), dtype=str)
        self.assertEqual(sorted(short.study_id), ["s1", "s2", "s3", "s4"])
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        other = clips[clips.view == "OTHER"]
        self.assertEqual(len(other), 12)
        self.assertEqual(set(other.form), {"view_type_review"})
        self.assertEqual(set(clips[clips.view != "OTHER"].form), {"column_review"})
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
        # s6 (train, no PSAX) is the only other training study; s5 is a validation patient.
        self.assertEqual(set(reference.study_id), {"s6"})
        self.assertEqual(len(reference), 20)
        self.run_cmd("shortlist", "--videos", os.path.join(self.tmp, "videos.parquet"),
                     "--ef-labels", os.path.join(self.tmp, "ef.parquet"),
                     "--rvsp-labels", os.path.join(self.tmp, "rvsp.parquet"),
                     "--reports", os.path.join(self.tmp, "reports.parquet"), "--out-dir", self.out,
                     "--min-clips", "12", "--reference-per-layout", "3")
        reference = pd.read_csv(os.path.join(self.out, "reference_clips.csv"))
        self.assertEqual(reference.groupby(["vendor", "model", "acq_category"]).size().tolist(), [3, 3])
        self.assertTrue(reference.global_key.str.startswith("radar-pilot-").all())
        with open(os.path.join(self.out, "shortlist.json")) as f:
            self.assertEqual(json.load(f)["clips_removed_by_metadata_rule"], {})

    def test_uncertain_clips_are_left_out_listed_and_never_count_for_a_study(self):
        self.shortlist()
        clips = pd.read_csv(os.path.join(self.out, "shortlist_clips.csv"))
        flags_path = os.path.join(self.tmp, "flags.parquet")
        # Every clip uncertain: no study qualifies on them, the stop says why, and nothing is written.
        uncertain = clips[["video_path"]].assign(decision="uncertain", decision_reasons="measurement_trace",
                                                 exclude=False, exclude_reason="")
        write_flags(uncertain, flags_path, self.out)
        with self.assertRaises(SystemExit) as stop:
            self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        self.assertIn("'uncertain': ", str(stop.exception))
        self.assertFalse(set(os.listdir(self.out)) & set(rp.SELECTION_OUTPUTS))
        # One clip uncertain: left out of the pilot, listed with its reasons, counted apart from the excluded ones.
        flags = uncertain.assign(decision="keep", decision_reasons="")
        flags.loc[0, ["decision", "decision_reasons"]] = ["uncertain", "measurement_trace"]
        write_flags(flags, flags_path, self.out)
        self.run_cmd("select", "--out-dir", self.out, "--flags", flags_path)
        pilot = pd.read_csv(os.path.join(self.out, "pilot_clips.csv"))
        listed = pd.read_csv(os.path.join(self.out, "uncertain_clips.csv"))
        self.assertEqual(listed.global_key.tolist(), [clips.global_key.iloc[0]])
        self.assertEqual(listed.decision_reasons.tolist(), ["measurement_trace"])
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
        still = clips.video_path.str.endswith("/s3/A4C_0.mp4")
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
        self.assertEqual(len(candidates), 3 * len(FAMILY_VIEWS) * 4 - 1)  # less the still clip.
        self.assertNotIn("OTHER", {c["view"] for c in candidates})
        items = pd.read_csv(os.path.join(self.out, "view_type_items.csv"))
        # The OTHER videos of the chosen studies with any (s3, and s1 when chosen) get the view and type review.
        self.assertEqual(len(items), 6 * len({"s1", "s3"} & set(studies.study_id)))
        keys_map = pd.read_csv(os.path.join(self.out, "global_keys.csv"))
        self.assertEqual(set(keys_map[keys_map.form == "view_type_review"].global_key), set(items.global_key))
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
        valve = clips.video_path.str.endswith("/s4/A4C_0.mp4")
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
