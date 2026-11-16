# tests/data/test_radar_normality_audit.py

"""`data/radar_normality_audit.py` on a synthetic caption table.

Coverage: counts with denominators, the funnel to the first epoch's bags (the same bags the training dataset draws),
reasons for unknown, identical captions with different flags, and a table that no longer matches its rules is refused.
Sample: blind, seeded, stratified, weighted so that the rates are rates among rows, with rare wording reached.
Score: false normals by structure, a known weighted answer, and refusal of an unfinished or mismatched sheet.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import build_radar_captions as builder  # noqa: E402
import radar_captions as rc  # noqa: E402
import radar_normality as rn  # noqa: E402
import radar_normality_audit as audit  # noqa: E402

from src.datasets.radar_study_dataset import RadarStudyDataset, file_sha256  # noqa: E402
from src.datasets.radar_token_cache import TokenCache  # noqa: E402
from src.models.radar.routing import load_routing  # noqa: E402
from tests.models.radar_synthetic import DIM, GRID, make_synthetic  # noqa: E402

POLICY = rn.load_policy()
ROUTING = load_routing()


MITRAL = next(s.report_column for s in ROUTING.trained if s.name == "mitral")
NOTHING_EXCLUDED = pd.DataFrame(columns=builder.EXCLUSION_COLUMNS)


def synthetic_table():
    """(table, ledger). 60 common normal pericardium rows, 6 of rarer wording, 10 abnormal, 4 unknown; 10 LV rows, some
    in conflict; four cells removed in full, two of them with text the rules cannot explain."""
    reports = []
    captions = (["Pas d'épanchement péricardique."] * 60 + ["Péricarde sec."] * 3 + ["Absence d'épanchement."] * 3
                + ["Épanchement péricardique modéré."] * 10 + ["Clause jamais vue."] * 4)
    columns = {s.report_column: None for s in ROUTING.trained}
    columns.update({c: None for s in ROUTING.trained for c in rn.report_columns(s.name, POLICY)})
    for i, text in enumerate(captions):
        row = {**columns, "study_id": f"S{i:03d}", "Pericardium findings": text}
        if i < 10:
            row["Left ventricule findings"] = "VG non dilaté. Fonction systolique normale."
            row["LVID Diastole (2D)"] = 6.5 if i < 3 else 4.8
        if i < 4:
            row[MITRAL] = "Sténose mitrale calcifiée probable." if i < 2 else "Pas de sténose mitrale."
        reports.append(row)
    frame, ledger = builder.build_tables(pd.DataFrame(reports), ROUTING, POLICY)
    frame["boundary"] = frame.boundary.astype(bool)
    return frame, ledger


class AuditCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.table, cls.ledger = synthetic_table()
        cls.clips = os.path.join(cls.root, "clips.csv")
        pd.DataFrame({"study_id": cls.table.study_id.unique()}).assign(
            clip_id=lambda t: t.study_id + "-c", view="A4C", split="train").to_csv(cls.clips, index=False)
        cls.path = cls.seal(cls.table, "built")

    @classmethod
    def seal(cls, table, name, ledger=None, **changes):
        """Write `table` as the builder does, with its ledger and `build_info.json` beside it."""
        folder = os.path.join(cls.root, name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "captions.csv")
        table.to_csv(path, index=False)
        exclusions = os.path.join(folder, "exclusions.csv")
        (cls.ledger if ledger is None else ledger).to_csv(exclusions, index=False)
        info = {"split": "train", "routing_sha256": ROUTING.sha256, "clips_sha256": file_sha256(cls.clips),
                "captions_sha256": file_sha256(path), "exclusions_sha256": file_sha256(exclusions),
                "studies_linked": table.study_id.nunique(), **changes}
        with open(os.path.join(folder, "build_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f)
        return path

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)


class TestCoverage(AuditCase):

    def test_counts_reasons_and_denominators(self):
        report = audit.coverage(audit.read_captions(self.path), ROUTING, POLICY)
        peri = report["structures"]["pericardium"]
        self.assertEqual(peri["rows"], 80)
        self.assertEqual(peri["flags"]["normal"], {"n": 66, "fraction": 0.825})
        self.assertEqual(peri["flags"]["abnormal"]["n"], 10)
        self.assertEqual(peri["flags"]["unknown"]["n"], 4)
        self.assertEqual(peri["unknown_rows_with_reason"]["unrecognized_text"]["n"], 4)
        lv = report["structures"]["lv"]
        self.assertEqual(lv["conflicts"], {"text_structured": 3})
        self.assertEqual(lv["identical_captions_with_different_flags"], 1)    # size conflict in 3 of 10
        self.assertEqual(report["policy_status"], "provisional")

    def test_the_funnel_follows_the_training_bags(self):
        root = tempfile.mkdtemp(dir=self.root)
        paths = make_synthetic(os.path.join(root, "synthetic"))
        clips = pd.read_csv(paths["clips"], dtype={"study_id": str, "clip_id": str})
        captions = pd.read_csv(paths["captions"], dtype={"study_id": str})
        cache = TokenCache(paths["cache_dir"], token_grid=GRID, embed_dim=DIM)
        dataset = RadarStudyDataset(clips, captions, ROUTING, cache, num_clips=2, seed=3)
        eligibility = audit.bag_eligibility(clips, ROUTING, 2, 3)
        views = dict(zip(clips.clip_id, clips.view))
        for index, (study, _) in enumerate(dataset.studies):
            bag = {views[c] for c in dataset[index]["clip_ids"]}
            expected = {s.name for s in ROUTING.trained if bag & set(s.views)}
            self.assertEqual(eligibility[study][1], expected, study)

    def test_funnel_counts(self):
        table = audit.read_captions(self.path)
        clips = pd.DataFrame({"study_id": table.study_id.unique(), "split": "train"})
        clips = clips.assign(clip_id=clips.study_id + "-c", view=["A4C" if i % 2 else "PLAX"
                                                                  for i in range(len(clips))])
        report = audit.coverage(table, ROUTING, POLICY, clips, num_clips=2)
        funnel = report["structures"]["lv"]["funnel"]
        self.assertEqual(funnel["studies_in_split"], 80)
        self.assertEqual(funnel["with_caption"], 10)
        self.assertEqual(funnel["with_eligible_clip"], 10)                 # the LV appears in both views
        ivc = report["structures"]["ivc"]["funnel"]
        self.assertEqual((ivc["with_caption"], ivc["with_eligible_clip_in_bag"]), (0, 0))
        self.assertIsNotNone(report["unknown_fraction"])

    def test_captions_outside_the_split_are_an_error(self):
        table = audit.read_captions(self.path)
        clips = pd.DataFrame({"study_id": ["S000"], "clip_id": ["c"], "view": ["A4C"], "split": ["train"]})
        with self.assertRaisesRegex(SystemExit, "not in split"):
            audit.coverage(table, ROUTING, POLICY, clips)

    def test_a_table_that_disagrees_with_its_caption_is_found(self):
        table = audit.read_captions(self.path)
        self.assertEqual(audit.table_problems(table, POLICY), {})
        edited = table.copy()
        edited.loc[edited.caption == "Clause jamais vue.", "caption"] = "Épanchement important."
        problems = audit.table_problems(edited, POLICY)
        for kind in ("flag_differs_from_caption", "caption_differs_from_source", "caption_id_differs_from_caption"):
            self.assertEqual(problems[kind], 4)
        other = table.assign(policy_digest="older")
        self.assertEqual(audit.table_problems(other, POLICY), {"other_policy_or_rules": len(table)})

    def test_each_part_of_a_row_is_checked_on_its_own(self):
        table = audit.read_captions(self.path)
        for changes, kind in (({"caption_id": 7}, "caption_id_differs_from_caption"),
                              ({"evidence": "structured"}, "evidence_differs_from_caption"),
                              ({"boundary": True}, "evidence_differs_from_caption"),
                              ({"dropped_parts": 5}, "caption_differs_from_source"),
                              ({"source": "Texte inconnu."}, "caption_differs_from_source")):
            with self.subTest(kind, changes=changes):
                self.assertEqual(audit.table_problems(table.iloc[:1].assign(**changes), POLICY), {kind: 1})
        longer = table[table.structure != "lv"].iloc[:1]
        longer = longer.assign(caption=f"{longer.caption.iloc[0]} Pas d'épanchement.")
        longer = longer.assign(caption_id=[rc.caption_id(longer.caption.iloc[0])])
        self.assertEqual(audit.table_problems(longer, POLICY).get("caption_differs_from_source"), 1)
        twice = pd.concat([table.iloc[:1]] * 2)
        self.assertEqual(audit.table_problems(twice, POLICY), {"study_and_structure_twice": 1})
        self.assertEqual(audit.table_problems(table.drop(columns=["source"]), POLICY),
                         {"no_source_column": len(table)})

    def test_unrecognized_wording_keeps_only_repeated_text(self):
        wording = audit.unrecognized_wording(audit.read_captions(self.path), min_count=4)
        self.assertEqual(wording.to_dict("records"),
                         [{"structure": "pericardium", "wording": "clause jamais vue", "count": 4}])
        self.assertTrue(audit.unrecognized_wording(audit.read_captions(self.path), min_count=5).empty)


class TestFidelity(AuditCase):

    def pairs(self):
        rows = [{"structure": "lv", "source": f"Source {i}.", "caption": f"Texte {i}.", "dropped_parts": 1,
                 "stranded_parts": int(i < 10)} for i in range(40)]
        return pd.DataFrame(rows + [{"structure": "lv", "source": "Rien.", "caption": "Rien.", "dropped_parts": 0,
                                     "stranded_parts": 0}])

    def test_it_pairs_rows_that_lost_text_half_of_them_quietly(self):
        sheet = audit.fidelity_sample(self.pairs(), NOTHING_EXCLUDED, rows=12, seed=0)
        self.assertEqual(list(sheet.columns), ["structure", "source", "caption", "reader_fidelity", "reader_note"])
        self.assertEqual(len(sheet), 12)
        self.assertNotIn("Rien.", set(sheet.source))
        numbers = sheet.source.str.extract(r"(\d+)")[0].astype(int)
        self.assertEqual(int((numbers < 10).sum()), 6)

    def test_a_short_side_gives_its_share_to_the_other(self):
        self.assertEqual(len(audit.fidelity_sample(self.pairs(), NOTHING_EXCLUDED, rows=38, seed=0)), 38)
        self.assertEqual(len(audit.fidelity_sample(self.pairs(), NOTHING_EXCLUDED, rows=500, seed=0)), 40)

    def test_the_same_pair_is_read_once(self):
        table = pd.concat([self.pairs()] * 3, ignore_index=True)
        self.assertEqual(len(audit.fidelity_sample(table, NOTHING_EXCLUDED, rows=500, seed=0)), 40)

    def excluded(self, stranded=20, quiet=20):
        rows = [{"study_id": f"S{i}", "structure": "aortic",
                 "reason": "stranded_text" if i < stranded else "doppler_only", "dropped_parts": 1,
                 "stranded_parts": int(i < stranded), "source": f"Parti {i}."} for i in range(stranded + quiet)]
        return pd.DataFrame(rows, columns=builder.EXCLUSION_COLUMNS)

    def test_a_quarter_are_cells_removed_in_full_half_of_them_with_unexplained_text(self):
        sheet = audit.fidelity_sample(self.pairs(), self.excluded(), rows=16, seed=0)
        gone = sheet[sheet.caption == ""]
        self.assertEqual(len(sheet), 16)
        self.assertEqual(len(gone), 4)
        self.assertEqual(int((gone.source.str.extract(r"(\d+)")[0].astype(int) < 20).sum()), 2)
        self.assertEqual(int(sheet.source.str.startswith("Source").sum()), 12)

    def test_a_short_ledger_gives_its_share_to_the_captions(self):
        sheet = audit.fidelity_sample(self.pairs(), self.excluded(1, 0), rows=16, seed=0)
        self.assertEqual(len(sheet), 16)
        self.assertEqual(int((sheet.caption == "").sum()), 1)

    def test_a_source_removed_in_many_reports_is_read_once(self):
        ledger = pd.concat([self.excluded(2, 0)] * 3, ignore_index=True)
        self.assertEqual(int((audit.fidelity_sample(self.pairs(), ledger, rows=500, seed=0).caption == "").sum()), 2)

    def test_what_was_removed_in_full_is_counted_on_its_own(self):
        sheet = pd.DataFrame({"reader_fidelity": ["faithful", "changed", "changed", "unclear"],
                              "caption": ["Texte.", "", " ", "Texte."]})
        self.assertEqual(audit.fidelity_counts(sheet)["removed_in_full"],
                         {"pairs_read": 2, "faithful": 0, "changed": 2, "unclear": 0})

    def test_counts_and_an_unfinished_sheet(self):
        sheet = pd.DataFrame({"reader_fidelity": ["faithful", "Faithful ", "changed", "unclear"]})
        self.assertEqual(audit.fidelity_counts(sheet), {"pairs_read": 4, "faithful": 2, "changed": 1, "unclear": 1})
        with self.assertRaisesRegex(SystemExit, "1 reader_fidelity"):
            audit.fidelity_counts(pd.DataFrame({"reader_fidelity": ["faithful", ""]}))


class TestTrainableRows(AuditCase):

    def test_only_rows_training_can_read_are_sampled(self):
        table = audit.read_captions(self.path)
        clips = pd.read_csv(self.clips, dtype=str)
        eligible = audit.split_eligibility(table, clips, ROUTING, 12, 0, "train")
        self.assertEqual(len(audit.trainable(table, eligible)), len(table))
        seen = clips.assign(view="SUBCOSTAL")                 # shows neither the LV nor the pericardium here
        eligible = audit.split_eligibility(table, seen, ROUTING, 12, 0, "train")
        expected = [s for s in table.structure if "SUBCOSTAL" in ROUTING.structure(s).views]
        self.assertEqual(list(audit.trainable(table, eligible).structure), expected)

    def test_a_blank_or_misspelled_view_is_refused(self):
        table = audit.read_captions(self.path)
        for view in ("", "Subcostal"):
            with self.subTest(view), self.assertRaisesRegex(SystemExit, "does not know"):
                clips = pd.read_csv(self.clips, dtype=str).assign(view=view)
                audit.split_eligibility(table, clips, ROUTING, 12, 0, "train")


class TestAuditIds(unittest.TestCase):

    def test_an_id_belongs_to_one_structure_and_one_text(self):
        ids = {audit.audit_id(*pair) for pair in [("lv", "un"), ("lv", "deux"), ("rv", "un")]}
        self.assertEqual(len(ids), 3)
        self.assertEqual(audit.audit_id("lv", "un"), audit.audit_id("lv", "un"))


class TestSample(AuditCase):

    def setUp(self):
        self.captions = audit.read_captions(self.path)

    def test_the_sheet_is_blind_and_lists_each_text_once(self):
        sheet, key = audit.draw_sample(self.captions)
        self.assertEqual(list(sheet.columns), ["audit_id", "structure", "caption", "reader_flag", "reader_note"])
        self.assertFalse(sheet.duplicated(["structure", "caption"]).any())
        self.assertEqual(set(sheet.audit_id), set(key.audit_id))
        self.assertIn("normality", key.columns)

    def test_it_is_seeded(self):
        first, second = audit.draw_sample(self.captions, seed=1), audit.draw_sample(self.captions, seed=1)
        pd.testing.assert_frame_equal(first[1], second[1])
        self.assertFalse(first[1].equals(audit.draw_sample(self.captions, seed=2)[1]))

    def test_quotas_favour_predicted_normals_and_reach_rare_wording(self):
        _, key = audit.draw_sample(self.captions, normal=12, other=2)
        peri = key[key.structure == "pericardium"]
        counts = peri.normality.value_counts()
        self.assertEqual((counts["normal"], counts["abnormal"], counts["unknown"]), (12, 2, 2))
        normal = peri[peri.normality == "normal"]
        self.assertEqual(set(normal.wording), {"common", "rare"})
        self.assertEqual(normal[normal.wording == "rare"].stratum_rows.iloc[0], 6)
        self.assertGreaterEqual((normal.wording == "rare").sum(), 3)       # far above its share of the rows

    def test_conflicts_and_values_near_a_cutoff_are_their_own_strata(self):
        captions = self.captions.copy()
        near = captions.index[(captions.structure == "lv") & (captions.conflicts == "[]")][:3]
        captions.loc[near, "boundary"] = True
        _, key = audit.draw_sample(captions, normal=2, other=8)
        self.assertTrue(key[(key.structure == "lv")].conflict.any())
        self.assertTrue(key[(key.structure == "lv")].boundary.any())

    def test_a_small_group_gives_what_it_has(self):
        small = self.captions[self.captions.caption == "Clause jamais vue."]
        _, key = audit.draw_sample(small, normal=12, other=12)
        self.assertEqual(len(key), 4)
        self.assertEqual(key.audit_id.nunique(), 1)                        # one text, read once, four sampled rows

    def test_nothing_to_sample(self):
        with self.assertRaises(SystemExit):
            audit.draw_sample(self.captions.iloc[0:0])


class TestScore(unittest.TestCase):

    def key(self, rows):
        """rows: (audit_id, flag, wording, stratum_rows, stratum_sampled)."""
        return pd.DataFrame([{"audit_id": a, "structure": "lv", "normality": f, "evidence": "text", "conflict": False,
                              "boundary": False, "wording": w, "caption": a, "policy_digest": "d", "stratum_rows": n,
                              "stratum_sampled": k, "group_rows": 1000} for a, f, w, n, k in rows])

    def sheet(self, answers):
        return pd.DataFrame([{"audit_id": a, "structure": "lv", "caption": a, "reader_flag": r}
                             for a, r in answers.items()])

    def test_a_sheet_scored_against_another_keys_texts_is_refused(self):
        key = self.key([("A1", "normal", "common", 10, 1), ("A2", "normal", "common", 10, 1)])
        swapped = self.sheet({"A1": "normal", "A2": "normal"}).assign(caption=["A2", "A1"])
        with self.assertRaisesRegex(SystemExit, "2 audit ids hold another text"):
            audit.score(key, swapped)
        with self.assertRaisesRegex(SystemExit, "no \\['caption'\\] column"):
            audit.score(key, self.sheet({"A1": "normal", "A2": "normal"}).drop(columns=["caption"]))

    def test_a_refusal_does_not_print_what_the_reader_typed(self):
        key = self.key([("A1", "normal", "common", 10, 1)])
        with self.assertRaises(SystemExit) as stopped:
            audit.score(key, self.sheet({"A1": "épanchement modéré"}))
        self.assertNotIn("épanchement", str(stopped.exception))

    def test_weights_bring_the_rates_back_to_the_rows(self):
        key = self.key([("A1", "normal", "common", 900, 2), ("A2", "normal", "common", 900, 2),
                        ("A3", "normal", "rare", 100, 2), ("A4", "normal", "rare", 100, 2)])
        report = audit.score(key, self.sheet({"A1": "normal", "A2": "normal", "A3": "normal", "A4": "abnormal"}))
        normal = report["structures"]["lv"]["predicted_normal"]
        self.assertEqual((normal["captions_read"], normal["false_normal"]), (4, 1))
        precision = normal["precision_unclear_counted_as_error"]
        self.assertAlmostEqual(precision["estimate"], 0.95)                # one rare error in four, but rare is 10 %
        self.assertEqual((precision["rows_in_sampled_strata"], precision["rows_in_group"]), (1000, 1000))
        self.assertLess(precision["effective_n"], 4)
        self.assertEqual(report["structures"]["lv"]["false_normal_by_wording"], {"common": 0, "rare": 1})

    def test_unclear_is_never_counted_as_right(self):
        key = self.key([("A1", "normal", "common", 10, 2), ("A2", "normal", "common", 10, 2)])
        report = audit.score(key, self.sheet({"A1": "normal", "A2": "unclear"}))
        normal = report["structures"]["lv"]["predicted_normal"]
        self.assertEqual((normal["false_normal"], normal["unclear"]), (0, 1))
        self.assertEqual(normal["precision_unclear_counted_as_error"]["estimate"], 0.5)

    def test_other_flags_are_reported_apart(self):
        key = self.key([("A1", "abnormal", "common", 10, 1), ("A2", "unknown", "common", 10, 1)])
        report = audit.score(key, self.sheet({"A1": "normal", "A2": "normal"}))
        lv = report["structures"]["lv"]
        self.assertEqual(lv["predicted_abnormal"]["false_abnormal"], 1)
        self.assertEqual(lv["predicted_unknown"]["readers_said"], {"normal": 1, "abnormal": 0, "unclear": 0})
        self.assertIsNone(lv["predicted_normal"]["precision_unclear_counted_as_error"]["estimate"])
        self.assertNotIn("accuracy", json.dumps(report))

    def test_zero_errors_still_has_an_upper_limit_on_them(self):
        key = self.key([(f"A{i}", "normal", "common", 12, 12) for i in range(12)])
        report = audit.score(key, self.sheet({f"A{i}": "normal" for i in range(12)}))
        precision = report["structures"]["lv"]["predicted_normal"]["precision_unclear_counted_as_error"]
        self.assertEqual(precision["estimate"], 1.0)
        self.assertLess(precision["interval_95"][0], 0.8)                  # 12 of 12 is far from proving 100 %

    def test_an_unfinished_or_mismatched_sheet_is_refused(self):
        key = self.key([("A1", "normal", "common", 10, 2), ("A2", "normal", "common", 10, 2)])
        for sheet, message in [(self.sheet({"A1": "normal", "A2": ""}), "no reader_flag"),
                               (self.sheet({"A1": "normal", "A2": "fine"}), "not one of"),
                               (self.sheet({"A1": "normal"}), "same audit ids"),
                               (self.sheet({"A1": "normal", "A2": "normal", "A3": "normal"}), "same audit ids")]:
            with self.subTest(message), self.assertRaisesRegex(SystemExit, message):
                audit.score(key, sheet)
        mixed = key.assign(policy_digest=["d", "e"])
        with self.assertRaisesRegex(SystemExit, "several policies"):
            audit.score(mixed, self.sheet({"A1": "normal", "A2": "normal"}))


class TestLedger(AuditCase):

    def test_what_became_of_every_cell_adds_up_to_the_split(self):
        clips = pd.read_csv(self.clips, dtype={"study_id": str, "clip_id": str})
        studies = self.table.study_id.nunique()
        report = audit.coverage(self.table, ROUTING, POLICY, clips, exclusions=self.ledger, linked=studies - 1)
        mitral = report["structures"]["mitral"]["sources"]
        self.assertEqual(mitral, {"no_report": 1, "empty_cell": studies - 5, "with_caption": 0,
                                  "removed_doppler_only": 2, "removed_stranded_text": 2, "removed_no_words": 0})
        for structure in ROUTING.trained:
            self.assertEqual(sum(report["structures"][structure.name]["sources"].values()), studies)

    def test_a_ledger_that_follows_from_its_sources_has_no_problems(self):
        self.assertEqual(audit.exclusion_problems(self.ledger, self.table), {})

    def test_a_ledger_that_does_not_is_found(self):
        ledger, first = self.ledger, self.ledger.iloc[:1]
        cases = {"excluded_but_kept_by_these_rules": first.assign(source="Valve fine."),
                 "ledger_differs_from_source": first.assign(reason="doppler_only"),
                 "study_and_structure_twice": pd.concat([first, first])}
        for problem, table in cases.items():
            with self.subTest(problem):
                self.assertIn(problem, audit.exclusion_problems(table, self.table))
        self.assertIn("ledger_differs_from_source",
                      audit.exclusion_problems(first.assign(stranded_parts=0), self.table))
        kept = self.table.iloc[0]
        captioned = ledger.iloc[:1].assign(structure=kept.structure, study_id=kept.study_id)
        self.assertIn("excluded_and_captioned", audit.exclusion_problems(captioned, self.table))
        self.assertEqual(audit.exclusion_problems(ledger.drop(columns=["reason"]), self.table),
                         {"ledger_columns_missing": 1})


class TestCommandLine(AuditCase):

    def run_audit(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            audit.main(list(args))
        return out.getvalue()

    def test_coverage_sample_and_score(self):
        out = os.path.join(self.root, "audit")
        self.run_audit("coverage", *self.inputs(), "--out-dir", out, "--min-count", "2")
        with open(os.path.join(out, "coverage.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["rows"], len(self.table))
        self.assertTrue(os.path.exists(os.path.join(out, "unrecognized_wording.csv")))
        self.run_audit("sample", *self.inputs(), "--out-dir", out)
        sheet = audit.read_sheet(os.path.join(out, "audit_sheet.csv"))
        filled = os.path.join(self.root, "filled.csv")
        sheet.assign(reader_flag="normal").to_csv(filled, index=False)
        self.run_audit("score", "--key", os.path.join(out, "audit_key.csv"), "--sheet", filled, "--out-dir", out)
        with open(os.path.join(out, "audit_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["policy_digest"], POLICY.digest)           # what an approval must quote

    def inputs(self, path=None):
        return "--captions", path or self.path, "--clips", self.clips

    def test_the_fidelity_sheet_reaches_what_was_removed_in_full(self):
        out = os.path.join(self.root, "ledger_sheet")
        self.run_audit("sample", *self.inputs(), "--removed", "8", "--out-dir", out)
        sheet = audit.read_sheet(os.path.join(out, "fidelity_sheet.csv"))
        self.assertEqual(sorted(sheet[sheet.caption == ""].source),
                         ["Pas de sténose mitrale.", "Sténose mitrale calcifiée probable."])
        self.run_audit("coverage", *self.inputs(), "--out-dir", out)
        with open(os.path.join(out, "coverage.json"), encoding="utf-8") as f:
            sources = json.load(f)["structures"]["mitral"]["sources"]
        self.assertEqual((sources["removed_doppler_only"], sources["removed_stranded_text"]), (2, 2))

    def test_a_cell_no_clip_of_its_study_can_reach_is_not_read(self):
        out = os.path.join(self.root, "no_clip_sheet")
        extra = self.ledger.iloc[:1].assign(structure="aortic", source="Sténose aortique calcifiée probable.")
        path = self.seal(self.table, "no_clip", pd.concat([self.ledger, extra], ignore_index=True))
        self.run_audit("sample", *self.inputs(path), "--removed", "40", "--out-dir", out)
        sheet = audit.read_sheet(os.path.join(out, "fidelity_sheet.csv"))
        self.assertNotIn("aortic", set(sheet.structure))

    def test_a_build_with_nothing_removed_in_full_still_samples(self):
        out = os.path.join(self.root, "empty_ledger_sheet")
        path = self.seal(self.table, "empty_ledger", self.ledger.iloc[:0])
        self.run_audit("sample", *self.inputs(path), "--out-dir", out)
        self.assertFalse((audit.read_sheet(os.path.join(out, "fidelity_sheet.csv")).caption == "").any())

    def test_a_ledger_that_is_not_this_builds_is_refused(self):
        path = self.seal(self.table, "edited_ledger")
        ledger = os.path.join(os.path.dirname(path), "exclusions.csv")
        with open(ledger, "a", encoding="utf-8") as f:
            f.write("S000,mitral,doppler_only,1,0,Pas de sténose mitrale.\n")
        gone = self.seal(self.table, "no_ledger")
        os.remove(os.path.join(os.path.dirname(gone), "exclusions.csv"))
        for table in (path, gone):
            for command in ("coverage", "sample"):
                with self.subTest(command), self.assertRaisesRegex(SystemExit, "exclusion ledger of this build"):
                    self.run_audit(command, *self.inputs(table), "--out-dir", os.path.join(self.root, "refused"))

    def test_a_sealed_ledger_these_rules_would_not_write_is_refused(self):
        path = self.seal(self.table, "wrong_ledger", self.ledger.assign(source="Valve fine."))
        with self.assertRaisesRegex(SystemExit, "exclusion ledger does not match"):
            self.run_audit("coverage", *self.inputs(path), "--out-dir", os.path.join(self.root, "refused"))

    def test_a_clip_study_id_that_is_not_plain_is_refused(self):
        clips = pd.read_csv(self.clips, dtype=str)
        quoted = os.path.join(self.root, "quoted.csv")
        clips.assign(study_id="'" + clips.study_id + "'").to_csv(quoted, index=False)
        with self.assertRaisesRegex(SystemExit, "study id"):
            audit.split_eligibility(self.table, pd.read_csv(quoted, dtype=str), ROUTING, 2, 0, "train")

    def test_coverage_and_sample_refuse_a_table_that_is_not_this_builds(self):
        table = audit.read_captions(self.path)
        cases = {"Rebuild": self.seal(table.assign(policy_digest="older"), "stale"),
                 "another routing": self.seal(table, "routing", routing_sha256="other"),
                 "another clip table": self.seal(table, "clips", clips_sha256="other"),
                 "another split": self.seal(table, "split", split="val")}
        edited = self.seal(table, "edited")
        with open(edited, "a", encoding="utf-8") as f:
            f.write("\n")
        cases["changed since"] = edited
        for command in ("coverage", "sample"):
            for message, path in cases.items():
                with self.subTest(command, message=message), self.assertRaisesRegex(SystemExit, message):
                    self.run_audit(command, *self.inputs(path), "--out-dir", os.path.join(self.root, "refused"))

    def test_a_table_without_its_build_record_is_refused(self):
        lone = os.path.join(self.root, "lone")
        os.makedirs(lone)
        path = os.path.join(lone, "captions.csv")
        self.table.to_csv(path, index=False)
        with self.assertRaisesRegex(SystemExit, "No build_info.json"):
            self.run_audit("sample", *self.inputs(path), "--out-dir", os.path.join(self.root, "refused"))

    def test_a_new_sample_never_replaces_a_key(self):
        out = os.path.join(self.root, "once")
        self.run_audit("sample", *self.inputs(), "--out-dir", out)
        with self.assertRaisesRegex(SystemExit, "already holds an audit key"):
            self.run_audit("sample", *self.inputs(), "--seed", "1", "--out-dir", out)

    def test_ids_belong_to_the_text_so_a_sheet_never_meets_another_samples_key(self):
        first, second = os.path.join(self.root, "first"), os.path.join(self.root, "second")
        self.run_audit("sample", *self.inputs(), "--seed", "0", "--normal", "2", "--other", "1", "--out-dir", first)
        self.run_audit("sample", *self.inputs(), "--seed", "1", "--normal", "2", "--other", "1", "--out-dir", second)
        sheets = [audit.read_sheet(os.path.join(f, "audit_sheet.csv")) for f in (first, second)]
        ids = [dict(zip(s.audit_id, s.caption)) for s in sheets]
        for shared in set(ids[0]) & set(ids[1]):
            self.assertEqual(ids[0][shared], ids[1][shared])

    def test_a_key_of_other_rules_is_refused(self):
        out = os.path.join(self.root, "older_key")
        self.run_audit("sample", *self.inputs(), "--out-dir", out)
        key = os.path.join(out, "audit_key.csv")
        pd.read_csv(key, keep_default_na=False).assign(policy_digest="older").to_csv(key, index=False)
        sheet = os.path.join(out, "audit_sheet.csv")
        with self.assertRaisesRegex(SystemExit, "current rules"):
            self.run_audit("score", "--key", key, "--sheet", sheet, "--out-dir", out)

    def test_a_sheet_saved_by_a_french_spreadsheet_is_read(self):
        path = os.path.join(self.root, "semicolons.csv")
        table = pd.DataFrame({"audit_id": ["A1"], "caption": ["Épanchement, léger"], "reader_flag": ["normal"]})
        table.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
        self.assertEqual(audit.read_sheet(path).to_dict("records"), table.to_dict("records"))
        table.to_csv(path, index=False, sep=";", encoding="cp1252")
        with self.assertRaisesRegex(SystemExit, "not UTF-8"):
            audit.read_sheet(path)

    def test_output_inside_the_repository_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_audit("sample", *self.inputs(), "--out-dir", os.path.join(REPO, "audit_here"))


if __name__ == "__main__":
    unittest.main()
