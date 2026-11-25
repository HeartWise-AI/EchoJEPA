"""The view labels export: each clip's label status and answer from a synthetic export, the checks that refuse an export
of another project, form or upload, the kept export, and the summary. No request is made."""

import csv
import io
import json
import shutil
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from tools.labelbox.export_view_labels import (
    COLUMNS,
    TARGET,
    ExportError,
    build_rows,
    export_project,
    form_changes,
    form_views,
    kept_export_path,
    kept_form_path,
    labels_csv,
    main,
    read_picked,
    read_picked_lists,
    summary_lines,
)
from tools.labelbox import view_review
from tools.labelbox.pilot_upload import VIEW_KEY_PREFIX, LabelboxUploadApi, Workspace

PICKED_FIELDS = ["patient_id", "study_id", "view", "video_path", "dicom_path", "vendor", "model", "acq_category",
                 "mode_source", "n_frames", "form", "global_key", "video_sha256"]
EMAIL = "reviewer@example.invalid"
COMMENT = "Text of a reviewer comment"


def key(char):
    return VIEW_KEY_PREFIX + char * 24


def view_answer(value):
    # As Labelbox exports a radio answer (seen in real exports, 2026-10-20): display text as `name`, id as `value`.
    return {"feature_id": "f", "feature_schema_id": "s", "name": "Which view does this clip show?", "value": "view",
            "radio_answer": {"feature_id": "f", "feature_schema_id": "s", "name": f"Label of {value}",
                             "value": value, "classifications": []}}


def comment_answer(text):
    return {"feature_id": "f", "feature_schema_id": "s", "name": "Comments", "value": "comments",
            "text_answer": {"content": text, "classifications": []}}


def label(view=None, *, skipped=False, created_at="2026-11-10T15:00:00.000+00:00", comment=None):
    classifications = ([view_answer(view)] if view else []) + ([comment_answer(comment)] if comment else [])
    return {"id": "label", "label_kind": "Video", "version": "1.0.0",
            "label_details": {"created_at": created_at, "updated_at": created_at,
                              "content_last_updated_at": created_at, "created_by": EMAIL, "reviews": []},
            "performance_details": {"seconds_to_create": 9, "seconds_to_review": 0, "skipped": skipped},
            "annotations": {"frames": {}, "segments": {}, "key_frame_feature_map": {},
                            "classifications": classifications}}


def exported(global_key, labels=(), workflow="DONE", project=TARGET.project_name):
    details = {"workflow_status": workflow, "task_name": "Done", "workflow_history": [{"created_by": EMAIL}]}
    return {"data_row": {"id": "row", "global_key": global_key, "external_id": global_key, "row_data": "https://x"},
            "projects": {"project-id": {"name": project, "labels": list(labels), "project_details": details}}}


# Each clip, its predicted view, its exported row and what the CSV must say: (human_view, label_status, n_labels).
CASES = {
    "a": ("A4C", [label("A4C")], "DONE", ("A4C", "done", 1)),
    "b": ("PLAX_ZOOM", [label("PLAX_ZOOM_AV")], "IN_REVIEW", ("PLAX_ZOOM_AV", "done", 1)),
    "c": ("PSAX", [label("unsure")], "DONE", ("UNSURE", "done", 1)),
    "d": ("PSAX", [], "TO_LABEL", ("", "unlabelled", 0)),
    "e": ("A2C", [label(skipped=True)], "IN_REVIEW", ("", "skipped", 0)),
    "f": ("A3C", [label("A3C")], "IN_REWORK", ("", "in_rework", 1)),
    "1": ("A5C", [label("A5C")], "TO_LABEL", ("", "in_rework", 1)),
    "2": ("PSAX", [label("PSAX_MID"), label("PSAX_BASAL", created_at="2026-11-11T02:00:00Z")], "DONE",
          ("", "conflict", 2)),
    "3": ("PSAX", [label("PSAX_MID"), label("PSAX_MID")], "DONE", ("PSAX_MID", "done", 2)),
    "4": ("OTHER", [label(skipped=True), label("OTHER", comment=COMMENT)], "DONE", ("OTHER", "done", 1)),
}


def write_picked(path, cases=CASES):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PICKED_FIELDS)
        writer.writeheader()
        for char, (view, *_rest) in cases.items():
            writer.writerow({"patient_id": f"patient-{char}", "study_id": f"study-{char}", "view": view,
                             "video_path": f"/videos/{char}.mp4", "dicom_path": f"/dicom/{char}.dcm", "vendor": "v",
                             "model": "m", "acq_category": "bmode", "mode_source": "s", "n_frames": "40",
                             "form": "view_review", "global_key": key(char), "video_sha256": "0" * 64})
    return path


def export_of(cases=CASES):
    return [exported(key(char), labels, workflow) for char, (_view, labels, workflow, _want) in cases.items()]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.picked_path = write_picked(self.folder / "picked_clips.csv")
        self.picked = read_picked(self.picked_path)

    def tearDown(self):
        shutil.rmtree(self.folder)


class StatusTests(TempDirTest):
    def test_each_clip_gets_its_answer_and_status(self):
        rows = {row["global_key"]: row for row in build_rows(export_of(), self.picked)}
        for char, (view, _labels, _workflow, want) in CASES.items():
            row = rows[key(char)]
            self.assertEqual((row["human_view"], row["label_status"], row["n_labels"]), want, char)
            self.assertEqual(row["predicted_view"], view)

    def test_labelled_at_is_the_latest_counted_label_in_utc(self):
        rows = {row["global_key"]: row for row in build_rows(export_of(), self.picked)}
        self.assertEqual(rows[key("a")]["labelled_at"], "2026-11-10T15:00:00+00:00")
        self.assertEqual(rows[key("2")]["labelled_at"], "2026-11-11T02:00:00+00:00")
        self.assertEqual(rows[key("e")]["labelled_at"], "2026-11-10T15:00:00+00:00")  # the skip's.
        self.assertEqual(rows[key("d")]["labelled_at"], "")

    def test_workflow_status_is_read_in_either_spelling(self):
        cases = {"a": ("A4C", [label("A4C")], "InRework", None)}
        write_picked(self.picked_path, cases)
        row, = build_rows(export_of(cases), read_picked(self.picked_path))
        self.assertEqual(row["label_status"], "in_rework")

    def test_rows_follow_the_picked_order_and_carry_its_clip(self):
        rows = build_rows(list(reversed(export_of())), self.picked)
        self.assertEqual([row["global_key"] for row in rows], list(self.picked))
        self.assertEqual((rows[0]["patient_id"], rows[0]["study_id"], rows[0]["video_path"], rows[0]["acq_category"]),
                         ("patient-a", "study-a", "/videos/a.mp4", "bmode"))


class RefusalTests(TempDirTest):
    def refused(self, rows, message):
        with self.assertRaisesRegex(ExportError, message):
            build_rows(rows, self.picked)

    def test_a_clip_picked_clips_does_not_list(self):
        self.refused([*export_of(), exported(key("9"))], "do not list")

    def test_a_picked_clip_missing_from_the_export(self):
        self.refused(export_of()[1:], "lacks 1 picked clips")

    def test_a_clip_exported_twice(self):
        self.refused([*export_of(), exported(key("a"))], "twice")

    def test_another_project(self):
        self.refused([exported(key("a"), project="RADAR Caption Pilot"), *export_of()[1:]], "not 'RADAR View")

    def test_a_question_of_another_form(self):
        other = label("A4C")
        other["annotations"]["classifications"].append({"name": "What is this item?", "value": "input_type",
                                                        "radio_answer": {"name": "Echo", "value": "cine_loop"}})
        self.refused([exported(key("a"), [other]), *export_of()[1:]], "does not ask")

    def test_a_view_the_form_does_not_offer(self):
        self.refused([exported(key("a"), [label("PSAX")]), *export_of()[1:]], "does not offer")

    def test_an_export_without_workflow_status(self):
        row = exported(key("a"), [label("A4C")])
        del row["projects"]["project-id"]["project_details"]
        self.refused([row, *export_of()[1:]], "no workflow status")

    def test_an_unknown_workflow_status(self):
        self.refused([exported(key("a"), [label("A4C")], workflow="ARCHIVED"), *export_of()[1:]], "unknown workflow")

    def test_a_label_that_neither_skips_nor_answers(self):
        self.refused([exported(key("a"), [label()]), *export_of()[1:]], "neither skipped nor answers")

    def test_a_label_without_skip_flag(self):
        unflagged = label("A4C")
        del unflagged["performance_details"]
        self.refused([exported(key("a"), [unflagged]), *export_of()[1:]], "performance details")

    def test_picked_clips_of_another_kind_of_key(self):
        with open(self.picked_path, encoding="utf-8") as f:
            text = f.read()
        self.picked_path.write_text(text.replace(VIEW_KEY_PREFIX, "radar-pilot-"), encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "view relabelling key"):
            read_picked(self.picked_path)


class BatchTests(TempDirTest):
    """The project holds several batches, each with its own picked_clips.csv: the export is checked against their
    union and each row names its batch."""

    FIRST = {char: CASES[char] for char in "abcdef"}
    SECOND = {char: CASES[char] for char in "1234"}

    def setUp(self):
        super().setUp()
        self.first = write_picked(self.folder / "first.csv", self.FIRST)
        self.second = write_picked(self.folder / "second.csv", self.SECOND)

    def test_rows_name_their_batch_in_list_order(self):
        picked = read_picked_lists([self.first, self.second])
        rows = build_rows(list(reversed(export_of())), picked)
        self.assertEqual([row["global_key"] for row in rows], [key(char) for char in "abcdef1234"])
        self.assertEqual([row["batch"] for row in rows], [1] * 6 + [2] * 4)
        rows = {row["global_key"]: row for row in rows}
        for char, (_view, _labels, _workflow, want) in CASES.items():
            row = rows[key(char)]
            self.assertEqual((row["human_view"], row["label_status"], row["n_labels"]), want, char)

    def test_one_list_is_batch_one(self):
        self.assertEqual({clip["batch"] for clip in read_picked(self.picked_path).values()}, {1})
        self.assertEqual({row["batch"] for row in build_rows(export_of(), self.picked)}, {1})

    def test_an_export_of_both_batches_needs_both_lists(self):
        with self.assertRaisesRegex(ExportError, "do not list.*missing from --picked"):
            build_rows(export_of(), read_picked_lists([self.first]))

    def test_every_listed_clip_of_a_later_batch_must_be_exported(self):
        rows = [row for row in export_of() if row["data_row"]["global_key"] != key("3")]
        with self.assertRaisesRegex(ExportError, "lacks 1 picked clips"):
            build_rows(rows, read_picked_lists([self.first, self.second]))

    def test_a_clip_in_two_lists_is_refused(self):
        overlap = write_picked(self.folder / "overlap.csv", {"a": CASES["a"], **self.SECOND})
        with self.assertRaisesRegex(ExportError, "1 clips of list 2 .* also in list 1"):
            read_picked_lists([self.first, overlap])
        with self.assertRaisesRegex(ExportError, "also in list 1"):
            read_picked_lists([self.first, self.first])
        with self.assertRaisesRegex(ExportError, "at least one"):
            read_picked_lists([])

    def test_a_list_with_its_own_batch_column_is_refused(self):
        with open(self.first, encoding="utf-8") as f:
            text = f.read().splitlines()
        text = [text[0] + ",batch"] + [line + ",7" for line in text[1:]]
        self.first.write_text("\n".join(text) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "has a `batch` column"):
            read_picked(self.first)

    def run_main(self, *args, api=None, environ=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args), environ={"LABELBOX_API_KEY": "key"} if environ is None else environ,
                        api_factory=lambda _key: api)
        return code, out.getvalue(), err.getvalue()

    def test_main_takes_several_lists_live_and_from_a_kept_export(self):
        out = self.folder / "labels.csv"
        code, printed, err = self.run_main("--picked", str(self.first), str(self.second), "--out", str(out),
                                           api=FakeApi(export_of()))
        self.assertEqual(code, 0, err)
        rows = list(csv.DictReader(io.StringIO(out.read_text())))
        self.assertEqual(tuple(rows[0]), COLUMNS)
        self.assertEqual([row["batch"] for row in rows], ["1"] * 6 + ["2"] * 4)
        self.assertEqual([row["workflow_status"] for row in rows],
                         ["DONE", "IN_REVIEW", "DONE", "TO_LABEL", "IN_REVIEW", "IN_REWORK",
                          "TO_LABEL", "DONE", "DONE", "DONE"])
        self.assertIn(f"batch 1: {self.first} (6 clips)", printed)
        self.assertIn(f"batch 2: {self.second} (4 clips)", printed)
        self.assertIn("  batch 1: 6 clips, done 3, unlabelled 1, skipped 1, in_rework 1", printed)
        self.assertIn("  batch 2: 4 clips, done 2, in_rework 1, conflict 1", printed)
        again = self.folder / "again.csv"
        code, _, _ = self.run_main("--picked", str(self.first), str(self.second), "--out", str(again),
                                   "--from-export", str(kept_export_path(out)), environ={})
        self.assertEqual((code, again.read_text()), (0, out.read_text()))
        # The kept export holds both batches: batch 1's list alone is refused, and nothing is written.
        alone = self.folder / "alone.csv"
        code, _, err = self.run_main("--picked", str(self.first), "--out", str(alone),
                                     "--from-export", str(kept_export_path(out)), environ={})
        self.assertEqual(code, 1)
        self.assertIn("missing from --picked", err)
        self.assertFalse(alone.exists())


class CsvTests(TempDirTest):
    def test_exactly_the_contract_columns_and_nothing_about_reviewers(self):
        text = labels_csv(build_rows(export_of(), self.picked))
        reader = csv.DictReader(io.StringIO(text))
        self.assertEqual(tuple(reader.fieldnames), COLUMNS)
        rows = list(reader)
        self.assertEqual(len(rows), len(CASES))
        self.assertEqual({row["label_status"] for row in rows}, {"done", "unlabelled", "skipped", "in_rework",
                                                                 "conflict"})
        self.assertNotIn(EMAIL, text)
        self.assertNotIn(COMMENT, text)

    def test_summary_counts_and_names_the_commented_clip_without_its_text(self):
        lines = summary_lines(build_rows(export_of(), self.picked))
        text = "\n".join(lines)
        self.assertIn("label_status: done 5, unlabelled 1, skipped 1, in_rework 2, conflict 1", text)
        self.assertIn("done by workflow status: DONE 4, IN_REVIEW 1", text)
        psax = next(line for line in lines if line.strip().startswith("PSAX "))
        self.assertEqual(psax.split()[:6], ["PSAX", "4", "2", "0", "1", "1"])  # clips, done, agree, unsure, other.
        self.assertIn("PSAX_MID 1", psax)
        a4c = next(line for line in lines if line.strip().startswith("A4C "))
        self.assertEqual(a4c.split()[:6], ["A4C", "1", "1", "1", "0", "0"])
        self.assertIn(f"  {key('4')}", lines)
        self.assertNotIn(COMMENT, text)
        self.assertNotIn(EMAIL, text)


class FakeProject:
    def __init__(self, ontology):
        self._ontology = ontology

    def ontology(self):
        return self._ontology


def edited_form():
    """The committed form as edited in Labelbox: one option added, one display text changed, as a JSON string."""
    form = view_review.build_form()
    options = form["classifications"][0]["options"]
    options.append({**options[0], "value": "NEW_VIEW", "label": "NEW_VIEW: a view added in Labelbox"})
    options[1] = {**options[1], "label": "edited display text"}
    form["projectInstructions"] = "edited instructions"
    return json.dumps(form)


class FakeApi:
    def __init__(self, rows, connected="form", normalized=None):
        self.rows, self.asked = rows, []
        self.form = SimpleNamespace(uid="form")
        normalized = view_review.build_form() if normalized is None else normalized
        self.project = FakeProject(SimpleNamespace(uid=connected, normalized=normalized) if connected else None)

    def ensure_workspace(self, forms, *, create, datasets_checked, form_checked):
        assert not (datasets_checked or form_checked)  # An export reads the project only, and its form as it is.
        self.asked.append(("workspace", tuple(forms), create))
        return Workspace({}, {form: self.project for form in forms}, {form: self.form for form in forms})

    def export_rows(self, project, *, project_details=False):
        self.asked.append(("export", project_details))
        return self.rows


class LiveExportTests(TempDirTest):
    def run_main(self, *args, api=None, environ=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--picked", str(self.picked_path), *args],
                        environ={"LABELBOX_API_KEY": "key"} if environ is None else environ,
                        api_factory=lambda _key: api)
        return code, out.getvalue(), err.getvalue()

    def test_exports_with_workflow_status_keeps_the_export_then_writes_the_labels(self):
        api, out = FakeApi(export_of()), self.folder / "labels" / "view_labels.csv"
        code, printed, _ = self.run_main("--out", str(out), api=api)
        self.assertEqual(code, 0)
        self.assertEqual(api.asked, [("workspace", ("view_review",), False), ("export", True)])
        kept = kept_export_path(out)
        self.assertEqual([json.loads(line) for line in kept.read_text().splitlines()], export_of())
        self.assertEqual(json.loads(kept_form_path(kept).read_text()), view_review.build_form())
        self.assertIn("form: view options as committed", printed)
        for path in (out, kept, kept_form_path(kept)):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(out.read_text(), labels_csv(build_rows(export_of(), self.picked)))
        self.assertIn("label_status: done 5", printed)

    def test_a_kept_export_is_read_again_without_a_key(self):
        api, first = FakeApi(export_of()), self.folder / "view_labels.csv"
        self.run_main("--out", str(first), api=api)
        again = self.folder / "again.csv"
        code, _, _ = self.run_main("--out", str(again), "--from-export", str(kept_export_path(first)), environ={})
        self.assertEqual(code, 0)
        self.assertEqual(again.read_text(), first.read_text())
        self.assertFalse(kept_export_path(again).exists())

    def test_never_overwrites_labels_or_an_export(self):
        api, out = FakeApi(export_of()), self.folder / "view_labels.csv"
        self.run_main("--out", str(out), api=api)
        before = out.read_bytes()
        code, _, err = self.run_main("--out", str(out), api=api)
        self.assertEqual((code, out.read_bytes()), (1, before))
        self.assertIn("give a new name", err)
        self.assertEqual(sum(step[0] == "export" for step in api.asked), 1)  # the refused run exported nothing.

    def test_refuses_an_output_in_the_repository(self):
        api = FakeApi(export_of())
        code, _, err = self.run_main("--out", str(Path(__file__).resolve().parent / "view_labels.csv"), api=api)
        self.assertEqual((code, api.asked), (1, []))
        self.assertIn("outside the repository", err)

    def test_needs_a_key_to_export(self):
        code, _, err = self.run_main("--out", str(self.folder / "view_labels.csv"), environ={})
        self.assertEqual(code, 2)
        self.assertIn("LABELBOX_API_KEY", err)

    def test_a_form_edited_in_labelbox_is_read_listed_and_kept(self):
        rows = export_of()
        rows[0]["projects"] = json.loads(json.dumps(rows[0]["projects"]).replace(
            '"value": "A4C"', '"value": "NEW_VIEW"'))
        api, out = FakeApi(rows, normalized=edited_form()), self.folder / "view_labels.csv"
        code, printed, err = self.run_main("--out", str(out), api=api)
        self.assertEqual(code, 0, err)
        self.assertIn("view options added in Labelbox: NEW_VIEW", printed)
        self.assertIn("view options with new display text", printed)
        self.assertIn("NEW_VIEW", out.read_text())
        again = self.folder / "again.csv"
        code, printed, _ = self.run_main("--out", str(again), "--from-export", str(kept_export_path(out)), environ={})
        self.assertEqual((code, again.read_text()), (0, out.read_text()))
        self.assertIn("view options added in Labelbox: NEW_VIEW", printed)

    def test_an_answer_the_form_in_labelbox_does_not_offer_is_refused(self):
        rows = export_of()
        rows[0]["projects"] = json.loads(json.dumps(rows[0]["projects"]).replace(
            '"value": "A4C"', '"value": "NOT_OFFERED"'))
        code, _, err = self.run_main("--out", str(self.folder / "view_labels.csv"),
                                     api=FakeApi(rows, normalized=edited_form()))
        self.assertEqual(code, 1)
        self.assertIn("does not offer", err)

    def test_form_views_and_changes(self):
        self.assertEqual(form_views(view_review.build_form()), view_review.view_choices())
        self.assertEqual(form_changes(view_review.view_choices()), [])
        removed = dict(list(view_review.view_choices().items())[1:])
        self.assertTrue(any(line.startswith("view options removed") for line in form_changes(removed)))
        reordered = dict(reversed(list(view_review.view_choices().items())))
        self.assertEqual(form_changes(reordered), ["view options reordered in Labelbox"])
        with self.assertRaisesRegex(ExportError, "single `view` question"):
            form_views({"classifications": []})

    def test_refuses_a_project_with_another_form(self):
        api = FakeApi(export_of(), connected="another form")
        with self.assertRaisesRegex(ExportError, "does not use the form"):
            export_project(api)
        self.assertNotIn(("export", True), api.asked)

    def test_a_refused_export_is_kept_and_the_labels_are_not_written(self):
        api, out = FakeApi(export_of()[1:]), self.folder / "view_labels.csv"
        code, _, err = self.run_main("--out", str(out), api=api)
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())
        self.assertTrue(kept_export_path(out).exists())
        self.assertIn("--from-export", err)

    def test_the_sdk_export_asks_for_workflow_status_only_when_told(self):
        class Task:
            status = "COMPLETE"

            def wait_till_done(self):
                return None

            def has_errors(self):
                return False

            def has_result(self):
                return False

        asked = []
        project = SimpleNamespace(export=lambda params: asked.append(params) or Task())
        LabelboxUploadApi.export_rows(project)
        LabelboxUploadApi.export_rows(project, project_details=True)
        self.assertEqual([params["project_details"] for params in asked], [False, True])
        self.assertTrue(all(params["label_details"] and params["performance_details"] for params in asked))


if __name__ == "__main__":
    unittest.main()
