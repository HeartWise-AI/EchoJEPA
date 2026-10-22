import copy
import csv
import importlib.util
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.labelbox.column_review import build_candidate, build_form, prelabels
from tools.labelbox.create_pilot_project import DATASET_DESCRIPTION, PROJECT_DESCRIPTION
from tools.labelbox.pilot_upload import (
    BATCH_DELETE_WAITS,
    BATCH_PREFIX,
    CLIPS,
    COLUMN_REVIEW,
    ITEMS,
    NOTICE,
    PREDICTIONS_NAME,
    PREVIEW,
    STUDY_KEY_PREFIX,
    TEST_KEY_PREFIX,
    VIEW_TYPE_REVIEW,
    WORKSPACES,
    LabelboxUploadApi,
    StepFailed,
    UploadError,
    Workspace,
    _answers_from_label,
    ensure_upload,
    main,
    preview,
    read_plan,
    sha256,
    summarize_export,
)

REPORT = {
    "Left ventricule findings": "Dysfonction légère du ventricule gauche. // Ventricule gauche non-dilaté",
    "Mitral valve findings": "Insuffisance mitrale légère",
    "Visually Estimated EF": 48,
}
A4C = TEST_KEY_PREFIX + "a" * 24
COLOR = TEST_KEY_PREFIX + "b" * 24
OTHER = TEST_KEY_PREFIX + "c" * 24
STUDY = STUDY_KEY_PREFIX + "d" * 24
TARGETS = WORKSPACES["pilot"]  # the workspace the SDK adapter tests use; main defaults to the test workspace.


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_folder(root: Path, *, candidates=None, listed=None, other=None) -> Path:
    (root / CLIPS).mkdir()
    (root / ITEMS).mkdir()
    forms = {A4C: COLUMN_REVIEW, COLOR: COLUMN_REVIEW, OTHER: VIEW_TYPE_REVIEW}
    rows = []
    for key, form in forms.items():
        path = root / CLIPS / f"{key}.mp4"
        path.write_bytes(f"video {key}".encode())
        rows.append({"global_key": key, "form": form, "study_key": STUDY, "copy_sha256": sha256(path)})
    if candidates is None:
        candidates = [build_candidate(REPORT, "A4C", "bmode", global_key=A4C, input_kind="video"),
                      build_candidate(REPORT, "A4C", "color_flow", global_key=COLOR, input_kind="video")]
    (root / ITEMS / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
    write_csv(root / ITEMS / "view_type_items.csv", [{"global_key": key} for key in (other or [OTHER])],
              ["global_key"])
    write_csv(root / ITEMS / "global_keys.csv", listed or rows, ["global_key", "form", "study_key", "copy_sha256"])
    return root


# Exported answers as Labelbox writes them: each question and answer carries its display text as `name` and its id
# as `value` (seen in a real export of both forms, 2026-10-20).
def answer(value, children=()):
    return {"feature_id": "f", "feature_schema_id": "s", "name": f"Label of {value}", "value": value,
            "classifications": list(children)}


def radio(question, value, children=()):
    return {"feature_id": "f", "feature_schema_id": "s", "name": f"Text of {question}?", "value": question,
            "radio_answer": answer(value, children)}


def checklist(question, answers):
    return {"feature_id": "f", "feature_schema_id": "s", "name": f"Text of {question}?", "value": question,
            "checklist_answers": list(answers)}


def column_review_label(candidate):
    columns = [answer(c["concept"], [radio(c["concept"] + "__value", c["value"]),
                                     radio(c["concept"] + "__status", "yes")]) for c in candidate["columns"]]
    return [
        radio("uninterpretable", "no"), radio("view", candidate["view"]), radio("modality", candidate["modality"]),
        radio("psax_level", candidate["level"]),
        checklist("visible_targets", [answer(t) for t in candidate["suggested_targets"]]),
        checklist("review_columns", columns),
        radio("missing_information", "no"), radio("unsupported_information", "no"),
        radio("review_completed", "yes"),
    ]


def server_form():
    """The column review form as Labelbox stores it, with a schema id on every question and answer."""
    form = copy.deepcopy(build_form())

    def add_ids(questions, path="root"):
        for question_index, question in enumerate(questions):
            question["featureSchemaId"] = f"{path}.q{question_index}"
            for option_index, option in enumerate(question.get("options", [])):
                option["featureSchemaId"] = f"{path}.q{question_index}.o{option_index}"
                add_ids(option.get("options", []), option["featureSchemaId"])

    add_ids(form["classifications"])
    return form


def exported(key, classifications=None):
    row = {"data_row": {"global_key": key}, "projects": {}}
    if classifications is not None:
        row["projects"]["project"] = {"labels": [{"label_details": {"created_at": "2026-10-20T00:00:00Z"},
                                                  "annotations": {"classifications": classifications}}]}
    return row


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(self.folder)

    def test_reads_rows_and_attaches_only_the_notice_and_the_context(self):
        plan = read_plan(make_folder(self.folder))

        self.assertEqual([row.global_key for row in plan.rows_of(COLUMN_REVIEW)], [A4C, COLOR])
        self.assertEqual([row.global_key for row in plan.rows_of(VIEW_TYPE_REVIEW)], [OTHER])
        other = plan.rows_of(VIEW_TYPE_REVIEW)[0]
        self.assertEqual(other.attachments, (NOTICE,))
        a4c = plan.rows_of(COLUMN_REVIEW)[0]
        self.assertEqual([attachment["name"] for attachment in a4c.attachments],
                         ["Workflow test", "DeepECHO review v3 - original proposals"])

    def test_preview_holds_no_report_text_or_local_path(self):
        plan = read_plan(make_folder(self.folder))
        sent = json.dumps(preview(plan, TARGETS), ensure_ascii=False)

        self.assertGreater(len(plan.candidates[A4C]["columns"]), 0)
        self.assertNotIn("Dysfonction", sent)
        self.assertNotIn("Insuffisance", sent)
        self.assertNotIn("source_text", sent)
        self.assertNotIn("report_context", sent)
        self.assertNotIn(str(self.folder), sent)
        rows = preview(plan, TARGETS)["rows"]
        self.assertEqual(rows[0]["file"], f"{A4C}.mp4")
        self.assertEqual(rows[0]["prelabel_records"], len(prelabels(plan.candidates[A4C])))
        self.assertEqual(rows[2]["prelabel_records"], 0)
        self.assertTrue(all(row["batch"].startswith(BATCH_PREFIX) for row in rows))
        self.assertNotIn(STUDY, sent)

    def test_one_batch_per_study_in_each_project(self):
        plan = read_plan(make_folder(self.folder))

        self.assertEqual(list(plan.batches(COLUMN_REVIEW).values()), [[A4C, COLOR]])
        self.assertEqual(list(plan.batches(VIEW_TYPE_REVIEW).values()), [[OTHER]])

    def test_refuses_a_changed_or_missing_copy(self):
        folder = make_folder(self.folder)
        (folder / CLIPS / f"{A4C}.mp4").write_bytes(b"another video")
        with self.assertRaisesRegex(UploadError, "missing or differs"):
            read_plan(folder)
        (folder / CLIPS / f"{A4C}.mp4").unlink()
        with self.assertRaisesRegex(UploadError, "missing or differs"):
            read_plan(folder)

    def test_refuses_keys_outside_the_test_prefix(self):
        pilot_key = "radar-pilot-" + "a" * 24
        rows = [{"global_key": pilot_key, "form": COLUMN_REVIEW, "study_key": STUDY, "copy_sha256": "0" * 64}]
        folder = make_folder(self.folder, listed=rows)
        with self.assertRaisesRegex(UploadError, "workflow-test global key"):
            read_plan(folder)

    def test_refuses_a_plain_study_id(self):
        rows = [{"global_key": A4C, "form": COLUMN_REVIEW, "study_key": "1.2.840.113619.2.1", "copy_sha256": "0" * 64}]
        folder = make_folder(self.folder, listed=rows)
        with self.assertRaisesRegex(UploadError, "opaque study key"):
            read_plan(folder)

    def test_refuses_items_that_do_not_match_the_listed_forms(self):
        folder = make_folder(self.folder, candidates=[
            build_candidate(REPORT, "A4C", "bmode", global_key=A4C, input_kind="video")])
        with self.assertRaisesRegex(UploadError, "column_review items"):
            read_plan(folder)

    def test_refuses_a_folder_without_items(self):
        with self.assertRaisesRegex(UploadError, "no review items"):
            read_plan(self.folder)

    def test_one_form_alone(self):
        plan = read_plan(make_folder(self.folder)).only(VIEW_TYPE_REVIEW)

        self.assertEqual([row.global_key for row in plan.rows], [OTHER])
        self.assertEqual(plan.candidates, {})
        self.assertEqual(plan.forms(), [VIEW_TYPE_REVIEW])


class FakeApi:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.calls = []
        self.uploaded = []
        self.batches = {}
        self.prelabeled = None

    def ensure_workspace(self, forms, *, create):
        self.calls.append(("workspace", tuple(forms), create))
        return Workspace(datasets={form: f"dataset-{form}" for form in forms},
                         projects={form: f"project-{form}" for form in forms},
                         ontologies={form: f"ontology-{form}" for form in forms})

    def existing_global_keys(self, workspace, rows):
        return self.existing & {row.global_key for row in rows}

    def upload_rows(self, workspace, rows):
        self.calls.append("upload")
        self.uploaded.extend(row.global_key for row in rows)
        self.existing.update(row.global_key for row in rows)

    def ensure_batches(self, project, plan):
        self.calls.append(("batches", project))
        known = self.batches.setdefault(project, {})
        new = {name: keys for name, keys in plan.items() if name not in known}
        known.update(new)
        return len(new)

    def ensure_prelabels(self, workspace, candidates, *, refresh=False):
        self.calls.append(("prelabels", refresh))
        if self.prelabeled is not None and not refresh:
            return False
        self.prelabeled = [candidate["global_key"] for candidate in candidates]
        return True


class OrchestrationTests(unittest.TestCase):
    def test_uploads_missing_rows_then_batches_then_prelabels(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        api = FakeApi(existing={COLOR})

        result = ensure_upload(api, plan)

        self.assertEqual(api.uploaded, [A4C, OTHER])
        self.assertEqual(result.uploaded, 2)
        self.assertEqual(api.calls, [("workspace", (COLUMN_REVIEW, VIEW_TYPE_REVIEW), True), "upload",
                                     ("batches", f"project-{COLUMN_REVIEW}"),
                                     ("batches", f"project-{VIEW_TYPE_REVIEW}"), ("prelabels", True)])
        self.assertEqual(api.prelabeled, [A4C, COLOR])
        self.assertTrue(result.prelabels_imported)

    def test_second_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        api = FakeApi()
        ensure_upload(api, plan)

        result = ensure_upload(api, plan)

        self.assertEqual(result.uploaded, 0)
        self.assertEqual(result.created_batches, {COLUMN_REVIEW: 0, VIEW_TYPE_REVIEW: 0})
        self.assertFalse(result.prelabels_imported)
        self.assertEqual(api.calls[-1], ("prelabels", False))

    def test_the_view_and_type_form_alone_needs_no_prelabels(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder))).only(VIEW_TYPE_REVIEW)
        api = FakeApi()

        result = ensure_upload(api, plan)

        self.assertEqual(api.uploaded, [OTHER])
        self.assertFalse(result.prelabels_imported)
        self.assertNotIn(("prelabels", True), api.calls)


class FakeProject:
    def __init__(self, name, description=PROJECT_DESCRIPTION, media_type="video", ontology=None, labels=0):
        self.name, self.description, self.media_type = name, description, media_type
        self.uid = f"project-{name}"
        self._ontology = ontology
        self.labels = labels
        self.batch_list = []
        self.import_list = []

    def ontology(self):
        return self._ontology

    def connect_ontology(self, ontology):
        self._ontology = ontology

    def get_label_count(self):
        return self.labels

    def batches(self):
        return list(self.batch_list)

    def get_mal_prediction_imports(self):
        return list(self.import_list)


def fake_dataset(name, description=DATASET_DESCRIPTION):
    return SimpleNamespace(name=name, description=description, uid=f"dataset-{name}", iam_integration=lambda: None)


class FakeClient:
    """A workspace an administrator bootstrapped: each form's project and dataset exist, no form yet."""

    def __init__(self, projects=None, datasets=None, ontologies=()):
        self.projects = projects if projects is not None else [FakeProject(t.project_name) for t in TARGETS.values()]
        self.datasets = datasets if datasets is not None else [fake_dataset(t.dataset_name) for t in TARGETS.values()]
        self.ontologies = list(ontologies)
        self.created = []
        self.rows = {}

    def get_datasets(self, where=None):
        return list(self.datasets)

    def get_projects(self, where=None):
        return list(self.projects)

    def get_ontologies(self, name_contains):
        return [ontology for ontology in self.ontologies if name_contains in ontology.name]

    def create_ontology(self, name, form, media_type):
        self.created.append(("form", name))
        ontology = SimpleNamespace(name=name, uid=f"ontology-{name}", normalized=copy.deepcopy(form))
        self.ontologies.append(ontology)
        return ontology

    def create_project(self, **_):
        raise AssertionError("created a project")

    def create_dataset(self, **_):
        raise AssertionError("created a dataset")

    def project(self, form):
        return next(p for p in self.projects if p.name == TARGETS[form].project_name)

    def dataset(self, form):
        return next(d for d in self.datasets if d.name == TARGETS[form].dataset_name)


class OfflineApi(LabelboxUploadApi):
    """The SDK adapter with data-row lookups served by the fake client, so no SDK is needed."""

    def _data_row(self, key):
        return self.client.rows.get(key)


def sdk_api(client, sleep=lambda _seconds: None):
    field = SimpleNamespace(name="name")
    return OfflineApi(client, project_type=field, dataset_type=field, media_type="video", targets=TARGETS,
                      sleep=sleep)


FORMS = (COLUMN_REVIEW, VIEW_TYPE_REVIEW)


class WorkspaceTests(unittest.TestCase):
    def test_the_workspaces_share_forms_and_nothing_else(self):
        test, pilot = WORKSPACES["test"], WORKSPACES["pilot"]
        for form in FORMS:
            self.assertEqual(test[form].ontology_name, pilot[form].ontology_name)
            self.assertNotEqual(test[form].project_name, pilot[form].project_name)
            self.assertNotEqual(test[form].dataset_name, pilot[form].dataset_name)
        names = [t.project_name for w in WORKSPACES.values() for t in w.values()]
        self.assertEqual(len(names), len(set(names)))

    def test_uses_the_bootstrapped_projects_and_creates_only_the_forms(self):
        client = FakeClient()
        workspace = sdk_api(client).ensure_workspace(FORMS, create=True)

        self.assertEqual(client.created, [("form", TARGETS[form].ontology_name) for form in FORMS])
        self.assertEqual(TARGETS[COLUMN_REVIEW].ontology_name, "DeepECHO column review v3")
        for form in FORMS:
            self.assertEqual(workspace.projects[form].ontology().name, TARGETS[form].ontology_name)
            self.assertIs(workspace.datasets[form], client.dataset(form))
        sdk_api(client).ensure_workspace(FORMS, create=True)
        self.assertEqual(len(client.created), 2)

    def test_the_column_review_goes_to_the_pilot_project(self):
        self.assertEqual(TARGETS[COLUMN_REVIEW].project_name, "RADAR Caption Pilot")
        self.assertEqual(TARGETS[COLUMN_REVIEW].dataset_name, "RADAR Caption Pilot Dataset")

    def test_a_missing_project_names_the_bootstrap_command(self):
        client = FakeClient(projects=[FakeProject(TARGETS[COLUMN_REVIEW].project_name)])
        with self.assertRaisesRegex(UploadError, "create_pilot_project.py --project-name"):
            sdk_api(client).ensure_workspace(FORMS, create=True)
        self.assertEqual(client.created, [])
        sdk_api(client).ensure_workspace([COLUMN_REVIEW], create=True)  # the other form can go ahead.

    def test_refuses_resources_the_bootstrap_did_not_make(self):
        other_project = FakeClient(projects=[FakeProject(t.project_name, description="someone else's")
                                             for t in TARGETS.values()])
        with self.assertRaisesRegex(UploadError, "not a pilot video project"):
            sdk_api(other_project).ensure_workspace(FORMS, create=True)
        other_dataset = FakeClient(datasets=[fake_dataset(t.dataset_name, "someone else's") for t in TARGETS.values()])
        with self.assertRaisesRegex(UploadError, "not an unconnected pilot dataset"):
            sdk_api(other_dataset).ensure_workspace(FORMS, create=True)

    def test_refuses_a_form_that_differs_from_the_committed_one(self):
        form = build_form()
        form["classifications"] = form["classifications"][1:]
        ontology = SimpleNamespace(name=TARGETS[COLUMN_REVIEW].ontology_name, uid="old", normalized=form)
        client = FakeClient(ontologies=[ontology])
        with self.assertRaisesRegex(UploadError, "differs from the committed form"):
            sdk_api(client).ensure_workspace(FORMS, create=True)
        self.assertEqual(client.created, [])

    def test_refuses_a_project_with_another_form(self):
        other = SimpleNamespace(name="another form", uid="other", normalized={"tools": [], "classifications": [{}]})
        client = FakeClient()
        client.project(COLUMN_REVIEW)._ontology = other
        with self.assertRaisesRegex(UploadError, "uses another form"):
            sdk_api(client).ensure_workspace(FORMS, create=True)

    def test_replaces_an_empty_default_form_without_labels(self):
        empty = SimpleNamespace(name="default", uid="empty", normalized={"tools": [], "classifications": []})
        client = FakeClient()
        client.project(COLUMN_REVIEW)._ontology = empty

        sdk_api(client).ensure_workspace([COLUMN_REVIEW], create=True)

        self.assertEqual(client.project(COLUMN_REVIEW).ontology().name, TARGETS[COLUMN_REVIEW].ontology_name)

    def test_lookup_alone_creates_and_connects_nothing(self):
        client = FakeClient()
        workspace = sdk_api(client).ensure_workspace(FORMS, create=False)

        self.assertEqual(client.created, [])
        self.assertEqual(workspace.ontologies, {COLUMN_REVIEW: None, VIEW_TYPE_REVIEW: None})
        self.assertIsNone(client.project(COLUMN_REVIEW).ontology())


class FakeTask:
    status = "COMPLETE"
    errors = []

    def wait_till_done(self):
        return None


class SdkPayloadTests(unittest.TestCase):
    def test_upload_sends_each_copy_under_its_key_to_its_forms_dataset(self):
        uploaded, sent = [], {}
        client = SimpleNamespace(upload_file=lambda path: uploaded.append(path) or "https://uploads.example.invalid/x")
        datasets = {form: SimpleNamespace(create_data_rows=lambda rows, form=form: sent.setdefault(form, rows)
                                          and FakeTask()) for form in FORMS}
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
            sdk_api(client).upload_rows(Workspace(datasets, {}, {}), plan.rows)

        self.assertEqual([Path(path).name for path in uploaded], [f"{row.global_key}.mp4" for row in plan.rows])
        self.assertEqual([row["global_key"] for row in sent[COLUMN_REVIEW]], [A4C, COLOR])
        self.assertEqual([row["global_key"] for row in sent[VIEW_TYPE_REVIEW]], [OTHER])
        self.assertTrue(all(row["external_id"] == row["global_key"] for rows in sent.values() for row in rows))
        self.assertEqual(sent[VIEW_TYPE_REVIEW][0]["attachments"], [NOTICE])
        self.assertNotIn(folder, json.dumps(sent))

    def test_upload_refuses_a_copy_changed_after_the_checks(self):
        client = SimpleNamespace(upload_file=lambda path: self.fail("uploaded a changed copy"))
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
            plan.rows[0].path.write_bytes(b"changed")
            with self.assertRaisesRegex(UploadError, "changed after the checks"):
                sdk_api(client).upload_rows(Workspace({}, {}, {}), plan.rows)

    def test_existing_rows_must_be_in_their_forms_dataset(self):
        client = FakeClient()
        workspace = sdk_api(client).ensure_workspace(FORMS, create=False)
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        client.rows[A4C] = SimpleNamespace(dataset=lambda: client.dataset(COLUMN_REVIEW))
        self.assertEqual(sdk_api(client).existing_global_keys(workspace, plan.rows), {A4C})
        client.rows[OTHER] = SimpleNamespace(dataset=lambda: client.dataset(COLUMN_REVIEW))
        with self.assertRaisesRegex(UploadError, "belongs to another dataset"):
            sdk_api(client).existing_global_keys(workspace, plan.rows)

    def test_export_reads_the_rows_from_the_stream_not_the_result_url(self):
        rows = [exported(A4C), exported(OTHER)]

        class StreamedExport:
            status = "COMPLETE"
            result = "https://storage.example.invalid/export-result.ndjson"  # what `result` holds for this task.

            def wait_till_done(self):
                return None

            def has_errors(self):
                return False

            def has_result(self):
                return True

            def get_buffered_stream(self):
                return iter([SimpleNamespace(json=row) for row in rows])

        project = SimpleNamespace(export=lambda params: StreamedExport())

        self.assertEqual(LabelboxUploadApi.export_rows(project), rows)

        failed = StreamedExport()
        failed.has_errors = lambda: True
        with self.assertRaisesRegex(UploadError, "export failed"):
            LabelboxUploadApi.export_rows(SimpleNamespace(export=lambda params: failed))

    def test_batches_are_created_reused_and_replaced_only_without_labels(self):
        batches, deleted = [], []

        def create_batch(name, global_keys):
            batch = SimpleNamespace(name=name, size=len(global_keys))
            batch.delete = lambda: (deleted.append(name), batches.remove(batch))
            batches.append(batch)

        project = SimpleNamespace(batches=lambda: list(batches), create_batch=create_batch, get_label_count=lambda: 0)
        plan = {f"{BATCH_PREFIX} study": [A4C, COLOR]}

        self.assertEqual(LabelboxUploadApi.ensure_batches(project, plan), 1)
        self.assertEqual(LabelboxUploadApi.ensure_batches(project, plan), 0)
        self.assertEqual(LabelboxUploadApi.ensure_batches(project, {f"{BATCH_PREFIX} study": [A4C]}), 1)
        self.assertEqual(deleted, [f"{BATCH_PREFIX} study"])
        project.get_label_count = lambda: 1
        with self.assertRaisesRegex(UploadError, "submitted labels"):
            LabelboxUploadApi.ensure_batches(project, plan)

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK not installed")
    def test_prelabels_are_imported_by_schema_id(self):
        created = []

        class Job:
            state = SimpleNamespace(value="FINISHED")
            errors = []

            def wait_till_done(self):
                return None

        class PredictionImport:
            @classmethod
            def create_from_objects(cls, **kwargs):
                created.append(kwargs)
                return Job()

        project = SimpleNamespace(uid="project", get_mal_prediction_imports=lambda: [],
                                  enable_model_assisted_labeling=lambda enabled: self.assertTrue(enabled))
        workspace = Workspace({}, {COLUMN_REVIEW: project}, {COLUMN_REVIEW: SimpleNamespace(normalized=server_form())})
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        with mock.patch("labelbox.schema.annotation_import.MALPredictionImport", PredictionImport):
            self.assertTrue(sdk_api(None).ensure_prelabels(workspace, list(plan.candidates.values())))

        self.assertEqual(created[0]["name"], PREDICTIONS_NAME)
        predictions = created[0]["predictions"]
        self.assertEqual(len(predictions), sum(len(prelabels(c)) for c in plan.candidates.values()))
        self.assertTrue(all("schemaId" in p and "name" not in p for p in predictions))
        self.assertEqual({p["dataRow"]["globalKey"] for p in predictions}, {A4C, COLOR})


class OperationNotAllowedException(Exception):
    """Named like the SDK's, which the uploader recognises by name."""


class CleanupTests(unittest.TestCase):
    """What `--delete` finds and removes in a workspace that also holds the pilot's own work."""

    def setUp(self):
        self.events, self.sleeps = [], []
        self.client = FakeClient()
        folder = tempfile.mkdtemp()
        self.plan = read_plan(make_folder(Path(folder)))
        for form in FORMS:
            project = self.client.project(form)
            for name in [*self.plan.batches(form), "pilot study batch"]:
                project.batch_list.append(self.batch(project, name))
        column = self.client.project(COLUMN_REVIEW)
        for name in (PREDICTIONS_NAME, "pilot prelabels"):
            item = SimpleNamespace(name=name)
            item.delete = lambda item=item: self.events.append(("delete import", item.name))
            column.import_list.append(item)
        for row in self.plan.rows:
            data_row = SimpleNamespace(dataset=lambda form=row.form: self.client.dataset(form))
            data_row.delete = lambda key=row.global_key: self.events.append(("delete row", key))
            self.client.rows[row.global_key] = data_row

    def batch(self, project, name):
        batch = SimpleNamespace(name=name, refusals=[])
        batch.delete_labels = lambda: self.events.append(("delete labels", name))

        def delete():
            if batch.refusals:
                raise batch.refusals.pop(0)
            self.events.append(("delete batch", name))
            project.batch_list.remove(batch)

        batch.delete = delete
        return batch

    def cleanup(self, apply):
        api = sdk_api(self.client, sleep=self.sleeps.append)
        return api.cleanup(api.ensure_workspace(FORMS, create=False), self.plan, apply=apply)

    def test_listing_counts_only_this_folders_work_and_deletes_nothing(self):
        self.assertEqual(self.cleanup(apply=False), {"study batches": 2, "prelabel imports": 1, "data rows": 3})
        self.assertEqual(self.events, [])

    def test_deletes_labels_batches_import_then_rows_and_nothing_of_the_pilot(self):
        self.cleanup(apply=True)

        kinds = [kind for kind, _ in self.events]
        self.assertEqual(kinds, ["delete labels"] * 2 + ["delete batch"] * 2 + ["delete import"]
                         + ["delete row"] * 3)
        names = [name for _, name in self.events]
        self.assertNotIn("pilot study batch", names)
        self.assertNotIn("pilot prelabels", names)
        self.assertEqual({name for kind, name in self.events if kind == "delete row"}, {A4C, COLOR, OTHER})
        self.assertEqual(self.client.project(COLUMN_REVIEW).ontology(), None)  # forms are never touched.

    def refuse(self, errors):
        """The first test batch's next deletions raise these errors."""
        batch = next(b for b in self.client.project(COLUMN_REVIEW).batch_list if b.name != "pilot study batch")
        batch.refusals.extend(errors)
        return batch.name

    def test_waits_while_labelbox_is_still_deleting_a_batchs_labels(self):
        name = self.refuse([OperationNotAllowedException("You can't delete batches that have labels.")] * 2)

        self.cleanup(apply=True)

        self.assertEqual(self.sleeps, list(BATCH_DELETE_WAITS[:2]))
        self.assertIn(("delete batch", name), self.events)
        self.assertEqual(len([e for e in self.events if e[0] == "delete row"]), 3)

    def test_stops_before_the_rows_when_labelbox_keeps_refusing(self):
        self.refuse([OperationNotAllowedException("You can't delete batches that have labels.")]
                    * (len(BATCH_DELETE_WAITS) + 1))

        with self.assertRaises(StepFailed) as raised:
            self.cleanup(apply=True)

        self.assertEqual((raised.exception.step, raised.exception.kind),
                         ("deleting the study batches", "OperationNotAllowedException"))
        self.assertEqual(self.sleeps, list(BATCH_DELETE_WAITS))
        self.assertEqual([kind for kind, _ in self.events], ["delete labels"] * 2)

    def test_does_not_wait_on_other_errors(self):
        self.refuse([RuntimeError("server down")])

        with self.assertRaises(StepFailed) as raised:
            self.cleanup(apply=True)

        self.assertEqual(raised.exception.kind, "RuntimeError")
        self.assertEqual(self.sleeps, [])

    def test_a_rerun_after_deletion_finds_only_what_is_left(self):
        self.cleanup(apply=True)
        self.client.rows.clear()
        self.client.project(COLUMN_REVIEW).import_list = [i for i in self.client.project(COLUMN_REVIEW).import_list
                                                         if i.name != PREDICTIONS_NAME]

        self.assertEqual(self.cleanup(apply=False), {"study batches": 0, "prelabel imports": 0, "data rows": 0})

    def test_refuses_to_delete_a_key_found_in_another_dataset(self):
        self.client.rows[OTHER] = SimpleNamespace(dataset=lambda: fake_dataset("someone else's dataset"))
        with self.assertRaisesRegex(UploadError, "belongs to another dataset"):
            self.cleanup(apply=True)
        self.assertEqual(self.events, [])


class ExportTests(unittest.TestCase):
    def test_scores_both_forms_and_keeps_unreviewed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        view_type = [radio("input_type", "cine_loop", [radio("view", "A4C"), radio("modality", "bmode")]),
                     radio("identifier_visible", "no")]
        summary = summarize_export({
            COLUMN_REVIEW: [exported(A4C, column_review_label(plan.candidates[A4C])), exported(COLOR)],
            VIEW_TYPE_REVIEW: [exported(OTHER, view_type)],
        }, plan)

        status = {item["global_key"]: item["status"] for item in summary["items"]}
        self.assertEqual(status, {A4C: "valid", COLOR: "unreviewed", OTHER: "valid"})
        self.assertNotIn("source_text", json.dumps(summary))
        self.assertNotIn("Dysfonction", json.dumps(summary, ensure_ascii=False))
        other = next(item for item in summary["items"] if item["global_key"] == OTHER)
        self.assertEqual(other["answers"]["view"], "A4C")

    def test_answers_are_keyed_by_question_id_not_display_text(self):
        label = {"annotations": {"classifications": [
            radio("input_type", "cine_loop", [radio("view", "A2C"), radio("modality", "color_flow")]),
            radio("identifier_visible", "no"),
            {"name": "Comments (no patient details)", "value": "comments", "text_answer": {"content": "fine"}},
            checklist("visible_targets", [answer("mitral"), answer("la")]),
        ]}}

        self.assertEqual(_answers_from_label(label), {
            "input_type": "cine_loop", "view": "A2C", "modality": "color_flow", "identifier_visible": "no",
            "comments": "fine", "visible_targets": ["mitral", "la"]})

    def test_incomplete_reviews_are_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        summary = summarize_export({VIEW_TYPE_REVIEW: [exported(OTHER, [radio("input_type", "cine_loop")])]}, plan)

        self.assertEqual(summary["items"][0]["status"], "invalid")
        self.assertTrue(summary["items"][0]["problems"])

    def test_leaves_out_the_pilots_own_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        summary = summarize_export({COLUMN_REVIEW: [exported("radar-pilot-" + "e" * 24), exported(A4C)]}, plan)

        self.assertEqual([item["global_key"] for item in summary["items"]], [A4C])

    def test_refuses_a_row_found_in_the_other_forms_project(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = read_plan(make_folder(Path(folder)))
        with self.assertRaisesRegex(UploadError, "other form's project"):
            summarize_export({COLUMN_REVIEW: [exported(OTHER)]}, plan)


class MainTests(unittest.TestCase):
    def run_main(self, *args, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args), **kwargs)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_writes_the_private_preview_and_makes_no_client(self):
        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            code, out, _ = self.run_main("--folder", folder, environ={},
                                         api_factory=lambda _key: self.fail("made a client in a dry run"))
            mode = stat.S_IMODE((Path(folder) / PREVIEW).stat().st_mode)
            sent = json.loads((Path(folder) / PREVIEW).read_text())

        self.assertEqual(code, 0)
        self.assertIn("no Labelbox requests", out)
        self.assertEqual(mode, 0o600)
        self.assertEqual(sent["forms"][COLUMN_REVIEW]["project"], "RADAR Workflow Test - column review v3")
        self.assertEqual(len(sent["rows"]), 3)

    def test_the_pilot_workspace_is_chosen_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            self.run_main("--folder", folder, "--workspace", "pilot", environ={})
            sent = json.loads((Path(folder) / PREVIEW).read_text())
        self.assertEqual(sent["forms"][COLUMN_REVIEW]["project"], "RADAR Caption Pilot")

    def test_form_limits_the_preview_and_the_run(self):
        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            code, out, _ = self.run_main("--folder", folder, "--form", COLUMN_REVIEW, environ={})
            sent = json.loads((Path(folder) / PREVIEW).read_text())
        self.assertEqual(code, 0)
        self.assertIn("2 column_review", out)
        self.assertEqual([row["global_key"] for row in sent["rows"]], [A4C, COLOR])

    def test_apply_and_delete_need_a_key(self):
        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            for action in (["--apply"], ["--delete"]):
                code, _, err = self.run_main("--folder", folder, *action, environ={})
                self.assertEqual(code, 2)
                self.assertIn("LABELBOX_API_KEY", err)

    def test_export_runs_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            with self.assertRaises(SystemExit):
                self.run_main("--folder", folder, "--apply", "--export-summary", f"{folder}/summary.json")

    def test_failures_print_only_their_type(self):
        def failing(_key):
            raise RuntimeError(f"secret-key at {folder}/local/path")

        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            code, out, err = self.run_main("--folder", folder, "--apply", environ={"LABELBOX_API_KEY": "secret-key"},
                                           api_factory=failing)
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError", err)
        self.assertNotIn("secret-key", out + err)
        self.assertNotIn("local/path", out + err)

    def test_a_refused_step_is_named_without_the_server_message(self):
        class AuthorizationError(Exception):
            pass

        class NoForms(FakeClient):
            def create_ontology(self, name, form, media_type):
                raise AuthorizationError("secret-server-detail")

        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            code, out, err = self.run_main("--folder", folder, "--apply", environ={"LABELBOX_API_KEY": "key"},
                                           api_factory=lambda _key: sdk_api(NoForms()))
        self.assertEqual(code, 1)
        self.assertIn("while creating the form 'DeepECHO column review v3' (AuthorizationError)", err)
        self.assertIn("not allowed", err)
        self.assertNotIn("secret-server-detail", out + err)

    def test_apply_reports_counts(self):
        api = FakeApi()
        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            code, out, _ = self.run_main("--folder", folder, "--apply", environ={"LABELBOX_API_KEY": "key"},
                                         api_factory=lambda _key: api)
        self.assertEqual(code, 0)
        self.assertIn("uploaded now: 3", out)
        self.assertIn("prelabels: imported", out)

    def test_delete_lists_first_and_deletes_only_with_apply(self):
        cleanup = CleanupTests("test_listing_counts_only_this_folders_work_and_deletes_nothing")
        cleanup.setUp()
        with tempfile.TemporaryDirectory() as folder:
            make_folder(Path(folder))
            code, out, _ = self.run_main("--folder", folder, "--delete", environ={"LABELBOX_API_KEY": "key"},
                                         api_factory=lambda _key: sdk_api(cleanup.client))
            self.assertEqual(code, 0)
            self.assertIn("nothing was deleted", out)
            self.assertEqual(cleanup.events, [])
            code, out, _ = self.run_main("--folder", folder, "--delete", "--apply",
                                         environ={"LABELBOX_API_KEY": "key"},
                                         api_factory=lambda _key: sdk_api(cleanup.client))
        self.assertEqual(code, 0)
        self.assertIn("Deleted what this folder uploaded: 2 study batches, 1 prelabel imports, 3 data rows", out)
        self.assertEqual(len([e for e in cleanup.events if e[0] == "delete row"]), 3)


if __name__ == "__main__":
    unittest.main()
