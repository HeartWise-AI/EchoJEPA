import copy
import importlib.util
import io
import json
import logging
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import imageio.v3 as iio

from tools.labelbox.column_review import build_form, prelabels
from tools.labelbox.synthetic_pipeline import (
    CANDIDATE_ATTACHMENT,
    GLOBAL_KEY_PREFIX,
    LEGACY_SANDBOX_ONTOLOGY_NAMES,
    SANDBOX_BATCH_PREFIX,
    SANDBOX_DATASET_DESCRIPTION,
    SANDBOX_PREDICTIONS_NAME,
    SUMMARY_SCHEMA,
    LabelboxSandboxApi,
    SandboxError,
    Workspace,
    _canonical_form,
    _difference_paths,
    _render_video,
    _safe_error_summary,
    _study_batch_plan,
    ensure_pipeline,
    main,
    summarize_export,
    synthetic_cases,
)


def server_form():
    form = copy.deepcopy(build_form())

    def add_ids(questions, path="root"):
        for question_index, question in enumerate(questions):
            question_path = f"{path}.q{question_index}"
            question["featureSchemaId"] = question_path
            for option_index, option in enumerate(question.get("options", [])):
                option_path = f"{question_path}.o{option_index}"
                option["featureSchemaId"] = option_path
                add_ids(option.get("options", []), option_path)

    add_ids(form["classifications"])
    return form


class FakeApi:
    def __init__(self, *, existing=(), export_rows=()):
        self.workspace = Workspace(
            project=SimpleNamespace(name="project", uid="synthetic-project-id"),
            dataset=SimpleNamespace(name="dataset"),
            ontology=SimpleNamespace(name="ontology"),
        )
        self.existing = set(existing)
        self.rows = []
        self.study_batches = None
        self.exported = list(export_rows)
        self.prediction_cases = None
        self.calls = []

    def ensure_workspace(self, form, *, create):
        self.form = form
        self.create = create
        return self.workspace

    def existing_global_keys(self, workspace, keys):
        self.queried_keys = list(keys)
        return self.existing & set(keys)

    def upload_rows(self, workspace, cases, paths):
        self.rows.extend(cases)
        self.paths = dict(paths)
        for case in cases:
            self.existing.add(case.global_key)

    def ensure_study_batches(self, workspace, cases):
        self.calls.append("study_batches")
        if self.study_batches is not None:
            return 0
        grouped = {}
        for case in cases:
            grouped.setdefault(case.study_key, []).append(case.global_key)
        self.study_batches = grouped
        return len(grouped)

    def ensure_prelabels(self, workspace, cases, *, refresh=False):
        self.calls.append("prelabels")
        self.prelabels_refresh = refresh
        if self.prediction_cases is not None:
            return False
        self.prediction_cases = list(cases)
        return True

    def export_rows(self, workspace):
        return self.exported


class FakeTask:
    status = "COMPLETE"
    errors = []

    def wait_till_done(self):
        return None


class CapturingDataset:
    def __init__(self):
        self.items = None

    def create_data_rows(self, items):
        self.items = items
        return FakeTask()


class CapturingSdkClient:
    def __init__(self):
        self.uploaded_paths = []

    def upload_file(self, path):
        self.uploaded_paths.append(path)
        return "https://uploads.example.invalid/synthetic-video"


class MissingRowsSdkClient:
    def get_data_row_by_global_key(self, _key):
        from labelbox.schema.data_row import DataRow
        from lbox.exceptions import ResourceNotFoundError

        raise ResourceNotFoundError(DataRow, {})


class SyntheticCasesTests(unittest.TestCase):
    def test_cases_are_reserved_and_candidate_payloads_are_synthetic(self):
        cases = synthetic_cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({case.global_key for case in cases}), len(cases))
        for case in cases:
            candidate = case.candidate()
            self.assertTrue(case.global_key.startswith(GLOBAL_KEY_PREFIX))
            self.assertIs(candidate["synthetic"], True)
            self.assertEqual(candidate["global_key"], case.global_key)
            self.assertEqual(candidate["study_key"], case.study_key)
            self.assertEqual(candidate["clip_key"], case.clip_key)
            self.assertEqual(candidate["schema"], "deepecho-column-review")
            self.assertEqual(candidate["version"], 2)
            self.assertGreater(len(candidate["columns"]), 0)
            self.assertTrue(set(candidate["report_context"]).issubset(case.report))

        self.assertEqual(len(build_form()["classifications"]), 11)

    def test_renderer_creates_a_readable_mp4(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.mp4"
            _render_video(synthetic_cases()[0], path)
            metadata = iio.immeta(path)

            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(metadata["size"], (320, 256))

    def test_canonical_form_normalizes_only_inactive_server_metadata(self):
        expected_question = {
            "name": "x",
            "schemaNodeId": None,
            "attributes": None,
            "options": [{"value": "yes", "featureSchemaId": None, "options": []}],
        }
        server_question = {
            "name": "x",
            "schemaNodeId": "server",
            "kind": "RadioQuestion",
            "archived": 0,
            "options": [{"value": "yes", "featureSchemaId": "id", "kind": "RadioOption", "archived": False,}],
        }
        left = {"classifications": [expected_question]}
        right = {
            "id": "server-ontology-id",
            "name": "server ontology resource name",
            "files": [],
            "relationships": [],
            "classifications": [server_question],
        }
        self.assertEqual(_canonical_form(left), _canonical_form(json.dumps(right)))
        self.assertNotEqual(
            _canonical_form(left), _canonical_form({**right, "classifications": [{**server_question, "name": "y"}]}),
        )
        self.assertNotEqual(
            _canonical_form(left),
            _canonical_form({**right, "classifications": [{**server_question, "archived": True}]}),
        )
        self.assertNotEqual(
            _canonical_form(left),
            _canonical_form({**right, "classifications": [{**server_question, "attributes": {"key": "value"}}]}),
        )

    def test_difference_paths_are_bounded_and_never_include_values(self):
        paths = _difference_paths(
            {"questions": [{"name": "expected-secret", "required": True}]},
            {"questions": [{"name": "actual-secret", "required": False}], "server": "server-secret"},
            limit=2,
        )

        self.assertEqual(paths, ["root.questions[0].name", "root.questions[0].required"])
        self.assertNotIn("secret", json.dumps(paths))

    def test_server_error_summary_redacts_urls_paths_and_tokens(self):
        summary = _safe_error_summary(
            {
                "message": (
                    "Invalid attachment at /sensitive/local/path using "
                    "https://uploads.example.invalid/private?token=secret and "
                    "abcdefghijklmnopqrstuvwxyz0123456789"
                )
            }
        )

        self.assertIn("Invalid attachment", summary)
        self.assertIn("<path>", summary)
        self.assertIn("<url>", summary)
        self.assertIn("<token>", summary)
        self.assertNotIn("sensitive", summary)
        self.assertNotIn("secret", summary)

class OrchestrationTests(unittest.TestCase):
    @staticmethod
    def renderer(case, path):
        path.write_bytes(b"synthetic video bytes")

    def test_uploads_only_missing_rows_and_creates_one_batch_per_study(self):
        first = synthetic_cases()[0].global_key
        api = FakeApi(existing={first})

        result = ensure_pipeline(api, renderer=self.renderer)

        self.assertTrue(api.create)
        self.assertEqual(result.uploaded_rows, 3)
        self.assertEqual(result.total_rows, 4)
        self.assertEqual(result.project_id, "synthetic-project-id")
        self.assertEqual(result.total_study_batches, 3)
        self.assertEqual(result.created_study_batches, 3)
        self.assertTrue(result.prelabels_imported)
        self.assertEqual(api.calls, ["study_batches", "prelabels"])
        self.assertEqual(len(api.study_batches), 3)
        a4c_cases = [case for case in synthetic_cases() if case.study_key.endswith("01")]
        self.assertEqual(
            api.study_batches[a4c_cases[0].study_key],
            [case.global_key for case in a4c_cases],
        )
        self.assertEqual(set(api.paths), {case.global_key for case in synthetic_cases()[1:]})
        self.assertTrue(all(path.name.endswith(".mp4") for path in api.paths.values()))

    def test_second_run_is_idempotent(self):
        keys = [case.global_key for case in synthetic_cases()]
        api = FakeApi(existing=keys)
        api.study_batches = {
            case.study_key: [item.global_key for item in synthetic_cases() if item.study_key == case.study_key]
            for case in synthetic_cases()
        }
        api.prediction_cases = list(synthetic_cases())

        result = ensure_pipeline(api, renderer=lambda *_: self.fail("rendered an existing row"))

        self.assertEqual(result.uploaded_rows, 0)
        self.assertEqual(result.total_study_batches, 3)
        self.assertEqual(result.created_study_batches, 0)
        self.assertFalse(result.prelabels_imported)
        self.assertFalse(api.prelabels_refresh)
        self.assertEqual(api.rows, [])

    def test_refuses_non_reserved_or_duplicate_keys(self):
        case = synthetic_cases()[0]
        duplicate = type(case)(**{**case.__dict__, "slug": case.slug})
        with self.assertRaises(SandboxError):
            ensure_pipeline(FakeApi(), cases=(case, duplicate), renderer=self.renderer)


class SdkPayloadTests(unittest.TestCase):
    def test_sdk_adapter_creates_and_reuses_one_batch_per_study(self):
        batches = []
        project = SimpleNamespace(
            batches=lambda: batches,
            create_batch=lambda name, global_keys: batches.append(
                SimpleNamespace(name=name, global_keys=list(global_keys), size=len(global_keys))
            ),
        )
        workspace = Workspace(project=project, dataset=object(), ontology=object())
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        self.assertEqual(api.ensure_study_batches(workspace, synthetic_cases()), 3)
        self.assertEqual(api.ensure_study_batches(workspace, synthetic_cases()), 0)
        self.assertEqual(len(batches), 3)
        self.assertEqual(sorted(batch.size for batch in batches), [1, 1, 2])
        shared_study_keys = {
            case.global_key for case in synthetic_cases() if case.study_key.endswith("01")
        }
        self.assertIn(shared_study_keys, [set(batch.global_keys) for batch in batches])

    def test_sdk_adapter_replaces_stale_unlabeled_study_batch(self):
        deleted = []
        cases = [case for case in synthetic_cases() if case.study_key.endswith("01")]
        batch_name = next(
            iter(_study_batch_plan([case.candidate() for case in cases], prefix=SANDBOX_BATCH_PREFIX))
        )
        stale = SimpleNamespace(
            name=batch_name,
            size=0,
            delete=lambda: deleted.append(True),
        )
        batches = [stale]

        def create_batch(name, global_keys):
            batches.append(
                SimpleNamespace(name=name, global_keys=list(global_keys), size=len(global_keys))
            )

        project = SimpleNamespace(
            batches=lambda: batches,
            create_batch=create_batch,
            get_label_count=lambda: 0,
        )
        workspace = Workspace(project=project, dataset=object(), ontology=object())
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        self.assertEqual(api.ensure_study_batches(workspace, cases), 1)
        self.assertEqual(deleted, [True])
        self.assertEqual(batches[-1].size, len(cases))

    def test_sdk_adapter_refuses_to_replace_labeled_study_batch(self):
        cases = [case for case in synthetic_cases() if case.study_key.endswith("01")]
        batch_name = next(
            iter(_study_batch_plan([case.candidate() for case in cases], prefix=SANDBOX_BATCH_PREFIX))
        )
        stale = SimpleNamespace(
            name=batch_name,
            size=0,
            delete=lambda: self.fail("deleted a batch containing reviewed work"),
        )
        project = SimpleNamespace(
            batches=lambda: [stale],
            create_batch=lambda *_args, **_kwargs: self.fail("recreated a labeled batch"),
            get_label_count=lambda: 1,
        )
        workspace = Workspace(project=project, dataset=object(), ontology=object())
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        with self.assertRaisesRegex(SandboxError, "project has submitted labels"):
            api.ensure_study_batches(workspace, cases)

    def test_replaces_only_the_unlabeled_legacy_sandbox_ontology(self):
        legacy = SimpleNamespace(uid="legacy-id", name=next(iter(LEGACY_SANDBOX_ONTOLOGY_NAMES)))
        english = SimpleNamespace(uid="english-id", name="English ontology")
        connected = []
        project = SimpleNamespace(
            ontology=lambda: legacy, get_label_count=lambda: 0, connect_ontology=connected.append,
        )

        LabelboxSandboxApi._connect_expected_ontology(project, english)

        self.assertEqual(connected, [english])

    def test_refuses_to_replace_a_labeled_or_unknown_ontology(self):
        english = SimpleNamespace(uid="english-id", name="English ontology")
        legacy = SimpleNamespace(uid="legacy-id", name=next(iter(LEGACY_SANDBOX_ONTOLOGY_NAMES)))
        labeled = SimpleNamespace(
            ontology=lambda: legacy,
            get_label_count=lambda: 1,
            connect_ontology=lambda _: self.fail("replaced a labeled ontology"),
        )
        unknown = SimpleNamespace(
            ontology=lambda: SimpleNamespace(uid="other-id", name="Another ontology"),
            get_label_count=lambda: 0,
            connect_ontology=lambda _: self.fail("replaced an unknown ontology"),
        )

        with self.assertRaisesRegex(SandboxError, "another ontology or already has labels"):
            LabelboxSandboxApi._connect_expected_ontology(labeled, english)
        with self.assertRaisesRegex(SandboxError, "another ontology or already has labels"):
            LabelboxSandboxApi._connect_expected_ontology(unknown, english)

    def test_existing_dataset_calls_the_sdk_iam_relationship(self):
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")
        dataset = SimpleNamespace(description=SANDBOX_DATASET_DESCRIPTION, iam_integration=lambda: None,)

        api._validate_existing(None, dataset, None, {})

        dataset.iam_integration = lambda: object()
        with self.assertRaisesRegex(SandboxError, "unconnected synthetic sandbox"):
            api._validate_existing(None, dataset, None, {})

    @unittest.skipUnless(importlib.util.find_spec("lbox"), "Labelbox SDK not installed")
    def test_missing_global_keys_use_the_pinned_sdk_exception(self):
        api = LabelboxSandboxApi(MissingRowsSdkClient(), project_type=None, dataset_type=None, media_type="video")
        workspace = Workspace(project=object(), dataset=SimpleNamespace(uid="synthetic-dataset"), ontology=object(),)

        self.assertEqual(api.existing_global_keys(workspace, [f"{GLOBAL_KEY_PREFIX}missing"]), set())

    def test_upload_payload_contains_only_safe_synthetic_context(self):
        sdk = CapturingSdkClient()
        dataset = CapturingDataset()
        api = LabelboxSandboxApi(sdk, project_type=None, dataset_type=None, media_type="video")
        workspace = Workspace(project=object(), dataset=dataset, ontology=object())
        case = synthetic_cases()[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "local-secret-path.mp4"
            path.write_bytes(b"video")
            api.upload_rows(workspace, [case], {case.global_key: path})

        self.assertEqual(sdk.uploaded_paths, [str(path)])
        self.assertEqual(len(dataset.items), 1)
        row = dataset.items[0]
        serialized = json.dumps(row)
        self.assertEqual(row["global_key"], case.global_key)
        for attachment in row["attachments"]:
            self.assertTrue(attachment["name"].isascii())
        self.assertIn(CANDIDATE_ATTACHMENT, serialized)
        self.assertIn("SYNTHETIC DATA ONLY", serialized)
        self.assertNotIn("local-secret-path", serialized)
        self.assertNotIn("/tmp/", serialized)
        self.assertIn("DeepECHO review v2", serialized)
        self.assertIn(case.study_key, serialized)
        self.assertIn(case.clip_key, serialized)
        self.assertNotIn("report_context", serialized)
        self.assertNotIn("source_text", serialized)

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK not installed")
    def test_complete_v2_predictions_are_imported_as_prelabels(self):
        created = []

        class PredictionJob:
            state = SimpleNamespace(value="FINISHED")
            errors = []

            def wait_till_done(self):
                return None

        class PredictionImport:
            @classmethod
            def create_from_objects(cls, **kwargs):
                created.append(kwargs)
                return PredictionJob()

        project = SimpleNamespace(
            uid="project-id",
            get_mal_prediction_imports=lambda: [],
            enable_model_assisted_labeling=lambda enabled: self.assertTrue(enabled),
        )
        workspace = Workspace(
            project=project,
            dataset=object(),
            ontology=SimpleNamespace(normalized=server_form()),
        )
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        with mock.patch("labelbox.schema.annotation_import.MALPredictionImport", PredictionImport):
            self.assertTrue(api.ensure_prelabels(workspace, synthetic_cases()))

        self.assertEqual(created[0]["name"], SANDBOX_PREDICTIONS_NAME)
        predictions = created[0]["predictions"]
        self.assertEqual(
            len({prediction["uuid"] for prediction in predictions}),
            sum(len(prelabels(case.candidate())) for case in synthetic_cases()),
        )
        self.assertTrue(all("schemaId" in prediction for prediction in predictions))
        self.assertTrue(all("name" not in prediction for prediction in predictions))
        self.assertTrue(any(isinstance(prediction["answer"], list) for prediction in predictions))
        self.assertTrue(any(isinstance(prediction["answer"], str) for prediction in predictions))

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK not installed")
    def test_failed_prediction_import_is_replaced_only_without_labels(self):
        from labelbox.schema.enums import AnnotationImportState

        deleted = []
        created = []
        job = SimpleNamespace(
            name=SANDBOX_PREDICTIONS_NAME,
            state=AnnotationImportState.FINISHED,
            errors=[{"errors": [{"name": "synthetic-test-error"}]}],
            wait_till_done=lambda: None,
            delete=lambda: deleted.append(SANDBOX_PREDICTIONS_NAME),
        )
        project = SimpleNamespace(
            uid="project-id",
            get_mal_prediction_imports=lambda: [job],
            get_label_count=lambda: 0,
            enable_model_assisted_labeling=lambda enabled: self.assertTrue(enabled),
        )
        workspace = Workspace(project=project, dataset=object(), ontology=SimpleNamespace(normalized=server_form()))
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        class PredictionImport:
            @classmethod
            def create_from_objects(cls, **kwargs):
                created.append(kwargs)
                return SimpleNamespace(
                    state=AnnotationImportState.FINISHED,
                    errors=[],
                    wait_till_done=lambda: None,
                )

        with mock.patch("labelbox.schema.annotation_import.MALPredictionImport", PredictionImport):
            self.assertTrue(api.ensure_prelabels(workspace, synthetic_cases()))
        self.assertEqual(deleted, [SANDBOX_PREDICTIONS_NAME])
        self.assertEqual(len(created), 1)

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK not installed")
    def test_refresh_replaces_successful_prediction_import_without_labels(self):
        from labelbox.schema.enums import AnnotationImportState

        deleted = []
        created = []
        job = SimpleNamespace(
            name=SANDBOX_PREDICTIONS_NAME,
            state=AnnotationImportState.FINISHED,
            errors=[],
            wait_till_done=lambda: None,
            delete=lambda: deleted.append(SANDBOX_PREDICTIONS_NAME),
        )
        project = SimpleNamespace(
            uid="project-id",
            get_mal_prediction_imports=lambda: [job],
            get_label_count=lambda: 0,
            enable_model_assisted_labeling=lambda enabled: self.assertTrue(enabled),
        )
        workspace = Workspace(
            project=project,
            dataset=object(),
            ontology=SimpleNamespace(normalized=server_form()),
        )
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        class PredictionImport:
            @classmethod
            def create_from_objects(cls, **kwargs):
                created.append(kwargs)
                return SimpleNamespace(
                    state=AnnotationImportState.FINISHED,
                    errors=[],
                    wait_till_done=lambda: None,
                )

        with mock.patch("labelbox.schema.annotation_import.MALPredictionImport", PredictionImport):
            self.assertTrue(api.ensure_prelabels(workspace, synthetic_cases(), refresh=True))
        self.assertEqual(deleted, [SANDBOX_PREDICTIONS_NAME])
        self.assertEqual(len(created), 1)

    def test_legacy_prediction_migration_refuses_submitted_labels(self):
        legacy = SimpleNamespace(
            name="DeepECHO column review v2 synthetic prelabels v7",
            wait_till_done=lambda: None,
            delete=lambda: self.fail("A prediction import with submitted labels must not be deleted"),
        )
        project = SimpleNamespace(
            get_mal_prediction_imports=lambda: [legacy],
            get_label_count=lambda: 1,
        )
        workspace = Workspace(project=project, dataset=object(), ontology=object())
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        with self.assertRaisesRegex(SandboxError, "project has submitted labels"):
            api.ensure_prelabels(workspace, synthetic_cases())


def _export_classification(record):
    answer = record["answer"]
    item = {"name": record["name"]}
    if isinstance(answer, str):
        item["text_answer"] = {"content": answer}
    elif isinstance(answer, list):
        item["checklist_answers"] = [
            {
                "value": selected["name"],
                "classifications": [
                    _export_classification(child) for child in selected.get("classifications", [])
                ],
            }
            for selected in answer
        ]
    else:
        item["radio_answer"] = {
            "value": answer["name"],
            "classifications": [
                _export_classification(child) for child in answer.get("classifications", [])
            ],
        }
    return item


def exported_row(case, reviewed=False):
    labels = []
    if reviewed:
        records = prelabels(case.candidate())
        records.append({"name": "input_quality", "answer": {"name": "interpretable"}})
        records.append({"name": "review_completed", "answer": {"name": "yes"}})
        labels.append(
            {
                "label_details": {"created_at": "2026-10-10T10:00:00Z"},
                "annotations": {
                    "classifications": [_export_classification(record) for record in records]
                },
            }
        )
    return {
        "data_row": {"global_key": case.global_key},
        "attachments": [{"name": CANDIDATE_ATTACHMENT, "value": json.dumps(case.candidate())},],
        "projects": {"project-id": {"labels": labels}},
    }


class ExportTests(unittest.TestCase):
    def test_scores_reviewed_rows_and_keeps_unreviewed_rows(self):
        reviewed, unreviewed = synthetic_cases()[:2]

        summary = summarize_export([exported_row(reviewed, reviewed=True), exported_row(unreviewed)])

        self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
        by_key = {item["global_key"]: item for item in summary["items"]}
        self.assertEqual(by_key[reviewed.global_key]["status"], "scored")
        self.assertEqual(by_key[reviewed.global_key]["study_key"], reviewed.study_key)
        self.assertEqual(by_key[reviewed.global_key]["clip_key"], reviewed.clip_key)
        self.assertEqual(by_key[reviewed.global_key]["score"]["support_rate"], 1.0)
        self.assertTrue(by_key[reviewed.global_key]["score"]["ready_for_validation"])
        self.assertEqual(by_key[unreviewed.global_key]["status"], "unreviewed")

    def test_rejects_non_synthetic_or_duplicate_candidates(self):
        case = synthetic_cases()[0]
        row = exported_row(case)
        candidate_attachment = row["attachments"][0]
        candidate_attachment["value"] = json.dumps({**case.candidate(), "synthetic": False})
        with self.assertRaises(SandboxError):
            summarize_export([row])
        candidate_attachment["value"] = json.dumps([])
        with self.assertRaises(SandboxError):
            summarize_export([row])
        with self.assertRaises(SandboxError):
            summarize_export([exported_row(case), exported_row(case)])

    def test_null_label_details_do_not_break_scoring(self):
        case = synthetic_cases()[0]
        row = exported_row(case, reviewed=True)
        row["projects"]["project-id"]["labels"][0]["label_details"] = None

        summary = summarize_export([row])

        self.assertEqual(summary["items"][0]["status"], "scored")


class MainTests(unittest.TestCase):
    def test_dry_run_makes_no_client_and_needs_no_key(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main([], environ={}, api_factory=lambda _: self.fail("client created"))
        self.assertEqual(status, 0)
        self.assertIn("Dry run", stdout.getvalue())
        self.assertIn("4 synthetic videos", stdout.getvalue())

    def test_apply_requires_key(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["--apply"], environ={}, api_factory=lambda _: self.fail("client created"))
        self.assertEqual(status, 2)
        self.assertIn("LABELBOX_API_KEY is required", stderr.getvalue())

    def test_apply_does_not_print_key_or_paths(self):
        secret = "secret-test-key"
        received = []
        stdout = io.StringIO()
        api = FakeApi()
        with redirect_stdout(stdout):
            status = main(
                ["--apply"], environ={"LABELBOX_API_KEY": secret}, api_factory=lambda key: received.append(key) or api,
            )
        self.assertEqual(status, 0)
        self.assertEqual(received, [secret])
        self.assertNotIn(secret, stdout.getvalue())
        self.assertNotIn("/tmp/", stdout.getvalue())
        self.assertIn("rows=4", stdout.getvalue())
        self.assertIn("id=synthetic-project-id", stdout.getvalue())
        self.assertIn("study batches: total=3; created=3; reused=0", stdout.getvalue())

    def test_errors_are_sanitized(self):
        secret = "secret-test-key"
        stdout, stderr = io.StringIO(), io.StringIO()
        previous_log_disable = logging.root.manager.disable

        def fail(_):
            logging.getLogger("labelbox-test").error("%s at /sensitive/path", secret)
            print(f"{secret} at /sensitive/path")
            print(f"{secret} at /sensitive/path", file=sys.stderr)
            raise RuntimeError(f"{secret} at /sensitive/path")

        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(["--apply"], environ={"LABELBOX_API_KEY": secret}, api_factory=fail)
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "Synthetic Labelbox pipeline failed (RuntimeError).\n")
        self.assertEqual(logging.root.manager.disable, previous_log_disable)

    def test_controlled_safety_errors_remain_actionable(self):
        stdout, stderr = io.StringIO(), io.StringIO()

        def fail(_):
            raise SandboxError("The fixed-name ontology differs from the committed column-review v2 form.")

        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(["--apply"], environ={"LABELBOX_API_KEY": "secret-test-key"}, api_factory=fail)

        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            "Synthetic Labelbox pipeline stopped by safety check: "
            "The fixed-name ontology differs from the committed column-review v2 form.\n",
        )

    def test_export_writes_only_summary(self):
        case = synthetic_cases()[0]
        api = FakeApi(export_rows=[exported_row(case)])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "summary.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(
                    ["--export-summary", str(output)],
                    environ={"LABELBOX_API_KEY": "secret-test-key"},
                    api_factory=lambda _: api,
                )
            summary = json.loads(output.read_text())
        self.assertEqual(status, 0)
        self.assertFalse(api.create)
        self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
        self.assertEqual(summary["items"][0]["status"], "unreviewed")
        self.assertIn("reviewed=0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
