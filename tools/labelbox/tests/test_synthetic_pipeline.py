import importlib.util
import io
import json
import logging
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import imageio.v3 as iio

from tools.labelbox.synthetic_pipeline import (
    CANDIDATE_ATTACHMENT,
    GLOBAL_KEY_PREFIX,
    LEGACY_SANDBOX_ONTOLOGY_NAMES,
    SANDBOX_BATCH_NAME,
    SANDBOX_DATASET_DESCRIPTION,
    SANDBOX_VIEW_PREDICTIONS_NAME,
    SUMMARY_SCHEMA,
    LabelboxSandboxApi,
    SandboxError,
    Workspace,
    _canonical_form,
    _difference_paths,
    _render_video,
    _view_schema_ids,
    ensure_pipeline,
    main,
    summarize_export,
    synthetic_cases,
)


class FakeApi:
    def __init__(self, *, existing=(), export_rows=()):
        self.workspace = Workspace(
            project=SimpleNamespace(name="project"),
            dataset=SimpleNamespace(name="dataset"),
            ontology=SimpleNamespace(name="ontology"),
        )
        self.existing = set(existing)
        self.rows = []
        self.batch_keys = None
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

    def ensure_batch(self, workspace, global_keys):
        self.calls.append("batch")
        if self.batch_keys is not None:
            return False
        self.batch_keys = list(global_keys)
        return True

    def ensure_view_predictions(self, workspace, cases):
        self.calls.append("predictions")
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
            self.assertLessEqual(len(candidate["claims"]), 5)
            self.assertNotIn("report_context", candidate)

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

    def test_view_schema_ids_resolve_question_and_candidate_answers(self):
        ontology = SimpleNamespace(
            normalized={
                "classifications": [
                    {
                        "name": "view",
                        "featureSchemaId": "view-schema-id",
                        "options": [
                            {"value": "A4C", "featureSchemaId": "a4c-schema-id"},
                            {"value": "PLAX", "featureSchemaId": "plax-schema-id"},
                        ],
                    }
                ]
            }
        )

        question_id, answer_ids = _view_schema_ids(ontology, ["A4C", "PLAX"])

        self.assertEqual(question_id, "view-schema-id")
        self.assertEqual(answer_ids, {"A4C": "a4c-schema-id", "PLAX": "plax-schema-id"})

    def test_view_schema_ids_reject_missing_candidate_answer(self):
        ontology = SimpleNamespace(
            normalized=json.dumps(
                {
                    "classifications": [
                        {
                            "name": "view",
                            "featureSchemaId": "view-schema-id",
                            "options": [{"value": "A4C", "featureSchemaId": "a4c-schema-id"}],
                        }
                    ]
                }
            )
        )

        with self.assertRaisesRegex(SandboxError, "missing a candidate-view answer schema ID"):
            _view_schema_ids(ontology, ["PLAX"])


class OrchestrationTests(unittest.TestCase):
    @staticmethod
    def renderer(case, path):
        path.write_bytes(b"synthetic video bytes")

    def test_uploads_only_missing_rows_and_creates_one_batch(self):
        first = synthetic_cases()[0].global_key
        api = FakeApi(existing={first})

        result = ensure_pipeline(api, renderer=self.renderer)

        self.assertTrue(api.create)
        self.assertEqual(result.uploaded_rows, 3)
        self.assertEqual(result.total_rows, 4)
        self.assertTrue(result.batch_created)
        self.assertTrue(result.view_predictions_imported)
        self.assertEqual(api.calls, ["predictions", "batch"])
        self.assertEqual(api.batch_keys, [case.global_key for case in synthetic_cases()])
        self.assertEqual(set(api.paths), {case.global_key for case in synthetic_cases()[1:]})
        self.assertTrue(all(path.name.endswith(".mp4") for path in api.paths.values()))

    def test_second_run_is_idempotent(self):
        keys = [case.global_key for case in synthetic_cases()]
        api = FakeApi(existing=keys)
        api.batch_keys = keys
        api.prediction_cases = list(synthetic_cases())

        result = ensure_pipeline(api, renderer=lambda *_: self.fail("rendered an existing row"))

        self.assertEqual(result.uploaded_rows, 0)
        self.assertFalse(result.batch_created)
        self.assertFalse(result.view_predictions_imported)
        self.assertEqual(api.rows, [])

    def test_refuses_non_reserved_or_duplicate_keys(self):
        case = synthetic_cases()[0]
        duplicate = type(case)(**{**case.__dict__, "slug": case.slug})
        with self.assertRaises(SandboxError):
            ensure_pipeline(FakeApi(), cases=(case, duplicate), renderer=self.renderer)


class SdkPayloadTests(unittest.TestCase):
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

        self.assertEqual(api.existing_global_keys(workspace, ["radar-synthetic-v2-missing"]), set())

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
        self.assertIn(CANDIDATE_ATTACHMENT, serialized)
        self.assertIn("SYNTHETIC DATA ONLY", serialized)
        self.assertNotIn("local-secret-path", serialized)
        self.assertNotIn("/tmp/", serialized)
        self.assertNotIn('"name": "Candidate view"', serialized)

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK not installed")
    def test_candidate_view_predictions_are_single_choice_prelabels(self):
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
            ontology=SimpleNamespace(
                normalized={
                    "classifications": [
                        {
                            "name": "view",
                            "featureSchemaId": "view-schema-id",
                            "options": [
                                {"value": "A4C", "featureSchemaId": "a4c-schema-id"},
                                {"value": "PLAX", "featureSchemaId": "plax-schema-id"},
                            ],
                        }
                    ]
                }
            ),
        )
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        with mock.patch("labelbox.schema.annotation_import.MALPredictionImport", PredictionImport):
            self.assertTrue(api.ensure_view_predictions(workspace, synthetic_cases()))

        self.assertEqual(created[0]["name"], SANDBOX_VIEW_PREDICTIONS_NAME)
        self.assertEqual(
            len({prediction["uuid"] for prediction in created[0]["predictions"]}),
            len(synthetic_cases()),
        )
        self.assertEqual(
            created[0]["predictions"],
            [
                {
                    "dataRow": {"globalKey": case.global_key},
                    "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"echojepa:{case.global_key}:candidate-view:v4")),
                    "schemaId": "view-schema-id",
                    "answer": {"schemaId": f"{case.view.lower()}-schema-id"},
                }
                for case in synthetic_cases()
            ],
        )

    @unittest.skipUnless(importlib.util.find_spec("labelbox"), "Labelbox SDK not installed")
    def test_existing_prediction_import_with_row_errors_is_not_reported_as_reused(self):
        from labelbox.schema.enums import AnnotationImportState

        job = SimpleNamespace(
            name=SANDBOX_VIEW_PREDICTIONS_NAME,
            state=AnnotationImportState.FINISHED,
            errors=[{"errors": [{"name": "synthetic-test-error"}]}],
            wait_till_done=lambda: None,
        )
        project = SimpleNamespace(
            get_mal_prediction_imports=lambda: [job],
            enable_model_assisted_labeling=lambda enabled: self.assertTrue(enabled),
        )
        workspace = Workspace(project=project, dataset=object(), ontology=object())
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")

        with self.assertRaisesRegex(SandboxError, "existing candidate-view prediction import failed"):
            api.ensure_view_predictions(workspace, synthetic_cases())


def exported_row(case, answers=None):
    labels = []
    if answers is not None:
        classifications = []
        for name, value in answers.items():
            if name in {"missing_detail", "optional_correction"}:
                classifications.append({"name": name, "text_answer": {"content": value}})
            else:
                classifications.append({"name": name, "radio_answer": {"value": value}})
        labels.append(
            {
                "label_details": {"created_at": "2026-10-10T10:00:00Z"},
                "annotations": {"classifications": classifications},
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
        answers = {
            "view": "A4C",
            "modality_ok": "yes",
            "claim_1_ok": "yes",
            "claim_2_ok": "yes",
            "claim_3_ok": "no_claim",
            "claim_4_ok": "no_claim",
            "claim_5_ok": "no_claim",
            "important_missing": "no",
        }

        summary = summarize_export([exported_row(reviewed, answers), exported_row(unreviewed)])

        self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
        by_key = {item["global_key"]: item for item in summary["items"]}
        self.assertEqual(by_key[reviewed.global_key]["status"], "scored")
        self.assertTrue(by_key[reviewed.global_key]["score"]["all_supported"])
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
        row = exported_row(case, {})
        row["projects"]["project-id"]["labels"][0]["label_details"] = None

        with self.assertRaisesRegex(ValueError, "Missing or invalid view"):
            summarize_export([row])


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
        self.assertIn(SANDBOX_BATCH_NAME, stdout.getvalue())

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
            raise SandboxError("The fixed-name ontology differs from the committed simple review form.")

        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(["--apply"], environ={"LABELBOX_API_KEY": "secret-test-key"}, api_factory=fail)

        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            "Synthetic Labelbox pipeline stopped by safety check: "
            "The fixed-name ontology differs from the committed simple review form.\n",
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
