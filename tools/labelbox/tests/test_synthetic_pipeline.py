import io
import json
import logging
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio

from tools.labelbox.synthetic_pipeline import (
    CANDIDATE_ATTACHMENT,
    GLOBAL_KEY_PREFIX,
    SANDBOX_BATCH_NAME,
    SANDBOX_DATASET_DESCRIPTION,
    SUMMARY_SCHEMA,
    LabelboxSandboxApi,
    SandboxError,
    Workspace,
    _canonical_form,
    _render_video,
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
        if self.batch_keys is not None:
            return False
        self.batch_keys = list(global_keys)
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

    def test_canonical_form_ignores_only_server_ids(self):
        left = {"name": "x", "schemaNodeId": None, "options": [{"value": "yes", "featureSchemaId": None}]}
        right = {"name": "x", "schemaNodeId": "server", "options": [{"value": "yes", "featureSchemaId": "id"}]}
        self.assertEqual(_canonical_form(left), _canonical_form(json.dumps(right)))
        self.assertNotEqual(_canonical_form(left), _canonical_form({"name": "y"}))


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
        self.assertEqual(api.batch_keys, [case.global_key for case in synthetic_cases()])
        self.assertEqual(set(api.paths), {case.global_key for case in synthetic_cases()[1:]})
        self.assertTrue(all(path.name.endswith(".mp4") for path in api.paths.values()))

    def test_second_run_is_idempotent(self):
        keys = [case.global_key for case in synthetic_cases()]
        api = FakeApi(existing=keys)
        api.batch_keys = keys

        result = ensure_pipeline(api, renderer=lambda *_: self.fail("rendered an existing row"))

        self.assertEqual(result.uploaded_rows, 0)
        self.assertFalse(result.batch_created)
        self.assertEqual(api.rows, [])

    def test_refuses_non_reserved_or_duplicate_keys(self):
        case = synthetic_cases()[0]
        duplicate = type(case)(**{**case.__dict__, "slug": case.slug})
        with self.assertRaises(SandboxError):
            ensure_pipeline(FakeApi(), cases=(case, duplicate), renderer=self.renderer)


class SdkPayloadTests(unittest.TestCase):
    def test_existing_dataset_calls_the_sdk_iam_relationship(self):
        api = LabelboxSandboxApi(object(), project_type=None, dataset_type=None, media_type="video")
        dataset = SimpleNamespace(description=SANDBOX_DATASET_DESCRIPTION, iam_integration=lambda: None,)

        api._validate_existing(None, dataset, None, {})

        dataset.iam_integration = lambda: object()
        with self.assertRaisesRegex(SandboxError, "unconnected synthetic sandbox"):
            api._validate_existing(None, dataset, None, {})

    def test_missing_global_keys_use_the_pinned_sdk_exception(self):
        api = LabelboxSandboxApi(MissingRowsSdkClient(), project_type=None, dataset_type=None, media_type="video")
        workspace = Workspace(project=object(), dataset=SimpleNamespace(uid="synthetic-dataset"), ontology=object(),)

        self.assertEqual(api.existing_global_keys(workspace, ["radar-synthetic-v1-missing"]), set())

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
            "view_ok": "yes",
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

        with self.assertRaisesRegex(ValueError, "Missing or invalid view_ok"):
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
