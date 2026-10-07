import io
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass

from tools.labelbox.create_pilot_project import (
    AmbiguousResourceError,
    LabelboxPilotClient,
    ensure_pilot_resources,
    main,
)


@dataclass
class FakeResource:
    name: str
    uid: str


class FakeClient:
    def __init__(self, *, projects=(), datasets=()):
        self.projects = list(projects)
        self.datasets = list(datasets)
        self.created = []

    def find_projects(self, name):
        return [resource for resource in self.projects if resource.name == name]

    def find_datasets(self, name):
        return [resource for resource in self.datasets if resource.name == name]

    def create_video_project(self, name, description):
        self.created.append(("project", name, description))
        resource = FakeResource(name=name, uid="project-created")
        self.projects.append(resource)
        return resource

    def create_dataset(self, name, description):
        self.created.append(("dataset", name, description))
        resource = FakeResource(name=name, uid="dataset-created")
        self.datasets.append(resource)
        return resource


class FakeField:
    def __eq__(self, value):
        return ("equals", value)


class FakeModel:
    name = FakeField()


class FakeSdkClient:
    def __init__(self):
        self.calls = []

    def get_projects(self, *, where):
        self.calls.append(("get_projects", where))
        return [FakeResource("project", "project-id")]

    def get_datasets(self, *, where):
        self.calls.append(("get_datasets", where))
        return [FakeResource("dataset", "dataset-id")]

    def create_project(self, **kwargs):
        self.calls.append(("create_project", kwargs))
        return FakeResource(kwargs["name"], "project-created")

    def create_dataset(self, **kwargs):
        self.calls.append(("create_dataset", kwargs))
        return FakeResource(kwargs["name"], "dataset-created")


class EnsurePilotResourcesTests(unittest.TestCase):
    def test_creates_both_resources_when_missing(self):
        client = FakeClient()

        result = ensure_pilot_resources(client)

        self.assertTrue(result.project.created)
        self.assertEqual(result.project.uid, "project-created")
        self.assertTrue(result.dataset.created)
        self.assertEqual(result.dataset.uid, "dataset-created")
        self.assertEqual([call[0] for call in client.created], ["project", "dataset"])

    def test_reuses_existing_resources(self):
        client = FakeClient(
            projects=[FakeResource("RADAR Caption Pilot", "project-existing")],
            datasets=[FakeResource("RADAR Caption Pilot Dataset", "dataset-existing")],
        )

        result = ensure_pilot_resources(client)

        self.assertFalse(result.project.created)
        self.assertEqual(result.project.uid, "project-existing")
        self.assertFalse(result.dataset.created)
        self.assertEqual(result.dataset.uid, "dataset-existing")
        self.assertEqual(client.created, [])

    def test_creates_only_the_missing_resource(self):
        client = FakeClient(projects=[FakeResource("RADAR Caption Pilot", "project-existing")])

        result = ensure_pilot_resources(client)

        self.assertFalse(result.project.created)
        self.assertTrue(result.dataset.created)
        self.assertEqual([call[0] for call in client.created], ["dataset"])

    def test_rejects_duplicate_project_names_before_writing(self):
        client = FakeClient(
            projects=[
                FakeResource("RADAR Caption Pilot", "project-one"),
                FakeResource("RADAR Caption Pilot", "project-two"),
            ]
        )

        with self.assertRaises(AmbiguousResourceError):
            ensure_pilot_resources(client)

        self.assertEqual(client.created, [])

    def test_rejects_duplicate_dataset_names_before_writing(self):
        client = FakeClient(
            datasets=[
                FakeResource("RADAR Caption Pilot Dataset", "dataset-one"),
                FakeResource("RADAR Caption Pilot Dataset", "dataset-two"),
            ]
        )

        with self.assertRaises(AmbiguousResourceError):
            ensure_pilot_resources(client)

        self.assertEqual(client.created, [])


class LabelboxPilotClientTests(unittest.TestCase):
    def setUp(self):
        self.sdk_client = FakeSdkClient()
        self.client = LabelboxPilotClient(
            self.sdk_client,
            project_type=FakeModel,
            dataset_type=FakeModel,
            video_media_type="video",
        )

    def test_searches_for_exact_names(self):
        self.client.find_projects("pilot project")
        self.client.find_datasets("pilot dataset")

        self.assertEqual(
            self.sdk_client.calls,
            [
                ("get_projects", ("equals", "pilot project")),
                ("get_datasets", ("equals", "pilot dataset")),
            ],
        )

    def test_creates_a_video_project_and_dataset(self):
        self.client.create_video_project("pilot project", "project description")
        self.client.create_dataset("pilot dataset", "dataset description")

        self.assertEqual(
            self.sdk_client.calls,
            [
                (
                    "create_project",
                    {
                        "name": "pilot project",
                        "description": "project description",
                        "media_type": "video",
                    },
                ),
                (
                    "create_dataset",
                    {
                        "name": "pilot dataset",
                        "description": "dataset description",
                        "iam_integration": None,
                    },
                ),
            ],
        )


class MainTests(unittest.TestCase):
    def test_dry_run_needs_no_key_or_client(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            status = main([], environ={}, client_factory=lambda _: self.fail("client created"))

        self.assertEqual(status, 0)
        self.assertIn("Dry run", stdout.getvalue())

    def test_apply_requires_an_api_key(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            status = main(["--apply"], environ={}, client_factory=lambda _: self.fail("client created"))

        self.assertEqual(status, 2)
        self.assertIn("LABELBOX_API_KEY is required", stderr.getvalue())

    def test_apply_uses_key_without_printing_it(self):
        secret = "secret-test-key"
        client = FakeClient()
        received = []
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            status = main(
                ["--apply"],
                environ={"LABELBOX_API_KEY": secret},
                client_factory=lambda key: received.append(key) or client,
            )

        self.assertEqual(status, 0)
        self.assertEqual(received, [secret])
        self.assertNotIn(secret, stdout.getvalue())
        self.assertIn("project: created", stdout.getvalue())
        self.assertIn("dataset: created", stdout.getvalue())

    def test_apply_sanitizes_api_errors(self):
        secret = "secret-test-key"
        stderr = io.StringIO()

        def fail(_):
            logging.getLogger("labelbox-test").error("server rejected %s at /sensitive/path", secret)
            raise RuntimeError(f"server rejected {secret} at /sensitive/path")

        with redirect_stderr(stderr):
            status = main(
                ["--apply"],
                environ={"LABELBOX_API_KEY": secret},
                client_factory=fail,
            )

        self.assertEqual(status, 1)
        self.assertEqual(stderr.getvalue(), "Labelbox setup failed (RuntimeError).\n")
        self.assertNotIn(secret, stderr.getvalue())
        self.assertNotIn("/sensitive/path", stderr.getvalue())

    def test_rejects_control_characters_in_resource_names(self):
        stderr = io.StringIO()

        with self.assertRaises(SystemExit), redirect_stderr(stderr):
            main(["--project-name", "unsafe\nname"], environ={})

        self.assertIn("contain no control characters", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
