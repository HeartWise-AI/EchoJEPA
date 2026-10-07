from dataclasses import dataclass
import unittest

from tools.labelbox.create_pilot_project import AmbiguousResourceError, ensure_pilot_resources


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


if __name__ == "__main__":
    unittest.main()
