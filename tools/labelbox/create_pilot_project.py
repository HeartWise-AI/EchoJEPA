"""Create or reuse the Labelbox resources for the RADAR caption pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


DEFAULT_PROJECT_NAME = "RADAR Caption Pilot"
DEFAULT_DATASET_NAME = "RADAR Caption Pilot Dataset"
PROJECT_DESCRIPTION = "View-specific RADAR caption pilot managed by EchoJEPA issue #18."
DATASET_DESCRIPTION = "Synthetic or approved de-identified videos for the RADAR caption pilot."


class RemoteResource(Protocol):
    """The safe subset of a Labelbox project or dataset used by this tool."""

    uid: str
    name: str


class PilotClient(Protocol):
    """Operations required from Labelbox, separated for offline unit testing."""

    def find_projects(self, name: str) -> Sequence[RemoteResource]:
        ...

    def find_datasets(self, name: str) -> Sequence[RemoteResource]:
        ...

    def create_video_project(self, name: str, description: str) -> RemoteResource:
        ...

    def create_dataset(self, name: str, description: str) -> RemoteResource:
        ...


class AmbiguousResourceError(RuntimeError):
    """Raised when a name identifies more than one Labelbox resource."""


@dataclass(frozen=True)
class ResourceResult:
    """A safe summary of a project or dataset creation decision."""

    name: str
    uid: str
    created: bool


@dataclass(frozen=True)
class PilotResult:
    """The resources that make up the Labelbox pilot workspace."""

    project: ResourceResult
    dataset: ResourceResult


def _validate_unique(resources: Sequence[RemoteResource], kind: str, name: str) -> None:
    if len(resources) > 1:
        raise AmbiguousResourceError(
            f"Found multiple Labelbox {kind}s named {name!r}; rename or remove duplicates before retrying."
        )


def _result(resource: RemoteResource, *, created: bool) -> ResourceResult:
    return ResourceResult(name=resource.name, uid=resource.uid, created=created)


def ensure_pilot_resources(
    client: PilotClient, *, project_name: str = DEFAULT_PROJECT_NAME, dataset_name: str = DEFAULT_DATASET_NAME,
) -> PilotResult:
    """Create missing pilot resources and reuse unambiguous exact-name matches.

    Both names are checked before the first write. This prevents a known
    duplicate from leaving behind a partially created project or dataset.
    """

    projects = list(client.find_projects(project_name))
    datasets = list(client.find_datasets(dataset_name))
    _validate_unique(projects, "project", project_name)
    _validate_unique(datasets, "dataset", dataset_name)

    if projects:
        project = _result(projects[0], created=False)
    else:
        project = _result(client.create_video_project(project_name, PROJECT_DESCRIPTION), created=True,)

    if datasets:
        dataset = _result(datasets[0], created=False)
    else:
        dataset = _result(client.create_dataset(dataset_name, DATASET_DESCRIPTION), created=True,)

    return PilotResult(project=project, dataset=dataset)
