"""Create or reuse the Labelbox resources for the RADAR caption pilot."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

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

    def find_projects(self, name: str) -> Sequence[RemoteResource]: ...

    def find_datasets(self, name: str) -> Sequence[RemoteResource]: ...

    def create_video_project(self, name: str, description: str) -> RemoteResource: ...

    def create_dataset(self, name: str, description: str) -> RemoteResource: ...


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


class LabelboxPilotClient:
    """Small adapter around the Labelbox SDK used by the pilot setup."""

    def __init__(self, client, *, project_type, dataset_type, video_media_type):
        self._client = client
        self._project_type = project_type
        self._dataset_type = dataset_type
        self._video_media_type = video_media_type

    @classmethod
    def from_api_key(cls, api_key: str) -> "LabelboxPilotClient":
        # Keep Labelbox optional for training and for credential-free CI tests.
        import labelbox as lb
        from labelbox.schema.dataset import Dataset
        from labelbox.schema.project import Project

        return cls(
            lb.Client(api_key=api_key),
            project_type=Project,
            dataset_type=Dataset,
            video_media_type=lb.MediaType.Video,
        )

    def find_projects(self, name: str) -> Sequence[RemoteResource]:
        return list(self._client.get_projects(where=self._project_type.name == name))

    def find_datasets(self, name: str) -> Sequence[RemoteResource]:
        return list(self._client.get_datasets(where=self._dataset_type.name == name))

    def create_video_project(self, name: str, description: str) -> RemoteResource:
        return self._client.create_project(
            name=name,
            description=description,
            media_type=self._video_media_type,
        )

    def create_dataset(self, name: str, description: str) -> RemoteResource:
        # Do not silently attach the organization's default cloud storage.
        # Storage can be connected later after the governance review.
        return self._client.create_dataset(
            name=name,
            description=description,
            iam_integration=None,
        )


def _validate_unique(resources: Sequence[RemoteResource], kind: str, name: str) -> None:
    if len(resources) > 1:
        raise AmbiguousResourceError(
            f"Found multiple Labelbox {kind}s named {name!r}; rename or remove duplicates before retrying."
        )


def _result(resource: RemoteResource, *, created: bool) -> ResourceResult:
    return ResourceResult(name=resource.name, uid=resource.uid, created=created)


def ensure_pilot_resources(
    client: PilotClient,
    *,
    project_name: str = DEFAULT_PROJECT_NAME,
    dataset_name: str = DEFAULT_DATASET_NAME,
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
        project = _result(
            client.create_video_project(project_name, PROJECT_DESCRIPTION),
            created=True,
        )

    if datasets:
        dataset = _result(datasets[0], created=False)
    else:
        dataset = _result(
            client.create_dataset(dataset_name, DATASET_DESCRIPTION),
            created=True,
        )

    return PilotResult(project=project, dataset=dataset)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or reuse the Labelbox resources for the RADAR caption pilot.",
    )
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME, type=_resource_name)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, type=_resource_name)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Connect to Labelbox and create missing resources. Without this flag, only print the plan.",
    )
    return parser


def _resource_name(value: str) -> str:
    value = value.strip()
    if not value or any(not character.isprintable() for character in value):
        raise argparse.ArgumentTypeError("resource names must be non-empty and contain no control characters")
    return value


def _print_result(kind: str, resource: ResourceResult) -> None:
    action = "created" if resource.created else "reused"
    print(f"{kind}: {action}; name={resource.name!r}; id={resource.uid}")


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    client_factory: Callable[[str], PilotClient] | None = None,
) -> int:
    args = _parser().parse_args(argv)

    if not args.apply:
        print("Dry run: no Labelbox requests were made.")
        print(f"Would ensure video project: {args.project_name!r}")
        print(f"Would ensure dataset: {args.dataset_name!r}")
        print("Run again with --apply after setting LABELBOX_API_KEY.")
        return 0

    api_key = environ.get("LABELBOX_API_KEY", "").strip()
    if not api_key:
        print("LABELBOX_API_KEY is required with --apply.", file=sys.stderr)
        return 2

    if client_factory is None:
        client_factory = LabelboxPilotClient.from_api_key

    previous_log_disable = logging.root.manager.disable
    try:
        # Third-party API errors may include request or storage details. Keep
        # the operation silent and emit a sanitized result below.
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as discarded_output:
            with redirect_stdout(discarded_output), redirect_stderr(discarded_output):
                result = ensure_pilot_resources(
                    client_factory(api_key),
                    project_name=args.project_name,
                    dataset_name=args.dataset_name,
                )
    except Exception as exc:
        # API errors can contain request details. Keep logs safe and actionable
        # without reproducing server messages, credentials, or storage paths.
        print(f"Labelbox setup failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_log_disable)

    _print_result("project", result.project)
    _print_result("dataset", result.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
