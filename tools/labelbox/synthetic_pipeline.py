"""Exercise the simple RADAR review workflow in a Labelbox sandbox.

This command accepts no input media or clinical metadata. It generates four
small synthetic videos, imports the simple review proposal from PR #26,
uploads safe context attachments, and creates one labeling batch in resources
whose fixed names and descriptions identify them as synthetic-only.

Usage::

    python -m tools.labelbox.synthetic_pipeline
    python -m tools.labelbox.synthetic_pipeline --apply
    python -m tools.labelbox.synthetic_pipeline --export-summary summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from tools.labelbox.simple_review import build_form, score_review

SANDBOX_PROJECT_NAME = "RADAR Caption Pilot Sandbox"
SANDBOX_DATASET_NAME = "RADAR Caption Pilot Sandbox Dataset"
SANDBOX_ONTOLOGY_NAME = "RADAR simple review proposal v1 - synthetic sandbox"
SANDBOX_BATCH_NAME = "RADAR synthetic pipeline v1"
SANDBOX_PROJECT_DESCRIPTION = "Synthetic-only RADAR Labelbox pipeline test managed by EchoJEPA issue #22."
SANDBOX_DATASET_DESCRIPTION = "Generated synthetic videos only; no clinical data or source file paths."
GLOBAL_KEY_PREFIX = "radar-synthetic-v1-"
CANDIDATE_ATTACHMENT = "radar_candidate_v1"
SUMMARY_SCHEMA = "radar-synthetic-review-summary-v1"


class SandboxError(RuntimeError):
    """The sandbox is missing, ambiguous, incompatible, or failed remotely."""


@dataclass(frozen=True)
class SyntheticCase:
    """One entirely synthetic review item and its safe display context."""

    slug: str
    title: str
    scenario: str
    view: str
    modality: str
    claims: tuple[Mapping[str, Any], ...]
    marker: str
    marker_outside_crop: bool = False

    @property
    def global_key(self) -> str:
        return f"{GLOBAL_KEY_PREFIX}{self.slug}"

    def candidate(self) -> dict[str, Any]:
        return {
            "schema": "radar-simple-candidate-v1",
            "synthetic": True,
            "global_key": self.global_key,
            "view": self.view,
            "modality": self.modality,
            "claims": [dict(claim) for claim in self.claims],
            "input_provenance_verified": True,
            "crop_geometry": {"x": 48, "y": 16, "width": 224, "height": 224},
            "selected_frame_indices": list(range(12)),
        }


def synthetic_cases() -> tuple[SyntheticCase, ...]:
    """Return deterministic cases that cover supported, rejected, unresolved and routing behavior."""

    return (
        SyntheticCase(
            slug="supported",
            title="Synthetic supported claims",
            scenario="The two candidate findings are represented inside the model-input square.",
            view="A4C",
            modality="bmode",
            claims=({"concept": "lv_function", "value": "normal"}, {"concept": "lv_size", "value": "normal"},),
            marker="SUPPORTED",
        ),
        SyntheticCase(
            slug="unsupported",
            title="Synthetic unsupported claim",
            scenario="The candidate effusion claim is intentionally not represented in the synthetic loop.",
            view="PLAX",
            modality="bmode",
            claims=({"concept": "pericardial_effusion", "value": "present"},),
            marker="ABSENT",
        ),
        SyntheticCase(
            slug="outside-crop",
            title="Synthetic claim outside the model crop",
            scenario="The marker exists only outside the square and should be answered Cannot assess.",
            view="PLAX",
            modality="bmode",
            claims=({"concept": "pericardial_effusion", "value": "present"},),
            marker="OUTSIDE",
            marker_outside_crop=True,
        ),
        SyntheticCase(
            slug="wrong-modality",
            title="Synthetic modality mismatch",
            scenario="A color-flow claim is paired with a B-mode candidate to exercise routing safeguards.",
            view="A4C",
            modality="bmode",
            claims=({"concept": "mitral_regurgitant_jet", "value": "demonstrated"},),
            marker="MODE",
        ),
    )


@dataclass(frozen=True)
class Workspace:
    project: Any
    dataset: Any
    ontology: Any


@dataclass(frozen=True)
class PipelineResult:
    project_name: str
    dataset_name: str
    ontology_name: str
    total_rows: int
    uploaded_rows: int
    batch_created: bool


class SandboxApi(Protocol):
    """Operations isolated from orchestration so tests make no external requests."""

    def ensure_workspace(self, form: Mapping[str, Any], *, create: bool) -> Workspace:
        ...

    def existing_global_keys(self, workspace: Workspace, keys: Sequence[str]) -> set[str]:
        ...

    def upload_rows(self, workspace: Workspace, cases: Sequence[SyntheticCase], paths: Mapping[str, Path],) -> None:
        ...

    def ensure_batch(self, workspace: Workspace, global_keys: Sequence[str]) -> bool:
        ...

    def export_rows(self, workspace: Workspace) -> Sequence[Mapping[str, Any]]:
        ...


def _exactly_one(resources: Sequence[Any], kind: str, name: str) -> Any | None:
    matches = [resource for resource in resources if resource.name == name]
    if len(matches) > 1:
        raise SandboxError(f"Multiple {kind}s use the fixed sandbox name.")
    return matches[0] if matches else None


def _canonical_form(value: Any) -> Any:
    """Remove server-assigned schema identifiers before comparing forms."""

    if isinstance(value, str):
        value = json.loads(value)

    return _canonical_form_value(value)


def _canonical_form_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_canonical_form_value(item) for item in value]
    if isinstance(value, Mapping):
        ignored = {"schemaNodeId", "featureSchemaId", "attributes"}
        return {key: _canonical_form_value(item) for key, item in value.items() if key not in ignored}
    return value


class LabelboxSandboxApi:
    """Small adapter around the pinned Labelbox SDK."""

    def __init__(self, client, *, project_type, dataset_type, media_type):
        self.client = client
        self.project_type = project_type
        self.dataset_type = dataset_type
        self.media_type = media_type

    @classmethod
    def from_api_key(cls, api_key: str) -> "LabelboxSandboxApi":
        import labelbox as lb
        from labelbox.schema.dataset import Dataset
        from labelbox.schema.project import Project

        return cls(
            lb.Client(api_key=api_key), project_type=Project, dataset_type=Dataset, media_type=lb.MediaType.Video
        )

    def _resources(self) -> tuple[Any | None, Any | None, Any | None]:
        projects = list(self.client.get_projects(where=self.project_type.name == SANDBOX_PROJECT_NAME))
        datasets = list(self.client.get_datasets(where=self.dataset_type.name == SANDBOX_DATASET_NAME))
        ontologies = list(self.client.get_ontologies(name_contains=SANDBOX_ONTOLOGY_NAME))
        return (
            _exactly_one(projects, "project", SANDBOX_PROJECT_NAME),
            _exactly_one(datasets, "dataset", SANDBOX_DATASET_NAME),
            _exactly_one(ontologies, "ontology", SANDBOX_ONTOLOGY_NAME),
        )

    def _validate_existing(self, project, dataset, ontology, form: Mapping[str, Any]) -> None:
        if project is not None and (
            project.media_type != self.media_type or project.description != SANDBOX_PROJECT_DESCRIPTION
        ):
            raise SandboxError("The fixed-name project is not the synthetic sandbox.")
        if dataset is not None and (
            dataset.description != SANDBOX_DATASET_DESCRIPTION or dataset.iam_integration() is not None
        ):
            raise SandboxError("The fixed-name dataset is not the unconnected synthetic sandbox.")
        if ontology is not None and _canonical_form(ontology.normalized) != _canonical_form(form):
            raise SandboxError("The fixed-name ontology differs from the committed simple review form.")

    def ensure_workspace(self, form: Mapping[str, Any], *, create: bool) -> Workspace:
        project, dataset, ontology = self._resources()
        self._validate_existing(project, dataset, ontology, form)
        if not create and any(resource is None for resource in (project, dataset, ontology)):
            raise SandboxError("The synthetic sandbox has not been created.")
        if ontology is None:
            ontology = self.client.create_ontology(SANDBOX_ONTOLOGY_NAME, dict(form), media_type=self.media_type)
        if project is None:
            project = self.client.create_project(
                name=SANDBOX_PROJECT_NAME, description=SANDBOX_PROJECT_DESCRIPTION, media_type=self.media_type,
            )
        if dataset is None:
            dataset = self.client.create_dataset(
                name=SANDBOX_DATASET_NAME, description=SANDBOX_DATASET_DESCRIPTION, iam_integration=None,
            )
        connected = project.ontology()
        if connected is None:
            project.connect_ontology(ontology)
        elif connected.uid != ontology.uid:
            raise SandboxError("The synthetic sandbox project uses another ontology.")
        return Workspace(project=project, dataset=dataset, ontology=ontology)

    def existing_global_keys(self, workspace: Workspace, keys: Sequence[str]) -> set[str]:
        from lbox.exceptions import ResourceNotFoundError

        existing = set()
        for key in keys:
            try:
                row = self.client.get_data_row_by_global_key(key)
            except ResourceNotFoundError:
                continue
            if row.dataset().uid != workspace.dataset.uid:
                raise SandboxError("A synthetic global key belongs to another dataset.")
            existing.add(key)
        return existing

    def upload_rows(self, workspace: Workspace, cases: Sequence[SyntheticCase], paths: Mapping[str, Path],) -> None:
        rows = []
        for case in cases:
            url = self.client.upload_file(str(paths[case.global_key]))
            claims = "\n".join(
                f"{index}. {claim['concept']} = {claim['value']}" for index, claim in enumerate(case.claims, start=1)
            )
            rows.append(
                {
                    "row_data": url,
                    "global_key": case.global_key,
                    "external_id": case.global_key,
                    "attachments": [
                        {"type": "RAW_TEXT", "name": "Synthetic notice", "value": "SYNTHETIC DATA ONLY"},
                        {"type": "RAW_TEXT", "name": "Review scenario", "value": case.scenario},
                        {"type": "RAW_TEXT", "name": "Candidate view", "value": case.view},
                        {"type": "RAW_TEXT", "name": "Candidate modality", "value": case.modality},
                        {"type": "RAW_TEXT", "name": "Numbered candidate claims", "value": claims},
                        {
                            "type": "RAW_TEXT",
                            "name": CANDIDATE_ATTACHMENT,
                            "value": json.dumps(case.candidate(), ensure_ascii=True, sort_keys=True),
                        },
                    ],
                }
            )
        task = workspace.dataset.create_data_rows(rows)
        task.wait_till_done()
        if task.status == "FAILED" or task.errors:
            raise SandboxError("Synthetic data-row creation failed.")

    def ensure_batch(self, workspace: Workspace, global_keys: Sequence[str]) -> bool:
        batches = [batch for batch in workspace.project.batches() if batch.name == SANDBOX_BATCH_NAME]
        if len(batches) > 1:
            raise SandboxError("Multiple batches use the fixed sandbox name.")
        if batches:
            if batches[0].size != len(global_keys):
                raise SandboxError("The existing synthetic batch has an unexpected size.")
            return False
        workspace.project.create_batch(SANDBOX_BATCH_NAME, global_keys=list(global_keys))
        return True

    def export_rows(self, workspace: Workspace) -> Sequence[Mapping[str, Any]]:
        task = workspace.project.export(
            params={"label_details": True, "performance_details": True, "attachments": True}
        )
        task.wait_till_done()
        if task.status == "FAILED":
            raise SandboxError("Synthetic project export failed.")
        return list(task.result or [])


def _render_video(case: SyntheticCase, path: Path) -> None:
    """Generate a deterministic, non-clinical MP4 with the model crop drawn."""

    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw

    frames = []
    for frame_index in range(12):
        image = Image.new("RGB", (320, 256), color=(10, 18, 30))
        draw = ImageDraw.Draw(image)
        draw.rectangle((48, 16, 271, 239), outline=(255, 215, 0), width=3)
        draw.text((10, 8), "SYNTHETIC", fill=(255, 255, 255))
        draw.text((58, 26), case.marker, fill=(180, 220, 255))
        phase = frame_index if frame_index < 6 else 11 - frame_index
        if case.marker_outside_crop:
            x, y = 20 + phase, 118
        else:
            x, y = 130 + phase * 3, 110
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(40, 190, 220), outline=(255, 255, 255))
        draw.text((58, 218), "yellow square = model input", fill=(255, 215, 0))
        frames.append(np.asarray(image, dtype=np.uint8))
    iio.imwrite(path, np.stack(frames), fps=6, codec="libx264", pixelformat="yuv420p")
    if not path.is_file() or path.stat().st_size == 0:
        raise SandboxError("Synthetic video generation failed.")


def ensure_pipeline(
    api: SandboxApi,
    *,
    cases: Sequence[SyntheticCase] | None = None,
    renderer: Callable[[SyntheticCase, Path], None] = _render_video,
) -> PipelineResult:
    cases = tuple(cases or synthetic_cases())
    keys = [case.global_key for case in cases]
    if len(keys) != len(set(keys)) or any(not key.startswith(GLOBAL_KEY_PREFIX) for key in keys):
        raise SandboxError("Synthetic cases need unique, reserved global keys.")
    workspace = api.ensure_workspace(build_form(), create=True)
    existing = api.existing_global_keys(workspace, keys)
    missing = [case for case in cases if case.global_key not in existing]
    if missing:
        with tempfile.TemporaryDirectory(prefix="radar-synthetic-") as folder:
            paths = {}
            for case in missing:
                path = Path(folder) / f"{case.slug}.mp4"
                renderer(case, path)
                paths[case.global_key] = path
            api.upload_rows(workspace, missing, paths)
    batch_created = api.ensure_batch(workspace, keys)
    return PipelineResult(
        project_name=SANDBOX_PROJECT_NAME,
        dataset_name=SANDBOX_DATASET_NAME,
        ontology_name=SANDBOX_ONTOLOGY_NAME,
        total_rows=len(keys),
        uploaded_rows=len(missing),
        batch_created=batch_created,
    )


def _walk(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _candidate_from_export(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for item in _walk(row):
        if item.get("name") == CANDIDATE_ATTACHMENT and isinstance(item.get("value"), str):
            candidate = json.loads(item["value"])
            if not isinstance(candidate, Mapping) or candidate.get("synthetic") is not True:
                raise SandboxError("The sandbox export contains a non-synthetic candidate.")
            return candidate
    return None


def _latest_label(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    labels = []
    projects = row.get("projects")
    if isinstance(projects, Mapping):
        for project in projects.values():
            if isinstance(project, Mapping):
                labels.extend(label for label in project.get("labels", []) if isinstance(label, Mapping))
    if not labels:
        return None

    def created_at(label: Mapping[str, Any]) -> str:
        details = label.get("label_details")
        return str(details.get("created_at", "")) if isinstance(details, Mapping) else ""

    return max(labels, key=created_at)


def _answers_from_label(label: Mapping[str, Any]) -> dict[str, str]:
    answers = {}
    for item in _walk(label.get("annotations", {})):
        name = item.get("name")
        if not isinstance(name, str):
            continue
        radio = item.get("radio_answer")
        text = item.get("text_answer")
        if isinstance(radio, Mapping):
            value = radio.get("value") or radio.get("name")
            if isinstance(value, str):
                answers[name] = value
        elif isinstance(text, Mapping) and isinstance(text.get("content"), str):
            answers[name] = text["content"]
    return answers


def summarize_export(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert a raw sandbox export into a de-identified scorer summary."""

    items = []
    seen = set()
    for row in rows:
        candidate = _candidate_from_export(row)
        if candidate is None:
            continue
        key = candidate.get("global_key")
        if not isinstance(key, str) or not key.startswith(GLOBAL_KEY_PREFIX) or key in seen:
            raise SandboxError("The sandbox export has an invalid or repeated synthetic key.")
        seen.add(key)
        label = _latest_label(row)
        if label is None:
            items.append({"global_key": key, "status": "unreviewed"})
            continue
        score = score_review(candidate, _answers_from_label(label))
        items.append({"global_key": key, "status": "scored", "score": score})
    return {"schema": SUMMARY_SCHEMA, "items": sorted(items, key=lambda item: item["global_key"])}


def export_summary(api: SandboxApi, output: Path) -> int:
    workspace = api.ensure_workspace(build_form(), create=False)
    summary = summarize_export(api.export_rows(workspace))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sum(item["status"] == "scored" for item in summary["items"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exercise the RADAR simple-review workflow with synthetic data only.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--apply", action="store_true", help="Create/reuse the sandbox and upload missing synthetic rows."
    )
    action.add_argument(
        "--export-summary", type=Path, metavar="PATH", help="Export and score completed sandbox reviews."
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    api_factory: Callable[[str], SandboxApi] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if not args.apply and args.export_summary is None:
        print("Dry run: no Labelbox requests were made.")
        print(f"Would use synthetic-only project: {SANDBOX_PROJECT_NAME!r}")
        print(f"Would import ontology: {SANDBOX_ONTOLOGY_NAME!r}")
        print(f"Would generate and upload {len(synthetic_cases())} synthetic videos.")
        print("Run again with --apply after setting LABELBOX_API_KEY.")
        return 0

    api_key = environ.get("LABELBOX_API_KEY", "").strip()
    if not api_key:
        print("LABELBOX_API_KEY is required for Labelbox operations.", file=sys.stderr)
        return 2
    if api_factory is None:
        api_factory = LabelboxSandboxApi.from_api_key

    previous_log_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as discarded:
            with redirect_stdout(discarded), redirect_stderr(discarded):
                api = api_factory(api_key)
                if args.apply:
                    result = ensure_pipeline(api)
                else:
                    reviewed = export_summary(api, args.export_summary)
    except SandboxError as exc:
        print(f"Synthetic Labelbox pipeline stopped by safety check: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Synthetic Labelbox pipeline failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_log_disable)

    if args.apply:
        print(f"sandbox: ready; rows={result.total_rows}; uploaded={result.uploaded_rows}")
        print(f"batch: {'created' if result.batch_created else 'reused'}; name={SANDBOX_BATCH_NAME!r}")
    else:
        print(f"summary: written; reviewed={reviewed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
