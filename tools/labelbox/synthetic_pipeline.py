"""Exercise the complete DeepECHO column-review v2 workflow in a Labelbox sandbox.

This command accepts no input media or clinical metadata. It generates four
small synthetic videos, compiles fictional reports with the production v2
compiler from PR #26, imports the complete nested prelabels, and creates one
labeling batch per fictional study in resources whose fixed names and descriptions identify them
as synthetic-only.

Usage::

    python -m tools.labelbox.synthetic_pipeline
    python -m tools.labelbox.synthetic_pipeline --apply
    python -m tools.labelbox.synthetic_pipeline --export-summary summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from tools.labelbox.column_review import (
    build_candidate,
    build_form,
    context_attachment,
    prelabels,
    score_review,
)

SANDBOX_PROJECT_NAME = "RADAR Caption Pilot Sandbox"
SANDBOX_DATASET_NAME = "RADAR Caption Pilot Sandbox Dataset"
SANDBOX_ONTOLOGY_NAME = "DeepECHO column review v2 - editable view-mode synthetic sandbox"
LEGACY_SANDBOX_ONTOLOGY_NAMES = {
    "DeepECHO column review v2 - clip evidence synthetic sandbox",
    "DeepECHO column review v2 - compact finding labels synthetic sandbox",
    "DeepECHO column review v2 - reordered synthetic sandbox",
    "DeepECHO column review v2 - synthetic sandbox",
    "DeepECHO column review v2 - streamlined synthetic sandbox",
    "RADAR simple review proposal v1 - synthetic sandbox",
    "RADAR simple review proposal v2 - English synthetic sandbox",
    "RADAR simple review proposal v3 - preselected view synthetic sandbox",
}
SANDBOX_BATCH_PREFIX = "DeepECHO column review v2 synthetic study batch v1"
SANDBOX_PREDICTIONS_NAME = "DeepECHO column review v2 synthetic prelabels v8"
LEGACY_SANDBOX_PREDICTION_NAMES = {
    "DeepECHO column review v2 synthetic prelabels v6",
    "DeepECHO column review v2 synthetic prelabels v7",
}
SANDBOX_PROJECT_DESCRIPTION = "Synthetic-only RADAR Labelbox pipeline test managed by EchoJEPA issue #22."
SANDBOX_DATASET_DESCRIPTION = "Generated synthetic videos only; no clinical data or source file paths."
GLOBAL_KEY_PREFIX = "deepecho-v2-synthetic-v3-"
STUDY_KEY_PREFIX = "deepecho-v2-synthetic-study-"
CANDIDATE_ATTACHMENT = "deepecho_column_review_v2_candidate"
SUMMARY_SCHEMA = "deepecho-column-review-v2-synthetic-summary-v1"


class SandboxError(RuntimeError):
    """The sandbox is missing, ambiguous, incompatible, or failed remotely."""


@dataclass(frozen=True)
class SyntheticCase:
    """One entirely synthetic review item and its safe display context."""

    slug: str
    study_slug: str
    title: str
    scenario: str
    view: str
    modality: str
    report: Mapping[str, Any]
    marker: str
    targets: tuple[str, ...] | None = None
    level: str = "not_applicable"
    marker_outside_crop: bool = False

    @property
    def global_key(self) -> str:
        return f"{GLOBAL_KEY_PREFIX}{self.slug}"

    @property
    def study_key(self) -> str:
        return f"{STUDY_KEY_PREFIX}{self.study_slug}"

    @property
    def clip_key(self) -> str:
        return self.global_key

    def candidate(self) -> dict[str, Any]:
        candidate = build_candidate(
            self.report,
            self.view,
            self.modality,
            global_key=self.global_key,
            targets=list(self.targets) if self.targets is not None else None,
            level=self.level,
            input_kind="video",
            input_provenance_verified=True,
        )
        candidate.update(
            study_key=self.study_key,
            clip_key=self.clip_key,
            synthetic=True,
            synthetic_title=self.title,
            synthetic_scenario=self.scenario,
            crop_geometry={"x": 48, "y": 16, "width": 224, "height": 224},
            selected_frame_indices=list(range(12)),
        )
        return candidate


def _labelbox_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the safe candidate context stored with a synthetic data row."""

    safe = {
        key: value
        for key, value in candidate.items()
        if key != "report_context"
    }
    safe["columns"] = [
        {
            key: value
            for key, value in column.items()
            if key != "source_text"
        }
        for column in candidate["columns"]
    ]
    return safe


def _study_batch_plan(
    candidates: Sequence[Mapping[str, Any]], *, prefix: str
) -> dict[str, list[str]]:
    """Group clips by opaque study key without exposing that key in batch names."""

    grouped: dict[str, list[str]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["study_key"], []).append(candidate["global_key"])
    return {
        f"{prefix} {hashlib.sha256(study_key.encode()).hexdigest()[:12]}": keys
        for study_key, keys in sorted(grouped.items())
    }


def _validate_synthetic_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Validate production prelabels plus synthetic study and clip identities."""

    global_keys: set[str] = set()
    clip_keys: set[str] = set()
    for candidate in candidates:
        global_key = candidate.get("global_key")
        study_key = candidate.get("study_key")
        clip_key = candidate.get("clip_key")
        if (
            not isinstance(global_key, str)
            or not global_key.startswith(GLOBAL_KEY_PREFIX)
            or global_key in global_keys
            or not isinstance(study_key, str)
            or not study_key.startswith(STUDY_KEY_PREFIX)
            or not isinstance(clip_key, str)
            or clip_key in clip_keys
        ):
            raise ValueError("Synthetic candidates need unique reserved identities")
        global_keys.add(global_key)
        clip_keys.add(clip_key)
        prelabels(candidate)


def synthetic_cases() -> tuple[SyntheticCase, ...]:
    """Return fictional reports covering representative v2 routing and fields."""

    return (
        SyntheticCase(
            slug="a4c-chambers",
            study_slug="01",
            title="Synthetic A4C chamber and wall-motion proposals",
            scenario="UI-only example covering chamber function, size, wall motion and a valve proposal.",
            view="A4C",
            modality="bmode",
            report={
                "Left ventricule findings": (
                    "Dysfonction légère du ventricule gauche. // Ventricule gauche non-dilaté"
                ),
                "Right ventricule findings": "Fonction systolique normale du ventricule droit",
                "Tricuspid valve findings": "Valve tricuspide normale",
                "Visually Estimated EF": 48,
                "WMS Basal Inferoseptal": 2,
            },
            marker="A4C / CHAMBERS",
        ),
        SyntheticCase(
            slug="plax-morphology",
            study_slug="02",
            title="Synthetic PLAX valve and aorta proposals",
            scenario="UI-only example covering valve appearance, motion, aorta and pericardium fields.",
            view="PLAX",
            modality="bmode",
            report={
                "Mitral valve findings": "Feuillets épaissis. // Prolapsus mitral",
                "Aortic valve findings": "Valve aortique calcifiée avec ouverture réduite",
                "Aorta findings": "Dilatation de la racine aortique. // Aorte ascendante non-dilatée",
                "Pericardium findings": "Pas d’épanchement péricardique",
            },
            marker="PLAX / MORPHOLOGY",
        ),
        SyntheticCase(
            slug="a4c-color",
            study_slug="01",
            title="Synthetic A4C Color Doppler proposals",
            scenario="UI-only example covering demonstrated and not-demonstrated regurgitant jets.",
            view="A4C",
            modality="color_flow",
            report={
                "Mitral valve findings": "Insuffisance mitrale légère",
                "Tricuspid valve findings": "Pas d’insuffisance tricuspidienne",
            },
            marker="A4C / COLOR",
        ),
        SyntheticCase(
            slug="psax-av",
            study_slug="03",
            title="Synthetic PSAX aortic-valve proposals",
            scenario="UI-only example covering aortic cusp count, appearance and leaflet motion.",
            view="PSAX_AV",
            modality="bmode",
            report={
                "Aortic valve findings": "Valve aortique tricuspide, calcifiée, avec ouverture réduite",
                "Tricuspid valve findings": "Valve tricuspide normale",
                "Pulmonary valve findings": "Valve pulmonaire normale",
            },
            marker="PSAX AV",
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
    project_id: str
    dataset_name: str
    ontology_name: str
    total_rows: int
    uploaded_rows: int
    total_study_batches: int
    created_study_batches: int
    prelabels_imported: bool


class SandboxApi(Protocol):
    """Operations isolated from orchestration so tests make no external requests."""

    def ensure_workspace(self, form: Mapping[str, Any], *, create: bool) -> Workspace:
        ...

    def existing_global_keys(self, workspace: Workspace, keys: Sequence[str]) -> set[str]:
        ...

    def upload_rows(self, workspace: Workspace, cases: Sequence[SyntheticCase], paths: Mapping[str, Path],) -> None:
        ...

    def ensure_study_batches(self, workspace: Workspace, cases: Sequence[SyntheticCase]) -> int:
        ...

    def ensure_prelabels(
        self,
        workspace: Workspace,
        cases: Sequence[SyntheticCase],
        *,
        refresh: bool = False,
    ) -> bool:
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

    canonical = _canonical_form_value(value)
    if isinstance(canonical, Mapping):
        root_metadata = {"files", "id", "name", "relationships"}
        canonical = {key: item for key, item in canonical.items() if key not in root_metadata}
    return canonical


def _canonical_form_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_canonical_form_value(item) for item in value]
    if isinstance(value, Mapping):
        ignored = {"schemaNodeId", "featureSchemaId", "kind"}
        canonical = {}
        for key, item in value.items():
            if key in ignored:
                continue
            if key == "archived" and item in (False, 0, None):
                continue
            if key == "attributes" and item is None:
                continue
            normalized = _canonical_form_value(item)
            if key == "options" and normalized == []:
                continue
            canonical[key] = normalized
        return canonical
    return value


def _difference_paths(expected: Any, actual: Any, path: str = "root", limit: int = 8) -> list[str]:
    """Return bounded schema paths only, without exposing ontology values."""

    if limit <= 0:
        return []
    if type(expected) is not type(actual):
        return [path]
    if isinstance(expected, Mapping):
        paths = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected or key not in actual:
                paths.append(child)
            else:
                paths.extend(_difference_paths(expected[key], actual[key], child, limit - len(paths)))
            if len(paths) >= limit:
                break
        return paths
    if isinstance(expected, list):
        paths = []
        if len(expected) != len(actual):
            paths.append(path)
        for index, (left, right) in enumerate(zip(expected, actual)):
            paths.extend(_difference_paths(left, right, f"{path}[{index}]", limit - len(paths)))
            if len(paths) >= limit:
                break
        return paths
    return [] if expected == actual else [path]


def _safe_error_summary(errors: Any) -> str:
    """Return bounded server messages with URLs, paths and opaque tokens removed."""

    messages = []
    for item in _walk(errors):
        for key in ("message", "error", "description", "reason"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            value = re.sub(r"https?://[^\s'\"]+", "<url>", value)
            value = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s/'\"]+/)*[^\s/'\"]+", "<path>", value)
            value = re.sub(r"\b[A-Za-z0-9_-]{32,}\b", "<token>", value)
            messages.append(" ".join(value.split())[:240])
            if len(messages) == 3:
                break
        if len(messages) == 3:
            break
    return "; ".join(dict.fromkeys(messages)) or "no safe server message available"


def _schema_id_predictions(
    payloads: Sequence[Mapping[str, Any]], normalized_ontology: Any
) -> list[dict[str, Any]]:
    """Replace stable column-review names with connected server schema IDs."""

    if isinstance(normalized_ontology, str):
        normalized_ontology = json.loads(normalized_ontology)

    def unique(nodes: Sequence[Mapping[str, Any]], key: str, value: Any) -> Mapping[str, Any]:
        matches = [node for node in nodes if node.get(key) == value]
        if len(matches) != 1:
            raise ValueError("Connected ontology has no unique feature for a v2 prelabel")
        return matches[0]

    def convert_answer(
        selected: Mapping[str, Any], options: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        matches = [
            option
            for option in options
            if selected.get("name") in {option.get("value"), option.get("name")}
        ]
        if len(matches) != 1:
            raise ValueError("Connected ontology has no unique answer for a v2 prelabel")
        option = matches[0]
        schema_id = option.get("featureSchemaId")
        if not schema_id:
            raise ValueError("Connected ontology answer is missing its server schema ID")
        converted = {
            key: value
            for key, value in selected.items()
            if key not in {"name", "classifications"}
        }
        converted["schemaId"] = schema_id
        children = selected.get("classifications", [])
        if children:
            converted["classifications"] = [
                convert_question(child, option.get("options", [])) for child in children
            ]
        return converted

    def convert_question(
        record: Mapping[str, Any], questions: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        question = unique(questions, "name", record.get("name"))
        schema_id = question.get("featureSchemaId")
        if not schema_id:
            raise ValueError("Connected ontology question is missing its server schema ID")
        converted = {
            key: value for key, value in record.items() if key not in {"name", "answer"}
        }
        converted["schemaId"] = schema_id
        answer = record.get("answer")
        if isinstance(answer, list):
            converted["answer"] = [
                convert_answer(selected, question.get("options", [])) for selected in answer
            ]
        elif isinstance(answer, Mapping):
            converted["answer"] = convert_answer(answer, question.get("options", []))
        else:
            converted["answer"] = answer
        return converted

    questions = normalized_ontology.get("classifications", [])
    return [convert_question(payload, questions) for payload in payloads]


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
        if ontology is not None:
            expected = _canonical_form(form)
            actual = _canonical_form(ontology.normalized)
            if actual != expected:
                paths = ", ".join(_difference_paths(expected, actual))
                raise SandboxError(
                    f"The fixed-name ontology differs from the committed column-review v2 form at: {paths}."
                )

    @staticmethod
    def _connect_expected_ontology(project, ontology) -> None:
        connected = project.ontology()
        if connected is None:
            project.connect_ontology(ontology)
            return
        if connected.uid == ontology.uid:
            return
        if connected.name in LEGACY_SANDBOX_ONTOLOGY_NAMES and project.get_label_count() == 0:
            project.connect_ontology(ontology)
            return
        raise SandboxError("The synthetic sandbox project uses another ontology or already has labels.")

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
        self._connect_expected_ontology(project, ontology)
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
            candidate = case.candidate()
            safe_candidate = _labelbox_candidate(candidate)
            context = context_attachment(candidate)
            rows.append(
                {
                    "row_data": url,
                    "global_key": case.global_key,
                    "external_id": case.global_key,
                    "attachments": [
                        {"type": "RAW_TEXT", "name": "Synthetic notice", "value": "SYNTHETIC DATA ONLY"},
                        {"type": "RAW_TEXT", "name": "Review scenario", "value": case.scenario},
                        {
                            "type": context["attachment_type"],
                            "name": context["attachment_name"],
                            "value": context["attachment_value"],
                        },
                        {
                            "type": "RAW_TEXT",
                            "name": CANDIDATE_ATTACHMENT,
                            "value": json.dumps(safe_candidate, ensure_ascii=True, sort_keys=True),
                        },
                    ],
                }
            )
        task = workspace.dataset.create_data_rows(rows)
        task.wait_till_done()
        errors = task.errors
        if task.status == "FAILED" or errors:
            raise SandboxError(f"Synthetic data-row creation failed: {_safe_error_summary(errors)}.")

    def ensure_study_batches(self, workspace: Workspace, cases: Sequence[SyntheticCase]) -> int:
        plan = _study_batch_plan(
            [case.candidate() for case in cases], prefix=SANDBOX_BATCH_PREFIX
        )
        existing = list(workspace.project.batches())
        created = 0
        for name, global_keys in plan.items():
            batches = [batch for batch in existing if batch.name == name]
            if len(batches) > 1:
                raise SandboxError("Multiple batches use a fixed synthetic study-batch name.")
            if batches:
                if batches[0].size != len(global_keys):
                    if workspace.project.get_label_count() != 0:
                        raise SandboxError(
                            "A stale synthetic study batch cannot be replaced because "
                            "the project has submitted labels."
                        )
                    batches[0].delete()
                else:
                    continue
            workspace.project.create_batch(name, global_keys=global_keys)
            created += 1
        return created

    def ensure_prelabels(
        self,
        workspace: Workspace,
        cases: Sequence[SyntheticCase],
        *,
        refresh: bool = False,
    ) -> bool:
        from labelbox.schema.annotation_import import MALPredictionImport
        from labelbox.schema.enums import AnnotationImportState

        all_imports = list(workspace.project.get_mal_prediction_imports())
        imports = [
            item
            for item in all_imports
            if item.name == SANDBOX_PREDICTIONS_NAME
        ]
        if len(imports) > 1:
            raise SandboxError("Multiple imports use the fixed synthetic v2 prelabel name.")
        legacy_imports = [item for item in all_imports if item.name in LEGACY_SANDBOX_PREDICTION_NAMES]
        current_failed = False
        if imports:
            imports[0].wait_till_done()
            current_failed = imports[0].state == AnnotationImportState.FAILED or bool(imports[0].errors)
        obsolete = legacy_imports + (imports if current_failed or refresh else [])
        if obsolete:
            if workspace.project.get_label_count() != 0:
                raise SandboxError("Synthetic prelabel migration refused because the project has submitted labels.")
            for item in obsolete:
                item.wait_till_done()
                item.delete()
            if current_failed or refresh:
                imports = []
        workspace.project.enable_model_assisted_labeling(True)
        if imports:
            return False

        named_predictions = [prediction for case in cases for prediction in prelabels(case.candidate())]
        predictions = _schema_id_predictions(named_predictions, workspace.ontology.normalized)
        job = MALPredictionImport.create_from_objects(
            client=self.client,
            project_id=workspace.project.uid,
            name=SANDBOX_PREDICTIONS_NAME,
            predictions=predictions,
        )
        job.wait_till_done()
        errors = job.errors
        if job.state == AnnotationImportState.FAILED or errors:
            raise SandboxError(f"Synthetic v2 prelabel import failed: {_safe_error_summary(errors)}.")
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
    candidates = [case.candidate() for case in cases]
    try:
        _validate_synthetic_candidates(candidates)
    except ValueError as exc:
        raise SandboxError("Synthetic cases need unique, valid clip identities.") from exc
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
    batch_plan = _study_batch_plan(candidates, prefix=SANDBOX_BATCH_PREFIX)
    created_study_batches = api.ensure_study_batches(workspace, cases)
    prelabels_imported = api.ensure_prelabels(
        workspace,
        cases,
        refresh=created_study_batches > 0,
    )
    return PipelineResult(
        project_name=SANDBOX_PROJECT_NAME,
        project_id=workspace.project.uid,
        dataset_name=SANDBOX_DATASET_NAME,
        ontology_name=SANDBOX_ONTOLOGY_NAME,
        total_rows=len(keys),
        uploaded_rows=len(missing),
        total_study_batches=len(batch_plan),
        created_study_batches=created_study_batches,
        prelabels_imported=prelabels_imported,
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


def _answers_from_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten submitted Labelbox radio, checklist and text answers for the v2 scorer."""

    answers: dict[str, Any] = {}

    def visit(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if not isinstance(name, str):
                continue
            radio = item.get("radio_answer")
            checklist = item.get("checklist_answers")
            text = item.get("text_answer")
            if isinstance(radio, Mapping):
                value = radio.get("value") or radio.get("name")
                if isinstance(value, str):
                    answers[name] = value
                visit(radio.get("classifications"))
            elif isinstance(checklist, list):
                values = []
                for selected in checklist:
                    if not isinstance(selected, Mapping):
                        continue
                    value = selected.get("value") or selected.get("name")
                    if isinstance(value, str):
                        values.append(value)
                    visit(selected.get("classifications"))
                answers[name] = values
            elif isinstance(text, Mapping) and isinstance(text.get("content"), str):
                answers[name] = text["content"]

    annotations = label.get("annotations", {})
    if isinstance(annotations, Mapping):
        visit(annotations.get("classifications"))
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
            items.append({
                "global_key": key,
                "study_key": candidate["study_key"],
                "clip_key": candidate["clip_key"],
                "status": "unreviewed",
            })
            continue
        score = score_review(candidate, _answers_from_label(label), submitted=True)
        items.append({
            "global_key": key,
            "study_key": candidate["study_key"],
            "clip_key": candidate["clip_key"],
            "status": "scored",
            "score": score,
        })
    return {"schema": SUMMARY_SCHEMA, "items": sorted(items, key=lambda item: item["global_key"])}


def export_summary(api: SandboxApi, output: Path) -> int:
    workspace = api.ensure_workspace(build_form(), create=False)
    summary = summarize_export(api.export_rows(workspace))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sum(item["status"] == "scored" for item in summary["items"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exercise DeepECHO column-review v2 with synthetic data only.")
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
        print(f"Would import the complete 46-variable ontology: {SANDBOX_ONTOLOGY_NAME!r}")
        print(f"Would generate and upload {len(synthetic_cases())} synthetic videos.")
        print("Would import view, modality, targets, clinical proposals and caption as v2 prelabels.")
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
        print(
            f"sandbox: ready; project={result.project_name!r}; id={result.project_id}; "
            f"rows={result.total_rows}; uploaded={result.uploaded_rows}"
        )
        print(
            f"study batches: total={result.total_study_batches}; "
            f"created={result.created_study_batches}; "
            f"reused={result.total_study_batches - result.created_study_batches}"
        )
        print(f"v2 prelabels: {'imported' if result.prelabels_imported else 'existing import reused'}")
    else:
        print(f"summary: written; reviewed={reviewed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
