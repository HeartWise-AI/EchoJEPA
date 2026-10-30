"""Upload prepared real clips to a RADAR review workspace in Labelbox, and optionally remove them afterward.

Reads playable clips (`clips/<global key>.mp4`) and review items (`candidates.json`, `view_type_items.csv`,
`global_keys.csv`) prepared by `tools.labelbox.workflow_test` (a workflow test) or `tools.labelbox.pilot_prepare`
(the pilot). Run in the Labelbox environment:

    python -m tools.labelbox.pilot_upload --folder TEST                       # dry run: checks and preview only
    python -m tools.labelbox.pilot_upload --folder TEST --apply               # upload, batch, prelabel
    python -m tools.labelbox.pilot_upload --folder TEST --apply --replace-form  # an empty project gets the new form
    python -m tools.labelbox.pilot_upload --folder TEST --export-summary OUT  # score the submitted reviews
    python -m tools.labelbox.pilot_upload --folder TEST --delete              # list what this folder uploaded
    python -m tools.labelbox.pilot_upload --folder TEST --delete --apply      # delete it

Use `--form column_review` or `--form view_type_review` to target one form. The folder's keys decide its workspace:
a workflow test's go to the test projects, the pilot's to the pilot projects (`--workspace` may only confirm it). A
pilot upload is deleted only while its project holds no submitted review.

Validation and privacy: all checks run before uploading, including during dry runs. Each run writes
`TEST/upload_preview.json` with the exact upload metadata. Only playable copies, opaque keys, and
attachments leave the server; report text and source clauses remain local.

Compatibility: reuses `synthetic_pipeline` conventions for form validation, batch naming, schema-ID
prelabels, and review exports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Mapping, Sequence

from tools.labelbox import column_review, view_type_review
from tools.labelbox.create_pilot_project import (
    DATASET_DESCRIPTION,
    DEFAULT_DATASET_NAME,
    DEFAULT_PROJECT_NAME,
    PROJECT_DESCRIPTION,
)
from tools.labelbox.import_column_review import ONTOLOGY_NAME as COLUMN_REVIEW_FORM_NAME
from tools.labelbox.import_column_review import VERSION as COLUMN_REVIEW_VERSION
from tools.labelbox.synthetic_pipeline import (
    _canonical_form,
    _difference_paths,
    _latest_label,
    _safe_error_summary,
    _schema_id_predictions,
    _study_batch_plan,
)

COLUMN_REVIEW, VIEW_TYPE_REVIEW = "column_review", "view_type_review"
TEST_KEY_PREFIX = "radar-workflow-test-"
STUDY_KEY_PREFIX = "radar-workflow-test-study-"
GLOBAL_KEY = re.compile(re.escape(TEST_KEY_PREFIX) + r"[0-9a-f]{24}")
STUDY_KEY = re.compile(re.escape(STUDY_KEY_PREFIX) + r"[0-9a-f]{24}")
CLIPS, ITEMS, PREVIEW = "clips", "items", "upload_preview.json"

BATCH_PREFIX = "RADAR workflow test study"
PREDICTIONS_NAME = f"DeepECHO column review v{COLUMN_REVIEW_VERSION} workflow test prelabels v1"
NOTICE = {"type": "RAW_TEXT", "name": "Workflow test",
          "value": "Workflow test of the RADAR review (EchoJEPA issue #22) on a real de-identified clip. "
                   "Reviews made here are not pilot data; the clip is deleted after the test."}
SUMMARY_SCHEMA = "radar-workflow-test-summary-v1"
# The pilot's keys: `data/radar_pilot.py` names each clip `radar-pilot-<HMAC digest>`, and `pilot_prepare` each study.
PILOT_KEY_PREFIX = "radar-pilot-"
PILOT_STUDY_KEY_PREFIX = "radar-pilot-study-"
# Seconds between attempts to delete a study batch whose labels Labelbox is still deleting, about 5 minutes in all.
BATCH_DELETE_WAITS = (5, 10, 15, 30, 60, 60, 60, 60)


@dataclass(frozen=True)
class Kind:
    """A kind of folder: the keys of its clips and studies, the workspace it goes to, and the names its batches,
    prelabel import, notice and summary carry there."""

    name: str
    workspace: str
    global_key: re.Pattern
    study_key: re.Pattern
    batch_prefix: str
    predictions_name: str
    notice: Mapping[str, str]
    summary_schema: str
    keeps_reviews: bool  # deleting the upload is refused once its project holds a submitted review.


WORKFLOW_TEST = Kind("workflow test", "test", GLOBAL_KEY, STUDY_KEY, BATCH_PREFIX, PREDICTIONS_NAME, NOTICE,
                     SUMMARY_SCHEMA, keeps_reviews=False)
PILOT = Kind(
    "pilot", "pilot", re.compile(re.escape(PILOT_KEY_PREFIX) + r"[0-9a-f]{24}"),
    re.compile(re.escape(PILOT_STUDY_KEY_PREFIX) + r"[0-9a-f]{24}"), "RADAR pilot study",
    f"DeepECHO column review v{COLUMN_REVIEW_VERSION} pilot prelabels v1",
    {"type": "RAW_TEXT", "name": "RADAR caption pilot",
     "value": "RADAR caption pilot (EchoJEPA issue #22) on a real de-identified clip. "
              "Reviews made here are pilot data."},
    "radar-pilot-summary-v1", keeps_reviews=True)
KINDS = (WORKFLOW_TEST, PILOT)


@dataclass(frozen=True)
class Target:
    """The Labelbox project, dataset and form of one review form."""

    project_name: str
    dataset_name: str
    ontology_name: str
    build_form: Callable[[], dict]


FORMS = (COLUMN_REVIEW, VIEW_TYPE_REVIEW)
WORKSPACES = {
    "test": {
        # The project keeps its name across form versions: `--replace-form` gives it a new form once emptied.
        COLUMN_REVIEW: Target("RADAR Workflow Test - column review", "RADAR Workflow Test - column review dataset",
                              COLUMN_REVIEW_FORM_NAME, column_review.build_form),
        VIEW_TYPE_REVIEW: Target("RADAR Workflow Test - view and type review v1",
                                 "RADAR Workflow Test - view and type dataset", view_type_review.FORM_NAME,
                                 view_type_review.build_form),
    },
    "pilot": {
        COLUMN_REVIEW: Target(DEFAULT_PROJECT_NAME, DEFAULT_DATASET_NAME, COLUMN_REVIEW_FORM_NAME,
                              column_review.build_form),
        VIEW_TYPE_REVIEW: Target("RADAR Caption Pilot - view and type review",
                                 "RADAR Caption Pilot - view and type dataset", view_type_review.FORM_NAME,
                                 view_type_review.build_form),
    },
}


class UploadError(RuntimeError):
    """The folder or a Labelbox resource fails a check, or Labelbox reported a failure."""


class StepFailed(RuntimeError):
    """A Labelbox request raised: names the step and the error type, never the server's message."""

    def __init__(self, step: str, kind: str):
        super().__init__(f"{step} ({kind})")
        self.step, self.kind = step, kind


@contextmanager
def _step(name: str) -> Iterator[None]:
    """Report an SDK error raised inside as a failure of this named step; checks pass through unchanged."""
    try:
        yield
    except (UploadError, StepFailed):
        raise
    except Exception as exc:
        raise StepFailed(name, type(exc).__name__) from exc


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Row:
    global_key: str
    form: str
    study_key: str
    path: Path
    sha256: str
    attachments: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class Plan:
    rows: tuple[Row, ...]
    candidates: Mapping[str, Mapping[str, Any]]  # column review candidates by global key.
    kind: Kind = WORKFLOW_TEST

    def forms(self) -> list[str]:
        return [form for form in FORMS if self.rows_of(form)]

    def rows_of(self, form: str) -> list[Row]:
        return [row for row in self.rows if row.form == form]

    def batches(self, form: str) -> dict[str, list[str]]:
        rows = [{"study_key": row.study_key, "global_key": row.global_key} for row in self.rows_of(form)]
        return _study_batch_plan(rows, prefix=self.kind.batch_prefix)

    def only(self, form: str) -> "Plan":
        return Plan(rows=tuple(self.rows_of(form)), candidates=self.candidates if form == COLUMN_REVIEW else {},
                    kind=self.kind)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_plan(folder: Path) -> Plan:
    """The rows to upload, after every check that needs no request."""
    items = folder / ITEMS
    try:
        candidates = json.loads((items / "candidates.json").read_text(encoding="utf-8"))
        other = [row["global_key"] for row in _read_csv(items / "view_type_items.csv")]
        listed = _read_csv(items / "global_keys.csv")
    except FileNotFoundError:
        raise UploadError("The folder has no review items: run `workflow_test items` or `pilot_prepare` first.")
    except KeyError:
        raise UploadError("view_type_items.csv needs a global_key column.")
    if not listed:
        raise UploadError("global_keys.csv lists no clip.")
    keys = [row.get("global_key", "") for row in listed]
    kinds = [kind for kind in KINDS if all(kind.global_key.fullmatch(key) for key in keys)]
    if len(keys) != len(set(keys)) or len(kinds) != 1:
        raise UploadError("Every listed clip needs its own global key, all of a workflow test or all of the pilot.")
    kind = kinds[0]
    if any(row.get("form") not in FORMS or not kind.study_key.fullmatch(row.get("study_key", "")) for row in listed):
        raise UploadError(f"Every listed clip needs a known review form and an opaque {kind.name} study key.")
    forms = {row["global_key"]: row["form"] for row in listed}
    candidate_keys = [candidate.get("global_key") for candidate in candidates]
    for form, keys_of_form in ((COLUMN_REVIEW, candidate_keys), (VIEW_TYPE_REVIEW, other)):
        if len(keys_of_form) != len(set(keys_of_form)) or set(keys_of_form) != {
                key for key, listed_form in forms.items() if listed_form == form}:
            raise UploadError(f"The {form} items and the clips global_keys.csv lists for that form differ.")

    by_key = {}
    for candidate in candidates:
        if candidate.get("input_kind") != "video":
            raise UploadError("Every column review candidate must describe a video.")
        if candidate.get("version") != COLUMN_REVIEW_VERSION:
            raise UploadError(f"The candidates were built for column review v{candidate.get('version')}, not the "
                              f"committed v{COLUMN_REVIEW_VERSION}: build them again with this code.")
        try:
            column_review.prelabels(candidate)  # validates the candidate against the form.
        except (ValueError, KeyError) as exc:
            raise UploadError(f"A column review candidate is invalid ({type(exc).__name__}).") from exc
        by_key[candidate["global_key"]] = candidate

    rows = []
    for row in listed:
        key = row["global_key"]
        path = folder / CLIPS / f"{key}.mp4"
        if not path.is_file() or sha256(path) != row.get("copy_sha256"):
            raise UploadError(f"The copy of {key} is missing or differs from global_keys.csv.")
        attachments = [kind.notice]
        if row["form"] == COLUMN_REVIEW:
            context = column_review.context_attachment(by_key[key])
            attachments.append({"type": context["attachment_type"], "name": context["attachment_name"],
                                "value": context["attachment_value"]})
        rows.append(Row(key, row["form"], row["study_key"], path, row["copy_sha256"], tuple(attachments)))
    return Plan(rows=tuple(rows), candidates=by_key, kind=kind)


def preview(plan: Plan, targets: Mapping[str, Target]) -> dict[str, Any]:
    """Every row exactly as it is sent, with its dataset, project and batch."""
    batch_of = {key: name for form in plan.forms() for name, keys in plan.batches(form).items() for key in keys}
    return {
        "forms": {form: {"project": targets[form].project_name, "dataset": targets[form].dataset_name,
                         "form": targets[form].ontology_name} for form in plan.forms()},
        "rows": [{
            "global_key": row.global_key,
            "form": row.form,
            "dataset": targets[row.form].dataset_name,
            "project": targets[row.form].project_name,
            "batch": batch_of[row.global_key],
            "file": row.path.name,
            "sha256": row.sha256,
            "attachments": [dict(attachment) for attachment in row.attachments],
            "prelabel_records": len(column_review.prelabels(plan.candidates[row.global_key]))
            if row.form == COLUMN_REVIEW else 0,
        } for row in plan.rows],
    }


@dataclass(frozen=True)
class Workspace:
    datasets: Mapping[str, Any]
    projects: Mapping[str, Any]
    ontologies: Mapping[str, Any]  # None for a form not created yet, when looked up without creating.


@dataclass(frozen=True)
class UploadResult:
    rows: int
    uploaded: int
    batches: Mapping[str, int]
    created_batches: Mapping[str, int]
    prelabels_imported: bool
    replaced: int = 0  # rows Labelbox had not processed, deleted and uploaded again.


def _one(resources: Sequence[Any], kind: str, name: str) -> Any | None:
    matches = [resource for resource in resources if resource.name == name]
    if len(matches) > 1:
        raise UploadError(f"Several {kind}s are named {name!r}.")
    return matches[0] if matches else None


def _failed_keys(errors: Any, keys: Sequence[str]) -> list[str]:
    """The keys, of `keys` and in their order, that a data-row task's errors name (`failedDataRows`)."""
    named = {row.get("globalKey") for error in (errors if isinstance(errors, list) else [errors])
             if isinstance(error, Mapping) for row in error.get("failedDataRows") or [] if isinstance(row, Mapping)}
    return [key for key in keys if key in named]


def _empty_form(ontology) -> bool:
    form = ontology.normalized
    form = json.loads(form) if isinstance(form, str) else form
    return not form.get("tools") and not form.get("classifications")


def _bootstrap_command(target: Target) -> str:
    return (f"python tools/labelbox/create_pilot_project.py --project-name {target.project_name!r} "
            f"--dataset-name {target.dataset_name!r} --apply")


class LabelboxUploadApi:
    """Small adapter around the pinned Labelbox SDK."""

    def __init__(self, client, *, project_type, dataset_type, media_type, targets: Mapping[str, Target],
                 sleep: Callable[[float], None] = time.sleep):
        self.client = client
        self.sleep = sleep
        self.targets = targets
        self.project_type = project_type
        self.dataset_type = dataset_type
        self.media_type = media_type

    @classmethod
    def from_api_key(cls, api_key: str, targets: Mapping[str, Target]) -> "LabelboxUploadApi":
        with _step("connecting to Labelbox"):
            import labelbox as lb
            from labelbox.schema.dataset import Dataset
            from labelbox.schema.project import Project

            return cls(lb.Client(api_key=api_key), project_type=Project, dataset_type=Dataset,
                       media_type=lb.MediaType.Video, targets=targets)

    def ensure_workspace(self, forms: Sequence[str], *, create: bool, replace_form: bool = False) -> Workspace:
        """Find each form's project and dataset, which must exist, and its form; check every match. With `create`,
        then create a missing form and connect each form to its project. A project with another form is refused,
        unless `replace_form` and the project is empty (no batch, no label): it then gets the committed form."""
        datasets, projects, ontologies = {}, {}, {}
        with _step("looking up the projects, datasets and forms"):
            for form in forms:
                target = self.targets[form]
                project = _one(list(self.client.get_projects(where=self.project_type.name == target.project_name)),
                               "project", target.project_name)
                dataset = _one(list(self.client.get_datasets(where=self.dataset_type.name == target.dataset_name)),
                               "dataset", target.dataset_name)
                if project is None or dataset is None:
                    raise UploadError(f"The project {target.project_name!r} or its dataset {target.dataset_name!r} "
                                      f"does not exist. An administrator creates both with: "
                                      f"{_bootstrap_command(target)}")
                if project.media_type != self.media_type or project.description != PROJECT_DESCRIPTION:
                    raise UploadError(f"The project {target.project_name!r} is not a pilot video project made by "
                                      "create_pilot_project.py.")
                if dataset.description != DATASET_DESCRIPTION or dataset.iam_integration() is not None:
                    raise UploadError(f"The dataset {target.dataset_name!r} is not an unconnected pilot dataset made "
                                      "by create_pilot_project.py.")
                ontology = _one(list(self.client.get_ontologies(name_contains=target.ontology_name)),
                                "form", target.ontology_name)
                if ontology is not None:
                    expected, actual = _canonical_form(target.build_form()), _canonical_form(ontology.normalized)
                    if actual != expected:
                        paths = ", ".join(_difference_paths(expected, actual))
                        raise UploadError(f"The form {target.ontology_name!r} differs from the committed form "
                                          f"at: {paths}.")
                connected = project.ontology()
                if connected is not None and not (
                        (ontology is not None and connected.uid == ontology.uid)
                        or (_empty_form(connected) and project.get_label_count() == 0)):
                    emptied = project.get_label_count() == 0 and not list(project.batches())
                    if not replace_form:
                        raise UploadError(f"The project {target.project_name!r} uses another form. Once it is empty, "
                                          "rerun with --replace-form to give it the committed form.")
                    if not emptied:
                        raise UploadError(f"The project {target.project_name!r} uses another form and still has "
                                          "batches or labels: empty it first (--delete --apply).")
                datasets[form], projects[form], ontologies[form] = dataset, project, ontology
        if not create:
            return Workspace(datasets=datasets, projects=projects, ontologies=ontologies)
        for form in forms:
            target = self.targets[form]
            if ontologies[form] is None:
                with _step(f"creating the form {target.ontology_name!r}"):
                    ontologies[form] = self.client.create_ontology(target.ontology_name, target.build_form(),
                                                                   media_type=self.media_type)
            with _step(f"connecting the form {target.ontology_name!r} to the project {target.project_name!r}"):
                connected = projects[form].ontology()
                if connected is None or connected.uid != ontologies[form].uid:
                    projects[form].connect_ontology(ontologies[form])
        return Workspace(datasets=datasets, projects=projects, ontologies=ontologies)

    def _data_row(self, key: str):
        """The data row of a global key, or None."""
        from lbox.exceptions import ResourceNotFoundError

        try:
            return self.client.get_data_row_by_global_key(key)
        except ResourceNotFoundError:
            return None

    def _rows_in_labelbox(self, workspace: Workspace, rows: Sequence[Row]) -> dict[str, Any]:
        """{global key: data row} of the rows Labelbox has; a key in a dataset other than its form's stops the run."""
        found = {}
        for row in rows:
            data_row = self._data_row(row.global_key)
            if data_row is None:
                continue
            if data_row.dataset().uid != workspace.datasets[row.form].uid:
                raise UploadError("A global key of this folder belongs to another dataset.")
            found[row.global_key] = data_row
        return found

    def existing_global_keys(self, workspace: Workspace, rows: Sequence[Row]) -> set[str]:
        with _step("checking which clips Labelbox already has"):
            return set(self._rows_in_labelbox(workspace, rows))

    def _processed(self, keys: Sequence[str]) -> bool:
        """Whether Labelbox processed (read the video of) the data row of every key: the check `create_batch` repeats
        for up to an hour before it creates a batch (the SDK's `Project.__check_data_rows_have_been_processed`)."""
        response = self.client.execute(
            "query AllDataRowsProcessed($globalKeys: [ID!]) { queryAllDataRowsHaveBeenProcessed("
            "globalKeys: $globalKeys) { allDataRowsHaveBeenProcessed } }", {"globalKeys": list(keys)})
        return bool(response["queryAllDataRowsHaveBeenProcessed"]["allDataRowsHaveBeenProcessed"])

    def remove_unprocessed(self, keys: Collection[str]) -> set[str]:
        """Delete the rows of these keys that Labelbox created but never processed, and return their keys, so that the
        run uploads them again. Such a row (Labelbox could not read one clip's video on 2026-10-30) would hold its
        study batch back for an hour, then fail it. A labelled one stops the run instead."""
        with _step("checking that Labelbox processed the clips it has"):
            if not keys or self._processed(sorted(keys)):
                return set()
            unprocessed = {key: self._data_row(key) for key in sorted(keys) if not self._processed([key])}
            labelled = [key for key, data_row in unprocessed.items() if list(data_row.labels())]
        if labelled:
            raise UploadError(f"Labelbox never processed {len(labelled)} labelled rows of this folder "
                              f"({', '.join(labelled)}): remove them in Labelbox, then run again.")
        with _step("deleting the rows Labelbox could not process"):
            for data_row in unprocessed.values():
                data_row.delete()
        return set(unprocessed)

    def upload_rows(self, workspace: Workspace, rows: Sequence[Row]) -> None:
        for form in FORMS:
            payload = []
            for row in (row for row in rows if row.form == form):
                if sha256(row.path) != row.sha256:
                    raise UploadError("A copy changed after the checks: run again.")
                with _step("uploading the video files"):
                    url = self.client.upload_file(str(row.path))
                payload.append({
                    "row_data": url,
                    "global_key": row.global_key,
                    "external_id": row.global_key,
                    "attachments": [dict(attachment) for attachment in row.attachments],
                })
            if not payload:
                continue
            with _step(f"creating the data rows in {self.targets[form].dataset_name!r}"):
                task = workspace.datasets[form].create_data_rows(payload)
                task.wait_till_done()
                errors = task.errors
            if task.status == "FAILED" or errors:
                failed = _failed_keys(errors, [item["global_key"] for item in payload])
                which = f" for {len(failed)} of {len(payload)} rows ({', '.join(failed)})" if failed else ""
                raise UploadError(f"Data-row creation failed{which}: {_safe_error_summary(errors)}. A rerun uploads "
                                  "again the rows Labelbox does not have or did not process.")

    @staticmethod
    def ensure_batches(project, plan: Mapping[str, Sequence[str]]) -> int:
        """One batch per study; a batch of another size is replaced only while the project has no labels."""
        with _step("adding the clips to their project as study batches"):
            existing = list(project.batches())
            created = 0
            for name, keys in plan.items():
                batches = [batch for batch in existing if batch.name == name]
                if len(batches) > 1:
                    raise UploadError("Several batches use one study's batch name.")
                if batches:
                    if batches[0].size == len(keys):
                        continue
                    if project.get_label_count() != 0:
                        raise UploadError("A changed study batch cannot be replaced: the project has submitted "
                                          "labels.")
                    batches[0].delete()
                project.create_batch(name, global_keys=list(keys))
                created += 1
            return created

    def ensure_prelabels(self, workspace: Workspace, candidates: Sequence[Mapping[str, Any]], *,
                         name: str = PREDICTIONS_NAME, refresh: bool = False) -> bool:
        """Import the column review prelabels once, as `name`; a failed import, or one made before the batches
        changed, is replaced only while the project has no labels."""
        with _step("importing the column review prelabels"):
            from labelbox.schema.annotation_import import MALPredictionImport
            from labelbox.schema.enums import AnnotationImportState

            project = workspace.projects[COLUMN_REVIEW]
            imports = [item for item in project.get_mal_prediction_imports() if item.name == name]
            if len(imports) > 1:
                raise UploadError("Several imports use this folder's prelabel name.")
            if imports:
                imports[0].wait_till_done()
                failed = imports[0].state == AnnotationImportState.FAILED or bool(imports[0].errors)
                if failed or refresh:
                    if project.get_label_count() != 0:
                        raise UploadError("Prelabels cannot be replaced: the project has submitted labels.")
                    imports[0].delete()
                    imports = []
            project.enable_model_assisted_labeling(True)
            if imports:
                return False
            named = [record for candidate in candidates for record in column_review.prelabels(candidate)]
            job = MALPredictionImport.create_from_objects(
                client=self.client, project_id=project.uid, name=name,
                predictions=_schema_id_predictions(named, workspace.ontologies[COLUMN_REVIEW].normalized))
            job.wait_till_done()
            if job.state == AnnotationImportState.FAILED or job.errors:
                raise UploadError(f"Prelabel import failed: {_safe_error_summary(job.errors)}.")
            return True

    @staticmethod
    def export_rows(project) -> list[Mapping[str, Any]]:
        """Every exported row as a dict. `Project.export` is a streamed export: its `result` is the URL of the
        result file, not the rows, which come from its buffered stream."""
        with _step("exporting the reviews"):
            task = project.export(params={"label_details": True, "attachments": True})
            task.wait_till_done()
            if task.status == "FAILED" or task.has_errors():
                raise UploadError("Project export failed.")
            if not task.has_result():
                return []
            return [output.json for output in task.get_buffered_stream()]

    def cleanup(self, workspace: Workspace, plan: Plan, *, apply: bool) -> dict[str, int]:
        """What this folder uploaded: its study batches, its prelabel import and its data rows, counted; with `apply`,
        deleted in that order (each batch's labels first, which a batch with labels needs; see `_delete_batch`).
        Nothing else is touched. A rerun continues where an interrupted one stopped. A folder whose kind keeps its
        reviews (the pilot's) is deleted only while its projects hold no submitted review."""
        with _step("listing what this folder uploaded"):
            batches = [batch for form in plan.forms() for batch in workspace.projects[form].batches()
                       if batch.name in plan.batches(form)]
            imports = ([item for item in workspace.projects[COLUMN_REVIEW].get_mal_prediction_imports()
                        if item.name == plan.kind.predictions_name] if COLUMN_REVIEW in plan.forms() else [])
            data_rows = list(self._rows_in_labelbox(workspace, plan.rows).values())
            reviewed = plan.kind.keeps_reviews and any(workspace.projects[form].get_label_count() != 0
                                                       for form in plan.forms())
        counts = {"study batches": len(batches), "prelabel imports": len(imports), "data rows": len(data_rows)}
        if not apply:
            return counts
        if reviewed:
            raise UploadError(f"The {plan.kind.name} projects hold submitted reviews, which deleting would remove: "
                              "nothing was deleted.")
        with _step("deleting the labels of the study batches"):
            for batch in batches:
                batch.delete_labels()
        with _step("deleting the study batches"):
            for batch in batches:
                self._delete_batch(batch)
        with _step("deleting the prelabel import"):
            for item in imports:
                item.delete()
        with _step("deleting the data rows"):
            for data_row in data_rows:
                data_row.delete()
        return counts

    def _delete_batch(self, batch) -> None:
        """Delete a batch whose labels were just deleted. Labelbox deletes them in the background and refuses to
        delete the batch until they are gone (OperationNotAllowedException), which took minutes on 2026-10-20; the
        batch is tried again after each of `BATCH_DELETE_WAITS`."""
        for wait in (*BATCH_DELETE_WAITS, None):
            try:
                batch.delete()
                return
            except Exception as exc:  # the SDK's exception classes are not imported outside the SDK calls
                if wait is None or type(exc).__name__ != "OperationNotAllowedException":
                    raise
            self.sleep(wait)


def ensure_upload(api, plan: Plan, *, replace_form: bool = False) -> UploadResult:
    """Check the projects and datasets, create and connect the forms (replacing an emptied project's form with
    `replace_form`), upload missing rows (and again those Labelbox never processed), batch them by study (rows must
    be batched before their prelabels are imported), then import the column review prelabels."""
    forms = plan.forms()
    workspace = api.ensure_workspace(forms, create=True, replace_form=replace_form)
    existing = api.existing_global_keys(workspace, plan.rows)
    replaced = api.remove_unprocessed(existing)
    missing = [row for row in plan.rows if row.global_key not in existing - replaced]
    if missing:
        api.upload_rows(workspace, missing)
    created = {form: api.ensure_batches(workspace.projects[form], plan.batches(form)) for form in forms}
    imported = bool(plan.candidates) and api.ensure_prelabels(
        workspace, list(plan.candidates.values()), name=plan.kind.predictions_name,
        refresh=created.get(COLUMN_REVIEW, 0) > 0)
    return UploadResult(rows=len(plan.rows), uploaded=len(missing),
                        batches={form: len(plan.batches(form)) for form in forms}, created_batches=created,
                        prelabels_imported=imported, replaced=len(replaced))


def _answers_from_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """A submitted label's radio, checklist and text answers, keyed by question id, as the forms' checks read them.

    Labelbox exports each question and each answer with its display text as `name` and its id as `value` (`"name":
    "What is this item?", "value": "input_type"`), so both are read from `value`; `name` is a fallback only for an
    entry without one."""
    answers: dict[str, Any] = {}

    def identifier(item: Mapping[str, Any]) -> str | None:
        value = item.get("value") or item.get("name")
        return value if isinstance(value, str) else None

    def visit(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, Mapping) or identifier(item) is None:
                continue
            question = identifier(item)
            radio, checklist, text = item.get("radio_answer"), item.get("checklist_answers"), item.get("text_answer")
            if isinstance(radio, Mapping):
                if identifier(radio) is not None:
                    answers[question] = identifier(radio)
                visit(radio.get("classifications"))
            elif isinstance(checklist, list):
                selected = [answer for answer in checklist if isinstance(answer, Mapping)]
                answers[question] = [identifier(answer) for answer in selected if identifier(answer) is not None]
                for answer in selected:
                    visit(answer.get("classifications"))
            elif isinstance(text, Mapping) and isinstance(text.get("content"), str):
                answers[question] = text["content"]

    annotations = label.get("annotations")
    if isinstance(annotations, Mapping):
        visit(annotations.get("classifications"))
    return answers


def _without_source_text(value: Any) -> Any:
    """A score without the report clauses its columns came from, so the summary can be shared."""
    if isinstance(value, list):
        return [_without_source_text(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _without_source_text(item) for key, item in value.items() if key != "source_text"}
    return value


def summarize_export(exports: Mapping[str, Sequence[Mapping[str, Any]]], plan: Plan) -> dict[str, Any]:
    """The review status of each of this folder's exported rows: column reviews scored with `score_review`, view and
    type reviews checked with `review_problems`. Other rows of the project (the pilot's) are left out."""
    planned = {row.global_key: row.form for row in plan.rows}
    items = []
    for form, rows in exports.items():
        for row in rows:
            data_row = row.get("data_row")
            key = data_row.get("global_key") if isinstance(data_row, Mapping) else None
            if key not in planned:
                continue
            if planned[key] != form:
                raise UploadError("The export holds one of this folder's rows in the other form's project.")
            label = _latest_label(row)
            if label is None:
                items.append({"global_key": key, "form": form, "status": "unreviewed"})
                continue
            answers = _answers_from_label(label)
            if form == COLUMN_REVIEW:
                try:
                    score = column_review.score_review(plan.candidates[key], answers, submitted=True)
                except ValueError as exc:
                    items.append({"global_key": key, "form": form, "status": "invalid", "problems": [str(exc)]})
                    continue
                items.append({"global_key": key, "form": form, "status": "invalid" if score["pending"] else "valid",
                              "score": _without_source_text(score)})
            else:
                problems = view_type_review.review_problems(answers)
                items.append({"global_key": key, "form": form, "status": "invalid" if problems else "valid",
                              "answers": view_type_review.with_not_applicable(answers), "problems": problems})
    return {"schema": plan.kind.summary_schema, "items": sorted(items, key=lambda item: item["global_key"])}


def export_summary(api, plan: Plan, output: Path) -> dict[str, int]:
    workspace = api.ensure_workspace(plan.forms(), create=False)
    summary = summarize_export({form: api.export_rows(workspace.projects[form]) for form in plan.forms()}, plan)
    _write_private(output, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    counts: dict[str, int] = {}
    for item in summary["items"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, required=True,
                        help="the folder `workflow_test` or `pilot_prepare` prepared")
    parser.add_argument("--workspace", choices=sorted(WORKSPACES),
                        help="the folder's own workspace, which its keys decide: the workflow test's projects for a "
                             "workflow test, the pilot's for the pilot; another is refused")
    parser.add_argument("--form", choices=FORMS, help="only this form's rows")
    parser.add_argument("--apply", action="store_true",
                        help="upload missing rows; with --delete, delete what this folder uploaded")
    parser.add_argument("--delete", action="store_true", help="list what this folder uploaded (delete with --apply)")
    parser.add_argument("--replace-form", action="store_true",
                        help="with --apply: give an empty project that has another form the committed form")
    parser.add_argument("--export-summary", type=Path, metavar="PATH", help="export and score submitted reviews")
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] = os.environ,
         api_factory: Callable[[str], Any] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.export_summary is not None and (args.apply or args.delete):
        parser.error("--export-summary runs alone")
    if args.replace_form and (not args.apply or args.delete):
        parser.error("--replace-form goes with --apply, for an upload")
    try:
        plan = read_plan(args.folder)
        if args.workspace not in (None, plan.kind.workspace):
            raise UploadError(f"A {plan.kind.name} folder goes to the {plan.kind.workspace!r} workspace only.")
    except UploadError as exc:
        print(f"Upload stopped by a check: {exc}", file=sys.stderr)
        return 1
    if args.form:
        plan = plan.only(args.form)
        if not plan.rows:
            print(f"The folder holds no {args.form} rows.", file=sys.stderr)
            return 1
    targets = WORKSPACES[plan.kind.workspace]
    _write_private(args.folder / PREVIEW, json.dumps(preview(plan, targets), ensure_ascii=False, indent=2) + "\n")
    counts = {form: len(plan.rows_of(form)) for form in plan.forms()}
    description = ", ".join(f"{n} {form}" for form, n in counts.items())
    if not args.apply and not args.delete and args.export_summary is None:
        print(f"Dry run: no Labelbox requests were made. Checked {len(plan.rows)} rows ({description}) of a "
              f"{plan.kind.name} folder, for the {plan.kind.workspace} projects.")
        print(f"Read {args.folder / PREVIEW}: every row as it would be sent.")
        print("Run again with --apply after setting LABELBOX_API_KEY.")
        return 0

    api_key = environ.get("LABELBOX_API_KEY", "").strip()
    if not api_key:
        print("LABELBOX_API_KEY is required for Labelbox operations.", file=sys.stderr)
        return 2
    api_factory = api_factory or (lambda key: LabelboxUploadApi.from_api_key(key, targets))
    if args.delete and args.apply:
        print("Deleting. Labelbox deletes labels in the background, so this can take a few minutes.", flush=True)
    previous_log_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as discarded, redirect_stdout(discarded), redirect_stderr(discarded):
            api = api_factory(api_key)
            if args.delete:
                workspace = api.ensure_workspace(plan.forms(), create=False)
                found = api.cleanup(workspace, plan, apply=args.apply)
            elif args.apply:
                result = ensure_upload(api, plan, replace_form=args.replace_form)
            else:
                statuses = export_summary(api, plan, args.export_summary)
    except UploadError as exc:
        print(f"Upload stopped by a check: {exc}", file=sys.stderr)
        return 1
    except StepFailed as exc:
        print(f"Upload failed while {exc.step} ({exc.kind}). Nothing after this step ran; a rerun reuses what "
              "the earlier steps did.", file=sys.stderr)
        if exc.kind in {"AuthorizationError", "AuthenticationError"}:
            print("Your Labelbox key or role is not allowed to do this step.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Upload failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_log_disable)

    if args.delete:
        listed = ", ".join(f"{n} {kind}" for kind, n in found.items())
        if args.apply:
            print(f"Deleted what this folder uploaded: {listed}. Projects, datasets and forms are kept.")
        else:
            print(f"This folder's upload holds {listed}; nothing was deleted. Run with --delete --apply to delete.")
    elif args.apply:
        print(f"rows: {result.rows} ({description}); uploaded now: {result.uploaded}")
        if result.replaced:
            print(f"of which {result.replaced} again: Labelbox had not processed their videos")
        for form in result.batches:
            print(f"{targets[form].project_name}: study batches {result.batches[form]}, "
                  f"created now {result.created_batches[form]}")
        if plan.candidates:
            print(f"column review prelabels: {'imported' if result.prelabels_imported else 'existing import kept'}")
    else:
        print("summary written; " + ", ".join(f"{status}: {n}" for status, n in sorted(statuses.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
