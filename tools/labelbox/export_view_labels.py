"""Export the view relabelling project's answers as one row per clip, for training the view classifier.

    python -m tools.labelbox.export_view_labels --picked PICKED... --out OUT                     # export, write OUT
    python -m tools.labelbox.export_view_labels --picked PICKED... --out OUT --from-export KEPT   # no Labelbox request

PICKED is the `picked_clips.csv` of a `data/view_relabel.py draw` folder, one per batch uploaded to the project, in
batch order (batch 1 first): the project holds their union. Every exported clip must be in one list, every listed clip
must be exported once, and no clip may be in two lists. Run in the Labelbox environment. The export
checks that the project `RADAR View Relabelling` uses the form `DeepECHO view review v1`, exports every row with its
workflow status and each label's details, and keeps the export unchanged next to OUT (`<OUT name>.export.ndjson`),
private, because it holds reviewers' names and comments. The form may have been edited in Labelbox since it was
committed: the console lists the view options added or removed, answers are checked against the form as it is in
Labelbox, and that form is kept next to the export (`<OUT name>.form.json`) for `--from-export`. `--from-export` writes OUT again
from a kept export, without a key or a request. Neither overwrites a file, and both refuse a path in the repository.

OUT has one row per picked clip, with the columns of COLUMNS:

- `predicted_view`: the view classifier's class, the clip's prefill (`PSAX` clips had none: it is not a choice).
- `human_view`: the answer's option value, `UNSURE` for `unsure`; filled only when `label_status` is `done`.
- `label_status`:
  `done`, the row's labels all answer one view (it may still wait in a review step, which the summary counts);
  `unlabelled`, no label; `skipped`, only skips; `conflict`, labels that answer different views;
  `in_rework`, the row is in Rework, or back in the labelling queue with an answered label.
- `n_labels`: the labels that answer the view (skips are not counted).
- `labelled_at`: when the latest of those labels was created (of the skips, for a skipped row), in UTC.
- `batch`: which `--picked` list holds the clip, 1-based in the order given (1 = the first list).
- `workflow_status`: the row's step in the project's workflow, TO_LABEL, IN_REVIEW, IN_REWORK or DONE (a `done` row
  still IN_REVIEW is submitted but not approved).

Nothing about reviewers or their comments is written. The console shows counts and the opaque keys of the clips with a
comment, which may report identifying text, so that someone reads it in Labelbox.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import logging
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from tools.labelbox import column_review, view_review
from tools.labelbox.pilot_upload import (
    VIEW_RELABEL,
    VIEW_REVIEW,
    WORKSPACES,
    LabelboxUploadApi,
    StepFailed,
    UploadError,
    _answers_from_label,
    _read_csv,
    _step,
    _write_private,
    sha256,
)

COLUMNS = ("global_key", "patient_id", "study_id", "video_path", "acq_category", "predicted_view", "human_view",
           "label_status", "n_labels", "labelled_at", "batch", "workflow_status")
STATUSES = ("done", "unlabelled", "skipped", "in_rework", "conflict")
UNSURE = "UNSURE"
PICKED_COLUMNS = ("global_key", "patient_id", "study_id", "video_path", "acq_category", "view")
TARGET = WORKSPACES[VIEW_RELABEL.workspace][VIEW_REVIEW]
QUESTIONS = ("view", "comments")
# A row's workflow status, by its letters alone: Labelbox's export filters spell them `InRework`, its export documents
# `IN_REWORK`.
WORKFLOW = {"TOLABEL": "TO_LABEL", "INREVIEW": "IN_REVIEW", "INREWORK": "IN_REWORK", "DONE": "DONE"}


class ExportError(UploadError):
    """The export, `picked_clips.csv` or an output path fails a check; the messages name opaque keys only."""


def kept_export_path(out: Path) -> Path:
    """Where the live run keeps the export: next to OUT, `<OUT name>.export.ndjson`."""
    return out.with_name(f"{out.stem}.export.ndjson")


def kept_form_path(export: Path) -> Path:
    """Where the live run keeps the form as it is in Labelbox: next to the export, `<OUT name>.form.json`."""
    return export.with_name(export.name.removesuffix(".export.ndjson") + ".form.json")


def form_views(form: Any) -> dict[str, str]:
    """{option value: display text} of the form's `view` question."""
    form = json.loads(form) if isinstance(form, str) else form
    questions = [item for item in (form.get("classifications") or []) if isinstance(item, Mapping)
                 and item.get("name") == "view"] if isinstance(form, Mapping) else []
    if len(questions) != 1:
        raise ExportError(f"The form of {TARGET.project_name!r} has no single `view` question.")
    options = {option.get("value"): option.get("label", "") for option in questions[0].get("options") or []
               if isinstance(option, Mapping)}
    if not options or not all(isinstance(value, str) for value in options):
        raise ExportError(f"The `view` question of {TARGET.project_name!r} has no option values.")
    return options


def form_changes(views: Mapping[str, str]) -> list[str]:
    """The view options added, removed or relabelled in Labelbox, against the committed form."""
    committed = view_review.view_choices()
    lines = []
    if added := [value for value in views if value not in committed]:
        lines.append(f"view options added in Labelbox: {', '.join(added)}")
    if removed := [value for value in committed if value not in views]:
        lines.append(f"view options removed in Labelbox: {', '.join(removed)}")
    if relabelled := [value for value in views if value in committed and views[value] != committed[value]]:
        lines.append(f"view options with new display text: {', '.join(relabelled)}")
    if [value for value in views if value in committed] != [value for value in committed if value in views]:
        lines.append("view options reordered in Labelbox")
    return lines


def read_picked(path: Path, batch: int = 1) -> dict[str, dict[str, Any]]:
    """{global key: clip} of one `picked_clips.csv`, in its order; each clip carries `batch`."""
    rows = _read_csv(path)
    if not rows or set(PICKED_COLUMNS) - set(rows[0]):
        raise ExportError(f"{path.name} needs the columns {', '.join(PICKED_COLUMNS)}.")
    if "batch" in rows[0]:
        raise ExportError(f"{path.name} has a `batch` column: the batch is the list's place in --picked.")
    keys = [row["global_key"] for row in rows]
    if len(keys) != len(set(keys)) or not all(VIEW_RELABEL.global_key.fullmatch(key) for key in keys):
        raise ExportError(f"{path.name} must list each clip once, under a view relabelling key ({VIEW_RELABEL.name}).")
    return {row["global_key"]: {**row, "batch": batch} for row in rows}


def read_picked_lists(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """{global key: clip} of several `picked_clips.csv`, one per batch of the project, batch 1 first: their union, in
    list order then each list's order. A clip in two lists is refused."""
    if not paths:
        raise ExportError("Name at least one picked_clips.csv.")
    picked: dict[str, dict[str, Any]] = {}
    for batch, path in enumerate(paths, start=1):
        clips = read_picked(path, batch)
        if shared := [key for key in clips if key in picked]:
            raise ExportError(f"{len(shared)} clips of list {batch} ({path}) are also in list "
                              f"{picked[shared[0]]['batch']} ({shared[0]}, ...): each clip belongs to one batch.")
        picked.update(clips)
    return picked


def _workflow_status(entry: Mapping[str, Any], key: str) -> str:
    details = entry.get("project_details")
    status = details.get("workflow_status") if isinstance(details, Mapping) else None
    if not isinstance(status, str):
        raise ExportError(f"{key} has no workflow status: the export needs project details, as the live run asks.")
    if re.sub(r"[^A-Z]", "", status.upper()) not in WORKFLOW:
        raise ExportError(f"{key} has an unknown workflow status {status!r}.")
    return WORKFLOW[re.sub(r"[^A-Z]", "", status.upper())]


def _skipped(label: Mapping[str, Any], key: str) -> bool:
    flags = [details["skipped"] for details in (label.get("performance_details"), label.get("performance_details_v2"))
             if isinstance(details, Mapping) and isinstance(details.get("skipped"), bool)]
    if not flags:
        raise ExportError(f"A label of {key} does not say whether it was skipped: the export needs performance details.")
    return any(flags)


def _created_at(label: Mapping[str, Any], key: str) -> str:
    """The label's creation time in UTC, to the second."""
    details = label.get("label_details")
    text = details.get("created_at") if isinstance(details, Mapping) else None
    try:
        moment = datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        moment = None
    if moment is None or moment.tzinfo is None:
        raise ExportError(f"A label of {key} has no creation time with a time zone: the export needs label details.")
    return moment.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


def _answers(label: Mapping[str, Any], key: str, views: Mapping[str, str]) -> tuple[str | None, str]:
    """(the view's option value or None, the comment) of one label; anything the form (`views`, as in Labelbox) does
    not ask or offer is refused, since then the export is not of this form."""
    answers = _answers_from_label(label)
    if set(answers) - set(QUESTIONS):
        raise ExportError(f"A label of {key} answers questions {TARGET.ontology_name!r} does not ask: is the export of "
                          f"{TARGET.project_name!r}?")
    view, comment = answers.get("view"), answers.get("comments", "")
    if view is not None and view not in views:
        raise ExportError(f"A label of {key} answers the view {view!r}, which {TARGET.ontology_name!r} does not offer.")
    return view, comment if isinstance(comment, str) else ""


def clip_labels(entry: Mapping[str, Any], key: str, views: Mapping[str, str]) -> dict[str, Any]:
    """One row's label columns, its workflow status and whether a label carries a comment."""
    workflow = _workflow_status(entry, key)
    labels = entry.get("labels")
    if not isinstance(labels, list):
        raise ExportError(f"{key} has no list of labels.")
    answered, skips, comment = [], [], False
    for label in labels:
        view, text = _answers(label, key, views)
        comment = comment or bool(text.strip())
        if _skipped(label, key):
            skips.append(_created_at(label, key))
        elif view is None:
            raise ExportError(f"A label of {key} is neither skipped nor answers the view.")
        else:
            answered.append((view, _created_at(label, key)))
    views = {view for view, _ in answered}
    if workflow == "IN_REWORK" or (workflow == "TO_LABEL" and answered):
        status = "in_rework"
    elif len(views) > 1:
        status = "conflict"
    elif views:
        status = "done"
    else:
        status = "skipped" if skips else "unlabelled"
    times = [time for _, time in answered] or skips
    human = ""
    if status == "done":
        view, = views
        human = UNSURE if view == view_review.UNSURE else view
    return {"human_view": human, "label_status": status, "n_labels": len(answered),
            "labelled_at": max(times, default=""), "workflow_status": workflow, "comment": comment}


def build_rows(exported: Sequence[Mapping[str, Any]], picked: Mapping[str, Mapping[str, str]],
               views: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """One record per picked clip, in picked order: COLUMNS, plus `comment` for the summary.
    The export must be of the view relabelling project and hold every picked clip, and nothing else, once. `views` are
    the view options of the form as it is in Labelbox (the committed form's by default)."""
    views = view_review.view_choices() if views is None else views
    entries: dict[str, Mapping[str, Any]] = {}
    for row in exported:
        data_row = row.get("data_row")
        key = data_row.get("global_key") if isinstance(data_row, Mapping) else None
        projects = row.get("projects")
        if not isinstance(key, str) or not isinstance(projects, Mapping) or len(projects) != 1:
            raise ExportError("Every exported row needs a global key and exactly one project.")
        if key in entries:
            raise ExportError(f"The export holds {key} twice.")
        entry, = projects.values()
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if name != TARGET.project_name:
            raise ExportError(f"The export is of the project {name!r}, not {TARGET.project_name!r}.")
        entries[key] = entry
    unknown = sorted(set(entries) - set(picked))
    if unknown:
        raise ExportError(f"The export holds {len(unknown)} clips the picked lists do not list ({unknown[0]}, ...): "
                          "is a batch's picked_clips.csv missing from --picked?")
    missing = [key for key in picked if key not in entries]
    if missing:
        raise ExportError(f"The export lacks {len(missing)} picked clips ({missing[0]}, ...): it is not the whole "
                          f"{TARGET.project_name!r} project.")
    return [{"global_key": key, "patient_id": clip["patient_id"], "study_id": clip["study_id"],
             "video_path": clip["video_path"], "acq_category": clip["acq_category"], "predicted_view": clip["view"],
             **clip_labels(entries[key], key, views), "batch": clip["batch"]} for key, clip in picked.items()]


def labels_csv(records: Sequence[Mapping[str, Any]]) -> str:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return text.getvalue()


def _counts(counter: Counter, order: Sequence[str] = ()) -> str:
    keys = [k for k in order if counter[k]] + sorted(k for k in counter if k not in order)
    return ", ".join(f"{k} {counter[k]}" for k in keys) or "none"


def summary_lines(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Counts per status and per answer, predicted against human per predicted class, and the clips with a comment."""
    done = [r for r in records if r["label_status"] == "done"]
    lines = [f"clips: {len(records)}",
             f"label_status: {_counts(Counter(r['label_status'] for r in records), STATUSES)}"]
    batches = sorted({r["batch"] for r in records})
    if len(batches) > 1:
        lines += [f"  batch {batch}: {sum(r['batch'] == batch for r in records)} clips, "
                  f"{_counts(Counter(r['label_status'] for r in records if r['batch'] == batch), STATUSES)}"
                  for batch in batches]
    lines += [f"  done by workflow status: {_counts(Counter(r['workflow_status'] for r in done))}",
             f"human_view (done): {_counts(Counter(r['human_view'] for r in done))}",
             "predicted vs human (done rows; agree = the answer is the predicted class):",
             f"  {'predicted':<14}{'clips':>6}{'done':>6}{'agree':>7}{'unsure':>8}{'other':>7}  other answers"]
    for view in sorted({r["predicted_view"] for r in records}):
        own = [r for r in records if r["predicted_view"] == view]
        answers = Counter(r["human_view"] for r in own if r["label_status"] == "done")
        agree, unsure = answers.pop(view, 0), answers.pop(UNSURE, 0)
        lines.append(f"  {view:<14}{len(own):>6}{sum(answers.values()) + agree + unsure:>6}{agree:>7}{unsure:>8}"
                     f"{sum(answers.values()):>7}  {', '.join(f'{v} {n}' for v, n in answers.most_common(4))}")
    lines += ["  PSAX is not a choice, so a PSAX clip never agrees: its other answers are its levels.",
              "  A prefilled clip submitted unchanged and one a reviewer chose the same view for look alike in the "
              "export: agreement counts both."]
    comments = [r["global_key"] for r in records if r["comment"]]
    lines.append(f"clips with a comment, to read in Labelbox (it may report identifying text): {len(comments)}")
    lines += [f"  {key}" for key in comments]
    return lines


def check_outputs(out: Path, *, keep_export: bool) -> None:
    paths = [out, kept_export_path(out), kept_form_path(kept_export_path(out))] if keep_export else [out]
    if any(path.resolve().is_relative_to(column_review.ROOT) for path in paths):
        raise ExportError("--out must be outside the repository: the labels name patients and video paths.")
    if any(path.exists() for path in paths):
        raise ExportError(f"{out.name} or its export exists: every export is kept, so give a new name.")


def export_project(api) -> tuple[list[Mapping[str, Any]], Any]:
    """(the view relabelling project's rows, its form as it is in Labelbox), after checking that the project uses the
    view review form. Neither the dataset nor the form's content is checked against the committed one: an export reads
    the project only, and the answers are checked against the form in Labelbox."""
    workspace = api.ensure_workspace([VIEW_REVIEW], create=False, datasets_checked=False, form_checked=False)
    project, form = workspace.projects[VIEW_REVIEW], workspace.ontologies[VIEW_REVIEW]
    with _step("checking the project's form"):
        connected = project.ontology()
    if form is None or connected is None or connected.uid != form.uid:
        raise ExportError(f"The project {TARGET.project_name!r} does not use the form {TARGET.ontology_name!r}.")
    with _step("reading the project's form"):
        normalized = connected.normalized
    normalized = json.loads(normalized) if isinstance(normalized, str) else normalized
    return api.export_rows(project, project_details=True), normalized


def read_kept_export(path: Path) -> list[Mapping[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@contextmanager
def _quiet() -> Iterator[None]:
    """The SDK's own logs and prints stay out of the console: they can hold server messages."""
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as discarded, redirect_stdout(discarded), redirect_stderr(discarded):
            yield
    finally:
        logging.disable(previous)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--picked", type=Path, nargs="+", required=True, metavar="PICKED",
                        help="picked_clips.csv of each view relabelling batch in the project, batch 1 first")
    parser.add_argument("--out", type=Path, required=True, help="the labels CSV to write, outside the repository")
    parser.add_argument("--from-export", type=Path, metavar="KEPT",
                        help="write the labels from an export this script kept, without a Labelbox request")
    return parser


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] = os.environ,
         api_factory: Callable[[str], Any] | None = None) -> int:
    args = _parser().parse_args(argv)
    kept = kept_export_path(args.out)
    try:
        picked = read_picked_lists(args.picked)
        check_outputs(args.out, keep_export=args.from_export is None)
        if args.from_export is not None:
            exported, source = read_kept_export(args.from_export), args.from_export
            kept_form = kept_form_path(args.from_export)
            form = json.loads(kept_form.read_text(encoding="utf-8")) if kept_form.exists() else None
        else:
            api_key = environ.get("LABELBOX_API_KEY", "").strip()
            if not api_key:
                print("LABELBOX_API_KEY is required to export; --from-export reads a kept export.", file=sys.stderr)
                return 2
            factory = api_factory or (lambda key: LabelboxUploadApi.from_api_key(key, WORKSPACES["view_relabel"]))
            try:
                with _quiet():
                    exported, form = export_project(factory(api_key))
            except (UploadError, StepFailed):
                raise
            except Exception as exc:
                print(f"Export failed ({type(exc).__name__}).", file=sys.stderr)
                return 1
            _write_private(kept, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported))
            _write_private(kept_form_path(kept), json.dumps(form, ensure_ascii=False, indent=1) + "\n")
            source = kept
        views = view_review.view_choices() if form is None else form_views(form)
        changes = form_changes(views)
        records = build_rows(exported, picked, views)
    except UploadError as exc:
        print(f"Export stopped by a check: {exc}", file=sys.stderr)
        if args.from_export is None and kept.exists():
            print(f"The export is kept in {kept}: once fixed, rerun with --from-export {kept}.", file=sys.stderr)
        return 1
    except StepFailed as exc:
        print(f"Export failed while {exc.step} ({exc.kind}); nothing was written.", file=sys.stderr)
        return 1
    _write_private(args.out, labels_csv(records))
    print(f"labels written: {args.out}")
    print(f"from export: {source} ({len(exported)} rows, sha256 {sha256(source)})")
    for batch, path in enumerate(args.picked, start=1):
        print(f"batch {batch}: {path} ({sum(r['batch'] == batch for r in records)} clips)")
    if form is None:
        print("form: none kept next to the export; answers checked against the committed form")
    for line in changes or ["form: view options as committed"]:
        print(line)
    for line in summary_lines(records):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
