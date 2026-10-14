"""The view and type review: a second Labelbox dataset that relabels clips the view classifier calls `OTHER`.

Many `OTHER` items are not cine loops but still frames, often measurement frames. A reviewer gives each item its
view, what kind of item it is, its acquisition mode, and whether identifying text is visible. The answers measure
what `OTHER` holds, and `route` sends a relabelled clip back into the column review v2
(`tools.labelbox.column_review.build_candidate`).

    python -m tools.labelbox.view_type_review            # write the form JSON and the generated table of its doc.
    python -m tools.labelbox.view_type_review --check    # exit 1 if either is out of date.

Every question is a global classification, so the form fits an image or a video project. The view labels come from
the RADAR ontology and the acquisition modes from the column review v2 specification, so both review forms share
their ids.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from app.radar.ontology import REPO_ROOT, UNCLASSIFIED_VIEW, VIEWS, load_ontology
from tools.labelbox.column_review import load_spec as load_column_spec
from tools.labelbox.review_ontology import _classification, _option

FORM_NAME = "DeepECHO view and type review v1"
FORM_PATH = os.path.join(REPO_ROOT, "configs", "radar", "labelbox", "view_type_review_v1.json")
DOC_PATH = os.path.join(REPO_ROOT, "docs", "radar", "view_type_review_v1.md")

INPUT_TYPES = {
    "cine_loop": "Cine loop (moving echo images)",
    "still_image": "Still frame of an echo image",
    "measurement_still": "Still frame with measurements, calipers or tracings",
    "analysis_screen": "Analysis or quantification screen (strain, 3D, EF tool)",
    "text_page": "Report, worksheet or other text page",
    "other": "Other, no echo image",
}
ECHO_IMAGE_TYPES = ("cine_loop", "still_image", "measurement_still")  # always show an echo view.
NON_IMAGE_TYPES = ("text_page", "other")  # never do. An analysis screen may or may not.
CANNOT_TELL = "cannot_tell"
NO_VIEW = "no_view"
NO_ULTRASOUND = "not_applicable"
# Items the column review v2 can take: its prefills need a cine loop or a plain still in B-mode or color Doppler.
ROUTED_INPUT_KINDS = {"cine_loop": "video", "still_image": "image"}
ROUTED_MODALITIES = ("bmode", "color_flow")
REQUIRED = ("input_type", "view", "modality", "identifier_visible")


def view_choices() -> dict[str, str]:
    views = load_ontology().views
    choices = {v: f"{v}: {views[v].labels['en']}" for v in VIEWS}
    choices[UNCLASSIFIED_VIEW] = f"{UNCLASSIFIED_VIEW}: off-axis or non-standard echo view"
    choices[CANNOT_TELL] = "Cannot tell the view"
    choices[NO_VIEW] = "No echo view (text, screen or no ultrasound image)"
    return choices


def modality_choices() -> dict[str, str]:
    return {**load_column_spec()["modalities"], NO_ULTRASOUND: "Not applicable: no ultrasound image"}


def identifier_choices() -> dict[str, str]:
    return {"no": "No", "yes": "Yes: the item must be deleted"}


def questions() -> list[tuple[str, str, dict[str, str] | None, bool]]:
    """(id, question, choices, required) for each question, in form order; free text has no choices."""
    return [
        ("input_type", "What is this item?", INPUT_TYPES, True),
        ("view", "Which view does it show?", view_choices(), True),
        ("modality", "Acquisition mode", modality_choices(), True),
        ("identifier_visible", "Is identifying text visible (a name, a record number, a date of birth)?",
         identifier_choices(), True),
        ("comments", "Comments (no patient details)", None, False),
    ]


def build_form() -> dict:
    classifications = []
    for name, text, choices, required in questions():
        if choices is None:
            classifications.append(_classification("text", name, text, required=required, top_level=True))
            continue
        out = _classification("radio", name, text, required=required, top_level=True,
                              options=[_option(value, label) for value, label in choices.items()])
        if name == "view":
            out["uiMode"] = "searchable"
        classifications.append(out)
    return {"tools": [], "classifications": classifications}


def review_problems(answers: dict) -> list[str]:
    """What is wrong with one decoded review (`column_review.decode_annotations`); empty when it is valid."""
    problems = []
    for name, _, choices, required in questions():
        value = answers.get(name)
        if value is None:
            if required:
                problems.append(f"{name} is required")
        elif choices is not None and value not in choices:
            problems.append(f"{name} has no answer {value!r}")
    if problems:
        return problems
    kind, view, modality = answers["input_type"], answers["view"], answers["modality"]
    if kind in NON_IMAGE_TYPES and view != NO_VIEW:
        problems.append(f"{kind} cannot show a view")
    if kind in ECHO_IMAGE_TYPES and view == NO_VIEW:
        problems.append(f"{kind} always shows an echo view: give it, or {CANNOT_TELL!r}")
    if (view == NO_VIEW) != (modality == NO_ULTRASOUND):
        problems.append(f"{NO_VIEW!r} and {NO_ULTRASOUND!r} go together: no ultrasound image means no view")
    return problems


def route(answers: dict) -> dict | None:
    """The `view`, `modality` and `input_kind` arguments of `build_candidate` for a relabelled item, or None.

    Only a valid review of a cine loop or a plain still, in a view the column review routes, in B-mode or color
    Doppler, and with no identifying text, goes to the column review. A measurement still does not: its calipers and
    numbers are not something a caption should describe.
    """
    if review_problems(answers) or answers["identifier_visible"] != "no":
        return None
    kind, view, modality = answers["input_type"], answers["view"], answers["modality"]
    if kind not in ROUTED_INPUT_KINDS or modality not in ROUTED_MODALITIES:
        return None
    route_columns = load_column_spec()["views"].get(view, {}).get("columns")
    if view == UNCLASSIFIED_VIEW or not route_columns:
        return None
    return {"view": view, "modality": modality, "input_kind": ROUTED_INPUT_KINDS[kind]}


def render_form_table() -> str:
    lines = ["| Question | Id | Required | Answers |", "|---|---|---|---|"]
    for name, text, choices, required in questions():
        answers = "free text" if choices is None else "; ".join(f"`{v}` {label}" for v, label in choices.items())
        lines.append(f"| {text} | `{name}` | {'yes' if required else 'no'} | {answers} |")
    return "\n".join(lines) + "\n"


_BLOCK = re.compile(r"(<!-- BEGIN GENERATED: view_type_form -->\n).*?(<!-- END GENERATED: view_type_form -->)", re.S)


def synced_doc(text: str) -> str:
    if not _BLOCK.search(text):
        raise ValueError(f"{DOC_PATH} has no generated view_type_form block")
    return _BLOCK.sub(lambda m: m.group(1) + render_form_table() + m.group(2), text)


def _form_json() -> str:
    return json.dumps(build_form(), ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the form JSON or its doc is out of date")
    args = parser.parse_args(argv)
    with open(DOC_PATH, encoding="utf-8") as f:
        doc = f.read()
    form_json, new_doc = _form_json(), synced_doc(doc)
    if args.check:
        stale = [path for path, current, wanted in ((FORM_PATH, _read(FORM_PATH), form_json), (DOC_PATH, doc, new_doc))
                 if current != wanted]
        for path in stale:
            print(f"Out of date: {os.path.relpath(path, REPO_ROOT)}", file=sys.stderr)
        return 1 if stale else 0
    for path, text in ((FORM_PATH, form_json), (DOC_PATH, new_doc)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
