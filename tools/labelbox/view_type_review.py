"""The view and type review: a second Labelbox dataset that relabels clips the view classifier calls `OTHER`.

Still clips, clips with visible measurements and single-frame objects are removed before upload (issue #22), so every
item is a video. A reviewer says what kind of item it is and whether identifying text is visible; only an echo cine
loop is also asked its view and acquisition mode, which `with_not_applicable` records as not applicable for any other
item. An item the screen should have removed is reported in the comments. The answers measure what `OTHER` holds,
and `route` sends a relabelled cine loop back into the column review v2
(`tools.labelbox.column_review.build_candidate`).

    python -m tools.labelbox.view_type_review            # write the form JSON.
    python -m tools.labelbox.view_type_review --check    # exit 1 if it is out of date.

The view and the acquisition mode are nested under the cine loop answer, the way Labelbox shows a question only after
a given answer; the other questions are global classifications. The view labels come from the RADAR ontology and the
acquisition modes from the column review v2 specification, so both review forms share their ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from app.radar.ontology import REPO_ROOT, UNCLASSIFIED_VIEW, VIEWS, load_ontology
from tools.labelbox.column_review import load_spec as load_column_spec
from tools.labelbox.review_ontology import _classification, _option

FORM_NAME = "DeepECHO view and type review v1"
FORM_PATH = os.path.join(REPO_ROOT, "configs", "radar", "labelbox", "view_type_review_v1.json")

CINE_LOOP = "cine_loop"
INPUT_TYPES = {
    CINE_LOOP: "Echo cine loop (moving echo images)",
    "analysis_screen": "Analysis or quantification screen (strain, 3D, EF tool)",
    "no_echo_image": "No echo image",
}
CANNOT_TELL = "cannot_tell"
NOT_APPLICABLE = "not_applicable"  # the view and acquisition mode of an item that is not a cine loop.
# Cine loops the column review v2 can take: its prefills need B-mode or color Doppler.
ROUTED_MODALITIES = ("bmode", "color_flow")


@dataclass(frozen=True)
class Question:
    name: str
    text: str
    choices: dict[str, str] | None  # None for free text.
    required: bool
    asked_after: tuple[str, str] | None = None  # the (question, answer) it is nested under; None when always asked.


def view_choices() -> dict[str, str]:
    views = load_ontology().views
    choices = {v: f"{v}: {views[v].labels['en']}" for v in VIEWS}
    choices[UNCLASSIFIED_VIEW] = f"{UNCLASSIFIED_VIEW}: off-axis or non-standard echo view"
    choices[CANNOT_TELL] = "Cannot tell the view"
    return choices


def modality_choices() -> dict[str, str]:
    return dict(load_column_spec()["modalities"])


def identifier_choices() -> dict[str, str]:
    return {"no": "No", "yes": "Yes: the item must be deleted"}


def questions() -> list[Question]:
    """Every question, in form order: the view and the acquisition mode follow the cine loop answer."""
    after_cine_loop = ("input_type", CINE_LOOP)
    return [
        Question("input_type", "What is this item?", INPUT_TYPES, True),
        Question("view", "Which view does it show?", view_choices(), True, after_cine_loop),
        Question("modality", "Acquisition mode", modality_choices(), True, after_cine_loop),
        Question("identifier_visible", "Is identifying text visible (a name, a record number, a date of birth)?",
                 identifier_choices(), True),
        Question("comments", "Comments (no patient details)", None, False),
    ]


def build_form() -> dict:
    asked = questions()

    def classification(q: Question) -> dict:
        top_level = q.asked_after is None
        if q.choices is None:
            return _classification("text", q.name, q.text, required=q.required, top_level=top_level)
        options = [_option(value, label, [classification(n) for n in asked if n.asked_after == (q.name, value)])
                   for value, label in q.choices.items()]
        out = _classification("radio", q.name, q.text, required=q.required, top_level=top_level, options=options)
        if q.name == "view":
            out["uiMode"] = "searchable"
        return out

    return {"tools": [], "classifications": [classification(q) for q in asked if q.asked_after is None]}


def _is_asked(q: Question, answers: dict) -> bool:
    return q.asked_after is None or answers.get(q.asked_after[0]) == q.asked_after[1]


def review_problems(answers: dict) -> list[str]:
    """What is wrong with one decoded review (`column_review.decode_annotations`); empty when it is valid."""
    problems = []
    for q in questions():
        value = answers.get(q.name)
        if value is None:
            if q.required and _is_asked(q, answers):
                problems.append(f"{q.name} is required")
        elif not _is_asked(q, answers):
            problems.append(f"{q.name} is asked only when {q.asked_after[0]} is {q.asked_after[1]!r}")
        elif q.choices is not None and value not in q.choices:
            problems.append(f"{q.name} has no answer {value!r}")
    return problems


def with_not_applicable(answers: dict) -> dict:
    """A review's answers with every question it was not asked recorded as not applicable: the view and the
    acquisition mode of an item that is not a cine loop."""
    return {**answers, **{q.name: NOT_APPLICABLE for q in questions() if not _is_asked(q, answers)}}


def route(answers: dict) -> dict | None:
    """The `view`, `modality` and `input_kind` arguments of `build_candidate` for a relabelled item, or None.

    Only a valid review of a cine loop, in a view the column review routes, in B-mode or color Doppler, and with no
    identifying text, goes to the column review.
    """
    if review_problems(answers) or answers["identifier_visible"] != "no" or answers["input_type"] != CINE_LOOP:
        return None
    view, modality = answers["view"], answers["modality"]
    route_columns = load_column_spec()["views"].get(view, {}).get("columns")
    if modality not in ROUTED_MODALITIES or view == UNCLASSIFIED_VIEW or not route_columns:
        return None
    return {"view": view, "modality": modality, "input_kind": "video"}


def _form_json() -> str:
    return json.dumps(build_form(), ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the form JSON is out of date")
    args = parser.parse_args(argv)
    form_json = _form_json()
    if args.check:
        if _read(FORM_PATH) != form_json:
            print(f"Out of date: {os.path.relpath(FORM_PATH, REPO_ROOT)}", file=sys.stderr)
            return 1
        return 0
    with open(FORM_PATH, "w", encoding="utf-8") as f:
        f.write(form_json)
    return 0


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
