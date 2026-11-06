"""The view review: one question per clip, to relabel views and validate the view classifier's classes.

Each clip opens with the classifier's predicted view. `PSAX` is not a choice, because a PSAX clip's label is its level:
it opens with no view, and the required question keeps Submit disabled until the reviewer gives one. A clip that is not
clear or of poor quality is `unsure`. Comments are optional, and the place to report identifying text.

    python -m tools.labelbox.view_review            # write the form JSON.
    python -m tools.labelbox.view_review --check    # exit 1 if it is out of date.

`data/view_relabel.py` draws and prepares the clips, and `tools.labelbox.pilot_upload` uploads them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from app.radar.ontology import REPO_ROOT
from tools.labelbox import view_type_review
from tools.labelbox.review_ontology import _classification, _option

FORM_NAME = "DeepECHO view review v1"
FORM_PATH = os.path.join(REPO_ROOT, "configs", "radar", "labelbox", "view_review_v1.json")
UNSURE = "unsure"
NO_LEVEL = "PSAX"  # the classifier's class without a level: never a prefill, never an answer.


def view_choices() -> dict[str, str]:
    """The view and type review's views, without `PSAX` and "Cannot tell", which `unsure` replaces."""
    choices = view_type_review.view_choices()
    return {**{view: label for view, label in choices.items() if view not in (NO_LEVEL, view_type_review.CANNOT_TELL)},
            UNSURE: "unsure (not clear / poor quality)"}


def build_form() -> dict:
    view = _classification("radio", "view", "Which view does this clip show? (prefilled by the view classifier)",
                           required=True, top_level=True,
                           options=[_option(value, label) for value, label in view_choices().items()])
    view["uiMode"] = "searchable"
    comments = _classification("text", "comments", "Comments (report any identifying text here; no patient details)",
                               required=False, top_level=True)
    return {"tools": [], "classifications": [view, comments]}


def prelabels(item: dict) -> list[dict]:
    """The prefill of one clip (`global_key`, `view`, the classifier's prediction): its view, unless it has no level."""
    if item["view"] == NO_LEVEL:
        return []
    if item["view"] not in view_choices():
        raise ValueError("A predicted view is not a choice of the view review")
    return [{"name": "view", "answer": {"name": item["view"]}, "dataRow": {"globalKey": item["global_key"]},
             "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{FORM_NAME}:{item['global_key']}:view"))}]


def review_problems(answers: dict) -> list[str]:
    """Why a submitted review cannot be counted; empty when it is valid."""
    return [] if answers.get("view") in view_choices() else ["view is required"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed form is out of date")
    args = parser.parse_args(argv)
    text = json.dumps(build_form(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        with open(FORM_PATH, encoding="utf-8") as f:
            if f.read() != text:
                print(f"{FORM_PATH} is out of date: run python -m tools.labelbox.view_review", file=sys.stderr)
                return 1
        return 0
    with open(FORM_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
