"""The Labelbox review form for RADAR candidate captions, built from the `review` section of the caption ontology.

    python -m tools.labelbox.review_ontology            # write `configs/radar/labelbox/<ontology tag>.json`.
    python -m tools.labelbox.review_ontology --check    # exit 1 if the committed JSON is out of date.

Every question is a global (whole-video) classification:

- uninterpretable_clip     radio      no; yes -> a reason (assessment limitations).
- view_correct             radio      yes; no -> the correct view class; unsure; not applicable.
- caption_supported        radio      yes; no; unsure; not applicable.
- missing_information      radio      no; yes -> missing concepts and free text; not applicable.
- unsupported_information  radio      no; yes -> unsupported statements (not supported / wrong value); not applicable.
- unconfirmed_statements   checklist  optional; each statement with a reason (uncertain or a limitation).
- corrected_caption        text       required (preloaded with the candidate caption).
- comments                 text       optional.

Statements are the acquisition mode plus the concepts a caption can state (`Ontology.statement_concepts`), so a
reviewer can flag a visibility claim as well as a finding. The JSON is laid out exactly as the Labelbox SDK's
`OntologyBuilder.asdict()` writes it, so a round trip through `OntologyBuilder.from_dict` changes nothing; its extra
top-level keys (`name`, `review_version`, `concept_ontology`) are ours and the SDK drops them. Nothing here talks to
Labelbox.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from app.radar.ontology import ACQUISITION_STATEMENT, REPO_ROOT, VIEWS, Ontology, load_ontology

LABELBOX_DIR = os.path.join(REPO_ROOT, "configs", "radar", "labelbox")
UNSUPPORTED_KIND_SUFFIX = "__unsupported_kind"
UNCONFIRMED_REASON_SUFFIX = "__unconfirmed_reason"
LIMITATION_SUFFIX = "__reason"
CORRECT_VIEW = "correct_view"
MISSING_CONCEPTS = "missing_concepts"
MISSING_OTHER = "missing_other"
UNSUPPORTED_STATEMENTS = "unsupported_statements"


def review_version(ontology: Ontology) -> str:
    """The version stamped on every review: the form is part of the ontology, so it is the ontology's tag."""
    return ontology.tag


def default_json_path(ontology: Ontology) -> str:
    return os.path.join(LABELBOX_DIR, f"{ontology.tag}.json")


def bilingual(labels) -> str:
    return f"{labels['fr']} / {labels['en']}"


def _classification(kind: str, name: str, instructions: str, *, required: bool, options=(), top_level=False) -> dict:
    """One classification, with the SDK's keys in the SDK's order (nested ones have no scope)."""
    out = {"type": kind, "instructions": instructions, "name": name, "required": required, "options": list(options),
           "schemaNodeId": None, "featureSchemaId": None, "attributes": None}
    if top_level:
        out["scope"] = "global"
    return out


def _option(value: str, label: str, nested=()) -> dict:
    return {"schemaNodeId": None, "featureSchemaId": None, "label": label, "value": value, "options": list(nested)}


def statement_ids(ontology: Ontology) -> tuple[str, ...]:
    return (ACQUISITION_STATEMENT,) + ontology.statement_concepts


def statement_label(ontology: Ontology, statement_id: str) -> str:
    if statement_id == ACQUISITION_STATEMENT:
        return bilingual(ontology.review.acquisition_statement)
    return bilingual(ontology.concept(statement_id).labels)


def _detail(ontology: Ontology, detail: str, question_id: str) -> list[dict]:
    """The classifications nested under an answer whose `detail` is `detail`."""
    spec = ontology.review
    labels = spec.detail_labels
    if detail == "limitation":
        return [_classification("radio", f"{question_id}{LIMITATION_SUFFIX}", bilingual(labels["limitation"]),
                                required=True,
                                options=[_option(r, bilingual(lab)) for r, lab in ontology.limitations.items()])]
    if detail == "correct_view":
        return [_classification("radio", CORRECT_VIEW, bilingual(labels["correct_view"]), required=True,
                                options=[_option(v, f"{v}: {bilingual(ontology.views[v].labels)}") for v in VIEWS])]
    if detail == "missing":
        return [
            _classification("checklist", MISSING_CONCEPTS, bilingual(labels["missing"]), required=False,
                            options=[_option(c, statement_label(ontology, c)) for c in ontology.statement_concepts]),
            _classification("text", MISSING_OTHER, bilingual(labels["missing_other"]), required=False),
        ]
    if detail == "unsupported":
        kinds = [_option(k, bilingual(lab)) for k, lab in spec.unsupported_kinds.items()]
        options = [
            _option(s, statement_label(ontology, s), [_classification(
                "radio", f"{s}{UNSUPPORTED_KIND_SUFFIX}", bilingual(labels["unsupported_kind"]), required=True,
                options=kinds)])
            for s in statement_ids(ontology)
        ]
        return [_classification("checklist", UNSUPPORTED_STATEMENTS, bilingual(labels["unsupported"]), required=True,
                                options=options)]
    raise ValueError(f"unknown answer detail {detail!r}")


def unconfirmed_reasons(ontology: Ontology) -> dict:
    """Reason id -> labels for an unconfirmed statement: the review's own reasons, then the assessment limitations."""
    return {**ontology.review.unconfirmed_reasons, **ontology.limitations}


def build_review_ontology(ontology: Ontology) -> dict:
    classifications = []
    for q in ontology.review.questions:
        if q.type == "radio":
            options = [_option(a.id, bilingual(a.labels), _detail(ontology, a.detail, q.id) if a.detail else ())
                       for a in q.answers]
        elif q.type == "checklist":  # the unconfirmed statements, each with a reason
            reasons = [_option(r, bilingual(lab)) for r, lab in unconfirmed_reasons(ontology).items()]
            options = [
                _option(s, statement_label(ontology, s), [_classification(
                    "radio", f"{s}{UNCONFIRMED_REASON_SUFFIX}",
                    bilingual(ontology.review.detail_labels["unconfirmed_reason"]), required=True, options=reasons)])
                for s in statement_ids(ontology)
            ]
        else:
            options = []
        classifications.append(_classification(q.type, q.id, bilingual(q.text), required=q.required,
                                               options=options, top_level=True))
    return {
        "name": f"RADAR caption review ({ontology.tag})",
        "review_version": review_version(ontology),
        "concept_ontology": {"name": ontology.name, "version": ontology.version, "sha256": ontology.sha256},
        "tools": [],
        "classifications": classifications,
    }


def serialize(form: dict) -> str:
    return json.dumps(form, indent=2, ensure_ascii=False) + "\n"


def walk(classifications, depth: int = 1):
    """(classification, depth) for every classification, nested ones included, depth-first."""
    for c in classifications:
        yield c, depth
        for option in c.get("options", []):
            yield from walk(option.get("options", []), depth + 1)


def feature_names(review_ontology: dict) -> set[str]:
    """Every classification name, nested ones included, for checking exports against the form."""
    return {c["name"] for c, _ in walk(review_ontology["classifications"])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write or check the committed Labelbox review form JSON.")
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed JSON is out of date")
    args = parser.parse_args(argv)
    ontology = load_ontology()
    path = default_json_path(ontology)
    text = serialize(build_review_ontology(ontology))
    current = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            current = f.read()
    if args.check:
        if current != text:
            print(f"{os.path.relpath(path, REPO_ROOT)} is out of date: run python -m tools.labelbox.review_ontology")
            return 1
        return 0
    if current != text:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
