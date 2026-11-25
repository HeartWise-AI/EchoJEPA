"""The Labelbox review form for RADAR candidate captions, built from the concept ontology.

One global (whole-video) classification per question:

- caption_satisfactory     radio      is the caption acceptable as written?
- clip_assessability       radio      can the clip be assessed at all (quality, crop, modality)?
- missing_visible_concepts checklist  visible, clinically relevant concepts the caption leaves out
- unsupported_concepts     checklist  claimed concepts the video does not support (hallucinations)
- wrong_value_concepts     checklist  claimed concepts with a wrong value; each has a nested corrected value
- uncertain_concepts       checklist  concepts the reviewer is unsure about; each has a nested reason
- unassessable_concepts    checklist  concepts that cannot be judged from this clip; each has a nested reason
- corrected_caption        text       the caption as it should read (preloaded with the candidate)
- comments                 text

Visibility ("mitral valve visible") and findings ("moderate mitral regurgitation") are separate options, so a
reviewer can confirm a structure without endorsing a finding. The dictionary mirrors the Labelbox SDK's
`OntologyBuilder.asdict()` layout so it can be loaded with `OntologyBuilder.from_dict` once access is approved;
nothing here talks to Labelbox.
"""

from __future__ import annotations

from app.radar.ontology import Ontology
from app.radar.records import UNASSESSABLE_REASONS

REVIEW_FORM_VERSION = "radar-review/0.2"

SATISFACTION = {"yes": True, "no": False}
ASSESSABILITY = {  # option value -> Review.clip_unassessable_reason
    "assessable": None,
    "limited_quality": "quality",
    "limited_crop": "crop",
    "wrong_modality": "modality",
    "other_limitation": "other",
}
CONCEPT_CHECKLISTS = (
    "missing_visible_concepts", "unsupported_concepts", "wrong_value_concepts", "uncertain_concepts",
    "unassessable_concepts",
)
CORRECTED_VALUE_SUFFIX = "__corrected_value"
# Nested reason radios are named per checklist so that every feature name in the form is unique.
REASON_SUFFIXES = {"uncertain_concepts": "__uncertain_reason", "unassessable_concepts": "__unassessable_reason"}


def review_version(ontology: Ontology) -> str:
    """The version stamped on every review: the form version plus the concept ontology it lists."""
    return f"{REVIEW_FORM_VERSION}+{ontology.tag}"


def _option(value: str, label: str, nested: list | None = None) -> dict:
    return {"schemaNodeId": None, "featureSchemaId": None, "label": label, "value": value, "options": nested or []}


def _classification(kind: str, name: str, instructions: str, *, required: bool = False,
                    options: list | None = None) -> dict:
    return {
        "type": kind, "name": name, "instructions": instructions, "required": required, "options": options or [],
        "schemaNodeId": None, "featureSchemaId": None, "scope": "global",
    }


def concept_label(ontology: Ontology, concept_id: str) -> str:
    c = ontology.concept(concept_id)
    return f"{c.labels['fr']} / {c.labels['en']}"


def _reason_radio(checklist: str):
    labels = {"quality": "Image quality", "crop": "Crop or zoom", "modality": "Modality", "other": "Other"}

    def make(concept_id: str) -> dict:
        return _classification(
            "radio", f"{concept_id}{REASON_SUFFIXES[checklist]}", "Why?", required=True,
            options=[_option(r, labels[r]) for r in UNASSESSABLE_REASONS],
        )

    return make


def build_review_ontology(ontology: Ontology) -> dict:
    findings = [cid for cid, c in ontology.concepts.items() if c.kind == "finding"]
    every = list(ontology.concept_ids)

    def checklist(name, instructions, ids, nested=None):
        options = [_option(cid, concept_label(ontology, cid), [nested(cid)] if nested else None) for cid in ids]
        return _classification("checklist", name, instructions, options=options)

    classifications = [
        _classification(
            "radio", "caption_satisfactory",
            "Does the caption describe this clip correctly and completely enough to keep as written?", required=True,
            options=[_option("yes", "Yes, keep as written"), _option("no", "No, it needs changes")],
        ),
        _classification(
            "radio", "clip_assessability", "Can this clip be assessed?", required=True,
            options=[
                _option("assessable", "Assessable"),
                _option("limited_quality", "Not assessable: image quality"),
                _option("limited_crop", "Not assessable: crop or zoom"),
                _option("wrong_modality", "Not assessable: modality (e.g. not the stated mode)"),
                _option("other_limitation", "Not assessable: other"),
            ],
        ),
        checklist("missing_visible_concepts",
                  "Visible, clinically relevant structures or findings that the caption leaves out.", every),
        checklist("unsupported_concepts", "Concepts the caption states that this video does not support.", findings),
        checklist("wrong_value_concepts", "Concepts the caption states with a wrong value or severity.", findings,
                  nested=lambda cid: _classification("text", f"{cid}{CORRECTED_VALUE_SUFFIX}",
                                                     "Correct value (as you would write it in the caption)")),
        checklist("uncertain_concepts", "Concepts you are unsure about in this clip.", every,
                  nested=_reason_radio("uncertain_concepts")),
        checklist("unassessable_concepts", "Concepts that cannot be judged from this clip.", every,
                  nested=_reason_radio("unassessable_concepts")),
        _classification("text", "corrected_caption",
                        "The caption as it should read. Leave the preloaded text unchanged if it is correct. "
                        "If you flagged a concept as unsupported, uncertain or unassessable, edit it out of the "
                        "caption: an unchanged caption with any flag is not approved."),
        _classification("text", "comments", "Anything else the caption team should know (no patient details)."),
    ]
    return {
        "name": f"RADAR caption review {REVIEW_FORM_VERSION} ({ontology.tag})",
        "review_version": review_version(ontology),
        "concept_ontology": {"name": ontology.name, "version": ontology.version, "sha256": ontology.sha256},
        "tools": [],
        "classifications": classifications,
    }


def feature_names(review_ontology: dict) -> set[str]:
    """Every classification name, nested ones included, for checking exports against the form."""
    names = set()

    def walk(classifications):
        for c in classifications:
            names.add(c["name"])
            for option in c.get("options", []):
                walk(option.get("options", []))

    walk(review_ontology["classifications"])
    return names
