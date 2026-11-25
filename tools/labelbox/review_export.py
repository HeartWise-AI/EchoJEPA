"""Map a Labelbox project export (NDJSON, export v2 layout) back onto RADAR clip records.

Each exported data row is matched to its record by `global_key` (the pseudonymous record id). Its single label's
global classifications become a `Review`; a label whose workflow status is Done becomes `reviewed`, anything else
`in_review`. Approved captions are then derived by `app.radar.records.apply_review`, so approval exists only where
returned review data supports it.

Reviewer accounts are e-mail addresses in Labelbox. They are replaced by pseudonymous handles from an explicit map,
or by a hash prefix when no map is given; addresses never reach the records.

Anything unexpected (an unknown global key, a feature the review form does not define, several labels on one data
row, a Done label without the required answers) stops the import instead of being skipped.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any, Iterable, Mapping

from app.radar.findings import map_code, parse_number
from app.radar.ontology import Ontology, normalize_text
from app.radar.records import ClipRecord, ConceptReason, Review, WrongValue, apply_review
from tools.labelbox.review_ontology import (
    ASSESSABILITY,
    CORRECTED_VALUE_SUFFIX,
    REASON_SUFFIXES,
    SATISFACTION,
    build_review_ontology,
    feature_names,
    review_version,
)

DONE_STATUSES = {"DONE"}


class ExportError(ValueError):
    """The export cannot be mapped onto the records without guessing."""


def _key(answer: Mapping) -> str:
    """Exports carry both a display `name` and the schema `value`; the form's identifiers are the values."""
    value = answer.get("value") or answer.get("name")
    if not isinstance(value, str):
        raise ExportError("An exported answer has no name or value.")
    return value


def _utc(value: Any) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ExportError("An exported timestamp is not ISO-8601.") from None
    if parsed.tzinfo is None:
        raise ExportError("An exported timestamp has no time zone.")
    return parsed.astimezone(datetime.timezone.utc).replace(microsecond=0).isoformat()


def pseudonymous_reviewer(account: str, reviewer_map: Mapping[str, str] | None) -> str:
    if reviewer_map is not None:
        if account not in reviewer_map:
            raise ExportError("A reviewer account is missing from the reviewer map.")
        handle = reviewer_map[account]
        if "@" in handle:
            raise ExportError("Reviewer handles must not be e-mail addresses.")
        return handle
    return "reviewer-" + hashlib.sha256(account.strip().lower().encode()).hexdigest()[:12]


def parse_corrected_value(text: str, ontology: Ontology, concept_id: str) -> Any:
    """A reviewer's corrected value in ontology terms when it can be read unambiguously, else the text as given."""
    c = ontology.concept(concept_id)
    raw = (text or "").strip()
    if c.value_type == "numeric":
        number = parse_number(raw)
        return number if number is not None and not ontology.value_error(concept_id, number) else raw
    if c.value_type == "boolean":
        word = normalize_text(raw)
        return {"oui": True, "yes": True, "present": True, "non": False, "no": False, "absent": False}.get(word, raw)
    mapped = map_code(raw, ontology, concept_id)
    return mapped if mapped is not None else raw


def _answers(classifications: list, known: set) -> dict:
    """Feature value -> answer object, for the global classifications of one label."""
    out = {}
    for c in classifications or []:
        name = _key(c)
        if name not in known:
            raise ExportError(f"The export has a feature {name!r} that this review form does not define.")
        if name in out:
            raise ExportError(f"The feature {name!r} is answered twice in one label.")
        out[name] = c
    return out


def _radio(answer: Mapping | None) -> str | None:
    if answer is None:
        return None
    radio = answer.get("radio_answer")
    return _key(radio) if radio else None


def _checklist(answer: Mapping | None) -> list[Mapping]:
    if answer is None:
        return []
    return list(answer.get("checklist_answers") or [])


def _text(answer: Mapping | None) -> str | None:
    if answer is None:
        return None
    content = (answer.get("text_answer") or {}).get("content")
    return content.strip() if isinstance(content, str) and content.strip() else None


def _nested(option: Mapping, name: str, known: set) -> Mapping | None:
    return _answers(option.get("classifications") or [], known).get(name)


def label_to_review(label: Mapping, workflow_status: str | None, record: ClipRecord, ontology: Ontology, *,
                    known: set, form_version: str, reviewer_map: Mapping[str, str] | None) -> Review:
    annotations = label.get("annotations") or {}
    for frame in (annotations.get("frames") or {}).values():
        if frame.get("classifications"):
            raise ExportError("Frame-level classifications are not part of the review form; answer globally.")
    answers = _answers(annotations.get("classifications"), known)
    details = label.get("label_details") or {}
    account = details.get("created_by")
    if not isinstance(account, str) or not account:
        raise ExportError("A label has no reviewer account.")
    status = "reviewed" if (workflow_status or "").upper().replace("_", "") in DONE_STATUSES else "in_review"

    satisfied = _radio(answers.get("caption_satisfactory"))
    assess = _radio(answers.get("clip_assessability"))
    if satisfied is not None and satisfied not in SATISFACTION or assess is not None and assess not in ASSESSABILITY:
        raise ExportError("A radio answer is not one of the form's options.")
    if status == "reviewed" and (satisfied is None or assess is None):
        raise ExportError(f"Label for {record.record_id} is Done without the required answers.")

    def concept_ids(name):
        ids = [_key(o) for o in _checklist(answers.get(name))]
        unknown = [cid for cid in ids if cid not in ontology.concepts]
        if unknown:
            raise ExportError(f"{name} lists concepts outside {ontology.tag}: {unknown}")
        return ids

    claimed = {c.concept_id: c.value for c in record.candidate_caption.claims} if record.candidate_caption else {}
    wrong = []
    for option in _checklist(answers.get("wrong_value_concepts")):
        cid = _key(option)
        if cid not in ontology.concepts:
            raise ExportError(f"wrong_value_concepts lists a concept outside {ontology.tag}: {cid!r}")
        text = _text(_nested(option, f"{cid}{CORRECTED_VALUE_SUFFIX}", known))
        corrected = parse_corrected_value(text, ontology, cid) if text else None
        wrong.append(WrongValue(cid, claimed.get(cid), corrected))

    def reasons(name):
        out = []
        for option in _checklist(answers.get(name)):
            cid = _key(option)
            if cid not in ontology.concepts:
                raise ExportError(f"{name} lists a concept outside {ontology.tag}: {cid!r}")
            reason = _radio(_nested(option, f"{cid}{REASON_SUFFIXES[name]}", known)) or "other"
            out.append(ConceptReason(cid, reason))
        return out

    return Review(
        status=status,
        reviewer_id=pseudonymous_reviewer(account, reviewer_map),
        review_version=form_version,
        source="labelbox_export",
        label_ref=f"labelbox:{label.get('id')}",
        started_utc=_utc(details.get("created_at")),
        completed_utc=_utc(details.get("updated_at")) if status == "reviewed" else None,
        satisfied=SATISFACTION.get(satisfied) if satisfied is not None else None,
        missing_concepts=concept_ids("missing_visible_concepts"),
        unsupported_concepts=concept_ids("unsupported_concepts"),
        wrong_values=wrong,
        uncertain=reasons("uncertain_concepts"),
        unassessable=reasons("unassessable_concepts"),
        clip_unassessable_reason=ASSESSABILITY.get(assess) if assess is not None else None,
        corrected_caption=_text(answers.get("corrected_caption")),
        comments=_text(answers.get("comments")),
    )


def read_export(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        try:
            return [json.loads(line) for line in f if line.strip()]
        except json.JSONDecodeError:
            raise ExportError("The export is not valid NDJSON.") from None


def apply_export(records: list[ClipRecord], export_rows: Iterable[Mapping], ontology: Ontology, *,
                 expected_review_version: str, project_id: str | None = None,
                 reviewer_map: Mapping[str, str] | None = None) -> tuple[list[ClipRecord], dict]:
    """Records updated with the export's reviews, and counts of what happened (no identifiers)."""
    form_version = review_version(ontology)
    if expected_review_version != form_version:
        raise ExportError(f"The export is declared as {expected_review_version!r} but this ontology's review form "
                          f"is {form_version!r}; import it with the ontology it was collected under.")
    known = feature_names(build_review_ontology(ontology))
    by_id = {r.record_id: r for r in records}
    reviews = {}
    for row in export_rows:
        key = (row.get("data_row") or {}).get("global_key")
        if key not in by_id:
            raise ExportError("The export has a data row whose global key is not in the records.")
        if key in reviews:
            raise ExportError(f"Data row {key} appears twice in the export.")
        projects = row.get("projects") or {}
        if project_id is not None:
            projects = {pid: p for pid, p in projects.items() if pid == project_id}
        if len(projects) != 1:
            raise ExportError(f"Data row {key} is in {len(projects)} projects; pass the project id to choose one.")
        project = next(iter(projects.values()))
        labels = project.get("labels") or []
        if not labels:
            continue
        if len(labels) > 1:
            raise ExportError(f"Data row {key} has {len(labels)} labels; review {form_version} expects one.")
        workflow = (project.get("project_details") or {}).get("workflow_status")
        reviews[key] = label_to_review(labels[0], workflow, by_id[key], ontology, known=known,
                                       form_version=form_version, reviewer_map=reviewer_map)
    updated = [apply_review(r, reviews[r.record_id], ontology) if r.record_id in reviews else r for r in records]
    counts = {
        "records": len(records),
        "reviews": len(reviews),
        "reviewed": sum(1 for v in reviews.values() if v.status == "reviewed"),
        "in_review": sum(1 for v in reviews.values() if v.status == "in_review"),
        "approved_captions": sum(1 for r in updated if r.approved_caption is not None),
    }
    return updated, counts
