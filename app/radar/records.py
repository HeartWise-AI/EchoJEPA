"""Local RADAR clip records: one JSON object per clip, stored as JSONL outside the repository.

A record ties a video to its study's report findings, a candidate caption and the review of that caption. Its
`study_key` and `patient_key` are local grouping keys: the loss uses them to avoid treating same-study clips as
negatives. They are never model inputs, never caption text and never leave the local data store. `record_id` is the
pseudonymous key used everywhere else (Labelbox global keys, logs).

A record's `split` is its patient's split in the shared archive patient split (`patient_splits.csv` of
`data/build_manifests.py`), whose SHA-256 it carries as `split_source`. A file holds one split source and one split
per patient and per study (`split_problems`); the split guard of `app/radar/data.py` relies on both.

Supervision is layered and never mixed silently:
  report_findings   -> what the report says about the study (with provenance);
  candidate_caption -> an *unvalidated* caption built from those findings and the clip's view/acquisition prior;
  review            -> what a reviewer returned after inspecting the video;
  approved_caption  -> derived only from returned review data (`apply_review`), never written by hand.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import hmac
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Iterable, Optional

from app.radar.ontology import (
    ACQUISITION_CATEGORIES,
    GAP_REASONS,
    LANGUAGES,
    RELEVANT,
    REPO_ROOT,
    VIEWS,
    Ontology,
)

SCHEMA_VERSION = "radar-clip-record/0.1"
SPLITS = ("train", "val", "test")
REVIEW_STATUSES = ("unreviewed", "in_review", "reviewed")
REVIEW_SOURCES = ("labelbox_export", "local_review")
CANDIDATE_STATUS = "unvalidated"
GENERATORS = ("rule_based", "llm")
ACQ_CONFIDENCE = ("header", "unknown")
UNASSESSABLE_REASONS = ("quality", "crop", "modality", "other")
FINDING_METHODS = ("structured_column", "phrase_hint", "manual", "synthetic")
APPROVAL_SOURCES = ("corrected_caption", "candidate_accepted")
MAX_CAPTION_CHARS = 1000
RECORD_ID = re.compile(r"^rc-[0-9a-f]{16,64}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class RecordError(ValueError):
    """A clip record (or a JSONL file of them) is malformed; the message lists every problem found."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def make_record_id(secret: str, video_key: str, length: int = 24) -> str:
    """A stable pseudonymous id: HMAC-SHA256 of the video's local key under a secret kept outside the repository.

    Without the secret the id cannot be linked back to the video, study or patient.
    """
    if not secret:
        raise ValueError("A non-empty secret is required to derive pseudonymous record ids.")
    digest = hmac.new(secret.encode(), str(video_key).encode(), hashlib.sha256).hexdigest()
    return f"rc-{digest[:length]}"


# --------------------------------------------------------------------------- dataclasses


@dataclass
class Acquisition:
    """The acquisition index's verdict for this video (shared contract)."""

    category: str
    confidence: str = "unknown"
    rule: Optional[str] = None
    has_tissue_2d: Optional[bool] = None
    has_color_flow: Optional[bool] = None
    has_pw_doppler: Optional[bool] = None
    has_cw_doppler: Optional[bool] = None
    has_mmode: Optional[bool] = None


@dataclass
class Focus:
    """What the clip is zoomed or cropped on, when known. All fields are optional."""

    zoomed: Optional[bool] = None
    target_structure: Optional[str] = None
    crop_box: Optional[list] = None  # [x0, y0, x1, y1] as fractions of the frame
    source: Optional[str] = None  # how it was determined, e.g. "view_label" or "manual"


@dataclass
class Provenance:
    source_field: str  # report-export column, e.g. "Referring MR Grade"
    method: str  # one of FINDING_METHODS
    extractor_version: str
    span: Optional[list] = None  # [start, end) character offsets in the source field, never the text itself
    hint: Optional[str] = None  # the ontology phrase hint that matched, for phrase_hint findings


@dataclass
class ReportFinding:
    concept_id: str
    value: Any
    provenance: Provenance
    severity: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class Claim:
    concept_id: str
    value: Any


@dataclass
class ConceptGap:
    concept_id: str
    reason: str
    required_modality: list = field(default_factory=list)


@dataclass
class CandidateCaption:
    text: str
    language: str
    generator: str
    generator_version: str
    ontology_version: str
    created_utc: str
    prompt_version: Optional[str] = None
    status: str = CANDIDATE_STATUS
    claims: list = field(default_factory=list)  # [Claim]: findings the caption states
    gaps: list = field(default_factory=list)  # [ConceptGap]: findings this clip cannot ground
    out_of_view: list = field(default_factory=list)  # [concept_id]: findings outside this view's prior
    expected_structures: list = field(default_factory=list)  # [concept_id]: visibility prior, for reviewers


@dataclass
class ConceptReason:
    concept_id: str
    reason: str  # one of UNASSESSABLE_REASONS


@dataclass
class WrongValue:
    concept_id: str
    claimed_value: Any
    corrected_value: Any = None


@dataclass
class Review:
    status: str = "unreviewed"
    reviewer_id: Optional[str] = None  # pseudonymous reviewer handle, never an e-mail address
    review_version: Optional[str] = None
    source: Optional[str] = None
    label_ref: Optional[str] = None
    started_utc: Optional[str] = None
    completed_utc: Optional[str] = None
    satisfied: Optional[bool] = None
    missing_concepts: list = field(default_factory=list)  # visible, relevant, absent from the caption
    unsupported_concepts: list = field(default_factory=list)  # claimed but not supported by the video
    wrong_values: list = field(default_factory=list)  # [WrongValue]
    uncertain: list = field(default_factory=list)  # [ConceptReason]: reviewer unsure
    unassessable: list = field(default_factory=list)  # [ConceptReason]: cannot be judged from this clip
    clip_unassessable_reason: Optional[str] = None
    corrected_caption: Optional[str] = None
    comments: Optional[str] = None


@dataclass
class ApprovedCaption:
    text: str
    language: str
    derived_from: str
    review_version: str
    reviewer_id: str
    label_ref: str
    approved_utc: str
    ontology_version: str
    concepts: list = field(default_factory=list)  # [Claim] after the review's corrections


@dataclass
class ClipRecord:
    record_id: str
    study_key: str
    patient_key: str
    split: str
    video_path: str
    view: str
    acquisition: Acquisition
    view_confidence: Optional[float] = None
    focus: Optional[Focus] = None
    report_findings: list = field(default_factory=list)  # [ReportFinding]
    candidate_caption: Optional[CandidateCaption] = None
    review: Review = field(default_factory=Review)
    approved_caption: Optional[ApprovedCaption] = None
    split_source: Optional[str] = None  # SHA-256 of the archive patient_splits.csv that gave `split`
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def finding(self, concept_id: str) -> ReportFinding | None:
        return next((f for f in self.report_findings if f.concept_id == concept_id), None)


_LIST_ITEMS = {
    (ClipRecord, "report_findings"): ReportFinding,
    (CandidateCaption, "claims"): Claim,
    (CandidateCaption, "gaps"): ConceptGap,
    (Review, "wrong_values"): WrongValue,
    (Review, "uncertain"): ConceptReason,
    (Review, "unassessable"): ConceptReason,
    (ApprovedCaption, "concepts"): Claim,
}


def _decode(cls, data, where: str):
    """Strictly decode a JSON object into a dataclass: unknown or missing required fields are errors."""
    if not isinstance(data, dict):
        raise RecordError(f"{where} must be an object")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise RecordError(f"{where} has unknown fields {unknown}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise RecordError(f"{where}.{f.name} is required")
            continue
        value = data[f.name]
        item_cls = _LIST_ITEMS.get((cls, f.name))
        if item_cls is not None:
            if not isinstance(value, list):
                raise RecordError(f"{where}.{f.name} must be a list")
            value = [_decode(item_cls, v, f"{where}.{f.name}[{i}]") for i, v in enumerate(value)]
        else:
            target = hints[f.name]
            args = typing.get_args(target)
            if typing.get_origin(target) in (typing.Union, types.UnionType):
                target = next((a for a in args if a is not type(None)), target)
            if is_dataclass(target) and value is not None:
                value = _decode(target, value, f"{where}.{f.name}")
        kwargs[f.name] = value
    return cls(**kwargs)


def record_from_dict(data: dict) -> ClipRecord:
    return _decode(ClipRecord, data, "record")


# --------------------------------------------------------------------------- validation


def _is_utc(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == datetime.timedelta(0)


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def identifier_values(record: ClipRecord) -> list[str]:
    """Local values that must never appear in caption text or anything sent outside the data store."""
    values = [record.study_key, record.patient_key, record.video_path]
    base = os.path.basename(record.video_path or "")
    values += [base, os.path.splitext(base)[0]]
    return [v for v in values if isinstance(v, str) and len(v.strip()) >= 4]


def _check_text(text, record: ClipRecord, where: str, problems: list) -> None:
    if not _nonempty(text):
        problems.append(f"{where} is empty")
        return
    if len(text) > MAX_CAPTION_CHARS:
        problems.append(f"{where} is longer than {MAX_CAPTION_CHARS} characters")
    if any(v in text for v in identifier_values(record)):
        problems.append(f"{where} contains a local identifier or path")


def _check_claims(claims, ontology: Ontology, where: str, problems: list) -> None:
    seen = set()
    for i, claim in enumerate(claims):
        if claim.concept_id not in ontology.concepts:
            problems.append(f"{where}[{i}]: unknown concept {claim.concept_id!r}")
            continue
        if claim.concept_id in seen:
            problems.append(f"{where}[{i}]: concept {claim.concept_id!r} repeated")
        seen.add(claim.concept_id)
        if claim.value is not None:
            error = ontology.value_error(claim.concept_id, claim.value)
            if error:
                problems.append(f"{where}[{i}]: {error}")


def _check_concept_ids(ids, ontology: Ontology, where: str, problems: list) -> None:
    for cid in ids:
        if cid not in ontology.concepts:
            problems.append(f"{where}: unknown concept {cid!r}")


def record_problems(record: ClipRecord, ontology: Ontology) -> list[str]:
    """Every semantic problem of one record. Messages never include paths, keys or caption text."""
    p: list[str] = []
    if record.schema_version != SCHEMA_VERSION:
        p.append(f"schema_version {record.schema_version!r} is not {SCHEMA_VERSION!r}")
    if not isinstance(record.record_id, str) or not RECORD_ID.match(record.record_id):
        p.append("record_id must look like rc-<16-64 hex characters> (see make_record_id)")
    for name in ("study_key", "patient_key", "video_path"):
        if not _nonempty(getattr(record, name)):
            p.append(f"{name} is required")
    if _nonempty(record.study_key) and _nonempty(record.record_id) and record.study_key in record.record_id:
        p.append("record_id must not embed the study key")
    if record.split not in SPLITS:
        p.append(f"split {record.split!r} not in {list(SPLITS)}")
    if record.split_source is not None and not (isinstance(record.split_source, str)
                                                and SHA256_HEX.match(record.split_source)):
        p.append("split_source must be the SHA-256 (64 lowercase hex characters) of the patient split table")
    if record.view not in VIEWS:
        p.append(f"view {record.view!r} is not a classifier view")
    vc = record.view_confidence
    if vc is not None and (isinstance(vc, bool) or not isinstance(vc, (int, float)) or not 0 <= vc <= 1):
        p.append("view_confidence must be in [0, 1]")
    acq = record.acquisition
    if acq.category not in ACQUISITION_CATEGORIES:
        p.append(f"acquisition.category {acq.category!r} not in {list(ACQUISITION_CATEGORIES)}")
    if acq.confidence not in ACQ_CONFIDENCE:
        p.append(f"acquisition.confidence {acq.confidence!r} not in {list(ACQ_CONFIDENCE)}")
    for flag in ("has_tissue_2d", "has_color_flow", "has_pw_doppler", "has_cw_doppler", "has_mmode"):
        if getattr(acq, flag) not in (None, True, False):
            p.append(f"acquisition.{flag} must be true, false or null")
    if record.focus is not None and record.focus.crop_box is not None:
        box = record.focus.crop_box
        ok = isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)
        if not ok or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            p.append("focus.crop_box must be [x0, y0, x1, y1] fractions with x0 < x1 and y0 < y1")

    seen = set()
    for i, f in enumerate(record.report_findings):
        where = f"report_findings[{i}]"
        if f.concept_id not in ontology.concepts:
            p.append(f"{where}: unknown concept {f.concept_id!r}")
            continue
        if ontology.concept(f.concept_id).kind != "finding":
            p.append(f"{where}: {f.concept_id} is a visibility concept, not a report finding")
        if f.concept_id in seen:
            p.append(f"{where}: a study has one value per concept; {f.concept_id} repeats")
        seen.add(f.concept_id)
        error = ontology.value_error(f.concept_id, f.value)
        if error:
            p.append(f"{where}: {error}")
        allowed = ontology.severity_values(f.concept_id)
        if f.severity is not None and (allowed is None or f.severity not in allowed):
            p.append(f"{where}: severity {f.severity!r} not allowed for {f.concept_id}")
        prov = f.provenance
        if not _nonempty(prov.source_field) or not _nonempty(prov.extractor_version):
            p.append(f"{where}: provenance needs source_field and extractor_version")
        if prov.method not in FINDING_METHODS:
            p.append(f"{where}: provenance.method {prov.method!r} not in {list(FINDING_METHODS)}")
        if prov.span is not None and not (
            isinstance(prov.span, list) and len(prov.span) == 2 and all(isinstance(v, int) for v in prov.span)
            and 0 <= prov.span[0] < prov.span[1]
        ):
            p.append(f"{where}: provenance.span must be [start, end) offsets")

    cap = record.candidate_caption
    if cap is not None:
        where = "candidate_caption"
        if cap.status != CANDIDATE_STATUS:
            p.append(f"{where}.status must stay {CANDIDATE_STATUS!r}; approval lives in approved_caption")
        if cap.language not in LANGUAGES:
            p.append(f"{where}.language {cap.language!r} not in {list(LANGUAGES)}")
        if cap.generator not in GENERATORS:
            p.append(f"{where}.generator {cap.generator!r} not in {list(GENERATORS)}")
        if not _nonempty(cap.generator_version):
            p.append(f"{where}.generator_version is required")
        if cap.generator == "llm" and not _nonempty(cap.prompt_version):
            p.append(f"{where}.prompt_version is required for LLM captions")
        if cap.ontology_version != ontology.version:
            p.append(f"{where} was built with ontology {cap.ontology_version!r}, not {ontology.version!r}")
        if not _is_utc(cap.created_utc):
            p.append(f"{where}.created_utc must be an ISO-8601 UTC time")
        _check_text(cap.text, record, f"{where}.text", p)
        _check_claims(cap.claims, ontology, f"{where}.claims", p)
        for i, claim in enumerate(cap.claims):
            if claim.concept_id not in ontology.concepts:
                continue
            finding = record.finding(claim.concept_id)
            if finding is None or finding.value != claim.value:
                p.append(f"{where}.claims[{i}]: {claim.concept_id} is not a report finding with that value")
            elif record.view in VIEWS and acq.category in ACQUISITION_CATEGORIES:
                outcome = ontology.relevance(claim.concept_id, record.view, acq.category)
                if outcome != RELEVANT:
                    p.append(f"{where}.claims[{i}]: {claim.concept_id} cannot be stated for this clip ({outcome})")
        for i, gap in enumerate(cap.gaps):
            if gap.concept_id not in ontology.concepts or gap.reason not in GAP_REASONS:
                p.append(f"{where}.gaps[{i}]: unknown concept or reason {gap.reason!r}")
        _check_concept_ids(cap.out_of_view, ontology, f"{where}.out_of_view", p)
        _check_concept_ids(cap.expected_structures, ontology, f"{where}.expected_structures", p)

    rev = record.review
    where = "review"
    if rev.status not in REVIEW_STATUSES:
        p.append(f"{where}.status {rev.status!r} not in {list(REVIEW_STATUSES)}")
    answered = any(
        getattr(rev, f.name) not in (None, [], "") for f in fields(Review) if f.name != "status"
    )
    if rev.status == "unreviewed" and answered:
        p.append(f"{where} is unreviewed but carries review data")
    if rev.status != "unreviewed" and cap is None:
        p.append(f"{where}: a review needs a candidate caption to review")
    if rev.status == "reviewed":
        for name in ("reviewer_id", "review_version", "source", "label_ref"):
            if not _nonempty(getattr(rev, name)):
                p.append(f"{where}.{name} is required once reviewed")
        if rev.source not in REVIEW_SOURCES:
            p.append(f"{where}.source {rev.source!r} not in {list(REVIEW_SOURCES)}")
        if not _is_utc(rev.completed_utc):
            p.append(f"{where}.completed_utc must be an ISO-8601 UTC time once reviewed")
        if not isinstance(rev.satisfied, bool):
            p.append(f"{where}.satisfied must be true or false once reviewed")
        if rev.reviewer_id and "@" in rev.reviewer_id:
            p.append(f"{where}.reviewer_id must be a pseudonymous handle, not an e-mail address")
    if rev.started_utc is not None and not _is_utc(rev.started_utc):
        p.append(f"{where}.started_utc must be an ISO-8601 UTC time")
    _check_concept_ids(rev.missing_concepts, ontology, f"{where}.missing_concepts", p)
    _check_concept_ids(rev.unsupported_concepts, ontology, f"{where}.unsupported_concepts", p)
    _check_concept_ids([w.concept_id for w in rev.wrong_values], ontology, f"{where}.wrong_values", p)
    for name in ("uncertain", "unassessable"):
        for i, item in enumerate(getattr(rev, name)):
            if item.concept_id not in ontology.concepts or item.reason not in UNASSESSABLE_REASONS:
                p.append(f"{where}.{name}[{i}]: unknown concept or reason {item.reason!r}")
    if rev.clip_unassessable_reason not in (None,) + UNASSESSABLE_REASONS:
        p.append(f"{where}.clip_unassessable_reason not in {list(UNASSESSABLE_REASONS)}")
    if rev.corrected_caption is not None:
        _check_text(rev.corrected_caption, record, f"{where}.corrected_caption", p)

    app = record.approved_caption
    if app is not None:
        where = "approved_caption"
        if rev.status != "reviewed":
            p.append(f"{where} exists without a completed review; approval comes only from returned review data")
        for name in ("reviewer_id", "label_ref", "review_version"):
            if getattr(app, name) != getattr(rev, name):
                p.append(f"{where}.{name} does not match the review")
        if app.derived_from not in APPROVAL_SOURCES:
            p.append(f"{where}.derived_from {app.derived_from!r} not in {list(APPROVAL_SOURCES)}")
        if app.language not in LANGUAGES:
            p.append(f"{where}.language {app.language!r} not in {list(LANGUAGES)}")
        if app.ontology_version != ontology.version:
            p.append(f"{where} was approved under ontology {app.ontology_version!r}, not {ontology.version!r}")
        if not _is_utc(app.approved_utc):
            p.append(f"{where}.approved_utc must be an ISO-8601 UTC time")
        _check_text(app.text, record, f"{where}.text", p)
        _check_claims(app.concepts, ontology, f"{where}.concepts", p)
        # Every field, the concept values included: the loss groups captions by those concepts.
        expected = derive_approved_caption(record, ontology, approved_utc=app.approved_utc)
        if expected is None or expected != app:
            p.append(f"{where} does not follow from the review (see derive_approved_caption)")
    return p


def validate_record(record: ClipRecord, ontology: Ontology) -> ClipRecord:
    problems = record_problems(record, ontology)
    if problems:
        rid = record.record_id if isinstance(record.record_id, str) and RECORD_ID.match(record.record_id) else "?"
        raise RecordError(f"Clip record {rid} is invalid:\n- " + "\n- ".join(problems))
    return record


# --------------------------------------------------------------------------- reviews and approval


def derive_approved_caption(record: ClipRecord, ontology: Ontology, approved_utc: str | None = None):
    """The caption a completed review approves, or None when the review does not approve one.

    The candidate is approved as-is only when the reviewer was satisfied and flagged nothing: no missing,
    unsupported, wrong, uncertain or unassessable concept. A corrected caption is the reviewer's resolution of their
    flags: it is approved, and every concept marked unsupported, unassessable or uncertain leaves its concept list
    (the review form asks the corrected caption not to state them). A "corrected" caption identical to the candidate
    (the form preloads it) is no correction. Any flag without a correction leaves the clip without supervision.
    """
    rev, cap = record.review, record.candidate_caption
    if rev.status != "reviewed" or cap is None or rev.clip_unassessable_reason is not None:
        return None
    corrected = rev.corrected_caption.strip() if _nonempty(rev.corrected_caption) else None
    if corrected is not None and corrected.split() != cap.text.split():
        text, source = corrected, "corrected_caption"
    elif rev.satisfied and not (rev.missing_concepts or rev.unsupported_concepts or rev.wrong_values
                                or rev.uncertain or rev.unassessable):
        text, source = cap.text, "candidate_accepted"
    else:
        return None
    dropped = (set(rev.unsupported_concepts) | {u.concept_id for u in rev.unassessable}
               | {u.concept_id for u in rev.uncertain})
    corrected = {w.concept_id: w.corrected_value for w in rev.wrong_values}
    concepts = []
    for claim in cap.claims:
        if claim.concept_id in dropped:
            continue
        if claim.concept_id in corrected:
            value = corrected[claim.concept_id]
            if value is None or ontology.value_error(claim.concept_id, value):
                continue  # a wrong value without a usable correction is no longer a known concept value
            concepts.append(Claim(claim.concept_id, value))
        else:
            concepts.append(Claim(claim.concept_id, claim.value))
    claimed = {c.concept_id for c in concepts}
    concepts += [Claim(cid, None) for cid in rev.missing_concepts if cid not in claimed and cid not in dropped]
    return ApprovedCaption(
        text=text, language=cap.language, derived_from=source, review_version=rev.review_version,
        reviewer_id=rev.reviewer_id, label_ref=rev.label_ref, approved_utc=approved_utc or utc_now(),
        ontology_version=ontology.version, concepts=concepts,
    )


def apply_review(record: ClipRecord, review: Review, ontology: Ontology) -> ClipRecord:
    """A copy of `record` with `review` and the approved caption it implies (or none)."""
    updated = dataclasses.replace(record, review=review, approved_caption=None)
    updated.approved_caption = derive_approved_caption(updated, ontology)
    return validate_record(updated, ontology)


def caption_concept_key(record: ClipRecord) -> tuple:
    """What the training caption asserts, as a hashable key: view, acquisition and (concept, value) claims.

    Used by the loss to find semantically equivalent pairs even when their wording differs.
    """
    if record.approved_caption is not None:
        claims = record.approved_caption.concepts
    elif record.candidate_caption is not None:
        claims = record.candidate_caption.claims
    else:
        claims = []
    items = tuple(sorted((c.concept_id, json.dumps(c.value, sort_keys=True)) for c in claims))
    return (record.view, record.acquisition.category, items)


# --------------------------------------------------------------------------- JSONL I/O


def split_problems(records: Iterable[ClipRecord]) -> list[str]:
    """Problems of a set of records taken together: a patient or a study under more than one split, a study under
    more than one patient, or splits from more than one patient split table (`split_source`, unbound included).
    Messages give counts only."""
    patient_splits, study_splits, study_patients, sources = {}, {}, {}, set()
    for r in records:
        patient_splits.setdefault(r.patient_key, set()).add(r.split)
        study_splits.setdefault(r.study_key, set()).add(r.split)
        study_patients.setdefault(r.study_key, set()).add(r.patient_key)
        sources.add(r.split_source)
    p = []
    for what, groups in (("patients have records in more than one split", patient_splits),
                         ("studies have records in more than one split", study_splits),
                         ("studies have records of more than one patient", study_patients)):
        n = sum(len(v) > 1 for v in groups.values())
        if n:
            p.append(f"{n} {what}")
    if len(sources) > 1:
        p.append(f"the records' splits come from {len(sources)} different split tables (split_source"
                 + (", some unbound)" if None in sources else ")"))
    return p


def read_records(path: str, ontology: Ontology) -> list[ClipRecord]:
    """Read and validate a JSONL file; every problem of every line is reported before raising."""
    records, problems, ids = [], [], set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = record_from_dict(json.loads(line))
            except (json.JSONDecodeError, RecordError, TypeError) as e:
                problems.append(f"line {lineno}: {type(e).__name__}: {str(e).splitlines()[0]}")
                continue
            issues = record_problems(record, ontology)
            problems += [f"line {lineno}: {issue}" for issue in issues]
            if record.record_id in ids:
                problems.append(f"line {lineno}: duplicate record_id")
            ids.add(record.record_id)
            records.append(record)
    problems += [f"file: {issue}" for issue in split_problems(records)]
    if problems:
        shown = problems[:50] + ([f"... and {len(problems) - 50} more"] if len(problems) > 50 else [])
        raise RecordError(f"{os.path.basename(path)} has invalid clip records:\n- " + "\n- ".join(shown))
    return records


def write_records(path: str, records: Iterable[ClipRecord], ontology: Ontology) -> None:
    """Validate, then write atomically so a reader never sees a half-written file."""
    records = [validate_record(r, ontology) for r in records]
    ids = [r.record_id for r in records]
    if len(set(ids)) != len(ids):
        raise RecordError("record_ids must be unique within a file.")
    problems = split_problems(records)
    if problems:
        raise RecordError("The records cannot share a file: " + "; ".join(problems) + ".")
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def ensure_outside_repo(path: str, flag: str) -> None:
    """Refuse output inside the repository, with the same rule as `data/build_manifests.ensure_outside_repo`."""
    build_manifests().ensure_outside_repo(path, flag)


def build_manifests():
    """`data/build_manifests.py`, loaded from its file: data/ holds standalone scripts rather than a package (and
    `data` may resolve to tests/data under unittest discovery)."""
    module = sys.modules.get("build_manifests")
    if module is None or not hasattr(module, "effective_acq_category"):
        spec = importlib.util.spec_from_file_location("build_manifests", os.path.join(REPO_ROOT, "data",
                                                                                      "build_manifests.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules.setdefault("build_manifests", module)
    return module

