"""Labelbox-ready local payloads for the RADAR caption review, and the guard every payload must pass.

`build_data_rows` makes one NDJSON data row per clip record: the pseudonymous record id as `global_key`, a video URL
from an explicit template (by default a placeholder on the reserved `.invalid` domain, to be replaced by an approved
de-identified location), and the candidate caption plus view/acquisition context as text attachments.
`build_prelabels` preloads the candidate caption into the `corrected_caption` field so reviewers edit it in place.

Nothing here uploads. `assert_payload_safe` refuses any payload containing an identifier-like field name, a local
path, a file name, a DICOM UID, a date, an e-mail address, a long digit run, or any record's local keys and paths.
Its errors name the JSON location and the rule, never the offending value.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Iterable

from app.radar.ontology import Ontology
from app.radar.records import RECORD_ID, ClipRecord, identifier_values

PAYLOAD_VERSION = "radar-labelbox-payload/0.1"
PLACEHOLDER_URL_TEMPLATE = "https://deidentified-video-location.invalid/radar/{global_key}.mp4"
_PRELABEL_NAMESPACE = uuid.UUID("0b9c6a5e-3a51-4f0e-9a39-6f2f7f0d6b11")

# Field names that hold identifiers or local locations in the archive, metadata and report export.
FORBIDDEN_KEYS = {
    "mrn", "patient", "patient_id", "patientid", "patient_key", "patient_name", "patientname", "dossier",
    "study", "study_id", "studyid", "study_key", "study_uid", "studyinstanceuid", "accession", "accessionnumber",
    "accession_number", "series_uid", "sop_instance_uid", "sopinstanceuid", "file_id", "dcm_path", "avi_path",
    "video_path", "path", "filepath", "file_path", "filename", "birth_date", "birthdate", "dob", "date",
    "study_date", "report", "report_text", "findings_text",
}
_RULES = (
    ("local path", re.compile(r"(^|[\s'\"(=])(~|\.{1,2})?/(?!/)|^[A-Za-z]:[\\/]|\\\\|file:")),
    ("file name", re.compile(r"\.(avi|dcm|dicom|mp4|mov|mkv|nii|npy|npz|csv|parquet|pt|pth|tar|jsonl?)\b", re.I)),
    ("DICOM UID", re.compile(r"\d+(\.\d+){4,}")),
    ("date", re.compile(r"\b(19|20)\d{2}[-/.]?(0[1-9]|1[0-2])[-/.]?(0[1-9]|[12]\d|3[01])\b")),
    ("e-mail address", re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
    ("long digit run", re.compile(r"\d{7,}")),
)


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class UnsafePayloadError(ValueError):
    """A payload holds something that must not leave the local data store."""


def validate_url_template(template: str) -> str:
    if not template.startswith("https://") or "{global_key}" not in template:
        raise ValueError("The video URL template must be an https:// URL containing {global_key}.")
    host_and_path = template[len("https://"):]
    if template.count("{") != 1 or any(rule.search(host_and_path.replace("{global_key}.mp4", "x"))
                                       for name, rule in _RULES if name != "file name"):
        raise ValueError("The video URL template may only hold a host, a path and {global_key}.")
    return template


def _strings(obj: Any, where: str = "$"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield f"{where}.{key}", key, None
            yield from _strings(value, f"{where}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _strings(value, f"{where}[{i}]")
    elif isinstance(obj, str):
        yield where, None, obj


def payload_problems(payload: Any, forbidden_values: Iterable[str] = (), url_template: str | None = None) -> list:
    """Every reason `payload` is unsafe, as "<json location>: <rule>" strings without the values."""
    forbidden = [v for v in forbidden_values if isinstance(v, str) and len(v.strip()) >= 4]
    url_prefix = url_template.split("{global_key}")[0] if url_template else None
    problems = []
    for where, key, text in _strings(payload):
        if key is not None:
            if key.lower().replace("-", "_") in FORBIDDEN_KEYS:
                problems.append(f"{where}: identifier field name")
            continue
        # A pseudonymous id is an HMAC hex digest: it can contain a digit run or, by chance, a short local
        # identifier as a substring without revealing anything, so its exact format is the check.
        if where.endswith(".row_data") and url_prefix and text.startswith(url_prefix):
            suffix = url_template.split("{global_key}")[1]
            key_part = text[len(url_prefix):len(text) - len(suffix)] if text.endswith(suffix) else ""
            if not RECORD_ID.fullmatch(key_part):
                problems.append(f"{where}: video URL does not match its template")
            continue
        if where.endswith((".global_key", ".globalKey")):
            if not RECORD_ID.fullmatch(text):
                problems.append(f"{where}: not a pseudonymous record id")
            continue
        if where.endswith(".uuid"):
            if not _UUID.fullmatch(text):
                problems.append(f"{where}: not a UUID")
            continue
        if any(v in text for v in forbidden):
            problems.append(f"{where}: local identifier or path")
        for name, rule in _RULES:
            if rule.search(text):
                problems.append(f"{where}: {name}")
    return problems


def assert_payload_safe(payload: Any, forbidden_values: Iterable[str] = (), url_template: str | None = None) -> None:
    problems = payload_problems(payload, forbidden_values, url_template)
    if problems:
        shown = problems[:20] + ([f"... and {len(problems) - 20} more"] if len(problems) > 20 else [])
        raise UnsafePayloadError("Refusing to build a Labelbox payload:\n- " + "\n- ".join(shown))


def _context(record: ClipRecord, ontology: Ontology, language: str) -> str:
    view = ontology.view_labels[language][record.view]
    acq = ontology.acquisition_labels[language][record.acquisition.category]
    return f"{view} ({record.view}); {acq}; concepts {ontology.tag}"


def _gaps(record: ClipRecord, ontology: Ontology, language: str) -> str:
    cap = record.candidate_caption
    if not cap.gaps:
        return "-"
    # Concept names and reasons only: the report's values for these findings stay local.
    return "; ".join(f"{ontology.label(g.concept_id, language)} ({g.reason})" for g in cap.gaps)


def _check_record(record: ClipRecord, allowed_splits: tuple) -> None:
    if not RECORD_ID.match(record.record_id or ""):
        raise UnsafePayloadError("A record id is not a pseudonymous rc-<hex> id.")
    if record.candidate_caption is None:
        raise ValueError(f"Record {record.record_id} has no candidate caption to review; generate one first.")
    if record.split not in allowed_splits:
        raise ValueError(f"Record {record.record_id} is in the {record.split} split; only {list(allowed_splits)} "
                         "may be sent for review (pass the split explicitly to allow it).")


def build_data_rows(records: list[ClipRecord], ontology: Ontology, *, language: str = "fr",
                    url_template: str = PLACEHOLDER_URL_TEMPLATE, allowed_splits: tuple = ("train",)) -> list[dict]:
    validate_url_template(url_template)
    rows = []
    for r in records:
        _check_record(r, allowed_splits)
        rows.append({
            "global_key": r.record_id,
            "row_data": url_template.format(global_key=r.record_id),
            "media_type": "VIDEO",
            "attachments": [
                {"type": "RAW_TEXT", "name": "candidate_caption", "value": r.candidate_caption.text},
                {"type": "RAW_TEXT", "name": "clip_context", "value": _context(r, ontology, language)},
                {"type": "RAW_TEXT", "name": "not_groundable_in_this_clip", "value": _gaps(r, ontology, language)},
            ],
        })
    assert_payload_safe(rows, [v for r in records for v in identifier_values(r)], url_template)
    return rows


def prelabel_uuid(record: ClipRecord) -> str:
    digest = hashlib.sha256(record.candidate_caption.text.encode()).hexdigest()[:16]
    return str(uuid.uuid5(_PRELABEL_NAMESPACE, f"{record.record_id}:{digest}"))


def build_prelabels(records: list[ClipRecord], *, allowed_splits: tuple = ("train",)) -> list[dict]:
    """Model-assisted-labeling NDJSON that preloads `corrected_caption` with the candidate text."""
    rows = []
    for r in records:
        _check_record(r, allowed_splits)
        rows.append({
            "uuid": prelabel_uuid(r),
            "name": "corrected_caption",
            "answer": r.candidate_caption.text,
            "dataRow": {"globalKey": r.record_id},
        })
    assert_payload_safe(rows, [v for r in records for v in identifier_values(r)])
    return rows
