"""Extract structured report findings for one study from a row of the ICM report export.

Structured columns ("Referring MR Grade", "RV Systolic Pressure", ...) are parsed first; free-text findings columns
(French in ~all reports) are searched only for concepts the columns did not resolve, with the ontology's phrase
hints, a short negation window and the nearest severity word in the same clause. Anything ambiguous (an unknown
grade code, two values for one concept, a mention without a severity) is reported as an `ExtractionIssue` and never
guessed. Findings keep their provenance (column, method, character span), never the report text itself.

This is a pilot tool: alias tables and hints in the ontology must be checked against the real value distribution
of each column before extraction is trusted, and every extracted finding is still only a candidate for review.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from app.radar.ontology import Ontology, is_code_alias, normalize_text
from app.radar.records import Provenance, ReportFinding

EXTRACTOR_VERSION = "radar-findings/0.1"
NEGATION_WINDOW = 30  # characters before a mention searched for a negation cue
_CLAUSE_BREAK = re.compile(r"[.;:\n,()]")


@dataclass(frozen=True)
class ExtractionIssue:
    """Why a concept was not extracted from a source. `raw_value` is kept only for short structured codes."""

    concept_id: str
    source_field: str
    reason: str  # unmapped_value | date_like_value | out_of_range | conflict | unresolved_mention | numeric_in_text
    raw_value: str | None = None


# Spellings of a missing value once a column has been read as text. They mean "unknown", never "none" or zero.
_MISSING_SPELLINGS = {"nan", "none", "null", "nat", "na", "n/a", "<na>"}
_DATE_LIKE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{2}:\d{2}")
# Raw values short and code-like enough to show in an issue (e.g. "0.5", "Trace", "2+"). Anything else, dates
# above all, is reported by reason only.
_SHOWABLE_CODE = re.compile(r"[^\W\d]{1,12}|\d{1,2}([.,]\d{1,2})?\+?")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and (not value.strip() or value.strip().lower() in _MISSING_SPELLINGS)


def _unmapped(cid: str, column: str, raw: Any) -> ExtractionIssue:
    """An unparseable structured value: unknown, with its raw form only when it is a short harmless code."""
    text = str(raw).strip()
    if _DATE_LIKE.search(text):
        return ExtractionIssue(cid, column, "date_like_value")
    return ExtractionIssue(cid, column, "unmapped_value", text if _SHOWABLE_CODE.fullmatch(text) else None)


def parse_number(value: Any) -> float | None:
    """A plain number, with a French decimal comma and an optional trailing %; anything else (ranges,
    inequalities, words) is None rather than a guess."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    text = str(value).strip().replace(",", ".")
    text = text[:-1].strip() if text.endswith("%") else text
    return float(text) if re.fullmatch(r"[-+]?\d+(\.\d+)?", text) else None


def _alias_table(ontology: Ontology, concept_id: str) -> tuple[dict, tuple]:
    c = ontology.concept(concept_id)
    if c.value_type == "ordinal":
        scale = ontology.scales[c.scale]
        return scale.aliases, scale.values
    return c.aliases, c.values or ()


def map_code(raw: Any, ontology: Ontology, concept_id: str) -> str | None:
    """Map a whole structured-column value onto a scale or categorical value by exact normalized alias.

    A float read from a numeric column is compared in its integer spelling when it is whole (2.0 -> "2"), so a
    code alias matches however pandas parsed the column; 0.5 stays "0.5" and maps only if an alias says so.
    """
    aliases, values = _alias_table(ontology, concept_id)
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    key = normalize_text(str(raw)).strip()
    if key in values:
        return key
    return aliases.get(key)


def _structured(row: Mapping[str, Any], ontology: Ontology, cid: str, issues: list) -> ReportFinding | None:
    c = ontology.concept(cid)
    candidates = []  # (value, column)
    worst = False
    for spec, column in ontology.matching_columns(cid, list(row)):
        raw = row[column]
        if _is_empty(raw):
            continue
        worst = worst or spec.get("aggregate") == "worst"
        if spec["parse"] == "numeric":
            number = parse_number(raw)
            if number is None:
                issues.append(_unmapped(cid, column, raw))
                continue
            if ontology.value_error(cid, number):
                issues.append(ExtractionIssue(cid, column, "out_of_range"))
                continue
            candidates.append((number, column))
        else:
            value = map_code(raw, ontology, cid)
            if value is None:
                issues.append(_unmapped(cid, column, raw))
                continue
            candidates.append((value, column))
    if not candidates:
        return None
    if worst:
        _, order = _alias_table(ontology, cid)
        value, column = max(candidates, key=lambda vc: order.index(vc[0]))
    else:
        distinct = {v for v, _ in candidates}
        if len(distinct) > 1:
            for _, column in candidates:
                issues.append(ExtractionIssue(cid, column, "conflict"))
            return None
        value, column = candidates[0]
    severity = value if c.value_type == "ordinal" else None
    return ReportFinding(
        concept_id=cid, value=value, severity=severity, unit=c.unit,
        provenance=Provenance(source_field=column, method="structured_column", extractor_version=EXTRACTOR_VERSION),
    )


def _word_regex(phrase: str) -> re.Pattern:
    """A phrase as a word-start prefix pattern over normalized text, tolerant to repeated spaces."""
    tokens = [re.escape(t) for t in phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(tokens))


def find_mentions(text: str, ontology: Ontology, concept_id: str) -> list[tuple[int, int, str]]:
    """(start, end, hint) of each phrase-hint (normalized prefix) or acronym (exact whole word) mention."""
    sources = ontology.concept(concept_id).report_sources
    norm = normalize_text(text)
    found = []
    for hint in sources.get("phrase_hints") or []:
        for m in _word_regex(normalize_text(hint)).finditer(norm):
            found.append((m.start(), m.end(), hint))
    for acronym in sources.get("acronyms") or []:
        for m in re.finditer(r"(?<!\w)" + re.escape(acronym) + r"(?!\w)", text):
            found.append((m.start(), m.end(), acronym))
    # Overlapping hints describe one mention: keep the longest.
    found.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    kept = []
    for m in found:
        if kept and m[0] < kept[-1][1]:
            continue
        kept.append(m)
    return kept


def _clause(norm: str, start: int, end: int) -> tuple[int, int]:
    left = max((m.end() for m in _CLAUSE_BREAK.finditer(norm, 0, start)), default=0)
    right = next((m.start() for m in _CLAUSE_BREAK.finditer(norm, end)), len(norm))
    return left, right


def _negated(norm: str, clause_start: int, start: int, ontology: Ontology) -> bool:
    window = norm[max(clause_start, start - NEGATION_WINDOW):start]
    return any(re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", window) for cue in ontology.negation_cues)


def _nearest_alias(norm: str, span: tuple[int, int], clause: tuple[int, int], aliases: Mapping[str, str]):
    """The alias value nearest to the mention within its clause, longest aliases first; None if none."""
    taken = [span]
    hits = []
    for alias in sorted((a for a in aliases if not is_code_alias(a)), key=len, reverse=True):
        for m in _word_regex(alias).finditer(norm, clause[0], clause[1]):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            distance = m.start() - span[1] if m.start() >= span[1] else span[0] - m.end()
            hits.append((distance, aliases[alias]))
    return min(hits)[1] if hits else None


def _mention_value(text: str, mention, ontology: Ontology, cid: str):
    """The value one mention states, or None when it cannot be resolved without guessing."""
    c = ontology.concept(cid)
    norm = normalize_text(text)
    start, end, _ = mention
    clause = _clause(norm, start, end)
    negated = _negated(norm, clause[0], start, ontology)
    if c.value_type == "boolean":
        return not negated
    if c.value_type == "numeric":
        return None
    aliases, values = _alias_table(ontology, cid)
    if negated:
        if c.value_type == "ordinal":
            return ontology.scales[c.scale].negated_value
        return "none" if "none" in values else None
    return _nearest_alias(norm, (start, end), clause, aliases)


def _from_text(row: Mapping[str, Any], ontology: Ontology, cid: str, issues: list) -> ReportFinding | None:
    c = ontology.concept(cid)
    for column in c.report_sources.get("text_fields") or []:
        text = row.get(column)
        if _is_empty(text):
            continue
        text = str(text)
        mentions = find_mentions(text, ontology, cid)
        if not mentions:
            continue
        if c.value_type == "numeric":
            issues.append(ExtractionIssue(cid, column, "numeric_in_text"))
            continue
        resolved = [(m, _mention_value(text, m, ontology, cid)) for m in mentions]
        values = {v for _, v in resolved if v is not None}
        if len(values) > 1:
            issues.append(ExtractionIssue(cid, column, "conflict"))
            continue
        if not values:
            issues.append(ExtractionIssue(cid, column, "unresolved_mention"))
            continue
        value = values.pop()
        mention = next(m for m, v in resolved if v == value)
        return ReportFinding(
            concept_id=cid, value=value, severity=value if c.value_type == "ordinal" else None, unit=c.unit,
            provenance=Provenance(source_field=column, method="phrase_hint", extractor_version=EXTRACTOR_VERSION,
                                  span=[mention[0], mention[1]], hint=mention[2]),
        )
    return None


def extract_findings(row: Mapping[str, Any], ontology: Ontology) -> tuple[list[ReportFinding], list[ExtractionIssue]]:
    """One study's findings, in ontology order: structured columns first, then free-text fields."""
    findings, issues = [], []
    for cid, concept in ontology.concepts.items():
        if concept.kind != "finding":
            continue
        finding = _structured(row, ontology, cid, issues)
        if finding is None:
            finding = _from_text(row, ontology, cid, issues)
        if finding is not None:
            findings.append(finding)
    return findings, issues
