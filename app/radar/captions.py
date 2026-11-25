"""Candidate captions for one clip: a deterministic rule-based generator and an LLM prompt/parse contract.

Both start from the same clip plan (`plan_clip`): the study's report findings filtered by the ontology's view x
acquisition prior. A finding enters a caption only when the clip's view is in the concept's prior, its acquisition
is one the concept can be supported by, and the value is judged visually. Findings that need an absent modality
(e.g. RVSP in a color loop), a measurement, or a known acquisition become explicit gaps; findings outside the view's
prior are simply out of view. Captions are built from structured values and ontology labels, so no report text is
ever copied into them.

Every caption is an *unvalidated candidate*: the prior establishes possible relevance, not visibility in this video.

The LLM path is offline: `build_llm_prompt` prepares the request for a future authorized endpoint and
`parse_llm_output` validates whatever comes back against a strict JSON contract. No network call is made here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from app.radar.ontology import DEFAULT_ONTOLOGY_PATH, GAP_REASONS, NOT_IN_VIEW, RELEVANT, Ontology, load_ontology
from app.radar.records import (
    MAX_CAPTION_CHARS,
    CandidateCaption,
    Claim,
    ClipRecord,
    ConceptGap,
    ensure_outside_repo,
    identifier_values,
    read_records,
    utc_now,
    write_records,
)

RULE_GENERATOR_VERSION = "radar-rule-captions/0.1"
PROMPT_VERSION = "radar-caption-prompt/0.1"
LLM_OUTPUT_SCHEMA_VERSION = "radar-llm-caption/0.1"
DEFAULT_LANGUAGE = "fr"  # ICM reports are French; reviewers read French.


class LLMOutputError(ValueError):
    """An LLM response does not satisfy the caption contract; it must not become a candidate."""


@dataclass(frozen=True)
class ClipPlan:
    claims: tuple  # (Claim, ...) in ontology order: findings the clip may state
    gaps: tuple  # (ConceptGap, ...): findings this clip cannot ground
    out_of_view: tuple  # (concept_id, ...)
    expected_structures: tuple  # (concept_id, ...): visibility concepts in the clip's prior


def plan_clip(record: ClipRecord, ontology: Ontology) -> ClipPlan:
    view, acq = record.view, record.acquisition.category
    claims, gaps, out_of_view = [], [], []
    by_concept = {f.concept_id: f for f in record.report_findings}
    for cid in ontology.concept_ids:
        finding = by_concept.get(cid)
        if finding is None:
            continue
        outcome = ontology.relevance(cid, view, acq)
        if outcome == RELEVANT:
            claims.append(Claim(cid, finding.value))
        elif outcome in GAP_REASONS:
            gaps.append(ConceptGap(cid, outcome, list(ontology.concept(cid).required_modality)))
        else:
            out_of_view.append(cid)
    expected = tuple(
        cid for cid, c in ontology.concepts.items()
        if c.kind == "structure_visibility" and ontology.relevance(cid, view, acq) == RELEVANT
    )
    return ClipPlan(tuple(claims), tuple(gaps), tuple(out_of_view), expected)


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def render_claim(claim: Claim, ontology: Ontology, language: str) -> str:
    label = ontology.label(claim.concept_id, language)
    if ontology.concept(claim.concept_id).value_type == "boolean":
        if claim.value:
            return label
        if language == "fr":
            article = "d'" if label[:1].lower() in "aeéèêëiîïoôuûyh" else "de "
            return f"Absence {article}{_lower_first(label)}"
        return f"No {_lower_first(label)}"
    sep = " : " if language == "fr" else ": "
    return f"{label}{sep}{ontology.value_label(claim.concept_id, claim.value, language)}"


def render_caption(view: str, acquisition: str, claims, ontology: Ontology, language: str) -> str:
    head = f"{ontology.view_labels[language][view]} ({view}), {ontology.acquisition_labels[language][acquisition]}."
    return " ".join([head] + [render_claim(c, ontology, language) + "." for c in claims])


def rule_based_caption(record: ClipRecord, ontology: Ontology, language: str = DEFAULT_LANGUAGE,
                       created_utc: str | None = None) -> CandidateCaption:
    plan = plan_clip(record, ontology)
    text = render_caption(record.view, record.acquisition.category, plan.claims, ontology, language)
    return CandidateCaption(
        text=text, language=language, generator="rule_based", generator_version=RULE_GENERATOR_VERSION,
        prompt_version=None, ontology_version=ontology.version, created_utc=created_utc or utc_now(),
        claims=list(plan.claims), gaps=list(plan.gaps), out_of_view=list(plan.out_of_view),
        expected_structures=list(plan.expected_structures),
    )


# --------------------------------------------------------------------------- LLM contract

LLM_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": LLM_OUTPUT_SCHEMA_VERSION,
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "language", "caption", "claims", "omitted"],
    "properties": {
        "schema_version": {"const": LLM_OUTPUT_SCHEMA_VERSION},
        "language": {"enum": ["fr", "en"]},
        "caption": {"type": "string", "minLength": 1, "maxLength": MAX_CAPTION_CHARS},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["concept_id", "value"],
                "properties": {"concept_id": {"type": "string"}, "value": {}},
            },
        },
        "omitted": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["concept_id", "reason"],
                "properties": {
                    "concept_id": {"type": "string"},
                    "reason": {"enum": list(GAP_REASONS) + [NOT_IN_VIEW]},
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You write short, view-specific captions for single echocardiography video clips.
Rules:
1. Write in {language_name}. One or two sentences, at most {max_chars} characters.
2. Start with the view and acquisition mode given in `clip`.
3. State only findings listed in `allowed_findings`, with exactly the given value. Do not add findings, numbers,
   comparisons, diagnoses or recommendations that are not listed.
4. Never mention findings listed in `gaps`: this clip cannot show them (wrong modality, a measurement is needed, or
   the acquisition is unknown). Report them in `omitted` with the given reason.
5. Never include names, dates, identifiers, file names or any text that is not about this clip.
6. The caption is a candidate that a cardiologist will check against the video; do not claim certainty.
Return only one JSON object that matches this schema, with no surrounding text:
{schema}"""


def build_llm_prompt(record: ClipRecord, ontology: Ontology, language: str = DEFAULT_LANGUAGE,
                     report_excerpts: Mapping[str, str] | None = None, allow_test_split: bool = False) -> dict:
    """The request for an authorized endpoint: system and user messages plus the response schema.

    Only structured findings that the clip plan allows are sent, never identifiers or paths. `report_excerpts` (short
    free-text passages, keyed by report field) are optional and must only be passed to an endpoint approved for
    clinical text; they are not needed for the structured path. Test-split records are refused unless
    `allow_test_split` is set for the frozen final evaluation: their reports must not shape prompt development.
    """
    if record.split == "test" and not allow_test_split:
        raise ValueError(f"Record {record.record_id} is in the test split; prompts are developed on train/val only.")
    plan = plan_clip(record, ontology)
    payload = {
        "clip": {
            "view": record.view,
            "view_label": ontology.view_labels[language][record.view],
            "acquisition": record.acquisition.category,
            "acquisition_label": ontology.acquisition_labels[language][record.acquisition.category],
        },
        "language": language,
        "allowed_findings": [
            {
                "concept_id": c.concept_id,
                "label": ontology.label(c.concept_id, language),
                "value": c.value,
                "value_label": ontology.value_label(c.concept_id, c.value, language),
            }
            for c in plan.claims
        ],
        "gaps": [{"concept_id": g.concept_id, "reason": g.reason} for g in plan.gaps],
    }
    if report_excerpts:
        payload["report_excerpts"] = dict(report_excerpts)
    system = SYSTEM_PROMPT.format(
        language_name={"fr": "French", "en": "English"}[language], max_chars=MAX_CAPTION_CHARS,
        schema=json.dumps(LLM_OUTPUT_SCHEMA, ensure_ascii=False),
    )
    return {
        "prompt_version": PROMPT_VERSION,
        "ontology_version": ontology.version,
        "system": system,
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "response_schema": LLM_OUTPUT_SCHEMA,
    }


def _exact_keys(obj: Any, keys: set, where: str) -> None:
    if not isinstance(obj, dict):
        raise LLMOutputError(f"{where} must be a JSON object")
    if set(obj) != keys:
        raise LLMOutputError(f"{where} must have exactly the keys {sorted(keys)}, not {sorted(obj)}")


def parse_llm_output(raw: str, record: ClipRecord, ontology: Ontology, *, model_id: str,
                     language: str = DEFAULT_LANGUAGE, created_utc: str | None = None) -> CandidateCaption:
    """Validate an LLM response strictly and turn it into an unvalidated candidate caption.

    Rejected: anything that is not exactly one JSON object of the schema, unknown concepts, values not in the
    ontology or different from the report, claims the clip plan does not allow (gaps, out of view), repeated
    claims, captions with identifiers or long digit runs. Gaps come from the deterministic plan, not the model.
    """
    if not isinstance(raw, str):
        raise LLMOutputError("the response must be text")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMOutputError(f"the response is not valid JSON ({e.msg})") from None
    _exact_keys(obj, {"schema_version", "language", "caption", "claims", "omitted"}, "the response")
    if obj["schema_version"] != LLM_OUTPUT_SCHEMA_VERSION:
        raise LLMOutputError(f"schema_version must be {LLM_OUTPUT_SCHEMA_VERSION!r}")
    if obj["language"] != language:
        raise LLMOutputError(f"language must be {language!r}")
    caption = obj["caption"]
    if not isinstance(caption, str) or not caption.strip() or len(caption) > MAX_CAPTION_CHARS:
        raise LLMOutputError(f"caption must be non-empty text of at most {MAX_CAPTION_CHARS} characters")
    if any(v in caption for v in identifier_values(record)) or re.search(r"\d{6,}", caption):
        raise LLMOutputError("caption contains an identifier, a path or a long number")
    plan = plan_clip(record, ontology)
    allowed = {c.concept_id: c.value for c in plan.claims}
    if not isinstance(obj["claims"], list) or not isinstance(obj["omitted"], list):
        raise LLMOutputError("claims and omitted must be lists")
    claims, seen = [], set()
    for i, item in enumerate(obj["claims"]):
        _exact_keys(item, {"concept_id", "value"}, f"claims[{i}]")
        cid, value = item["concept_id"], item["value"]
        if cid not in ontology.concepts:
            raise LLMOutputError(f"claims[{i}]: unknown concept {cid!r}")
        if cid in seen:
            raise LLMOutputError(f"claims[{i}]: {cid} is claimed twice")
        seen.add(cid)
        error = ontology.value_error(cid, value)
        if error:
            raise LLMOutputError(f"claims[{i}]: {error}")
        if cid not in allowed:
            reason = next((g.reason for g in plan.gaps if g.concept_id == cid), None)
            raise LLMOutputError(f"claims[{i}]: {cid} is not an allowed finding for this clip"
                                 + (f" ({reason})" if reason else ""))
        if allowed[cid] != value:
            raise LLMOutputError(f"claims[{i}]: {cid} value differs from the report finding")
        claims.append(Claim(cid, value))
    for i, item in enumerate(obj["omitted"]):
        _exact_keys(item, {"concept_id", "reason"}, f"omitted[{i}]")
        if item["concept_id"] not in ontology.concepts:
            raise LLMOutputError(f"omitted[{i}]: unknown concept {item['concept_id']!r}")
        if item["reason"] not in GAP_REASONS + (NOT_IN_VIEW,):
            raise LLMOutputError(f"omitted[{i}]: unknown reason {item['reason']!r}")
    order = {cid: i for i, cid in enumerate(ontology.concept_ids)}
    claims.sort(key=lambda c: order[c.concept_id])
    return CandidateCaption(
        text=caption.strip(), language=language, generator="llm", generator_version=model_id,
        prompt_version=PROMPT_VERSION, ontology_version=ontology.version, created_utc=created_utc or utc_now(),
        claims=claims, gaps=list(plan.gaps), out_of_view=list(plan.out_of_view),
        expected_structures=list(plan.expected_structures),
    )


def main(argv=None) -> int:
    """Add rule-based candidate captions to a records file (unreviewed records only)."""
    parser = argparse.ArgumentParser(description="Add rule-based RADAR candidate captions to clip records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True, help="updated records JSONL, outside the repository")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=("fr", "en"))
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args(argv)
    ensure_outside_repo(args.out, "--out")
    ontology = load_ontology(args.ontology)
    records, kept = [], 0
    for r in read_records(args.records, ontology):
        if r.review.status != "unreviewed":
            kept += 1  # a caption under review is never replaced underneath its reviewer
            records.append(r)
        else:
            records.append(dataclasses.replace(r, candidate_caption=rule_based_caption(r, ontology, args.language)))
    write_records(args.out, records, ontology)
    print(f"Captioned {len(records) - kept} records; kept {kept} captions already under review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
