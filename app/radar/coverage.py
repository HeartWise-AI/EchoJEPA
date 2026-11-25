"""Study-level coverage ledger: which observable report findings are grounded by which clips.

For each study and each report finding, the ledger lists the clips whose view x acquisition prior makes them
candidates, what review returned for them, which view groups a multi-view finding still lacks, and, when no clip
can ground the finding, an explicit gap with its reason (absent modality, measurement-only value, unknown
acquisition, no clip in a relevant view). Gaps stay gaps: nothing here inserts a spectral finding into a color-loop
caption.

The per-study ledger carries local study keys and record ids, so it is written only outside the repository. The
summary (`summarize`) is aggregate counts with no keys.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from app.radar.ontology import (
    ACQUISITION_UNKNOWN,
    DEFAULT_ONTOLOGY_PATH,
    MEASUREMENT_REQUIRED,
    MODALITY_ABSENT,
    NOT_IN_VIEW,
    RELEVANT,
    Ontology,
    load_ontology,
)
from app.radar.records import ClipRecord, ensure_outside_repo, read_records

# Finding statuses, from best to worst grounded.
SUPPORTED = "supported"  # at least one completed review supports it
CANDIDATE = "candidate"  # some candidate clips, not all reviewed, none supporting yet
NOT_SUPPORTED = "not_supported_after_review"  # every candidate clip reviewed, none supports it
GAP = "gap"  # no clip can ground it; see gap_reason
NO_RELEVANT_VIEW = "no_relevant_view"
STATUSES = (SUPPORTED, CANDIDATE, NOT_SUPPORTED, GAP)
# A study-level gap reports the most actionable reason found among its clips.
_GAP_PRIORITY = (MEASUREMENT_REQUIRED, MODALITY_ABSENT, ACQUISITION_UNKNOWN, NO_RELEVANT_VIEW)


@dataclass
class FindingCoverage:
    concept_id: str
    value: object
    source_field: str
    status: str
    gap_reason: str | None = None
    required_modality: list = field(default_factory=list)
    candidate_clips: list = field(default_factory=list)  # record ids whose prior allows the finding
    supported_by: list = field(default_factory=list)  # record ids a completed review supports
    not_supported_by: list = field(default_factory=list)  # reviewed: unsupported, wrong value or unassessable
    uncertain_in: list = field(default_factory=list)
    requires_views_from: list = field(default_factory=list)
    view_groups_covered: list = field(default_factory=list)
    view_groups_missing: list = field(default_factory=list)

    @property
    def needs_several_views(self) -> bool:
        return bool(self.requires_views_from)


@dataclass
class StudyCoverage:
    study_key: str
    split: str
    clips: list
    findings: list  # [FindingCoverage]
    inconsistent_findings: list = field(default_factory=list)  # concept ids with different values across clips


def _review_verdict(record: ClipRecord, cid: str) -> str | None:
    """'supported', 'not_supported', 'uncertain' or None (not reviewed / not addressed) for one clip."""
    rev = record.review
    if rev.status != "reviewed":
        return None
    if rev.clip_unassessable_reason is not None:
        return "not_supported"
    if (cid in rev.unsupported_concepts or any(w.concept_id == cid for w in rev.wrong_values)
            or any(u.concept_id == cid for u in rev.unassessable)):
        return "not_supported"
    if any(u.concept_id == cid for u in rev.uncertain):
        return "uncertain"
    cap = record.candidate_caption
    claimed = cap is not None and any(c.concept_id == cid for c in cap.claims)
    if claimed or cid in rev.missing_concepts:
        return "supported"
    return None


def study_coverage(study_key: str, records: list[ClipRecord], ontology: Ontology) -> StudyCoverage:
    splits = {r.split for r in records}
    if len(splits) != 1:
        raise ValueError("A study's clips must share one split.")
    values = defaultdict(set)
    first = {}
    for r in records:
        for f in r.report_findings:
            values[f.concept_id].add(repr(f.value))
            first.setdefault(f.concept_id, f)
    inconsistent = sorted(cid for cid, vals in values.items() if len(vals) > 1)
    findings = []
    for cid in ontology.concept_ids:
        if cid not in first or cid in inconsistent:
            continue
        concept = ontology.concept(cid)
        cov = FindingCoverage(
            concept_id=cid, value=first[cid].value, source_field=first[cid].provenance.source_field, status=CANDIDATE,
            required_modality=list(concept.required_modality), requires_views_from=list(concept.requires_views_from),
        )
        reasons = set()
        covered_views = set()
        for r in records:
            outcome = ontology.relevance(cid, r.view, r.acquisition.category)
            if outcome == RELEVANT:
                cov.candidate_clips.append(r.record_id)
            elif outcome != NOT_IN_VIEW:
                reasons.add(outcome)
            verdict = _review_verdict(r, cid)
            if verdict == "supported":
                cov.supported_by.append(r.record_id)
            elif verdict == "not_supported" and outcome == RELEVANT:
                cov.not_supported_by.append(r.record_id)
            elif verdict == "uncertain":
                cov.uncertain_in.append(r.record_id)
            if outcome == RELEVANT or verdict == "supported":
                covered_views.add(r.view)
        for group in concept.requires_views_from:
            target = cov.view_groups_covered if covered_views & set(ontology.view_groups[group]) else \
                cov.view_groups_missing
            target.append(group)
        if cov.supported_by:
            cov.status = SUPPORTED
        elif cov.candidate_clips:
            reviewed = [r for r in records if r.record_id in cov.candidate_clips and r.review.status == "reviewed"]
            cov.status = NOT_SUPPORTED if len(reviewed) == len(cov.candidate_clips) else CANDIDATE
        else:
            cov.status = GAP
            reasons = reasons or {NO_RELEVANT_VIEW}
            cov.gap_reason = next(reason for reason in _GAP_PRIORITY if reason in reasons)
        findings.append(cov)
    return StudyCoverage(study_key=study_key, split=splits.pop(), clips=[r.record_id for r in records],
                         findings=findings, inconsistent_findings=inconsistent)


def build_ledger(records: list[ClipRecord], ontology: Ontology) -> list[StudyCoverage]:
    by_study = defaultdict(list)
    for r in records:
        by_study[r.study_key].append(r)
    return [study_coverage(key, clips, ontology) for key, clips in sorted(by_study.items())]


def ledger_to_dicts(ledger: list[StudyCoverage]) -> list[dict]:
    return [asdict(study) for study in ledger]


def summarize(ledger: list[StudyCoverage]) -> dict:
    """Aggregate counts only: no study keys, record ids or values."""
    status = Counter()
    gaps = Counter()
    per_concept = defaultdict(Counter)
    multi = Counter()
    for study in ledger:
        for f in study.findings:
            status[f.status] += 1
            per_concept[f.concept_id][f.status] += 1
            if f.gap_reason:
                gaps[f.gap_reason] += 1
            if f.needs_several_views:
                multi["needs_several_views"] += 1
                multi["view_groups_complete" if not f.view_groups_missing else "view_groups_incomplete"] += 1
    total = sum(status.values())
    return {
        "studies": len(ledger),
        "clips": sum(len(s.clips) for s in ledger),
        "findings": total,
        "by_status": {s: status.get(s, 0) for s in STATUSES},
        "gap_reasons": dict(sorted(gaps.items())),
        "multi_view": dict(sorted(multi.items())),
        "supported_fraction": (status[SUPPORTED] / total) if total else None,
        "inconsistent_findings": sum(len(s.inconsistent_findings) for s in ledger),
        "by_concept": {cid: dict(sorted(c.items())) for cid, c in sorted(per_concept.items())},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RADAR study-level coverage ledger.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--out-dir", required=True, help="writes ledger.json (local keys) and summary.json")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    ontology = load_ontology(args.ontology)
    ledger = build_ledger(read_records(args.records, ontology), ontology)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ledger.json"), "w", encoding="utf-8") as f:
        json.dump(ledger_to_dicts(ledger), f, ensure_ascii=False, indent=2, default=str)
    summary = summarize(ledger)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("studies", "clips", "findings", "by_status", "gap_reasons")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
