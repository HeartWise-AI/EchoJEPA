"""Synthetic RADAR fixtures: records, findings and tiny videos. No real clinical data."""

import dataclasses
import os

import numpy as np

from app.radar.ontology import load_ontology
from app.radar.records import (
    Acquisition,
    CandidateCaption,
    Claim,
    ClipRecord,
    Provenance,
    ReportFinding,
    Review,
    make_record_id,
)

ONTOLOGY = load_ontology()
T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-02T00:00:00+00:00"


def finding(concept_id, value, field="synthetic field"):
    c = ONTOLOGY.concept(concept_id)
    return ReportFinding(concept_id=concept_id, value=value, severity=value if c.value_type == "ordinal" else None,
                         unit=c.unit, provenance=Provenance(field, "synthetic", "synthetic/0"))


def record(n, *, view="A4C", acq="bmode", split="train", study="study-a", patient="patient-a", findings=(),
           video_path=None, caption=None):
    return ClipRecord(
        record_id=make_record_id("unit-test-secret", f"video-{n}"), study_key=study, patient_key=patient,
        split=split, video_path=video_path or f"/synthetic/videos/video-{n:04d}.mp4", view=view, view_confidence=0.9,
        acquisition=Acquisition(category=acq, confidence="header" if acq != "unknown" else "unknown"),
        report_findings=list(findings), candidate_caption=caption,
    )


def candidate(text, claims=(), generator="rule_based"):
    return CandidateCaption(text=text, language="fr", generator=generator, generator_version="synthetic/0",
                            ontology_version=ONTOLOGY.version, created_utc=T0, claims=[Claim(*c) for c in claims],
                            prompt_version="p/0" if generator == "llm" else None)


def reviewed(**changes):
    base = dict(status="reviewed", reviewer_id="reviewer-a", review_version="radar-review/0.1+test",
                source="local_review", label_ref="local:1", started_utc=T0, completed_utc=T1, satisfied=True)
    base.update(changes)
    return Review(**base)


def with_review(rec, review):
    from app.radar.records import apply_review

    return apply_review(rec, review, ONTOLOGY)


def write_video(path, frames=8, fps=8, size=32, seed=0):
    """A tiny clip whose frames are never black, so content is told apart from padding."""
    import cv2

    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(frames):
        frame = rng.integers(40, 220, size=(size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def approved_clip_records(tmp, n=6, studies=3, split="train", captions=None, approve=True, start=0):
    """`n` records with real tiny videos, approved captions (unless approve=False) and `studies` studies."""
    out = []
    for i in range(start, start + n):
        path = write_video(os.path.join(tmp, f"clip{i}.mp4"), seed=i)
        text = captions[i - start] if captions else f"Coupe apicale 4 cavités (A4C), mode B. Clip {i}."
        rec = record(i, study=f"study-{i % studies}", patient=f"patient-{i % studies}", split=split,
                     video_path=path, caption=candidate(text))
        if approve:
            rec = with_review(rec, reviewed(label_ref=f"local:{i}"))
        out.append(rec)
    return out


def replace(rec, **changes):
    return dataclasses.replace(rec, **changes)
