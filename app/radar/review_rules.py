"""Validate RADAR caption reviews for inclusion in pilot metrics.

A review contains question answers, conditional details, and a corrected caption.

Validation consists of three checks:
    - answer_problems: Check required answers and allowed values.
    - consistency_problems: Check cross-field consistency with the candidate view.
    - unclaimed_flags: Check flags against the candidate caption's statements.
`review_problems` runs all three. A review counts as valid (V) only if no problems are found;
otherwise, it is returned for correction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

import yaml

from app.radar.ontology import ACQUISITION_STATEMENT, REPO_ROOT, VIEWS, Ontology

NOT_APPLICABLE = "not_applicable"
DEFAULT_EXAMPLES_PATH = os.path.join(REPO_ROOT, "configs", "radar", "review_examples_v1.yaml")
EXAMPLES_SCHEMA = "radar-review-examples"
# Expected schema name for that YAML file.
CLIP_QUESTIONS = ("view_correct", "caption_supported", "missing_information", "unsupported_information")


@dataclass(frozen=True)
class Review:
    """The structured representation of one human review, which stores all the information the reviewer submitted."""
    answers: Mapping[str, str]          # radio question id -> answer id.
    limitation: str | None = None       # uninterpretable_clip "yes".
    correct_view: str | None = None     # view_correct "no".
    missing: tuple[str, ...] = ()       # missing_information "yes": concept ids.
    missing_other: str = ""             # missing_information "yes": free text.
    unsupported: Mapping[str, str] = field(default_factory=dict)  # statement id -> unsupported kind.
    unconfirmed: Mapping[str, str] = field(default_factory=dict)  # statement id -> reason.
    corrected_caption: str = ""         # required; preloaded with the candidate caption.

    def __post_init__(self):
        """An export gives `null` for an empty answer: read it as empty, so the checks report it instead of failing."""
        for name, empty in (("answers", {}), ("missing", ()), ("missing_other", ""), ("unsupported", {}),
                            ("unconfirmed", {}), ("corrected_caption", "")):
            if getattr(self, name) is None:
                object.__setattr__(self, name, empty)  # the dataclass is frozen.


def answer_problems(review: Review, ontology: Ontology) -> list[str]:
    """Checks whether the review follows the form's schema.

    Checks:
        - Are all required answers provided?
        - Does the review contain only answers, concepts, and flags allowed by the ontology?
    """
    problems = []
    radios = {q.id: q for q in ontology.review.questions if q.type == "radio"}
    unknown = sorted(set(review.answers) - set(radios))
    if unknown:
        problems.append(f"Answers to questions the form does not ask: {unknown}.")
    for qid, q in radios.items():
        answer = review.answers.get(qid)
        if answer is None and q.required:
            problems.append(f"{qid} is required.")
        elif answer is not None and answer not in [a.id for a in q.answers]:
            problems.append(f"{qid} has no answer {answer!r}.")
    if review.limitation is not None and review.limitation not in ontology.limitations:
        problems.append(f"{review.limitation!r} is not an assessment limitation.")
    if review.correct_view is not None and review.correct_view not in VIEWS:
        problems.append(f"{review.correct_view!r} is not a view class.")
    bad = sorted(set(review.missing) - set(ontology.statement_concepts))
    if bad:
        problems.append(f"Missing concepts the form does not offer: {bad}.")
    statements = {ACQUISITION_STATEMENT, *ontology.statement_concepts}
    kinds, reasons = ontology.review.unsupported_kinds, {*ontology.review.unconfirmed_reasons, *ontology.limitations}
    for name, flags, allowed, what in (("unsupported", review.unsupported, kinds, "kind"),
                                       ("unconfirmed", review.unconfirmed, reasons, "reason")):
        bad = sorted(set(flags) - statements)
        if bad:
            problems.append(f"{name} statements the form does not offer: {bad}.")
        bad = sorted(s for s, value in flags.items() if value not in allowed)
        if bad:
            problems.append(f"{name} statements without an allowed {what}: {bad}.")
    if ontology.review.question("corrected_caption").required and not review.corrected_caption.strip():
        problems.append("`corrected_caption` is required.")
    return problems


def consistency_problems(review: Review, candidate_view: str) -> list[str]:
    """Verifies relationships between fields to enforce logical rules."""
    a = review.answers
    problems = []
    uninterpretable = a.get("uninterpretable_clip") == "yes"
    not_applicable = [q for q in CLIP_QUESTIONS if a.get(q) == NOT_APPLICABLE]
    if uninterpretable and len(not_applicable) != len(CLIP_QUESTIONS):
        problems.append("An uninterpretable clip answers not applicable to all four other questions.")
    if not uninterpretable and not_applicable:
        problems.append(f"An interpretable clip answers every question; not applicable: {not_applicable}.")
    if uninterpretable and not review.limitation:
        problems.append("An uninterpretable clip needs a reason.")
    if not uninterpretable and review.limitation:
        problems.append("A reason is given only for an uninterpretable clip.")
    if a.get("view_correct") == "no" and not review.correct_view:
        problems.append("A wrong view needs the correct view.")
    if a.get("view_correct") != "no" and review.correct_view:
        problems.append("A correct view is given only for a wrong view.")
    if a.get("view_correct") == "no" and review.correct_view == candidate_view:
        problems.append(f"A wrong view is corrected to another class than the stated view {candidate_view}.")
    supported, unsupported = a.get("caption_supported"), a.get("unsupported_information")
    if supported == "yes" and (unsupported != "no" or review.unconfirmed):
        problems.append("A supported caption has no unsupported and no unconfirmed statement.")
    if (supported == "no") != (unsupported == "yes"):
        problems.append('"caption supported: no" goes with "unsupported information: yes", and only with it.')
    if unsupported == "yes" and not review.unsupported:
        problems.append("Unsupported information needs at least one unsupported statement.")
    if unsupported != "yes" and review.unsupported:
        problems.append("Unsupported statements are listed only under unsupported information: yes.")
    if supported == "unsure" and (unsupported != "no" or not review.unconfirmed):
        problems.append('"caption supported: unsure" needs no unsupported and at least one unconfirmed statement.')
    missing = a.get("missing_information")
    if missing == "yes" and not (review.missing or review.missing_other.strip()):
        problems.append("Missing information needs a missing concept or a free-text item.")
    if missing != "yes" and (review.missing or review.missing_other.strip()):
        problems.append("Missing concepts are listed only under missing information: yes.")
    both = sorted(set(review.unsupported) & set(review.unconfirmed))
    if both:
        problems.append(f"Statements both unsupported and unconfirmed: {both}.")
    return problems


def unclaimed_flags(review: Review, statements) -> list[str]:
    """Finds flags for statements not in the candidate caption and concepts incorrectly marked as missing."""
    stated = set(statements)
    flagged = set(review.unsupported) | set(review.unconfirmed)
    flags = [f"{s} is flagged but not stated" for s in sorted(flagged - stated)]
    flags += [f"{c} is listed as missing but is stated" for c in sorted(set(review.missing) & stated)]
    return flags


def review_problems(review: Review, ontology: Ontology, candidate_view: str, statements) -> list[str]:
    """Returns all validation errors based on the review, candidate view, and stated concepts (including acquisition).
    An empty list means the review is valid."""
    return (answer_problems(review, ontology) + consistency_problems(review, candidate_view)
            + unclaimed_flags(review, statements))


def load_examples(ontology: Ontology, path: str = DEFAULT_EXAMPLES_PATH) -> list[dict]:
    """Load worked review examples (synthetic clips) and verify that they match the current ontology."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw.get("schema") != EXAMPLES_SCHEMA or raw.get("ontology") != ontology.tag:
        raise ValueError(f"{os.path.basename(path)} is not a {EXAMPLES_SCHEMA} file for {ontology.tag}.")
    return list(raw["examples"])


def example_review(example: Mapping) -> Review:
    r = example["review"]
    return Review(answers=dict(r["answers"]), limitation=r.get("limitation"), correct_view=r.get("correct_view"),
                  missing=tuple(r.get("missing") or ()), missing_other=r.get("missing_other") or "",
                  unsupported=dict(r.get("unsupported") or {}), unconfirmed=dict(r.get("unconfirmed") or {}),
                  corrected_caption=example["corrected_caption"] or example["caption"])  # `null`: left as preloaded.
