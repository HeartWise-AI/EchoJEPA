"""Load and validate the versioned RADAR caption ontology.

The ontology defines a shared vocabulary for report extraction, caption generation, clinical review,
coverage tracking, and contrastive learning. Versioning ensures compatibility across generated artifacts.

`relevance()` determines whether a concept may be visible in a clip based on its view and acquisition mode,
without confirming its presence.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ONTOLOGY_PATH = os.path.join(REPO_ROOT, "configs", "radar", "radar-caption-ontology-v1.yaml")

ONTOLOGY_SCHEMA = "radar-caption-ontology"
# These constants define the view classifier's classes and acquisition index's categories. Duplicated in the YAML so
# it can be edited independently. The loader verifies that both definitions match.
VIEWS = (
    "A2C", "A2C_LV", "A2C_ZOOM", "A3C", "A3C_LV", "A3C_ZOOM", "A4C", "A4C_LV", "A4C_ZOOM", "A5C", "A5C_ZOOM",
    "PLAX", "PLAX_DEEP", "PLAX_ZOOM", "PSAX", "PSAX_AV", "RVINF", "SUBCOSTAL", "SUPRASTERNAL", "OTHER",
)
UNCLASSIFIED_VIEW = "OTHER"     # belongs to no view groups and carries no priors.
ACQUISITION_CATEGORIES = ("bmode", "color_flow", "spectral_doppler", "mmode", "other", "unknown")
# Categories that can carry evidence. "other" and "unknown" never satisfy a modality requirement (fail closed).
EVIDENCE_MODALITIES = ("bmode", "color_flow", "spectral_doppler", "mmode")
KINDS = ("structure_visibility", "finding")
VALUE_TYPES = ("boolean", "ordinal", "categorical", "numeric")
EVIDENCE_TYPES = ("visual", "measurement")
LANGUAGES = ("fr", "en")
COLUMN_PARSERS = ("numeric", "scale")
QUESTION_TYPES = ("radio", "checklist", "text")
# Define the additional information requested after a review answer (see the `review` section of the YAML); the form builder implements each.
ANSWER_DETAILS = ("limitation", "correct_view", "missing", "unsupported")
CHECKLIST_DETAILS = ("unconfirmed",)
DETAIL_LABELS = ("limitation", "correct_view", "missing", "missing_other", "unsupported", "unsupported_kind",
                 "unconfirmed_reason")
ACQUISITION_STATEMENT = "acquisition"  # the statement id of a caption's acquisition mode.

# Outcomes of `Ontology.relevance`. Only `RELEVANT` findings may enter a caption. The three gap outcomes are reported
# as explicit gaps; `NOT_IN_VIEW` is simply out of scope for the clip.
RELEVANT = "relevant"
NOT_IN_VIEW = "not_in_view"
MODALITY_ABSENT = "modality_absent"
MEASUREMENT_REQUIRED = "measurement_required"
ACQUISITION_UNKNOWN = "acquisition_unknown"
GAP_REASONS = (MODALITY_ABSENT, MEASUREMENT_REQUIRED, ACQUISITION_UNKNOWN)

_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class OntologyError(ValueError):
    """The ontology file is malformed; the message lists every problem found."""


def normalize_text(text: str) -> str:
    """Converts text to lowercase, remove accents, and standardize apostrophes. Preserves the original text
    length so character positions in the normalized text still correspond to positions in the original text."""
    out = []
    for ch in str(text):
        if ch in "’‘`´":
            out.append("'")
            continue
        base = unicodedata.normalize("NFKD", ch)
        base = "".join(c for c in base if not unicodedata.combining(c)) or ch
        low = base.lower()
        out.append(low if len(low) == 1 else base[0])
    return "".join(out)


def is_code_alias(alias: str) -> bool:
    """Identify numeric grade aliases (e.g., "2+", "1-2") used in structured report columns rather than
    free-text reports."""
    return bool(re.fullmatch(r"[\d\s+\-/.]+", alias))


@dataclass(frozen=True)
class Scale:
    name: str
    values: tuple[str, ...]
    labels: Mapping[str, Mapping[str, str]]
    aliases: Mapping[str, str]      # normalized alias -> value.
    negated_value: str | None = None
    unranked: tuple[str, ...] = ()  # valid scale values that have no severity ranking, such as `indeterminate`.

    @property
    def ranked(self) -> tuple[str, ...]:
        """Returns the scale values in severity order, excluding unranked values."""
        return tuple(v for v in self.values if v not in self.unranked)


@dataclass(frozen=True)
class View:
    id: str
    window: str
    variant: str
    labels: Mapping[str, str]       # language -> label.
    description: Mapping[str, str]  # language -> what the standard view shows.


@dataclass(frozen=True)
class Structure:
    id: str
    category: str
    labels: Mapping[str, str]       # language -> label.


@dataclass(frozen=True)
class Concept:
    id: str
    labels: Mapping[str, str]
    structures: tuple[str, ...]
    kind: str
    value_type: str
    evidence: str
    required_modality: tuple[str, ...]
    visible_in: frozenset           # {(view, acquisition)}
    prior_views: frozenset          # views of `visible_in`, any acquisition.
    scale: str | None = None
    values: tuple[str, ...] | None = None  # categorical values
    value_labels: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # categorical labels.
    aliases: Mapping[str, str] = field(default_factory=dict)  # categorical normalized alias -> value.
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    requires_views_from: tuple[str, ...] = ()   # view-group names that must each be covered.
    report_sources: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class Answer:
    id: str
    labels: Mapping[str, str]           # language -> label.
    detail: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    required: bool
    text: Mapping[str, str]             # language -> question.
    answers: tuple[Answer, ...] = ()    # radio questions.
    detail: str | None = None           # checklist questions.


@dataclass(frozen=True)
class ReviewSpec:
    questions: tuple[Question, ...]
    detail_labels: Mapping[str, Mapping[str, str]]
    unsupported_kinds: Mapping[str, Mapping[str, str]]
    unconfirmed_reasons: Mapping[str, Mapping[str, str]]  # followed by the assessment limitations.
    acquisition_statement: Mapping[str, str]
    undetermined_view: Mapping[str, str]  # the view assigned to a corrected caption when the reviewer cannot identify it.

    def question(self, question_id: str) -> Question:
        return next(q for q in self.questions if q.id == question_id)


@dataclass(frozen=True)
class Ontology:
    name: str
    version: int
    path: str
    sha256: str
    views: Mapping[str, View]                       # in VIEWS order.
    view_groups: Mapping[str, tuple[str, ...]]
    view_labels: Mapping[str, Mapping[str, str]]    # language -> view -> label.
    structures: Mapping[str, Structure]             # in file order.
    acquisition_labels: Mapping[str, Mapping[str, str]]
    limitations: Mapping[str, Mapping[str, str]]    # reason -> language -> label, in file order.
    review: ReviewSpec
    scales: Mapping[str, Scale]
    negation_cues: tuple[str, ...]
    concepts: Mapping[str, Concept]                 # in file order.

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(self.concepts)

    @property
    def statement_concepts(self) -> tuple[str, ...]:
        """Returns concepts that can appear in captions, in their original file order.

        Includes structure visibility concepts and visually assessed findings. Measurement-based findings
        are never considered relevant to individual clips and therefore cannot be included in captions.
        """
        return tuple(cid for cid, c in self.concepts.items()
                     if c.kind == "structure_visibility" or c.evidence == "visual")

    @property
    def tag(self) -> str:
        """Returns the ontology identifier in `<name>-v<version>` format. This identifier is recorded
        in every artifact generated from the ontology."""
        return f"{self.name}-v{self.version}"

    def concept(self, concept_id: str) -> Concept:
        try:
            return self.concepts[concept_id]
        except KeyError:
            raise KeyError(f"Unknown concept {concept_id!r} for ontology {self.tag}.") from None

    def allowed_values(self, concept_id: str) -> tuple | None:
        """Returns the allowed values for a concept. For numeric concepts, return `None` because validity is
        determined by the minimum and maximum bounds."""
        c = self.concept(concept_id)
        if c.value_type == "boolean":
            return (True, False)
        if c.value_type == "ordinal":
            return self.scales[c.scale].values
        if c.value_type == "categorical":
            return c.values
        return None

    def value_error(self, concept_id: str, value: Any) -> str | None:
        """Returns an error message if the value is invalid for the concept. Returns `None` if the value is valid."""
        c = self.concept(concept_id)
        if c.value_type == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{concept_id} needs a number, not {type(value).__name__}"
            if value != value or (c.minimum is not None and value < c.minimum) or (
                c.maximum is not None and value > c.maximum
            ):
                return f"{concept_id} value {value} is outside [{c.minimum}, {c.maximum}] {c.unit or ''}".rstrip()
            return None
        if c.value_type == "boolean":
            return None if isinstance(value, bool) else f"{concept_id} needs true or false"
        allowed = self.allowed_values(concept_id)
        return None if value in allowed else f"{concept_id} value {value!r} is not one of {list(allowed)}"

    def severity_values(self, concept_id: str) -> tuple[str, ...] | None:
        """Returns the ordered severity levels of an ordinal concept. Exclude unranked values such as `indeterminate`,
        which are valid but do not represent a severity level. Return `None` for non-ordinal concepts."""
        c = self.concept(concept_id)
        return self.scales[c.scale].ranked if c.value_type == "ordinal" else None

    def label(self, concept_id: str, language: str) -> str:
        return self.concept(concept_id).labels[language]

    def value_label(self, concept_id: str, value: Any, language: str) -> str:
        c = self.concept(concept_id)
        if c.value_type == "ordinal":
            return self.scales[c.scale].labels[language][value]
        if c.value_type == "categorical":
            return c.value_labels[language][value]
        if c.value_type == "boolean":
            return {"fr": {True: "oui", False: "non"}, "en": {True: "yes", False: "no"}}[language][value]
        number = int(value) if float(value).is_integer() else round(float(value), 2)
        unit = c.unit or ""
        sep = "" if unit == "%" and language == "en" else " "
        return f"{number}{sep}{unit}".strip()

    def view_group_of(self, view: str) -> tuple[str, ...]:
        return tuple(name for name, views in self.view_groups.items() if view in views)

    def concepts_of_structure(self, structure: str) -> tuple[str, ...]:
        """Ids of the concepts about `structure`, in file order."""
        return tuple(cid for cid, c in self.concepts.items() if structure in c.structures)

    def visibility_concept(self, structure: str) -> str | None:
        """Returns the visibility concept associated with a structure. Returns `None` if the structure
        has no visibility concept."""
        found = [cid for cid in self.concepts_of_structure(structure)
                 if self.concepts[cid].kind == "structure_visibility"]
        return found[0] if found else None

    def relevance(self, concept_id: str, view: str, acquisition: str) -> str:
        """Determines whether a concept could be assessed from a clip based on its view and acquisition mode.

        The order of checks matters:
            - Views outside the concept's prior are considered out of scope.
            - For views within the prior, unknown acquisition modes, missing required modalities, and
            measurement-only findings are reported as explicit gaps.
        """
        if view not in VIEWS:
            raise ValueError(f"Unknown view {view!r}; expected one of {list(VIEWS)}.")
        if acquisition not in ACQUISITION_CATEGORIES:
            raise ValueError(f"Unknown acquisition category {acquisition!r}; expected {list(ACQUISITION_CATEGORIES)}.")
        c = self.concept(concept_id)
        if view not in c.prior_views:
            return NOT_IN_VIEW
        if acquisition not in EVIDENCE_MODALITIES:
            return ACQUISITION_UNKNOWN
        if acquisition not in c.required_modality:
            return MODALITY_ABSENT
        if (view, acquisition) not in c.visible_in:
            return NOT_IN_VIEW
        if c.evidence == "measurement":
            return MEASUREMENT_REQUIRED
        return RELEVANT

    def structured_columns(self, concept_id: str) -> list[dict]:
        return list(self.concept(concept_id).report_sources.get("columns", []))

    def matching_columns(self, concept_id: str, available: list[str]) -> list[tuple[dict, str]]:
        """Finds report-export columns associated with a concept.

        Returns matching columns in the order of their specifications, using either exact column names
        or glob patterns.
        """
        found = []
        for spec in self.structured_columns(concept_id):
            if "name" in spec:
                if spec["name"] in available:
                    found.append((spec, spec["name"]))
            else:
                found.extend((spec, col) for col in available if fnmatch.fnmatchcase(col, spec["pattern"]))
        return found


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _aliases(raw: Mapping, values: tuple, where: str, problems: list) -> dict:
    out = {}
    for value, aliases in (raw or {}).items():
        if value not in values:
            problems.append(f"{where}: alias target {value!r} is not a value")
            continue
        for alias in aliases or []:
            key = normalize_text(str(alias)).strip()
            if key in out and out[key] != value:
                problems.append(f"{where}: alias {alias!r} maps to both {out[key]!r} and {value!r}")
            out[key] = value
    return out


def _labels(raw: Mapping, values: tuple, where: str, problems: list) -> dict:
    out = {}
    for lang in LANGUAGES:
        table = (raw or {}).get(lang)
        if not isinstance(table, Mapping):
            problems.append(f"{where}: missing {lang} labels")
            continue
        missing = [v for v in values if v not in table]
        if missing:
            problems.append(f"{where}: {lang} labels missing for {missing}")
        out[lang] = {v: str(table[v]) for v in values if v in table}
    return out


def _expand_views(entries, groups: Mapping, where: str, problems: list) -> list:
    views = []
    for entry in entries or []:
        if entry in groups:
            views.extend(groups[entry])
        elif entry in VIEWS:
            views.append(entry)
        else:
            problems.append(f"{where}: {entry!r} is neither a view nor a view group")
    return views


def _bilingual(raw, where: str, problems: list) -> dict:
    if not isinstance(raw, Mapping) or any(not isinstance(raw.get(lang), str) or not raw.get(lang)
                                           for lang in LANGUAGES):
        problems.append(f"{where}: needs {list(LANGUAGES)} text")
        return {}
    return {lang: raw[lang] for lang in LANGUAGES}


def _labelled_ids(raw, where: str, problems: list) -> dict:
    out = {}
    if not isinstance(raw, Mapping) or not raw:
        problems.append(f"{where}: must list at least one id")
        return out
    for key, labels in raw.items():
        if not isinstance(key, str) or not _ID.match(key):
            problems.append(f"{where}: {key!r} must be a snake_case string id")
            continue
        out[key] = _bilingual(labels, f"{where}.{key}", problems)
    return out


def _review_spec(raw, limitations: Mapping, problems: list) -> ReviewSpec | None:
    if not isinstance(raw, Mapping):
        problems.append("review: the review form section is missing")
        return None
    questions, seen_details = [], []
    for i, qraw in enumerate(raw.get("questions") or []):
        qraw = qraw or {}
        qid = qraw.get("id")
        where = f"review question {qid or f'#{i}'}"
        if not isinstance(qid, str) or not _ID.match(qid) or qid in {q.id for q in questions}:
            problems.append(f"{where}: id must be a unique snake_case string")
            continue
        qtype = qraw.get("type")
        if qtype not in QUESTION_TYPES:
            problems.append(f"{where}: type {qtype!r} not in {list(QUESTION_TYPES)}")
        if not isinstance(qraw.get("required"), bool):
            problems.append(f"{where}: required must be true or false")
        answers = []
        for araw in qraw.get("answers") or []:
            araw = araw or {}
            aid = araw.get("id")
            if not isinstance(aid, str) or not _ID.match(aid) or aid in {a.id for a in answers}:
                problems.append(f"{where}: answer {aid!r} must be a unique snake_case string (quote yes and no)")
                continue
            detail = araw.get("detail")
            if detail is not None and detail not in ANSWER_DETAILS:
                problems.append(f"{where}: answer {aid}: detail {detail!r} not in {list(ANSWER_DETAILS)}")
            seen_details.append(detail)
            answers.append(Answer(id=aid, labels=_bilingual(araw.get("labels"), f"{where}.{aid}", problems),
                                  detail=detail))
        if qtype == "radio" and len(answers) < 2:
            problems.append(f"{where}: a radio question needs at least two answers")
        if qtype != "radio" and qraw.get("answers"):
            problems.append(f"{where}: only radio questions list answers")
        detail = qraw.get("detail")
        if (qtype == "checklist") != (detail in CHECKLIST_DETAILS):
            problems.append(f"{where}: a checklist question, and only one, has a detail in {list(CHECKLIST_DETAILS)}")
        seen_details.append(detail)
        questions.append(Question(id=qid, type=qtype, required=bool(qraw.get("required")),
                                  text=_bilingual(qraw.get("text"), f"{where}.text", problems),
                                  answers=tuple(answers), detail=detail))
    used = [d for d in seen_details if d is not None]
    for detail in ANSWER_DETAILS + CHECKLIST_DETAILS:
        if used.count(detail) != 1:
            problems.append(f"review: detail {detail!r} must be used exactly once, not {used.count(detail)} times")
    detail_labels = _labelled_ids(raw.get("detail_labels"), "review.detail_labels", problems)
    if set(detail_labels) != set(DETAIL_LABELS):
        problems.append(f"review.detail_labels must label exactly {list(DETAIL_LABELS)}")
    unsupported_kinds = _labelled_ids(raw.get("unsupported_kinds"), "review.unsupported_kinds", problems)
    unconfirmed = _labelled_ids(raw.get("unconfirmed_reasons"), "review.unconfirmed_reasons", problems)
    clash = set(unconfirmed) & set(limitations)
    if clash:
        problems.append(f"review.unconfirmed_reasons repeat assessment limitations {sorted(clash)}")
    return ReviewSpec(
        questions=tuple(questions), detail_labels=detail_labels, unsupported_kinds=unsupported_kinds,
        unconfirmed_reasons=unconfirmed,
        acquisition_statement=_bilingual(raw.get("acquisition_statement"), "review.acquisition_statement", problems),
        undetermined_view=_bilingual(raw.get("undetermined_view"), "review.undetermined_view", problems),
    )


def validate_ontology(raw: Mapping, path: str = "<memory>", sha256: str = "") -> Ontology:
    """Build an `Ontology` from parsed YAML, collecting every problem before raising `OntologyError`."""
    problems: list[str] = []
    if not isinstance(raw, Mapping):
        raise OntologyError(f"{path}: the ontology must be a mapping.")
    if raw.get("schema") != ONTOLOGY_SCHEMA:
        problems.append(f"schema must be {ONTOLOGY_SCHEMA!r}")
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        problems.append(f"version {version!r} must be a positive integer")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        problems.append("name is required")
    raw_views = raw.get("views") if isinstance(raw.get("views"), Mapping) else {}
    if tuple(raw_views) != VIEWS:
        problems.append(f"views must be exactly {list(VIEWS)}, in that order")
    if tuple(raw.get("acquisition_categories") or ()) != ACQUISITION_CATEGORIES:
        problems.append(f"acquisition_categories must be exactly {list(ACQUISITION_CATEGORIES)}")

    windows, variants = tuple(raw.get("view_windows") or ()), tuple(raw.get("view_variants") or ())
    view_defs = {}
    for vid, vraw in raw_views.items():
        vraw = vraw or {}
        where = f"view {vid}"
        if vraw.get("window") not in windows:
            problems.append(f"{where}: window {vraw.get('window')!r} not in view_windows {list(windows)}")
        if vraw.get("variant") not in variants:
            problems.append(f"{where}: variant {vraw.get('variant')!r} not in view_variants {list(variants)}")
        labels, description = vraw.get("labels") or {}, vraw.get("description") or {}
        for lang in LANGUAGES:
            if not isinstance(labels.get(lang), str) or not isinstance(description.get(lang), str):
                problems.append(f"{where}: needs a {lang} label and description")
        view_defs[vid] = View(id=vid, window=vraw.get("window"), variant=vraw.get("variant"), labels=dict(labels),
                              description=dict(description))
    view_labels = {lang: {vid: v.labels.get(lang) for vid, v in view_defs.items()} for lang in LANGUAGES}

    groups = {}
    for gname, gviews in (raw.get("view_groups") or {}).items():
        if gname in VIEWS:
            problems.append(f"view group {gname!r} shadows a view name")
        unknown = [v for v in gviews or [] if v not in VIEWS]
        if unknown or not gviews:
            problems.append(f"view group {gname!r} has unknown or no views {unknown}")
        groups[gname] = tuple(v for v in gviews or [] if v in VIEWS)
    for vid in VIEWS:
        n = sum(vid in g for g in groups.values())
        if vid == UNCLASSIFIED_VIEW and n:
            problems.append(f"{vid} is the unclassified view and belongs to no view group")
        elif vid != UNCLASSIFIED_VIEW and n != 1:
            problems.append(f"view {vid} must belong to exactly one view group, not {n}")

    categories = tuple(raw.get("structure_categories") or ())
    structures = {}
    for sid, sraw in (raw.get("structures") or {}).items():
        sraw = sraw or {}
        where = f"structure {sid}"
        if not isinstance(sid, str) or not _ID.match(sid):
            problems.append(f"{where}: id must be snake_case")
            continue
        if sraw.get("category") not in categories:
            problems.append(f"{where}: category {sraw.get('category')!r} not in structure_categories")
        labels = sraw.get("labels") or {}
        if any(not isinstance(labels.get(lang), str) or not labels.get(lang) for lang in LANGUAGES):
            problems.append(f"{where}: labels need {list(LANGUAGES)}")
        structures[sid] = Structure(id=sid, category=sraw.get("category"), labels=dict(labels))
    if not structures:
        problems.append("no structures")
    acquisition_labels = {}
    for lang in LANGUAGES:
        table = (raw.get("acquisition_labels") or {}).get(lang) or {}
        missing = [a for a in ACQUISITION_CATEGORIES if a not in table]
        if missing:
            problems.append(f"acquisition_labels.{lang} missing {missing}")
        acquisition_labels[lang] = dict(table)

    limitations = {}
    for reason, labels in (raw.get("assessment_limitations") or {}).items():
        if not isinstance(reason, str) or not _ID.match(reason):
            problems.append(f"assessment limitation {reason!r} must be snake_case")
        elif any(not isinstance((labels or {}).get(lang), str) for lang in LANGUAGES):
            problems.append(f"assessment limitation {reason!r} needs {list(LANGUAGES)} labels")
        else:
            limitations[reason] = {lang: labels[lang] for lang in LANGUAGES}
    if not limitations:
        problems.append("assessment_limitations must list at least one reason")

    scales = {}
    for sname, sraw in (raw.get("scales") or {}).items():
        values = tuple(str(v) for v in (sraw or {}).get("values") or ())
        where = f"scale {sname}"
        if not values or len(set(values)) != len(values):
            problems.append(f"{where}: values must be non-empty and unique")
        negated = sraw.get("negated_value")
        if negated is not None and negated not in values:
            problems.append(f"{where}: negated_value {negated!r} is not a value")
        unranked = tuple(str(v) for v in sraw.get("unranked") or ())
        if not set(unranked) <= set(values) - {negated} or len(unranked) >= len(values):
            problems.append(f"{where}: unranked values must be values other than negated_value, and not all of them")
        scales[sname] = Scale(
            name=sname, values=values, labels=_labels(sraw.get("labels"), values, where, problems),
            aliases=_aliases(sraw.get("aliases"), values, where, problems), negated_value=negated, unranked=unranked,
        )

    concepts = {}
    for i, craw in enumerate(raw.get("concepts") or []):
        cid = (craw or {}).get("id")
        where = f"concept {cid or f'#{i}'}"
        if not isinstance(cid, str) or not _ID.match(cid):
            problems.append(f"{where}: id must be snake_case")
            continue
        if cid in concepts:
            problems.append(f"{where}: duplicate id")
            continue
        labels = craw.get("labels") or {}
        if any(not isinstance(labels.get(lang), str) or not labels.get(lang) for lang in LANGUAGES):
            problems.append(f"{where}: labels need {list(LANGUAGES)}")
        kind = craw.get("kind")
        if kind not in KINDS:
            problems.append(f"{where}: kind {kind!r} not in {list(KINDS)}")
        value = craw.get("value") or {}
        vtype = value.get("type")
        if vtype not in VALUE_TYPES:
            problems.append(f"{where}: value.type {vtype!r} not in {list(VALUE_TYPES)}")
        if kind == "structure_visibility" and vtype != "boolean":
            problems.append(f"{where}: a structure_visibility concept is boolean")
        cstructures = craw.get("structures")
        if not isinstance(cstructures, list) or not cstructures or len(set(cstructures)) != len(cstructures):
            problems.append(f"{where}: structures must be a non-empty list without repeats")
            cstructures = []
        unknown_structures = [s for s in cstructures if s not in structures]
        if unknown_structures:
            problems.append(f"{where}: unknown structures {unknown_structures}")
        if kind == "structure_visibility" and len(cstructures) != 1:
            problems.append(f"{where}: a structure_visibility concept names exactly one structure")
        scale, values, value_labels, aliases = None, None, {}, {}
        if vtype == "ordinal":
            scale = value.get("scale")
            if scale not in scales:
                problems.append(f"{where}: unknown scale {scale!r}")
        elif vtype == "categorical":
            values = tuple(str(v) for v in value.get("values") or ())
            if not values or len(set(values)) != len(values):
                problems.append(f"{where}: categorical values must be non-empty and unique")
            value_labels = _labels(value.get("labels"), values, where, problems)
            aliases = _aliases(value.get("aliases"), values, where, problems)
        elif vtype == "numeric":
            if not value.get("unit"):
                problems.append(f"{where}: numeric concepts need a unit")
            lo, hi = value.get("min"), value.get("max")
            if lo is None or hi is None or not lo < hi:
                problems.append(f"{where}: numeric concepts need min < max")
        evidence = craw.get("evidence")
        if evidence not in EVIDENCE_TYPES:
            problems.append(f"{where}: evidence {evidence!r} not in {list(EVIDENCE_TYPES)}")
        if kind == "structure_visibility" and evidence != "visual":
            problems.append(f"{where}: visibility is judged visually")
        required = tuple(craw.get("required_modality") or ())
        bad = [m for m in required if m not in EVIDENCE_MODALITIES]
        if not required or bad:
            problems.append(f"{where}: required_modality must list some of {list(EVIDENCE_MODALITIES)}; bad {bad}")
        visible = set()
        for j, entry in enumerate(craw.get("potentially_visible_in") or []):
            views = _expand_views(entry.get("views"), groups, f"{where}.potentially_visible_in[{j}]", problems)
            if UNCLASSIFIED_VIEW in views:
                problems.append(f"{where}: {UNCLASSIFIED_VIEW} is the unclassified view and carries no prior")
            acqs = entry.get("acquisitions") or []
            for acq in acqs:
                if acq not in ACQUISITION_CATEGORIES:
                    problems.append(f"{where}: unknown acquisition {acq!r}")
                elif acq not in required:
                    # A concept's visibility prior must only include modalities that can support the finding.
                    # Otherwise, captions could contain findings that cannot be assessed from the clip.
                    problems.append(f"{where}: prior lists {acq!r}, which is not in required_modality")
            visible.update((v, a) for v in views for a in acqs if a in required)
        if not visible:
            problems.append(f"{where}: potentially_visible_in is empty")
        needs = tuple(craw.get("requires_views_from") or ())
        unknown_groups = [g for g in needs if g not in groups]
        if unknown_groups:
            problems.append(f"{where}: requires_views_from names unknown view groups {unknown_groups}")
        prior_views = frozenset(v for v, _ in visible)
        for g in needs:
            if g in groups and not prior_views & set(groups[g]):
                problems.append(f"{where}: requires views from {g!r} but its prior has none of them")
        sources = dict(craw.get("report_sources") or {})
        for spec in sources.get("columns", []) or []:
            if ("name" in spec) == ("pattern" in spec):
                problems.append(f"{where}: a report column needs exactly one of name or pattern")
            if spec.get("parse") not in COLUMN_PARSERS:
                problems.append(f"{where}: column parse {spec.get('parse')!r} not in {list(COLUMN_PARSERS)}")
            if spec.get("parse") == "numeric" and vtype != "numeric":
                problems.append(f"{where}: numeric column for a {vtype} concept")
            if spec.get("parse") == "scale" and vtype not in ("ordinal", "categorical"):
                problems.append(f"{where}: scale column for a {vtype} concept")
        if sources.get("text_fields") and not (sources.get("phrase_hints") or sources.get("acronyms")):
            problems.append(f"{where}: text_fields without phrase_hints or acronyms")
        if kind == "finding" and not sources:
            problems.append(f"{where}: a finding needs report_sources")
        concepts[cid] = Concept(
            id=cid, labels=dict(labels), structures=tuple(cstructures), kind=kind, value_type=vtype,
            evidence=evidence, required_modality=required, visible_in=frozenset(visible), prior_views=prior_views,
            scale=scale, values=values, value_labels=value_labels, aliases=aliases, unit=value.get("unit"),
            minimum=value.get("min"), maximum=value.get("max"), requires_views_from=needs, report_sources=sources,
            notes=str(craw.get("notes") or ""),
        )
    if not concepts:
        problems.append("no concepts")
    if ACQUISITION_STATEMENT in concepts:
        problems.append(f"concept id {ACQUISITION_STATEMENT!r} is reserved for the acquisition statement")
    review = _review_spec(raw.get("review"), limitations, problems)
    for sid in structures:
        visibility = [cid for cid, c in concepts.items() if c.kind == "structure_visibility" and sid in c.structures]
        if len(visibility) > 1:
            problems.append(f"structure {sid}: more than one visibility concept {visibility}")
    if problems:
        raise OntologyError(f"{os.path.basename(path)} is not a valid RADAR ontology:\n- " + "\n- ".join(problems))
    return Ontology(
        name=name, version=version, path=path, sha256=sha256, views=view_defs, view_groups=groups,
        view_labels=view_labels, structures=structures, acquisition_labels=acquisition_labels, limitations=limitations,
        review=review, scales=scales,
        negation_cues=tuple(normalize_text(c) for c in raw.get("negation_cues") or ()), concepts=concepts,
    )


def load_ontology(path: str = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return validate_ontology(raw, path=path, sha256=_file_sha256(path))
