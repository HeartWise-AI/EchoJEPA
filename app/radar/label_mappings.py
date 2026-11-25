"""Load and validate label mappings from DeepECHO, EchoPrime, and PanEcho (`configs/radar/label_mappings_v1.yaml`)
to the shared caption ontology.

Validates mapping definitions against the ontology and checks that all source labels and task classes are covered
without missing or extra entries.

Usage:
    python -m app.radar.label_mappings  # validate, check completeness, print a summary.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml

from app.radar.ontology import ACQUISITION_CATEGORIES, REPO_ROOT, VIEWS, Concept, Ontology, load_ontology

DEFAULT_MAPPINGS_PATH = os.path.join(REPO_ROOT, "configs", "radar", "label_mappings_v1.yaml")
MAPPINGS_SCHEMA = "radar-label-mappings"
RELATIONS = (
    "exact",        # directly corresponds to a concept, view, or value without changing its meaning.
    "grouped",      # combines several more-specific values or views (source uses a coarser representation).
    "transformed",  # a measurement requires a transformation to match the concept.
    "related",      # connected to an ontology concept but not equivalent.
    "unmapped"      # no suitable concept-level mapping is defined.
)
SOURCE_KINDS = ("view", "task")
UNSPECIFIED = "unspecified"  # a view label whose acquisition mode the source does not define.
UNITS = ("%", "mmHg", "cm", "mm", "mL", "cm/s", "m/s", "ratio")

# Source vocabulary paths (PanEcho, EchoPrime).
MODELCUSTOM = os.path.join(REPO_ROOT, "evals", "video_classification_frozen", "modelcustom")
ECHOPRIME_VIEWS_PATH = os.path.join(MODELCUSTOM, "EchoPrime", "utils", "utils.py")
ECHOPRIME_FINDINGS_PATH = os.path.join(MODELCUSTOM, "EchoPrime", "assets", "per_section.json")
PANECHO_TASKS_PATH = os.path.join(MODELCUSTOM, "PanEcho", "content", "tasks.pkl")
ECHOPRIME_BINARY_CLASSES = ("positive", "negative")


class MappingError(ValueError):
    """The mapping file is malformed or inconsistent with the ontology; the message lists every problem."""


@dataclass(frozen=True)
class Entry:
    """An `Entry` represents one source label and its mapping information."""
    label: str                                  # original label name.
    relation: str                               # how it relates to the ontology.
    concepts: tuple[str, ...] = ()              # mapped or related ontology concept ids.
    views: tuple[str, ...] = ()                 # view names or group names from YAML.
    expanded_views: tuple[str, ...] = ()        # view entries: canonical individual views after expanding groups, in taxonomy order.
    acquisitions: tuple[str, ...] | str = ()    # view entries: acquisition categories, or `UNSPECIFIED`.
    # Source class names mapped to values or numeric intervals; for related and unmapped labels every class maps to `None`.
    classes: Mapping[str, Any] = field(default_factory=dict)
    unassigned: tuple = ()                      # ontology values not covered by source classes.
    unit: str | None = None                     # source task measurement unit.
    transform: str | None = None                # description of a required transformation.
    also_related: tuple[str, ...] = ()          # additional related ontology concepts.
    structures: tuple[str, ...] = ()            # anatomical structures associated with the mapped concepts.
    reason: str = ""                            # explanation for leaving a task unmapped.
    notes: str = ""                             # additional human-readable notes.


@dataclass(frozen=True)
class Source:
    """A collection of `Entry`'s that represents an entire group of labels from one model."""
    name: str
    kind: str
    description: str
    vocabulary: str
    entries: Mapping[str, Entry]  # label -> entry, in file order.


@dataclass(frozen=True)
class LabelMappings:
    """Represents the complete validated mapping config."""
    version: int
    ontology_tag: str
    path: str
    sources: Mapping[str, Source]


def _expand(ontology: Ontology, items, where: str, problems: list) -> tuple[str, ...]:
    """Converts a collection of view labels and view-group labels into a list of individual canonical views."""
    # Use `set()` because the source might mention overlapping views.
    views = set()
    for item in items or []:
        if item in ontology.view_groups:
            views.update(ontology.view_groups[item])
        elif item in VIEWS:
            views.add(item)
        else:
            problems.append(f"{where}: {item!r} is neither a view nor a view group.")
    # Ensure the result follows the canonical order defined by `VIEWS`.
    return tuple(v for v in VIEWS if v in views)


def _bounds(concept: Concept, interval: Mapping) -> tuple:
    """The half-open range [lo, hi) of a numeric class; a missing bound is the concept's min or max."""
    return interval.get("at_least", concept.minimum), interval.get("below", concept.maximum)


def _interval(ontology: Ontology, concept_id: str, raw, where: str, problems: list) -> dict | None:
    """Validates source classes represented by intervals of numeric values. None if the interval is invalid."""
    if not isinstance(raw, Mapping) or not raw or set(raw) - {"at_least", "below"}:
        problems.append(f"{where}: a numeric class maps to {{at_least: x}} and/or {{below: y}}.")
        return None
    c = ontology.concept(concept_id)
    # Only keys `at_least` (lower bound, inclusive) and `below` (upper bound, exclusive) are allowed.
    # If a bound is missing, it uses the concept's min or max.
    lo, hi = _bounds(c, raw)
    if any(isinstance(b, bool) or not isinstance(b, (int, float)) for b in (lo, hi)):
        problems.append(f"{where}: interval bounds {dict(raw)} must be numbers.")
        return None
    # Validate the boundaries.
    if not (c.minimum <= lo < hi <= c.maximum):
        problems.append(f"{where}: interval {dict(raw)} is empty or outside [{c.minimum}, {c.maximum}].")
        return None
    return dict(raw)


def _overlapping_classes(concept: Concept, classes: Mapping) -> list[list[str]]:
    """Pairs of classes that share a value or part of an interval. Classes that failed validation are skipped."""
    valid = [(name, target) for name, target in classes.items() if target]
    pairs = []
    for i, (a, target_a) in enumerate(valid):
        for b, target_b in valid[i + 1:]:
            if concept.value_type == "numeric":
                (lo_a, hi_a), (lo_b, hi_b) = _bounds(concept, target_a), _bounds(concept, target_b)
                shared = lo_a < hi_b and lo_b < hi_a    # half-open.
            else:
                shared = {repr(v) for v in target_a} & {repr(v) for v in target_b}
            if shared:
                pairs.append([a, b])
    return pairs


def _values(ontology: Ontology, concept_id: str, raw, where: str, problems: list) -> tuple:
    """Validates categorical class mappings."""
    if not isinstance(raw, list) or not raw or len(set(map(repr, raw))) != len(raw):
        problems.append(f"{where}: a class maps to a non-empty list of values without repeats.")
        return ()
    allowed = ontology.allowed_values(concept_id)
    bad = [v for v in raw if v not in allowed or isinstance(v, bool) != isinstance(allowed[0], bool)]
    if bad:
        problems.append(f"{where}: {bad} are not values of {concept_id} {list(allowed)}.")
    return tuple(raw)


def _entry(ontology: Ontology, kind: str, label: str, raw: Mapping, where: str, problems: list) -> Entry:
    """Validates one mapping entry."""
    raw = dict(raw or {})
    relation = raw.get("relation")
    if relation not in RELATIONS:
        problems.append(f"{where}: relation {relation!r} not in {list(RELATIONS)}.")
    allowed_keys = {"relation", "notes"}
    out: dict[str, Any] = {"label": label, "relation": relation, "notes": str(raw.get("notes") or "").strip()}

    # Validate view mappings.
    if kind == "view":
        allowed_keys |= {"views", "acquisitions"}
        expanded = _expand(ontology, raw.get("views"), where, problems)
        if not expanded:
            problems.append(f"{where}: a view label needs views.")
        if relation not in ("exact", "grouped"):
            problems.append(f"{where}: a view label is exact or grouped.")
        if relation == "exact" and len(expanded) != 1:
            problems.append(f"{where}: an exact view label maps to one view, not {list(expanded)}.")
        if relation == "grouped" and len(expanded) < 2:
            problems.append(f"{where}: a grouped view label covers several views.")
        acqs = raw.get("acquisitions", UNSPECIFIED)
        if acqs != UNSPECIFIED and (not isinstance(acqs, list) or not acqs
                                    or any(a not in ACQUISITION_CATEGORIES for a in acqs)):
            problems.append(f"{where}: acquisitions must be {UNSPECIFIED!r} or categories of "
                            f"{list(ACQUISITION_CATEGORIES)}.")
        out.update(views=tuple(raw.get("views") or ()), expanded_views=expanded,
                   acquisitions=acqs if acqs == UNSPECIFIED else tuple(acqs or ()))
        unknown = set(raw) - allowed_keys
        if unknown:
            problems.append(f"{where}: unknown fields {sorted(unknown)}.")
        return Entry(**out)

    # Validate task units.
    unit = raw.get("unit")
    if unit is not None and unit not in UNITS:
        problems.append(f"{where}: unit {unit!r} not in {list(UNITS)}.")
    out["unit"] = unit
    allowed_keys |= {"unit"}
    if relation in ("exact", "grouped", "transformed"):
        allowed_keys |= {"concept", "classes", "unassigned", "also_related", "transform"}
        cid = raw.get("concept")
        if cid not in ontology.concepts:
            problems.append(f"{where}: unknown concept {cid!r}.")
            return Entry(**out)
        concept = ontology.concept(cid)
        also = tuple(raw.get("also_related") or ())
        for other in also:
            if other not in ontology.concepts or other == cid:
                problems.append(f"{where}: also_related {other!r} is unknown or the mapped concept.")
        out.update(concepts=(cid,), also_related=also,
                   structures=tuple(dict.fromkeys(s for c in (cid, *also) if c in ontology.concepts
                                                  for s in ontology.concept(c).structures)))
        classes = raw.get("classes")
        if classes is not None:
            if not isinstance(classes, Mapping) or not classes:
                problems.append(f"{where}: classes must be a non-empty mapping.")
                classes = {}
            parsed = {}
            for name, target in classes.items():
                cwhere = f"{where} class {name!r}"
                if concept.value_type == "numeric":
                    parsed[str(name)] = _interval(ontology, cid, target, cwhere, problems)
                else:
                    parsed[str(name)] = _values(ontology, cid, target, cwhere, problems)
            out["classes"] = parsed
            # A source predicts one class at a time, so no two of its classes may assert the same value.
            overlapping = _overlapping_classes(concept, parsed)
            if overlapping:
                problems.append(f"{where}: classes {overlapping} overlap; a value belongs to at most one class.")
            if concept.value_type != "numeric":
                covered = {repr(v) for values in parsed.values() for v in values}
                uncovered = [v for v in ontology.allowed_values(cid) if repr(v) not in covered]
                unassigned = list(raw.get("unassigned") or [])
                if [repr(v) for v in unassigned] != [repr(v) for v in uncovered]:
                    problems.append(f"{where}: values no class covers are {uncovered}; `unassigned` must list exactly "
                                    f"those, in scale order.")
                out["unassigned"] = tuple(unassigned)
                if relation == "exact":
                    # `singles`: every source class maps to exactly one ontology value.
                    # `distinct`: different source classes cannot share an ontology value (checked as overlap above).
                    # `uncovered`: every possible ontology value must be represented.
                    singles = all(len(v) == 1 for v in parsed.values())
                    if not singles or uncovered:
                        problems.append(f"{where}: an exact mapping gives each class one value and covers them all.")
            elif raw.get("unassigned"):
                problems.append(f"{where}: unassigned applies to non-numeric concepts only.")
        if relation == "exact" and classes is None and unit != concept.unit:
            problems.append(f"{where}: an exact regression has the concept's unit {concept.unit!r}, not {unit!r}.")
        if relation == "transformed":
            if classes is not None or not raw.get("transform") or not unit:
                problems.append(f"{where}: a transformed mapping is a regression with a unit and a transform.")
            out["transform"] = str(raw.get("transform") or "")
        elif raw.get("transform"):
            problems.append(f"{where}: only a transformed mapping has a transform.")
        if relation == "grouped" and classes is None:
            problems.append(f"{where}: a grouped mapping needs classes.")
    elif relation in ("related", "unmapped"):
        allowed_keys |= {"classes"}
        names = raw.get("classes")
        if names is not None:
            if not isinstance(names, list) or not names or len(set(map(str, names))) != len(names):
                problems.append(f"{where}: a {relation} label lists its source classes as names without repeats.")
                names = []
            out["classes"] = {str(name): None for name in names}
    if relation == "related":
        allowed_keys |= {"concepts"}
        concepts = tuple(raw.get("concepts") or ())
        unknown = [c for c in concepts if c not in ontology.concepts]
        if not concepts or unknown:
            problems.append(f"{where}: a related mapping needs known concepts; unknown {unknown}.")
        concepts = tuple(c for c in concepts if c in ontology.concepts)
        out.update(concepts=concepts,
                   structures=tuple(dict.fromkeys(s for c in concepts for s in ontology.concept(c).structures)))
    elif relation == "unmapped":
        allowed_keys |= {"structures", "reason"}
        structures = tuple(raw.get("structures") or ())
        unknown = [s for s in structures if s not in ontology.structures]
        if not structures or unknown:
            problems.append(f"{where}: an unmapped label needs known structures; unknown {unknown}.")
        if not raw.get("reason"):
            problems.append(f"{where}: an unmapped label needs a reason.")
        out.update(structures=structures, reason=str(raw.get("reason") or "").strip())
    unknown = set(raw) - allowed_keys
    if unknown:
        problems.append(f"{where}: unknown fields {sorted(unknown)}.")
    return Entry(**out)


def validate_mappings(raw: Mapping, ontology: Ontology, path: str = "<memory>") -> LabelMappings:
    problems: list[str] = []
    if not isinstance(raw, Mapping):
        raise MappingError(f"{path}: the mapping file must be a mapping.")
    if raw.get("schema") != MAPPINGS_SCHEMA:
        problems.append(f"Schema must be {MAPPINGS_SCHEMA!r}.")
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        problems.append(f"Version {version!r} must be a positive integer.")
    if raw.get("ontology") != ontology.tag:
        problems.append(f"Ontology {raw.get('ontology')!r} is not the loaded ontology {ontology.tag!r}.")
    sources = {}
    for name, sraw in (raw.get("sources") or {}).items():
        sraw = sraw or {}
        kind = sraw.get("kind")
        if kind not in SOURCE_KINDS:
            problems.append(f"Source {name}: kind {kind!r} not in {list(SOURCE_KINDS)}.")
            continue
        if not sraw.get("description") or not sraw.get("vocabulary"):
            problems.append(f"Source {name}: needs a description and a vocabulary.")
        entries = {str(label): _entry(ontology, kind, str(label), eraw, f"{name}.{label}", problems)
                   for label, eraw in (sraw.get("labels") or {}).items()}
        if not entries:
            problems.append(f"Source {name}: no labels.")
        sources[name] = Source(name=name, kind=kind, description=" ".join(str(sraw.get("description")).split()),
                               vocabulary=str(sraw.get("vocabulary")), entries=entries)
    if not sources:
        problems.append("No sources.")
    if problems:
        raise MappingError(f"{os.path.basename(path)} is not a valid label mapping:\n- " + "\n- ".join(problems))
    return LabelMappings(version=version, ontology_tag=ontology.tag, path=path, sources=sources)


def load_mappings(ontology: Ontology, path: str = DEFAULT_MAPPINGS_PATH) -> LabelMappings:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return validate_mappings(raw, ontology, path=path)


def _echoprime_coarse_views(path: str) -> tuple[str, ...]:
    """Extracts `COARSE_VIEWS` from EchoPrime's `utils.py`, read without importing the module (it needs cv2 and torch)."""
    with open(path, encoding="utf-8") as f, warnings.catch_warnings():
        # EchoPrime source code contains escape sequences that may generate warnings, so suppresses `SyntaxWarning`
        # messages while parsing the file.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "COARSE_VIEWS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"`COARSE_VIEWS` not found in {path}.")


def source_vocabularies() -> dict[str, dict[str, tuple[str, ...] | None]]:
    """Each source's labels -> their class names (`None` for views and regression tasks), from the vendored assets."""
    import pandas as pd     # for PanEcho task file.

    with open(ECHOPRIME_FINDINGS_PATH, encoding="utf-8") as f:
        echoprime = json.load(f)
    panecho = pd.read_pickle(PANECHO_TASKS_PATH)
    return {
        "deepecho_views": {v: None for v in VIEWS},
        "echoprime_views": {v: None for v in _echoprime_coarse_views(ECHOPRIME_VIEWS_PATH)},
        "echoprime_findings": {task: ECHOPRIME_BINARY_CLASSES if spec["mode"] == "binary" else None
                               for task, spec in echoprime.items()},
        "panecho_tasks": {task: tuple(str(c) for c in spec["class_names"]) or None for task, spec in panecho.items()},
    }


def completeness_problems(mappings: LabelMappings, vocabularies: Mapping[str, Mapping]) -> list[str]:
    """Verifies that the label mapping configuration contains exactly the labels and class names defined by
    the original models, per source. Empty when the mapping is complete."""
    problems = []
    if set(mappings.sources) != set(vocabularies):
        problems.append(f"Sources {sorted(mappings.sources)} differ from {sorted(vocabularies)}.")
    for name, vocab in vocabularies.items():
        source = mappings.sources.get(name)
        if source is None:
            continue
        missing, extra = set(vocab) - set(source.entries), set(source.entries) - set(vocab)
        if missing or extra:
            problems.append(f"{name}: missing labels {sorted(missing)}, unknown labels {sorted(extra)}.")
        for label, classes in vocab.items():
            entry = source.entries.get(label)
            if entry is None:
                continue
            if classes is None and entry.classes:
                problems.append(f"{name}.{label}: the source has no classes but the mapping lists "
                                f"{list(entry.classes)}.")
            if classes is not None and set(entry.classes) != set(classes):
                problems.append(f"{name}.{label}: classes {sorted(entry.classes)} differ from the source's "
                                f"{sorted(classes)}.")
    return problems


def main(argv=None) -> int:
    ontology = load_ontology()
    mappings = load_mappings(ontology)
    problems = completeness_problems(mappings, source_vocabularies())
    for name, source in mappings.sources.items():
        counts = {r: sum(e.relation == r for e in source.entries.values()) for r in RELATIONS}
        print(f"{name}: {len(source.entries)} labels, " + ", ".join(f"{r} {n}" for r, n in counts.items() if n))
    for p in problems:
        print(f"INCOMPLETE: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
