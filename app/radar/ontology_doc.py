"""Generate and synchronize the tables of `docs/radar/ontology_v1.md` from the ontology and label-mapping YAML files:
the taxonomies, findings and label mappings. The reviewer instructions are generated from the review forms by
`tools.labelbox.instructions`.

Usage:
    python -m app.radar.ontology_doc          # rewrite the generated parts.
    python -m app.radar.ontology_doc --check  # exit `1` if any part is out of date.
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import sys
from typing import Callable, Mapping

from app.radar.label_mappings import RELATIONS, UNSPECIFIED, Entry, LabelMappings, Source, load_mappings
from app.radar.ontology import REPO_ROOT, VIEWS, Concept, Ontology, load_ontology

DOCS_DIR = os.path.join(REPO_ROOT, "docs", "radar")
DEFAULT_DOC_PATH = os.path.join(DOCS_DIR, "ontology_v1.md")
# Identify important pieces between `<!-- BEGIN GENERATED: <name> -->` and `<!-- END GENERATED: <name> -->`.
_BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z_]+) -->\n)(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)", re.S
)


def _table(header: list[str], rows: list[list[str]]) -> str:
    """Creates a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _code(items) -> str:
    """Formats identifiers as inline code: converts a collection of identifiers into
    comma-separated markdown inline code."""
    return ", ".join(f"`{i}`" for i in items)


def _compress_views(ontology: Ontology, views: set) -> list[str]:
    """Replaces complete view groups with their names and lists remaining views in taxonomy order."""
    out, covered = [], set()
    for name, members in ontology.view_groups.items():
        if set(members) <= views:
            out.append(name)
            covered.update(members)
    return out + [v for v in VIEWS if v in views and v not in covered]


def prior_text(ontology: Ontology, concept: Concept) -> str:
    """Describes where an anatomical structure or finding may be visible, in terms of both
    echocardiographic views and acquisition modes, e.g. "`apical_4c`, `PSAX_AV` (bmode, color_flow)"."""
    by_view: dict[str, tuple] = {}
    for view, acq in concept.visible_in:
        by_view[view] = by_view.get(view, ()) + (acq,)
    by_acqs: dict[tuple, set] = {}
    for view, acqs in by_view.items():
        key = tuple(a for a in ("bmode", "color_flow", "spectral_doppler", "mmode") if a in acqs)
        by_acqs.setdefault(key, set()).add(view)
    parts = [f"{_code(_compress_views(ontology, views))} ({', '.join(acqs)})" for acqs, views in by_acqs.items()]
    return "; ".join(sorted(parts))


def value_text(ontology: Ontology, concept: Concept) -> str:
    """Describes a concept's allowed values."""
    if concept.value_type == "boolean":
        return "yes / no"
    if concept.value_type == "ordinal":
        return f"scale `{concept.scale}`"
    if concept.value_type == "categorical":
        return _code(concept.values)
    return f"{concept.unit}, {concept.minimum:g} to {concept.maximum:g}"


def render_views(ontology: Ontology) -> str:
    """Generates the table of echocardiographic views."""
    rows = []
    for vid, v in ontology.views.items():
        group = ", ".join(ontology.view_group_of(vid)) or "none"
        rows.append([f"`{vid}`", f"`{group}`" if group != "none" else "none", v.window, v.variant,
                     v.labels["en"], v.labels["fr"], v.description["en"]])
    return _table(["View", "Group", "Window", "Variant", "English", "French", "The standard view shows"], rows)


def render_acquisitions(ontology: Ontology) -> str:
    """Creates a bilingual table of acquisition categories."""
    rows = [[f"`{a}`", ontology.acquisition_labels["en"][a], ontology.acquisition_labels["fr"][a]]
            for a in ontology.acquisition_labels["en"]]
    return _table(["Category", "English", "French"], rows)


def render_structures(ontology: Ontology) -> str:
    """Generates a table of anatomical structures and their associated concepts.
    Note: a structure is an anatomical object.
    """
    rows = []
    for sid, s in ontology.structures.items():
        visibility = ontology.visibility_concept(sid)
        findings = [cid for cid in ontology.concepts_of_structure(sid) if cid != visibility]
        rows.append([
            f"`{sid}`", s.category, s.labels["en"], s.labels["fr"], f"`{visibility}`" if visibility else "none",
            prior_text(ontology, ontology.concept(visibility)) if visibility else "no prior in v1",
            _code(findings),
        ])
    return _table(["Structure", "Category", "English", "French", "Visibility concept",
                   "May be visible in (views, acquisitions)", "Findings"], rows)


def render_findings(ontology: Ontology) -> str:
    """Generates a table of findings. Each finding has information about its associated structures,
    allowed values, evidence, modalities, visibility priors, view-group requirements, and notes.
    Note: a finding is a property or abnormality involving a structure.
    """
    rows = []
    for cid, c in ontology.concepts.items():
        if c.kind != "finding":
            continue
        rows.append([
            f"`{cid}`", c.labels["en"], c.labels["fr"], _code(c.structures), value_text(ontology, c), c.evidence,
            ", ".join(c.required_modality), prior_text(ontology, c), _code(c.requires_views_from) or "", c.notes,
        ])
    return _table(["Finding", "English", "French", "Structures", "Value", "Evidence", "Modalities",
                   "Prior (views, acquisitions)", "Needs views from", "Notes"], rows)


def render_scales(ontology: Ontology) -> str:
    """Documents ordinal rating systems."""
    # `unranked`: values that are valid but do not fit into the ordered progression.
    # `negated_value`: specifies what a negated statement should mean.
    rows = [[f"`{name}`", " < ".join(f"`{v}`" for v in s.ranked), _code(s.unranked),
             f"`{s.negated_value}`" if s.negated_value else "unresolved"] for name, s in ontology.scales.items()]
    return _table(["Scale", "Values, in order", "Outside the order", "Negated mention"], rows)


def _target_text(target) -> str:
    """Expresses a mapping target in readable text.
    
    Two supported shapes:
        Dictionary-style target: represents numeric conditions.
        Collection-style target: represents a list of allowed values.
    """
    if isinstance(target, dict):
        return " and ".join(f"{'at least' if k == 'at_least' else 'below'} {v:g}" for k, v in target.items())
    return ", ".join(f"`{str(v).lower() if isinstance(v, bool) else v}`" for v in target)


def mapping_text(entry: Entry) -> str:
    """Explain how external labels relate to RADAR concepts."""
    if entry.relation == "unmapped":
        return f"structures {_code(entry.structures)}: {entry.reason}"
    if entry.relation == "related":
        listed = f" (classes {_code(entry.classes)} imply no value)" if entry.classes else ""
        return f"related to {_code(entry.concepts)}{listed}"
    parts = [f"`{entry.concepts[0]}`"]
    if entry.classes:
        parts.append("; ".join(f"`{name}` → {_target_text(target)}" for name, target in entry.classes.items()))
    if entry.unassigned:
        parts.append(f"no class: {_target_text(entry.unassigned)}")
    if entry.transform:
        parts.append(entry.transform)
    if entry.also_related:
        parts.append(f"also related to {_code(entry.also_related)}")
    return ": ".join(parts[:2]) + "".join(f"; {p}" for p in parts[2:])


def render_mapping_summary(mappings: LabelMappings) -> str:
    """Counts how many mapping entries belong to each relationship category."""
    rows = []
    for name, source in mappings.sources.items():
        counts = [sum(e.relation == r for e in source.entries.values()) for r in RELATIONS]
        rows.append([f"`{name}`", source.kind, len(source.entries), *counts])
    return _table(["Source", "Kind", "Labels", *RELATIONS], rows)


def _render_view_source(source: Source) -> str:
    """Renders a source's view-label mappings as a Markdown table or a concise summary."""
    entries = source.entries.values()
    if all(e.relation == "exact" and e.expanded_views == (e.label,) for e in entries):
        return f"Every class maps exactly to the view of the same name: {_code(source.entries)}.\n"
    rows = [[f"`{e.label}`", e.relation, _code(e.views),
             e.acquisitions if e.acquisitions == UNSPECIFIED else ", ".join(e.acquisitions), e.notes] for e in entries]
    return _table(["Label", "Relation", "Views", "Acquisitions", "Notes"], rows)


def _render_task_source(source: Source) -> str:
    """Renders a source's clinical task-label mappings as a Markdown table."""
    rows = [[f"`{e.label}`", e.relation, e.unit or "", mapping_text(e), e.notes] for e in source.entries.values()]
    return _table(["Label", "Relation", "Unit", "Maps to", "Notes"], rows)


def _source_section(name: str):
    def render(mappings: LabelMappings) -> str:
        source = mappings.sources[name]
        body = _render_view_source(source) if source.kind == "view" else _render_task_source(source)
        return f"{source.description} Source: `{source.vocabulary}`.\n\n{body}"

    return render


ONTOLOGY_SECTIONS = {
    "views": render_views,
    "acquisitions": render_acquisitions,
    "structures": render_structures,
    "findings": render_findings,
    "scales": render_scales,
}
MAPPING_SECTIONS = {
    "mapping_summary": render_mapping_summary,
    "mapping_deepecho_views": _source_section("deepecho_views"),
    "mapping_echoprime_views": _source_section("echoprime_views"),
    "mapping_echoprime_findings": _source_section("echoprime_findings"),
    "mapping_panecho_tasks": _source_section("panecho_tasks"),
}


def document_sections(ontology: Ontology, mappings: LabelMappings) -> dict[str, dict]:
    """Maps each document path to its named generated-section renderers."""
    return {DEFAULT_DOC_PATH: {
        **{name: functools.partial(fn, ontology) for name, fn in ONTOLOGY_SECTIONS.items()},
        **{name: functools.partial(fn, mappings) for name, fn in MAPPING_SECTIONS.items()},
    }}


def sync(text: str, sections: Mapping[str, Callable[[], str]]) -> str:
    """Regenerate marked sections while preserving manual text, requiring each section exactly once."""
    found = [m.group("name") for m in _BLOCK.finditer(text)]
    if sorted(found) != sorted(sections):
        raise ValueError(f"generated sections {sorted(found)} do not match {sorted(sections)}")
    return _BLOCK.sub(lambda m: m.group(1) + sections[m.group("name")]() + m.group(4), text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Exit `1` if any document is out of date.")
    args = parser.parse_args(argv)
    ontology = load_ontology()
    stale = []
    for path, sections in document_sections(ontology, load_mappings(ontology)).items():
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new = sync(text, sections)
        if new != text:
            stale.append(os.path.relpath(path, REPO_ROOT))
            if not args.check:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
    if args.check and stale:
        print(f"Out of date: {', '.join(stale)}: run `python -m app.radar.ontology_doc`.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
