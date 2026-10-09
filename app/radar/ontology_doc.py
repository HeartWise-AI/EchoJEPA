"""Generate and synchronize RADAR documentation from YAML definitions.

- docs/radar/ontology_v1.md: ontology taxonomies, findings and label mappings.
- docs/radar/labeling_instructions_v1.md and .fr.md: the review form, statements and worked examples, in
  English and French.

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
from app.radar.ontology import ACQUISITION_STATEMENT, REPO_ROOT, VIEWS, Concept, Ontology, load_ontology
from app.radar.review_rules import load_examples

DOCS_DIR = os.path.join(REPO_ROOT, "docs", "radar")
DEFAULT_DOC_PATH = os.path.join(DOCS_DIR, "ontology_v1.md")
INSTRUCTIONS_PATHS = {"en": os.path.join(DOCS_DIR, "labeling_instructions_v1.md"),
                      "fr": os.path.join(DOCS_DIR, "labeling_instructions_v1.fr.md")}
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


def render_limitations(ontology: Ontology) -> str:
    """Creates a bilingual table of limitation reasons."""
    rows = [[f"`{r}`", labels["en"], labels["fr"]] for r, labels in ontology.limitations.items()]
    return _table(["Reason", "English", "French"], rows)


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
    "limitations": render_limitations,
}
MAPPING_SECTIONS = {
    "mapping_summary": render_mapping_summary,
    "mapping_deepecho_views": _source_section("deepecho_views"),
    "mapping_echoprime_views": _source_section("echoprime_views"),
    "mapping_echoprime_findings": _source_section("echoprime_findings"),
    "mapping_panecho_tasks": _source_section("panecho_tasks"),
}
# The few words the instruction tables need besides the YAML's own labels.
WORDS = {
    "en": {"question": "Question", "id": "Id", "required": "Required", "answers": "Answers", "yes": "yes",
           "no": "no", "views": "one of the view classes", "statements": "the statements below",
           "each": "each", "free_text": "free text", "statement": "Statement", "label": "Label", "kind": "Kind",
           "acquisition": "acquisition mode", "visibility": "visibility", "finding": "finding",
           "clip": "Clip (synthetic)", "stated": "Classifier view and acquisition", "caption": "Candidate caption",
           "answers_given": "Answers",
           "corrected": "Corrected caption", "unchanged": "unchanged", "why": "Why", "none": "none", "colon": ":"},
    "fr": {"question": "Question", "id": "Id", "required": "Obligatoire", "answers": "Réponses", "yes": "oui",
           "no": "non", "views": "une des classes de coupe", "statements": "les énoncés ci-dessous",
           "each": "chacun", "free_text": "texte libre", "statement": "Énoncé", "label": "Libellé", "kind": "Type",
           "acquisition": "mode d'acquisition", "visibility": "visibilité", "finding": "constatation",
           "clip": "Clip (synthétique)", "stated": "Coupe et acquisition du classifieur",
           "caption": "Légende proposée", "answers_given": "Réponses", "corrected": "Légende corrigée",
           "unchanged": "inchangée", "why": "Pourquoi", "none": "aucun", "colon": " :"},
}


def _statement_label(ontology: Ontology, statement: str, lang: str) -> str:
    """Returns the localized label for an acquisition statement or ontology concept."""
    if statement == ACQUISITION_STATEMENT:
        return ontology.review.acquisition_statement[lang]
    return ontology.concept(statement).labels[lang]


def _detail_text(ontology: Ontology, detail: str, lang: str) -> str:
    """Describes additional information required from a reviewer."""
    spec, w = ontology.review, WORDS[lang]
    label = spec.detail_labels[detail][lang] if detail in spec.detail_labels else ""
    if detail == "limitation":
        return f"{label}{w['colon']} " + ", ".join(lab[lang] for lab in ontology.limitations.values())
    if detail == "correct_view":
        return f"{label} ({w['views']})"
    if detail == "missing":
        return f"{label} ({w['statements']}) + {spec.detail_labels['missing_other'][lang]} ({w['free_text']})"
    if detail == "unsupported":
        kinds = " / ".join(lab[lang] for lab in spec.unsupported_kinds.values())
        return f"{label} ({w['statements']}), {w['each']}{w['colon']} {kinds}"
    if detail == "unconfirmed":
        reasons = [lab[lang] for lab in {**spec.unconfirmed_reasons, **ontology.limitations}.values()]
        return (f"{w['statements']}, {w['each']}{w['colon']} {spec.detail_labels['unconfirmed_reason'][lang]} "
                f"({' / '.join(reasons)})")
    raise ValueError(f"unknown detail {detail!r}")


def render_review_form(ontology: Ontology, lang: str) -> str:
    """Iterates over all questions defined in the ontology's review specification and generates
    the review-form table."""
    w, rows = WORDS[lang], []
    for i, q in enumerate(ontology.review.questions, 1):
        if q.type == "radio":
            answers = "; ".join(f"`{a.id}` {a.labels[lang]}"
                                + (f" → {_detail_text(ontology, a.detail, lang)}" if a.detail else "")
                                for a in q.answers)
        elif q.type == "checklist":
            answers = _detail_text(ontology, q.detail, lang)
        else:
            answers = w["free_text"]
        rows.append([i, q.text[lang], f"`{q.id}`", w["yes"] if q.required else w["no"], answers])
    return _table(["#", w["question"], w["id"], w["required"], w["answers"]], rows)


def render_statements(ontology: Ontology, lang: str) -> str:
    """Generates a reference list of statements that can be reviewed."""
    w = WORDS[lang]
    rows = [[f"`{ACQUISITION_STATEMENT}`", ontology.review.acquisition_statement[lang], w["acquisition"]]]
    rows += [[f"`{cid}`", ontology.concept(cid).labels[lang],
              w["visibility"] if ontology.concept(cid).kind == "structure_visibility" else w["finding"]]
             for cid in ontology.statement_concepts]
    return _table([w["statement"], w["label"], w["kind"]], rows)


def _answer_lines(ontology: Ontology, example: dict, lang: str) -> list[str]:
    """Formats a worked example's review answers as localized Markdown bullet points."""
    spec, w, review = ontology.review, WORDS[lang], example["review"]
    lines = []
    for q in spec.questions:
        if q.type != "radio":
            continue
        answer = next(a for a in q.answers if a.id == review["answers"][q.id])
        text = f"- {q.text[lang]} **{answer.labels[lang]}**"
        if answer.detail == "limitation":
            text += f" ({ontology.limitations[review['limitation']][lang]})"
        elif answer.detail == "correct_view":
            view = review["correct_view"]
            text += f" ({view}: {ontology.views[view].labels[lang]})"
        elif answer.detail == "missing":
            items = [_statement_label(ontology, c, lang) for c in review.get("missing", [])]
            items += [review["missing_other"]] if review.get("missing_other") else []
            text += f" ({'; '.join(items)})"
        elif answer.detail == "unsupported":
            text += " (" + "; ".join(
                f"{_statement_label(ontology, s, lang)}{w['colon']} {spec.unsupported_kinds[k][lang]}"
                for s, k in review["unsupported"].items()) + ")"
        lines.append(text)
    checklist = next(q for q in spec.questions if q.type == "checklist")
    reasons = {**spec.unconfirmed_reasons, **ontology.limitations}
    unconfirmed = "; ".join(f"{_statement_label(ontology, s, lang)}{w['colon']} {reasons[r][lang]}"
                            for s, r in review.get("unconfirmed", {}).items())
    lines.append(f"- {checklist.text[lang]}{w['colon']} **{unconfirmed or w['none']}**")
    return lines


def render_examples(ontology: Ontology, examples: list, lang: str) -> str:
    """Renders worked review examples with captions, answers, corrections and explanations."""
    w, out = WORDS[lang], []

    def quoted(text: str) -> str:
        return f"« {text} »" if lang == "fr" else f"\"{text}\""

    def item(key: str, value: str) -> str:
        return f"- **{w[key]}{w['colon']}** {value}".rstrip()

    for i, e in enumerate(examples, 1):
        corrected = e["corrected_caption"]
        out += [
            f"### {i}. {e['title'][lang]}", "",
            item("clip", e["clip"][lang]),
            item("stated", f"`{e['view']}`, {ontology.acquisition_labels[lang][e['acquisition']]}"),
            item("caption", quoted(e["caption"])),
            item("answers_given", ""), "",
            *("  " + line for line in _answer_lines(ontology, e, lang)), "",
            item("corrected", w["unchanged"] if corrected is None else quoted(corrected)),
            item("why", " ".join(e["why"][lang].split())), "",
        ]
    return "\n".join(out)


def document_sections(ontology: Ontology, mappings: LabelMappings, examples: list) -> dict[str, dict]:
    """Maps each document path to its named generated-section renderers."""
    sections = {DEFAULT_DOC_PATH: {
        **{name: functools.partial(fn, ontology) for name, fn in ONTOLOGY_SECTIONS.items()},
        **{name: functools.partial(fn, mappings) for name, fn in MAPPING_SECTIONS.items()},
    }}
    for lang, path in INSTRUCTIONS_PATHS.items():
        sections[path] = {
            "review_form": functools.partial(render_review_form, ontology, lang),
            "statements": functools.partial(render_statements, ontology, lang),
            "examples": functools.partial(render_examples, ontology, examples, lang),
        }
    return sections


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
    for path, sections in document_sections(ontology, load_mappings(ontology), load_examples(ontology)).items():
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
