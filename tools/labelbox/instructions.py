"""The generated parts of the English reviewer instructions: the questions of both review forms, the column review's
reasons and the worked examples.

    python -m tools.labelbox.instructions            # rewrite the generated parts of the instructions.
    python -m tools.labelbox.instructions --check    # exit 1 if any part is out of date.

The worked examples (`configs/radar/review_examples_v1.yaml`) describe synthetic clips in words. A column review
example is built the way a reviewer meets it: the candidate's prefilled form, the answers the reviewer changes, then
"Review completed". `load_examples` refuses an example that is not a valid review under its form's own checks
(`review_problems` of `column_review` and of `view_type_review`), and the rebuilt caption shown is the scorer's. Form
Labels stay in English, as reviewers see them in Labelbox.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import yaml

from app.radar.ontology import REPO_ROOT
from app.radar.ontology_doc import sync
from tools.labelbox import column_review as cr
from tools.labelbox import view_type_review as vt

EXAMPLES_PATH = os.path.join(REPO_ROOT, "configs", "radar", "review_examples_v1.yaml")
EXAMPLES_SCHEMA = "radar-review-examples"
INSTRUCTIONS_PATHS = {"en": os.path.join(REPO_ROOT, "docs", "radar", "labeling_instructions_v1.md")}
TEXTS = ("title", "clip", "why")
# Nested answers whose choices depend on the column, by the end of their id.
COLUMN_DEPENDENT = (("__value", "column_values"), ("__missing_value", "column_values"),
                    ("__correction", "remove_or_value"), ("__error", "error_reason"),
                    ("__unsupported_kind", "error_reason"), ("__limitation", "limitation"),
                    ("uninterpretable_reason", "uninterpretable_reason"))
# The words the generated parts need besides the forms' own labels.
WORDS = {
    "en": {"question": "Question", "id": "Id", "required": "Required", "answers": "Answers", "yes": "yes", "no": "no",
           "free_text": "free text", "views": "one of the 20 view classes", "each": "each",
           "review_columns": "the prefilled columns", "missing_columns": "any column the view carries",
           "unsupported_columns": "any column or statement type", "statement_type_also": "a statement type also needs",
           "column_values": "one of the column's values",
           "remove_or_value": "`remove` Remove this claim, or one of the column's values",
           "error_reason": "an error reason", "limitation": "an assessment limitation",
           "uninterpretable_reason": "a reason below", "label": "Label",
           "error_kinds": "Error reasons (an incorrect finding, an additional statement)",
           "cannot_assess_reasons": "Assessment limitations (a finding that cannot be assessed)",
           "uninterpretable_reasons": "Uninterpretable clips",
           "unsupported_columns_title":
               "Additional statement types (a statement the caption makes outside its columns)",
           "clip": "Clip (synthetic)", "item": "Item (synthetic)", "proposed": "Proposed",
           "prefilled": "Prefilled findings", "candidate_caption": "Candidate caption",
           "changes": "What the reviewer changes", "none": "none", "nothing": "nothing", "rebuilt": "Rebuilt caption",
           "none_uninterpretable": "none: the clip is uninterpretable",
           "none_rerouted": "none: the view or acquisition mode was corrected",
           "none_empty": "none: no finding was confirmed", "why": "Why", "answers_given": "Answers", "next": "Next",
           "routed": "back to the column review as `{view}`, {modality}",
           "not_routed": "stays out of the column review", "deleted": "deleted from Labelbox: identifying text",
           "asked_after": ", asked when `{question}` is `{answer}`",
           "not_asked": "view and acquisition mode not asked, recorded as not applicable",
           "colon": ":"},
}


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]) + "\n"


def _walk(classifications):
    for q in classifications:
        yield q
        for option in q.get("options", []):
            yield from _walk(option.get("options", []))


def form_index(form: dict) -> dict:
    """Every classification of the form, nested ones included, by its id (ids are unique in the form)."""
    return {q["name"]: q for q in _walk(form["classifications"])}


# The column review form.

def _nested(q: dict, w: dict) -> str:
    return f"{q['instructions']} ({_answers(q, w)})"


def _answers(q: dict, w: dict) -> str:
    """The answers of one question, with what each asks next."""
    if q["type"] == "text":
        return w["free_text"]
    for suffix, key in COLUMN_DEPENDENT:
        if q["name"].endswith(suffix):
            return w[key]
    if q["name"] == "view":
        return w["views"]
    if q["name"] in ("review_columns", "missing_columns", "unsupported_columns"):
        nested = [n for n in q["options"][0]["options"]]  # the same questions for every column
        text = f"{w[q['name']]}, {w['each']}{w['colon']} " + " + ".join(_nested(n, w) for n in nested)
        typed = [n for o in q["options"] for n in o["options"] if n["type"] == "text"][:1]
        if typed:
            text += f"; {w['statement_type_also']} {_nested(typed[0], w)}"
        return text
    parts = []
    for option in q["options"]:
        part = f"`{option['value']}` {option['label']}"
        if option["options"]:
            part += " → " + " + ".join(_nested(n, w) for n in option["options"])
        parts.append(part)
    return "; ".join(parts)


def render_column_form(lang: str) -> str:
    w = WORDS[lang]
    rows = [[i, q["instructions"], f"`{q['name']}`", w["yes"] if q["required"] else w["no"], _answers(q, w)]
            for i, q in enumerate(cr.build_form()["classifications"], 1)]
    return _table(["#", w["question"], w["id"], w["required"], w["answers"]], rows)


def render_column_reasons(lang: str) -> str:
    spec, w, out = cr.load_spec(), WORDS[lang], []
    for key, title in (("error_kinds", "error_kinds"), ("cannot_assess_reasons", "cannot_assess_reasons"),
                       ("uninterpretable_reasons", "uninterpretable_reasons"),
                       ("unsupported_columns", "unsupported_columns_title")):
        out += [f"**{w[title]}**", "", _table([w["id"], w["label"]], [[f"`{k}`", v] for k, v in spec[key].items()])]
    return "\n".join(out)


def render_view_type_form(lang: str) -> str:
    w, rows = WORDS[lang], []
    for q in vt.questions():
        answers = (w["free_text"] if q.choices is None
                   else "; ".join(f"`{v}` {label}" for v, label in q.choices.items()))
        required = w["yes"] if q.required else w["no"]
        if q.asked_after:
            required += w["asked_after"].format(question=q.asked_after[0], answer=q.asked_after[1])
        rows.append([q.text, f"`{q.name}`", required, answers])
    return _table([w["question"], w["id"], w["required"], w["answers"]], rows)


# The worked examples.

def _check_texts(example: dict) -> None:
    for key in TEXTS:
        if not isinstance(example.get(key), str) or not example[key].strip():
            raise ValueError(f"example {example.get('id')!r}: {key} needs English text")


def _column_example(example: dict, spec: dict, form: dict) -> dict:
    """The candidate, the submitted answers and their score, refusing an example that is not a valid review."""
    _check_texts(example)
    view, modality = example["view"], example["modality"]
    candidate = cr.build_candidate({}, view, modality, global_key=f"example-{example['id']}",
                                   level=example.get("level"), spec=spec)
    allowed = cr.allowed_columns(view, modality, targets=candidate["suggested_targets"], level=candidate["level"],
                                 spec=spec)
    for key, value in example["columns"].items():
        if key not in allowed or not cr.value_allowed(key, value, view, spec):
            raise ValueError(f"example {example['id']!r}: {key} = {value!r} cannot be prefilled for this clip")
    candidate["columns"] = [dict(concept=k, value=v, source_column="synthetic", source_text="",
                                 origin="explicit_report") for k, v in example["columns"].items()]
    candidate["candidate_caption"] = " ".join(cr.finding_sentence(k, v, spec) for k, v in example["columns"].items())
    answers = {**cr.decode_annotations(cr.prelabels(candidate, spec)), **example["changes"], "review_completed": "yes"}
    problems = cr.review_problems(candidate, cr.encode_answers(answers, form), spec)
    if problems:
        raise ValueError(f"example {example['id']!r} is not a valid review: {problems}")
    return {**example, "candidate": candidate, "answers": answers,
            "score": cr.score_review(candidate, answers, submitted=True, spec=spec)}


def _view_type_example(example: dict) -> dict:
    _check_texts(example)
    answers, form = example["answers"], vt.build_form()
    cr.validate_annotations(cr.encode_answers(answers, form), form, require_complete=True)
    problems = vt.review_problems(answers)
    if problems:
        raise ValueError(f"example {example['id']!r} is not a valid review: {problems}")
    return {**example, "route": vt.route(answers)}


@functools.lru_cache(maxsize=None)
def load_examples(path: str = EXAMPLES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("schema") != EXAMPLES_SCHEMA:
        raise ValueError(f"{os.path.basename(path)}: schema must be {EXAMPLES_SCHEMA!r}")
    ids = [e["id"] for e in raw["column_review"] + raw["view_type_review"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{os.path.basename(path)}: example ids must be unique")
    spec, form = cr.load_spec(), cr.build_form()
    return {"column_review": [_column_example(e, spec, form) for e in raw["column_review"]],
            "view_type_review": [_view_type_example(e) for e in raw["view_type_review"]]}


def _quote(text: str, lang: str) -> str:
    return f"\"{text}\""


def _item(w: dict, key: str, value: str) -> str:
    return f"- **{w[key]}{w['colon']}** {value}".rstrip()


def _asked(text: str, answer: str, w: dict) -> str:
    """A question and its answer; a question that ends with "?" takes no colon."""
    return f"{text}{' ' if text.endswith('?') else w['colon'] + ' '}{answer}"


def _label(q: dict, value) -> str:
    labels = {o["value"]: o["label"] for o in q["options"]}
    return ", ".join(labels[v] for v in value) if isinstance(value, list) else labels[value]


def _change_lines(example: dict, lang: str, spec: dict, index: dict) -> list[str]:
    """One line per changed answer, naming its column when it belongs to one."""
    w, lines = WORDS[lang], []
    for name, value in example["changes"].items():
        q = index[name]
        column = name.split("__")[0] if "__" in name else None
        column_label = spec["fields"].get(column, {}).get("label") or spec["unsupported_columns"].get(column)
        prefix = f"{column_label}, " if column_label else ""
        answer = _quote(value, lang) if q["type"] == "text" else f"**{_label(q, value)}**"
        lines.append(f"  - {prefix}{_asked(q['instructions'], answer, w)}")
    return lines


def _rebuilt(example: dict, lang: str) -> str:
    w, score = WORDS[lang], example["score"]
    if score["uninterpretable"]:
        return w["none_uninterpretable"]
    if not score["routing"]:
        return w["none_rerouted"]
    return _quote(score["caption"], lang) if score["caption"] else w["none_empty"]


def render_column_examples(lang: str) -> str:
    spec, w, out = cr.load_spec(), WORDS[lang], []
    index = form_index(cr.build_form(spec))
    for i, e in enumerate(load_examples()["column_review"], 1):
        c = e["candidate"]
        prefilled = "; ".join(f"`{col['concept']}` {spec['fields'][col['concept']]['values'][col['value']]}"
                              for col in c["columns"])
        changes = _change_lines(e, lang, spec, index)
        out += [f"### {i}. {e['title']}", "",
                _item(w, "clip", " ".join(e["clip"].split())),
                _item(w, "proposed", f"`{c['view']}`, {spec['modalities'][c['modality']]}"),
                _item(w, "prefilled", prefilled or w["none"]),
                _item(w, "candidate_caption", _quote(c["candidate_caption"], lang)),
                _item(w, "changes", "" if changes else w["nothing"]), *changes,
                _item(w, "rebuilt", _rebuilt(e, lang)),
                _item(w, "why", " ".join(e["why"].split())), ""]
    return "\n".join(out)


def render_view_type_examples(lang: str) -> str:
    w, modalities, out = WORDS[lang], vt.modality_choices(), []
    for i, e in enumerate(load_examples()["view_type_review"], 1):
        answers = e["answers"]
        given = [_asked(q.text, f"**{q.choices[answers[q.name]]}**", w) for q in vt.questions()
                 if q.choices is not None and q.name in answers]
        if any(value == vt.NOT_APPLICABLE for value in vt.with_not_applicable(answers).values()):
            given.append(w["not_asked"])
        given = "; ".join(given)
        if answers["identifier_visible"] == "yes":
            then = w["deleted"]
        elif e["route"]:
            then = w["routed"].format(view=e["route"]["view"], modality=modalities[e["route"]["modality"]])
        else:
            then = w["not_routed"]
        out += [f"### {i}. {e['title']}", "",
                _item(w, "item", " ".join(e["clip"].split())),
                _item(w, "answers_given", given),
                _item(w, "next", then),
                _item(w, "why", " ".join(e["why"].split())), ""]
    return "\n".join(out)


def document_sections(lang: str) -> dict:
    """The named generated parts of the instructions in `lang`."""
    return {name: functools.partial(render, lang) for name, render in (
        ("column_form", render_column_form), ("column_reasons", render_column_reasons),
        ("column_examples", render_column_examples), ("view_type_form", render_view_type_form),
        ("view_type_examples", render_view_type_examples))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if any part is out of date")
    args = parser.parse_args(argv)
    stale = []
    for lang, path in INSTRUCTIONS_PATHS.items():
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new = sync(text, document_sections(lang))
        if new != text:
            stale.append(os.path.relpath(path, REPO_ROOT))
            if not args.check:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
    if args.check and stale:
        print(f"Out of date: {', '.join(stale)}: run `python -m tools.labelbox.instructions`.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
