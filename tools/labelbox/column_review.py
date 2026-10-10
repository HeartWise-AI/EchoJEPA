"""Named DeepECHO columns, conservative report prefills and Labelbox MAL drafts.

Offline: the ontology and site share IDs, values and error reasons.
A prelabel is never a submitted annotation.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
import uuid
from pathlib import Path

from tools.labelbox.review_ontology import _classification, _option

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/radar/column_review_v2.json"
FORM = ROOT / "configs/radar/labelbox/column_review_v2.json"


def load_spec():
    return json.loads(SPEC.read_text())


def _choice(name, label, choices, *, required=True, top=False):
    out = _classification("radio", name, label, required=required,
                          options=[_option(k, v) for k, v in choices.items()], top_level=top)
    out["uiMode"] = "searchable"
    return out


def _checklist(name, label, options, *, required=False, top=False):
    out = _classification("checklist", name, label, required=required, options=options, top_level=top)
    out["uiMode"] = "searchable"
    return out


def build_form(spec=None):
    spec = spec or load_spec()
    review_columns, missing_columns, unsupported_columns = [], [], []
    for key, field in spec["fields"].items():
        correction = {"remove": "Remove this claim", **field["values"]}
        status = _classification("radio", key+"__status", "Confirm this finding", required=True, options=[
            _option("yes", "Supported in this input"),
            _option("no", "Incorrect / should not be here", [
                _choice(key+"__error", "Why should it not be here?", spec["error_kinds"]),
                _choice(key+"__correction", "Correction", correction),
            ]),
            _option("cannot_assess", "Cannot assess", [
                _choice(key+"__limitation", "Why can it not be assessed?", spec["cannot_assess_reasons"])
            ]),
        ])
        review_columns.append(_option(key, field["label"], [
            _choice(key+"__value", "Proposed value (original retained in context)", field["values"]), status]))
        missing_columns.append(_option(key, field["label"], [
            _choice(key+"__missing_value", "Missing visible value", field["values"]),
            _choice(key+"__missing_visible", "Visible in this input?", {"yes": "Yes", "cannot_assess": "Cannot assess"})]))
        unsupported_columns.append(_option(key, field["label"], [
            _choice(key+"__unsupported_kind", "Why should this statement not be here?", spec["error_kinds"])]))
    for key, label in spec["unsupported_columns"].items():
        unsupported_columns.append(_option(key, label, [
            _choice(key+"__unsupported_kind", "Why should this statement not be here?", spec["error_kinds"]),
            _classification("text", key+"__statement", "Copy the offending statement", required=True)]))
    questions = [
        _choice("view", "Proposed view", {v: v for v in spec["views"]}, top=True),
        _choice("modality", "Acquisition mode", spec["modalities"], top=True),
        _choice("view_ok", "View correct?", {"yes": "Yes", "no": "No — reroute", "cannot_assess": "Cannot assess"}, top=True),
        _choice("modality_ok", "Acquisition mode correct?", {"yes": "Yes", "no": "No — reroute", "cannot_assess": "Cannot assess"}, top=True),
        _checklist("visible_targets", "Confirm targets actually covered and assessable", [_option(k, v) for k, v in spec["targets"].items()], top=True),
        _choice("psax_level", "PSAX slice / SUBCOSTAL subtype (when relevant)",
                {"not_applicable": "Not applicable", "unresolved": "Unresolved", "mid": "PSAX mid / papillary level", "basal": "PSAX basal", "apical": "PSAX apical", "mitral": "PSAX mitral leaflet level", "ivc": "IVC focused", "four_chamber": "Subcostal four chamber"}, top=True),
        _checklist("review_columns", "Prefilled findings — confirm or correct selected columns", review_columns, top=True),
        _classification("radio", "missing_information", "Important visible finding missing from the candidate?", required=True, options=[
            _option("no", "No"), _option("yes", "Yes", [_checklist("missing_columns", "Select missing columns and values", missing_columns, required=True)])], top_level=True),
        _classification("radio", "unsupported_information", "Additional statement that should not be here?", required=True, options=[
            _option("no", "No"), _option("yes", "Yes", [_checklist("unsupported_columns", "Select offending columns / statements", unsupported_columns, required=True)])], top_level=True),
        _classification("text", "candidate_caption", "Original caption (kept in immutable context; generated from proposals)", required=False, top_level=True),
        _classification("radio", "review_completed", "I reviewed this input, checked the proposed values and targets, and confirmed or corrected the suggestions.", required=True, options=[_option("yes", "Review completed")], top_level=True),
    ]
    return {"tools": [], "classifications": questions}


def _normalize(value):
    text = unicodedata.normalize("NFKD", str(value or "").lower().replace("’", "'"))
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def _clauses(report, sources):
    for source in sources:
        text = report.get(source)
        if not isinstance(text, str):
            continue
        for clause in re.split(r"//|;|\n|\.(?!\d)", text):
            normalized = _normalize(clause)
            if normalized and not re.search(r"suspect|possible|probable|cannot exclude|non evalu|not assess|incertain", normalized):
                yield source, clause.strip(), normalized


def allowed_columns(view, modality, *, targets=None, level="not_applicable", input_kind="video", spec=None):
    spec = spec or load_spec()
    route = spec["views"].get(view)
    if not route or modality not in {"bmode", "color_flow"}:
        return []
    columns = []
    for key in route["columns"]:
        field = spec["fields"][key]
        if modality not in field["modalities"] or (targets is not None and field["target"] not in targets):
            continue
        if input_kind == "image" and (key.startswith("wall_") or key.endswith("_function") or key.endswith("_motion") or key == "ivc_collapse"):
            continue
        if view == "PSAX":
            if key.startswith("wall_") and level != "mid":
                continue
            if field["target"] == "mitral" and level != "mitral":
                continue
        if view == "SUBCOSTAL":
            if level not in {"ivc", "four_chamber"}:
                continue
            if (field["target"] == "ivc") != (level == "ivc"):
                continue
        columns.append(key)
    return columns


def value_allowed(key, value, view, spec=None):
    spec = spec or load_spec()
    return (key in spec["fields"] and value in spec["fields"][key]["values"]
            and not (key == "mitral_motion" and value == "prolapse" and str(view).split("_")[0] not in {"PLAX", "A3C"}))


def _text_value(key, text):
    """Narrow explicit assertions; no numeric thresholds or general NLP inference."""
    if key in {"lv_function", "rv_function"}:
        if re.search(r"pas de dysfonction|sans dysfonction|no dysfunction|not reduced", text):
            return None
        for severity, value in [("severe|severely", "severely_reduced"), ("modere|moderate", "moderately_reduced"), ("legere?|mild", "mildly_reduced")]:
            if re.search(r"(?:dysfonction|dysfunction|reduced).*?(?:"+severity+r")|(?:"+severity+r").*?(?:dysfonction|dysfunction|reduced)", text):
                return value
        if re.search(r"fonction systolique.{0,35}\bnormal|normal systolic function|systolic function.*normal", text):
            return "normal"
        if key == "lv_function" and re.search(r"hyperdynamique|hyperdynamic", text):
            return "hyperdynamic"
    if key in {"lv_size", "rv_size", "la_size", "ra_size", "ivc_size"}:
        if re.search(r"non[- ]dilate|pas de dilatation|not dilated|dimension normale|volume normal|normal (?:size|dimension|volume)", text):
            return "normal"
        if re.search(r"dilatation|dilate|dilated", text):
            return "dilated"
    if key == "lv_wall_thickness":
        if re.search(r"epaisseur.*normale|normal wall thickness", text):
            return "not_increased"
        if re.search(r"augmentation.*epaisseur|epaississement.*paroi|increased.*wall thickness", text):
            return "increased"
    if key.endswith("_appearance"):
        for pattern, value in [("calcifi", "calcified"), ("sclerose|sclerotic", "sclerotic"), ("epaissi|thickened", "thickened"),
                               (r"^normale?$|feuillets.*normaux|leaflets.*normal|normal.*leaflets|cusps.*normal|valve.{0,30}\bnormal|normal (?:tricuspid|mitral|aortic|pulmonic|pulmonary) valve|(?:tricuspide|tricuspid|mitrale|mitral|aortique|aortic|pulmonic)[: ]+normal", "normal")]:
            if re.search(r"pas de|sans |no |not ", text):
                continue
            for match in re.finditer(pattern, text):
                if value == "normal" and (
                    re.search(r"motion|mobilite|mobility|opening|ouverture|function|fonction", match.group())
                    or re.match(r"e?\s+(?:(?:leaflet|cusp)\s+)?(?:motion|opening|function|mobilite|mobility|ouverture|fonction)\b", text[match.end():])
                ):
                    continue
                return value
    if key.endswith("_motion") and not key.startswith("wall_"):
        for pattern, value in [("prolaps", "prolapse"), ("flail|eversion", "flail"), ("tether|traction", "tethered"), ("mobilite.*reduite|ouverture.*reduite|restricted", "restricted"), ("mobilite.*normale|ouverture.*normale|normal.*(?:motion|opening)|(?:motion|opening).*normal", "normal")]:
            if re.search(pattern, text) and not re.search(r"pas de|sans |no |not ", text):
                return value
    if key.endswith("_device"):
        for pattern, value in [("mitraclip|triclip|edge.to.edge", "edge_to_edge_clip"), ("annuloplast|annuloplasty", "annuloplasty_ring"), ("bioprothese|bioprosthe", "bioprosthesis"), ("prothese mecanique|mechanical prosthe", "mechanical_prosthesis")]:
            if re.search(pattern, text) and not re.search(r"pas de|sans |no |not ", text):
                return value
    if key == "aortic_cusp_count":
        if re.search(r"tricuspide|three.cusp|3 cusps", text):
            return "three"
        if re.search(r"bicuspide|bicuspid|two.cusp|2 cusps", text):
            return "two"
    if key.endswith("_regurgitant_jet") and re.search(r"insuffisance|regurgitation|regurgitant", text):
        # A negated stenosis in the same clause must not negate regurgitation.
        if re.search(r"pas de stenose|no stenosis", text):
            return None
        return "not_demonstrated" if re.search(r"pas d'|absence|sans |no |not |aucune", text) else "demonstrated"
    if key == "pericardial_effusion" and re.search(r"epanchement pericardique|pericardial effusion", text):
        return "not_demonstrated" if re.search(r"pas d'|absence|sans |no |not |aucun", text) else "present"
    if key == "ivc_collapse" and re.search(r"collaps|respiratoire|respiratory", text):
        if re.search(r"reduit|reduced|absent|minimal", text):
            return "reduced"
        if re.search(r"preserv|normal", text):
            return "preserved"
    if key.startswith("aorta_"):
        location = key.split("_")[1]
        relevant = {"root": r"sinus de valsalva|aortic root|racine", "ascending": r"ascendante|ascending", "arch": r"crosse|aortic arch"}[location]
        if re.search(relevant, text):
            if re.search(r"pas de dilatation|not dilated|non[- ]dilate|normal", text):
                return "normal"
            if re.search(r"dilatation|dilate|dilated", text):
                return "dilated"
    return None


def build_candidate(report, view, modality, *, global_key, targets=None, level="not_applicable", input_kind="video", input_provenance_verified=False, spec=None):
    spec = spec or load_spec()
    if view not in spec["views"] or modality not in spec["modalities"]:
        raise ValueError("Unknown view or acquisition mode")
    route = spec["views"][view]
    # Focused/zoom targets are offered in the form, not assumed in a prelabel.
    suggested_targets = list(targets) if targets is not None else [t for t in route["targets"] if t not in route["conditional_targets"]]
    if targets is None and view == "PSAX":
        suggested_targets = ["lv", "pericardium"] if level in {"mid", "basal", "apical"} else ["mitral"] if level == "mitral" else []
    if targets is None and view == "SUBCOSTAL":
        suggested_targets = ["ivc"] if level == "ivc" else [t for t in route["targets"] if t != "ivc"] if level == "four_chamber" else []
    columns = []
    for key in allowed_columns(view, modality, targets=suggested_targets, level=level, input_kind=input_kind, spec=spec):
        field = spec["fields"][key]
        if key.startswith("wall_"):
            source = route["wall_sources"].get(key.removeprefix("wall_"))
            raw = report.get(source) if source else None
            try:
                number = float(raw)
                value = spec["wms_scale"].get(str(int(number))) if number.is_integer() else None
            except (TypeError, ValueError, OverflowError):
                value = None
            if value:
                columns.append(dict(concept=key, value=value, source_column=source, source_text=str(raw), origin="report_wms_ase_assumption"))
            continue
        matches = [(value, source, clause) for source, clause, text in _clauses(report, field["sources"])
                   if (value := _text_value(key, text)) and value_allowed(key, value, view, spec)]
        if key == "rv_function":
            raw = report.get("MHI VD fonction systolique")
            try:
                number = float(raw)
                code = str(int(number)) if number.is_integer() else None
            except (TypeError, ValueError, OverflowError):
                code = None
            value = field["verified_codes"].get(code)
            if value:
                matches.append((value, "MHI VD fonction systolique", str(raw)))
        if matches and len({m[0] for m in matches}) == 1:
            value, source, text = matches[0]
            columns.append(dict(concept=key, value=value, source_column=source, source_text=text, origin="explicit_report"))
    candidate = dict(schema=spec["schema"], version=spec["version"], global_key=global_key, view=view, modality=modality,
                     suggested_targets=suggested_targets, level=level, columns=columns,
                     input_kind=input_kind, input_provenance_verified=input_provenance_verified)
    context_columns = {source for key in route["columns"] if not key.startswith("wall_")
                       for source in spec["fields"][key]["sources"]+spec["fields"][key].get("context_columns", [])}
    context_columns.update(route["wall_sources"].values())
    candidate["report_context"] = {source: report[source] for source in sorted(context_columns)
                                   if source in report and report[source] is not None and report[source] == report[source]}
    candidate["candidate_caption"] = " ".join(finding_sentence(c["concept"], c["value"], spec) for c in columns)
    return candidate


def finding_sentence(key, value, spec=None):
    spec = spec or load_spec()
    if key in {"lv_function", "rv_function"}:
        chamber = key[:2].upper()
        description = spec["fields"][key]["values"][value].lower()
        return f"{chamber} systolic function appears {description} in this input."
    if key in {"lv_size", "rv_size", "la_size", "ra_size"}:
        chamber = {"lv_size": "LV cavity", "rv_size": "RV cavity", "la_size": "left atrium", "ra_size": "right atrium"}[key]
        return f"The visible {chamber} is {'not dilated' if value == 'normal' else 'dilated'}."
    if key.endswith("_appearance"):
        valve = key.split("_")[0]
        return f"The visible {valve} valve has a {value} appearance." if value == "normal" else f"The visible {valve} valve appears {value}."
    if key.endswith("_motion") and not key.startswith("wall_"):
        valve = key.split("_")[0]
        leaflets = "cusps" if valve in {"aortic", "pulmonic"} else "leaflets"
        return f"The visible {valve} {leaflets} show {value} motion."
    if key.endswith("_regurgitant_jet"):
        valve = key.split("_")[0]
        return f"A {valve} regurgitant jet is demonstrated in this color acquisition." if value == "demonstrated" else f"No {valve} regurgitant jet is demonstrated in this acquisition."
    if key == "pericardial_effusion":
        return "Pericardial effusion is demonstrated in the imaged region." if value == "present" else "No pericardial effusion is demonstrated in the imaged region."
    if key.startswith("wall_"):
        name = key.removeprefix("wall_").replace("_", " ")
        motion = {"normal": "normal motion and thickening", "normal_or_hyperkinetic": "normal or hyperkinetic motion", "akinetic_or_severely_hypokinetic": "akinesis or severe hypokinesis", "akinetic": "akinesis", "hypokinetic": "hypokinesis", "severely_hypokinetic": "severe hypokinesis", "hyperkinetic": "hyperkinesis", "dyskinetic": "dyskinesis", "aneurysmal": "aneurysmal deformation"}[value]
        return f"The visible {name} segment shows {motion}."
    return f"{spec['fields'][key]['label']}: {spec['fields'][key]['values'][value].lower()} in this input."


def _answer(name, value, children=()):
    return {"name": name, "answer": {"name": value, **({"classifications": list(children)} if children else {})}}


def prelabels(candidate, spec=None):
    """Global video classifications; review_completed is intentionally absent."""
    spec = spec or load_spec()
    _validate_candidate(candidate, spec)
    selected = [{"name": c["concept"], "classifications": [
        _answer(c["concept"]+"__value", c["value"]), _answer(c["concept"]+"__status", "yes")]} for c in candidate["columns"]]
    records = [
        _answer("view", candidate["view"]), _answer("modality", candidate["modality"]),
        _answer("view_ok", "yes"), _answer("modality_ok", "yes"), _answer("psax_level", candidate["level"]),
        {"name": "visible_targets", "answer": [{"name": t} for t in candidate["suggested_targets"]]},
        {"name": "review_columns", "answer": selected}, _answer("missing_information", "no"),
        _answer("unsupported_information", "no"), {"name": "candidate_caption", "answer": candidate["candidate_caption"]},
    ]
    for record in records:
        record["dataRow"] = {"globalKey": candidate["global_key"]}
        record["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{spec['schema']}:{spec['version']}:{candidate['global_key']}:{record['name']}"))
    validate_annotations(records, build_form(spec))
    return records


def _validate_candidate(candidate, spec):
    if candidate.get("schema") != spec["schema"] or candidate.get("version") != spec["version"]:
        raise ValueError("Candidate ontology version mismatch")
    if not isinstance(candidate.get("global_key"), str) or not candidate["global_key"].strip():
        raise ValueError("Opaque global_key is required")
    if candidate.get("view") not in spec["views"] or candidate.get("modality") not in spec["modalities"]:
        raise ValueError("Unknown view/modality")
    seen = set()
    for column in candidate["columns"]:
        key = column["concept"]
        if key in seen or not value_allowed(key, column["value"], candidate["view"], spec):
            raise ValueError("Duplicate or invalid candidate column")
        seen.add(key)


def validate_annotations(annotations, form, *, require_complete=False):
    """Validate the full nested path, not whether a name exists somewhere."""
    def walk(records, questions):
        lookup = {q["name"]: q for q in questions}
        seen = set()
        for record in records:
            name = record["name"]
            if name not in lookup or name in seen:
                raise ValueError("Unknown, duplicate or incorrectly nested classification")
            seen.add(name)
            q, answer = lookup[name], record["answer"]
            if q["type"] == "text":
                if not isinstance(answer, str):
                    raise ValueError("Text answer must be a string")
                continue
            options = {o["value"]: o for o in q["options"]}
            answers = answer if q["type"] == "checklist" else [answer]
            if not isinstance(answers, list) or (q["required"] and require_complete and not answers):
                raise ValueError("Required classification is empty")
            used = set()
            for selected in answers:
                value = selected["name"]
                if value not in options or value in used:
                    raise ValueError("Unknown or duplicate selected value")
                used.add(value)
                walk(selected.get("classifications", []), options[value].get("options", []))
        if require_complete and any(q["required"] and q["name"] not in seen for q in questions):
            raise ValueError("Required review classification missing")
    walk(annotations, form["classifications"])


def decode_annotations(annotations):
    out = {}
    def walk(records):
        for record in records:
            name, answer = record["name"], record["answer"]
            if name in out:
                raise ValueError("Duplicate classification")
            if isinstance(answer, str):
                out[name] = answer
            elif isinstance(answer, list):
                out[name] = [a["name"] for a in answer]
                for a in answer:
                    walk(a.get("classifications", []))
            else:
                out[name] = answer["name"]
                walk(answer.get("classifications", []))
    walk(annotations)
    return out


def score_review(candidate, answers, *, submitted=False, spec=None):
    spec = spec or load_spec()
    _validate_candidate(candidate, spec)
    original = {c["concept"]: c for c in candidate["columns"]}
    if set(answers.get("review_columns", [])) != set(original):
        raise ValueError("Original columns must be reviewed; additions go in missing_columns")
    supported, rejected, unresolved, corrections, omissions, errors, pending = [], [], [], [], [], [], []
    for key, column in original.items():
        status = answers.get(key+"__status")
        if answers.get(key+"__value") != column["value"]:
            raise ValueError("Original value changed: use the explicit correction control")
        if status == "yes":
            supported.append(column)
        elif status == "cannot_assess":
            if answers.get(key+"__limitation") not in spec["cannot_assess_reasons"]:
                pending.append(key+"__limitation")
            unresolved.append(column)
        elif status == "no":
            rejected.append(column)
            kind, value = answers.get(key+"__error"), answers.get(key+"__correction")
            if kind not in spec["error_kinds"] or (value != "remove" and not value_allowed(key, value, answers.get("view"), spec)):
                pending.append(key+"__correction")
            else:
                errors.append(dict(concept=key, kind=kind))
                if value != "remove":
                    corrections.append(dict(concept=key, value=value, origin="reviewer_correction"))
        else:
            pending.append(key+"__status")
    if answers.get("missing_information") == "yes":
        keys = answers.get("missing_columns", [])
        if not keys or len(keys) != len(set(keys)):
            pending.append("missing_columns")
        for key in keys:
            value = answers.get(key+"__missing_value")
            if key in original:
                raise ValueError("Existing columns use correction, not omission")
            if not value_allowed(key, value, answers.get("view"), spec) or answers.get(key+"__missing_visible") != "yes":
                pending.append(key+"__missing_visible")
            else:
                omissions.append(dict(concept=key, value=value, origin="reviewer_omission"))
    elif answers.get("missing_information") != "no":
        pending.append("missing_information")
    if answers.get("unsupported_information") == "yes":
        keys = answers.get("unsupported_columns", [])
        if not keys or len(keys) != len(set(keys)):
            pending.append("unsupported_columns")
        for key in keys:
            if key not in spec["fields"] and key not in spec["unsupported_columns"]:
                raise ValueError("Unknown unsupported column")
            kind = answers.get(key+"__unsupported_kind")
            if kind not in spec["error_kinds"] or (key in spec["unsupported_columns"] and not str(answers.get(key+"__statement", "")).strip()):
                pending.append(key+"__unsupported_kind")
            else:
                errors.append(dict(concept=key, kind=kind))
                if key in original and answers.get(key+"__status") == "yes":
                    pending.append(key+"__status_conflict")
    elif answers.get("unsupported_information") != "no":
        pending.append("unsupported_information")
    actual = supported + corrections + omissions
    allowed = allowed_columns(answers.get("view"), answers.get("modality"), targets=answers.get("visible_targets", []), level=answers.get("psax_level"), input_kind=candidate.get("input_kind", "video"), spec=spec)
    routing = (answers.get("view_ok") == answers.get("modality_ok") == "yes"
               and answers.get("view") == candidate["view"] and answers.get("modality") == candidate["modality"]
               and candidate["view"] != "OTHER"
               and all(c["concept"] in allowed and value_allowed(c["concept"], c["value"], candidate["view"], spec) for c in actual))
    completed = submitted is True and answers.get("review_completed") == "yes"
    caption = " ".join(finding_sentence(c["concept"], c["value"], spec) for c in actual) if completed and routing and not pending else ""
    resolved = len(supported) + len(rejected)
    return dict(supported=supported, rejected=rejected, cannot_assess=unresolved, corrections=corrections, omissions=omissions,
                errors=errors, pending=pending, original_count=len(original), resolved_count=resolved,
                support_rate=len(supported)/len(original) if original else None,
                error_rate=len(rejected)/resolved if resolved else None,
                routing=routing, submitted=completed, caption=caption,
                ready_for_validation=bool(caption) and candidate.get("input_provenance_verified") is True)


def context_attachment(candidate, spec=None):
    spec = spec or load_spec()
    lines = [f"DeepECHO column review v{spec['version']} — original, unverified proposals",
             f"View: {candidate['view']}; acquisition: {candidate['modality']}",
             "Confirm values only in the displayed input. Report numbers are study context."]
    for c in candidate["columns"]:
        lines.append(f"{c['concept']} | {spec['fields'][c['concept']]['values'][c['value']]} | {c.get('source_column', 'candidate')} | {c.get('source_text', '')}")
    if candidate.get("report_context"):
        allowed = {source for f in spec["fields"].values() for source in f["sources"]+f.get("context_columns", [])}
        if any(key not in allowed for key in candidate["report_context"]):
            raise ValueError("Report context contains a column outside the clinical allowlist")
        lines.append("Report context (not local claims):")
        lines.extend(f"{key}: {value}" for key, value in candidate["report_context"].items())
    lines += ["Original caption: "+candidate["candidate_caption"], "Suggestions require submitted human review."]
    return dict(global_key=candidate["global_key"], attachment_type="RAW_TEXT", attachment_name="DeepECHO review v2 — original proposals", attachment_value="\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--candidates", type=Path, help="JSON list of deidentified structured candidates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    form = build_form()
    if args.candidates:
        if not args.output:
            parser.error("--output is required with --candidates")
        candidates = json.loads(args.candidates.read_text())
        keys = [c["global_key"] for c in candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate candidate global keys")
        payloads = [p for c in candidates for p in prelabels(c)]
        attachments = [context_attachment(c) for c in candidates]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"ontology.json").write_text(json.dumps(form, ensure_ascii=False, indent=2)+"\n")
        (args.output/"prelabels.ndjson").write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in payloads))
        (args.output/"attachments.json").write_text(json.dumps(attachments, ensure_ascii=False, indent=2)+"\n")
        print(f"Prepared {len(candidates)} drafts and {len(payloads)} global classifications; no Labelbox requests.")
    elif args.check:
        if not FORM.exists() or json.loads(FORM.read_text()) != form:
            raise SystemExit("Committed v2 form is out of date")
    else:
        FORM.write_text(json.dumps(form, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
