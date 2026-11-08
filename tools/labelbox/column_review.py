"""Named DeepECHO columns, conservative report prefills and Labelbox MAL drafts.

Offline: the ontology and site share IDs, values and error reasons.
A prelabel is never a submitted annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import uuid
from pathlib import Path

from tools.labelbox.review_ontology import _classification, _option

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/radar/deepecho_video_clip_annotation_specification.json"
FORM = ROOT / "configs/radar/labelbox/deepecho_video_clip_annotation_ontology.json"


POSTERIOR_WALL_VIEWS = {"PLAX", "PLAX_DEEP", "A3C", "A3C_LV", "A3C_ZOOM", "PSAX_BASAL"}  # the views that show that wall.


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
        _classification("radio", "uninterpretable", "Is this clip uninterpretable?", required=True, options=[
            _option("no", "No, it can be interpreted"),
            _option("yes", "Yes, nothing in it can be judged", [
                _choice("uninterpretable_reason", "Why?", spec["uninterpretable_reasons"])])], top_level=True),
        _choice("view", "View (prefilled - correct if needed)", {v: route.get("label", v) for v, route in spec["views"].items()}, top=True),
        _choice("modality", "Acquisition mode (prefilled - correct if needed)", spec["modalities"], top=True),
        _checklist("visible_targets", "Structures visible in this clip, even partly (prefilled: untick any not visible)", [_option(k, v) for k, v in spec["targets"].items()], top=True),
        _choice("subcostal_level", "Subcostal subtype (required for subcostal clips)", spec["levels"], top=True),
        _checklist("review_columns", "Prefilled findings — confirm or correct selected columns", review_columns, top=True),
        _classification("radio", "missing_information", "Important visible finding missing from the candidate?", required=True, options=[
            _option("no", "No"), _option("yes", "Yes", [_checklist("missing_columns", "Select missing columns and values", missing_columns, required=True)])], top_level=True),
        _classification("radio", "unsupported_information", "Additional statement that should not be here?", required=True, options=[
            _option("no", "No"), _option("yes", "Yes", [_checklist("unsupported_columns", "Select offending columns / statements", unsupported_columns, required=True)])], top_level=True),
        _classification("text", "candidate_caption", "Original caption (for reference only; edits are ignored)", required=False, top_level=True),
        _classification("text", "comments", "Comments (no patient details)", required=False, top_level=True),
        _classification("radio", "review_completed", "I reviewed this input, checked the proposed values and structures, and confirmed or corrected the suggestions.", required=True, options=[_option("yes", "Review completed")], top_level=True),
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


def level_choices(view, spec=None):
    """The levels a clip of this view may have: a subcostal view's own subtypes, or "Cannot tell the subtype";
    "Not applicable" for every other view."""
    spec = spec or load_spec()
    levels = spec["views"].get(view, {}).get("levels")
    return [*levels, "unresolved"] if levels else ["not_applicable"]


def default_level(view, spec=None):
    """The level prefilled for a clip of this view: "Not applicable" for a view without levels, else the view's default
    level (subcostal four-chamber)."""
    spec = spec or load_spec()
    route = spec["views"][view]
    return route["default_level"] if route.get("levels") else "not_applicable"


def refines(view, parent, spec=None):
    """Whether `view` narrows the classifier's class `parent` (PSAX_MID refines PSAX). A class does not refine itself."""
    spec = spec or load_spec()
    return spec["views"].get(view, {}).get("refines") == parent


def view_targets(view, level, spec=None):
    """The structures a clip of this view and level may show, all prefilled: the view's own, then its level's."""
    spec = spec or load_spec()
    route = spec["views"][view]
    extra = route.get("levels", {}).get(level, {}).get("targets", [])
    return route["targets"] + [t for t in extra if t not in route["targets"]]


def allowed_columns(view, modality, *, targets=None, level=None, input_kind="video", spec=None):
    """The findings a clip may claim: the view's columns about a structure of the view and level (only those in
    `targets` when given) in a mode that shows them. A column one of the view's levels lists is claimed only at that
    level."""
    spec = spec or load_spec()
    route = spec["views"].get(view)
    if not route or modality not in {"bmode", "color_flow"}:
        return []
    structures = set(view_targets(view, level, spec))
    if targets is not None:
        structures &= set(targets)
    levels = route.get("levels", {})
    level_only = {key for each in levels.values() for key in each["columns"]}
    columns = []
    for key in route["columns"]:
        field = spec["fields"][key]
        if modality not in field["modalities"] or field["target"] not in structures:
            continue
        if input_kind == "image" and (key.startswith("wall_") or key.endswith("_function") or key.endswith("_motion") or key == "ivc_collapse"):
            continue
        if key in level_only and key not in levels.get(level, {}).get("columns", []):
            continue
        columns.append(key)
    return columns


def value_allowed(key, value, view, spec=None):
    spec = spec or load_spec()
    return (key in spec["fields"] and value in spec["fields"][key]["values"]
            and not (key == "mitral_motion" and value == "prolapse" and str(view).split("_")[0] not in {"PLAX", "A3C"})
            and not (key == "lv_wall_thickness" and value == "septum_only"
                     and "ivs" not in spec["views"].get(view, {}).get("targets", []))
            and not (key == "lv_wall_thickness" and value == "posterior_only" and view not in POSTERIOR_WALL_VIEWS))


# A negation word right before a finding (at most two words between, none a conjunction: "sans signe de dilatation",
# "no significant mitral regurgitation", not "non dilaté mais dilatation apicale"), or a negating word right after it
# ("regurgitation not seen", "insuffisance mitrale : aucune", not "légère et aucune sténose"). A negation elsewhere in
# the clause is about something else: "insuffisance mitrale modérée sans prolapsus".
NOT_CONJUNCTION = r"(?!(?:mais|but|et|and|avec|with)\b)"
NEGATION_BEFORE = (r"(?:\bnon[- ]|\bpas d[e']\s?|\bsans |\baucune? |\babsence d[e']\s?|\bni (?:d[e']\s?)?|\bno |\bnot |\bwithout )"
                   r"(?:" + NOT_CONJUNCTION + r"[\w']+ ){0,2}$")
NEGATION_AFTER = (r"^\w*(?: " + NOT_CONJUNCTION + r"[\w']+){0,3}?\s?:?\s"
                  r"(?:aucune?|absente?|none|not (?:seen|demonstrated|visuali[sz]ed|present|detected)"
                  r"|non (?:vue?s?|visualisee?s?|demontree?s?|objectivee?s?))\b")
# Diastolic function is not the systolic function these columns take: "dysfonction diastolique légère" is dropped up
# to the next comma or conjunction, so that its severity cannot be read as a reduced systolic function.
DIASTOLIC = r"(?:\bdys)?fonctions? diastoliques?(?:(?!,| et | mais | avec ).)*|\bdiastolic (?:dys)?function(?:(?!,| and | but | with ).)*"
# Attributes that share one "normal": "de dimension, de masse et de fonction normales", "normal size and function".
ATTRIBUTE = r"\b(?:dimensions?|tailles?|volumes?|masses?|fonctions?(?: systoliques?)?|epaisseurs?(?: parietales?)?|contractilite)"
NORMAL_ATTRIBUTES = rf"((?:(?:de |d'|en )?{ATTRIBUTE}(?:, | et |, et ))*(?:de |d'|en )?{ATTRIBUTE}) normal(?:e|es|s)?\b"
ATTRIBUTE_EN = r"\b(?:size|dimensions?|volume|mass|wall thickness|(?:systolic )?function)"
NORMAL_ATTRIBUTES_EN = rf"\bnormal ((?:{ATTRIBUTE_EN}(?:, | and |, and ))*{ATTRIBUTE_EN})\b"
NORMAL_EF = r"\b(?:fevg|fe vg|fraction d'ejection(?: du vg| ventriculaire gauche)?) (?:(?:est|estimee|visuellement|globale|jugee) )*normale?\b"


def _negations(subject, text):
    """For each mention of `subject` in the clause, whether that mention is negated."""
    return [bool(re.search(NEGATION_BEFORE, text[:m.start()]) or re.search(NEGATION_AFTER, text[m.end():]))
            for m in re.finditer(subject, text)]


def _normal_attributes(text):
    """The attributes the clause calls normal together: {"size", "function"}."""
    words = " ".join(m.group(1) for pattern in (NORMAL_ATTRIBUTES, NORMAL_ATTRIBUTES_EN) for m in re.finditer(pattern, text))
    return ({"size"} if re.search(r"dimension|taille|volume|\bsize", words) else set()) | (
        {"function"} if re.search(r"fonction|contractilite|function", words) else set())


def _text_value(key, text):
    """Narrow explicit assertions; no numeric thresholds or general NLP inference."""
    if key in {"lv_function", "rv_function"}:
        text = re.sub(DIASTOLIC, " ", text)
        if re.search(r"pas de dysfonction|sans dysfonction|no dysfunction|not reduced", text):
            return None
        for severity, value in [("severe|severely", "severely_reduced"), ("modere|moderate", "moderately_reduced"), ("legere?|mild", "mildly_reduced")]:
            if re.search(r"(?:dysfonction|dysfunction|reduced).*?(?:"+severity+r")|(?:"+severity+r").*?(?:dysfonction|dysfunction|reduced)", text):
                return value
        if (re.search(r"fonction systolique.{0,35}\bnormal|normal systolic function|systolic function.*normal", text)
                or (key == "lv_function" and re.search(NORMAL_EF, text)) or "function" in _normal_attributes(text)):
            return "normal"
        if key == "lv_function" and re.search(r"hyperdynamique|hyperdynamic", text):
            return "hyperdynamic"
    if key in {"lv_size", "rv_size", "la_size", "ra_size", "ivc_size"}:
        negations = _negations(r"\bdilat\w*", text)
        normal = (any(negations) or "size" in _normal_attributes(text)
                  or re.search(r"dimension normale|volume normal|taille normale|normal (?:size|dimension|volume)", text))
        dilated = not all(negations)  # a dilatation the clause does not negate
        if normal and dilated:  # "VG non dilaté mais dilatation apicale": not one value
            return None
        if normal:
            return "normal"
        if dilated:
            return "dilated"
    if key == "septal_bulge" and re.search(r"bourrelet[^,]{0,25}sept|sept[^,]{0,25}bourrelet", text):
        # Only a negation of the bulge itself: "pas d'hypertrophie (bourrelet septal 13 mm)" states one.
        return None if re.search(r"(?:\bpas d[e']|absence d[e']|aucun|sans|\bni(?: de)?|\bplus (?:le|de)) ?bourrelet", text) else "present"
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
        # Only a negation of the regurgitation itself: "insuffisance mitrale modérée sans prolapsus" states one.
        negations = _negations(r"\binsuffisance|\bregurgita\w*", text)
        return None if len(set(negations)) != 1 else "not_demonstrated" if negations[0] else "demonstrated"
    if key == "pericardial_effusion" and re.search(r"epanchement pericardique|pericardial effusion", text):
        # "Épanchement péricardique minime sans tamponnade" states an effusion.
        negations = _negations(r"\bepanchement pericardique|\bpericardial effusion", text)
        return None if len(set(negations)) != 1 else "not_demonstrated" if negations[0] else "present"
    if key == "ivc_collapse" and re.search(r"collaps|respiratoire|respiratory", text):
        if re.search(r"reduit|reduced|absent|minimal", text):
            return "reduced"
        if re.search(r"preserv|normal", text):
            return "preserved"
    if key.startswith("aorta_"):
        location = key.split("_")[1]
        relevant = {"root": r"sinus de valsalva|aortic root|racine", "ascending": r"ascendante|ascending", "arch": r"crosse|aortic arch"}[location]
        # Only the part of the clause naming this segment: "racine aortique dilatée, aorte ascendante normale".
        values = set()
        for part in text.split(","):
            if re.search(relevant, part):
                negations = _negations(r"\bdilat\w*", part)
                if any(negations) or re.search(r"normal", part):
                    values.add("normal")
                if not all(negations):
                    values.add("dilated")
        return values.pop() if len(values) == 1 else None
    return None


POSTERIOR_WALL = "LVIW Diastolic Thickness (2D)"  # the report's measured wall opposite the septum.
WALL_NEGATED = r"(?:\bpas\b|\bsans\b|absence|aucune?|\bno\b|\bnot\b)[^,]*(?:augmentation|epaississement|increased)"
WALL_NORMAL = (r"epaisseure? (?:parietale )?normale|parois?[^,]{0,25}d'epaisseure? normale"
               r"|epaisseure? des parois[^,]{0,40}normale|normal wall thickness")
# "LV wall thickness is normal." takes every wall: not one wall, a borderline value, a contrast or a stated thickening.
WALL_NOT_ALL = (r"limite|\bmais\b|sauf|sept|posterieur|lateral|anterieur|\bbase\b|basal|apex|autres|par ailleurs"
                r"|augmentation|epaississement")
WALL_THICK = r"augmentation(?:(?!masse)[^,])*epaisseur|epaississement[^,]*paroi|increased[^,]*wall thickness"
CONCENTRIC_HYPERTROPHY = r"(?:hypertrophie|hvg)[^,]{0,35}concentrique|concentrique[^,]{0,20}hypertrophie"


def _wall_statement(text):
    """What one clause says about wall thickness: the `septum`, the `posterior` wall or `all` the walls are thick, all
    of them are `normal`, or None: a negated clause, one region of the wall, asymmetric thickening, or a septal bulge,
    which is its own finding."""
    if "bourrelet" in text or re.search(WALL_NEGATED, text):
        return None
    normal = re.search(WALL_NORMAL, text)
    if normal and not re.search(WALL_NOT_ALL, text):
        return "normal"
    if normal or "asymetrique" in text or not re.search(WALL_THICK, text):
        return None
    if re.search(r"symetrique|diffuse", text):
        return "all"
    septum, posterior = bool(re.search(r"septum|septal", text)), bool(re.search(r"posterieure|infero-?laterale", text))
    if septum != posterior:  # "pariétale" with one wall named is every wall, thickest there: not that wall alone.
        return None if re.search(r"parietale|\bparois\b", text) else "septum" if septum else "posterior"
    return None if not septum and re.search(r"apical|apex|inferieure|anterieure|laterale", text) else "all"


def _concentric_hypertrophy(text):
    return bool(re.search(CONCENTRIC_HYPERTROPHY, text)
                and not re.search(r"(?:pas d'|sans |absence d')(?:hvg|hypertrophie)", text))


def _wall_thickness(report, field):
    """The value, source column and clause of the whole report's wall thickness, or None. The reports name the septum
    and the posterior wall separately, so a clause is not enough: "normal elsewhere" needs the posterior wall measured
    and not called thick, and no concentric hypertrophy in the report: statements of the report, not a thickness
    threshold of ours."""
    clauses = list(_clauses(report, field["sources"]))
    found = [(said, source, clause) for source, clause, text in clauses if (said := _wall_statement(text))]
    said = {statement for statement, _, _ in found}
    concentric = any(_concentric_hypertrophy(text) for _, _, text in clauses)
    if said == {"normal"} and not concentric:
        value = "not_increased"
    elif "normal" not in said and ("all" in said or {"septum", "posterior"} <= said):
        value = "increased"
    elif said == {"septum"} and _measured(report.get(POSTERIOR_WALL)) and not concentric:
        value = "septum_only"
    elif said == {"posterior"}:
        value = "posterior_only"
    else:
        return None
    return value, found[0][1], found[0][2]


def _measured(raw):
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError):
        return False
    return number == number and number > 0  # NaN is no measurement, and neither is a placeholder 0.


def build_candidate(report, view, modality, *, global_key, targets=None, level=None, input_kind="video", input_provenance_verified=False, spec=None):
    """A clip's candidate: its prefilled structures (every structure of its view and level unless `targets` names
    them) and the findings the report states explicitly about them. `level` None takes the view's default."""
    spec = spec or load_spec()
    if view not in spec["views"] or modality not in spec["modalities"]:
        raise ValueError("Unknown view or acquisition mode")
    route = spec["views"][view]
    if level is None:
        level = default_level(view, spec)
    elif level not in level_choices(view, spec):
        raise ValueError("Level not allowed for this view")
    suggested_targets = list(targets) if targets is not None else view_targets(view, level, spec)
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
        if key == "lv_wall_thickness":
            wall = _wall_thickness(report, field)
            matches = [wall] if wall and value_allowed(key, wall[0], view, spec) else []
        else:
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


AORTA_SEGMENTS = {"aorta_root_size": "Aortic root", "aorta_ascending_size": "Ascending aorta", "aorta_arch_size": "Aortic arch"}
DEVICES = {"edge_to_edge_clip": "{valve} edge-to-edge clip", "annuloplasty_ring": "{valve} annuloplasty ring",
           "bioprosthesis": "bioprosthetic {valve} valve", "mechanical_prosthesis": "mechanical {valve} valve prosthesis"}
WALL_THICKNESS = {"not_increased": "LV wall thickness is normal.", "increased": "Increased LV wall thickness.",
                  "septum_only": "Thick septum with normal LV wall thickness elsewhere.",
                  "posterior_only": "Thickened LV posterior wall."}
CHAMBERS = {"lv_size": "LV cavity", "rv_size": "RV cavity", "la_size": "Left atrium", "ra_size": "Right atrium", "ivc_size": "IVC"}


def finding_sentence(key, value, spec=None):
    """The caption sentence of one finding, as the medical reviewer worded it: a plain statement, with no "The visible"
    and no "in this input". Every column has its own wording, so that a column added to the specification without one
    is refused here, not captioned from its form labels."""
    spec = spec or load_spec()
    if key in {"lv_function", "rv_function"}:
        chamber = key[:2].upper()
        description = spec["fields"][key]["values"][value].lower()
        return f"{chamber} systolic function appears {description}."
    if key in CHAMBERS:
        return f"{CHAMBERS[key]} is {'not dilated' if value == 'normal' else 'dilated'}."
    if key == "lv_wall_thickness":
        return WALL_THICKNESS[value]
    if key == "septal_bulge":
        return "Focal basal septal thickening (septal bulge)."
    if key == "aortic_cusp_count":
        return f"Aortic valve has {value} cusps."
    if key in AORTA_SEGMENTS:
        return f"{AORTA_SEGMENTS[key]} size is {'not dilated' if value == 'normal' else 'dilated'}."
    if key == "ivc_collapse":
        return f"IVC collapse is {value}."
    if key.endswith("_device"):
        device = DEVICES[value].format(valve=key.split("_")[0])
        return f"{device[0].upper()}{device[1:]} is present."
    if key.endswith("_appearance"):
        valve = key.split("_")[0].capitalize()
        return f"{valve} valve has a {value} appearance." if value == "normal" else f"{valve} valve appears {value}."
    if key.endswith("_motion") and not key.startswith("wall_"):
        valve = key.split("_")[0]
        if value == "prolapse":
            return f"{valve.capitalize()} valve prolapse is present."
        if value == "flail":
            return f"Flail {valve} leaflet is present."
        leaflets = "cusps" if valve in {"aortic", "pulmonic"} else "leaflets"
        return f"{valve.capitalize()} {leaflets} show {value} motion."
    if key.endswith("_regurgitant_jet"):
        valve = key.split("_")[0]
        # What this color clip shows, not whether the patient has regurgitation.
        return f"{valve.capitalize()} regurgitant jet is visible on color Doppler." if value == "demonstrated" else f"No {valve} regurgitant jet is seen on color Doppler."
    if key == "pericardial_effusion":
        return "Pericardial effusion is present." if value == "present" else "No pericardial effusion."
    if key.startswith("wall_"):
        name = key.removeprefix("wall_").replace("_", " ").capitalize()
        motion = {"normal": "normal motion and thickening", "normal_or_hyperkinetic": "normal or hyperkinetic motion", "akinetic_or_severely_hypokinetic": "akinesis or severe hypokinesis", "akinetic": "akinesis", "hypokinetic": "hypokinesis", "severely_hypokinetic": "severe hypokinesis", "hyperkinetic": "hyperkinesis", "dyskinetic": "dyskinesis", "aneurysmal": "aneurysmal deformation"}[value]
        return f"{name} segment shows {motion}."
    raise ValueError(f"No caption sentence for {key}")


def _answer(name, value, children=()):
    return {"name": name, "answer": {"name": value, **({"classifications": list(children)} if children else {})}}


def prelabels(candidate, spec=None):
    """Global video classifications; review_completed is intentionally absent, and so is the view of a class the
    reviewer has to narrow (a PSAX clip's level)."""
    spec = spec or load_spec()
    _validate_candidate(candidate, spec)
    form_revision = hashlib.sha256(
        json.dumps(build_form(spec), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selected = [{"name": c["concept"], "classifications": [
        _answer(c["concept"]+"__value", c["value"]), _answer(c["concept"]+"__status", "yes")]} for c in candidate["columns"]]
    records = [
        _answer("uninterpretable", "no"),
        *([_answer("view", candidate["view"])] if spec["views"][candidate["view"]].get("prefill_view", True) else []),
        _answer("modality", candidate["modality"]), _answer("subcostal_level", candidate["level"]),
        {"name": "visible_targets", "answer": [{"name": t} for t in candidate["suggested_targets"]]},
        {"name": "review_columns", "answer": selected}, _answer("missing_information", "no"),
        _answer("unsupported_information", "no"), {"name": "candidate_caption", "answer": candidate["candidate_caption"]},
    ]
    for record in records:
        record["dataRow"] = {"globalKey": candidate["global_key"]}
        record["uuid"] = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{spec['schema']}:{spec['version']}:{form_revision}:{candidate['global_key']}:{record['name']}",
        ))
    validate_annotations(records, build_form(spec))
    return records


def _validate_candidate(candidate, spec):
    if candidate.get("schema") != spec["schema"] or candidate.get("version") != spec["version"]:
        raise ValueError("Candidate ontology version mismatch")
    if not isinstance(candidate.get("global_key"), str) or not candidate["global_key"].strip():
        raise ValueError("Opaque global_key is required")
    if candidate.get("view") not in spec["views"] or candidate.get("modality") not in spec["modalities"]:
        raise ValueError("Unknown view/modality")
    if candidate.get("level") not in level_choices(candidate["view"], spec):
        raise ValueError("Candidate level not allowed for its view")
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


def encode_answers(answers, form):
    """The nested annotation records of decoded answers on this form: the inverse of `decode_annotations`. An answer
    the selected options do not reach is refused rather than dropped."""
    def records(questions):
        out = []
        for q in questions:
            if q["name"] not in answers:
                continue
            value = answers[q["name"]]
            if q["type"] == "text":
                out.append({"name": q["name"], "answer": value})
                continue
            options = {o["value"]: o for o in q["options"]}
            selected = []
            for chosen in (value if q["type"] == "checklist" else [value]):
                if chosen not in options:
                    raise ValueError(f"{q['name']} has no answer {chosen!r}")
                nested = records(options[chosen]["options"])
                selected.append({"name": chosen, **({"classifications": nested} if nested else {})})
            out.append({"name": q["name"], "answer": selected if q["type"] == "checklist" else selected[0]})
        return out
    encoded = records(form["classifications"])
    unreached = set(answers) - set(decode_annotations(encoded))
    if unreached:
        raise ValueError(f"Answers outside the selected options: {sorted(unreached)}")
    return encoded


def _structure_warning(key, spec):
    """What to tell the reviewer of a claim left on a structure they unticked. The form cannot compare two answers, so
    this is read from the review afterwards, not shown at submission."""
    field = spec["fields"][key]
    return (f"{spec['targets'][field['target']]} is marked as not visible, but \"{field['label']}\" is still claimed. "
            "Mark the finding Incorrect and remove the claim, or tick the structure again.")


def score_review(candidate, answers, *, submitted=False, spec=None):
    spec = spec or load_spec()
    _validate_candidate(candidate, spec)
    original = {c["concept"]: c for c in candidate["columns"]}
    completed = submitted is True and answers.get("review_completed") == "yes"
    if answers.get("uninterpretable") == "yes":
        # Nothing else is read: the clip counts in no column rate and gets no caption.
        reason = answers.get("uninterpretable_reason")
        explained = reason != "other" or str(answers.get("comments", "")).strip()
        pending = [] if reason in spec["uninterpretable_reasons"] and explained else ["uninterpretable_reason"]
        return dict(uninterpretable=True, uninterpretable_reason=reason, supported=[], rejected=[], cannot_assess=[],
                    out_of_view=[], corrections=[], omissions=[], errors=[], pending=pending, warnings=[],
                    original_count=len(original),
                    resolved_count=0, support_rate=None, error_rate=None, view_correct=None,
                    modality_correct=None, routing=False, submitted=completed,
                    caption="", ready_for_validation=False)
    view, level = answers.get("view"), answers.get("subcostal_level")
    refined = refines(view, candidate["view"], spec)
    # A reviewer who narrows the view (PLAX_ZOOM to PLAX_ZOOM_AV) does not have to reject what the classifier's broader
    # class prefilled: the findings the view does not claim leave the candidate, neither confirmed nor rejected.
    out_of_view = [c for key, c in original.items() if refined and key not in spec["views"][view]["columns"]]
    original = {key: c for key, c in original.items() if c not in out_of_view}
    ticked = answers.get("review_columns", [])
    if not set(original) <= set(ticked):
        raise ValueError("A prefilled finding was unticked: mark it Incorrect and remove it instead")
    # Labelbox shows every column in the prefilled list, so a reviewer can tick one that was not prefilled: it is an
    # addition, counted like a confirmed missing finding.
    added = [key for key in ticked if key not in original and key not in {c["concept"] for c in out_of_view}]
    supported, rejected, unresolved, corrections, omissions, errors, pending = [], [], [], [], [], [], []
    if answers.get("uninterpretable") != "no":
        pending.append("uninterpretable")
    if level not in level_choices(view, spec):
        pending.append("subcostal_level")
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
    for key in added:
        value = answers.get(key+"__value")
        if answers.get(key+"__status") == "yes" and value_allowed(key, value, answers.get("view"), spec):
            omissions.append(dict(concept=key, value=value, origin="reviewer_addition"))
        else:
            pending.append(key+"__status")
    if answers.get("missing_information") == "yes":
        keys = answers.get("missing_columns", [])
        if not keys or len(keys) != len(set(keys)) or set(keys) & set(added):
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
                if key in ticked and answers.get(key+"__status") == "yes":
                    pending.append(key+"__status_conflict")
    elif answers.get("unsupported_information") != "no":
        pending.append("unsupported_information")
    actual = supported + corrections + omissions
    structures = set(answers.get("visible_targets", []))
    if view in spec["views"]:
        # The structures not prefilled for the candidate's view that its confirmed view or level shows (a PSAX level's,
        # an IVC sweep's) come with the answer: the reviewer never adds a structure.
        own = spec["views"][candidate["view"]]["targets"]
        structures |= {t for t in view_targets(view, level, spec) if t not in own and t not in candidate["suggested_targets"]}
    allowed = allowed_columns(view, answers.get("modality"), targets=structures, level=level, input_kind=candidate.get("input_kind", "video"), spec=spec)
    # A finding the reviewer adds must be one this view and these structures carry: it is sent back to be corrected, not
    # left to cost the clip its caption without a word.
    pending += [c["concept"]+"__not_in_view" for c in omissions if c["concept"] not in allowed]
    # A structure the reviewer unticked cannot keep a claim: the review says both. The answer is not rewritten to settle
    # it, which would change the reviewer's judgment; it goes back to be resolved. Only what was unticked counts: a
    # corrected view leaves its other findings judged, without a caption.
    conflicts = [c["concept"] for c in supported + corrections if spec["fields"][c["concept"]]["target"] not in structures]
    pending += [key+"__structure_unticked" for key in conflicts]
    view_correct = view == candidate["view"] or refined
    modality_correct = answers.get("modality") == candidate["modality"]
    routing = (view_correct and modality_correct
               and candidate["view"] != "OTHER"
               and all(c["concept"] in allowed and value_allowed(c["concept"], c["value"], view, spec) for c in actual))
    caption = " ".join(finding_sentence(c["concept"], c["value"], spec) for c in actual) if completed and routing and not pending else ""
    resolved = len(supported) + len(rejected)
    return dict(uninterpretable=False, uninterpretable_reason=None, supported=supported, rejected=rejected,
                cannot_assess=unresolved, out_of_view=out_of_view, corrections=corrections, omissions=omissions,
                errors=errors, pending=pending, warnings=[_structure_warning(key, spec) for key in conflicts],
                original_count=len(original), resolved_count=resolved,
                support_rate=len(supported)/len(original) if original else None,
                error_rate=len(rejected)/resolved if resolved else None,
                view_correct=view_correct, modality_correct=modality_correct,
                routing=routing, submitted=completed, caption=caption,
                ready_for_validation=bool(caption) and candidate.get("input_provenance_verified") is True)


def review_problems(candidate, annotations, spec=None):
    """Why one submitted review cannot be counted; empty when it is valid. Valid reviews are the population of the
    pilot metrics: every required answer in its place in the form, no detail left pending, and nothing that
    contradicts the candidate. The original candidate view and mode remain available for comparison with the
    reviewer-confirmed values."""
    spec = spec or load_spec()
    try:
        validate_annotations(annotations, build_form(spec), require_complete=True)
        answers = decode_annotations(annotations)
        score = score_review(candidate, answers, submitted=True, spec=spec)
    except ValueError as e:
        return [str(e)]
    problems = list(score["pending"])
    return problems


def numeric_context(report_context):
    """The report context values that are finite numbers (EF, dimensions, codes); free text is left out."""
    out = {}
    for key, value in report_context.items():
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number == number and abs(number) != float("inf"):
            out[key] = int(number) if number.is_integer() else number
    return out


def context_attachment(candidate, spec=None):
    """The text uploaded next to each video. It holds no report text: each proposal shows its report field's name,
    not the clause it came from, and the report context keeps only numbers. The source text stays in the local
    candidate."""
    spec = spec or load_spec()
    lines = [f"DeepECHO column review v{spec['version']} — original, unverified proposals",
             f"View: {candidate['view']}; acquisition: {candidate['modality']}",
             "Confirm values only in the displayed input. Report numbers are study context."]
    for c in candidate["columns"]:
        lines.append(f"{c['concept']} | {spec['fields'][c['concept']]['values'][c['value']]} | "
                     f"{c.get('source_column', 'candidate')}")
    if candidate.get("report_context"):
        allowed = {source for f in spec["fields"].values() for source in f["sources"]+f.get("context_columns", [])}
        if any(key not in allowed for key in candidate["report_context"]):
            raise ValueError("Report context contains a column outside the clinical allowlist")
        numbers = numeric_context(candidate["report_context"])
        if numbers:
            lines.append("Report numbers (not local claims):")
            lines.extend(f"{key}: {value}" for key, value in numbers.items())
    lines += ["Original caption: "+candidate["candidate_caption"], "Suggestions require submitted human review."]
    return dict(global_key=candidate["global_key"], attachment_type="RAW_TEXT",
                attachment_name=f"DeepECHO review v{spec['version']} - original proposals",
                attachment_value="\n".join(lines))


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
            raise SystemExit(f"Committed v{load_spec()['version']} form is out of date")
    else:
        FORM.write_text(json.dumps(form, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
