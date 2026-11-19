# data/build_radar_captions_merged.py

"""RADAR caption table, second version: English sentences from the column review, checked and completed by the parser.

`build_radar_captions.py` keeps the report's own (French) text. This builder writes each (study, structure) caption
as short English sentences, one per finding, from four sources:

1. The column review prefill (`tools/labelbox/column_review.py`, form v5, on the column review branch): its findings
   for the views of the study's clips that may show the structure, worded by its `finding_sentence`. A finding is
   dropped when the report's other source contradicts it or the RADAR parser reads the text the other way: exactly
   the checks of `compare_column_prefill.py`, always with its strict rule for an LV function next to normal.
2. The RADAR parser (`radar_normality.py`), where the prefill says nothing of a column and the parser decides: the
   column's normal sentence, or a plain abnormal one (`ABNORMAL_FILL`), since the parser keeps the side of a finding
   and not its grade. A dilated chamber is written only when the parser named a dilatation, except the left atrium
   (`DILATED_IF_ABNORMAL`): its size is abnormal in these reports only as an increase. Nothing is filled for a
   device, a valve's motion, the wall thickness, the cusp count or an aorta segment on the abnormal side.
3. The RADAR parser again, for what the form has no column for (`EXTRA_SENTENCES`): LV hypertrophy stated or denied,
   LV remodelling, the LV mass (normal or increased), and the right atrial pressure worded with the vena cava (normal
   or elevated). Without them an LV caption says nothing of hypertrophy, and an IVC caption nothing of the pressure. A
   sentence is written only when every text claim of that kind agrees and none is uncertain; measurements write none.
   For a valve, also the procedures the form has no value for (`procedure_sentences`): a homograft the parser reads as
   a device, and a surgical repair in the reports' fixed wording (`statut post-plastie tricuspidienne chirurgicale`),
   written on the device side. Without them a valve whose cell states only such a procedure has no caption.
4. The visual EF, for the LV function only, when the prefill says nothing: the grade of the EF's band in this lab
   (`configs/radar/prefill_check_limits_v0.yaml`), unless the EF is in a grey zone or the parser reads the text the
   other way. The EF number itself is not written, except with item 7.
5. Only with `--regurgitation-default-from YEAR` (off by default): `No tricuspid regurgitation.` or `No pulmonic
   regurgitation.` for a valve left without a caption, when no source says anything of it (`REGURGITATION_DEFAULT`).
6. Only with `--regurgitation-grade-valves VALVE ...` (off by default): the structured regurgitation grade, in the
   ontology's words, for a valve whose study has a routed colour-flow clip, in a second row of `scope` `color_flow`
   (`regurgitation_grade`). B-mode rows are never changed: a regurgitation jet is seen on colour Doppler only.
7. Only with `--ef-sentence` (off by default): the LV ejection fraction as a number, `LV ejection fraction is 45%.`
   (`ef_entry`), right after the LV function sentence, which is kept. The EF stated in the report text is preferred:
   the LV cell's retained text, else the conclusion, read by `stated_ef` (a point, a closed interval's midpoint);
   else the structured visual EF. The sentence is on no side (`EF_SIDE`): the row's flag is the flag
   without it. Its source is `ef_text`, `ef_conclusion` or `ef_structured`.

The 16 wall segments are one sentence when all are normal, and one sentence per abnormal segment otherwise. A caption
is the union over the views of every eligible clip of the study: an upper bound for a sampled bag, as in the
comparison tool.

`normality` is `abnormal` when a kept sentence is abnormal or names a device, `normal` when every sentence is normal
and the parser's own flag for the report text is normal, else `unknown`; the EF sentence (item 7) is left out of it.
It is PROVISIONAL (`normality_status`), so the loss arms that read it (A2, A3) refuse this table. `sources` lists,
per sentence, the column, the value and where it came from (prefill_text, prefill_code, parser, ef_band, default,
grade, ef_text, ef_conclusion, ef_structured): categories and the EF value only, no report text. `dropped.csv` lists
the prefill findings left out and why.

The table and `build_info.json` have the layout `evals/radar_frozen/train.py` checks. Outputs go outside the
repository. Nothing printed holds report text; counts from 1 to 4 print as `<5`.

Example:
    python data/build_radar_captions_merged.py --column-review-root <column review checkout> --reports <reports> \\
        --pacs <pacs index> --clips <clip table> --splits <cohort studies.csv> --out-dir <folder>
"""

import argparse
import dataclasses
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter

import compare_column_prefill as ccp
import pandas as pd
import radar_captions as rc
import radar_normality as rn

COLUMNS = ("study_id", "structure", "caption", "caption_id", "normality", "normality_status", "policy_digest",
           "sentences", "from_prefill", "from_parser", "from_ef", "sources")
DROPPED_COLUMNS = ("study_id", "structure", "column", "value", "reason")
DROPPED_CHECKS = ("contradiction", "off_by_one_across_normal")
WALLS_NORMAL = "LV regional wall motion is normal."
WALLS_ABNORMAL = "LV regional wall motion abnormality is present."
WORDING = {  # a structure-level wording where the column review's sentence speaks of one clip
    ("pericardial_effusion", "present"): "Pericardial effusion is present.",
    ("pericardial_effusion", "not_demonstrated"): "No pericardial effusion.",
}
ABNORMAL_FILL = {  # a parser-only abnormal finding: the column's value when it has one abnormal value, else a sentence
    "lv_function": "LV systolic function appears reduced.",
    "rv_function": "RV systolic function appears reduced.",
    "pericardial_effusion": "present",
    "ivc_collapse": "reduced",
    "mitral_appearance": "Mitral valve appears abnormal.",
    "aortic_appearance": "Aortic valve appears abnormal.",
    "tricuspid_appearance": "Tricuspid valve appears abnormal.",
    "pulmonic_appearance": "Pulmonic valve appears abnormal.",
    ccp.WALL_COLUMN: WALLS_ABNORMAL,
}
DILATED_IF_NAMED = ("lv_size", "rv_size", "la_size", "ivc_size")    # filled as dilated only when the parser said so
# A size the reports call abnormal only as an increase (`volume indexé légèrement augmenté`): dilated without the word.
# Checked on the training split: every left atrium caption with an abnormal size state says `augmenté`, none smaller.
DILATED_IF_ABNORMAL = ("la_size",)
# Parser findings the form has no column for: structure -> (name, parser attribute, parser findings, {value: sentence}).
# Unqualified hypertrophy is no wall thickness column value (`ccp.speaks_for_column`), so it is worded as itself. An
# abnormal mass state is an increase in these reports (checked on the training split, as above).
EXTRA_SENTENCES = {
    "lv": (
        ("lv_hypertrophy", "lv_wall_thickness", ("hypertrophy", "concentric_hypertrophy"),
         {"abnormal": "LV hypertrophy is present.", "normal": "No LV hypertrophy."}),
        ("lv_remodelling", "lv_geometry", ("remodelling",), {"abnormal": "LV remodelling is present."}),
        ("lv_mass", "lv_mass", ("state",), {"normal": "LV mass is normal.", "abnormal": "LV mass is increased."}),
    ),
    # The right atrial pressure the report words next to the vena cava. Abnormal is elevated: on the training split
    # every study with such a statement has a recorded pressure of 8 mmHg or more.
    "ivc": (
        ("ra_pressure", "ivc_pressure", ("state",), {"normal": "Estimated right atrial pressure is normal.",
                                                     "abnormal": "Estimated right atrial pressure is elevated."}),
    ),
}
# Valve procedures the form has no value for, read in the valve's own cell after the Doppler removal: a homograft,
# which the parser takes for a device without saying which (so the parser fill writes nothing for it), and a
# surgical repair in the reports' fixed wording, which neither reader knows. Both are written as devices are, on the
# device side. A homograft needs the parser's device status too, so that a denied one writes nothing; a repair
# clause must hold no negation and no hedge.
VALVE_WORDS = {"mitral": "mitrale", "aortic": "aortique", "tricuspid": "tricuspidienne", "pulmonic": "pulmonaire"}
HOMOGRAFT = re.compile(r"\bhomogreffes?\b")
REPAIR = r"\bstatut post-?plastie {valve} chirurgicale\b"
# An option (`--regurgitation-default-from`), off by default: a tricuspid or pulmonic valve with no caption is
# captioned as having no regurgitation when nothing in the report says otherwise. That is when the study is dated
# from the given year on (the valve cells are filled from 2020; an earlier empty cell says nothing), the valve's own
# cell is empty (a cell whose text was all removed as Doppler stated something, mostly a regurgitation), the
# structured grade is missing or 0 (any other grade states a jet, trace (0.5) included, and an unreadable one is not
# taken for 0), and the conclusion does not name the valve. The sentence is on the normal side, but the row's flag
# stays `unknown` (no text was read), and its source is `default`. Grade columns: 0 none, 0.5 trace, 1 mild,
# 2 moderate, 3 moderately severe, 4 severe.
YEAR_COLUMN, CONCLUSION_COLUMN = "StudyDate_Year", "Report conclusions findings"
REGURGITATION_DEFAULT = {  # structure: (structured grade column, the valve named in the conclusion)
    "tricuspid": ("Referring TR Grade", re.compile(r"\btricuspidien\w*|\bvalv(?:e|ule)s? tricuspides?\b|\bit\b")),
    "pulmonic": ("Referring PR Grade", re.compile(
        r"\b(?:valv(?:e|ule)s?|insuffisance|fuite|stenose|regurgitation) pulmonaires?\b|\bpulmoniques?\b"
        r"|\bip\b|\bvp\b")),
}
TRACE_GRADE = 0.5
# An option (`--regurgitation-grade-valves`), off by default: the valve's structured grade as a sentence, written only
# in a row of scope `color_flow` and only when the study has a routed colour-flow clip of the valve. A training bag
# must hold such a clip for the row to be read (see `regurgitation_grade`). Codes as above; the words are the
# ontology's (configs/radar/radar-caption-ontology-v2.yaml, scale `regurgitation`, English labels), and only the
# codes listed are read. The ontology itself still leaves the numeric codes unmapped (its v1 note): this mapping is
# the one the report text agrees with on the training split.
GRADE_COLUMNS = {"mitral": "Referring MR Grade", "aortic": "Referring AR Grade",
                 "tricuspid": "Referring TR Grade", "pulmonic": "Referring PR Grade"}
GRADE_CODES = {0.0: "none", 0.5: "trace", 1.0: "mild", 2.0: "moderate", 3.0: "moderate_to_severe", 4.0: "severe"}
GRADE_LABELS = {"none": "none", "trace": "trace", "mild": "mild", "mild_to_moderate": "mild to moderate",
                "moderate": "moderate", "moderate_to_severe": "moderate to severe", "severe": "severe"}
NORMAL_GRADES = ("none", "trace")      # the sentence's side; any other grade is on the abnormal side
COLOR_FLOW = "color_flow"              # the clip table's acquisition class, and the scope of the graded rows
# How the report text states a valve's regurgitation (`text_regurgitation`), on plain text (`rn._plain`): a valve's
# name or abbreviation, a regurgitation word, the ontology's grade aliases (composite grades first), a negation of the
# regurgitation, and grade words the ontology has no value for (read as uncertain, never as a grade).
VALVE_NAMES = {
    "mitral": re.compile(r"\bmitral\w*|\bim\b"),
    "aortic": re.compile(r"\b(?:valv(?:e|ule)s?|insuffisance|fuite|regurgitation|stenose|(?:bio)?prothese)"
                         r" aortiques?\b|\bia\b|\biao\b"),
    "tricuspid": re.compile(r"\btricuspidien\w*|\bvalv(?:e|ule)s? tricuspides?\b|\bit\b"),
    "pulmonic": re.compile(r"\b(?:valv(?:e|ule)s?|insuffisance|fuite|stenose|regurgitation) pulmonaires?\b"
                           r"|\bpulmoniques?\b|\bip\b|\bvp\b"),
}
_REGURGITATION_WORD = r"insuffisance\w*|insuffisante?s?\b|fuites?\b|regurgit\w*|(?:im|ia|iao|it|ip)\b"
REGURGITATION_WORD = re.compile(rf"\b(?:{_REGURGITATION_WORD})")
REGURGITATION_DENIED = re.compile(
    rf"\b(?:pas|absence|sans|aucune?|ni)\b(?:\s+[\w']+){{0,3}}?\s+(?:{_REGURGITATION_WORD})"
    rf"|\b(?:{_REGURGITATION_WORD})(?:\s+[\w']+){{0,2}}?\s*:?\s*\b(?:aucune?|absente?s?|nulle?s?)\b")
GRADE_PATTERNS = (
    ("moderate_to_severe", re.compile(r"\bmoderee?s? a severes?\b|\bmoderee?s?-severes?\b|\bmoderement severes?\b")),
    ("mild_to_moderate", re.compile(r"\blegere?s? a moderee?s?\b|\blegere?s?-moderee?s?\b")),
    ("severe", re.compile(r"\bseveres?\b|\bimportante?s?\b|\bmassive?s?\b")),
    ("moderate", re.compile(r"\bmoderee?s?\b")),
    ("mild", re.compile(r"\blegere?s?\b")),
    ("trace", re.compile(r"\btraces?\b|\bminimes?\b|\btriviale?s?\b|\bphysiologiques?\b")),
)
UNMAPPED_GRADE = re.compile(r"\blegerement\b|\bdiscrete?s?\b|\bsignificative?s?\b|\bpetite?s?\b|\bmodeste?s?\b"
                            r"|\bminimale?s?\b|\bgrades?\b|\b\d(?:[.,]\d)? ?/ ?4\b|\bvoire\b|\bou\b")
TRACE_WORD = re.compile(r"\btraces?\b")     # a grade here, a hedge to the parser (`rn._HEDGE`)
# Segments of a clause: a comma that leads to an alternative (`, voire modérée`) does not end one.
SEGMENT_SPLIT = re.compile(r",(?!\d)(?!\s*(?:voire|ou)\b)|\bet\b|\bavec\b|\bainsi que\b")
# An option (`--ef-sentence`), off by default: the LV row's EF as a number (`ef_entry`). Its key sorts right after the
# LV function column; its side is none of normal, abnormal or device, so the row's flag never reads it.
EF_COLUMN, EF_AFTER, EF_SIDE = "lv_ef", "lv_function", "value"
EF_TEXT_READS = ("ef_text", "ef_conclusion")     # sources of a text EF: the LV cell, then the conclusion
EF_SENTENCE = "LV ejection fraction is {}%."     # an integer; `src/models/radar/loss.py` reads it back (soft EF)
EF_SPREAD = 5.0                                  # text statements further apart than this give no value
# An EF labelled with another method than the visual estimate, in the words between the EF label and its value
# (`calculée à`, `calculée par méthode Simpson`, 3D): left out of the text reading, as a strain value is.
OTHER_METHOD = re.compile(r"simpson|biplan|\b[23]\s?d\b|teichholz|quinones|\bcalcul|\bmesur", re.IGNORECASE)


def sentence(cr, spec, column, value):
    return WORDING.get((column, value)) or cr.finding_sentence(column, value, spec)


def normal_value(spec, column):
    """The column's normal value, or None when it has none (a device, a finding that is only ever present)."""
    return next((v for v in spec["fields"][column]["values"] if ccp.POLARITY[v] == "normal"), None)


def extra_sentences(facts, name):
    """{name: (sentence, side)} of the parser's text claims the form has no column for (`EXTRA_SENTENCES`). A kind
    whose claims disagree, or hold an uncertain one, writes nothing."""
    found = {}
    for extra, attribute, findings, wording in EXTRA_SENTENCES.get(name, ()):
        values = {f["value"] for f in facts if f["source"] == "text" and f["attribute"] == attribute
                  and f.get("finding") in findings}
        if len(values) == 1 and next(iter(values)) in wording:
            side = next(iter(values))
            found[extra] = (wording[side], side)
    return found


def procedure_sentences(caption, statuses, name):
    """{name: (sentence, side)} of the valve procedures (`VALVE_WORDS`) a structure's retained report text states:
    `caption` is that text (the parser's caption), `statuses` the parser's statuses of it."""
    if name not in VALVE_WORDS:
        return {}
    valve, found = name.capitalize(), {}
    clauses = [rn._plain(clause) for clause in rc.text_clauses(caption or "")]
    if statuses.get(f"{name}_device") == "device" and any(HOMOGRAFT.search(c) for c in clauses):
        found[f"{name}_homograft"] = (f"{valve} valve homograft is present.", "device")
    repair = re.compile(REPAIR.format(valve=VALVE_WORDS[name]))
    if any(repair.search(c) and not rn._NEGATION.search(c) and not rn._HEDGE.search(c) for c in clauses):
        found[f"{name}_surgical_repair"] = (f"Status post surgical {name} valve repair.", "device")
    return found


def number(value):
    """A report cell as a float, NaN when it holds no number."""
    return pd.to_numeric(pd.Series([value], dtype=object), errors="coerce").iloc[0]


def regurgitation_default(report, structure, from_year):
    """Why a (study, valve) without a caption gets no default sentence (`REGURGITATION_DEFAULT`), or None when it
    gets one."""
    if structure.name not in REGURGITATION_DEFAULT:
        return "not_a_default_valve"
    grade_column, named = REGURGITATION_DEFAULT[structure.name]
    year = number(report.get(YEAR_COLUMN))
    if pd.isna(year):
        return "no_year"
    if year < from_year:
        return "before_year"
    if rc.caption_text(report.get(structure.report_column)):
        return "cell_filled"
    raw = report.get(grade_column)
    if not (raw is None or pd.isna(raw) or not str(raw).strip()):
        grade = number(str(raw).strip())
        if pd.isna(grade):
            return "grade_unreadable"
        if grade != 0:
            return "grade_trace" if grade == TRACE_GRADE else "grade_regurgitation"
    if named.search(rn._plain(rc.caption_text(report.get(CONCLUSION_COLUMN)))):
        return "named_in_conclusion"
    return None


def doubtful(segment):
    """Whether a clause segment (plain text) is hedged, not assessed, compared with an earlier study, or uses a grade
    word the ontology has no value for (`UNMAPPED_GRADE`). A trace is a grade here, not the parser's hedge."""
    return bool(UNMAPPED_GRADE.search(segment) or rn._HEDGE.search(TRACE_WORD.sub(" ", segment))
                or rn._NOT_ASSESSED.search(segment) or rn._PRIOR.search(segment))


def segment_regurgitation(segment, names_valve, clause_states):
    """What one clause segment (plain text) states of a regurgitation: None (nothing), `uncertain`, `ungraded` (a
    regurgitation without a grade) or an ontology grade (`none` for a denied one). A segment counts when it has a
    regurgitation word or a denial, or a grade word and the valve's name in a clause with a regurgitation word."""
    grades, rest = [], segment
    for grade, pattern in GRADE_PATTERNS:
        if pattern.search(rest):
            grades.append(grade)
            rest = pattern.sub(" ", rest)
    denied = bool(REGURGITATION_DENIED.search(segment))
    if not (REGURGITATION_WORD.search(segment) or denied or (grades and names_valve and clause_states)):
        return None
    if doubtful(segment):
        return "uncertain"
    if denied:
        return "uncertain" if grades else "none"
    return grades[0] if len(grades) == 1 else "uncertain" if grades else "ungraded"


def text_regurgitation(report, structure):
    """(grades, ungraded, uncertain) the report text states of the valve's regurgitation: the set of ontology grades
    (`none` for a denial), whether it states one without a grade, and whether a statement is doubtful (`doubtful`) or
    names two grades or two valves at once.

    Read are the valve's own cell (`structure.report_column`, before any Doppler removal) and the conclusion. Clauses
    (`rc.text_clauses`) are cut into segments (`SEGMENT_SPLIT`: commas, `et`, `avec`, `ainsi que`). A segment naming
    the valve is read; one naming only another valve is not; one naming no valve is read in the valve's own cell
    when its clause names no other valve, and never in the conclusion. A segment naming the valve and another one is
    uncertain, and so is a statement whose clause has a doubtful segment that is read (`IT légère, modérée à l'examen
    précédent`)."""
    name, grades, ungraded, uncertain = structure.name, set(), False, False
    others = [pattern for valve, pattern in VALVE_NAMES.items() if valve != name]
    for column, own in ((structure.report_column, True), (CONCLUSION_COLUMN, False)):
        for clause in rc.text_clauses(report.get(column)):
            plain = rn._plain(clause)
            states = bool(REGURGITATION_WORD.search(plain) or REGURGITATION_DENIED.search(plain))
            if not states:
                continue
            clause_other = any(p.search(plain) for p in others)
            readings, doubt = [], False
            for segment in SEGMENT_SPLIT.split(plain):
                this = bool(VALVE_NAMES[name].search(segment))
                other = any(p.search(segment) for p in others)
                if (other and not this) or (not this and (not own or clause_other)):
                    continue
                doubt = doubt or doubtful(segment)
                readings.append("uncertain" if this and other else segment_regurgitation(segment, this, states))
            readings = [r for r in readings if r is not None]
            if readings and doubt:
                readings = ["uncertain"]
            for reading in readings:
                if reading == "uncertain":
                    uncertain = True
                elif reading == "ungraded":
                    ungraded = True
                else:
                    grades.add(reading)
    return grades, ungraded, uncertain


def structured_grade(report, name):
    """(ontology grade or None, why not) of the valve's structured grade column (`GRADE_COLUMNS`)."""
    raw = report.get(GRADE_COLUMNS[name])
    text = "" if raw is None or (not isinstance(raw, str) and pd.isna(raw)) else str(raw).strip()
    if not text:
        return None, "grade_missing"
    code = number(text)
    grade = None if pd.isna(code) else GRADE_CODES.get(float(code))
    return (grade, None) if grade is not None else (None, "grade_unreadable")


def grade_sentence(name, grade):
    """The ontology's words for a grade of the valve's regurgitation: `No`, `Trace`, `Mild to moderate`..."""
    words = "No" if grade == "none" else GRADE_LABELS[grade].capitalize()
    return f"{words} {name} regurgitation."


def regurgitation_grade(report, structure, has_color):
    """(reason, sentence entry or None) of the graded regurgitation sentence of one (study, valve) (`GRADE_COLUMNS`).

    Written (`written`, entry = (sentence, side, grade)) only when, in this order: the study has at least one
    colour-flow clip routed to the valve (`no_color_clip`); the structured grade is filled (`grade_missing`) and one of
    `GRADE_CODES` (`grade_unreadable`); the text (`text_regurgitation`) holds no uncertain statement
    (`text_uncertain`); the text agrees with the grade (`text_conflict`); and, for a trace, the text states a trace
    (`trace_not_in_text`).

    Agree: every grade the text states is the structured grade after mapping both to the ontology (`mild_to_moderate`
    in the text against code 1 or 2 is a conflict, as are two grades), and a regurgitation stated without a grade
    conflicts with `none` only. Text that states nothing is no conflict, except for a trace, which is written only
    when the text says trace too: a code 0.5 is the most common code and is often not in the text."""
    if not has_color:
        return "no_color_clip", None
    grade, why = structured_grade(report, structure.name)
    if grade is None:
        return why, None
    grades, ungraded, uncertain = text_regurgitation(report, structure)
    if uncertain:
        return "text_uncertain", None
    if grades - {grade} or (ungraded and grade == "none"):
        return "text_conflict", None
    if grade == "trace" and grades != {"trace"}:
        return "trace_not_in_text", None
    side = "normal" if grade in NORMAL_GRADES else "abnormal"
    return "written", (grade_sentence(structure.name, grade), side, grade)


def stated_ef(text, spread=EF_SPREAD):
    """(value, status, statements) of the EF a report text states, by `radar_captions` (`find_ef`, clause by clause,
    its clause-wide hedge rule included). `statements`: those read, without those labelled with another method than
    the visual estimate (`OTHER_METHOD`), which count as unstated. `status`: `value` (a point, a closed interval's
    midpoint, or the mean of the midpoints of statements no more than `spread` points apart), `uncertain` (a hedge),
    `bound` (an open or one-sided range), `disagree` (statements further apart), `other_method` (only such
    statements), `unread` (EF wording whose value the reader does not take, `estimée à` after a comma) or `none`.
    `value` is None unless the status is `value`."""
    text = unicodedata.normalize("NFC", rc.caption_text(text)).translate(rc._APOSTROPHES)
    statements, other = [], False
    for clause in rc._CLAUSE_SPLIT.split(text):
        found = []
        for match, statement in rc.ef_expressions(clause):
            if OTHER_METHOD.search(match.group("gap")):
                other = True
            else:
                found.append(statement)
        if found and rc._qualified(rc._EF.sub(" ", clause)):
            found = [dataclasses.replace(st, certain=False) for st in found]
        statements.extend(found)
    if not statements:
        return None, "other_method" if other else "unread" if rc._EF_UNREAD.search(text) else "none", []
    if not all(s.certain for s in statements):
        return None, "uncertain", statements
    if any(s.low_open or s.high_open or (s.low != s.high and (s.low == rc.EF_RANGE[0] or s.high == rc.EF_RANGE[1]))
           for s in statements):
        return None, "bound", statements
    mids = [(s.low + s.high) / 2.0 for s in statements]
    if max(mids) - min(mids) > spread:
        return None, "disagree", statements
    return sum(mids) / len(mids), "value", statements


def ef_integer(value):
    """An EF rounded half up (57.5 -> 58; a stored 64.9999976 -> 65)."""
    return int(math.floor(float(value) + 0.5))


def ef_entry(report, lv_text, policy):
    """(sentence entry or None, reason) of the LV row's EF sentence (`--ef-sentence`). `lv_text`: the LV cell's
    retained text (the parser's caption). Entry = (sentence, side `EF_SIDE`, integer value, source).

    The text is read first (`stated_ef`): the LV cell, then, when the cell states no EF of its own (no statement,
    only one of another method, or wording the reader does not take), the conclusion. A text value is written
    (`ef_text`, `ef_conclusion`). A statement without a value (a hedge, a bound, two values more than 5 points apart)
    stops the reading there: the structured visual EF (`policy.ef`: column, valid range) is written only when it lies
    within `policy.ef["tolerance"]` points of every such statement, else nothing (`text_conflict`). Otherwise the
    structured EF (`ef_structured`), else nothing (`no_ef`). The reason of a structured EF names the status of each
    text read (`ef_structured:lv_cell=unread,conclusion=none`)."""
    structured = rc.finite(report.get(policy.ef["column"]))
    low, high = policy.ef["valid"]
    if structured is not None and not low <= structured <= high:
        structured = None
    tolerance, read = float(policy.ef["tolerance"]), []
    for text, source, place in ((lv_text, EF_TEXT_READS[0], "lv_cell"),
                                (report.get(CONCLUSION_COLUMN), EF_TEXT_READS[1], "conclusion")):
        value, status, statements = stated_ef(text)
        read.append(f"{place}={status}")
        if status == "value":
            return (EF_SENTENCE.format(ef_integer(value)), EF_SIDE, ef_integer(value), source), source
        if statements:
            if structured is not None and all(s.low - tolerance <= structured <= s.high + tolerance
                                              for s in statements):
                break
            return None, "text_conflict"
    if structured is None:
        return None, "no_ef"
    value = ef_integer(structured)
    return (EF_SENTENCE.format(value), EF_SIDE, value, "ef_structured"), "ef_structured:" + ",".join(read)


def segment_values(report, views, name, acquisition, spec, cr, cache):
    """{wall segment column: value} of the prefill for the structure's views; a segment the views disagree on is left
    out. Uses the candidates `ccp.prefill_columns` cached."""
    found = {}
    for view in sorted(views):
        candidate = cache[(view, ccp.level_for(view, name, spec, cr))]
        for column in candidate["columns"]:
            concept = column["concept"]
            if concept.startswith("wall_") and ccp.PREFILL_TARGETS.get(spec["fields"][concept]["target"]) == name:
                found.setdefault(concept, set()).add(column["value"])
    return {concept: next(iter(values)) for concept, values in found.items() if len(values) == 1}


def merged_row(study_id, structure, report, views, policy, spec, cr, limits, acquisition="bmode",
               regurgitation_default_from=None, ef_sentence=False):
    """(caption row or None, dropped prefill findings) of one (study, structure) whose clips have `views`. With
    `regurgitation_default_from` (a year), a valve left without a caption may get the default sentence
    (`REGURGITATION_DEFAULT`); with `ef_sentence`, the LV gets its EF sentence (`ef_entry`)."""
    kept, flag, dropped = merged_sentences(study_id, structure, report, views, policy, spec, cr, limits, acquisition,
                                           regurgitation_default_from, ef_sentence)
    return caption_row(study_id, structure.name, kept, flag, policy, spec), dropped


def merged_sentences(study_id, structure, report, views, policy, spec, cr, limits, acquisition="bmode",
                     regurgitation_default_from=None, ef_sentence=False, ef_reasons=None):
    """(kept sentences {column: (sentences, side, value, source)}, the parser's flag, dropped prefill findings) of one
    (study, structure): what `merged_row` writes. `ef_reasons` (a Counter) counts the LV's `ef_entry` reasons."""
    import build_radar_captions as brc

    name, cache = structure.name, {}
    prefill = ccp.prefill_columns(report, views, name, acquisition, spec, cr, cache)
    without_ef = {**report, policy.ef["column"]: None}
    row = brc.caption_row(study_id, structure, without_ef, policy)
    if row is None:
        statuses, own, findings, flag, facts = {}, {}, {}, "unknown", []
        silent = "removed" if brc.exclusion_row(study_id, structure, report) is not None else "empty"
    else:
        covered = {ccp.parser_attribute(c) for c in ccp.COLUMNS_OF.get(name, ())} - {None}
        facts = json.loads(row["facts"])
        statuses, own, findings = ccp.parser_reading(facts, covered | {"aorta_size"})
        silent, flag = "not_stated", row["normality"]

    kept, dropped = {}, []      # kept: column -> (sentences, side, value, source)
    for column, (value, source) in prefill.items():
        attribute = ccp.parser_attribute(column)
        status = ccp.parser_status(statuses, attribute, silent) if attribute else "no_attribute"
        compared = ccp.outcome(value, status, findings.get(attribute, ())) if value != "mixed" else "mixed"
        check = ccp.structured_check(column, value, source, report, own, limits, strict_normal=True)
        reason = ("mixed" if compared == "mixed" else check if check in DROPPED_CHECKS
                  else compared if compared in ccp.DROPPED_BY_PARSER else None)
        if reason:
            dropped.append({"study_id": study_id, "structure": name, "column": column, "value": value,
                            "reason": reason})
            continue
        if column == ccp.WALL_COLUMN:
            segments = segment_values(report, views, name, acquisition, spec, cr, cache)
            abnormal = [sentence(cr, spec, c, v) for c, v in segments.items() if ccp.POLARITY[v] != "normal"]
            texts = abnormal if value != "normal" and abnormal else [WALLS_NORMAL if value == "normal"
                                                                     else WALLS_ABNORMAL]
        else:
            texts = [sentence(cr, spec, column, value)]
        kept[column] = (texts, ccp.prefill_side(value), value, f"prefill_{source or 'text'}")

    for column in sorted(ccp.COLUMNS_OF.get(name, ())):
        attribute = ccp.parser_attribute(column)
        if column in prefill or not attribute:
            continue
        status = ccp.parser_status(statuses, attribute, silent)
        if column == "lv_function":
            ef = ccp.ef_value(report, limits)
            grades = ccp.ef_grades(ef, limits) if ef is not None else set()
            if len(grades) == 1:
                band = ccp.GRADE_VALUE[next(iter(grades))]
                side = "normal" if band == "normal" else "abnormal"
                if status not in ccp.DECIDED or status == side:
                    kept[column] = ([sentence(cr, spec, column, band)], side, band, "ef_band")
                continue
        if status == "normal":
            value = normal_value(spec, column) if column != ccp.WALL_COLUMN else "normal"
            if value is not None:
                text = WALLS_NORMAL if column == ccp.WALL_COLUMN else sentence(cr, spec, column, value)
                kept[column] = ([text], "normal", value, "parser")
        elif status == "abnormal":
            named = set(findings.get(attribute, ()))
            dilated = column in DILATED_IF_ABNORMAL or (column in DILATED_IF_NAMED and "dilated" in named)
            fill = "dilated" if dilated else ABNORMAL_FILL.get(column)
            if fill is not None:
                text = fill if fill.endswith(".") else sentence(cr, spec, column, fill)
                kept[column] = ([text], "abnormal", fill if not fill.endswith(".") else "abnormal", "parser")

    for extra, (text, side) in extra_sentences(facts, name).items():
        kept[extra] = ([text], side, side, "parser")
    for procedure, (text, side) in procedure_sentences(row and row["caption"], statuses, name).items():
        kept[procedure] = ([text], side, "present", "parser")
    if (not kept and regurgitation_default_from is not None
            and regurgitation_default(report, structure, regurgitation_default_from) is None):
        kept[f"{name}_regurgitation"] = ([f"No {name} regurgitation."], "normal", "none", "default")
    if ef_sentence and name == "lv":
        entry, reason = ef_entry(report, row["caption"] if row is not None else "", policy)
        if ef_reasons is not None:
            ef_reasons[reason] += 1
        if entry is not None:
            text, side, value, source = entry
            kept[EF_COLUMN] = ([text], side, value, source)
    return kept, flag, dropped


def caption_row(study_id, name, kept, flag, policy, spec):
    """The caption row of the sentences `kept` (see `merged_sentences`), or None when they say nothing."""
    order = {c: i for i, c in enumerate(spec["fields"])}
    if EF_COLUMN in kept:       # the EF sentence right after the LV function sentence
        order[EF_COLUMN] = order[EF_AFTER] + 0.5
    # The wall summary and the extra sentences come last.
    columns = sorted(kept, key=lambda c: order.get(c, len(order)))
    texts = [text for column in columns for text in kept[column][0]]
    caption = " ".join(texts)
    if not rc.canonical_caption(caption):
        return None
    sides = {kept[c][1] for c in columns} - {EF_SIDE}
    normality = ("abnormal" if sides & {"abnormal", "device"} else
                 "normal" if sides == {"normal"} and flag == "normal" else "unknown")
    sources = [kept[c][3] for c in columns]
    return {"study_id": study_id, "structure": name, "caption": caption, "caption_id": rc.caption_id(caption),
            "normality": normality, "normality_status": "provisional", "policy_digest": policy.digest,
            "sentences": len(texts), "from_prefill": sum(s.startswith("prefill") for s in sources),
            "from_parser": sources.count("parser"), "from_ef": sources.count("ef_band"),
            "sources": json.dumps([[c, kept[c][2], kept[c][3]] for c in columns])}


def build_tables(linked, clips, routing, policy, spec, cr, limits, acquisition="bmode",
                 regurgitation_default_from=None, default_reasons=None, grade_valves=(), color_clips=None,
                 grade_reasons=None, ef_sentence=False, ef_reasons=None):
    """(caption table, dropped prefill findings) of the linked studies with clips. With `regurgitation_default_from`,
    `default_reasons` (a dict) gets, per default valve, the count of each reason a row without another caption got no
    default sentence, and `written`.

    With `grade_valves` (names in `GRADE_COLUMNS`) and `color_clips` (the split's colour-flow clips: study_id, view),
    a valve whose study has a colour clip routed to it may get a second row, of scope `color_flow`: the sentences of
    its B-mode row (none when no B-mode clip is routed to it) and the graded sentence (`regurgitation_grade`), which
    takes the place of a default sentence. The table then has a `scope` column, `bmode` on every other row, and
    `grade_reasons` (a dict) gets, per graded valve with a routed clip of either kind, the count of each reason.
    A row is per study, a bag is not: a reader must use a `color_flow` row only for a bag that holds a colour clip
    routed to the structure, and the B-mode row otherwise.

    With `ef_sentence`, every LV row with an EF gets the EF sentence (`ef_entry`), and `ef_reasons` (a Counter) gets
    the count of each reason over the studies with an LV clip."""
    views_of = clips.groupby("study_id").view.agg(lambda v: set(map(str, v))).to_dict()
    color_of = ({} if not grade_valves or color_clips is None else
                color_clips.groupby("study_id").view.agg(lambda v: set(map(str, v))).to_dict())
    rows, dropped = [], []
    for report in linked.to_dict("records"):
        study_id = str(report["study_id"])
        for structure in routing.trained:
            name = structure.name
            views = views_of.get(study_id, set()) & set(structure.views)
            color = color_of.get(study_id, set()) & set(structure.views) if name in grade_valves else set()
            if not views and name not in grade_valves:
                continue
            kept, flag = {}, "unknown"
            if views:
                kept, flag, out = merged_sentences(study_id, structure, report, views, policy, spec, cr, limits,
                                                   acquisition, regurgitation_default_from, ef_sentence, ef_reasons)
                row = caption_row(study_id, name, kept, flag, policy, spec)
                dropped += out
                if (regurgitation_default_from is not None and default_reasons is not None
                        and name in REGURGITATION_DEFAULT
                        and (row is None or json.loads(row["sources"])[0][2] == "default")):
                    reason = regurgitation_default(report, structure, regurgitation_default_from)
                    default_reasons.setdefault(name, Counter())[reason or "written"] += 1
                if row is not None:
                    rows.append({**row, "scope": "bmode"} if grade_valves else row)
            elif not color:
                continue
            if name in grade_valves:
                reason, entry = regurgitation_grade(report, structure, bool(color))
                if grade_reasons is not None:
                    grade_reasons.setdefault(name, Counter())[reason] += 1
                if entry is not None:
                    text, side, grade = entry
                    graded = {**kept, f"{name}_regurgitation": ([text], side, grade, "grade")}
                    rows.append({**caption_row(study_id, name, graded, flag, policy, spec), "scope": COLOR_FLOW})
    columns = [*COLUMNS, "scope"] if grade_valves else COLUMNS
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(dropped, columns=DROPPED_COLUMNS)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--column-review-root", required=True,
                   help="A checkout holding tools/labelbox/column_review.py and its specification (form v5).")
    p.add_argument("--reports", required=True, help="The cleaned DeepECHO report export (.parquet or .csv).")
    p.add_argument("--pacs", required=True, help="PACS index (AccessionNumber, StudyInstanceUID, PatientID).")
    p.add_argument("--clips", required=True, help="The eligible clip table (study_id, view, acquisition, split).")
    p.add_argument("--splits", required=True,
                   help="The full cohort split, before clip screening (study_id, split, patient_id if it has one).")
    p.add_argument("--split", default="train", help="The split to build.")
    p.add_argument("--acquisition", default="bmode", help="The acquisition mode the prefill is asked about.")
    p.add_argument("--check-limits", default=ccp.DEFAULT_LIMITS)
    p.add_argument("--out-dir", required=True, help="Output folder; must be outside this repository.")
    p.add_argument("--min-linked", type=float, default=0.95,
                   help="The share of the split's studies that must link to a report, or the build stops.")
    p.add_argument("--routing", default=None)
    p.add_argument("--regurgitation-default-from", type=int, default=None, metavar="YEAR",
                   help="Off by default. From this study year on, a tricuspid or pulmonic valve left without a "
                        "caption is captioned 'No <valve> regurgitation.' when its cell is empty, its structured "
                        "grade is missing or 0 and the conclusion does not name it (REGURGITATION_DEFAULT).")
    p.add_argument("--regurgitation-grade-valves", nargs="+", default=None, choices=sorted(GRADE_COLUMNS),
                   metavar="VALVE",
                   help="Off by default. For these valves, a row of scope color_flow with the structured "
                        "regurgitation grade as a sentence, when the study has a colour-flow clip routed to the valve "
                        "in --clips and the text agrees (regurgitation_grade). Adds a scope column.")
    p.add_argument("--ef-sentence", action="store_true",
                   help="Off by default. The LV row also states its EF as a number ('LV ejection fraction is 45%%.'), "
                        "from the report text (LV cell, then conclusion), else the structured visual EF (ef_entry).")
    p.add_argument("--policy", default=rn.DEFAULT_POLICY)
    p.add_argument("--report-accession-col", default="AccessionNumber")
    p.add_argument("--report-patient-col", default="Dossier")
    p.add_argument("--pacs-accession-col", default="AccessionNumber")
    p.add_argument("--pacs-study-col", default="StudyInstanceUID")
    p.add_argument("--pacs-patient-col", default="PatientID")
    return p.parse_args(argv)


def counts(values):
    return {str(k): ccp.suppressed(int(n)) for k, n in Counter(values).items()}


def main(argv=None):
    args = parse_args(argv)
    cr, spec = ccp.load_column_review(args.column_review_root)
    ccp.index_columns(spec)
    import build_radar_captions as brc     # puts this repository on the path, after the column review import
    from build_manifests import ensure_outside_repo
    from make_study_targets import link_studies, read_reports

    from src.datasets.radar_study_dataset import file_sha256
    from src.datasets.study_bags import check_study_ids
    from src.models.radar.routing import DEFAULT_ROUTING, load_routing

    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    routing, policy = load_routing(args.routing or DEFAULT_ROUTING), rn.load_policy(args.policy)
    limits = ccp.load_limits(args.check_limits)
    table = pd.read_csv(args.clips, dtype=str, usecols=["study_id", "view", "acquisition", "split"])
    try:
        check_study_ids(table.study_id)
    except ValueError as error:
        raise SystemExit(f"{args.clips}: {error}")
    clips = table[(table.split == args.split) & (table.acquisition == args.acquisition)]
    studies = set(table.loc[table.split == args.split, "study_id"])
    if clips.empty:
        raise SystemExit(f"No {args.acquisition} clip of split {args.split!r} in {args.clips}.")
    valves = tuple(args.regurgitation_grade_valves or ())
    untrained = sorted(set(valves) - {s.name for s in routing.trained})
    if untrained:
        raise SystemExit(f"--regurgitation-grade-valves: {untrained} have no trained query in the routing registry.")
    color_clips = table[(table.split == args.split) & (table.acquisition == COLOR_FLOW)] if valves else None
    unknown_views = sorted(set(clips.view).union(*([] if color_clips is None else [color_clips.view]))
                           - set(spec["views"]))
    if unknown_views:
        raise SystemExit(f"Views the specification does not know: {unknown_views}.")
    pacs = pd.read_csv(args.pacs, dtype=str, usecols=[args.pacs_accession_col, args.pacs_study_col,
                                                      args.pacs_patient_col])
    splits = pd.read_csv(args.splits, dtype=str)
    if not {"study_id", "split"} <= set(splits.columns):
        raise SystemExit("--splits needs study_id and split columns.")
    patients = brc.check_patients(studies, args.split, splits, pacs, args, set(table.study_id) - studies)
    reports = read_reports(args.reports, args)
    needed = ccp.report_columns(routing, policy, spec, limits)
    if args.regurgitation_default_from is not None:
        needed |= {YEAR_COLUMN, CONCLUSION_COLUMN, *(column for column, _ in REGURGITATION_DEFAULT.values())}
    if valves:
        needed |= {CONCLUSION_COLUMN, *(GRADE_COLUMNS[valve] for valve in valves)}
    if args.ef_sentence:
        needed |= {CONCLUSION_COLUMN, policy.ef["column"]}
    missing = sorted(needed - set(reports.columns))
    if missing:
        raise SystemExit(f"The reports have no column {missing}: a missing column would look like missing evidence.")
    linked = link_studies(reports, pacs, args)
    linked = linked[linked.study_id.isin(studies)]
    if linked.study_id.nunique() < args.min_linked * len(studies):
        raise SystemExit(f"Only {linked.study_id.nunique()} of the split's {len(studies)} studies link to a report.")
    default_reasons, grade_reasons, ef_reasons = {}, {}, Counter()
    captions, dropped = build_tables(linked, clips, routing, policy, spec, cr, limits, args.acquisition,
                                     args.regurgitation_default_from, default_reasons, valves, color_clips,
                                     grade_reasons, args.ef_sentence, ef_reasons)
    path = os.path.join(args.out_dir, "captions.csv")
    brc.write_atomic(path, lambda temporary: captions.to_csv(temporary, index=False))
    brc.write_atomic(os.path.join(args.out_dir, "dropped.csv"),
                     lambda temporary: dropped.to_csv(temporary, index=False))
    info = {"builder": "merged", "language": "en", "split": args.split, "studies_in_split": len(studies),
            "studies_linked": int(linked.study_id.nunique()), "patients": patients, "rows": len(captions),
            "policy_status": "provisional", "policy_digest": policy.digest, "rule_version": policy.rule_version,
            "flags": {k: int(v) for k, v in captions.normality.value_counts().items()},
            "rows_per_structure": {k: int(v) for k, v in captions.structure.value_counts().items()},
            "sentences": {"prefill": int(captions.from_prefill.sum()), "parser": int(captions.from_parser.sum()),
                          "ef_band": int(captions.from_ef.sum())},
            "dropped": counts(dropped.reason),
            "spec": {"version": spec["version"], "sha256": ccp.sha256(cr.SPEC)},
            "column_review": {"commit": ccp.git_commit(args.column_review_root),
                              "module_sha256": ccp.sha256(cr.__file__)},
            "check_limits_sha256": ccp.sha256(args.check_limits),
            "routing_sha256": routing.sha256, "captions_sha256": file_sha256(path),
            "clips_sha256": file_sha256(args.clips), "splits_sha256": file_sha256(args.splits),
            "pacs_sha256": file_sha256(args.pacs), "reports_sha256": file_sha256(args.reports)}
    if args.regurgitation_default_from is not None:     # absent when off, so an unchanged build is unchanged
        info["regurgitation_default"] = {"from_year": args.regurgitation_default_from,
                                         "per_valve": {name: {k: ccp.suppressed(int(n)) for k, n in reasons.items()}
                                                       for name, reasons in sorted(default_reasons.items())}}
    if valves:      # absent when off, as above
        info["regurgitation_grade"] = {
            "valves": list(valves), "scope": COLOR_FLOW, "color_clips": len(color_clips), "trace_needs_text": True,
            "codes": {f"{code:g}": grade for code, grade in GRADE_CODES.items()},
            "scopes": {k: int(v) for k, v in captions.scope.value_counts().items()},
            "per_valve": {name: {k: ccp.suppressed(int(n)) for k, n in sorted(reasons.items())}
                          for name, reasons in sorted(grade_reasons.items())}}

    if args.ef_sentence:        # absent when off, as above
        lv = captions[captions.structure == "lv"]
        values = [e[1] for s in lv.sources for e in json.loads(s) if e[0] == EF_COLUMN]
        info["ef_sentence"] = {
            "sentence": EF_SENTENCE, "order": ["lv_cell", "conclusion", "structured"],
            "tolerance": policy.ef["tolerance"], "lv_rows": len(lv), "lv_rows_with_ef": len(values),
            "reasons": {k: ccp.suppressed(int(n)) for k, n in sorted(ef_reasons.items())},
            "distinct_lv_captions": int(lv.caption.nunique())}

    def write_info(temporary):
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    brc.write_atomic(os.path.join(args.out_dir, "build_info.json"), write_info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    sys.exit(main())
