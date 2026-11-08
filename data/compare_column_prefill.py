# data/compare_column_prefill.py

"""Column review prefill, checked finding by finding: against the report's other source and against the RADAR parser.
Counts only, on one split.

RADAR structures are to be captioned from the column review's report prefill (`tools/labelbox/column_review.py`, v4,
on the column review branch). Every finding it derives is checked against the report's other source: a finding read
from the text against the structured column for the same attribute (`STRUCTURED_CHECKS`), a finding read from a
structured column (the RV function code, the wall motion scores) against the text. Where no other source exists, the
check is skipped. Separately, the RADAR parser (`radar_normality.py`) reads the same text, and its reading is compared
with the prefill's: a second reader catches misreadings, though not a report that contradicts itself. This tool
measures all of that before captions are built from it.

For every study of the split that links to a report, and every trained structure of the routing registry with at least
one of the study's clips in its views:

- Prefill side: `build_candidate(report, view, acquisition)` for every distinct view of those clips, run as written,
  and the union of the columns whose target belongs to the structure: the caption of a bag holding every eligible clip
  (an upper bound for a sampled bag). A view with levels (subcostal) is read at the level whose targets include the
  structure, else at its default level. The 17 wall segments are summarised as one `wall_segments` column (abnormal if
  any segment is, else hyperkinetic if any is, else normal). Each value keeps its source: `text`, or `code` when it
  came from a structured column.
- Parser side: `caption_row` of `build_radar_captions.py` on the report without its EF, and per attribute the status
  its text facts give (`_resolve` without a measurement). The EF is left out because the prefill-based captions this
  tool evaluates will not append it; the current caption builder still does, so these outputs describe the planned
  captions, not the table training reads today. The whole-structure statements are resolved first: a whole that
  resolves to normal (`VG normal`) counts for the attributes it covers, a conflicted or uncertain one makes them
  conflicted or uncertain unless they have statements of their own, and a whole abnormal says nothing of any one
  attribute; never a device, the aortic cusp count or wall motion. Each aorta part is its own attribute
  (`aorta_size:root`), and a whole-aorta statement stands in for a part only when no statement names that part. The
  named abnormal findings of an attribute (`calcified`, `prolapse`...) are kept too; a state word (`augmentées`)
  names none.

The structured check of one finding (`structured_check`) gives:

- `consistent` (for a text RV function with a code: the prefill itself keeps it only then); `grey` (a measurement
  between the limits, which neither confirms nor contradicts); `no_measurement`; `not_checkable` (no other source for
  this column: valves, septal bulge, effusion, IVC, aorta);
- `contradiction`: dropped. For a text finding, the measurement lies where at least 90% of this lab's statements say
  the opposite (`configs/radar/prefill_check_limits_v0.yaml`, set on the training split: LV, RV and LA size, each LV
  wall); for LV function, the EF is two or more grades from every grade it may carry; for a code finding, the parser
  reads the text the other way;
- `grade_off_by_one`: LV function one grade from the nearest grade its EF may carry, on the same side: kept, recorded;
- `off_by_one_across_normal`: LV function normal one grade from a reduced-only EF, or reduced one grade from a
  normal-only EF: kept, recorded apart, and dropped instead with `--strict-normal`, because a kept one may be a false
  normal.

EF: the limits file gives, per EF range, the grades this lab writes there (ASE 2015, with grey zones where the men's
and women's limits differ). `hyperdynamic` counts as normal for the check. When the prefill has no LV function, an EF
in a range of one grade would fill it: `ef_fill.csv` counts those fills, the EFs in a grey zone that fill nothing, and
how often the parser reads the text the other way. IVC diameter, TAPSE and the aortic diameters are not checked. The
RV function code is already checked by the prefill itself, which drops the column when text and code differ.

The parser comparison of one finding (`outcome`): `agree`; `finding_mismatch` (both abnormal, but the parser named
other findings: calcified against sclerotic; a parser state with no named finding agrees); `disagree` (opposite
sides); `prefill_only`; `parser_only`; `parser_unsure` (uncertain, conflicted or not assessed); `prefill_other`
(`hyperdynamic`, `hyperkinetic`). Graded columns are compared by side only: the parser keeps the finding
(`dysfunction`), not its grade.

Outputs, in `--out-dir` (outside the repository):

- `structured_checks.csv`: per (structure, column, source), the structured check outcomes;
- `agreement.csv`: per (structure, column, source), the parser comparison outcomes;
- `survival.csv`: per (structure, column), prefill findings, kept after the structured checks, and kept after the
  parser comparison too (`disagree` and `finding_mismatch` dropping the finding);
- `crosstab.csv`: per (structure, column, source), prefill value against parser status;
- `ef_fill.csv`: LV function fills from the EF band, and those the parser contradicts;
- `fill_in.csv`: parser decisions on attributes no prefill column covers;
- `coverage.csv`: per structure, pairs with any prefill finding, with a parser caption, with any parser decision;
- `flagged.csv`: study, structure, column, source, prefill value, what flagged it and how: every dropped finding and
  every recorded grade difference. Study identifiers and categories, no report text and no measurement.
- `summary.json`: inputs and their digests, the specification and its commit, the policy digest, the options.

Every count from 1 to 4 is written `<5`, in the files and on screen. No report text is written or printed. The tables
are about patients: keep them outside the repository and get agreement before sharing even the counts. Develop on the
training split only.

Example:
    python data/compare_column_prefill.py \\
        --column-review-root /volume/DeepECHO/EchoJEPA-column-v3 \\
        --reports /media/data1/ravram/DeepECHO/TTE_reports_concat/df_report_2006_2026_clean.parquet \\
        --pacs /media/data1/datasets/echo/database_tte_2009_2024.csv \\
        --clips /volume/DeepECHO/experiments/radar_v0/clips_p0_train.csv \\
        --out-dir /volume/DeepECHO/experiments/radar_v0/column_prefill_compare
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace

import pandas as pd
import radar_captions as rc
import radar_normality as rn
import yaml

PREFILL_TARGETS = {  # a prefill target, the RADAR structure whose caption it belongs to
    "lv": "lv", "ivs": "lv", "rv": "rv", "la": "la", "ra": "ra", "mitral": "mitral", "aortic": "aortic",
    "tricuspid": "tricuspid", "pulmonic": "pulmonic", "pericardium": "pericardium", "ivc": "ivc",
    "aorta_root": "aorta", "aorta_ascending": "aorta", "aorta_arch": "aorta",
}
WALL_COLUMN = "wall_segments"
RV_CODE = "MHI VD fonction systolique"
POLARITY = {
    "normal": "normal", "not_increased": "normal", "not_demonstrated": "normal", "preserved": "normal",
    "three": "normal", "normal_or_hyperkinetic": "normal",
    "dilated": "abnormal", "mildly_reduced": "abnormal", "moderately_reduced": "abnormal",
    "severely_reduced": "abnormal", "increased": "abnormal", "septum_only": "abnormal", "posterior_only": "abnormal",
    "present": "abnormal", "calcified": "abnormal", "sclerotic": "abnormal", "thickened": "abnormal",
    "prolapse": "abnormal", "flail": "abnormal", "tethered": "abnormal", "restricted": "abnormal", "two": "abnormal",
    "reduced": "abnormal", "hypokinetic": "abnormal", "severely_hypokinetic": "abnormal", "akinetic": "abnormal",
    "akinetic_or_severely_hypokinetic": "abnormal", "dyskinetic": "abnormal", "aneurysmal": "abnormal",
    "demonstrated": "abnormal",
    "edge_to_edge_clip": "device", "annuloplasty_ring": "device", "bioprosthesis": "device",
    "mechanical_prosthesis": "device",
    "hyperdynamic": "other", "hyperkinetic": "other",
}
# A prefill value, the parser findings that say the same. A value not listed is compared by side only.
SAME_FINDING = {
    "calcified": {"calcified"}, "sclerotic": {"sclerotic"}, "thickened": {"thickened"},
    "prolapse": {"prolapse"}, "flail": {"flail"}, "restricted": {"restricted"},
    "dilated": {"dilated"}, "present": {"effusion"}, "two": {"bicuspid"}, "reduced": {"reduced_collapse", "stiff"},
}
# The structured column checking a text finding: the policy's measure of the same attribute. IVC diameter, TAPSE and
# the aorta diameters are left out of this run.
STRUCTURED_CHECKS = {"lv_size": "measure", "rv_size": "measure", "la_size": "measure",
                     "lv_wall_thickness": "walls", "lv_function": "ef", "rv_function": "rv_code"}
RV_CODES = {0, 1, 2, 3}     # the RV function code's grades; -1 is missing
GRADE = {"normal": 0, "hyperdynamic": 0, "mildly_reduced": 1, "moderately_reduced": 2, "severely_reduced": 3}
GRADE_VALUE = {0: "normal", 1: "mildly_reduced", 2: "moderately_reduced", 3: "severely_reduced"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIMITS = os.path.join(REPO, "configs", "radar", "prefill_check_limits_v0.yaml")
LIMITS_SCHEMA = "radar-prefill-check-limits"
THICK_WALLS_OF = ("septum", "posterior")
THICK_WALLS = {"not_increased": set(), "increased": {"septum", "posterior"}, "septum_only": {"septum"},
               "posterior_only": {"posterior"}}
DECIDED = ("normal", "abnormal", "device")
UNSURE = ("uncertain", "conflict", "not_assessed")
OUTCOMES = ("agree", "finding_mismatch", "disagree", "prefill_only", "parser_only", "parser_unsure", "prefill_other")
CHECKS = ("consistent", "grey", "no_measurement", "not_checkable", "grade_off_by_one", "off_by_one_across_normal",
          "contradiction")
DROPPED_BY_PARSER = ("disagree", "finding_mismatch")
# Fact labels that name no finding: a state word (`augmentées`, `anormal`) or a whole-structure rule moved onto an
# attribute (`anomalie de la fonction`). They give the side only.
GENERIC_FINDINGS = frozenset({"", "state", "cue"} | {rule.name for rule in rn._COMMON})


def speaks_for_column(claim):
    """Whether a parser claim says something about the prefill column of its attribute. Two kinds say less than the
    caption flag takes from them: hypertrophy that is not concentric (`pas d'HVG`, `HVG excentrique`) is about the LV
    mass, and the walls of this lab's reports vary under it; and a denied finding of a valve (`pas de prolapsus`)
    leaves the valve's other findings open, so it is no normal appearance or motion."""
    if claim.attribute == "lv_wall_thickness" and claim.finding == "hypertrophy":
        return False
    valve_part = claim.attribute.endswith(("_appearance", "_motion")) and claim.attribute != "lv_wall_motion"
    return not (valve_part and claim.negated)


def parser_attribute(column):
    """The parser attribute a prefill column is compared with, or None when the parser has none."""
    if column == WALL_COLUMN:
        return "lv_wall_motion"
    if column == "aortic_cusp_count":
        return "aortic_cusps"
    if column.startswith("aorta_") and column.endswith("_size"):
        return "aorta_size:" + column.split("_")[1]     # aorta_root_size: aorta_size:root
    if column in ("septal_bulge", "ra_size") or column.endswith("_regurgitant_jet"):
        return None
    return column


def wall_summary(values):
    """One value for the wall segments the prefill gave: abnormal if any is, else `hyperkinetic` (other) if any is,
    else normal."""
    sides = {POLARITY[v] for v in values}
    if "abnormal" in sides:
        return "abnormal_segment"
    return "hyperkinetic" if "other" in sides else "normal"


def prefill_side(value):
    if value == "abnormal_segment":
        return "abnormal"
    if value not in POLARITY:
        raise ValueError(f"No polarity for the prefill value {value!r}: add it to POLARITY.")
    return POLARITY[value]


def value_source(column, source_column):
    """`code` for a value read from a structured column, `text` for one read from report text."""
    return "code" if column == WALL_COLUMN or source_column == RV_CODE or str(source_column).startswith("WMS") \
        else "text"


# ------------------------------------------------------------------------------------------------- parser side

def parser_reading(facts, covered=()):
    """What the parser read in a caption's text facts (dicts as the caption table stores them):
    statuses {attribute: status} as `assess` resolves them without measurements; `own`, the statuses of attributes
    stated by facts of their own; and {attribute: named abnormal findings}, without the generic markers
    (`GENERIC_FINDINGS`). Claims that do not speak for a column (`speaks_for_column`) are left out. The
    whole-structure statements are resolved first. Only a whole that resolves to normal joins the attributes' own
    facts, as in `assess`; a whole that resolves to conflict, uncertain or not assessed is the status of a `covered`
    attribute with no facts of its own, and a whole abnormal says nothing of any one attribute. `covered`: attributes the whole structure speaks for, never a device, the aortic cusp count or wall
    motion."""
    claims = [c for c in (SimpleNamespace(**f) for f in facts if f["source"] == "text") if speaks_for_column(c)]
    whole = [c for c in claims if c.attribute == rn.OVERALL]
    whole_status = rn._resolve(rn.OVERALL, whole, None)[0] if whole else None
    whole_normal = [c for c in whole if c.value == "normal"] if whole_status == "normal" else []
    by_attribute, findings = defaultdict(list), defaultdict(set)
    for claim in claims:
        if claim.attribute != rn.OVERALL:
            by_attribute[claim.attribute].append(claim)
            if claim.value == "abnormal" and claim.finding and claim.finding not in GENERIC_FINDINGS:
                findings[claim.attribute].add(claim.finding)
    own = {}
    for attribute, mine in by_attribute.items():
        own[attribute] = _status(attribute, mine + ([] if _no_whole(attribute) else whole_normal))
    statuses = dict(own)
    spread = whole_status if whole_status == "normal" or whole_status in UNSURE else None
    if spread:
        for attribute in covered:
            if attribute not in statuses and not _no_whole(attribute):
                statuses[attribute] = spread
    return statuses, own, dict(findings)


def _no_whole(attribute):
    return attribute.endswith("_device") or attribute in ("aortic_cusps", "lv_wall_motion")


def _status(attribute, claims):
    status, _ = rn._resolve(attribute, claims, None)
    return "device" if attribute.endswith("_device") and status == "abnormal" else status


def parser_status(statuses, attribute, silent):
    """The parser's status for a prefill column's attribute: an aorta part falls back to a whole-aorta statement."""
    if attribute in statuses:
        return statuses[attribute]
    if attribute and attribute.startswith("aorta_size:") and "aorta_size" in statuses:
        return statuses["aorta_size"]
    return silent


def outcome(prefill_value, status, findings=()):
    """The parser comparison of one finding, or None when neither side says anything."""
    if prefill_value is None:
        return "parser_only" if status in DECIDED else None
    side = prefill_side(prefill_value)
    if side == "other":
        return "prefill_other"
    if status in DECIDED:
        if side != status:
            return "disagree"
        same = SAME_FINDING.get(prefill_value)
        return "finding_mismatch" if same and findings and not same & set(findings) else "agree"
    if status in UNSURE:
        return "parser_unsure"
    return "prefill_only"


# --------------------------------------------------------------------------------------------- structured side

def load_limits(path=DEFAULT_LIMITS):
    """The check limits (`configs/radar/prefill_check_limits_v0.yaml`), checked for shape."""
    with open(path, encoding="utf-8") as f:
        limits = yaml.safe_load(f)
    if not isinstance(limits, dict) or limits.get("schema") != LIMITS_SCHEMA:
        raise SystemExit(f"{path}: not a {LIMITS_SCHEMA} file.")
    sources = [s for m in limits["measures"].values() for s in m["sources"]] + list(limits["walls"].values())
    for source in sources:
        if not {"column", "scale", "valid", "abnormal_up_to", "normal_from"} <= set(source):
            raise SystemExit(f"{path}: a source needs column, scale, valid, abnormal_up_to and normal_from.")
        if source.get("contradicts", "both") not in ("both", "normal_only"):
            raise SystemExit(f"{path}: {source['column']}: contradicts must be both or normal_only.")
        if source["abnormal_up_to"] >= source["normal_from"]:
            raise SystemExit(f"{path}: {source['column']}: abnormal_up_to must be below normal_from.")
    lowest = [r[0] for r in limits["ef"]["ranges"]]
    if lowest != sorted(lowest, reverse=True) or lowest[-1] != 0:
        raise SystemExit(f"{path}: EF ranges must go from the highest down to 0.")
    if set(limits["measures"]) != {"lv_size", "la_size", "rv_size"} or set(limits["walls"]) != set(THICK_WALLS_OF):
        raise SystemExit(f"{path}: measures must be lv_size, la_size and rv_size, walls septum and posterior.")
    return limits


def measurement(report, source):
    """A report column's value in the limits' unit, or None when missing or outside its valid range."""
    value = rc.finite(report.get(source["column"]))
    if value is None:
        return None
    value *= source["scale"]
    low, high = source["valid"]
    return value if low <= value <= high else None


def check_value(side, value, source):
    """A normal or abnormal statement against one measurement: contradicted at or below `abnormal_up_to` (abnormal,
    unless the source `contradicts` a normal statement only) or from `normal_from` (normal), grey between the two."""
    if side == "abnormal" and value <= source["abnormal_up_to"] and source.get("contradicts") != "normal_only":
        return "contradiction"
    if side == "normal" and value >= source["normal_from"]:
        return "contradiction"
    if source["abnormal_up_to"] < value < source["normal_from"]:
        return "grey"
    return "consistent"


def check_measure(column, value, report, limits):
    """The check of a size against the first of its sources the report measured."""
    for source in limits["measures"][column]["sources"]:
        measured = measurement(report, source)
        if measured is not None:
            return check_value(POLARITY[value], measured, source)
    return "no_measurement"


def check_walls(value, report, limits):
    """The check of a wall thickness value, wall by wall: a contradiction on either wall, else grey on either, else
    consistent."""
    results = []
    for wall, source in limits["walls"].items():
        measured = measurement(report, source)
        if measured is not None:
            side = "abnormal" if wall in THICK_WALLS[value] else "normal"
            results.append(check_value(side, measured, source))
    if not results:
        return "no_measurement"
    for result in ("contradiction", "grey"):
        if result in results:
            return result
    return "consistent"


def ef_value(report, limits):
    """The report's visual EF, or None when missing or outside its valid range."""
    ef = rc.finite(report.get(limits["ef"]["column"]))
    low, high = limits["ef"]["valid"]
    return ef if ef is not None and low <= ef <= high else None


def ef_grades(ef, limits):
    """The grades an EF may carry in this lab: one, or two in a grey zone."""
    for lowest, grades in limits["ef"]["ranges"]:
        if ef >= lowest:
            return set(grades)
    raise ValueError(f"No EF range holds {ef}.")


def check_ef(value, ef, limits, strict_normal=False):
    """The check of an LV function value against the grades its EF may carry."""
    if ef is None:
        return "no_measurement"
    stated, allowed = GRADE[value], ef_grades(ef, limits)
    if stated in allowed:
        return "consistent"
    nearest = min(allowed, key=lambda g: abs(g - stated))
    if abs(nearest - stated) > 1:
        return "contradiction"
    if 0 in (nearest, stated):
        return "contradiction" if strict_normal else "off_by_one_across_normal"
    return "grade_off_by_one"


def check_against_text(value, status):
    """The check of a finding read from a structured column against the parser's reading of the text."""
    if status not in DECIDED:
        return "no_measurement"     # no text to check it against
    return "consistent" if prefill_side(value) == status else "contradiction"


def rv_code(report):
    """The report's RV function code, or None when missing (-1) or not a grade."""
    code = rc.finite(report.get(RV_CODE))
    return int(code) if code is not None and code in RV_CODES else None


def structured_check(column, value, source, report, own, limits, strict_normal=False):
    """The check of one prefill finding against the report's other source."""
    if value == "mixed":
        return "not_checkable"
    if source == "code":
        if prefill_side(value) == "other":
            return "not_checkable"
        return check_against_text(value, own.get(parser_attribute(column), "not_stated"))
    kind = STRUCTURED_CHECKS.get(column)
    if kind is None:
        return "not_checkable"
    if kind == "ef":
        return check_ef(value, ef_value(report, limits), limits, strict_normal)
    if kind == "rv_code":   # the prefill keeps a text RV function only when a present code agrees with it
        return "no_measurement" if rv_code(report) is None else "consistent"
    if kind == "walls":
        return check_walls(value, report, limits)
    return check_measure(column, value, report, limits)


# --------------------------------------------------------------------------------------------- the comparison

def suppressed(n):
    return "<5" if 0 < n < 5 else n


def suppress_frame(frame):
    numeric = frame.select_dtypes("number").columns
    return frame.astype({c: object for c in numeric}).apply(
        lambda col: col.map(suppressed) if col.name in numeric else col)


def load_column_review(root):
    """The column review module of the checkout at `root`, and its specification. `tools` is a package in this
    repository too, so the checkout goes first on the path and the module must come from it."""
    root = os.path.abspath(root)
    for name in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
        del sys.modules[name]
    sys.path.insert(0, root)
    try:
        cr = importlib.import_module("tools.labelbox.column_review")
    finally:
        sys.path.remove(root)   # the checkout has its own `src`: nothing else may be imported from it
    if not os.path.abspath(cr.__file__).startswith(root + os.sep):
        raise SystemExit(f"tools.labelbox.column_review came from {cr.__file__}, not from {root}.")
    return cr, cr.load_spec()


def level_for(view, structure, spec, cr):
    """The level to read a clip of `view` at for `structure`: the one whose targets include the structure, else the
    view's default."""
    for level, entry in spec["views"][view].get("levels", {}).items():
        if any(PREFILL_TARGETS.get(t) == structure for t in entry.get("targets", [])):
            return level
    return cr.default_level(view, spec)


def prefill_columns(report, views, structure, acquisition, spec, cr, cache):
    """{column: (value, source)} of the prefill for one (study, structure): the union over `views`, wall segments
    summarised."""
    found, sources = defaultdict(set), {}
    for view in sorted(views):
        level = level_for(view, structure, spec, cr)
        key = (view, level)
        if key not in cache:
            cache[key] = cr.build_candidate(report, view, acquisition, global_key="compare", level=level, spec=spec)
        for column in cache[key]["columns"]:
            if PREFILL_TARGETS.get(spec["fields"][column["concept"]]["target"]) == structure:
                found[column["concept"]].add(column["value"])
                sources.setdefault(column["concept"], value_source(column["concept"], column.get("source_column")))
    values, walls = {}, set()
    for column, seen in found.items():
        if column.startswith("wall_"):
            walls |= seen
        else:
            # values come from the report, so views should never differ
            values[column] = (next(iter(seen)) if len(seen) == 1 else "mixed", sources[column])
    if walls:
        values[WALL_COLUMN] = (wall_summary(walls), "code")
    return values


COLUMNS_OF = {}     # structure: the prefill columns that can belong to it, filled by `index_columns`
MAPPED = set()      # parser attributes some prefill column is compared with


def index_columns(spec):
    COLUMNS_OF.clear()
    MAPPED.clear()
    for column, field in spec["fields"].items():
        structure = PREFILL_TARGETS.get(field["target"])
        if structure is None:
            continue
        name = WALL_COLUMN if column.startswith("wall_") else column
        COLUMNS_OF.setdefault(structure, set()).add(name)
        if parser_attribute(name):
            MAPPED.add(parser_attribute(name))
    MAPPED.add("aorta_size")      # a whole-aorta statement stands in for the parts
    missing = {v for field in spec["fields"].values() for v in field["values"]} - set(POLARITY)
    if missing:
        raise SystemExit(f"Prefill values with no polarity: {sorted(missing)}. Add them to POLARITY.")


def compare(linked, clips, routing, policy, spec, cr, limits, acquisition="bmode", strict_normal=False):
    """(finding records, flagged rows, fill-in counts, EF fill counts, coverage counts) of the linked studies."""
    import build_radar_captions as brc

    views_of = clips.groupby("study_id").view.agg(lambda v: set(map(str, v))).to_dict()
    records, flagged = [], []
    fill_in, ef_fill, coverage = Counter(), Counter(), Counter()
    for report in linked.to_dict("records"):
        study_id = str(report["study_id"])
        study_views = views_of.get(study_id, set())
        without_ef = {**report, policy.ef["column"]: None}    # no EF is appended to a caption any more
        ef = ef_value(report, limits)
        cache = {}
        for structure in routing.trained:
            views = study_views & set(structure.views)
            if not views:
                continue
            name = structure.name
            coverage[(name, "eligible")] += 1
            prefill = prefill_columns(report, views, name, acquisition, spec, cr, cache)
            row = brc.caption_row(study_id, structure, without_ef, policy)
            if row is None:
                statuses, own, findings = {}, {}, {}
                silent = "removed" if brc.exclusion_row(study_id, structure, report) is not None else "empty"
            else:
                covered = {parser_attribute(c) for c in COLUMNS_OF.get(name, ())} - {None}
                statuses, own, findings = parser_reading(json.loads(row["facts"]), covered | {"aorta_size"})
                silent = "not_stated"
                coverage[(name, "parser_caption")] += 1
            coverage[(name, "prefill_any")] += bool(prefill)
            coverage[(name, "parser_decided_any")] += any(s in DECIDED for s in statuses.values())
            for column in sorted(set(prefill) | COLUMNS_OF.get(name, set())):
                attribute = parser_attribute(column)
                value, source = prefill.get(column, (None, ""))
                status = parser_status(statuses, attribute, silent) if attribute else "no_attribute"
                compared = (outcome(value, status, findings.get(attribute, ())) if value != "mixed"
                            else "prefill_other")
                check = structured_check(column, value, source, report, own, limits, strict_normal) \
                    if value is not None else ""
                if compared is None:
                    continue
                records.append((name, column, source, attribute or "", value or "none", status, compared, check))
                for kind, detail in (("structured", check), ("parser", compared)):
                    if detail in ("contradiction", "grade_off_by_one", "off_by_one_across_normal",
                                  *DROPPED_BY_PARSER):
                        flagged.append({"study_id": study_id, "structure": name, "column": column, "source": source,
                                        "prefill_value": value, "flagged_by": kind, "detail": detail,
                                        "parser_status": status})
            for attribute, status in own.items():
                if attribute not in MAPPED and status in DECIDED:
                    fill_in[(name, attribute, status)] += 1
            if name == "lv" and "lv_function" not in prefill and ef is not None:
                grades = ef_grades(ef, limits)
                if len(grades) > 1:
                    ef_fill[("grey_zone", "not_filled")] += 1
                else:
                    band = GRADE_VALUE[next(iter(grades))]
                    ef_fill[(band, "filled")] += 1
                    text = own.get("lv_function")
                    if text in DECIDED and text != ("normal" if band == "normal" else "abnormal"):
                        ef_fill[(band, "parser_reads_the_other_side")] += 1
    return records, flagged, fill_in, ef_fill, coverage


RECORD_COLUMNS = ["structure", "column", "source", "parser_attribute", "prefill_value", "parser_status", "outcome",
                  "check"]


def tables(records, flagged, fill_in, ef_fill, coverage):
    frame = pd.DataFrame(records, columns=RECORD_COLUMNS)
    eligible = {s: n for (s, k), n in coverage.items() if k == "eligible"}
    keys = ["structure", "column", "source"]
    agreement = (frame.groupby(keys + ["outcome"]).size().unstack("outcome", fill_value=0)
                 .reindex(columns=list(OUTCOMES), fill_value=0).reset_index())
    agreement.insert(3, "eligible", agreement.structure.map(eligible))
    found = frame[frame.prefill_value != "none"]
    checks = (found.groupby(keys + ["check"]).size().unstack("check", fill_value=0)
              .reindex(columns=list(CHECKS), fill_value=0).reset_index())
    kept_structured = found.check != "contradiction"
    kept_both = kept_structured & ~found.outcome.isin(DROPPED_BY_PARSER)
    survival = (found.assign(kept_structured=kept_structured, kept_both=kept_both)
                .groupby(["structure", "column"])
                .agg(prefill=("check", "size"), kept_structured=("kept_structured", "sum"),
                     kept_both=("kept_both", "sum")).reset_index())
    crosstab = (frame.groupby(keys + ["prefill_value", "parser_status"]).size().rename("count").reset_index())
    fill = pd.DataFrame([(s, a, st, n) for (s, a, st), n in sorted(fill_in.items())],
                        columns=["structure", "attribute", "status", "count"])
    ef = pd.DataFrame([(b, k, n) for (b, k), n in sorted(ef_fill.items())], columns=["band", "measure", "count"])
    cover = (pd.DataFrame([(s, k, n) for (s, k), n in coverage.items()], columns=["structure", "measure", "count"])
             .pivot(index="structure", columns="measure", values="count").fillna(0).astype(int).reset_index())
    cover = cover.reindex(columns=["structure", "eligible", "prefill_any", "parser_caption", "parser_decided_any"],
                          fill_value=0)
    flags = pd.DataFrame(flagged, columns=["study_id", "structure", "column", "source", "prefill_value", "flagged_by",
                                           "detail", "parser_status"])
    return {"structured_checks": checks, "agreement": agreement, "survival": survival, "crosstab": crosstab,
            "ef_fill": ef, "fill_in": fill, "coverage": cover}, flags


def report_columns(routing, policy, spec, limits):
    """Every report column either side reads: a missing one would look like missing evidence."""
    columns = {c for s in routing.trained for c in [s.report_column, *rn.report_columns(s.name, policy)]}
    columns.update(s["column"] for m in limits["measures"].values() for s in m["sources"])
    columns.update(s["column"] for s in limits["walls"].values())
    columns.add(limits["ef"]["column"])
    for field in spec["fields"].values():
        columns.update(field["sources"])
        columns.update(field.get("context_columns", []))
    for route in spec["views"].values():
        columns.update(route.get("wall_sources", {}).values())
    return columns


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(root):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--column-review-root", required=True,
                   help="A checkout holding tools/labelbox/column_review.py and its specification (the column "
                        "review branch, v4).")
    p.add_argument("--reports", required=True, help="The cleaned DeepECHO report export (.parquet or .csv).")
    p.add_argument("--pacs", required=True, help="PACS index (AccessionNumber, StudyInstanceUID, PatientID).")
    p.add_argument("--clips", required=True, help="The eligible clip table (study_id, view, acquisition, split).")
    p.add_argument("--split", default="train", help="The split to compare; develop on train only.")
    p.add_argument("--acquisition", default="bmode",
                   help="The acquisition mode the prefill is asked about (V1: bmode).")
    p.add_argument("--strict-normal", action="store_true",
                   help="Drop an LV function one EF band away across normal, instead of keeping and recording it.")
    p.add_argument("--check-limits", default=DEFAULT_LIMITS, help="The limits measurements are checked against.")
    p.add_argument("--out-dir", required=True, help="Output folder; must be outside this repository.")
    p.add_argument("--routing", default=None)
    p.add_argument("--policy", default=rn.DEFAULT_POLICY)
    p.add_argument("--report-accession-col", default="AccessionNumber")
    p.add_argument("--report-patient-col", default="Dossier")
    p.add_argument("--pacs-accession-col", default="AccessionNumber")
    p.add_argument("--pacs-study-col", default="StudyInstanceUID")
    p.add_argument("--pacs-patient-col", default="PatientID")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cr, spec = load_column_review(args.column_review_root)
    index_columns(spec)
    import build_radar_captions  # noqa: F401  (puts this repository on the path, after the column review import)
    from build_manifests import ensure_outside_repo
    from make_study_targets import link_studies, read_reports

    from src.models.radar.routing import DEFAULT_ROUTING, load_routing

    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    routing, policy = load_routing(args.routing or DEFAULT_ROUTING), rn.load_policy(args.policy)
    limits = load_limits(args.check_limits)
    clips = pd.read_csv(args.clips, dtype=str, usecols=["study_id", "view", "acquisition", "split"])
    clips = clips[(clips.split == args.split) & (clips.acquisition == args.acquisition)]
    if clips.empty:
        raise SystemExit(f"No {args.acquisition} clip of split {args.split!r} in {args.clips}.")
    unknown_views = sorted(set(clips.view) - set(spec["views"]))
    if unknown_views:
        raise SystemExit(f"Views the specification does not know: {unknown_views}.")
    pacs = pd.read_csv(args.pacs, dtype=str, usecols=[args.pacs_accession_col, args.pacs_study_col,
                                                      args.pacs_patient_col])
    reports = read_reports(args.reports, args)
    missing = sorted(report_columns(routing, policy, spec, limits) - set(reports.columns))
    if missing:
        raise SystemExit(f"The reports have no column {missing}: a missing column would look like missing evidence.")
    linked = link_studies(reports, pacs, args)
    linked = linked[linked.study_id.isin(set(clips.study_id))]
    records, flagged, fill_in, ef_fill, coverage = compare(linked, clips, routing, policy, spec, cr, limits,
                                                           args.acquisition, args.strict_normal)
    frames, flags = tables(records, flagged, fill_in, ef_fill, coverage)
    for name, frame in frames.items():
        suppress_frame(frame).to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    flags.to_csv(os.path.join(args.out_dir, "flagged.csv"), index=False)
    summary = {"split": args.split, "acquisition": args.acquisition, "strict_normal": args.strict_normal,
               "check_limits": {"path": os.path.abspath(args.check_limits), "sha256": sha256(args.check_limits),
                                "version": limits["version"]},
               "studies_in_clips": int(clips.study_id.nunique()), "studies_linked": int(linked.study_id.nunique()),
               "spec": {"schema": spec["schema"], "version": spec["version"], "sha256": sha256(cr.SPEC)},
               "column_review": {"root": os.path.abspath(args.column_review_root),
                                 "commit": git_commit(args.column_review_root),
                                 "module_sha256": sha256(cr.__file__)},
               "policy_digest": policy.digest, "rule_version": policy.rule_version,
               "routing_sha256": routing.sha256, "clips_sha256": sha256(args.clips),
               "reports_sha256": sha256(args.reports), "pacs_sha256": sha256(args.pacs),
               "flagged": {k: suppressed(int(n)) for k, n in flags.detail.value_counts().items()}}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with pd.option_context("display.max_rows", None, "display.width", 220):
        for name in ("coverage", "survival", "structured_checks", "agreement", "ef_fill"):
            print(f"\n== {name}")
            print(suppress_frame(frames[name]).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
