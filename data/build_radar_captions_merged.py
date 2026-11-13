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
   LV remodelling, and the LV mass (normal or increased). Without them an LV caption says nothing of hypertrophy. A
   sentence is written only when every text claim of that kind agrees and none is uncertain; measurements write none.
4. The visual EF, for the LV function only, when the prefill says nothing: the grade of the EF's band in this lab
   (`configs/radar/prefill_check_limits_v0.yaml`), unless the EF is in a grey zone or the parser reads the text the
   other way. The EF number itself is not written.

The 16 wall segments are one sentence when all are normal, and one sentence per abnormal segment otherwise. A caption
is the union over the views of every eligible clip of the study: an upper bound for a sampled bag, as in the
comparison tool.

`normality` is `abnormal` when a kept sentence is abnormal or names a device, `normal` when every sentence is normal
and the parser's own flag for the report text is normal, else `unknown`. It is PROVISIONAL (`normality_status`), so
the loss arms that read it (A2, A3) refuse this table. `sources` lists, per sentence, the column, the value and where
it came from (prefill_text, prefill_code, parser, ef_band): categories only, no report text. `dropped.csv` lists the
prefill findings left out and why.

The table and `build_info.json` have the layout `evals/radar_frozen/train.py` checks. Outputs go outside the
repository. Nothing printed holds report text; counts from 1 to 4 print as `<5`.

Example:
    python data/build_radar_captions_merged.py --column-review-root <column review checkout> --reports <reports> \\
        --pacs <pacs index> --clips <clip table> --splits <cohort studies.csv> --out-dir <folder>
"""

import argparse
import json
import os
import sys
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
}


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


def merged_row(study_id, structure, report, views, policy, spec, cr, limits, acquisition="bmode"):
    """(caption row or None, dropped prefill findings) of one (study, structure) whose clips have `views`."""
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

    order = {c: i for i, c in enumerate(spec["fields"])}
    # The wall summary and the extra sentences come last.
    columns = sorted(kept, key=lambda c: order.get(c, len(order)))
    texts = [text for column in columns for text in kept[column][0]]
    caption = " ".join(texts)
    if not rc.canonical_caption(caption):
        return None, dropped
    sides = {kept[c][1] for c in columns}
    normality = ("abnormal" if sides & {"abnormal", "device"} else
                 "normal" if sides == {"normal"} and flag == "normal" else "unknown")
    sources = [kept[c][3] for c in columns]
    return {"study_id": study_id, "structure": name, "caption": caption, "caption_id": rc.caption_id(caption),
            "normality": normality, "normality_status": "provisional", "policy_digest": policy.digest,
            "sentences": len(texts), "from_prefill": sum(s.startswith("prefill") for s in sources),
            "from_parser": sources.count("parser"), "from_ef": sources.count("ef_band"),
            "sources": json.dumps([[c, kept[c][2], kept[c][3]] for c in columns])}, dropped


def build_tables(linked, clips, routing, policy, spec, cr, limits, acquisition="bmode"):
    """(caption table, dropped prefill findings) of the linked studies with clips."""
    views_of = clips.groupby("study_id").view.agg(lambda v: set(map(str, v))).to_dict()
    rows, dropped = [], []
    for report in linked.to_dict("records"):
        study_id = str(report["study_id"])
        for structure in routing.trained:
            views = views_of.get(study_id, set()) & set(structure.views)
            if not views:
                continue
            row, out = merged_row(study_id, structure, report, views, policy, spec, cr, limits, acquisition)
            dropped += out
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS), pd.DataFrame(dropped, columns=DROPPED_COLUMNS)


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
    unknown_views = sorted(set(clips.view) - set(spec["views"]))
    if unknown_views:
        raise SystemExit(f"Views the specification does not know: {unknown_views}.")
    pacs = pd.read_csv(args.pacs, dtype=str, usecols=[args.pacs_accession_col, args.pacs_study_col,
                                                      args.pacs_patient_col])
    splits = pd.read_csv(args.splits, dtype=str)
    if not {"study_id", "split"} <= set(splits.columns):
        raise SystemExit("--splits needs study_id and split columns.")
    patients = brc.check_patients(studies, args.split, splits, pacs, args, set(table.study_id) - studies)
    reports = read_reports(args.reports, args)
    missing = sorted(ccp.report_columns(routing, policy, spec, limits) - set(reports.columns))
    if missing:
        raise SystemExit(f"The reports have no column {missing}: a missing column would look like missing evidence.")
    linked = link_studies(reports, pacs, args)
    linked = linked[linked.study_id.isin(studies)]
    if linked.study_id.nunique() < args.min_linked * len(studies):
        raise SystemExit(f"Only {linked.study_id.nunique()} of the split's {len(studies)} studies link to a report.")
    captions, dropped = build_tables(linked, clips, routing, policy, spec, cr, limits, args.acquisition)
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

    def write_info(temporary):
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    brc.write_atomic(os.path.join(args.out_dir, "build_info.json"), write_info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    sys.exit(main())
