# data/build_radar_captions.py

"""The RADAR caption table of one split, from the report export.

For every study of the clip table's split and every trained structure of the routing registry: the report's findings
cell for that structure (`caption_text`), without what the policy's scope cannot ground (`retain_caption`), for the LV
with the structured EF written in (`lv_caption`), then judged as a whole (`assess`). A structure whose caption ends up
empty gets no row, because training has nothing to read; if its report cell held text, that cell goes to the exclusion
ledger instead, so that what was deleted can still be read. The caption in the table is the exact text that was judged,
and A0 and A1 train on the same text.

The flag is in `normality`; `normality_status` says whether its policy is approved, and training refuses A2 and A3
unless it is. Each row keeps what produced its flag: `facts`, `conflicts`, `reasons` and `unrecognized` as JSON, then
`evidence`, `boundary`, `ef_status`, `dropped_parts` and `stranded_parts`, with the policy's digest. `source` is the
report cell itself, for reading a caption beside what it came from.

`exclusions.csv` is that ledger: study, structure, source, the counts of what left, and why nothing is left:
`doppler_only` (every word was accounted for), `stranded_text` (something the rules cannot explain went with it: a
mixed statement like `Sténose aortique calcifiée` may have held B-mode evidence), or `no_words`. Training never reads
it; the audit does. A report cell that was empty is in neither file.

The clip table's study ids are taken as written, by every reader: one that is blank or has spaces or quotes around it
stops the build (`check_study_ids`). Report and PACS ids go through `id_text`. Zeros at the front of an accession
count, so an accession that a table stored as a number cannot link; those of a patient id do not.

Build one table per split, and develop or audit the rules on the training split only.

Patients: `--splits` is the full cohort split, from before clip screening, because a held-out patient can have a study
that screening removed. The build stops if a patient of this table has a study in another split, if a study of this
table is not in this split, or if a study's patient cannot be found, in that table or in PACS: it cannot then show that
the table is patient-disjoint. Studies of another split in the clip table count as held out too, and a table with no
held-out study at all stops the build: there would be nothing to compare. An id of only punctuation and zeros (a blank,
quotes, `-`, `000`, `0.0`) or a word of `PLACEHOLDER_IDS` names no one, so it counts as not found; any other id, of
letters or of digits, is someone's. The split is fixed upstream, so an overlap is an error to correct there, not rows
to drop. The split table can place a study this way, but it says nothing of whose report a report is: linking a report
takes a usable patient id in the report and in PACS, and the same one.

Doppler stills (`--doppler-captions`, off by default; needs a routing registry with `still_structs`, version 1): for
every structure a still may feed, a second row with `scope` = `full` is added next to the B-mode row (`scope` =
`bmode`): the B-mode caption followed by the report cell's Doppler clauses (`radar_doppler_captions.full_caption`), or
for Hemo, which has no report column, a caption templated from the report's RVSP, RA pressure and E/e'
(`hemo_caption`). The training dataset reads the full row of a structure only when the study's bag holds a still of
it. Full rows are flagged by provisional rules (`normality_status` = provisional), so A2 and A3 refuse such a table
until a policy with Doppler scope is approved; A0 and A1 run on it. Without the flag the table is exactly as before
(no `scope` column).

Example:
    python data/build_radar_captions.py \\
        --reports /media/data1/ravram/DeepECHO/TTE_reports_concat/df_report_2006_2026_clean.parquet \\
        --pacs /media/data1/datasets/echo/database_tte_2009_2024.csv \\
        --clips <manifests>/clips.csv --splits <manifests>/studies.csv --out-dir <captions_dir>

Writes `captions.csv`, `exclusions.csv` and `build_info.json` (counts, and the digests of what went in and came out,
which the audit and the trainer check). A build that links under `--min-linked` of the split's studies stops. The two
tables hold report text and study identifiers: keep them outside the repository, and never send them to an external
service.
"""

import argparse
import json
import os
import sys

import pandas as pd
import radar_captions as rc
import radar_doppler_captions as rdc
import radar_normality as rn
from build_manifests import REPO_ROOT, ensure_outside_repo
from make_study_targets import link_studies, normalize_id, patient_key, read_reports, usable_ids

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from src.datasets.radar_study_dataset import file_sha256  # noqa: E402
from src.datasets.study_bags import check_study_ids  # noqa: E402
from src.models.radar.routing import DEFAULT_ROUTING, load_routing  # noqa: E402

COLUMNS = ("study_id", "structure", "caption", "caption_id", "normality", "normality_status", "policy_digest",
           "evidence", "boundary", "ef_status", "dropped_parts", "stranded_parts", "facts", "conflicts", "reasons",
           "unrecognized", "source")
EXCLUSION_COLUMNS = ("study_id", "structure", "reason", "dropped_parts", "stranded_parts", "source")


def _json(items):
    return json.dumps(items, ensure_ascii=False, default=vars)


def caption_row(study_id, structure, report, policy):
    """The caption table's row of one (study, structure), or None without a caption."""
    caption, dropped, stranded = rn.retain_caption(report.get(structure.report_column))
    ef_status = ""
    if structure.name == "lv":
        caption, ef_status = rc.lv_caption(caption, report.get(policy.ef["column"]), policy.ef["tolerance"])
    if not rc.canonical_caption(caption):
        return None
    structured = rn.structured_facts(structure.name, report, policy)
    result = rn.assess(structure.name, caption, policy, structured, ef_status or None, stranded)
    return {"study_id": study_id, "structure": structure.name, "caption": caption,
            "caption_id": rc.caption_id(caption), "normality": result.flag, "normality_status": policy.status,
            "policy_digest": policy.digest, "evidence": result.evidence, "boundary": result.boundary,
            "ef_status": ef_status, "dropped_parts": dropped, "stranded_parts": stranded,
            "facts": _json(result.facts), "conflicts": _json(result.conflicts), "reasons": _json(result.reasons),
            "unrecognized": _json(result.unrecognized), "source": rc.caption_text(report.get(structure.report_column))}


def exclusion_reason(dropped, stranded):
    """Why a report cell with text left no caption."""
    return "stranded_text" if stranded else "doppler_only" if dropped else "no_words"


def exclusion_row(study_id, structure, report):
    """The ledger's row of a (study, structure) without a caption, or None if its report cell was empty."""
    source = rc.caption_text(report.get(structure.report_column))
    if not source:
        return None
    _, dropped, stranded = rn.retain_caption(source)
    return {"study_id": study_id, "structure": structure.name, "reason": exclusion_reason(dropped, stranded),
            "dropped_parts": dropped, "stranded_parts": stranded, "source": source}


def full_row(study_id, structure, report, bmode, policy):
    """The `full` scope row of a structure fed by Doppler stills, or None without Doppler text (see the docstring)."""
    if structure.caption_source == "structured":
        caption, flag = rdc.hemo_caption(report), rdc.hemo_flag(report)
        source = ""
    else:
        source = rc.caption_text(report.get(structure.report_column))
        caption, flag = rdc.full_caption(bmode["caption"] if bmode else "", source), "unknown"
    if not rc.canonical_caption(caption):
        return None
    row = {c: "" for c in COLUMNS}
    row.update({"study_id": study_id, "structure": structure.name, "caption": caption,
                "caption_id": rc.caption_id(caption), "normality": flag, "normality_status": rdc.FULL_STATUS,
                "policy_digest": policy.digest, "dropped_parts": 0, "stranded_parts": 0, "facts": "[]",
                "conflicts": "[]", "reasons": "[]", "unrecognized": "[]", "source": source, "scope": "full"})
    return row


def build_tables(reports, routing, policy, doppler=False):
    """(caption table, exclusion ledger) of `reports`, one row per study with a `study_id` column. `doppler` adds
    the `full` scope rows of the structures that Doppler stills feed."""
    fed = {name for targets in routing.still_structs.values() for name in targets}
    if doppler and not fed:
        raise SystemExit("--doppler-captions needs a routing registry with still_structs (version 1).")
    wanted = {c for s in routing.trained for c in [s.report_column, *rn.report_columns(s.name, policy)]} - {None}
    if doppler:
        wanted |= set(rdc.hemo_columns()) if any(s.caption_source == "structured" for s in routing.trained) else set()
    missing = sorted(wanted - set(reports.columns))
    if missing:
        raise SystemExit(f"The reports have no column {missing}: a missing column would look like missing evidence.")
    rows, excluded = [], []
    for record in reports.to_dict("records"):
        for structure in routing.trained:
            study_id = str(record["study_id"])
            row = caption_row(study_id, structure, record, policy) if structure.report_column else None
            if row is not None:
                rows.append({**row, "scope": "bmode"} if doppler else row)
            elif structure.report_column and (exclusion := exclusion_row(study_id, structure, record)) is not None:
                excluded.append(exclusion)
            if doppler and structure.name in fed and (full := full_row(study_id, structure, record, row, policy)):
                rows.append(full)
    columns = [*COLUMNS, "scope"] if doppler else COLUMNS
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(excluded, columns=EXCLUSION_COLUMNS)


def study_patients(splits, pacs, args):
    """(study_id, patient) pairs, from PACS and from the split table's `patient_id` when it has one. A pair with an
    unusable id is left out, so its study stays without a patient unless the other source gives one."""
    pairs = pacs[[args.pacs_study_col, args.pacs_patient_col]].set_axis(["study_id", "patient"], axis=1)
    if "patient_id" in splits.columns:
        own = splits[["study_id", "patient_id"]].rename(columns={"patient_id": "patient"})
        pairs = pd.concat([pairs, own], ignore_index=True)
    pairs = pairs.assign(study_id=usable_ids(pairs.study_id), patient=patient_key(pairs.patient))
    return pairs.dropna().drop_duplicates()


def check_patients(built, split, splits, pacs, args, elsewhere=()):
    """Counts of the patient check of a table of `built` studies, `elsewhere` being studies of another split in the
    clip table; see the module docstring for what it refuses."""
    splits = splits.assign(study_id=normalize_id(splits.study_id), split=splits.split.astype(str))
    built = set(normalize_id(pd.Series(sorted(built))))
    outside = built - set(splits.loc[splits.split == split, "study_id"])
    if outside:
        raise SystemExit(f"{len(outside)} studies of this table are not in split {split!r} of --splits.")
    held_out = set(splits.loc[splits.split != split, "study_id"]) | set(elsewhere)
    if not held_out:
        raise SystemExit(f"--splits holds no study outside split {split!r}: nothing to check the table against.")
    pairs = study_patients(splits, pacs, args)
    unknown = (built | held_out) - set(pairs.study_id)
    if unknown:
        raise SystemExit(f"{len(unknown)} studies have no usable patient id in --splits or in PACS: the table cannot "
                         f"be shown to be patient-disjoint.")
    patients = set(pairs.loc[pairs.study_id.isin(built), "patient"])
    others = set(pairs.loc[pairs.study_id.isin(held_out), "patient"])
    shared = patients & others
    if shared:
        raise SystemExit(f"{len(shared)} patients of split {split!r} also have a held-out study: correct the split.")
    return {"in_table": len(patients), "held_out": len(others), "held_out_studies": len(held_out)}


def write_atomic(path, write):
    """`write` to a temporary file beside `path`, then move it into place: a failed run leaves the old file whole."""
    temporary = f"{path}.tmp"
    write(temporary)
    os.replace(temporary, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", required=True, help="The cleaned DeepECHO report export (.parquet or .csv).")
    p.add_argument("--pacs", required=True, help="PACS index (AccessionNumber, StudyInstanceUID, PatientID).")
    p.add_argument("--clips", required=True, help="The final eligible clip table (study_id, split).")
    p.add_argument("--splits", required=True,
                   help="The full cohort split, before clip screening (study_id, split, patient_id if it has one).")
    p.add_argument("--split", default="train", help="The split to build.")
    p.add_argument("--out-dir", required=True, help="Output folder; must be outside this repository.")
    p.add_argument("--min-linked", type=float, default=0.95,
                   help="The share of the split's studies that must link to a report, or the build stops.")
    p.add_argument("--routing", default=DEFAULT_ROUTING)
    p.add_argument("--policy", default=rn.DEFAULT_POLICY)
    p.add_argument("--doppler-captions", action="store_true",
                   help="Add the full-scope rows of the structures Doppler stills feed (needs a v1 routing).")
    p.add_argument("--report-accession-col", default="AccessionNumber")
    p.add_argument("--report-patient-col", default="Dossier")
    p.add_argument("--pacs-accession-col", default="AccessionNumber")
    p.add_argument("--pacs-study-col", default="StudyInstanceUID")
    p.add_argument("--pacs-patient-col", default="PatientID")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    routing, policy = load_routing(args.routing), rn.load_policy(args.policy)
    clips = pd.read_csv(args.clips, dtype={"study_id": str}, usecols=["study_id", "split"])
    try:
        check_study_ids(clips.study_id)
    except ValueError as error:
        raise SystemExit(f"{args.clips}: {error}")
    clips = clips.assign(split=clips.split.astype(str))
    studies = set(clips.loc[clips.split == args.split, "study_id"])
    if not studies:
        raise SystemExit(f"No study of split {args.split!r} in {args.clips}.")
    pacs = pd.read_csv(args.pacs, dtype=str, usecols=[args.pacs_accession_col, args.pacs_study_col,
                                                      args.pacs_patient_col])
    splits = pd.read_csv(args.splits, dtype=str)
    if not {"study_id", "split"} <= set(splits.columns):
        raise SystemExit("--splits needs study_id and split columns.")
    patients = check_patients(studies, args.split, splits, pacs, args, set(clips.study_id) - studies)
    reports = read_reports(args.reports, args)
    linked = link_studies(reports, pacs, args)
    linked = linked[linked.study_id.isin(studies)]
    if linked.study_id.nunique() < args.min_linked * len(studies):
        raise SystemExit(f"Only {linked.study_id.nunique()} of the split's {len(studies)} studies link to a report: "
                         f"check the identifiers, or lower --min-linked if that is expected.")
    table, exclusions = build_tables(linked, routing, policy, doppler=args.doppler_captions)
    path = os.path.join(args.out_dir, "captions.csv")
    write_atomic(path, lambda temporary: table.to_csv(temporary, index=False))
    ledger = os.path.join(args.out_dir, "exclusions.csv")
    write_atomic(ledger, lambda temporary: exclusions.to_csv(temporary, index=False))
    info = {"split": args.split, "studies_in_split": len(studies), "studies_linked": int(linked.study_id.nunique()),
            "patients": patients, "rows": len(table), "policy_status": policy.status,
            "excluded": {k: int(v) for k, v in exclusions.reason.value_counts().items()},
            "policy_digest": policy.digest, "rule_version": policy.rule_version,
            "flags": {k: int(v) for k, v in table.normality.value_counts().items()},
            **({"scopes": {k: int(v) for k, v in table.scope.value_counts().items()}}
               if args.doppler_captions else {}),
            "routing_sha256": routing.sha256, "captions_sha256": file_sha256(path),
            "exclusions_sha256": file_sha256(ledger),
            "clips_sha256": file_sha256(args.clips), "splits_sha256": file_sha256(args.splits),
            "pacs_sha256": file_sha256(args.pacs), "reports_sha256": file_sha256(args.reports)}

    def write_info(temporary):
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    write_atomic(os.path.join(args.out_dir, "build_info.json"), write_info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
