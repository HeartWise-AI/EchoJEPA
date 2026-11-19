# data/radar_normality_audit.py

"""Coverage and accuracy of the RADAR caption flags.

Synthetic tests show how the rules behave. How often they are right on real reports takes a person reading a sample of
complete captions, so this tool prepares that reading and scores it. It runs on the training split's caption table.

- `coverage`: per structure, how many flags are normal, abnormal or unknown, why the unknown ones are unknown, and the
  conflicts, with denominators. With the clip table it follows the funnel: studies in the split, with a caption, with
  an eligible clip, with one in the first epoch's bag. `sources` says what became of every study of the split: no
  report, an empty report cell, a caption, or text that was removed in full, by reason. The unknown fraction is what
  decides whether a local language model is worth adding. It also lists identical captions with different flags (a flag
  then rests on something the text does not say). `unrecognized_wording.csv` lists the most frequent wording the rules
  did not understand: where they grow.
- `sample`: a blinded stratified sample of complete captions. It samples the rows training can read (a study with a
  clip the structure may pool), so the rates it gives are rates among training rows, in strata of structure, flag,
  evidence, conflict, whether a value sits near a cutoff, and common or rare wording (without that last one the few
  most frequent captions would take the whole sample, and rare wording is where the errors hide). Predicted normal
  captions get the largest share, because a false normal is the error that matters most. `audit_sheet.csv` lists each
  sampled text once and holds no flag, so the reader is not anchored; `audit_key.csv` keeps the flags and the sampling
  weights, and stays away from the reader. An existing key is never overwritten: its sheet would be scored against
  other texts.
  `fidelity_sheet.csv` sets rows that lost text beside the report cell they came from, half of them rows that claim
  nothing of substance was lost, because no reading of final captions can show what was deleted. A quarter of it, when
  there are that many, are report cells removed in full, from the exclusion ledger, with an empty caption: half of
  those lost text the rules could not explain, the mixed statements that may have held B-mode evidence.
- `score`: reads the filled sheet against the key, after checking that every id still holds the text of the key and
  that the key comes from the current rules. Per structure: false normals, precision of the normal group, false
  abnormals, and what the readers say of the unknown captions, each with counts. Precision is weighted back to the
  table's strata, with a Wilson interval on the effective sample size, and says how many rows the sampled strata cover
  (a small quota cannot reach every stratum). An overall accuracy is deliberately not given: it can hide a poor normal
  group, and the normal group is what A2 turns into positives. With `--fidelity` it counts the filled source sheet,
  and on their own the cells removed in full.

`coverage` and `sample` first check the table: it must be the file the builder wrote for this registry, clip table and
split, and every flag, id and count must follow from its source and these rules. A table that does not is refused, and
so is an exclusion ledger that is not the builder's, or that holds a cell these rules would keep.

A sample of about 200 is a first check, not a validation of every rare finding. A structure with few checked captions
has a wide interval, and the report says so rather than rounding it away.

The reader answers one question per caption: "Read as written, does this caption describe this structure as normal?"
with `normal`, `abnormal` or `unclear` in `reader_flag`. No images: this audits the text rules, not the reports. On the
source sheet the question is whether the caption says what its source says about B-mode findings, with `faithful`,
`changed` or `unclear` in `reader_fidelity`. For an empty caption that means whether the source held no B-mode finding.

Examples:
    python data/radar_normality_audit.py coverage --captions <dir>/captions.csv --clips <manifests>/clips.csv \\
        --out-dir <audit_dir>
    python data/radar_normality_audit.py sample --captions <dir>/captions.csv --clips <manifests>/clips.csv \\
        --out-dir <audit_dir>
    python data/radar_normality_audit.py score --key <audit_dir>/audit_key.csv --sheet <filled>.csv \\
        --fidelity <filled_source>.csv --out-dir <audit_dir>

Sheets and `unrecognized_wording.csv` hold report text, and every output is about patients: keep them outside the
repository, never send them to an external service, and get agreement before publishing even aggregate counts.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter

import pandas as pd
import radar_captions as rc
import radar_normality as rn
from build_manifests import REPO_ROOT, ensure_outside_repo
from build_radar_captions import EXCLUSION_COLUMNS, exclusion_reason

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from src.datasets.radar_study_dataset import check_caption_build, check_views, file_sha256  # noqa: E402
from src.datasets.study_bags import check_study_ids, choose_bag, training_rng  # noqa: E402
from src.models.radar.routing import DEFAULT_ROUTING, load_routing  # noqa: E402

READER_FLAGS = ("normal", "abnormal", "unclear")
FIDELITY = ("faithful", "changed", "unclear")
STRATUM = ["structure", "normality", "evidence", "conflict", "boundary", "wording"]
REMOVED = ("doppler_only", "stranded_text", "no_words")     # why a report cell with text left no caption
COMMON_SHARE = 0.5   # the most frequent captions that together make up this share of a group's rows are `common`
READER_QUESTION = "Read as written, does this caption describe this structure as normal?"


def read_captions(path):
    table = pd.read_csv(path, dtype={"study_id": str}, keep_default_na=False)
    table["boundary"] = table.boundary.astype(str) == "True"
    return table


def read_exclusions(captions_path, info):
    """The exclusion ledger beside a caption table, after checking that it is the one the builder wrote with it."""
    path = os.path.join(os.path.dirname(os.path.abspath(captions_path)), "exclusions.csv")
    if not os.path.exists(path) or info.get("exclusions_sha256") != file_sha256(path):
        raise SystemExit(f"No exclusion ledger of this build beside {captions_path}: rebuild the captions.")
    return pd.read_csv(path, dtype={"study_id": str}, keep_default_na=False)


def exclusion_problems(exclusions, table):
    """Ledger rows that do not follow from their source: what the ledger says left must have left."""
    problems = Counter()
    missing = set(EXCLUSION_COLUMNS) - set(exclusions.columns)
    if missing:
        return {"ledger_columns_missing": len(missing)}
    keys = list(zip(exclusions.study_id, exclusions.structure))
    captioned = set(zip(table.study_id, table.structure))
    problems["study_and_structure_twice"] = len(keys) - len(set(keys))
    problems["excluded_and_captioned"] = len(set(keys) & captioned)
    for row in exclusions.itertuples():
        kept, dropped, stranded = rn.retain_caption(row.source)
        problems["excluded_but_kept_by_these_rules"] += bool(kept)
        problems["ledger_differs_from_source"] += (
            (dropped, stranded) != (int(row.dropped_parts), int(row.stranded_parts))
            or row.reason != exclusion_reason(dropped, stranded))
    return {k: v for k, v in problems.items() if v}


def table_problems(table, policy):
    """Rows that do not follow from their source, the policy and the rules: the check that a flag, a caption and a
    caption id belong together, and that the caption is what its report cell gives."""
    problems = Counter()
    if "source" not in table.columns:
        return {"no_source_column": len(table)}
    duplicates = int(table.duplicated(["study_id", "structure"]).sum())
    if duplicates:
        problems["study_and_structure_twice"] = duplicates
    for row in table.itertuples():
        if row.policy_digest != policy.digest:
            problems["other_policy_or_rules"] += 1
            continue
        structured = tuple(rn.Fact(**f) for f in json.loads(row.facts) if f["source"] == "structured")
        derived = rn.assess(row.structure, row.caption, policy, structured, row.ef_status or None, row.stranded_parts)
        kept, dropped, stranded = rn.retain_caption(row.source)
        if derived.flag != row.normality:
            problems["flag_differs_from_caption"] += 1
        if (derived.evidence, bool(derived.boundary)) != (row.evidence, bool(row.boundary)):
            problems["evidence_differs_from_caption"] += 1
        if rc.caption_id(row.caption) != int(row.caption_id):
            problems["caption_id_differs_from_caption"] += 1
        same_text = row.caption.startswith(kept.rstrip(" .")) if row.ef_status == "appended" else row.caption == kept
        if (dropped, stranded) != (int(row.dropped_parts), int(row.stranded_parts)) or not same_text:
            problems["caption_differs_from_source"] += 1
    return dict(problems)


def bag_eligibility(clips, routing, num_clips, seed):
    """{study: (structures with an eligible clip in the study, structures with one in the first epoch's bag)}."""
    out = {}
    for study, rows in clips.groupby(clips.study_id.astype(str), sort=True):
        members = dict(zip(rows.clip_id.astype(str), rows.view.astype(str)))
        chosen = choose_bag(list(members.items()), num_clips, "view_balanced", training_rng(seed, 0, study),
                            routing.view_families)
        views, bag = set(members.values()), {members[c] for c in chosen}
        out[study] = ({s.name for s in routing.trained if views & set(s.views)},
                      {s.name for s in routing.trained if bag & set(s.views)})
    return out


def split_eligibility(table, clips, routing, num_clips, seed, split):
    """`bag_eligibility` of the split's clips, refusing a bad view or a captioned study outside the split."""
    if "split" in clips.columns:
        clips = clips[clips.split.astype(str) == split]
    try:
        check_views(clips, routing)
        check_study_ids(clips.study_id)
    except ValueError as error:
        raise SystemExit(str(error))
    eligible = bag_eligibility(clips, routing, num_clips, seed)
    outside = set(table.study_id) - set(eligible)
    if outside:
        raise SystemExit(f"{len(outside)} captioned studies are not in split {split!r} of the clip table.")
    return eligible


def trainable(table, eligible):
    """The rows training can read: their study has a clip the structure may pool."""
    return table.loc[[row.structure in eligible[row.study_id][0] for row in table.itertuples()]]


def _fractions(counts, total):
    return {k: {"n": int(v), "fraction": round(v / total, 4) if total else None} for k, v in counts.items()}


def _structure_summary(rows):
    """The flag, reason and conflict counts of one structure's rows."""
    n = len(rows)
    flags = Counter(rows.normality)
    reasons, conflicts = Counter(), Counter()
    for row in rows[rows.normality == "unknown"].itertuples():
        reasons.update({reason.split(":")[0] for reason in json.loads(row.reasons)})
    for text in rows.conflicts:
        conflicts.update(c["kind"] for c in json.loads(text))
    mixed = rows.groupby("caption_id").normality.nunique()
    return {"rows": n, "flags": _fractions({k: flags.get(k, 0) for k in rn.FLAGS}, n),
            "unknown_rows_with_reason": _fractions(reasons, n), "conflicts": dict(conflicts),
            "evidence": dict(Counter(rows.evidence)), "near_a_cutoff": int(rows.boundary.sum()),
            "rows_with_parts_removed": int((rows.dropped_parts.astype(int) > 0).sum()),
            "distinct_captions": int(rows.caption_id.nunique()),
            "identical_captions_with_different_flags": int((mixed > 1).sum())}


def source_funnel(table, exclusions, structure, studies_in_split, studies_linked):
    """What became of one structure's report cell in every study of the split. The counts add up to the split."""
    excluded = Counter(exclusions.reason[exclusions.structure == structure])
    captioned = int((table.structure == structure).sum())
    return {"no_report": studies_in_split - studies_linked,
            "empty_cell": studies_linked - captioned - sum(excluded.values()),
            "with_caption": captioned, **{f"removed_{k}": excluded.get(k, 0) for k in REMOVED}}


def coverage(table, routing, policy, clips=None, num_clips=12, seed=0, split="train", exclusions=None, linked=None):
    report = {"policy_status": policy.status, "policy_digest": policy.digest, "rule_version": policy.rule_version,
              "rows": len(table), "structures": {}}
    eligible = None
    if clips is not None:
        eligible = split_eligibility(table, clips, routing, num_clips, seed, split)
        report["studies_in_split"] = len(eligible)
    usable = []
    for structure in routing.trained:
        rows = table[table.structure == structure.name]
        summary = _structure_summary(rows)
        if eligible is not None:
            in_study = rows.study_id.isin({s for s, (names, _) in eligible.items() if structure.name in names})
            in_bag = rows.study_id.isin({s for s, (_, names) in eligible.items() if structure.name in names})
            rows = rows[in_bag]
            usable.append(rows)
            summary["funnel"] = {
                "studies_in_split": len(eligible), "with_caption": summary["rows"],
                "with_eligible_clip": int(in_study.sum()), "with_eligible_clip_in_bag": int(in_bag.sum())}
            if exclusions is not None:
                summary["sources"] = source_funnel(table, exclusions, structure.name, len(eligible), linked)
            summary["flags_in_bag"] = _fractions({k: (rows.normality == k).sum() for k in rn.FLAGS}, len(rows))
        report["structures"][structure.name] = summary
    final = pd.concat(usable) if usable else table
    report["unknown_fraction"] = round(float((final.normality == "unknown").mean()), 4) if len(final) else None
    return report


def unrecognized_wording(table, min_count=5):
    """The wording not understood, by structure, at least `min_count` times: rare wording may identify someone."""
    counts = Counter()
    for row in table.itertuples():
        counts.update((row.structure, text) for text in json.loads(row.unrecognized))
    rows = [{"structure": s, "wording": w, "count": n} for (s, w), n in counts.most_common() if n >= min_count]
    return pd.DataFrame(rows, columns=["structure", "wording", "count"])


def _wording(table):
    """common or rare, per row, within each (structure, flag)."""
    common = set()
    for (structure, flag), rows in table.groupby(["structure", "normality"]):
        counts = rows.caption_id.value_counts().sort_index().sort_values(ascending=False, kind="stable")
        before = counts.cumsum().shift(fill_value=0) / len(rows)
        common.update((structure, flag, c) for c in counts.index[before < COMMON_SHARE])
    keys = zip(table.structure, table.normality, table.caption_id)
    return ["common" if key in common else "rare" for key in keys]


def draw_sample(table, normal=12, other=4, seed=0):
    """(sheet, key): per structure, `normal` rows predicted normal and `other` rows of each other flag, shared out in
    turn over the strata of that group."""
    table = table.assign(conflict=table.conflicts != "[]", wording=_wording(table))
    rng = random.Random(f"radar-audit:{seed}")
    strata = {key: rows.index.tolist() for key, rows in table.groupby(STRATUM, sort=True)}
    groups = {}
    for key, positions in strata.items():
        remaining = list(positions)
        rng.shuffle(remaining)
        groups.setdefault(key[:2], []).append((key, remaining))
    picked = {}
    for (structure, flag), members in sorted(groups.items()):
        rng.shuffle(members)
        quota = normal if flag == "normal" else other
        while quota and any(remaining for _, remaining in members):
            for key, remaining in members:
                if quota and remaining:
                    picked.setdefault(key, []).append(remaining.pop())
                    quota -= 1
    if not picked:
        raise SystemExit("Nothing to sample.")
    group_rows = table.groupby(["structure", "normality"]).size()
    rows = []
    for stratum, positions in sorted(picked.items()):
        for position in positions:
            row = table.loc[position]
            rows.append({**{name: row[name] for name in STRATUM}, "caption": row.caption,
                         "policy_digest": row.policy_digest, "stratum_rows": len(strata[stratum]),
                         "stratum_sampled": len(positions), "group_rows": int(group_rows[stratum[:2]])})
    key = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    key.insert(0, "audit_id", [audit_id(s, c) for s, c in zip(key.structure, key.caption)])
    sheet = key.drop_duplicates("audit_id")[["audit_id", "structure", "caption"]].reset_index(drop=True)
    return sheet.assign(reader_flag="", reader_note=""), key


def audit_id(structure, caption):
    """An id that is the text's own: another sample cannot give it to another text."""
    return "A" + hashlib.sha256(f"{structure}\0{caption}".encode()).hexdigest()[:12]


def _shared_out(first, second, rows, seed):
    """`rows` rows of two groups, half of them from `first`; a short side gives its share to the other."""
    take = min(len(first), max(rows // 2, rows - len(second)))
    return pd.concat([first.sample(take, random_state=seed),
                      second.sample(min(len(second), rows - take), random_state=seed)])


def fidelity_sample(table, exclusions, rows=100, seed=0):
    """Rows that lost text, each distinct (structure, source, caption) once, in random order. A quarter are cells of
    the exclusion ledger, which have no caption, and half of those lost unexplained text. Half of the others are rows
    that claim nothing of substance left: the claims this sheet exists to test."""
    gone = exclusions.drop_duplicates(["structure", "source"]).assign(caption="")
    picked_gone = _shared_out(gone[gone.reason == "stranded_text"], gone[gone.reason != "stranded_text"], rows // 4,
                              seed)
    changed = table[table.dropped_parts.astype(int) > 0].drop_duplicates(["structure", "source", "caption"])
    quiet = changed[changed.stranded_parts.astype(int) == 0]
    counted = changed[changed.stranded_parts.astype(int) > 0]
    picked = pd.concat([_shared_out(quiet, counted, rows - len(picked_gone), seed), picked_gone])
    picked = picked.sample(frac=1.0, random_state=seed)[["structure", "source", "caption"]].reset_index(drop=True)
    return picked.assign(reader_fidelity="", reader_note="")


def wilson(p, n, z=1.96):
    """The Wilson interval of a proportion `p` over an effective `n`; None without data."""
    if not n:
        return None
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _weighted(rows, positive):
    """Weighted proportion of `positive` rows, its effective sample size, Wilson interval and reach."""
    if rows.empty:
        return {"estimate": None, "effective_n": 0, "interval_95": None}
    weights = rows.stratum_rows / rows.stratum_sampled
    estimate = float((weights * positive).sum() / weights.sum())
    effective = float(weights.sum() ** 2 / (weights ** 2).sum())
    return {"estimate": round(estimate, 4), "effective_n": round(effective, 1),
            "interval_95": wilson(estimate, effective),
            "rows_in_sampled_strata": int(rows.drop_duplicates(STRATUM).stratum_rows.sum()),
            "rows_in_group": int(rows.group_rows.iloc[0])}


def _texts(rows, reader):
    """How many distinct sampled texts the reader judged `reader`."""
    return int(rows[rows.reader == reader].audit_id.nunique())


def score(key, sheet):
    """The audit report of a filled sheet."""
    missing = {"audit_id", "structure", "caption", "reader_flag"} - set(sheet.columns)
    if missing:
        raise SystemExit(f"The sheet has no {sorted(missing)} column.")
    if sheet.audit_id.duplicated().any() or set(key.audit_id) != set(sheet.audit_id):
        raise SystemExit("The sheet and the key do not hold the same audit ids.")
    texts = {r.audit_id: (r.structure, r.caption.strip()) for r in key.drop_duplicates("audit_id").itertuples()}
    moved = sum(texts[r.audit_id] != (r.structure, r.caption.strip()) for r in sheet.itertuples())
    if moved:
        raise SystemExit(f"{moved} audit ids hold another text than in the key: this sheet is not this key's.")
    flags = sheet.reader_flag.fillna("").astype(str).str.strip().str.lower()
    if (flags == "").any():
        raise SystemExit(f"{int((flags == '').sum())} captions have no reader_flag: finish the sheet first.")
    bad = int((~flags.isin(READER_FLAGS)).sum())
    if bad:
        raise SystemExit(f"{bad} reader_flag values are not one of {READER_FLAGS}.")
    if key.policy_digest.nunique() != 1:
        raise SystemExit("The key mixes flags from several policies.")
    merged = key.assign(reader=key.audit_id.map(dict(zip(sheet.audit_id, flags))))
    report = {"policy_digest": key.policy_digest.iloc[0], "captions_read": int(key.audit_id.nunique()),
              "rows_sampled": len(key), "reader_question": READER_QUESTION, "structures": {}}
    for structure, rows in merged.groupby("structure", sort=True):
        normal, abnormal = rows[rows.normality == "normal"], rows[rows.normality == "abnormal"]
        unknown = rows[rows.normality == "unknown"]
        report["structures"][structure] = {
            "predicted_normal": {
                "captions_read": int(normal.audit_id.nunique()), "false_normal": _texts(normal, "abnormal"),
                "unclear": _texts(normal, "unclear"),
                "precision_unclear_counted_as_error": _weighted(normal, normal.reader == "normal")},
            "predicted_abnormal": {
                "captions_read": int(abnormal.audit_id.nunique()), "false_abnormal": _texts(abnormal, "normal"),
                "unclear": _texts(abnormal, "unclear")},
            "predicted_unknown": {"captions_read": int(unknown.audit_id.nunique()),
                                  "readers_said": {r: _texts(unknown, r) for r in READER_FLAGS}},
            "false_normal_by_evidence": {k: _texts(v, "abnormal") for k, v in normal.groupby("evidence")},
            "false_normal_near_a_cutoff": _texts(normal[normal.boundary.astype(bool)], "abnormal"),
            "false_normal_by_wording": {k: _texts(v, "abnormal") for k, v in normal.groupby("wording")}}
    return report


def fidelity_counts(sheet):
    """How many pairs of a filled source sheet the reader judged each way."""
    answers = sheet.reader_fidelity.fillna("").astype(str).str.strip().str.lower()
    bad = int((~answers.isin(FIDELITY)).sum())
    if bad:
        raise SystemExit(f"{bad} reader_fidelity values are not one of {FIDELITY}: finish the sheet first.")
    counts = {"pairs_read": len(sheet), **{k: int((answers == k).sum()) for k in FIDELITY}}
    if "caption" in sheet.columns:
        gone = sheet.caption.fillna("").astype(str).str.strip() == ""
        counts["removed_in_full"] = {"pairs_read": int(gone.sum()),
                                     **{k: int((answers[gone] == k).sum()) for k in FIDELITY}}
    return counts


def read_sheet(path):
    """A filled sheet as a spreadsheet saves it: with or without a byte order mark, with commas or semicolons."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            header = f.readline()
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                           sep=";" if header.count(";") > header.count(",") else ",")
    except UnicodeDecodeError:
        raise SystemExit(f"{path} is not UTF-8: save it as CSV UTF-8.")


def checked_table(args, routing, policy):
    """(caption table, exclusion ledger, build info), if they are what the builder wrote for this run and they follow
    from the rules."""
    try:
        info = check_caption_build(args.captions, routing, args.clips, args.split)
    except ValueError as error:
        raise SystemExit(str(error))
    table = read_captions(args.captions)
    if table.empty:
        raise SystemExit("The caption table is empty.")
    problems = table_problems(table, policy)
    if problems:
        raise SystemExit(f"The caption table does not match these rules and this policy: {problems}. Rebuild it.")
    exclusions = read_exclusions(args.captions, info)
    problems = exclusion_problems(exclusions, table)
    if problems:
        raise SystemExit(f"The exclusion ledger does not match these rules: {problems}. Rebuild it.")
    return table, exclusions, info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--min-count", type=int, default=5)
    s = sub.add_parser("sample")
    s.add_argument("--normal", type=int, default=12, help="Predicted normal captions per structure.")
    s.add_argument("--other", type=int, default=4, help="Captions of each other flag per structure.")
    s.add_argument("--removed", type=int, default=100, help="Pairs of source and caption for the source sheet.")
    for command in (c, s):
        command.add_argument("--captions", required=True)
        command.add_argument("--clips", required=True, help="The clip table the captions were built from.")
        command.add_argument("--split", default="train")
        command.add_argument("--num-clips", type=int, default=12)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--routing", default=DEFAULT_ROUTING)
    r = sub.add_parser("score")
    r.add_argument("--key", required=True)
    r.add_argument("--sheet", required=True)
    r.add_argument("--fidelity", help="The filled source sheet.")
    for command in (c, s, r):
        command.add_argument("--policy", default=rn.DEFAULT_POLICY)
        command.add_argument("--out-dir", required=True, help="Output folder; must be outside this repository.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    policy = rn.load_policy(args.policy)
    if args.command != "score":
        routing = load_routing(args.routing)
        table, exclusions, info = checked_table(args, routing, policy)
        clips = pd.read_csv(args.clips, dtype={"study_id": str, "clip_id": str}, keep_default_na=False)
    if args.command == "coverage":
        report = coverage(table, routing, policy, clips, args.num_clips, args.seed, args.split, exclusions,
                          info["studies_linked"])
        with open(os.path.join(args.out_dir, "coverage.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        unrecognized_wording(table, args.min_count).to_csv(os.path.join(args.out_dir, "unrecognized_wording.csv"),
                                                           index=False)
        print(f"unknown fraction: {report['unknown_fraction']}")
    elif args.command == "sample":
        if os.path.exists(os.path.join(args.out_dir, "audit_key.csv")):
            raise SystemExit(f"{args.out_dir} already holds an audit key, and its sheet belongs to it. Use another.")
        eligible = split_eligibility(table, clips, routing, args.num_clips, args.seed, args.split)
        rows = trainable(table, eligible)
        sheet, key = draw_sample(rows, args.normal, args.other, args.seed)
        sheet.to_csv(os.path.join(args.out_dir, "audit_sheet.csv"), index=False, encoding="utf-8-sig")
        key.to_csv(os.path.join(args.out_dir, "audit_key.csv"), index=False)
        fidelity = fidelity_sample(rows, trainable(exclusions, eligible), args.removed, args.seed)
        fidelity.to_csv(os.path.join(args.out_dir, "fidelity_sheet.csv"), index=False, encoding="utf-8-sig")
        print(f"{len(sheet)} captions. Question for the reader: {READER_QUESTION}")
    else:
        key = pd.read_csv(args.key, dtype={"audit_id": str}, keep_default_na=False)
        if set(key.policy_digest) != {policy.digest}:
            raise SystemExit("The key was not drawn from the current rules and policy: draw a new sample.")
        report = score(key, read_sheet(args.sheet))
        if args.fidelity:
            report["source_fidelity"] = fidelity_counts(read_sheet(args.fidelity))
        with open(os.path.join(args.out_dir, "audit_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"{report['captions_read']} captions scored; policy_digest {report['policy_digest']}")


if __name__ == "__main__":
    main()
