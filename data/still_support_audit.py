# data/still_support_audit.py

"""How often a Doppler caption states a finding that none of the study's stills can show. Counts only.

Still routing is by structure: an LVOT still feeds the aortic valve, so a bag holding only an LVOT still switches on
the aortic valve's Doppler caption, which may speak of aortic stenosis, although the stenosis is read on the AV
continuous-wave still, not on the LVOT one. This is a known limitation of the first experiments. This tool measures
it, so that the decision to keep structure-level routing or to move to finding-level support rests on a number.

`FINDING_SUPPORT` is the finding-support table: per structure, each Doppler finding, the wording that states it
(negated or not: `pas de sténose` is a statement about stenosis too) and the still tags that show it. For every
(study, structure) with a Doppler caption (`scope` full or doppler) and at least one still that feeds the structure:

- `stated`: captions that state the finding;
- `supported`: of those, the study has a still of a supporting tag;
- `unsupported`: of those, it has none: every bag of the study that reads this caption trains the statement
  against stills that cannot show it;
- `unsupported_by_tags`: the unsupported ones by the feeding tags the study does have (for example `LVOT`).

The study's whole set of stills is used, which is the best case: a bag draws at most `num_stills` of them, so the
share of mismatched bags is at least the share reported here. Counts from 1 to 4 are written `<5`. If `unsupported`
is a small share of `stated`, document it and go on; if it is frequent, write the Doppler clauses per finding and
give the dataset this table, as it now does for Hemo's fields (`caption_evidence`).

Usage:
    python data/still_support_audit.py --captions <captions.csv built with --doppler-captions> \\
        --stills <stills.csv> --split train --out <folder outside the repository>/still_support.json
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd
from build_manifests import REPO_ROOT, ensure_outside_repo

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from src.models.radar.routing import load_routing  # noqa: E402

import radar_doppler_captions as rdc  # noqa: E402

STENOSIS, REGURGITATION = rdc.FINDING_PATTERNS["stenosis"], rdc.FINDING_PATTERNS["regurgitation"]
FINDING_SUPPORT = {
    "aortic": {"stenosis": (STENOSIS, ("AV", "AVA")), "regurgitation": (REGURGITATION, ("AR",))},
    "mitral": {"stenosis": (STENOSIS, ("MV", "MVA")), "regurgitation": (REGURGITATION, ("MR",))},
    "tricuspid": {"stenosis": (STENOSIS, ("TV",)), "regurgitation": (REGURGITATION, ("TR",))},
    "pulmonic": {"stenosis": (STENOSIS, ("PV",)), "regurgitation": (REGURGITATION, ("PR",))},
}


def plain(text):
    """Lower case without accents, as the wording patterns expect."""
    return "".join(c for c in unicodedata.normalize("NFD", str(text).lower()) if unicodedata.category(c) != "Mn")


def suppressed(n):
    return "<5" if 0 < n < 5 else int(n)


def audit(captions, stills, routing):
    """{structure: {finding: counts}} (see the module docstring). `captions`: study_id, structure, caption, scope;
    `stills`: study_id, still_tag, of one split."""
    tags_of = stills[stills.still_tag.astype(str) != ""].groupby("study_id").still_tag.agg(lambda t: set(map(str, t)))
    tags_of = tags_of.to_dict()
    feeding = {name: {tag for tag, targets in routing.still_structs.items() if name in targets}
               for name in FINDING_SUPPORT}
    rows = captions[captions.scope.isin(["full", "doppler"]) & captions.structure.isin(list(FINDING_SUPPORT))]
    rows = rows.drop_duplicates(["study_id", "structure"])        # full and doppler rows hold the same clauses
    report = {}
    for name, findings in FINDING_SUPPORT.items():
        mine = rows[rows.structure == name]
        entry = {"captions_with_feeding_stills": 0}
        counts = {f: Counter() for f in findings}
        by_tags = {f: Counter() for f in findings}
        for study, caption in zip(mine.study_id.astype(str), mine.caption):
            fed = tags_of.get(study, set()) & feeding[name]
            if not fed:
                continue
            entry["captions_with_feeding_stills"] += 1
            text = plain(caption)
            for finding, (pattern, supporting) in findings.items():
                if not re.search(pattern, text):
                    continue
                counts[finding]["stated"] += 1
                if fed & set(supporting):
                    counts[finding]["supported"] += 1
                else:
                    counts[finding]["unsupported"] += 1
                    by_tags[finding]["+".join(sorted(fed))] += 1
        entry["captions_with_feeding_stills"] = suppressed(entry["captions_with_feeding_stills"])
        for finding in findings:
            stated, unsupported = counts[finding]["stated"], counts[finding]["unsupported"]
            entry[finding] = {"stated": suppressed(stated), "supported": suppressed(counts[finding]["supported"]),
                              "unsupported": suppressed(unsupported),
                              "unsupported_share": round(unsupported / stated, 3) if stated >= 5 else None,
                              "unsupported_by_tags": {k: suppressed(v) for k, v in sorted(by_tags[finding].items())}}
        report[name] = entry
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captions", required=True, help="Caption table built with --doppler-captions (has `scope`).")
    p.add_argument("--stills", required=True, help="The still table (study_id, still_tag, split).")
    p.add_argument("--split", default="train")
    p.add_argument("--routing", required=True, help="The registry the captions were built with (version 1).")
    p.add_argument("--out", required=True, help="The JSON report; outside this repository.")
    args = p.parse_args(argv)
    ensure_outside_repo(os.path.dirname(os.path.abspath(args.out)), "--out")
    captions = pd.read_csv(args.captions, dtype={"study_id": str}, keep_default_na=False)
    if "scope" not in captions.columns:
        raise SystemExit("--captions has no scope column: build it with --doppler-captions.")
    stills = pd.read_csv(args.stills, dtype={"study_id": str, "still_tag": str}, keep_default_na=False)
    stills = stills[stills.split.astype(str) == args.split]
    report = {"split": args.split, "structures": audit(captions, stills, load_routing(args.routing))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
