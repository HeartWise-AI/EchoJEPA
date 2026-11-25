"""Run the RADAR caption pipeline on the synthetic demo set in configs/radar/demo/.

    python -m app.radar.demo                    # rule-based captions, one LLM prompt, LLM parse outcomes, coverage
    python -m app.radar.demo --write-expected   # regenerate demo_expected.json after an intended change

The demo records are synthetic (no patient, study or report data) and their video paths are placeholders, so this
is the one output the repository may hold.
"""

from __future__ import annotations

import argparse
import json
import os

from app.radar.captions import LLMOutputError, build_llm_prompt, parse_llm_output, rule_based_caption
from app.radar.coverage import build_ledger, summarize
from app.radar.ontology import REPO_ROOT, load_ontology
from app.radar.records import read_records

DEMO_DIR = os.path.join(REPO_ROOT, "configs", "radar", "demo")
DEMO_RECORDS = os.path.join(DEMO_DIR, "demo_records.jsonl")
DEMO_LLM_OUTPUTS = os.path.join(DEMO_DIR, "demo_llm_outputs.json")
DEMO_EXPECTED = os.path.join(DEMO_DIR, "demo_expected.json")
DEMO_TIME = "2026-01-01T00:00:00+00:00"  # fixed so outputs are reproducible


def expected_outputs(ontology, records) -> dict:
    out = {"ontology_version": ontology.version, "records": []}
    for r in records:
        entry = {"record_id": r.record_id, "view": r.view, "acquisition": r.acquisition.category}
        for language in ("fr", "en"):
            caption = rule_based_caption(r, ontology, language=language, created_utc=DEMO_TIME)
            entry[f"caption_{language}"] = caption.text
        entry["claims"] = [[c.concept_id, c.value] for c in caption.claims]
        entry["gaps"] = [[g.concept_id, g.reason] for g in caption.gaps]
        entry["out_of_view"] = list(caption.out_of_view)
        entry["expected_structures"] = list(caption.expected_structures)
        out["records"].append(entry)
    return out


def llm_outcomes(ontology, records) -> list[dict]:
    with open(DEMO_LLM_OUTPUTS, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    results = []
    for case in cases:
        record = records[case["record"]]
        try:
            caption = parse_llm_output(case["response"], record, ontology, model_id="synthetic-demo",
                                       language=case["language"], created_utc=DEMO_TIME)
            results.append({"name": case["name"], "outcome": "accepted",
                            "claims": [[c.concept_id, c.value] for c in caption.claims]})
        except LLMOutputError as e:
            results.append({"name": case["name"], "outcome": "rejected", "error": str(e)})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-expected", action="store_true", help="rewrite configs/radar/demo/demo_expected.json")
    args = parser.parse_args(argv)
    ontology = load_ontology()
    records = read_records(DEMO_RECORDS, ontology)
    expected = expected_outputs(ontology, records)
    if args.write_expected:
        with open(DEMO_EXPECTED, "w", encoding="utf-8") as f:
            json.dump(expected, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {len(expected['records'])} expected demo captions.")
        return 0
    for entry in expected["records"]:
        print(f"[{entry['view']} / {entry['acquisition']}] {entry['caption_fr']}")
        if entry["gaps"]:
            print("    gaps: " + ", ".join(f"{c} ({r})" for c, r in entry["gaps"]))
    prompt = build_llm_prompt(records[1], ontology)
    print(f"\nLLM prompt {prompt['prompt_version']} for record 1, user message:\n{prompt['user']}\n")
    for result in llm_outcomes(ontology, records):
        print(f"LLM case {result['name']}: {result['outcome']} {result.get('error', '')}".rstrip())
    for r in records:
        r.candidate_caption = rule_based_caption(r, ontology, created_utc=DEMO_TIME)
    print("\nCoverage summary:\n" + json.dumps(summarize(build_ledger(records, ontology)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
