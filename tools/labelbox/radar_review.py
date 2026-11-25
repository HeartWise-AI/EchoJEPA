"""Local export/import workflow for the RADAR caption review (no Labelbox requests are made).

    python -m tools.labelbox.radar_review ontology [--out FILE]
    python -m tools.labelbox.radar_review payload --records RECORDS.jsonl [--out-dir DIR] [--video-url-template URL]
    python -m tools.labelbox.radar_review import-export --records RECORDS.jsonl --export EXPORT.ndjson \
        --review-version VERSION --out UPDATED.jsonl [--reviewer-map MAP.json] [--project-id ID]

`payload` is a dry run unless `--out-dir` is given: it builds and checks the data rows and caption prelabels and
prints only counts. Uploading is deliberately not implemented: it needs an approved workspace, approved
de-identified video storage (to replace the placeholder video URL) and a reviewed upload change. Outputs that hold
captions or identifiers must be written outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from app.radar.ontology import DEFAULT_ONTOLOGY_PATH, load_ontology
from app.radar.records import SPLITS, ensure_outside_repo, read_records, write_records
from tools.labelbox.payloads import PLACEHOLDER_URL_TEMPLATE, build_data_rows, build_prelabels
from tools.labelbox.review_export import apply_export, read_export
from tools.labelbox.review_ontology import build_review_ontology, review_version


def _write_ndjson(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RADAR caption review: local Labelbox payloads and imports.")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY_PATH, help="concept ontology YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ontology", help="print or write the Labelbox review form")
    p.add_argument("--out", help="write the form JSON here instead of printing it")

    p = sub.add_parser("payload", help="build data rows and caption prelabels (dry run without --out-dir)")
    p.add_argument("--records", required=True)
    p.add_argument("--out-dir", help="write data_rows.ndjson and prelabels.ndjson here (outside the repository)")
    p.add_argument("--video-url-template", default=PLACEHOLDER_URL_TEMPLATE,
                   help="https URL with {global_key}; the default is a placeholder that never resolves")
    p.add_argument("--language", default="fr", choices=("fr", "en"))
    p.add_argument("--splits", nargs="+", default=["train"], choices=SPLITS,
                   help="splits allowed in the payload (default: train; the RADAR pilot uses training studies)")

    p = sub.add_parser("import-export", help="map a Labelbox export back onto the records")
    p.add_argument("--records", required=True)
    p.add_argument("--export", required=True, help="Labelbox export NDJSON")
    p.add_argument("--review-version", required=True, help="the review form version the project was built with")
    p.add_argument("--out", required=True, help="updated records JSONL (outside the repository)")
    p.add_argument("--reviewer-map", help="JSON object mapping reviewer accounts to pseudonymous handles")
    p.add_argument("--project-id", help="the Labelbox project to read when rows belong to several")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ontology = load_ontology(args.ontology)

    if args.command == "ontology":
        form = build_review_ontology(ontology)
        text = json.dumps(form, ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"Wrote review form {form['review_version']} with {len(form['classifications'])} questions.")
        else:
            print(text)
        return 0

    if args.command == "payload":
        records = read_records(args.records, ontology)
        rows = build_data_rows(records, ontology, language=args.language, url_template=args.video_url_template,
                               allowed_splits=tuple(args.splits))
        prelabels = build_prelabels(records, allowed_splits=tuple(args.splits))
        print(f"Checked {len(rows)} data rows and {len(prelabels)} caption prelabels "
              f"(review form {review_version(ontology)}).")
        if args.video_url_template == PLACEHOLDER_URL_TEMPLATE:
            print("Video URLs are placeholders: replace them with an approved de-identified location before upload.")
        if not args.out_dir:
            print("Dry run: nothing written. Pass --out-dir to write the NDJSON files.")
            return 0
        ensure_outside_repo(args.out_dir, "--out-dir")
        os.makedirs(args.out_dir, exist_ok=True)
        _write_ndjson(os.path.join(args.out_dir, "data_rows.ndjson"), rows)
        _write_ndjson(os.path.join(args.out_dir, "prelabels.ndjson"), prelabels)
        print("Wrote data_rows.ndjson and prelabels.ndjson. No upload was made.")
        return 0

    ensure_outside_repo(args.out, "--out")
    records = read_records(args.records, ontology)
    reviewer_map = None
    if args.reviewer_map:
        with open(args.reviewer_map, encoding="utf-8") as f:
            reviewer_map = json.load(f)
    updated, counts = apply_export(records, read_export(args.export), ontology,
                                   expected_review_version=args.review_version, project_id=args.project_id,
                                   reviewer_map=reviewer_map)
    write_records(args.out, updated, ontology)
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError) as e:  # validation errors carry no values, paths or identifiers
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)
