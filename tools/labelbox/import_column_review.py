"""Apply the v2 ontology and prelabels to an explicitly selected video project.

Existing data rows only. No asset upload, ground-truth import, invitations, or
provisioning. Dry run is the default; API credentials stay in the environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools.labelbox.column_review import build_form, context_attachment, prelabels

ONTOLOGY_NAME = "DeepECHO column review v2"


def schema_shape(value):
    """Ignore server-assigned identifiers while checking the full feature tree."""
    if isinstance(value, list):
        return [schema_shape(v) for v in value]
    if isinstance(value, dict):
        keys = {"tools", "classifications", "type", "name", "instructions", "required",
                "options", "label", "value", "scope", "uiMode"}
        return {k: schema_shape(v) for k, v in value.items() if k in keys}
    return value


def apply_bundle(client, project_id, candidates, *, mal_import):
    """Check every row and schema before the first remote mutation."""
    import labelbox as lb

    payloads = [p for c in candidates for p in prelabels(c)]
    keys = [c["global_key"] for c in candidates]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Candidates must have distinct global keys")
    project = client.get_project(project_id)
    if project.media_type != lb.MediaType.Video:
        raise ValueError("The selected project must use video assets")
    rows = [client.get_data_row_by_global_key(key) for key in keys]
    if any(not row.media_attributes or not row.media_attributes.get("frameCount") for row in rows):
        raise ValueError("All existing rows must be video assets")
    form = build_form()
    expected = schema_shape(form)
    attached = project.ontology()
    if attached is not None and not attached.normalized.get("tools") and not attached.normalized.get("classifications") and project.data_row_count == 0:
        attached = None
    if attached is not None and schema_shape(attached.normalized) != expected:
        raise ValueError("Selected project has a different ontology; use a v2 project")
    existing = list(client.get_ontologies(ONTOLOGY_NAME))
    exact = [o for o in existing if o.name == ONTOLOGY_NAME]
    if len(exact) > 1:
        raise ValueError("Multiple v2 ontologies found")
    if exact and schema_shape(exact[0].normalized) != expected:
        raise ValueError("Named v2 ontology has a different schema")
    digest = hashlib.sha256(json.dumps(payloads, sort_keys=True).encode()).hexdigest()[:16]
    name = "deepecho-v2-"+digest
    jobs = [j for j in project.get_mal_prediction_imports() if j.name == name]
    batches = [b for b in project.batches() if b.name == name]
    if len(jobs) > 1 or len(batches) > 1:
        raise ValueError("Multiple matching v2 import jobs or batches")
    for candidate, row in zip(candidates, rows):
        # A duplicate context name with different content is an audit mismatch.
        desired = context_attachment(candidate)
        matching = [a for a in row.attachments() if a.attachment_name == desired["attachment_name"]]
        if len(matching) > 1 or (matching and matching[0].attachment_value != desired["attachment_value"]):
            raise ValueError("Existing original-proposal context differs")
    ontology = attached or (exact[0] if exact else client.create_ontology(
        ONTOLOGY_NAME, form, media_type=lb.MediaType.Video))
    if attached is None:
        project.connect_ontology(ontology)
    if not batches and not jobs:
        project.create_batch(name=name, global_keys=keys)
    for candidate, row in zip(candidates, rows):
        desired = context_attachment(candidate)
        if not any(a.attachment_name == desired["attachment_name"] for a in row.attachments()):
            row.create_attachment(attachment_type=desired["attachment_type"],
                                  attachment_value=desired["attachment_value"],
                                  attachment_name=desired["attachment_name"])
    # Stable names identify retries; no second job is made when it already exists.
    job = jobs[0] if jobs else mal_import.create_from_objects(
        client=client, project_id=project_id, name=name, predictions=payloads)
    job.wait_until_done()
    if job.errors:
        raise RuntimeError("MAL import reported annotation errors")
    return {"project_id": project_id, "ontology_id": ontology.uid,
            "job_id": job.uid, "data_rows": len(rows), "classifications": len(payloads)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    candidates = json.loads(args.candidates.read_text())
    payloads = [p for c in candidates for p in prelabels(c)]
    if not args.apply:
        print(f"Dry run: v2 ontology, {len(candidates)} existing video rows, {len(payloads)} draft classifications.")
        print("No Labelbox requests. Review completed is not prefilled.")
        return 0
    key = os.environ.get("LABELBOX_API_KEY", "").strip()
    if not key:
        print("LABELBOX_API_KEY is required with --apply.", file=sys.stderr)
        return 2
    previous = logging.root.manager.disable
    try:
        import labelbox as lb
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            result = apply_bundle(lb.Client(api_key=key), args.project_id, candidates, mal_import=lb.MALPredictionImport)
    except Exception as exc:
        print(f"Labelbox v2 import failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
