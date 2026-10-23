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
# The form the predictions are sent in: 1 named questions and answers, 2 their schema ids. An import job is named
# after the prelabels and this form, so a job of an earlier form, which may have failed, never stands for this one.
PREDICTION_FORMAT = 2


class ImportStopped(RuntimeError):
    """A stop whose message names only job ids and what to do, so it is safe to print."""


def schema_shape(value):
    """Ignore server-assigned identifiers while checking the full feature tree."""
    if isinstance(value, list):
        return [schema_shape(v) for v in value]
    if isinstance(value, dict):
        keys = {"tools", "classifications", "type", "name", "instructions", "required",
                "options", "label", "value", "scope", "uiMode"}
        return {k: schema_shape(v) for k, v in value.items() if k in keys}
    return value


def comparable_shape(value):
    """`schema_shape` of a form as Labelbox returns it: the ontology's own name at the top and empty option lists,
    which Labelbox adds or leaves out, do not count."""
    def strip(v):
        if isinstance(v, list):
            return [strip(x) for x in v]
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items() if not (k == "options" and x == [])}
        return v

    shape = schema_shape(value)
    return strip({k: v for k, v in shape.items() if k != "name"} if isinstance(shape, dict) else shape)


def with_schema_ids(records, classifications):
    """Name-based prelabels rewritten with the schema ids of an ontology's classifications, as Labelbox returns them.
    An empty checklist answer is dropped; a name the ontology lacks is refused."""
    out = []
    for record in records:
        matches = [q for q in classifications if q.get("name") == record["name"]]
        if len(matches) != 1 or not matches[0].get("featureSchemaId"):
            raise ValueError(f"The ontology has no unique schema id for {record['name']!r}")
        question, answer = matches[0], record["answer"]
        if question.get("type") == "checklist" and not answer:
            continue
        rewritten = {k: v for k, v in record.items() if k not in ("name", "answer")}
        rewritten["schemaId"] = question["featureSchemaId"]
        if question.get("type") == "text":
            rewritten["answer"] = answer
        elif question.get("type") == "checklist":
            rewritten["answer"] = [_option_with_schema_id(a, question) for a in answer]
        else:
            rewritten["answer"] = _option_with_schema_id(answer, question)
        out.append(rewritten)
    return out


def _option_with_schema_id(answer, question):
    matches = [o for o in question.get("options", []) if o.get("value") == answer["name"]]
    if len(matches) != 1 or not matches[0].get("featureSchemaId"):
        raise ValueError(f"The ontology has no unique schema id for {question['name']!r} = {answer['name']!r}")
    out = {"schemaId": matches[0]["featureSchemaId"]}
    if answer.get("classifications"):
        out["classifications"] = with_schema_ids(answer["classifications"], matches[0].get("options", []))
    return out


def import_outcome(job):
    """'done' when an import job finished without annotation errors, 'failed' when it failed or reported some, and
    'unreadable' when its errors can no longer be read (Labelbox keeps them 24 hours)."""
    job.wait_till_done()
    if getattr(getattr(job, "state", None), "name", "") == "FAILED":
        return "failed"
    try:
        return "failed" if job.errors else "done"
    except Exception:
        return "unreadable"


def apply_bundle(client, project_id, candidates, *, mal_import, retry_failed=False):
    """Check every row, schema and earlier import job before the first remote mutation.

    The batch is named after the prelabels; each import job of them after the prelabels and PREDICTION_FORMAT. A
    job that finished cleanly is reused, whatever its form: its prelabels are in. A failed job of an earlier form
    is superseded. A failed job of this form, or one whose errors can no longer be read, stops the import unless
    `retry_failed`, which submits a new attempt under a new name."""
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
    expected = comparable_shape(form)
    attached = project.ontology()
    if attached is not None and not attached.normalized.get("tools") and not attached.normalized.get("classifications") and project.data_row_count == 0:
        attached = None
    if attached is not None and comparable_shape(attached.normalized) != expected:
        raise ValueError("Selected project has a different ontology; use a v2 project")
    existing = list(client.get_ontologies(ONTOLOGY_NAME))
    exact = [o for o in existing if o.name == ONTOLOGY_NAME]
    if len(exact) > 1:
        raise ValueError("Multiple v2 ontologies found")
    if exact and comparable_shape(exact[0].normalized) != expected:
        raise ValueError("Named v2 ontology has a different schema")
    digest = hashlib.sha256(json.dumps(payloads, sort_keys=True).encode()).hexdigest()[:16]
    name = "deepecho-v2-"+digest  # the batch; before PREDICTION_FORMAT, the import job too.
    job_name = f"{name}-f{PREDICTION_FORMAT}"
    jobs = [j for j in project.get_mal_prediction_imports() if j.name == name or j.name.startswith(name + "-")]
    batches = [b for b in project.batches() if b.name == name]
    if len(batches) > 1:
        raise ValueError("Multiple matching v2 batches")
    for candidate, row in zip(candidates, rows):
        # A duplicate context name with different content is an audit mismatch.
        desired = context_attachment(candidate)
        matching = [a for a in row.attachments() if a.attachment_name == desired["attachment_name"]]
        if len(matching) > 1 or (matching and matching[0].attachment_value != desired["attachment_value"]):
            raise ValueError("Existing original-proposal context differs")
    outcomes = [(job, import_outcome(job)) for job in jobs]
    done = [job for job, outcome in outcomes if outcome == "done"]
    attempts = [job for job in jobs if job.name == job_name or job.name.startswith(job_name + "-r")]
    blocking = [job for job, outcome in outcomes
                if outcome == "unreadable" or (outcome == "failed" and job in attempts)]
    if not done and blocking and not retry_failed:
        raise ImportStopped(
            f"Earlier import jobs of these prelabels failed or no longer show their errors "
            f"({', '.join(job.uid for job in blocking)}); nothing was changed. Check them in Labelbox (one that "
            "succeeded holds the prelabels already), then rerun with --retry-failed to import them again.")
    ontology = attached or (exact[0] if exact else client.create_ontology(
        ONTOLOGY_NAME, form, media_type=lb.MediaType.Video))
    predictions = with_schema_ids(payloads, ontology.normalized["classifications"])
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
    if done:  # no second import of prelabels that are in.
        job = done[0]
    else:
        attempt = f"{job_name}-r{len(attempts) + 1}" if attempts else job_name
        job = mal_import.create_from_objects(client=client, project_id=project_id, name=attempt,
                                             predictions=predictions)
        if import_outcome(job) != "done":
            raise ImportStopped(f"Import job {job.uid} failed or reported annotation errors; check it in Labelbox.")
    return {"project_id": project_id, "ontology_id": ontology.uid,
            "job_id": job.uid, "data_rows": len(rows), "classifications": len(predictions)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="import again after a failed import job of the same prelabels, under a new job name.")
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
            result = apply_bundle(lb.Client(api_key=key), args.project_id, candidates,
                                  mal_import=lb.MALPredictionImport, retry_failed=args.retry_failed)
    except ImportStopped as exc:
        print(f"Labelbox v2 import stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Labelbox v2 import failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
