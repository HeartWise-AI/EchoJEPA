"""Verify medical approval for versioned RADAR caption ontologies.

Each version has a YAML approval record at `docs/radar/approval_v<N>.md`. Pending records have empty approval
fields; approved records contain the reviewed Git commit, file hashes, medical sign-off, threshold agreement,
and Labelbox SDK version.

Approval requires all files to match their recorded hashes in both the reviewed commit and working tree, and
the loaded ontology to match the approved YAML. The commit must be an ancestor of `HEAD`. Missing Git history
invalidates approval, so CI must fetch the full history. Any file change requires a new ontology version and approval.

Usage:
    python -m app.radar.approval            # Check approval status.
    python -m app.radar.approval --hashes   # Generate file hashes, which will be written into the record at approval time.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import subprocess
import sys
from typing import Mapping

import yaml

from app.radar.ontology import REPO_ROOT, Ontology, load_ontology

RECORD_SCHEMA = "radar-caption-approval"
PENDING, APPROVED, MISSING = "pending", "approved", "missing"
MEDICAL_REVIEWER = "robertavram-md"         # GitHub login of the cardiologist who approves.
_YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_SIGN_OFF = re.compile(r"^https://github\.com/HeartWise-AI/EchoJEPA/(pull|issues)/\d+(#\S+)?$")


class ApprovalError(ValueError):
    """The approval record is malformed or no longer matches the files it approves."""


def approved_files(ontology: Ontology) -> tuple[str, ...]:
    """Repository paths a medical approval covers, for this ontology version."""
    v = ontology.version
    return (
        f"configs/radar/{ontology.tag}.yaml",
        f"configs/radar/label_mappings_v{v}.yaml",
        f"configs/radar/review_examples_v{v}.yaml",
        f"configs/radar/labelbox/{ontology.tag}.json",
        f"docs/radar/ontology_v{v}.md",
        f"docs/radar/labeling_instructions_v{v}.md",
        f"docs/radar/labeling_instructions_v{v}.fr.md",
        f"docs/radar/pilot_metrics_v{v}.md",
    )


def record_path(ontology: Ontology, repo_root: str = REPO_ROOT) -> str:
    return os.path.join(repo_root, "docs", "radar", f"approval_v{ontology.version}.md")


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def current_hashes(ontology: Ontology, repo_root: str = REPO_ROOT) -> dict[str, str]:
    return {p: file_sha256(os.path.join(repo_root, p)) for p in approved_files(ontology)}


def read_record(path: str) -> dict:
    """Loads and parses the approval information from the `.md` document."""
    with open(path, encoding="utf-8") as f:
        # Extract YAML content.
        blocks = _YAML_BLOCK.findall(f.read())
    if len(blocks) != 1:
        raise ApprovalError(f"{os.path.basename(path)} must hold exactly one yaml block, not {len(blocks)}")
    try:
        # Parse YAML: converts YAML into python objects.
        record = yaml.safe_load(blocks[0]) or {}
    # PyYAML raises ValueError on an impossible unquoted date.
    except (yaml.YAMLError, ValueError) as e:
        raise ApprovalError(f"{os.path.basename(path)}: the record cannot be read: {e}") from None
    if not isinstance(record, Mapping):
        raise ApprovalError(f"{os.path.basename(path)}: the record must be a mapping")
    return record


def _is_date(value) -> bool:
    """A real calendar day written YYYY-MM-DD (YAML reads an unquoted one as a `datetime.date`, printed the same)."""
    try:
        return bool(_DATE.match(str(value))) and bool(datetime.date.fromisoformat(str(value)))
    except ValueError:
        return False


def record_problems(record: Mapping, ontology: Ontology, repo_root: str = REPO_ROOT) -> list[str]:
    """Receives the parsed approval record and the ontology being checked and returns a list of
    problems. An empty list means no violations were detected by these checks.

    Note: rather than immediately raising an exception for every issue, the function collects errors
    so the user can see multiple problems at once.
    """
    problems = []
    if record.get("schema") != RECORD_SCHEMA:
        problems.append(f"Schema must be {RECORD_SCHEMA!r}.")
    if record.get("ontology") != ontology.tag:
        problems.append(f"The record is for {record.get('ontology')!r}, not {ontology.tag!r}.")
    status = record.get("status")
    sections = {}
    for name in ("medical_approval", "thresholds_agreement", "labelbox_sdk_check"):
        sections[name] = record.get(name) or {}
        if not isinstance(sections[name], Mapping):
            problems.append(f"{name} must be a mapping.")
            sections[name] = {}
    medical, thresholds, sdk = sections.values()
    fields = {
        "content_commit": record.get("content_commit"), "files": record.get("files"),
        "medical_approval.reviewer": medical.get("reviewer"), "medical_approval.link": medical.get("link"),
        "medical_approval.date": medical.get("date"), "thresholds_agreement.link": thresholds.get("link"),
        "thresholds_agreement.date": thresholds.get("date"), "labelbox_sdk_check.version": sdk.get("version"),
    }
    if status == PENDING:
        filled = [name for name, value in fields.items() if value]
        if filled:
            problems.append(f"A pending record has no approval fields; filled: {filled}.")
        return problems
    if status != APPROVED:
        return problems + [f"Status must be {PENDING!r} or {APPROVED!r}, not {status!r}."]
    empty = [name for name, value in fields.items() if not value]
    if empty:
        problems.append(f"An approved record needs {empty}.")
    if fields["content_commit"] and not _COMMIT.match(str(fields["content_commit"])):
        problems.append("`content_commit` must be a full 40-character commit hash.")
    for name in ("medical_approval.link", "thresholds_agreement.link"):
        if fields[name] and not _SIGN_OFF.match(str(fields[name])):
            problems.append(f"{name} must link to a HeartWise-AI/EchoJEPA pull request or issue.")
    if fields["medical_approval.reviewer"] and fields["medical_approval.reviewer"] != MEDICAL_REVIEWER:
        problems.append(f"`medical_approval.reviewer` must be {MEDICAL_REVIEWER!r}.")
    for name in ("medical_approval.date", "thresholds_agreement.date"):
        if fields[name] and not _is_date(fields[name]):
            problems.append(f"{name} must be a calendar date written YYYY-MM-DD.")
    version = fields["labelbox_sdk_check.version"]
    if version and not _VERSION.match(str(version)):
        problems.append("`labelbox_sdk_check.version` must be a release number such as 7.12.0.")
    files = fields["files"] if isinstance(fields["files"], Mapping) else {}
    expected = approved_files(ontology)
    if set(files) != set(expected):
        problems.append(f"Files must be exactly {list(expected)}.")
    for path, digest in files.items():
        if path not in expected:  # reported above
            continue
        full = os.path.join(repo_root, path)
        if not _SHA256.match(str(digest)):
            problems.append(f"{path}: the recorded hash is not a SHA-256.")
        elif not os.path.exists(full):
            problems.append(f"{path}: approved but missing.")
        elif file_sha256(full) != digest:
            problems.append(f"{path}: changed since its approval; a change needs a new version and approval.")
    ontology_file = expected[0]  # the ontology YAML
    if files.get(ontology_file) and ontology.sha256 != files[ontology_file]:
        problems.append(f"The ontology checked was not loaded from the approved {ontology_file} (its SHA-256 differs).")
    if fields["content_commit"] and _COMMIT.match(str(fields["content_commit"])):
        problems += commit_problems(record, repo_root)
    return problems


def approval_state(ontology: Ontology, repo_root: str = REPO_ROOT) -> str:
    """Return the ontology's approval status: `approved`, `pending`, or `missing`.
    Raise `ApprovalError` if the record is invalid."""
    path = record_path(ontology, repo_root)
    if not os.path.exists(path):
        return MISSING
    record = read_record(path)
    problems = record_problems(record, ontology, repo_root)
    if problems:
        raise ApprovalError(f"{os.path.relpath(path, repo_root)}:\n- " + "\n- ".join(problems))
    return record["status"]


def require_approval(ontology: Ontology, repo_root: str = REPO_ROOT) -> None:
    """Verify that this exact ontology version is medically approved before a clinical upload.
    Raise `ApprovalError` if approval is missing, pending, or invalid."""
    state = approval_state(ontology, repo_root)
    if state != APPROVED:
        raise ApprovalError(f"{ontology.tag} is not approved (approval record: {state}).")


def _git(repo_root: str, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", repo_root, *args], capture_output=True)
    except OSError:  # no git executable: fail closed.
        return subprocess.CompletedProcess(args, 127, b"", b"")


def commit_problems(record: Mapping, repo_root: str = REPO_ROOT) -> list[str]:
    """The content commit must be an ancestor of `HEAD` and hold exactly the recorded files."""
    commit = str(record.get("content_commit"))
    files = record.get("files") if isinstance(record.get("files"), Mapping) else {}
    if _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        return [f"Content commit {commit} is not in this repository's history (CI must fetch the full history)."]
    if _git(repo_root, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return [f"Content commit {commit} is not an ancestor of `HEAD` (merge the pull request without squashing)."]
    problems = []
    for path, digest in files.items():
        shown = _git(repo_root, "show", f"{commit}:{path}")
        if shown.returncode != 0:
            problems.append(f"{path}: not in the content commit.")
        elif hashlib.sha256(shown.stdout).hexdigest() != digest:
            problems.append(f"{path}: the recorded hash is not the file at the content commit.")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report the medical approval state of the caption ontology.")
    parser.add_argument("--hashes", action="store_true", help="Print the current hashes of the approved files.")
    args = parser.parse_args(argv)
    ontology = load_ontology()
    if args.hashes:
        print(yaml.safe_dump({"files": current_hashes(ontology)}, sort_keys=False), end="")
        return 0
    print(f"{ontology.tag}: {approval_state(ontology)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
