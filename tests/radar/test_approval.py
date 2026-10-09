"""The approval record: the shipped record is valid in whichever state it is in, an approved record must match the
approved files and its content commit, and a change or a version bump leaves the ontology unapproved. Synthetic
records in temporary git repositories only."""

import dataclasses
import datetime
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import yaml

from app.radar.approval import (
    APPROVED,
    MISSING,
    PENDING,
    ApprovalError,
    approval_state,
    approved_files,
    current_hashes,
    record_path,
    record_problems,
    require_approval,
)
from app.radar.ontology import REPO_ROOT, load_ontology

ONTOLOGY = load_ontology()
LINK = "https://github.com/HeartWise-AI/EchoJEPA/issues/21#issuecomment-1"
PENDING_RECORD = {
    "schema": "radar-caption-approval", "ontology": ONTOLOGY.tag, "status": PENDING, "content_commit": None,
    "files": {}, "medical_approval": {"reviewer": None, "link": None, "date": None},
    "thresholds_agreement": {"link": None, "date": None}, "labelbox_sdk_check": {"version": None},
}
STATUS_TEXT = re.compile(
    r"\b(?:status|statut)\s*:|\b(?:not approved|non approuvée?s?|draft|brouillon|proposed thresholds)\b", re.I)


def git(root, *args):
    return subprocess.run(["git", "-C", root, "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def write_record(root, record):
    with open(record_path(ONTOLOGY, root), "w", encoding="utf-8") as f:
        f.write("# Record\n\n```yaml\n" + yaml.safe_dump(record, sort_keys=False) + "```\n")


class ApprovedRepo(unittest.TestCase):
    """A temporary repository: the approved files in a content commit, then the approval record in the next commit."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        git(self.root, "init", "-q")
        for path in approved_files(ONTOLOGY):
            os.makedirs(os.path.dirname(os.path.join(self.root, path)), exist_ok=True)
            shutil.copy(os.path.join(REPO_ROOT, path), os.path.join(self.root, path))
        git(self.root, "add", ".")
        git(self.root, "commit", "-q", "-m", "content")
        self.content = git(self.root, "rev-parse", "HEAD")
        self.record = {
            **PENDING_RECORD, "status": APPROVED, "content_commit": self.content,
            "files": current_hashes(ONTOLOGY, self.root),
            "medical_approval": {"reviewer": "robertavram-md", "link": LINK, "date": "2026-10-20"},
            "thresholds_agreement": {"link": LINK, "date": "2026-10-19"},
            "labelbox_sdk_check": {"version": "7.12.0"},
        }
        self.approve(self.record)

    def approve(self, record):
        write_record(self.root, record)
        git(self.root, "add", ".")
        git(self.root, "commit", "-q", "-m", "approval")

    def edit_metrics(self):
        with open(os.path.join(self.root, "docs", "radar", "pilot_metrics_v1.md"), "a", encoding="utf-8") as f:
            f.write("\nM3 > 40%\n")

    def refused(self, message):
        with self.assertRaises(ApprovalError) as ctx:
            require_approval(ONTOLOGY, self.root)
        self.assertIn(message, str(ctx.exception))


class TestApprovalGate(ApprovedRepo):
    def test_a_complete_approval_passes(self):
        self.assertEqual(approval_state(ONTOLOGY, self.root), APPROVED)
        require_approval(ONTOLOGY, self.root)

    def test_an_approved_file_cannot_change(self):
        self.edit_metrics()
        self.refused("pilot_metrics_v1.md: changed since its approval")

    def test_refreshing_the_hashes_does_not_re_approve(self):
        self.edit_metrics()
        git(self.root, "commit", "-q", "-am", "edit")
        self.approve({**self.record, "files": current_hashes(ONTOLOGY, self.root)})
        self.refused("pilot_metrics_v1.md: the recorded hash is not the file at the content commit")

    def test_the_content_commit_must_exist(self):
        self.approve({**self.record, "content_commit": "0" * 40})
        self.refused("is not in this repository's history")

    def test_the_content_commit_must_be_an_ancestor(self):
        tree = git(self.root, "rev-parse", f"{self.content}^{{tree}}")
        detached = git(self.root, "commit-tree", tree, "-m", "same files, other history")
        self.approve({**self.record, "content_commit": detached})
        self.refused("is not an ancestor of `HEAD`")

    def test_without_git_the_record_is_invalid(self):
        shutil.rmtree(os.path.join(self.root, ".git"))
        self.refused("is not in this repository's history")

    def test_only_the_approved_ontology_is_approved(self):
        """An ontology with the approved tag but other content (here loaded from another path) is refused."""
        with open(ONTOLOGY.path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw["concepts"][0]["labels"]["en"] = "Edited label"
        path = os.path.join(self.root, "other-ontology.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
        other = load_ontology(path)
        self.assertEqual(other.tag, ONTOLOGY.tag)
        with self.assertRaises(ApprovalError) as ctx:
            require_approval(other, self.root)
        self.assertIn("was not loaded from the approved configs/radar/radar-caption-ontology-v1.yaml",
                      str(ctx.exception))

    def test_an_unreadable_record_is_refused(self):
        for block, message in (("medical_approval: {date: 2026-99-99}", "the record cannot be read"),
                               ("- a list", "the record must be a mapping")):
            with open(record_path(ONTOLOGY, self.root), "w", encoding="utf-8") as f:
                f.write(f"```yaml\n{block}\n```\n")
            self.refused(message)

    def test_malformed_fields_are_problems_not_crashes(self):
        for record in (PENDING_RECORD, self.record):
            for key in ("medical_approval", "thresholds_agreement", "labelbox_sdk_check", "files", "content_commit"):
                for value in ("text", ["a list"], 5):
                    problems = record_problems({**record, key: value}, ONTOLOGY, self.root)
                    self.assertTrue(problems, (record["status"], key, value))
        self.assertIn("medical_approval must be a mapping.",
                      record_problems({**PENDING_RECORD, "medical_approval": "text"}, ONTOLOGY))

    def test_a_new_version_is_unapproved(self):
        v2 = dataclasses.replace(ONTOLOGY, version=2)
        self.assertEqual(approval_state(v2, self.root), MISSING)
        with self.assertRaises(ApprovalError):
            require_approval(v2, self.root)
        self.assertIn("is for 'radar-caption-ontology-v1', not 'radar-caption-ontology-v2'",
                      "\n".join(record_problems(self.record, v2, self.root)))

    def test_refusals(self):
        def problems(**changes):
            record = {**self.record}
            for key, value in changes.items():  # the sign-off fields are updated, every other field replaced
                nested = key in ("medical_approval", "thresholds_agreement", "labelbox_sdk_check")
                record[key] = {**record[key], **value} if nested else value
            return "\n".join(record_problems(record, ONTOLOGY, self.root))

        self.assertEqual(problems(), "")
        self.assertIn("needs ['medical_approval.link']", problems(medical_approval={"link": None}))
        self.assertIn("must link to a HeartWise-AI/EchoJEPA",
                      problems(medical_approval={"link": "https://example.org/ok"}))
        self.assertIn("Files must be exactly", problems(files=dict(list(self.record["files"].items())[1:])))
        self.assertIn("full 40-character commit", problems(content_commit="abc123"))
        self.assertIn("must be a calendar date", problems(thresholds_agreement={"date": "Oct 19"}))
        self.assertIn("must be a calendar date", problems(medical_approval={"date": "2026-99-99"}))
        self.assertEqual(problems(medical_approval={"date": datetime.date(2026, 10, 20)}), "")  # an unquoted date
        self.assertIn("`medical_approval.reviewer` must be 'robertavram-md'",
                      problems(medical_approval={"reviewer": "someone_else"}))
        self.assertIn("needs ['labelbox_sdk_check.version']", problems(labelbox_sdk_check={"version": None}))
        self.assertIn("must be a release number", problems(labelbox_sdk_check={"version": "banana"}))
        self.assertIn("Status must be", problems(status="signed"))


class TestPendingRecords(unittest.TestCase):
    def test_a_pending_record_is_valid_and_not_approved(self):
        self.assertEqual(record_problems(PENDING_RECORD, ONTOLOGY), [])

    def test_a_pending_record_has_no_partial_approval(self):
        partial = {**PENDING_RECORD, "medical_approval": {**PENDING_RECORD["medical_approval"],
                                                          "reviewer": "robertavram-md"}}
        self.assertIn("A pending record has no approval fields", "\n".join(record_problems(partial, ONTOLOGY)))


class TestShippedRecord(unittest.TestCase):
    def test_valid_in_its_state(self):
        state = approval_state(ONTOLOGY)  # raises on any problem, content commit included once approved
        self.assertIn(state, (PENDING, APPROVED))
        if state == APPROVED:
            require_approval(ONTOLOGY)
        else:
            with self.assertRaises(ApprovalError):
                require_approval(ONTOLOGY)

    def test_every_approved_file_exists(self):
        for path in approved_files(ONTOLOGY):
            self.assertTrue(os.path.exists(os.path.join(REPO_ROOT, path)), path)

    def test_approved_files_do_not_state_the_approval_status(self):
        """Approval freezes them, so a status written in them would be stale once the record is approved."""
        for path in approved_files(ONTOLOGY):
            with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
                self.assertEqual(STATUS_TEXT.findall(f.read()), [], path)


if __name__ == "__main__":
    unittest.main()
