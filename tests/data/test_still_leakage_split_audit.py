"""CPU-only patient-identity contracts for PR #47, using invented identifiers.

Strict xfails are unmet acceptance criteria, not evidence of a safe cohort. Run
with --runxfail to see the failures. No clinical files are read.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
import build_still_manifest as manifest


def metadata_row(study, patient, folder):
    return dict(study_uid=study, mrn=patient, dcm_path=f"/synthetic/{folder}/clip.dcm",
                sex="F", birthdate="19700101", date="20240101",
                patient_weight_kg=70.0, patient_height_m=1.7)


def test_repeat_study_inherits_existing_patient_split():
    metadata = pd.DataFrame([metadata_row("1.2.1", "P001", "old"),
                             metadata_row("1.2.2", "P001", "new")])
    splits = pd.DataFrame([dict(study_id="1.2.1", patient_id="P001", split="test")])
    stills = pd.DataFrame([dict(study_id="1.2.2", patient="P001")])
    assigned, counts = manifest.assign_splits(stills, splits, metadata, "audit", (.8, .1, .1))
    assert assigned.split.tolist() == ["test"]
    assert counts["patients_inherited"] == 1


def test_existing_patient_in_two_splits_is_rejected():
    splits = pd.DataFrame([dict(study_id="1.2.1", patient_id="P001", split="test"),
                           dict(study_id="1.2.2", patient_id="P001", split="train")])
    with pytest.raises(SystemExit, match="two splits"):
        manifest.assign_splits(pd.DataFrame([dict(study_id="1.2.3", patient="P001")]),
                               splits, pd.DataFrame(), "audit", (.8, .1, .1))


@pytest.mark.xfail(strict=True, reason="Split-table patient_id is not reconciled with metadata.mrn")
def test_disagreeing_patient_sources_must_not_assign_repeat_study_to_training():
    # Select an invented patient that the hash would place in training. The
    # canonical metadata places this patient's earlier study in the test set.
    patient = next(f"P{i:03d}" for i in range(100)
                   if manifest.hashed_split(f"P{i:03d}", "audit", (.8, .1, .1)) == "train")
    metadata = pd.DataFrame([metadata_row("1.2.1", patient, "old"),
                             metadata_row("1.2.2", patient, "new")])
    splits = pd.DataFrame([dict(study_id="1.2.1", patient_id="WRONG-PATIENT", split="test")])
    stills = pd.DataFrame([dict(study_id="1.2.2", patient=patient)])
    # Safe outcomes: refuse the identity conflict, or inherit the canonical
    # patient's held-out assignment. A separate study-ID check cannot catch it.
    try:
        assigned, _ = manifest.assign_splits(stills, splits, metadata, "audit", (.8, .1, .1))
    except (ValueError, SystemExit):
        return
    assert assigned.split.tolist() == ["test"]


@pytest.mark.xfail(strict=True, reason="Folder linking validates study UID but silently keeps the first MRN")
def test_conflicting_patient_identity_in_one_folder_must_be_quarantined():
    metadata = pd.DataFrame([metadata_row("1.2.1", "PATIENT-A", "same"),
                             metadata_row("1.2.1", "PATIENT-B", "same")])
    stills = pd.DataFrame([dict(still_id="invented-still", dcm_path="/synthetic/same/still.dcm")])
    try:
        linked, _, _ = manifest.link_studies(stills, metadata, "audit")
    except (ValueError, SystemExit):
        return
    assert linked.empty, "Ambiguous patient identity must not become an arbitrary patient split"
