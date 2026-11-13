# tests/data/test_still_split_audit.py

"""`data/still_split_audit.py` on invented tables: a clean table passes, and each kind of patient in two splits is
counted."""

import os
import sys
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import still_split_audit as audit  # noqa: E402

STILLS = pd.DataFrame({"study_id": ["1.1", "1.2", "1.3", "1.4"], "patient_hash": ["a", "a", "b", "c"],
                       "split": ["train", "train", "val", "test"]})
METADATA = pd.DataFrame({"study_uid": ["1.1", "1.2", "1.3", "1.4", "2.1"], "mrn": ["01", "1", "2", "3", "3"],
                         "birthdate": ["1950-01-01", "1950-01-01", "1960-02-02", "1970-03-03", "1970-03-03"]})
PACS = pd.DataFrame({"StudyInstanceUID": ["1.1", "1.2", "1.3", "1.4"], "PatientID": ["1", "1", "2", "3"],
                     "PatientName": ["DOE^JANE", "DOE^JANE", "ROE^RICHARD", "POE^ANNE"]})
PATIENTS = pd.DataFrame({"patient_id": ["1", "2", "3"], "split": ["train", "val", "test"]})


class TestSplitAudit(unittest.TestCase):

    def test_a_clean_table_passes(self):
        clips = pd.DataFrame({"study_id": ["1.1", "2.1"], "split": ["train", "test"]})
        report = audit.audit(STILLS, METADATA, PACS, PATIENTS, [clips])
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["metadata_and_pacs_disagree"], 0)
        self.assertEqual(report["absent_from_patient_splits"], 0)

    def test_each_kind_of_patient_in_two_splits_is_counted(self):
        moved = STILLS.assign(split=["train", "val", "val", "test"])              # patient a in two splits
        self.assertIn("patient_hash_in_two_splits", audit.audit(moved, METADATA)["violations"])
        pacs = PACS.assign(PatientID=["1", "1", "1", "3"])                         # PACS: 1.3 is patient 1 too
        report = audit.audit(STILLS, METADATA, pacs)
        self.assertEqual((report["metadata_and_pacs_disagree"], report["pacs_patient_in_two_splits"]), (1, 1))
        same = PACS.assign(PatientName=["DOE^JANE", "DOE^JANE", "Doe Jane", "POE^ANNE"])
        metadata = METADATA.assign(birthdate=["1950-01-01"] * 3 + ["1970-03-03"] * 2)   # one person, numbers 1 and 2
        report = audit.audit(STILLS, metadata, same)
        self.assertEqual((report["same_person_two_numbers"], report["same_person_two_numbers_in_two_splits"]), (1, 1))
        report = audit.audit(STILLS, METADATA, PACS, PATIENTS.assign(split=["train", "train", "test"]).iloc[:2])
        self.assertEqual((report["differs_from_patient_splits"], report["absent_from_patient_splits"]), (1, 1))
        clips = pd.DataFrame({"study_id": ["1.1", "2.1"], "split": ["val", "train"]})
        report = audit.audit(STILLS, METADATA, PACS, PATIENTS, [clips])
        self.assertEqual(report["clip_study_in_another_split"], 1)
        self.assertEqual(report["clip_patient_in_two_splits"], 2)                  # patient 1 and patient 3


if __name__ == "__main__":
    unittest.main()
