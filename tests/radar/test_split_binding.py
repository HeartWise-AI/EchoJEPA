"""RADAR records and the shared patient split: one split per patient and study across a whole records file, one
split table (`split_source`), validation only from bound records, and pilot records that take their split and their
acquisition (the pretraining B-mode vendor rule) from the archive's own tables. Synthetic data only."""

import hashlib
import json
import os
import tempfile
import unittest

import pandas as pd

from app.radar.build_pilot_records import VIEW_COLUMNS, build
from app.radar.data import SplitGuardError, select_records
from app.radar.records import RecordError, read_records, write_records
from tests.radar.synthetic import ONTOLOGY, candidate, record, replace, reviewed, with_review

SOURCE = "a" * 64
OTHER_SOURCE = "b" * 64


def approved(n, split, patient, study, source=SOURCE):
    rec = record(n, split=split, patient=patient, study=study, caption=candidate(f"Coupe A4C, mode B. {n}."))
    return replace(with_review(rec, reviewed(label_ref=f"local:{n}")), split_source=source)


class TestSplitBinding(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.train = [approved(0, "train", "patient-a", "study-a"), approved(1, "train", "patient-a", "study-a")]
        self.val = [approved(2, "val", "patient-b", "study-b"), approved(3, "val", "patient-b", "study-b")]

    def write_lines(self, records):
        path = os.path.join(self.tmp, "records.jsonl")
        with open(path, "w") as f:  # bypasses write_records, as a hand-edited or concatenated file would
            f.writelines(json.dumps(r.to_dict()) + "\n" for r in records)
        return path

    def test_bound_train_and_val_records_select(self):
        items, stats = select_records(self.train + self.val, split="val", purpose="validation_loss",
                                      split_source=SOURCE)
        self.assertEqual((len(items), stats["split_source"]), (2, SOURCE))

    def test_a_patient_or_study_in_two_splits_is_refused_everywhere(self):
        cases = {
            "patients have records in more than one split": approved(4, "val", "patient-a", "study-c"),
            "studies have records in more than one split": approved(4, "val", "patient-c", "study-a"),
            "studies have records of more than one patient": approved(4, "train", "patient-c", "study-a"),
        }
        for message, extra in cases.items():
            records = self.train + self.val + [extra]
            with self.subTest(message):
                # The train split alone looks clean; the whole file does not.
                with self.assertRaisesRegex(SplitGuardError, message):
                    select_records(records, split="train", purpose="training")
                with self.assertRaisesRegex(RecordError, message):
                    write_records(os.path.join(self.tmp, "out.jsonl"), records, ONTOLOGY)
                with self.assertRaisesRegex(RecordError, message):
                    read_records(self.write_lines(records), ONTOLOGY)

    def test_validation_needs_one_bound_split_table(self):
        unbound = [replace(r, split_source=None) for r in self.train + self.val]
        with self.assertRaisesRegex(SplitGuardError, "bound to the shared archive patient split"):
            select_records(unbound, split="val", purpose="validation_loss")
        # Training alone may use unbound records (synthetic demos); the stats say so.
        _, stats = select_records(unbound[:2], split="train", purpose="training")
        self.assertIsNone(stats["split_source"])
        mixed = self.train + [replace(r, split_source=OTHER_SOURCE) for r in self.val]
        with self.assertRaisesRegex(SplitGuardError, "2 different split tables"):
            select_records(mixed, split="train", purpose="training")
        with self.assertRaisesRegex(RecordError, "some unbound"):
            read_records(self.write_lines(self.train + unbound[2:]), ONTOLOGY)
        with self.assertRaisesRegex(SplitGuardError, "expected patient split table"):
            select_records(self.train + self.val, split="train", purpose="training", split_source=OTHER_SOURCE)

    def test_split_source_must_be_a_sha256(self):
        with self.assertRaisesRegex(RecordError, "split_source must be the SHA-256"):
            read_records(self.write_lines([replace(r, split_source="patient_splits.csv") for r in self.train]),
                         ONTOLOGY)


class TestPilotRecords(unittest.TestCase):
    """build_pilot_records on a synthetic pilot: one train and one val study, GE and non-GE clips."""

    CLIPS = [  # (patient, study, view, dcm, brand, category, confidence)
        ("0000001", "1.2.999.1", "A4C", "d1", "GE Vingmed Ultrasound", "bmode", "header"),
        ("0000001", "1.2.999.1", "A4C", "d2", "Philips Medical Systems", "bmode", "header"),
        ("0000001", "1.2.999.1", "PSAX", "d3", "Philips Medical Systems", "color_flow", "header"),
        ("0000002", "1.2.999.2", "A4C", "d4", "GE Vingmed Ultrasound", "unknown", "unknown"),
    ]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.pilot = os.path.join(self.tmp, "pilot")
        os.makedirs(self.pilot)
        clips = pd.DataFrame(
            [(p, s, v, f"/synthetic/{d}.mp4", f"/synthetic/{d}.dcm") for p, s, v, d, *_ in self.CLIPS],
            columns=["patient_id", "study_id", "predicted_class", "avi_path", "dcm_path"])
        clips.to_csv(os.path.join(self.pilot, "pilot_clips.csv"), index=False)
        studies = clips.drop_duplicates("study_id")[["study_id"]]
        studies.to_csv(os.path.join(self.pilot, "pilot_studies.csv"), index=False)
        studies.assign(accession=["A1", "A2"]).to_csv(os.path.join(self.pilot, "shortlist.csv"), index=False)
        self.reports = os.path.join(self.tmp, "reports.csv")
        pd.DataFrame({"AccessionNumber": ["A1", "A2"]}).to_csv(self.reports, index=False)
        meta = clips[["avi_path"]].assign(brand=[c[4] for c in self.CLIPS], **{v: 0.04 for v in VIEW_COLUMNS})
        meta["A4C"] = 0.9
        self.metadata = os.path.join(self.tmp, "meta.parquet")
        meta.to_parquet(self.metadata, index=False)
        acq = pd.DataFrame({"dcm_path": clips.dcm_path, "acq_category": [c[5] for c in self.CLIPS],
                            "acq_confidence": [c[6] for c in self.CLIPS], "acq_rule": "synthetic",
                            "has_tissue_2d": True, "has_color_flow": [c[5] == "color_flow" for c in self.CLIPS],
                            "has_pw_doppler": False, "has_cw_doppler": False, "has_mmode": False})
        self.acq = os.path.join(self.tmp, "acq.parquet")
        acq.to_parquet(self.acq, index=False)
        self.splits = os.path.join(self.tmp, "patient_splits.csv")
        pd.DataFrame({"patient_id": ["0000001", "0000002", "0000003"], "split": ["train", "val", "test"],
                      "origin": "new", "in_archive": True}).to_csv(self.splits, index=False)

    def build(self, splits=None):
        return build(self.pilot, self.metadata, self.acq, self.reports, "unit-test-secret", ONTOLOGY,
                     patient_splits=splits or self.splits)

    def test_records_take_the_archive_split_and_its_checksum(self):
        records, _ = self.build()
        with open(self.splits, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual([r.split for r in records], ["train", "train", "train", "val"])
        self.assertEqual({r.split_source for r in records}, {digest})
        path = os.path.join(self.tmp, "records.jsonl")
        write_records(path, records, ONTOLOGY)
        items, _ = select_records(read_records(path, ONTOLOGY), split="val", purpose="validation_loss",
                                  split_source=digest)
        self.assertEqual(items, [])  # no approved caption yet; the guard passed

    def test_header_bmode_from_an_unvalidated_vendor_is_unresolved(self):
        records = {r.video_path: r for r in self.build()[0]}
        ge, other, colour = (records[f"/synthetic/{d}.mp4"].acquisition for d in ("d1", "d2", "d3"))
        self.assertEqual((ge.category, ge.confidence, ge.has_color_flow), ("bmode", "header", False))
        self.assertEqual((other.category, other.confidence, other.rule, other.has_color_flow),
                         ("unknown", "unknown", "bmode_vendor_unvalidated", None))
        self.assertEqual(colour.category, "color_flow")
        # The candidate caption names the acquisition the record holds, not the header's B-mode.
        caption = records["/synthetic/d2.mp4"].candidate_caption.text
        self.assertIn(ONTOLOGY.acquisition_labels["fr"]["unknown"], caption)
        self.assertNotIn(ONTOLOGY.acquisition_labels["fr"]["bmode"], caption)

    def test_a_patient_missing_from_the_split_table_stops_the_build(self):
        partial = os.path.join(self.tmp, "partial.csv")
        pd.DataFrame({"patient_id": ["0000001"], "split": ["train"]}).to_csv(partial, index=False)
        with self.assertRaisesRegex(ValueError, "1 pilot clips belong to patients absent"):
            self.build(partial)


if __name__ == "__main__":
    unittest.main()
