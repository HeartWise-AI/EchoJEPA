# tests/datasets/test_still_leakage_contracts.py

"""Leakage contracts of the Doppler still stream that the code enforces (synthetic data, CPU only).

Three of the open points of the leakage audit of the still stream (docs/radar/RADAR_stills_leakage_audit_*.md):

- a conditioning column must be an input known before a measurement, whatever it is called;
- a still token cache must record an accepted preprocessing recipe;
- a patient whose identity the tables disagree on gets no split.

The pixel points of that audit (white, grey or tint-coloured calipers, a value outside the label box, the mark a
filled box leaves) are not covered here: they are open.
"""

import os
import random
import sys
import tempfile
import unittest

import pandas as pd
import torch

from src.datasets.radar_stills import INFERENCE_INPUTS, PREPROCESS_ACCEPTED, StillSource, known_preprocess
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
from src.models.radar.routing import load_routing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data"))
import build_still_manifest as bsm  # noqa: E402

V1_PATH = "configs/radar/radar_routing_v1.yaml"
ENCODER = {"encoder_sha256": "synthetic", "encoder": {"module_name": "fixture"}}


class MemoryCache:
    """A token cache held in memory, without any record of where it came from."""

    shape = (1, 2)

    def __init__(self, provenance=None):
        self.meta = {"token_grid": [1, 1, 1], "provenance": provenance or {}}

    def __contains__(self, key):
        return key in set("abcd")

    def load(self, key):
        return torch.zeros(self.shape)


def still_table():
    target = [1.5, 2.0, 2.5, 3.0]
    return pd.DataFrame({"study_id": list("ABCD"), "still_id": list("abcd"), "still_tag": ["AV"] * 4,
                         "still_type": ["cw_AV"] * 4, "y_AV_VMAX": target, "v_top_ms": [5.0] * 4,
                         "v_bot_ms": [-1.0] * 4, "raw_maxpg": [4 * v ** 2 for v in target],
                         "n_rows": [100 * v for v in target]})


def source(columns, cache=None):
    routing = load_routing(V1_PATH)
    return StillSource(still_table(), routing, cache or MemoryCache(), [s.name for s in routing.trained],
                       num_stills=1, cond_columns=columns, target_columns=["y_AV_VMAX"])


class TestConditioningInputs(unittest.TestCase):

    def test_a_column_that_follows_from_the_measurement_is_refused_whatever_its_name(self):
        for column in ("raw_maxpg", "n_rows"):          # 4 Vmax^2, and a count of measurement rows
            with self.assertRaisesRegex(ValueError, "known before a measurement"):
                source([column])
            with self.assertRaisesRegex(ValueError, "known before a measurement"):
                source(["v_top_ms", column])

    def test_a_target_is_still_refused_first(self):
        with self.assertRaisesRegex(ValueError, "never an input"):
            source(["y_AV_VMAX"])

    def test_the_listed_inputs_pass(self):
        item = source(["v_top_ms", "v_bot_ms"]).item("A", random.Random(0))
        self.assertEqual(item["still_cond"][0].tolist(), [5.0, -1.0])
        self.assertFalse(any(c.startswith("y_") for c in INFERENCE_INPUTS))


class TestPreprocessRecord(unittest.TestCase):

    def test_a_cache_without_an_accepted_recipe_is_refused(self):
        for provenance in (dict(ENCODER), {**ENCODER, "preprocess": "unmasked"}, {**ENCODER, "preprocess": "m0"},
                           {"split": "train"}):
            with self.assertRaisesRegex(ValueError, "preprocessing recipe"):
                known_preprocess({"provenance": provenance})
            with self.assertRaisesRegex(ValueError, "preprocessing recipe"):
                source(["v_top_ms"], MemoryCache(provenance))

    def test_an_accepted_recipe_and_a_synthetic_cache_pass(self):
        recipe = PREPROCESS_ACCEPTED[0]
        self.assertEqual(known_preprocess({"provenance": {**ENCODER, "preprocess": recipe}}), recipe)
        self.assertEqual(source(["v_top_ms"], MemoryCache({**ENCODER, "preprocess": recipe})).preprocess, recipe)
        for provenance in ({}, None, {"synthetic": True}):
            self.assertIsNone(known_preprocess({"provenance": provenance}))

    def test_training_s_reader_refuses_a_cache_on_disk_without_the_recipe(self):
        from evals.radar_frozen.train import read_stills
        routing = load_routing(V1_PATH)
        with tempfile.TemporaryDirectory() as root:
            table = os.path.join(root, "stills.csv")
            pd.DataFrame({"study_id": ["A"], "still_id": ["a"], "still_tag": ["AV"], "still_type": ["cw_AV"],
                          "split": ["train"]}).to_csv(table, index=False)
            for name, record in (("none", dict(ENCODER)), ("unmasked", {**ENCODER, "preprocess": "unmasked"}),
                                 ("ok", {**ENCODER, "preprocess": PREPROCESS_ACCEPTED[0]})):
                writer = TokenCacheWriter(os.path.join(root, name), (1, 1, 1), 2, provenance=record)
                writer.add("a", torch.zeros(1, 2))
                writer.close()
            for name in ("none", "unmasked"):
                with self.assertRaisesRegex(ValueError, "preprocessing recipe"):
                    read_stills({"table": table, "cache_dir": os.path.join(root, name)}, routing, "train")
            kept = read_stills({"table": table, "cache_dir": os.path.join(root, "ok")}, routing, "train")
            self.assertEqual(kept.studies, {"A"})
            self.assertIsInstance(kept.cache, TokenCache)


def metadata_row(study, patient, folder):
    return dict(study_uid=study, mrn=patient, dcm_path=f"/synthetic/{folder}/clip.dcm", sex="F",
                birthdate="19700101", date="20240101", patient_weight_kg=70.0, patient_height_m=1.7)


class TestPatientIdentity(unittest.TestCase):

    FRACTIONS = (0.8, 0.1, 0.1)

    def test_a_split_table_patient_the_metadata_does_not_give_the_study_stops_the_build(self):
        patient = next(f"P{i:03d}" for i in range(100)
                       if bsm.hashed_split(f"P{i:03d}", "salt", self.FRACTIONS) == "train")
        metadata = pd.DataFrame([metadata_row("1.2.1", patient, "old"), metadata_row("1.2.2", patient, "new")])
        splits = pd.DataFrame([dict(study_id="1.2.1", patient_id="OTHER", split="test")])
        stills = pd.DataFrame([dict(study_id="1.2.2", patient=patient)])
        with self.assertRaisesRegex(SystemExit, "metadata does not give"):
            bsm.assign_splits(stills, splits, metadata, "salt", self.FRACTIONS)
        agreeing = splits.assign(patient_id=patient)
        out, counts = bsm.assign_splits(stills, agreeing, metadata, "salt", self.FRACTIONS)
        self.assertEqual(out.split.tolist(), ["test"])          # the later study follows the earlier one's split
        self.assertEqual(counts["patients_inherited"], 1)

    def test_a_folder_or_a_study_with_two_patients_links_no_still(self):
        stills = pd.DataFrame([dict(still_id="s1", dcm_path="/synthetic/same/still.dcm")])
        two_in_folder = pd.DataFrame([metadata_row("1.2.1", "PATIENT-A", "same"),
                                      metadata_row("1.2.1", "PATIENT-B", "same")])
        linked, _, unlinked = bsm.link_studies(stills, two_in_folder, "salt")
        self.assertTrue(linked.empty)
        self.assertEqual(unlinked, 1)
        two_in_study = pd.DataFrame([metadata_row("1.2.1", "PATIENT-A", "same"),
                                     metadata_row("1.2.1", "PATIENT-B", "elsewhere")])
        linked, _, unlinked = bsm.link_studies(stills, two_in_study, "salt")
        self.assertTrue(linked.empty)
        one = pd.DataFrame([metadata_row("1.2.1", "PATIENT-A", "same"), metadata_row("1.2.1", "PATIENT-A", "same")])
        linked, _, unlinked = bsm.link_studies(stills, one, "salt")
        self.assertEqual((len(linked), unlinked), (1, 0))


if __name__ == "__main__":
    unittest.main()
