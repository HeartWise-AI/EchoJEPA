# tests/models/test_radar_mine_prompts.py

"""Prompts sampled from training captions: sides, grades and values follow the labels; refusals; reads back."""

import json
import os
import tempfile
import unittest

import pandas as pd
import yaml

from evals.radar_frozen import mine_prompts as mp
from src.datasets.radar_study_dataset import file_sha256
from src.models.radar.zero_shot import load_prompt_tasks

TASKS = {"tasks": {
    "effusion": {"mode": "binary", "structure": "pericardium", "label_sources": ["effusion"],
                 "negative_sources": ["no effusion"]},
    "lv_size": {"mode": "categorical", "structure": "lv", "label_sources": ["normal lv", "dilated lv"],
                "values": [0, 1]},
    "ef": {"mode": "regression", "structure": "lv", "target": "lvef", "label_sources": ["ef <#>"],
           "range": [50, 52]},
    "rare": {"mode": "binary", "structure": "aorta", "label_sources": ["aneurysm"], "negative_sources": ["normal"]},
}}


class TestMinePrompts(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        rows, labels = [], []
        for n in range(30):
            study = f"s{n:02d}"
            effusion, size, ef = n % 2, n % 2, 50 + n % 3
            rows.append((study, "pericardium", f"peri {'effusion' if effusion else 'clear'} {n % 4}"))
            rows.append((study, "lv", f"lv size{size} ef{ef} w{n % 5}"))
            rows.append((study, "aorta", "aorta " + " ".join(["long"] * (300 if n == 0 else 1))))
            labels.append({"study_id": study, "effusion": effusion, "lv_size": size, "lvef": ef + 0.2,
                           "rare": int(n < 2)})
        labels.append({"study_id": "v99", "effusion": 1, "lv_size": 1, "lvef": 51, "rare": 1})   # not a train study
        labels[3]["effusion"] = ""                                                              # a missing label
        self.captions = os.path.join(root, "captions.csv")
        pd.DataFrame(rows, columns=["study_id", "structure", "caption"]).to_csv(self.captions, index=False)
        with open(os.path.join(root, "build_info.json"), "w") as f:
            json.dump({"captions_sha256": file_sha256(self.captions), "split": "train", "rule_version": "t"}, f)
        self.labels = os.path.join(root, "labels.csv")
        pd.DataFrame(labels).to_csv(self.labels, index=False)
        self.prompts = os.path.join(root, "tasks.yaml")
        with open(self.prompts, "w") as f:
            yaml.safe_dump(TASKS, f)
        self.out = os.path.join(root, "out", "mined.yaml")

    def tearDown(self):
        self.tmp.cleanup()

    def test_sides_grades_and_values_follow_the_labels(self):
        content, counts = mp.mine(self.prompts, self.captions, self.labels, per_set=6, per_value=4)
        tasks = content["tasks"]
        self.assertEqual(len(tasks["effusion"]["label_sources"]), 6)
        self.assertTrue(all("effusion" in p for p in tasks["effusion"]["label_sources"]))
        self.assertTrue(all("clear" in p for p in tasks["effusion"]["negative_sources"]))
        self.assertEqual(counts["effusion"]["captions_with_label"], 29)             # the blank label is left out
        for prompt, value in zip(tasks["lv_size"]["label_sources"], tasks["lv_size"]["values"]):
            self.assertIn(f"size{value}", prompt)
        self.assertEqual(sorted(set(tasks["ef"]["values"])), [50, 51, 52])
        for prompt, value in zip(tasks["ef"]["label_sources"], tasks["ef"]["values"]):
            self.assertIn(f"ef{value}", prompt)                                    # 50.2 rounds to 50
        self.assertNotIn("range", tasks["ef"])
        self.assertEqual(max(tasks["ef"]["values"].count(v) for v in (50, 51, 52)), 4)
        self.assertEqual(content["mined_from"]["kept_as_written"], ["rare"])       # 1 positive after the long one
        self.assertEqual(tasks["rare"]["label_sources"], ["aneurysm"])
        again, _ = mp.mine(self.prompts, self.captions, self.labels, per_set=6, per_value=4)
        self.assertEqual(content, again)
        other, _ = mp.mine(self.prompts, self.captions, self.labels, per_set=6, per_value=4, seed=1)
        self.assertNotEqual(content["tasks"], other["tasks"])

    def test_written_file_reads_back(self):
        mp.main(["--prompts", self.prompts, "--captions", self.captions, "--labels", self.labels, "--out", self.out,
                 "--per_set", "6", "--per_value", "4"])
        tasks = load_prompt_tasks(self.out)
        self.assertEqual(tasks["ef"].mode, "regression")
        self.assertEqual(len(tasks["ef"].values), len(tasks["ef"].label_sources))
        self.assertEqual(tasks["ef"].value_range, ())

    def test_refuses_another_split_a_changed_table_and_the_repository(self):
        with self.assertRaises(ValueError):
            mp.mine(self.prompts, self.captions, self.labels, split="val")
        with self.assertRaises(ValueError):
            mp.main(["--prompts", self.prompts, "--captions", self.captions, "--labels", self.labels,
                     "--out", os.path.join(mp.REPO, "mined.yaml")])
        with open(self.captions, "a") as f:
            f.write("s00,rv,rv ok\n")
        with self.assertRaises(ValueError):
            mp.mine(self.prompts, self.captions, self.labels)


class TestSampledRegressionSchema(unittest.TestCase):

    def load(self, spec):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "p.yaml")
            with open(path, "w") as f:
                yaml.safe_dump({"tasks": {"ef": {"mode": "regression", "structure": "lv", **spec}}}, f)
            return load_prompt_tasks(path)["ef"]

    def test_values_replace_the_slot(self):
        task = self.load({"label_sources": ["a", "b", "c"], "values": [40, 55, 55]})
        self.assertEqual(task.values, (40, 55, 55))
        for bad in ({"label_sources": ["a", "b"], "values": [40]},
                    {"label_sources": ["a <#>", "b"], "values": [40, 50]},
                    {"label_sources": ["a", "b"], "values": [40, 40]},
                    {"label_sources": ["a", "b"], "values": [40, "x"]},
                    {"label_sources": ["a", "b"]}):
            with self.assertRaises(ValueError):
                self.load(bad)


if __name__ == "__main__":
    unittest.main()
