# tests/datasets/test_radar_stills.py

"""The still stream of RADAR study items and training with it, on synthetic tokens:

- a study with stills but no clip is an item, whose B-mode structures have no input;
- a structure gets its `full` caption only when the bag holds a still it may use, its `bmode` caption otherwise;
- stills are drawn after the clips from the same seeded stream, so the clip bag is the v0 bag;
- measured values are refused as conditioning inputs; targets and age are z-scored with the training statistics;
- one epoch of training with the still stream, the heat-map bias and the measurement heads runs, logs the heads, and
  a config without stills keeps its v0 fingerprint.
"""

import json
import os
import random
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from evals.radar_frozen import train as radar_train
from src.datasets.radar_stills import StillSource
from src.datasets.radar_study_dataset import RadarStudyDataset, collate, file_sha256, read_table
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
from src.models.radar.routing import DEFAULT_ROUTING, load_routing
from src.models.radar.text_encoder import HashTokenizer
from tests.models.radar_synthetic import DIM, make_synthetic, synthetic_config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import radar_captions as rc  # noqa: E402

V1_PATH = os.path.join(os.path.dirname(DEFAULT_ROUTING), "radar_routing_v1.yaml")
STILL_GRID = (1, 2, 2)
TARGETS = ["y_AV_VMAX", "y_TR_VMAX"]
COND = ["v_top_ms", "v_bot_ms", "echonet_present", "echonet_vmax_ms"]


def make_stills(root, studies):
    """Two stills per study of `studies` (AV and TR) plus a still-only study `X99`; a token cache; demographics;
    an EchoNet heat-map for the first still."""
    writer = TokenCacheWriter(os.path.join(root, "still_cache"), STILL_GRID, DIM)
    g = torch.Generator().manual_seed(3)
    heat = os.path.join(root, "heat.npy")
    np.save(heat, np.eye(4, dtype=np.float32))
    rows = []
    for i, study in enumerate([*studies, "X99"]):
        for j, (tag, kind, value) in enumerate((("AV", "cw_AV", 2.0 + 0.1 * i), ("TR", "cw_TR", 2.5 + 0.05 * i))):
            still = f"{study}-s{j}"
            writer.add(still, torch.randn(4, DIM, generator=g))
            rows.append({"still_id": still, "study_id": study, "split": "train", "still_tag": tag,
                         "still_type": kind, "v_top_ms": 1.0, "v_bot_ms": -5.0, "echonet_present": int(i == 0),
                         "echonet_vmax_ms": value if i == 0 else np.nan,
                         "echonet_heatmap": heat if (i == 0 and j == 0) else "",
                         "y_AV_VMAX": value if tag == "AV" else np.nan,
                         "y_TR_VMAX": value if tag == "TR" else np.nan})
    writer.close()
    pd.DataFrame(rows).to_csv(os.path.join(root, "stills.csv"), index=False)
    ids = [*studies, "X99"]
    pd.DataFrame({"study_id": ids, "age_years": [50.0 + i for i in range(len(ids))],
                  "sex": [("M", "F")[i % 2] for i in range(len(ids))]}).to_csv(
        os.path.join(root, "demographics.csv"), index=False)
    return {"table": os.path.join(root, "stills.csv"), "cache_dir": os.path.join(root, "still_cache"),
            "demographics": os.path.join(root, "demographics.csv")}


def add_full_captions(paths, routing):
    """The synthetic caption table with a `scope` column, a full aortic caption for S00 and S03, Hemo captions for
    S00 and X99, sealed for the v1 registry."""
    table = read_table(paths["captions"], id_columns=("study_id",)).assign(scope="bmode")
    extra = []
    for study, structure, text in (("S00", "aortic", "Valve aortique de structure normale. Pas de sténose aortique."),
                                   ("S03", "aortic", "Valve aortique calcifiée. Sténose aortique serrée."),
                                   ("S00", "hemo", "Pression systolique ventriculaire droite estimée à 30 mmHg."),
                                   ("X99", "hemo", "Pression systolique ventriculaire droite estimée à 50 mmHg.")):
        extra.append({"study_id": study, "structure": structure, "caption": text, "caption_id": rc.caption_id(text),
                      "normality": "unknown", "normality_status": "provisional", "policy_digest": "synthetic",
                      "scope": "full"})
    table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
    table.to_csv(paths["captions"], index=False)
    info = {"split": "train", "routing_sha256": routing.sha256, "clips_sha256": file_sha256(paths["clips"]),
            "captions_sha256": file_sha256(paths["captions"]), "policy_digest": "synthetic"}
    with open(os.path.join(os.path.dirname(paths["captions"]), "build_info.json"), "w") as f:
        json.dump(info, f)


class TestStillItems(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(cls.root)
        cls.routing = load_routing(V1_PATH)
        cls.clips = read_table(cls.paths["clips"])
        cls.studies = sorted(set(cls.clips.study_id))[:4]
        cls.still_paths = make_stills(cls.root, cls.studies)
        add_full_captions(cls.paths, cls.routing)
        cls.captions = read_table(cls.paths["captions"], id_columns=("study_id",))
        cls.cache = TokenCache(cls.paths["cache_dir"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def source(self, **kw):
        table = pd.read_csv(self.still_paths["table"], dtype={"study_id": str, "still_id": str})
        args = dict(num_stills=2, cond_columns=COND, target_columns=TARGETS, heatmap_column="echonet_heatmap",
                    demographics=pd.read_csv(self.still_paths["demographics"], dtype={"study_id": str}))
        args.update(kw)
        return StillSource(table, self.routing, TokenCache(self.still_paths["cache_dir"]),
                           [s.name for s in self.routing.trained], **args)

    def dataset(self, stills=None, routing=None):
        return RadarStudyDataset(self.clips, self.captions, routing or self.routing, self.cache, num_clips=4,
                                 stills=stills)

    def test_still_only_study_and_captions(self):
        data = self.dataset(self.source())
        names = [s.name for s in self.routing.trained]
        items = {data[i]["study_id"]: data[i] for i in range(len(data))}
        x = items["X99"]
        self.assertFalse(x["clip_valid"].any())
        self.assertEqual(x["still_valid"].tolist(), [True, True])
        self.assertTrue(x["has_caption"][names.index("hemo")])
        self.assertTrue(x["still_eligible"][names.index("hemo")].any())
        s00 = items["S00"]
        self.assertIn("Pas de sténose", s00["captions"][names.index("aortic")])     # full: AV still in the bag
        bag = self.dataset(self.source(num_stills=1))
        one = {bag[i]["study_id"]: bag[i] for i in range(len(bag))}["S00"]
        uses_av = bool(one["still_eligible"][names.index("aortic")].any())
        self.assertEqual("Pas de sténose" in one["captions"][names.index("aortic")], uses_av)
        batch = collate([x, s00], HashTokenizer(vocab_size=50, max_length=16))
        self.assertEqual(tuple(batch["still_tokens"].shape), (2, 2, 4, DIM))
        self.assertEqual(tuple(batch["still_y"].shape), (2, 2, 2))

    def test_clip_bag_is_v0_and_no_full_caption_without_stills(self):
        v0 = load_routing()
        plain = RadarStudyDataset(self.clips, self.captions, v0, self.cache, num_clips=4)
        with_stills = self.dataset(self.source())
        plain_items = {plain[i]["study_id"]: plain[i] for i in range(len(plain))}
        for i in range(len(with_stills)):
            item = with_stills[i]
            if item["study_id"] in plain_items:
                self.assertEqual(item["clip_ids"], plain_items[item["study_id"]]["clip_ids"])
        self.assertNotIn("X99", plain_items)
        aortic = [s.name for s in v0.trained].index("aortic")
        self.assertNotIn("Pas de sténose", plain_items["S00"]["captions"][aortic])
        self.assertNotIn("still_tokens", plain_items["S00"])

    def test_leak_guard_and_statistics(self):
        with self.assertRaisesRegex(ValueError, "never an input"):
            self.source(cond_columns=["v_top_ms", "y_AV_VMAX"])
        source = self.source()
        self.assertEqual(set(source.scaler), set(TARGETS))
        item = source.item("S00", random.Random(0))
        self.assertEqual(item["demo"].shape, (4,))
        self.assertAlmostEqual(float(item["demo"][1] + item["demo"][2]), 1.0)
        av = int(item["still_tag_ids"].tolist().index(self.routing.still_tag_index("AV")))
        self.assertTrue(item["still_y_mask"][av, 0] and not item["still_y_mask"][av, 1])
        self.assertLess(float(item["still_bias"][av].max()), 1e-6)
        self.assertLess(float(item["still_bias"][av].min()), -1.0)       # S00's AV still has a heat-map


class TestTrainingWithStills(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_one_epoch_and_v0_fingerprint(self):
        paths = make_synthetic(self.root)
        base = synthetic_config(paths, epochs=1)
        base["loss"] = {"use_exact": True}
        v0_model, _ = radar_train.build_model(base, load_routing())
        self.assertFalse(hasattr(v0_model.pooler, "still_marker"))
        routing = load_routing(V1_PATH)
        clips = read_table(paths["clips"])
        stills = make_stills(self.root, sorted(set(clips.study_id))[:4])
        add_full_captions(paths, routing)
        config = synthetic_config(paths, epochs=1)
        config["loss"] = {"use_exact": True}
        config["data"].update(routing=V1_PATH, stills={**stills, "num_stills": 2, "cond_columns": COND,
                                                       "heatmap_column": "echonet_heatmap"})
        config["model"]["stills"] = {"token_grid": list(STILL_GRID), "heatmap_bias": True, "demo_film": True}
        config["heads"] = {"targets": TARGETS, "weight": 0.5, "hidden_dim": 8}
        summary = radar_train.train(config, os.path.join(self.root, "run"))
        self.assertEqual(summary["epochs_completed"], 1)
        self.assertTrue(all(np.isfinite(summary["losses"])))
        lines = [json.loads(line) for line in open(os.path.join(self.root, "run", "log.jsonl"))]
        self.assertIn("measurement", lines[0])
        record = json.load(open(os.path.join(self.root, "run", "fingerprint.json")))
        self.assertIn("stills", record)
        self.assertEqual(set(record["stills"]["scaler"]), set(TARGETS))
        measured = [line for line in lines if any(v["n"] > 0 for v in line["measurement"].values())]
        self.assertTrue(measured and all(line["contributed"] for line in measured))   # a measured still is a step
        with self.assertRaisesRegex(ValueError, "go together"):
            bad = json.loads(json.dumps(config))
            bad["model"].pop("stills")
            radar_train.build_model(bad, routing)


if __name__ == "__main__":
    unittest.main()
