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
from evals.radar_frozen import zero_shot_eval as zse
from src.datasets.radar_stills import StillSource, same_encoder
from src.datasets.radar_study_dataset import RadarStudyDataset, collate, file_sha256, read_table
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
from src.models.radar.routing import DEFAULT_ROUTING, load_routing
from src.models.radar.text_encoder import HashTokenizer
from tests.models.radar_synthetic import DIM, make_synthetic, synthetic_config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import radar_captions as rc  # noqa: E402

V1_PATH = os.path.join(os.path.dirname(DEFAULT_ROUTING), "radar_routing_v1_hemo.yaml")
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
                      "scope": "ev:rvsp" if structure == "hemo" else "full:stenosis"})   # the bags hold a TR and an AV still
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
        s00, aortic = items["S00"], names.index("aortic")
        bag = self.dataset(self.source(num_stills=1))
        one = {bag[i]["study_id"]: bag[i] for i in range(len(bag))}["S00"]
        seen = set()
        for item in (s00, items["S03"], one):       # full: an aortic clip and an AV still in the bag, both
            clip, still = bool(item["eligible"][aortic].any()), bool(item["still_eligible"][aortic].any())
            seen.add((clip, still))
            self.assertEqual("ténose" in item["captions"][aortic], clip and still, item["study_id"])
            if still and not clip:                  # stills alone, and no `doppler` row: no caption
                self.assertFalse(item["has_caption"][aortic])
        self.assertTrue(seen & {(True, True), (False, True)})
        batch = collate([x, s00], HashTokenizer(vocab_size=50, max_length=16))
        self.assertEqual(tuple(batch["still_tokens"].shape), (2, 2, 4, DIM))
        self.assertEqual(tuple(batch["still_y"].shape), (2, 2, 2))

    def test_no_bmode_caption_against_stills_alone_and_no_slot_for_an_unrouted_still(self):
        names = [s.name for s in self.routing.trained]
        text = "Valve aortique calcifiée."
        row = {"study_id": "X99", "structure": "aortic", "caption": text, "caption_id": rc.caption_id(text),
               "normality": "unknown", "normality_status": "provisional", "policy_digest": "synthetic",
               "scope": "bmode"}
        captions = pd.concat([self.captions, pd.DataFrame([row])], ignore_index=True)
        data = RadarStudyDataset(self.clips, captions, self.routing, self.cache, num_clips=4, stills=self.source())
        x = {data[i]["study_id"]: data[i] for i in range(len(data))}["X99"]
        self.assertTrue(x["still_eligible"][names.index("aortic")].any())      # an AV still, and no clip
        self.assertFalse(x["has_caption"][names.index("aortic")])
        self.assertTrue(x["has_caption"][names.index("hemo")])
        table = pd.read_csv(self.still_paths["table"], dtype={"study_id": str, "still_id": str})
        table.loc[table.still_id == "X99-s0", "still_tag"] = np.nan
        source = StillSource(table, self.routing, TokenCache(self.still_paths["cache_dir"]), names, num_stills=1)
        self.assertEqual((source.unrouted, source.unrouted_with_targets), (1, 0))
        kept = StillSource(table, self.routing, TokenCache(self.still_paths["cache_dir"]), names, num_stills=2,
                           target_columns=TARGETS)                 # X99-s0 has a measured AV Vmax: kept for the heads
        self.assertEqual((kept.unrouted, kept.unrouted_with_targets), (0, 1))
        item = kept.item("X99", random.Random(0))
        slot = item["still_tag_ids"].tolist().index(-1)
        self.assertTrue(item["still_valid"][slot] and item["still_y_mask"][slot].any())
        self.assertFalse(item["still_eligible"][:, slot].any())
        tr = self.routing.still_tag_index("TR")
        self.assertEqual({int(source.item("X99", random.Random(seed))["still_tag_ids"][0]) for seed in range(20)},
                         {tr})

    def test_stills_alone_read_the_doppler_caption(self):
        names = [s.name for s in self.routing.trained]
        text = "Sténose aortique serrée."
        rows = [{"study_id": "X99", "structure": "aortic", "caption": c, "caption_id": rc.caption_id(c),
                 "normality": "unknown", "normality_status": "provisional", "policy_digest": "synthetic", "scope": s}
                for c, s in ((text, "doppler:stenosis"), ("Valve aortique calcifiée. " + text, "full:stenosis"))]
        captions = pd.concat([self.captions, pd.DataFrame(rows)], ignore_index=True)
        data = RadarStudyDataset(self.clips, captions, self.routing, self.cache, num_clips=4, stills=self.source())
        x = {data[i]["study_id"]: data[i] for i in range(len(data))}["X99"]
        self.assertEqual(x["captions"][names.index("aortic")], text)
        with self.assertRaisesRegex(ValueError, "scope must be"):
            RadarStudyDataset(self.clips, captions.assign(scope="other"), self.routing, self.cache, num_clips=4)

    @staticmethod
    def items(data):
        return [data[i] for i in range(len(data))]

    def test_an_lvot_still_does_not_switch_on_the_stenosis_clause(self):
        """Finding support: with LVOT stills instead of AV ones the aortic embedding is still fed by stills, but the
        caption says no more than the clips show; a still-only study then has no aortic caption."""
        names = [s.name for s in self.routing.trained]
        a = names.index("aortic")
        table = pd.read_csv(self.still_paths["table"], dtype={"study_id": str, "still_id": str})
        table.loc[table.still_tag == "AV", ["still_tag", "still_type"]] = ["LVOT", "pw_LVOT"]
        lvot = StillSource(table, self.routing, TokenCache(self.still_paths["cache_dir"]), names, num_stills=2)
        with_av = {i["study_id"]: i for i in self.items(self.dataset(self.source()))}
        with_lvot = {i["study_id"]: i for i in self.items(self.dataset(lvot))}
        self.assertIn("Sténose aortique serrée", with_av["S03"]["captions"][a])
        self.assertTrue(with_lvot["S03"]["still_eligible"][a].any())
        self.assertNotIn("ténose", with_lvot["S03"]["captions"][a])
        bmode = {i["study_id"]: i for i in self.items(self.dataset())}
        self.assertEqual(with_lvot["S03"]["captions"][a], bmode["S03"]["captions"][a])
        self.assertFalse(with_lvot["X99"]["has_caption"][a])

    def test_heatmap_contents_are_in_the_record(self):
        before = self.source().record()
        heat = pd.read_csv(self.still_paths["table"]).echonet_heatmap.dropna().iloc[0]
        saved = np.load(heat)
        try:
            np.save(heat, np.ones_like(saved))
            self.assertNotEqual(self.source().record()["heatmaps_sha256"], before["heatmaps_sha256"])
        finally:
            np.save(heat, saved)
        self.assertEqual(self.source().record(), before)
        self.assertIsNone(self.source(heatmap_column=None).record()["heatmaps_sha256"])

    def test_heatmap_repeats_over_the_temporal_positions_of_a_16_frame_still(self):
        root = os.path.join(self.root, "frames16")
        writer = TokenCacheWriter(os.path.join(root, "cache"), (8, 2, 2), DIM)       # 16 frames, tubelet 2
        writer.add("a", torch.randn(32, DIM))
        writer.close()
        table = pd.DataFrame({"study_id": ["X1"], "still_id": ["a"], "still_tag": ["AV"], "still_type": ["cw_AV"],
                              "echonet_heatmap": [pd.read_csv(self.still_paths["table"]).echonet_heatmap.dropna()
                                                  .iloc[0]]})
        source = StillSource(table, self.routing, TokenCache(os.path.join(root, "cache")),
                             [s.name for s in self.routing.trained], num_stills=1, heatmap_column="echonet_heatmap")
        bias = source.item("X1", random.Random(0))["still_bias"][0]
        self.assertEqual(tuple(bias.shape), (32,))
        self.assertTrue(torch.equal(bias.reshape(8, 4), bias[:4].expand(8, 4)))
        self.assertLess(float(bias.min()), -1.0)

    def test_the_two_caches_must_come_from_one_encoder(self):
        def meta(sha=None, dim=DIM, **encoder):
            provenance = {} if sha is None else {"encoder_sha256": sha, "encoder": encoder, "split": "train"}
            return {"embed_dim": dim, "provenance": provenance}
        self.assertIsNone(same_encoder(meta(), meta()))
        self.assertTrue(same_encoder(meta("a", module_name="m"), meta("a", module_name="m")))
        for still, message in ((meta("b", module_name="m"), "different encoders"),
                               (meta("a", module_name="other"), "different encoders"),
                               (meta(), "Only one"), (meta("a", dim=DIM + 1, module_name="m"), "embed_dim")):
            with self.assertRaisesRegex(ValueError, message):
                same_encoder(meta("a", module_name="m"), still)

    def test_age_statistics_are_the_source_s_own(self):
        demo = pd.read_csv(self.still_paths["demographics"], dtype={"study_id": str})
        other = pd.DataFrame({"study_id": ["VAL1", "VAL2"], "age_years": [5.0, 99.0], "sex": ["M", "F"]})
        both = self.source(demographics=pd.concat([demo, other], ignore_index=True))
        self.assertEqual(both.demo_stats, self.source().demo_stats)

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
        self.evaluation(config, routing, paths, stills)
        self.head_evaluation(config, stills)
        self.head_experiments(config, routing)

    def head_evaluation(self, config, stills):
        """The heads are scored per target in the targets' units, with the outside prediction on the same stills."""
        from evals.radar_frozen import eval_heads
        checkpoint = os.path.join(self.root, "run", "latest.pt")
        out = tempfile.mkdtemp()
        try:
            with self.assertRaisesRegex(ValueError, "trained on"):
                eval_heads.evaluate(config, checkpoint, stills["table"], out, split="train")
            report = eval_heads.evaluate(config, checkpoint, stills["table"], out, split="train", samples=20,
                                         allow_training_stills=True)
            self.assertEqual(set(report["targets"]), set(TARGETS))
            av = report["targets"]["y_AV_VMAX"]
            self.assertEqual(av["head"]["n"], 5)                         # one AV still per study
            self.assertGreater(av["head"]["metrics"]["mae"], 0.0)
            self.assertEqual(av["paired_stills"], 1)                     # the one study with an outside prediction
            self.assertFalse(report["baseline_overlap_checked"])
            self.assertEqual(report["head_inputs"], {"visual": True, "cond_columns": COND, "demographics": True})
            table = pd.read_csv(os.path.join(out, "head_predictions.csv"))
            self.assertEqual(len(table), 10)
            self.assertTrue((table.measured > 0).all() and (table.prediction > 0).all())
            with self.assertRaises(ValueError):
                eval_heads.evaluate(config, checkpoint, stills["table"], os.path.join(REPO, "out"), split="val")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def head_experiments(self, config, routing):
        """The three head experiments are configs of one code path: chosen side inputs, or side inputs alone."""
        visual = json.loads(json.dumps(config))
        visual["heads"].update(cond_columns=COND[:2], demographics=False)
        model, _ = radar_train.build_model(visual, routing)
        self.assertEqual(model.head_cond_index, [0, 1])
        self.assertIsNone(model.heads.film)
        self.assertEqual(model.heads.cond.fc1.in_features, 2)
        side = json.loads(json.dumps(config))
        side["heads"]["visual"] = False
        self.assertFalse(radar_train.build_model(side, routing)[0].heads.use_visual)
        bad = json.loads(json.dumps(config))
        bad["heads"]["cond_columns"] = ["y_AV_VMAX"]
        with self.assertRaisesRegex(ValueError, "not in data.stills.cond_columns"):
            radar_train.build_model(bad, routing)

    def evaluation(self, config, routing, paths, stills):
        """The evaluator reads the still stream of a checkpoint trained with it, or refuses."""
        checkpoint = os.path.join(self.root, "run", "latest.pt")
        device = torch.device("cpu")
        model, _, state = zse.load_model(config, checkpoint, routing, device)
        with self.assertRaisesRegex(ValueError, "training split"):      # X99 trained through its stills alone
            zse.check_not_trained_on(config, state, ["X99"])
        self.assertIn("X99", zse.check_not_trained_on(config, state, ["HELD-OUT"]))
        moved = json.loads(json.dumps(config))
        moved["data"]["stills"]["table"] = paths["clips"]
        with self.assertRaisesRegex(ValueError, "still table the checkpoint trained on"):
            zse.check_not_trained_on(moved, state, ["HELD-OUT"])
        out = tempfile.mkdtemp()
        try:
            for kwargs, message in (({}, "trained with Doppler stills"), ({"clip_only": True}, "demographics")):
                with self.assertRaisesRegex(ValueError, message):
                    zse.evaluate(config, checkpoint, paths["clips"], "unused.csv", out, **kwargs)
        finally:
            shutil.rmtree(out, ignore_errors=True)
        source = zse.read_eval_stills(config, state, routing, model, stills["table"], "train")
        self.assertEqual(source.demo_stats, state["fingerprint"]["stills"]["demo_stats"])
        clips = read_table(paths["clips"])
        data = RadarStudyDataset(clips, pd.DataFrame(columns=list(zse.CAPTION_COLUMNS)), routing,
                                 TokenCache(paths["cache_dir"]), num_clips=config["data"].get("num_clips", 12),
                                 training=False, stills=source)
        z, has = zse.embed_studies(model, data, device, num_bags=2, batch_size=3)     # demo_film: no crash
        x, hemo = [s for s, _ in data.studies].index("X99"), [s.name for s in routing.trained].index("hemo")
        self.assertTrue(has[:, x, hemo].all())                         # supported by its TR still
        self.assertTrue(torch.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
