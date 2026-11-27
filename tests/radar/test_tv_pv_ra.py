"""Tricuspid, pulmonic and right atrium training use: the loss's uniform fallback (`loss.uniform_fallback`) and
per-structure `min_pairs`, its fallback counts in `epochs.jsonl`, routing registry v2 (generic PSAX for the pulmonic
and tricuspid valves, a trained right atrium, SUBCOSTAL for the right heart and the pericardium), and colour-flow
caption rows read by the study dataset (`color_flow_captions`)."""

import collections
import os
import shutil
import tempfile
import unittest

import pandas as pd
import torch
import torch.nn.functional as F

from evals.radar_frozen.train import FALLBACK_KINDS, SKIP_REASONS, fingerprint, structure_summary
from src.datasets.radar_study_dataset import RadarStudyDataset
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
from src.models.radar.loss import StructureContrastiveLoss, contrastive_loss
from src.models.radar.routing import load_routing

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS = os.path.join(REPO, "configs", "radar")
V0, V1, V2 = (os.path.join(CONFIGS, f"radar_routing_v{v}.yaml") for v in (0, 1, 2))


def rows(n, s, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = F.normalize(torch.randn(n, s, 8, generator=g), dim=-1)
    t = F.normalize(torch.randn(n, s, 8, generator=g), dim=-1)
    return z, t


class TestUniformFallback(unittest.TestCase):

    def test_off_by_default_skips_as_before(self):
        z, t = rows(4, 1)
        loss = StructureContrastiveLoss(["tricuspid"], use_exact=True)
        value, stats = loss(z, t, torch.ones(4, 1, dtype=torch.bool), torch.full((4, 1), 7))
        self.assertEqual(stats["tricuspid"]["skipped"], "uniform_targets")
        self.assertIsNone(stats["tricuspid"]["fallback"])
        self.assertEqual(float(value), 0.0)

    def test_identical_captions_fall_back_without_moving_the_temperature(self):
        z, t = rows(4, 1)
        t = t[:1].expand(4, 1, 8).clone()          # one caption: the same text embedding on every row
        loss = StructureContrastiveLoss(["tricuspid"], use_exact=True, uniform_fallback="identity")
        value, stats = loss(z, t, torch.ones(4, 1, dtype=torch.bool), torch.full((4, 1), 7))
        self.assertIsNone(stats["tricuspid"]["skipped"])
        self.assertEqual(stats["tricuspid"]["fallback"], "identity_identical")
        expected = contrastive_loss(z[:, 0], t[:, 0], torch.eye(4), loss.effective_temperature().detach())
        self.assertAlmostEqual(float(value), float(expected), places=5)
        value.backward()
        self.assertEqual(float(loss.temperature.grad), 0.0)

    def test_distinct_captions_fall_back_with_the_temperature(self):
        """Under N, all-normal rows with different captions are uniform: identity targets, temperature trained."""
        z, t = rows(4, 1)
        loss = StructureContrastiveLoss(["tricuspid"], use_exact=True, use_normal=True, uniform_fallback="identity")
        normal = torch.ones(4, 1, dtype=torch.bool)
        value, stats = loss(z, t, normal, torch.arange(4).view(4, 1), normal, ~normal)
        self.assertEqual(stats["tricuspid"]["fallback"], "identity")
        value.backward()
        self.assertNotEqual(float(loss.temperature.grad), 0.0)

    def test_min_distinct_and_per_structure(self):
        z, t = rows(4, 2)
        ids = torch.tensor([[7, 1], [7, 1], [7, 1], [7, 1]])
        valid = torch.ones(4, 2, dtype=torch.bool)
        loss = StructureContrastiveLoss(["tricuspid", "pulmonic"], uniform_fallback={"pulmonic": "identity"})
        _, stats = loss(z, t, valid, ids)
        self.assertEqual(stats["tricuspid"]["skipped"], "uniform_targets")
        self.assertEqual(stats["pulmonic"]["fallback"], "identity_identical")
        strict = StructureContrastiveLoss(["tricuspid", "pulmonic"], uniform_fallback="identity",
                                          uniform_fallback_min_distinct=2)
        _, stats = strict(z, t, valid, ids)
        self.assertEqual({s["skipped"] for s in stats.values()}, {"uniform_targets"})

    def test_a_non_uniform_step_is_unchanged(self):
        z, t = rows(4, 1)
        ids = torch.tensor([[1], [1], [2], [3]])
        valid = torch.ones(4, 1, dtype=torch.bool)
        base, _ = StructureContrastiveLoss(["tricuspid"])(z, t, valid, ids)
        with_option, stats = StructureContrastiveLoss(["tricuspid"], uniform_fallback="identity")(z, t, valid, ids)
        self.assertEqual(float(base), float(with_option))
        self.assertIsNone(stats["tricuspid"]["fallback"])

    def test_min_pairs_per_structure(self):
        z, t = rows(3, 2)
        ids = torch.tensor([[1, 1], [2, 2], [3, 3]])
        loss = StructureContrastiveLoss(["lv", "ra"], min_pairs={"ra": 4})
        _, stats = loss(z, t, torch.ones(3, 2, dtype=torch.bool), ids)
        self.assertIsNone(stats["lv"]["skipped"])
        self.assertEqual(stats["ra"]["skipped"], "too_few_rows")

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "not trained"):
            StructureContrastiveLoss(["lv"], uniform_fallback={"ra": "identity"})
        with self.assertRaisesRegex(ValueError, "identity"):
            StructureContrastiveLoss(["lv"], uniform_fallback="exact")
        with self.assertRaisesRegex(ValueError, "not trained"):
            StructureContrastiveLoss(["lv"], min_pairs={"ra": 3})
        with self.assertRaisesRegex(ValueError, "at least 2"):
            StructureContrastiveLoss(["lv"], min_pairs={"lv": 1})

    def test_summary_counts_fallbacks(self):
        contributing = {"tricuspid": [(3, 3, 0), (4, 6, 0), (5, 1, 0)], "lv": [(40, 2, 0)]}
        skips = {"tricuspid": collections.Counter({"too_few_rows": 1}), "lv": collections.Counter()}
        fallbacks = {"tricuspid": collections.Counter({"identity_identical": 2}), "lv": collections.Counter()}
        out = structure_summary(contributing, skips, 4, fallbacks)
        self.assertEqual(out["tricuspid"]["fallback_identity_identical"], 2)
        self.assertEqual(out["tricuspid"]["fallback_identity"], 0)
        self.assertEqual(out["tricuspid"]["use_fraction"], 0.75)
        self.assertEqual(set(FALLBACK_KINDS), {"identity", "identity_identical"})
        plain = structure_summary(contributing, skips, 4)
        self.assertFalse(any(k.startswith("fallback_") for entry in plain.values() for k in entry))
        self.assertTrue(all(f"skipped_{r}" in plain["lv"] for r in SKIP_REASONS))


class TestRoutingV2(unittest.TestCase):

    def setUp(self):
        self.v1, self.v2 = load_routing(V1), load_routing(V2)

    def test_earlier_registries_are_unchanged(self):
        self.assertFalse(self.v1.structure("ra").train)
        self.assertEqual(self.v1.structure("pulmonic").views, ("PSAX_AV",))
        self.assertNotIn("PSAX", self.v1.structure("tricuspid").views)
        self.assertEqual(load_routing(V0).version, 0)
        self.assertTrue(all(not s.weak_views for s in self.v1.structures))

    def test_v2_routes(self):
        r = self.v2
        self.assertEqual(r.version, 2)
        self.assertEqual(set(r.structure("pulmonic").views), {"PSAX", "PSAX_AV"})
        self.assertEqual(r.structure("pulmonic").weak_views, ("PSAX",))
        self.assertIn("PSAX", r.structure("tricuspid").views)
        self.assertEqual(r.structure("tricuspid").weak_views, ("PSAX",))
        ra = r.structure("ra")
        self.assertTrue(ra.train)
        self.assertEqual(ra.views[0], "A4C")
        self.assertNotIn("A4C_LV", ra.views)
        for name in ("rv", "pericardium", "ra"):
            self.assertIn("SUBCOSTAL", r.structure(name).views)
            self.assertIn("SUBCOSTAL", r.structure(name).weak_views)
        self.assertEqual(r.still_structs, self.v1.still_structs)
        self.assertEqual([s.name for s in r.trained if s.name != "ra"], [s.name for s in self.v1.trained])

    def test_eligibility(self):
        names = ["tricuspid", "pulmonic", "ra"]
        for view, expected_v2 in (("PSAX", [True, True, False]), ("A4C", [True, False, True]),
                                  ("PSAX_AV", [True, True, True])):
            ids = torch.tensor([[self.v2.view_index(view)]])
            got = self.v2.clip_eligibility(ids, torch.ones(1, 1, dtype=torch.bool), names)[0, :, 0].tolist()
            self.assertEqual(got, expected_v2, view)
        ids = torch.tensor([[self.v1.view_index("PSAX")]])
        got = self.v1.clip_eligibility(ids, torch.ones(1, 1, dtype=torch.bool), ["tricuspid", "pulmonic"])
        self.assertEqual(got[0, :, 0].tolist(), [False, False])

    def test_weak_views_must_be_routed(self):
        root = tempfile.mkdtemp()
        try:
            with open(V2, encoding="utf-8") as f:
                text = f.read()
            bad = os.path.join(root, "bad.yaml")
            with open(bad, "w", encoding="utf-8") as f:
                f.write(text.replace("views: [PSAX, PSAX_AV]\n    weak_views: [PSAX]",
                                     "views: [PSAX_AV]\n    weak_views: [PSAX]"))
            with self.assertRaisesRegex(ValueError, "weak_views"):
                load_routing(bad)
        finally:
            shutil.rmtree(root, ignore_errors=True)


GRID, DIM = (1, 2, 2), 4
BMODE, COLOR = "bmode", "color_flow"


class TestColorFlowCaptions(unittest.TestCase):
    """A valve's `color_flow` row is read only for a bag that holds a colour clip routed to it."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        writer = TokenCacheWriter(os.path.join(cls.root, "cache"), GRID, DIM)
        clips = []
        for study, entries in {"S0": [("A4C", COLOR), ("PLAX", BMODE)], "S1": [("A4C", BMODE), ("PLAX", BMODE)],
                               "S2": [("PLAX", COLOR), ("A4C", BMODE)]}.items():
            for j, (view, acquisition) in enumerate(entries):
                clip = f"{study}-c{j}"
                writer.add(clip, torch.zeros(GRID[0] * GRID[1] * GRID[2], DIM))
                clips.append({"study_id": study, "clip_id": clip, "view": view, "acquisition": acquisition})
        writer.close()
        cls.cache = TokenCache(os.path.join(cls.root, "cache"))
        cls.clips = pd.DataFrame(clips)
        caption = []
        for study in ("S0", "S1", "S2"):
            caption.append({"study_id": study, "structure": "tricuspid", "scope": BMODE, "caption_id": 1,
                            "caption": "Tricuspid valve has a normal appearance.", "normality": "normal"})
            caption.append({"study_id": study, "structure": "tricuspid", "scope": COLOR, "caption_id": 10,
                            "caption": "Tricuspid valve has a normal appearance. Mild tricuspid regurgitation.",
                            "normality": "abnormal"})
        cls.captions = pd.DataFrame(caption)
        cls.routing = load_routing(V2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def dataset(self, **kwargs):
        return RadarStudyDataset(self.clips, self.captions, self.routing, self.cache, num_clips=4, training=False,
                                 **kwargs)

    def test_off_refuses_the_rows(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            self.dataset()
        plain = self.dataset_without_color_rows()
        s = plain.structures.index("tricuspid")
        self.assertEqual([int(plain[i]["caption_ids"][s]) for i in range(3)], [1, 1, 1])

    def dataset_without_color_rows(self):
        return RadarStudyDataset(self.clips, self.captions[self.captions.scope == BMODE], self.routing, self.cache,
                                 num_clips=4, training=False)

    def test_on_reads_the_row_of_a_routed_colour_clip(self):
        data = self.dataset(color_flow_captions=True)
        s = data.structures.index("tricuspid")
        by_study = {data[i]["study_id"]: data[i] for i in range(len(data))}
        self.assertEqual(int(by_study["S0"]["caption_ids"][s]), 10)    # colour A4C: routed to the valve
        self.assertTrue(bool(by_study["S0"]["abnormal"][s]))
        self.assertEqual(int(by_study["S1"]["caption_ids"][s]), 1)     # no colour clip
        self.assertEqual(int(by_study["S2"]["caption_ids"][s]), 1)     # colour PLAX: not routed to the valve

    def test_fingerprint_keys_off_are_absent(self):
        """A config with the new keys set off has the fingerprint of a config without them."""
        class Cache:
            meta = {}
        root = self.root
        clips = os.path.join(root, "c.csv")
        captions = os.path.join(root, "k.csv")
        self.clips.to_csv(clips, index=False)
        self.captions.to_csv(captions, index=False)
        base = {"data": {"clips": clips, "captions": captions}, "loss": {"use_exact": True}}
        off = {"data": {"clips": clips, "captions": captions, "color_flow_captions": False},
               "loss": {"use_exact": True, "uniform_fallback": None}}
        on = {"data": {"clips": clips, "captions": captions}, "loss": {"use_exact": True,
                                                                        "uniform_fallback": "identity"}}
        a, b, c = (fingerprint(cfg, self.routing, Cache(), 3) for cfg in (base, off, on))
        self.assertEqual(a["config"], b["config"])
        self.assertNotEqual(a["config"], c["config"])


if __name__ == "__main__":
    unittest.main()
