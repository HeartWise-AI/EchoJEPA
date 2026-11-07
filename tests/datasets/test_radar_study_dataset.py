# tests/datasets/test_radar_study_dataset.py

"""RADAR data path: the token cache (round trip, conversion to the storage dtype before the finiteness check,
content identity and its checks, refusals), view-balanced clip choice, study items (zero padding, routing
eligibility, captions and flags, seeded bags that depend neither on the reader nor on the clip table's row order),
the global-batch sampler and collation."""

import os
import random
import shutil
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from evals.radar_frozen.train import fingerprint
from src.datasets.radar_study_dataset import RadarStudyDataset, StudyBatchSampler, check_views, collate, read_table
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
from src.datasets.study_bags import check_study_ids, select_clips
from src.models.radar.routing import load_routing
from src.models.radar.text_encoder import HashTokenizer
from tests.models.radar_synthetic import DIM, GRID, make_synthetic

L = GRID[0] * GRID[1] * GRID[2]


class TestTokenCache(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_round_trip_and_names(self):
        writer = TokenCacheWriter(self.root, GRID, DIM, provenance={"encoder_sha256": "abc"})
        tokens = torch.randn(L, DIM)
        writer.add("study/1234567/clip.mp4", tokens, file_sha256="f" * 64)
        meta = writer.close()
        self.assertEqual(meta["n_clips"], 1)
        cache = TokenCache(self.root, token_grid=GRID, embed_dim=DIM, dtype="float16")
        self.assertIn("study/1234567/clip.mp4", cache)
        self.assertTrue(torch.allclose(cache.load("study/1234567/clip.mp4"), tokens.half().float()))
        files = [f for _, _, names in os.walk(self.root) for f in names if f.endswith(".npy")]
        self.assertTrue(all("1234567" not in f and "clip" not in f for f in files))
        self.assertEqual(cache.index.file_sha256.tolist(), ["f" * 64])

    def test_refusals(self):
        writer = TokenCacheWriter(self.root, GRID, DIM)
        with self.assertRaisesRegex(ValueError, "shape"):
            writer.add("a", torch.randn(L + 1, DIM))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            writer.add("a", torch.full((L, DIM), float("nan")))
        writer.add("a", torch.randn(L, DIM))
        with self.assertRaisesRegex(ValueError, "already"):
            writer.add("a", torch.randn(L, DIM))
        with self.assertRaises(FileNotFoundError):
            TokenCache(self.root)                         # meta.json is written last
        writer.close()
        with self.assertRaises(FileExistsError):
            TokenCacheWriter(self.root, GRID, DIM)
        with self.assertRaisesRegex(ValueError, "embed_dim"):
            TokenCache(self.root, embed_dim=DIM + 1)
        with self.assertRaisesRegex(ValueError, "token_grid"):
            TokenCache(self.root, token_grid=(1, 2, 2))

    def test_reserved_index_fields(self):
        writer = TokenCacheWriter(self.root, GRID, DIM)
        with self.assertRaisesRegex(ValueError, "written by the cache"):
            writer.add("a", torch.randn(L, DIM), token_sha256="x")

    def test_conversion_before_validation(self):
        writer = TokenCacheWriter(self.root, GRID, DIM)
        bf16 = torch.randn(L, DIM).bfloat16()
        writer.add("bf16", bf16)                                    # NumPy has no bfloat16: converted first
        writer.add("float64", np.random.default_rng(0).standard_normal((L, DIM)))
        with self.assertRaisesRegex(ValueError, "Non-finite.*float16"):
            writer.add("overflow", torch.full((L, DIM), 1e6))      # finite in float32, infinite in float16
        writer.add("overflow", torch.full((L, DIM), 1e3))          # nothing was recorded for the refused clip
        writer.close()
        cache = TokenCache(self.root)
        self.assertTrue(torch.equal(cache.load("bf16"), bf16.float().half().float()))
        self.assertTrue(torch.isfinite(cache.load("overflow")).all())
        self.assertEqual(len(cache), 3)
        cache.verify()

    def test_content_identity(self):
        a, b = torch.randn(L, DIM), torch.randn(L, DIM)
        roots = [os.path.join(self.root, name) for name in ("one", "swapped")]
        for root, (first, second) in zip(roots, [(a, b), (b, a)]):
            writer = TokenCacheWriter(root, GRID, DIM)
            writer.add("x", first)
            writer.add("y", second)
            writer.close()
        one, swapped = (TokenCache(r) for r in roots)
        self.assertNotEqual(one.meta["index_sha256"], swapped.meta["index_sha256"])
        tables = {}
        for name in ("clips", "captions"):
            tables[name] = os.path.join(self.root, f"{name}.csv")
            with open(tables[name], "w") as f:
                f.write("study_id\nS0\n")
        config = {"data": {**tables, "cache_dir": roots[0]}}
        routing = load_routing()
        self.assertNotEqual(fingerprint(config, routing, one, 1)["sha256"],
                            fingerprint(config, routing, swapped, 1)["sha256"])

        index_path = os.path.join(roots[0], "index.csv")           # swap the two clips' files in the index
        index = pd.read_csv(index_path, dtype=str)
        index["file"] = index["file"].iloc[::-1].tolist()
        index.to_csv(index_path, index=False)
        with self.assertRaisesRegex(ValueError, "digest"):
            TokenCache(roots[0])

        token_path = os.path.join(roots[1], swapped.files["x"])    # change one token file behind the index
        np.save(token_path, np.zeros((L, DIM), dtype=np.float16))
        with self.assertRaisesRegex(ValueError, "1 token files differ.*'x'"):
            TokenCache(roots[1]).verify()


class TestSelection(unittest.TestCase):

    def test_view_balanced_covers_views_first(self):
        clips = [("a", "A4C"), ("b", "A4C"), ("c", "A4C"), ("d", "PLAX"), ("e", "PSAX_AV")]
        chosen = select_clips(clips, 3, "view_balanced", random.Random(0))
        self.assertEqual(sorted(clips[i][1] for i in chosen), ["A4C", "PLAX", "PSAX_AV"])
        self.assertEqual(len(select_clips(clips, 9, "view_balanced", random.Random(0))), 5)
        self.assertEqual(select_clips(clips, 4, "random", random.Random(3)),
                         select_clips(clips, 4, "random", random.Random(3)))
        with self.assertRaises(ValueError):
            select_clips(clips, 2, "first", random.Random(0))

    def test_registry_families_keep_psax_av_in_the_bag(self):
        families = load_routing().view_families
        clips = [("a", "A4C"), ("b", "A4C_ZOOM"), ("c", "A4C_LV"), ("d", "PSAX"), ("e", "PSAX"), ("f", "PSAX_AV")]
        grouped = [(clip, families.get(view, view)) for clip, view in clips]
        for seed in range(20):
            views = [clips[i][1] for i in select_clips(grouped, 3, "view_balanced", random.Random(seed))]
            self.assertIn("PSAX_AV", views)
            self.assertIn("PSAX", views)
            self.assertEqual(sum(view.startswith("A4C") for view in views), 1)


class TestStudyDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(cls.root)
        cls.routing = load_routing()
        cls.cache = TokenCache(cls.paths["cache_dir"], token_grid=GRID, embed_dim=DIM)
        cls.clips = read_table(cls.paths["clips"])
        cls.captions = read_table(cls.paths["captions"], id_columns=("study_id",))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def dataset(self, routing=None, **kw):
        return RadarStudyDataset(self.clips, self.captions, routing or self.routing, self.cache, num_clips=4, **kw)

    def test_a_study_id_that_is_not_plain_stops_the_dataset(self):
        for study in ("'S00'", " S00", "S00 ", ""):
            with self.subTest(study), self.assertRaisesRegex(ValueError, "study id"):
                clips = self.clips.assign(study_id=self.clips.study_id.where(self.clips.study_id != "S00", study))
                RadarStudyDataset(clips, self.captions, self.routing, self.cache)

    def test_a_view_the_registry_does_not_know_stops_the_dataset(self):
        with self.assertRaisesRegex(ValueError, "does not know"):
            RadarStudyDataset(self.clips.assign(view="Subcostal"), self.captions, self.routing, self.cache)

    def test_item(self):
        ds = self.dataset()
        item = ds[0]                                       # S00: captioned, A4C + 2 clips, one padded slot
        S = len(self.routing.trained)
        self.assertEqual(item["tokens"].shape, (4, L, DIM))
        self.assertEqual(item["clip_valid"].tolist(), [True, True, True, False])
        self.assertTrue(torch.equal(item["tokens"][3], torch.zeros(L, DIM)))
        self.assertEqual(item["view_ids"][3].item(), -1)
        self.assertEqual(item["eligible"].shape, (S, 4))
        self.assertFalse(item["eligible"][:, 3].any())
        lv = ds.structures.index("lv")
        a4c = item["view_ids"].tolist().index(self.routing.view_index("A4C"))
        self.assertTrue(item["eligible"][lv, a4c])
        self.assertEqual(item["has_caption"].sum().item(), 3)   # lv, pericardium, aortic
        peri = ds.structures.index("pericardium")
        self.assertTrue(item["normal"][peri] and not item["abnormal"][peri])
        self.assertTrue(item["captions"][lv].endswith("FEVG estimée visuellement : 40 %."))
        uncaptioned = ds[15]
        self.assertFalse(uncaptioned["has_caption"].any())
        self.assertEqual(uncaptioned["captions"], [""] * S)

    def test_bags_depend_on_seed_epoch_and_study_only(self):
        a, b = self.dataset(seed=1), self.dataset(seed=1)
        self.assertEqual(a[12]["clip_ids"], b[12]["clip_ids"])
        for num_clips in (2, 4):                                   # bags do not depend on the table's row order
            kw = {"num_clips": num_clips, "seed": 1}
            straight = RadarStudyDataset(self.clips, self.captions, self.routing, self.cache, **kw)
            reordered = RadarStudyDataset(self.clips.iloc[::-1], self.captions, self.routing, self.cache, **kw)
            shuffled = RadarStudyDataset(self.clips.sample(frac=1.0, random_state=3), self.captions, self.routing,
                                         self.cache, **kw)
            for i in range(len(straight)):
                self.assertEqual(reordered[i]["clip_ids"], straight[i]["clip_ids"])
                self.assertEqual(shuffled[i]["clip_ids"], straight[i]["clip_ids"])
        epochs = set()
        for epoch in range(6):
            a.set_epoch(epoch)
            epochs.add(tuple(a[13]["clip_ids"]))
        self.assertGreater(len(epochs), 1)
        ev = self.dataset(training=False, eval_seed=3)
        ev.set_epoch(5)
        self.assertEqual(ev[13]["clip_ids"], self.dataset(training=False, eval_seed=3)[13]["clip_ids"])

    def test_view_families_steer_the_choice(self):
        apical = {"A2C": "A4C", "A3C": "A4C"}
        ds = self.dataset(routing=replace(self.routing, view_families=apical))
        ds.num_clips = 2
        for i, (study, clips) in enumerate(ds.studies):
            families = {apical.get(v, v) for _, v, _ in clips}
            item = ds[i]
            chosen = [apical.get(v, v) for c, v, _ in clips if c in item["clip_ids"]]
            self.assertEqual(len(set(chosen)), min(2, len(families)), study)
            views = {c: v for c, v, _ in clips}
            for slot, clip in enumerate(item["clip_ids"]):        # eligibility still uses the clip's own view
                self.assertEqual(item["view_ids"][slot].item(), self.routing.view_index(views[clip]))

    def test_refusals(self):
        bad = self.captions.copy()
        bad.loc[0, "normality"] = "fine"
        with self.assertRaisesRegex(ValueError, "normality"):
            RadarStudyDataset(self.clips, bad, self.routing, self.cache)
        dup = pd.concat([self.captions, self.captions.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "more than one caption"):
            RadarStudyDataset(self.clips, dup, self.routing, self.cache)
        empty = self.captions.copy()
        empty.loc[0, "caption"] = np.nan
        with self.assertRaisesRegex(ValueError, "empty"):
            RadarStudyDataset(self.clips, empty, self.routing, self.cache)
        with self.assertRaisesRegex(ValueError, "no clip"):
            RadarStudyDataset(self.clips[self.clips.study_id != "S00"], self.captions, self.routing, self.cache)
        extra = pd.concat([self.clips, pd.DataFrame([{"study_id": "S99", "clip_id": "nope", "view": "A4C",
                                                      "acquisition": "bmode", "split": "train"}])])
        with self.assertRaisesRegex(ValueError, "not in the token cache"):
            RadarStudyDataset(extra, self.captions, self.routing, self.cache)

    def test_collate(self):
        ds = self.dataset()
        batch = collate([ds[0], ds[15]], HashTokenizer(200, 32))
        S = len(self.routing.trained)
        self.assertEqual(batch["tokens"].shape, (2, 4, L, DIM))
        self.assertEqual(batch["text_ids"].shape[:2], (2, S))
        self.assertEqual(batch["text_mask"][1].sum().item(), S)     # every slot of S15 is [CLS] only
        self.assertEqual(batch["study_id"], ["S00", "S15"])


class TestReaders(unittest.TestCase):

    def test_a_caption_that_reads_like_a_missing_value_stays_text(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "captions.csv")
            table = pd.DataFrame({"study_id": ["0012", "NA"], "caption": ["NA", "None"], "note": ["", "x"]})
            table.to_csv(path, index=False)
            table = read_table(path, ("study_id",))
        self.assertEqual(table.study_id.tolist(), ["0012", "NA"])
        self.assertEqual(table.caption.tolist(), ["NA", "None"])
        self.assertEqual(table.note.tolist(), ["", "x"])

    def test_a_plain_study_id_passes_and_anything_around_it_does_not(self):
        check_study_ids(["1.2.840", "S00", "O'Neil", "0012", 12])
        for study in ("'S0", "S0'", '"S0', 'S0"', " S0", "S0 ", "\tS0", "S0\n", "", None, float("nan")):
            with self.subTest(study), self.assertRaisesRegex(ValueError, "1 clips have a study id"):
                check_study_ids(["S1", study])

    def test_a_view_the_registry_does_not_know_is_refused(self):
        routing = load_routing()
        check_views(pd.DataFrame({"view": list(routing.view_classes)}), routing)
        for view in ("Subcostal", "", None):
            with self.subTest(view), self.assertRaisesRegex(ValueError, "does not know"):
                check_views(pd.DataFrame({"view": ["A4C", view]}), routing)


class TestSampler(unittest.TestCase):

    def test_ranks_split_each_global_batch(self):
        single = list(StudyBatchSampler(10, 4, seed=2))
        self.assertEqual(len(single), 2)                             # the partial last batch is dropped
        ranks = [list(StudyBatchSampler(10, 4, rank=r, world_size=2, seed=2)) for r in range(2)]
        for k, batch in enumerate(single):
            self.assertEqual(sorted(ranks[0][k] + ranks[1][k]), sorted(batch))
        sampler = StudyBatchSampler(10, 4, seed=2)
        sampler.set_epoch(1)
        self.assertNotEqual(list(sampler), single)
        with self.assertRaises(ValueError):
            StudyBatchSampler(10, 3, world_size=2)
        with self.assertRaises(ValueError):
            StudyBatchSampler(3, 4)


if __name__ == "__main__":
    unittest.main()
