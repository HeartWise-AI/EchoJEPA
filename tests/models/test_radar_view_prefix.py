# tests/models/test_radar_view_prefix.py

"""Captions with a view prefix (`data.view_prefix`, `src/models/radar/view_prefix.py`), on synthetic data:

- the phrase file names every view family of the registry once, and a prefix lists the distinct phrases of the
  feeding views in the file's order ("Apical four-chamber view.", "A and b views.", "A, b and c views."), with
  nothing for an unphrased view;
- a caption id with a prefix is equal only for the same views and the same caption (A1's exact-caption positives);
- `possible_prefixes` holds the prefix of every bag that can be drawn, and only reachable ones when every family is
  in every bag;
- the dataset prefixes each caption with the views that feed its structure in the bag, leaves everything else of the
  item unchanged, and without the option leaves items as they were;
- training: the fingerprint of a run without the option is unchanged (absent or false), a frozen encoder holds every
  prefixed caption before the first step, and caption lengths are measured with the longest prefix;
- zero-shot: `prompt_view_prefix` scores each study's bag with prompts after that bag's prefix, the report says which
  prompts were used, and a checkpoint trained without a prefix gets the report it always had.
"""

import copy
import json
import os
import random
import shutil
import tempfile
import unittest
from itertools import permutations
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

from evals.radar_frozen import train as radar_train
from evals.radar_frozen import zero_shot_eval as zs
from src.datasets.radar_study_dataset import RadarStudyDataset, read_table
from src.datasets.radar_token_cache import TokenCache, TokenCacheWriter
from src.datasets.study_bags import choose_bag
from src.models.radar.loss import caption_targets
from src.models.radar.routing import load_routing
from src.models.radar.view_prefix import (
    DEFAULT_VIEW_PHRASES,
    ViewPhrases,
    longest_prefixes,
    prefixed,
    prefixed_caption_id,
    view_phrases,
)
from src.models.radar.zero_shot import load_prompt_tasks
from tests.models.radar_synthetic import DIM, GRID, make_synthetic, synthetic_config

ROUTING = load_routing()
FROZEN = {"backbone": "tiny", "vocab_size": 200, "hidden_size": 16, "num_layers": 2, "num_heads": 2,
          "max_length": 64, "frozen": True, "pooling": "mean", "head": "mlp", "head_hidden_dim": 12}
A1 = {"use_exact": True, "use_normal": False, "use_abnormal_teacher": False}
PROMPTS = {
    "ef": {"mode": "regression", "structure": "lv", "target": "lvef",
           "label_sources": ["FEVG estimée visuellement : <#> %."], "range": [30, 60]},
    "effusion": {"mode": "binary", "structure": "pericardium",
                 "label_sources": ["Épanchement péricardique modéré."],
                 "negative_sources": ["Pas d'épanchement péricardique."]},
    "av": {"mode": "categorical", "structure": "aortic", "target": "av_grade",
           "label_sources": ["Valve aortique calcifiée.", "Valve aortique de structure normale."], "values": [1, 0]},
}


def write_yaml(path, content):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)
    return path


class TestViewPhrases(unittest.TestCase):

    def setUp(self):
        self.vp = ViewPhrases(DEFAULT_VIEW_PHRASES, ROUTING)
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_view_class_takes_the_phrase_of_its_family(self):
        for view in ("A4C", "A4C_LV", "A4C_ZOOM"):
            self.assertEqual(self.vp.phrase(view), "apical four-chamber")
        self.assertEqual(self.vp.phrase("PSAX_AV"), "parasternal short-axis aortic valve level")
        self.assertNotEqual(self.vp.phrase("PSAX_AV"), self.vp.phrase("PSAX"))
        self.assertIsNone(self.vp.phrase("OTHER"))
        self.assertIsNone(self.vp.phrase("NOT_A_VIEW"))
        routed = {v for s in ROUTING.structures for v in s.views}
        self.assertTrue(all(self.vp.phrase(v) for v in routed))          # every routed view has a phrase

    def test_prefix_text_for_one_two_and_three_views_in_the_fixed_order(self):
        self.assertEqual(self.vp.prefix(["A4C"]), "Apical four-chamber view.")
        self.assertEqual(self.vp.prefix(["A4C", "A4C_LV", "A4C_ZOOM"]), "Apical four-chamber view.")
        self.assertEqual(self.vp.prefix(["A4C", "PLAX_ZOOM"]), "Parasternal long-axis and apical four-chamber views.")
        three = "Parasternal long-axis, apical four-chamber and apical two-chamber views."
        for order in permutations(["A2C", "A4C", "PLAX", "A2C_LV"]):
            self.assertEqual(self.vp.prefix(order), three)
        self.assertEqual(self.vp.prefix(["OTHER"]), "")
        self.assertEqual(self.vp.prefix([]), "")
        self.assertEqual(self.vp.prefix(["OTHER", "SUBCOSTAL"]), "Subcostal view.")
        self.assertEqual(prefixed("Subcostal view.", "IVC normal."), "Subcostal view. IVC normal.")
        self.assertEqual(prefixed("", "IVC normal."), "IVC normal.")

    def test_a_phrase_file_that_does_not_match_the_registry_is_refused(self):
        with open(DEFAULT_VIEW_PHRASES, encoding="utf-8") as f:
            good = yaml.safe_load(f)

        def changed(name, change):
            raw = copy.deepcopy(good)
            change(raw)
            return write_yaml(os.path.join(self.root, name), raw)

        bad = [changed("missing.yaml", lambda r: r["families"].pop("A3C")),
               changed("extra.yaml", lambda r: r["families"].update(A4C_LV="apical four-chamber LV")),
               changed("twice.yaml", lambda r: r["families"].update(A3C="apical two-chamber")),
               changed("both.yaml", lambda r: r["unphrased"].append("A2C")),
               changed("schema.yaml", lambda r: r.update(schema="other")),
               changed("blank.yaml", lambda r: r["families"].update(A3C=" "))]
        for path in bad:
            with self.assertRaises(ValueError, msg=path):
                ViewPhrases(path, ROUTING)
        self.assertEqual(ViewPhrases(changed("same.yaml", lambda r: None), ROUTING).sha256, self.vp.sha256)
        reordered = ViewPhrases(changed("order.yaml", lambda r: r.update(
            families=dict(reversed(list(r["families"].items()))))), ROUTING)
        self.assertNotEqual(reordered.sha256, self.vp.sha256)
        self.assertEqual(reordered.prefix(["A4C", "PLAX"]), "Apical four-chamber and parasternal long-axis views.")

    def test_a_prefixed_caption_id_needs_the_same_views_and_the_same_caption(self):
        a, b = "Apical four-chamber view.", "Parasternal long-axis and apical four-chamber views."
        self.assertEqual(prefixed_caption_id(123, ""), 123)
        self.assertEqual(prefixed_caption_id(123, a), prefixed_caption_id(123, a))
        self.assertNotEqual(prefixed_caption_id(123, a), prefixed_caption_id(123, b))
        self.assertNotEqual(prefixed_caption_id(123, a), prefixed_caption_id(124, a))
        self.assertNotEqual(prefixed_caption_id(123, a), 123)
        for value in (prefixed_caption_id(-(2 ** 63), a), prefixed_caption_id(2 ** 63 - 1, b)):
            self.assertTrue(-(2 ** 63) <= value < 2 ** 63)
            torch.tensor([value], dtype=torch.long)

    def test_the_config_switch(self):
        self.assertIsNone(view_phrases({}, ROUTING))
        self.assertIsNone(view_phrases({"view_prefix": False}, ROUTING))
        self.assertIsInstance(view_phrases({"view_prefix": True}, ROUTING), ViewPhrases)
        self.assertEqual(view_phrases({"view_prefix": True, "view_phrases": DEFAULT_VIEW_PHRASES}, ROUTING).sha256,
                         self.vp.sha256)
        for bad in ({"view_prefix": "yes"}, {"view_prefix": 1}, {"view_phrases": DEFAULT_VIEW_PHRASES},
                    {"view_prefix": True, "stills": {"table": "x"}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                view_phrases(bad, ROUTING)

    def test_longest_prefixes_name_every_view_of_a_structure(self):
        longest = longest_prefixes(self.vp, ROUTING, ["lv", "pulmonic", "ivc"])
        self.assertEqual(longest["pulmonic"], "Parasternal short-axis aortic valve level view.")
        self.assertEqual(longest["ivc"], "Subcostal view.")
        self.assertEqual(longest["lv"].count(","), 4)                  # six families: "a, b, c, d, e and f views."

    def test_possible_prefixes_hold_every_bag_that_can_be_drawn(self):
        rng = random.Random(0)
        views = list(ROUTING.view_classes)
        table = ROUTING.eligibility([s.name for s in ROUTING.structures])
        for case in range(150):
            clip_views = [rng.choice(views) for _ in range(rng.randint(1, 14))]
            num_clips, selection = rng.choice([1, 2, 4, 12]), rng.choice(["view_balanced", "random"])
            clips = [(f"c{i:02d}", v) for i, v in enumerate(clip_views)]
            for j, structure in enumerate(ROUTING.structures):
                possible = set(self.vp.possible_prefixes(clip_views, structure.views, num_clips, selection))
                for seed in range(15):
                    chosen = choose_bag(clips, num_clips, selection, random.Random(f"{case}:{seed}"),
                                        ROUTING.view_families)
                    own = dict(clips)
                    fed = [own[c] for c in chosen if table[ROUTING.view_classes.index(own[c]), j]]
                    prefix = self.vp.prefix(fed)
                    self.assertTrue(prefix == "" or prefix in possible, (clip_views, structure.name, prefix))

    def test_with_every_family_in_every_bag_only_a_mixed_family_varies(self):
        la = ROUTING.structure("la").views                      # reads A2C, not A2C_LV
        self.assertNotIn("A2C_LV", la)
        self.assertEqual(self.vp.possible_prefixes(["A4C", "A2C", "PLAX"], la, 12, "view_balanced"),
                         ["Parasternal long-axis, apical four-chamber and apical two-chamber views."])
        mixed = ["A4C", "A2C", "A2C_LV", "A2C_LV"]
        expected = ["Apical four-chamber and apical two-chamber views.", "Apical four-chamber view."]
        self.assertEqual(self.vp.possible_prefixes(mixed, la, 2, "view_balanced"), expected)
        clips = [(f"c{i}", v) for i, v in enumerate(mixed)]
        reached = set()
        for seed in range(60):
            chosen = choose_bag(clips, 2, "view_balanced", random.Random(seed), ROUTING.view_families)
            reached.add(self.vp.prefix([dict(clips)[c] for c in chosen if dict(clips)[c] in la]))
        self.assertEqual(sorted(reached), expected)
        self.assertEqual(self.vp.possible_prefixes(["A2C_LV"], la, 2, "view_balanced"), [])


class TestPrefixedDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(cls.root)
        cls.cache = TokenCache(cls.paths["cache_dir"], token_grid=GRID, embed_dim=DIM)
        cls.clips = read_table(cls.paths["clips"])
        cls.captions = read_table(cls.paths["captions"], id_columns=("study_id",))
        cls.vp = ViewPhrases(DEFAULT_VIEW_PHRASES, ROUTING)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def dataset(self, **kw):
        return RadarStudyDataset(self.clips, self.captions, ROUTING, self.cache, num_clips=4, **kw)

    def test_the_prefix_names_the_views_feeding_the_structure_and_nothing_else_changes(self):
        plain, prefixed_ds = self.dataset(), self.dataset(view_prefix=self.vp)
        views = dict(zip(self.clips.clip_id, self.clips.view))
        table = dict(((r.study_id, r.structure), (r.caption, int(r.caption_id))) for r in self.captions.itertuples())
        checked = 0
        for epoch in range(3):
            plain.set_epoch(epoch)
            prefixed_ds.set_epoch(epoch)
            for i in range(len(plain)):
                a, b = plain[i], prefixed_ds[i]
                for key in ("tokens", "clip_valid", "view_ids", "eligible", "has_caption", "normal", "abnormal"):
                    self.assertTrue(torch.equal(a[key], b[key]), key)
                self.assertEqual(a["clip_ids"], b["clip_ids"])
                prefixes = prefixed_ds.view_prefixes(i)
                for s, name in enumerate(prefixed_ds.structures):
                    fed = [views[c] for k, c in enumerate(b["clip_ids"]) if bool(b["eligible"][s, k])]
                    self.assertEqual(prefixes[s], self.vp.prefix(fed))
                    if not b["has_caption"][s]:
                        self.assertEqual(b["captions"][s], a["captions"][s])
                        continue
                    caption, caption_id = table[(b["study_id"], name)]
                    self.assertEqual(a["captions"][s], caption)                       # off: the table's text
                    self.assertEqual(int(a["caption_ids"][s]), caption_id)
                    self.assertEqual(bool(prefixes[s]), bool(b["eligible"][s].any()))
                    if not prefixes[s]:                       # no clip feeds it: never a trained row, text as is
                        self.assertEqual(b["captions"][s], caption)
                        continue
                    self.assertEqual(b["captions"][s], f"{prefixes[s]} {caption}")
                    self.assertEqual(int(b["caption_ids"][s]), prefixed_caption_id(caption_id, prefixes[s]))
                    checked += 1
        self.assertGreater(checked, 20)

    def test_prefixed_texts_cover_every_bag(self):
        data = self.dataset(view_prefix=self.vp)
        texts = data.prefixed_texts()
        for epoch in range(25):
            data.set_epoch(epoch)
            for i in range(len(data)):
                item = data[i]
                for s in range(len(data.structures)):
                    if item["has_caption"][s]:
                        self.assertIn(item["captions"][s], texts)
        captions = set(self.captions.caption)
        self.assertTrue(captions <= texts)
        self.assertTrue(all(t[0].isupper() and " view" in t.split(". ")[0] for t in texts - captions))
        with self.assertRaises(ValueError):
            self.dataset().prefixed_texts()
        with self.assertRaises(ValueError):
            self.dataset().view_prefixes(0)

    def test_stills_are_refused(self):
        with self.assertRaises(ValueError):
            self.dataset(view_prefix=self.vp, stills=mock.Mock(studies=set()))


class TestExactCaptionsWithPrefix(unittest.TestCase):
    """Three studies with one pericardium caption: two seen in A4C only, one in PLAX only."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        writer = TokenCacheWriter(os.path.join(self.root, "cache"), GRID, DIM)
        clips = []
        for study, view in (("S0", "A4C"), ("S1", "A4C"), ("S2", "PLAX")):
            writer.add(f"{study}-c0", torch.randn(GRID[0] * GRID[1] * GRID[2], DIM))
            clips.append({"study_id": study, "clip_id": f"{study}-c0", "view": view, "acquisition": "bmode"})
        writer.close()
        self.clips = pd.DataFrame(clips)
        self.captions = pd.DataFrame([{"study_id": s, "structure": "pericardium", "caption": "No effusion.",
                                       "caption_id": 77, "normality": "normal"} for s in ("S0", "S1", "S2")])
        self.cache = TokenCache(os.path.join(self.root, "cache"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def ids(self, view_prefix):
        data = RadarStudyDataset(self.clips, self.captions, ROUTING, self.cache, num_clips=4,
                                 view_prefix=view_prefix)
        s = data.structures.index("pericardium")
        return torch.stack([data[i]["caption_ids"][s] for i in range(3)])

    def test_exact_positives_need_the_same_views(self):
        plain = caption_targets(self.ids(None), use_exact=True)
        self.assertTrue(torch.allclose(plain, torch.full((3, 3), 1 / 3)))     # one caption: uniform targets
        with_prefix = caption_targets(self.ids(ViewPhrases(DEFAULT_VIEW_PHRASES, ROUTING)), use_exact=True)
        expected = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(torch.allclose(with_prefix, expected))


class TestPrefixedTrainingAndZeroShot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        base = synthetic_config(cls.paths, text=dict(FROZEN), epochs=2)
        base["text"].pop("trainable_layers")
        base["loss"] = dict(A1)
        base["data"]["text_cache_dir"] = os.path.join(cls.root, "text_cache")
        cls.off = base
        cls.on = copy.deepcopy(base)
        cls.on["data"]["view_prefix"] = True
        cls.trained = {name: radar_train.train(config, os.path.join(cls.root, name))
                       for name, config in (("off", cls.off), ("on", cls.on))}
        studies = sorted(set(read_table(cls.paths["clips"]).study_id))
        cls.labels = os.path.join(cls.root, "labels.csv")
        pd.DataFrame({"study_id": studies, "lvef": [35 + 2 * i for i in range(len(studies))],
                      "effusion": [i % 2 for i in range(len(studies))],
                      "av_grade": [(i // 2) % 2 for i in range(len(studies))]}).to_csv(cls.labels, index=False)
        cls.prompts = write_yaml(os.path.join(cls.root, "prompts.yaml"), {"tasks": PROMPTS})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def evaluate(self, name, out, **kw):
        config = self.on if name == "on" else self.off
        return zs.evaluate(config=config, checkpoint=os.path.join(self.root, name, "latest.pt"),
                           clips=self.paths["clips"], labels=self.labels, out_dir=os.path.join(self.root, out),
                           prompts=self.prompts, split="train", samples=20, allow_training_studies=True, **kw)

    def test_a_run_without_the_option_keeps_its_fingerprint(self):
        cache = radar_train.TokenCache(self.paths["cache_dir"])
        a = radar_train.fingerprint(self.off, ROUTING, cache, 16)
        explicit = copy.deepcopy(self.off)
        explicit["data"]["view_prefix"] = False
        self.assertEqual(radar_train.fingerprint(explicit, ROUTING, cache, 16)["sha256"], a["sha256"])
        self.assertNotIn("view_prefix", a)
        self.assertNotIn("view_prefix", a["config"]["data"])
        vp = ViewPhrases(DEFAULT_VIEW_PHRASES, ROUTING)
        on = radar_train.fingerprint(self.on, ROUTING, cache, 16, view_prefix=vp.record())
        self.assertNotEqual(on["sha256"], a["sha256"])
        changed = {**vp.record(), "phrases_sha256": "x"}
        other = radar_train.fingerprint(self.on, ROUTING, cache, 16, view_prefix=changed)
        self.assertNotEqual(other["sha256"], on["sha256"])
        saved = torch.load(os.path.join(self.root, "on", "latest.pt"), weights_only=True)["fingerprint"]
        self.assertEqual(saved["view_prefix"], vp.record())
        self.assertNotIn("view_prefix", torch.load(os.path.join(self.root, "off", "latest.pt"),
                                                   weights_only=True)["fingerprint"])

    def test_the_frozen_store_holds_every_prefixed_caption_and_lengths_include_the_prefix(self):
        model = self.trained["on"]["model"]
        self.assertTrue(model.text.store.sealed)
        self.assertGreater(self.trained["on"]["optimizer_steps"], 0)
        vp = ViewPhrases(DEFAULT_VIEW_PHRASES, ROUTING)
        captions = read_table(self.paths["captions"], id_columns=("study_id",))
        clips = read_table(self.paths["clips"])
        data = RadarStudyDataset(clips, captions, ROUTING, TokenCache(self.paths["cache_dir"]), num_clips=4,
                                 view_prefix=vp)
        self.assertTrue(data.prefixed_texts() <= set(model.text.store.index))
        lengths = {}
        for name in ("off", "on"):
            with open(os.path.join(self.root, name, "caption_lengths.json")) as f:
                lengths[name] = json.load(f)["structures"]
        for structure, entry in lengths["on"].items():
            self.assertGreater(entry["longest"], lengths["off"][structure]["longest"])

    def test_a_trained_caption_encoder_reads_prefixed_captions_too(self):
        config = synthetic_config(self.paths, epochs=1)
        config["loss"] = dict(A1)
        config["data"]["view_prefix"] = True
        summary = radar_train.train(config, os.path.join(self.root, "trained_text"))
        self.assertGreater(summary["optimizer_steps"], 0)

    def test_prefixed_prompts_score_each_bag_with_its_own_prefix(self):
        report = self.evaluate("on", "zs_prefixed", prompt_view_prefix=True)
        self.assertIs(report["prompt_view_prefix"], True)
        vp = ViewPhrases(DEFAULT_VIEW_PHRASES, ROUTING)
        self.assertEqual(report["view_prefix"], vp.record())
        predictions = pd.read_csv(os.path.join(self.root, "zs_prefixed", "predictions.csv"), dtype={"study_id": str})

        # by hand: embed every bag, score each study's bag against the prompts after that bag's prefix
        model, tokenizer, _ = zs.load_model(self.on, os.path.join(self.root, "on", "latest.pt"), ROUTING,
                                            torch.device("cpu"))
        clips = read_table(self.paths["clips"])
        data = RadarStudyDataset(clips, pd.DataFrame(columns=list(zs.CAPTION_COLUMNS)), ROUTING,
                                 TokenCache(self.paths["cache_dir"]), num_clips=4, training=False, view_prefix=vp)
        z, has = zs.embed_studies(model, data, torch.device("cpu"), num_bags=3)
        temperature = float(model.loss.effective_temperature().detach())
        tasks = load_prompt_tasks(self.prompts)
        structures = [s.name for s in ROUTING.trained]
        varied = 0
        for name, task in tasks.items():
            s = structures.index(task.structure)
            rows = predictions[predictions.task == name].set_index("study_id")
            for i, (study, _) in enumerate(data.studies):
                values = []
                for b in range(3):
                    data.eval_bag_index = b
                    prefix = data.view_prefixes(i)[s]
                    self.assertEqual(prefix != "", bool(has[b, i, s]))
                    if prefix:
                        encoded = zs.encode_task(task, model.text, tokenizer, prefix)
                        values.append(zs.score_task(task, z[b, i:i + 1, s], encoded, temperature).double()[0])
                if not values:
                    self.assertFalse(rows.loc[study].prediction_supported)
                    continue
                pooled = torch.stack(values).mean(dim=0)
                if task.mode == "categorical":
                    grades = sorted(set(task.values))
                    self.assertEqual(rows.loc[study].prediction, grades[int(pooled.argmax())])
                else:
                    self.assertAlmostEqual(rows.loc[study].prediction, float(pooled), places=5)
                varied += 1
        self.assertGreater(varied, 20)
        data.eval_bag_index = 0

        plain = self.evaluate("on", "zs_plain")
        self.assertIs(plain["prompt_view_prefix"], False)
        self.assertEqual(plain["view_prefix"], vp.record())
        other = pd.read_csv(os.path.join(self.root, "zs_plain", "predictions.csv"), dtype={"study_id": str})
        ef = predictions.task == "ef"
        self.assertFalse(np.allclose(predictions[ef].prediction.dropna(), other[ef].prediction.dropna()))

    def test_without_a_prefix_checkpoint_the_report_is_unchanged_and_the_option_refused(self):
        report = self.evaluate("off", "zs_off")
        self.assertNotIn("prompt_view_prefix", report)
        self.assertNotIn("view_prefix", report)
        with open(os.path.join(self.root, "zs_off", "thresholds.json")) as f:
            self.assertNotIn("prompt_view_prefix", json.load(f)["source"])
        with self.assertRaisesRegex(ValueError, "trained with data.view_prefix"):
            self.evaluate("off", "zs_off_refused", prompt_view_prefix=True)
        with self.assertRaisesRegex(ValueError, "dump_study"):
            self.evaluate("on", "zs_dump", prompt_view_prefix=True, dump_studies=["S00"])

    def test_thresholds_of_the_other_prompt_setting_are_refused(self):
        source = {"checkpoint_sha256": "c", "clips_sha256": "val", "labels_sha256": "l", "prompts_sha256": "p",
                  "split": "val", "pooling": "mean_similarity"}
        tasks = load_prompt_tasks(self.prompts)
        path = os.path.join(self.root, "thresholds.json")
        for chosen, used in ((True, False), (False, True)):
            saved = {**source, "clips_sha256": "tune", "split": "tune"}
            if chosen:
                saved["prompt_view_prefix"] = True
            with open(path, "w") as f:
                json.dump({"source": saved, "thresholds": {"effusion": 0.4}}, f)
            current = {**source, "prompt_view_prefix": True} if used else dict(source)
            with self.assertRaisesRegex(ValueError, "prompt_view_prefix"):
                zs.read_thresholds(path, current, tasks)
        with open(path, "w") as f:
            json.dump({"source": {**source, "clips_sha256": "tune", "split": "tune", "prompt_view_prefix": True},
                       "thresholds": {"effusion": 0.4}}, f)
        self.assertEqual(zs.read_thresholds(path, {**source, "prompt_view_prefix": True}, tasks), {"effusion": 0.4})


if __name__ == "__main__":
    unittest.main()
