# tests/models/test_radar_frozen_text.py

"""The frozen caption encoder with cached caption features (`text.frozen: true`), on a tiny random backbone:

- the frozen backbone gets no gradient, stays in eval mode and is outside the module (no parameter in the optimizer,
  the checkpoint or DDP); only the head trains;
- cached features equal those computed directly, also after a reload from the cache file, which holds no text;
- the cache key changes with the backbone, pooling, layer, template, max_length or dtype; a file whose key does not
  match is rebuilt, not reused; a head trained on other features is refused at load;
- zero-shot prompts go through the same store (encoded once, then reused), and a sealed store refuses to encode;
- pooling: `mean` leaves special tokens out, `last` reads the last caption token whatever the padding;
- training: caption lengths are measured with the frozen tokenizer, the store is filled before the first step and
  sealed, the fingerprint records the feature key, a resume reproduces straight training, loader workers (spawn)
  read the row index only, and two gloo processes reproduce one.
"""

import copy
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from evals.radar_frozen import train as radar_train
from src.models.radar import frozen_text as ft
from src.models.radar.text_encoder import MomentumTeacher, build_text_encoder, build_tokenizer
from src.models.radar.zero_shot import encode_prompts
from tests.models.radar_synthetic import make_synthetic, synthetic_config

FROZEN = {"backbone": "tiny", "vocab_size": 200, "hidden_size": 16, "num_layers": 2, "num_heads": 2,
          "max_length": 32, "frozen": True, "pooling": "mean", "head": "mlp", "head_hidden_dim": 12}
TEXTS = ["Normal left ventricular size.", "Severe mitral regurgitation.", "No pericardial effusion.", ""]


def frozen(**changes):
    cfg = dict(FROZEN)
    cfg.update(changes)
    return cfg


def _state(summary):
    return {k: v.detach().clone() for k, v in summary["model"].state_dict().items()}


def _same(a, b, atol):
    return a.keys() == b.keys() and all(torch.allclose(a[k].float(), b[k].float(), rtol=0.0, atol=atol) for k in a)


def _worker(rank, world_size, init_file, config, out_dir, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        summary = radar_train.train(config, out_dir)
        torch.save({"losses": summary["losses"], "state": _state(summary)}, os.path.join(result_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


class TestFrozenEncoder(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_backbone_gets_no_gradient_and_stays_in_eval_mode(self):
        encoder, tokenizer = build_text_encoder(frozen(), proj_dim=8)
        encoder.train()
        ids, mask = tokenizer(TEXTS)
        encoder(ids, mask).sum().backward()
        backbone = encoder.store.backbone._model
        self.assertIsNotNone(backbone)
        self.assertFalse(backbone.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in backbone.parameters()))
        self.assertTrue(all(p.grad is not None for p in encoder.parameters()))
        self.assertTrue(all(name.startswith("head.") for name, _ in encoder.named_parameters()))
        self.assertEqual(set(encoder.state_dict()) - {"feature_key"},
                         {name for name, _ in encoder.named_parameters()})
        teacher = MomentumTeacher(encoder)
        self.assertIs(teacher.model.store, encoder.store)          # shared, not copied
        self.assertTrue(torch.allclose(teacher(ids, mask), encoder(ids, mask)))

    def test_heads(self):
        linear, _ = build_text_encoder(frozen(head="linear", input_norm=False), proj_dim=8)
        self.assertEqual([type(m) for m in linear.head], [torch.nn.Linear])
        mlp, _ = build_text_encoder(frozen(), proj_dim=8)
        self.assertEqual([type(m) for m in mlp.head], [torch.nn.LayerNorm, torch.nn.Linear, torch.nn.GELU,
                                                       torch.nn.Linear])
        self.assertEqual(mlp.head[1].out_features, 12)

    def test_cached_features_equal_direct_ones_and_survive_a_reload(self):
        cache = os.path.join(self.root, "cache")
        encoder, tokenizer = build_text_encoder(frozen(), proj_dim=8, cache_dir=cache)
        out = encoder(*tokenizer(TEXTS))
        backbone = encoder.store.backbone
        ids, mask, special = backbone._tokenize(TEXTS)
        direct = ft.pool(backbone.load("cpu")(ids, mask), mask, special, "mean")
        self.assertTrue(torch.allclose(encoder.store.features[tokenizer(TEXTS)[0][:, 0]], direct, atol=1e-6))
        self.assertTrue(torch.allclose(out, torch.nn.functional.normalize(encoder.head(direct), dim=-1), atol=1e-6))

        again, tokenizer2 = build_text_encoder(frozen(), proj_dim=8, cache_dir=cache)
        again.load_state_dict(encoder.state_dict())
        with mock.patch.object(ft.FrozenBackbone, "encode", side_effect=AssertionError("encoded again")):
            reloaded = again(*tokenizer2(list(reversed(TEXTS))))
        self.assertTrue(torch.allclose(reloaded, out.flip(0), atol=1e-6))
        files = [f for f in os.listdir(cache) if f.startswith("text_features_")]
        self.assertEqual(len(files), 1)
        with open(os.path.join(cache, files[0]), "rb") as f:
            raw = f.read()
        self.assertNotIn(b"mitral", raw)
        saved = torch.load(os.path.join(cache, files[0]), weights_only=True)
        self.assertEqual(sorted(saved["hashes"]), sorted(ft.text_sha256(t) for t in TEXTS))

    def test_the_key_changes_with_what_fixes_the_features(self):
        base = ft.frozen_feature_key(frozen())
        self.assertEqual(base, ft.frozen_feature_key(frozen(head="linear", head_hidden_dim=99)))   # head: not a key
        for change in [{"pooling": "last"}, {"pooling": "cls"}, {"layer": 1}, {"template": "Report: {}"},
                       {"max_length": 16}, {"dtype": "bfloat16"}, {"init_seed": 1}, {"hidden_size": 8},
                       {"num_layers": 3}]:
            with self.subTest(**change):
                self.assertNotEqual(ft.frozen_feature_key(frozen(**change)), base)
        self.assertIsNone(ft.frozen_feature_key({"backbone": "tiny"}))

    def test_writing_keeps_the_rows_of_other_runs(self):
        cache = os.path.join(self.root, "cache")
        _, captions = build_text_encoder(frozen(), proj_dim=8, cache_dir=cache)
        captions(TEXTS[:2])
        _, prompts = build_text_encoder(frozen(), proj_dim=8, cache_dir=cache)
        prompts(["Ejection fraction 40 %."])                      # another run adds a prompt
        saved = torch.load(prompts.store.path, weights_only=True)
        expected = sorted(ft.text_sha256(t) for t in TEXTS[:2] + ["Ejection fraction 40 %."])
        self.assertEqual(sorted(saved["hashes"]), expected)

    def test_a_cache_file_with_another_key_is_rebuilt(self):
        cache = os.path.join(self.root, "cache")
        encoder, tokenizer = build_text_encoder(frozen(), proj_dim=8, cache_dir=cache)
        tokenizer(TEXTS)
        path = encoder.store.path
        saved = torch.load(path, weights_only=True)
        saved["key"] = {**saved["key"], "pooling": "last"}           # a file written for other settings
        saved["features"] = torch.zeros_like(saved["features"])
        torch.save(saved, path)
        again, tokenizer2 = build_text_encoder(frozen(), proj_dim=8, cache_dir=cache)
        with self.assertWarnsRegex(UserWarning, "rebuilt"):
            rows = tokenizer2(TEXTS)[0][:, 0]
        self.assertTrue(torch.allclose(again.store.features[rows], encoder.store.features[tokenizer(TEXTS)[0][:, 0]]))
        self.assertEqual(torch.load(path, weights_only=True)["key"], again.store.key)       # rewritten

    def test_a_head_trained_on_other_features_is_refused(self):
        encoder, _ = build_text_encoder(frozen(), proj_dim=8)
        other, _ = build_text_encoder(frozen(pooling="last"), proj_dim=8)
        with self.assertRaisesRegex(ValueError, "other caption features"):
            other.load_state_dict(encoder.state_dict())

    def test_zero_shot_prompts_use_the_store(self):
        encoder, tokenizer = build_text_encoder(frozen(), proj_dim=8)
        prompts = [f"The left ventricular ejection fraction is {n}%." for n in range(5, 15)]
        with mock.patch.object(ft.FrozenBackbone, "encode", wraps=encoder.store.backbone.encode) as spy:
            first = encode_prompts(encoder, tokenizer, prompts, batch_size=4)
            second = encode_prompts(encoder, tokenizer, prompts, batch_size=3)
        self.assertEqual(sum(len(call.args[0]) for call in spy.call_args_list), len(prompts))     # each once
        self.assertTrue(torch.allclose(first, second, atol=1e-6))
        rows = tokenizer(prompts)[0][:, 0]
        expected = torch.nn.functional.normalize(encoder.head(encoder.store.features[rows]), dim=-1)
        self.assertTrue(torch.allclose(first, expected, atol=1e-6))
        tokenizer.seal()
        self.assertTrue(torch.allclose(encode_prompts(encoder, tokenizer, prompts), first, atol=1e-6))
        with self.assertRaisesRegex(ValueError, "sealed"):
            encode_prompts(encoder, tokenizer, ["An unseen prompt."])

    def test_pooling(self):
        hidden = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        special = torch.tensor([[True, False, False, False], [True, False, False, True]])
        result = SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden * 0, hidden))
        self.assertEqual(ft.pool(result, mask, special, "mean").flatten().tolist(), [1.5, 5.5])
        self.assertEqual(ft.pool(result, mask, special, "last").flatten().tolist(), [2.0, 6.0])
        self.assertEqual(ft.pool(result, mask, special, "cls").flatten().tolist(), [0.0, 4.0])
        self.assertEqual(ft.pool(result, mask, special, "mean", layer=0).flatten().tolist(), [0.0, 0.0])
        empty = ft.pool(result, torch.tensor([[1, 0, 0, 0]]), torch.tensor([[True, False, False, False]]), "mean",
                        layer=-1)
        self.assertTrue(torch.isfinite(empty).all())

    def test_features_do_not_depend_on_the_batch(self):
        store = ft.CaptionFeatureStore(ft.FrozenBackbone(frozen()))
        together = store.backbone.encode(TEXTS)
        alone = torch.cat([store.backbone.encode([t]) for t in TEXTS])
        self.assertTrue(torch.allclose(together, alone, atol=1e-5))

    def test_invalid_configs_are_refused(self):
        for change, message in [({"trainable_layers": 1}, "trainable_layers"), ({"pooling": "max"}, "pooling"),
                                ({"head": "deep"}, "head"), ({"template": "no slot"}, "slot"),
                                ({"pooling": "pooler", "layer": 3}, "layer"), ({"dtype": "fp16"}, "dtype")]:
            with self.subTest(**change), self.assertRaisesRegex(ValueError, message):
                build_text_encoder(frozen(**change), proj_dim=8)

    def test_lengths_come_from_the_frozen_tokenizer_and_its_template(self):
        plain = build_tokenizer(frozen())
        wrapped = build_tokenizer(frozen(template="This finding : {} means"))
        self.assertEqual(plain.max_length, 32)
        self.assertEqual(plain.lengths(["a b c"]), [4])
        self.assertEqual(wrapped.lengths(["a b c"]), [4 + 4])


class TestFrozenTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(os.path.join(cls.root, "data"))
        cls.config = synthetic_config(cls.paths, text={**FROZEN, "trainable_layers": None})
        cls.config["data"]["text_cache_dir"] = os.path.join(cls.root, "text_cache")
        cls.config["text"].pop("trainable_layers")
        cls.straight = radar_train.train(cls.config, os.path.join(cls.root, "straight"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_only_the_head_is_trained_and_checkpointed(self):
        model = self.straight["model"]
        text_keys = [k for k in model.state_dict() if k.startswith("text.")]
        self.assertTrue(text_keys and all(k.startswith(("text.head.", "text.feature_key")) for k in text_keys))
        self.assertTrue(model.text.store.sealed)
        self.assertIsNone(model.text.store.backbone._model)               # released after the captions
        self.assertGreater(self.straight["optimizer_steps"], 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(self.straight["losses"]))))

    def test_the_fingerprint_records_the_feature_key_and_not_the_cache_folder(self):
        state = torch.load(os.path.join(self.root, "straight", "latest.pt"), weights_only=True)
        record = state["fingerprint"]
        self.assertEqual(record["text_features"], ft.frozen_feature_key(self.config["text"]))
        self.assertNotIn("text_cache_dir", record["config"]["data"])
        moved = copy.deepcopy(self.config)
        moved["data"]["text_cache_dir"] = os.path.join(self.root, "elsewhere")
        routing = radar_train.load_routing()
        cache = radar_train.TokenCache(self.paths["cache_dir"])
        a = radar_train.fingerprint(self.config, routing, cache, 16,
                                    text_features=ft.frozen_feature_key(self.config["text"]))
        b = radar_train.fingerprint(moved, routing, cache, 16,
                                    text_features=ft.frozen_feature_key(moved["text"]))
        c = radar_train.fingerprint(self.config, routing, cache, 16,
                                    text_features=ft.frozen_feature_key({**self.config["text"], "pooling": "last"}))
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertNotEqual(a["sha256"], c["sha256"])
        self.assertNotIn("text_features", radar_train.fingerprint(synthetic_config(self.paths), routing, cache, 16))

    def test_resume_reproduces_straight_training(self):
        out = os.path.join(self.root, "resumed")
        first = radar_train.train(self.config, out, stop_after_epoch=1)
        with mock.patch.object(ft.FrozenBackbone, "encode", side_effect=AssertionError("encoded again")):
            second = radar_train.train(self.config, out)                # captions come from the cache file
        self.assertEqual(first["losses"] + second["losses"], self.straight["losses"])
        self.assertTrue(_same(_state(second), _state(self.straight), 1e-6))

    def test_zero_shot_evaluation_encodes_prompts_through_the_cache(self):
        from evals.radar_frozen import zero_shot_eval as zs
        from src.models.radar.zero_shot import PromptTask
        task = PromptTask("ef", "regression", "lv", ("Ejection fraction <#> %.",), value_range=(5, 20))
        checkpoint = os.path.join(self.root, "straight", "latest.pt")
        routing = radar_train.load_routing()
        model, tokenizer, _ = zs.load_model(self.config, checkpoint, routing, torch.device("cpu"))
        first = zs.encode_task(task, model.text, tokenizer)
        model, tokenizer, _ = zs.load_model(self.config, checkpoint, routing, torch.device("cpu"))
        with mock.patch.object(ft.FrozenBackbone, "encode", side_effect=AssertionError("encoded again")):
            second = zs.encode_task(task, model.text, tokenizer)          # prompts now come from the cache file
        self.assertTrue(torch.allclose(first["prompts"], second["prompts"], atol=1e-6))
        self.assertEqual(first["prompts"].shape[0], 16)
        other = copy.deepcopy(self.config)
        other["text"]["pooling"] = "last"
        with self.assertRaises(ValueError):            # the checkpoint's config differs (and so would its features)
            zs.load_model(other, checkpoint, routing, torch.device("cpu"))

    def test_captions_over_the_frozen_limit_are_measured_and_left_out(self):
        captions = pd.read_csv(self.paths["captions"])
        kept, report = radar_train.fit_captions(captions, build_tokenizer(frozen(max_length=6)), max_fraction=1.0)
        self.assertEqual(report["max_length"], 6)
        self.assertLess(len(kept), len(captions))

    def test_loader_workers_read_the_row_index_only(self):
        config = copy.deepcopy(self.config)
        config["optimization"].update(num_workers=1, epochs=1)
        reference = copy.deepcopy(self.config)
        reference["optimization"]["epochs"] = 1
        with_workers = radar_train.train(config, os.path.join(self.root, "workers"))
        without = radar_train.train(reference, os.path.join(self.root, "no_workers"))
        self.assertEqual(with_workers["losses"], without["losses"])

    def test_two_processes_match_one(self):
        result_dir = tempfile.mkdtemp(dir=self.root)
        mp.spawn(_worker, args=(2, os.path.join(result_dir, "init"), self.config, os.path.join(self.root, "ddp"),
                                result_dir), nprocs=2, join=True)
        for rank in range(2):
            with self.subTest(rank=rank):
                result = torch.load(os.path.join(result_dir, f"r{rank}.pt"))
                self.assertTrue(torch.allclose(torch.tensor(result["losses"]), torch.tensor(self.straight["losses"]),
                                               atol=1e-5))
                self.assertTrue(_same(result["state"], _state(self.straight), 1e-5))


if __name__ == "__main__":
    unittest.main()
