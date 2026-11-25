# tests/models/test_radar_partial_freeze.py

"""A partly frozen caption encoder and a temperature floor, as configured for a BERT ablation run, on a tiny random
BERT (`hf_random`) and the synthetic training set:

- `text.trainable_layers: k` on an L-layer BERT freezes the embeddings and the first L - k encoder layers: they hold
  no gradient, are left out of every optimizer group (the coverage check passes) and keep their exact initial values
  through training, while the top k layers and the projection change;
- `loss.temperature_bounds: [floor, upper]` is a floor on the learnable temperature: below it the loss uses the floor,
  a step that takes the stored value under it ends on it, a run started just above it never logs a value under it,
  and the bounds are part of the run's fingerprint;
- a config without either key builds the encoder fully trainable with the default bounds (0.001, 0.5).
"""

import copy
import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers.utils import is_torch_available

from evals.radar_frozen import train as radar_train
from src.models.radar.loss import StructureContrastiveLoss
from src.models.radar.routing import load_routing
from src.utils.optimizer_coverage import check_optimizer_coverage
from tests.models.radar_synthetic import make_synthetic, synthetic_config

LAYERS = 4
BERT = {"backbone": "hf_random", "model_type": "bert", "max_length": 32,
        "config": {"vocab_size": 200, "hidden_size": 16, "num_hidden_layers": LAYERS, "num_attention_heads": 2,
                   "intermediate_size": 32, "max_position_embeddings": 64}}


def _frozen_part(name):
    """True for a text parameter that `trainable_layers: 2` must freeze on the 4-layer BERT."""
    if not name.startswith("text.backbone."):
        return False
    rest = name[len("text.backbone."):]
    return rest.startswith("embeddings.") or rest.startswith(("encoder.layer.0.", "encoder.layer.1."))


def _a1(config, text=None, loss=None):
    config = copy.deepcopy(config)
    config["text"] = {**BERT, **(text or {})}
    config["loss"] = {"use_exact": True, "use_normal": False, "use_abnormal_teacher": False, **(loss or {})}
    return config


@unittest.skipUnless(is_torch_available(), "transformers cannot load PyTorch models with this torch version")
class TestPartlyFrozenText(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.paths = make_synthetic(cls.root)
        cls.base = synthetic_config(cls.paths, epochs=1)
        cls.routing = load_routing()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def build(self, config):
        torch.manual_seed(0)
        return radar_train.build_model(config, self.routing)[0]

    def test_embeddings_and_bottom_layers_are_frozen_and_left_out_of_the_optimizer(self):
        config = _a1(self.base, text={"trainable_layers": 2})
        model = self.build(config)
        text = {n: p for n, p in model.named_parameters() if n.startswith("text.")}
        frozen = sorted(n for n, p in text.items() if not p.requires_grad)
        self.assertEqual(frozen, sorted(n for n in text if _frozen_part(n)))
        trained = {n for n, p in text.items() if p.requires_grad}
        for prefix in ("text.proj.", "text.backbone.encoder.layer.2.", "text.backbone.encoder.layer.3."):
            self.assertTrue(any(n.startswith(prefix) for n in trained), prefix)
        groups = radar_train.param_groups(model, config["optimization"])
        listed = {id(p) for g in groups for p in g["params"]}
        self.assertFalse(any(id(text[n]) in listed for n in frozen))
        check_optimizer_coverage(model, torch.optim.AdamW(groups, lr=1e-3))

    def test_frozen_part_keeps_its_values_through_training(self):
        config = _a1(self.base, text={"trainable_layers": 2})
        before = {n: p.detach().clone() for n, p in self.build(config).named_parameters()}
        summary = radar_train.train(config, os.path.join(self.root, "partly_frozen"), resume=False)
        self.assertGreater(summary["optimizer_steps"], 0)
        after = dict(summary["model"].named_parameters())
        unchanged = {n for n in before if torch.equal(before[n], after[n].detach().cpu())}
        self.assertEqual({n for n in before if _frozen_part(n)} - unchanged, set())
        moved = [n for n in before if n.startswith(("text.proj.", "text.backbone.encoder.layer.3."))
                 and n not in unchanged]
        self.assertTrue(moved)

    def test_the_default_trains_the_whole_encoder(self):
        model = self.build(_a1(self.base))
        self.assertTrue(all(p.requires_grad for p in model.text.parameters()))
        self.assertEqual(model.loss.temperature_bounds, (0.001, 0.5))


class TestTemperatureFloor(unittest.TestCase):

    def test_floor_is_used_in_the_loss_and_restored_after_a_step(self):
        loss_fn = StructureContrastiveLoss(["a"], init_temperature=0.005, temperature_bounds=(0.01, 0.5))
        self.assertAlmostEqual(loss_fn.effective_temperature().item(), 0.01)
        self.assertAlmostEqual(loss_fn.temperature.item(), 0.005)
        loss_fn.clamp_temperature_()
        self.assertAlmostEqual(loss_fn.temperature.item(), 0.01)
        loss_fn = StructureContrastiveLoss(["a"], init_temperature=0.07, temperature_bounds=(0.01, 0.5))
        loss_fn.clamp_temperature_()
        self.assertAlmostEqual(loss_fn.temperature.item(), 0.07)     # inside the bounds: untouched

    def test_a_step_that_would_cross_the_floor_ends_on_it(self):
        # matching image and caption features: a lower temperature lowers the loss, so a step pushes it down
        z = F.normalize(torch.randn(6, 1, 8, generator=torch.Generator().manual_seed(0)), dim=-1)
        loss_fn = StructureContrastiveLoss(["a"], use_exact=True, init_temperature=0.05,
                                           temperature_bounds=(0.01, 0.5))
        optimizer = torch.optim.SGD([loss_fn.temperature], lr=1.0)
        loss, _ = loss_fn(z, z.clone(), torch.ones(6, 1, dtype=torch.bool), torch.arange(6)[:, None])
        loss.backward()
        self.assertGreater(loss_fn.temperature.grad.item(), 0.0)
        optimizer.step()
        self.assertLess(loss_fn.temperature.item(), 0.01)
        loss_fn.clamp_temperature_()
        self.assertAlmostEqual(loss_fn.temperature.item(), 0.01, places=7)

    def test_training_never_logs_a_temperature_under_the_floor(self):
        root = tempfile.mkdtemp()
        try:
            paths = make_synthetic(root)
            config = synthetic_config(paths, epochs=1)
            config["loss"] = {"use_exact": True, "init_temperature": 0.0105, "temperature_bounds": [0.01, 0.5]}
            out = os.path.join(root, "floor")
            radar_train.train(config, out, resume=False)
            with open(os.path.join(out, "log.jsonl")) as f:
                temps = [json.loads(line)["temperature"] for line in f]
            self.assertTrue(temps)
            self.assertGreaterEqual(min(temps), 0.01 - 1e-7)
            with open(os.path.join(out, "fingerprint.json")) as f:
                floored = json.load(f)
            self.assertEqual(floored["config"]["loss"]["temperature_bounds"], [0.01, 0.5])
            config["loss"]["temperature_bounds"] = [0.001, 0.5]
            other = os.path.join(root, "default_floor")
            radar_train.train(config, other, resume=False, stop_after_epoch=0)
            with open(os.path.join(other, "fingerprint.json")) as f:
                self.assertNotEqual(json.load(f)["sha256"], floored["sha256"])
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
