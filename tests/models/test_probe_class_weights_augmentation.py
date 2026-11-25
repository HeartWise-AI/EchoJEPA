# tests/models/test_probe_class_weights_augmentation.py

"""The study probe's per-head class weights (inverse prevalence, capped) and its configurable
training augmentation (legacy defaults when the config leaves it out)."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from evals.study_multitask_frozen.eval import LEGACY_AUGMENTATION, augmentation_options, resolve_class_weights
from src.models.study_probe import head_loss, head_specs

CLASSIFIER = {
    "target_label": ["lvef", "tr_modsev", "tr_grade"],
    "head_task": {"lvef": "regression", "tr_modsev": "binary", "tr_grade": "classification"},
    "head_structure": {"tr_grade": 3},
}
# 20 train studies: tr_modsev 2 positives / 18 negatives (one missing); tr_grade 15 / 4 / 1.
TARGETS = np.array([[50.0, 1.0 if i < 2 else 0.0, 0 if i < 15 else (1 if i < 19 else 2)] for i in range(20)])
TARGETS[5, 1] = np.nan


class TestClassWeights(unittest.TestCase):

    def specs(self, **classifier):
        specs = head_specs({**CLASSIFIER, **classifier})
        resolve_class_weights(specs, TARGETS, {**CLASSIFIER, **classifier})
        return specs

    def test_unset_leaves_the_specs_unchanged(self):
        self.assertNotIn("class_weight", self.specs()["tr_modsev"])

    def test_auto_is_inverse_prevalence_capped(self):
        specs = self.specs(class_weights="auto", class_weight_cap=5)
        self.assertEqual(specs["tr_modsev"]["class_weight"], [5.0])            # 17 / 2 = 8.5, capped at 5
        self.assertEqual(specs["tr_grade"]["class_weight"], [1.0, 3.75, 5.0])  # 15/15, 15/4, 15/1 capped
        self.assertNotIn("class_weight", specs["lvef"])

    def test_explicit_values_and_refusals(self):
        specs = self.specs(class_weights={"tr_modsev": 3.0, "tr_grade": [1, 2, 4]})
        self.assertEqual(specs["tr_modsev"]["class_weight"], [3.0])
        self.assertEqual(specs["tr_grade"]["class_weight"], [1.0, 2.0, 4.0])
        for weights, message in (({"lvef": 2.0}, "regression"), ({"tr_grade": [1, 2]}, "3 positive"),
                                 ({"unknown": 1.0}, "does not list"), ({"tr_modsev": -1.0}, "positive")):
            with self.assertRaisesRegex(ValueError, message):
                self.specs(class_weights=weights)

    def test_the_loss_uses_the_weights(self):
        specs = self.specs(class_weights={"tr_modsev": 4.0, "tr_grade": [1, 2, 4]})
        logits, target = torch.tensor([[0.3], [-0.2]]), torch.tensor([1.0, 0.0])
        value, count = head_loss(specs["tr_modsev"], logits, target)
        expected = F.binary_cross_entropy_with_logits(logits.squeeze(-1), target, reduction="sum",
                                                      pos_weight=torch.tensor([4.0]))
        self.assertEqual(count, 2)
        self.assertTrue(torch.allclose(value, expected))
        logits, target = torch.randn(3, 3), torch.tensor([0.0, 1.0, 2.0])
        value, _ = head_loss(specs["tr_grade"], logits, target)
        expected = F.cross_entropy(logits, target.long(), weight=torch.tensor([1.0, 2.0, 4.0]), reduction="sum")
        self.assertTrue(torch.allclose(value, expected))


class TestAugmentation(unittest.TestCase):

    def test_unset_keeps_the_legacy_augmentation(self):
        self.assertEqual(augmentation_options(None), LEGACY_AUGMENTATION)

    def test_partial_settings_fill_from_the_legacy_values(self):
        out = augmentation_options({"random_resize_scale": [0.5, 1.0], "auto_augment": False})
        self.assertEqual(out["random_resize_scale"], (0.5, 1.0))
        self.assertFalse(out["auto_augment"])
        self.assertEqual(out["reprob"], LEGACY_AUGMENTATION["reprob"])

    def test_invalid_settings_are_refused(self):
        for augmentation in ({"random_resize_scale": [0.5, 1.2]}, {"random_resize_scale": [0.9, 0.5]},
                             {"reprob": 2}, {"auto_augment": "yes"}, {"hflip": True}, [0.5, 1.0]):
            with self.assertRaises(ValueError):
                augmentation_options(augmentation)


if __name__ == "__main__":
    unittest.main()
