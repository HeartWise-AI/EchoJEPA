"""Positive/negative/ignore masks and the multi-positive symmetric InfoNCE, on hand-checkable examples."""

import math
import unittest

import torch

from app.radar.loss import (
    IGNORE,
    NEGATIVE,
    POSITIVE,
    ContrastiveLoss,
    build_pair_mask,
    multi_positive_infonce,
    normalize_caption,
)

P, N, I = POSITIVE, NEGATIVE, IGNORE


class TestMask(unittest.TestCase):
    def test_identical_captions_are_positive_and_same_study_is_ignored(self):
        captions = [normalize_caption(c) for c in ("Coupe A4C, mode B.", "coupe  a4c, MODE b", "Coupe PLAX.", "x")]
        mask = build_pair_mask(captions, ["s1", "s2", "s1", "s3"])
        self.assertEqual(mask.tolist(), [
            [P, P, I, N],
            [P, P, N, N],
            [I, N, P, N],
            [N, N, N, P],
        ])

    def test_diagonal_is_positive_even_within_a_study(self):
        mask = build_pair_mask(["a", "b", "c"], ["s", "s", "s"])
        self.assertEqual(mask.tolist(), [[P, I, I], [I, P, I], [I, I, P]])
        self.assertFalse(bool((mask == NEGATIVE).any()))  # same-study pairs are never negatives

    def test_policies(self):
        captions, studies, concepts = ["a", "b", "c"], ["s1", "s1", "s2"], ["k", "j", "k"]
        self.assertEqual(build_pair_mask(captions, studies, same_study="positive").tolist(),
                         [[P, P, N], [P, P, N], [N, N, P]])
        self.assertEqual(build_pair_mask(captions, studies, concepts).tolist(),
                         [[P, I, I], [I, P, N], [I, N, P]])
        self.assertEqual(build_pair_mask(captions, studies, concepts, equal_concepts="positive").tolist(),
                         [[P, I, P], [I, P, N], [P, N, P]])
        self.assertEqual(build_pair_mask(captions, studies, concepts, equal_concepts="off").tolist(),
                         [[P, I, N], [I, P, N], [N, N, P]])
        with self.assertRaises(ValueError):
            build_pair_mask(captions, studies, same_study="negative")
        with self.assertRaises(ValueError):
            build_pair_mask(captions, studies[:2])


class TestInfoNCE(unittest.TestCase):
    def test_plain_clip_loss_when_all_off_diagonal_are_negative(self):
        logits = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
        mask = torch.tensor([[P, N], [N, P]], dtype=torch.int8)
        row0 = -(2.0 - math.log(math.exp(2.0) + math.exp(0.0)))
        row1 = -(3.0 - math.log(math.exp(1.0) + math.exp(3.0)))
        self.assertAlmostEqual(float(multi_positive_infonce(logits, mask)), (row0 + row1) / 2, places=6)
        self.assertAlmostEqual(float(torch.nn.functional.cross_entropy(logits, torch.arange(2))), (row0 + row1) / 2,
                               places=6)

    def test_multiple_positives_are_averaged(self):
        logits = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mask = build_pair_mask(["a", "a", "b"], ["s1", "s2", "s3"])
        # Rows 0 and 1: two positives each with logit 1 over denominator e + e + 1.
        row01 = -(1.0 - math.log(2 * math.e + 1))
        row2 = -(1.0 - math.log(math.e + 2))
        self.assertAlmostEqual(float(multi_positive_infonce(logits, mask)), (2 * row01 + row2) / 3, places=6)

    def test_ignored_entries_leave_the_denominator(self):
        mask = build_pair_mask(["a", "b", "c"], ["s1", "s1", "s2"])
        logits = torch.tensor([[1.0, 5.0, 0.0], [5.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        expected_row0 = -(1.0 - math.log(math.e + 1.0))  # the 5.0 same-study logit is ignored
        loss = multi_positive_infonce(logits, mask)
        changed = logits.clone()
        changed[0, 1] = changed[1, 0] = -40.0
        self.assertAlmostEqual(float(loss), float(multi_positive_infonce(changed, mask)), places=6)
        row2 = -(1.0 - math.log(math.e + 2))
        self.assertAlmostEqual(float(loss), (2 * expected_row0 + row2) / 3, places=6)

    def test_rows_need_a_positive(self):
        with self.assertRaises(ValueError):
            multi_positive_infonce(torch.zeros(2, 2), torch.full((2, 2), N, dtype=torch.int8))


class TestContrastiveLoss(unittest.TestCase):
    def test_symmetric_value_and_learnable_temperature(self):
        loss_fn = ContrastiveLoss(init_temperature=0.5)
        video = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
        mask = loss_fn.mask(["a", "b"], ["s1", "s2"])
        loss, stats = loss_fn(video, text, mask)
        logits = 2.0 * video @ text.t()
        expected = (multi_positive_infonce(logits, mask) + multi_positive_infonce(logits.t(), mask.t())) / 2
        self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=5)
        self.assertAlmostEqual(stats["temperature"], 0.5, places=5)
        loss.backward()
        self.assertIsNotNone(loss_fn.logit_scale.grad)
        self.assertNotEqual(float(loss_fn.logit_scale.grad), 0.0)

    def test_scale_is_clamped_and_temperature_can_be_fixed(self):
        loss_fn = ContrastiveLoss(init_temperature=0.001, max_logit_scale=100.0, learnable_temperature=False)
        self.assertAlmostEqual(float(loss_fn.scale()), 100.0, places=4)
        self.assertFalse(loss_fn.logit_scale.requires_grad)


if __name__ == "__main__":
    unittest.main()
