# tests/utils/test_loss.py

"""Tests for the per-sample aggregates `jepa_loss` reports alongside the objective."""

import unittest

import torch

from app.vjepa.loss import jepa_loss


def _bucket(per_sample_errors, num_masks=1, num_patches=1, dim=1):
    """Build (z, h, masks) for one fpc bucket with a chosen error per sample.

    With `loss_exp=1` every mask term comes out as the mean of `per_sample_errors`,
    so the arithmetic under test is visible in the numbers.
    """
    b = len(per_sample_errors)
    h = torch.zeros(b, num_patches, dim)
    z = torch.tensor(per_sample_errors, dtype=torch.float32).view(b, 1, 1).expand(b, num_patches, dim).clone()
    masks = [torch.zeros(b, num_patches, dtype=torch.long) for _ in range(num_masks)]
    return [z] * num_masks, h, masks


class TestJepaLoss(unittest.TestCase):

    def test_unequal_buckets_report_the_true_sample_mean(self):
        """The reviewer's counterexample: 1 clip at 0.0 and 3 clips at 4.0."""
        z_a, h_a, m_a = _bucket([0.0])
        z_b, h_b, m_b = _bucket([4.0, 4.0, 4.0])

        loss, sample_loss_sum, num_samples = jepa_loss([z_a, z_b], [h_a, h_b], [m_a, m_b], loss_exp=1)

        # The objective is unchanged: both buckets weigh the same.
        self.assertAlmostEqual(float(loss), 2.0, places=6)
        # The metrics path sees the mean over the four clips instead.
        self.assertEqual(num_samples, 4)
        self.assertAlmostEqual(float(sample_loss_sum), 12.0, places=6)
        self.assertAlmostEqual(float(sample_loss_sum) / num_samples, 3.0, places=6)
        # Which is exactly what the old `loss * batch_size` total got wrong.
        self.assertNotAlmostEqual(float(loss) * num_samples, float(sample_loss_sum), places=6)

    def test_equal_buckets_agree_with_the_objective(self):
        """With equally sized buckets the two weightings coincide."""
        z_a, h_a, m_a = _bucket([0.0, 0.0])
        z_b, h_b, m_b = _bucket([4.0, 4.0])

        loss, sample_loss_sum, num_samples = jepa_loss([z_a, z_b], [h_a, h_b], [m_a, m_b], loss_exp=1)

        self.assertEqual(num_samples, 4)
        self.assertAlmostEqual(float(sample_loss_sum) / num_samples, float(loss), places=6)

    def test_single_bucket_agrees_with_the_objective(self):
        """The shipped config has one fpc bucket, so nothing may shift there."""
        z, h, m = _bucket([1.0, 2.0, 3.0, 6.0], num_masks=3, num_patches=4, dim=2)

        loss, sample_loss_sum, num_samples = jepa_loss([z], [h], [m], loss_exp=1)

        self.assertEqual(num_samples, 4)
        self.assertAlmostEqual(float(sample_loss_sum) / num_samples, float(loss), places=6)

    def test_masks_within_a_bucket_are_averaged_not_summed(self):
        """Adding masks must not inflate a bucket's per-clip loss."""
        one_mask = jepa_loss(*[[x] for x in _bucket([2.0, 2.0], num_masks=1)], loss_exp=1)
        three_masks = jepa_loss(*[[x] for x in _bucket([2.0, 2.0], num_masks=3)], loss_exp=1)

        self.assertAlmostEqual(float(one_mask[1]), float(three_masks[1]), places=6)
        self.assertEqual(one_mask[2], three_masks[2])

    def test_aggregates_are_detached_and_leave_the_objective_differentiable(self):
        z_a, h_a, m_a = _bucket([0.0])
        z_b, h_b, m_b = _bucket([4.0, 4.0, 4.0])
        z_a = [zi.requires_grad_(True) for zi in z_a]
        z_b = [zi.requires_grad_(True) for zi in z_b]

        loss, sample_loss_sum, _ = jepa_loss([z_a, z_b], [h_a, h_b], [m_a, m_b], loss_exp=1)

        # Nothing in the metrics path can reach the gradient path.
        self.assertFalse(sample_loss_sum.requires_grad)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(z_b[0].grad)


if __name__ == "__main__":
    unittest.main()
