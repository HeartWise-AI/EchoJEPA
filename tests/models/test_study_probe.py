# tests/models/test_study_probe.py

"""The study probe: head specs from DeepCORO_CLIP-style config keys, every pooling mode masks the
padded video slots out, and the multi-task loss leaves missing targets out."""

import unittest

import torch

from src.models.study_probe import StudyProbe, head_loss, head_specs, multitask_loss

CLASSIFIER = {
    "target_label": ["lvef", "as_present", "severity"],
    "head_task": {"lvef": "regression", "as_present": "binary", "severity": "classification"},
    "head_structure": {"lvef": 1, "as_present": 1, "severity": 3},
    "loss_structure": {"lvef": "huber", "as_present": "bce", "severity": "ce"},
    "head_weights": {"lvef": 2.0, "as_present": 1.0},
    "head_dropout": {"lvef": 0.1},
}
D, N, TOKENS = 16, 3, 4  # embed dim, video slots, tokens per video


class TestHeadSpecs(unittest.TestCase):

    def test_specs_follow_target_label_order_with_defaults(self):
        specs = head_specs(CLASSIFIER)
        self.assertEqual(list(specs), ["lvef", "as_present", "severity"])
        self.assertEqual(specs["lvef"], {"task": "regression", "outputs": 1, "loss": "huber", "weight": 2.0,
                                         "dropout": 0.1})
        self.assertEqual(specs["severity"], {"task": "classification", "outputs": 3, "loss": "ce", "weight": 1.0,
                                             "dropout": 0.0})

    def test_invalid_specs_are_refused(self):
        for change, message in (
            ({"head_task": {**CLASSIFIER["head_task"], "lvef": "ranking"}}, "head_task"),
            ({"head_structure": {**CLASSIFIER["head_structure"], "severity": 1}}, "number of classes"),
            ({"head_structure": {**CLASSIFIER["head_structure"], "lvef": 2}}, "one output"),
            ({"loss_structure": {**CLASSIFIER["loss_structure"], "lvef": "bce"}}, "loss_structure"),
            ({"head_weights": {"unknown": 1.0}}, "does not list"),
            ({"target_label": ["lvef", "lvef"]}, "repeats"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                head_specs({**CLASSIFIER, **change})


class TestStudyProbe(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.specs = head_specs(CLASSIFIER)
        self.tokens = torch.randn(2, N * TOKENS, D)
        self.mask = torch.tensor([[True, True, False], [True, False, False]])

    def probe(self, mode):
        torch.manual_seed(1)
        return StudyProbe(D, self.specs, pooling_mode=mode, num_heads=4, depth=2, attention_hidden=8).eval()

    def test_outputs_have_one_row_per_study_and_the_head_width(self):
        for mode in ("attentive", "mean", "max", "attention"):
            out = self.probe(mode)(self.tokens, self.mask)
            self.assertEqual({k: tuple(v.shape) for k, v in out.items()},
                             {"lvef": (2, 1), "as_present": (2, 1), "severity": (2, 3)}, mode)

    def test_padded_slots_do_not_change_the_outputs(self):
        """Whatever the padded slots' tokens hold, the study's outputs are the same."""
        other = self.tokens.clone()
        other[0, 2 * TOKENS:] = 10 * torch.randn(TOKENS, D)
        other[1, TOKENS:] = -7.0
        for mode in ("attentive", "mean", "max", "attention"):
            probe = self.probe(mode)
            a, b = probe(self.tokens, self.mask), probe(other, self.mask)
            for head in a:
                torch.testing.assert_close(a[head], b[head], msg=mode)

    def test_mean_pooling_is_the_mean_over_the_real_videos(self):
        probe = self.probe("mean")
        videos = self.tokens.reshape(2, N, TOKENS, D).mean(dim=2)
        expected = torch.stack([videos[0, :2].mean(dim=0), videos[1, 0]])
        torch.testing.assert_close(probe.pool(self.tokens, self.mask), expected)

    def test_a_study_without_a_real_video_is_refused(self):
        mask = torch.tensor([[True, True, False], [False, False, False]])
        with self.assertRaisesRegex(ValueError, "at least one real video"):
            self.probe("mean")(self.tokens, mask)

    def test_view_embedding_marks_each_slot_by_its_view(self):
        torch.manual_seed(1)
        probe = StudyProbe(D, self.specs, pooling_mode="mean", num_view_classes=2).eval()
        views = torch.tensor([[0, 1, -1], [1, -1, -1]])
        with self.assertRaisesRegex(ValueError, "view_ids"):
            probe(self.tokens, self.mask)
        with self.assertRaisesRegex(ValueError, "View classes"):
            probe(self.tokens, self.mask, torch.tensor([[0, 2, -1], [1, -1, -1]]))
        videos = self.tokens.reshape(2, N, TOKENS, D).mean(dim=2)
        e = probe.view_embed.weight
        expected = torch.stack([(videos[0, 0] + e[0] + videos[0, 1] + e[1]) / 2, videos[1, 0] + e[1]])
        torch.testing.assert_close(probe.pool(self.tokens, self.mask, views), expected)
        # Another view for a video changes the study embedding; the padded slots' views do not matter.
        a = probe.pool(self.tokens, self.mask, views)
        self.assertFalse(torch.allclose(a, probe.pool(self.tokens, self.mask, torch.tensor([[1, 1, -1], [0, -1, -1]]))))
        torch.testing.assert_close(a, probe.pool(self.tokens, self.mask, torch.tensor([[0, 1, 1], [1, 0, 0]])))
        for mode in ("attentive", "attention", "max"):
            torch.manual_seed(1)
            out = StudyProbe(D, self.specs, pooling_mode=mode, num_heads=4, depth=2, attention_hidden=8,
                             num_view_classes=2).eval()(self.tokens, self.mask, views)
            self.assertEqual(tuple(out["severity"].shape), (2, 3), mode)

    def test_unknown_pooling_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pooling_mode"):
            StudyProbe(D, self.specs, pooling_mode="sum")


class TestLoss(unittest.TestCase):

    def setUp(self):
        self.specs = head_specs(CLASSIFIER)

    def test_missing_targets_are_left_out(self):
        outputs = {"lvef": torch.tensor([[0.5], [2.0], [1.0]]), "as_present": torch.tensor([[0.0], [3.0], [-3.0]]),
                   "severity": torch.zeros(3, 3)}
        targets = torch.tensor([[0.0, 1.0, float("nan")], [float("nan"), float("nan"), 2.0],
                                [float("nan"), float("nan"), float("nan")]])
        loss, parts = multitask_loss(self.specs, outputs, targets)
        self.assertEqual({h: c for h, (_, c) in parts.items()}, {"lvef": 1, "as_present": 1, "severity": 1})
        huber = 0.5 * 0.5**2
        bce = float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(0.0), torch.tensor(1.0)))
        ce = float(torch.nn.functional.cross_entropy(torch.zeros(1, 3), torch.tensor([2])))
        self.assertAlmostEqual(float(loss), 2.0 * huber + bce + ce, places=5)

    def test_a_head_without_targets_adds_nothing_but_keeps_the_graph(self):
        weight = torch.ones(2, 1, requires_grad=True)
        outputs = {"lvef": weight * 1.0, "as_present": weight * 2.0, "severity": torch.zeros(2, 3)}
        targets = torch.tensor([[float("nan"), 1.0, float("nan")], [float("nan"), 0.0, float("nan")]])
        loss, parts = multitask_loss(self.specs, outputs, targets)
        self.assertEqual(parts["lvef"][1], 0)
        loss.backward()
        self.assertIsNotNone(weight.grad)

    def test_regression_losses(self):
        output, target = torch.tensor([[3.0], [1.0]]), torch.tensor([1.0, 1.0])
        for loss, expected in (("huber", 1.5), ("mse", 4.0), ("l1", 2.0)):
            value, count = head_loss({"task": "regression", "loss": loss}, output, target)
            self.assertEqual((float(value), count), (expected, 2), loss)
