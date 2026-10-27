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


class TestStudyAblations(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)
        self.specs = head_specs(CLASSIFIER)
        self.tokens = torch.randn(2, 3 * 2 * 8, 24)  # 3 videos, 2 clips/video, grid 2x2x2
        self.mask = torch.tensor([[True, True, False], [True, False, False]])
        self.views = torch.tensor([[0, 1, -1], [1, -1, -1]])

    def model(self, **kwargs):
        options = dict(pooling_mode="hierarchical", num_heads=4, depth=2, query_mode="task",
                       num_view_classes=2, local_depth=1, local_num_queries=3, local_use_rope=True,
                       token_grid=(2, 2, 2), num_clips_per_video=2)
        options.update(kwargs)
        return StudyProbe(24, self.specs, **options)

    def test_video_order_and_clip_order_do_not_change_hierarchical_predictions(self):
        model = self.model().eval()
        expected = model(self.tokens, self.mask, self.views)
        videos = self.tokens.reshape(2, 3, 2, 8, 24)
        order = [2, 0, 1]
        reordered = videos[:, order].flip(2).reshape_as(self.tokens)
        actual = model(reordered, self.mask[:, order], self.views[:, order])
        for head in expected:
            torch.testing.assert_close(expected[head], actual[head], atol=1e-6, rtol=1e-5)
        # Padded content may be arbitrary, even nonfinite, without contaminating the study.
        other = videos.clone()
        other[~self.mask] = float("nan")
        actual = model(other.reshape_as(self.tokens), self.mask, self.views)
        for head in expected:
            torch.testing.assert_close(expected[head], actual[head])

    def test_task_queries_route_gradients_to_the_corresponding_readout(self):
        for pooling in ("attentive", "hierarchical"):
            model = self.model(pooling_mode=pooling, local_use_rope=pooling == "hierarchical")
            output = model(self.tokens, self.mask, self.views)
            output["as_present"].sum().backward()
            grad = model.pooler.query_tokens.grad[0]
            self.assertGreater(grad[1].abs().sum().item(), 0)
            self.assertEqual(grad[[0, 2]].abs().sum().item(), 0)
            if pooling == "hierarchical":
                self.assertGreater(model.clip_pooler.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)

    def test_group_queries_share_only_the_requested_groups(self):
        model = self.model(query_mode="group", task_groups={"lvef": "lv", "as_present": "av", "severity": "av"})
        self.assertEqual(model.head_query, {"lvef": 0, "as_present": 1, "severity": 1})
        self.assertEqual(model.pooler.query_tokens.shape[1], 2)
        self.assertEqual(model(self.tokens, self.mask, self.views)["severity"].shape, (2, 3))
        with self.assertRaisesRegex(ValueError, "every target"):
            self.model(query_mode="group", task_groups={"lvef": "lv"})

    def test_soft_view_probabilities_match_hard_labels_and_mixtures(self):
        model = self.model(pooling_mode="mean", query_mode="shared", local_use_rope=False).eval()
        onehot = torch.nn.functional.one_hot(self.views.clamp(min=0), 2).float()
        hard = model.pool(self.tokens, self.mask, self.views)
        torch.testing.assert_close(hard, model.pool(self.tokens, self.mask, view_probs=onehot))
        mixed = torch.full_like(onehot, 0.5)
        expected = (self.tokens.reshape(2, 3, -1, 24).mean(2) * self.mask.unsqueeze(-1)).sum(1)
        expected /= self.mask.sum(1, keepdim=True)
        expected += model.view_embed.weight[:2].mean(0)
        torch.testing.assert_close(expected, model.pool(self.tokens, self.mask, view_probs=mixed))
        with self.assertRaisesRegex(ValueError, "summing to one"):
            model(self.tokens, self.mask, view_probs=mixed * 2)

    def test_view_dropout_is_per_video_training_only_and_types_are_independent(self):
        model = self.model(view_dropout=1, num_acquisition_classes=2)
        types = torch.tensor([[0, 1, -1], [0, -1, -1]])
        swapped = self.views.clone()
        swapped[self.mask] = 1 - swapped[self.mask]
        a = model.pool(self.tokens, self.mask, self.views, acquisition_ids=types)
        b = model.pool(self.tokens, self.mask, swapped, acquisition_ids=types)
        torch.testing.assert_close(a, b)
        model.eval()
        a = model.pool(self.tokens, self.mask, self.views, acquisition_ids=types)
        b = model.pool(self.tokens, self.mask, swapped, acquisition_ids=types)
        self.assertFalse(torch.allclose(a, b))
        other_types = types.clone()
        other_types[self.mask] = 1 - other_types[self.mask]
        c = model.pool(self.tokens, self.mask, self.views, acquisition_ids=other_types)
        self.assertFalse(torch.allclose(a, c))

    def test_rope_is_local_and_changes_spatial_attention(self):
        from src.models.utils.modules import RoPEAttention
        model = self.model().eval()
        self.assertIsInstance(model.clip_pooler.blocks[0].attn, RoPEAttention)
        self.assertNotIsInstance(model.pooler.blocks[0].attn, RoPEAttention)
        plain = self.model(local_use_rope=False).eval()
        plain.load_state_dict(model.state_dict())
        a = model.clip_pooler(self.tokens[:, :8])
        b = plain.clip_pooler(self.tokens[:, :8])
        self.assertGreater((a - b).abs().max().item(), 1e-8)
        with self.assertRaisesRegex(ValueError, "patch tokens"):
            model(self.tokens[:, :24], self.mask, self.views)
        with self.assertRaisesRegex(ValueError, "hierarchical"):
            self.model(pooling_mode="attentive")
        with self.assertRaisesRegex(ValueError, "self-attention"):
            self.model(local_depth=0)

    def test_local_rotations_preserve_norm_and_relative_positions(self):
        from src.models.utils.modules import rotate_queries_or_keys
        for dim in (2, 4, 8):
            q, k = torch.randn(2, 4, 8, dim), torch.randn(2, 4, 8, dim)
            m, n = torch.arange(8), torch.arange(8).flip(0)
            rotate = lambda x, p: rotate_queries_or_keys(x, p, interleaved=True)
            qm, kn = rotate(q, m), rotate(k, n)
            torch.testing.assert_close(qm.square().sum(-1), q.square().sum(-1))
            torch.testing.assert_close((qm * kn).sum(-1), (rotate(q, m + 3) * rotate(k, n + 3)).sum(-1),
                                       atol=2e-6, rtol=1e-5)

    def test_hierarchy_trains_with_bfloat16_and_activation_checkpointing(self):
        model = self.model()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = model(self.tokens, self.mask, self.views)
            loss = sum(value.float().square().mean() for value in outputs.values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


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

    def test_global_counts_scale_each_rank_like_ddp_averaging(self):
        """Under DDP the gradients are averaged over ranks: with the global count and world size,
        the sum over ranks of the local losses is the mean over every labeled study."""
        outputs = {"lvef": torch.tensor([[3.0], [1.0]]), "as_present": torch.tensor([[0.0], [0.0]]),
                   "severity": torch.zeros(2, 3)}
        targets = torch.tensor([[1.0, 1.0, float("nan")], [1.0, float("nan"), float("nan")]])
        local, _ = multitask_loss(self.specs, outputs, targets)
        scaled, _ = multitask_loss(self.specs, outputs, targets, counts={"lvef": 4, "as_present": 2, "severity": 0},
                                   world_size=2)
        huber_sum, bce = 1.5 + 0.0, float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(0.0), torch.tensor(1.0)))
        self.assertAlmostEqual(float(local), 2.0 * huber_sum / 2 + bce / 1, places=5)
        self.assertAlmostEqual(float(scaled), 2.0 * huber_sum * 2 / 4 + bce * 2 / 2, places=5)

    def test_invalid_categorical_targets_are_refused(self):
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            head_loss(self.specs["as_present"], torch.zeros(2, 1), torch.tensor([0.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "whole numbers"):
            head_loss(self.specs["severity"], torch.zeros(2, 3), torch.tensor([0.0, 1.5]))
        with self.assertRaisesRegex(ValueError, "whole numbers"):
            head_loss(self.specs["severity"], torch.zeros(2, 3), torch.tensor([0.0, 3.0]))
        value, count = head_loss(self.specs["severity"], torch.zeros(2, 3), torch.tensor([2.0, float("nan")]))
        self.assertEqual(count, 1)

    def test_regression_losses(self):
        output, target = torch.tensor([[3.0], [1.0]]), torch.tensor([1.0, 1.0])
        for loss, expected in (("huber", 1.5), ("mse", 4.0), ("l1", 2.0)):
            value, count = head_loss({"task": "regression", "loss": loss}, output, target)
            self.assertEqual((float(value), count), (expected, 2), loss)
