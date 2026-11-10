# tests/models/test_radar_loss.py

"""RADAR's structure loss: the target matrix is the released construction (union, diagonal one, row-normalised,
soft abnormal weights), the loss is the soft cross-entropy in both directions, skips happen only for uninformative
structures (the release's abnormal gate only when asked), and every parameter stays in the graph. Also the step
schedule, the optimizer coverage check, and the text encoder with its momentum teacher."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.utils import is_torch_available

from src.models.radar.loss import StructureContrastiveLoss, caption_targets, contrastive_loss
from src.models.radar.schedule import step_schedule
from src.models.radar.text_encoder import (
    HashTokenizer,
    HFTokenizer,
    MomentumTeacher,
    TextEncoder,
    TinyTextBackbone,
    build_text_encoder,
    build_tokenizer,
)
from src.utils.optimizer_coverage import check_optimizer_coverage, optimizer_coverage_problems


def unit(*shape, seed=0):
    return F.normalize(torch.randn(*shape, generator=torch.Generator().manual_seed(seed)), dim=-1)


class TestTargets(unittest.TestCase):

    def test_identical_normal_captions_are_a_union_not_a_sum(self):
        ids = torch.tensor([7, 7, 7])
        flags = torch.tensor([True, True, True])
        Q = caption_targets(ids, flags, ~flags, use_exact=True, use_normal=True)
        self.assertTrue(torch.allclose(Q, torch.full((3, 3), 1 / 3)))   # a literal I + N + E would give 3/7, 2/7

    def test_identity_only_is_infonce(self):
        Q = caption_targets(torch.tensor([1, 1, 2]), use_exact=False)
        self.assertTrue(torch.equal(Q, torch.eye(3)))

    def test_exact_and_normal_parts(self):
        ids = torch.tensor([1, 1, 2, 3, 4])
        normal = torch.tensor([False, False, True, True, False])
        abnormal = torch.tensor([False, False, False, False, True])   # rows 0 and 1 are unknown
        Q = caption_targets(ids, normal, abnormal, use_exact=True, use_normal=False)
        self.assertTrue(torch.allclose(Q[0], torch.tensor([0.5, 0.5, 0, 0, 0])))
        self.assertTrue(torch.allclose(Q[2], torch.tensor([0, 0, 1.0, 0, 0])))
        Q = caption_targets(ids, normal, abnormal, use_exact=True, use_normal=True)
        self.assertTrue(torch.allclose(Q[2], torch.tensor([0, 0, 0.5, 0.5, 0])))
        self.assertTrue(torch.allclose(Q[0], torch.tensor([0.5, 0.5, 0, 0, 0])))   # unknown: identity + exact only

    def test_abnormal_soft_weights_follow_the_release(self):
        ids = torch.tensor([1, 2, 3, 4])
        normal = torch.tensor([True, False, False, False])
        abnormal = torch.tensor([False, True, True, False])            # row 3 unknown
        teacher, tau = unit(4, 8), 0.1
        Q = caption_targets(ids, normal, abnormal, teacher, tau, use_exact=True, use_normal=True)
        P = F.softmax(teacher @ teacher.T / tau, dim=1)
        R = torch.eye(4)
        R[1, 2], R[2, 1] = P[1, 2], P[2, 1]
        self.assertTrue(torch.allclose(Q, R / R.sum(1, keepdim=True), atol=1e-6))
        self.assertFalse(torch.allclose(Q, Q.T))                       # the teacher term is not symmetric

    def test_loss_is_soft_cross_entropy_both_ways(self):
        z, t, tau = unit(4, 8, seed=1), unit(4, 8, seed=2), 0.07
        Q = caption_targets(torch.tensor([1, 1, 2, 3]))
        logits = z @ t.T / tau
        expected = (-(Q * logits.log_softmax(-1)).sum(-1).mean() - (Q * logits.T.log_softmax(-1)).sum(-1).mean()) / 2
        self.assertTrue(torch.allclose(contrastive_loss(z, t, Q, tau), expected))


class TestStructureLoss(unittest.TestCase):

    def setUp(self):
        self.B, self.S, self.P = 4, 3, 8
        self.z = unit(self.B, self.S, self.P, seed=3).requires_grad_()
        self.t = unit(self.B, self.S, self.P, seed=4).requires_grad_()
        self.ids = torch.arange(self.B * self.S).reshape(self.B, self.S)
        self.normal = torch.zeros(self.B, self.S, dtype=torch.bool)
        self.abnormal = torch.zeros(self.B, self.S, dtype=torch.bool)

    def test_sum_over_structures_and_skips(self):
        valid = torch.ones(self.B, self.S, dtype=torch.bool)
        valid[1:, 1] = False                    # structure 1: one row
        ids = self.ids.clone()
        ids[:, 2] = 5                           # structure 2: one caption everywhere -> uniform targets
        loss_fn = StructureContrastiveLoss(["a", "b", "c"])
        loss, stats = loss_fn(self.z, self.t, valid, ids)
        self.assertEqual(stats["b"]["skipped"], "too_few_rows")
        self.assertEqual(stats["c"]["skipped"], "uniform_targets")
        expected = contrastive_loss(self.z[:, 0], self.t[:, 0], torch.eye(self.B), loss_fn.temperature)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNotNone(loss_fn.temperature.grad)
        self.assertTrue(torch.equal(self.z.grad[:, 1:], torch.zeros_like(self.z.grad[:, 1:])))

    def test_unknown_captions_are_not_skipped_when_n_is_on(self):
        valid = torch.ones(self.B, self.S, dtype=torch.bool)
        on = StructureContrastiveLoss(["a", "b", "c"], use_normal=True)
        _, stats = on(self.z, self.t, valid, self.ids, self.normal, self.abnormal)   # every row unknown
        self.assertTrue(all(s["skipped"] is None for s in stats.values()))
        gated = StructureContrastiveLoss(["a", "b", "c"], use_normal=True, release_gate=True)
        _, stats = gated(self.z, self.t, valid, self.ids, self.normal, self.abnormal)
        self.assertTrue(all(s["skipped"] == "no_abnormal_row" for s in stats.values()))

    def test_exact_pairs_count_valid_rows_the_same_in_a0_and_a1(self):
        valid = torch.ones(self.B, self.S, dtype=torch.bool)
        valid[3, 0] = False                     # an invalid row with a shared caption counts for nothing
        valid[1:, 1] = False                    # one row: no pair, skipped
        ids = self.ids.clone()
        ids[:, 0] = torch.tensor([1, 1, 1, 1])  # valid rows 0-2 share it: 3 pairs
        ids[:, 2] = torch.tensor([4, 4, 5, 5])  # 2 pairs
        counts = {}
        for name, use_exact in (("a0", False), ("a1", True)):
            loss_fn = StructureContrastiveLoss(["a", "b", "c"], use_exact=use_exact)
            _, stats = loss_fn(self.z, self.t, valid, ids)
            counts[name] = {s: st["exact_pairs"] for s, st in stats.items()}
            self.assertIsNone(stats["c"]["skipped"])
            self.assertEqual(stats["a"]["skipped"], None if name == "a0" else "uniform_targets")
        self.assertEqual(counts["a0"], {"a": 3, "b": 0, "c": 2})
        self.assertEqual(counts["a0"], counts["a1"])

    def test_normal_pairs_count_valid_normal_rows_in_every_arm_given_the_flags(self):
        valid = torch.ones(self.B, self.S, dtype=torch.bool)
        valid[3, 0] = False                     # an invalid normal row counts for nothing
        normal = torch.zeros(self.B, self.S, dtype=torch.bool)
        normal[:, 0] = True                     # valid rows 0-2 are normal: 3 pairs
        normal[:2, 2] = True                    # 1 pair; structure b has no normal row
        abnormal = ~normal
        for use_normal in (False, True):
            _, stats = StructureContrastiveLoss(["a", "b", "c"], use_normal=use_normal)(
                self.z, self.t, valid, self.ids, normal, abnormal)
            self.assertEqual({s: st["normal_pairs"] for s, st in stats.items()}, {"a": 3, "b": 0, "c": 1})
        _, stats = StructureContrastiveLoss(["a", "b", "c"])(self.z, self.t, valid, self.ids)   # no flags given
        self.assertEqual({st["normal_pairs"] for st in stats.values()}, {None})

    def test_temperature_is_clamped_without_in_place_changes(self):
        loss_fn = StructureContrastiveLoss(["a", "b", "c"], init_temperature=5.0)
        valid = torch.ones(self.B, self.S, dtype=torch.bool)
        self.assertAlmostEqual(loss_fn.effective_temperature().item(), 0.5)
        first, _ = loss_fn(self.z, self.t, valid, self.ids)
        second, _ = loss_fn(self.z, self.t, valid, self.ids)
        (first + second).backward()                       # two forwards, one backward (micro-batches, bags)
        self.assertAlmostEqual(loss_fn.temperature.item(), 5.0)
        loss_fn.clamp_temperature_()                      # after the optimizer step
        self.assertAlmostEqual(loss_fn.temperature.item(), 0.5)

    def test_float32_under_bfloat16_autocast(self):
        z, t, tau = unit(8, 16, seed=5), unit(8, 16, seed=6), torch.tensor(0.001)
        Q = caption_targets(torch.tensor([1, 1, 2, 3, 4, 5, 6, 7]), teacher=unit(8, 16, seed=7), tau=tau,
                            abnormal=torch.ones(8, dtype=torch.bool))
        reference = contrastive_loss(z, t, Q, tau)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            Q_amp = caption_targets(torch.tensor([1, 1, 2, 3, 4, 5, 6, 7]), teacher=unit(8, 16, seed=7), tau=tau,
                                    abnormal=torch.ones(8, dtype=torch.bool))
            under = contrastive_loss(z, t, Q_amp, tau)
        self.assertEqual(under.dtype, torch.float32)
        self.assertTrue(torch.equal(Q, Q_amp))
        self.assertTrue(torch.allclose(under, reference, rtol=1e-6))

    def test_requires_flags_and_teacher(self):
        valid = torch.ones(self.B, self.S, dtype=torch.bool)
        with self.assertRaises(ValueError):
            StructureContrastiveLoss(["a", "b", "c"], use_normal=True)(self.z, self.t, valid, self.ids)
        with self.assertRaises(ValueError):
            StructureContrastiveLoss(["a", "b", "c"], use_abnormal_teacher=True)(
                self.z, self.t, valid, self.ids, self.normal, self.abnormal)

    def test_everything_skipped_still_reaches_every_input(self):
        valid = torch.zeros(self.B, self.S, dtype=torch.bool)
        loss_fn = StructureContrastiveLoss(["a", "b", "c"])
        loss, _ = loss_fn(self.z, self.t, valid, self.ids)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(self.z.grad)
        self.assertIsNotNone(self.t.grad)
        self.assertIsNotNone(loss_fn.temperature.grad)


class TestSchedule(unittest.TestCase):

    def test_steps_come_from_the_study_sampler(self):
        self.assertEqual(step_schedule(7000, 48, 20),
                         {"steps_per_epoch": 145, "total_steps": 2900, "warmup_steps": 145})
        self.assertEqual(step_schedule(7000, 96, 20)["steps_per_epoch"], 72)
        self.assertEqual(step_schedule(10, 4, 1, drop_last=False)["steps_per_epoch"], 3)
        self.assertEqual(step_schedule(4, 4, 1, warmup_fraction=0.9)["warmup_steps"], 0)     # warmup ends early
        self.assertEqual(step_schedule(8, 4, 1, warmup_fraction=0.9)["warmup_steps"], 1)

    def test_refusals(self):
        for args in [(10, 20, 1), (0, 4, 1), (10, 6, 1, 0.05, 4), (10, 4, 1, 1.0)]:
            with self.subTest(args), self.assertRaises(ValueError):
                step_schedule(*args)


class TestOptimizerCoverage(unittest.TestCase):

    def test_detects_missing_duplicated_and_frozen(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        model[1].bias.requires_grad_(False)
        good = torch.optim.SGD([model[0].weight, model[0].bias, model[1].weight], lr=0.1)
        check_optimizer_coverage(model, good)
        # torch refuses a parameter in two groups, so a stand-in shows what the check reads.
        bad = SimpleNamespace(param_groups=[{"params": [model[0].weight, model[1].bias]},
                                            {"params": [model[0].weight]}])
        problems = optimizer_coverage_problems(model, bad)
        self.assertEqual(problems["missing"], ["0.bias", "1.weight"])
        self.assertEqual(problems["duplicated"], ["0.weight"])
        self.assertEqual(problems["not_trainable"], ["1.bias"])
        with self.assertRaises(ValueError):
            check_optimizer_coverage(model, bad)


class TestTextEncoder(unittest.TestCase):
    """The wrapper and teacher on the pure-PyTorch backbone, so they are covered whatever transformers supports."""

    def tiny(self, **kw):
        torch.manual_seed(0)
        backbone = TinyTextBackbone(vocab_size=50, hidden_size=16, num_layers=3, num_heads=2, max_length=32)
        return TextEncoder(backbone, 16, proj_dim=8, **kw)

    def test_output_and_freezing(self):
        enc = self.tiny(trainable_layers=1)
        ids = torch.randint(0, 50, (3, 6), generator=torch.Generator().manual_seed(0))
        out = enc(ids, torch.ones_like(ids))
        self.assertEqual(out.shape, (3, 8))
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5))
        trainable = {n.split(".")[0] if n.startswith("proj") else ".".join(n.split(".")[1:4])
                     for n, p in enc.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"proj", "encoder.layer.2"})
        (out.sum()).backward()
        self.assertIsNone(enc.backbone.token_embed.weight.grad)
        with self.assertRaises(ValueError):
            self.tiny(trainable_layers=4)

    def test_teacher_is_an_ema_without_gradients(self):
        enc = self.tiny()
        teacher = MomentumTeacher(enc, momentum=0.9)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        before = teacher.model.proj.weight.clone()
        with torch.no_grad():
            enc.proj.weight.add_(1.0)
        teacher.update(enc)
        self.assertTrue(torch.allclose(teacher.model.proj.weight, 0.9 * before + 0.1 * enc.proj.weight))
        teacher.train()
        self.assertFalse(teacher.model.training)
        ids = torch.randint(0, 50, (2, 5), generator=torch.Generator().manual_seed(1))
        self.assertFalse(teacher(ids, torch.ones_like(ids)).requires_grad)

    def test_hash_tokenizer(self):
        tok = HashTokenizer(vocab_size=50, max_length=4)
        ids, mask = tok(["Pas d'épanchement", "", "a b c d e f"])
        self.assertEqual(ids.shape, (3, 4))
        self.assertEqual(ids[:, 0].tolist(), [1, 1, 1])
        self.assertEqual(mask.sum(1).tolist(), [3, 1, 4])
        self.assertTrue(torch.equal(tok(["Pas d'épanchement"])[0], ids[:1, :3]))

    def test_lengths_are_what_a_tokenizer_needs_before_truncating(self):
        tok = HashTokenizer(vocab_size=50, max_length=4)
        self.assertEqual(tok.lengths(["Pas d'épanchement", "", "a b c d e f"]), [3, 1, 7])

    def test_hf_lengths_ask_for_no_truncation(self):
        tok = HFTokenizer.__new__(HFTokenizer)
        calls = []

        def fake(texts, truncation):
            calls.append(truncation)
            return {"input_ids": [t.split() for t in texts]}

        tok.tokenizer = fake
        self.assertEqual(tok.lengths(["a b", "c"]), [2, 1])
        self.assertEqual(calls, [False])

    def test_a_tokenizer_can_be_built_without_its_encoder(self):
        tok = build_tokenizer({"backbone": "tiny", "vocab_size": 50, "max_length": 4})
        self.assertEqual((tok.vocab_size, tok.max_length), (50, 4))
        with self.assertRaisesRegex(ValueError, "text backbone"):
            build_tokenizer({"backbone": "other"})


@unittest.skipUnless(is_torch_available(), "transformers cannot load PyTorch models with this torch version")
class TestHuggingFaceBackbone(unittest.TestCase):

    def test_random_bert_through_the_builder(self):
        torch.manual_seed(0)
        enc, tok = build_text_encoder({"backbone": "hf_random", "model_type": "bert", "trainable_layers": 1,
                                       "config": {"vocab_size": 50, "hidden_size": 16, "num_hidden_layers": 2,
                                                  "num_attention_heads": 2, "intermediate_size": 32}}, proj_dim=8)
        ids, mask = tok(["vg non dilaté", "pas d'épanchement"])
        self.assertEqual(enc(ids, mask).shape, (2, 8))
        self.assertTrue(any(p.requires_grad for p in enc.backbone.parameters()))
        self.assertFalse(enc.backbone.embeddings.word_embeddings.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
