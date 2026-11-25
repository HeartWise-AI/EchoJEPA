# tests/models/test_radar_pooler.py

"""The structure pooler: a structure's embedding depends only on its eligible, real clips; a structure without any
eligible, real clip stays finite, is flagged and depends on no token; only such a structure sees the null token, which
is a fixed zero; each structure has its own projection; gradients reach every part."""

import unittest

import torch

from src.models.radar.structure_pooler import StructurePooler

D, GRID, S, C = 16, (2, 2, 2), 3, 4
L = 8


def make(**kw):
    torch.manual_seed(0)
    return StructurePooler(embed_dim=D, num_structures=S, token_grid=GRID, num_view_classes=5,
                           num_acquisition_classes=2, depth=2, num_heads=4, proj_dim=8, **kw).eval()


def noise(*shape):
    """A random perturbation (a constant shift would be removed by the keys' LayerNorm)."""
    return 3.0 * torch.randn(*shape, generator=torch.Generator().manual_seed(7))


def inputs(B=2):
    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(B, C, L, D, generator=g)
    valid = torch.tensor([[True, True, True, False]] * B)
    eligible = torch.zeros(B, S, C, dtype=torch.bool)
    eligible[:, 0, 0] = True          # structure 0: clip 0 only
    eligible[:, 1, 1:] = True         # structure 1: clips 1, 2 and the padded slot 3
    # structure 2: no clip at all
    views = torch.tensor([[0, 1, 2, -1]] * B)
    acq = torch.tensor([[0, 0, 1, -1]] * B)
    return tokens, valid, eligible, views, acq


class TestStructurePooler(unittest.TestCase):

    def test_shapes_norms_and_flags(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        features, emb, has_clip = pooler(tokens, valid, eligible, views, acq)
        self.assertEqual(features.shape, (2, S, D))
        self.assertEqual(emb.shape, (2, S, 8))
        self.assertTrue(torch.allclose(emb.norm(dim=-1), torch.ones(2, S), atol=1e-5))
        self.assertEqual(has_clip.tolist(), [[True, True, False]] * 2)
        self.assertTrue(torch.isfinite(features).all() and torch.isfinite(emb).all())

    def test_ineligible_and_padded_clips_do_not_change_a_structure(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        _, before, _ = pooler(tokens, valid, eligible, views, acq)
        changed = tokens.clone()
        changed[:, 2] += noise(2, L, D)   # eligible for structure 1 only
        changed[:, 3] += noise(2, L, D)   # padded slot
        _, after, _ = pooler(changed, valid, eligible, views, acq)
        self.assertTrue(torch.allclose(before[:, 0], after[:, 0], atol=1e-6))   # structure 0 uses clip 0 only
        self.assertTrue(torch.allclose(before[:, 2], after[:, 2], atol=1e-6))   # structure 2 uses nothing
        self.assertFalse(torch.allclose(before[:, 1], after[:, 1], atol=1e-4))  # clip 2 is structure 1's

    def test_null_token_reaches_only_structures_without_a_clip(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        _, before, has_clip = pooler(tokens, valid, eligible, views, acq)
        pooler.null_token.copy_(noise(1, 1, D))
        _, after, _ = pooler(tokens, valid, eligible, views, acq)
        self.assertTrue(torch.equal(before[has_clip], after[has_clip]))
        self.assertFalse(torch.allclose(before[~has_clip], after[~has_clip], atol=1e-4))

    def test_null_token_is_a_fixed_zero(self):
        pooler = make()
        self.assertTrue(torch.equal(pooler.null_token, torch.zeros(1, 1, D)))
        self.assertNotIn("null_token", dict(pooler.named_parameters()))
        self.assertNotIn("null_token", pooler.state_dict())

    def test_a_padded_slot_is_not_a_clip(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        _, reference, _ = pooler(tokens, valid, eligible, views, acq)
        only_padded = eligible.clone()
        only_padded[:, 2, 3] = True                      # structure 2: the padded slot 3 alone
        poisoned = tokens.clone()
        poisoned[:, 3] = float("nan")
        features, emb, has_clip = pooler(poisoned, valid, only_padded, views, acq)
        self.assertEqual(has_clip.tolist(), [[True, True, False]] * 2)
        self.assertTrue(torch.isfinite(features).all() and torch.isfinite(emb).all())
        self.assertTrue(torch.allclose(emb[:, 2], reference[:, 2], atol=1e-6))

    def test_a_row_without_a_clip_depends_on_no_token(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        none = torch.zeros_like(valid)                   # a study with no real clip at all
        none[0] = valid[0]
        _, before, has_clip = pooler(tokens, none, eligible, views, acq)
        self.assertEqual(has_clip.tolist(), [[True, True, False], [False, False, False]])
        self.assertTrue(torch.isfinite(before).all())
        _, after, _ = pooler(tokens + noise(2, C, L, D), valid, eligible, views, acq)
        self.assertTrue(torch.allclose(before[0, 2], after[0, 2], atol=1e-6))
        self.assertTrue(torch.allclose(before[1, 2], after[1, 2], atol=1e-6))

    def test_views_of_other_clips_do_not_leak(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        _, before, _ = pooler(tokens, valid, eligible, views, acq)
        other = views.clone()
        other[:, 1] = 4
        _, after, _ = pooler(tokens, valid, eligible, other, acq)
        self.assertTrue(torch.allclose(before[:, 0], after[:, 0], atol=1e-6))
        self.assertFalse(torch.allclose(before[:, 1], after[:, 1], atol=1e-4))

    def test_studies_are_independent(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        _, before, _ = pooler(tokens, valid, eligible, views, acq)
        changed = tokens.clone()
        changed[1] += noise(C, L, D)
        _, after, _ = pooler(changed, valid, eligible, views, acq)
        self.assertTrue(torch.allclose(before[0], after[0], atol=1e-6))

    def test_each_structure_has_its_own_projection(self):
        pooler = make()
        tokens, valid, _, views, acq = inputs()
        everything = torch.ones(2, S, C, dtype=torch.bool)
        with torch.no_grad():
            pooler.queries.copy_(pooler.queries[:1].expand(S, D))   # same query everywhere
        features, emb, _ = pooler(tokens, valid, everything, views, acq)
        self.assertTrue(torch.allclose(features[:, 0], features[:, 1], atol=1e-6))
        self.assertFalse(torch.allclose(emb[:, 0], emb[:, 1], atol=1e-4))

    def test_gradients_reach_every_parameter_with_checkpointing(self):
        pooler = make(use_activation_checkpointing=True).train()
        tokens, valid, eligible, views, acq = inputs()
        features, emb, has_clip = pooler(tokens, valid, eligible, views, acq)
        (emb[has_clip].sum() + 0.0 * emb.sum()).backward()
        missing = [n for n, p in pooler.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])

    def test_padded_slots_are_zeroed_whatever_they_hold(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        clean = tokens.clone()
        clean[:, 3] = 0.0
        _, reference, _ = pooler(clean, valid, eligible, views, acq)
        poisoned = tokens.clone()
        poisoned[:, 3] = float("nan")                    # padded slot 3 holds NaN
        features, emb, _ = pooler(poisoned, valid, eligible, views, acq)
        self.assertTrue(torch.isfinite(features).all() and torch.isfinite(emb).all())
        self.assertTrue(torch.allclose(emb, reference, atol=1e-6))

    def test_non_finite_tokens_in_a_real_clip_are_refused(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        tokens[0, 1, 2, 3] = float("inf")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            pooler(tokens, valid, eligible, views, acq)

    def test_refuses_bad_shapes(self):
        pooler = make()
        tokens, valid, eligible, views, acq = inputs()
        with self.assertRaises(ValueError):
            pooler(tokens[:, :, :4], valid, eligible, views, acq)
        with self.assertRaises(ValueError):
            pooler(tokens, valid, eligible[:, :2], views, acq)
        with self.assertRaises(ValueError):
            pooler(tokens, valid, eligible, None, acq)


if __name__ == "__main__":
    unittest.main()
