# tests/models/test_radar_pooler_stills.py

"""The pooler's optional still stream: without it the pooler is v0 (same parameters, same outputs); a structure's
embedding depends only on the stills it may use; a structure fed only by a still has an input; the conditioning, the
heat-map bias and the demographic FiLM all start as the identity and all receive gradients; stills given to a pooler
without the stream are refused."""

import unittest

import torch

from src.models.radar.structure_pooler import StructurePooler

D, GRID, S, C = 16, (2, 2, 2), 3, 2
L = 8
STILL_GRID, K, LS, COND = (1, 2, 2), 3, 4, 2


def make(stills=True, **kw):
    torch.manual_seed(0)
    extra = dict(still_token_grid=STILL_GRID, num_still_types=4, still_cond_dim=COND, still_heatmap_bias=True,
                 demo_dim=4) if stills else {}
    extra.update(kw)
    return StructurePooler(embed_dim=D, num_structures=S, token_grid=GRID, num_view_classes=5,
                           num_acquisition_classes=2, depth=2, num_heads=4, proj_dim=8, **extra).eval()


def clip_inputs(B=2):
    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(B, C, L, D, generator=g)
    valid = torch.tensor([[True, False]] * B)
    eligible = torch.zeros(B, S, C, dtype=torch.bool)
    eligible[:, 0, 0] = True                 # structure 0: clip 0; structures 1, 2: no clip
    return tokens, valid, eligible, torch.tensor([[0, -1]] * B), torch.tensor([[0, -1]] * B)


def still_inputs(B=2):
    g = torch.Generator().manual_seed(2)
    still = torch.randn(B, K, LS, D, generator=g)
    valid = torch.tensor([[True, True, False]] * B)
    eligible = torch.zeros(B, S, K, dtype=torch.bool)
    eligible[:, 1, 0] = True                 # structure 1: still 0 only
    eligible[:, 0, 1] = True                 # structure 0: its clip and still 1
    eligible[:, 2, 2] = True                 # structure 2: only the padded still
    return dict(still_tokens=still, still_valid=valid, still_eligible=eligible,
                still_type_ids=torch.tensor([[0, 3, -1]] * B), still_cond=torch.randn(B, K, COND, generator=g),
                still_bias=torch.randn(B, K, LS, generator=g), demo=torch.randn(B, 4, generator=g))


class TestStillStream(unittest.TestCase):

    def test_without_stream_is_v0(self):
        v0, opt_in = make(stills=False), make(stills=True)
        names0 = [n for n, _ in v0.named_parameters()]
        self.assertTrue(set(names0) < {n for n, _ in opt_in.named_parameters()})
        for name, p in v0.named_parameters():
            self.assertTrue(torch.equal(p, dict(opt_in.named_parameters())[name]), name)
        a = v0(*clip_inputs())
        b = opt_in(*clip_inputs(), demo=torch.zeros(2, 4))        # FiLM starts at the identity
        for x, y in zip(a, b):
            self.assertTrue(torch.allclose(x.float(), y.float(), atol=1e-6))

    def test_has_input_and_isolation(self):
        pooler = make()
        _, z, has = pooler(*clip_inputs(), **still_inputs())
        self.assertEqual(has[0].tolist(), [True, True, False])   # structure 2: its only still is padded
        moved = still_inputs()
        moved["still_tokens"][:, 0] += 3.0 * torch.randn(2, LS, D)    # still 0 feeds structure 1 only
        _, z2, _ = pooler(*clip_inputs(), **moved)
        self.assertTrue(torch.allclose(z[:, 0], z2[:, 0], atol=1e-6))
        self.assertTrue(torch.allclose(z[:, 2], z2[:, 2], atol=1e-6))
        self.assertFalse(torch.allclose(z[:, 1], z2[:, 1], atol=1e-4))
        padded = still_inputs()
        padded["still_tokens"][:, 2] = float("nan")             # a padded still never reaches anything
        _, z3, _ = pooler(*clip_inputs(), **padded)
        self.assertTrue(torch.allclose(z, z3, atol=1e-6))

    def test_identity_at_init_and_gradients(self):
        pooler = make()
        inputs = still_inputs()
        base = pooler(*clip_inputs(), **inputs)[1]
        no_bias = dict(inputs, still_bias=torch.zeros_like(inputs["still_bias"]),
                       still_cond=torch.zeros_like(inputs["still_cond"]))
        self.assertTrue(torch.allclose(base, pooler(*clip_inputs(), **no_bias)[1], atol=1e-6))
        pooler.train()
        _, z, has = pooler(*clip_inputs(), **inputs)
        (z[has] ** 2).sum().mul(0).add(z[has].sum()).backward()
        for name in ("heatmap_scale", "still_cond.fc2.weight", "demo_film.weight", "still_marker",
                     "still_type_embed.weight"):
            grad = dict(pooler.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "without a still stream"):
            make(stills=False)(*clip_inputs(), **{k: v for k, v in still_inputs().items() if k != "demo"})
        with self.assertRaisesRegex(ValueError, "need still_token_grid"):
            make(stills=False, still_heatmap_bias=True)
        with self.assertRaisesRegex(ValueError, "still_bias is required"):
            make()(*clip_inputs(), **dict(still_inputs(), still_bias=None))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            bad = still_inputs()
            bad["still_tokens"][:, 0, 0, 0] = float("inf")
            make()(*clip_inputs(), **bad)


if __name__ == "__main__":
    unittest.main()
