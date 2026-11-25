# tests/utils/test_rvsp_probe_models.py

"""The multi-view probe's models (`evals/video_classification_frozen_multi/models.py`), on CPU: a
masked slot position has no influence on the prediction (whatever it holds, and no gradient
reaches it), slot embeddings tell positions apart, the late-averaging baseline is the mean of the
present positions, and the frozen V-JEPA 2.1 encoder loads strictly, places each video's tokens at
its position and never changes while the probe trains."""

import os
import tempfile
import unittest

import torch

from app.vjepa_2_1.utils import init_video_model
from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen_multi.models import (EarlyFusionRegressor, LateAveragingRegressor, SlotEncoder,
                                                            make_probe)
from tests.utils.test_vjepa_2_1_eval_encoder import OPTIONS
from tests.utils.test_vjepa_2_1_utils import QuietLogs

LAYOUT = [("A4C", 0), ("A4C", 1), ("RVINF", 0)]
B, N, D = 2, 8, 32
PRESENT = torch.tensor([[True, False, True], [True, True, False]])


def probe(cls=EarlyFusionRegressor, depth=2, **kwargs):
    torch.manual_seed(0)
    return cls(embed_dim=D, num_heads=4, depth=depth, layout=LAYOUT, **kwargs).eval()


class TestSlotProbes(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1)
        self.tokens = torch.randn(B, len(LAYOUT), N, D)

    def scrambled(self):
        """The same tokens with something else (large values) at every masked position."""
        other = self.tokens.clone()
        other[~PRESENT] = 100.0 * torch.randn(int((~PRESENT).sum()), N, D)
        return other

    def test_a_masked_position_has_no_influence(self):
        for cls in (EarlyFusionRegressor, LateAveragingRegressor):
            with self.subTest(cls.__name__):
                head = probe(cls)
                with torch.no_grad():
                    out = head(self.tokens, PRESENT)
                    self.assertEqual(tuple(out.shape), (B, 1))
                    self.assertTrue(torch.allclose(out, head(self.scrambled(), PRESENT), atol=1e-6))
                    changed = self.tokens.clone()
                    changed[0, 0] += torch.randn(N, D)  # a present position does count
                    self.assertFalse(torch.allclose(out, head(changed, PRESENT)))

    def test_no_gradient_reaches_a_masked_position(self):
        head = probe().train()  # training mode, with activation checkpointing
        tokens = self.scrambled().requires_grad_(True)
        head(tokens, PRESENT).sum().backward()
        self.assertEqual(float(tokens.grad[~PRESENT].abs().sum()), 0.0)
        self.assertGreater(float(tokens.grad[PRESENT].abs().sum()), 0.0)

    def test_slot_embeddings_tell_positions_apart(self):
        head = probe()
        everywhere = torch.ones(B, len(LAYOUT), dtype=torch.bool)
        swapped = self.tokens[:, [2, 1, 0]]  # the A4C and RVINF videos exchanged
        with torch.no_grad():
            self.assertFalse(torch.allclose(head(self.tokens, everywhere), head(swapped, everywhere)))
            # Without the embeddings, early fusion is blind to which position holds which video.
            head.slot_embed.zero_()
            head.clip_embed.zero_()
            self.assertTrue(torch.allclose(head(self.tokens, everywhere), head(swapped, everywhere), atol=1e-5))

    def test_late_averaging_is_the_mean_of_the_present_positions(self):
        head = probe(LateAveragingRegressor)
        with torch.no_grad():
            x = head.embed(self.tokens)
            each = head.regressor(head.pooler(x.flatten(0, 1)).squeeze(1)).view(B, len(LAYOUT))
            expected = torch.stack([each[b][PRESENT[b]].mean() for b in range(B)])[:, None]
            self.assertTrue(torch.allclose(head(self.tokens, PRESENT), expected, atol=1e-6))

    def test_invalid_probes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            EarlyFusionRegressor(embed_dim=30, num_heads=4, depth=1, layout=LAYOUT)
        with self.assertRaisesRegex(ValueError, "built for 3"):
            probe()(self.tokens[:, :2], PRESENT[:, :2])
        with self.assertRaisesRegex(ValueError, "fusion"):
            make_probe("mean", embed_dim=D, num_heads=4, depth=1, layout=LAYOUT)
        self.assertEqual(probe(depth=4).pooler.blocks.__len__(), 3)  # 3 self-attention blocks + cross-attention


class TestSlotEncoder(QuietLogs):

    FRAMES, CROP = 4, 32

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        torch.manual_seed(0)
        self.trained, _ = init_video_model(
            torch.device("cpu"), model_name="vit_tiny", crop_size=self.CROP, max_num_frames=self.FRAMES, pred_depth=4,
            pred_embed_dim=96, use_mask_tokens=True, return_all_tokens=True, **OPTIONS,
        )
        self.trained.eval()

    def save(self, name, prefix="module.", drop=None):
        """A checkpoint as the trainer (and the published EchoJEPA checkpoint) saves it: `module.backbone.*`."""
        state = {f"{prefix}{k}": v for k, v in self.trained.state_dict().items() if k != drop}
        path = os.path.join(self.tmp, name)
        torch.save({"target_encoder": state, "epoch": 1}, path)
        return path

    def load(self, path):
        encoder = {"model_name": "vit_tiny", "checkpoint_key": "target_encoder", **OPTIONS}
        return SlotEncoder(init_module(
            module_name="evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder", device=torch.device("cpu"),
            frames_per_clip=self.FRAMES, resolution=self.CROP, checkpoint=path, model_kwargs={"encoder": encoder},
            wrapper_kwargs={}))

    def test_strict_load_of_trainer_and_unwrapped_checkpoints(self):
        for prefix in ("module.", ""):
            with self.subTest(prefix=prefix):
                encoder = self.load(self.save(f"ckpt{prefix}.pt", prefix))
                for k, v in self.trained.backbone.state_dict().items():
                    self.assertTrue(torch.equal(encoder.encoder.model.state_dict()[k], v), k)
        missing = next(iter(self.trained.state_dict()))
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.load(self.save("partial.pt", drop=missing))

    def videos(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        clips = [torch.randn(n, 3, self.FRAMES, self.CROP, self.CROP, generator=g) for _ in range(2)]
        return clips, [torch.arange(self.FRAMES).expand(n, self.FRAMES) for _ in range(2)]

    def test_each_video_is_encoded_and_placed_at_its_position(self):
        encoder = self.load(self.save("ckpt.pt"))
        clips, indices = self.videos(int(PRESENT.sum()))
        tokens = encoder(clips, indices, PRESENT)
        tokens_per_video = 2 * (self.FRAMES // 2) * (self.CROP // 16) ** 2  # 2 clips x 2 tubelets x 2 x 2 patches
        self.assertEqual(tuple(tokens.shape), (B, len(LAYOUT), tokens_per_video, 192))
        with torch.no_grad():
            backbone = self.trained.backbone.eval()
            expected = [torch.cat([backbone(c[j:j + 1]) for c in clips], dim=1)[0] for j in range(int(PRESENT.sum()))]
        for j, (b, p) in enumerate(PRESENT.nonzero().tolist()):
            self.assertTrue(torch.allclose(tokens[b, p], expected[j], atol=1e-5))
        self.assertEqual(float(tokens[~PRESENT].abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "present slot positions"):
            encoder(*self.videos(2), PRESENT)

    def test_the_encoder_stays_frozen_while_the_probe_trains(self):
        encoder = self.load(self.save("ckpt.pt"))
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        encoder.train()
        self.assertFalse(any(m.training for m in encoder.modules()))
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        head = make_probe("early", embed_dim=192, num_heads=16, depth=4, layout=LAYOUT).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
        head_before = {k: v.clone() for k, v in head.state_dict().items()}
        clips, indices = self.videos(int(PRESENT.sum()))
        tokens = encoder(clips, indices, PRESENT)
        self.assertFalse(tokens.requires_grad)
        torch.nn.functional.smooth_l1_loss(head(tokens, PRESENT), torch.ones(B, 1)).backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        for k, v in encoder.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        self.assertFalse(all(torch.equal(v, head_before[k]) for k, v in head.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
