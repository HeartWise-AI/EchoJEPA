# tests/utils/test_vjepa_2_1_utils.py

"""Tests for the V-JEPA 2.1 training helpers EchoJEPA adds or changes: predictor
normalization, strict weight initialisation and resume, and the choice of checkpoint
to load."""

import logging
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from app.vjepa_2_1 import train
from app.vjepa_2_1.utils import (
    init_video_model,
    load_checkpoint,
    load_pretrained_weights,
    normalize_nested,
    select_load_path,
)

EMBED_DIM = 192  # vit_tiny


def tiny_model(n_output_distillation):
    """A vit_tiny encoder and a 4-block predictor on 4-frame 32 px clips: 8 tokens."""
    return init_video_model(
        torch.device("cpu"), model_name="vit_tiny", crop_size=32, max_num_frames=4,
        pred_depth=4, pred_embed_dim=96, use_rope=True, use_mask_tokens=True,
        return_all_tokens=True, n_output_distillation=n_output_distillation,
    )


class QuietLogs(unittest.TestCase):
    """init_video_model prints every model it builds, and the trainer its settings."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.INFO)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)


class TestNormalizePredictor(QuietLogs):
    """normalize_predictor: true, on the predictor's real output, for both output counts."""

    def test_predictions_are_normalised_like_the_targets(self):
        clips = [torch.randn(2, 3, 4, 32, 32)]
        masks_enc = [[torch.tensor([[0, 1, 2, 3]] * 2)]]
        masks_pred = [[torch.tensor([[4, 5, 6, 7]] * 2)]]
        for n in (1, 4):
            with self.subTest(n_output_distillation=n):
                encoder, predictor = tiny_model(n)
                with torch.no_grad():
                    z = encoder(clips, masks_enc, training_mode=True)
                    z_pred, z_context = predictor(z, masks_enc, masks_pred)
                    normed = normalize_nested(z_pred, EMBED_DIM)

                self.assertEqual(len(normed), 1)
                self.assertEqual(len(normed[0]), 1)
                out = normed[0][0]
                self.assertEqual(tuple(out.shape), (2, 4, n * EMBED_DIM))
                # The same per-layer LayerNorm the training loop applies to the targets.
                expected = torch.cat(
                    [F.layer_norm(c, (EMBED_DIM,)) for c in z_pred[0][0].split(EMBED_DIM, dim=-1)], dim=-1
                )
                self.assertTrue(torch.allclose(out, expected, atol=1e-5))
                self.assertEqual(tuple(normalize_nested(z_context, EMBED_DIM)[0][0].shape), (2, 4, n * EMBED_DIM))


class TestLoadPretrainedWeights(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(0)
        cls.source = tiny_model(1)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.dir = tmp.name

    def checkpoint(self, name, ema_key="target_encoder", prefix="module.", drop=None, source=None, ema=None):
        encoder, predictor = source or self.source
        entries = {"encoder": encoder, "predictor": predictor, ema_key: ema or encoder}
        ckpt = {k: {prefix + p: t for p, t in m.state_dict().items()} for k, m in entries.items() if k != drop}
        ckpt["epoch"] = 3
        path = os.path.join(self.dir, name)
        torch.save(ckpt, path)
        return path

    def assert_loaded(self, model, source):
        for (name, a), b in zip(model.state_dict().items(), source.state_dict().values()):
            self.assertTrue(torch.equal(a, b), name)

    def test_ddp_prefixed_checkpoint_loads_every_tensor(self):
        path = self.checkpoint("ddp.pt")
        encoder, predictor = tiny_model(1)
        target = tiny_model(1)[0]
        load_pretrained_weights(path, encoder, predictor, target)
        self.assert_loaded(encoder, self.source[0])
        self.assert_loaded(predictor, self.source[1])
        self.assert_loaded(target, self.source[0])

    def test_unprefixed_keys_load_too(self):
        path = self.checkpoint("plain.pt", prefix="")
        encoder, predictor = tiny_model(1)
        load_pretrained_weights(path, encoder, predictor, tiny_model(1)[0])
        self.assert_loaded(encoder, self.source[0])

    def test_ema_encoder_is_read_when_there_is_no_target_encoder(self):
        ema = tiny_model(1)[0]  # weights of its own, unlike the online encoder
        path = self.checkpoint("ema.pt", ema_key="ema_encoder", ema=ema)
        encoder, predictor = tiny_model(1)
        target = tiny_model(1)[0]
        load_pretrained_weights(path, encoder, predictor, target)
        self.assert_loaded(target, ema)
        self.assert_loaded(encoder, self.source[0])

    def test_missing_entry_names_what_is_missing(self):
        path = self.checkpoint("no_predictor.pt", drop="predictor")
        encoder, predictor = tiny_model(1)
        with self.assertRaisesRegex(KeyError, "no `predictor` weights"):
            load_pretrained_weights(path, encoder, predictor, tiny_model(1)[0])

    def test_four_output_checkpoint_does_not_load_into_a_one_output_model(self):
        path = self.checkpoint("four.pt", source=tiny_model(4))
        encoder, predictor = tiny_model(1)
        with self.assertRaisesRegex(RuntimeError, "`predictor` does not match .* `n_output_distillation`"):
            load_pretrained_weights(path, encoder, predictor, tiny_model(1)[0])

    def test_a_missing_tensor_fails_instead_of_being_skipped(self):
        path = self.checkpoint("partial.pt")
        ckpt = torch.load(path)
        ckpt["encoder"].pop(next(iter(ckpt["encoder"])))
        torch.save(ckpt, path)
        encoder, predictor = tiny_model(1)
        with self.assertRaisesRegex(RuntimeError, "`encoder` does not match"):
            load_pretrained_weights(path, encoder, predictor, tiny_model(1)[0])


class TestLoadCheckpoint(QuietLogs):
    """A resume restores weights, optimizer, scaler and epoch exactly, or fails."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        torch.manual_seed(0)
        self.encoder, self.predictor = tiny_model(1)
        self.target = tiny_model(1)[0]
        self.opt = self.optimizer(self.encoder, self.predictor)
        for p in self.opt.param_groups[0]["params"]:
            p.grad = torch.ones_like(p)
        self.opt.step()  # gives the optimizer state to restore

    def optimizer(self, *models):
        return torch.optim.AdamW([p for m in models for p in m.parameters()], lr=1e-3)

    def save(self, drop=(), **changes):
        ckpt = {
            "encoder": self.encoder.state_dict(), "predictor": self.predictor.state_dict(),
            "target_encoder": self.target.state_dict(), "opt": self.opt.state_dict(), "scaler": None, "epoch": 5,
        }
        ckpt.update(changes)
        path = os.path.join(self.dir, "latest.pth.tar")
        torch.save({k: v for k, v in ckpt.items() if k not in drop}, path)
        return path

    def resume(self, path, n_output_distillation=1, opt=None, scaler=None, is_anneal=False):
        encoder, predictor = tiny_model(n_output_distillation)
        target = tiny_model(n_output_distillation)[0]
        opt = opt or self.optimizer(encoder, predictor)
        return load_checkpoint(path, encoder, predictor, target, opt, scaler, is_anneal=is_anneal)

    def test_a_checkpoint_of_the_same_run_restores_everything(self):
        encoder, predictor, target, opt, _, epoch = self.resume(self.save())
        self.assertEqual(epoch, 5)
        for a, b in zip(encoder.state_dict().values(), self.encoder.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(len(opt.state_dict()["state"]), len(self.opt.state_dict()["state"]))

    def test_a_cooldown_start_resets_the_epoch(self):
        self.assertEqual(self.resume(self.save(), is_anneal=True)[-1], 0)

    def test_a_missing_tensor_fails(self):
        encoder = dict(self.encoder.state_dict())
        encoder.pop(next(iter(encoder)))
        with self.assertRaisesRegex(RuntimeError, "`encoder` does not match"):
            self.resume(self.save(encoder=encoder))

    def test_a_mis_shaped_tensor_fails(self):
        with self.assertRaisesRegex(RuntimeError, "`predictor` does not match"):
            self.resume(self.save(), n_output_distillation=4)

    def test_a_mismatched_optimizer_fails(self):
        encoder, predictor = tiny_model(1)
        with self.assertRaisesRegex(ValueError, "optimizer state does not match"):
            self.resume(self.save(), opt=self.optimizer(encoder))

    def test_the_gradient_scaler_must_match(self):
        with self.assertRaisesRegex(ValueError, "without a gradient scaler .* uses one"):
            self.resume(self.save(), scaler=torch.amp.GradScaler("cpu"))

    def test_a_weights_only_file_is_refused(self):
        with self.assertRaisesRegex(KeyError, "not a training checkpoint.*init_checkpoint"):
            self.resume(self.save(drop=("opt", "scaler", "epoch")))


class TestSelectLoadPath(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.latest = os.path.join(tmp.name, "latest.pth.tar")
        self.anneal = os.path.join(tmp.name, "pretrain_latest.pth.tar")
        self.missing = os.path.join(tmp.name, "absent.pth.tar")

    def touch(self, path):
        open(path, "w").close()

    def select(self, load_model=True, r_file=None, is_anneal=False, resume_anneal=False, anneal_ckpt=None):
        return select_load_path(self.latest, load_model, r_file, is_anneal, resume_anneal, anneal_ckpt)

    def test_cooldown_loads_anneal_ckpt_even_with_load_checkpoint_false(self):
        self.touch(self.anneal)
        self.assertEqual(self.select(load_model=False, is_anneal=True, anneal_ckpt=self.anneal), (self.anneal, False))

    def test_cooldown_without_anneal_ckpt_fails_whatever_load_checkpoint_says(self):
        for load_model in (True, False):
            with self.assertRaisesRegex(FileNotFoundError, "anneal_ckpt"):
                self.select(load_model=load_model, is_anneal=True, anneal_ckpt=self.missing)

    def test_cooldown_resumes_its_own_latest_checkpoint(self):
        self.touch(self.latest)
        selected = self.select(is_anneal=True, resume_anneal=True, anneal_ckpt=self.missing)
        self.assertEqual(selected, (self.latest, True))

    def test_cooldown_without_its_own_checkpoint_starts_from_anneal_ckpt(self):
        self.touch(self.anneal)
        selected = self.select(is_anneal=True, resume_anneal=True, anneal_ckpt=self.anneal)
        self.assertEqual(selected, (self.anneal, False))

    def test_pretraining_loads_only_when_asked_and_present(self):
        self.assertEqual(self.select(load_model=True), (None, False))
        self.touch(self.latest)
        self.assertEqual(self.select(load_model=False), (None, False))
        self.assertEqual(self.select(load_model=True), (self.latest, False))
        self.touch(self.anneal)
        self.assertEqual(self.select(load_model=True, r_file=self.anneal), (self.anneal, False))

    def test_a_cooldown_refuses_read_checkpoint_instead_of_ignoring_it(self):
        self.touch(self.anneal)
        with self.assertRaisesRegex(ValueError, "remove `meta.read_checkpoint`"):
            self.select(r_file=self.anneal, is_anneal=True, anneal_ckpt=self.anneal)

    def test_read_checkpoint_without_load_checkpoint_is_refused_instead_of_ignored(self):
        self.touch(self.anneal)
        with self.assertRaisesRegex(ValueError, "needs `meta.load_checkpoint: true`"):
            self.select(load_model=False, r_file=self.anneal)

    def test_a_missing_read_checkpoint_fails_instead_of_starting_fresh(self):
        self.touch(self.latest)  # not a fallback for an explicit read_checkpoint
        with self.assertRaisesRegex(FileNotFoundError, "read_checkpoint"):
            self.select(load_model=True, r_file=self.missing)


class TestConfigChecks(QuietLogs):

    def test_n_output_distillation_other_than_1_or_4_is_refused(self):
        for n in (0, 2, 3, "4"):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "n_output_distillation must be 1 or 4"):
                train.main({"folder": "unused", "meta": {"dtype": "float32"}, "model": {"n_output_distillation": n}})


if __name__ == "__main__":
    unittest.main()
