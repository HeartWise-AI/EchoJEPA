# tests/utils/test_vjepa_2_1_utils.py

"""Tests for the V-JEPA 2.1 training helpers EchoJEPA adds or changes: the padded-token
loss mask, predictor normalization, strict weight initialisation and resume, and the
choice of checkpoint to load."""

import copy
import inspect
import logging
import math
import os
import socket
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from app.vjepa_2_1 import train
from app.vjepa_2_1.utils import (
    clip_grads_with_norm_,
    feature_spread,
    init_video_model,
    load_checkpoint,
    load_pretrained_weights,
    masked_l1_loss,
    normalize_nested,
    select_load_path,
    token_validity,
)

EMBED_DIM = 192  # vit_tiny


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _checkpoint_failure_worker(rank, world_size, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        def save():
            raise OSError("simulated rank-0 checkpoint failure")

        try:
            train._run_rank_zero_checkpoint_save(save, rank, torch.device("cpu"))
        except RuntimeError as exc:
            queue.put((rank, str(exc)))
        else:
            queue.put((rank, None))
    finally:
        dist.destroy_process_group()


def tiny_model(n_output_distillation):
    """A vit_tiny encoder and a 4-block predictor on 4-frame 32 px clips: 8 tokens."""
    return init_video_model(
        torch.device("cpu"), model_name="vit_tiny", crop_size=32, max_num_frames=4,
        pred_depth=4, pred_embed_dim=96, use_rope=True, use_mask_tokens=True,
        return_all_tokens=True, n_output_distillation=n_output_distillation,
    )


def upstream_loss(z, h, loss_exp, d_weights=None):
    """The loss term upstream computed for one mask, before padding was masked."""
    err = torch.abs(z - h) ** loss_exp
    if d_weights is not None:
        err = err * (1 / d_weights.unsqueeze(2))
    return torch.mean(err) / loss_exp


class QuietLogs(unittest.TestCase):
    """init_video_model prints every model it builds, and the trainer its settings."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.INFO)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)


class TestTokenValidity(unittest.TestCase):
    # 4 frames, tubelets of 2, a 2 x 2 grid: tokens 0-3 are tubelet 0, 4-7 tubelet 1.

    def validity(self, *clips):
        return token_validity(torch.tensor(clips), tubelet_size=2, grid_size=2).squeeze(-1).tolist()

    def test_real_padded_and_partly_padded_tubelets(self):
        rows = self.validity(
            [0, 1, 2, 3],     # all real
            [0, 1, -1, -1],   # tubelet 1 all padding
            [0, -1, -1, -1],  # tubelet 0 has one real frame: kept
            [-1, -1, -1, -1], # all padding
        )
        self.assertEqual(rows, [
            [1.0] * 8,
            [1.0] * 4 + [0.0] * 4,
            [1.0] * 4 + [0.0] * 4,
            [0.0] * 8,
        ])

    def test_shape_matches_the_token_sequence(self):
        v = token_validity(torch.zeros(3, 16, dtype=torch.long), tubelet_size=2, grid_size=14)
        self.assertEqual(tuple(v.shape), (3, 8 * 14 * 14, 1))

    def test_frames_that_do_not_fill_whole_tubelets_are_refused(self):
        with self.assertRaisesRegex(ValueError, "multiple of `data.tubelet_size`"):
            token_validity(torch.zeros(3, 15, dtype=torch.long), tubelet_size=2, grid_size=14)


class TestMaskedL1Loss(unittest.TestCase):

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.z = torch.randn(2, 6, 5, generator=g)
        self.h = torch.randn(2, 6, 5, generator=g)
        self.d = torch.rand(2, 6, generator=g) + 0.5
        self.valid = torch.ones(2, 6, 1)
        self.valid[1, 4:] = 0  # the last two tokens of sample 1 are padding

    def test_equals_the_upstream_loss_when_every_token_is_real(self):
        for p in (1, 2):
            for d in (None, self.d):
                expected = upstream_loss(self.z, self.h, p, d)
                self.assertTrue(torch.allclose(masked_l1_loss(self.z, self.h, p, None, d), expected))
                self.assertTrue(torch.allclose(masked_l1_loss(self.z, self.h, p, torch.ones(2, 6, 1), d), expected))

    def test_padded_tokens_do_not_count(self):
        keep = self.valid.squeeze(-1).bool()
        for p in (1, 2):
            for d in (None, self.d):
                err = torch.abs(self.z - self.h) ** p
                if d is not None:
                    err = err / d.unsqueeze(2)
                expected = err[keep].mean() / p  # mean over the real tokens only
                loss = masked_l1_loss(self.z, self.h, p, self.valid, d)
                self.assertTrue(torch.allclose(loss, expected))

                z = self.z.clone()
                z[1, 4:] += 100.0  # a prediction on padding changes nothing
                self.assertTrue(torch.allclose(masked_l1_loss(z, self.h, p, self.valid, d), loss))

    def test_an_all_padding_batch_gives_zero_not_nan(self):
        loss = masked_l1_loss(self.z, self.h, 1, torch.zeros(2, 6, 1))
        self.assertEqual(float(loss), 0.0)


class TestFeatureSpread(unittest.TestCase):
    """The collapse check: how alike the target encoder makes different clips."""

    def test_identical_clips_are_fully_collapsed(self):
        h = torch.randn(1, 6, 8).expand(4, 6, 8)
        clip_cosine, effective_rank = feature_spread(h)
        self.assertAlmostEqual(clip_cosine, 1.0, places=5)
        self.assertEqual(effective_rank, 0.0)

    def test_orthogonal_clips_use_every_direction(self):
        # Clip i has every token on axis i: no similarity, and n clips span n - 1 centred directions.
        h = torch.eye(5, 8).unsqueeze(1).expand(5, 6, 8)
        clip_cosine, effective_rank = feature_spread(h)
        self.assertAlmostEqual(clip_cosine, 0.0, places=6)
        self.assertAlmostEqual(effective_rank, 4.0, places=4)

    def test_padding_tokens_are_left_out(self):
        # Two clips alike on their real tokens, different on their padding.
        h = torch.randn(1, 6, 8).repeat(2, 1, 1)
        h[0, 4:], h[1, 4:] = 5.0, -5.0
        valid = torch.tensor([1.0] * 4 + [0.0] * 2).view(1, 6, 1).expand(2, 6, 1)
        self.assertAlmostEqual(feature_spread(h, valid)[0], 1.0, places=5)
        self.assertLess(feature_spread(h)[0], 0.99)

    def test_clips_of_every_length_are_pooled_together(self):
        short, long = torch.eye(3, 8)[:2].unsqueeze(1).expand(2, 4, 8), torch.eye(3, 8)[2:].unsqueeze(1).expand(1, 8, 8)
        clip_cosine, effective_rank = feature_spread([short, long], [None, None])
        self.assertAlmostEqual(clip_cosine, 0.0, places=6)
        self.assertAlmostEqual(effective_rank, 2.0, places=4)

    def test_one_clip_has_no_spread(self):
        self.assertTrue(all(math.isnan(v) for v in feature_spread(torch.randn(1, 6, 8))))


class TestClipGradsWithNorm(unittest.TestCase):
    """Gradient clipping against a norm the trainer already computed (and logs)."""

    def params(self):
        a, b, unused = (torch.nn.Parameter(torch.zeros(2)) for _ in range(3))
        a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0])  # total norm 5
        return [a, b, unused]

    def test_gradients_above_the_limit_are_scaled_to_it(self):
        params = self.params()
        clip_grads_with_norm_(params, 2.5, torch.tensor(5.0))
        self.assertAlmostEqual(float(torch.cat([params[0].grad, params[1].grad]).norm()), 2.5, places=5)
        self.assertIsNone(params[2].grad)

    def test_gradients_within_the_limit_are_unchanged(self):
        params = self.params()
        clip_grads_with_norm_(params, 10.0, torch.tensor(5.0))
        self.assertEqual(params[0].grad.tolist(), [3.0, 0.0])
        self.assertEqual(params[1].grad.tolist(), [0.0, 4.0])

    def test_matches_pytorch_where_pytorch_has_it(self):
        if not hasattr(torch.nn.utils, "clip_grads_with_norm_"):
            self.skipTest("this PyTorch has no clip_grads_with_norm_")
        for max_norm, total_norm in ((2.5, 5.0), (10.0, 5.0), (1.0, float("inf"))):
            with self.subTest(max_norm=max_norm, total_norm=total_norm):
                ours, theirs = self.params(), self.params()
                clip_grads_with_norm_(ours, max_norm, torch.tensor(total_norm))
                torch.nn.utils.clip_grads_with_norm_(theirs[:2], max_norm, torch.tensor(total_norm))
                for p, q in zip(ours[:2], theirs[:2]):
                    self.assertTrue(torch.equal(p.grad, q.grad))


class TestNonFiniteLoss(unittest.TestCase):

    def test_finite_loss_continues_when_every_rank_is_finite(self):
        device = torch.device("cpu")
        with mock.patch.object(train, "any_rank_failed", return_value=False) as vote:
            train._raise_if_non_finite_loss(torch.tensor(1.0), device)

        vote.assert_called_once_with(False, device=device)

    def test_nan_and_infinities_stop_before_the_update(self):
        device = torch.device("cpu")
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), mock.patch.object(
                train, "any_rank_failed", side_effect=lambda failed, device: failed
            ) as vote, self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
                train._raise_if_non_finite_loss(torch.tensor(value), device)
            vote.assert_called_once_with(True, device=device)

    def test_a_remote_failure_stops_a_rank_with_a_finite_loss(self):
        device = torch.device("cpu")
        with mock.patch.object(train, "any_rank_failed", return_value=True) as vote:
            with self.assertRaisesRegex(FloatingPointError, "stopping all ranks"):
                train._raise_if_non_finite_loss(torch.tensor(1.0), device)

        vote.assert_called_once_with(False, device=device)

    def test_non_finite_loss_leaves_model_optimizer_and_ema_unchanged(self):
        source = inspect.getsource(train.main)
        guard = source.index("_raise_if_non_finite_loss(loss, device)")
        for mutation in ("loss.backward()", "optimizer.step()", "torch._foreach_mul_"):
            self.assertLess(guard, source.index(mutation, guard), mutation)

        torch.manual_seed(0)
        encoder = torch.nn.Linear(2, 1, bias=False)
        target_encoder = copy.deepcopy(encoder)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.1)
        inputs = torch.tensor([[1.0, -2.0]])

        # Populate AdamW's moments and make the online and EMA models different,
        # so mutations to every state below would be observable.
        encoder(inputs).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertFalse(torch.equal(encoder.weight, target_encoder.weight))

        encoder_before = copy.deepcopy(encoder.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        target_before = copy.deepcopy(target_encoder.state_dict())

        non_finite_loss = encoder(inputs).square().mean() + torch.tensor(float("inf"))
        with self.assertRaises(FloatingPointError):
            train._raise_if_non_finite_loss(non_finite_loss, torch.device("cpu"))
            non_finite_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                for online, target in zip(encoder.parameters(), target_encoder.parameters()):
                    target.mul_(0.9).add_(online, alpha=0.1)

        torch.testing.assert_close(encoder.state_dict(), encoder_before, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), optimizer_before, rtol=0, atol=0)
        torch.testing.assert_close(target_encoder.state_dict(), target_before, rtol=0, atol=0)


class TestDataLoaderErrorLogging(unittest.TestCase):

    def test_retry_log_contains_only_the_error_type_and_retry_count(self):
        sensitive_path = "/clinical-storage/patient-identifier/video.dcm"
        message = f"decoder failed for {sensitive_path}"
        error = RuntimeError(message)

        with mock.patch.object(train.logger, "warning") as warning:
            train._log_data_loading_retry(error, retry_number=2, max_retries=5)

        warning.assert_called_once_with(
            "Data loading failed with %s; retrying (%d/%d).",
            "RuntimeError",
            2,
            5,
        )
        logged = warning.call_args.args[0] % warning.call_args.args[1:]
        self.assertNotIn(sensitive_path, logged)
        self.assertNotIn(message, logged)

    def test_exhausted_retries_keep_the_original_exception_as_the_cause(self):
        error = OSError("private storage detail")

        with self.assertRaisesRegex(RuntimeError, r"Exceeded max retries \(5\)") as raised:
            train._raise_data_loading_retries_exhausted(error, max_retries=5)

        self.assertIs(raised.exception.__cause__, error)
        self.assertNotIn(str(error), str(raised.exception))


class TestAtomicCheckpointSave(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name
        self.path = os.path.join(self.directory, "latest.pth.tar")
        with open(self.path, "wb") as checkpoint:
            checkpoint.write(b"previous checkpoint")

    @staticmethod
    def _fail_after_partial_write(_save_dict, temporary_file):
        temporary_file.write(b"partial replacement")
        raise OSError("simulated write failure")

    def test_a_failed_write_preserves_the_previous_checkpoint(self):
        with mock.patch.object(train.torch, "save", side_effect=self._fail_after_partial_write):
            with self.assertRaisesRegex(OSError, "simulated write failure"):
                train._atomic_torch_save({"epoch": 2}, self.path)

        with open(self.path, "rb") as checkpoint:
            self.assertEqual(checkpoint.read(), b"previous checkpoint")

    def test_a_successful_write_atomically_replaces_the_checkpoint(self):
        real_replace = os.replace
        with mock.patch.object(train.os, "replace", wraps=real_replace) as replace:
            train._atomic_torch_save({"epoch": 2}, self.path)

        replace.assert_called_once()
        temporary_path, destination = replace.call_args.args
        self.assertEqual(os.path.dirname(temporary_path), self.directory)
        self.assertEqual(destination, self.path)
        self.assertEqual(torch.load(self.path, weights_only=True)["epoch"], 2)

    def test_a_failed_write_cleans_up_the_temporary_file(self):
        with mock.patch.object(train.torch, "save", side_effect=self._fail_after_partial_write):
            with self.assertRaises(OSError):
                train._atomic_torch_save({"epoch": 2}, self.path)

        self.assertEqual(os.listdir(self.directory), ["latest.pth.tar"])


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(),
    "requires torch.distributed with gloo",
)
class TestDistributedCheckpointSave(unittest.TestCase):

    def test_a_rank_zero_save_failure_stops_every_rank(self):
        world_size = 3
        context = mp.get_context("spawn")
        queue = context.Queue()

        mp.spawn(
            _checkpoint_failure_worker,
            args=(world_size, _free_port(), queue),
            nprocs=world_size,
            join=True,
        )

        results = dict(queue.get(timeout=60) for _ in range(world_size))
        self.assertEqual(sorted(results), list(range(world_size)))
        for rank, message in results.items():
            self.assertEqual(
                message,
                "Checkpoint save failed on rank 0; stopping all ranks.",
                msg=f"rank {rank}",
            )


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
        encoder, predictor, target, opt, _, epoch, _ = self.resume(self.save())
        self.assertEqual(epoch, 5)
        for a, b in zip(encoder.state_dict().values(), self.encoder.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(len(opt.state_dict()["state"]), len(self.opt.state_dict()["state"]))

    def test_a_cooldown_start_resets_the_epoch(self):
        self.assertEqual(self.resume(self.save(), is_anneal=True)[5], 0)

    def test_the_wandb_run_id_comes_back_and_is_none_when_absent(self):
        self.assertEqual(self.resume(self.save(wandb_run_id="abc123"))[-1], "abc123")
        self.assertIsNone(self.resume(self.save())[-1])

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
