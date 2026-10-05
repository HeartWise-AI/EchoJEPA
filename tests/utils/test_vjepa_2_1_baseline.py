# tests/utils/test_vjepa_2_1_baseline.py

"""Tests for the pilot pretraining baseline (docs/pretraining_baseline.md): the shipped
ViT-B configs of the two arms, the checkpoint smoke test, and weights-only
initialization versus resuming a run, and the launcher's exit status."""

import ast
import contextlib
import copy
import inspect
import io
import logging
import os
import re
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import torch
import yaml

from app import main as launcher
from app.vjepa_2_1 import check_checkpoint, train
from app.vjepa_2_1.utils import masked_l1_loss
from src.masks.utils import apply_masks

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "train_2_1", "vitb16"
)
CHECKPOINT_NAME = "vjepa2_1_vitb_mimic_pt169_c60.pt"


def shipped(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def flatten(cfg, prefix=""):
    """{"a": {"b": 1}} -> {"a.b": 1}; lists (the mask blocks) stay whole values."""
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            flat.update(flatten(v, f"{prefix}{k}."))
        else:
            flat[f"{prefix}{k}"] = v
    return flat


def differing(a, b):
    a, b = flatten(a), flatten(b)
    return {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}


def tiny_config(folder, init_checkpoint=None, **model):
    """The shipped continued-arm config shrunk to a vit_tiny on 4-frame 64 px clips: 2 x 4 x 4
    tokens, enough room for the shipped masks to leave context tokens."""
    cfg = shipped("pretrain-continued-224px-16f.yaml")
    cfg["folder"] = folder
    cfg["meta"]["init_checkpoint"] = init_checkpoint
    cfg["model"].update(model_name="vit_tiny", pred_depth=4, pred_embed_dim=96, use_activation_checkpointing=False)
    cfg["model"].update(model)
    cfg["data"].update(crop_size=64, dataset_fpcs=[4])
    return cfg


class Stop(Exception):
    """Raised by a patched step to end train.main once the behaviour under test has run."""


class QuietLogs(unittest.TestCase):
    """init_video_model prints every model it builds, and the trainer its settings."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.INFO)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)


@contextlib.contextmanager
def trainer_on_cpu():
    """Run train.main in this process without a GPU and without changing the global
    multiprocessing start method."""
    with mock.patch("torch.cuda.is_available", return_value=False), mock.patch.object(train.mp, "set_start_method"):
        yield


class TestShippedConfigs(unittest.TestCase):

    ARMS = ("continued", "scratch")
    PHASES = ("pretrain", "cooldown")

    def config(self, phase, arm):
        return shipped(f"{phase}-{arm}-224px-16f.yaml")

    def every_config(self):
        return [(f"{p}-{a}", self.config(p, a)) for p in self.PHASES for a in self.ARMS]

    def test_pretraining_arms_differ_only_in_initialisation_and_output_folder(self):
        self.assertEqual(
            differing(self.config("pretrain", "continued"), self.config("pretrain", "scratch")),
            {"folder", "meta.init_checkpoint"},
        )

    def test_cooldown_arms_differ_only_in_their_source_run_and_output_folder(self):
        self.assertEqual(
            differing(self.config("cooldown", "continued"), self.config("cooldown", "scratch")),
            {"folder", "optimization.anneal_ckpt"},
        )

    def test_each_cooldown_continues_its_own_arm(self):
        for arm in self.ARMS:
            with self.subTest(arm=arm):
                pretrain, cooldown = self.config("pretrain", arm), self.config("cooldown", arm)
                self.assertEqual(
                    cooldown["optimization"]["anneal_ckpt"], os.path.join(pretrain["folder"], "latest.pth.tar")
                )
                self.assertNotEqual(cooldown["folder"], pretrain["folder"])

    def test_cooldown_keeps_the_data_masks_and_model_of_pretraining(self):
        ramp = {"lambda_progressive", "lambda_start_iter", "lambda_end_iter"}
        for arm in self.ARMS:
            with self.subTest(arm=arm):
                pretrain, cooldown = self.config("pretrain", arm), self.config("cooldown", arm)
                for section in ("data", "data_aug", "mask"):
                    self.assertEqual(cooldown[section], pretrain[section], section)
                strip = lambda model: {k: v for k, v in model.items() if k not in ramp}  # noqa: E731
                self.assertEqual(strip(cooldown["model"]), strip(pretrain["model"]))
                self.assertEqual(cooldown["meta"]["seed"], pretrain["meta"]["seed"])

    def test_clips_are_16_frames_at_8_fps(self):
        for name, cfg in self.every_config():
            with self.subTest(config=name):
                self.assertEqual(cfg["data"]["dataset_fpcs"], [16])
                self.assertEqual(cfg["data"]["fps"], 8)

    def test_train_csv_is_the_only_training_data(self):
        for name, cfg in self.every_config():
            with self.subTest(config=name):
                self.assertEqual(len(cfg["data"]["datasets"]), 1)
                self.assertEqual(os.path.basename(cfg["data"]["datasets"][0]), "train.csv")
                self.assertNotIn("datasets_weights", cfg["data"])

    def test_only_the_continued_arm_starts_from_the_echojepa_checkpoint(self):
        continued = self.config("pretrain", "continued")["meta"]["init_checkpoint"]
        self.assertEqual(os.path.basename(continued), CHECKPOINT_NAME)
        self.assertIsNone(self.config("pretrain", "scratch")["meta"]["init_checkpoint"])
        for name, cfg in self.every_config():
            with self.subTest(config=name):
                self.assertIsNone(cfg["meta"]["read_checkpoint"])
                if name.startswith("cooldown"):
                    self.assertIsNone(cfg["meta"]["init_checkpoint"])

    def test_model_matches_the_published_vitb_checkpoint(self):
        # Shapes of vjepa2_1_vitb_mimic_pt169_c60.pt; see docs/pretraining_baseline.md.
        expected_model = dict(
            model_name="vit_base", pred_depth=12, pred_embed_dim=384, pred_num_heads=12, n_output_distillation=1,
            use_rope=True, modality_embedding=True, img_temporal_dim_size=1, use_mask_tokens=True, n_registers=0,
            n_registers_predictor=0, has_cls_first=False,
        )
        for name, cfg in self.every_config():
            with self.subTest(config=name):
                self.assertEqual({k: cfg["model"][k] for k in expected_model}, expected_model)
                self.assertEqual((cfg["data"]["patch_size"], cfg["data"]["tubelet_size"]), (16, 2))
                self.assertEqual(len(cfg["mask"]), 2)  # 2 mask tokens in the predictor

    def test_no_input_clips_are_uploaded(self):
        # #13: visual examples are reviewed and anonymized before any upload.
        for name, cfg in self.every_config():
            with self.subTest(config=name):
                self.assertEqual(cfg["meta"]["num_log_videos"], 0)

    def test_paths_are_placeholders(self):
        for name, cfg in self.every_config():
            for key, value in flatten(cfg).items():
                values = value if isinstance(value, list) else [value]
                for v in values:
                    if isinstance(v, str) and v.startswith("/"):
                        with self.subTest(config=name, key=key):
                            self.assertTrue(v.startswith("/your_"), v)


class TestCheckCheckpoint(QuietLogs):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.cfg = tiny_config(self.tmp)
        encoder, predictor, target_encoder = check_checkpoint.build_models(self.cfg)
        prefixed = lambda model: {f"module.{k}": v for k, v in model.state_dict().items()}  # noqa: E731
        self.checkpoint = {
            "encoder": prefixed(encoder),
            "predictor": prefixed(predictor),
            "target_encoder": prefixed(target_encoder),
            "opt": {"state": {}, "param_groups": [{"lr": 1e-6}]},
            "scaler": None,
            "epoch": 60,
        }
        self.path = self.save(self.checkpoint)

    def save(self, checkpoint, name="source.pt"):
        path = os.path.join(self.tmp, name)
        torch.save(checkpoint, path)
        return path

    def test_a_matching_checkpoint_passes(self):
        report = check_checkpoint.check(
            self.cfg, self.path, expected_sha256=check_checkpoint.sha256sum(self.path), forward=True
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["strict_load"], "ok")
        for name, m in report["models"].items():
            self.assertEqual((m["missing"], m["unexpected"], m["mismatched"]), ([], [], []), name)
        self.assertEqual(report["models"]["target_encoder"]["source"], "target_encoder")
        forward = report["forward"]
        self.assertTrue(forward["finite"])
        self.assertEqual(len(forward["tokens"]), 2)  # one (context, predicted) pair per mask
        for context, predicted in forward["tokens"]:
            self.assertGreater(context, 0)
            self.assertLessEqual(context + predicted, 2 * 4 * 4)  # trimmed to the batch's shortest mask
        self.assertIsNotNone(forward["loss"]["context"])
        self.assertEqual(report["training_state"], {"epoch": 60, "opt_lr": 1e-6})

    def test_a_wrong_checksum_fails_before_the_file_is_loaded(self):
        with mock.patch.object(check_checkpoint, "robust_checkpoint_loader") as load, \
                mock.patch.object(check_checkpoint, "load_pretrained_weights") as init_weights:
            report = check_checkpoint.check(self.cfg, self.path, expected_sha256="0" * 64, forward=True)
        load.assert_not_called()
        init_weights.assert_not_called()
        self.assertFalse(report["ok"])
        self.assertNotIn("forward", report)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            check_checkpoint.print_report(report)
        self.assertIn("Not loaded", out.getvalue())
        self.assertIn("RESULT: FAIL", out.getvalue())

    def forward(self, cfg=None):
        cfg = cfg or self.cfg
        return check_checkpoint.masked_forward(cfg, *check_checkpoint.build_models(cfg))

    def test_the_forward_pass_is_repeatable(self):
        models = check_checkpoint.build_models(self.cfg)
        torch.manual_seed(1)
        first = check_checkpoint.masked_forward(self.cfg, *models)
        torch.manual_seed(2)  # the caller's random state must not matter
        state = torch.get_rng_state()
        self.assertEqual(check_checkpoint.masked_forward(self.cfg, *models), first)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))  # and is left as it was

    def test_the_forward_loss_adds_the_weighted_context_loss(self):
        loss = self.forward()["loss"]
        self.assertEqual(loss["context_weight"], self.cfg["model"]["lambda_value_vid"])
        self.assertAlmostEqual(loss["total"], loss["prediction"] + loss["context_weight"] * loss["context"], places=5)

    def test_distance_weighting_follows_the_config(self):
        for weighted in (True, False):
            with self.subTest(weight_distance_loss=weighted):
                cfg = copy.deepcopy(self.cfg)
                cfg["loss"]["weight_distance_loss"] = weighted
                with mock.patch.object(
                    check_checkpoint, "compute_mask_distance", wraps=check_checkpoint.compute_mask_distance
                ) as distance:
                    loss = self.forward(cfg)["loss"]
                self.assertEqual(loss["distance_weighted"], weighted)
                self.assertEqual(distance.call_count, int(weighted))
                if weighted:
                    grid_size, offset = distance.call_args.args[2:]
                    self.assertEqual((grid_size, offset), (64 // 16, False))

    def test_differences_are_listed_per_model(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        dropped = sorted(checkpoint["predictor"])[0]
        del checkpoint["predictor"][dropped]
        checkpoint["encoder"]["module.backbone.extra"] = torch.zeros(1)
        reshaped = "module.backbone.norms_block.0.weight"
        checkpoint["target_encoder"][reshaped] = torch.zeros(7)
        report = check_checkpoint.check(self.cfg, self.save(checkpoint, "tampered.pt"))
        self.assertFalse(report["ok"])
        self.assertIsNone(report["strict_load"])
        self.assertEqual(report["models"]["predictor"]["missing"], [dropped.removeprefix("module.")])
        self.assertEqual(report["models"]["encoder"]["unexpected"], ["backbone.extra"])
        self.assertEqual(
            report["models"]["target_encoder"]["mismatched"], [("backbone.norms_block.0.weight", (7,), (192,))]
        )

    def test_ema_encoder_stands_in_for_a_missing_target_encoder(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["ema_encoder"] = checkpoint.pop("target_encoder")
        report = check_checkpoint.check(self.cfg, self.save(checkpoint, "ema.pt"))
        self.assertTrue(report["ok"])
        self.assertEqual(report["models"]["target_encoder"]["source"], "ema_encoder")

    def test_a_forward_pass_that_fails_fails_the_check(self):
        with mock.patch.object(check_checkpoint, "masked_forward", side_effect=RuntimeError("shape mismatch")):
            report = check_checkpoint.check(self.cfg, self.path, forward=True)
        self.assertEqual(report["strict_load"], "ok")
        self.assertFalse(report["ok"])

    def test_a_four_level_config_does_not_fit_a_one_level_checkpoint(self):
        report = check_checkpoint.check(tiny_config(self.tmp, n_output_distillation=4), self.path)
        self.assertFalse(report["ok"])
        self.assertTrue(report["models"]["predictor"]["mismatched"])

    def test_exit_status(self):
        fname = os.path.join(self.tmp, "config.yaml")
        with open(fname, "w") as f:
            yaml.safe_dump(self.cfg, f)
        sha = check_checkpoint.sha256sum(self.path)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check_checkpoint.main(["--fname", fname, "--checkpoint", self.path, "--sha256", sha]), 0)
            self.assertEqual(check_checkpoint.main(["--fname", fname, "--checkpoint", self.path, "--sha256", "0"]), 1)
        self.assertIn("RESULT: PASS", out.getvalue())
        self.assertIn("RESULT: FAIL", out.getvalue())

    def test_builds_the_model_the_trainer_builds(self):
        cfg = shipped("pretrain-continued-224px-16f.yaml")
        cfg["folder"], cfg["meta"]["init_checkpoint"] = self.tmp, None
        with trainer_on_cpu(), mock.patch.object(train, "init_video_model", side_effect=Stop) as init_video_model:
            with self.assertRaises(Stop):
                train.main(copy.deepcopy(cfg))
        trainer_kwargs = dict(init_video_model.call_args.kwargs)
        trainer_kwargs.pop("device")
        self.assertEqual(trainer_kwargs, check_checkpoint.model_kwargs(cfg))


class TestLossMatchesTheTrainer(unittest.TestCase):
    """check_checkpoint.loss_fn is a copy of the `loss_fn` closure in train.py's training
    step; run both on the same tensors, in each of its branches, with and without padding."""

    def trainer_loss_fn(self, loss_exp):
        source = inspect.getsource(train)
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "loss_fn")
        namespace = {"torch": torch, "apply_masks": apply_masks, "masked_l1_loss": masked_l1_loss,
                     "loss_exp": loss_exp}
        exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
        return namespace["loss_fn"]

    def test_every_branch_matches(self):
        g = torch.Generator().manual_seed(0)
        batch, tokens, width, sizes = 2, 12, 8, (5, 7)
        masks = [[torch.stack([torch.randperm(tokens, generator=g)[:n] for _ in range(batch)]) for n in sizes]]
        z = [[torch.randn(batch, n, width, generator=g) for n in sizes]]
        z_cls = [[torch.randn(batch, n + 1, width, generator=g) for n in sizes]]
        h = [torch.randn(batch, tokens, width, generator=g)]
        h_cls = [torch.randn(batch, tokens + 1, width, generator=g)]
        d_weights = [[1 + torch.rand(batch, n, generator=g) for n in sizes]]
        real = [torch.ones(batch, tokens, 1)]
        padded = [torch.ones(batch, tokens, 1)]
        padded[0][1, tokens // 2:] = 0  # the second half of the second clip is padding
        for loss_exp in (1.0, 2.0):
            trainer = self.trainer_loss_fn(loss_exp)
            for name, args in (
                ("plain", (z, h, masks, False, None, real)),
                ("distance-weighted", (z, h, masks, False, d_weights, real)),
                ("class token first", (z_cls, h_cls, masks, True, None, real)),
                ("plain, padded", (z, h, masks, False, None, padded)),
                ("distance-weighted, padded", (z, h, masks, False, d_weights, padded)),
            ):
                with self.subTest(branch=name, loss_exp=loss_exp):
                    zi, hi, mi, cls_loss, d, v = args
                    self.assertTrue(torch.equal(
                        check_checkpoint.loss_fn(zi, hi, mi, loss_exp, cls_loss=cls_loss, d_weights=d, valid=v),
                        trainer(zi, hi, mi, cls_loss=cls_loss, d_weights=d, valid=v),
                    ))


class TestInitialisationIsNotAResume(QuietLogs):
    """meta.init_checkpoint copies weights only; the run's own latest.pth.tar is the one
    and only thing a restart resumes (optimizer, scaler, epoch and schedules)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name
        self.published = os.path.join(self.folder, "published.pt")

    def run_until_data(self, cfg):
        """Run train.main up to the data loader; return the mock of the weights-only loader."""
        with trainer_on_cpu(), mock.patch.object(train, "load_pretrained_weights") as init_weights, \
                mock.patch.object(train, "init_data", side_effect=Stop):
            with self.assertRaises(Stop):
                train.main(cfg)
        return init_weights

    def test_a_new_run_takes_weights_from_init_checkpoint(self):
        init_weights = self.run_until_data(tiny_config(self.folder, init_checkpoint=self.published))
        init_weights.assert_called_once()
        self.assertEqual(init_weights.call_args.args[0], self.published)

    def test_a_restarted_run_resumes_its_own_checkpoint_instead(self):
        open(os.path.join(self.folder, "latest.pth.tar"), "w").close()
        init_weights = self.run_until_data(tiny_config(self.folder, init_checkpoint=self.published))
        init_weights.assert_not_called()

    def test_the_scratch_arm_loads_no_weights(self):
        init_weights = self.run_until_data(tiny_config(self.folder))
        init_weights.assert_not_called()

    def test_init_checkpoint_and_read_checkpoint_together_are_refused(self):
        cfg = tiny_config(self.folder, init_checkpoint=self.published)
        cfg["meta"]["read_checkpoint"] = self.published
        with self.assertRaisesRegex(ValueError, "init_checkpoint.*read_checkpoint.*not both"):
            train.main(cfg)


def rank_exits(rank, fname, world_size, devices):
    """Stands in for a training rank; `fname` names the rank that fails."""
    sys.exit(1 if str(rank) == fname else 0)


def rank_fails_or_hangs(rank, fname, world_size, devices):
    """Rank 0 fails at once; the others would wait forever on it."""
    if rank == 0:
        sys.exit(1)
    time.sleep(600)


class TestLauncher(unittest.TestCase):
    """python -m app.main exits non-zero when a rank fails, so a failed run stops the next one."""

    def test_clean_ranks_report_no_failure(self):
        self.assertEqual(launcher.launch("none", ["cuda:0", "cuda:1"], target=rank_exits), [])

    def test_a_failed_rank_is_reported(self):
        self.assertEqual(launcher.launch("1", ["cuda:0", "cuda:1", "cuda:2"], target=rank_exits), [1])

    def test_a_failed_rank_stops_the_others(self):
        start = time.monotonic()
        failed = launcher.launch("unused", ["cuda:0", "cuda:1"], target=rank_fails_or_hangs)
        self.assertEqual(failed, [0])  # rank 1 was stopped, not failed
        self.assertLess(time.monotonic() - start, 60)


class TestPilotEnvironment(unittest.TestCase):
    """requirements-pilot.txt pins the pilot's environment, which the launch script checks."""

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CONFIG_DIR)))  # configs/train_2_1/vitb16

    def lines(self, name):
        with open(os.path.join(self.ROOT, name)) as f:
            return [line.strip() for line in f]

    def pins(self):
        return [line for line in self.lines("requirements-pilot.txt") if line and not line.startswith("#")]

    @staticmethod
    def normalise(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    def test_python_and_every_package_are_pinned_exactly(self):
        self.assertIn("# python: 3.12", self.lines("requirements-pilot.txt"))
        for line in self.pins():
            with self.subTest(line=line):
                # name==version only: no ranges, URLs or local paths
                self.assertRegex(line, r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+!_-]+$")

    def test_every_declared_requirement_is_pinned(self):
        pinned = {self.normalise(line.split("==")[0]) for line in self.pins()}
        for line in self.lines("requirements.txt"):
            if line and not line.startswith("#"):
                name = self.normalise(re.split(r"[<>=!~\[; ]", line)[0])
                with self.subTest(requirement=name):
                    self.assertIn(name, pinned)


if __name__ == "__main__":
    unittest.main()
