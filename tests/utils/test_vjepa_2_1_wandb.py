# tests/utils/test_vjepa_2_1_wandb.py

"""The V-JEPA 2.1 trainer's wandb logging, on CPU with a fake wandb: what one short run logs,
that private environment capture is disabled, that a restart reattaches to its run, and that a
cooldown starting from `anneal_ckpt` opens a new one."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from app.vjepa_2_1 import train
from src.utils import wandb_logging
from tests.utils.test_vjepa_2_1_baseline import QuietLogs, shipped, tiny_config, trainer_on_cpu

FRAMES, CROP = 4, 64
REAL = np.arange(FRAMES)
PADDED = np.array([0, 1, -1, -1])  # a 2-frame video: its second tubelet is all padding


class FakeRun(SimpleNamespace):
    def define_metric(self, *args, **kwargs):
        pass

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        self.finished = True


class FakeWandb:
    """Records `init` calls and hands out runs that keep what they log."""

    def __init__(self):
        self.calls, self.runs = [], []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        run = FakeRun(id=kwargs["id"] or f"run-{len(self.calls)}", entity=kwargs["entity"],
                      project=kwargs["project"], url="https://example.invalid/run", logged=[], finished=False)
        self.runs.append(run)
        return run

    @staticmethod
    def Video(frames, fps, format):
        return ("video", frames.shape)


class Clips(torch.utils.data.Dataset):
    """Pairs of clips as VideoDataset returns them: one full, one padded at the end."""

    def __len__(self):
        return 4

    def __getitem__(self, i):
        indices = PADDED if i % 2 else REAL
        return [torch.randn(3, FRAMES, CROP, CROP)], 0, [indices]


def fake_init_data(batch_size, collator, **_):
    loader = torch.utils.data.DataLoader(Clips(), batch_size=batch_size, collate_fn=collator, drop_last=True)
    return loader, SimpleNamespace(set_epoch=lambda epoch: None)


def config(folder, epochs=1, wandb=True):
    cfg = tiny_config(folder)
    cfg["meta"].update(dtype="float32", save_every_freq=-1)
    if wandb:
        cfg["meta"].update(wandb_project="project", wandb_entity="entity")
    cfg["data"].update(batch_size=2, num_workers=0)
    cfg["optimization"].update(ipe=2, epochs=epochs, warmup=0)
    return cfg


def cooldown_config(folder, anneal_ckpt):
    cfg = shipped("cooldown-continued-224px-16f.yaml")
    small = config(folder)
    for block in ("meta", "data", "model"):
        cfg[block].update(small[block])
    cfg["folder"] = folder
    cfg["optimization"].update(ipe=2, epochs=1, warmup=0, anneal_ckpt=anneal_ckpt)
    return cfg


class TestTrainerWandb(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # DDP needs a process group: one gloo rank, so every collective is local.
        cls.store = tempfile.TemporaryDirectory()
        dist.init_process_group("gloo", init_method=f"file://{cls.store.name}/store", rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.store.cleanup()
        super().tearDownClass()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.wandb = FakeWandb()

    def run_trainer(self, cfg):
        os.makedirs(cfg["folder"], exist_ok=True)  # as the launcher does
        # The trainer's console line reads CUDA memory even on CPU, which fails once another test
        # has initialised CUDA.
        with trainer_on_cpu(), mock.patch.object(train, "init_data", side_effect=fake_init_data), \
                mock.patch.object(wandb_logging, "wandb", self.wandb), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0):
            train.main(cfg)
        return self.wandb.runs[-1] if self.wandb.runs else None

    def test_the_run_config_holds_no_path(self):
        cfg = config(os.path.join(self.tmp, "pretrain"))
        self.run_trainer(cfg)
        sent = self.wandb.calls[0]["config"]
        self.assertEqual(sent["folder"], "pretrain")  # its folder name, not where it is

        def strings(value):
            if isinstance(value, dict):
                return [s for v in value.values() for s in strings(v)]
            if isinstance(value, list):
                return [s for v in value for s in strings(v)]
            return [value] if isinstance(value, str) else []

        self.assertFalse([s for s in strings(sent) if "/" in s or self.tmp in s])

    def test_the_run_disables_private_environment_capture(self):
        self.run_trainer(config(os.path.join(self.tmp, "pretrain")))
        self.assertEqual(self.wandb.calls[0]["settings"], wandb_logging.PRIVATE_WANDB_SETTINGS)

    def test_the_shipped_configs_upload_no_input_clips(self):
        run = self.run_trainer(config(os.path.join(self.tmp, "pretrain")))  # `num_log_videos: 0`, as shipped
        self.assertTrue(run.logged)
        self.assertFalse([k for p in run.logged for k in p if k.startswith("train/input_clips")])

    def test_a_run_logs_its_inputs_losses_and_padding(self):
        folder = os.path.join(self.tmp, "pretrain")
        cfg = config(folder)
        cfg["meta"]["num_log_videos"] = 2  # input clips on, as the trainer's default
        run = self.run_trainer(cfg)
        self.assertEqual(self.wandb.calls[0]["resume"], "never")
        self.assertTrue(run.finished)

        clips = [p for p in run.logged if "train/input_clips/0" in p]
        self.assertEqual(len(clips), 1)  # the first step of the job only
        self.assertEqual(clips[0]["train/global_step"], 0)
        self.assertAlmostEqual(clips[0]["data/padded_frame_frac"], 0.25)  # 2 of the 8 frames

        scalars = [p for p in run.logged if "train/loss" in p]
        self.assertEqual([p["train/global_step"] for p in scalars], [0, 1])  # step 0 and the epoch's last
        for p in scalars:
            for key in ("train/loss", "train/loss_pred", "train/loss_context", "optimization/grad_norm"):
                self.assertTrue(np.isfinite(p[key]), key)
            # One clip of each pair is half padding: 3 of 4 tubelets count, 2 of 8 frames are black.
            self.assertAlmostEqual(p["data/valid_token_frac"], 0.75)
            self.assertAlmostEqual(p["data/batch_padded_frame_frac"], 0.25)
            self.assertEqual(p["train/lambda_context"], 0.0)  # before the context-loss ramp
            # Below the ramp, the step's loss is the prediction loss alone.
            self.assertAlmostEqual(p["train/loss"], p["train/loss_pred"], places=5)
            # The collapse check over the batch's 2 clips: at most 1 centred direction.
            self.assertTrue(-1.0 <= p["features/clip_cosine"] <= 1.0)
            self.assertTrue(0.0 <= p["features/effective_rank"] <= 1.0 + 1e-5)
            # Logged every step here, so the largest norm since the last log is this step's.
            self.assertEqual(p["optimization/grad_norm_max"], p["optimization/grad_norm"])
            self.assertNotIn("optimization/clipped_steps", p)  # clipping is off

        saved = torch.load(os.path.join(folder, "latest.pth.tar"), map_location="cpu", weights_only=False)
        self.assertEqual(saved["wandb_run_id"], run.id)

    def test_a_restart_reattaches_to_its_run(self):
        folder = os.path.join(self.tmp, "pretrain")
        first = self.run_trainer(config(folder, epochs=1))
        second = self.run_trainer(config(folder, epochs=2))
        self.assertEqual(self.wandb.calls[1]["id"], first.id)
        self.assertEqual(self.wandb.calls[1]["resume"], "must")
        # The restarted job logs its own first step (epoch 2) onwards.
        steps = [p["train/global_step"] for p in second.logged if "train/loss" in p]
        self.assertEqual(steps, [2, 3])

    def test_a_cooldown_from_anneal_ckpt_is_a_new_run(self):
        pretrain = os.path.join(self.tmp, "pretrain")
        first = self.run_trainer(config(pretrain))
        cooldown = self.run_trainer(cooldown_config(os.path.join(self.tmp, "cooldown"),
                                                    os.path.join(pretrain, "latest.pth.tar")))
        self.assertIsNone(self.wandb.calls[1]["id"])
        self.assertEqual(self.wandb.calls[1]["resume"], "never")
        self.assertNotEqual(cooldown.id, first.id)

    def test_gradient_spikes_between_logs_are_seen_and_clipped(self):
        folder = os.path.join(self.tmp, "pretrain")
        cfg = config(folder)
        cfg["optimization"].update(ipe=4, clip_grad=2.5)
        norms = [1.0, 5.0, 2.0, 3.0]  # the gradient norm of steps 0 to 3
        with mock.patch.object(train, "log_freq", 2), \
                mock.patch.object(train, "grad_norm", side_effect=[torch.tensor(n) for n in norms]), \
                mock.patch.object(train, "clip_grads_with_norm_", wraps=train.clip_grads_with_norm_) as clipped:
            run = self.run_trainer(cfg)
        # Every step is clipped against its own norm.
        self.assertEqual([(c.args[1], float(c.args[2])) for c in clipped.call_args_list],
                         [(2.5, n) for n in norms])
        logged = {p["train/global_step"]: p for p in run.logged if "train/loss" in p}
        self.assertEqual(sorted(logged), [0, 2, 3])  # every 2 steps and the epoch's last
        # Step 2's log covers steps 1 and 2: the spike at step 1 shows though it was not logged.
        self.assertEqual([logged[s]["optimization/grad_norm"] for s in (0, 2, 3)], [1.0, 2.0, 3.0])
        self.assertEqual([logged[s]["optimization/grad_norm_max"] for s in (0, 2, 3)], [1.0, 5.0, 3.0])
        self.assertEqual([logged[s]["optimization/clipped_steps"] for s in (0, 2, 3)], [0, 1, 1])

    def test_clip_grad_must_be_positive(self):
        cfg = config(os.path.join(self.tmp, "pretrain"))
        cfg["optimization"]["clip_grad"] = 0
        with self.assertRaisesRegex(ValueError, "clip_grad"):
            self.run_trainer(cfg)

    def test_without_a_project_nothing_is_logged(self):
        folder = os.path.join(self.tmp, "pretrain")
        self.assertIsNone(self.run_trainer(config(folder, wandb=False)))
        self.assertEqual(self.wandb.calls, [])
        saved = torch.load(os.path.join(folder, "latest.pth.tar"), map_location="cpu", weights_only=False)
        self.assertIsNone(saved["wandb_run_id"])


if __name__ == "__main__":
    unittest.main()
