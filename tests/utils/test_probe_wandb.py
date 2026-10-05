# tests/utils/test_probe_wandb.py

"""The frozen probe's wandb logging (`evals/video_classification_frozen/eval.py`), on CPU with a
tiny V-JEPA 2.1 encoder, synthetic clips and a fake wandb: what each epoch logs, what the run's
config holds (and that it holds no filesystem path), that a resumed probe reattaches to its run,
that nothing is logged without a project, and the test's plots."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from app.vjepa_2_1.utils import init_video_model
from evals.video_classification_frozen import eval as probe
from evals.video_classification_frozen import provenance
from src.utils import wandb_logging
from tests.utils.test_vjepa_2_1_eval_encoder import OPTIONS
from tests.utils.test_vjepa_2_1_utils import QuietLogs
from tests.utils.test_vjepa_2_1_wandb import FakeRun, FakeWandb

FRAMES, CROP, BATCHES = 4, 32, 2
VIDEOS = {"train": 6, "val": 5, "test": 7}  # odd counts: the last batch of 2 is short
STUDY_Z = (-2.0, 0.0, 1.0)  # z-scored EF of studies 0, 1, 2, ...: EF 35, 55, 65 at mean 55, std 10
GRID = [
    {"start_lr": 1e-4, "lr": 1e-4, "final_lr": 0.0, "warmup": 0.0, "weight_decay": 0.01, "final_weight_decay": 0.01},
    {"start_lr": 5e-5, "lr": 5e-5, "final_lr": 0.0, "warmup": 0.0, "weight_decay": 0.4, "final_weight_decay": 0.4},
]


class FakeVideos:
    """A split as `VideoDataset` reads it: one clip per video, two videos per study sharing the
    study's z-scored EF, and an unreadable video where `broken` says."""

    def __init__(self, split, n=None, broken=()):
        self.split, self.broken = split, set(broken)
        self.samples = [f"{split}_{i}.mp4" for i in range(VIDEOS[split] if n is None else n)]
        self.labels = [STUDY_Z[(i // 2) % len(STUDY_Z)] for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def get_item_video(self, index):
        if index in self.broken:
            return None
        g = torch.Generator().manual_seed(index + 1000 * (self.split == "test"))
        clip = torch.randn(3, FRAMES, CROP, CROP, generator=g)
        return [[clip]], self.labels[index], [torch.arange(FRAMES)]


def video_index(path):
    """The video index of the fake splits: study i // 2, one patient per study."""
    rows = [(split, f"{split}_{i}.mp4", f"{split}-s{i // 2}", f"{split}-p{i // 2}")
            for split, n in VIDEOS.items() for i in range(n)]
    pd.DataFrame(rows, columns=provenance.INDEX_COLUMNS).to_csv(path, index=False)


class Interrupted(Exception):
    """Stops a probe between two epochs, as a killed job would."""


def stop_after(epochs, run_one_epoch):
    """`run_one_epoch`, raising `Interrupted` when training epoch `epochs + 1` would start."""
    started = []

    def wrapped(**kwargs):
        if kwargs.get("training"):
            if len(started) == epochs:
                raise Interrupted
            started.append(True)
        return run_one_epoch(**kwargs)

    return wrapped


class FakeLoader(list):
    """Training batches as the eval's loader yields them (clips [segment][view], z-scored
    labels, clip indices), with the split it reads as `dataset`."""

    def __init__(self, batches, dataset=None):
        super().__init__(batches)
        self.dataset = dataset


def fake_make_dataloader(root_path, batch_size, splits=None, **_):
    """The eval's `make_dataloader`: random training batches; val and test as `FakeVideos`
    (`splits` replaces them)."""
    g = torch.Generator().manual_seed(0)
    batches = [
        ([[torch.randn(batch_size, 3, FRAMES, CROP, CROP, generator=g)]], torch.randn(batch_size, generator=g),
         [torch.arange(FRAMES).expand(batch_size, FRAMES)])
        for _ in range(BATCHES)
    ]
    split = os.path.splitext(os.path.basename(root_path[0]))[0]
    dataset = (splits or {}).get(split) or (FakeVideos(split) if split in VIDEOS else None)
    return FakeLoader(batches, dataset), SimpleNamespace(set_epoch=lambda epoch: None)


class ProbeFakeRun(FakeRun):
    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class ProbeFakeWandb(FakeWandb):
    """A fake wandb with the objects the probe logs: runs with a summary and artifacts, tables,
    plots and histograms."""

    def init(self, **kwargs):
        run = ProbeFakeRun(**vars(super().init(**kwargs)), summary={}, artifacts=[])
        self.runs[-1] = run
        return run

    class Artifact:
        def __init__(self, name, type, metadata=None):
            self.name, self.type, self.metadata, self.files = name, type, metadata, {}

        @contextlib.contextmanager
        def new_file(self, name):
            buffer = io.StringIO()
            yield buffer
            self.files[name] = buffer.getvalue()

    class Table:
        # wandb keeps a Table in an artifact whose manifest records a local path: the probe logs none.
        def __init__(self, *args, **kwargs):
            raise AssertionError("The probe must not log a wandb.Table.")

    class Image:
        def __init__(self, data, caption=None):
            self.data, self.caption = np.asarray(data), caption

    @staticmethod
    def Histogram(values):
        return ("histogram", list(values))


class TestTestPlots(unittest.TestCase):

    def test_the_scatter_plot_draws_each_pair_where_it_belongs(self):
        image = wandb_logging.scatter_image([20.0], [80.0])
        self.assertEqual((image.shape, image.dtype), ((480, 480, 3), np.uint8))
        left, right, top, bottom = 44, 468, 12, 444  # the plot area of a 480-pixel image
        x, y = round(left + 0.2 * (right - left)), round(bottom - 0.8 * (bottom - top))
        r, g, b = image[y, x].astype(int)
        self.assertGreater(b, r)  # the point, in blue
        x, y = round(left + 0.7 * (right - left)), round(bottom - 0.3 * (bottom - top))
        self.assertTrue((image[y, x] == 255).all())  # no point (and no grid line) there

    def test_range_labels_become_wandb_keys(self):
        self.assertEqual([wandb_logging.range_key(r) for r in ("< 30", "30-40", ">= 60")],
                         ["below_30", "30_to_40", "60_and_above"])


class TestProbeWandb(QuietLogs):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        encoder, _ = init_video_model(
            torch.device("cpu"), model_name="vit_tiny", crop_size=CROP, max_num_frames=FRAMES, pred_depth=4,
            pred_embed_dim=96, use_mask_tokens=True, return_all_tokens=True, **OPTIONS,
        )
        self.checkpoint = os.path.join(self.tmp, "cooldown.pth.tar")
        torch.save({"target_encoder": {f"module.{k}": v for k, v in encoder.state_dict().items()}}, self.checkpoint)
        self.index = os.path.join(self.tmp, "video_index.csv")
        video_index(self.index)
        # The manifests, as files: the probe records their checksums.
        self.manifests = {}
        for split in VIDEOS:
            videos = FakeVideos(split)
            self.manifests[split] = os.path.join(self.tmp, f"{split}.csv")
            with open(self.manifests[split], "w") as f:
                f.writelines(f"{path} {label}\n" for path, label in zip(videos.samples, videos.labels))
        self.wandb = ProbeFakeWandb()

    def config(self, epochs, wandb=True, index=True):
        cfg = {
            "folder": os.path.join(self.tmp, "run"),
            "tag": "lvef",
            "num_workers": 0,
            "resume_checkpoint": True,
            "meta": {"seed": 0, **({"wandb_project": "project", "wandb_entity": "entity"} if wandb else {})},
            "experiment": {
                "classifier": {"task_type": "regression", "num_heads": 4, "num_probe_blocks": 1, "num_targets": 1},
                "data": {"dataset_train": self.manifests["train"], "dataset_val": self.manifests["val"],
                         "dataset_test": self.manifests["test"], "resolution": CROP,
                         "frames_per_clip": FRAMES, "frame_step": 2, "num_segments": 1,
                         "num_views_per_segment": 1, "target_mean": 55.0, "target_std": 10.0},
                "optimization": {"batch_size": 2, "num_epochs": epochs, "use_bfloat16": False,
                                 "multihead_kwargs": GRID},
            },
            "model_kwargs": {
                "checkpoint": self.checkpoint,
                "module_name": "evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder",
                "pretrain_kwargs": {
                    "encoder": {"model_name": "vit_tiny", "checkpoint_key": "target_encoder", **OPTIONS},
                },
                "wrapper_kwargs": {},
            },
        }
        if index:
            cfg["experiment"]["data"]["video_index"] = self.index
            cfg["experiment"]["evaluation"] = {"low_ef_below": 40, "targets": {"level": "study", "mae_below": 5.0,
                                                                                 "auroc_above": 0.95}}
        return cfg

    def run_probe(self, cfg, splits=None):
        def make_dataloader(**kwargs):
            return fake_make_dataloader(splits=splits, **kwargs)

        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0), \
                mock.patch.object(probe.mp, "set_start_method"), \
                mock.patch.object(probe, "make_dataloader", side_effect=make_dataloader), \
                mock.patch.object(wandb_logging, "wandb", self.wandb):
            probe.main(cfg)
        return self.wandb.runs[-1] if self.wandb.runs else None

    def test_each_epoch_logs_the_best_head_and_every_head(self):
        run = self.run_probe(self.config(epochs=2))
        self.assertEqual(self.wandb.calls[0]["resume"], "never")
        self.assertTrue(run.finished)
        self.assertEqual([p["train/global_step"] for p in run.logged], [BATCHES, 2 * BATCHES])
        heads = ["head0_lr0.0001_wd0.01", "head1_lr5e-05_wd0.4"]
        for epoch, p in enumerate(run.logged, start=1):
            self.assertEqual(p["probe/epoch"], epoch)
            for key in ("train_mae", "val_mae", "train_loss", "val_loss", "val_study_mae"):
                per_head = [p[f"probe/{key}/{h}"] for h in heads]
                self.assertEqual(p[f"probe/{key}"], min(per_head))  # the best head
            self.assertEqual(p["probe/val_failed_videos"], 0)
            # The cosine schedule has brought each head's learning rate down by the epoch's end.
            self.assertLess(p["probe/lr/head0_lr0.0001_wd0.01"], 1e-4)
        # Selected per study (the config has a video index).
        self.assertEqual(run.logged[-1]["probe/val_study_mae_best"],
                         min(p["probe/val_study_mae"] for p in run.logged))
        best = min(run.logged, key=lambda p: p["probe/val_study_mae"])["probe/epoch"]
        self.assertEqual(run.logged[-1]["probe/best_epoch"], best)

    def test_relative_paths_are_cut_to_their_last_component(self):
        cfg = {"folder": "patient_123/run", "experiment": {"data": {
            "dataset_train": "data/patient_123/train.csv", "video_index": "../sensitive/video_index.csv",
            "dataset_val": "val.csv"}}, "other": "./patient_9/x", "home": "~/patient_9/y",
            "windows": "C:\\patients\\p1\\a.csv", "lists": ["rel/a.csv", "/abs/b.csv"],
            "module_name": "evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder", "tag": "lvef-a4c"}
        self.assertEqual(provenance.public_config(cfg), {
            "folder": "run", "experiment": {"data": {"dataset_train": "train.csv", "video_index": "video_index.csv",
                                                     "dataset_val": "val.csv"}},
            "other": "x", "home": "y", "windows": "a.csv", "lists": ["a.csv", "b.csv"],
            "module_name": "evals.video_classification_frozen.modelcustom.vjepa_2_1_encoder", "tag": "lvef-a4c"})

    def test_the_config_records_the_run_without_paths(self):
        self.run_probe(self.config(epochs=1))
        [call] = self.wandb.calls
        self.assertEqual(call["settings"], provenance.PRIVATE_WANDB_SETTINGS)  # no console, metadata or git
        config = call["config"]

        def strings(value):
            if isinstance(value, dict):
                return [s for v in value.values() for s in strings(v)]
            if isinstance(value, list):
                return [s for v in value for s in strings(v)]
            return [value] if isinstance(value, str) else []

        self.assertFalse([s for s in strings(config) if self.tmp in s or os.path.isabs(s)])
        self.assertEqual(config["model_kwargs"]["checkpoint"], "cooldown.pth.tar")
        info = config["run_info"]
        self.assertEqual(info["seed"], 0)
        self.assertEqual(info["encoder_checkpoint"]["sha256"], provenance.sha256(self.checkpoint))
        self.assertEqual(info["git_commit"], provenance.git_state()[0])
        self.assertEqual(info["manifests"]["video_index"]["sha256"], provenance.sha256(self.index))
        self.assertEqual(info["splits"]["val"], {"videos": 5, "studies": 3, "patients": 3})
        params = info["parameters"]
        self.assertGreater(params["encoder_frozen"], 0)
        self.assertEqual(params["probe_trainable_total"], 2 * params["probe_trainable_per_head"])
        self.assertEqual(info["optimization"]["global_batch_size"], 2)
        self.assertEqual(info["targets"]["auroc_above"], 0.95)
        # The full config, with paths, stays with the probe.
        folder = os.path.join(self.tmp, "run", "video_classification_frozen", "lvef")
        self.assertTrue(os.path.exists(os.path.join(folder, "params-probe.yaml")))
        self.assertTrue(os.path.exists(os.path.join(folder, "run_info.json")))

    def interrupted(self, cfg, after):
        """Run `cfg` until it is stopped after `after` epochs; returns its wandb run."""
        with mock.patch.object(probe, "run_one_epoch", side_effect=stop_after(after, probe.run_one_epoch)), \
                self.assertRaises(Interrupted):
            self.run_probe(cfg)
        return self.wandb.runs[-1] if self.wandb.runs else None

    def test_a_resumed_probe_reattaches_to_its_run(self):
        first = self.interrupted(self.config(epochs=2), after=1)
        second = self.run_probe(self.config(epochs=2))
        self.assertEqual(self.wandb.calls[1]["id"], first.id)
        self.assertEqual(self.wandb.calls[1]["resume"], "must")
        self.assertEqual([p["probe/epoch"] for p in second.logged], [2])
        self.assertEqual(second.logged[0]["probe/best_epoch"], min(
            (p for p in first.logged + second.logged), key=lambda p: p["probe/val_study_mae"])["probe/epoch"])

    def test_without_a_project_nothing_is_logged(self):
        self.assertIsNone(self.run_probe(self.config(epochs=1, wandb=False)))
        self.assertEqual(self.wandb.calls, [])

    def test_a_regression_probe_needs_its_normalization(self):
        for mean, std, message in ((None, 10.0, "Set `experiment.data.target_mean`"),
                                   (55.0, 0.0, "standard deviation positive")):
            cfg = self.config(epochs=1)
            cfg["experiment"]["data"].update(target_mean=mean, target_std=std)
            with self.subTest(mean=mean, std=std), self.assertRaisesRegex(ValueError, message):
                self.run_probe(cfg)

    def test_the_normalization_matches_the_manifests_probe_info(self):
        with open(os.path.join(self.tmp, "probe_info.json"), "w") as f:
            json.dump({"target_mean": 55.0, "target_std": 9.0}, f)  # next to the train manifest
        with self.assertRaisesRegex(ValueError, "differ from the probe_info.json"):
            self.run_probe(self.config(epochs=1))  # target_std 10.0

    def test_study_targets_need_the_video_index(self):
        cfg = self.config(epochs=1, index=False)
        cfg["experiment"]["evaluation"] = {"targets": {"level": "study", "mae_below": 5.0}}
        with self.assertRaisesRegex(ValueError, "video_index"):
            self.run_probe(cfg)


if __name__ == "__main__":
    unittest.main()
