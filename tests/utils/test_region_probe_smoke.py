# tests/utils/test_region_probe_smoke.py

"""An end-to-end smoke run of the study probe with `pooling_mode: region` on CPU: tiny real
videos, a tiny encoder read by the `deepcoro_clip_echojepa_encoder` adapter, a region decoder
file, two epochs with the encoder's top block training, a resume, a warm start and the test."""

import contextlib
import copy
import json
import os
import pickle
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import torch

import src.models.vision_transformer as vit
from evals.study_multitask_frozen import eval as probe
from src.models.region_probe import RegionMaskDecoder, RegionStudyProbe
from tests.models.test_region_probe import CLASSES, CLASSIFIER, FRAMES, RES, SOURCE
from tests.utils.test_deepcoro_clip_encoder import OPTIONS
from tests.utils.test_probe_wandb import GRID, Interrupted
from tests.utils.test_study_probe_smoke import RunsOnLoad, stop_after, write_video
from tests.utils.test_vjepa_2_1_utils import QuietLogs

# (split, videos) of each study; the targets are drawn with a fixed seed.
STUDIES = [("train", 3), ("train", 2), ("train", 3), ("train", 1), ("val", 2), ("val", 3), ("val", 2),
           ("test", 2), ("test", 3), ("test", 1)]
BINARY = [h for h, task in CLASSIFIER["head_task"].items() if task == "binary"]


class TestRegionProbeSmoke(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = tmp.name
        rng = np.random.default_rng(0)
        rows = []
        for s, (split, videos) in enumerate(STUDIES):
            targets = {"a_size": float(rng.integers(0, 100)), "b_size": float(rng.integers(0, 100)),
                       "flow": float(rng.uniform(0.6, 1.0)), **{h: (s + i) % 2 for i, h in enumerate(BINARY)}}
            if s == 1:
                targets["flow"] = targets["flow_low"] = None  # a study without the pressure targets.
            for v in range(videos):
                path = os.path.join(cls.tmp, "videos", f"s{s}", f"{v:04d}.mp4")
                write_video(path, brightness=40 * s + 20 * v)
                rows.append({"video_path": path, "study_id": f"s{s}", "patient_id": f"p{s}", "split": split, **targets})
        cls.manifest = os.path.join(cls.tmp, "videos.csv")
        pd.DataFrame(rows).to_csv(cls.manifest, index=False)

        torch.manual_seed(0)
        cls.encoder_state = vit.vit_tiny(img_size=RES, num_frames=FRAMES, **OPTIONS).state_dict()
        cls.checkpoint = os.path.join(cls.tmp, "encoder.pt")
        torch.save({"encoder": cls.encoder_state}, cls.checkpoint)
        # A decoder that labels every pixel class 1: group `a` is found in every view, group `b` in none.
        decoder = RegionMaskDecoder(192, CLASSES, width=16, blocks=[1, 3]).state_dict()
        decoder["out.weight"].zero_()
        decoder["out.bias"].copy_(torch.tensor([0.0, 5.0, 0.0, 0.0]))
        cls.decoder_state = decoder
        cls.decoder = os.path.join(cls.tmp, "decoder.pt")
        torch.save({"region_decoder": decoder}, cls.decoder)

    def config(self, folder, trainable_blocks=1, **region):
        classifier = copy.deepcopy(CLASSIFIER)
        classifier["region"]["decoder"]["checkpoint"] = self.decoder
        classifier["region"].update(region)
        return {
            "folder": folder, "tag": "region", "num_workers": 0, "resume_checkpoint": True, "meta": {"seed": 0},
            "experiment": {
                "data": {"data_filename": self.manifest, "split_column": "split", "groupby_column": "study_id",
                         "num_videos": 3, "shuffle_videos": True, "num_clips_per_video": 1,
                         "frames_per_clip": FRAMES, "frame_step": 2, "resolution": RES},
                "encoder": {"trainable_blocks": trainable_blocks, "lr": 1e-4, "weight_decay": 0.01},
                "classifier": {**classifier, "target_mean": {"a_size": 0.0, "b_size": 0.0},
                               "target_std": {"a_size": 100.0, "b_size": 100.0}},
                "evaluation": {"selection": "weighted_loss"},
                "optimization": {"batch_size": 2, "num_epochs": 2, "use_bfloat16": False, "multihead_kwargs": GRID[:1]},
            },
            "model_kwargs": {
                "checkpoint": self.checkpoint,
                "module_name": "evals.video_classification_frozen.modelcustom.deepcoro_clip_echojepa_encoder",
                "pretrain_kwargs": {"encoder": {"model_name": "vit_tiny", "state_dict_key": "encoder",
                                                "encoder_prefix": "", **OPTIONS}},
                "wrapper_kwargs": {},
            },
        }

    @contextlib.contextmanager
    def on_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(probe.mp, "set_start_method"):
            yield

    def folder(self, cfg):
        return os.path.join(cfg["folder"], probe.EVAL_NAME, cfg["tag"])

    def load(self, cfg, name="best.pt"):
        return torch.load(os.path.join(self.folder(cfg), name), map_location="cpu", weights_only=True)

    def test_the_region_probe_trains_the_encoder_and_is_tested(self):
        cfg = self.config(os.path.join(self.tmp, "first"))
        passes = []
        forward_study = RegionStudyProbe.forward_study

        def counted(self, *args, **kwargs):
            outputs, regions = forward_study(self, *args, **kwargs)
            passes.append((regions["encoder_passes"], len(regions["crops"])))
            return outputs, regions

        with self.on_cpu(), mock.patch.object(RegionStudyProbe, "forward_study", counted):
            probe.main(dict(cfg))
            probe.main(dict(cfg, test_only=True))
        # Every study takes two encoder passes: its views, then at most two crops of group `a`.
        self.assertEqual({p for p, _ in passes}, {2})
        self.assertEqual({crops for _, crops in passes}, {1, 2})
        folder = self.folder(cfg)
        with open(os.path.join(folder, "test_metrics.json")) as f:
            result = json.load(f)
        self.assertEqual((result["test"]["studies"], result["test"]["failed"]), (3, 0))
        self.assertEqual(sorted(result["test"]["heads"]), sorted(CLASSIFIER["target_label"]))
        self.assertEqual(result["run_info"]["parameters"]["encoder_trainable_blocks"], 1)
        predictions = pd.read_csv(os.path.join(folder, "test_predictions.csv"))
        self.assertTrue(predictions.flow_prediction.between(0.0, 1.05).all())  # `output_range`.
        self.assertTrue(predictions.any_bright_prediction.between(0.0, 1.0).all())
        for name in ("params-probe.yaml", "run_info.json", "log_r0.csv", "latest.pt", "val_predictions.csv"):
            self.assertTrue(os.path.exists(os.path.join(folder, name)), name)

        best = self.load(cfg)
        settings = best["fingerprint"]["settings"]["classifier"]
        self.assertEqual(settings["pooling_mode"], "region")
        self.assertEqual(settings["region"]["decoder_sha256"], probe.provenance.sha256(self.decoder))
        self.assertIsNone(settings["region"]["init_sha256"])
        self.assertNotIn(self.tmp, json.dumps(settings))  # hashes, no paths.
        # The decoder is saved as it was read; the encoder's top block and the probe trained.
        state = best["probes"][0]
        for name, value in self.decoder_state.items():
            self.assertTrue(torch.equal(state[f"region_decoder.{name}"], value), name)
        self.assertEqual(len(best["opt"]), 2)  # the probe's optimizer, then the encoder's.
        moved = {k: not torch.equal(v, self.encoder_state[k.removeprefix("model.")])
                 for k, v in best["encoder"].items() if k.startswith("model.blocks.")}
        self.assertTrue(any(m for k, m in moved.items() if k.startswith("model.blocks.11.")))
        self.assertFalse(any(m for k, m in moved.items() if not k.startswith("model.blocks.11.")))
        self.assertTrue(bool(state["heads.a_bright.1.weight"].abs().sum() > 0))  # starts at zero.

        # The run is continued and tested only with the decoder it was trained with.
        other = os.path.join(self.tmp, "other-decoder.pt")
        torch.save({"region_decoder": {k: v + 1.0 for k, v in self.decoder_state.items()}}, other)
        changed = self.config(cfg["folder"])
        changed["experiment"]["classifier"]["region"]["decoder"]["checkpoint"] = other
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "decoder_sha256"):
            probe.main(dict(changed, val_only=True))

    def test_a_resumed_region_probe_continues_as_an_uninterrupted_one(self):
        def train(name, interrupt):
            cfg = self.config(os.path.join(self.tmp, name))
            with self.on_cpu():
                if interrupt:
                    with mock.patch.object(probe, "train_one_epoch",
                                           side_effect=stop_after(1, probe.train_one_epoch)), \
                            self.assertRaises(Interrupted):
                        probe.main(dict(cfg))
                probe.main(dict(cfg))
            with open(os.path.join(self.folder(cfg), "log_r0.csv")) as f:
                return f.read(), self.load(cfg, "latest.pt")

        log, latest = train("uninterrupted", interrupt=False)
        resumed_log, resumed = train("resumed", interrupt=True)
        self.assertEqual(resumed_log, log)
        for part in ("probes", "encoder"):
            a, b = (latest[part], resumed[part]) if part == "encoder" else (latest[part][0], resumed[part][0])
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_a_warm_start_reads_the_probe_entry_strictly(self):
        first = self.trained()
        trained = {k: v for k, v in self.load(first)["probes"][0].items() if not k.startswith("region_decoder.")}
        path = os.path.join(self.tmp, "init.pt")
        torch.save({"region_decoder": self.decoder_state, "probe": trained}, path)

        seen = []
        train_one_epoch = probe.train_one_epoch

        def first_state(**kwargs):
            if not seen:
                seen.append({k: v.clone() for k, v in kwargs["probes"][0].state_dict().items()})
            return train_one_epoch(**kwargs)

        cfg = self.config(os.path.join(self.tmp, "warm"), trainable_blocks=0, init_checkpoint=path)
        with self.on_cpu(), mock.patch.object(probe, "train_one_epoch", side_effect=first_state):
            probe.main(dict(cfg))
        for name, value in trained.items():
            self.assertTrue(torch.equal(seen[0][name], value), name)
        region = self.load(cfg)["fingerprint"]["settings"]["classifier"]["region"]
        self.assertEqual(region["init_sha256"], probe.provenance.sha256(path))
        self.assertIsNone(self.load(cfg)["encoder"])  # frozen encoder: nothing of it is saved.

        # A file without the entry, and a probe of another shape, are refused.
        with self.on_cpu(), self.assertRaisesRegex(KeyError, "no `probe` entry"):
            probe.main(self.config(os.path.join(self.tmp, "warm-missing"), init_checkpoint=self.decoder))
        partial = os.path.join(self.tmp, "partial.pt")
        torch.save({"probe": {k: v for k, v in trained.items() if not k.startswith("study_pooler.")}}, partial)
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "do not fit this probe"):
            probe.main(self.config(os.path.join(self.tmp, "warm-partial"), init_checkpoint=partial))

    def test_wrong_region_settings_and_files_are_refused(self):
        def refused(error, message, change, name="refused"):
            cfg = self.config(os.path.join(self.tmp, name))
            change(cfg["experiment"])
            with self.on_cpu(), self.assertRaisesRegex(error, message):
                probe.main(cfg)

        refused(ValueError, "read only with `pooling_mode: region`",
                lambda e: e["classifier"].update(pooling_mode="attentive"))
        refused(ValueError, "is one of .*'region'", lambda e: e["classifier"].update(pooling_mode="regions"))
        refused(ValueError, "is a mapping with the keys", lambda e: e["classifier"].pop("region"))
        refused(ValueError, "has unknown keys", lambda e: e["classifier"]["region"].update(crop_size=48))
        refused(ValueError, "one setting", lambda e: e["optimization"].update(multihead_kwargs=GRID))
        refused(ValueError, "outside the encoder's 12 blocks",
                lambda e: e["classifier"]["region"]["decoder"].update(blocks=[1, 12]))
        refused(KeyError, "no `region_decoder` entry",
                lambda e: e["classifier"]["region"]["decoder"].update(checkpoint=self.checkpoint))
        wide = os.path.join(self.tmp, "wide.pt")
        torch.save({"region_decoder": RegionMaskDecoder(192, CLASSES, width=32, blocks=[1, 3]).state_dict()}, wide)
        refused(RuntimeError, "size mismatch", lambda e: e["classifier"]["region"]["decoder"].update(checkpoint=wide))
        runs_code = os.path.join(self.tmp, "runs-code.pt")
        torch.save({"region_decoder": self.decoder_state, "payload": RunsOnLoad()}, runs_code)
        with mock.patch("src.utils.checkpoint_loader.time.sleep"):
            refused(pickle.UnpicklingError, "",
                    lambda e: e["classifier"]["region"]["decoder"].update(checkpoint=runs_code))

    def test_clip_grad_bounds_the_joint_gradient_norm_of_probe_and_encoder(self):
        cfg = self.config(os.path.join(self.tmp, "clipped"))
        cfg["experiment"]["optimization"].update(clip_grad=1e-3, num_epochs=1)
        clipped = []
        clip = torch.nn.utils.clip_grad_norm_

        def recorded(parameters, max_norm):
            parameters = list(parameters)
            before = clip(parameters, max_norm)
            after = torch.stack([p.grad.norm() for p in parameters if p.grad is not None]).norm()
            clipped.append((len(parameters), float(max_norm), float(before), float(after)))
            return before

        with self.on_cpu(), mock.patch.object(torch.nn.utils, "clip_grad_norm_", recorded):
            probe.main(dict(cfg))
        best = self.load(cfg)
        self.assertEqual(len(clipped), 2)  # 4 training studies in batches of 2.
        probe_parameters = sum(1 for k in best["probes"][0] if not k.startswith("region_decoder."))
        for count, max_norm, before, after in clipped:
            self.assertGreater(count, probe_parameters)  # the probe's and the encoder's top block's.
            self.assertEqual(max_norm, 1e-3)
            self.assertGreater(before, 1e-3)
            self.assertLessEqual(after, 1e-3 * (1 + 1e-4))
        self.assertEqual(best["fingerprint"]["settings"]["optimization"]["clip_grad"], 1e-3)
        # Without the key the settings do not name it; a wrong value and a grid are refused.
        self.assertNotIn("clip_grad", self.load(self.trained())["fingerprint"]["settings"]["optimization"])
        for value in (0, -1.0, True, "1"):
            other = self.config(os.path.join(self.tmp, "clip-refused"))
            other["experiment"]["optimization"]["clip_grad"] = value
            with self.on_cpu(), self.assertRaisesRegex(ValueError, "positive number or null"):
                probe.main(other)
        grid = self.config(os.path.join(self.tmp, "clip-grid"), trainable_blocks=0)
        classifier = grid["experiment"]["classifier"]
        for key in ("region", "query_mode", "task_groups", "attention_hidden"):
            classifier.pop(key)
        classifier.update(pooling_mode="mean")
        grid["experiment"]["optimization"].update(clip_grad=1.0, multihead_kwargs=GRID)
        with self.on_cpu(), self.assertRaisesRegex(ValueError, "clips one probe's step"):
            probe.main(grid)

    def trained(self):
        """A finished frozen-encoder run of the region probe, trained once for the tests that read it."""
        cfg = self.config(os.path.join(self.tmp, "frozen"), trainable_blocks=0)
        if not os.path.exists(os.path.join(self.folder(cfg), "best.pt")):
            with self.on_cpu():
                probe.main(dict(cfg))
        return cfg

    def test_whole_frames_are_resized_not_cropped_and_stay_in_unit_range(self):
        frames = [np.full((RES, 2 * RES, 3), 255, np.uint8) for _ in range(FRAMES)]
        for frame in frames:
            frame[:, :RES] = 0  # the left half is black: a center crop would lose part of it.
        (clip,) = probe.WholeFrames(SOURCE)(frames)
        self.assertEqual(tuple(clip.shape), (3, FRAMES, SOURCE, SOURCE))
        self.assertEqual((float(clip.min()), float(clip.max())), (0.0, 1.0))
        self.assertAlmostEqual(float(clip.mean()), 0.5, places=2)
        # Frames already at the size pass through unchanged.
        rng = np.random.default_rng(0)
        same = [rng.integers(0, 256, (SOURCE, SOURCE, 3), dtype=np.uint8) for _ in range(FRAMES)]
        (clip,) = probe.WholeFrames(SOURCE)(same)
        self.assertTrue(torch.equal(clip, torch.from_numpy(np.stack(same)).permute(3, 0, 1, 2).float() / 255.0))
        self.assertEqual(pickle.loads(pickle.dumps(probe.WholeFrames(SOURCE))).transform.transforms[0].size,
                         (SOURCE, SOURCE))

    def test_other_pooling_modes_keep_their_settings(self):
        cfg = self.config(os.path.join(self.tmp, "attentive"), trainable_blocks=0)
        classifier = cfg["experiment"]["classifier"]
        for key in ("region", "query_mode", "task_groups", "attention_hidden"):
            classifier.pop(key)
        classifier.update(pooling_mode="attentive", num_heads=4, num_probe_blocks=1)
        with self.on_cpu():
            probe.main(dict(cfg))
        settings = self.load(cfg)["fingerprint"]["settings"]["classifier"]
        self.assertNotIn("region", settings)
        self.assertEqual(sorted(settings), sorted([
            "pooling_mode", "num_heads", "num_probe_blocks", "attention_hidden", "dropout", "heads", "labels_map",
            "use_view_embedding", *probe.study_probe_options(classifier, cfg["experiment"]["data"],
                                                             cfg["model_kwargs"])]))
