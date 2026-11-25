# tests/datasets/test_frame_sampling.py

"""Fractional frame steps (`fractional_frame_step`, src/datasets/utils/frame_sampling.py): the index rule for
various native frame rates, lengths and targets; the default (whole-step) sampler unchanged, byte for byte; the
padding share at 12 fps against 8 fps; both padding policies (`temporal_padding`) on fractional clips; the option
reaching every loader that samples frames and every probe that builds one; and pretraining resumes refused when the
clip sampling changed."""

import hashlib
import os
import tempfile
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import torch

from src.datasets.utils import frame_sampling as fs
from src.datasets.video_dataset import VideoDataset
from src.datasets.video_group_dataset import VideoGroupDataset
from src.datasets.video_study_dataset import make_clip_loader

FPC = 16

# sha256 of the default sampler's clip indices over the grid below, computed with main's code without
# `fractional_frame_step`, for each `temporal_padding` policy. A change to the default sampling changes these.
DEFAULT_VIDEO_DIGEST = "a42b641ee8d1d455393ef1bd74cb16a5d38738de234f70cc04b75aefa313fba2"
DEFAULT_LOOP_DIGEST = "ba21736da13a47cb54e64133df4a7b4f8fe0c2ba6a3922432d64eeeb71675946"


class Reader:
    """A decoder stand-in: `n` frames at `fps`; frame i is filled with (i % 250) + 1, so never black."""

    def __init__(self, n, fps):
        self.n, self.fps, self.requested = n, fps, []

    def __len__(self):
        return self.n

    def get_avg_fps(self):
        return self.fps

    def seek(self, _):
        pass

    def get_batch(self, indices):
        indices = [int(i) for i in indices]
        assert all(0 <= i < self.n for i in indices), indices
        self.requested.append(indices)
        frames = np.broadcast_to((np.asarray(indices) % 250 + 1).astype(np.uint8).reshape(-1, 1, 1, 1),
                                 (len(indices), 2, 2, 3))
        return types.SimpleNamespace(asnumpy=lambda: frames.copy())


class NoFpsReader(Reader):
    """A decoder whose frame rate cannot be read."""

    def get_avg_fps(self):
        raise RuntimeError("no frame rate")


def video_dataset(**kw):
    kw.setdefault("allow_clip_overlap", False)
    return VideoDataset(data_paths=[], frames_per_clip=FPC, **kw)


def group_dataset(tmp, **kw):
    """A VideoGroupDataset over a one-study manifest in `tmp`; its videos are read by `ds.reader`."""
    path = os.path.join(tmp, "group.csv")
    with open(path, "w") as f:
        f.write("study_id,A4C[0],label\nS1,clip.mp4,0.5\n")
    kw.setdefault("frames_per_clip", FPC)
    return VideoGroupDataset(path, **kw)


class TestIndexRule(unittest.TestCase):
    def test_step(self):
        self.assertAlmostEqual(fs.native_step(25.0, FPC, fps=12), 25 / 12)
        self.assertAlmostEqual(fs.native_step(29.97, FPC, fps=12), 29.97 / 12)
        self.assertAlmostEqual(fs.native_step(30.0, FPC, duration=2.0), 30 * 2 / 16)
        self.assertEqual(fs.native_step(None, FPC, frame_step=3), 3.0)
        # Slower than the target: every frame once, never a frame twice.
        self.assertEqual(fs.native_step(10.0, FPC, fps=12), 1.0)
        # No usable frame rate: every frame, as the default sampler.
        for bad in (float("nan"), 0.0, -1.0, None):
            self.assertEqual(fs.native_step(bad, FPC, fps=12), 1.0)

    def test_full_clip_at_the_exact_rate(self):
        for native in (19.0, 25.0, 29.97, 30.0, 44.0, 50.0, 60.0, 74.0, 100.0):
            for target in (8, 12, 15):
                step = fs.native_step(native, FPC, fps=target)
                n = fs.span(step, FPC) + 1  # just long enough
                (indices, n_real), = fs.clip_indices(n, FPC, step)
                self.assertEqual(n_real, FPC)
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], n - 1)
                # Each frame is the nearest native frame to k * step: never more than half a frame off.
                self.assertLessEqual(np.abs(indices - np.arange(FPC) * step).max(), 0.5 + 1e-9)
                self.assertTrue((np.diff(indices) >= 1).all(), "duplicate frame")

    def test_short_video_is_read_from_the_start_then_padded(self):
        # A 31-frame loop at 25 fps (a common single-beat loop): 12 fps covers 30 frames in 15 steps of 25/12.
        (indices, n_real), = fs.clip_indices(31, FPC, 25 / 12)
        self.assertEqual(n_real, 15)
        self.assertEqual(indices[:n_real].tolist(), [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 25, 27, 29])
        self.assertTrue((indices[n_real:] == indices[n_real - 1]).all())
        # At 8 fps (step 25/8): 10 real frames.
        self.assertEqual(fs.clip_indices(31, FPC, 25 / 8)[0][1], 10)
        # One frame: one real frame.
        self.assertEqual(fs.clip_indices(1, FPC, 2.5)[0][1], 1)

    def test_no_duplicates_and_within_range_over_a_grid(self):
        rng = np.random.RandomState(0)
        for n in range(1, 200):
            for step in (1.0, 1.01, 1.5, 2.0, 25 / 12, 2.4975, 2.5, 3.125, 6.25):
                for num_clips in (1, 3):
                    for overlap in (False, True):
                        for random_start in (False, True):
                            for indices, n_real in fs.clip_indices(n, FPC, step, num_clips=num_clips,
                                                                   random_start=random_start,
                                                                   allow_clip_overlap=overlap, rng=rng):
                                self.assertEqual(len(indices), FPC)
                                self.assertTrue(1 <= n_real <= FPC)
                                self.assertTrue(((indices >= 0) & (indices < n)).all())
                                self.assertTrue((np.diff(indices[:n_real]) >= 1).all())

    def test_evaluation_is_deterministic_and_training_starts_anywhere_that_fits(self):
        step = 30 / 12
        last = fs.span(step, FPC)
        n = last + 1 + 20
        evaluation = {tuple(fs.clip_indices(n, FPC, step)[0][0]) for _ in range(5)}
        self.assertEqual(len(evaluation), 1)
        self.assertEqual(next(iter(evaluation))[0], 0)
        np.random.seed(0)
        starts = {int(fs.clip_indices(n, FPC, step, random_start=True)[0][0][0]) for _ in range(400)}
        self.assertEqual(starts, set(range(21)))  # every start that leaves room for the clip

    def test_several_clips(self):
        step = 2.0
        (a, na), (b, nb) = fs.clip_indices(100, FPC, step, num_clips=2)
        self.assertEqual((na, nb), (FPC, FPC))
        self.assertEqual((a[0], b[0]), (0, 50))  # one partition each
        (a, na), (b, nb) = fs.clip_indices(40, FPC, step, num_clips=2, allow_clip_overlap=True)
        self.assertEqual((na, nb), (FPC, FPC))
        self.assertEqual((a[0], b[-1]), (0, 39))  # starts spread over the video

    def test_contiguous_clips_keep_the_rate_and_start_anywhere_that_fits_in_training(self):
        step, k = 2.5, 2
        layout = np.floor(np.arange(k * FPC) * step + 0.5).astype(np.int64)
        last = int(layout[-1])
        n = last + 1 + 20
        evaluation = np.concatenate(fs.contiguous_indices(n, FPC, k, step))
        self.assertEqual(evaluation.tolist(), layout.tolist())
        np.random.seed(0)
        starts = set()
        for _ in range(400):
            both = np.concatenate(fs.contiguous_indices(n, FPC, k, step, random_start=True))
            self.assertEqual((both - both[0]).tolist(), layout.tolist())  # same spacing, clips back to back
            starts.add(int(both[0]))
        self.assertEqual(starts, set(range(21)))
        # Too short for both clips: from frame 0, the last frame repeated past the end.
        short = np.concatenate(fs.contiguous_indices(40, FPC, k, step, random_start=True))
        self.assertEqual(short.tolist(), np.minimum(layout, 39).tolist())

    def test_rejects_a_step_below_one(self):
        with self.assertRaises(ValueError):
            fs.clip_indices(30, FPC, 0.5)


class TestDefaultUnchanged(unittest.TestCase):
    def default_digest(self, **extra):
        modes = [dict(fps=8, frame_step=None), dict(fps=12, frame_step=None), dict(frame_step=2),
                 dict(frame_step=4), dict(duration=2.0, frame_step=None)]
        digest = hashlib.sha256()
        np.random.seed(0)
        for mode in modes:
            for overlap in (False, True):
                for num_clips in (1, 2):
                    for random_clip_sampling in (False, True):
                        ds = VideoDataset(data_paths=[], frames_per_clip=FPC, num_clips=num_clips,
                                          random_clip_sampling=random_clip_sampling, allow_clip_overlap=overlap,
                                          **mode, **extra)
                        for fps in (19.0, 25.0, 29.97, 30.0, 50.0, 74.0):
                            for n in range(1, 130):
                                _, clip_indices = ds._sample_from_vr(Reader(n, fps), FPC)
                                for c in clip_indices:
                                    digest.update(np.asarray(c, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def test_video_dataset_default_indices_are_byte_equal(self):
        self.assertEqual(self.default_digest(), DEFAULT_VIDEO_DIGEST)
        self.assertEqual(self.default_digest(temporal_padding="loop"), DEFAULT_LOOP_DIGEST)

    def test_group_dataset_default_reader_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = group_dataset(tmp, frame_step=4).reader
        self.assertEqual((reader.frame_step, reader.fps, reader.fractional_frame_step), (4, None, False))

    def test_option_is_off_by_default(self):
        self.assertFalse(video_dataset(fps=8, frame_step=None).fractional_frame_step)
        self.assertFalse(make_clip_loader(FPC, 2).fractional_frame_step)


class TestPadding(unittest.TestCase):
    """Synthetic single-beat loops: 0.8 to 1.6 s at 25 and 30 fps."""

    LOOPS = [(round(seconds * fps), fps) for fps in (25.0, 30.0) for seconds in np.arange(0.8, 1.61, 0.05)]

    def padding_share(self, **mode):
        ds = video_dataset(random_clip_sampling=False, **mode)
        shares = []
        for n, fps in self.LOOPS:
            _, clip_indices = ds._sample_from_vr(Reader(n, fps), FPC)
            shares.append(float((np.asarray(clip_indices[0]) == -1).mean()))
        return float(np.mean(shares))

    def test_padding_drops_at_12_fps(self):
        whole_8 = self.padding_share(fps=8, frame_step=None)
        fractional_8 = self.padding_share(fps=8, frame_step=None, fractional_frame_step=True)
        fractional_12 = self.padding_share(fps=12, frame_step=None, fractional_frame_step=True)
        self.assertGreater(whole_8, 0.25)
        self.assertGreater(fractional_8, 0.25)
        self.assertLess(fractional_12, 0.6 * whole_8)

    def test_a_31_frame_loop_at_25_fps(self):
        def pads(**mode):
            _, ci = video_dataset(random_clip_sampling=False, **mode)._sample_from_vr(Reader(31, 25.0), FPC)
            return int((np.asarray(ci[0]) == -1).sum())
        self.assertEqual(pads(fps=8, frame_step=None), 6)
        self.assertEqual(pads(fps=12, frame_step=None, fractional_frame_step=True), 1)


class TestLoaders(unittest.TestCase):
    def test_video_dataset_blacks_out_padding_and_reads_the_indices(self):
        ds = video_dataset(fps=12, frame_step=None, fractional_frame_step=True, random_clip_sampling=False)
        reader = Reader(31, 25.0)
        buffer, clip_indices = ds._sample_from_vr(reader, FPC)
        ci = np.asarray(clip_indices[0])
        self.assertEqual(ci.tolist(), [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1])
        self.assertTrue((buffer[-1] == 0).all())
        self.assertEqual(buffer[:15, 0, 0, 0].tolist(), (ci[:15] + 1).tolist())

    def test_video_dataset_training_draws_a_start_and_keeps_the_rate(self):
        ds = video_dataset(fps=12, frame_step=None, fractional_frame_step=True, random_clip_sampling=True)
        np.random.seed(3)
        starts = set()
        for _ in range(50):
            _, clip_indices = ds._sample_from_vr(Reader(120, 30.0), FPC)
            ci = np.asarray(clip_indices[0])
            starts.add(int(ci[0]))
            self.assertLessEqual(np.abs(ci - ci[0] - np.arange(FPC) * 2.5).max(), 0.5)
        self.assertGreater(len(starts), 10)

    def test_filter_short_videos_uses_the_fractional_span(self):
        ds = video_dataset(fps=12, frame_step=None, fractional_frame_step=True, filter_short_videos=True)
        span = fs.span(25 / 12, FPC)
        self.assertEqual(ds._sample_from_vr(Reader(span, 25.0), FPC), ([], None))
        self.assertEqual(len(ds._sample_from_vr(Reader(span + 1, 25.0), FPC)[0]), FPC)

    def test_study_probe_clip_loader(self):
        loader = make_clip_loader(FPC, fps=12, fractional_frame_step=True, random_clip_sampling=False)
        self.assertTrue(loader.fractional_frame_step)
        _, clip_indices = loader._sample_from_vr(Reader(31, 25.0), FPC)
        self.assertEqual(int((np.asarray(clip_indices[0]) == -1).sum()), 1)

    def test_loop_padding_repeats_the_real_frames_of_a_fractional_clip(self):
        ds = video_dataset(fps=12, frame_step=None, fractional_frame_step=True, random_clip_sampling=False,
                           temporal_padding="loop")
        buffer, clip_indices = ds._sample_from_vr(Reader(31, 25.0), FPC)
        ci = np.asarray(clip_indices[0])
        self.assertEqual(ci.tolist(), [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 25, 27, 29, 0])
        self.assertEqual(buffer[:, 0, 0, 0].tolist(), (ci + 1).tolist())             # real frames, none black

    def test_group_dataset_reads_each_video_with_the_fractional_single_view_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = group_dataset(tmp, frame_step=4, fps=12, fractional_frame_step=True, num_segments=2,
                               random_clip_sampling=False, temporal_padding="loop")
        reader = ds.reader
        self.assertEqual((reader.frame_step, reader.fps, reader.fractional_frame_step, reader.temporal_padding),
                         (None, 12, True, "loop"))
        _, clip_indices = reader._sample_from_vr(Reader(200, 30.0), FPC)
        for c in clip_indices:
            c = np.asarray(c)
            self.assertLessEqual(np.abs(c - c[0] - np.arange(FPC) * 2.5).max(), 0.5)  # 12 fps from 30 fps

    def test_token_cache_timing_and_provenance(self):
        from evals.radar_frozen import build_token_cache as btc

        self.assertEqual(btc.clip_timing({}), {"frame_step": 2, "duration": None, "fps": None,
                                               "fractional_frame_step": False})
        self.assertEqual(btc.clip_timing({"fps": 12, "frame_step": 2, "fractional_frame_step": True}),
                         {"frame_step": None, "duration": None, "fps": 12, "fractional_frame_step": True})
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, clips = os.path.join(tmp, "encoder.pt"), os.path.join(tmp, "clips.csv")
            for path in (checkpoint, clips):
                with open(path, "w") as f:
                    f.write("x")
            args = types.SimpleNamespace(clips=clips, split="train", exclude=[], chunks=4)
            # A config without the new keys (or with the option off) keeps the record of caches written before
            # them, key for key and value for value.
            for data_section, frame_step in (({"frame_step": 2}, 2), ({}, 2), ({"frame_step": 3}, 3),
                                             ({"frame_step": 3, "fractional_frame_step": False}, 3)):
                config = {"encoder": {"checkpoint": checkpoint}, "cache": {}, "data": data_section}
                self.assertEqual(btc.provenance(config, args, 1, 0)["data"], {
                    "frames_per_clip": 16, "frame_step": frame_step, "resolution": 224,
                    "normalization": [list(v) for v in btc.DEFAULT_NORMALIZATION],
                    "clip_sampling": "first_window", "transform": "eval"})
            config["data"] = {"fps": 12, "fractional_frame_step": True}
            data = btc.provenance(config, args, 1, 0)["data"]
            self.assertEqual((data["fps"], data["frame_step"], data["fractional_frame_step"]), (12, None, True))



class TestProbeEntryPoints(unittest.TestCase):
    """`fps`, `clip_duration` and `fractional_frame_step` reach the loaders of every probe."""

    def test_frozen_probe_make_dataloader(self):
        from evals.video_classification_frozen import eval as probe

        with mock.patch.object(probe, "init_data", return_value=("loader", "sampler")) as init:
            probe.make_dataloader(root_path=["x.csv"], batch_size=1, world_size=1, rank=0, frame_step=None,
                                  fps=12, fractional_frame_step=True)
        self.assertEqual({k: init.call_args.kwargs[k] for k in ("frame_sample_rate", "fps", "fractional_frame_step")},
                         {"frame_sample_rate": None, "fps": 12, "fractional_frame_step": True})


class TestMultiViewProbe(unittest.TestCase):
    """The multi-view probe reads `fps` and `fractional_frame_step` from `experiment.data` into its clip settings
    (also its fingerprint), only when set, and its datasets give them to the group reader."""

    def test_settings_without_the_keys_are_unchanged(self):
        from evals.video_classification_frozen_multi.eval import read_clip_settings

        self.assertEqual(read_clip_settings({"frame_step": 2}), {
            "resolution": 224, "frames_per_clip": 16, "frame_step": 2, "num_segments": 1, "normalization": None,
            "temporal_padding": None})

    def test_fps_replaces_frame_step_and_the_option_reaches_the_reader(self):
        from evals.video_classification_frozen_multi.eval import make_dataset, read_clip_settings

        settings = read_clip_settings({"frame_step": 2, "fps": 12, "fractional_frame_step": True,
                                       "frames_per_clip": FPC})
        self.assertEqual((settings["frame_step"], settings["fps"], settings["fractional_frame_step"]),
                         (None, 12, True))
        with tempfile.TemporaryDirectory() as tmp:
            path = group_dataset(tmp).manifest
            reader = make_dataset(path, training=False, **settings).reader
        self.assertEqual((reader.frame_step, reader.fps, reader.fractional_frame_step), (None, 12, True))


class TestResumeCheck(unittest.TestCase):
    OFF, ON = fs.sampling_record(8, False), fs.sampling_record(12, True)

    def test_same_settings_resume(self):
        for current in (self.OFF, self.ON):
            fs.check_resume({fs.CHECKPOINT_KEY: dict(current)}, current)

    def test_other_settings_are_refused(self):
        for saved, current in ((self.OFF, self.ON), (self.ON, self.OFF), (self.OFF, fs.sampling_record(12, False)),
                               (self.ON, fs.sampling_record(8, True))):
            with self.assertRaisesRegex(ValueError, "clip sampling"):
                fs.check_resume({fs.CHECKPOINT_KEY: saved}, current)

    def test_a_checkpoint_without_a_record_resumes_only_with_the_option_off(self):
        fs.check_resume({}, self.OFF)
        fs.check_resume({}, fs.sampling_record(None, False))
        with self.assertRaisesRegex(ValueError, "fractional_frame_step"):
            fs.check_resume({}, self.ON)

    def checkpoint(self, tmp, record):
        torch.manual_seed(0)
        models = [torch.nn.Linear(2, 2) for _ in range(3)]
        opt = torch.optim.SGD(models[0].parameters(), lr=0.1)
        saved = {"encoder": models[0].state_dict(), "predictor": models[1].state_dict(),
                 "target_encoder": models[2].state_dict(), "opt": opt.state_dict(), "scaler": None, "epoch": 3}
        if record is not None:
            saved[fs.CHECKPOINT_KEY] = record
        path = os.path.join(tmp, "latest.pt")
        torch.save(saved, path)
        return path, models, opt

    def test_load_checkpoint_of_both_trainers(self):
        from app.vjepa import utils as v1
        from app.vjepa_2_1 import utils as v21

        for load in (v1.load_checkpoint, v21.load_checkpoint):
            for record, current, refused in ((self.OFF, self.OFF, False), (self.OFF, self.ON, True),
                                             (None, self.OFF, False), (None, self.ON, True),
                                             (self.ON, self.ON, False), (self.ON, self.OFF, True)):
                with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print"):
                    path, models, opt = self.checkpoint(tmp, record)
                    call = lambda: load(r_path=path, encoder=models[0], predictor=models[1],
                                        target_encoder=models[2], opt=opt, scaler=None, clip_sampling=current)
                    if refused:
                        with self.assertRaises(ValueError):
                            call()
                    else:
                        call()

    def test_a_cooldown_from_anneal_ckpt_is_not_a_resume(self):
        from app.vjepa_2_1 import utils as v21

        with tempfile.TemporaryDirectory() as tmp:
            path, models, opt = self.checkpoint(tmp, self.OFF)
            v21.load_checkpoint(r_path=path, encoder=models[0], predictor=models[1], target_encoder=models[2],
                                opt=opt, scaler=None, is_anneal=True, clip_sampling=self.ON)


if __name__ == "__main__":
    unittest.main()
