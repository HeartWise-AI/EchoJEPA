# tests/datasets/test_temporal_padding.py

"""The temporal-padding policy of short clips (docs/temporal_padding.md): "black" (the baseline,
unchanged) and "loop" (the clip's real frames repeated from its start, with true source indices).
Synthetic videos are written to a temporary directory; frame k encodes k in its pixel values, so the
order of the decoded frames can be checked. Also covers the `clip_index_stats` diagnostics, the
option's plumbing through `init_data` and the V-JEPA 2.1 trainer (config, wandb keys, provenance),
and the junction-review CLI.
"""

import contextlib
import csv
import io
import os
import tempfile
import types
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist
from decord import VideoReader

import src.datasets.video_dataset as video_dataset
from app.vjepa_2_1 import review_loop_junctions, train
from app.vjepa_2_1.utils import token_validity
from src.datasets.data_manager import init_data
from src.datasets.video_dataset import (
    VideoDataset,
    clip_index_stats,
    loop_clip_indices,
    make_videodataset,
    recycled_frame_mask,
)
from src.utils import wandb_logging
from tests.utils.test_vjepa_2_1_baseline import QuietLogs, shipped, trainer_on_cpu
from tests.utils.test_vjepa_2_1_wandb import FakeWandb
from tests.utils.test_vjepa_2_1_wandb import config as trainer_config

FPC = 16  # frames per clip, as in pretraining
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_video(path, frames, fps):
    """Frame k is uniform with value 12 + 10 * (k % 24): never black, and `decode_k` recovers k % 24
    despite the codec's small colour shifts."""
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for k in range(frames):
        writer.write(np.full((32, 32, 3), 12 + 10 * (k % 24), np.uint8))
    writer.release()


def decode_k(clip):
    """The source frame number (mod 24) each decoded frame encodes."""
    means = np.asarray(clip).reshape(len(clip), -1).mean(axis=1)
    return np.rint((means - 10) / 10).astype(int) % 24


def reader(fps):
    """decord's reader, reporting `fps` (or raising it, if an exception class) as the frame rate."""

    class Reader(VideoReader):
        def get_avg_fps(self):
            if isinstance(fps, type) and issubclass(fps, Exception):
                raise fps("no frame rate")
            return fps

    return Reader


class FrameCounter:
    """A decoder stand-in with `n` frames, each filled with its index + 1, so never black."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def seek(self, _):
        pass

    def get_batch(self, indices):
        frames = np.broadcast_to(np.asarray(indices, np.int64).reshape(-1, 1, 1, 1) + 1, (len(indices), 2, 2, 3))
        return types.SimpleNamespace(asnumpy=lambda: frames.copy())


def upstream_whole_clip_indices(n_frames, fpc, fstp, num_clips, random_clip_sampling):
    """Upstream's frame indices for a video long enough for whole clips, drawing from np.random as it does."""
    clip_len, partition_len = fpc * fstp, n_frames // num_clips
    clips = []
    for i in range(num_clips):
        end_indx = np.random.randint(clip_len, partition_len) if random_clip_sampling else clip_len
        indices = np.linspace(end_indx - clip_len, end_indx, num=fpc)
        clips.append(np.clip(indices, end_indx - clip_len, end_indx - 1).astype(np.int64) + i * partition_len)
    return clips


def looped(unit, length=FPC):
    return [unit[j % len(unit)] for j in range(length)]


class VideoCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def video(self, frames, video_fps):
        path = os.path.join(self.tmp, f"{frames}f_{video_fps}fps.mp4")
        if not os.path.exists(path):
            write_video(path, frames, video_fps)
        return path

    def manifest(self, *paths, name="manifest.csv"):
        manifest = os.path.join(self.tmp, name)
        with open(manifest, "w") as f:
            f.writelines(f"{p} 0\n" for p in paths)
        return manifest

    def load_clips(self, frames, video_fps, policy, **sampling):
        """Every clip of the video under `policy`, and the frame indices of each."""
        sampling = {"fps": 8, "frame_step": None, **sampling}
        dataset = VideoDataset(data_paths=[self.manifest(self.video(frames, video_fps))], frames_per_clip=FPC,
                               temporal_padding=policy, **sampling)
        buffer, _, clip_indices = dataset.get_item_video(0)
        return [np.asarray(c) for c in buffer], [np.asarray(i) for i in clip_indices]

    def load(self, frames, video_fps, policy="loop", **sampling):
        clips, indices = self.load_clips(frames, video_fps, policy, **sampling)
        return clips[0], indices[0]

    def assert_looped(self, clip, indices, unit):
        """`indices` are `unit` repeated from its start, every frame is the decoded source frame it names."""
        np.testing.assert_array_equal(indices, looped(unit))
        np.testing.assert_array_equal(decode_k(clip), np.asarray(looped(unit)) % 24)
        self.assertTrue((clip.reshape(FPC, -1).max(axis=1) > 0).all(), "a looped frame is black")


class TestLoopSampling(VideoCase):

    def test_known_frame_order_across_wraps(self):
        # 30 fps at 8 fps: step 3; 12 frames give 4 real frames, linspace(0, 12, 4) clipped to 11.
        clip, indices = self.load(12, 30)
        self.assert_looped(clip, indices, [0, 4, 8, 11])
        _, black = self.load(12, 30, policy="black")
        np.testing.assert_array_equal(black, [0, 4, 8, 11] + [-1] * 12)
        # Wraps land at positions 4, 8 and 12: last real -> first real, never a reversal or a hold.
        self.assertEqual([j for j in range(1, FPC) if indices[j] < indices[j - 1]], [4, 8, 12])

    def test_fewer_than_16_source_frames(self):
        # At 8 fps the step is 1: every source frame is real, then the clip starts again.
        for frames in (2, 5, 9, 15):
            with self.subTest(frames=frames):
                self.assert_looped(*self.load(frames, 8), unit=list(range(frames)))

    def test_very_short_videos(self):
        # Step 3: fewer than 2 * 3 frames give one real frame (a static clip); 6 frames give 2.
        for frames in (2, 4, 5):
            with self.subTest(frames=frames):
                clip, indices = self.load(frames, 30)
                self.assert_looped(clip, indices, [0])
                self.assertTrue((clip == clip[0]).all(), "a static clip repeats its one frame")
        self.assert_looped(*self.load(6, 30), unit=[0, 5])

    def test_one_frame_video_is_that_frame_16_times(self):
        clip, indices = self.load(1, 30)
        self.assert_looped(clip, indices, [0])
        self.assertEqual(clip_index_stats(indices)["loop_junctions"][0], 0)

    def test_non_integer_fps(self):
        # The step is ceil(video fps) // 8: 29.97 and 23.976 fps give 3, as 30 fps does; 7.5 fps gives 1.
        for video_fps, unit in ((29.97, [0, 4, 8, 11]), (23.976, [0, 4, 8, 11]), (7.5, list(range(12)))):
            with self.subTest(fps=video_fps), mock.patch.object(video_dataset, "VideoReader", reader(video_fps)):
                self.assert_looped(*self.load(12, 30), unit=unit)
        # A float target fps samples as the integer one.
        np.testing.assert_array_equal(self.load(12, 30, fps=8.0)[1], self.load(12, 30, fps=8)[1])

    def test_missing_fps_samples_every_frame(self):
        for video_fps in (float("nan"), 0.0, -30.0, RuntimeError):
            with self.subTest(fps=video_fps), mock.patch.object(video_dataset, "VideoReader", reader(video_fps)):
                with self.assertWarnsRegex(UserWarning, "sampling every frame"):
                    self.assert_looped(*self.load(12, 30), unit=list(range(12)))

    def test_exact_boundary_length_needs_no_loop(self):
        # 48 frames at 25 fps: step 3, so exactly one 48-frame clip; both policies give the same 16 real frames.
        clip, indices = self.load(48, 25)
        black_clip, black = self.load(48, 25, policy="black")
        np.testing.assert_array_equal(indices, black)
        np.testing.assert_array_equal(clip, black_clip)
        self.assertEqual(len(set(indices.tolist())), FPC)
        # One frame shorter: 15 real frames, and the last position takes the first one.
        _, indices = self.load(47, 25)
        self.assertEqual(indices[-1], 0)
        self.assertEqual(len(set(indices[:-1].tolist())), FPC - 1)

    def test_long_video_windows_are_the_same_under_both_policies(self):
        for frames, video_fps, num_clips, random_clip_sampling in (
            (120, 30, 1, True), (120, 30, 1, False), (200, 25, 2, True), (100, 8, 2, False),
        ):
            sampling = dict(num_clips=num_clips, random_clip_sampling=random_clip_sampling)
            with self.subTest(frames=frames, fps=video_fps, **sampling):
                got = {}
                for policy in ("black", "loop"):
                    np.random.seed(3)
                    got[policy] = self.load_clips(frames, video_fps, policy, **sampling)
                np.random.seed(3)
                expected = upstream_whole_clip_indices(frames, FPC, int(np.ceil(video_fps)) // 8, num_clips,
                                                       random_clip_sampling)
                for loop_idx, black_idx, want in zip(got["loop"][1], got["black"][1], expected):
                    np.testing.assert_array_equal(loop_idx, want)
                    np.testing.assert_array_equal(black_idx, want)
                for loop_clip, black_clip in zip(got["loop"][0], got["black"][0]):
                    np.testing.assert_array_equal(loop_clip, black_clip)

    def test_two_clips_loop_within_their_own_segment(self):
        # 24 frames, 2 clips: two 12-frame segments, each sampled from its start and looped on its own.
        clips, indices = self.load_clips(24, 30, "loop", num_clips=2)
        self.assert_looped(clips[0], indices[0], [0, 4, 8, 11])
        self.assert_looped(clips[1], indices[1], [12, 16, 20, 23])

    def test_overlapping_clips_loop_the_same_way(self):
        # 12 frames at step 2 with overlap: linspace(0, 11, 6) -> 0, 2, 4, 6, 8, 11.
        unit = [0, 2, 4, 6, 8, 11]
        self.assert_looped(*self.load(12, 30, fps=None, frame_step=2, allow_clip_overlap=True), unit=unit)
        # A video no longer than a clip gives every overlapping clip the same start.
        clips, indices = self.load_clips(12, 30, "loop", fps=None, frame_step=2, allow_clip_overlap=True, num_clips=2)
        for clip, idx in zip(clips, indices):
            self.assert_looped(clip, idx, unit)

    def test_validity_masks(self):
        # Tubelets of 2 frames: black padding after 4 real frames leaves 2 of 8 tubelets valid, after 5
        # real frames 3 (a tubelet with one real frame counts); looping makes every tubelet valid.
        grid = 2
        for frames, video_fps, valid_tubelets in ((12, 30, 2), (5, 8, 3)):
            with self.subTest(frames=frames):
                _, black = self.load(frames, video_fps, policy="black")
                _, loop = self.load(frames, video_fps)
                v_black = token_validity(torch.as_tensor(black)[None], 2, grid)
                v_loop = token_validity(torch.as_tensor(loop)[None], 2, grid)
                expected = torch.arange(FPC // 2) < valid_tubelets
                self.assertTrue(torch.equal(v_black.view(FPC // 2, grid * grid).bool().all(dim=1), expected))
                self.assertTrue(bool((v_loop == 1).all()))

    def test_init_data_builds_a_looping_loader(self):
        loader, _ = init_data(batch_size=1, data="VideoDataset", root_path=[self.manifest(self.video(12, 30))],
                              clip_len=FPC, fps=8, num_workers=0, temporal_padding="loop")
        self.assertEqual(loader.dataset.temporal_padding, "loop")
        buffer, _, clip_indices = loader.dataset.get_item_video(0)
        self.assert_looped(np.asarray(buffer[0]), np.asarray(clip_indices[0]), [0, 4, 8, 11])
        # The default stays black padding.
        loader, _ = init_data(batch_size=1, data="VideoDataset", root_path=[self.manifest(self.video(12, 30))],
                              clip_len=FPC, fps=8, num_workers=0)
        self.assertEqual(loader.dataset.temporal_padding, "black")


class TestLoopSweep(unittest.TestCase):
    """Every length up to three whole clips, steps 1-3, 1-3 clips, with and without overlap."""

    @staticmethod
    def sample(policy, frames, **sampling):
        """`_sample_from_vr` on FrameCounter videos of each length in `frames`; `policy=None` omits the option."""
        if policy is not None:
            sampling["temporal_padding"] = policy
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "m.csv")
            with open(manifest, "w") as f:
                f.write("unused.mp4 0\n")
            dataset = VideoDataset(data_paths=[manifest], frames_per_clip=FPC, **sampling)
        return dataset, [dataset._sample_from_vr(FrameCounter(n), FPC) for n in frames]

    def test_loop_reuses_the_baseline_sample_and_black_is_unchanged(self):
        failures = []
        for overlap in (False, True):
            for num_clips in (1, 2, 3):
                for step in (1, 2, 3):
                    sampling = dict(frame_step=step, num_clips=num_clips, allow_clip_overlap=overlap)
                    frames = range(1, 3 * FPC * step + 2)
                    results = {}
                    for policy in (None, "black", "loop"):
                        np.random.seed(0)
                        results[policy] = self.sample(policy, frames, **sampling)[1]
                    for n, default, black, loop in zip(frames, results[None], results["black"], results["loop"]):
                        where = f"{n} frames, {sampling}"
                        # Explicit "black" is the default, frame for frame.
                        np.testing.assert_array_equal(default[0], black[0], where)
                        np.testing.assert_array_equal(np.stack(default[1]), np.stack(black[1]), where)
                        for clip, b_idx, l_idx in zip(np.split(loop[0], num_clips), black[1], loop[1]):
                            n_real = int((b_idx >= 0).sum())
                            ok = (
                                (l_idx >= 0).all()
                                and np.array_equal(l_idx[:n_real], b_idx[:n_real])
                                and np.array_equal(l_idx, loop_clip_indices(l_idx, n_real))
                                and np.array_equal(clip[:, 0, 0, 0], l_idx + 1)  # the decoded frame of each index
                                and (n_real == 1 or (np.diff(l_idx) != 0).all())  # no frame shown twice in a row
                                and clip_index_stats(l_idx)["unique_frame_frac"][0] * FPC == n_real
                                and clip_index_stats(l_idx)["loop_junctions"][0]
                                == (0 if n_real == 1 else -(-FPC // n_real) - 1)
                            )
                            if not ok:
                                failures.append(f"{where}: black {b_idx.tolist()}, loop {l_idx.tolist()}")
        self.assertEqual(failures[:3], [], f"{len(failures)} looped clips differ from the specification")

    def test_black_padding_frames_stay_black(self):
        _, ((buffer, indices),) = self.sample("black", [12], frame_step=3)
        self.assertTrue((buffer[4:] == 0).all())
        np.testing.assert_array_equal(indices[0], [0, 4, 8, 11] + [-1] * 12)


class TestClipIndexStats(unittest.TestCase):

    def stats(self, indices):
        return {k: v.tolist() for k, v in clip_index_stats(np.asarray(indices)).items()}

    def test_hand_built_clips(self):
        rows = [
            ([0, 1, 2, 3], (1.0, 0.0, 0.0, 0)),  # all unique
            ([0, 4, -1, -1], (0.5, 0.0, 0.5, 0)),  # black padding
            ([0, 4, 0, 4], (0.5, 0.5, 0.0, 1)),  # looped twice: one junction
            ([5, 9, 11, 5, 9, 11, 5, 9], (3 / 8, 5 / 8, 0.0, 2)),
            ([7, 7, 7, 7], (0.25, 0.75, 0.0, 0)),  # static: repeats, but no junction
            ([3, -1, 1, -1], (0.5, 0.0, 0.5, 1)),  # a decrease across padding is still a junction
            ([-1, -1, -1, -1], (0.0, 0.0, 1.0, 0)),
        ]
        for clip, (unique, recycled, padded, junctions) in rows:
            with self.subTest(clip=clip):
                stats = self.stats([clip])
                self.assertAlmostEqual(stats["unique_frame_frac"][0], unique)
                self.assertAlmostEqual(stats["recycled_frame_frac"][0], recycled)
                self.assertAlmostEqual(stats["padded_frame_frac"][0], padded)
                self.assertEqual(stats["loop_junctions"][0], junctions)
                self.assertAlmostEqual(unique + recycled + padded, 1.0)
        # A batch gives one value per clip.
        batch = self.stats([clip for clip, _ in rows if len(clip) == 4])
        self.assertEqual(batch["loop_junctions"], [0, 0, 1, 0, 1, 0])

    def test_inputs(self):
        # One clip (T,), a batch, a CPU tensor as the trainer passes it.
        self.assertEqual(self.stats([0, 0])["recycled_frame_frac"], [0.5])
        tensor = clip_index_stats(torch.tensor([[0, 1, 0, 1], [0, 1, -1, -1]]))
        self.assertEqual(tensor["loop_junctions"].tolist(), [1.0, 0.0])
        np.testing.assert_array_equal(recycled_frame_mask(torch.tensor([2, 5, 2, -1])), [[False, False, True, False]])
        for bad in ([[0, -2]], [[[0]]], np.zeros((1, 0), int), [[0.5, 1.0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                clip_index_stats(np.asarray(bad))

    def test_loop_clip_indices(self):
        np.testing.assert_array_equal(loop_clip_indices([3, 6, 9, 9, 9], 3), [3, 6, 9, 3, 6])
        for n_real in (0, 6):
            with self.assertRaises(ValueError):
                loop_clip_indices([3, 6, 9, 9, 9], n_real)


class TestPolicyOption(VideoCase):

    def test_invalid_values_are_refused(self):
        manifest = self.manifest(self.video(12, 30))
        for bad in ("repeat", "Loop", "", None, 1):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "temporal_padding"):
                    VideoDataset(data_paths=[manifest], frames_per_clip=FPC, fps=8, frame_step=None,
                                 temporal_padding=bad)
                with self.assertRaisesRegex(ValueError, "temporal_padding"):
                    make_videodataset(data_paths=[manifest], batch_size=1, frames_per_clip=FPC, fps=8,
                                      frame_step=None, num_workers=0, temporal_padding=bad)
                with self.assertRaisesRegex(ValueError, "temporal_padding"):
                    init_data(batch_size=1, data="VideoDataset", root_path=[manifest], clip_len=FPC, fps=8,
                              num_workers=0, temporal_padding=bad)

    def test_other_datasets_refuse_to_loop(self):
        for data in ("ImageNet", "VideoGroupDataset"):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "only supported by VideoDataset"):
                init_data(batch_size=1, data=data, root_path=["unused"], temporal_padding="loop")

    def test_shipped_configs_keep_black_padding_explicitly(self):
        # The launcher saves the config file as given (params-pretrain.yaml), so the key must be in it.
        for phase in ("pretrain", "cooldown"):
            for arm in ("continued", "scratch"):
                with self.subTest(phase=phase, arm=arm):
                    self.assertEqual(shipped(f"{phase}-{arm}-224px-16f.yaml")["data"]["temporal_padding"], "black")


class TestLogInputClipsRecycled(unittest.TestCase):

    def test_looped_frames_are_not_reported_as_frozen(self):
        clips = torch.randn(1, 3, 4, 8, 8)
        clips[:, :, 2:] = clips[:, :, :2]  # looped: frames 2, 3 repeat frames 0, 1
        indices = torch.tensor([[0, 1, 0, 1]])
        logged = []
        run = SimpleNamespace(log=logged.append)
        fake_wandb = SimpleNamespace(Video=lambda data, fps, format: ("video", data.shape))
        with mock.patch.object(wandb_logging, "wandb", fake_wandb), \
                self.assertNoLogs(wandb_logging.logger, level="WARNING"):
            wandb_logging.log_input_clips(run, [clips], step=0, normalize=((0.5,) * 3, (0.5,) * 3), num_videos=1,
                                          padded=[indices == -1],
                                          recycled=[torch.from_numpy(recycled_frame_mask(indices))])
        self.assertEqual(logged[0]["data/unique_frame_frac"], 1.0)
        self.assertEqual(logged[0]["data/padded_frame_frac"], 0.0)
        # Without the mask the repeats count, and the frozen-frame warning fires.
        with mock.patch.object(wandb_logging, "wandb", fake_wandb), \
                self.assertLogs(wandb_logging.logger, level="WARNING"):
            wandb_logging.log_input_clips(run, [clips], step=0, normalize=((0.5,) * 3, (0.5,) * 3), num_videos=1,
                                          padded=[indices == -1])
        self.assertEqual(logged[1]["data/unique_frame_frac"], 0.5)


FRAMES, CROP = 4, 64


class LoopedClips(torch.utils.data.Dataset):
    """Pairs of clips as VideoDataset returns them under "loop": a 2-frame video looped, a whole one."""

    def __len__(self):
        return 4

    def __getitem__(self, i):
        indices = np.array([0, 1, 0, 1]) if i % 2 else np.arange(FRAMES)
        return [torch.randn(3, FRAMES, CROP, CROP)], 0, [indices]


class TestTrainerTemporalPadding(QuietLogs):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
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
        self.init_data_kwargs = []

    def fake_init_data(self, batch_size, collator, **kwargs):
        self.init_data_kwargs.append(kwargs)
        loader = torch.utils.data.DataLoader(LoopedClips(), batch_size=batch_size, collate_fn=collator,
                                             drop_last=True)
        return loader, SimpleNamespace(set_epoch=lambda epoch: None)

    def run_trainer(self, cfg):
        os.makedirs(cfg["folder"], exist_ok=True)
        with trainer_on_cpu(), mock.patch.object(train, "init_data", side_effect=self.fake_init_data), \
                mock.patch.object(wandb_logging, "wandb", self.wandb), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0):
            train.main(cfg)
        return self.wandb.runs[-1]

    def test_loop_reaches_the_loader_the_run_config_and_the_logs(self):
        cfg = trainer_config(os.path.join(self.tmp, "pretrain"))
        cfg["data"]["temporal_padding"] = "loop"
        run = self.run_trainer(cfg)
        self.assertEqual(self.init_data_kwargs[0]["temporal_padding"], "loop")
        self.assertEqual(self.wandb.calls[0]["config"]["data"]["temporal_padding"], "loop")
        scalars = [p for p in run.logged if "train/loss" in p]
        self.assertTrue(scalars)
        for p in scalars:
            # Per pair: unique 2/4 and 4/4, recycled 2/4 and 0, junctions 1 and 0; no padding, all tokens valid.
            self.assertAlmostEqual(p["data/valid_token_frac"], 1.0)
            self.assertAlmostEqual(p["data/batch_padded_frame_frac"], 0.0)
            self.assertAlmostEqual(p["data/unique_source_frame_frac"], 0.75)
            self.assertAlmostEqual(p["data/recycled_frame_frac"], 0.25)
            self.assertAlmostEqual(p["data/loop_junctions_per_clip"], 0.5)

    def test_a_config_without_the_key_records_black(self):
        cfg = trainer_config(os.path.join(self.tmp, "pretrain"))
        cfg["data"].pop("temporal_padding", None)
        self.run_trainer(cfg)
        self.assertEqual(self.init_data_kwargs[0]["temporal_padding"], "black")
        self.assertEqual(self.wandb.calls[0]["config"]["data"]["temporal_padding"], "black")
        self.assertNotIn("temporal_padding", cfg["data"])  # the caller's config is not modified

    def test_an_invalid_policy_stops_the_run_before_loading_data(self):
        cfg = trainer_config(os.path.join(self.tmp, "pretrain"))
        cfg["data"]["temporal_padding"] = "mirror"
        with self.assertRaisesRegex(ValueError, "temporal_padding"):
            self.run_trainer(cfg)
        self.assertEqual(self.init_data_kwargs, [])


class TestReviewLoopJunctions(VideoCase):

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            review_loop_junctions.main(list(argv))
        return out.getvalue()

    def test_out_dir_inside_the_repository_is_refused(self):
        manifest = self.manifest(self.video(12, 30))
        with self.assertRaises(SystemExit):
            self.run_cli("--manifest", manifest, "--out-dir", os.path.join(REPO_ROOT, "junction_review"))
        self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "junction_review")))

    def test_renders_looped_videos_only(self):
        from PIL import Image

        broken = os.path.join(self.tmp, "broken.mp4")
        with open(broken, "wb") as f:
            f.write(b"not a video")
        # Lines 1-4: looped (n_real 4), long (no loop), static (one frame), unreadable.
        manifest = self.manifest(self.video(12, 30), self.video(120, 30), self.video(1, 30), broken)
        out_dir = os.path.join(self.tmp, "review")
        printed = self.run_cli("--manifest", manifest, "--out-dir", out_dir, "--num", "10", "--seed", "1")
        with open(os.path.join(out_dir, "junctions.csv")) as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["manifest_line"]))
        self.assertEqual([(r["manifest_line"], r["n_frames"], r["n_real"], r["junctions"]) for r in rows],
                         [("1", "12", "4", "3"), ("3", "1", "1", "0")])
        self.assertEqual((rows[0]["first_index"], rows[0]["last_index"]), ("0", "11"))
        self.assertAlmostEqual(float(rows[0]["fps"]), 30.0, places=1)
        for r in rows:
            with Image.open(os.path.join(out_dir, r["png"])) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertGreater(image.width, FPC * 16)
        self.assertIn("unreadable 1", printed)
        self.assertIn("no loop 1", printed)
        self.assertNotIn(self.tmp, printed)
        self.assertNotIn(".mp4", printed)
        # A second review never mixes with the first.
        with self.assertRaises(SystemExit):
            self.run_cli("--manifest", manifest, "--out-dir", out_dir)


if __name__ == "__main__":
    unittest.main()
