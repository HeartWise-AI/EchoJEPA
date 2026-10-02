# tests/datasets/test_video_padding.py

"""Test `VideoDataset` clip sampling with tiny synthetic videos in a temporary directory. Short videos
are padded with black frames marked by index `-1`, low or invalid FPS uses step 1, long videos follow
upstream sampling, float target FPS is treated as an integer, local paths stream correctly, sampled
frames match their indices, and a single manifest path behaves like a one-item list.
"""

import io
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from decord import VideoReader

import src.datasets.video_dataset as video_dataset
from app.vjepa.transforms import make_transforms
from src.datasets.data_manager import init_data
from src.datasets.video_dataset import VideoDataset

# The manifest scripts live in data/ as standalone scripts, not a package.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"))
from build_manifests import needs_padding  # noqa: E402

FPC = 16  # frames per clip, as in pretraining


def write_video(path, frames, fps):
    """A clip whose frames are never black, so padding is told apart from content."""
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), 40 + 10 * (i % 20), np.uint8))
    writer.release()


def reader(requested=None, fps=None, length=None):
    """The loader's decoder, recording the frame indices it is asked to decode into
    `requested`, reporting `fps` instead of the file's frame rate when given (raising it
    when it is an exception class), and `length` frames when given."""

    class Reader(VideoReader):
        def get_batch(self, indices):
            if requested is not None:
                requested.extend(int(i) for i in indices)
            return super().get_batch(indices)

        def get_avg_fps(self):
            if isinstance(fps, type) and issubclass(fps, Exception):
                raise fps("no frame rate")
            return super().get_avg_fps() if fps is None else fps

        def __len__(self):
            return super().__len__() if length is None else length

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
        frames = np.broadcast_to(np.asarray(indices, np.uint8).reshape(-1, 1, 1, 1) + 1, (len(indices), 2, 2, 3))
        return types.SimpleNamespace(asnumpy=lambda: frames.copy())


def upstream_whole_clip_indices(n_frames, fpc, fstp, num_clips, random_clip_sampling):
    """The frame indices upstream's sampler gives a video long enough for whole clips,
    drawing from np.random as it does. This path must not change."""
    clip_len, partition_len = fpc * fstp, n_frames // num_clips
    clips = []
    for i in range(num_clips):
        end_indx = np.random.randint(clip_len, partition_len) if random_clip_sampling else clip_len
        start_indx = end_indx - clip_len
        indices = np.linspace(start_indx, end_indx, num=fpc)
        indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
        clips.append(indices + i * partition_len)
    return clips


class TestShortClipPadding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def video(self, frames, video_fps):
        path = os.path.join(self.tmp.name, f"{frames}f_{video_fps}fps.mp4")
        write_video(path, frames, video_fps)
        return path

    def manifest(self, *paths):
        manifest = os.path.join(self.tmp.name, "manifest.csv")
        with open(manifest, "w") as f:
            f.writelines(f"{p} 0\n" for p in paths)
        return manifest

    def dataset(self, frames, video_fps, **sampling):
        """A VideoDataset over a one-video manifest."""
        return VideoDataset(data_paths=[self.manifest(self.video(frames, video_fps))], frames_per_clip=FPC, **sampling)

    def load_clips(self, frames, video_fps, **sampling):
        """Every clip of the video, and the frame indices of each."""
        buffer, _, clip_indices = self.dataset(frames, video_fps, **sampling).get_item_video(0)
        return [np.asarray(c) for c in buffer], [np.asarray(i) for i in clip_indices]

    def load(self, frames, video_fps, **sampling):
        """The first clip of the video, and its frame indices."""
        clips, indices = self.load_clips(frames, video_fps, **sampling)
        return clips[0], indices[0]

    def assert_padded_after(self, clip, indices, n_real):
        """The first n_real frames are real; every frame after them is black with index -1."""
        self.assertEqual((clip.shape[0], len(indices)), (FPC, FPC))
        self.assertTrue((indices[:n_real] >= 0).all(), indices)
        self.assertTrue((indices[n_real:] == -1).all(), indices)
        self.assertTrue((clip[n_real:] == 0).all(), "padded frames are not black")
        self.assertTrue((clip[:n_real].reshape(n_real, -1).max(axis=1) > 0).all(), "a real frame is black")

    def test_short_clip_at_the_pretraining_fps(self):
        # 30 fps sampled at 8 fps takes every 3rd frame: 12 frames give 4 real frames.
        self.assert_padded_after(*self.load(12, 30, fps=8, frame_step=None), n_real=4)

    def test_short_clip_at_a_fixed_frame_step(self):
        # The probe setting: every 2nd frame, so 12 frames give 6 real frames.
        self.assert_padded_after(*self.load(12, 30, frame_step=2), n_real=6)

    def test_video_below_the_target_fps_is_read_frame_by_frame(self):
        # 5 fps // 8 fps is 0; the step is floored at 1 instead of failing.
        self.assert_padded_after(*self.load(12, 5, fps=8, frame_step=None), n_real=12)
        # Long enough at step 1: a whole clip, no padding.
        self.assert_padded_after(*self.load(20, 5, fps=8, frame_step=None), n_real=FPC)

    def test_a_float_target_fps_samples_as_the_integer_one(self):
        # A config may give the target fps as 8.0: the frame step must still be a whole
        # number, for a short (padded) video and a long one alike.
        for frames in (12, 120):
            with self.subTest(frames=frames):
                np.random.seed(0)
                clip, indices = self.load(frames, 30, fps=8.0, frame_step=None)
                np.random.seed(0)
                want_clip, want_indices = self.load(frames, 30, fps=8, frame_step=None)
                np.testing.assert_array_equal(indices, want_indices)
                np.testing.assert_array_equal(clip, want_clip)
        self.assert_padded_after(*self.load(12, 30, fps=8.0, frame_step=None), n_real=4)

    def test_a_one_frame_video_is_its_frame_then_black_padding(self):
        self.assert_padded_after(*self.load(1, 30, fps=8, frame_step=None), n_real=1)

    def test_videos_shorter_than_a_clip_are_padded_to_exactly_16_frames(self):
        # At 8 fps the step is 1, so every frame of the video is real.
        for frames in (2, 5, 9, 15):
            with self.subTest(frames=frames):
                self.assert_padded_after(*self.load(frames, 8, fps=8, frame_step=None), n_real=frames)

    def test_clips_long_enough_are_not_padded(self):
        self.assert_padded_after(*self.load(48, 25, fps=8, frame_step=None), n_real=FPC)  # exactly one clip
        self.assert_padded_after(*self.load(60, 8, fps=8, frame_step=None), n_real=FPC)

    def test_long_videos_are_sampled_as_upstream(self):
        # Random and fixed windows, one clip and two.
        for frames, video_fps, num_clips, random_clip_sampling in (
            (120, 30, 1, True), (120, 30, 1, False), (200, 25, 2, True), (100, 8, 2, False),
        ):
            with self.subTest(frames=frames, video_fps=video_fps, num_clips=num_clips, random=random_clip_sampling):
                np.random.seed(3)
                _, indices = self.load_clips(frames, video_fps, fps=8, frame_step=None, num_clips=num_clips,
                                             random_clip_sampling=random_clip_sampling)
                np.random.seed(3)
                expected = upstream_whole_clip_indices(frames, FPC, math.ceil(video_fps) // 8, num_clips,
                                                       random_clip_sampling)
                for got, want in zip(indices, expected):
                    np.testing.assert_array_equal(got, want)

    def test_video_with_fewer_frames_than_clips(self):
        # Each clip gets the first frame, then padding; no frame index is ever negative,
        # so a real frame is never reported as padding.
        for frames, num_clips, overlap in ((1, 2, False), (3, 4, False), (1, 1, True)):
            requested = []
            with mock.patch.object(video_dataset, "VideoReader", reader(requested)):
                clips, indices = self.load_clips(frames, 30, fps=8, frame_step=None,
                                                 num_clips=num_clips, allow_clip_overlap=overlap)
            self.assertGreaterEqual(min(requested), 0, (frames, num_clips, overlap))
            self.assertEqual(len(clips), num_clips)
            for clip, idx in zip(clips, indices):
                self.assert_padded_after(clip, idx, n_real=1)

    def test_video_without_a_usable_fps_is_read_every_frame(self):
        # Missing (the decoder raises), zero, negative or non-finite: the fallback frame step
        # is 1, so 12 frames give 12 real frames, then padding, and the manifest flag (fps
        # recorded as unknown) agrees.
        for fps in (float("nan"), float("inf"), float("-inf"), 0.0, -30.0, RuntimeError):
            with self.subTest(fps=fps), mock.patch.object(video_dataset, "VideoReader", reader(fps=fps)):
                with self.assertWarnsRegex(UserWarning, "sampling every frame"):
                    self.assert_padded_after(*self.load(12, 30, fps=8, frame_step=None), n_real=12)
        clip = {"frames_per_clip": FPC, "fps": 8}
        self.assertTrue(needs_padding(pd.Series([12], dtype="Int64"), pd.Series([np.nan]), clip).iloc[0])

    def test_overlapping_clips_count_real_frames_from_the_length(self):
        # 12 frames at step 2 give 6 real frames, with or without clip overlap.
        self.assert_padded_after(*self.load(12, 30, frame_step=2, allow_clip_overlap=True), n_real=6)
        # 40 frames hold a whole 32-frame clip at step 2: two overlapping clips, no padding.
        clips, indices = self.load_clips(40, 30, frame_step=2, num_clips=2, allow_clip_overlap=True)
        for clip, idx in zip(clips, indices):
            self.assert_padded_after(clip, idx, n_real=FPC)

    def test_no_real_frame_is_read_twice(self):
        # Every length up to three whole clips, with and without clip overlap: a clip's real
        # frames are distinct, in order and inside the video, and a video no longer than one
        # clip at step 1 is read in full. Upstream's overlap branch clipped the indices to one
        # below the window's last frame, so its last two samples could land on one frame.
        manifest = self.manifest("unused.mp4")
        failures = []
        for overlap in (False, True):
            for num_clips in (1, 2, 3):
                for step in (1, 2, 3):
                    dataset = VideoDataset(data_paths=[manifest], frames_per_clip=FPC, frame_step=step,
                                           num_clips=num_clips, allow_clip_overlap=overlap)
                    for frames in range(1, 3 * FPC * step + 2):
                        _, clip_indices = dataset._sample_from_vr(FrameCounter(frames), FPC)
                        for idx in clip_indices:
                            real = idx[idx >= 0]
                            full = step > 1 or num_clips > 1 or frames > FPC or np.array_equal(real, np.arange(frames))
                            if not (len(real) and (np.diff(real) > 0).all() and 0 <= real.min() and real.max() < frames
                                    and full):
                                failures.append(f"{frames} frames, step {step}, {num_clips} clips, "
                                                f"overlap {overlap}: {real.tolist()}")
        self.assertEqual(failures[:3], [], f"{len(failures)} clips repeat or miss frames")

    def test_padding_is_black_never_the_last_frame(self):
        clip, indices = self.load(12, 30, fps=8, frame_step=None)
        self.assertEqual(clip.dtype, np.uint8)  # the decoded frames' dtype
        self.assertEqual(clip.shape, (FPC, 32, 32, 3))  # and their height, width and channels
        for frame in clip[4:]:
            self.assertTrue((frame == 0).all())
            self.assertFalse(np.array_equal(frame, clip[3]))

    def test_every_consumer_of_the_loader_gets_black_padding(self):
        # init_data builds the loader of every trainer and eval; none can ask for repeats.
        loader, _ = init_data(batch_size=1, data="VideoDataset", root_path=[self.manifest(self.video(12, 30))],
                              clip_len=FPC, fps=8, num_workers=0)
        buffer, _, clip_indices = loader.dataset.get_item_video(0)
        self.assert_padded_after(np.asarray(buffer[0]), np.asarray(clip_indices[0]), n_real=4)

    def test_the_transform_pipeline_takes_a_padded_clip(self):
        transform = make_transforms(random_horizontal_flip=False, crop_size=32)
        buffer, _, clip_indices = self.dataset(12, 30, fps=8, frame_step=None, transform=transform).get_item_video(0)
        clip = buffer[0]
        self.assertEqual(tuple(clip.shape), (3, FPC, 32, 32))  # (C, T, H, W), as the model takes it
        self.assertEqual(clip.dtype, torch.float32)
        self.assertTrue(torch.isfinite(clip).all())
        padded = torch.as_tensor(np.asarray(clip_indices[0]) == -1)
        self.assertEqual(int(padded.sum()), FPC - 4)
        self.assertTrue(torch.equal(clip[:, padded].amax(dim=1), clip[:, padded].amin(dim=1)))  # one black frame

    def test_unreadable_and_zero_frame_videos_fail_cleanly(self):
        # No clip is made up for them: the loader returns nothing, and __getitem__ moves on.
        broken = os.path.join(self.tmp.name, "broken.mp4")
        with open(broken, "wb") as f:
            f.write(b"not a video")
        self.assertIsNone(VideoDataset(data_paths=[self.manifest(broken)], frames_per_clip=FPC, fps=8,
                                       frame_step=None).get_item_video(0))
        with mock.patch.object(video_dataset, "VideoReader", reader(length=0)):
            with self.assertWarnsRegex(UserWarning, "without frames"):
                self.assertIsNone(self.dataset(12, 30, fps=8, frame_step=None).get_item_video(0))

    def test_s3_videos_are_sampled_and_padded_the_same_way(self):
        path = self.video(12, 30)
        with open(path, "rb") as f:
            data = f.read()
        s3 = mock.Mock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3.exceptions.ClientError = type("ClientError", (Exception,), {})
        s3.head_object.return_value = {"ContentLength": len(data)}
        s3.get_object.side_effect = lambda **_: {"Body": io.BytesIO(data)}
        dataset = VideoDataset(data_paths=[self.manifest("s3://bucket/video.mp4")], frames_per_clip=FPC, fps=8,
                               frame_step=None)
        dataset.s3_client = s3
        s3_buffer, _, s3_indices = dataset.get_item_video(0)
        local, local_indices = self.load(12, 30, fps=8, frame_step=None)
        self.assert_padded_after(np.asarray(s3_buffer[0]), np.asarray(s3_indices[0]), n_real=4)
        np.testing.assert_array_equal(np.asarray(s3_indices[0]), local_indices)
        np.testing.assert_array_equal(np.asarray(s3_buffer[0]), local)

    def test_local_videos_are_streamed_from_their_path(self):
        # decord gets the path, as upstream, so the video is not read into memory whole.
        opened = []

        class Recording(VideoReader):
            def __init__(self, uri, *args, **kwargs):
                opened.append(uri)
                super().__init__(uri, *args, **kwargs)

        with mock.patch.object(video_dataset, "VideoReader", Recording):
            self.load(12, 30, fps=8, frame_step=None)
        self.assertEqual([type(uri) for uri in opened], [str])

    def test_clip_frames_are_the_frames_at_their_indices(self):
        path = self.video(120, 30)
        for sampling in ({"fps": 8, "frame_step": None}, {"frame_step": 2, "num_clips": 2}):
            with self.subTest(**sampling):
                clips, indices = self.load_clips(120, 30, **sampling)
                decoded = VideoReader(path).get_batch(np.concatenate(indices)).asnumpy()
                np.testing.assert_array_equal(np.concatenate(clips), decoded)

    def test_a_single_manifest_path_is_read_like_a_list(self):
        manifest = self.manifest(self.video(12, 30), self.video(20, 8))
        one = VideoDataset(data_paths=manifest, frames_per_clip=FPC, fps=8, frame_step=None)
        listed = VideoDataset(data_paths=[manifest], frames_per_clip=FPC, fps=8, frame_step=None)
        self.assertEqual(len(one), 2)
        self.assertEqual((one.samples, one.data_paths), (listed.samples, listed.data_paths))

    def test_manifest_padding_flag_agrees_with_the_loader(self):
        for target in (8, 8.0):
            clip = {"frames_per_clip": FPC, "fps": target}
            for fps, frames in ((30, 12), (25, 47), (25, 48), (8, 15), (8, 16), (5, 12), (60, 100), (60, 112)):
                _, indices = self.load(frames, fps, fps=target, frame_step=None)
                flag = needs_padding(pd.Series([frames], dtype="Int64"), pd.Series([float(fps)]), clip).iloc[0]
                self.assertEqual(bool(flag), bool((indices == -1).any()), f"{frames} frames at {fps} fps, {target=}")


if __name__ == "__main__":
    unittest.main()
