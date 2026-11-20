"""Clip frame indices with a fractional frame step (opt-in: `fractional_frame_step: true`).

The default sampler turns a target rate into a whole frame step, `ceil(native fps) // fps`. At 8 fps a 25 fps video
gets step 3 (8.3 fps) and a 30 fps video step 3 (10 fps); at 12 fps a 25 fps video gets step 2 (12.5 fps). The clip
then spans more or less time than asked for, and a short loop (one heart beat) runs out of frames and is padded with
black frames.

With `fractional_frame_step` the step is the exact ratio, `native fps / target fps` (or `duration * native fps /
frames_per_clip`, or `frame_step` itself), never below 1 so that no frame is read twice. Frame `k` of a clip is the
native frame nearest to `start + k * step` (halves round up), so every clip covers the same time at the same rate
whatever the native frame rate. A step of at least 1 gives strictly increasing indices: no duplicates.

- A video long enough for the clip (`(frames_per_clip - 1) * step` frames after the start) gets a full clip. In
  training the start is a random whole frame among the positions that fit; in evaluation it is frame 0.
- A shorter video is read from frame 0 at the same step, as far as it goes; the rest of the clip is padding
  (`n_real < frames_per_clip`), which the loaders fill with black frames (clip index -1).
- `num_clips > 1` cuts the video into equal partitions with one clip each, or, with `allow_clip_overlap`, spreads
  the clips' starts evenly over the video, as the default sampler does.
"""

import math

import numpy as np


def native_step(video_fps, frames_per_clip, *, frame_step=None, fps=None, duration=None):
    """The fractional frame step (float, at least 1) for one video. A video without a usable frame rate is read
    frame by frame (step 1), as the default sampler does."""
    if fps is None and duration is None:
        step = float(frame_step)
    elif not (video_fps is not None and math.isfinite(video_fps) and video_fps > 0):
        step = 1.0
    elif fps is not None:
        step = float(video_fps) / float(fps)
    else:
        step = float(duration) * float(video_fps) / frames_per_clip
    return max(1.0, step)


def _offsets(step, count):
    """Frame offsets of `count` frames at `step` from frame 0: the nearest frame, halves rounded up."""
    return np.floor(np.arange(count) * step + 0.5).astype(np.int64)


def span(step, frames_per_clip):
    """Frames from the first to the last frame of a full clip (the last offset)."""
    return int(_offsets(step, frames_per_clip)[-1])


def clip_indices(num_frames, frames_per_clip, step, *, num_clips=1, random_start=False, allow_clip_overlap=False,
                 rng=np.random):
    """[(indices [frames_per_clip] int64, n_real)] for each of `num_clips` clips of a `num_frames`-frame video.

    `indices[:n_real]` are the frames to show; `indices[n_real:]` repeat the last real frame and are padding (the
    caller blacks them out or keeps the repeat, as its loader does). `rng` draws the training start
    (`rng.randint(low, high)`, high exclusive), so `np.random` seeding reproduces it."""
    if num_frames < 1:
        raise ValueError("A video needs at least one frame.")
    if step < 1:
        raise ValueError(f"The frame step must be at least 1, got {step}.")
    offsets = _offsets(step, frames_per_clip)
    last = int(offsets[-1])
    partition = num_frames // num_clips
    out = []
    for i in range(num_clips):
        if partition > last:
            # Room for a full clip in this partition.
            start = rng.randint(0, partition - last) if random_start else 0
            indices, n_real = start + offsets + i * partition, frames_per_clip
        else:
            if allow_clip_overlap:
                length, shift = num_frames, 0
                if num_clips > 1 and num_frames - 1 > last:
                    shift = (num_frames - 1 - last) // (num_clips - 1)
                base = i * shift
            else:
                # As the default sampler: a partition that is not empty, at least one frame.
                length, base = max(1, partition), i * partition
            n_real = int(np.count_nonzero(offsets <= length - 1))
            indices = np.minimum(offsets, offsets[n_real - 1]) + base
        indices = np.clip(indices, 0, num_frames - 1).astype(np.int64)
        out.append((indices, int(n_real)))
    return out


def contiguous_indices(num_frames, frames_per_clip, num_clips, step, *, random_start=False, rng=np.random):
    """[indices [frames_per_clip] int64] of `num_clips` back-to-back clips (the group loader's layout): clip i
    starts where clip i-1 would have taken its next frame, all at `step`. In training (`random_start`) the first
    clip starts at a random whole frame among the positions where all clips fit; otherwise, or when they do not
    fit, at frame 0. Indices past the video's end repeat its last frame."""
    if num_frames < 1:
        raise ValueError("A video needs at least one frame.")
    offsets = _offsets(step, num_clips * frames_per_clip)
    last = int(offsets[-1])
    start = rng.randint(0, num_frames - last) if (random_start and num_frames > last) else 0
    indices = np.clip(start + offsets, 0, num_frames - 1).astype(np.int64)
    return [indices[i * frames_per_clip:(i + 1) * frames_per_clip] for i in range(num_clips)]


# Pretraining checkpoints record the clip sampling they were trained with under this key, so that a resume cannot
# silently switch between whole and fractional frame steps, or change the frame rate.
CHECKPOINT_KEY = "clip_sampling"


def sampling_record(fps=None, fractional_frame_step=False):
    """The clip-sampling settings a pretraining checkpoint records."""
    return {"fps": fps, "fractional_frame_step": bool(fractional_frame_step)}


def check_resume(checkpoint, current, path="the checkpoint"):
    """Refuse to resume `checkpoint` (a dict) with other clip-sampling settings than it was trained with.
    `current` is this run's `sampling_record`. A checkpoint without a record was written before the option
    existed, with whole frame steps: it resumes only when this run keeps `fractional_frame_step` off."""
    saved = checkpoint.get(CHECKPOINT_KEY)
    if saved is None:
        if current["fractional_frame_step"]:
            raise ValueError(
                f"{path} records no clip sampling, so it was trained with whole frame steps; it cannot be resumed "
                "with `fractional_frame_step: true`. Resume it with the option off, or start a new run from its "
                "weights only.")
        return
    saved = sampling_record(saved.get("fps"), saved.get("fractional_frame_step", False))
    if saved != current:
        raise ValueError(
            f"{path} was trained with clip sampling {saved} and this run asks for {current}. Resume it with the "
            "same `fps` and `fractional_frame_step`, or start a new run from its weights only.")
