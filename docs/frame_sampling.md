# Clip frame rate: fractional frame steps

## Why

A clip is a fixed number of frames (`frames_per_clip`, usually 16). Asked for a frame rate (`fps`), the loaders
have always turned it into a whole frame step, `ceil(native fps) // fps`. So the rate a clip really gets depends on
the video's native rate: at `fps: 8`, a 25 fps video is read every 3rd frame (8.3 fps) and a 30 fps video also
every 3rd frame (10 fps); at `fps: 12`, a 25 fps video is read every 2nd frame (12.5 fps) and a 30 fps video every
2nd frame (15 fps).

The second problem is length. Many echo clips are a single heart beat of about one second. Sixteen frames at
8 fps need two seconds, so most clips run out of frames and the rest of the clip is black padding. For example, a
31-frame loop at 25 fps gives 10 real frames and 6 black ones at 8 fps, but 15 real frames and 1 black one at
12 fps.

## What `fractional_frame_step` does

With `fractional_frame_step: true` the step is the exact ratio `native fps / fps` (or `clip_duration × native fps /
frames_per_clip`), so a 25 fps video gets a step of 25/12 ≈ 2.08 at 12 fps. Frame `k` of a clip is the native frame
nearest to `start + k × step`. Every clip then covers the same time at the same rate, whatever the native rate.

- No frame is read twice: the step is never below 1. A video slower than the target is read frame by frame.
- Training draws a random start among the positions where the whole clip fits; evaluation always starts at frame 0,
  so it is deterministic.
- A video too short for the clip is read from frame 0 at the same step as far as it goes, then padded with black
  frames (clip index -1), as before.
- A video without a usable frame rate is read frame by frame, as before.

The option is **off by default**. Without it every loader reads exactly the frames it read before (a test checks the
indices byte for byte), so earlier runs reproduce. The code is in `src/datasets/utils/frame_sampling.py`.

## Config keys

Set both keys together. `fps` (or `clip_duration`) replaces `frame_step` (a `frame_step` left in the probe or cache
config is ignored when `fps` or `clip_duration` is set).

| Use | Where | Keys |
| --- | --- | --- |
| Pretraining (`app/vjepa_2_1`, `app/vjepa`) | `data:` | `fps: 12`, `fractional_frame_step: true` |
| Study probe (`evals/study_multitask_frozen`) | `experiment.data:` | `fps: 12`, `fractional_frame_step: true` (drop `frame_step`) |
| Frozen clip probe (`evals/video_classification_frozen`) | `experiment.data:` | `fps: 12`, `fractional_frame_step: true` (drop `frame_step`) |
| Multi-view probe (`evals/video_classification_frozen_multi`) | `experiment.data:` | `fps: 12`, `fractional_frame_step: true` (drop `frame_step`) |
| RADAR token cache (`evals/radar_frozen/build_token_cache.py`) | `data:` | `fps: 12`, `fractional_frame_step: true` (drop `frame_step`) |

The multi-view probe takes `fps`, not `clip_duration`. Its group loader (`src/datasets/video_group_dataset.py`)
reads every video with the single-view loader (`VideoDataset`), so each view gets exactly the clips the frozen clip
probe would read: the same rate, the same short-video padding and the same training/evaluation positions. The
upstream `init_data(data="VideoGroupDataset")` path is refused; use the multi-view probe.

Short clips are filled as `temporal_padding` says ([temporal_padding.md](temporal_padding.md)), with the option on
as without it: black frames at clip index -1, or under `loop` the clip's own real frames again from its start.

## What changes when it is on

- **Token caches and any other stored clip features must be rebuilt.** A cache holds the tokens of the frames it
  read; with the option on, other frames are read. The cache records `fps` and `fractional_frame_step` in its
  provenance (only when set, so the records of older caches do not change), and chunks with different settings do
  not merge.
- **Probe fingerprints change** when the option is on (the probes record `fps` and `fractional_frame_step` only when
  set), so a probe trained without it cannot be resumed or tested with it, and the reverse.
- **Pretraining checkpoints record their clip sampling** (`fps` and `fractional_frame_step`, key
  `clip_sampling`), and a resume with other settings is refused. A checkpoint written before the record resumes only
  with the option off; to switch an existing run to the option, start a new run from its weights.
- **Encoder and probe should agree.** An encoder pretrained at 12 fps should be probed, and its token cache built,
  at 12 fps with the same option.
- The manifest builder's `needs_padding` flag (`data/build_manifests.py`) still uses the whole-step rule.
