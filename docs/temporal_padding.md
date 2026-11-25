# Temporal padding of short clips

Every pretraining clip has `frames_per_clip` frames (16). Many echo cines are shorter than one
clip at the target rate, so part of their clip is padding. `data.temporal_padding` decides how
such a clip is filled:

| Policy | Positions after the real frames | Clip index | Loss |
| --- | --- | --- | --- |
| `black` (default, the baseline) | black frames | `-1` | tubelets made only of padding are excluded |
| `loop` (ablation candidate) | the clip's own real frames again, from its start | true source index | every tubelet is a target |

The option exists on `VideoDataset`, `make_videodataset` and `init_data` (`temporal_padding=`), and
as `data.temporal_padding` in the V-JEPA 2.1 trainer (`app/vjepa_2_1/train.py`). Values are checked
strictly: anything other than the strings `black` and `loop` (a typo, other capitalisation, `null`)
raises `ValueError`. A config without the key uses `black`. Only `VideoDataset` implements the
choice; `init_data` refuses `loop` for every other dataset type. Code: `src/datasets/video_dataset.py`
(`loop_clip_indices`, `clip_index_stats`); tests: `tests/datasets/test_temporal_padding.py`.

## How the real frames are chosen (both policies)

`loop` does not resample anything. It takes exactly the indices, the number of real frames
`n_real` and the random window that `black` takes, and draws the same random numbers, then fills the
remaining positions differently. Notation: the video has `L` frames, split into `num_clips` segments
of `P = L // num_clips` frames; the source frame rate is `F`, the target rate `f` (`data.fps`, 8).

- **Frame step.** `s = max(1, ceil(F) // f)` (integer). A missing, zero, negative or non-finite
  `F` gives `s = 1` with a warning: every frame is sampled. With `frame_step` instead of `fps`, `s`
  is that value. Because of the integer step, the nominal rate is `F / s`, not exactly `f`:
  30 and 29.97 fps give `s = 3` (about 10 fps), 23.976 fps gives `s = 3` (about 8 fps), 25 fps gives
  `s = 3` (8.3 fps), 7.5 fps gives `s = 1` (7.5 fps). One clip covers `clip_len = 16 * s` source frames.
- **Long videos** (`P > clip_len`): a window `[e - clip_len, e)` with `e` drawn uniformly from
  `[clip_len, P)` (`e = clip_len` without random sampling), 16 evenly spaced indices in it. There is
  no padding, so both policies return identical indices and pixels for the same RNG state.
- **Short videos** (`P <= clip_len`): `n_real = max(1, P // s)` real frames at
  `floor(linspace(0, P, n_real))`, the last one clipped to `P - 1`, offset by `i * P` for clip `i`.
  Short clips always start at the first frame of their segment: there is no random phase. The real
  frames span the whole segment, so their spacing is `P / (n_real - 1)` source frames, wider than
  the nominal step `s` (12 frames at 30 fps: `s = 3`, real frames 0, 4, 8, 11: steps of 4, 4, 3).
- **`allow_clip_overlap`** (short videos): `n_real = max(1, min(clip_len, L) // s)` real frames at
  `floor(linspace(0, min(clip_len, L) - 1, n_real))`. Clip `i` is offset by
  `i * (L - clip_len) // (num_clips - 1)` when `L > clip_len` and `num_clips > 1`, else by 0, so
  every clip of a video no longer than one clip is the same clip.

These real indices are strictly increasing (no frame is read twice), which the tests check for every
length up to three clips, steps 1 to 3, one to three clips, with and without overlap.

## The loop

Position `j` of the clip takes the source frame of position `j % n_real`:

```
looped[j] = indices[j % n_real]        for j = 0 .. 15
```

The *unit* is the clip's `n_real` real frames in chronological order; it is repeated from its start
until the clip is full, and the last repetition is cut off at position 15. Frames are never
shuffled, reversed, taken from another part of the video, or taken from another video. With
`num_clips > 1`, each clip loops its own real frames.

| Video (target 8 fps) | `s` | Unit | Looped clip | Junctions |
| --- | --- | --- | --- | --- |
| 12 frames at 30 fps | 3 | 0 4 8 11 | 0 4 8 11 0 4 8 11 0 4 8 11 0 4 8 11 | 3 |
| 5 frames at 8 fps | 1 | 0 1 2 3 4 | 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 | 3 |
| 6 frames at 30 fps | 3 | 0 5 | 0 5 0 5 ... 0 5 | 7 |
| 1 to 5 frames at 30 fps | 3 | 0 | frame 0, 16 times (static) | 0 |
| 47 frames at 25 fps | 3 | 0 3 6 10 ... 43 46 (15 frames) | the unit, then 0 | 1 |
| 48 frames at 25 fps | 3 | 16 frames, 0 ... 47 | no loop: the clip is whole | 0 |

Edge cases, precisely:

- **Wraparound step.** At each junction the clip goes from the last real frame (`P - 1`) back to
  the first (`0`): a jump of `P - 1` frames backwards in the source. Read as a cyclic cine, that step
  is 1 source frame, while the cadence inside the unit is about `P / (n_real - 1)` frames, so the
  junction step does not equal the cadence in general. A short cine is not automatically a whole
  number of cardiac cycles: the junction usually joins two different cardiac phases, and any probe
  movement during the cine appears as a jump. Junctions sit at positions `n_real, 2 n_real, ...`;
  with tubelet size 2 and an odd `n_real`, some fall inside a tubelet (last and first real frame in
  one token).
- **No extra endpoint duplicate.** The unit holds each index once. After the last real frame comes
  the first real frame, so the last frame is not held and the start/end of the segment is not shown
  twice (the `linspace` endpoint `P` is clipped to `P - 1`, it does not wrap to frame 0). Adjacent
  positions repeat a frame only when `n_real == 1`.
- **Videos shorter than one sampling interval.** `n_real = max(1, P // s)`, so any segment shorter
  than `2 s` frames (and a 1-frame video) yields one real frame: under `loop`, a static clip of that
  frame repeated 16 times, with no junction; under `black`, that frame then 15 black frames.
- **Fewer than 16 source frames.** At `s = 1` (a source rate up to 15 fps at target 8, or missing
  FPS) every source frame is real, then the clip starts again.
- **Non-integer FPS** only matters through `ceil(F) // f`, as above.
- **Missing FPS**: `s = 1`, every frame (the manifest's `needs_padding` flag assumes the same).
- **Exact boundary.** `P == clip_len` takes the short-video branch with `n_real = 16`: no padding,
  so no loop. `P = clip_len - 1` already gives 15 real frames and one recycled (or black) position.
- **Long videos** are unaffected, with or without `num_clips > 1`.

Recycled positions carry their true source index in `clip_indices` (never `-1`) and the decoded
pixels of that frame, and go through the same transform as the rest of the clip. Under `loop`,
`clip_indices` contains no `-1`, so `token_validity` marks every token valid.

## What changes for training

Under `black`, padding-only tubelets are left out of the prediction and context losses (the encoder
still attends to them). Under `loop`, the recycled frames are real inputs and valid loss targets.
The policies therefore differ in what the loss is computed on, as well as in the input: **JEPA
losses are not comparable across policies** and are not a measure of representation quality between
them. Compare policies with the same frozen-probe evaluation on identical downstream data. The
encoder receives no timestamps under either policy; the effective frame rate (`F` over the real
frames' spacing) varies between videos, and storing FPS in a manifest does not make it explicit to
the model.

## Diagnostics

`clip_index_stats(clip_indices)` derives per-clip values from the indices alone. Each position is
exactly one of: a unique source frame (first occurrence of its index), a recycled frame (a repeat of
an index seen earlier in the clip), or black padding (`-1`); the three fractions sum to 1. A
junction is a real index smaller than the previous real index (padding in between is skipped); a
static clip has none.

The trainer logs, as means over all ranks of the logged step's batch:

| W&B key | Meaning |
| --- | --- |
| `data/valid_token_frac` | tokens counted in the loss (existing) |
| `data/batch_padded_frame_frac` | black padding frames (existing) |
| `data/unique_source_frame_frac` | distinct source frames |
| `data/recycled_frame_frac` | repeats of earlier frames (loop) |
| `data/loop_junctions_per_clip` | wraparound junctions per clip |

`data/unique_frame_frac` (existing) is a different measure: pixel-level uniqueness of input clips
uploaded for review, which leaves out padding and, under `loop`, recycled frames, so it still flags
frozen source videos. The console line adds `[frames unique/recycled/black: u/r/b] [junctions: j]`,
rank-local running means over the epoch like `valid`.

## Provenance

- `params-pretrain.yaml`: the launcher (`app/main.py`) saves the config file as given, so the
  shipped `configs/train_2_1/vitb16/*.yaml` set `temporal_padding: black` explicitly. A config
  without the key records nothing there.
- The W&B run config receives the resolved value under `data.temporal_padding`, including when the
  config file omits it; the training log prints `temporal_padding=...` at start.
- Checkpoints do not store it: resume and cool down with the same config.

## Reviewing junctions

`app/vjepa_2_1/review_loop_junctions.py` renders looped clips for clinical review of phase
mismatch and probe movement at the junctions:

```
python -m app.vjepa_2_1.review_loop_junctions --manifest /path/to/train.csv \
    --out-dir <REVIEW_ROOT>/loop_junctions --num 24 --seed 0 --frames-per-clip 16 --fps 8
```

It visits manifest rows in a seeded random order and keeps the first `--num` videos that `loop`
actually loops, sampled as the trainer samples them (without augmentation). `--out-dir` is refused
inside the repository (the images are clinical data) and must not already contain a review. Each
`line<manifest line>.png` shows the frames around the first junction, then the whole looped clip;
tiles are labelled `p<clip position> src <source frame index>` and junctions are marked in red.
Every junction of a clip joins the same two frames, so the first stands for all of them.
`junctions.csv` lists manifest line, `n_frames`, `fps`, `n_real`, junctions, recycled fraction, the
unit's first and last index and the PNG name. The console shows aggregate counts only, never paths.
Nothing is uploaded.

## Choosing a policy

`black` stays the default. `loop` replaces it only if a bounded ablation (identical cohorts,
initialization, schedule and frozen-probe inputs; validation only, never the test split) shows a
convincing validation benefit without unacceptable junction artifacts in clinical review or
clinical-task degradation. Inconclusive evidence keeps `black`. Available-duration sampling, higher
fixed rates, variable-length inputs and cardiac-cycle-aware looping are not implemented.
