# Pilot pretraining baseline: V-JEPA 2.1 ViT-B, scratch vs. EchoJEPA checkpoint

This document records the code, model, checkpoint and configuration used by the two
initial pretraining experiments, so that another developer can reproduce both runs.
The two runs are ViT-B pretraining **from random initialization** ("scratch") and
**continued from the published EchoJEPA ViT-B checkpoint** ("continued").
It answers issue #9. The data comes from issue #6. Training also requires the loader fix
of issue #8, a separate PR; see [Prerequisite](#prerequisite-the-8-loader-fix).

## Decision

| | |
|---|---|
| Upstream code | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) `app/vjepa_2_1` at commit `204698b45b3712590f06245fbfba32d3be539812` (2026-03-23). V-JEPA 2.1 landed in `45d025f`. Because the repository has no tags or releases, commit hashes are used to identify the exact upstream version. |
| HeartWise code | Commit `ffcc8345128241d434b90e4ed39f07a3518f4f71` imported that upstream `app/vjepa_2_1` directory unchanged. The adaptations described below were made from that baseline. Each run records its exact local code version at launch using `git rev-parse HEAD` (see [Launch](#launch)). |
| Architecture | V-JEPA 2.1 `vit_base` (ViT-B/16) encoder, 86,833,152 parameters (published as "80M"), with a 12-layer, 384-wide predictor of 22,182,528 parameters. |
| Checkpoint | `vjepa2_1_vitb_mimic_pt169_c60.pt` from the EchoJEPA authors, SHA-256 `40384a17d3d32375142e3bd3a43d2f5bfa2b174b44c6960958bffd731e9624c1`. |
| Loading | Strict weight loading: the encoder, predictor, and target encoder must have zero missing, unexpected, or shape-mismatched keys. Only model weights are restored; the optimizer and training schedules are initialized fresh. |
| Configs | [configs/train_2_1/vitb16/](../configs/train_2_1/vitb16/): `pretrain-{scratch,continued}-224px-16f.yaml`, then `cooldown-{scratch,continued}-224px-16f.yaml`. Within each stage, the two arms are identical except for the checkpoint that initializes that arm (`meta.init_checkpoint` for pretraining and `optimization.anneal_ckpt` for cooldown), plus each run’s output folder. |
| Confirmed | The checkpoint's SHA-256, and its strict load into this model with a finite training loss ([Checkpoint loading](#checkpoint-loading)). |
| Newer versions | None found on 2026-09-25. |
| Duration | 3,900 pretraining + 900 cooldown steps per arm, at a global batch of 256. |
| Environment | Python 3.12 with the exact package versions of [requirements-pilot.txt](../requirements-pilot.txt); the launch script refuses any other version. |
| Prerequisite | The #8 loader fix (separate PR) is merged before either run starts. |

## Code baseline

**HeartWise `main` is not aligned with V-JEPA 2.1 on its own.** `main` (`c783da6`) is
[bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA) at `77208e1` (2026-06-18)
plus HeartWise's own PRs. It contains only V-JEPA 2 trainers (`app/vjepa`, and
`app/vjepa_droid` for the action-conditioned model). The
EchoJEPA authors released V-JEPA 2.1 checkpoints but not the 2.1 training code they used.
This branch therefore imports Meta's official 2.1 trainer:

- `ffcc834` copies all 10 files of `app/vjepa_2_1` from `204698b`. They are verified
  byte-identical to upstream.
- The trainer imports shared utilities from `src/`, which originate from EchoJEPA’s fork of an older `vjepa2` revision rather than directly from Meta’s V-JEPA 2.1 commit. Meta’s 2.1 changes to `src/` do not affect this video-only training path.

### Differences from upstream

In `app/vjepa_2_1/train.py` and `utils.py`:

1. **`model.n_output_distillation`** (1 or 4) sets how many encoder layers are used as prediction
   targets and is applied consistently to the encoder, predictor, and target normalization. Upstream always built the model with 4; its now-unused `levels_predictor` option affected only target normalization. The EchoJEPA checkpoint uses 1 (the last
   layer only).
2. **`meta.init_checkpoint`** strictly initializes the encoder, target encoder, and predictor from another run’s weights while starting fresh training state. It cannot be used for cooldown runs or together with
   `meta.read_checkpoint`; see [Checkpoint loading](#checkpoint-loading).
3. **Checkpoint selection.** A cooldown always loads its checkpoint or fails; upstream
   silently trained from random weights when `anneal_ckpt` was missing. `meta.read_checkpoint` is invalid during cooldown or when `meta.load_checkpoint` is `False`, cases that upstream silently ignored. Any explicitly specified `read_checkpoint` must exist. This also avoids the upstream bug where the presence of `latest.pth.tar` could trigger a load attempt with a `None` path when checkpoint loading was disabled.
4. **Resumes are strict.** Restoring a run (a restart, or a cooldown starting from
   `anneal_ckpt`) fails on any missing, unexpected or mis-shaped tensor, on optimizer
   groups that do not match, and on a gradient scaler present on only one side. Upstream
   skipped such tensors, kept fresh weights for mis-shaped ones and silently started a new
   optimizer.
5. **Horizontal flips are off**, through `data_aug.random_horizontal_flip`. Upstream
   always flipped, which produces mirrored echocardiographic views that are not produced by the scanner.
6. **The context-loss ramp is configurable** (`model.lambda_start_iter` and
   `model.lambda_end_iter`). Upstream fixed it at steps 15k→30k, which were sized for
   runs of about 300k steps.
7. **`data.persistent_workers` reaches the loader.** Upstream ignored it, so each rank
   restarted its loader workers at every pass over the data.
8. **Fixes.**
   - The encoder width is read from the backbone; upstream failed for `vit_base`.
   - `normalize_nested` recursed one list level too far. It only runs with
     `model.normalize_predictor`, which is off here.
   - Periodic checkpoint filenames now match the number of completed epochs stored in them: `e5.pth.tar` after
     5 epochs, where upstream wrote `e4.pth.tar`.

In the shared launcher (`app/main.py`, also used by `app/vjepa`):

9. **`python -m app.main` waits for every rank and exits non-zero if any fails.** Once a
   rank fails, the others are stopped. Upstream started the ranks and returned success
   after launching the ranks, so downstream commands could run even when training had failed.

Not used or not ported: the image branch (`img_data`, ImageNet), the Gram-loss options present in upstream cooldown configs but ignored by this trainer, and upstream’s V-JEPA 2.1 evaluation configs.

### Prerequisite: the #8 loader fix

Issue #6 retains low-FPS and short videos in the manifests; correct loading of those cases is handled separately by issue #8, so this PR does not change their loading behavior. Without #8, the shared loader fails on videos at 7 fps or below, causing the trainer to skip the entire batch, and pads short clips by repeating the final frame. It also logs manifest paths when datasets are created and when videos fail to load, which may identify patients. The #8 PR removes them; a failed video is reported by its
manifest line instead. **Merge #8 before starting either run.** The trainer still does not mask padding-derived tokens from the loss; changing that would modify the training objective and should be decided separately.

## Architecture

| | Value |
|---|---|
| Encoder | `vit_base`: width 768, 12 blocks, 12 heads, patch 16, tubelet 2, RoPE (`use_rope`, `interpolate_rope`), `uniform_power`, modality embeddings. |
| Encoder parameters | 86,833,152, of which 591,360 serve only the image path (`patch_embed_img`, `img_mod_embed`). These are loaded but unused in video-only training. |
| Predictor | depth 12, width 384, 12 heads, 2 mask tokens, dense loss over all tokens (`predict_all`), `n_output_distillation: 1`; 22,182,528 parameters. |
| Target encoder | EMA copy of the encoder (momentum 0.99925). |
| Input (this pilot) | 16 frames at 8 fps, 224 × 224, ImageNet mean/std → 8 × 14 × 14 = 1,568 tokens. The checkpoint's original training inputs are unknown; see [Published checkpoint](#published-checkpoint). |
| Masking | Upstream 2.1 multi-block masks: 8 blocks at spatial scale 0.15 and 2 blocks at 0.7, full temporal extent. |

## Published checkpoint

| | |
|---|---|
| File | `vjepa2_1_vitb_mimic_pt169_c60.pt`, 1,651,185,340 bytes |
| SHA-256 | `40384a17d3d32375142e3bd3a43d2f5bfa2b174b44c6960958bffd731e9624c1` |
| Official source | [EchoJEPA checkpoints](https://drive.google.com/drive/folders/1RFEXMe8TTcABMBz4H_qtiLB43K9jD_lf) → `vjepa 2.1/` ([file](https://drive.google.com/file/d/102kUdwIAv9Sw0QlSyThHQwyrPtVMe9lh/view), uploaded 2026-06-18). The link comes from the bowang-lab/EchoJEPA README at `77208e1` |
| Architecture | V-JEPA 2.1 ViT-B/16 with the predictor above. Keys carry the DDP `module.` prefix, which the loader strips |
| Contents | `encoder`, `target_encoder`, `predictor`, plus the training state `opt`, `scaler`, `epoch` = 60, `itr`, `loss` = 0.589, `batch_size` = 128, `world_size` = 8 and `lr` = 1.75e-4 (final optimizer LR 1e-6) |
| Training domain | As stated by the authors, not verifiable from the weights: [MIMIC-IV-ECHO](https://physionet.org/content/mimic-iv-echo/0.1/), about 525k echo videos from 4,579 patients (Beth Israel Deaconess, 2017–2019), per the file name and the EchoJEPA README. By the README's naming convention, the name encodes 169 pretraining and 60 cooldown epochs; the stored `epoch` = 60 is consistent with the cooldown count |
| Original training inputs | **Unknown.** EchoJEPA did not publish the ViT-B 2.1 config, and the weights cannot reveal the frame rate, clip length, resolution or normalisation they were trained with. They fix only 3-channel input, 16 px patches and a 2-frame tubelet: with RoPE there is no learned position table, and no weight shape depends on resolution or frame count. The pilot's 224 px, 16 frames at 8 fps and ImageNet normalisation are **inferred** from EchoJEPA's ViT-L MIMIC config (`configs/train/vitl16/pretrain-mimic-224px-16f.yaml`). Its batch 128 × 8 GPUs and LR 1.75e-4 match the values stored in this checkpoint, which is consistent with, but does not prove, the same inputs. The pilot's own inputs are fixed and documented, so its runs are reproducible either way. If the checkpoint was trained on other inputs, only the continued arm starts off-distribution |

**License: unknown.** The EchoJEPA README identifies this download, but no licence or
redistribution terms are stated for the weight file, and neither of the documents below
settles them:
- The bowang-lab/EchoJEPA repository holds only `APACHE-LICENSE`, while its README says
  most of the code is MIT, inherited from V-JEPA 2. Both are code licences; neither
  mentions the checkpoint files.
- The weights are trained on MIMIC-IV-ECHO, which is released under the
  [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimic-iv-echo/view-dua/0.1/).
  That data use agreement limits the data to lawful scientific research and forbids
  sharing access or attempting re-identification. It governs the data, and does not state
  terms for the weight file.

The following is our precaution, not a licence and not a confirmed requirement. It applies
until the authors state terms:
- use the checkpoint for internal research only;
- do not redistribute the file;
- cite EchoJEPA;
- ask the authors before any commercial, clinical or public release built on it.

Separately, Meta's file headers reference an MIT `LICENSE` file that this repository
lacks (bowang-lab's lacks it too).

### Checkpoint loading

The smoke test builds the model exactly as the trainer does and checks the checkpoint against it.
A file whose SHA-256 differs from `--sha256` fails without being loaded. With `--forward`
it also runs one masked training forward pass on two random clips, in float32 on CPU:
- using the config's masks, the target encoder, the context encoder and the predictor;
- the training loss: the prediction loss plus the context loss, distance-weighted as
  configured, at the context-loss weight the ramp ends on (0.5). The loss function is a
  copy of the trainer's, and a test runs both on the same inputs.

The fixed seed makes the synthetic clips, masks, and resulting numbers reproducible. Because the inputs are random, the resulting loss only verifies that the computation succeeds and remains finite; it does not measure checkpoint quality. The test needs no GPU:

```bash
python -m app.vjepa_2_1.check_checkpoint \
    --fname configs/train_2_1/vitb16/pretrain-continued-224px-16f.yaml \
    --checkpoint /path/to/vjepa2_1_vitb_mimic_pt169_c60.pt \
    --sha256 40384a17d3d32375142e3bd3a43d2f5bfa2b174b44c6960958bffd731e9624c1 --forward
```

Result on 2026-09-25 (excerpt; two runs gave identical output):

```
encoder <- `encoder`: 0 missing, 0 unexpected, 0 mis-shaped keys
predictor <- `predictor`: 0 missing, 0 unexpected, 0 mis-shaped keys
target_encoder <- `target_encoder`: 0 missing, 0 unexpected, 0 mis-shaped keys
Strict load with the trainer's loader: ok.
Masked forward on random clips (2, 3, 16, 224, 224), seed 0: mask 1: 416 context / 1136 predicted tokens; mask 2: 320 context / 1248 predicted tokens.
    prediction loss 0.6061 + 0.5 x context loss 0.3703 (distance-weighted) = 0.7913
    finite=True
RESULT: PASS
```

Each mask's token counts are cut to the shortest in the batch, as in training, so the
context and predicted tokens add up to at most 1,568.

**The comparison uses model weights only.** What each run loads:

| Run | Loads | Optimizer, scaler, epoch; LR, WD, EMA and mask schedules |
|---|---|---|
| Scratch, first start | Nothing: random initialisation, seed 239. | Fresh, epoch 0. |
| Continued, first start | `init_checkpoint`: encoder ← `encoder`, target encoder ← `target_encoder`, predictor ← `predictor`, strict. | Fresh, epoch 0. The file's epoch 60, annealed LR, Adam moments and scaler are never read. |
| Pretraining, restarted (either arm) | Its own `folder/latest.pth.tar` (`load_checkpoint: true`); `init_checkpoint` is ignored. | Weights, optimizer, scaler and epoch restored strictly (any mismatch fails the run); schedules advanced to that epoch. Not bit-exact: see below. |
| Cooldown, first start | `anneal_ckpt`, the same arm's pretraining `latest.pth.tar`: weights, optimizer and scaler, strict. | Epoch reset to 0; the cooldown's own LR decay and schedules start. |
| Cooldown, restarted | Its own `latest.pth.tar`. | As for a pretraining restart. |

**A restart is not bit-exact.** Checkpoints are saved only at epoch boundaries, so a restart may repeat up to 300 steps. Model, optimizer, scaler, LR/weight-decay/EMA schedules, and the mask-size seed counter are restored, but the DataLoader position and RNG states are not. The restarted epoch begins a newly shuffled pass over train.csv, so sample order and subsequent training can differ from an uninterrupted run and from the other experimental arm. Any restart should therefore be reported.

`meta.read_checkpoint` is a full resume of a named file. Never point it at the published
checkpoint: the run would restore that file's optimizer (LR already 1e-6), scaler and
epoch counter 60, past this config's 13 epochs, instead of starting fresh. The trainer
refuses `read_checkpoint` together with `init_checkpoint`, in a cooldown, and without
`meta.load_checkpoint: true`.

The 2.1 trainer has no W&B logging, so no W&B run is created or resumed. Its logs are
`folder/log_r<rank>.csv` (one per GPU) and stdout.

## Shared configuration

Both arms use the same configuration. Besides each run's `folder`, the pretraining configs
differ only in `meta.init_checkpoint` and the cooldown configs only in
`optimization.anneal_ckpt`, as `tests/utils/test_vjepa_2_1_baseline.py` checks:

| | Both arms |
|---|---|
| Data | One `train.csv` from the issue #6 manifests (patient-level split; val/test patients never listed). No `datasets_weights`, so each pass is a shuffle of `train.csv` split across the 4 GPUs, not a draw with replacement. |
| Clips | `dataset_fpcs: [16]` (the 2.1 trainer's frames-per-clip), `fps: 8`, 224 px, tubelet 2. |
| Augmentation | Random resized crop with scale [0.5, 1.0] and aspect ratio [0.9, 1.1] (EchoJEPA's narrowed ranges); no flips, auto-augment, motion shift or random erasing. |
| Masking | As in [Architecture](#architecture). |
| Objective | Pretraining uses dense L1 prediction against the final encoder layer. Context loss starts disabled, ramps from 0 to 0.5 between steps 200 and 400, then stays at 0.5, with upstream-style distance weighting. Cooldown uses the same 0.5 context-loss weight from the start but does not use the pretraining distance weighting. |
| Batch / hardware | 64 per GPU × 4 GPUs = 256; bfloat16; activation checkpointing (without it, batch 64 does not fit a 48 GB GPU). |
| Optimizer | AdamW, weight decay 0.04, EMA momentum 0.99925. LR 1e-5 → 4.375e-5 over 1.5 epochs (450 steps), then constant. Note: 4.375e-5 is EchoJEPA's 5.25e-4 at batch 3072 scaled linearly to 256 (Meta's ViT-B recipe uses 6e-4). The checkpoint's stored LR and batch (1.75e-4 at 1024) fit the same rule. |
| Duration | Pretraining 13 epochs × 300 steps = 3,900 steps. Cooldown 3 × 300 = 900 steps, with the LR decaying linearly from 4.375e-5 to 1e-6. |
| Seed | `meta.seed: 239` in both arms. Each pass is shuffled from `DistributedSampler`'s fixed seed (0, not `meta.seed`) and the number of the epoch in which the pass starts. The order does not depend on the model, so both arms see the same videos in the same order, unless a video fails to load or a run is restarted. Crops and masks come from the same distributions but are not guaranteed identical batch for batch: loader workers draw mask seeds from a shared counter in whatever order they run |

Compared with upstream's `vitb16` configs, this recipe changes:

- the data: echo `train.csv` instead of Kinetics-710, SSv2, HowTo100M and ImageNet;
- the clips: 224 px at 8 fps instead of 256 px at 4 fps, and 16 frames in the cooldown
  instead of 64;
- the objective: one distillation level instead of four;
- the augmentation: narrower ranges, no flips;
- the schedule: an LR scaled to batch 256, and a far shorter duration.

Both arms train for 3,900 pretraining and 900 cooldown steps. The budget was sized from a
throughput test on 4 × A6000. A different budget must change `epochs` (or
`ipe`) identically in all four configs and keep the ramp at about 5–10 % of the
pretraining steps.

## Launch

Save the script below outside the repository, and fill in the three paths (outputs, data and
checkpoints stay outside the repository). Activate an environment with only
[requirements-pilot.txt](../requirements-pilot.txt) installed, then
run the script with `bash` from inside the checkout. Do not paste it into an interactive shell: `set -e` would close the
shell at the first failure.

```bash
#!/usr/bin/env bash
set -euo pipefail

RUNS=/path/to/runs/pilot          # outputs
MANIFESTS=/path/to/manifests      # issue #6 output directory: train.csv, manifest_info.json
CKPT_DIR=/path/to/checkpoints     # holds vjepa2_1_vitb_mimic_pt169_c60.pt
CKPT_SHA256=40384a17d3d32375142e3bd3a43d2f5bfa2b174b44c6960958bffd731e9624c1
GPUS=(cuda:0 cuda:1 cuda:2 cuda:3)

cd "$(git rev-parse --show-toplevel)"

# Only committed code runs: nothing modified, untracked or ignored in the checkout except
# Python caches, since any other file (a stray torch.py, sitecustomize.py or .pyc) could
# shadow an import.
dirty=$(git status --porcelain --ignored | grep -v -E '^!! (.+/)?__pycache__/$' || true)
if [ -n "$dirty" ]; then
  echo "The checkout is not clean; use a fresh clone or git worktree. Found:" >&2
  echo "$dirty" >&2
  exit 1
fi

# The environment: the Python version, every package pinned in requirements-pilot.txt, and
# nothing on the import path besides this checkout and the environment itself (no
# PYTHONPATH, user site-packages or installed copy of this repository).
python - requirements-pilot.txt <<'PY'
import os, platform, sys
from importlib.metadata import PackageNotFoundError, version
env = tuple(os.path.realpath(p) + os.sep for p in {sys.prefix, sys.base_prefix})
wrong = [f"import path outside the environment: {p}" for p in sys.path[1:]
         if p and not (os.path.realpath(p) + os.sep).startswith(env)]
for line in open(sys.argv[1]):
    if line.startswith("# python: "):
        pinned = line.split(":")[1].strip()
        if not platform.python_version().startswith(pinned + "."):
            wrong.append(f"python {platform.python_version()}, pinned {pinned}")
    elif line.strip() and not line.startswith("#"):
        name, pinned = line.strip().split("==")
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not installed"
        if installed != pinned:
            wrong.append(f"{name} {installed}, pinned {pinned}")
sys.exit("Not the pilot environment (requirements-pilot.txt):\n" + "\n".join(wrong) if wrong else 0)
PY

# The four configs with placeholders resolved, kept aside until the checks below pass.
resolved=$(mktemp -d)
trap 'rm -rf "$resolved"' EXIT
for f in configs/train_2_1/vitb16/*-224px-16f.yaml; do
  sed -e "s#/your_folder#$RUNS#g" -e "s#/your_manifests#$MANIFESTS#g" -e "s#/your_checkpoints#$CKPT_DIR#g" \
      "$f" > "$resolved/$(basename "$f")"
done

# Inputs: the published checkpoint and the frozen train.csv of issue #6.
echo "$CKPT_SHA256  $CKPT_DIR/vjepa2_1_vitb_mimic_pt169_c60.pt" | sha256sum -c
train_sha256=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['outputs']['train.csv']['sha256'])" \
    "$MANIFESTS/manifest_info.json")
echo "$train_sha256  $MANIFESTS/train.csv" | sha256sum -c

# Provenance: the commit, the inputs and the resolved configs. The first run of $RUNS
# records them; every re-run (a restart) must match them exactly. A first run needs an
# absent or empty $RUNS: the trainer would resume any latest.pth.tar already there.
provenance="$(git rev-parse HEAD)  commit
$CKPT_SHA256  vjepa2_1_vitb_mimic_pt169_c60.pt
$(sha256sum < "$MANIFESTS/manifest_info.json" | cut -d' ' -f1)  manifest_info.json
$train_sha256  train.csv
$(cd "$resolved" && sha256sum -- *.yaml)"
if [ -f "$RUNS/provenance.txt" ]; then
  if [ "$provenance" != "$(cat "$RUNS/provenance.txt")" ]; then
    echo "$RUNS was started with other code, inputs or configs (< recorded, > now):" >&2
    diff "$RUNS/provenance.txt" - <<< "$provenance" >&2 || true
    exit 1
  fi
elif [ -e "$RUNS" ] && [ -n "$(ls -A "$RUNS")" ]; then
  echo "$RUNS is not empty but has no provenance.txt; use a new, empty RUNS folder." >&2
  exit 1
fi

# The trainer's strict load of the checkpoint, and one masked training forward pass.
python -m app.vjepa_2_1.check_checkpoint --fname "$resolved/pretrain-continued-224px-16f.yaml" \
    --sha256 "$CKPT_SHA256" --forward

mkdir -p "$RUNS"
echo "$provenance" > "$RUNS/provenance.txt"
python -m pip list --format=freeze > "$RUNS/environment.txt"
mkdir -p "$RUNS/configs"
cp "$resolved"/*.yaml "$RUNS/configs/"

# A run is complete when its latest.pth.tar holds all the epochs of its config.
complete() {
  python - "$1" <<'PY'
import os, sys, torch, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
path = os.path.join(cfg["folder"], "latest.pth.tar")
done = torch.load(path, map_location="cpu", weights_only=False)["epoch"]
sys.exit(0 if done == cfg["optimization"]["epochs"] else f"{path}: {done} of {cfg['optimization']['epochs']} epochs")
PY
}

# Training. Each command waits for its run and fails if any rank fails.
for arm in scratch continued; do
  for phase in pretrain cooldown; do
    cfg="$RUNS/configs/$phase-$arm-224px-16f.yaml"
    python -m app.main --fname "$cfg" --devices "${GPUS[@]}"
    complete "$cfg"
  done
done
```

- **Run folders.** `$RUNS/configs` holds the four resolved configs, and `app.main` also
  writes each run's config to `params-pretrain.yaml` in its run folder.
- **Provenance.** `$RUNS/provenance.txt` records the commit, the checksums of the
  checkpoint, `manifest_info.json` and `train.csv`, and those of the four resolved configs.
  The script writes it once the checks pass, just before training starts. Reusing the same run directory is allowed only if all recorded provenance still matches; otherwise the launcher stops and reports the differences. A first run requires a new or empty `$RUNS` directory. Non-empty directories without `provenance.txt` are rejected to avoid accidentally resuming older checkpoints. Any intentional input or config change should use a new `$RUNS` directory. Only `train.csv` is hashed, not the video files it references.
- **Restarts.** Re-running the script resumes each run from its own `latest.pth.tar`,
  saved at the end of every epoch (300 steps); a finished run only reloads and exits.
  A restart is not bit-exact; see [Checkpoint loading](#checkpoint-loading).
- **Environment.** `requirements.txt` allows any `torch>=2` and leaves most packages
  unpinned, so the pilot pins its own: `requirements-pilot.txt` lists Python 3.12 and the
  exact version of every package in the environment that ran the smoke test and the
  tests. The script refuses any other version, and writes the full
  package list to `$RUNS/environment.txt`. It also refuses an import path that reaches
  code outside the checkout and the environment: `PYTHONPATH`, user site-packages, or an
  installed copy of this repository. An editable install (`pip install -e .`) in another
  working copy puts that copy's `src/` on every process's import path. The GPU driver and
  the operating system are not pinned.
- **NCCL hangs.** If NCCL hangs while setting up shared memory, add `export NCCL_CUMEM_ENABLE=0` near the top of the script.
- **Tests:** `python -m unittest tests.utils.test_vjepa_2_1_baseline tests.utils.test_vjepa_2_1_utils`.

## Newer-version assessment (checked 2026-09-25)

**No V-JEPA version newer than 2.1 is published.** Sources reviewed:

| Source | Finding |
|---|---|
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | Last commit `204698b` on 2026-03-23, a figure fix. The CHANGELOG's latest entry is "0.0.2 – Release of V-JEPA 2.1". No tags or releases. The side branches `vjepa2_1`, `security-fix-black` and `Adrien987k-patch-1` hold only March 2026 README, config and formatting commits |
| [V-JEPA 2.1 paper, arXiv 2603.14482](https://arxiv.org/abs/2603.14482) | The latest revision is v3 (2026-06-11); no successor paper |
| [facebookresearch repositories matching "jepa"](https://github.com/orgs/facebookresearch/repositories?q=jepa) | `eb_jepa` (an examples library), `jepa-wms` (world models for planning), `jepa-intuitive-physics`, `td_jepa` (reinforcement learning), `locate-3d`, `jepa` (V-JEPA 1) and `ijepa`. None is a newer video-encoder generation |
| [Hugging Face, facebook](https://huggingface.co/facebook) | V-JEPA 2 models from 2025 only; no 2.1 or later |
| [Meta AI V-JEPA page](https://ai.meta.com/research/vjepa/) | Presents V-JEPA 2 |
| [bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA), [EchoJEPA paper, arXiv 2602.02603](https://arxiv.org/abs/2602.02603) | The latest commit, `77208e1` (2026-06-18), added the V-JEPA 2.1 checkpoints to the README. The paper (v4, 2026-02-10) covers the V-JEPA 2 models only |
| Web search for "V-JEPA 3" and "V-JEPA 2.2" | No release or announcement |

Within 2.1, Meta also publishes a ViT-B checkpoint, `vjepa2_1_vitb_dist_vitG_384.pt`. It
was trained on natural video and images at 384 px, distilled from a ViT-G. It is not a
drop-in initialisation for the continued arm, for two reasons:
- its predictor outputs ViT-G teacher features (1664-d), not ViT-B features;
- it has never seen echocardiography.