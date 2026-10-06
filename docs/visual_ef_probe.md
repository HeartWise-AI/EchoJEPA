# Frozen Visual EF probe

This document describes how a V-JEPA 2.1 encoder is evaluated on Visual EF (issue #13): an
attentive regression head is trained on top of the **frozen** encoder, chosen on validation,
and scored once on the held-out test split. The pilot compares three encoders with it: the
published EchoJEPA ViT-B checkpoint as is, that checkpoint after continued pretraining on our
data, and a ViT-B pretrained from scratch on our data
([docs/pretraining_baseline.md](pretraining_baseline.md)).

## Protocol

| | |
|---|---|
| Data | The issue #6 manifests (PR #7). [`data/make_probe_manifests.py`](../data/make_probe_manifests.py) keeps the A4C videos of each split and labels each with its study's Visual EF, z-scored with the train split's mean and standard deviation. A video without an EF, or with an EF outside (0, 100], is left out and counted per split in `probe_info.json`. A patient found in more than one split stops the script; the probe checks the video index again when it starts. `probe_info.json` records the SHA-256 of the source `videos.csv` and of every file written. |
| Encoder | V-JEPA 2.1 `vit_base`, built as in the pretraining configs and loaded strictly from the checkpoint's `target_encoder` ([`vjepa_2_1_encoder.py`](../evals/video_classification_frozen/modelcustom/vjepa_2_1_encoder.py)): any missing, unexpected or mis-shaped key stops the run. It runs in eval mode, without gradients. The probe stops if any encoder parameter requires a gradient or sits in an optimizer. |
| Head | Attentive probe: one query, 4 blocks, 16 attention heads, then a linear layer to one output. It is trained on the z-scored EF with the smooth L1 (Huber) loss. |
| Settings | Six heads train side by side on the same frozen features, one per learning rate {1e-4, 5e-5} × weight decay {0.01, 0.1, 0.4}. AdamW, learning rate and weight decay on cosine schedules, no warmup, 20 epochs, batch 4 per GPU on 4 GPUs. Each GPU's last incomplete training batch is dropped. |
| Clips | Each video gives 2 clips of 16 frames (every second frame, 224 px), one from each half of the video: at a random position in training, at the start of each half in validation and test, so evaluation reads the same frames every time. Training adds random augmentation; evaluation has none. |
| Aggregation | The encoder embeds each clip separately; the two clips' tokens are concatenated along time, and the attentive probe pools them into one prediction per video (`ClipAggregation`). Training takes one loss step per video, on that prediction. A study's prediction is the mean over its videos. Every video of a study carries the study's EF. |
| Selection | After each epoch, every validation video is scored once by every head. The epoch and head with the lowest **per-study validation MAE** are kept in `best.pt`, so selection matches the per-study results and targets (`evaluation.selection`: `study`, or `video` for per-video MAE). The pilot selected per video. |
| Threshold | Reduced EF is a reference EF below 40. A predicted EF below the threshold flags it. The threshold maximises sensitivity + specificity − 1 over the validation studies (videos, when `targets.level` is `video`), scored with the selected head; ties go to the threshold closest to 40. |
| Fingerprint | Every checkpoint saves the probe's fingerprint: the SHA-256 of the encoder checkpoint, of the train, validation and test manifests and of the video index; the EF normalization; and the settings: how the encoder is built (`model_kwargs` without the checkpoint path, so also which weights of the checkpoint are loaded and options such as `is_causal`), head, clips, optimization, selection rule, seed and number of GPUs; and the protocol, how each split's clips are drawn and batched, with a version the code raises when that changes. A resumed probe must match `latest.pt`, and the test must match `best.pt`, the number of GPUs aside. Anything else stops the run, so a probe is never continued or tested with another encoder, other data or under another protocol (one saved before the protocol was recorded, when evaluation drew random clip positions, is refused). `dataset_test` is therefore set before training. |
| Manifests | Before training, each manifest is checked against the video index: a video the index lacks or files under another split stops the run. The test manifest is checked the same way before the test. |
| Test | Run once, after the last epoch (`--test_only`). It refuses an unfinished probe, and a probe that already has `test_metrics.json`. Each test video is scored exactly once; a video that fails to load is counted, not replaced. The guard is per probe folder: it cannot stop another folder from scoring the same test split. |
| Metrics | Per study (primary) and per video: MAE and Pearson r (primary), RMSE, R² and bias (mean of prediction − reference), in EF points. For EF < 40: AUROC, AUPRC, and sensitivity and specificity at the validation threshold. Error and bias by reference EF range: < 30, 30–40, 40–50, 50–60, ≥ 60. |
| Targets | Per study: MAE < 5 and AUROC > 0.95 for EF < 40. Agreed on 2026-10-02 and 2026-10-03, after the preliminary test results of the pilot had been seen. `targets.level` (`study` or `video`) sets the level the targets are checked at; the error by EF range and the plots use the same level. |
| Seed | `meta.seed` (0): probe initialisation, data order and augmentation. Every checkpoint saves each GPU's random number state, so a resumed probe continues exactly as an uninterrupted one. On CPU a run, and a run resumed after an interruption, repeat exactly ([smoke test](#tests)). GPU runs are seeded but not bit-exact. |

The shipped config is [`configs/eval/vitb16/lvef-a4c-224px-16f.yaml`](../configs/eval/vitb16/lvef-a4c-224px-16f.yaml).
Its `/your_...` placeholders and `data.target_mean` / `data.target_std` (from
`probe_info.json`) must be filled in.

## Outputs

A probe writes to `<folder>/video_classification_frozen/<tag>/`:

| File | Contents |
|---|---|
| `params-probe.yaml`, `run_info.json` | The full config, and the run's record: code commit, encoder checkpoint and manifest checksums, normalization, split sizes, seed, parameter counts, settings, selection rule and targets. Written when the probe starts; a resumed probe keeps them. |
| `log_r0.csv` | Per epoch: training MAE and the validation MAE selection uses (`val_study_mae`), of the best head. |
| `latest.pt`, `epoch_NNN.pt`, `best.pt` | The six heads with their optimizer states, the probe's fingerprint and each GPU's random number state. Each also holds every head's validation MAE per study and per video for its epoch; `best.pt` is the epoch selected. About 2 GB each for ViT-B. Each is written to a temporary file, then renamed; `latest.pt` last, so it records an epoch only once that epoch's other files are written. |
| `test_metrics.json` | Epoch and head tested, the validation MAE they were selected on, the SHA-256 of `best.pt`, the threshold and the level it was chosen at, validation and test metrics per study and per video, error by EF range, targets met, and the run record of the test. |
| `test_predictions.csv`, `val_predictions.csv` | One row per video: path, study, reference and predicted EF. **Paths and study IDs can identify patients: these files stay with the data.** |

## Weights & Biases

With `meta.wandb_project` and `meta.wandb_entity` set, the probe logs to its own run. A
resumed probe, and its test, rejoin that run.

- **Config**: the probe config with every path, absolute or relative, cut to its file or folder
  name, and `run_info`, as in `run_info.json`. `init_wandb` applies the same rule to every run,
  the V-JEPA 2.1 trainer's included.
- **Each epoch**: training and validation MAE and loss, validation MAE per study, learning
  rate, the best validation MAE per study so far (`probe/val_study_mae_best`) and its epoch, and
  the number of validation videos that failed to load. Each value is logged for every head, and for the best head.
- **Test**: every test metric (`probe/test/study_*`, `probe/test/video_*`), the threshold and
  the targets met. Also, at the targets' level, a predicted-versus-reference scatter plot (an
  image), the residual histogram, and the error by EF range as values
  (`probe/test/study/by_reference_range/<range>/{n,mae,bias}`).
- **The trained probe**: `probe_checkpoint/*` in the run summary (file name, SHA-256 of
  `best.pt`, its epoch and head, the head's validation MAE per study, and the encoder's SHA-256). Also a
  `probe-checkpoint` artifact whose metadata holds that record. It has no file: wandb's
  manifest of an artifact file records the path it was staged from on the machine. Both are logged at the
  end of training and again by the test.

The run receives no filesystem path, video path, patient, study or accession identifier.
Console output, system metadata (command line, working directory, host), the git remote and
the package list are not sent. The commit is recorded in `run_info`. The scatter plot is an
image whose points have no identifier or order. Nothing is logged as a wandb Table: wandb keeps
each Table in an artifact whose manifest records the folder it was staged from on the machine.

## Reproduce

```bash
# 1. Probe manifests from the issue #6 output (outside the repository).
python data/make_probe_manifests.py --manifests <issue_6_output_dir> --out-dir <probe_dir> --views A4C

# 2. A copy of the config for each encoder: set folder, model_kwargs.checkpoint, the dataset_*
#    files and data.video_index, and copy target_mean and target_std from <probe_dir>/probe_info.json.
#    For the published EchoJEPA checkpoint, set model_kwargs.checkpoint to vjepa2_1_vitb_mimic_pt169_c60.pt:
#    its target_encoder loads like a cooldown checkpoint's.

# 3. Train the probe, then test it once.
python -m evals.main --fname <probe config> --devices cuda:0 cuda:1 cuda:2 cuda:3
python -m evals.main --fname <probe config> --devices cuda:0 cuda:1 cuda:2 cuda:3 --test_only
```

Both commands exit non-zero if any GPU's process fails. The others are then stopped, so a
failed probe never looks finished. A stopped probe resumes from `latest.pt` when the same
command is run again: an epoch stopped before its `latest.pt` was written is trained again, and
`best.pt` is put back to the best epoch `latest.pt` records if that epoch had replaced it. The test
refuses a probe whose `best.pt` is not that epoch.

## Tests

| File | Covers |
|---|---|
| [`tests/utils/test_probe_metrics.py`](../tests/utils/test_probe_metrics.py) | Metrics on cases worked out by hand: MAE, RMSE, R², Pearson r, study aggregation, AUROC with ties, AUPRC, sensitivity and specificity, the threshold rule, ranges. |
| [`tests/utils/test_probe_protocol.py`](../tests/utils/test_probe_protocol.py) | Metrics weighted by every video (7 errors of 1 and one of 9 give 2.0, not 5.0); validation scoring each video once, on one rank and on two; encoder weights unchanged after training; a trainable encoder refused; the head's output shape; the launcher failing with a rank; a video index with an empty value or an unknown split refused. |
| [`tests/utils/test_probe_test_split.py`](../tests/utils/test_probe_test_split.py) | The test uses the epoch and head chosen on validation: per study, or per video when configured, with validation scores where the two rules disagree; a probe not tested under the other rule; study-level metrics, the threshold from validation, failed videos counted, unfinished and already-tested probes refused, the validation record across a resume. A checkpoint replaced whole or not at all; an epoch whose `best.pt` failed to save trained again; a `best.pt` written after the last `latest.pt` put back on resume, and one that is not the selected epoch refused at the test. A probe resumed or tested with another encoder, encoder setting (`checkpoint_key`, `is_causal`), normalization, manifest or protocol version refused, as is one saved before the protocol was recorded or without a fingerprint; the record a probe started with kept; a train manifest listing validation videos refused; the wandb reference to `best.pt`; targets per video, with the threshold chosen on validation videos. |
| [`tests/utils/test_probe_wandb.py`](../tests/utils/test_probe_wandb.py) | What each epoch logs, the run record, no path in the config (relative paths included), the privacy settings, resume, the normalization required and checked against `probe_info.json`. |
| [`tests/utils/test_probe_smoke.py`](../tests/utils/test_probe_smoke.py) | End to end on a small fixture of real videos: manifests, the video loader, two epochs and the test. Two runs with the same seed give the same probe and numbers, and so does a run interrupted after its first epoch and resumed. Evaluation reads the same frames every time; training drops a last incomplete batch, evaluation keeps every video; a training manifest without a full batch is refused. |
| [`tests/data/test_make_probe_manifests.py`](../tests/data/test_make_probe_manifests.py) | View selection, z-scoring, exclusion counts, the patient-leakage check, the video index. |
