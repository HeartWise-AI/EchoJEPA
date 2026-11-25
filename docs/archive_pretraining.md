# Archive-scale ICM pretraining: protocol

This document fixes, before any run, how the archive-scale V-JEPA 2.1 ViT-B experiments are built,
compared and evaluated. It continues the 10k-study pilot ([pretraining_baseline.md](pretraining_baseline.md),
[visual_ef_probe.md](visual_ef_probe.md)); the pilot's cohorts, results and selection rules stay as they are.
Numbers that depend on measurements are filled in only from recorded runs.

## Decisions

| Item | Decision |
|---|---|
| Corpus | Standard TTE (`study_type == "ECHO CARDIAQUE AVEC DOPPLER"`), every eligible study of every training patient, longitudinal studies included, no EF label required, usable `OTHER` views kept. |
| Split | One patient-level split shared by JEPA, clinical probes and RADAR, ~70/10/20. The 10k cohort's patients keep their split; new patients are assigned deterministically towards the global target (`data/build_manifests.py`, `selection.mode: all_studies`). |
| Arms | Inclusive TTE videos vs confirmed B-mode-only videos (acquisition mode from DICOM ultrasound regions, `data/scan_acquisition.py`). Same patients, same split. |
| Initialization | Random for both arms, same seed. Published EchoJEPA stays the external reference encoder and the first RADAR backbone. |
| Model | V-JEPA 2.1 ViT-B/16, 224 px, 16 frames at ~8 fps, tubelet 2, objective and augmentations of the validated pilot recipe. |
| Fairness | Identical optimizer updates, global batch, schedules, seed, objective and augmentations across arms. Dataset size and exposures per unique video differ and are reported. |
| Encoder selection | The final checkpoint of a prespecified schedule and cooldown. Intermediate probes are diagnostics only. |
| Endpoints | Study-level EF MAE (percentage points) and RVSP MAE (mmHg), analysed separately; no composite. |

## Temporal padding ablation (decided before it runs)

Many videos are shorter than a 16-frame clip at ~8 fps, so part of each such clip is black padding.
Two policies are compared once, and one is applied to both corpus arms:

- **black** (baseline): chronological ~8 fps sampling, black padding at clip index `-1`, padding-only
  tubelets excluded from the loss.
- **loop** (candidate): the same real frames repeated from the clip start to fill the clip; recycled
  frames are real inputs and valid targets. See [temporal_padding.md](temporal_padding.md).

Setup: identical random initialization (seed 239), the pilot schedule of 3,900 pretraining + 900 cooldown
updates at global batch 256, the 10k cohort's `train.csv` (its patients are global training patients),
and the #13 EF probe on the 10k cohort's probe train/val sets, scored on **validation only**. RVSP
validation is added if its evaluator and labels are ready when the probes run. No test set is scored to
choose the policy.

Selection rule, fixed now:

1. Primary: study-level validation EF MAE of each arm's selected probe (#13 selection rules).
2. Adopt **loop** only if its study-level validation MAE is lower by at least 0.15 EF points **and** the
   paired study-level bootstrap 95% interval of (black - loop) excludes 0, **and** RVSP validation MAE (if
   available) is not worse by more than 0.5 mmHg, **and** the junction review
   (`app/vjepa_2_1/review_loop_junctions.py`, at least 30 short videos) finds no systematic artefact the
   reviewers judge unacceptable.
3. Otherwise, including an inconclusive or missing comparison, keep **black**.

Limitations: one seed, a short scratch schedule, a frozen probe, and a training loss that is not
comparable across policies because loop changes which tokens are targets.

## Evaluation protocol for archive-scale encoders

- Encoders: inclusive arm, B-mode arm, and published EchoJEPA as the external reference, each frozen and
  wrapped identically.
- EF: the #13 single-view A4C attentive probe (4 blocks, 16 heads, learning rate {1e-4, 5e-5} x weight
  decay {0.01, 0.1, 0.4}); head and epoch selected by study-level validation MAE.
- RVSP: early-fusion multi-view probe on grouped clips per study ([rvsp_multiview_probe.md](rvsp_multiview_probe.md)),
  one target per study, same grid and selection rule. RVSP is computed in reports from CW Doppler
  (4·TRV² + RA pressure); spectral traces are not in the video corpus, so this is prediction from available
  image features, not a Doppler measurement.
- Probe train/val sets: the 10k cohort's train/val studies (comparable with the pilot).
- Tests: the legacy 10k test set has been inspected; it is kept for historical comparison and labelled as
  such. Confirmatory results use a **fresh test set** of 2,000 patients newly assigned to the global test
  pool, frozen (with checksums) before any archive-scale encoder or probe exists. Each encoder/probe is
  scored on it once.
- Diagnostics: RMSE, bias, Pearson r, error by reference range, paired patient-level bootstrap intervals.
  Regurgitation severity is not an endpoint: the report grade fields mix encodings and are not yet reliable.

## Schedule

Epochs are the trainer's logical blocks of 300 updates (`optimization.ipe`), not passes over the data.
Both arms use the same number of updates, so the smaller B-mode arm (a subset of the inclusive arm) sees
each video more often.

Resolved recipe for both arms (the validated pilot recipe; only lengths change): LR 1e-5 -> 4.375e-5 linear
warmup, then constant (linear scaling of EchoJEPA's 1.75e-4 at batch 1024); cooldown decays it linearly to
1e-6 with a fixed context-loss weight and no distance weighting, loading the pretraining checkpoint
strictly; context-loss weight 0.5, ramped over 5%-10% of pretraining updates (the pilot's 200-400 of 3,900);
EMA 0.99925 constant; weight decay 0.04; `latest.pth.tar` every 300 updates and `e<N>.pth.tar` every 6,000
for diagnostics only.

Restarts: a relaunch resumes `latest.pth.tar` (model, optimizer, scaler, epoch; schedules are fast-forwarded)
but not the exact data order or RNG state (see [pretraining_baseline.md](pretraining_baseline.md)), so up to
300 updates may repeat with different batches.
