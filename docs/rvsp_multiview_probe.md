# Frozen multi-view RVSP probe

This document describes how a V-JEPA 2.1 encoder is evaluated on right ventricular systolic
pressure (RVSP): an early-fusion attentive regression head is trained on top of the **frozen**
encoder over several views of each study, chosen on validation, and scored once on the held-out
test split. It follows the protocol of the frozen Visual EF probe ([docs/visual_ef_probe.md](visual_ef_probe.md),
issue #13) and imports that probe's selection, fingerprint, checkpoint, resume and test-once code
([`evals/video_classification_frozen/eval.py`](../evals/video_classification_frozen/eval.py)), which it leaves unchanged.
What differs is the unit: one manifest row, one prediction and one target per **study**.

## What the probe sees

RVSP in the reports is derived from continuous-wave Doppler, 4·TRV² + RAP (tricuspid regurgitation
peak velocity, plus an estimated right atrial pressure). The probe does not receive that
measurement: its inputs are the videos of the view slots below, so a result is a prediction of RVSP
from the image features of those videos, not a Doppler-pressure measurement. A slot with
`acq: any` admits every acquisition category, including spectral Doppler if such videos exist under
its view label; `probe_info.json` counts the acquisition categories of the chosen videos per split
(`splits.<split>.acquisition`), and results must be reported with those counts. A good RVSP result
alone does not show that an encoder reads spectral measurements.

## Protocol

| | |
|---|---|
| Data | [`data/make_rvsp_probe_manifests.py`](../data/make_rvsp_probe_manifests.py) reads the master video index (`patient_id`, `study_id`, `split`, `view`, `video_path`, `reason`, `acq_category`, `acq_confidence`, `acq_vendor_validated`), the study labels (`study_id`, `rvsp` in mmHg, one row per study) and the probe's study list (`study_id`, `split`). A study whose split differs from the index's, a study absent from the index, a study or label listed twice, a non-numeric RVSP, and a patient with probe studies in two splits stop the script. Only verified videos (`reason` null) are used. |
| Slots | An ordered list of slots, each a set of view labels, the acquisition categories it admits (`any` or a list) and the number of videos it holds (`clips`). Categories are the effective ones of pretraining (`effective_acq_category` in `data/build_manifests.py`): a B-mode the DICOM header does not confirm, or that comes from a vendor outside `acquisition.bmode_vendors`, is unresolved (`unknown`) and fills no B-mode slot. Default: `A4C_bmode` (A4C, B-mode), `A4C_color` (A4C, colour flow), `RVINF` (any), `PSAX_bmode` (PSAX, B-mode), `PSAX_AV` (any), `SUBCOSTAL` (any), one video each. `--slots` takes a YAML or JSON file of the same form. A view label no verified video has stops the script (a typo would otherwise leave a slot always empty). |
| Choice | Within a slot, a study's candidate videos are ranked by SHA-256 of (`--seed`, slot name, video path) and the first `clips` are taken: deterministic, independent of row order and of the other studies, and identical for every encoder. Slots are filled in order; a video fills at most one position. |
| Target | One RVSP per study, z-scored with the mean and standard deviation (ddof 0) of the probe **train** studies kept. A study is left out, and counted per split, when its RVSP is missing, outside [5, 150] mmHg (`--min-rvsp`, `--max-rvsp`, the label table's own filter), or when none of its slots is filled. No validation or test label statistic is computed. |
| Missing views | A position without a video holds `MISS`. It is never read and never replaced by a stand-in clip: the loader returns which positions hold a video, the encoder runs only on those, and the probe's key padding mask removes the others from every attention layer. A listed video that cannot be read is treated as missing and counted (`failed_videos`); a study none of whose videos can be read is counted (`failed_studies`) in evaluation, and replaced by another study in training, as the EF probe does with videos. |
| Encoder | V-JEPA 2.1 `vit_base`, built and loaded strictly from `target_encoder` by the EF probe's [`vjepa_2_1_encoder.py`](../evals/video_classification_frozen/modelcustom/vjepa_2_1_encoder.py) (published EchoJEPA checkpoint and our trainer's checkpoints alike). It runs in eval mode, without gradients, wrapped by `SlotEncoder`; the probe stops if any encoder parameter requires a gradient or sits in an optimizer. |
| Clips | Per video, exactly as the EF probe: 2 clips of 16 frames (every second frame, 224 px), one from each half, read by `VideoDataset`'s loader (its short-video padding included); random positions and augmentation in training, the start of each half and no augmentation in evaluation. The tokens of a video's 2 clips are concatenated along time. |
| Head | Early fusion ([`models.py`](../evals/video_classification_frozen_multi/models.py)): each position's tokens get a learned slot (view) embedding and a clip embedding (the video's index within its slot); the tokens of all positions are concatenated and pooled by one attentive probe (one query, 4 blocks, 16 heads), then a linear layer to one output. Smooth L1 loss on the z-scored RVSP. `fusion: late` (a development baseline, not the probe) pools each present position alone and averages their predictions. |
| Settings | As the EF probe: six heads side by side on the same frozen tokens, learning rate {1e-4, 5e-5} × weight decay {0.01, 0.1, 0.4}, AdamW with cosine schedules, no warmup, 20 epochs, batch 4 **studies** per GPU, each GPU's last incomplete training batch dropped, seed 0. |
| Selection | After each epoch every validation study is scored once by every head. The epoch whose best head has the lowest **per-study validation MAE (mmHg)** is `best.pt`; that head is tested. On a tie the earlier epoch is kept. |
| Fingerprint | Every checkpoint saves the SHA-256 of the encoder checkpoint, of the train and validation manifests, of the video index and of `probe_info.json`; the normalization; and the settings (encoder construction, head and fusion, slot specification and positions, clips and temporal padding, optimization, ranges and targets, selection, protocol version, seed, number of GPUs). A resumed probe must match `latest.pt`, a tested one `best.pt` (the number of GPUs aside). |
| Test split | Training and validation never open the test manifest. Its checksum is fixed before training through `probe_info.json`, which the fingerprint covers. `--test_only` refuses an unfinished probe, a probe that already has `test_metrics.json`, and a test manifest that is not the `test.csv` written with the training manifests. Each test study is scored exactly once. Every manifest is checked against the video index first: a video the index lacks, files under another split, or assigns to another study stops the run. |
| Metrics | Per study, in mmHg: MAE (primary), RMSE, R², Pearson r and bias (prediction − reference); error and bias by reference RVSP range (`rvsp_ranges`, default < 30, 30–40, 40–50, 50–60, ≥ 60) and by the number of videos read for the study. No target is prespecified (`evaluation.targets: {}`); set `mae_below` only once agreed, before any test result is seen. |

The shipped config is [`configs/eval/vitb16/rvsp-multiview-224px-16f.yaml`](../configs/eval/vitb16/rvsp-multiview-224px-16f.yaml).
Its `/your_...` placeholders and `data.target_mean` / `data.target_std` (from `probe_info.json`)
must be filled in. Its encoder, clip, head and optimization settings equal the EF config's (a test
checks this), so every encoder is probed identically.

Cost: with the default 6 slots, a study is up to 6 videos × 2 clips × 1,568 tokens = 18,816 tokens,
which the 4-block probe attends over (activation checkpointing is on). Reduce `num_segments` or the
slots only as a new protocol, for every encoder.

## Outputs

`make_rvsp_probe_manifests.py` writes, outside the repository: `train.csv`, `val.csv`, `test.csv`
(`study_id`, the slot positions `<slot>[<k>]`, `label`), `video_index.csv` (split, video, study and
patient of every chosen video) and `probe_info.json` (input checksums, slots, seed, normalization,
counts left out, per-slot presence and acquisition categories per split, output checksums).

A probe writes to `<folder>/video_classification_frozen_multi/<tag>/` the files of the EF probe:
`params-probe.yaml`, `run_info.json`, `log_r0.csv` (training MAE, `val_study_mae`), `latest.pt`,
`epoch_NNN.pt`, `best.pt` (written atomically, `latest.pt` last), and after the test
`test_metrics.json` (written last), `test_predictions.csv` and `val_predictions.csv` (one row per
study: study ID, reference, prediction, error, videos read). **Study IDs can identify patients: these
files stay with the data.** With `meta.wandb_project` set, the probe logs to a private W&B run as
the EF probe does (no path, identifier or table; test: scalars, residual histogram, error by range).

## Validation-only scoring

A comparison that must stay on validation (encoders compared by per-study validation MAE with a paired
bootstrap) takes the selected probe's validation predictions from a separate command that never opens the test
split, as the EF probe's does ([visual_ef_probe.md](visual_ef_probe.md#validation-only-scoring)). It runs the
probe's own `main()` in a scoring mode: the probe is rebuilt from its config and checked against the fingerprint
in `best.pt` (the number of GPUs aside), its epoch must be finished, and the head is selected and every
validation study scored once exactly as the test step does. Training scored the same studies with the same
function, so the recomputed MAE of the selected head must equal the one `best.pt` recorded within `--tolerance`
mmHg (default 0.01; bfloat16 and another number of GPUs move it slightly); otherwise nothing is written. Nothing
is sent to wandb.

```bash
python -m evals.video_classification_frozen_multi.score_validation --fname <probe config> \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 [--overwrite] [--tolerance 0.01]
```

It writes to the probe's folder `val_selected_predictions.csv` (the columns of `val_predictions.csv`; **study
IDs: it stays with the data**) and, last, `val_selected_metrics.json` (epoch, head, checksums of `best.pt` and of
the encoder, the stored-score check, per-study metrics, error by reference range and by videos read, the failed
manifest rows, the numbers of GPUs, the git commit). Existing outputs are replaced only with `--overwrite`.

## Reproduce

```bash
# 1. Per-study manifests (outside the repository).
python data/make_rvsp_probe_manifests.py --video-index <master_video_index.parquet> --labels <rvsp_labels.parquet> \
    --probe-splits <probe_studies.csv> --out-dir <rvsp_probe_dir> [--slots <slots.yaml>]

# 2. A copy of the config per encoder: folder, model_kwargs.checkpoint, the dataset_* files, data.video_index,
#    and target_mean / target_std from <rvsp_probe_dir>/probe_info.json.

# 3. Train, then test once.
python -m evals.main --fname <probe config> --devices cuda:0 cuda:1 cuda:2 cuda:3
python -m evals.main --fname <probe config> --devices cuda:0 cuda:1 cuda:2 cuda:3 --test_only
```

## Tests

| File | Covers |
|---|---|
| [`tests/data/test_make_rvsp_probe_manifests.py`](../tests/data/test_make_rvsp_probe_manifests.py) | Rows to slots with view and acquisition filters, unverified videos left out, the seeded hash choice independent of row order and of other studies, several videos per slot without reuse, z-scores from the train split only, exclusions and presence counted, the refusals listed above, outputs outside the repository. |
| [`tests/utils/test_rvsp_dataset.py`](../tests/utils/test_rvsp_dataset.py) | The manifest layout and malformed manifests; missing positions never read; each video read exactly as `VideoDataset` reads it (padding included); an unreadable video masked and counted; batch order; fixed evaluation frames. |
| [`tests/utils/test_rvsp_probe_models.py`](../tests/utils/test_rvsp_probe_models.py) | A masked position has no effect on the prediction and receives no gradient; slot embeddings tell positions apart; late averaging; strict loading of trainer-style and unwrapped checkpoints; tokens placed at their positions; the encoder unchanged by a probe step. |
| [`tests/utils/test_rvsp_protocol.py`](../tests/utils/test_rvsp_protocol.py) | End to end on synthetic videos: repeatable, and a resumed run equal to an uninterrupted one; selection by per-study validation MAE; resume and test refused with another encoder, seed or manifests; the test manifest never opened by training, not needed to exist, tested once, only when finished and only as written with the training manifests; a manifest with another split's video refused; the shipped config matches the EF config. Validation-only scoring: the selected epoch and head, the validation predictions the test step later writes, the test manifest never opened and nothing sent to wandb; refused for an unfinished probe, another encoder, scores that are not the stored ones, existing outputs (unless `--overwrite`) and together with the test. |
