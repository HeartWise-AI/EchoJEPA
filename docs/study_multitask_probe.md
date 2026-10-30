# Study-level multi-task probe

This document describes `evals/study_multitask_frozen`: a probe that reads a **study** (a bag
of its videos) with a V-JEPA / EchoJEPA encoder and predicts several targets at once, the way
DeepCORO_CLIP's multi-instance linear probing does (`multi_video`, `num_videos`,
`shuffle_videos`, one head per target). It complements the per-video Visual EF probe
([docs/visual_ef_probe.md](visual_ef_probe.md)) and the fixed-slot multi-view probe
(`evals/video_classification_frozen_multi`, one column per view).

## Protocol

| | |
|---|---|
| Manifest | A long CSV with a header: one row per video, with the video path (`datapoint_loc_label`, default `video_path`), the study (`groupby_column`, default `study_id`), the target columns, and optionally a view column and a split column. Either one file (`data_filename` + `split_column`) or one per split (`dataset_train`, `dataset_val`, `dataset_test`). The issue #6 `videos.csv` ([data/build_manifests.py](../data/build_manifests.py)) already has this layout; add the target columns to it. |
| Studies | The rows of a study form one item. Targets are per study: every row must carry the same value (a missing value is allowed and masked), checked before any view filter. `views` keeps only the listed views; `min_videos` leaves out smaller studies. Overlapping studies or patients stop the run. The patient id column (`patient_column`, default `patient_id`) is required, and each study must have exactly one nonmissing patient id. Identifier columns (video, study, view, patient) are read as text, so a patient `0123` is never `123` and a study id has one type in every split's file. |
| Bags | `num_videos` slots per study, each holding `num_clips_per_video` clips. `sampling_mode: random` redraws training bags and uses deterministic evaluation bags seeded by study id, `eval_seed` and bag index. `view_balanced` draws across available view families without replacement. `fixed` retains legacy path-order selection. Training clip positions and augmentation are random; evaluation clip positions are fixed without augmentation. Short studies and unreadable videos are padded and masked. A study whose selected videos all fail is replaced during training and counted as failed during evaluation. |
| Encoder | Built and loaded as in the other probes (`model_kwargs`); `vit_encoder_multiclip` for the EchoJEPA checkpoints, `vjepa_2_1_encoder` for the V-JEPA 2.1 ones. Every clip of every slot is embedded separately; the tokens are concatenated slot by slot. Frozen by default (`experiment.encoder.trainable_blocks: 0`): it runs in eval mode, without gradients, and the probe stops if any encoder parameter requires a gradient or sits in an optimizer. |
| Encoder fine-tuning | `experiment.encoder.trainable_blocks: N` lets the top N transformer blocks and the output norm train (`norm`, or V-JEPA 2.1's `norms_block[-1]`, the norm its forward applies to the last layer), with their own AdamW setting (`lr`, `weight_decay`, `start_lr`, `final_lr`, `warmup`, same schedules as the probe), as the EJ-FT stenosis probe did (20 of ViT-L's 24 blocks at lr 6.4e-6). The encoder is in training mode while it trains and in evaluation mode for scoring; the frozen bottom blocks keep no graph. Only one probe setting is allowed then (several probes would share the encoder's gradients), and every checkpoint holds the fine-tuned encoder. |
| View embedding | With `classifier.use_view_embedding`, `data.view_labels_map` ({view: class}) gives each video its view class and the probe adds a learned embedding of that class to the video's tokens (a view the map lacks takes `Other`; a padded slot has its own embedding). The pooling then knows which view each video shows, as DeepCORO_CLIP's view embeddings. |
| Pooling | `attentive`: the V-JEPA attentive probe (`num_probe_blocks` blocks, `num_heads` heads, one query cross-attending to every token of every video; the padded slots' tokens are masked). `mean` / `max`: each video's tokens are averaged, then the videos are averaged / max-pooled. `attention`: gated attention over the videos' mean embeddings (Ilse et al., 2018; DeepCORO_CLIP's `attention` pooling, `attention_hidden`, `dropout`). The attentive mode attends over `num_videos x num_clips_per_video x tokens per clip` tokens (about 12,500 for 8 videos x 1 clip of 16 frames at 224 px), so it costs far more memory than the others. |
| Heads | One linear head per `target_label`, DeepCORO_CLIP style: `head_task` (`regression`, `binary`, `classification`), `head_structure` (outputs; the number of classes for classification), `loss_structure` (`huber` / `mse` / `l1`, `bce`, `ce`), `head_weights`, `head_dropout`, `labels_map` (string labels to integers). The loss is the weighted sum of the heads' mean losses over the studies whose target is present, counted over every GPU (so the normalization does not depend on how the sampler spread the labeled studies); a head without a target in a batch contributes zero but stays in the graph. Binary targets must be 0 or 1, classification targets whole numbers below the number of classes. Regression targets are z-scored with the train split's mean and population std (or `target_mean` / `target_std` per head); metrics are reported in the targets' units. |
| Settings | Several probes train side by side on the same features, one per `optimization.multihead_kwargs` setting (AdamW, cosine learning-rate and weight-decay schedules, `batch_size` studies per GPU, `num_epochs`, bf16 autocast). Each GPU's last incomplete training batch is dropped. |
| Validation | After each epoch, every validation study is scored with `evaluation.num_bags` deterministic bags (default 1). Regression outputs and classification probabilities are averaged before metrics. A failure in any bag marks that study failed. Per head: regression MAE, RMSE, R², Pearson r and bias; binary AUROC, AUPRC, accuracy at 0.5 and prevalence; classification accuracy and one-vs-rest AUROC; and loss. |
| Selection | `evaluation.selection`: `weighted_loss` or `<head>/<metric>` such as `lvef/mae`. `best.pt` keeps the selected epoch and probe. Any nonfinite output from any head makes the probe ineligible, even if selection uses a different head. An epoch with no finite selection score stops the run. Training ends by rescoring the selected checkpoint on validation; `--val_only` repeats that step without reading the test split or a separate test manifest. |
| Fingerprint | Every checkpoint saves the SHA-256 of the encoder checkpoint and of the manifests, the normalization and the settings (encoder construction, heads, data, bags, optimization, encoder training, selection, protocol, seed). A resumed probe must match `latest.pt` and the test must match `best.pt`, the number of GPUs aside. The protocol has a version, raised when the code changes how a probe trains or is scored: version 3 has both version-2 changes, made in parallel: seeded evaluation bags (`sampling_mode`) and identifier columns read as text (numeric study ids then sort as text, which reorders training). Probes of either version 2 are not resumed or tested. |
| Test | `--test_only`, once, after the last epoch: it refuses an unfinished probe and a probe that already has `test_metrics.json`. The validation split is scored again with the selected probe, then the test split; a study that fails to load is counted, not replaced. |
| Seed | `meta.seed`: initialization, training bags, data order, clip positions and augmentation. `data.eval_seed`: evaluation bag selection, independent of initialization and workers. Every checkpoint saves each GPU's random number state for resumption. |

The shipped config is [`configs/eval/vitl/study_multitask_frozen.yaml`](../configs/eval/vitl/study_multitask_frozen.yaml).

## Coronary angiograms: fine-tuning the DeepCORO_CLIP EchoJEPA encoder

The same probe runs on DeepCORO's angiogram studies, with the EchoJEPA ViT-L that
DeepCORO_CLIP's contrastive stage produced as the encoder:

- [`data/make_coronary_manifest.py`](../data/make_coronary_manifest.py) turns DeepCORO_CLIP's
  report CSV (one row per cine, `α`-separated) into the probe's long manifest: `video_path`,
  `study_id`, `patient_id`, `split`, `structure` (VasoVision's left/right coronary, the view
  column), the 18 per-segment stenoses (%) and per-vessel findings (`rca_ge70`, `lad_ge70`,
  `lcx_ge70`, `left_main_ge50`, `any_ge70`). A study whose cines disagree on a target loses
  it. On the 2025-06-01 export: 22,764 / 4,881 / 4,828 train / val / test studies, a median of
  5 cines per study, `any_ge70` prevalence 0.60.
- [`deepcoro_clip_echojepa_encoder`](../evals/video_classification_frozen/modelcustom/deepcoro_clip_echojepa_encoder.py)
  reads the encoder out of the DeepCORO_CLIP checkpoint (`video_encoder` → `model.encoder.*`,
  the aggregator and projection left out) and loads it strictly into the ViT.
- [`configs/eval/vitl/study_multitask_deepcoro_finetune.yaml`](../configs/eval/vitl/study_multitask_deepcoro_finetune.yaml):
  10 cines per study drawn at random per read, a left/right view embedding, the attentive probe,
  18 segment regressions plus the 5 findings (DeepCORO_CLIP's distal/branch weighting), and the
  top 20 of 24 encoder blocks fine-tuned at 6.4e-6 under one probe setting (the EJ-FT recipe).
  Set `trainable_blocks: 0` and several `multihead_kwargs` for the frozen comparison.

```bash
python data/make_coronary_manifest.py --reports <DeepCORO_CLIP report CSV> --out-dir <manifest_dir>
python -m evals.main --fname configs/eval/vitl/study_multitask_deepcoro_finetune.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3
```

## Targets from the DeepECHO report

[`data/make_study_targets.py`](../data/make_study_targets.py) derives 45 study-level targets from
the cleaned DeepECHO report export (`df_report_2006_2026_clean.parquet`), links them to studies
through the PACS index (accession number to StudyInstanceUID, as `link_ef_labels.py` does) and
joins them onto the issue #6 `videos.csv`, giving `videos_with_targets.csv`, the manifest the
probe reads, and `targets_info.json`, every target's task, units and distribution per split.

| Group | Targets | Rule |
|---|---|---|
| Regression (report units) | `lvef`, `rvsp`, `lvidd`, `lvids`, `ivsd`, `lvpwd`, `lvot_diameter`, `la_dimension`, `lavi` (mL/m², the export's L/m² × 1000), `rv_basal_diameter`, `tapse`, `rv_s_prime`, `av_peak_velocity` and `tr_peak_velocity` (m/s), `av_mean_gradient`, `av_area`, `e_over_e_prime` (mean of lateral and septal), `aortic_root` (sinus of Valsalva), `ivc_diameter` | The report's value; PanEcho's units where they differ. |
| Binary (0/1) | `reduced_ef_40` (EF < 40), `lvef_le_45`, `lvh_any` (both walls ≥ 1.1 cm), `lvh_modsev` (≥ 1.3), `rv_dysfunction` (`MHI VD fonction systolique` ≥ 1; −1 is missing), `rv_dilated` (RV basal ≥ 4.2), `rap_elevated` (≥ 8 mmHg), `lvot_obstruction` (4v² ≥ 20 mmHg; 10 positives in the whole export, left out of the config), `mv_stenosis` (area ≤ 1.5, mean gradient ≥ 5 or PHT ≥ 0.15 s), `as_present` / `as_severe` (ASE: area, mean gradient, peak velocity), `mr_modsev` / `tr_modsev` / `ar_modsev` / `pr_modsev` (referring grade ≥ 2; `0.5` and `Trace` are trace, dates are missing), `pericardial_effusion` (pericardium text; verify against a chart review), `shd_composite` (EchoNext's composite without its effusion component; an unmeasured component counts as absent) | The MHI - PanEcho mapping's cutoffs. |
| Classification (PanEcho class order) | `lv_systolic_function` (EF ≥ 54 normal, ≤ 40 moderately/severely decreased, else mild), `lv_size` (LVIDd ≤ 5.2 / 6.4), `rv_size` (RV basal 4.2 / 5.0), `la_size` (LAVI 34 / 41, else LA dimension 4.0 / 4.6), `lv_diastolic_function` (grade 0 normal, grade 1 or indeterminate, grades 2 to 4), `av_stenosis` (none, mild/moderate, severe), `mr_grade` / `tr_grade` / `ar_grade` (none/trace ≤ 0.5, mild < 2, moderate/severe) | Class indices follow PanEcho's `tasks.pkl` order so the probe's classes compare with PanEcho's outputs. |

A value a rule cannot decide stays empty and the probe masks it. The config for this target set
is [`configs/eval/vitl/study_multitask_deepecho.yaml`](../configs/eval/vitl/study_multitask_deepecho.yaml).

```bash
python data/make_study_targets.py \
    --reports /media/data1/ravram/DeepECHO/TTE_reports_concat/df_report_2006_2026_clean.parquet \
    --pacs /media/data1/datasets/echo/database_tte_2009_2024.csv \
    --videos <probe_manifests>/videos.csv --out-dir <targets_dir>
```

## Outputs

A probe writes to `<folder>/study_multitask_frozen/<tag>/`:

| File | Contents |
|---|---|
| `params-probe.yaml`, `run_info.json` | The full config, and the run's record: code commit, encoder and manifest checksums, split sizes and how each split's bags are drawn, heads, normalization, parameter counts, optimization, protocol and selection. |
| `log_r0.csv` | Per epoch: the best probe's training loss, the validation selection score and the probe it belongs to. |
| `latest.pt`, `best.pt` | The probes with their optimizer states (and the fine-tuned encoder when it trains), every probe's validation metrics for the epoch, the selection state, the normalization, the fingerprint and each GPU's random number state. `best.pt` is the selected epoch. They hold tensors and plain values only and are read back (resume, `--val_only`, `--test_only`) with `weights_only=True`, so a file that holds anything else, such as pickled code, is refused rather than run. |
| `test_metrics.json` | The selected epoch and probe, their validation score, validation and test metrics per head, the normalization, the run record, the failed test studies, and two fingerprints: `training_fingerprint` (the one `best.pt` kept, with the training GPU count) and `scoring_fingerprint` (this scoring run). |
| `val_metrics.json` | Validation rescored from `best.pt` at training completion or with `--val_only`; used by the sweep summary. With multiple bags, includes mean prediction standard deviation across bags per head. Holds the same two fingerprints as `test_metrics.json`. |
| `test_predictions.csv`, `val_predictions.csv` | One row per study: its id, and each head's label and prediction (regression in the target's units, binary as a probability, classification as the class plus its probabilities). **Study ids can identify patients: these files stay with the data.** |

## Weights & Biases

With `meta.wandb_project` and `meta.wandb_entity` set, the probe logs to its own run (a resumed
probe and its test rejoin it): per epoch, the training loss and validation score of the best probe
and of each probe, the best probe's validation metrics per head, its training loss per head, the
learning rates and the number of failed validation studies; at the test, every test metric per head.
The config is logged with every path cut to its file or folder name; no path, study or patient
identifier reaches wandb.
Selected-checkpoint validation results use `probe/selected_val/*`, including `score`, so they
can be distinguished from the last training epoch. Multi-bag prediction CSVs include
`<head>_bag_std` (per class for classification): regression uses report units, classification
uses probabilities. These quantify sensitivity to bag selection, not calibrated uncertainty.

## Study aggregation experiments

The four changes are independent configuration options. They leave the encoder's existing
positional encoding in place; all new trainable components belong to the probe.

| Experiment | Settings | Behavior |
|---|---|---|
| 1. Task/anatomy queries | `classifier.query_mode: task` or `group` | Each target has its own learned cross-attention query, or shares a query with the targets assigned to its anatomical group. With `group`, `task_groups` must map every head to a nonempty group name, for example `{lvef: LV, as_present: AV, as_severe: AV}` for a three-head probe. Requires `attentive` or `hierarchical` pooling. Default `shared` retains one study embedding. |
| 2. View coverage | `data.sampling_mode: view_balanced` | Shuffle candidates within each available family and draw one per family per round, reshuffling family order each round, until the video budget is filled. No duplication or invention of absent views. `view_family_map` maps view labels to families; unmapped labels form their own families. The sweep groups zoom variants and PSAX levels. |
| 3. View conditioning | `classifier.use_view_embedding: true`, `view_dropout: 0.2` | Drop the whole view embedding for a video during training, consistently across its clips/tokens, without rescaling. Evaluation keeps it. Optional probability embeddings and acquisition-type embeddings are described below. |
| 4. Local position and hierarchy | `classifier.pooling_mode: hierarchical`, `local_use_rope: true`, `local_depth: 1`, `local_num_queries: 4` | Each clip gets local self-attention with 3D time/row/column RoPE and multiple learned summaries. Study attention operates on those summaries with no clip-order positional encoding. Coordinates reset for every clip; final shared/task/group queries use ordinary cross-attention. |

`num_probe_blocks` controls the study pooler's depth (self-attention blocks followed by one
cross-attention readout). `local_depth` counts local self-attention blocks before the clip
readout. For hierarchy, the patch grid is derived from `frames_per_clip / tubelet_size` and
`resolution / patch_size`; a token count mismatch fails explicitly. Supply patch tokens,
without extra class/register tokens. RoPE needs at least one local block and at least six
channels per attention head. Local RoPE uses adjacent rotation pairs, as V-JEPA 2.1 does.

For refined views, set `data.view_column` to the new annotation column and provide a complete
`view_labels_map` with contiguous class indices. Unknown labels require an `Other` entry.
No new labels or probabilities are generated by the probe. To use classifier probabilities,
provide one CSV column per class and map each class's label to its probability column:

```yaml
# Example for a three-class manifest; extend to every class in the actual view map.
experiment:
  data:
    view_column: refined_view
    view_labels_map: {A4C: 0, PLAX: 1, Other: 2}
    view_probability_columns: {A4C: p_A4C, PLAX: p_PLAX, Other: p_OTHER}
    acquisition_column: acquisition_type
    acquisition_labels_map: {BMODE: 0, COLOR: 1, CW: 2, Other: 3}
  classifier:
    use_view_embedding: true
    view_dropout: 0.2
```

Every probability row must be finite, nonnegative and sum to one. The embedding is the
probability-weighted sum of view embeddings; one-hot probabilities reproduce hard labels.
The hard view column still controls sampling. Acquisition embeddings are separate, optional,
and not dropped with the view signal. Hierarchical pooling adds both metadata embeddings to
the clip summaries before study attention.

## Sweep

[`configs/sweeps/study_probe.yaml`](../configs/sweeps/study_probe.yaml) defines a controlled
ablation: baseline, each of the four changes, all four together, a no-view control, and
hierarchy without RoPE. Three seeds give **24 runs**. `--design factorial` expands every
binary combination plus the two controls, giving **54 runs**. Each run retains the base
config's optimizer grid. All runs share the checkpoint, cohort, video budget, three seeded
evaluation bags and `weighted_loss` selection. To target AS, change the selection in the
sweep's `defaults` for all variants together. Choose the actual encoder and architecture
through `--base`; `--checkpoint` alone does not change the encoder architecture.

The default third factor tests view dropout with existing hard labels. Add refined labels,
probability columns and/or acquisition metadata to that factor when those columns exist in
the cohort. Data annotation or classifier retraining is separate from this sweep.

```bash
# Generate runnable configs without starting training.
python -m tools.study_probe_sweep generate \
  --output /your_folder/study-sweep/configs \
  --base /your_folder/actual-study-config.yaml \
  --checkpoint /your_folder/encoder.pt \
  --data /your_manifests/videos_with_targets.csv \
  --folder /your_folder/study-sweep/results

# Inspect commands; remove --dry-run to launch sequentially with the existing eval launcher.
python -m tools.study_probe_sweep run \
  --manifest /your_folder/study-sweep/configs/manifest.json \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --dry-run

# A scheduler can launch one run with --index 0 (through 23), assigning devices per job.
# Re-running resumes latest.pt and validates the full experiment fingerprint.
python -m tools.study_probe_sweep summarize \
  --manifest /your_folder/study-sweep/configs/manifest.json \
  --output /your_folder/study-sweep/summary
```

Generation refuses to overwrite different configs. Launch verifies config hashes, locks each
run against concurrent writers and propagates training failures. Results use the normal W&B
project/entity configuration, with unique names and the sweep factors in metadata.
`runs.csv` contains selected validation metrics per seed; `summary.json` contains mean,
sample standard deviation, paired differences from the baseline seed and missing-run indices.
No sweep command evaluates the test split. After choosing a final variant, use its generated
config with `python -m evals.main --fname ... --test_only` for the held-out test.

This is protocol version 3. Version 1 used first-video evaluation. Version 2 was raised twice in
parallel: for these seeded bags (identifiers read with inferred types), and for identifiers read
as text with `eval_videos` bags drawn by another hash. Earlier versions are deliberately rejected
for resume/scoring under this protocol; train the comparison baseline under version 3.

## Reproduce

```bash
# 1. A manifest with the target columns, e.g. the issue #6 videos.csv joined with the report values
#    (lvef, rvsp, ...) per study, outside the repository.

# 2. A copy of the config: set folder, model_kwargs.checkpoint, data.data_filename, the targets
#    (classifier.target_label and the head blocks), and the view map if use_view_embedding.

# 3. Train (4 GPUs), then test once.
python -m evals.main --fname configs/eval/vitl/study_multitask_frozen.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3
python -m evals.main --fname configs/eval/vitl/study_multitask_frozen.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3 --test_only
```

## Tests

- `tests/datasets/test_video_study_dataset.py`: grouping, reproducible and balanced bags,
  padding, failed videos, view labels, soft probabilities and acquisition metadata alignment.
- `tests/models/test_study_probe.py`: head specs from the config keys, every pooling mode with
  masked padded slots, the view embedding, the masked multi-task loss.
- `tests/utils/test_study_probe_smoke.py`: an end-to-end run on CPU with a tiny encoder, two
  epochs, the test, exact repetition with the same seed, a resume after an interruption, the other
  pooling modes and selection rules, the view embedding, and encoder fine-tuning.
- `tests/utils/test_study_probe_sweep.py`: ablation/factorial coverage, launch integrity,
  selected-validation summaries, probability averaging, nonfinite outputs and patient overlap.

```bash
python -m pytest tests/datasets/test_video_study_dataset.py tests/models/test_study_probe.py tests/utils/test_study_probe_smoke.py
```
