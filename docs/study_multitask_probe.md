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
| Studies | The rows of a study form one item. Targets are per study: every row must carry the same value (a missing value is allowed and masked). `views` keeps only the listed views; `min_videos` leaves out smaller studies. |
| Bags | `num_videos` slots per study, each holding `num_clips_per_video` clips of `frames_per_clip` frames (`frame_step`). **Training with `shuffle_videos`**: a random subset of the study's videos, in random order, drawn again at every read, so each epoch sees another bag of the same study (as DeepCORO_CLIP's `shuffle_videos`); clip positions are random and augmentation is on. **Evaluation**: the first `num_videos` videos in path order, fixed clip positions, no augmentation, so a split is read the same way every time. A study with fewer videos is padded with black clips; the padded slots are masked out of the pooling. A video that fails to load is skipped and its slot padded; a study whose selected videos all fail is replaced by another in training and counted as failed in evaluation. |
| Encoder | Built and loaded as in the other probes (`model_kwargs`); `vit_encoder_multiclip` for the EchoJEPA checkpoints, `vjepa_2_1_encoder` for the V-JEPA 2.1 ones. Every clip of every slot is embedded separately; the tokens are concatenated slot by slot. Frozen by default (`experiment.encoder.trainable_blocks: 0`): it runs in eval mode, without gradients, and the probe stops if any encoder parameter requires a gradient or sits in an optimizer. |
| Encoder fine-tuning | `experiment.encoder.trainable_blocks: N` lets the top N transformer blocks and the final norm train, with their own AdamW setting (`lr`, `weight_decay`, `start_lr`, `final_lr`, `warmup`, same schedules as the probe), as the EJ-FT stenosis probe did (20 of ViT-L's 24 blocks at lr 6.4e-6). The frozen bottom blocks keep no graph. Only one probe setting is allowed then (several probes would share the encoder's gradients), and every checkpoint holds the fine-tuned encoder. |
| View embedding | With `classifier.use_view_embedding`, `data.view_labels_map` ({view: class}) gives each video its view class and the probe adds a learned embedding of that class to the video's tokens (a view the map lacks takes `Other`; a padded slot has its own embedding). The pooling then knows which view each video shows, as DeepCORO_CLIP's view embeddings. |
| Pooling | `attentive`: the V-JEPA attentive probe (`num_probe_blocks` blocks, `num_heads` heads, one query cross-attending to every token of every video; the padded slots' tokens are masked). `mean` / `max`: each video's tokens are averaged, then the videos are averaged / max-pooled. `attention`: gated attention over the videos' mean embeddings (Ilse et al., 2018; DeepCORO_CLIP's `attention` pooling, `attention_hidden`, `dropout`). The attentive mode attends over `num_videos x num_clips_per_video x tokens per clip` tokens (about 12,500 for 8 videos x 1 clip of 16 frames at 224 px), so it costs far more memory than the others. |
| Heads | One linear head per `target_label`, DeepCORO_CLIP style: `head_task` (`regression`, `binary`, `classification`), `head_structure` (outputs; the number of classes for classification), `loss_structure` (`huber` / `mse` / `l1`, `bce`, `ce`), `head_weights`, `head_dropout`, `labels_map` (string labels to integers). The loss is the weighted sum of the heads' mean losses over the studies whose target is present. Regression targets are z-scored with the train split's mean and population std (or `target_mean` / `target_std` per head); metrics are reported in the targets' units. |
| Settings | Several probes train side by side on the same features, one per `optimization.multihead_kwargs` setting (AdamW, cosine learning-rate and weight-decay schedules, `batch_size` studies per GPU, `num_epochs`, bf16 autocast). Each GPU's last incomplete training batch is dropped. |
| Validation | After each epoch, every validation study is scored once by every probe. Per head: regression MAE, RMSE, R², Pearson r and bias; binary AUROC, AUPRC, accuracy at 0.5 and prevalence; classification accuracy and one-vs-rest AUROC averaged over the classes present; and every head's loss. |
| Selection | `evaluation.selection`: `weighted_loss` (the training objective on validation, lower is better) or `<head>/<metric>` (`lvef/mae`, `lv_dysfunction/auroc`, ...). The epoch and probe with the best score are kept in `best.pt`. |
| Fingerprint | Every checkpoint saves the SHA-256 of the encoder checkpoint and of the manifests, the normalization and the settings (encoder construction, heads, data, bags, optimization, encoder training, selection, protocol, seed). A resumed probe must match `latest.pt` and the test must match `best.pt`, the number of GPUs aside. |
| Test | `--test_only`, once, after the last epoch: it refuses an unfinished probe and a probe that already has `test_metrics.json`. The validation split is scored again with the selected probe, then the test split; a study that fails to load is counted, not replaced. |
| Seed | `meta.seed`: probe initialization, the videos drawn for each bag, data order, clip positions and augmentation. Every checkpoint saves each GPU's random number state, so a resumed probe continues as an uninterrupted one ([smoke test](#tests)). |

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

## Outputs

A probe writes to `<folder>/study_multitask_frozen/<tag>/`:

| File | Contents |
|---|---|
| `params-probe.yaml`, `run_info.json` | The full config, and the run's record: code commit, encoder and manifest checksums, split sizes and how each split's bags are drawn, heads, normalization, parameter counts, optimization, protocol and selection. |
| `log_r0.csv` | Per epoch: the best probe's training loss, the validation selection score and the probe it belongs to. |
| `latest.pt`, `best.pt` | The probes with their optimizer states (and the fine-tuned encoder when it trains), every probe's validation metrics for the epoch, the selection state, the normalization, the fingerprint and each GPU's random number state. `best.pt` is the selected epoch. |
| `test_metrics.json` | The selected epoch and probe, their validation score, validation and test metrics per head, the normalization, the run record and the failed test studies. |
| `test_predictions.csv`, `val_predictions.csv` | One row per study: its id, and each head's label and prediction (regression in the target's units, binary as a probability, classification as the class plus its probabilities). **Study ids can identify patients: these files stay with the data.** |

## Weights & Biases

With `meta.wandb_project` and `meta.wandb_entity` set, the probe logs to its own run (a resumed
probe and its test rejoin it): per epoch, the training loss and validation score of the best probe
and of each probe, the best probe's validation metrics per head, its training loss per head, the
learning rates and the number of failed validation studies; at the test, every test metric per head.
The config is logged with every path cut to its file or folder name; no path, study or patient
identifier reaches wandb.

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

- `tests/datasets/test_video_study_dataset.py`: grouping, study-level targets, the random bag in
  training and the fixed one in evaluation, padding and masks, failed videos, view filter and
  view classes, label maps.
- `tests/models/test_study_probe.py`: head specs from the config keys, every pooling mode with
  masked padded slots, the view embedding, the masked multi-task loss.
- `tests/utils/test_study_probe_smoke.py`: an end-to-end run on CPU with a tiny encoder, two
  epochs, the test, exact repetition with the same seed, a resume after an interruption, the other
  pooling modes and selection rules, the view embedding, and encoder fine-tuning.

```bash
python -m pytest tests/datasets/test_video_study_dataset.py tests/models/test_study_probe.py tests/utils/test_study_probe_smoke.py
```
