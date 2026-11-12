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
| Pooling | `attentive`: the V-JEPA attentive probe (`num_probe_blocks` blocks, `num_heads` heads, one query cross-attending to every token of every video; the padded slots' tokens are masked). `mean` / `max`: each video's tokens are averaged, then the videos are averaged / max-pooled. `attention`: gated attention over the videos' mean embeddings (Ilse et al., 2018; DeepCORO_CLIP's `attention` pooling, `attention_hidden`, `dropout`). The attentive mode attends over `num_videos x num_clips_per_video x tokens per clip` tokens (about 12,500 for 8 videos x 1 clip of 16 frames at 224 px), so it costs far more memory than the others. `region`: two encoder passes per study over the regions a decoder finds, described [below](#region-pooling-two-passes-over-the-models-own-regions-deepcoro-jepa). |
| Heads | One linear head per `target_label`, DeepCORO_CLIP style: `head_task` (`regression`, `binary`, `classification`), `head_structure` (outputs; the number of classes for classification), `loss_structure` (`huber` / `mse` / `l1`, `bce`, `ce`), `head_weights`, `head_dropout`, `labels_map` (string labels to integers). The loss is the weighted sum of the heads' mean losses over the studies whose target is present, counted over every GPU (so the normalization does not depend on how the sampler spread the labeled studies); a head without a target in a batch contributes zero but stays in the graph. Binary targets must be 0 or 1, classification targets whole numbers below the number of classes. Regression targets are z-scored with the train split's mean and population std (or `target_mean` / `target_std` per head); metrics are reported in the targets' units. |
| Settings | Several probes train side by side on the same features, one per `optimization.multihead_kwargs` setting (AdamW, cosine learning-rate and weight-decay schedules, `batch_size` studies per GPU, `num_epochs`, bf16 autocast). Each GPU's last incomplete training batch is dropped. |
| Validation | After each epoch, every validation study is scored with `evaluation.num_bags` deterministic bags (default 1). Regression outputs and classification probabilities are averaged before metrics. A failure in any bag marks that study failed. Per head: regression MAE, RMSE, R², Pearson r and bias; binary AUROC, AUPRC, accuracy at 0.5 and prevalence; classification accuracy and one-vs-rest AUROC; and loss. |
| Selection | `evaluation.selection`: `weighted_loss` or `<head>/<metric>` such as `lvef/mae`. `best.pt` keeps the selected epoch and probe. Any nonfinite output from any head makes the probe ineligible, even if selection uses a different head. An epoch with no finite selection score stops the run. Training ends by rescoring the selected checkpoint on validation; `--val_only` repeats that step without reading the test split or a separate test manifest. |
| Fingerprint | Every checkpoint saves the SHA-256 of the encoder checkpoint and of the manifests, the normalization and the settings (encoder construction, heads, data, bags, optimization, encoder training, selection, protocol, seed). A resumed probe must match `latest.pt` and the test must match `best.pt`, the number of GPUs aside. The protocol has a version, raised when the code changes how a probe trains or is scored: version 3 has both version-2 changes, made in parallel: seeded evaluation bags (`sampling_mode`) and identifier columns read as text (numeric study ids then sort as text, which reorders training). Probes of either version 2 are not resumed or tested. Region pooling and `optimization.clip_grad` are options that are off by default and leave the version at 3; when set, their settings and weight files join the fingerprint. |
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

## Region pooling: two passes over the model's own regions (DeepCORO-JEPA)

`classifier.pooling_mode: region` replaces the pooling by a probe that runs the encoder twice
per study ([`src/models/region_probe.py`](../src/models/region_probe.py)). It is off by default:
without that value the probe is unchanged, and a `classifier.region` block is then refused.

1. Pass 1 embeds the study's `num_videos` whole videos at `data.resolution`. A frozen region
   decoder reads the final tokens and the outputs of a few encoder blocks and gives every pixel
   of every frame one of `num_classes` region classes.
2. For each region group (a named set of classes), the videos are ranked by the group's mask
   area. In the best `crops_per_group` of them, a square box follows the group's mask over the
   frames; it is cut from the `source_resolution` frames and resized to `crop_resolution`.
3. Pass 2 embeds all those crops with the same encoder, in one call. A study without any
   region takes one encoder call, every other study exactly two.
4. Each group gets one feature: gated attention (`attention_hidden`, `dropout`) over its crops'
   tokens and over the videos' tokens weighted by the group's share of each token.
5. `readouts` turn the group features into the heads of `target_label`. Everything after the
   heads is the study probe's: masked multi-task loss, target normalization, metrics, selection,
   fingerprint, `weights_only` checkpoints, outputs and Weights & Biases logging.

The same encoder serves both passes and trains through `experiment.encoder.trainable_blocks`.
Nothing in the code names an anatomy: classes, groups, resolutions, tracking and readouts are
the `region` block of the config.

| Key (under `experiment.classifier.region` unless noted) | Meaning |
|---|---|
| `classifier.pooling_mode: region` | Turns the two passes on. Needs `query_mode: group` with `task_groups`, `num_clips_per_video: 1`, no view embedding, one `multihead_kwargs` setting and one GPU (`tasks_per_node: 1`). |
| `num_classes` | Classes of the region decoder, the background included. |
| `groups` | `{group: [classes]}`. The order is the order of the groups; a class may belong to several groups. |
| `decoder.blocks`, `decoder.width` | The 0-based encoder blocks the decoder taps besides the final tokens, and its width (a multiple of 8). |
| `decoder.checkpoint` | A file whose `region_decoder` entry is the decoder's state. Required and loaded strictly; the decoder is frozen and saved with the probe. |
| `init_checkpoint` | Optional file whose `probe` entry is the state of the rest of the probe (its keys without `region_decoder.*`), to continue trained heads. It must fit exactly, with one exception: it may lack the whole study pooler, which then starts from its initial values (zero offsets). Default `null`: new heads. |
| `source_resolution` | The videos are resized (not cropped, no augmentation) to this square size; the crops are cut from these frames. |
| `crop_resolution` | Side of the pass-2 crops. |
| `crops_per_group` | Most crops per group, each from another video; `0` skips pass 2. |
| `tracking.min_pixels`, `min_frames`, `margin`, `min_side` | A group needs `min_pixels` source pixels of mask in a video to get a crop there; a mask component seen in fewer than `min_frames` frames is ignored; the box is the mask's extent plus `margin` on each side and at least `min_side` wide (fractions of the frame side). |
| `tracking.line_filter` | Optional `{linearity, length, min_pixels}`: drops a mask component that is a straight line (in the coronary config, a catheter). Default: no filter. |
| `readouts.<name>.kind` | `group_mlp`: per group, LayerNorm, a linear layer to `hidden` and GELU, then the group's heads. `group_linear`: LayerNorm, then one linear layer per head that starts at zero with the initial `bias`. `groups_mlp`: the features of `groups` side by side through LayerNorm, a linear layer to `hidden` and GELU, then the heads. |
| `readouts.<name>.heads` | The heads the readout gives. Every head of `target_label` is read by exactly one readout; `task_groups` gives the group of each head of a `group_mlp` or `group_linear` readout. |
| `readouts.<name>.study_head` | For a `group_linear` readout: the binary head that is the study-level finding (see `study_pooler`). |
| `study_pooler` | `{width, queries, num_heads, bias}`, set exactly when a readout has a `study_head`. Learned queries attend over all pass-1 tokens of the study; per finding they give the study logit (initial `bias`) and one offset per group, zero at the start, added to that readout's heads. No further encoder call. |
| `output_range` | `{regression head: [low, high]}`: the head's value is `low + (high - low) * sigmoid(output)` in the target's units. The head needs `target_mean` and `target_std`. |
| `optimization.clip_grad` | Largest gradient norm of a step, over the probe and the trained encoder blocks together. Default `null`: no clipping. Needs one `multihead_kwargs` setting. Read in every pooling mode. |

An unknown key at any level of the block, a head read twice or by no readout, and weights that
do not fit are errors. `batch_size` studies form one optimizer step; they go through the two
passes one at a time and each study's share of the batch loss is backpropagated at once, so one
study's graphs are in memory at a time. The loader still holds whole batches: 8 studies of 12
videos of 16 frames at 512 px are about 4.8 GB of float32 per batch.

The protocol version stays 3: no existing probe trains or scores differently. In region mode the
resolved `region` settings and the SHA-256 of `decoder.checkpoint` and `init_checkpoint` join the
fingerprint; `clip_grad` joins it only when set.

### The coronary configuration

[`configs/eval/vitl/study_multitask_deepcoro_jepa.yaml`](../configs/eval/vitl/study_multitask_deepcoro_jepa.yaml)
is DeepCORO-JEPA: the EchoJEPA ViT-L, 26 region classes, 18 coronary segment groups, up to 12
cines at 224 px in pass 1 and up to 4 crops of 320 px per segment in pass 2, the top 6 encoder
blocks trained. Its 97 heads, each a manifest column that is constant within a study and empty
where unknown:

| Heads | Target |
|---|---|
| `<segment>_stenosis` (18) | Diameter stenosis of the segment, in %. |
| `<segment>_ge70`, `left_main_ge50` (18) | Stenosis of at least 70 % (left main: at least 50 %), the thresholds of the ACC/AHA/SCAI coronary revascularization guideline (Lawton et al., Circulation 2022;145:e18). |
| `<segment>_calcif`, `_thrombus`, `_cto` (54), `any_calcif`, `any_thrombus`, `any_cto` | Calcification, thrombus and chronic total occlusion, per segment and per study. |
| `ffr`, `ifr` | Fractional flow reserve and instantaneous wave-free ratio of the left anterior descending territory. |
| `ffr_le80`, `ifr_le89` | FFR at most 0.80 (De Bruyne et al., FAME 2, N Engl J Med 2012;367:991) and iFR at most 0.89 (Davies et al., DEFINE-FLAIR, N Engl J Med 2017;376:1824). |

The config sets no threshold: these are the definitions of the manifest's binary columns.

Weights come from a DeepCORO research bundle through
[`tools/convert_deepcoro_bundle.py`](../tools/convert_deepcoro_bundle.py), which writes one file
of tensors with three entries: `encoder` (for `model_kwargs.checkpoint`), `region_decoder` (for
`region.decoder.checkpoint`) and `probe` (for `region.init_checkpoint`). The converter refuses a
bundle and a config that describe different models. A bundle trained before the study pooler
converts with this config to a `probe` entry without a pooler: started from it, the probe
trains a new pooler on top of the bundle's heads.

```bash
python -m tools.convert_deepcoro_bundle --bundle <bundle.pt> --config configs/eval/vitl/study_multitask_deepcoro_jepa.yaml --out <weights.pt>
# In a copy of the config: folder, data.data_filename, and the three weights paths (init_checkpoint: null for new heads).
python -m evals.main --fname <config.yaml> --devices cuda:0
python -m evals.main --fname <config.yaml> --devices cuda:0 --test_only
```

**Checked against the research model.** The converted weights of the bundle this config
describes were run next to the research forward pass on the same tensors: 6 validation
studies, 36 cines, 5 to 59 crops per study, two encoder calls per study in both. Largest
absolute differences:

| Output | bfloat16 autocast | float32 |
|---|---|---|
| Region logits | 0 | 2.5e-5 |
| Crop boxes, in pixels of the 512 frame | 0 | 0.10 |
| Pooled group features | 0.13 | 0.015 |
| Stenosis, in points of 0 to 100 | 0.39 | 0.019 |
| P(significant stenosis) | 0.0019 | 0.00018 |
| Lesion probability of a group | 0.0024 | 0.000019 |
| Study-level lesion logit | 0 | 0 |
| FFR and iFR value | 0.00025 | 1.8e-7 |
| FFR and iFR logit | 0.0039 | 2.4e-7 |

The crop lists were identical in both modes, and each model repeated its own outputs exactly.
bfloat16 autocast is the mode both trainers use, and it is coarse: on the same studies the
research model differs from itself between float32 and bfloat16 by up to 19 stenosis points
on a segment. The float32 column is with TF32 off; with TF32 on (the default of the PyTorch
build used), 5 region labels in a million flip, a crop box moves by up to 30 pixels, and the
float32 differences rise to 0.23 stenosis points and 0.003 in probability.

**Differences from the research trainer.** Training here is not a replay of the research run:

- Loss normalization: each head is averaged over its labelled studies of the batch (the study
  probe's rule); the research trainer divides a sparsely labelled target's loss by all the
  studies of the step.
- Data: the shared video reader (decord, 16 frames at `frame_step: 2`, the first `num_videos`
  cines in path order with `shuffle_videos: false`) and frames resized to `source_resolution`
  before cropping; the research trainer picks its frames around peak opacification, reads
  curated bags and crops at the native resolution.
- Schedule: epochs over the train split; the research trainer draws a fixed number of steps
  with oversampling of rare labels.
- Each `group_linear` readout has its own LayerNorm and each head its own dropout mask; the
  research model shares one of each. Converted weights give the same outputs in evaluation.

**Not in this path:** training the region decoder (it is frozen here; the mask-retention loss
and its teacher caches stay in the research code), cohort building and oversampling, the
peak-opacification frame sampler, the procedure head, and more than one GPU.

### Fine-tuning on your own angiograms

[`configs/eval/vitl/study_multitask_deepcoro_jepa_finetune_example.yaml`](../configs/eval/vitl/study_multitask_deepcoro_jepa_finetune_example.yaml)
is the example to copy. It is the coronary configuration with `region.init_checkpoint` set, so
the heads continue from the trained model; its header repeats the list below. What you need:

| You need | Details |
|---|---|
| Videos | One file per cine run, in a container decord reads (for example `.avi` or `.mp4`). Grayscale X-ray coronary angiography, diagnostic contrast injections of the left or right coronary artery. The trained model has not seen PCI, stent or ventriculography runs: list only the runs you want read. Each cine gives 16 frames (every second frame), resized to 512 x 512 without cropping. The first 12 cines of a study in path order are read. |
| No segmentation masks | The model draws its own coronary segment masks. The mask decoder comes with the weights file and stays frozen. |
| A manifest | One comma-separated file with one row per cine: `video_path`, `study_id`, `patient_id`, `split` (`train`, `val` or `test`) and one column per head (97 columns, table above). A target has the same value on every row of a study and is empty where unknown. It holds identifiers: keep it outside the repository (`*.csv` is ignored by Git here). |
| Patient-level splits | `patient_id` is required and a patient may be in one split only. The run stops otherwise. |
| Labels | Per segment: the stenosis in % (0 to 100), its 0/1 finding (`_ge70`, `left_main_ge50`) and the 0/1 findings `_calcif`, `_thrombus`, `_cto`. Per study: `any_calcif`, `any_thrombus`, `any_cto` (0/1), `ffr` and `ifr` (the measured ratio, 0 to 1) and `ffr_le80`, `ifr_le89` (0/1), empty when not measured. |
| Partial labels | Every column must exist, but a column may be entirely empty: a head's loss and metrics use only the studies that have its label. With stenosis percentages alone, fill the 36 stenosis columns and leave the other 61 empty. A head without labels gets no training signal while the encoder under it changes, so do not rely on its outputs after fine-tuning. |
| Weights | One file with the `encoder`, `region_decoder` and `probe` entries, written by `tools/convert_deepcoro_bundle.py` (above). The example gives it as all three weights paths. |
| One GPU | Region pooling runs on one GPU. A smoke run of the coronary configuration (top 6 encoder blocks trained, studies of 3 to 8 cines) used 10.6 GB; studies with more cines need more. |

Two rows of one study, with most target columns left out:

```text
video_path,study_id,patient_id,split,prox_lad_stenosis,prox_lad_ge70,prox_lad_cto,ffr,ffr_le80
/your_videos/study_0001/run_01.avi,study_0001,patient_0001,train,80,1,0,0.76,1
/your_videos/study_0001/run_02.avi,study_0001,patient_0001,train,80,1,0,0.76,1
```

```bash
# In a copy of the example: the paths marked "SET THIS".
python -m evals.main --fname <your_copy.yaml> --devices cuda:0               # train, select on validation
python -m evals.main --fname <your_copy.yaml> --devices cuda:0 --test_only   # score the test split once
```

The learning rates, the 3 epochs and the loss weights are those of the research run. They have
not been tuned for a small data set: with a few hundred studies, watch the validation loss and
consider `encoder.trainable_blocks: 0`, which trains the heads only.

### Another modality

An echocardiogram model uses the same path with its own config and no code change: a region
decoder for its classes (chambers, valves) saved as the `region_decoder` entry of a file in
[`RegionMaskDecoder`](../src/models/region_probe.py)'s layout, then for example
`groups: {lv: [1], la: [2], mitral: [5]}`, a `group_mlp` readout whose heads are `lvef` and
`reduced_ef_40` with `task_groups: {lvef: lv, reduced_ef_40: lv}`, and the echo manifest and
encoder of the other study probe configs. No echo region decoder exists in this repository yet.

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
- `tests/models/test_region_probe.py`: masks to boxes, tracking, crops, the region decoder,
  exactly two encoder calls per study, padded slots, the readouts and the study pooler, the
  `region` block's validation, probe weights that must fit, the shipped DeepCORO-JEPA config,
  and its fine-tuning example: the same settings with a warm start, and a manifest as its header
  describes it, read with every label or with stenosis alone.
- `tests/utils/test_region_probe_smoke.py`: an end-to-end region run on CPU with a tiny
  encoder and decoder: training with gradient clipping, the test, a resume, a warm start with
  and without a study pooler, and refused settings and files.
- `tests/utils/test_convert_deepcoro_bundle.py`: the bundle converter on a synthetic bundle:
  every weight where the research formulas read it, and refused mismatches.

```bash
python -m pytest tests/datasets/test_video_study_dataset.py tests/models/test_study_probe.py tests/utils/test_study_probe_smoke.py \
    tests/models/test_region_probe.py tests/utils/test_region_probe_smoke.py tests/utils/test_convert_deepcoro_bundle.py
```
