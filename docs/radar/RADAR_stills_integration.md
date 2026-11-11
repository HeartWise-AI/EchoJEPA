# RADAR + Doppler stills

This document describes how Doppler still images are added to RADAR v0 (PR #46). It covers:

- the still image itself;
- structure-wise captions that keep the Doppler statements v0 removes, plus a templated Hemo caption;
- an optional EchoNet peak heat-map;
- per-still measurement regression.

Everything here is **off by default**. A v0 config builds the same model, reads the same items and keeps its v0 fingerprint.

## Why

RADAR v0 aligns frozen B-mode/colour clip tokens with structure captions. The caption builder removes every clause that only Doppler can ground (`scope: bmode`): stenosis, regurgitation, gradients, velocities, pressures and diastology. Those cells go to the `doppler_only` exclusion ledger. `hemo` is registered but untrained ("waits for the Doppler stream").

DeepEcho now has a still-image measurement dataset. It contains the CW and PW/TDI stills that the sonographer measured, with the values read from the burned-in measurement box: AV Vmax, LVOT VTI, TR Vmax, MV E/A/DT and so on. These stills can ground the removed statements and feed Hemo.

## How it works

```
 meas_<year>.parquet (one row per burned-in measurement)      master_metadata (video pipeline)     RADAR studies.csv
            |                                                          |                                 |
            v                                                          v                                 v
 data/build_still_manifest.py --------- study folder -> study_uid = RADAR study_id ----- patient split (inherit / hash)
            |   still_id (hash), still_tag (AV, LVOT, TR, MV_INFLOW, TDI...), still_type (cw_AV...),
            |   spectral box + velocity scale, label boxes, y_<STRUCT>_<Q> targets, EchoNet scalars
            v
 stills.csv + demographics.csv (age from dates, sex, BSA; no birth date, patient id hashed)
            |
            |   still PNG
            v
 src/datasets/still_preprocess.py:  mask label/value boxes -> crop spectral region -> inpaint coloured
            |                        calipers/traces -> grey 224x224 -> 2 identical frames
            v
 frozen EchoJEPA encoder (one tubelet) -> tokens [1x14x14, D] -> still TokenCache (separate from the clip cache)
            |
            |   per study item (src/datasets/radar_stills.py, drawn after the clip bag, balanced over tags)
            v
 +--------------------------------------------- StructurePooler -----------------------------------------------+
 |  keys = [ null | clip tokens (+pos, view, acquisition) | still tokens (+pos, still marker, still type,      |
 |                                                          + MLP(velocity scale, mode, EchoNet Vmax/conf)) ]  |
 |  mask : clip -> structure by VIEW (routing.views); still -> structure by TAG (routing v1 still_structs)    |
 |  bias : + alpha * log(EchoNet heat-map pooled to 14x14) on that still's keys   (optional, alpha starts 0)   |
 |  query: one per structure (+ FiLM(age, sex) on the queries: optional ablation, off by default)             |
 +---------------------------------------------------------------------------------------------------------------+
            |  z[study, structure]                                     still tokens + cond + FiLM(age, sex)
            v                                                                       v
 StructureContrastiveLoss vs caption[study, structure]            StillMeasurementHeads (per still)
   caption = "full" (B-mode + Doppler clauses) if the bag         -> z(log y_AV_VMAX), z(log y_TR_VMAX)...
   holds a still of the structure, else "bmode";                  -> Huber on present targets
   Hemo = templated RVSP / RA pressure / E/e'                     
            \___________________ loss = L_radar + heads.weight * L_measure ___________________/
```

1. **Manifest** (`data/build_still_manifest.py`). The still dataset's `StudyInstanceUID` is the extractor's own and never matches RADAR's `study_id` (0 of 6,687 studies in 2024). The study is therefore found through the DICOM study folder (`dirname(dcm_path)`) in the video metadata, which matched 100% of 2024.
   - The split is inherited from RADAR's full cohort split. A patient outside it gets a split drawn from a salted hash, and every study of a patient shares that split.
   - Measured values become wide targets, in fixed units and averaged over the still's beats. Implausible values are dropped and counted in `build_info.json`.
2. **Still encoding** (`src/datasets/still_preprocess.py`). The still is processed in this order:
   - the text boxes are masked;
   - it is cropped to the DICOM spectral region;
   - chromatic overlays are inpainted;
   - the result is converted to grey, resized, and repeated over 2 frames.

   With the encoder's tubelet of 2, that gives one temporal token, i.e. a `[1, 14, 14]` grid. The encoder and embedding space are the same as for the clips, and a still costs 8x fewer tokens than a 16-frame replicated clip. The velocity scale lost by cropping comes back as numbers (`v_top_ms`, `v_bot_ms`).
3. **Routing** (`configs/radar/radar_routing_v1.yaml`). It is v0 plus `still_structs`, `still_types` and a trained `hemo` with `caption_source: structured`. v0 is untouched and stays the default.

   | still tag | feeds |
   |---|---|
   | AV, LVOT, AVA, AR | aortic |
   | MV, MVA, MR | mitral |
   | MV_INFLOW | mitral, hemo |
   | TR | tricuspid, hemo |
   | TV | tricuspid |
   | PV, PR, RVOT | pulmonic |
   | DAO | aorta |
   | TDI | hemo |
4. **Captions** (`data/build_radar_captions.py --doppler-captions`, `data/radar_doppler_captions.py`).
   - The flag adds a `scope=full` row (B-mode caption + Doppler clauses) next to the `bmode` row of every structure a still can feed. It also adds a templated Hemo row.
   - It also adds a `scope=doppler` row (Doppler clauses alone). The dataset reads `full` when the bag holds a clip and a still of the structure, `doppler` when it holds stills alone, and `bmode` when it holds clips alone, so B-mode statements are never trained against stills alone.
   - With `--stills <stills.csv>`, the split's studies that have stills and no eligible clip are captioned too (`doppler` rows and the Hemo row only) and go through the same patient check.
   - Frames per still: the still stream reads whatever token grid the still cache has (`model.stills.token_grid`): `[1, 14, 14]` for a 2-frame still, `[8, 14, 14]` for 16 repeated frames (`static_clip(frames=16)`). The heat-map bias is the same spatial map at every temporal position.
   - The clip and still caches must come from one encoder: training and evaluation compare the caches' `encoder_sha256` and `encoder` provenance and refuse a mismatch. The still cache writer must record them as the clip cache writer does.
   - Evaluation (`zero_shot_eval.py`): a checkpoint trained with stills needs `--stills` (the evaluated studies' still table) or `--clip_only`; the training-overlap check covers the studies of the training still table.
   - The dataset reads the `full` row only when the study's bag holds a still of that structure, so a caption never states what the inputs cannot show.
   - The new rules live in a separate module, because the normality policy digest covers `radar_normality.py` and `radar_captions.py`.
   - Full rows are `provisional`, so A2/A3 refuse them until a policy with Doppler scope is approved. A0/A1 run.
5. **Model** (`src/models/radar/structure_pooler.py`, `heads.py`, `model.py`).
   - Stills are a second key stream for the same structure queries.
   - Conditioning, the heat-map bias and demographic FiLM all start as the identity (zero-initialised).
   - `StillMeasurementHeads` regress the still's own measurements as z-scores of log values, with an all-reduced count normalisation so DDP equals one process.
6. **Training** (`evals/radar_frozen/train.py`). The new optional sections are `data.stills`, `model.stills` and `heads`. The still stream's record (table digest, cache meta, target scaler, age stats) enters the fingerprint only when present.

### Config (opt-in)

```yaml
data:
  routing: configs/radar/radar_routing_v1.yaml
  captions: <captions built with --doppler-captions and the v1 registry>
  stills:
    table: <stills_dir>/stills.csv
    cache_dir: <still token cache>
    demographics: <stills_dir>/demographics.csv
    num_stills: 8
    cond_columns: [v_top_ms, v_bot_ms, echonet_present, echonet_vmax_ms, echonet_conf]
    heatmap_column: echonet_heatmap        # optional: .npy heat-maps already mapped to the crop
model:
  stills: {token_grid: [1, 14, 14], cond: true, heatmap_bias: false, demo_film: false}
heads:                                      # optional
  targets: [y_AV_VMAX, y_AV_VTI, y_AV_MEANPG, y_LVOT_VMAX, y_LVOT_VTI, y_MV_E, y_MV_A, y_MV_DT, y_TR_VMAX]
  weight: 0.5
  demographics: true
```

### On and off by default

| Piece | Default |
|---|---|
| Routing registry | v0 (`DEFAULT_ROUTING`); v1 only through `data.routing` |
| Still stream in the pooler | off (no `model.stills`): same parameters and outputs as v0 |
| Doppler and Hemo captions | off (no `--doppler-captions`): caption table identical to v0, no `scope` column |
| Measurement heads | off (no `heads`) |
| EchoNet heat-map bias | off (`model.stills.heatmap_bias: false`) |
| EchoNet scalars | only if listed in `data.stills.cond_columns` |
| Age/sex FiLM | heads: on when heads are on (`heads.demographics`); pooler: off (`model.stills.demo_film`) |
| Measured value as an input | not possible: conditioning columns that are targets or `y_*` are refused |

## Leakage controls

The measured value reaches the model in four ways. Each has a control:

1. **Burned-in text.**
   - Label boxes are masked from `text_boxes`. In 2024, 16% of label boxes overlap the spectral region, and 75% of Philips boxes do.
   - Crop to the spectral region.
   - *Gap:* the still dataset stores label boxes, not value boxes. `still_meas.py` should store value boxes too; until then the mask pads boxes and relies on the crop.
2. **Calipers and VTI traces.** Measured stills have about 50% more chromatic pixels in the spectrum (median 1.35% vs 0.88%). Chromatic pixels are inpainted on every still, measured or not, so masking itself is not a marker.
3. **Value as input.** Refused by `StillSource`. Some quantities determine each other: MaxPG = 4 Vmax² (true for ~100% of OCR pairs), and AVA comes from the VTIs. Never split such a group between input and target (`LEAK_GROUPS` in the manifest builder).
4. **Shortcuts in alignment.** Age and sex correlate with findings such as calcified valves and E/A < 1. Demographic FiLM is off in the pooler; report a demographics-only baseline whenever it is turned on.

**Audit before trusting results** (not in this PR):

- **"Is-measured" probe.** Train a linear probe on still tokens to tell measured stills from unmeasured ones of the same type and vendor. Target AUROC about 0.5.
- **Overlay-only regression probe.** Regress the value from the overlay pixels alone; it should fail.
- **Clean evaluation.** Score the heads on unmeasured stills against report values (`make_study_targets`: `av_peak_velocity`, `tr_peak_velocity`, `rvsp`).

## Data facts behind the design (2024 build)

| Fact | Value |
|---|---|
| Size | 78k measurement rows, 12.9k stills, 6.7k studies |
| Structure imbalance (rows) | LVOT 34k, AV 14k, MV 5k, TR 0.5k, MR 35 |
| Modes | CW and PW/TDI; no M-mode detected (the header rule never fires) |
| Vendors | 95% GE at one screen layout; 2025 is about 25% Philips |
| 2025 | not yet in `master_metadata`, so no study id or demographics until it is rebuilt |
| EchoNet AV Vmax | within 10% for about 74% of stills after width-1024 rescaling |
| EchoNet coverage | AV, LVOT, TR and MR Vmax only, so the heat-map path is optional |

## Not in this PR

- **Still token cache writer.** It depends on open question 1. `prepare_still` is the per-still input; `TokenCacheWriter(root, [1, 14, 14], D, provenance={..., "preprocess": PREPROCESS_VERSION})` stores the result.
- **EchoNet runner.** It should produce heat-maps in crop coordinates.
- **Value-as-input ("interpret") arm** with modality dropout.
- **Zero-shot Doppler prompts.**
- **The leak audits above.**

## Open questions

1. **Which encoder built the RADAR clip cache: EchoJEPA ViT-L (`vit_encoder_multiclip`) or the V-JEPA 2.1 ViT-B?**
   - ViT-L: a 2-frame static clip gives one tubelet.
   - V-JEPA 2.1: it has a native image path (`img_temporal_dim_size: 1`, image modality embedding), which may suit stills better.
   - Either way, D and the encoder checkpoint must equal the clip cache's.
2. **Are templated Hemo captions acceptable?** The template uses RVSP, RA pressure and mean E/e′, in French like the reports. RADAR otherwise trains on free text.
   - The template is deterministic. Many studies will share identical Hemo captions, which A1's exact-caption positives turn into positives.
   - The alternative is to leave Hemo untrained and use only the heads for haemodynamics.
3. Should Doppler clauses stay in the same structure caption (`full`, as here), or go to separate `<structure>_doppler` slots?
4. Should 2025 (more Philips) be held out as a temporal and vendor test once it has study ids?
