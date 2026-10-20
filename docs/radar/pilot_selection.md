# RADAR caption pilot: choosing studies and removing clips before upload (issue #22)

The pilot reviews every valid video of 2–3 report-linked studies from training-split patients, in one of two
forms:

- a moving B-mode or color Doppler video of a classified view gets the [column review v2](column_review_v2.md)
  and its caption;
- a video the view classifier labels `OTHER`, or does not classify, gets the
  [view and type review](view_type_review_v1.md) and no caption, because its view does not say which anatomy is
  shown (one `OTHER` clip shows the inferior vena cava, another flow through a ventricle).

## What is removed, and where

| Removed | Rule | Where |
|---|---|---|
| Single-frame DICOM objects: still B-mode measurement frames, spectral Doppler stills, report and measurement pages | They have no video in the archive | never candidates |
| Classified views in other acquisition modes | not B-mode or color Doppler, or not read from a validated header (GE only); `OTHER` videos keep any mode, which their reviewer gives | `radar_pilot.py shortlist` |
| Clips whose DICOM header gives no usable image region: on this archive, analysis screens (strain curves, segment values) and still images | no image region (`no_image_region`), or a header that cannot be used (`header_size_mismatch`, `header_unreadable`, `no_header`) | `clip_screen.py` |
| Still clips, including a frozen frame saved as several frames | almost nothing in the sector moves (`still`) | `clip_screen.py` |
| Clips with visible measurements over a regular image | not removed by pixels: the measurement scores are reported, and exclude only with `measurement_excludes` (see **Policy**); the audit bounds how many such clips are kept | — |

These removals are automatic: no person reviews a flagged clip, and the only human review is the caption review in
Labelbox. No clip is removed for being short: a clip of a few frames goes to its form like any other. If a reviewer
meets a clip that should have been removed, the review form's **Comments** say so, the clip is deleted from Labelbox
and the rule is fixed.

## The screen (`data/clip_screen.py`)

The screen samples up to 48 evenly spaced frames per clip and reads the ultrasound regions (tissue, color flow, PW/CW Doppler) from its DICOM header. Each video is decoded once using `decord`.

**Errors.** Clips rejected by the decoder (cannot open, no frames, or decode failure) are excluded as unreadable. All other failures—including missing files, missing dependencies, memory errors, worker crashes, and timeouts (`--clip-timeout`, default 600 s)—stop the run. Clip paths and error details are written only to private logs. Requires `decord`, `pydicom`, and `imageio-ffmpeg` for contact sheets.

**Replaced clips.** Archive files may be replaced without changing their paths. `features` therefore records each video's SHA-256 (`video_sha256`) and DICOM regions (`dicom_regions`). Subsequent steps (`sheet`, `calibrate`, `select`, `candidates`) reject changed or missing clips. Read failures stop the step and are logged privately. `candidates` exports hashes to `global_keys.csv`.

**Scan area.** The scan area consists of pixels within DICOM image regions that become sufficiently dark in at least one frame, excluding text and tick marks. Motion is not required. `dicom_status` explains missing scan areas. In 30 recent GE clips, scan areas occupied 16–50% of the frame.

**Still.** A clip is still if:
- `scan_motion_frac` $< 5\%$ when a scan area exists.
- `motion_frac` $< 1\%$ otherwise.

A pixel counts as moving if its temporal standard deviation exceeds 6 intensity levels in any color channel, excluding thin moving cursors. `still_basis` records the criterion used. For calibration, the screen also records motion fractions at thresholds 3 and 10, plus `duplicate_frac` (near-duplicate consecutive frames).

**Measurement.** Two pixel scores, computed and reported for every clip; they exclude a clip only with `measurement_excludes` (see **Policy**):

- `inside_glyphs`: Detects static bright or colored marks over moving tissue, including calipers (+/x), traced contours, dotted/dashed lines, and other ink. Solid straight lines are ignored. Detection uses the full clip and, for clips with $\geq 12$ sampled frames, overlapping half-clip windows spaced one quarter apart. The maximum count is retained; `inside_glyphs_clip` stores the full-clip count, and `inside_marks_raw` counts all ink pieces. Frozen windows and locally static patches are excluded to avoid mistaking tissue for ink. End-bar calipers are not detected, and a single trace falls below the two-mark threshold.

- `extra_cells`: Detects unusual text or results boxes outside the ultrasound image using 8x8-pixel cells. Each clip is compared against the normal ink pattern of its scanner layout (vendor, model, frame size, acquisition mode). Whole-clip and window-based checks detect even brief annotations.

**Layout references.** `shortlist` samples up to 200 reference clips per layout from other studies. Clean references must:
- Show sufficient motion ($\geq 20\%$ of scan area or $\geq 5\%$ of frame).
- Have no internal measurement marks.
- Have no unusual ink block of $\geq 8$ cells (`extra_ink`).

At least 25 clean references are required per layout. `flag` saves baseline images to `flags.parquet.baselines/` (permissions 0700) and reference-exclusion counts to `layouts.csv`.

**Policy.** `flag` excludes, in this order: unreadable clips; clips whose DICOM header gives no image region that applies to the video, each failure its own reason (`no_image_region`; `header_size_mismatch`, a header image size other than the video's; `header_unreadable`; `no_header`, no `dicom_path`); and still clips. The measurement flag (`measurement`) and its scores are computed and reported, but exclude only when the `measurement_excludes` threshold is set, and so does a missing layout baseline (`incomplete`: fewer than 25 clean references), which otherwise only leaves a clip without a results-box score. The header rule is an eligibility rule for this archive, set by the labelled sample below: there it captured the measured clips but one, but DICOM regions describe the image data, not whether anything was measured, so it does not show that every measured clip lacks them.

**A limit found in testing.** Through H.264 encoding (synthetic tissue, libx264 4:4:4 at CRF 18 and 23), the
pixels of a caliper over moving tissue vary by a median of 4 to 9 levels between frames, more than the 2 levels
static ink may vary. Four calipers present through the whole clip then gave 0 to 3 marks instead of 4, with or
without windows. How often this happens on the archive's own encoding only labelled clips can tell: `features`
also records `inside_glyphs_static4`, the marks with ink allowed to vary by 4 levels, and `calibrate` sweeps it.
Looser levels found marks in slowly moving bright tissue (6 levels: 8 to 10 false marks) in the same tests.

**A known blind spot: brief marks.** Windows are half the clip long, one every quarter, so a mark is found only when
it stays static through a whole such window: through at least three quarters of the clip, or through its first,
middle or last half. Calipers shown for less (in a synthetic 48-frame clip, frames 18 to 29) give no mark, and the
clip is kept. Shorter windows would find them, but on real clips most marks found in windows only are already small
specks that may be tissue, and shorter windows would add more; they are held until the labelled validation, whose
sheets should include clips with brief measurements.

**On real clips (counts only, no labels).** 300 training clips of the archive's most common GE model (636 x 434),
one per patient, 150 B-mode and 150 color, half of each as reference clips (75 per layout, where `shortlist` takes
200), at the provisional settings: the color layout gets no baseline, because 39 of its 75 reference clips have a
mark inside the image and 26 extra ink, so all 75 color clips are excluded (43 incomplete, 32 measurement); of the
B-mode clips, 14 of 75 are excluded as measurement. The marks are small (median 4 to 5 pixels), bright rather than
colored, and mostly found in windows only (13 of the 14 B-mode clips, 39 of the 61 color clips with two or more):
slow tissue, or overlays broken up by compression; only labels can tell. A larger minimum mark size alone does not
restore the color baseline: from 3 to 16 pixels, the color reference clips left out for marks fall from 39 to 9 but
those left out for extra ink rise from 26 to 45, leaving 10 to 21 usable. Caliper detection changed no count on
these clips, counting extra cells over the whole screen brought no B-mode clip to 8, and traced
contours added one mark in 2 clips and changed no exclusion.

**Labelled diagnostic sample (the pilot's shortlist).** 216 clips (97 screened, 119 reference) were labelled blind. At the earlier policy, under which the pixel measurement scores excluded, the screen excluded 77% of the 1,061 shortlisted clips, nearly all through `extra_cells`, and flagged none of the 27 clips labelled with measurements. None of the 162 regular B-mode and color clips labelled (1016 x 708, with image regions) had measurements, including the 20 drawn as results boxes. 26 of the 27 measured clips, and all 12 still ones, had no image region in their header: analysis screens such as strain displays (acquisition mode unknown, view `OTHER`). The 27th has a usable header at an unusual size (1236 x 854, its image region 7% of the frame), and the current policy keeps it. The still flag classified both still screened clips correctly, a small sample. With the current policy the screen excludes 10 of the 1,061 clips. About 10 clips were drawn per stratum, so a minority of measured regular clips is not ruled out (0 measured of 10 in a stratum of 423 clips leaves up to about a quarter at 95%): the audit below bounds it.

## Tuning the thresholds (outside the pipeline)

All screening thresholds are initially **provisional**:

- **Still:** motion $<5\%$ of scan area, or $<1\%$ of frame without a scan area.
- **Measurement:** `inside_glyphs` $\geq 2$ or `extra_cells` $\geq 8$.
- **Minimum mark size:** `inside_mark_px` = 3 pixels.
- **Policy:** `measurement_excludes` = false, so the measurement scores exclude nothing (see **Policy**); `calibrate --thresholds '{"measurement_excludes": true}'` evaluates the earlier policy on the same labels.
- **Clean references:** motion $\geq 20\%$ of scan area or $\geq 5\%$ of frame, with no unusual ink block of $\geq 8$ cells.

`select` warns about provisional flags; `candidates` rejects them unless `--provisional-screen` is set for a trial run.

`inside_mark_px` applies to both screened and reference clips, so it also affects layout baselines. Raising it may miss compression-fragmented calipers. `features` stores each detected mark's size and kind (`piece`, `caliper`, `trace`, or dotted `line`) for the full clip and each window (`inside_marks`), allowing size thresholds to be retuned without re-extraction.

### Calibration workflow

Calibration uses separate `clip_screen.py` commands, not the main pipeline:

1. **`sheet`: Generate a blinded review sample.**
   - Default audit: 50 flagged and 100 unflagged clips.
   - `--diagnostic`: up to `--n-per-stratum` (default 10) per role (screened/reference), acquisition mode, and case: still, results box, $\geq 2$ large marks ($\geq 12$ px), trace, dotted line, window-only small marks, other small marks, or clean.
   - Records inclusion probabilities. Private HTML pages show shuffled full clips under global keys, without detector results. Since frames may contain burned-in identifiers, pages stay in an owner-only server folder.
   - Each sheet needs a new folder. `sheet_clips.csv` and `sheet.info.json` are written last; `calibrate` rejects incomplete sheets. Decoder errors go to `decoder.log`.

2. **Manual review:** Answer four questions per clip: (a) still? (b) visible measurements (calipers, traces, measurement text, results box)? (c) another exclusion reason? (d) Doppler/M-mode cursor line over the image? The first three determine eligibility; the fourth enables separate cursor analysis. A cursor alone is not a measurement. Each answer contains the source video's SHA-256; answers for replaced videos are rejected.

3. **`calibrate`: Evaluate thresholds.** Weight clips by inverse inclusion probability (`1/p`) and recompute marks, reference eligibility, baselines, and flags from saved features, using the recorded thresholds or `--thresholds`. Report:
   - Detector sensitivity, specificity, and score sweeps.
   - Eligibility errors, exclusion rate, and the share of kept clips that are ineligible, with an exact one-sided 95% upper bound (potentially optimistic because clips from the same study are correlated).
   - Errors by stratum and acquisition mode, with cursor clips reported separately.
   - Misses, false exclusions, and per-layout baseline availability for minimum mark sizes 3, 5, 8, 12, and 16 px.
   - Reference-clip results separately for diagnostic samples.

   All sheet clips must be reviewed. Calibration rejects unrelated flags or (for diagnostic sheets) reference clips, and incompatible flag rules, pass-1 extractors, or library versions.

4. **Audit, then independent validation:** Tune on the pilot diagnostic sample. Rerun `flag` with the chosen policy, then label a fresh random audit sample of 100 kept clips (with the excluded ones): zero ineligible among them bounds the share of ineligible kept clips at 2.95% (exact one-sided 95% binomial bound; showing below 1% takes 299). This bounds prevalence among the pilot's kept clips, not a detector's sensitivity or other patients. Then freeze the policy and validate on different patients: `screen_validation.py --exclude-patients` with the pilot's `shortlist_clips.csv` and `reference_clips.csv`, followed by `features`, `flag`, `sheet --all-clips`, and `calibrate`.

5. **Finalize:** Make the chosen thresholds the defaults. Update `CALIBRATION` in `clip_screen.py` with the report date and SHA-256, evaluated screen, and thresholds. A test checks that these match the defaults. Only the evaluated screen/configuration is calibrated; changes to code, constants, library versions, or thresholds make flags provisional again.

`sheet --keys <csv-of-global_key>` displays selected clips with scan area, moving region, marks, and motion map. It is not a random sample and cannot be calibrated.

### Validation lists (`data/screen_validation.py`)

Because the pilot covers only recent GE studies, validation samples the full archive manifest (training split by default) in four groups:

- **Suspected frozen:** $\leq 10$ frames or FPS $\leq$`--frozen-max-fps`.
- **Other vendors:** each vendor other than the main one.
- **Low EF:** EF $\leq 35$.
- **General:** remaining clips.

Group sizes use `--n-frozen`, `--n-per-vendor`, `--n-low-ef`, and `--n-general`. Seeded sampling draws patients, then up to `--clips-per-patient` clips each (default 1), recording each clip's inclusion probability. Every nonempty group must contribute at least one clip; a zero target is rejected.

References come from unselected patients. Samples leaving no references are rejected; layouts with fewer than 25 clean references are counted and warned about. `sheet --all-clips` shows every readable sampled clip. `calibrate` uses inclusion-probability weights to estimate overall and per-group (`by_stratum`) performance. `sheet` rejects other sampling methods for these lists.

### Provenance

Each step deletes its previous outputs (and downstream outputs in `radar_pilot.py`), reads inputs once, and hashes inputs, outputs, and its code (`code_sha256`), alongside the Git commit. Provenance records (`<out>.info.json`, `shortlist.json`, `selection.json`) are written last. If an input changes mid-run, outputs are removed and the step stops. Downstream steps reject files lacking matching records.

- `features` records its pass-1 extractor, constants, and library versions; `flag` rejects incompatible features.
- `calibrate` rejects incompatible flag rules or library versions.
- `select` and `candidates` verify the chain back to the shortlist's clip and reference lists.
- Manifest dimensions are retained as `listed_<name>`; decoded dimensions take precedence, and mismatches are counted.

Outputs are owner-only (directories 0700, files 0600). Existing directories readable by others are rejected.

## Choosing the studies (`data/radar_pilot.py`)

### Eligible studies

A study must have:

- A patient in the training split and a TTE examination.
- Exactly one matching report row in the cleaned DeepECHO export.
- After screening, $\geq 20$ classified-view clips covering A4C, A2C, PLAX, and PSAX, with at least one view family represented in both B-mode and color Doppler.

By default, only recent GE studies are used because their acquisition mode can be read from validated DICOM headers.

`--videos` must point to the archive manifest containing DICOM acquisition columns, generated by `build_manifests.py` at commit `57204d1` (`icm/archive_pretraining`; `experiments/icm_archive/manifests`). The current branch's builder lacks these fields, so `shortlist` and `screen_validation.py` check for them and record the build commit.

### Categories (priority order)

1. **Normal:** EF $\geq 55$, AV mean gradient $<20$ mmHg, RVSP $<35$ mmHg, and no moderate-or-worse valve statement.
2. **LV dysfunction:** EF $\leq 35$.
3. **Valve disease:** AV mean gradient $\geq 40$ mmHg, or moderate/severe mitral (MR) or aortic regurgitation (AR) in report text.

MR/AR grade columns are ignored because their codebook is unverified. The text parser recognizes French and English names/abbreviations (e.g., `insuffisance mitrale`, `fuite aortique`, `IM`, `MR`, `AI`) and written/numeric grades (e.g., `modérée`, `sévère`, `grade 3`, `3/4`, `2+`; `1/4` is mild).

A grade counts only if it appears in the same sentence segment as the finding (segments split at commas and connectors such as `et`, `and`, `avec`, `with`, `mais`, `but`), is not negated or uncertain, cannot describe another finding (stenosis, calcification, prolapse, etc.), and is not an adverb (`sévèrement`). Other moderate-or-worse valve wording is **possible**, so the study is neither normal nor definite valve disease. This conservative rule also treats `insuffisance mitrale sévère sur prolapsus` as possible.

Of 249,084 cleaned reports (2006–2026), 88.7% contain no such moderate-or-worse valve statement; 14,485 mention significant MR or AR.

### Selection

`shortlist` samples 8 studies per category in seeded order for screening. Each clip gets an opaque global key: an HMAC of its path using a secret stored in the pilot folder.

`select` chooses one eligible study per category from distinct patients, after flagged clips are removed. It searches all valid combinations to maximize ranking without allowing one category to consume another's only eligible patient. It first verifies that the files still match the screened media. If no combination exists, it writes nothing and reports qualifying counts. Thresholds are recorded, with a warning if provisional.

### Candidates

`candidates` creates column-review-v2 prefills and view/type review items for retained classified clips, keyed by global key. It rejects incomplete selections, changed provenance since `select`, mismatched report files, and provisional screens unless `--provisional-screen` is supplied.

Its summary records SHA-256 hashes of the report and of the review spec (`SPEC` in `tools/labelbox/column_review.py`). Patient identities, report text, and the key map remain in the private pilot folder.

## Not done here

- **Upload.** Only de-identified copies may be uploaded, and they do not exist yet. The upload step will put each
  kept clip's copy under its global key; then `tools/labelbox/import_column_review.py` adds the form and the
  prefills.
- **Report text in Labelbox.** The v2 context attachment must carry no report text: each proposal names its report
  field, and the report context keeps only numbers (fixed in the column review builder on the issue #21 branch).
