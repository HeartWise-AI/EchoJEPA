# RADAR caption pilot: choosing studies and removing clips before upload (issue #22)

The pilot reviews every valid video of 2–3 report-linked studies from training-split patients, in one of two
forms:

- a moving B-mode or color Doppler video of a classified view gets the [column review v2](labeling_instructions_v1.md#column-review)
  and its caption;
- a video the view classifier labels `OTHER`, or does not classify, gets the
  [view and type review](labeling_instructions_v1.md#view-and-type-review-other-clips) and no caption, because its view does not say which anatomy is
  shown (one `OTHER` clip shows the inferior vena cava, another flow through a ventricle).

## What is removed, and where

| Removed | Rule | Where |
|---|---|---|
| Single-frame DICOM objects: still B-mode measurement frames, spectral Doppler stills, report and measurement pages | They have no video in the archive | never candidates |
| Classified views in other acquisition modes | not B-mode or color Doppler, or not read from a validated header (GE only); `OTHER` videos keep any mode, which their reviewer gives | `radar_pilot.py shortlist` |
| Unreadable clips | the decoder rejects the video (`unreadable`) | `clip_screen.py` |
| Still clips, including a frozen frame saved as several frames | almost nothing in the field of view moves (`still`) | `clip_screen.py` |
| Analysis screens: strain curves, segment values, plots | the header names no image data, no acquired image and no ECG, **and** static graphics lie beside the field of view (`analysis_screen`) | `clip_screen.py` |
| Clips with visible measurements | the header names a measurement (`measurement_header`), measurement text is read (`measurement_text`, only with an OCR engine), or a caliper pair is drawn (`measurement_calipers`) | `clip_screen.py` |
| Clips the screen is uncertain about | partial or conflicting evidence (see **Decision**): left out of the pilot and listed, not judged invalid | `clip_screen.py`, `radar_pilot.py select` |

These removals are automatic: no person reviews an excluded or uncertain clip, and the only human review is the caption review in
Labelbox. No clip is removed for being short: a clip of a few frames goes to its form like any other. If a reviewer
meets a clip that should have been removed, the review form's **Comments** say so, the clip is deleted from Labelbox
and the rule is fixed.

## The screen (`data/clip_screen.py`)

The screen samples up to 48 frames per clip, one per equal step, each moved by a deterministic jitter of up to 40% of the step: evenly spaced samples of a loop whose cycle divides the step all show one phase (a synthetic 941-frame clip repeating a 20-frame cycle read as still; jittered, its samples fall on 17 of the 20 phases). It reads the ultrasound regions (tissue, color flow, PW/CW Doppler) from its DICOM header. Each video is decoded once using `decord`.

**Errors.** Clips rejected by the decoder (cannot open, no frames, or decode failure) are excluded as unreadable. All other failures—including missing files, missing dependencies, memory errors, worker crashes, and timeouts (`--clip-timeout`, default 600 s)—stop the run. Clip paths and error details are written only to private logs. Requires `decord`, `pydicom`, and `imageio-ffmpeg` for contact sheets.

**Replaced clips.** Archive files may be replaced without changing their paths. `features` therefore records each video's SHA-256 (`video_sha256`), its DICOM regions (`dicom_regions`) and the header evidence the decision uses (image content, `MEASURE`, source image, ECG). Subsequent steps (`sheet`, `calibrate`, `select`, `candidates`) reject changed or missing clips; `sheet` and `select` also reread the headers, so a header rewritten to name a measurement, with the same regions, counts as changed, and so does a row that does not record the evidence. Read failures stop the step and are logged privately. `candidates` exports hashes to `global_keys.csv`.

The screen keeps a clip, excludes it, or leaves it **uncertain**, from three independent kinds of evidence. A clip is excluded only on evidence of a disallowed condition; a header without image regions, a missing reference baseline, missing OCR or many small static marks are recorded, never a reason on their own.

**Field of view.** Motion is read over the DICOM scan area (pixels inside the header's image regions that reach a low brightness in some frame, cleaned of text and tick marks); when the header gives no usable region, over a field of view found from the pixels (`image_field`): the largest component of that content, opened to drop text, lines and curves, with a plausible size and shape and speckle texture inside it. It uses appearance, not motion, so a frozen clip has one too. On 563 pilot clips that have regions it was found in all, and matched the DICOM scan area (median IoU 0.99; 4 below 0.8): agreement with the header, not labels. Without either, motion is read over the whole frame. `region_source` records which (`dicom`, `image`, `whole_frame`), and `dicom_status` why the header gave none.

**Still.** A pixel moves when its temporal standard deviation exceeds 6 levels in any color channel (thin moving lines such as cursors removed). A clip is still when less than 5% of its field of view moves (`region_motion_frac`); read over the whole frame, below 1%, and then it is excluded only if its frames also repeat (`duplicate_frac` $\geq 0.9$), else it is uncertain (`still_low_confidence`). The earlier shares (`scan_motion_frac`, `motion_frac`, and at 3 and 10 levels) stay recorded for calibration.

**Analysis screens.** Strain and other analysis displays are screens composed after acquisition. On the 1,661 pilot clips (shortlist and reference), the GE header told them apart: 108 clips name no image data in ImageType (fourth value `0000`, against `0001` for B-mode and `0011` for color), reference no acquired image (`SourceImageSequence`) and carry no ECG; all 108 have no ultrasound regions, an unknown acquisition mode and the `OTHER` view, and every labelled one was measured or still. 101 other clips without regions are ordinary recordings by their header (and the 25 labelled were clean). `ImageType` DERIVED marks every clip and says nothing. From the pixels, an analysis screen has static graphics beside its field of view (`outside_ink_frac`: plots, tables, segment maps): 0.3% to 3% of the frame on the 28 labelled analysis screens, below 0.07% on 90% of 387 clean clips and 0.65% at most. Both the header pattern and graphics on at least 0.3% of the frame exclude a clip (`analysis_screen`); either alone makes it uncertain.

**Measurement overlays.** Three kinds of evidence, each enough alone:
- the header names a measurement: GE writes `MEASURE` in ImageType (`measurement_header`; on the pilot, the 2 clips of a 1236 x 854 layout, one labelled measured);
- measurement text, read by an OCR engine when one is installed: a measurement name with its number, such as `EF 55`, `LVIDd 4.5`, `EDV 120`, excludes (`measurement_text`); a number with a measurement unit but no name read (ml, cm², mmHg) only makes a clip uncertain (`measurement_unit_text`); a velocity without a name (`Scale 64 cm/s`, `Nyquist 64 cm/s`, a color bar's limit) is a Doppler setting and counts for nothing; lines that start with a scanner setting, the heart rate or the depth are left out. No OCR engine is installed on the server, so `text_evidence` is `unavailable` for every clip: recorded as missing, never as no text;
- a caliper pair over the moving image (`measurement_calipers`): two calipers (+ or ×) static together over the whole clip or one half-clip window, on one line (the same ink component) or alike in size and color and at least 20 pixels apart; two calipers each on a line of its own are not a pair unless they are alike. A single caliper or a traced contour makes a clip uncertain.

The earlier pixel scores stay as diagnostics, and for the `pixel` policy:

- `inside_glyphs`: Detects static bright or colored marks over moving tissue, including calipers (+/x), traced contours, dotted/dashed lines, and other ink. Solid straight lines are ignored. Detection uses the full clip and, for clips with $\geq 12$ sampled frames, overlapping half-clip windows spaced one quarter apart. The maximum count is retained; `inside_glyphs_clip` stores the full-clip count, and `inside_marks_raw` counts all ink pieces. Each mark is kept with its size, kind, place and color (`inside_marks`). Frozen windows and locally static patches are excluded to avoid mistaking tissue for ink. End-bar calipers are not detected.

- `extra_cells`: Detects unusual text or results boxes outside the ultrasound image using 8x8-pixel cells. Each clip is compared against the normal ink pattern of its scanner layout (vendor, model, frame size, acquisition mode). Whole-clip and window-based checks detect even brief annotations.

**Layout references.** `shortlist` samples up to 200 reference clips per layout from other studies. Clean references must:
- Show sufficient motion ($\geq 20\%$ of scan area or $\geq 5\%$ of frame).
- Have no internal measurement marks.
- Have no unusual ink block of $\geq 8$ cells (`extra_ink`).

At least 25 clean references are required per layout. `flag` saves baseline images to `flags.parquet.baselines/` (permissions 0700) and reference-exclusion counts to `layouts.csv`.

**Decision.** Under the default `hybrid` policy, `flag` records for every clip `video_status`, `dicom_region_status`, `region_source`, `still_status`, `analysis_evidence`, `measurement_header`, `text_evidence`, `geometry_evidence`, `evidence_missing` (header, field of view, text), and the decision with its reasons (`decision`, `decision_reasons`):
- **exclude** when any of: `unreadable`; `still`; `analysis_screen`; `measurement_header`; `measurement_text`; `measurement_calipers`;
- else **uncertain** when any of: `still_low_confidence`; `analysis_header_only`; `analysis_visual_only`; `measurement_trace`; `measurement_single_caliper`; `measurement_unit_text`;
- else **keep**.

`exclude` and `exclude_reason` give the exclusions alone. The earlier policies stay to compare on the same labels: `header` (exclusion by a header without usable image regions, then still) and `pixel` (still, then the pixel scores, with a missing baseline excluded as `incomplete`); they keep or exclude, never leave a clip uncertain.

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
200), under the earlier `pixel` policy: the color layout gets no baseline, because 39 of its 75 reference clips have a
mark inside the image and 26 extra ink, so all 75 color clips are excluded (43 incomplete, 32 measurement); of the
B-mode clips, 14 of 75 are excluded as measurement. The marks are small (median 4 to 5 pixels), bright rather than
colored, and mostly found in windows only (13 of the 14 B-mode clips, 39 of the 61 color clips with two or more):
slow tissue, or overlays broken up by compression; only labels can tell. A larger minimum mark size alone does not
restore the color baseline: from 3 to 16 pixels, the color reference clips left out for marks fall from 39 to 9 but
those left out for extra ink rise from 26 to 45, leaving 10 to 21 usable. Caliper detection changed no count on
these clips, counting extra cells over the whole screen brought no B-mode clip to 8, and traced
contours added one mark in 2 clips and changed no exclusion.

**Labelled diagnostic sample (the pilot's shortlist).** 216 clips (97 screened, 119 reference) were labelled blind: 27 measured (26 analysis screens without regions, 1 recording with regions at 1236 x 854), 12 still (all analysis screens) and 187 clean (162 B-mode or color recordings with regions, 25 recordings without them). It is a stratified diagnostic sample, so these are counts, not estimates, and the hybrid policy was designed on it: in-sample results, optimistic. Its one numeric value, `ANALYSIS_OUTSIDE_INK`, set from either half of the patients (0.28% and 0.33%), gave the same decisions on the other half. Measured clips excluded / uncertain / kept, and clean clips the same:

| Policy | Measured (27) | Clean (187) | Clean clips excluded |
|---|---|---|---|
| `pixel` (earlier) | 27 / 0 / 0 | 124 / 0 / 63 | 66% |
| `header` (earlier) | 26 / 0 / 1 | 25 / 0 / 162 | 13% |
| `hybrid` | 26 / 1 / 0 | 0 / 3 / 184 | 0% (1.6% uncertain) |

Every policy excluded the 12 still clips, all analysis screens; the sample holds no still ordinary recording, so still detection on recordings rests on synthetic tests. On the 1,061 shortlisted clips, the hybrid policy keeps 1,043 (B-mode 627 of 630, color 412 of 421, unknown mode 4 of 10), leaves 10 uncertain (9 color clips with static graphics beside the field of view, 1 B-mode clip with a single caliper) and excludes 8 (6 analysis screens, the 2 clips whose header names a measurement); `pixel` excluded 818 and `header` 10.

**Known gaps: `keep` is no assurance that a clip shows no measurement.** The labels hold one measured ordinary recording, caught by its header (its pixels gave no mark), and no still ordinary recording, so the hybrid policy's sensitivity over ordinary loops is not known, nor how it carries to other vendors:
- results boxes and measurement text (an `EF 55%` beside the image) are caught only by OCR, which is not installed. `extra_cells` cannot stand in for it: at 8 cells or more it scores 745 of the pilot's 1,043 kept clips (172 at 20 or more) and none of the 27 labelled measured clips, so a fallback built on it would be neither selective nor validated;
- calipers drawn as end bars, shown for less than half the clip (two calipers over 12 of 48 frames leave no mark; brief overlays are on hold), or broken up by compression give no caliper pair; a single caliper or a traced contour only makes a clip uncertain;
- the header signals are those of GE Vivid E95 exports (the ImageType content value, `MEASURE`); other vendors and models need their own survey.

Closing these needs a validated fallback (OCR within the secure environment, or evidence of brief overlays) and blind challenge sets holding brief and compressed overlays; until then the screen stays provisional (**Finalize**, below).

## Tuning the thresholds (outside the pipeline)

All screening thresholds are initially **provisional**:

- **Still:** motion $<5\%$ of scan area, or $<1\%$ of frame without a scan area.
- **Measurement:** `inside_glyphs` $\geq 2$ or `extra_cells` $\geq 8$.
- **Minimum mark size:** `inside_mark_px` = 3 pixels.
- **Policy:** `policy` = `hybrid` (see **Decision**); `calibrate --thresholds '{"policy": "pixel"}'` or `"header"` evaluates an earlier policy on the same labels, and every report compares the three (`policies`).
- **Hybrid evidence constants** (in the code, part of the flag rule): static graphics beside the field of view on at least 0.3% of the frame (`ANALYSIS_OUTSIDE_INK`, the one value set from the labels); a caliper pair of areas within a factor 2, colors within 40 levels, at least 20 pixels apart; repeated frames ($\geq 90\%$) to call a whole-frame still reading certain.
- **Clean references:** motion $\geq 20\%$ of scan area or $\geq 5\%$ of frame, with no unusual ink block of $\geq 8$ cells.

`select` warns about provisional flags; `candidates` rejects them unless `--provisional-screen` is set for a trial run.

`inside_mark_px` applies to both screened and reference clips, so it also affects layout baselines. Raising it may miss compression-fragmented calipers. `features` stores each detected mark's size and kind (`piece`, `caliper`, `trace`, or dotted `line`) for the full clip and each window (`inside_marks`), allowing size thresholds to be retuned without re-extraction.

### Calibration workflow

Calibration uses separate `clip_screen.py` commands, not the main pipeline:

1. **`sheet`: Generate a blinded review sample.**
   - Default audit: one stratum per decision, 50 excluded (`--n-flagged`), 50 uncertain (`--n-uncertain`) and 100 kept clips (`--n-unflagged`); each stratum is drawn independently, and the kept one, a simple random sample of the kept clips, bounds the share of them that are ineligible.
   - `--diagnostic`: up to `--n-per-stratum` (default 10) per role (screened/reference), acquisition mode, and case: still, results box, $\geq 2$ large marks ($\geq 12$ px), trace, dotted line, window-only small marks, other small marks, or clean.
   - Records inclusion probabilities. Private HTML pages show shuffled full clips under global keys, without detector results. Since frames may contain burned-in identifiers, pages stay in an owner-only server folder.
   - Each sheet needs a new folder. `sheet_clips.csv` and `sheet.info.json` are written last; `calibrate` rejects incomplete sheets. Decoder errors go to `decoder.log`.

2. **Manual review:** Answer four questions per clip: (a) still? (b) visible measurements (calipers, traces, measurement text, results box)? (c) another exclusion reason? (d) Doppler/M-mode cursor line over the image? The first three determine eligibility; the fourth enables separate cursor analysis. A cursor alone is not a measurement. Each answer contains the source video's SHA-256; answers for replaced videos are rejected.

3. **`calibrate`: Evaluate thresholds.** Weight clips by inverse inclusion probability (`1/p`) and recompute marks, reference eligibility, baselines, and flags from saved features, using the recorded thresholds or `--thresholds`. Report:
   - Detector sensitivity, specificity, and score sweeps.
   - Eligibility errors, exclusion rate, and the share of kept clips that are ineligible. Its exact one-sided 95% upper bound follows the sample's design, not the clips the sample happens to hold. Only an audit sample draws its kept clips as a simple random sample of the clips the screen keeps, so only it gets a bound, and only when every clip kept at the evaluated thresholds comes from one of its strata (the sheet records each screened clip's stratum; strata are drawn independently). Kept clips of several strata, such as those another policy merges, are a stratified sample even at equal probabilities, and no bound for that is implemented. The bound then describes the clips this screen keeps, whatever their patients, not other patients' clips. A validation list draws patients, then clips: a cluster sample, where 100 clips of 5 patients can show none ineligible though a tenth are, and no design-based bound for it is implemented. Diagnostic samples and challenge sets get none either. The report says why it withholds the bound (`upper_95_withheld`) or what the bound describes (`upper_95_describes`).
   - Errors by stratum and acquisition mode, with cursor clips reported separately.
   - Misses, false exclusions, and per-layout baseline availability for minimum mark sizes 3, 5, 8, 12, and 16 px.
   - Reference-clip results separately for diagnostic samples.

   All sheet clips must be reviewed. Calibration rejects unrelated flags or (for diagnostic sheets) reference clips, and incompatible flag rules, pass-1 extractors, or library versions. A sheet records the media of the whole population it was drawn from (each readable clip's video SHA-256, DICOM regions and header evidence, sampled or not), and `calibrate` refuses flags whose clips hold other media: the weights would describe other clips. A sheet made before sheets recorded them is checked against the flags it was drawn from (`--drawn-flags`, matched by SHA-256), on the media columns both record. `flag --thresholds` and `calibrate --thresholds` are checked too: shares are finite numbers from 0 to 1, counts and sizes whole numbers of at least 1, the policy one of the three; NaN and Infinity are refused.

4. **Audit, challenge set, then independent validation:** Tune on the pilot diagnostic sample. Rerun `features` and `flag` with the chosen policy, then label a fresh random audit (all excluded and uncertain clips and 100 kept ones). Its kept stratum is a simple random sample of the pilot's kept clips: zero ineligible among 100 would bound their share at 2.95% (exact one-sided 95% binomial; showing below 1% takes 299). That bound describes the clips this pilot keeps (24 patients), not other patients' clips, and prevalence among kept clips, not a detector's sensitivity. A bound for the archive needs a design over its patients: a validation list's two-stage sample is a cluster sample, for which no design-based bound is implemented. Sensitivity to measurements over ordinary loops needs positives the random samples rarely hold: draw a challenge set (`challenge`) from a validation list's flags and label it blind. Then freeze the policy and validate on different patients: `screen_validation.py --exclude-patients` with the pilot's `shortlist_clips.csv` and `reference_clips.csv`, followed by `features`, `flag`, `sheet --all-clips`, and `calibrate`.

5. **Finalize,** only once the independent-patient validation and blind challenge sets with positives (measurements over ordinary loops, brief and compressed overlays) are labelled and reported; until then `CALIBRATION` stays `None`, every flag is provisional and `candidates` refuses it. Make the chosen thresholds the defaults. Update `CALIBRATION` in `clip_screen.py` with the report date and SHA-256, evaluated screen, and thresholds. A test checks that these match the defaults. Only the evaluated screen/configuration is calibrated; changes to code, constants, library versions, or thresholds make flags provisional again.

`sheet --keys <csv-of-global_key>` displays selected clips with scan area, moving region, marks, and motion map. It is not a random sample and cannot be calibrated.

**Challenge sets.** `clip_screen.py challenge --flags <flags> --out <challenge.csv>` draws, with a fixed seed, up to `--per-signal` (default 10) random clips per evidence signal: likely measurements the screen may catch or miss (a caliper pair, a single caliper, a traced contour, a header naming a measurement, measurement text, partial analysis evidence, a layout without reference clips) and kept clips that resemble measurements (many marks inside the image, a results-box score, a dotted line, color Doppler: hard negatives, or misses). `sheet --keys <challenge.csv> --blind` labels them without the detector's view, and `calibrate` counts the answers per signal and per policy; an enriched set is never weighed up to a population, so its report carries no share or bound. Draw it from the flags of patients the screen was not tuned on, such as a validation list's (its sampling probabilities are left aside: the set is only counted).

### Validation lists (`data/screen_validation.py`)

Because the pilot covers only recent GE studies, validation samples the full archive manifest (training split by default) in four groups:

- **Suspected frozen:** $\leq 10$ frames, or FPS $\leq$`--frozen-max-fps` when that is above 0 (the default 0 leaves the frame rate out, so the manifest may lack `fps`; above 0 it needs it).
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

`select` chooses one eligible study per category from distinct patients, on the clips the screen keeps: excluded clips are removed, and uncertain ones are left out and listed with their reasons (`uncertain_clips.csv`, every shortlisted study's), never taken for invalid. It searches all valid combinations to maximize ranking without allowing one category to consume another's only eligible patient. It first verifies that the files still match the screened media. If no combination exists, it writes nothing and reports the qualifying studies per category and the shortlisted clips' decisions. Flags without decisions (an older `flag`) are refused. Thresholds are recorded, with a warning if provisional.

### Candidates

`candidates` creates column-review-v2 prefills and view/type review items for retained classified clips, keyed by global key. It rejects incomplete selections, changed provenance since `select`, mismatched report files, and provisional screens unless `--provisional-screen` is supplied.

Its summary records SHA-256 hashes of the report and of the review spec (`SPEC` in `tools/labelbox/column_review.py`). Patient identities, report text, and the key map remain in the private pilot folder.

## Not done here

- **Upload.** Only de-identified copies may be uploaded, and they do not exist yet. The upload step will put each
  kept clip's copy under its global key; then `tools/labelbox/import_column_review.py` adds the form and the
  prefills.
- **Report text in Labelbox.** The v2 context attachment must carry no report text: each proposal names its report
  field, and the report context keeps only numbers (fixed in the column review builder on the issue #21 branch).
