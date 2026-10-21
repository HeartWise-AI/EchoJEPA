# RADAR caption pilot: choosing studies and removing clips before upload (issue #22)

The pilot reviews every valid video of 2–3 report-linked studies from training-split patients, one per report
category, and of one study per vendor slot (other vendors, **Selection**), in one of two forms:

- a moving B-mode or color Doppler video of a classified view gets the [column review v2](labeling_instructions_v1.md#column-review)
  and its caption;
- a video the view classifier labels `OTHER`, or does not classify, gets the
  [view and type review](labeling_instructions_v1.md#view-and-type-review-other-clips) and no caption, because its view does not say which anatomy is
  shown (one `OTHER` clip shows the inferior vena cava, another flow through a ventricle).

## What is removed, and where

| Removed | Rule | Where |
|---|---|---|
| Single-frame DICOM objects: still B-mode measurement frames, spectral Doppler stills, report and measurement pages | They have no video in the archive | never candidates |
| Classified views in other acquisition modes | the header declares neither B-mode nor color-flow imaging, or, for GE, the mode was not read from it; another vendor's B-mode or color flow is then told apart from the pixels (**Mode from the pixels**); `OTHER` videos keep any mode, which their reviewer gives | `radar_pilot.py shortlist` |
| Unreadable clips | the decoder rejects the video (`unreadable`) | `clip_screen.py` |
| Objects that are no ultrasound image: structured reports, secondary captures | the header's SOP class (`not_ultrasound_image`), any vendor | `clip_screen.py` |
| Still clips, including a frozen frame saved as several frames | almost nothing in the field of view moves (`still`) | `clip_screen.py` |
| Analysis screens: strain curves, segment values, plots | a GE header names no image data, no acquired image and no ECG; for another vendor, that **and** static graphics beside the field of view (`analysis_screen`) | `clip_screen.py` |
| Clips with visible measurements | the header names a measurement (`measurement_header`) or a caliper pair is drawn (`measurement_calipers`); measurement text read by OCR excludes where no header can name a measurement (another vendor's header, or none) and is recorded for GE headers (`measurement_text`, `text_decides`) | `clip_screen.py` |
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

**Analysis screens.** Strain and other analysis displays are screens composed after acquisition. On the 1,661 pilot clips (shortlist and reference), the GE header told them apart: 108 clips name no image data in ImageType (fourth value `0000`, against `0001` for B-mode and `0011` for color), reference no acquired image (`SourceImageSequence`) and carry no ECG; all 108 have no ultrasound regions, an unknown acquisition mode and the `OTHER` view, and every labelled one was measured or still. 101 other clips without regions are ordinary recordings by their header (and the 25 labelled were clean). `ImageType` DERIVED marks every clip and says nothing. For a GE header (see **Vendors**) this pattern alone excludes a clip (`analysis_screen`).

From the pixels, an analysis screen has static bright or colored graphics beside its field of view (`outside_ink_frac`: plots, tables, segment maps; the black background does not count): 0.3% to 3% of the frame on the 28 labelled analysis screens, below 0.07% on 90% of 387 clean clips but up to 0.65%, so the two overlap. "Beside" means outside the DICOM scan area, or, without regions (every analysis screen), outside the one field of view `image_field` picks; other small sectors and charts then count as beside it, and a screen where no field is plausible has no pixel evidence at all. On the pilot's GE clips this score never added an exclusion to the header and made 9 color recordings uncertain, so for GE it is recorded only. For another vendor, whose header does not carry the pattern, the header pattern and graphics on at least 0.3% of the frame together exclude a clip, and either alone makes it uncertain; that threshold is unvalidated there.

**Unusual image data.** A GE header names its image data in ImageType's fourth value: `0001` tissue, `0011` tissue and color flow, `0000` none (an analysis screen). Of 46,293 GE training headers surveyed (20,000 GE Vingmed clips of as many patients, 20,000 GE Healthcare, all 6,293 GEMS; headers only), 22 GE Vingmed clips (0.1%) named another value, `0041` (20) or `0051` (2), and GE Healthcare and GEMS none. All 22 were Vivid E9 or E95 clips of the `OTHER` view, without `MEASURE`, and the screen kept 21 of them. Labelled blind on one sheet with 22 clips of `0001` or `0011` of the same view and models, of other patients: all 22 were displays of a 3D probe (several panels, or a rendered volume), 11 measured and 7 more a rendered volume rather than a 2D recording, and 4 clean (two or more 2D panels); all 22 others were clean. The text pass read only scale labels (`40 mm`) on the 11 measured, which it rightly ignores: their measurements are drawn, not written. So such a header makes a clip uncertain (`header_content` `unusual`, `unusual_image_content`); another vendor's header, which does not use the value, gives no such evidence (`unavailable`).

**Measurement overlays.** Three kinds of evidence, each enough alone:
- the header names a measurement: GE writes `MEASURE` in ImageType (`measurement_header`; on the pilot, the 2 clips of a 1236 x 854 layout, one labelled measured);
- measurement text, read by the `text` pass (see **Measurement text**): a measurement name with its number, such as `EF 55`, `LVIDd 4.5`, `EDV 120` (`measurement_text`); a number with a measurement unit but no name read (ml, cm², mmHg: `measurement_unit_text`); a velocity without a name (`Scale 64 cm/s`, `Nyquist 64 cm/s`, a color bar's limit) is a Doppler setting and counts for nothing; lines that start with a scanner setting, the heart rate or the depth are left out. The evidence is recorded for every clip the pass read, and `unavailable` without it (missing, never no text). It decides for the clips whose header cannot name a measurement (`text_decides` `header_unavailable`, the default: another vendor's header, whose convention is unverified, or none), for which the text and the marks over the image are the only measurement evidence: a name excludes and a unit alone makes a clip uncertain. For a GE header, which names a measurement itself (`MEASURE`), it is recorded only; `all` makes it decide for every clip and `none` for none;
- a caliper pair over the moving image (`measurement_calipers`): two calipers (+ or ×) static together over the whole clip or one half-clip window, on one line (the same ink component) or alike in size and color and at least 20 pixels apart; two calipers each on a line of its own are not a pair unless they are alike. A single caliper or a traced contour makes a clip uncertain.

**Vendors.** The archive manifest is 98.7% GE Vingmed but also holds GE Healthcare (66,532 clips), Philips (13,762), GEMS (8,623) and Siemens (235) clips. A survey of the headers of up to 400 training clips per vendor (headers only, one clip per patient first):

| Vendor (models) | ImageType 4th value | references the acquired image | ECG | MEASURE | image regions |
|---|---|---|---|---|---|
| GE Vingmed (Vivid E9, E95, Vivid7) | `0001` / `0011` | 400 of 400 | 400 | 0 | 398 |
| GE Healthcare (Vivid S70) | `0001` / `0011` | 400 of 400 | 398 | 0 | 397 |
| GEMS (Vivid i) | `0001` / `0011` | 400 of 400 | 387 | 0 | 396 |
| Philips (iE33, EPIQ, Affiniti) | none | 0 of 400 | 399 | 0 | 399, all of type tissue |
| Siemens (SC2000, 5 patients) | `0001` / `0011` / `0040` | 4 of 199 | 198 | 0 | 198 |

The three GE vendor names write one convention (`GE_HEADER_VENDORS`, `header_convention` = `ge`): the analysis-screen pattern and `MEASURE` are evidence, and so is the absence of `MEASURE`. For every other vendor (`unverified`) these signals say nothing by their absence (`measurement_header` is `unavailable`, `evidence_missing` names `header_convention`), analysis screens rest on the pixels, and only a present `MEASURE` flag counts. The survey also found a Philips structured report and a single-frame secondary capture among the archive's videos, which the SOP class rule removes for any vendor. Their screening rules are unvalidated until other vendors' clips are labelled (a validation list's vendor groups, **Validation lists**).

**Mode from the pixels.** Only the three GE names' region headers were validated against the decoded videos (0 of 822 GE header-B-mode videos showed color); Philips and Siemens headers declare one tissue region over a color-flow loop, and all 13,706 classifiable Philips clips read as B-mode. Pass 1 therefore measures the share of the frame whose pixels are red or blue (colored by at least 60 levels, red or blue at least 40 above green, which the ECG trace is) and change over the clip (`moving_color_frac`; a static color bar does not change), and `flag` reads a clip as color flow from 0.5% of the frame (`pixel_mode`). On 80 training clips per group, one per patient: GE header-B-mode clips reached 0.14% at most (GE Healthcare 0.04%); 97.5% of GE Vingmed and 93.8% of GE Healthcare header-color clips reached 0.5%, and so did 33.8% of Philips clips. Measured by pass 1 itself on 25 other clips per group, GE header-B-mode clips reached 0.33% (none read as color), and 92% of GE Vingmed and 96% of GEMS header-color clips and 32% of Philips clips read as color: a color clip with little flow can read as B-mode, and a B-mode clip stays closer to the threshold than in the first sample. `flag` reports the reading against the header on every GE clip it screens (`pixel_mode_vs_header`: agreement, not labels). The reading decides nothing in the screen; `radar_pilot.py` uses it as the mode of clips whose header cannot give one (`mode_source` `pixels`), and the reviewer confirms or corrects it like the view.

**Measurement text.** `clip_screen.py text` reads each readable clip of `features` output with RapidOCR (`rapidocr-onnxruntime`, its PP-OCR models inside the pinned package, run on this machine) in the isolated OCR environment of the de-identification check (`deid_ocr`, built for the Doppler pilot's de-identification check), which has no `decord`: frames are decoded with OpenCV, in order. It reads the first, middle and last of the frames pass 1 read, upscaled twice as the de-identification check does, joins the words of a row into lines (a name and its number are often boxed apart; a wide gap starts another line, so a settings panel beside a results box stays apart), and keeps only the matched measurement expressions, never the text read. A clip whose file changed since `features` stops the run; one OpenCV cannot decode keeps its text unavailable. Its output is the `features` output with the text columns added and a record of what read them; `flag` takes it as its features and refuses text read by other code. On a synthetic results box RapidOCR dropped the space of `EF 55 %` (`EF55%`), so a name ends wherever a letter does not follow. On the 266 clips of the first validation list it read measurement text on none (**Validation lists**), so deciding for the clips without a GE header excluded no clean clip there; how much measurement text it catches still needs a positive control: clips with drawn and with real measurement text, labelled, and `calibrate` on them (`--thresholds '{"text_decides": "all"}'` to score it on every vendor). `flag` refuses features without the text pass when the text decides for any of their readable clips.

    python data/clip_screen.py features --clips <clips.csv> --out <screen_dir>/features.parquet
    <deid_ocr>/bin/python -E -s data/clip_screen.py text --features <screen_dir>/features.parquet \
        --out <screen_dir>/features_text.parquet --workers 6 --threads 2
    python data/clip_screen.py flag --features <screen_dir>/features_text.parquet --reference <screen_dir>/reference.parquet \
        --out <screen_dir>/flags.parquet

The earlier pixel scores stay as diagnostics, and for the `pixel` policy:

- `inside_glyphs`: Detects static bright or colored marks over moving tissue, including calipers (+/x), traced contours, dotted/dashed lines, and other ink. Solid straight lines are ignored. Detection uses the full clip and, for clips with $\geq 12$ sampled frames, overlapping half-clip windows spaced one quarter apart. The maximum count is retained; `inside_glyphs_clip` stores the full-clip count, and `inside_marks_raw` counts all ink pieces. Each mark is kept with its size, kind, place and color (`inside_marks`). Frozen windows and locally static patches are excluded to avoid mistaking tissue for ink. End-bar calipers are not detected.

- `extra_cells`: Detects unusual text or results boxes outside the ultrasound image using 8x8-pixel cells. Each clip is compared against the normal ink pattern of its scanner layout (vendor, model, frame size, acquisition mode). Whole-clip and window-based checks detect even brief annotations.

**Layout references.** `shortlist` samples up to 200 reference clips per layout from other studies. Clean references must:
- Show sufficient motion ($\geq 20\%$ of scan area or $\geq 5\%$ of frame).
- Have no internal measurement marks.
- Have no unusual ink block of $\geq 8$ cells (`extra_ink`).

At least 25 clean references are required per layout. `flag` saves baseline images to `flags.parquet.baselines/` (permissions 0700) and reference-exclusion counts to `layouts.csv`.

**Decision.** Under the default `hybrid` policy, `flag` records for every clip `video_status`, `dicom_region_status`, `dicom_object`, `header_convention`, `region_source`, `still_status`, `analysis_evidence`, `header_content`, `measurement_header`, `text_evidence`, `geometry_evidence`, `evidence_missing` (header, header convention, field of view, text), and the decision with its reasons (`decision`, `decision_reasons`):
- **exclude** when any of: `unreadable`; `not_ultrasound_image`; `still`; `analysis_screen` (a GE header's pattern, or another vendor's with graphics beside the field); `measurement_header`; `measurement_calipers`; `measurement_text` where `text_decides` (by default, clips without a GE header);
- else **uncertain** when any of: `still_low_confidence`; `analysis_header_only` and `analysis_visual_only` (other vendors only); `unusual_image_content` (a GE header naming image data other than tissue, tissue and color, or none: **Unusual image data**); `measurement_trace`; `measurement_single_caliper`; `measurement_unit_text` where `text_decides`;
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
| `hybrid`, analysis by header and pixels (earlier) | 26 / 1 / 0 | 0 / 3 / 184 | 0% (1.6% uncertain) |
| `hybrid`, a GE header alone | 27 / 0 / 0 | 0 / 1 / 186 | 0% (0.5% uncertain) |

Every policy excluded the 12 still clips, all analysis screens; the sample holds no still ordinary recording, so still detection on recordings rests on synthetic tests. On the 1,061 shortlisted clips (all GE), the hybrid policy keeps 1,052 (B-mode 627 of 630, color 421 of 421, unknown mode 4 of 10), leaves 1 uncertain (a B-mode clip with a single caliper) and excludes 8 (6 analysis screens, the 2 clips whose header names a measurement); scoring the analysis screens by header and pixels left 9 more color clips uncertain, `pixel` excluded 818 and `header` 10. These counts come from features extracted before the SOP class was recorded.

**Audit of the second shortlist.** The second draw shortlisted 1,973 clips (four categories and three vendor slots: GE Vingmed, GE Healthcare, GEMS, and 111 Philips clips), screened with the text pass: 1,946 kept, 15 uncertain, 12 excluded. An audit sheet of all 12 excluded and all 15 uncertain clips and a simple random sample of 100 kept ones was labelled without the detector's view:
- kept: none of the 100 measured, still or otherwise ineligible, so at most 2.95% of the 1,946 kept clips are (one-sided 95% bound; it describes these clips, not other patients');
- excluded: 11 of the 12 measured (10 analysis screens, 4 of them still too, and 1 clip whose header names a measurement) and 1 clean, a Philips clip in which a caliper pair was found among its marks;
- uncertain: all 15 clean, 10 with a single caliper and 5 with a traced contour, 10 of them Philips clips.

Single-caliper and traced-contour evidence has so far been wrong on every labelled clip it marked (15 here, 8 on the first validation list, 1 in the diagnostic sample) and has marked no measured clip: those were caught by their header. It costs clips (5 in the pilot's studies), not cleanliness; whether it catches anything waits for a caliper test set of measured loops. The text pass read named measurement text on 6 of the 10 analysis screens (units only on 3 more) and none on the other 1,963 clips.

**Known gaps: `keep` is no assurance that a clip shows no measurement.** The labels hold one measured ordinary recording, caught by its header (its pixels gave no mark), and no still ordinary recording, so the hybrid policy's sensitivity over ordinary loops is not known, nor how it carries to other vendors:
- results boxes and measurement text (an `EF 55%` beside the image) are caught only by OCR, which decides only where no header can name a measurement and whose sensitivity is not measured (**Measurement text**). `extra_cells` cannot stand in for it: at 8 cells or more it scores 745 of the pilot's 1,043 kept clips (172 at 20 or more) and none of the 27 labelled measured clips, so a fallback built on it would be neither selective nor validated;
- calipers drawn as end bars, shown for less than half the clip (two calipers over 12 of 48 frames leave no mark; brief overlays are on hold), or broken up by compression give no caliper pair; a single caliper or a traced contour only makes a clip uncertain. The caliper detector has not caught a labelled measured clip yet: the one measured recording was caught by its header;
- the header signals hold for GE's three vendor names by a header survey, and were labelled on GE Vingmed only; for other vendors they are unverified, and their analysis-screen and still rules are untested on labels (**Vendors**).

Closing these needs a validated text reading, evidence of brief overlays, blind challenge sets holding brief and compressed overlays, a caliper test set of real measured loops, and labelled clips of other vendors; until then the screen stays provisional (**Finalize**, below).

## Tuning the thresholds (outside the pipeline)

All screening thresholds are initially **provisional**:

- **Still:** motion $<5\%$ of scan area, or $<1\%$ of frame without a scan area.
- **Measurement:** `inside_glyphs` $\geq 2$ or `extra_cells` $\geq 8$.
- **Minimum mark size:** `inside_mark_px` = 3 pixels.
- **Policy:** `policy` = `hybrid` (see **Decision**); `calibrate --thresholds '{"policy": "pixel"}'` or `"header"` evaluates an earlier policy on the same labels, and every report compares the three (`policies`).
- **Text:** `text_decides` = `header_unavailable`: measurement text decides for clips whose header cannot name a measurement and is recorded for the rest; `calibrate --thresholds '{"text_decides": "all"}'` (or `"none"`) evaluates another scope.
- **Hybrid evidence constants** (in the code, part of the flag rule): for vendors other than GE, static graphics beside the field of view on at least 0.3% of the frame (`ANALYSIS_OUTSIDE_INK`, the one value set from the labels); the vendors whose header decides (`GE_HEADER_VENDORS`); the SOP classes of an ultrasound image (`ULTRASOUND_IMAGE_CLASSES`); a caliper pair of areas within a factor 2, colors within 40 levels, at least 20 pixels apart; repeated frames ($\geq 90\%$) to call a whole-frame still reading certain.
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

**Challenge sets.** `clip_screen.py challenge --flags <flags> --out <challenge.csv>` draws, with a fixed seed, up to `--per-signal` (default 10) random clips per evidence signal: likely measurements the screen may catch or miss (a caliper pair, a single caliper, a traced contour, a header naming a measurement, measurement text, partial analysis evidence, a GE header naming unusual image data, a layout without reference clips) and kept clips that resemble measurements (many marks inside the image, a results-box score, a dotted line, color Doppler: hard negatives, or misses), and kept clips of a vendor whose header convention is unverified. `sheet --keys <challenge.csv> --blind` labels them without the detector's view, and `calibrate` counts the answers per signal and per policy; an enriched set is never weighed up to a population, so its report carries no share or bound. Draw it from the flags of patients the screen was not tuned on, such as a validation list's (its sampling probabilities are left aside: the set is only counted).

### Validation lists (`data/screen_validation.py`)

Because the pilot covers only recent GE studies, validation samples the full archive manifest (training split by default) in four groups:

- **Suspected frozen:** $\leq 10$ frames, or FPS $\leq$`--frozen-max-fps` when that is above 0 (the default 0 leaves the frame rate out, so the manifest may lack `fps`; above 0 it needs it).
- **Other vendors:** each vendor other than the main one (GE Vingmed): GE Healthcare, GEMS, Philips and Siemens. Labelled, these are what the rules for other vendors (**Vendors**) are checked and built on; Siemens has 5 training patients, so its group holds at most `--clips-per-patient` clips of each.
- **Low EF:** EF $\leq 35$.
- **General:** remaining clips.

Group sizes use `--n-frozen`, `--n-per-vendor`, `--n-low-ef`, and `--n-general`. Seeded sampling draws patients, then up to `--clips-per-patient` clips each (default 1), recording each clip's inclusion probability. Every nonempty group must contribute at least one clip; a zero target is rejected.

References come from unselected patients. Samples leaving no references are rejected; layouts with fewer than 25 clean references are counted and warned about. `sheet --all-clips` shows every readable sampled clip. `calibrate` uses inclusion-probability weights to estimate overall and per-group (`by_stratum`) performance. `sheet` rejects other sampling methods for these lists.

**The first validation list.** 266 clips of 266 training patients, none of the first shortlist's or its reference clips' (30 suspected frozen, 30 low EF, 50 general, and of the other vendors 50 each of GE Healthcare, GEMS and Philips, the 5 Siemens patients and the one ARTEFACT clip), labelled blind on the sheet: 2 measured, 5 still, 259 clean. By vendor the list holds 106 GE Vingmed clips, 51 GE Healthcare, 51 GEMS, 52 Philips, 5 Siemens and 1 ARTEFACT. The hybrid policy, with the text pass, decided (excluded / uncertain / kept):

| Labelled | Excluded | Uncertain | Kept |
|---|---|---|---|
| Measured (2) | 1 | 0 | 1 |
| Still (5) | 4 | 0 | 1 |
| Clean (259) | 2 | 8 | 249 |

- The measured clip excluded was caught by its GE header (`MEASURE`); its traced contours gave no mark. The one kept, a GE Vivid E9 clip of 8 frames at 2 fps, names image data `0041` in ImageType (the other 207 GE clips `0001` or `0011`) and has graphics beside the image on 3.2% of the frame (clean GE clips 0.69% at most), with no `MEASURE`, no caliper pair and no text read. Such codes now make a clip uncertain (**Unusual image data**).
- The still clip kept is a 5-frame GE color clip with half its frames repeated and 19% of its field changing; a clean 8-frame color clip moving on 3.9% of its field, under the 5% line, was excluded as still. Short color clips sit on the line both ways.
- The other clean clip excluded is the archive's one ARTEFACT clip, a moving loop stored as a secondary capture (`not_ultrasound_image`).
- The 8 clean clips left uncertain hold a single caliper or a traced contour found among bright static marks: GE Healthcare 1 of 51 clips, GEMS 4 of 51, Philips 3 of 52, GE Vingmed none of 106; a second look at the two found over the whole clip saw no mark.
- OCR read measurement text on none of the 266 clips, so deciding by it for the clips without a GE header (now the default) changes no decision here. The pixel reading of the mode agreed with the GE headers on 126 of 127 B-mode and 72 of 79 color clips; of the 58 clips without a GE header it read 38 as B-mode and 20 as color.

No labelled clip of a vendor without a GE header was measured, so the measurement rules for those vendors are untested on positives; and the list is a cluster sample (patients, then clips), so the weighted estimates carry no upper bound.

### Provenance

Each step deletes its previous outputs (and downstream outputs in `radar_pilot.py`), reads inputs once, and hashes inputs, outputs, and its code (`code_sha256`), alongside the Git commit. Provenance records (`<out>.info.json`, `shortlist.json`, `selection.json`) are written last. If an input changes mid-run, outputs are removed and the step stops. Downstream steps reject files lacking matching records.

- `features` records its pass-1 extractor, constants, and library versions; `flag` rejects incompatible features.
- `text` carries that record over and adds its own (code, constants, the OCR environment's library versions); `flag` rejects text read by other code. Code is recorded as its source without docstrings and comments, which is the same in every Python release, so the OCR and training environments agree on it.
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

By default, the categories draw recent GE Vingmed studies (`--vendor`, `--years`). Studies of other vendors join as vendor slots (**Selection**).

`--videos` must point to the archive manifest containing DICOM acquisition columns, generated by `build_manifests.py` at commit `57204d1` (`icm/archive_pretraining`; `experiments/icm_archive/manifests`). The current branch's builder lacks these fields, so `shortlist` and `screen_validation.py` check for them and record the build commit.

### Categories (priority order)

1. **Normal:** EF $\geq 55$, AV mean gradient $<20$ mmHg, RVSP $<35$ mmHg, and no moderate-or-worse valve statement.
2. **LV dysfunction:** EF $\leq 35$.
3. **Valve disease:** AV mean gradient $\geq 40$ mmHg, or moderate/severe mitral (MR) or aortic regurgitation (AR) in report text.
4. **Right heart:** RVSP $\geq 50$ mmHg with EF $>35$, and no left-sided valve disease (AV mean gradient $\geq 40$ mmHg, moderate/severe MR or AR): a raised RVSP that the other categories do not account for. Tricuspid regurgitation, which often comes with it, is allowed. `select --n-studies 4` takes it (the default 3 takes the first three).

MR/AR grade columns are ignored because their codebook is unverified. The text parser recognizes French and English names/abbreviations (e.g., `insuffisance mitrale`, `fuite aortique`, `IM`, `MR`, `AI`) and written/numeric grades (e.g., `modérée`, `sévère`, `grade 3`, `3/4`, `2+`; `1/4` is mild).

A grade counts only if it appears in the same sentence segment as the finding (segments split at commas and connectors such as `et`, `and`, `avec`, `with`, `mais`, `but`), is not negated or uncertain, cannot describe another finding (stenosis, calcification, prolapse, etc.), and is not an adverb (`sévèrement`). Other moderate-or-worse valve wording is **possible**, so the study is neither normal nor definite valve disease. This conservative rule also treats `insuffisance mitrale sévère sur prolapsus` as possible.

Of 249,084 cleaned reports (2006–2026), 88.7% contain no such moderate-or-worse valve statement; 14,485 mention significant MR or AR.

### Selection

`shortlist` samples 8 studies per category in seeded order for screening. Each clip gets an opaque global key: an HMAC of its path using a secret stored in the pilot folder.

**Vendor slots.** `--vendor-studies 'GE Healthcare Ultrasound=1' 'GEMS Ultrasound=1' 'Philips Medical Systems=1'` adds, per vendor, that many studies outside the categories (`vendor:<vendor>` in `shortlist.csv`): training studies of that vendor alone with a linked report and enough classified clips of the required view families, of any year (the categories' `--years` would leave out every Philips study, all from 2011 to 2016), drawn from a random stream of their own, with their own reference clips, so that the category studies and their reference clips are those of a shortlist without slots. Of the training studies with a linked report and the required views, GE Healthcare has 89 (2019 to 2024), GEMS 13 (2010 to 2012) and Philips 14 (2011 to 2016); Siemens has none with a linked report. Such a study enters on its report, whatever its category (one of the 14 Philips studies falls in one). Its classified clips need the B-mode and color pairing too, checked at `select` on their resolved mode. `--exclude-patients <validation_clips.csv>` keeps a validation list's patients out of every draw, categories and reference clips included, so that the list stays independent of the pilot.

`select` chooses one eligible study per category and per vendor slot from distinct patients, on the clips the screen keeps: excluded clips are removed, and uncertain ones are left out and listed with their reasons (`uncertain_clips.csv`, every shortlisted study's), never taken for invalid. Each clip's mode is first resolved: the header's (`mode_source` `header`) or the screen's pixel reading (`pixels`); `pilot_clips.csv` keeps the header's as `header_acq_category`, and `selection.json` records the reading against the header and the clips per mode source. It searches all valid combinations to maximize ranking without allowing one category to consume another's only eligible patient. It first verifies that the files still match the screened media. If no combination exists, it writes nothing and reports the qualifying studies per category and the shortlisted clips' decisions. Flags without decisions (an older `flag`) are refused. Thresholds are recorded, with a warning if provisional.

### Candidates

`candidates` creates column-review-v2 prefills and view/type review items for retained classified clips, keyed by global key, each candidate at its resolved mode (`candidates_by_mode_source` in its summary). It rejects incomplete selections, changed provenance since `select`, mismatched report files, and provisional screens unless `--provisional-screen` is supplied.

Its summary records SHA-256 hashes of the report and of the review spec (`SPEC` in `tools/labelbox/column_review.py`). Patient identities, report text, and the key map remain in the private pilot folder.

## Not done here

- **Upload.** The archive's clips are treated as de-identified. The uploader of the Labelbox workflow test
  (`tools/labelbox/pilot_upload.py`, branch `issue_22/labelbox_workflow_test`) reads the same files `candidates`
  writes, and puts each kept clip's playable copy under its global key with the form and the prefills.
- **Report text in Labelbox.** The v2 context attachment must carry no report text: each proposal names its report
  field, and the report context keeps only numbers (`context_attachment` in `tools/labelbox/column_review.py`).
