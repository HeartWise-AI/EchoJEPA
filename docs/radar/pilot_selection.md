# RADAR Caption Pilot: Study Selection and Clip Screening (Issue #22)

Select 2–3 report-linked studies from training-split patients, one per report
category, with optional studies of named vendors (vendor slots) screened as backups. The pilot captions V1: classified B-mode clips, each reviewed in the
[Column Review](labeling_instructions_v1.md#column-review) with candidate captions. Color Doppler is left to V2.
`OTHER` and unclassified clips serve pretraining only: they are not captioned and do not reach the pilot, so the
[View and Type Review](labeling_instructions_v1.md#view-and-type-review-other-clips) serves only the view-classifier relabelling.

The workflow is `shortlist -> features -> text -> flag -> select -> candidates -> Labelbox upload`.

## What is removed, and where

| Removed | Rule | Where |
|---|---|---|
| Single-frame DICOM objects: still B-mode measurement frames, spectral Doppler stills, report and measurement pages | Not present in the video archive; never considered. | Before shortlist |
| `OTHER` and unclassified views | Pretraining only (`other_view`). | `radar_pilot.py shortlist` |
| Unsupported acquisition modes | The header must declare B-mode or color Doppler (`acquisition`). | `radar_pilot.py shortlist` |
| Color Doppler | From GE headers (`color_flow`), or from other vendors' pixels after the screen. | `radar_pilot.py shortlist`, `select` |
| Clips of 3 frames or fewer | Counted as stills (`few_frames`). | `radar_pilot.py shortlist` |
| Unreadable clips | Video decoding fails (`unreadable`). | `clip_screen.py` |
| Non-ultrasound objects | DICOM SOP class identifies a non-ultrasound object, regardless of visible content (`not_ultrasound_image`). | `clip_screen.py` |
| Still clips | Insufficient motion in the field of view, including frozen multi-frame videos (`still`). | `clip_screen.py` |
| Analysis screens | GE: matching header pattern. Other vendors: header evidence and static graphics outside the image (`analysis_screen`). | `clip_screen.py` |
| Measurement overlays | Measurement indicated by the header or named measurement text detected by OCR. Calipers and contours are recorded but do not independently trigger exclusion. | `clip_screen.py` |
| 3D or multi-panel displays | GE Vingmed image code `0041` or `0051` (`3d_or_multipanel`, see Unusual image data). | `radar_pilot.py select` |
| Uncertain clips | Partial or conflicting evidence; omitted from the pilot and listed separately, not classified as invalid. | `clip_screen.py`, `radar_pilot.py select` |

Every clip of a shortlisted study that is left out is listed with its reasons: `removed_clips.csv` (`shortlist`: `other_view`, `acquisition`, `color_flow`, `few_frames`, and `vendor` or `year` for a category study's clip outside `--vendor` or `--years`) and `not_kept_clips.csv` (`select`: those plus the excluded and uncertain clips after the screen, `color_flow` and `3d_or_multipanel` included, and clean clips of unchosen backup studies (`step=selection`, `decision=not_chosen`, `reasons=study_not_chosen`), with `step`, `decision`, `reasons` and `in_pilot`). A clip lists every reason that applies. Like every output they hold patient IDs and paths and stay in the private `--out-dir`.

Exclusions are automatic. Only retained clips reach Labelbox for human review. Short clips are kept unless they have 3 frames or fewer. Reviewers report missed exclusions in the form's Comments; affected clips are removed and the screening rule corrected.

## Clip screening (`data/clip_screen.py`)

The screen uses DICOM metadata, video motion, pixel evidence, and OCR to classify each clip as **keep**, **exclude**, or **uncertain**.

**Features.** Samples up to 48 frames per clip and extracts motion, DICOM evidence, image regions, and static marks.

**Text.** Uses RapidOCR on three sampled frames to detect measurement text. Named measurements and a heart rate (`HR 63`, `63 bpm`) trigger exclusion; other units alone or incomplete readings cause uncertainty. Scanner settings and isolated velocity scales are ignored, and so are OCR words taller than a tenth of the frame: OCR can read a structure of the image as a word (a PSAX aortic valve read as `LA`, joined with a depth label into `LA 0`), while no overlay text is that tall.

**Flag.** Combines motion, DICOM header, pixel, and OCR evidence to classify each clip as `keep`, `exclude`, or `uncertain`, with reasons. Uses the default `hybrid` policy; older `header` and `pixel` policies remain for comparison.

**Sheet.** Generates blinded HTML review pages where humans label clips for stillness, measurements, and other exclusion reasons.

**Challenge.** Samples difficult cases (e.g., calipers, unusual headers, and possible missed measurements) to test screening weaknesses.

**Calibrate.** Compares screening decisions against human labels to measure errors, evaluate thresholds, and compare policies.

Motion is measured within the DICOM scan area, a pixel-detected field of view, or the whole frame. A clip is considered still below 5% field-of-view motion; whole-frame readings require less than 1% motion and at least 90% repeated frames for exclusion. Changing Doppler color counts as motion, but moving cursors alone do not.

GE header conventions identify analysis screens and measurements. Other vendors require additional evidence because their header conventions are unverified. When acquisition mode is unavailable from the header, a pixel-based color-flow detector provides a fallback (`pixel_mode`), which reviewers confirm.

Calipers, contours, and static marks are recorded but do not determine eligibility: testing found low detection sensitivity and false positives.

### Run the screening pipeline

Extract clip and reference features, detect measurement text, then classify clips using the combined evidence:

    python data/clip_screen.py features --clips <clips.csv> --out <screen_dir>/features.parquet
    python data/clip_screen.py features --clips <reference_clips.csv> --out <screen_dir>/reference.parquet
    <deid_ocr>/bin/python -E -s data/clip_screen.py text --features <screen_dir>/features.parquet \
        --out <screen_dir>/features_text.parquet --workers 6 --threads 2
    python data/clip_screen.py flag --features <screen_dir>/features_text.parquet --reference <screen_dir>/reference.parquet \
        --out <screen_dir>/flags.parquet

The resulting `flags.parquet` contains each clip's screening decision and reasons.

### Errors

Decoder failures (cannot open, no frames, or decoding errors) exclude clips as `unreadable`. Other failures—including missing files, dependencies, memory errors, worker crashes, and timeouts (`--clip-timeout`, default 600 s)-stop the run. Sensitive paths and error details go to private logs. Requires `decord`, `pydicom`, and `imageio-ffmpeg` for contact sheets.

### Screening edge cases

**Unusual image data.** GE's fourth `ImageType` value identifies tissue (`0001`), tissue with color flow (`0011`), or no image data (`0000`, analysis screen). In 46,293 GE headers, 22 GE Vingmed clips (0.1%) instead had `0041` or `0051`, all `OTHER` views. Labelled blind, 7 were 3D renders and 15 displays of two or more panels (11 with measurements, 4 clean); none was a single 2D view, and not every one is 3D. All 22 matched conventional-image controls were clean. OCR missed the drawn measurements because only scale labels appeared. The screen makes unusual GE headers `uncertain` (`unusual_image_content`); `select` excludes the two labelled codes as `3d_or_multipanel`, while other unusual codes, never labelled, stay uncertain. Other vendors' headers provide no equivalent evidence (`unavailable`). 3D displays with an ordinary code (`0001`) are not caught by the code (see Known limitations).

**Brief measurement overlays.** Static-mark detection uses half-clip windows spaced one quarter-clip apart. Marks appearing briefly may be missed; for example, calipers visible only in frames 18–29 of a 48-frame clip were not detected. Shorter windows could detect them but increase false positives from tissue. This remains a known limitation pending labeled validation.

## Calibration and validation

**Screening thresholds remain provisional** until validated on human-labeled clips. The current rule and default thresholds are calibrated from its report on 266 validation clips of independent patients: 258 of 260 clean clips kept, all 4 still clips excluded, and both measured clips excluded. Positive cases came from 130 real loops (all 58 measured clips excluded, all 64 clean clips kept) and synthetic overlays (all 53 results boxes read). The known limitations below still apply.

The calibration reads the files as re-masked by the archive-wide fix, which replaced 2 validation clips in place; those 2 were answered again on a new sheet of the same 266 clips, and the other answers carried over (same video bytes). The second-shortlist audit and the diagnostic sample were left out of this calibration: their samples were drawn from flags of videos replaced since. Against the previous calibration, counting a heart rate as measurement text turned 9 measured clips from uncertain to excluded (1 validation, 8 unusual-code displays) and excluded the real-loop clip it kept; no clean clip changed.

The calibration workflow is:

1. Generate blinded review sheets (`sheet`) with excluded, uncertain, and kept clips.
2. Label stillness, measurements, other exclusion reasons, and cursor lines.
3. Run `calibrate` to compare detector performance, thresholds, and screening policies.
4. Label difficult cases using `challenge`, including measurement overlays, unusual headers, and clips from other vendors.
5. Validate the finalized policy on independent patients sampled with `screen_validation.py`.

Validation samples cover suspected frozen clips, other vendors, low-EF cases, and general clips. Sampling records inclusion probabilities; calibration reports missed exclusions, false exclusions, and performance by group. A population bound on ineligible kept clips is available only for a suitable random audit of that kept population, not diagnostic, challenge, or patient-clustered validation samples.

### Screening evaluation

**Real-clip pixel screening.** Under the earlier `pixel` policy, 300 GE clips (150 B-mode, 150 color) showed excessive exclusions:
- All 75 screened color clips were excluded (43 missing baselines, 32 measurement flags).
- 14 of 75 screened B-mode clips were flagged for measurements.
- Most detected marks were small (4–5 pixels) and appeared only in temporal windows, potentially reflecting tissue motion or compression artifacts.

Increasing the minimum mark size did not restore sufficient clean color references. These results motivated replacing pixel-only exclusions with the `hybrid` policy.

**Labelled diagnostic sample.** A blinded sample of 216 clips (97 screened, 119 references) contained 27 measured, 12 still (all analysis screens), and 187 clean clips. The `hybrid` policy was developed on this sample, so results are in-sample and potentially optimistic.

Counts below are **excluded / uncertain / kept**:

| Policy | Measured (27) | Clean (187) | Clean clips excluded |
|---|---|---|---|
| `pixel` (earlier) | 27 / 0 / 0 | 124 / 0 / 63 | 66% |
| `header` (earlier) | 26 / 0 / 1 | 25 / 0 / 162 | 13% |
| `hybrid` (header + pixels) | 26 / 1 / 0 | 0 / 3 / 184 | 0% (1.6% uncertain) |
| `hybrid` (GE header alone) | 27 / 0 / 0 | 0 / 1 / 186 | 0% (0.5% uncertain) |

All policies excluded the 12 still clips, but none was an ordinary recording, so still detection on ordinary videos required separate validation.

On the first shortlist (1,061 GE clips), the earlier GE-header-based hybrid rule kept 1,052, excluded 8 (6 analysis screens, 2 measurement headers), and marked 1 uncertain. That uncertain clip is kept under the current rule. These results predate SOP-class screening.

**Second-shortlist audit.** Of 1,973 clips, the previous screening rule kept 1,946, marked 15 uncertain, and excluded 12. Blinded review covered all excluded and uncertain clips plus 100 randomly sampled kept clips:

- **Kept:** All 100 were eligible. The one-sided 95% upper bound for ineligible clips was 2.95%, applicable only to the 1,946 clips kept by that earlier rule.
- **Excluded:** 11 were measured; 1 clean Philips clip was incorrectly excluded because of a detected caliper pair.
- **Uncertain:** All 15 were clean, flagged for single calipers or traced contours.

All 16 clips incorrectly flagged by measurement structures are now kept because those signals proved unreliable. The revised rule keeps 1,960 clips, but the previous 2.95% bound does not apply to this changed population.

OCR found named measurement text on 6 of the 10 analysis screens and none of the other 1,963 clips. Two clips with unusual image data remain held back separately.

### Known limitations

A `keep` decision does **not** guarantee that a clip has no measurement overlays.

- **Drawn measurements:** Calipers and contours are poorly detected without a measurement header. Structure detection found at most 5 of 53 synthetic copies per overlay type. Other vendors may lack the necessary header signals entirely.
- **Measurement text:** OCR found all 53 synthetic results boxes, but real measurement text was observed mainly on analysis screens already identified by headers. `extra_cells` is too nonspecific to replace OCR.
- **3D displays:** A measured 3D display with an ordinary GE header (code `0001`, no MEASURE) was missed in the real-loop validation until a heart rate counted as measurement text: it is now excluded only because it shows one (`HR58`). It was an `OTHER` clip, which the pilot removes anyway; a classified 3D display with such a header and no heart rate shown would reach the pilot.
- **Short color clips:** One clean 8-frame color clip with 3.9% motion was incorrectly excluded by the 5% stillness threshold.
- **Brief overlays:** Measurements appearing for only a short portion of the video may escape static-mark detection.
- **Other vendors:** Header conventions are supported by GE surveys but clinically labeled mainly on GE Vingmed. The 73 labeled clips without GE headers contained only 1 still and no measured clips, leaving measurement-screening sensitivity unverified.

## Study selection (`data/radar_pilot.py`)

Eligible studies must have:

- A training-split patient with a TTE examination.
- Clips of any vendor (no `--vendor`); with `--vendor GE`, of GE's three vendor names alone (GE Vingmed Ultrasound, GE Healthcare Ultrasound, GEMS Ultrasound), whose headers give the acquisition mode; with `--vendor "GE Vingmed Ultrasound"`, of GE Vingmed alone, the vendor the screen was mostly labelled on. Another vendor's mode is read from the pixels by the screen; each clip's source is `mode_source`.
- Consistent patient IDs and study dates across every training clip of each study. Conflicts stop shortlisting before patient exclusions, year/vendor filters or clip rules can hide a row; the error reports only the number of affected studies. Correct the source metadata and rebuild the manifest before retrying.
- One cleaned DeepECHO report, found through the PACS index by accession (`--pacs`), with the report, PACS and manifest agreeing on the patient and PACS and manifest on the date. Rows repeated exactly count once; an accession whose rows differ excludes its entire study even when another accession has a valid report. Multiple reports for a study are ambiguous before patient/date filtering, so an incompatible second report cannot be hidden by a compatible one. No EF label is needed: EF and RVSP are joined where they exist for that report, so a study without EF can still be valve disease.
- At least 20 retained classified-view B-mode clips (`--min-clips`) covering A4C, A2C, PLAX, and PSAX (`--view-families`; a family is a view's name before its first `_`). The clip rule has already left only such clips; for other vendors, B-mode is counted after the screen reads the mode.

The archive manifest must contain DICOM acquisition fields from `build_manifests.py` at commit `57204d1`.

Clinical categories, in priority order:

1. **Normal:** EF $\geq 55\%$, AV mean gradient $<20$ mmHg, RVSP $<35$ mmHg, and no moderate-or-worse valve statement.
2. **LV dysfunction:** EF $\leq 35\%$.
3. **Valve disease:** AV mean gradient $\geq 40$ mmHg, or definite moderate/severe mitral or aortic regurgitation.

The default selects one study of each category; `--n-studies 2` takes the first two. Vendor slots (`shortlist --vendor-studies`, for example one study each of GE Vingmed Ultrasound, GE Healthcare Ultrasound, GEMS Ultrasound and Philips Medical Systems, so that every vendor has a screened study) add studies of a named vendor, within `--vendor` when given, of any year and whatever their report category, from patients not drawn in the categories or an earlier slot. A slot is for its vendor: `select` never takes its study for a category. They are screened with the rest of the shortlist but reach the pilot only when `select --vendor-slots` names them. Report parsing supports French and English findings, but ambiguous or negated statements do not establish definite valve disease. Unverified MR/AR grade columns are ignored.

The selection workflow is:

1. **`shortlist`:** Seededly sample up to 5 studies per category (`--per-category`), plus optional vendor slots (`--vendor-studies`). Assign opaque `global_key` identifiers, select reference clips from other studies, and list the shortlisted studies' clips the screen will not see (`removed_clips.csv`).
2. **`features` $\rightarrow$ `text` $\rightarrow$ `flag`:** Screen the shortlisted and reference clips using the evidence rules above.
3. **`select`:** Choose one eligible study per category, and per vendor slot named in `--vendor-slots`, from distinct patients, using only kept clips; the other shortlisted studies stay screened backups. Resolve acquisition modes, verify media integrity, save uncertain clips separately (`uncertain_clips.csv`), and list every clip left out with its reasons (`not_kept_clips.csv`). Write the funnel (`funnel.csv`, also in `summary.md`; the shortlist's part is in `shortlist.json`): from the archive manifest to the pilot, the videos, studies and patients each step leaves, and what it leaves out by reason, counted in clips or studies. A vendor slot's row counts the shortlist with its study added, its own counts in the note. Each clip or study counts once, under its first reason in the rule's order, so the counts add up to what the step leaves out: `non_train` (the splits in the note); the clip rule's `other_view`, `acquisition`, `color_flow`, `few_frames`; `no_single_report` (the link's own reasons in the note); `min_clips`, `view_families`; the screen's and the pilot rule's reasons. Where a clip or study has several, the note also counts every reason. Aggregate only, for the CONSORT chart.
4. **`candidates`:** Generate column-review prefills, verifying selection, reports, and screening provenance (`view_type_items.csv` stays empty).

`select` refuses a shortlist drawn before the V1 rule, and `candidates` a selection made before it.
Artifacts must also record the current study identity and report-link policy. Shortlists and selections made before
these checks must be regenerated from `shortlist`, followed by screening and selection; they cannot be reused.

`--exclude-patients` prevents validation patients from entering pilot or reference selection.

## Provenance and data safety

**Replaced clips.** Archive files may change without changing paths. `features` records each video's SHA-256 (`video_sha256`), DICOM regions, and relevant header evidence (image content, measurements, source image, ECG). Downstream steps (`sheet`, `calibrate`, `select`, `candidates`) reject missing or changed media; `sheet` and `select` also recheck headers. Read failures stop the step. `candidates` exports hashes to `global_keys.csv`.

Every step records input, output, media, and code SHA-256 hashes alongside the Git commit. Changed videos, DICOM headers, incompatible extractors, incomplete outputs, and mismatched provenance are rejected. Records are written last so failed runs cannot be reused.

Outputs and review media remain in owner-only directories (`0700`, files `0600`) outside the repository (`--out-dir` inside it is refused), with sensitive errors in private logs. Patient identifiers, report text, and global-key mappings remain private.

Labelbox upload is handled separately by `tools/labelbox/pilot_upload.py` on branch `issue_22/labelbox_workflow_test`. Context attachments must contain numeric values and report-field references, **not raw report text**.

## Before uploading the pilot (issue #22)

Selection and screening do not establish de-identification. In the approved secure environment, verify every video and caption against the agreed de-identification check, inspect the exact outgoing metadata, and keep all source-to-pilot mappings, report clauses, identifiers, dates and local paths private. Upload only the approved video, caption and structured review metadata, including ontology and available caption-generation versions.

Issue #22 calls for 3–5 representative studies and approximately 100–150 clips. The two-study option is useful for a trial; it does not satisfy that requirement. Reconcile the final composition with issue #22 before upload. The selection tool does not enforce the total clip range.
