# RADAR Caption Pilot: Study Selection and Clip Screening (Issue #22)

Select 2–3 report-linked studies from training-split patients, one per report
category, with optional studies from other vendors. Review every eligible clip in Labelbox:

- [Column Review](labeling_instructions_v1.md#column-review): classified B-mode or color Doppler clips, with candidate captions.
- [View and Type Review](labeling_instructions_v1.md#view-and-type-review-other-clips): `OTHER` or unclassified clips, without captions.
  These may use any acquisition mode.

The workflow is `shortlist -> features -> text -> flag -> select -> candidates -> Labelbox upload`.

## What is removed, and where

| Removed | Rule | Where |
|---|---|---|
| Single-frame DICOM objects: still B-mode measurement frames, spectral Doppler stills, report and measurement pages | Not present in the video archive; never considered. | Before shortlist |
| Classified views in unsupported acquisition modes | Classified views must be B-mode or color Doppler, determined from GE headers or other vendors' pixels. `OTHER` accepts any mode. | `radar_pilot.py shortlist` |
| Unreadable clips | Video decoding fails (`unreadable`). | `clip_screen.py` |
| Non-ultrasound objects | DICOM SOP class identifies a non-ultrasound object, regardless of visible content (`not_ultrasound_image`). | `clip_screen.py` |
| Still clips | Insufficient motion in the field of view, including frozen multi-frame videos (`still`). | `clip_screen.py` |
| Analysis screens | GE: matching header pattern. Other vendors: header evidence and static graphics outside the image (`analysis_screen`). | `clip_screen.py` |
| Measurement overlays | Measurement indicated by the header or named measurement text detected by OCR. Calipers and contours are recorded but do not independently trigger exclusion. | `clip_screen.py` |
| Uncertain clips | Partial or conflicting evidence; omitted from the pilot and listed separately, not classified as invalid. | `clip_screen.py`, `radar_pilot.py select` |

Exclusions are automatic. Only retained clips reach Labelbox for human review. No clip is excluded solely for being short. Reviewers report missed exclusions in the form's Comments; affected clips are removed and the screening rule corrected.

## Clip screening (`data/clip_screen.py`)

The screen uses DICOM metadata, video motion, pixel evidence, and OCR to classify each clip as **keep**, **exclude**, or **uncertain**.

**Features.** Samples up to 48 frames per clip and extracts motion, DICOM evidence, image regions, and static marks.

**Text.** Uses RapidOCR on three sampled frames to detect measurement text. Named measurements trigger exclusion; units alone or incomplete readings cause uncertainty. Scanner settings and isolated velocity scales are ignored.

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

**Unusual image data.** GE's fourth `ImageType` value identifies tissue (`0001`), tissue with color flow (`0011`), or no image data (`0000`, analysis screen). In 46,293 GE headers, 22 GE Vingmed clips (0.1%) instead had `0041` or `0051`. All were `OTHER`-view 3D displays: 11 contained measurements, 7 showed rendered volumes, and 4 were clean multipanel images. All 22 matched conventional-image controls were clean. OCR missed the drawn measurements because only scale labels appeared. Such unusual GE headers therefore make clips `uncertain` (`unusual_image_content`). Other vendors' headers provide no equivalent evidence (`unavailable`).

**Brief measurement overlays.** Static-mark detection uses half-clip windows spaced one quarter-clip apart. Marks appearing briefly may be missed; for example, calipers visible only in frames 18–29 of a 48-frame clip were not detected. Shorter windows could detect them but increase false positives from tissue. This remains a known limitation pending labeled validation.

## Calibration and validation

**Screening thresholds remain provisional** until validated on human-labeled clips.

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
- **3D displays:** A measured 3D display with an ordinary GE header was missed in the real-loop validation. As an `OTHER` clip, it reaches view/type review, where reviewers can identify analysis screens.
- **Short color clips:** One clean 8-frame color clip with 3.9% motion was incorrectly excluded by the 5% stillness threshold.
- **Brief overlays:** Measurements appearing for only a short portion of the video may escape static-mark detection.
- **Other vendors:** Header conventions are supported by GE surveys but clinically labeled mainly on GE Vingmed. The 73 labeled clips without GE headers contained only 1 still and no measured clips, leaving measurement-screening sensitivity unverified.

## Study selection (`data/radar_pilot.py`)

Eligible studies must have:

- A training-split patient with a TTE examination.
- Exactly one matching cleaned DeepECHO report.
- At least 20 retained classified-view clips covering A4C, A2C, PLAX, and PSAX.
- At least one view family represented in both B-mode and color Doppler.

The archive manifest must contain DICOM acquisition fields from `build_manifests.py` at commit `57204d1`.

Clinical categories, in priority order:

1. **Normal:** EF $\geq 55\%$, AV mean gradient $<20$ mmHg, RVSP $<35$ mmHg, and no moderate-or-worse valve statement.
2. **LV dysfunction:** EF $\leq 35\%$.
3. **Valve disease:** AV mean gradient $\geq 40$ mmHg, or definite moderate/severe mitral or aortic regurgitation.
4. **Right heart:** RVSP $\geq 50$ mmHg, EF $> 35\%$, without significant left-sided valve disease.

The default selects the first three categories; `--n-studies 4` includes right heart. Report parsing supports French and English findings, but ambiguous or negated statements do not establish definite valve disease. Unverified MR/AR grade columns are ignored.

The selection workflow is:

1. **`shortlist`:** Seededly sample up to 8 studies per category, plus optional vendor slots (`--vendor-studies`). Assign opaque `global_key` identifiers and select reference clips from other studies.
2. **`features` $\rightarrow$ `text` $\rightarrow$ `flag`:** Screen the shortlisted and reference clips using the evidence rules above.
3. **`select`:** Choose one eligible study per category/vendor slot from distinct patients, using only kept clips. Resolve acquisition modes, verify media integrity, and save uncertain clips separately (`uncertain_clips.csv`).
4. **`candidates`:** Generate column-review prefills and view/type review items, verifying selection, reports, and screening provenance.

`--exclude-patients` prevents validation patients from entering pilot or reference selection.

## Provenance and data safety

**Replaced clips.** Archive files may change without changing paths. `features` records each video's SHA-256 (`video_sha256`), DICOM regions, and relevant header evidence (image content, measurements, source image, ECG). Downstream steps (`sheet`, `calibrate`, `select`, `candidates`) reject missing or changed media; `sheet` and `select` also recheck headers. Read failures stop the step. `candidates` exports hashes to `global_keys.csv`.

Every step records input, output, media, and code SHA-256 hashes alongside the Git commit. Changed videos, DICOM headers, incompatible extractors, incomplete outputs, and mismatched provenance are rejected. Records are written last so failed runs cannot be reused.

Outputs and review media remain in owner-only directories (`0700`, files `0600`), with sensitive errors in private logs. Patient identifiers, report text, and global-key mappings remain private.

Labelbox upload is handled separately by `tools/labelbox/pilot_upload.py` on branch `issue_22/labelbox_workflow_test`. Context attachments must contain numeric values and report-field references, **not raw report text**.