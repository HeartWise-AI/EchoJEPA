# RADAR caption review: instructions for reviewers (v1)

Version: `radar-caption-ontology-v2`, with column review v4 and view/type review v2.
The field names and choices below match the English labels shown in Labelbox.

## What you review

Every valid video of a pilot study goes to one of two Labelbox projects:

- **Column review**, for a clip of a classified view in B-mode or color Doppler. You see the clip, its proposed view
  and acquisition mode, its **prefilled findings** and a **candidate caption** with one sentence per finding. The
  findings are prefilled from what the study's report states explicitly: a report section that says nothing
  prefills nothing, and no finding is assumed normal. They are suggestions; your review decides what this clip
  supports.
- **View and type review**, for a clip the view classifier labels `OTHER`, or does not classify. You see the clip
  only, with no caption, and say what it is and which view it shows.

Still clips, clips with visible measurements and classified views in other acquisition modes are removed before
upload. If one reaches you anyway, say so in **Comments**.

Next to each clip of the column review, a text attachment lists the original proposals, each with the report field it
comes from, and the report's numbers (EF, dimensions, wall-motion scores) as study context. Those numbers describe the
study, not this clip, and are never findings to confirm.

The clip is shown whole, at its original resolution, after de-identification: identifying text is blacked out. The
model receives the same clip, every frame resized whole: nothing is cropped. Judge every finding on the whole image
and the whole loop (see [Model input and review display](ontology_v2.md#model-input-and-review-display)).

The terms used below:

- **Finding:** one column with its value, such as "LV function: Mildly reduced". Each gives one sentence of the
  caption.
- **Supported:** this clip shows it, at the stated value.
- **Incorrect:** this clip shows another value, or contradicts it.
- **Cannot assess:** this clip does not let you decide (see [Uncertainty](#uncertainty)).

## Column review

### How to work through a clip

1. Watch the whole loop, more than once if needed.
2. Decide whether the clip can be interpreted at all (question 1). If it cannot, give the reason, select **Review
   completed** and submit: nothing else is read.
3. Check the prefilled view and acquisition mode, correcting either value directly when needed; a PSAX clip has no
   prefilled view, so choose its level (questions 2 and 3).
4. Untick the prefilled structures the clip does not show and, for a subcostal clip, check its subtype (questions 4
   and 5).
5. Go through the prefilled findings one by one: supported, incorrect or cannot assess (question 6).
6. Look for an important visible finding the candidate leaves out (question 7), and for a statement of the caption
   outside its findings (question 8).
7. Add a comment only if needed, select **Review completed** (question 11) and submit.

Every answer except **Review completed** and the view of a PSAX clip is prefilled: the clip interpretable, the view
and mode correct, every structure of the view visible, every finding supported, nothing missing or extra. Change what
this clip does not support; an answer you leave unchanged is your confirmation.

### The form

<!-- BEGIN GENERATED: column_form -->
| # | Question | Id | Required | Answers |
|---|---|---|---|---|
| 1 | Is this clip uninterpretable? | `uninterpretable` | yes | `no` No, it can be interpreted; `yes` Yes, nothing in it can be judged → Why? (a reason below) |
| 2 | View (prefilled - correct if needed) | `view` | yes | one of the 24 views |
| 3 | Acquisition mode (prefilled - correct if needed) | `modality` | yes | `bmode` B-mode; `color_flow` Color Doppler; `spectral` Spectral Doppler; `m_mode` M-mode; `mixed` Mixed / analysis acquisition; `unresolved` Unresolved |
| 4 | Structures visible in this clip, even partly (prefilled: untick any not visible) | `visible_targets` | no | `lv` LV; `rv` RV; `la` LA; `ra` RA; `ivs` Interventricular septum; `ias` Interatrial septum; `mitral` Mitral valve; `tricuspid` Tricuspid valve; `aortic` Aortic valve; `pulmonic` Pulmonic valve; `papillary_muscles` Papillary muscles; `chordae` Chordae; `lvot` LVOT; `rvot` RVOT; `pulmonary_artery` Proximal pulmonary artery; `aorta_root` Aortic root; `aorta_ascending` Ascending aorta; `aorta_arch` Aortic arch; `aorta_descending` Descending aorta; `pericardium` Pericardium; `ivc` IVC; `hepatic_veins` Hepatic veins |
| 5 | Subcostal subtype (required for subcostal clips) | `subcostal_level` | yes | `not_applicable` Not applicable (not subcostal); `unresolved` Cannot tell the subtype; `four_chamber` Subcostal four-chamber; `ivc` Subcostal IVC / hepatic veins |
| 6 | Prefilled findings — confirm or correct selected columns | `review_columns` | no | the prefilled columns, each: Proposed value (original retained in context) (one of the column's values) + Confirm this finding (`yes` Supported in this input; `no` Incorrect / should not be here → Why should it not be here? (an error reason) + Correction (`remove` Remove this claim, or one of the column's values); `cannot_assess` Cannot assess → Why can it not be assessed? (an assessment limitation)) |
| 7 | Important visible finding missing from the candidate? | `missing_information` | yes | `no` No; `yes` Yes → Select missing columns and values (any column the view carries, each: Missing visible value (one of the column's values) + Visible in this input? (`yes` Yes; `cannot_assess` Cannot assess)) |
| 8 | Additional statement that should not be here? | `unsupported_information` | yes | `no` No; `yes` Yes → Select offending columns / statements (any column or statement type, each: Why should this statement not be here? (an error reason); a statement type also needs Copy the offending statement (free text)) |
| 9 | Original caption (for reference only; edits are ignored) | `candidate_caption` | no | free text |
| 10 | Comments (no patient details) | `comments` | no | free text |
| 11 | I reviewed this input, checked the proposed values and structures, and confirmed or corrected the suggestions. | `review_completed` | yes | `yes` Review completed |
<!-- END GENERATED: column_form -->

The finding columns, their values and the views that carry them are defined in the machine-readable
[`deepecho_video_clip_annotation_specification.json`](../../configs/radar/deepecho_video_clip_annotation_specification.json).
The reasons shown in the form are:

<!-- BEGIN GENERATED: column_reasons -->
**Error reasons (an incorrect finding, an additional statement)**

| Id | Label |
|---|---|
| `wrong_value` | Wrong value / diagnosis |
| `wrong_severity` | Wrong severity |
| `wrong_location` | Wrong structure / segment |
| `contradicted` | Contradicted by visible evidence |
| `not_visible` | Claimed structure/finding not visible |
| `wrong_view` | Wrong view / target |
| `wrong_modality` | Needs another acquisition mode |
| `study_only` | Study-level claim in a local caption |
| `unsupported_report` | No supporting source / invented assertion |

**Assessment limitations (a finding that cannot be assessed)**

| Id | Label |
|---|---|
| `poor_quality` | Poor image quality |
| `incomplete_cycle` | Incomplete cardiac/respiratory cycle |
| `target_outside` | Target outside the displayed input |
| `unresolved_view` | View / slice level unresolved |
| `insufficient_input` | Needs additional views or measurements |

**Uninterpretable clips**

| Id | Label |
|---|---|
| `poor_quality` | Poor image quality or artifacts throughout |
| `no_cardiac_structure` | No identifiable cardiac structure (crop, zoom or depth) |
| `not_cardiac` | Not a cardiac ultrasound acquisition |
| `other` | Other: say why in Comments |

**Additional statement types (a statement the caption makes outside its columns)**

| Id | Label |
|---|---|
| `exact_ef` | Exact numerical EF |
| `valve_severity` | Integrated valve severity |
| `numeric_measurement` | Numeric dimension / velocity / gradient / pressure |
| `other_statement` | Other unsupported statement |
<!-- END GENERATED: column_reasons -->

### Rules

#### Judge this clip only

Judge what this clip shows, not what you know about the patient: the report, its numbers, the other clips of the
study and your own measurements do not count. A finding that is real for the patient but not visible in this clip is
not supported here.

#### Supported, incorrect, cannot assess

- **Supported:** you can see it at the stated value, as you would grade it by eye. Leave "Supported in this input".
- **Incorrect:** choose why (the error reasons above), then the correct value or "Remove this claim". Give a value
  only when this clip shows it; otherwise remove the claim. Removing a claim never states its opposite.
- **Cannot assess:** choose the limitation. It is neither an error nor a negative finding, and the finding stays out
  of the caption.

**Proposed value** keeps the original value: a correction goes in **Correction**, never in **Proposed value**.

#### Uncertainty

When you cannot decide whether a finding is supported, do not guess: choose **Cannot assess** with the closest
limitation. "View / slice level unresolved" is for a finding that depends on a view or level you cannot tell; "Needs
additional views or measurements" for a finding this clip alone cannot settle; "Target outside the displayed input"
for a structure cut off at the edge of the sector, outside the color box, or outside a zoomed clip.

#### Absent findings

Some findings state an absence: "No mitral regurgitant jet is seen on color Doppler", "No pericardial effusion". Judge them like any other finding: supported only if this clip could
show the finding (right view, right mode, the structure in the image and, for a jet, color over the valve through
systole) and it is not there. If the clip could not show it, choose **Cannot assess**.

A report that does not mention a finding is no evidence of its absence, and an absent finding that is not prefilled
is not missing information.

#### Prefilled normal findings

A normal finding is prefilled only when the report states it ("Valve tricuspide normale"). Confirm it only if this
clip shows it. Normal leaflet appearance does not establish normal motion, and neither establishes the absence of a
jet.

#### Clinical severity and scope

Function and size values are clip-level visual impressions, not a complete examination's diagnosis. A regurgitant jet
is only demonstrated or not demonstrated in this acquisition: no finding grades it, and a caption that states an
integrated grade, an exact EF or any number makes an additional statement (question 8). Regional wall motion
describes only the segments this clip shows. Report wall-motion scores keep their ambiguity: score 1 gives "Normal /
hyperkinetic", score 3 "Akinetic / severely hypokinetic". Do not infer normal motion in unseen segments, a whole-study
EF, stenosis severity, filling pressures or pulmonary hypertension.

#### Structures

**Structures visible in this clip** is prefilled with every structure of the view (of its subtype, for a subcostal
clip). A tick means visible, even partly: untick only a structure that is not in the clip at all. Do not tick a
structure that is not prefilled. Each view's structures are fixed, so that the same structures can be given to
the clips nobody reviews; how often each is unticked shows which ones a view really shows.

Only the findings about a ticked structure count. Some structures have no findings yet (the septa, the LVOT and RVOT,
the proximal pulmonary artery, the papillary muscles, the chordae, the hepatic veins, the descending aorta): for
those, only say whether they are visible.

A finding about a structure that is not in the clip at all is **Incorrect**: "Claimed structure/finding not visible".
A structure only partly in the image, too cut off to judge, is **Cannot assess**: "Target outside the displayed
input".

When you untick a structure, resolve each of its findings: mark it **Incorrect** and remove the claim, or tick the
structure again. Nothing changes your answers for you, and the form cannot catch this as you submit, so a finding left
supported on an unticked structure sends the review back. **Incorrect** here means this clip does not support the
statement, not that the patient does not have the finding.

#### Zoomed and LV-focused clips

A zoomed clip is prefilled with its base view's structures: untick those outside the zoom. An LV-focused clip is
prefilled with the LV and the view's valves. In both, no chamber size is claimed, because the whole chamber may not be
in the image; LV function, valve and root findings and wall motion are. A zoomed PLAX clip is narrower still: the LV,
the RV and the pericardium are not judged there, only the valves, the root and the ascending aorta.

#### LV wall thickness

Thickening in different places has different causes, so each has its own answer: "Thick septum with normal LV wall
thickness elsewhere" when the septum alone is thick, "Thickened LV posterior wall" when the posterior wall alone is, and
"Increased LV wall thickness" when the walls are thick together. Correct one to another if the clip shows it. The first
is claimed only in the views that list the septum (A4C, A5C, PLAX, `PSAX_BASAL`, and their zoomed and deep forms); the
second only in PLAX, A3C and `PSAX_BASAL` clips, which show that wall.

A basal septal bulge is a separate finding, "Focal basal septal thickening (septal bulge)": a localized thickening, not
a thick septum.

#### PSAX level, PLAX zoomed on the aortic valve and subcostal subtype

A PSAX clip opens with no view, because the view classifier does not give the slice level: choose `PSAX_BASAL`
(mitral level), `PSAX_MID` (papillary level) or `PSAX_APICAL`. A clip at the aortic valve is a `PSAX_AV` clip. When you
cannot tell the level, choose `PSAX`. The RV and chamber size are not judged at the PSAX levels. LV wall thickness is judged
at `PSAX_BASAL` only, the mitral valve at `PSAX_BASAL` and `PSAX_MID`, and the papillary muscles, the chordae and the mid
wall segments at `PSAX_MID` only. These structures of a level are not prefilled and you do not tick them: they come
with the view you choose.

A `PLAX_ZOOM` clip zoomed on the aortic valve is a `PLAX_ZOOM_AV` clip: correct the view. Only the aortic valve, LVOT,
aortic root and ascending aorta are judged there, and the prefilled findings about anything else leave the caption
without an answer from you.

For a subcostal clip, the four-chamber subtype and its structures are prefilled: switch an IVC sweep to "Subcostal IVC
/ hepatic veins", which brings the IVC and the hepatic veins, and mark the prefilled chamber findings incorrect ("Wrong
view / target"). IVC and chamber findings never mix.

For every other view, keep "Not applicable". When you correct the view to or from subcostal, correct the subtype too.

#### Missing information

Answer "Yes" to question 7 only for an **important** finding that is visible in this clip, that the view carries, and
that no prefilled finding states: choose the column and its value, and confirm it is visible in this input. Do not add
normal values the report left out, absent findings, measurements, or what only other clips show. A prefilled finding
with a wrong value is corrected under question 6, not added as missing. A finding you tick in question 6 that was not
prefilled also counts as missing information, if you mark it supported; add it under question 7 instead.

#### Additional statements

The candidate caption is built from its findings, so question 8 is rarely "Yes". Use it when the caption states
something outside its findings, such as a grade, an exact EF or a number: choose the column or the statement type and
the reason, and copy the statement when the form asks.

#### Uninterpretable clips

Answer "Yes" to question 1 only when nothing in the clip can be judged: artifacts or poor image quality throughout, no
identifiable cardiac structure, or not a cardiac ultrasound acquisition. Give the reason; with "Other", say why in
**Comments**. The other answers are then not read, and you need not change them.

If only some findings cannot be judged, the clip is interpretable: use **Cannot assess** for those.

#### The view and the acquisition mode

Check question 2 against the view classes described in [ontology_v2.md](ontology_v2.md#views). If the clip belongs
to another class, replace the prefilled value with that class; use `OTHER` for off-axis or non-standard clips. If
nothing in the clip can be classified reliably, mark the clip uninterpretable rather than choosing at random. Check
and correct the acquisition mode directly in question 3.

Still judge the prefilled findings on the clip: they are counted either way. A corrected view or mode gives the clip
no caption.

#### The caption

There is no caption to edit. After review, the caption is rebuilt from your answers: the supported findings, the
corrected values and the confirmed missing findings, one sentence each. **Original caption** shows the candidate and
is not read.

#### When a review is sent back

A review is returned for correction when:

- a required answer is empty, or **Review completed** is not selected;
- an incorrect finding has no reason or no correction, or a finding that cannot be assessed has no limitation;
- question 7 is "Yes" without a column, its value and "Visible in this input? Yes", or lists a finding that is
  already prefilled;
- a finding you added is not one this view carries, or is about a structure you unticked (the form lists every finding
  on every view);
- a finding is left supported, or given a corrected value, on a structure you unticked;
- question 8 is "Yes" without a column or statement type and its reason, or names a finding you left supported;
- the view or the acquisition mode is answered "No" but left as proposed, or changed without answering "No";
- a prefilled finding is unticked, or its proposed value changed;
- a finding ticked in question 6 without being prefilled is not marked supported, or is also added under question 7;
- a subcostal clip has no subtype, or a clip of another view has one;
- the clip is uninterpretable without a reason, or with "Other" and no comment.

The full rules are in [pilot_metrics_v2.md](pilot_metrics_v2.md#which-reviews-count).

## Worked examples: column review

The clips are synthetic, described in words.

<!-- BEGIN GENERATED: column_examples -->
### 1. Supported findings

- **Clip (synthetic):** Apical 4-chamber B-mode loop: a normal-sized LV, an RV that contracts normally, thin tricuspid leaflets.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_size` Normal; `rv_function` Normal; `tricuspid_appearance` Normal
- **Candidate caption:** "LV cavity is not dilated. RV systolic function appears normal. Tricuspid valve has a normal appearance."
- **What the reviewer changes:** nothing
- **Rebuilt caption:** "LV cavity is not dilated. RV systolic function appears normal. Tricuspid valve has a normal appearance."
- **Why:** Each prefilled finding is visible in this clip at the stated value, and nothing important is left out: confirm and submit.

### 2. Wrong severity

- **Clip (synthetic):** Apical 4-chamber B-mode loop: the LV contracts clearly less than normal, a moderate reduction by eye.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_function` Mildly reduced
- **Candidate caption:** "LV systolic function appears mildly reduced."
- **What the reviewer changes:**
  - LV function, Confirm this finding: **Incorrect / should not be here**
  - LV function, Why should it not be here? **Wrong severity**
  - LV function, Correction: **Moderately reduced**
- **Rebuilt caption:** "LV systolic function appears moderately reduced."
- **Why:** The finding is right but its severity is not: "Wrong severity", with the value the clip shows. The rebuilt caption uses the correction; the original value stays recorded.

### 3. Contradicted finding

- **Clip (synthetic):** Parasternal long-axis B-mode loop: the posterior pericardium is clearly seen, with no fluid around the heart.
- **Proposed:** `PLAX`, B-mode
- **Prefilled findings:** `pericardial_effusion` Present
- **Candidate caption:** "Pericardial effusion is present."
- **What the reviewer changes:**
  - Pericardial effusion, Confirm this finding: **Incorrect / should not be here**
  - Pericardial effusion, Why should it not be here? **Contradicted by visible evidence**
  - Pericardial effusion, Correction: **Not demonstrated**
- **Rebuilt caption:** "No pericardial effusion."
- **Why:** The clip shows the pericardium well and no effusion: the prefilled finding is contradicted. Because this clip can show an effusion, the correction can say none is demonstrated; if it could not, remove the claim instead.

### 4. Missing finding

- **Clip (synthetic):** Apical 4-chamber color Doppler loop: a mitral regurgitant jet and a clearly visible tricuspid regurgitant jet.
- **Proposed:** `A4C`, Color Doppler
- **Prefilled findings:** `mitral_regurgitant_jet` Jet demonstrated
- **Candidate caption:** "Mitral regurgitant jet is visible on color Doppler."
- **What the reviewer changes:**
  - Important visible finding missing from the candidate? **Yes**
  - Select missing columns and values: **Tricuspid regurgitant jet**
  - Tricuspid regurgitant jet, Missing visible value: **Jet demonstrated**
  - Tricuspid regurgitant jet, Visible in this input? **Yes**
- **Rebuilt caption:** "Mitral regurgitant jet is visible on color Doppler. Tricuspid regurgitant jet is visible on color Doppler."
- **Why:** What is prefilled is supported; the tricuspid jet is visible, relevant and not prefilled, so it is added as missing. Neither jet gets a grade: a color loop shows whether a jet is demonstrated, not an integrated severity.

### 5. Absence that cannot be judged

- **Clip (synthetic):** Apical 4-chamber color Doppler loop with the color box over the tricuspid valve; the mitral valve is outside the color box.
- **Proposed:** `A4C`, Color Doppler
- **Prefilled findings:** `mitral_regurgitant_jet` No jet demonstrated in this acquisition
- **Candidate caption:** "No mitral regurgitant jet is seen on color Doppler."
- **What the reviewer changes:**
  - Mitral regurgitant jet, Confirm this finding: **Cannot assess**
  - Mitral regurgitant jet, Why can it not be assessed? **Target outside the displayed input**
- **Rebuilt caption:** none: no finding was confirmed
- **Why:** A jet can be called absent only where the color covers the valve. Here it does not, so the absence cannot be assessed: that is neither an error nor a negative finding.

### 6. Segment that cannot be assessed

- **Clip (synthetic):** Apical 2-chamber B-mode loop: a normal-sized LV; the endocardium of the mid inferior wall is not seen (dropout).
- **Proposed:** `A2C`, B-mode
- **Prefilled findings:** `lv_size` Normal; `wall_mid_inferior` Normal / hyperkinetic (report score 1)
- **Candidate caption:** "LV cavity is not dilated. Mid inferior segment shows normal or hyperkinetic motion."
- **What the reviewer changes:**
  - Mid inferior wall motion, Confirm this finding: **Cannot assess**
  - Mid inferior wall motion, Why can it not be assessed? **Poor image quality**
- **Rebuilt caption:** "LV cavity is not dilated."
- **Why:** A segment whose endocardium is not seen cannot be graded. "Cannot assess" with its limitation keeps the claim out of the caption without counting it as an error; the other finding stays.

### 7. PSAX level and a missing segment

- **Clip (synthetic):** Parasternal short-axis B-mode loop at the papillary muscles: the LV contracts normally, except the mid anterior wall, which moves clearly less.
- **Proposed:** `PSAX`, B-mode
- **Prefilled findings:** `lv_function` Normal
- **Candidate caption:** "LV systolic function appears normal."
- **What the reviewer changes:**
  - View (prefilled - correct if needed): **PSAX_MID (papillary level)**
  - Important visible finding missing from the candidate? **Yes**
  - Select missing columns and values: **Mid anterior wall motion**
  - Mid anterior wall motion, Missing visible value: **Hypokinetic**
  - Mid anterior wall motion, Visible in this input? **Yes**
- **Rebuilt caption:** "LV systolic function appears normal. Mid anterior segment shows hypokinesis."
- **Why:** A PSAX clip opens with no view, because the view classifier does not give its level: the papillary muscles show the mid level, so choose PSAX_MID. Its mid wall segments can be claimed only at that level, so the hypokinetic segment is added as missing once the view is set.

### 8. Wrong view

- **Clip (synthetic):** B-mode loop labelled apical 4-chamber, but the LV outflow tract and the aortic valve are open in the image: it is an apical 5-chamber view. Both ventricles are normal in size.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_size` Normal; `rv_size` Normal
- **Candidate caption:** "LV cavity is not dilated. RV cavity is not dilated."
- **What the reviewer changes:**
  - View (prefilled - correct if needed): **A5C**
- **Rebuilt caption:** none: the view or acquisition mode was corrected
- **Why:** Replace the prefilled view with the view shown by the clip. Still judge the prefilled findings on the clip: they are counted. The clip gets no caption under the proposed view.

### 9. Uninterpretable clip

- **Clip (synthetic):** Apical B-mode loop with heavy artifacts throughout: no wall or valve can be made out.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_function` Normal
- **Candidate caption:** "LV systolic function appears normal."
- **What the reviewer changes:**
  - Is this clip uninterpretable? **Yes, nothing in it can be judged**
  - Why? **Poor image quality or artifacts throughout**
- **Rebuilt caption:** none: the clip is uninterpretable
- **Why:** Nothing in this clip can be judged: answer "Yes", give the reason and submit; the other answers are not read. When only some findings cannot be judged, the clip is interpretable: use "Cannot assess" for those.
<!-- END GENERATED: column_examples -->

## View and type review (`OTHER` clips)

Each item is a clip the view classifier labels `OTHER` or does not classify. Say what it is and, for an echo cine
loop, which view it shows and its acquisition mode. Your answers measure what `OTHER` holds, and an echo cine loop of
a view the column review carries, in B-mode or color Doppler, goes back to the column review under that view.

<!-- BEGIN GENERATED: view_type_form -->
| Question | Id | Required | Answers |
|---|---|---|---|
| What is this item? | `input_type` | yes | `cine_loop` Echo cine loop (moving echo images); `analysis_screen` Analysis or quantification screen (strain, 3D, EF tool); `no_echo_image` No echo image |
| Which view does it show? | `view` | yes, asked when `input_type` is `cine_loop` | `A2C` A2C: Apical 2-chamber view; `A2C_LV` A2C_LV: Apical 2-chamber view, LV-focused; `A2C_ZOOM` A2C_ZOOM: Apical 2-chamber view, zoomed; `A3C` A3C: Apical 3-chamber view; `A3C_LV` A3C_LV: Apical 3-chamber view, LV-focused; `A3C_ZOOM` A3C_ZOOM: Apical 3-chamber view, zoomed; `A4C` A4C: Apical 4-chamber view; `A4C_LV` A4C_LV: Apical 4-chamber view, LV-focused; `A4C_ZOOM` A4C_ZOOM: Apical 4-chamber view, zoomed; `A5C` A5C: Apical 5-chamber view; `A5C_ZOOM` A5C_ZOOM: Apical 5-chamber view, zoomed; `PLAX` PLAX: Parasternal long-axis view; `PLAX_DEEP` PLAX_DEEP: Parasternal long-axis view, deep; `PLAX_ZOOM` PLAX_ZOOM: Parasternal long-axis view, zoomed; `PLAX_ZOOM_AV` PLAX_ZOOM_AV (zoomed on the aortic valve); `PSAX` PSAX (cannot tell the level); `PSAX_BASAL` PSAX_BASAL (mitral level); `PSAX_MID` PSAX_MID (papillary level); `PSAX_APICAL` PSAX_APICAL; `PSAX_AV` PSAX_AV: Parasternal short-axis view at the aortic valve; `RVINF` RVINF: Right ventricular inflow view; `SUBCOSTAL` SUBCOSTAL: Subcostal view; `SUPRASTERNAL` SUPRASTERNAL: Suprasternal view; `OTHER` OTHER: off-axis or non-standard echo view; `cannot_tell` Cannot tell the view |
| Acquisition mode | `modality` | yes, asked when `input_type` is `cine_loop` | `bmode` B-mode; `color_flow` Color Doppler; `spectral` Spectral Doppler; `m_mode` M-mode; `mixed` Mixed / analysis acquisition; `unresolved` Unresolved |
| Is identifying text visible (a name, a record number, a date of birth)? | `identifier_visible` | yes | `no` No; `yes` Yes: the item must be deleted |
| Comments (no patient details) | `comments` | no | free text |
<!-- END GENERATED: view_type_form -->

- **What is this item?** "Echo cine loop" for a moving echo image in any acquisition mode: only then are the view and
  the acquisition mode asked. For any other answer they are not asked and are recorded as not applicable.
- **View.** `OTHER` is for an off-axis or non-standard echo view; "Cannot tell the view" when you cannot decide.
- **Identifying text.** Answer "Yes" if a name, a record number or a date of birth is readable anywhere in the item.
  It is deleted.

## Worked examples: view and type review

<!-- BEGIN GENERATED: view_type_examples -->
### 1. Inferior vena cava labelled OTHER

- **Item (synthetic):** Subcostal B-mode loop of the inferior vena cava entering the right atrium.
- **Answers:** What is this item? **Echo cine loop (moving echo images)**; Which view does it show? **SUBCOSTAL: Subcostal view**; Acquisition mode: **B-mode**; Is identifying text visible (a name, a record number, a date of birth)? **No**
- **Next:** back to the column review as `SUBCOSTAL`, B-mode
- **Why:** A moving echo image of a view the column review routes: it goes back to the column review under that view.

### 2. Off-axis view

- **Item (synthetic):** B-mode loop between the apical 4- and 5-chamber planes, foreshortened, in no standard view.
- **Answers:** What is this item? **Echo cine loop (moving echo images)**; Which view does it show? **OTHER: off-axis or non-standard echo view**; Acquisition mode: **B-mode**; Is identifying text visible (a name, a record number, a date of birth)? **No**
- **Next:** stays out of the column review
- **Why:** OTHER is for off-axis or non-standard echo views; "Cannot tell the view" is for when you cannot decide.

### 3. Analysis screen

- **Item (synthetic):** Strain analysis screen: an apical 4-chamber image with colored segments and a bull's-eye plot.
- **Answers:** What is this item? **Analysis or quantification screen (strain, 3D, EF tool)**; Is identifying text visible (a name, a record number, a date of birth)? **No**; view and acquisition mode not asked, recorded as not applicable
- **Next:** stays out of the column review
- **Why:** An analysis screen is not a clip a caption should describe: its view and mode are not asked, and it does not go back to the column review.

### 4. Identifying text visible

- **Item (synthetic):** Apical 4-chamber B-mode loop in which a name is still readable in the top banner.
- **Answers:** What is this item? **Echo cine loop (moving echo images)**; Which view does it show? **A4C: Apical 4-chamber view**; Acquisition mode: **B-mode**; Is identifying text visible (a name, a record number, a date of birth)? **Yes: the item must be deleted**
- **Next:** deleted from Labelbox: identifying text
- **Why:** Answer "Yes" without copying the text anywhere. The item is deleted from Labelbox and the de-identification step is fixed.
<!-- END GENERATED: view_type_examples -->

## Privacy

Comments must not contain patient names, identifiers, dates or report text.

If you see identifying text in the image (a name, a record number, a date of birth): in the column review, do not
review the clip, skip it in Labelbox and tell the study team its Labelbox id, without copying the text; in the view
and type review, answer "Yes" to the identifying-text question. Either way the clip is removed from Labelbox and from
the caption dataset.
