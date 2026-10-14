# RADAR caption review: instructions for reviewers (v1)

Version: `radar-caption-ontology-v1`, with the [column review v2](column_review_v2.md) and the
[view and type review](view_type_review_v1.md). Their medical approval is recorded in [approval_v1.md](approval_v1.md);
no clinical data may be uploaded for review until that record is `approved`.
[Version française](labeling_instructions_v1.fr.md). Both versions describe the same forms, whose labels are in
English, as in Labelbox.

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
and the whole loop (see [Model input and review display](ontology_v1.md#model-input-and-review-display)).

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
3. Check the view and the acquisition mode (questions 2 to 5).
4. Check the targets the clip actually covers and, for a PSAX or subcostal clip, its level (questions 6 and 7).
5. Go through the prefilled findings one by one: supported, incorrect or cannot assess (question 8).
6. Look for an important visible finding the candidate leaves out (question 9), and for a statement of the caption
   outside its findings (question 10).
7. Add a comment only if needed, select **Review completed** (question 13) and submit.

Every answer except **Review completed** is prefilled: the clip interpretable, the view and mode correct, every
finding supported, nothing missing or extra. Change what this clip does not support; an answer you leave unchanged is
your confirmation.

### The form

<!-- BEGIN GENERATED: column_form -->
| # | Question | Id | Required | Answers |
|---|---|---|---|---|
| 1 | Is this clip uninterpretable? | `uninterpretable` | yes | `no` No, it can be interpreted; `yes` Yes, nothing in it can be judged → Why? (a reason below) |
| 2 | Proposed view | `view` | yes | one of the 20 view classes |
| 3 | Acquisition mode | `modality` | yes | `bmode` B-mode; `color_flow` Color Doppler; `spectral` Spectral Doppler; `m_mode` M-mode; `mixed` Mixed / analysis acquisition; `unresolved` Unresolved |
| 4 | View correct? | `view_ok` | yes | `yes` Yes; `no` No — reroute; `cannot_assess` Cannot assess |
| 5 | Acquisition mode correct? | `modality_ok` | yes | `yes` Yes; `no` No — reroute; `cannot_assess` Cannot assess |
| 6 | Confirm targets actually covered and assessable | `visible_targets` | no | `lv` LV; `rv` RV; `la` LA; `ra` RA; `mitral` Mitral; `tricuspid` Tricuspid; `pericardium` Pericardium; `aortic` Aortic; `aorta_root` Aorta root; `aorta_ascending` Aorta ascending; `pulmonic` Pulmonic; `ivc` IVC; `aorta_arch` Aorta arch |
| 7 | PSAX slice / SUBCOSTAL subtype (when relevant) | `psax_level` | yes | `not_applicable` Not applicable; `unresolved` Unresolved; `mid` PSAX mid / papillary level; `basal` PSAX basal; `apical` PSAX apical; `mitral` PSAX mitral leaflet level; `ivc` IVC focused; `four_chamber` Subcostal four chamber |
| 8 | Prefilled findings — confirm or correct selected columns | `review_columns` | no | the prefilled columns, each: Proposed value (original retained in context) (one of the column's values) + Confirm this finding (`yes` Supported in this input; `no` Incorrect / should not be here → Why should it not be here? (an error reason) + Correction (`remove` Remove this claim, or one of the column's values); `cannot_assess` Cannot assess → Why can it not be assessed? (an assessment limitation)) |
| 9 | Important visible finding missing from the candidate? | `missing_information` | yes | `no` No; `yes` Yes → Select missing columns and values (any column the view carries, each: Missing visible value (one of the column's values) + Visible in this input? (`yes` Yes; `cannot_assess` Cannot assess)) |
| 10 | Additional statement that should not be here? | `unsupported_information` | yes | `no` No; `yes` Yes → Select offending columns / statements (any column or statement type, each: Why should this statement not be here? (an error reason); a statement type also needs Copy the offending statement (free text)) |
| 11 | Original caption (kept in immutable context; generated from proposals) | `candidate_caption` | no | free text |
| 12 | Comments (no patient details) | `comments` | no | free text |
| 13 | I reviewed this input, checked the proposed values and targets, and confirmed or corrected the suggestions. | `review_completed` | yes | `yes` Review completed |
<!-- END GENERATED: column_form -->

The finding columns, their values and the views that carry them are listed in
[column_review_v2.md](column_review_v2.md#clinical-columns-and-choices). The reasons:

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

Some findings state an absence: "No mitral regurgitant jet is demonstrated in this acquisition", "No pericardial
effusion is demonstrated in the imaged region". Judge them like any other finding: supported only if this clip could
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
integrated grade, an exact EF or any number makes an additional statement (question 10). Regional wall motion
describes only the segments this clip shows. Report wall-motion scores keep their ambiguity: score 1 gives "Normal /
hyperkinetic", score 3 "Akinetic / severely hypokinetic". Do not infer normal motion in unseen segments, a whole-study
EF, stenosis severity, filling pressures or pulmonary hypertension.

#### Targets and structures that are not visible

**Confirm targets actually covered and assessable** is prefilled with the targets the view usually shows. Untick a
target this clip does not cover well enough to assess; tick a target that a focused or zoomed clip does cover.

A finding about a structure that is not in the clip at all is **Incorrect**: "Claimed structure/finding not visible".
A structure only partly in the image, too cut off to judge, is **Cannot assess**: "Target outside the displayed
input".

#### PSAX level and subcostal subtype

For a PSAX clip, choose the slice level; for a subcostal clip, IVC or four-chamber. Wall segments are judged only at
the mid (papillary) level and mitral leaflets only at the mitral level, and subcostal IVC and chamber findings never
mix. For other views, leave "Not applicable".

#### Missing information

Answer "Yes" to question 9 only for an **important** finding that is visible in this clip, that the view carries, and
that no prefilled finding states: choose the column and its value, and confirm it is visible in this input. Do not add
normal values the report left out, absent findings, measurements, or what only other clips show. A prefilled finding
with a wrong value is corrected under question 8, not added as missing.

#### Additional statements

The candidate caption is built from its findings, so question 10 is rarely "Yes". Use it when the caption states
something outside its findings, such as a grade, an exact EF or a number: choose the column or the statement type and
the reason, and copy the statement when the form asks.

#### Uninterpretable clips

Answer "Yes" to question 1 only when nothing in the clip can be judged: artifacts or poor image quality throughout, no
identifiable cardiac structure, or not a cardiac ultrasound acquisition. Give the reason; with "Other", say why in
**Comments**. The other answers are then not read, and you need not change them.

If only some findings cannot be judged, the clip is interpretable: use **Cannot assess** for those.

#### The view and the acquisition mode

Answer question 4 against the view classes described in [ontology_v1.md](ontology_v1.md#views). If the clip belongs
to another class, answer "No — reroute" and choose that class in **Proposed view**; use `OTHER` for off-axis or
non-standard clips. If you cannot tell the class, answer "Cannot assess": do not pick a class at random, and do not
use `OTHER` for it. The acquisition mode works the same way (question 5 and **Acquisition mode**).

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
- question 9 is "Yes" without a column, its value and "Visible in this input? Yes", or lists a finding that is
  already prefilled;
- question 10 is "Yes" without a column or statement type and its reason, or names a finding you left supported;
- the view or the acquisition mode is answered "No" but left as proposed, or changed without answering "No";
- a prefilled finding is unticked, or its proposed value changed;
- the clip is uninterpretable without a reason, or with "Other" and no comment.

The full rules are in [pilot_metrics_v1.md](pilot_metrics_v1.md#which-reviews-count).

## Worked examples: column review

The clips are synthetic, described in words.

<!-- BEGIN GENERATED: column_examples -->
### 1. Supported findings

- **Clip (synthetic):** Apical 4-chamber B-mode loop: a normal-sized LV, an RV that contracts normally, thin tricuspid leaflets.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_size` Normal; `rv_function` Normal; `tricuspid_appearance` Normal
- **Candidate caption:** "The visible LV cavity is not dilated. RV systolic function appears normal in this input. The visible tricuspid valve has a normal appearance."
- **What the reviewer changes:** nothing
- **Rebuilt caption:** "The visible LV cavity is not dilated. RV systolic function appears normal in this input. The visible tricuspid valve has a normal appearance."
- **Why:** Each prefilled finding is visible in this clip at the stated value, and nothing important is left out: confirm and submit.

### 2. Wrong severity

- **Clip (synthetic):** Apical 4-chamber B-mode loop: the LV contracts clearly less than normal, a moderate reduction by eye.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_function` Mildly reduced
- **Candidate caption:** "LV systolic function appears mildly reduced in this input."
- **What the reviewer changes:**
  - LV function, Confirm this finding: **Incorrect / should not be here**
  - LV function, Why should it not be here? **Wrong severity**
  - LV function, Correction: **Moderately reduced**
- **Rebuilt caption:** "LV systolic function appears moderately reduced in this input."
- **Why:** The finding is right but its severity is not: "Wrong severity", with the value the clip shows. The rebuilt caption uses the correction; the original value stays recorded.

### 3. Contradicted finding

- **Clip (synthetic):** Parasternal long-axis B-mode loop: the posterior pericardium is clearly seen, with no fluid around the heart.
- **Proposed:** `PLAX`, B-mode
- **Prefilled findings:** `pericardial_effusion` Present
- **Candidate caption:** "Pericardial effusion is demonstrated in the imaged region."
- **What the reviewer changes:**
  - Pericardial effusion, Confirm this finding: **Incorrect / should not be here**
  - Pericardial effusion, Why should it not be here? **Contradicted by visible evidence**
  - Pericardial effusion, Correction: **Not demonstrated**
- **Rebuilt caption:** "No pericardial effusion is demonstrated in the imaged region."
- **Why:** The clip shows the pericardium well and no effusion: the prefilled finding is contradicted. Because this clip can show an effusion, the correction can say none is demonstrated; if it could not, remove the claim instead.

### 4. Missing finding

- **Clip (synthetic):** Apical 4-chamber color Doppler loop: a mitral regurgitant jet and a clearly visible tricuspid regurgitant jet.
- **Proposed:** `A4C`, Color Doppler
- **Prefilled findings:** `mitral_regurgitant_jet` Jet demonstrated
- **Candidate caption:** "A mitral regurgitant jet is demonstrated in this color acquisition."
- **What the reviewer changes:**
  - Important visible finding missing from the candidate? **Yes**
  - Select missing columns and values: **Tricuspid regurgitant jet**
  - Tricuspid regurgitant jet, Missing visible value: **Jet demonstrated**
  - Tricuspid regurgitant jet, Visible in this input? **Yes**
- **Rebuilt caption:** "A mitral regurgitant jet is demonstrated in this color acquisition. A tricuspid regurgitant jet is demonstrated in this color acquisition."
- **Why:** What is prefilled is supported; the tricuspid jet is visible, relevant and not prefilled, so it is added as missing. Neither jet gets a grade: a color loop shows whether a jet is demonstrated, not an integrated severity.

### 5. Absence that cannot be judged

- **Clip (synthetic):** Apical 4-chamber color Doppler loop with the color box over the tricuspid valve; the mitral valve is outside the color box.
- **Proposed:** `A4C`, Color Doppler
- **Prefilled findings:** `mitral_regurgitant_jet` No jet demonstrated in this acquisition
- **Candidate caption:** "No mitral regurgitant jet is demonstrated in this acquisition."
- **What the reviewer changes:**
  - Mitral regurgitant jet, Confirm this finding: **Cannot assess**
  - Mitral regurgitant jet, Why can it not be assessed? **Target outside the displayed input**
- **Rebuilt caption:** none: no finding was confirmed
- **Why:** A jet can be called absent only where the color covers the valve. Here it does not, so the absence cannot be assessed: that is neither an error nor a negative finding.

### 6. Segment that cannot be assessed

- **Clip (synthetic):** Apical 2-chamber B-mode loop: a normal-sized LV; the endocardium of the mid inferior wall is not seen (dropout).
- **Proposed:** `A2C`, B-mode
- **Prefilled findings:** `lv_size` Normal; `wall_mid_inferior` Normal / hyperkinetic (report score 1)
- **Candidate caption:** "The visible LV cavity is not dilated. The visible mid inferior segment shows normal or hyperkinetic motion."
- **What the reviewer changes:**
  - Mid inferior wall motion, Confirm this finding: **Cannot assess**
  - Mid inferior wall motion, Why can it not be assessed? **Poor image quality**
- **Rebuilt caption:** "The visible LV cavity is not dilated."
- **Why:** A segment whose endocardium is not seen cannot be graded. "Cannot assess" with its limitation keeps the claim out of the caption without counting it as an error; the other finding stays.

### 7. Wrong view

- **Clip (synthetic):** B-mode loop labelled apical 4-chamber, but the LV outflow tract and the aortic valve are open in the image: it is an apical 5-chamber view. Both ventricles are normal in size.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_size` Normal; `rv_size` Normal
- **Candidate caption:** "The visible LV cavity is not dilated. The visible RV cavity is not dilated."
- **What the reviewer changes:**
  - View correct? **No — reroute**
  - Proposed view: **A5C**
- **Rebuilt caption:** none: the view or acquisition mode was corrected
- **Why:** Answer "No" and choose the view the clip shows in "Proposed view". Still judge the prefilled findings on the clip: they are counted. The clip gets no caption under the proposed view.

### 8. Uninterpretable clip

- **Clip (synthetic):** Apical B-mode loop with heavy artifacts throughout: no wall or valve can be made out.
- **Proposed:** `A4C`, B-mode
- **Prefilled findings:** `lv_function` Normal
- **Candidate caption:** "LV systolic function appears normal in this input."
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
| Which view does it show? | `view` | yes, asked when `input_type` is `cine_loop` | `A2C` A2C: Apical 2-chamber view; `A2C_LV` A2C_LV: Apical 2-chamber view, LV-focused; `A2C_ZOOM` A2C_ZOOM: Apical 2-chamber view, zoomed; `A3C` A3C: Apical 3-chamber view; `A3C_LV` A3C_LV: Apical 3-chamber view, LV-focused; `A3C_ZOOM` A3C_ZOOM: Apical 3-chamber view, zoomed; `A4C` A4C: Apical 4-chamber view; `A4C_LV` A4C_LV: Apical 4-chamber view, LV-focused; `A4C_ZOOM` A4C_ZOOM: Apical 4-chamber view, zoomed; `A5C` A5C: Apical 5-chamber view; `A5C_ZOOM` A5C_ZOOM: Apical 5-chamber view, zoomed; `PLAX` PLAX: Parasternal long-axis view; `PLAX_DEEP` PLAX_DEEP: Parasternal long-axis view, deep; `PLAX_ZOOM` PLAX_ZOOM: Parasternal long-axis view, zoomed; `PSAX` PSAX: Parasternal short-axis view; `PSAX_AV` PSAX_AV: Parasternal short-axis view at the aortic valve; `RVINF` RVINF: Right ventricular inflow view; `SUBCOSTAL` SUBCOSTAL: Subcostal view; `SUPRASTERNAL` SUPRASTERNAL: Suprasternal view; `OTHER` OTHER: off-axis or non-standard echo view; `cannot_tell` Cannot tell the view |
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
