# RADAR caption review: instructions for reviewers (v1)

Version: `radar-caption-ontology-v1`. Its medical approval is recorded in [approval_v1.md](approval_v1.md); no
clinical data may be uploaded for review under these instructions until that record is `approved`.
[Version française](labeling_instructions_v1.fr.md). Both versions use the same question and answer ids.

## What you review

Each item in Labelbox is one echo clip with a **candidate caption**: one or two short French sentences that state
the clip's view, its acquisition mode and a few findings. Candidate captions are drafted automatically from the
study's report and from what the clip's view and acquisition can show. They are unvalidated: your review decides
what this clip actually supports.

Next to the clip you also see the structures its view usually shows (the "expected structures"). They are a prior to
help you, not something the caption claims.

The terms used below:

- **Statement:** one thing the caption says, besides the view: the acquisition mode, a structure being visible, or a
  finding with its value ("Insuffisance mitrale : modérée", "Insuffisance mitrale : absente").
- **Supported:** this clip shows it, at the stated value or severity.
- **Unconfirmed:** this clip does not let you decide (see [Uncertainty](#uncertainty)).

The ontology behind the form, with every view, structure and finding, is in [ontology_v1.md](ontology_v1.md).

## How to work through a clip

1. Watch the whole loop, more than once if needed.
2. Decide whether the clip can be interpreted at all (question 1).
3. Check the view (question 2).
4. Go through the caption statement by statement: supported, unsupported (question 5) or unconfirmed (question 6).
   Then answer question 3, which summarizes them.
5. Look for visible, relevant information the caption leaves out (question 4).
6. Edit the corrected caption (question 7), and add a comment only if needed.

## The form

<!-- BEGIN GENERATED: review_form -->
| # | Question | Id | Required | Answers |
|---|---|---|---|---|
| 1 | Is this clip uninterpretable? | `uninterpretable_clip` | yes | `no` No, the clip can be interpreted; `yes` Yes, it is uninterpretable → Reason: Image quality, Crop or zoom, Modality, Other |
| 2 | Is the stated view correct? | `view_correct` | yes | `yes` Yes; `no` No → Correct view (one of the view classes); `unsure` Unsure; `not_applicable` Not applicable: uninterpretable clip |
| 3 | Apart from the view, is everything the caption states supported by this clip? | `caption_supported` | yes | `yes` Yes, every statement is supported; `no` No, at least one statement is not supported or is wrong; `unsure` Unsure: nothing is wrong, but at least one statement cannot be confirmed; `not_applicable` Not applicable: uninterpretable clip |
| 4 | Does the caption leave out visible, clinically relevant information? | `missing_information` | yes | `no` No; `yes` Yes → Missing concepts (the statements below) + Other missing information (not in the ontology) (free text); `not_applicable` Not applicable: uninterpretable clip |
| 5 | Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? | `unsupported_information` | yes | `no` No; `yes` Yes → Unsupported or wrong statements (the statements below), each: Not shown or contradicted by the clip (hallucinated) / Present, but the value or severity is wrong; `not_applicable` Not applicable: uninterpretable clip |
| 6 | Statements that cannot be confirmed from this clip (leave empty if none) | `unconfirmed_statements` | no | the statements below, each: Reason (Visible, but I am not sure / Image quality / Crop or zoom / Modality / Other) |
| 7 | The caption as it should read (preloaded with the candidate caption) | `corrected_caption` | yes | free text |
| 8 | Comments (no patient details) | `comments` | no | free text |
<!-- END GENERATED: review_form -->

The statements you can flag or list as missing:

<!-- BEGIN GENERATED: statements -->
| Statement | Label | Kind |
|---|---|---|
| `acquisition` | Acquisition mode | acquisition mode |
| `lv_visible` | Left ventricle visible | visibility |
| `rv_visible` | Right ventricle visible | visibility |
| `la_visible` | Left atrium visible | visibility |
| `ra_visible` | Right atrium visible | visibility |
| `mitral_valve_visible` | Mitral valve visible | visibility |
| `aortic_valve_visible` | Aortic valve visible | visibility |
| `tricuspid_valve_visible` | Tricuspid valve visible | visibility |
| `pulmonic_valve_visible` | Pulmonic valve visible | visibility |
| `aortic_root_visible` | Aortic root visible | visibility |
| `interatrial_septum_visible` | Interatrial septum visible | visibility |
| `pericardium_visible` | Pericardium visible | visibility |
| `ivc_visible` | Inferior vena cava visible | visibility |
| `lv_systolic_function` | Left ventricular systolic function | finding |
| `lv_ejection_fraction` | LV ejection fraction (visual estimate) | finding |
| `lv_size` | Left ventricular size | finding |
| `lv_wall_thickness` | Left ventricular hypertrophy | finding |
| `lv_regional_wall_motion` | LV regional wall motion (worst segment) | finding |
| `rv_systolic_function` | Right ventricular systolic function | finding |
| `rv_size` | Right ventricular size | finding |
| `la_size` | Left atrial size | finding |
| `ra_size` | Right atrial size | finding |
| `atrial_septal_hypertrophy` | Lipomatous atrial septal hypertrophy | finding |
| `mitral_regurgitation` | Mitral regurgitation | finding |
| `mitral_annular_calcification` | Mitral annular calcification | finding |
| `mitral_valve_device` | Mitral valve device or prosthesis | finding |
| `aortic_regurgitation` | Aortic regurgitation | finding |
| `bicuspid_aortic_valve` | Bicuspid aortic valve | finding |
| `aortic_valve_prosthesis` | Aortic valve prosthesis | finding |
| `aortic_root_dilation` | Aortic root dilation | finding |
| `tricuspid_regurgitation` | Tricuspid regurgitation | finding |
| `pulmonic_regurgitation` | Pulmonic regurgitation | finding |
| `ivc_dilation` | Inferior vena cava dilation | finding |
| `ivc_collapse` | IVC inspiratory collapse | finding |
| `pericardial_effusion` | Pericardial effusion | finding |
| `pacing_lead` | Intracardiac pacing lead | finding |
| `mechanical_circulatory_support` | Mechanical circulatory support | finding |
<!-- END GENERATED: statements -->

## Rules

### Judge this clip only

Judge what this clip shows, not what you know about the patient: the report, the other clips of the study and your
own measurements do not count. A finding that is real for the patient but not visible in this clip is not supported
here.

### Supported, not supported, wrong value

- **Supported:** you can see it in this clip, at the stated value or severity, as you would grade it visually.
- **Not supported (hallucinated):** the clip does not show it, or contradicts it. Flag it under question 5 with
  "not shown or contradicted".
- **Wrong value:** the finding is there but its value or severity is wrong (for example severe instead of moderate).
  Flag it under question 5 with "wrong value or severity", and give the right value in the corrected caption.
- The **acquisition mode** is a statement too: if the caption says color Doppler and the clip is B-mode, flag
  "acquisition mode" as not supported.

### Uncertainty

When you cannot decide whether a statement is supported, do not guess. List it under question 6 with a reason:

- **uncertain:** the finding is visible but you cannot judge it with confidence (for example an eccentric jet whose
  grade you cannot tell);
- **image quality, crop or zoom, modality, other:** something about the clip prevents the judgment.

If nothing in the caption is wrong but at least one statement is unconfirmed, answer question 3 "unsure". A statement
is either unsupported or unconfirmed, never both. For the view, answer "unsure" when you cannot tell the class.

### Absent findings

A caption may state that a finding is absent ("Insuffisance mitrale : absente"). Judge it like any other statement:
it is supported only if this clip could show the finding (right view, right modality, the structure in the sector)
and it is not there. If the clip could not show it, the statement is unconfirmed, not supported.

A finding that is absent and not mentioned in the caption is not missing information.

### Structures that are not visible

The expected structures are a prior, not a claim. An expected structure that is not visible in this clip is neither
missing information nor an error, unless the caption states it is visible: then that statement is not supported.

If a structure is only partly visible, it counts as visible when you can identify it with confidence; otherwise list
the statement as unconfirmed (crop or zoom).

### Missing information

Answer "yes" to question 4 only for information that is visible in this clip and clinically relevant, and that the
caption leaves out. Tick the matching statements; use the free-text field for anything the list does not have.
Do not list measurements (TAPSE, RVSP, PISA…), what only other clips show, or absent findings.

### Uninterpretable clips

Answer "yes" to question 1 only when nothing in the caption can be judged: severe artifacts, wrong modality for the
whole caption, a crop that leaves nothing identifiable. Give the reason, answer "not applicable" to questions 2 to 5,
and leave the corrected caption unchanged (it is ignored).

If only part of the caption cannot be judged, the clip is interpretable: use the unconfirmed statements. An
interpretable clip never answers "not applicable": when you cannot decide, answer "unsure" where the question
offers it, and list the statements you cannot confirm under question 6.

### The view

Answer question 2 against the view classes described in [ontology_v1.md](ontology_v1.md#views). If the clip belongs
to another class, answer "no" and choose that class, never the stated one; use `OTHER` for off-axis or non-standard
clips. If you cannot tell the class, answer "unsure": do not pick a class at random, and do not use `OTHER` for it.

The view is judged only here. Questions 3 and 5 are about the other statements: a wrong or uncertain view alone gives
"caption supported: yes" when every statement is supported, and is never listed as an unsupported statement. Judge
the statements as they are written either way.

### The corrected caption

The field is preloaded with the candidate caption. Rewrite it so it states only what this clip supports:

- remove unsupported and unconfirmed statements, except the acquisition mode, which every caption keeps (see below);
- fix wrong values;
- add missing information when the statements list has it;
- give the right view if question 2 was "no";
- keep the same form: the view, the acquisition mode, then the findings, in French;
- never add identifiers, dates, measurements or report text.

When you cannot tell the view or the acquisition mode, the corrected caption keeps its form and says so:

- question 2 "unsure": write "Coupe indéterminée" in place of the view ("Coupe indéterminée, Doppler couleur. …").
  The classifier's class stays recorded with the clip;
- acquisition mode unconfirmed: write "acquisition indéterminée" in place of the mode. If the stated mode is wrong,
  write the mode the clip shows, or "acquisition indéterminée" if you cannot tell it.

The statements the clip supports stay. If the caption is right as it is, leave it unchanged.

### When a review is sent back

A review is returned for correction when a required answer is empty, such as a blank corrected caption, or when its
answers contradict each other: for example "caption supported: yes" with an unsupported or unconfirmed statement, "no"
without an unsupported statement, "missing information: yes" without anything listed, "not applicable" on an
interpretable clip, a wrong view corrected to the view the caption already states, or a flag on a statement the caption
does not make. The full list is in [pilot_metrics_v1.md](pilot_metrics_v1.md#which-reviews-count).

### Privacy

Comments and corrected captions must not contain patient names, identifiers, dates or report text.

## Worked examples

The clips are synthetic, described in words.

<!-- BEGIN GENERATED: examples -->
### 1. Supported caption

- **Clip (synthetic):** Apical 4-chamber color Doppler loop; a moderate mitral regurgitation jet, well aligned.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : modérée."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **Yes, every statement is supported**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **No**
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** unchanged
- **Why:** Every statement is visible in this clip, and nothing relevant is left out.

### 2. Wrong severity

- **Clip (synthetic):** Apical 4-chamber color Doppler loop; the mitral regurgitation jet looks moderate.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : sévère."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **No, at least one statement is not supported or is wrong**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **Yes** (Mitral regurgitation: Present, but the value or severity is wrong)
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : modérée."
- **Why:** The regurgitation is present but its severity is wrong: "wrong value or severity", and the corrected caption gives the severity the clip shows.

### 3. Hallucinated finding

- **Clip (synthetic):** Parasternal long-axis B-mode loop; the pericardium is well seen, with no effusion.
- **Classifier view and acquisition:** `PLAX`, B-mode
- **Candidate caption:** "Coupe parasternale grand axe (PLAX), mode B. Épanchement péricardique : de moyenne abondance."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **No, at least one statement is not supported or is wrong**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **Yes** (Pericardial effusion: Not shown or contradicted by the clip (hallucinated))
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** "Coupe parasternale grand axe (PLAX), mode B. Épanchement péricardique : absent."
- **Why:** The clip shows the pericardium and no effusion: the statement is not supported (hallucinated). Because the clip can show an effusion, the corrected caption may state that there is none.

### 4. Missing information

- **Clip (synthetic):** Apical 4-chamber color Doppler loop; mild mitral regurgitation and a clearly visible moderate tricuspid regurgitation jet.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **Yes, every statement is supported**
  - Does the caption leave out visible, clinically relevant information? **Yes** (Tricuspid regurgitation)
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **No**
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère. Insuffisance tricuspide : modérée."
- **Why:** What the caption states is supported; the visible, relevant tricuspid regurgitation is missing. The two questions are independent.

### 5. Absence that cannot be confirmed

- **Clip (synthetic):** Apical 4-chamber color Doppler loop; the color box sits on the tricuspid valve and does not cover the mitral valve.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : absente."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **Unsure: nothing is wrong, but at least one statement cannot be confirmed**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **No**
  - Statements that cannot be confirmed from this clip (leave empty if none): **Mitral regurgitation: Crop or zoom**

- **Corrected caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur."
- **Why:** An absence is judged like any other finding: it is supported only if the clip could show the regurgitation. Here the color box misses the mitral valve, so the statement cannot be confirmed (crop) and leaves the corrected caption.

### 6. Expected structure that is not visible

- **Clip (synthetic):** Apical 4-chamber B-mode loop zoomed on the mitral valve; the tricuspid valve is outside the sector.
- **Classifier view and acquisition:** `A4C_ZOOM`, B-mode
- **Candidate caption:** "Coupe apicale 4 cavités zoomée (A4C_ZOOM), mode B. Valve tricuspide visible."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **No, at least one statement is not supported or is wrong**
  - Does the caption leave out visible, clinically relevant information? **Yes** (Mitral valve visible)
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **Yes** (Tricuspid valve visible: Not shown or contradicted by the clip (hallucinated))
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** "Coupe apicale 4 cavités zoomée (A4C_ZOOM), mode B. Valve mitrale visible."
- **Why:** The expected structures of a view are only a prior: an expected structure that is not visible is neither missing nor wrong, unless the caption says it is visible. Here it does, so the statement is not supported.

### 7. Uncertain grade

- **Clip (synthetic):** Apical 4-chamber color Doppler loop; an eccentric tricuspid jet, partly out of plane.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance tricuspide : légère."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **Unsure: nothing is wrong, but at least one statement cannot be confirmed**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **No**
  - Statements that cannot be confirmed from this clip (leave empty if none): **Tricuspid regurgitation: Visible, but I am not sure**

- **Corrected caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur."
- **Why:** The regurgitation is visible but its grade cannot be judged from this clip: "uncertain", never a guessed yes or no. A statement that cannot be confirmed leaves the corrected caption.

### 8. Wrong acquisition mode

- **Clip (synthetic):** Apical 4-chamber B-mode loop, without color Doppler.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **No, at least one statement is not supported or is wrong**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **Yes** (Acquisition mode: Not shown or contradicted by the clip (hallucinated))
  - Statements that cannot be confirmed from this clip (leave empty if none): **Mitral regurgitation: Modality**

- **Corrected caption:** "Coupe apicale 4 cavités (A4C), mode B."
- **Why:** The acquisition mode is a statement too, and here it is wrong. Mitral regurgitation cannot be judged without color Doppler, so it cannot be confirmed (modality). A clip can have both unsupported and unconfirmed statements.

### 9. Acquisition mode that cannot be confirmed

- **Clip (synthetic):** Apical 4-chamber loop with cropped edges: the color scale and the outline of any color box are outside the image, and the few colored pixels in the left ventricle could be color Doppler or noise; the left ventricle is well seen.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Ventricule gauche visible."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Yes**
  - Apart from the view, is everything the caption states supported by this clip? **Unsure: nothing is wrong, but at least one statement cannot be confirmed**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **No**
  - Statements that cannot be confirmed from this clip (leave empty if none): **Acquisition mode: Crop or zoom**

- **Corrected caption:** "Coupe apicale 4 cavités (A4C), acquisition indéterminée. Ventricule gauche visible."
- **Why:** The acquisition mode is a statement, and here it cannot be confirmed (crop): it is listed under question 6 and question 3 is "unsure". The corrected caption keeps its form: instead of dropping the mode, it writes "acquisition indéterminée" (unknown acquisition). The left ventricle is visible whatever the mode, so its statement stays.

### 10. Wrong view

- **Clip (synthetic):** Apical 5-chamber color Doppler loop (LV outflow tract and aortic valve visible); mild mitral regurgitation.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **No** (A5C: Apical 5-chamber view)
  - Apart from the view, is everything the caption states supported by this clip? **Yes, every statement is supported**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **No**
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** "Coupe apicale 5 cavités (A5C), Doppler couleur. Insuffisance mitrale : légère."
- **Why:** The view is judged on its own: "no" and the right class. The statements are judged as written; the corrected caption gives the right view.

### 11. Uncertain view

- **Clip (synthetic):** Apical color Doppler loop between a 4- and a 5-chamber view: the LV outflow tract appears in some beats only; mild mitral regurgitation.
- **Classifier view and acquisition:** `A4C`, color Doppler
- **Candidate caption:** "Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère."
- **Answers:**

  - Is this clip uninterpretable? **No, the clip can be interpreted**
  - Is the stated view correct? **Unsure**
  - Apart from the view, is everything the caption states supported by this clip? **Yes, every statement is supported**
  - Does the caption leave out visible, clinically relevant information? **No**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **No**
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** "Coupe indéterminée, Doppler couleur. Insuffisance mitrale : légère."
- **Why:** When the view class cannot be decided, answer "unsure": do not pick a class at random, and do not use OTHER, which is for off-axis or non-standard clips. The statements are judged as written. The corrected caption replaces the view with "Coupe indéterminée" (undetermined view); the classifier's class stays recorded with the clip.

### 12. Uninterpretable clip

- **Clip (synthetic):** Subcostal B-mode loop; major artifacts, no identifiable structure.
- **Classifier view and acquisition:** `SUBCOSTAL`, B-mode
- **Candidate caption:** "Coupe sous-costale (SUBCOSTAL), mode B. Veine cave inférieure visible."
- **Answers:**

  - Is this clip uninterpretable? **Yes, it is uninterpretable** (Image quality)
  - Is the stated view correct? **Not applicable: uninterpretable clip**
  - Apart from the view, is everything the caption states supported by this clip? **Not applicable: uninterpretable clip**
  - Does the caption leave out visible, clinically relevant information? **Not applicable: uninterpretable clip**
  - Apart from the view, does the caption state anything this clip does not support (unsupported or hallucinated information)? **Not applicable: uninterpretable clip**
  - Statements that cannot be confirmed from this clip (leave empty if none): **none**

- **Corrected caption:** unchanged
- **Why:** Nothing can be judged: "yes", the reason, then "not applicable" to the four other questions. The corrected caption stays unchanged; it is ignored. If only part of the caption cannot be judged, the clip is interpretable: use the unconfirmed statements.
<!-- END GENERATED: examples -->
