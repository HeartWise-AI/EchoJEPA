# RADAR caption pilot: metrics and go/no-go rules (v1)

**Need agreement on the thresholds below before any pilot clip is reviewed.** Until
[approval_v1.md](approval_v1.md) links to the agreement, the thresholds are proposals. Once agreed, this document is
frozen with the ontology, and the pilot outcome is reported under these rules whatever it is. A change made after
seeing pilot results needs a new version and is judged on new pilot studies, never on the clips already reviewed.

The pilot reviews every valid video of 2–3 training-split studies. It tests whether the caption-and-review workflow
is feasible under [`radar-caption-ontology-v1`](ontology_v1.md) with the [column review v2](column_review_v2.md). It
does not measure clinical accuracy or model performance, and its small size allows only coarse decisions. The
metrics follow the column review's scoring ([Scoring](column_review_v2.md#scoring), `score_review` in
`tools/labelbox/column_review.py`).

## What each review records

### Column review (clips of a classified view)

The metrics use only these answers and the candidate recorded at upload (view, acquisition mode and
prefilled columns).

| Item | Answers |
|---|---|
| **Uninterpretable clip?** (`uninterpretable`) | no; yes, with a reason |
| **View correct?** (`view_ok`) and **Acquisition mode correct?** (`modality_ok`) | yes; no, with the view or mode the clip shows in place of the proposal; cannot assess |
| Targets and level (`visible_targets`, `psax_level`) | the targets the clip covers; the PSAX level or subcostal subtype |
| **Each prefilled column** | supported; incorrect, with an error reason and a corrected value or "remove"; cannot assess, with a limitation |
| **Missing information?** (`missing_information`) | no; yes, with each missing column, its value and the confirmation that it is visible |
| **Additional statement?** (`unsupported_information`) | no; yes, with each column or statement type and its error reason |
| Comments (optional) | free text, no patient details |
| **Review completed** (`review_completed`) | required; never prefilled |

From Labelbox itself: the review status, a pseudonymous reviewer id and the label's timing (see M7). In Labelbox's
terms, filling this form is the labeling step: the pilot's reviewer is Labelbox's labeler, and a separate Labelbox
review step, if the project has one, is not part of the medical review.

A **column** is one of the candidate's prefilled columns, with its original value: the claims the candidate caption
makes, one sentence each. The view and the acquisition mode are judged by their own questions and are not columns.
Every answer except **Review completed** is prefilled, so an answer left unchanged counts as the reviewer's
confirmation; M7 and the second reader (below) show whether the forms were checked.

Columns are judged on the whole de-identified clip, which is also the whole model input: every frame resized to
224 × 224, never cropped (see [Model input and review display](ontology_v1.md#model-input-and-review-display)). The
rates below therefore measure captions against what the model sees, and an incorrect column is a caption error, not a
crop effect.

### View and type review (`OTHER` clips)

Each review records the item's type, its view, its acquisition mode, whether identifying text is visible, and optional
comments ([view_type_review_v1.md](view_type_review_v1.md)). These items have no caption and enter no metric below
except the counts listed under [Metrics](#metrics).

## Display readiness and training eligibility

Before pilot review, check on a synthetic clip that the uploaded copy is the whole de-identified clip, every frame at
its original resolution, with no overlay, and that the embedding is computed from the same copy. Clips skipped because
identifying text is visible, and view and type items answered "identifying text visible: yes", are counted separately;
they are deleted from Labelbox and never enter the caption dataset.

A workflow Go does not validate all generated captions. Only the rebuilt caption of a valid, submitted review whose
view and acquisition mode were confirmed, and whose uploaded copy is verified as the encoder's input
(`ready_for_validation`), may enter the verified local contrastive dataset; a column that cannot be assessed is left
out of it. Preserve candidates separately for pilot scoring. Study-report labels remain a separate weak-supervision
source. Adjudicate clinically important errors before releasing affected pairs, regardless of aggregate M3; agree the
second-reader subset and adjudication procedure before the pilot starts.

## Which reviews count

Every rate names its population:

| Population | Definition |
|---|---|
| Uploaded (U) | Clips uploaded to the column review. |
| Submitted (S) | Uploaded clips whose latest label is done in Labelbox. Skipped and unreviewed clips are counted but are not submitted. |
| Returned (R) | Submitted reviews that failed the review checks at least once, whether corrected later or not (from the kept check runs, see below). |
| Valid (V) | Submitted reviews that pass the review checks when the analysis is frozen. |
| Unresolved | Submitted reviews that are not valid at the freeze. They are excluded from every rate computed on V, I or C, and counted. |
| Interpretable (I) | Valid reviews answering "Uninterpretable clip? no". |
| Captioned (C) | Interpretable reviews whose candidate has at least one column. |

The **review checks** are three, and a review that fails any of them is returned for correction. `review_problems` in
`tools/labelbox/column_review.py` runs all three:

- **answers:** every required answer is given, in its place in the form (a nested answer under the answer it belongs
  to), with a choice the form offers, and **Review completed** is selected;
- **details:** every incorrect column has its error reason and its correction, every column that cannot be assessed
  its limitation, every missing column its value and "visible in this input: yes", every additional statement its
  reason (and its copied text for a statement type), and an uninterpretable clip its reason (and a comment for
  "Other");
- **consistency:** the reviewed columns are exactly the candidate's, with their original values; no missing column is
  already a candidate column; no additional statement names a column left supported; and a view or acquisition mode
  answered "no" differs from the proposal, while one that differs is answered "no".

An uninterpretable clip is checked only for its answers and its reason: its other answers are not read.

For the view and type review, `review_problems` in `tools/labelbox/view_type_review.py` checks that every required
question asked is answered, and that the view and acquisition mode are answered exactly for a cine loop.

What is kept while the pilot runs, because a final export shows only the last answers:

- **Exports.** The project is exported after each review session, and always before a review is returned, with the
  export parameters `label_details` and `performance_details` turned on (the SDK leaves both off by default). Every
  export is kept unchanged with the pilot's analysis records, outside the repository.
- **Check runs.** The review checks run on every label of each kept export, with its candidate as recorded at upload.
  Every run is kept: label id, export, and the problems found with the check each belongs to (none for a pass). R and
  M8 are counted from these runs.
- **Timing.** M7 reads each label from the first kept export that contains it.

The metrics describe the **candidate** as generated. Rebuilt captions are an output of the pilot, not something it
scores, apart from how often a correction was needed (M6).

## Metrics

Every metric is reported as counts (numerator / denominator), overall and **for each study**, and by view and
acquisition mode; the column metrics also by column and by target, the cardiac structure a column is about. With 2–3
studies, clips from one study are not independent, so no confidence interval treating clips as independent is
reported; the spread across studies is the measure of consistency.

| | Metric | Numerator / denominator |
|---|---|---|
| C1 | View coverage | I answering "View correct?" yes or no / I |
| C2 | Column coverage | Resolved columns / all columns of I. A column is resolved when it is supported or incorrect; one that cannot be assessed is unresolved. |
| M1 | View accuracy | "View correct? yes" / I answering yes or no. The confusions (proposed view → the view the clip shows) are listed. The acquisition mode is reported the same way. |
| M2 | Confirmed captions | C with every column supported, no missing information, no additional statement, and the view and the acquisition mode correct / C |
| M3 | Clips with an error (primary) | C with at least one incorrect column or one additional statement / C. Reported by error reason. |
| M3s | Column error rate (secondary) | Incorrect columns / resolved columns of I: the scorer's original error rate. Also reported: its original support rate, supported columns / all columns of I. A column that cannot be assessed never counts as supported. |
| M4 | Clips with missing information | "Missing information? yes" / I, with how often each column is missing. |
| M5 | Uninterpretable clips | "Uninterpretable clip? yes" / V, by reason. |
| M6 | Corrections | Incorrect columns given a corrected value, and those removed / incorrect columns. |
| M7 | Review time | Median and interquartile range, over S and per reviewer, of the time spent filling the form: the label's `performance_details.seconds_to_create`, in minutes, from the first kept export that contains it (its first submission). Time spent correcting a returned review is not added (M8 counts those reviews), and `seconds_to_review`, the time of a Labelbox review step, is never part of M7. A missing or zero value is missing; M7 is computed over the labels that have one, with their number, and is reported as missing when fewer than 80% of S have one. It is never estimated from timestamps. |
| M8 | Form problems | R / S, by the check failed (answers, details or consistency). |

The view and type review is reported in counts, without gating the decision: items by type; cine loops by view and
by acquisition mode; items sent back to the column review; and items with identifying text, each a
de-identification failure.

If a second reviewer reads a subset of clips (to be decided before the review starts), their clip-level answers and
their answer for each column are compared clip by clip and reported as counts of agreement. No agreement coefficient
is computed on so few clips.

## Go/no-go

The rules are applied in this order. All rates are over the whole pilot unless stated.

1. **Insufficient evidence** if any of:
   - fewer than 2 studies, or a pilot study with an uploaded clip that is not submitted;
   - V / U < 80%, or unresolved / S > 20%;
   - I < 30, or a study has no interpretable clip, or any denominator a rule below uses is zero;
   - C1 < 80% or C2 < 80%: too much was left unassessed to judge the captions.

   The pilot is extended with new studies before any decision.
2. **No-go** if any of:
   - M3 > 30% (clips with an error);
   - M1 < 80% (view accuracy).
3. **Go** if all of:
   - M3 ≤ 10%, and no single study above 25%;
   - M2 ≥ 70% (confirmed captions);
   - M1 ≥ 90%;
   - M8 ≤ 10%;
   - M4 ≤ 50%;
   - M5 ≤ 25%.
4. **Revise** otherwise.

What each outcome means:

| Outcome | Next step |
|---|---|
| Go | Scale candidate-caption generation and review under this ontology and these forms. |
| Revise | Change the prefill rules, ontology or forms (a new version, approved again) and run a new pilot on new studies. |
| No-go | Reconsider how candidate captions are produced before any further review. |
| Insufficient evidence | Add studies under the same rules; nothing is changed in between. |

M3s, M6 and M7 are reported but do not gate the decision. M3s and M6 guide what a revision should change; M7 shows
how much reviewer time scaling up would take.
