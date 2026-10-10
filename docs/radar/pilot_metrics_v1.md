# RADAR caption pilot: metrics and go/no-go rules (v1)

**Need agreement on the thresholds below before any pilot clip is reviewed.** Until
[approval_v1.md](approval_v1.md) links to the agreement, the thresholds are proposals. Once agreed, this document is
frozen with the ontology, and the pilot outcome is reported under these rules whatever it is. A change made after
seeing pilot results needs a new version and is judged on new pilot studies, never on the clips already reviewed.

The pilot (3–5 training-split studies) tests whether the caption-and-review workflow is feasible under
[`radar-caption-ontology-v1`](ontology_v1.md). It does not measure clinical accuracy or model performance, and its
small size allows only coarse decisions.

## What each review records

The Labelbox review form captures the following for every clip. The metrics use nothing else.

| Item | Answers |
|---|---|
| **Uninterpretable clip?** | no; yes, with a reason (image quality, crop, zoom or outside the square, modality, other) |
| **View correct?** | yes; no, with the correct view class; unsure; not applicable (uninterpretable clip) |
| **Caption supported?** | yes: every statement is supported; no: at least one statement is not supported or is wrong; unsure: nothing is wrong, but at least one statement cannot be confirmed; not applicable |
| **Missing information?** | no; yes, with the missing concepts, plus free text for anything the ontology has no concept for; not applicable |
| **Unsupported or hallucinated information?** | no; yes, with each unsupported statement marked "not supported by the clip" (hallucinated) or "wrong value or severity"; not applicable |
| Unconfirmed statements (optional) | statements that cannot be confirmed, each with a reason: uncertain, or one of the assessment limitations |
| **Corrected caption** | the caption as it should read |
| Comments (optional) | free text, no patient details |

From Labelbox itself: the review status, a pseudonymous reviewer id and the label's timing (see M7). In Labelbox's
terms, filling this form is the labeling step: the pilot's reviewer is Labelbox's labeler, and a separate Labelbox
review step, if the project has one, is not part of the medical review.

A statement is one of the structured claims of the candidate caption (see
[What a caption claims](ontology_v1.md#what-a-caption-claims)): the acquisition mode, a structure's visibility, or a
finding with its value. The view is judged by "View correct?" and is not counted again as a statement.

Statements are judged inside the square outline drawn on each clip, the part of the image the model uses (see the
[labeling instructions](labeling_instructions_v1.md#what-you-review)). A statement that only the area outside the
square shows is unconfirmed, with the reason "crop, zoom or outside the square", not unsupported. The rates below
therefore measure captions against what the model sees, and an unsupported statement is a caption error, not a crop
effect.

## Display readiness and training eligibility

Before pilot review, verify crop geometry and selected model-input frames on synthetic examples under the
[model-input review contract](ontology_v1.md#model-input-review). Items with unverifiable display provenance are
returned for display correction before medical labeling; report their number separately from medical form problems.
For findings visible only outside the selected frames, use an unconfirmed statement with reason `other` and comment
"outside model-input frames". Report this reason separately in the analysis records, without changing the form ids.

A workflow Go does not validate all generated captions. Only individually reviewed corrected captions with resolved
view/acquisition, no unsupported or unconfirmed remaining claim, and verified text/structured-claim agreement may
enter the verified local contrastive dataset. Preserve candidates separately for pilot scoring. Study-report labels
remain a separate weak-supervision source. Adjudicate clinically important errors before releasing affected pairs,
regardless of aggregate M3; agree the second-reader subset and adjudication procedure before the pilot starts.

## Which reviews count

Every rate names its population:

| Population | Definition |
|---|---|
| Uploaded (U) | Clips uploaded for review. |
| Submitted (S) | Uploaded clips whose latest label is done in Labelbox. Skipped and unreviewed clips are counted but are not submitted. |
| Returned (R) | Submitted reviews that failed the review checks at least once, whether corrected later or not (from the kept check runs, see below). |
| Valid (V) | Submitted reviews that pass the review checks when the analysis is frozen. |
| Unresolved | Submitted reviews that are not valid at the freeze. They are excluded from every rate computed on V or I, and counted. |
| Interpretable (I) | Valid reviews answering "Uninterpretable clip? no". |

The **review checks** are three, and a review that fails any of them is returned for correction:

- **answers:** every required answer is given, the corrected caption included, and every choice is one the form offers;
- **consistency:** the answers agree with each other, as listed below;
- **unclaimed flags:** no flag on a statement the candidate caption does not make (see below).

`review_problems` in `app/radar/review_rules.py` runs all three. The consistency checks:

- "Uninterpretable clip? yes" with all four other questions answered "not applicable", and "no" with none of them
  answered "not applicable";
- "View correct? no" with a correct view, other than the view the candidate caption states;
- "Caption supported? yes" with "Unsupported information? no" and no unconfirmed statement;
- "Caption supported? no" if and only if "Unsupported information? yes" with at least one statement;
- "Caption supported? unsure" with "Unsupported information? no" and at least one unconfirmed statement;
- "Missing information? yes" with at least one missing concept or a free-text item;
- no statement both unsupported and unconfirmed.

An **unclaimed flag** marks a statement the candidate caption does not contain (for example an unstated finding ticked
as unsupported), or lists as missing a concept the caption already states. The review is returned for correction. A
review that still carries one at the freeze is not valid: it is unresolved, so it counts in S, R and M8 but in no
rate computed on V or I. An unclaimed flag is thus never counted as a hallucination or as missing information, and
the review is never counted as error-free either.

What is kept while the pilot runs, because a final export shows only the last answers:

- **Exports.** The project is exported after each review session, and always before a review is returned, with the
  export parameters `label_details` and `performance_details` turned on (the SDK leaves both off by default). Every
  export is kept unchanged with the pilot's analysis records, outside the repository.
- **Check runs.** The three review checks run on every label of each kept export, with the view and statements of
  its candidate caption as recorded at upload. Every run is kept: label id, export, and the problems found with the
  check each belongs to (none for a pass). R and M8 are counted from these runs.
- **Timing.** M7 reads each label from the first kept export that contains it.

The metrics describe the **candidate** caption as generated. Corrected captions are an output of the pilot, not
something it scores, apart from how often a correction was needed (M6).

## Metrics

Every metric is reported as counts (numerator / denominator), overall and **for each study**, and by view group and
acquisition mode. With 3–5 studies, clips from one study are not independent, so no confidence interval treating
clips as independent is reported; the spread across studies is the measure of consistency.

| | Metric | Numerator / denominator |
|---|---|---|
| C1 | View coverage | I answering "View correct?" yes or no / I |
| C2 | Statement coverage | Resolved statements / all statements of I. A statement is resolved when it is not unconfirmed. |
| M1 | View accuracy | "View correct? yes" / I answering yes or no. The confusions (classifier class → corrected class) are listed. |
| M2 | Confirmed captions | "Caption supported? yes" / I, with the "no" and "unsure" shares. |
| M3 | Clips with unsupported information (primary) | "Unsupported information? yes" / I. Reported for each kind: not supported (hallucinated) and wrong value. |
| M3s | Unsupported statements (secondary) | Unsupported statements / resolved statements. Also reported: the observed unsupported fraction (unsupported / all statements, so labelled). An unconfirmed statement never counts as supported. |
| M4 | Clips with missing information | "Missing information? yes" / I, with how often each concept is missing and the free-text items (ontology feedback). |
| M5 | Uninterpretable clips | "Uninterpretable clip? yes" / V, by reason. |
| M6 | Corrections | Corrected caption differs from the candidate (ignoring whitespace) / I. |
| M7 | Review time | Median and interquartile range, over S and per reviewer, of the time spent filling the form: the label's `performance_details.seconds_to_create`, in minutes, from the first kept export that contains it (its first submission). Time spent correcting a returned review is not added (M8 counts those reviews), and `seconds_to_review`, the time of a Labelbox review step, is never part of M7. A missing or zero value is missing; M7 is computed over the labels that have one, with their number, and is reported as missing when fewer than 80% of S have one. It is never estimated from timestamps. |
| M8 | Form problems | R / S, by the check failed (answers, consistency or unclaimed flags). |

If a second reviewer reads a subset of clips (to be decided before the review starts), their answers to the five
multiple-choice questions are compared clip by clip and reported as counts of agreement. No agreement coefficient is
computed on so few clips.

## Go/no-go

The rules are applied in this order. All rates are over the whole pilot unless stated.

1. **Insufficient evidence** if any of:
   - fewer than 3 studies have every uploaded clip submitted;
   - V / U < 80%, or unresolved / S > 20%;
   - I < 30, or a study has no interpretable clip, or any denominator a rule below uses is zero;
   - C1 < 80% or C2 < 80%: too much was left unconfirmed to judge the captions.

   The pilot is extended with new studies before any decision.
2. **No-go** if any of:
   - M3 > 30% (clips with unsupported information);
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
| Go | Scale candidate-caption generation and review under this ontology and form. |
| Revise | Change the prompts, ontology or form (a new version, approved again) and run a new pilot on new studies. |
| No-go | Reconsider how candidate captions are produced before any further review. |
| Insufficient evidence | Add studies under the same rules; nothing is changed in between. |

M3s, M6 and M7 are reported but do not gate the decision. M3s and M6 guide what a revision should change; M7 shows
how much reviewer time scaling up would take.
