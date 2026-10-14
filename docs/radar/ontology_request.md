# Request: a clinically grounded echo caption ontology after EchoJEPA

Ref: [issue #21](https://github.com/HeartWise-AI/EchoJEPA/issues/21) and
[PR #25](https://github.com/HeartWise-AI/EchoJEPA/pull/25), reviewed at commit
`6d4c55afd6c52007c31cd4301f027a29b1058e49`.

Build a versioned, bilingual ontology and dataset that align the **video actually received by the encoder** with
its supported view, anatomy and findings. First pretrain the video encoder with EchoJEPA; then train video–text
projectors and structure-wise contrastive alignment. Use the existing v1 taxonomy, mappings and Labelbox form as
the starting point. The proposed changes and clinical choices below require medical review before v1 is frozen.
This document is a research specification and review request, not medical sign-off.

## Evidence and manuscript review

**RADAR — Zhang et al., Science, 2026.** [Paper](https://doi.org/10.1126/science.aec6129),
[author implementation](https://github.com/alibaba-damo-academy/damo-radar). The author release describes alignment
at anatomical granularity. Its [training code](https://github.com/alibaba-damo-academy/damo-radar/blob/0dbf0ece209b77d722588e5e7943014d3db48f7f/RADAR_train/lavis/models/radar_models/radar_pretrain.py) constructs targets using
normal-status flags, identical text and momentum-text similarities among abnormal descriptions. The transferable
idea is anatomy-specific alignment with duplicate-aware targets. Echo's view labels are uncertain routing priors;
they are not anatomical segmentation masks. Our echo adaptation must validate visibility rather than assume every
structure is present. The Science full text could not be retrieved (publisher returned HTTP 403); detailed
implementation claims here are grounded in the author code, rather than an asserted full-text review.

**EchoPrime — Vukadinovic et al., Nature, 2026.**
[Paper](https://doi.org/10.1038/s41586-025-09850-x). The supplied manuscript's contrastive-training methods pair a
video with its study report, followed by view-informed multivideo interpretation. This supports a video–text
baseline and study-level aggregation, but does not establish that every report sentence is visible in each clip.
Our contribution should be explicitly reviewed local captions and structure-wise evidence routing. Use view
probabilities as an ablation; never turn the classifier's expected anatomy into a verified visibility label.

**PanEcho — Holste et al., JAMA, 2025.**
[Paper](https://doi.org/10.1001/jama.2025.8731).
The supplied manuscript defines 39 report-derived tasks (18 classification, 21 regression), shares a video encoder
across tasks and evaluates interpretations at study level. Use its task vocabulary and supervised predictions as
comparators, not as clip-caption ground truth. Its merged classes must remain set-valued mappings: a
moderate-or-severe source label does not identify which grade is present. Patient-level separation is essential.

**Clinical references.** [ASE comprehensive TTE recommendations](https://www.asecho.org/guideline/comprehensive-tte-in-adults/),
[ASE native regurgitation recommendations](https://www.asecho.org/guideline/native-valvular-regurgitation-by-echo/),
[EACVI/ASE aortic stenosis recommendations](https://www.asecho.org/guideline/update-of-aortic-valve-stenosis/).
These inform our proposed distinction between a local visual observation and an integrated clinical diagnosis.
The latter may require additional views, spectral Doppler and measurements.

## Review of PR #25

The existing PR documents the view and structure taxonomies, source mappings, machine-readable form, EN/FR
instructions, synthetic examples and pilot denominators. It separates visibility from pathology and excludes
measurement-dependent concepts from captions. Its approval record is pending. The latest PR description already
specifies a crop outline, updating the earlier discussion quoted in the request.

Resolve these points before medical approval:

| Priority | Finding | Requested decision or change |
|---|---|---|
| High | A center-crop outline describes spatial support but not selected frames; pretraining also uses random crops. | Apply the display contract below and prevent training augmentations from removing evidence for an accepted caption. |
| High | MR/AR/TR/PR share a severity scale marked visual; a report's integrated grade may not be decidable from one color loop. | Distinguish visible jet/presence from integrated severity. Until that extension is approved, leave an indeterminate grade unconfirmed and remove it from corrected captions. |
| High | Numerical EF is caption-eligible despite multiview requirements, while reviewer instructions exclude measurements. | Decide whether to exclude numerical EF from local captions or retain only explicitly qualified visual estimates with sufficient multiview evidence. Never use a study EF as a verified one-clip value. |
| High | A workflow-feasibility “Go” permits unsupported candidate captions. | Release only individually validated corrected pairs for contrastive training; the pilot's Go is not approval of every candidate. |
| Medium | OTHER has no local finding prior, although the existing project proposal allows soft routing. | Keep conservative local-caption behavior and allow OTHER clips only in a separately documented weak study-level training arm. |
| Medium | SUBCOSTAL groups four-chamber and IVC images; aorta is represented chiefly as aortic root; LVOT has no visibility concept. | Confirm these boundaries and whether to add IVC-focused, RV-focused, PSAX level, LVOT, aortic arch/ascending aorta and septal localization in a subsequent schema revision. |
| Medium | Mapping report EF to visual EF conflates the quantity with its measurement method. | Keep provenance for visual, Simpson, 3D and unspecified report EF; do not imply method equivalence or clip support. |
| Medium | Small pilot has optional second-reader review and broad aggregate rates. | Agree a blinded double-read subset before review; examine disagreements and study/view/modality strata. Treat all thresholds as feasibility proposals. |

The approval verifier checks recorded content hashes and ancestry; it does not authenticate the reviewer behind a
link. The approval procedure correctly requires a human to verify the GitHub sign-off. Do not fill that record from
this review. Changes to v1 are allowed while pending; changes after approval require a new version and approval.

## Ontology and dataset contract

Keep three distinct layers:

1. **Acquisition and visibility:** view class and uncertainty, modality, anatomical identity, visibility and quality
   within the model input. A structure may be identifiable yet inadequate for functional assessment.
2. **Local clinical observations:** finding, value, assertion (present/absent/uncertain), scope and limitations.
   Use only observations supported by this input. Absence in the report, absence from the field of view and
   demonstrated absence are different states. Normality of unseen anatomy is never inferred.
3. **Study-level findings:** integrated grades and numeric measurements, with units, method and report provenance.
   Retain these for supervised evaluation or a separately flagged weak structure–report objective. Never silently
   promote them into verified local captions.

For an extension to the v1 schema, request fields for evidence scope (`local_input`, `multiview_study`,
`report_only`), assertion, measurement method, visible segment/location, supporting input ids and assessment
limitations. Define per-concept allowed scopes and modality requirements. Add separate concepts for regurgitant
jet observation and integrated regurgitation severity, so a reviewer can preserve presence without inventing a grade.
These fields are proposed additions, not already implemented v1 fields.

Every dataset pair must retain opaque study/patient grouping keys in secure storage, input id, ontology version and
hash, preprocessing version, crop geometry, selected source-frame indices, modality and its provenance, candidate
structured claims/text, review status and reviewer, corrected claims/text, and acceptance reason. Preserve the
candidate separately from the correction. Do not place patient data or source report text in repository artifacts.
The future importer must parse corrected captions back to validated claims and flag text/claim disagreement for
adjudication; the current free-text correction field alone cannot enforce that contract.

Only local pairs with checked display provenance, resolved view/acquisition, validated corrected text/claims and no
remaining unsupported or unconfirmed claim enter the verified contrastive dataset. Uninterpretable inputs,
unresolved reviews and report-only claims are excluded from that dataset. If the crop removes a real finding,
removing that finding is a visibility limitation; it must not produce a negative clinical label.

Synthetic examples for the requested extension:

- EN: “Apical four-chamber view, B-mode. Reduced LV contraction in the visible segments.”
  FR: « Coupe apicale 4 cavités, mode B. Contraction VG diminuée dans les segments visibles. » No numerical EF.
- EN: “Apical four-chamber view, color Doppler. Mitral regurgitant jet visible; severity cannot be determined.”
  FR: « Coupe apicale 4 cavités, Doppler couleur. Jet d'insuffisance mitrale visible ; sévérité indéterminée. »
  Use proposed jet-presence semantics; do not serialize this as an existing v1 severity value.
- A report mentions effusion, visible only outside the crop: flag the candidate effusion unconfirmed, remove it
  from the corrected local caption and retain the report label only at study scope. Never label “no effusion”.

## Review display contract

The proposed v1 display is a full loop with the exact crop outline plus an identified model-input frame sequence.
A synchronized crop-only panel is useful if Labelbox supports it, but must not substitute a different crop or
temporal window. The model-input panel is the evidence; the full loop helps identify anatomy and view.

The upload implementation must derive geometry from the actual transform and record frame indices, padding,
source/resized dimensions, interpolation and preprocessing version. The 256-short-side/224-center-crop description
matches the default frozen-evaluation transform in `evals/video_classification_frozen/utils.py`; it is not a
universal description of `app/vjepa/transforms.py`, which uses random resized crops. Input resolution is configurable.
Do not draw a fixed square on the assumption that every run uses the default.

Outline overlays and panel labels are review aids only; the encoder receives neither. Verify the display using
synthetic wide, tall and odd-dimension inputs and selected-frame examples before clinical review. A display whose
crop or temporal sequence cannot be verified is returned for correction, not scored as a medical caption error.
The upload/display implementation remains outside issue #21 and is not delivered by this documentation PR.

## EchoJEPA then contrastive learning

1. Freeze patient splits before any corpus construction, including repeat studies and local/weak labels. If a
   temporal test is used, exclude test patients from earlier training studies too. Pretrain EchoJEPA only on the
   training partition; document any transductive experiment separately.
2. Start with the pretrained encoder frozen and train normalized video/text projection heads on accepted local
   pairs. Evaluate a French clinical text encoder; translation needs its own audit. Compare against global
   study–report alignment using the same splits and compute budget.
3. Add view-probability and modality conditioning outside the encoder. Compare hard routing with soft priors;
   require visibility evidence for verified local pairs even when routing is soft.
4. Add one study-level query per structure over clip embeddings. Maintain a separate weak report objective for
   findings requiring several views or Doppler data. Missing spectral stills mean unavailable measurement evidence,
   not absent stenosis or normal pressure. Attention weights identify influential inputs, not proof of anatomy.
5. Prevent false negatives from templated text: treat identical accepted same-structure semantics as multipositives,
   excluding scope/assertion conflicts. Never equate missing/unassessable with normal. Ablate adaptive momentum
   targets against ordinary contrastive loss before adopting them; CT results do not establish benefit in echo.
6. Partly unfreeze only after a stable frozen baseline. Restrict spatial/temporal augmentation to evidence-preserving
   transforms for verified captions; otherwise revalidate or omit the affected pair. Label-free EchoJEPA augmentations
   do not automatically remain valid for caption supervision.

Report local retrieval and caption-support audits separately from held-out study classification/regression.
Use AUROC and AUPRC per finding, sensitivity/specificity at validation-chosen thresholds, and MAE in native units
for regression. Report patient-grouped uncertainty and view/modality/subgroup coverage on an adequately sized test
set. Retain ordinal grades and negation in hard retrieval negatives. Never tune thresholds on test results.

## Pilot and medical review request

The existing [pilot metrics](pilot_metrics_v1.md) are explicitly feasibility metrics for 3–5 training studies.
Their denominators, unconfirmed-statement handling and per-study reporting are useful. Add a display-readiness gate,
reasons for temporal exclusion, clinically important-error review and the verified-pair acceptance rule above.
Report proposed thresholds as engineering decisions, not guideline-derived clinical acceptance limits. A rare major
error warrants adjudication even when aggregate M3 meets Go. Predetermine the double-read subset and adjudication
procedure; a larger external validation is a separate experiment.

Request Dr. Avram's review of all v1 taxonomies, priors, mappings, EN/FR instructions and examples, then explicit
decisions on numerical EF, regurgitation scope, absent findings, focused/zoomed views, temporal input, and pilot
thresholds. Charlotte incorporates the decisions into YAML, generated documents and form; Jacques checks the input
and dataset provenance contract. Run generation checks, the Labelbox SDK round trip and an editor inspection, then
request medical approval naming the final content commit. The approval record remains pending until that evidence
exists. No clinical pilot upload or training run is authorized by this document.
