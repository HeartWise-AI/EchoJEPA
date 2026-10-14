# DeepECHO view → report fields → simple claim review

**Current Labelbox interface:** [column review v2](column_review_v2.md) uses 46 named fields,
prefilled values and captions (including explicit normals), and dropdowns for missing and unsupported findings.
Its [native ontology](../../configs/radar/labelbox/column_review_v2.json) supersedes the generic five-slot proposal
as the proposed review interface. This page retains the source-data audit and earlier mapping for reference.

**Proposal for clinical review.** This simplifies the reviewer interface; it does not medically approve an ontology
or replace the pending `radar-caption-ontology-v1`. Every video receives its actual predicted view, acquisition mode
and at most five numbered candidate findings. Reviewers confirm **Yes / No / Cannot assess** per populated finding.
Blank slots are marked “No claim.” No bounding-box drawing, report rewriting or numeric measurement is required.

## What was checked in the actual data

The cleaned DeepECHO report parquet was read directly: **249,084 rows, 170 columns**. The current metadata parquet
has **20,095,828 DICOM rows**, including **10,227,071 predicted views** and **9,868,757 without a prediction**.
These are DICOM-row counts, not a new count of unique physical videos or TTE-only clips. All twenty classifier labels
are present. This file has changed since the Notion snapshot; do not copy the older totals into a new cohort audit.

Dedicated fields actually exist for `Right ventricule findings`, `Right Atrial findings`, `Pulmonary valve findings`,
`Pericardium findings`, and view-specific wall-motion scores. Prefer those over searching the entire conclusion.
`Worksheet Inspiratory Collapse` is entirely empty in this export; use `Inferior vena cava findings` when it explicitly
describes respiratory collapse. RV function is not always observed: **55,276 rows use the missing sentinel -1**.

The DeepECHO report-processing code explicitly defines RV function as `0 = normal`, `1 = mild dysfunction`,
`2 = moderate dysfunction`, `3 = severe dysfunction`, `-1 = missing`. The MR/AR/TR/PR grade columns contain mixed
numeric encodings, words and malformed values; their numeric severity codebook has **not** been established here.
Do not guess that `1/2/3/4` means a particular grade. Use explicit, negation-aware text until the codebook is verified.

## The view-to-field table

Read this as “which report information should be considered for this video,” not “copy every value into every clip.”
The exact EF number, dimensions, gradients and pressures remain study-level context or separate supervised targets.
They are never automatically verified by a Yes on a qualitative clip claim.

| View label(s) | Main B-mode claims to prefill, when explicitly reported | Exact DeepECHO sources | Color Doppler only | What must not be copied into this clip |
|---|---|---|---|---|
| A4C | LV systolic function (EF-related qualitative impression); LV size; RV function/size; LA size. Optional RA size or visible-segment wall motion. | `Left ventricule findings`; `Visually Estimated EF` as context; `Right ventricule findings`; `MHI VD fonction systolique`; `Left atrial findings`; `Right Atrial findings`; 4C WMS fields. | Mitral jet from `Mitral valve findings`/`Referring MR Grade`; tricuspid jet from `Tricuspid valve findings`/`Referring TR Grade`, only if the color box covers that valve. | Routine aortic-valve claims; exact EF; TAPSE/RVSP; diastolic grade; global absence of wall-motion abnormalities. |
| A4C_LV | LV function, LV size, local wall motion. | `Left ventricule findings`; 4C WMS; `Visually Estimated EF` as context. | No default valve claim; add one only after confirming the focused target. | RV/RA/valves merely because the base class is A4C. |
| A2C | LV function/size; local anterior/inferior wall motion; LA size; an explicitly reported mitral device if visible. | `Left ventricule findings`; 2C WMS; `Left atrial findings`; `Mitral valve findings`. | Mitral regurgitant jet. | RV, TV, AV; exact biplane EF from this one clip. |
| A2C_LV | LV function/size and local wall motion. | `Left ventricule findings`; 2C WMS. | No default valve claim. | LA/MV unless actually included. |
| A3C | LV function; local anteroseptal/inferolateral motion; AV morphology; visible mitral device. | `Left ventricule findings`; LAX WMS; `Aortic valve findings`; `Mitral valve findings`. | Mitral and aortic regurgitant jets, if individually covered. | Aortic stenosis severity, LVOT gradient, exact EF. |
| A3C_LV | LV function and local motion. | `Left ventricule findings`; LAX WMS. | No default valve claim. | Assume AV/MV remain visible after focusing. |
| A5C | LV function/size; visible AV/LVOT morphology. | `Left ventricule findings`; `Aortic valve findings`. | Aortic or mitral jet, only when its valve is covered. | AV velocity/gradient/area or stenosis grade from grayscale/color alone. |
| PLAX | LV size/wall thickness; AV morphology; mitral device; root/ascending aorta morphology. Optional local effusion. | `Left ventricule findings`; `Aortic valve findings`; `Mitral valve findings`; `Aorta findings`; `Pericardium findings`. | Mitral/aortic regurgitant jet. | Exact EF; measurements or stenosis severity; arch dilation from a root image. |
| PLAX_DEEP | Pericardial fluid; localized aortic findings where visible. | `Pericardium findings`; `Aorta findings`. | No default jet claim. | Label pleural fluid as pericardial fluid; treat descending aorta as root/ascending aorta. |
| PSAX | LV contraction, wall thickness and motion in the visible slice. | `Left ventricule findings`; SAX WMS fields. | No default valve claim; the slice level must be confirmed first. | AV/TV/PV merely from generic PSAX; whole-LV normal motion from one slice. |
| PSAX_AV | AV morphology, especially an adequately seen en-face valve. | `Aortic valve findings`; optionally valve-device text. | Aortic, tricuspid or pulmonic jets from the matching valve field, only if identified and covered. | LV EF; integrated regurgitation/stenosis grades; bicuspid morphology when leaflet opening cannot be resolved. |
| RVINF | Local RV function/size; optional visible tricuspid device/lead. | `Right ventricule findings`; `MHI VD fonction systolique`; `Tricuspid valve findings`. | Tricuspid jet. | LV EF, pulmonary pressure, global RV quantitative assessment. |
| SUBCOSTAL | First identify subtype: four-chamber → chambers/pericardium/septum; IVC-focused → IVC size/collapse. | `Right ventricule findings`; `Pericardium findings`; `Inter-atrial septum findings`; `Inferior vena cava findings`. | Tricuspid or septal flow only after target confirmation; shunt interpretation needs its own reviewed concept. | All subcostal anatomy in every clip; numeric RA pressure; respiratory collapse from a loop without respiration. |
| SUPRASTERNAL | Arch morphology, if explicitly reported and visible. | `Aorta findings`; `Ao Arch Diameter` as context. | No default valve jet claim. | Root measurements, LV EF, Doppler obstruction or velocities from a grayscale loop. |
| A2C_ZOOM, A3C_ZOOM, A4C_ZOOM, A5C_ZOOM, PLAX_ZOOM | Confirm the zoom target, then retain only its claims from the base-view row. A valve-focused zoom normally gets valve/device claims, not whole-chamber function. | Only the report section of the confirmed target. | Only the jet at that confirmed valve. | Inherit every expected structure or the parent's EF field automatically. |
| OTHER | No automatically routed clinical caption. Correct the view or keep a separately flagged weak study-level pair. | None until the view/target is resolved. | None by default. | Any finding assigned from the study report solely because the clip belongs to that study. |

These rows cover all twenty labels. The [machine-readable proposal](../../configs/radar/simple_review_proposal_v1.json)
contains the per-label priorities and allowed simple values. Optional priorities and SUBCOSTAL subtypes require
target confirmation; they are not already a complete automatic visibility detector.

## Values and report columns

| Review concept | Simple value choices | Report source and context |
|---|---|---|
| LV function / EF-related impression | hyperdynamic; normal; mildly/moderately/severely reduced | Prefer explicit `Left ventricule findings`. Display `Visually Estimated EF` separately as reported context. Do not invent qualitative grades from arbitrary numeric EF cutoffs. |
| LV/RV/LA/RA size | normal; dilated | The corresponding chamber findings; RV function also uses its verified codebook. Numeric dimensions, LA volume index and RA area are context only. |
| LV wall thickness | not increased; increased | Explicit LV text. IVS/LVIW thickness is a separate numeric target; numeric thresholds require their own definition. |
| Local LV motion | normal in visible segments; abnormal in visible segments | Use only the relevant view/segment scores, with a verified WMS codebook; explicit text may route a candidate, but does not localize it by itself. |
| Mitral device | clip; annuloplasty ring; bioprosthesis; mechanical prosthesis | `Mitral valve findings`, explicit presence only. Missing report mention is not an absent device. |
| Aortic morphology | calcified; restricted opening; bicuspid/three-cusp morphology; prosthesis | `Aortic valve findings`, explicit findings only. AV stenosis severity remains study-level. |
| MR/AR/TR/PR jet | demonstrated; not demonstrated in this acquisition | Corresponding valve findings and an explicitly decoded report assertion; review requires color Doppler with adequate target coverage. Presence does not establish grade. |
| Pericardial fluid | present; not demonstrated locally | `Pericardium findings`. Do not infer full-study absence from unseen pericardial spaces. |
| IVC | normal/dilated; collapse preserved/reduced | `Inferior vena cava findings`; diameter as context. Respiratory collapse requires captured respiratory variation. |
| Aorta | dilation, with its location retained | `Aorta findings`; distinguish root, ascending aorta and arch. Diameters stay separate numeric targets. |

An explicit report “no MR” can propose “no jet demonstrated,” but the reviewer must confirm acquisition adequacy.
An empty report field supplies **no candidate**, never a normal value. A broad “no significant abnormality” sentence
is not decomposed into many specific normal labels without an explicit extraction rule and medical approval.

### Wall-motion routing from the real schema

| View | Post-2020 segment columns | Legacy fields to reconcile |
|---|---|---|
| A4C | `WMS 4C Apical Lateral`, `WMS 4C Mid Anterolateral`, `WMS 4C Basal Anterolateral`, `WMS Basal Inferoseptal`, `WMS 4C Mid Inferoseptal`, `WMS 4C Apical septum` | `A4C_BS`, `A4C_MS`, `A4C_AS`, `A4C_AL`, `A4C_ML`, `A4C_BL` |
| A2C | `WMS 2C Apical Anterior`, `WMS 2C Mid Anterior`, `WMS 2C Basal Anterior`, `WMS 2C Basal Inferior`, `WMS 2C Mid Inferior`, `WMS 2C Apical Inferior` | `A2C_BI`, `A2C_MI`, `A2C_AI`, `A2C_AA`, `A2C_MA`, `A2C_BA` |
| A3C / long axis | `WMS LAX Mid Anteroseptal`, `WMS LAX Basal Anteroseptal`, `WMS LAX Basal Inferolateral`, `WMS LAX Mid Inferolateral` | `LAX_MAS`, `LAX_BAS`, `LAX_BP`, `LAX_MP` |
| PSAX | `WMS SAX Anterior`, `WMS SAX Anterolateral`, `WMS SAX Inferolateral`, `WMS SAX Inferior`, `WMS SAX Inferoseptal`, `WMS SAX Anteroseptal` | `SAX_MAS2`, `SAX_MIL2`, `SAX_MIS2`, `SAX_MAL2`, `SAX_MI2`, `SAX_MA2` |

`WMS 4C Apical Cap`, `WMS 2C Apical Cap`, `APEX`, `APEX2`, `Rest Echo WMS` and `Rest Echo WMS Index` need separate
handling. A generic PSAX classifier label does not identify basal/mid/apical level. Column names establish routing
possibilities, not visibility or a verified code-to-segment equivalence across legacy and current exports.

## What the reviewer does in Labelbox

The [small form](../../configs/radar/labelbox/simple_review_proposal_v1.json) uses one schema for every view.
The upload context must show **view, modality and numbered claim cards 1–5**, including the proposed value and
evidence input. The field “claim 1” corresponds to card 1; generic slot questions cannot magically display a
different per-row clinical label inside the native ontology. Display/attachment integration is required before use.

1. Confirm the proposed view and modality.
2. For each of up to five populated cards, answer Yes / No / Cannot assess.
3. Flag important missing information only when needed. Optional corrections are free text and require adjudication.

Draft answers may be presented as suggestions, but **no item is accepted until the clinician reviews and submits**.
Unused slots can be preannotated “No claim”; clinically meaningful slots require explicit confirmation.

Score the original candidate: supported claims / populated claims; resolved claims / populated claims; rejected
claims / resolved claims. “Cannot assess” is neither supported nor a clinical negative. A rejected claim is removed,
not inverted into “disease absent.” If the view or modality is wrong/uncertain, fix routing before releasing a pair.
Missing information or a free-text correction sends the item for adjudication. Rebuild captions from supported claims
and validate their structured/text agreement; do not train on the report number shown in the context panel.

## Linked video examples

The original three TTE clips were selected from two separately report-linked studies and inspected as deidentified local display exports:
A4C, PLAX and PSAX_AV. Their classifier probabilities are approximately 0.974, 0.998 and 0.978, respectively.
These are classifier scores, not clinical confidence or medical approval. The examples are kept outside GitHub;
their private linkage manifest is not part of the repository.

Each clip uses its own linked report for EF context, qualitative LV function, segment scores and valve findings.
A4C uses one study; PLAX and PSAX_AV use the second. The review cards are routed separately:

| Example | Candidate cards | What stays out |
|---|---|---|
| A4C B-mode | LV function; LV size; visible-segment motion | AV findings; mitral regurgitant jet without color; exact numerical EF |
| PLAX B-mode | LV size/wall morphology; visible aortic/mitral morphology if explicitly reported | Copying the report EF or MR grade into a verified local caption |
| PSAX_AV B-mode | AV morphology when systolic leaflet opening is adequately seen | LV EF; color jet absence without color; stenosis severity |

The [published gallery](https://deepecho-three-view-demo.papirobbi.chatgpt.site) contains the five view demonstrations
and a complete study B with all 44 video loops, 33 cropped image exports and its report records,
with valve morphology, ASE segment maps and confirmation-based captions. Display deidentification changes the field
of view, so these exports are **illustrations, not validated production model inputs**. They are not uploaded to
Labelbox or used as pilot training pairs. Production review still needs exact spatial/temporal input provenance.

## Sources and implementation status

- Actual cleaned DeepECHO parquet schema and aggregate field checks, and actual current view metadata.
- [DeepECHO report-processing source](https://github.com/HeartWise-AI/DeepECHO/blob/main/utils/main.py) for RV code meanings.
- [ASE comprehensive TTE recommendations](https://www.asecho.org/guideline/comprehensive-tte-in-adults/).
- [ASE chamber quantification recommendations](https://www.asecho.org/wp-content/uploads/2016/02/2015_ChamberQuantificationREV.pdf).
- [ASE native regurgitation recommendations](https://www.asecho.org/guideline/native-valvular-regurgitation-by-echo/).

Implemented here: proposed routing/value specification, SDK-compatible small form, a submission scorer and local
examples. Still needed before clinical rollout: medical approval, verified MR/AR/TR/PR and WMS codebooks, robust
negation-aware extraction, target/visibility checks, Labelbox context integration and import/editor verification.
