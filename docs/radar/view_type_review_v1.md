# DeepECHO view and type review (v1): relabelling `OTHER` items

**The form for the clips the view classifier labels `OTHER`.** Not every `OTHER` clip is useless: one can show
the inferior vena cava, another flow through a ventricle. Captions cannot be generated for them, because the view
does not say which anatomy is shown, so they get this form instead of the caption review. It records the view each
item shows and what kind of item it is, and needs no report and no caption. It is approved with ontology v1 and the
column review ([approval_v1.md](approval_v1.md)), and reviewers follow the
[instructions](labeling_instructions_v1.md#view-and-type-review-other-clips).

It is used twice: for every `OTHER` video of the 2–3 pilot studies, next to the caption review of their classified
videos ([column review v2](column_review_v2.md)); and later for a larger sample (about 500 `OTHER` items, many of
them still or measurement frames) to decide whether the view classifier needs new classes.

The results measure what `OTHER` contains, give labelled examples for the view classifier, and send relabelled cine
loops and plain stills back into the column review v2.

## What the reviewer does

1. Specify the item type (`input_type`). A **measurement still** has calipers, tracings or measured values drawn on the
   image. An **analysis screen** is a quantification tool's output (strain, 3D, EF), with or without an echo image.
2. Give the view (`view`), from the same classes as the column review (described in
   [ontology_v1.md](ontology_v1.md#views)):
   - `OTHER` only for an echo image with an off-axis or non-standard view;
   - `cannot_tell` for an echo image whose view cannot be identified;
   - `no_view` when there is no ultrasound image at all (a text page or a screen without an echo image).
3. Give the acquisition mode (`modality`); `not_applicable` exactly when the view is `no_view`.
4. Indicate whether identifying text is visible (`identifier_visible`). Answer "yes" if a name, record number, or date of birth appears anywhere on the item. Never copy identifiers into comments. Items marked "yes" are deleted from Labelbox and all datasets.

## The form

Every question is a global classification. The table is generated from
[`tools/labelbox/view_type_review.py`](../../tools/labelbox/view_type_review.py), which also writes
[`configs/radar/labelbox/view_type_review_v1.json`](../../configs/radar/labelbox/view_type_review_v1.json):

<!-- BEGIN GENERATED: view_type_form -->
| Question | Id | Required | Answers |
|---|---|---|---|
| What is this item? | `input_type` | yes | `cine_loop` Cine loop (moving echo images); `still_image` Still frame of an echo image; `measurement_still` Still frame with measurements, calipers or tracings; `analysis_screen` Analysis or quantification screen (strain, 3D, EF tool); `text_page` Report, worksheet or other text page; `other` Other, no echo image |
| Which view does it show? | `view` | yes | `A2C` A2C: Apical 2-chamber view; `A2C_LV` A2C_LV: Apical 2-chamber view, LV-focused; `A2C_ZOOM` A2C_ZOOM: Apical 2-chamber view, zoomed; `A3C` A3C: Apical 3-chamber view; `A3C_LV` A3C_LV: Apical 3-chamber view, LV-focused; `A3C_ZOOM` A3C_ZOOM: Apical 3-chamber view, zoomed; `A4C` A4C: Apical 4-chamber view; `A4C_LV` A4C_LV: Apical 4-chamber view, LV-focused; `A4C_ZOOM` A4C_ZOOM: Apical 4-chamber view, zoomed; `A5C` A5C: Apical 5-chamber view; `A5C_ZOOM` A5C_ZOOM: Apical 5-chamber view, zoomed; `PLAX` PLAX: Parasternal long-axis view; `PLAX_DEEP` PLAX_DEEP: Parasternal long-axis view, deep; `PLAX_ZOOM` PLAX_ZOOM: Parasternal long-axis view, zoomed; `PSAX` PSAX: Parasternal short-axis view; `PSAX_AV` PSAX_AV: Parasternal short-axis view at the aortic valve; `RVINF` RVINF: Right ventricular inflow view; `SUBCOSTAL` SUBCOSTAL: Subcostal view; `SUPRASTERNAL` SUPRASTERNAL: Suprasternal view; `OTHER` OTHER: off-axis or non-standard echo view; `cannot_tell` Cannot tell the view; `no_view` No echo view (text, screen or no ultrasound image) |
| Acquisition mode | `modality` | yes | `bmode` B-mode; `color_flow` Color Doppler; `spectral` Spectral Doppler; `m_mode` M-mode; `mixed` Mixed / analysis acquisition; `unresolved` Unresolved; `not_applicable` Not applicable: no ultrasound image |
| Is identifying text visible (a name, a record number, a date of birth)? | `identifier_visible` | yes | `no` No; `yes` Yes: the item must be deleted |
| Comments (no patient details) | `comments` | no | free text |
<!-- END GENERATED: view_type_form -->

A review is valid when `review_problems` finds nothing: every required answer is given from the form's choices; a
text page or "other" item has `no_view`; a cine loop or still (with or without measurements) has a view or
`cannot_tell`; and `no_view` goes with `not_applicable`, never with an acquisition mode. Decode an export with
`decode_annotations` in `tools/labelbox/column_review.py` before checking it.

## Back into the column review

`route` turns a valid review into the `view`, `modality` and `input_kind` arguments of the column review's
`build_candidate`, for a cine loop (`video`) or a plain still (`image`), in a view the column review routes (not
`OTHER`), in B-mode or color Doppler, with no identifying text. The column review then gives a still no motion or
function proposals. Everything else stays out of the column review: measurement stills, analysis screens, text
pages, spectral and M-mode items, and items whose view cannot be told.

## Selecting and uploading the items (not in this change)

- The pilot's items are the `OTHER` videos of its 2–3 studies, screened like the caption clips (still clips and
  visible measurements removed).
- The larger sample is drawn from training-split patients only, never from validation or test patients, and spread
  over studies, years and scanner models, with a cap per study.
- Upload only de-identified copies, as for the column review.
- The form has only global classifications: the same JSON can be created as an image ontology (stills) and as a
  video ontology (cine loops).

## Commands

```bash
python -m tools.labelbox.view_type_review          # write the form JSON and the table above
python -m tools.labelbox.view_type_review --check   # exit 1 if either is out of date (also a unit test)
```
