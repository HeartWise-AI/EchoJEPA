# DeepECHO view and type review (v1): relabelling `OTHER` items

**The form for the clips the view classifier labels `OTHER`.** Not every `OTHER` clip is useless: one can show
the inferior vena cava, another flow through a ventricle. Captions cannot be generated for them, because the view
does not say which anatomy is shown, so they get this form instead of the caption review. It records what kind of
item each one is and, for an echo cine loop, its view and acquisition mode, and needs no report and no caption. It is approved with ontology v1 and the
column review ([approval_v1.md](approval_v1.md)), and reviewers follow the
[instructions](labeling_instructions_v1.md#view-and-type-review-other-clips).

It is used for every `OTHER` video of the 2–3 pilot studies, next to the caption review of their classified videos
([column review v2](column_review_v2.md)). Those videos are screened like the caption clips before upload: still
clips and clips with visible measurements are removed, and single-frame objects (measurement frames, report pages)
are not videos. The larger sample proposed later (about 500 `OTHER` items, many of them still or measurement frames)
needs either the same screening or a form version that has answers for those item types.

The results measure what `OTHER` contains, give labelled examples for the view classifier, and send relabelled cine
loops back into the column review v2.

## What the reviewer does

1. Specify the item type (`input_type`). An **echo cine loop** is a moving echo image in any acquisition mode. An
   **analysis screen** is a quantification tool's output (strain, 3D, EF), with or without an echo image. An item the
   screen before upload should have removed (a still clip, visible measurements, a text page) is reported in the
   comments, as in the column review.
2. For an echo cine loop only, give the view (`view`), from the same classes as the column review (described in
   [ontology_v1.md](ontology_v1.md#views)):
   - `OTHER` only for an off-axis or non-standard echo view;
   - `cannot_tell` for an echo view that cannot be identified.
3. For an echo cine loop only, give the acquisition mode (`modality`).
4. Indicate whether identifying text is visible (`identifier_visible`). Answer "yes" if a name, record number, or date of birth appears anywhere on the item. Never copy identifiers into comments. Items marked "yes" are deleted from Labelbox and all datasets.

For any other item type, Labelbox does not show the view and acquisition mode questions, and `with_not_applicable`
records both as `not_applicable`.

## The form

The view and the acquisition mode are nested under the cine loop answer; the other questions are global
classifications. The table is generated from
[`tools/labelbox/view_type_review.py`](../../tools/labelbox/view_type_review.py), which also writes
[`configs/radar/labelbox/view_type_review_v1.json`](../../configs/radar/labelbox/view_type_review_v1.json):

<!-- BEGIN GENERATED: view_type_form -->
| Question | Id | Required | Answers |
|---|---|---|---|
| What is this item? | `input_type` | yes | `cine_loop` Echo cine loop (moving echo images); `analysis_screen` Analysis or quantification screen (strain, 3D, EF tool); `no_echo_image` No echo image |
| Which view does it show? | `view` | yes, asked when `input_type` is `cine_loop` | `A2C` A2C: Apical 2-chamber view; `A2C_LV` A2C_LV: Apical 2-chamber view, LV-focused; `A2C_ZOOM` A2C_ZOOM: Apical 2-chamber view, zoomed; `A3C` A3C: Apical 3-chamber view; `A3C_LV` A3C_LV: Apical 3-chamber view, LV-focused; `A3C_ZOOM` A3C_ZOOM: Apical 3-chamber view, zoomed; `A4C` A4C: Apical 4-chamber view; `A4C_LV` A4C_LV: Apical 4-chamber view, LV-focused; `A4C_ZOOM` A4C_ZOOM: Apical 4-chamber view, zoomed; `A5C` A5C: Apical 5-chamber view; `A5C_ZOOM` A5C_ZOOM: Apical 5-chamber view, zoomed; `PLAX` PLAX: Parasternal long-axis view; `PLAX_DEEP` PLAX_DEEP: Parasternal long-axis view, deep; `PLAX_ZOOM` PLAX_ZOOM: Parasternal long-axis view, zoomed; `PSAX` PSAX: Parasternal short-axis view; `PSAX_AV` PSAX_AV: Parasternal short-axis view at the aortic valve; `RVINF` RVINF: Right ventricular inflow view; `SUBCOSTAL` SUBCOSTAL: Subcostal view; `SUPRASTERNAL` SUPRASTERNAL: Suprasternal view; `OTHER` OTHER: off-axis or non-standard echo view; `cannot_tell` Cannot tell the view |
| Acquisition mode | `modality` | yes, asked when `input_type` is `cine_loop` | `bmode` B-mode; `color_flow` Color Doppler; `spectral` Spectral Doppler; `m_mode` M-mode; `mixed` Mixed / analysis acquisition; `unresolved` Unresolved |
| Is identifying text visible (a name, a record number, a date of birth)? | `identifier_visible` | yes | `no` No; `yes` Yes: the item must be deleted |
| Comments (no patient details) | `comments` | no | free text |
<!-- END GENERATED: view_type_form -->

A review is valid when `review_problems` finds nothing: every required question that was asked is answered from the
form's choices, and the view and acquisition mode are answered exactly for a cine loop. Decode an export with
`decode_annotations` in `tools/labelbox/column_review.py` before checking it.

## Back into the column review

`route` turns a valid review into the `view`, `modality` and `input_kind` (`video`) arguments of the column review's
`build_candidate`, for a cine loop in a view the column review routes (not `OTHER`), in B-mode or color Doppler, with
no identifying text. Everything else stays out of the column review: analysis screens, items without an echo image,
spectral and M-mode loops, and loops whose view cannot be told.

## Selecting and uploading the items (not in this change)

- The pilot's items are the `OTHER` videos of its 2–3 studies, screened like the caption clips (still clips and
  visible measurements removed).
- The larger sample is drawn from training-split patients only, never from validation or test patients, and spread
  over studies, years and scanner models, with a cap per study.
- Upload only de-identified copies, as for the column review.
- The items are videos, so the form is created as a video ontology.

## Commands

```bash
python -m tools.labelbox.view_type_review          # write the form JSON and the table above
python -m tools.labelbox.view_type_review --check   # exit 1 if either is out of date (also a unit test)
```
