# RADAR pilot: view-specific captions, review workflow and training skeleton

RADAR pairs echo clips with short, view-specific captions and trains a video-text contrastive model on top of the
EchoJEPA encoder. This document covers the pieces prepared for the 3–5-study annotation pilot (handoff §9). The
pilot tests whether the workflow is feasible and helps refine the ontology and prompts. It does not measure clinical
accuracy or model generalization.

Status: the code paths are implemented and tested on synthetic data only. No real report has been captioned, no
review has been collected and no RADAR model has been trained. Several dependencies are still missing (see
[Outstanding dependencies](#outstanding-dependencies)).

## Pieces

| Piece | Location |
|---|---|
| Concept ontology v0 (49 concepts) | `configs/radar/concepts_v0.yaml`, loader/validator `app/radar/ontology.py` |
| Clip record schema and JSONL I/O | `app/radar/records.py` |
| Report-finding extraction | `app/radar/findings.py` |
| Candidate captions (rule-based, LLM contract) | `app/radar/captions.py` |
| Coverage ledger | `app/radar/coverage.py` |
| Labelbox review form, payloads, export import | `tools/labelbox/{review_ontology,payloads,review_export,radar_review}.py` |
| Training skeleton | `app/radar/{model,data,loss,train}.py`, `configs/radar/skeleton-vitb16.yaml` |
| Synthetic demo and expected outputs | `configs/radar/demo/`, `python -m app.radar.demo` |

## Ontology (`concepts_v0@0.1.0`)

There are 12 structure-visibility concepts and 37 findings. They are built from the PanEcho tasks, the EchoPrime
report sections and the ICM report-export fields. Every PanEcho or EchoPrime task a concept names is checked to
exist in those task files. Each concept has:

- an id;
- `fr`/`en` labels;
- a structure;
- a `kind` (`structure_visibility` or `finding`);
- a value: boolean, ordinal on a named severity scale, categorical, or numeric with a unit and physiologic bounds;
- report sources: structured columns, plus free-text fields with phrase hints and acronyms;
- `potentially_visible_in`, the views × acquisition categories where the concept may be seen;
- `required_modality` and `evidence` (`visual` or `measurement`);
- an optional `requires_views_from` (for example, wall motion needs apical 4-, 2- and 3-chamber views);
- related PanEcho and EchoPrime tasks;
- notes.

Visibility and disease are separate concepts. `mitral_valve_visible` says nothing about
`mitral_regurgitation`.

`Ontology.relevance(concept, view, acquisition)` turns the prior into one explicit outcome:

| Outcome | Meaning | In a caption? |
|---|---|---|
| `relevant` | view and modality can show it, judged visually | yes (as an unvalidated candidate) |
| `modality_absent` | the view fits but the clip's modality cannot show it (e.g. RVSP in a color loop) | no, listed as a gap |
| `measurement_required` | needs calipers, tracing or spectral measurement (TAPSE, TR ERO, RVSP …) | no, listed as a gap |
| `acquisition_unknown` | the clip's acquisition category is `unknown`/`other` | no, listed as a gap |
| `not_in_view` | outside the view's prior | no, listed as out of view |

The prior only says what a clip might show. It is not evidence that the finding is visible in a particular video.

**Report encodings.** The grade columns (`Referring MR/AR/TR/PR Grade`) mix several encodings: 0, 0.5, 1, 2, 3,
words such as "Trace", and date-like strings. Version 0 maps only word grades. Every numeric code, and every date-like
or unparseable value, is reported as an `ExtractionIssue` (`unmapped_value` / `date_like_value`). None of them is
read as "none" or zero. A numeric 0 (for example, RVSP) falls outside the physiologic bounds and is reported as
`out_of_range`. The digits in the WMS columns assume the standard 1–5 wall-motion score, which has not been verified
yet. Check every alias table against each column's real value distribution before trusting extraction. Add numeric
grade mappings only under a new ontology version.

Bump `version` whenever ids, values, scales or priors change. Captions, reviews, the review form and checkpoints all
record the version, and records reject captions built under another one.

## Clip records (`radar-clip-record/0.1`)

There is one JSON object per clip, stored in JSONL **outside the repository**. Decoding is strict: unknown or
missing fields are errors.

- `record_id`: pseudonymous `rc-<hex>`, computed as an HMAC of the video key (`make_record_id`) with a secret kept
  outside Git. It is the only key used in Labelbox, logs and checkpoints.
- `study_key`, `patient_key`: local grouping keys. They feed the loss mask and the ledger only. They are never model
  input or caption text, and validation rejects captions that contain them.
- `split` (`train`/`val`/`test`): the patient's split in the shared archive patient split, with that table's SHA-256
  as `split_source` (`build_pilot_records.py --patient-splits`); `video_path` (local), `view`, `view_confidence`,
  `acquisition` (category, confidence, rule, `has_*` flags from the acquisition index, under the pretraining B-mode
  rule: a header B-mode from a vendor outside `acquisition.bmode_vendors` is `unknown`, rule
  `bmode_vendor_unvalidated`, its `has_color_flow` null), and `focus` (nullable zoom/crop information).
- `report_findings`: `concept_id`, `value`, `severity` (ordinal concepts only), `unit`, and `provenance`.
  Provenance holds the source column, the method (`structured_column`, `phrase_hint`, `manual` or `synthetic`), the
  extractor version, and a character span plus the matched hint. It never holds the report text itself.
- `candidate_caption`: `text`, `language`, `generator`, `generator_version`, `prompt_version`, `ontology_version`,
  `created_utc`, and `status`, which is always `unvalidated`. It also holds the stated `claims`, the explicit
  `gaps`, the `out_of_view` concepts and the `expected_structures` (the visibility prior shown to reviewers). A
  candidate can claim only report findings, with their reported values, that are `relevant` to the clip.
- `review`: `status` (`unreviewed`, `in_review` or `reviewed`), a pseudonymous `reviewer_id`, `review_version`,
  `source`, `label_ref`, timestamps, `satisfied`, `missing_concepts`, `unsupported_concepts`, `wrong_values`,
  `uncertain` and `unassessable` (each with a quality, crop, modality or other reason),
  `clip_unassessable_reason`, `corrected_caption` and `comments`.
- `approved_caption`: produced only by `apply_review` from returned review data. The candidate is approved as-is
  only if the reviewer was satisfied and flagged nothing (no missing, unsupported, wrong, uncertain or unassessable
  concept). A corrected caption is the reviewer's resolution of their flags: it is approved, and every concept
  marked unsupported, uncertain or unassessable leaves its concept list (the form asks reviewers to edit those out
  of the text). A "corrected" caption identical to the candidate (the form preloads it) is no correction. A flag
  without a correction, or an unassessable clip, gives no approved caption. Validation re-derives the approved
  caption and rejects any record whose `approved_caption` differs from it in any field, concept values included.

## Candidate captions

Rule-based captions are deterministic. They are built from structured findings and ontology labels, so no report
text is ever copied. For example:

> Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : modérée. Insuffisance tricuspide : légère.

For the same study, this color clip lists LV function and EF as `modality_absent`, RVSP and TR velocity as
`modality_absent` (spectral), and TR ERO as `measurement_required`. The default language is French, matching the
reports and the reviewers. English is also available.

`build_llm_prompt` prepares a versioned request (`radar-caption-prompt/0.1`) for a future authorized endpoint. The
request contains only the allowed structured findings and the gaps, with no identifiers. It refuses test-split
records unless explicitly allowed for the frozen final evaluation. `parse_llm_output` enforces the strict JSON schema
`radar-llm-caption/0.1` and rejects any of the following:

- invalid JSON or wrapped JSON;
- extra keys;
- unknown concepts;
- values outside the ontology or different from the report;
- claims about gaps or out-of-view concepts;
- repeated claims;
- identifiers or long numbers.

Gaps always come from the deterministic plan. `configs/radar/demo/demo_llm_outputs.json` holds 11 synthetic responses
(2 accepted, 9 rejected).

```bash
python -m app.radar.demo                       # captions, a prompt, LLM parse outcomes, coverage on the demo
python -m app.radar.captions --records IN.jsonl --out OUT.jsonl   # add rule-based candidates (outside the repo)
python -m app.radar.coverage --records IN.jsonl --out-dir DIR     # ledger.json (local keys) + summary.json
```

## Coverage ledger

For each study and report finding, the ledger records:

- the candidate clips (prior);
- the clips a completed review supports, does not support, or is uncertain about;
- for multi-view findings, the view groups covered and missing;
- when no clip can ground the finding, an explicit gap with its reason (`modality_absent`, `measurement_required`,
  `acquisition_unknown` or `no_relevant_view`).

A finding's status is `supported`, `candidate`, `not_supported_after_review` or `gap`. Findings whose values differ
across a study's clips are flagged. The summary contains aggregate counts only.

## Labelbox review

See `tools/labelbox/README.md` ("RADAR caption review"). The review form is built from the ontology. Payloads are
local NDJSON files: data rows keyed by `record_id`, a placeholder video URL on the reserved `.invalid` domain, and
caption prelabels. Every payload passes a guard that refuses identifier field names, local paths, file names, DICOM
UIDs, dates, e-mail addresses, long digit runs and any record's local keys. The export importer maps returned labels
back onto records. Nothing is uploaded.

## Training skeleton

```bash
python -m app.radar.train --fname <copy of configs/radar/skeleton-vitb16.yaml with /your_ paths replaced>
```

- **Video encoder:** V-JEPA 2.1 ViT-B, built and strictly loaded exactly like the frozen EF probe (same options and
  prefix handling), from the published EchoJEPA checkpoint. Checked on this machine: `target_encoder` (epoch 60,
  width 768) loads strictly. The encoder is frozen by default. Pooling is attentive or mean, over non-padding tubelets
  only, followed by a linear or MLP projection. Switching to an ICM checkpoint means changing `visual.checkpoint`.
  Width, depth, patch and tubelet size are checked before loading; for example, a ViT-L config on the ViT-B checkpoint
  fails with `embed_dim: checkpoint 768 vs config 1024, depth: checkpoint 12 vs config 24`.
- **Text encoder:** either `huggingface` or `tiny`.
  - `huggingface` loads a local snapshot only (`local_files_only: true`). No model is hard-coded, because the reports
    are French; choose a French or multilingual encoder. With `text.freeze: true` it stays in evaluation mode (no
    dropout) while the rest of the model trains, as the frozen video encoder does.
  - `tiny` is a random-init transformer with a hashing tokenizer, for tests and smoke runs only.
- **Data:** JSONL records, with clips sampled by `VideoDataset` (same fps, padding policy and clip indices as
  pretraining). Training and validation load in `data.num_workers` worker processes with `data.loader_timeout`
  (default 900 s), so a stalled read ends the run, resumable from `latest.pt`, instead of hanging it; with no
  workers PyTorch cannot time loading out, and `provenance.json` (`loader`) records the effective timeout. Training uses approved captions only, unless `meta.dev_allow_unreviewed_candidates` is set; that
  flag and the count of unvalidated captions are recorded in provenance.
- **Split guard:** a training records file that contains any test-split record is refused. A records file must keep
  each patient and each study in one split and come from one split table (`split_source`); reading, writing and
  selecting records all check this over the whole file. Validation needs records bound to the shared archive split
  (`split_source` set). Validation-split uses are logged in `provenance.json` (`split_uses`).
- **Loss:** symmetric multi-positive InfoNCE with a learnable, clamped temperature. Each pair in the batch is marked
  positive, negative or ignored:
  - positive: the pair itself and identical normalized captions;
  - ignored by default: same-study pairs and equal concept sets (configurable to positive);
  - ignored pairs are left out of the denominator.
  Same-study and semantically equivalent pairs are never negatives.
- **Run outputs:** each run writes the following to `meta.output_dir` (refused inside the repository):
  - `latest.pt`: atomic; trained parameters only (frozen encoders are referenced by SHA-256), plus the optimizer,
    step and the CPU, CUDA, numpy and Python RNG states;
  - `provenance.json`: config hash, git commit and dirty flag, encoder checkpoint SHA-256, text-model files SHA-256
    (weights, configuration and tokenizer of the snapshot), records and patient-split SHA-256, ontology tag,
    text-encoder identity, dev flags and split uses;
  - `log.jsonl`: per-step metrics, with no identifiers.

  A resume continues from `latest.pt` only if the config, encoder checkpoint, text-model files, records and patient
  split are byte-identical. The tests check that an interrupted and resumed run equals an uninterrupted one bit for
  bit, with text dropout and validation on, interrupted mid-epoch or at an epoch boundary.

The skeleton runs on a single process. Distributed all-gather negatives, augmentation, retrieval evaluation and a
validated recipe are not implemented.

## Tests

```bash
python -m unittest discover -s tests/radar -p 'test_*.py'
python -m unittest discover -s tools/labelbox/tests -p 'test_*.py'
```

All tests are synthetic and CPU-only. They cover:

- ontology validation;
- record validation and approval rules;
- grade-encoding handling;
- caption filtering by view and acquisition, gaps, and no report-text copying;
- the LLM parser;
- the coverage ledger;
- the Labelbox form, guard and export parsing;
- mask construction and hand-computed loss values;
- a forward/backward step with `vit_tiny` and the tiny text encoder;
- checkpoint resume equivalence and provenance.

## Outstanding dependencies

- **Labelbox:** an approved workspace and storage configuration, a project API key (Jacques) and an approved
  de-identified video location to replace the placeholder URL. Uploading is deliberately not implemented.
- **LLM:** an established, authorized endpoint for clinical text. Until then only the prompt, schema, parser and
  synthetic demonstrations exist.
- **Text encoder:** a French-capable encoder must be chosen and its snapshot stored locally.
- **Report export:** the real column encodings (grade codes, WMS columns, free-text phrasing) must be verified, and
  the alias tables and hints in the ontology updated under a new version.
- **Pilot records:** the 3–5 pilot studies must be selected (training patients only), and clip records built from the
  metadata, acquisition index, patient splits and report export outside the repository.
- **Clinical review:** no caption is approved until cardiologist review data comes back.
