# Medical approval of `radar-caption-ontology-v1`

The `status` of the record at the end of this document is the approval state of this version, and the approved files
do not repeat it, so approving them changes none of them. No clinical data may be uploaded for review under this
ontology until the record is `approved`. `python -m app.radar.approval` reports the state, and a unit test fails if
the record is malformed or if an approved file changes afterwards.

## What is approved

The exact content of these files at the content commit:

- `configs/radar/radar-caption-ontology-v1.yaml`: views, structures, concepts, review form;
- `configs/radar/label_mappings_v1.yaml`: DeepECHO, EchoPrime and PanEcho mappings;
- `configs/radar/review_examples_v1.yaml`: the worked examples;
- `configs/radar/labelbox/radar-caption-ontology-v1.json`: the Labelbox form;
- `docs/radar/ontology_v1.md`, including its handwritten rules;
- `docs/radar/labeling_instructions_v1.md` and `docs/radar/labeling_instructions_v1.fr.md`;
- `docs/radar/pilot_metrics_v1.md`, with the agreed thresholds.

## Procedure

1. Agreement on the pilot thresholds in a GitHub comment.
2. The Labelbox SDK round trip runs once, not skipped (see `tools/labelbox/tests/test_review_ontology.py`).
3. The commit that holds the final versions of the files above is the **content commit**. This record is still
   `pending` in it, so the record never has to name its own commit.
4. Robert reviews the content commit and records his medical approval on GitHub (a pull request review or a comment
   on issue #21 that names the commit).
5. The next commit fills the record: status `approved`, the content commit, the hashes from
   `python -m app.radar.approval --hashes`, Robert's GitHub login (`robertavram-md`) as reviewer, both links, both
   dates and the SDK version. Whoever reviews that commit opens the medical approval link and checks that Robert wrote
   it and that it names the content commit: the verifier checks files, not who wrote a link.
6. The verifier then checks that the content commit is an ancestor of HEAD and that every approved file is identical
   in that commit, in the record and in the working tree. It fails closed when git or the commit is unavailable, so
   CI checks out the full history, and the pull request must be merged with a merge commit (not squashed or rebased)
   for the content commit to stay in the history.
7. From then on, any change to an approved file fails the tests, even if its recorded hash is refreshed. A change
   needs version 2: new files, a new record `approval_v2.md` and a new approval. Until then version 2 is
   unapproved.

## Record

```yaml
schema: radar-caption-approval
ontology: radar-caption-ontology-v1
status: pending
content_commit: null
files: {}
medical_approval: {reviewer: null, link: null, date: null}
thresholds_agreement: {link: null, date: null}
labelbox_sdk_check: {version: null}
```
