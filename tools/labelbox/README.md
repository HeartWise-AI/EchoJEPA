# Labelbox RADAR caption pilot

This folder bootstraps the empty Labelbox workspace for the view-specific RADAR
caption pilot. It gives Charlotte a stable place to add the ontology, data import,
caption preloading, export, and evaluation tools after access and governance are
approved.

## What the setup script does

`create_pilot_project.py` ensures that the current Labelbox organization contains:

- a video project named `RADAR Caption Pilot`; and
- a dataset named `RADAR Caption Pilot Dataset`.

It searches for both exact names before writing. An existing match is reused only if
its description identifies this bootstrap, the project is an empty video project,
and the dataset is empty with no IAM storage integration. Missing resources are
created. Duplicate or incompatible matches stop the command before it creates
anything. Approved storage can be connected later.

The script does **not** invite users, assign roles, define an ontology, connect cloud
storage, upload videos, preload captions, or create labeling batches. Complete access
provisioning in the Labelbox administration interface: Charlotte should have Project
Lead access to this project, Robert should have Reviewer access, and organization
administration should remain with Jacques.

## Install

From the repository root, install the locked optional dependency group:

```bash
uv sync --frozen --group labelbox
```

The Labelbox SDK is kept out of the default EchoJEPA training dependencies.

## Preview without changing Labelbox

The default mode only prints the plan. It does not need a key and makes no Labelbox
request:

```bash
uv run --frozen --group labelbox python tools/labelbox/create_pilot_project.py
```

Expected output:

```text
Dry run: no Labelbox requests were made.
Would ensure video project: 'RADAR Caption Pilot'
Would ensure dataset: 'RADAR Caption Pilot Dataset'
Run again with --apply after setting LABELBOX_API_KEY.
```

## Create or reuse the live resources

Use a personal, revocable key belonging to an account allowed to create projects and
datasets. Do not paste the key into chat, GitHub, Slack, Notion, source files, command
arguments, logs, screenshots, or W&B.

Read the key silently so it does not enter shell history, apply the setup, and remove
it from the shell immediately afterward:

```bash
read -rsp "Labelbox API key: " LABELBOX_API_KEY && echo
export LABELBOX_API_KEY

uv run --frozen --group labelbox python tools/labelbox/create_pilot_project.py --apply

unset LABELBOX_API_KEY
```

A successful run prints only safe names, actions, and resource identifiers:

```text
project: created; name='RADAR Caption Pilot'; id=<project-id>
dataset: created; name='RADAR Caption Pilot Dataset'; id=<dataset-id>
```

Running the command again reuses those identifiers rather than creating duplicates.
If an API operation fails, the command returns a non-zero status and prints only the
exception type. It intentionally omits the server message because that message could
contain credentials, storage locations, or other sensitive details.

An `AmbiguousResourceError` or `IncompatibleResourceError` means the same-name
resources must be inspected in Labelbox. Rename or remove resources belonging to
another workflow; do not delete or repurpose them without confirming ownership.

Custom non-sensitive names can be supplied when a separate sandbox is required:

```bash
uv run --frozen --group labelbox python tools/labelbox/create_pilot_project.py \
  --project-name "RADAR Caption Pilot Sandbox" \
  --dataset-name "RADAR Caption Pilot Sandbox Dataset" \
  --apply
```

## Data and credential safety

- Use synthetic or fully de-identified test clips until the workspace and storage
  configuration are explicitly approved for the intended data classification.
- Do not upload patient names, accession numbers, raw reports, local filesystem paths,
  or other identifiers.
- Keep `LABELBOX_API_KEY` in an approved secret store or a temporary environment
  variable. Local `.env` files are ignored as defense in depth, but they are not the
  preferred long-term secret store.
- Revoke and rotate a key immediately if it is exposed.
- Confirm the target Labelbox organization before applying the setup. The resources
  are created in the organization associated with the supplied key.

## Charlotte's next extensions

Build follow-up work in this folder as separate reviewed changes:

1. Define and version the view-specific RADAR caption ontology.
2. Import a 3–5-study synthetic or approved de-identified pilot.
3. Preload generated captions for human correction.
4. Export annotations with stable video and study identifiers.
5. Add agreement, coverage, and failure-analysis reports.

Do not add these behaviors to `create_pilot_project.py`; keeping provisioning separate
makes the potentially mutating data workflow easier to review and audit.

## Tests

The tests use fake SDK clients and never require a real key or make external calls:

```bash
python -m unittest discover -s tools/labelbox/tests -p 'test_*.py' -v
```

## RADAR caption review (local workflow)

These modules prepare the review without contacting Labelbox. They are separate from
`create_pilot_project.py`, whose behavior is unchanged. Background, schemas and the
ontology are described in [`docs/radar_pilot.md`](../../docs/radar_pilot.md).

| Module | Purpose |
|---|---|
| `review_ontology.py` | The review form, built from `configs/radar/concepts_v0.yaml`. It mirrors the SDK's `OntologyBuilder.asdict()` layout. |
| `payloads.py` | Data rows and caption prelabels as local NDJSON, plus the safety guard. |
| `review_export.py` | Maps a project export back onto clip records. |
| `radar_review.py` | Command line for the three steps. |

The form asks one global question per item:

- whether the caption is satisfactory;
- whether the clip can be assessed (image quality, crop, modality);
- which visible, relevant concepts are missing;
- which claimed concepts are unsupported;
- which claimed values are wrong (with a nested corrected value);
- which concepts are uncertain or unassessable (each with a nested reason);
- the corrected caption (preloaded with the candidate);
- comments.

Visibility concepts ("mitral valve visible") and findings ("mitral regurgitation") are
separate options. The review version is `radar-review/0.2+concepts_v0@<version>` and is
stamped on every imported review.

```bash
# The form (no identifiers; safe to keep anywhere).
python -m tools.labelbox.radar_review ontology --out /path/outside/repo/review_form.json

# Data rows + caption prelabels. Dry run by default: builds, checks and prints counts only.
python -m tools.labelbox.radar_review payload --records /path/outside/repo/records.jsonl
python -m tools.labelbox.radar_review payload --records /path/outside/repo/records.jsonl \
  --out-dir /path/outside/repo/labelbox_payload

# Map a Labelbox export back onto the records (approved captions only from returned reviews).
python -m tools.labelbox.radar_review import-export --records /path/outside/repo/records.jsonl \
  --export /path/outside/repo/export.ndjson --review-version "radar-review/0.2+concepts_v0@0.1.0" \
  --reviewer-map /path/outside/repo/reviewers.json --out /path/outside/repo/records.reviewed.jsonl
```

Each data row holds:

- `global_key`: the pseudonymous record id;
- `row_data`: a video URL from `--video-url-template`. The default is a placeholder on
  the reserved `.invalid` domain; replace it with an approved de-identified location
  before any upload.
- text attachments: the candidate caption, the view and acquisition, and the findings
  this clip cannot ground (concept names and reasons only; the report's values for them
  stay local).

Training-split records are the only ones accepted unless `--splits` says otherwise.

The guard refuses a payload containing any of:

- identifier-like field names;
- local paths or file names;
- DICOM UIDs or dates;
- e-mail addresses;
- runs of seven or more digits;
- any record's study key, patient key or video path.

Its error messages give the JSON location and the rule, never the value. Commands that
write captions or identifiers refuse output paths inside the repository.

The importer refuses any of the following:

- an unknown global key;
- a feature the form does not define;
- more than one label per data row;
- a Done label missing the required answers;
- a review-version mismatch.

Reviewer e-mail addresses are replaced with handles from `--reviewer-map`, or with a
hash prefix when no map is given.

Uploading is not implemented. It needs an approved workspace, approved de-identified
video storage and a reviewed upload change (`Dataset.create_data_rows` for the rows,
model-assisted labeling import for the prelabels).

```bash
python -m unittest discover -s tools/labelbox/tests -p 'test_*.py' -v
```
