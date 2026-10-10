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

## Test the review workflow with synthetic videos

`synthetic_pipeline.py` exercises the simple five-claim review form without using
clinical data. It creates fixed, clearly marked sandbox resources, generates four
small synthetic videos, uploads safe review context, and creates one labeling batch.
The cases cover a supported finding, an unsupported finding, a finding outside the
model crop, and a modality mismatch.

The view question is a single-choice list of the approved RADAR views. Each synthetic
row imports its candidate view as a Labelbox model-assisted pre-label immediately after
the batch adds the row to the project, so the reviewer can confirm it without changing
the answer or directly select the correct view.

The command accepts no media, report, metadata, or storage-path arguments. Its
default mode is a local preview that makes no Labelbox request:

```bash
uv run --frozen --group labelbox python -m tools.labelbox.synthetic_pipeline
```

To create or reuse the synthetic sandbox and upload only missing rows:

```bash
read -rsp "Labelbox API key: " LABELBOX_API_KEY && echo
export LABELBOX_API_KEY

uv run --frozen --group labelbox python -m tools.labelbox.synthetic_pipeline --apply

unset LABELBOX_API_KEY
```

After the synthetic rows have been reviewed in Labelbox, export a de-identified
scoring summary. The output contains only reserved synthetic global keys, review
status, and computed scores:

```bash
read -rsp "Labelbox API key: " LABELBOX_API_KEY && echo
export LABELBOX_API_KEY

uv run --frozen --group labelbox python -m tools.labelbox.synthetic_pipeline \
  --export-summary /tmp/radar-synthetic-review-summary.json

unset LABELBOX_API_KEY
```

This sandbox does not modify the existing `RADAR Caption Pilot` project or its
dataset. Successful synthetic execution is a pipeline check, not approval to upload
clinical data.

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
