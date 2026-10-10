# Labelbox RADAR caption pilot

The current proposed workflow is **DeepECHO column review v2**:
[instructions](../../docs/radar/column_review_v2.md),
[native ontology](../../configs/radar/labelbox/column_review_v2.json), and
[current selection](../../configs/radar/labelbox/current_review.json).
It prefills report values, including explicit normals, and a draft caption. Most items need one final
confirmation; exceptions use per-column dropdowns for corrections, omissions and error reasons.

The v2 builder generates the native schema and MAL drafts. The separate v2 importer previews or applies
installation into an explicitly selected compatible video project with existing deidentified data rows.
The resource bootstrap below remains separate.

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

1. Review the v2 column ontology and complete its clinical/input checks.
2. Import a 3–5-study synthetic or approved de-identified pilot.
3. Use the v2 MAL importer to preload values and captions for human confirmation.
4. Export annotations with stable video and study identifiers.
5. Add agreement, coverage, and failure-analysis reports.

Do not add these behaviors to `create_pilot_project.py`; keeping provisioning separate
makes the potentially mutating data workflow easier to review and audit.

## Tests

The tests use fake SDK clients and never require a real key or make external calls:

```bash
python -m unittest discover -s tools/labelbox/tests -p 'test_*.py' -v
```
