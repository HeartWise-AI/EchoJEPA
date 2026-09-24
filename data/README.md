# Dataset manifests

Two scripts turn the echo video metadata into one fixed, patient-level cohort that every
experiment reads: pretraining from scratch, continued pretraining from a checkpoint, and
frozen-backbone evaluation. The cohort is drawn once and then reused, never redrawn per
experiment.

| Stage | Script | Writes |
|---|---|---|
| 1 | `data/link_ef_labels.py` | One row per study with a verified Visual EF label |
| 2 | `data/build_manifests.py` | The cohort, its train/val/test split and the EchoJEPA manifests |

Both scripts read the same configuration, [`configs/data/manifests_tte_10k_ef.yaml`](../configs/data/manifests_tte_10k_ef.yaml).

## Generating the cohort

Inputs and outputs hold patient and study identifiers, so they must stay **outside this
repository** (see [Keeping patient data out of Git](#keeping-patient-data-out-of-git));
both scripts refuse an output location inside it.

```bash
# Stage 1: study-level EF labels.
python data/link_ef_labels.py --config configs/data/manifests_tte_10k_ef.yaml \
    --reports <report_export.csv> --pacs <pacs_index.csv> \
    --metadata <video_metadata.parquet> --out <labels_dir>/study_labels.parquet

# Stage 2, dry run: eligibility counts and consistency checks only. No video is
# decoded and nothing is written.
python data/build_manifests.py --config configs/data/manifests_tte_10k_ef.yaml \
    --metadata <video_metadata.parquet> --labels <labels_dir>/study_labels.parquet \
    --out-dir <output_dir> --dry-run

# Stage 2: select, verify, split and write the manifests.
python data/build_manifests.py --config configs/data/manifests_tte_10k_ef.yaml \
    --metadata <video_metadata.parquet> --labels <labels_dir>/study_labels.parquet \
    --out-dir <output_dir> --workers 32
```

Stage 2 decodes every frame of every video in the selected studies.

Other options of `build_manifests.py`:

| Option | Effect |
|---|---|
| `--seed N` | Overrides `selection.seed`; the value used is recorded in the outputs |
| `--workers N` | Processes that decode videos in parallel (default 32, or the CPU count if lower) |
| `--batch-size N` | Metadata rows read at a time (default 500,000) |
| `--overwrite` | Replace the manifests in a non-empty `--out-dir` |
| `--no-smoke-test` | Skip reading the manifests back with the EchoJEPA loader (which fails on an empty manifest) |

## Inputs

### Video metadata (`--metadata`, both stages)

One row per video, as `.parquet` or `.csv`. The `columns` block of the configuration maps
each role the scripts use to a column name, so another export needs only a new mapping:

| Role | Column (default config) | Meaning |
|---|---|---|
| `patient_id` | `mrn` | Patient identifier; the unit of the split |
| `study_id` | `study_uid` | Study instance UID; the unit of selection. Surrounding quotes are stripped |
| `exam_type` | `study_type` | Exam type, matched against `eligibility.exam_types` |
| `view` | `predicted_class` | Predicted view. Reported, never filtered: `OTHER` is kept |
| `video_path` | `avi_path` | Path of the video file; must not contain whitespace. A video with no path is excluded as `no_path` |
| `avi_status` | `avi_status` | Conversion status, matched against `eligibility.status_ok` |
| `study_date` | `date` | Optional, `YYYYMMDD`; used by `eligibility.years` and the reports |

A study must belong to one patient and one exam type, and a video path must not appear
under two different studies, patients, exam types or views; the scripts stop with an error
otherwise. A row that repeats another row with the same video path is ignored before anything is counted, so a duplicated export gives the
same counts. Rows without a video path (videos never converted) cannot be told apart, so
each is counted as a video.

### Report export (`--reports`, stage 1)

One row per report, with a patient ID, an accession number and the visually estimated EF
(`--report-patient-col`, `--report-accession-col`, `--report-label-col`; defaults
`Dossier`, `AccessionNumber`, `Visually Estimated EF`). A valid EF is a number in
(0, 100]; every other value is dropped and counted by reason (empty, invalid, zero,
negative, above 100).

### PACS index (`--pacs`, stage 1)

Maps accession numbers to studies: accession number, study instance UID, patient ID and
study date as `YYYYMMDD` (`--pacs-accession-col`, `--pacs-study-col`, `--pacs-patient-col`,
`--pacs-date-col`; defaults `AccessionNumber`, `StudyInstanceUID`, `PatientID`, `StudyDate`).

Stage 1 joins report → PACS index by accession number, then requires the study's patient ID
and date in the PACS index to equal those in the video metadata. Accessions or studies with
conflicting values are dropped and counted. It stops if report and PACS patient IDs agree
less often than `--min-id-agreement` (default 99.9%), or if any linked study disagrees with
the video metadata.

### Label table (`--labels`, stage 2)

Stage 1's output: one row per study, with the columns named in the `labels` block of the
configuration (`study_id`, `ef`). A label table from elsewhere gets the same check as the
report export: an EF that is not a number in (0, 100] is dropped and counted, and the
study counts as unlabelled.

## Configuration

| Block | Sets |
|---|---|
| `columns` | Role → column in the video metadata |
| `labels` | Columns of the label table. `low_threshold` (default 40, the reduced-EF cutoff) filters nothing: it adds each split's share of studies with an EF below it to the reports, to show whether reduced EF is balanced across splits; `null` leaves that column out |
| `eligibility` | Configure allowed exam types and statuses, whether a label is required, and an optional year filter |
| `selection` | Number of patients (one study each) and the seed |
| `split` | Patient fractions for train, val and test |
| `clip` | Clip sampling of the pretraining loader, used only for the `needs_padding` flag |

The configuration is checked when it is loaded, and a malformed one stops with a message
naming the problem. The split fractions must lie in [0, 1] and sum to 1. `n_patients`
must be a positive integer that gives every split with a positive fraction at least one
patient, and the seed a non-negative integer. `exam_types` and `status_ok` must be
non-empty lists, `require_label` true or false, `years` null or a non-empty list,
`low_threshold` null or a number in (0, 100], and `clip` must have a positive frame count
and fps. Each role maps to its own non-empty column name.

## Outputs

Written to `--out-dir`:

| File | Contents and use |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | EchoJEPA input: `<video_path> 0` per line, headerless and space-delimited, every valid video of the split's studies. The `0` is a placeholder label; nothing else is added to these files |
| `videos.csv` | One row per video in the manifests: `patient_id`, `study_id`, `exam_type`, `view`, `video_path`, `avi_status`, `study_date`, `label` (EF), `split`, `n_frames` (decoded), `fps` (empty when the decoder reports no usable frame rate), `needs_padding` |
| `studies.csv` | One row per selected study: split, patient, exam type, date, EF, videos kept and excluded |
| `excluded_videos.csv` | Videos of the selected studies that failed verification, with the reason: `missing`, `empty`, `undecodable`, `no_frames`, `corrupt_frames`, `whitespace_in_path` or `no_path` |
| `manifest_info.json` | Complete provenance record for the run, including the command line, input paths and checksums, resolved configuration and seed, Git commit, counts, split distributions, and output checksums. Because it contains data paths, store it alongside the protected data rather than in the repository |
| `summary.md` | Create a shareable summary containing the same aggregate statistics as the full run record, but remove anything that could reveal file locations, patient/study identities, or the exact command used |

`needs_padding` is true when a video is shorter than one clip of the pretraining loader:
`clip.frames_per_clip` frames taken every max(1, ceil(video fps) // `clip.fps`) frames, or
every frame (the fallback frame step of 1) when `fps` is empty. Such videos are kept; the
loader pads them.

## Using the manifests in EchoJEPA

A pretraining config, from scratch or from a checkpoint, lists **only `train.csv`**:

```yaml
data:
  dataset_type: VideoDataset
  datasets:
  - <output_dir>/train.csv
  datasets_weights:
  - 1.0
  dataset_fpcs:
  - 16    # keep equal to clip.frames_per_clip
  fps: 8  # keep equal to clip.fps
```

`val.csv` and `test.csv` hold the held-out patients and are referenced the same way, but
only in a config of their own, never next to `train.csv`:

```yaml
data:
  dataset_type: VideoDataset
  datasets:
  - <output_dir>/val.csv  # or <output_dir>/test.csv
```

Every file under `data.datasets` is training data: the trainer merges them into one
dataset, so listing `val.csv` or `test.csv` with `train.csv` would pretrain on the
held-out patients.

The Visual EF probe needs label files with the EF of each video. Build them from
`videos.csv` (column `label`), not from `val.csv` or `test.csv`, whose label is the
placeholder `0`.

## Reusing the frozen manifests

- Generate the cohort once, then point every experiment's config at the same
  `<output_dir>`. Do not regenerate it per experiment.
- `build_manifests.py` refuses a non-empty `--out-dir` unless given `--overwrite`, so an
  existing cohort is not replaced by accident.
- To confirm an experiment reads the frozen files, compare `sha256sum <output_dir>/train.csv`
  with the checksum in `manifest_info.json` or `summary.md`.
- A different seed, configuration or metadata snapshot is a different cohort: write it to a
  new directory and record which directory each experiment used.

## Keeping patient data out of Git

The source exports (video metadata, report export, PACS index) and everything the scripts
write (study labels, manifests, `manifest_info.json`) hold patient and study identifiers.
Keep all of them outside this repository and never commit them:

- Both scripts refuse an output location inside the repository.
- `.gitignore` ignores `*.csv` and `*.parquet` everywhere in the repository, as a safety
  net for a file copied in by mistake. A CSV that genuinely belongs in the repository has
  to be added deliberately with `git add -f`.
- Only `summary.md` is meant to be shared, for example attached to an issue.

## Tests

```bash
python -m unittest tests.data.test_build_manifests
```

The tests use synthetic metadata and tiny generated videos only. They also check that no
data path or study UID appears in the pipeline's files and that a run writes nothing into
the repository.
