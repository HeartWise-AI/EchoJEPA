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
| `--no-smoke-test` | Skip the smoke test below |

After the manifest files are written, the code performs a small end-to-end sanity check using the real EchoJEPA data loader, to make sure the generated manifests can actually be used for pretraining.

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
| `study_date` | `date` | `YYYYMMDD`. Required by stage 1, which checks each linked study's date against it; optional for stage 2, which uses it for `eligibility.years` and the reports |

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
less often than `--min-id-agreement` (default 99.9%). PACS-to-metadata failures are also
dropped and counted when their combined patient/date agreement remains above that threshold;
the run stops when it falls below the threshold.

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
| `clip` | Clip sampling of the pretraining loader, used for the `needs_padding` flag and by the smoke test; no video is excluded for it |

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
  dataset_fpcs:
  - 16    # keep equal to clip.frames_per_clip
  fps: 8  # keep equal to clip.fps
```

Leave out `datasets_weights`: with it, the loader's weighted sampler draws videos with
replacement, so a pass over the data repeats some videos and skips others.

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
placeholder `0`. `make_probe_manifests.py` does this for chosen views:

```bash
python data/make_probe_manifests.py --manifests <output_dir> --out-dir <probe_dir> --views A4C
```

It writes `train.csv`, `val.csv` and `test.csv` with one row per video, `<video_path>
<z-scored EF>`, z-scored with the train split's mean and standard deviation. Those two
values, saved in `probe_info.json`, are the probe config's `data.target_mean` and
`data.target_std`. Videos with missing EF or EF outside (0, 100] are excluded and counted by split in `probe_info.json` under `excluded`. The script stops if any patient appears in more than one split. `video_index.csv` records each video's path, split, study, and patient. The probe reads it through `data.video_index` for study-level metrics. Since it lists paths and identifiers, it should be stored with the data, like the manifests.

The probe trains on `train.csv` and picks its epoch and head on `val.csv`. `test.csv` is
read once, after training, by `python -m evals.main --fname <probe config> --test_only`: it
loads the best validation checkpoint from `best.pt`, scores each test video exactly once,
and writes `test_metrics.json` and `test_predictions.csv` (one row per video). The protocol,
the metrics and how to reproduce a run are in
[`docs/visual_ef_probe.md`](../docs/visual_ef_probe.md).

## Reusing the frozen manifests

- Generate the cohort once, then point every experiment's config at the same
  `<output_dir>`. Do not regenerate it per experiment.
- `build_manifests.py` refuses a non-empty `--out-dir` unless given `--overwrite`, so an
  existing cohort is not replaced by accident.
- To confirm an experiment reads the frozen files, compare `sha256sum <output_dir>/train.csv`
  with the checksum in `manifest_info.json` or `summary.md`.
- A different seed, configuration or metadata snapshot is a different cohort: write it to a
  new directory and record which directory each experiment used.

## Archive mode: every study, anchored patient split

[`configs/data/manifests_tte_archive.yaml`](../configs/data/manifests_tte_archive.yaml) sets
`selection.mode: all_studies` for full-archive pretraining. It keeps **every eligible study of
every patient** (longitudinal studies included), requires no EF label, and builds two training
arms from one master index. The default mode, `one_study_per_patient`, is unchanged.

```bash
# 1. Freeze the patient split early: eligibility and assignment only, nothing is decoded.
python data/build_manifests.py --config configs/data/manifests_tte_archive.yaml \
    --metadata <video_metadata.parquet> --anchor-splits <10k_output_dir>/studies.csv \
    --out-dir <split_dir> --assign-only

# 2. Benchmark verification: decode 2,000 pending videos, report the rate and the time left,
#    keep the results in the cache, write no manifest (`--verify-limit 0`: cache coverage only).
python data/build_manifests.py --config configs/data/manifests_tte_archive.yaml \
    --metadata <video_metadata.parquet> --anchor-splits <split_dir>/patient_splits.csv \
    --verify-cache <cache_dir> --out-dir <archive_dir> --workers 32 --verify-limit 2000

# 3. Full build: verify (resumable), join the acquisition index, write the arms and smoke-test them.
python data/build_manifests.py --config configs/data/manifests_tte_archive.yaml \
    --metadata <video_metadata.parquet> --anchor-splits <split_dir>/patient_splits.csv \
    --acquisition-index <acquisition_index> --verify-cache <cache_dir> \
    --out-dir <archive_dir> --workers 32
```

The acquisition index must list every eligible video of the verified splits, from the current scanner version
(`SCANNER_VERSION`), with a successful header read; the build checks this before it verifies anything and stops
otherwise, because a partial, stale or failed index would silently shrink `train_bmode.csv`. Failed reads that
persist after `scan_acquisition.py scan ... --retry-errors` can be admitted, once reviewed, with
`--accept-acquisition-read-errors N` (they stay out of `train_bmode.csv`). `--allow-partial-acquisition` admits
any index for development builds only. `manifest_info.json` (`counts.acquisition`: `coverage_complete`,
`stale_scanner_version`, `read_errors`, `ready`) and `summary.md` record all of it. Videos read successfully with
an unknown mode are results, not errors: they stay out of `train_bmode.csv`.

A build is written and smoke-tested in a staging folder next to `--out-dir`, then gets `outputs.sha256` (every
file, `sha256sum -c` format) and, last, the receipt `BUILD_COMPLETE.json`, and only then is renamed to `--out-dir`.
The smoke test of each manifest runs in a supervised worker process with a deadline (`--smoke-timeout`, default
900 s, one retry), so a stalled decoder or network read fails the build instead of hanging it. A failure at any
step leaves `--out-dir` as it was (the staging folder stays for inspection); if the final rename fails after an
earlier build was moved aside, that build is moved back. An earlier build replaced with `--overwrite` is kept as
`<out-dir>.replaced-<time>`. Only an abrupt end of the process between those two renames leaves `--out-dir`
missing; the newest `<out-dir>.replaced-*` is then the earlier build, and the next run says so. Consumers should
require the receipt and check the checksums (the production driver does).

`--dry-run` reads the metadata and the anchor and prints the split counts, writing nothing.

### Patient split

- `--anchor-splits` is required: the 10k build's `studies.csv` (its patients get origin
  `legacy`) or an earlier archive build's `patient_splits.csv` (origins carried over).
  Identifiers are read as text, so leading zeros survive. A patient with two splits stops the build.
- Every anchored patient keeps its split. Targets are exact patient counts over **all** eligible
  patients (70/10/20 by default); the anchored patients present fill part of each target, and
  the patients new to the anchor fill the rest in a seeded order of their sorted ids (a random
  stream of its own, independent of the one-study stages). Video fractions are only
  approximately proportional.
- Splits are assigned from metadata eligibility, before any video is decoded. A study or patient
  whose videos all fail verification keeps its split and is counted; nothing is redrawn.
- Anchored patients without an eligible study are counted, listed in
  `anchor_absent_patients.csv`, and kept in `patient_splits.csv` with `in_archive` false, so a
  patient who returns in a later export keeps the original split.
- Anchor every later build on the previous build's `patient_splits.csv`: the mapping is then
  unchanged by new studies, new years or metadata row order, and only the new patients are assigned.
  The anchor must not lie inside `--out-dir`.

### Verification cache

Verification is the same full decode as in the one-study mode (`verify_video`, unchanged). With
`--verify-cache <dir>` (outside the repository) every result is stored with the file's size and
modification time and `VERIFY_VERSION` plus the decoder version; a result is reused only when all
of them match, so a changed file, a new `verify_video` (bump `VERIFY_VERSION`) or a decoder
upgrade is verified again. Results are written atomically every `--verify-chunk` videos (default
20,000), so an interrupted run resumes after its last write. Pending videos are verified in a
fixed shuffled order, so a `--verify-limit` run measures a representative sample.
`--verify-splits` restricts verification to some splits; the other splits' videos are excluded
as `not_verified` and get no manifest.

### Supervised workers and failures

With `--workers` above 1 (the default is 32) each video is decoded in a worker process
(`data/supervised_pool.py`) under a deadline, `--verify-timeout` (default 300 s, which outlasts one NFS major timeout of the video
mount).
A worker that overruns the deadline or dies is killed and replaced; a thread or a cancelled future
could not stop a stalled decoder or network read, a process can be killed. `--workers 1` decodes in
the build's own process, without a deadline (debugging only).

| Situation | What happens |
|---|---|
| A worker times out or dies on a video | The video is tried again, up to `--verify-retries` more times (default 2) in this run, a few seconds later; each failed attempt is logged with its task index and counts only |
| A decode fails: `empty`, `undecodable`, `no_frames`, `corrupt_frames` | Tried again the same way before the failure is cached: a storage read error can cause it with the file unchanged. `missing`, `no_path` and `whitespace_in_path` are cached at once (`missing` without a fingerprint, so it is verified again once the file appears) |
| A video still has no result after every attempt (unresolved) | Never cached. Every completed result is cached, the unresolved videos are listed in `<verify_cache>/unresolved-<time>.csv`, and the build raises `VerificationIncomplete`: nonzero exit, nothing written to `--out-dir`. Rerunning the same command tries them again |
| The same videos stay unresolved run after run | Inspect them, then rerun with `--accept-verify-failures N` (N at least their number): they are excluded as `verify_timeout`, `verify_crash` or `verify_error` (counted in `manifest_info.json`, listed in `excluded_videos.csv`) for that build only, never cached. More than N still stops the build |
| Attempts keep failing with no result in between (2 x workers + retries + 1 in a row), or killed workers cannot be reaped | The storage or the workers are failing: the run stops (`PoolBroken`), keeping what it cached; rerun once the storage is back |
| A cached failure should be retried after a storage incident | `--retry-failed` verifies the cached decode failures (the four above) again, still reusing every cached success |

The stat of cached entries (`--stat-threads`) raises `PoolBroken` when nothing answers for
`--verify-timeout` seconds, instead of waiting on a stalled mount forever.

### Arms

| Manifest | Rows |
|---|---|
| `train_inclusive.csv` | Every verified video of a train patient, any view (`OTHER` included) and any acquisition mode, unknown included |
| `train_bmode.csv` | The subset whose DICOM header confirms B-mode: `acq_category` `bmode` with `acq_confidence` `header` in the acquisition index, from a vendor in `acquisition.bmode_vendors`. Color, spectral Doppler, M-mode, other, unknown, vendor-unvalidated and unscanned videos are left out. Written only with `--acquisition-index`. The same rule (`effective_acq_category`) gives the acquisition category the RVSP probe slots and RADAR records use |
| `val.csv`, `test.csv` | Every verified video of the split's patients (inclusive) |

The acquisition index (from `data/scan_acquisition.py`, a parquet file or directory) is joined on
`columns.dicom_path`; a DICOM path listed twice in it stops the build. A video absent from it is
`unknown` and counted as not scanned. Since the inclusive arm keeps unknown and other modes,
comparing the arms is not a pure color-flow ablation; the summary quantifies these modes.

### Outputs of an archive build

| File | Contents |
|---|---|
| `patient_splits.csv` | `patient_id`, `split`, `origin` (`legacy` or `new`), `in_archive`: the mapping every later pipeline reuses, and the anchor of the next build |
| `anchor_absent_patients.csv` | Anchored patients without an eligible study in this archive |
| `studies.csv` | Every eligible study: patient, split, origin, date, exam type, `n_videos_metadata`, `label` (with `--labels`), and after verification `n_ok`, `n_bmode` (valid and header-confirmed B-mode, in every split), `n_failed`, `n_not_verified` |
| `videos.parquet` | Master index of every eligible video, failed ones included: ids, split, origin, view, paths, vendor, model, verification `reason`, `n_frames`, `fps`, `needs_padding`, acquisition fields, `in_inclusive`, `in_bmode` |
| `excluded_videos.csv` | Every eligible video left out of a manifest, with the `arm` (`inclusive` or `bmode`) and the reason: the verification failure, `not_verified`, or `acq:<category>` for the B-mode arm |
| loader manifests | As in the table above |
| `manifest_info.json`, `summary.md` | Provenance (inputs with SHA-256, the anchor and acquisition index included) and aggregate counts: by split for each arm, by year, view, vendor, model and acquisition mode, verification outcomes and rate, studies and patients left without videos, and videos per patient and per study |

`--assign-only` writes the first three files, `manifest_info.json` and `summary.md`.

Pretraining samples manifest rows (clips) uniformly, so a patient weighs in proportion to the
number of videos over all their studies; the summary reports the share of training videos held by
the top 1% and 10% of patients.

Before writing, the build checks: no patient in two splits; every anchored patient keeps its
split and origin; no anchored val/test patient in a training manifest; all videos of a study share
its patient's split; `train_bmode` is a subset of `train_inclusive` with header-confirmed B-mode
only; no video or DICOM path under two videos, studies or patients; no whitespace in a manifest
path. The smoke test then loads each manifest with the EchoJEPA loader, with the special cases
(low frame rate, unknown frame rate, shorter than one clip) drawn from `videos.parquet`.

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
python -m unittest tests.data.test_build_manifests tests.data.test_make_probe_manifests
python -m unittest tests.data.test_build_archive_manifests   # archive mode
python -m unittest tests.data.test_supervision   # worker crash / stall injection, each run under a hard deadline
```

The tests use synthetic metadata and tiny generated videos only. They also check that no
data path or study UID appears in the pipeline's files and that a run writes nothing into
the repository.
