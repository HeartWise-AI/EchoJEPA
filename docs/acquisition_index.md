# Acquisition-mode index

`data/scan_acquisition.py` records, for every echo DICOM, which acquisition modes its header
declares (color flow, PW / CW spectral Doppler, M-mode, plain 2D B-mode), so the archive
pretraining can build an inclusive TTE arm and a confirmed B-mode-only arm.
`data/validate_acquisition.py` checks those header flags against the converted videos.

The scanner reads **headers only** (`pydicom.dcmread(..., stop_before_pixels=True,
specific_tags=[...])`), never pixel data. It keeps the independently observable facts (which
region types and formats exist, each flag as true / false / unknown) next to one derived
category, and a header that is missing, unreadable or contradictory gives **unknown** flags,
never a negative Doppler flag.

Every output holds file paths, so caches, indexes and review material must live **outside
this repository**; both scripts refuse an output location inside it. The reports (`report`,
`validation_summary.*`) hold aggregate counts only.

## Commands

```bash
# 1. Usable TTE videos (study_type "ECHO CARDIAQUE AVEC DOPPLER", avi_status "success", a dcm_path).
python data/scan_acquisition.py scan --metadata <master_metadata.parquet> \
    --select usable-tte --cache-dir <acq_cache> --workers 8

# 2. A bounded, stratified inventory of TTE records WITHOUT a usable video (conversion failed
#    or missing video), to see whether PW/CW material is stored there. A separate cache keeps
#    its report apart from the usable videos.
python data/scan_acquisition.py scan --metadata <master_metadata.parquet> \
    --select excluded-tte --sample 2000 --seed 0 --stratify year,brand \
    --cache-dir <acq_cache_excluded>

# Any explicit list of DICOMs (one dcm_path per line); with --metadata, every path must be in it.
python data/scan_acquisition.py scan --paths-file <paths.txt> --cache-dir <acq_cache>

# 3. Before consolidating the full scan: reread every read error and every DICOM changed on disk
#    (--check-changed stats every cached row, which is slow on a cold NAS).
python data/scan_acquisition.py scan --metadata <master_metadata.parquet> \
    --select usable-tte --cache-dir <acq_cache> --workers 8 --retry-errors --check-changed

#    One parquet file, latest row per DICOM (+ acquisition_index.json provenance).
python data/scan_acquisition.py consolidate --cache-dir <acq_cache>

# 4. Aggregate coverage / error report (markdown + json), optionally per year, brand/model, view.
python data/scan_acquisition.py report --index <acq_cache>/acquisition_index.parquet \
    --metadata <master_metadata.parquet> --select usable-tte --out-dir <report_dir>

# 5. Header flags vs decoded converted videos, on a stratified sample.
python data/validate_acquisition.py --index <acq_cache>/acquisition_index.parquet \
    --metadata <master_metadata.parquet> --out-dir <review_dir> --per-stratum 2 --seed 0
```

### Resuming, rescanning, benchmarking

| Option (`scan`) | Effect |
|---|---|
| (re-run the same command) | Resumes: only DICOMs without a cached row from the current `SCANNER_VERSION` are read. With nothing to do, nothing is read. |
| `--check-changed` | Also stats every cached file and rescans those whose size or mtime changed (or that appeared / disappeared). The stats raise instead of waiting when none answers for `--timeout` seconds. |
| `--retry-errors` | Also rescans cached rows whose read failed (e.g. a NAS outage outlasting the retries), accepted worker failures included. |
| `--limit-chunks N` | Stops after N chunks (status `partial`); re-running continues. |
| `--workers N` | Reader processes (default 8). Header reads are I/O bound. |
| `--chunk-size N` | Files per shard (default 5000). |
| `--max-retries N` | Retries of a transient I/O error (EIO, ESTALE, ETIMEDOUT, ...), with exponential backoff capped at 8 s (default 3). Missing / permission errors are not retried. |
| `--timeout S` | Deadline for one header read in its worker process (default 300 s, which outlasts one NFS major timeout). A worker that overruns it or dies is killed and replaced. |
| `--worker-retries N` | Further attempts for a file whose worker timed out or died (default 2). |
| `--accept-scan-failures N` | After inspection, records up to N files still unresolved after every attempt as read errors (below). |
| `--sample N --seed S --stratify a,b` | A deterministic stratified sample (below). |

Each header is read in a worker process (`data/supervised_pool.py`), never a thread: a thread
blocked in network I/O cannot be stopped, a process can be killed. A file whose worker overran
`--timeout` or died is retried; one still failing after its attempts is **unresolved**: it gets no
row (so a rerun reads it again), it is listed in `runs/unresolved-<run id>.csv`, the run's status is
`incomplete`, and the scan exits nonzero once every other row is cached. If the same files stay
unresolved, inspect them and rerun with `--accept-scan-failures N` (N at least their number): they
are then recorded as read errors (`scan_timeout`, `scan_crash`, `scan_error`), which `--retry-errors`
reads again and the archive build counts as failed reads. Attempts failing in a row with no result in
between (twice the workers plus one file's attempts), or killed workers that cannot be reaped, stop the
scan at once ("the storage or the workers are failing"); rerun when the storage is back.

Benchmark before raising concurrency, so the NAS is not saturated and training reads are not
starved: run `scan --sample 2000 --limit-chunks 1 --chunk-size 500` into a throwaway cache
directory once per `--workers` value (e.g. 4, 8, 16) and compare the files/s printed per
chunk. Use different sample seeds or fresh files per setting so the page cache does not flatter
later runs. The log prints counts and rates only, never paths or identifiers.

`SCANNER_VERSION` in the module must be bumped on **any** change to the tags read, the rules or
the encodings: rows from another version are rescanned automatically.

## Rules

Region codes follow DICOM PS3.3 C.8.5.5.1.1 (Region Spatial Format) and C.8.5.5.1.2 (Region
Data Type); the tables are constants in the module. Spatial formats: 0 none, 1 2D, 2 M-mode,
3 spectral, 4 waveform, 5 graphics. Data types: 0 none, 1 tissue, 2 color flow, 3 PW spectral
Doppler, 4 CW spectral Doppler, 5-7 Doppler mean / mode / max trace, 8 volume trace, 9 d(volume)/dt
trace (absent from the 2020a and current editions), 10 ECG, 11 pulse, 12 phonocardiogram,
13 gray bar, 14 color bar, 15 integrated backscatter, 16 area trace, 17 d(area)/dt, 18 other
physiological input.

**All** items of the Sequence of Ultrasound Regions are read: color flow is often not the
first item. A file is **unknown** (`acq_confidence` "unknown", every `has_*` flag NA) when the
first of these applies; `acq_rule` names it:

| `acq_rule` | Condition |
|---|---|
| `read_error` | The header could not be read (see `failure_reason`) |
| `no_region_sequence` | No Sequence of Ultrasound Regions |
| `empty_region_sequence` | The sequence has no item |
| `region_missing_type_or_format` | A region lacks Region Data Type or Region Spatial Format |
| `invalid_region_value` | A value is not one integer (multi-valued, unreadable) |
| `undefined_region_code` | A code outside the standard tables |
| `contradictory_type_format` | PW/CW outside the spectral format; tissue or color flow outside 2D / M-mode; a spectral-format region holding anything but PW/CW or a Doppler trace; a 2D / M-mode region of data type "none" |
| `doppler_kind_undetermined` | Spectral evidence (spectral format or a Doppler trace) without a PW or CW region, so PW vs CW is undecided |

Otherwise the flags are observed true / false (`has_tissue_2d`: a tissue region in 2D format;
`has_color_flow`: any color-flow region; `has_pw_doppler` / `has_cw_doppler`: any PW / CW
region; `has_mmode`: any region in M-mode format), `acq_confidence` is "header", and the
category is the first that applies:

| `acq_category` | `acq_rule` | Condition |
|---|---|---|
| `spectral_doppler` | `pw_or_cw_region` | Any PW or CW region |
| `color_flow` | `color_flow_region` | Any color-flow region (2D or M-mode) |
| `mmode` | `tissue_mmode_region` | Any tissue region in M-mode format |
| `other` | `no_tissue_2d_region` | No tissue 2D region |
| `bmode` | `tissue_2d_with_allowed_regions` | At least one tissue 2D region, and every other region is another tissue 2D region or an allow-listed companion: data type 10, 11, 12, 13 or 18 in format 0, 4 or 5, or data type 0 in format 0 or 5 |
| `other` | `region_outside_bmode_allowlist` | Anything else, e.g. a color bar without color flow, volume / area traces, integrated backscatter |

Only `acq_category == "bmode"` with `acq_confidence == "header"` may enter the B-mode-only arm.
A color-flow region can encode blood flow or tissue motion: the index never names a valve,
jet or finding. Photometric interpretation (RGB, palette, ...) is recorded but never decides
a flag, because plain B-mode often carries coloured overlays.

### Read errors

`read_status` is "ok" or "error"; `failure_reason` is one of `missing`, `permission_denied`,
`not_a_file`, `empty_file`, `not_dicom` (no DICOM preamble / meta header), `ended_before_pixel_data`
(the header runs to the end of the file: a truncated file or a non-image object; pydicom
otherwise returns such a header silently), `unparseable_header`, `io_error` (transient error
persisting after the retries, or another OS error), and, only with `--accept-scan-failures`,
`scan_timeout` / `scan_crash` / `scan_error` (a file whose every attempt overran the deadline, killed
its worker, or raised). Corrupt or truncated **pixel data** does not
matter: it is never read.

## Outputs

`--cache-dir` holds:

| Path | Content |
|---|---|
| `shards/shard-<run id>-<chunk>.parquet` | Index rows, one shard per chunk, written to a temporary file and renamed (atomic). The run id is a UTC timestamp, so file order is write order. |
| `progress.json` | Counts and rate of the current / last run, rewritten atomically after every chunk |
| `runs/run-<run id>.json` | One record per run: command, code version, pydicom version, metadata size / mtime / SHA-256, selection facts (including whether `file_id` is unique), plan, counts (failed attempts by kind, unresolved files) |
| `runs/unresolved-<run id>.csv` | Written by a run that left files unresolved: their paths, failure kind and attempts, for inspection |
| `scan.lock` | Present while a scan runs; one scan per cache directory. Delete it by hand after a crash. |
| `acquisition_index.parquet`, `.json` | Written by `consolidate`: latest row per `dcm_path`, sorted, with its provenance (rows per scanner version, shards, runs, SHA-256) |

The latest row per `dcm_path` wins across runs; two different rows for one path inside one
run mean a corrupt cache, and `consolidate` refuses them. Readers should use
`scan_acquisition.load_index(path)` (a consolidated file or a cache directory), which returns
nullable pandas dtypes (`boolean`, `Int64`, Arrow-backed strings).

Columns (the "Acquisition index" contract): `dcm_path`, `source_size`, `source_mtime_ns`
(from `fstat` of the file actually parsed), `scanner_version`, `read_status`, `failure_reason`,
`has_region_sequence`, `n_regions`, `region_data_types` and `region_spatial_formats` (sorted
unique codes joined by ","; "" when there are none), `has_tissue_2d`, `has_color_flow`,
`has_pw_doppler`, `has_cw_doppler`, `has_mmode`, `acq_category`, `acq_confidence`, `acq_rule`,
`n_frames_header`, `manufacturer`, `model_name`, `software_versions` (multiple values joined by
a backslash), `sop_class_uid`, `photometric_interpretation`. One diagnostic column is added:
`region_pairs`, each region's `type/format` pair ("?" when missing or invalid), because the two
code lists lose which type sat in which format.

### Key and joins

The index is keyed by `dcm_path`. `scan` raises when a `dcm_path` appears on more than one
selected metadata row (even identical rows), and when a `--paths-file` path is not in the
metadata it is joined with. `file_id` is not assumed unique: its uniqueness among the selected
rows is reported in the run record. `validate_acquisition.py` also refuses an `avi_path` shared
by two DICOMs.

### Sampling

`--sample` and the validation sample rank rows within each stratum by a seeded hash of their
`dcm_path`, so the draw is deterministic, independent of row order, and a row's draw does not
depend on other strata. Missing stratum values form their own stratum ("<missing>"). A total
`n` is spread over the strata as equally as possible: strata smaller than their share are
taken whole and the rest is shared by the larger ones. `year` is the first four characters of
`date`; any other metadata column can be a stratum.

## Validating the flags against the converted videos

`validate_acquisition.py` joins the index to the metadata on `dcm_path` (for `avi_path`, year,
brand, model, view), draws up to `--per-stratum` videos per stratum of `acq_category x year x
brand x predicted_class` by default (so color-flow, B-mode, other and unknown cases of every
year, scanner and view are drawn; `--max-videos` optionally caps the total), and decodes
`--frames` evenly spaced frames per video with OpenCV.

Statistics, robust to coloured overlays:

- A pixel is coloured when `max(R,G,B) - min(R,G,B) >= --chroma-threshold` (default 40) and
  `max(R,G,B) >= --value-threshold` (default 40). Gray-scale video decodes with a chroma of a
  few units.
- `full_fraction_*`: coloured share of the whole frame (median / max over frames).
  `center_fraction_*`: the same inside a central box spanning `--center-fraction` (default 0.6)
  of the height and width, where the sector is and overlays (ECG trace, colour bar, labels)
  mostly are not; `outside_fraction_median` measures the overlays.
- `center_dynamic`: mean share of central pixels whose coloured state changes between
  consecutive sampled frames. Color flow changes over the cardiac cycle; a static coloured
  marker does not.

Colour is **observed** when the 90th percentile of `center_fraction` reaches
`--min-center-fraction` (default 0.01) and, with at least two frames, `center_dynamic` reaches
`--min-dynamic` (default 0.002). The defaults are starting points to calibrate on real data:
read the per-category distributions in the summary, adjust, and re-run.

Each video is labelled `agree_color`, `agree_no_color`, `header_color_missing_in_video` (the
header has a color-flow region but the video shows none), `video_color_without_header_color`,
`header_unknown` or `video_undecodable`. **Conversion and cropping preservation** is assessed
by whether colour present in color-flow DICOMs is still present in the AVI: the summary
reports, among decoded videos whose header has a color-flow region, how many show colour.
A missing colour can mean a gray-scale conversion, a crop that removed the colour box, little
flow in the sampled frames, or a header the image does not bear out; the contact sheets let a
reviewer tell these apart. The check covers colour only: spectral Doppler and M-mode displays
are mostly gray and are not judged from pixels.

Outputs in `--out-dir`:

| File | Content |
|---|---|
| `review_sheet.csv` | One row per sampled video **with paths**, its header flags, statistics, verdict, contact sheet name, and empty `reviewer_color_present` / `reviewer_notes` columns. Local review only. |
| `contact_sheets/review_<id>.png` | For disagreements (at most `--contact-sheets`, default 50): sampled frames with the central box outlined, over their coloured-pixel masks. Named by review id only. Local review only. |
| `validation_summary.md`, `.json` | Aggregates only: sample design and coverage (available vs sampled by category, year, brand, view), thresholds, decoder, agreement tables, statistic distributions per category, the preservation count |

Clinical review may be needed for ambiguous cases (e.g. tissue-motion color, very low flow);
the review sheet is the place to record it.

## Tests

Synthetic only (tiny pydicom DICOMs and OpenCV MJPG videos written to a temporary directory):

```bash
python -m unittest tests.data.test_scan_acquisition tests.data.test_validate_acquisition
python -m unittest tests.data.test_supervision   # worker crash / stall injection, under hard deadlines
```

## Open points

- The B-mode allow-list is deliberately strict (e.g. a gray bar declared in 2D format makes a
  video `other`). Check the `region_pairs` table of the real report before relaxing it, and
  bump `SCANNER_VERSION` when changing it.
- Spectral evidence without a PW/CW region is unknown rather than guessed.
- An absent `SequenceOfUltrasoundRegions` (some vendors or exports) leaves a video unknown; the
  report shows how many, by manufacturer and year.

## B-mode vendor allow-list

The B-mode-only arm admits header B-mode only from vendors whose headers were checked against the
decoded videos with `validate_acquisition.py` (`acquisition.bmode_vendors` in
configs/data/manifests_tte_archive.yaml). Some vendors' headers can declare one tissue region over a
colour-flow acquisition, so header B-mode from any other vendor is unresolved
(`acq:bmode_vendor_unvalidated`) and stays in the inclusive arm only. Header-unknown and `other` videos
stay out of the B-mode arm.
