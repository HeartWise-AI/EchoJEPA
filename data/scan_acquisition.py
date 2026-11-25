# /data/scan_acquisition.py

"""Acquisition-mode index: which echo DICOMs hold color flow, PW/CW spectral Doppler,
M-mode or plain 2D B-mode, read from the DICOM headers only (never the pixels).

Every item of the Sequence of Ultrasound Regions is read, and the independently
observable flags are kept (`has_color_flow`, `has_pw_doppler`, ... as true / false /
unknown) next to one derived `acq_category`. A header that is missing, unreadable or
contradictory yields unknown flags, never a negative Doppler flag. See
docs/acquisition_index.md for the rules, the outputs and how to resume a scan.

Examples:
    # Scan the usable TTE videos (resumable; re-running scans only what is new or stale).
    python data/scan_acquisition.py scan --metadata <master_metadata.parquet> \\
        --select usable-tte --cache-dir <acq_cache_dir> --workers 8

    # A bounded, stratified inventory of TTE records without a usable video.
    python data/scan_acquisition.py scan --metadata <master_metadata.parquet> \\
        --select excluded-tte --sample 2000 --seed 0 --stratify year,brand \\
        --cache-dir <acq_cache_excluded_dir>

    # One parquet file with the latest row per DICOM, then the aggregate report.
    python data/scan_acquisition.py consolidate --cache-dir <acq_cache_dir>
    python data/scan_acquisition.py report --index <acq_cache_dir>/acquisition_index.parquet \\
        --metadata <master_metadata.parquet> --select usable-tte --out-dir <report_dir>
"""

import argparse
import datetime
import errno
import functools
import glob
import hashlib
import json
import logging
import os
import socket
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_manifests import BATCH_SIZE, ensure_outside_repo, git_commit, iter_table, markdown_table, sha256
from supervised_pool import KINDS, Failure, PoolBroken, SupervisedPool, thread_map

# Bump on ANY change to how a header becomes a row (tags read, rules, encodings): cached rows
# from another version are rescanned, so an old rule never silently survives in the index.
SCANNER_VERSION = "1"

TTE = "ECHO CARDIAQUE AVEC DOPPLER"
SELECTIONS = ("usable-tte", "excluded-tte")
MISSING = "<missing>"

# Only these tags are parsed; the pixel data is never read (stop_before_pixels).
HEADER_TAGS = ("SequenceOfUltrasoundRegions", "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
               "NumberOfFrames", "SOPClassUID", "PhotometricInterpretation")

# --------------------------------------------------------------------------- #
#  DICOM code tables
# --------------------------------------------------------------------------- #

# Region Spatial Format (0018,6012), DICOM PS3.3 C.8.5.5.1.1
# (https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.5.5.html, read 2026-10-07).
REGION_SPATIAL_FORMATS = {
    0: "none",       # None or not applicable
    1: "2d",         # 2D (tissue or flow)
    2: "m_mode",     # M-Mode (tissue or flow)
    3: "spectral",   # Spectral (CW or PW Doppler)
    4: "waveform",   # Wave form (physiological traces, Doppler traces, ...)
    5: "graphics",   # Graphics
}

# Region Data Type (0018,6014), DICOM PS3.3 C.8.5.5.1.2 (same source). The standard lists
# the codes in hex (000AH = 10, ..., 0012H = 18); pydicom returns them as integers.
REGION_DATA_TYPES = {
    0: "none",                       # None or not applicable
    1: "tissue",
    2: "color_flow",                 # blood flow OR tissue motion: never read as a valve, jet or finding
    3: "pw_spectral_doppler",
    4: "cw_spectral_doppler",
    5: "doppler_mean_trace",
    6: "doppler_mode_trace",
    7: "doppler_max_trace",
    8: "volume_trace",
    # 0009H is absent from the 2020a and current editions; kept as the d(volume)/dt trace of
    # older references. Either way it can never make a video B-mode.
    9: "d_volume_dt_trace",
    10: "ecg_trace",
    11: "pulse_trace",
    12: "phonocardiogram_trace",
    13: "gray_bar",
    14: "color_bar",
    15: "integrated_backscatter",
    16: "area_trace",
    17: "d_area_dt_trace",
    18: "other_physiological_input",  # Other Physiological (Amplitude vs. Time) input
}

F_NONE, F_2D, F_MMODE, F_SPECTRAL, F_WAVEFORM, F_GRAPHICS = 0, 1, 2, 3, 4, 5
T_NONE, T_TISSUE, T_COLOR, T_PW, T_CW = 0, 1, 2, 3, 4
DOPPLER_TRACES = frozenset({5, 6, 7})
IMAGE_FORMATS = frozenset({F_2D, F_MMODE})
SPECTRAL_CONTENT = frozenset({T_PW, T_CW}) | DOPPLER_TRACES

# Regions that may sit next to a tissue 2D region in a B-mode video: physiological traces
# (ECG, pulse, phono, other) and the gray bar drawn as waveform / graphics / no format, and
# "none" regions drawn as graphics or with no format. Everything else (color bars, volume
# or area traces, integrated backscatter, ...) makes a video `other`, never `bmode`.
BMODE_COMPANION_TYPES = frozenset({10, 11, 12, 13, 18})
BMODE_COMPANION_FORMATS = frozenset({F_NONE, F_WAVEFORM, F_GRAPHICS})
BMODE_NONE_FORMATS = frozenset({F_NONE, F_GRAPHICS})

CATEGORIES = ("bmode", "color_flow", "spectral_doppler", "mmode", "other", "unknown")
FLAGS = ("has_tissue_2d", "has_color_flow", "has_pw_doppler", "has_cw_doppler", "has_mmode")

# `acq_rule` values: which rule produced the category. Unknown first, in the order checked.
UNKNOWN_RULES = ("read_error", "no_region_sequence", "empty_region_sequence", "region_missing_type_or_format",
                 "invalid_region_value", "undefined_region_code", "contradictory_type_format",
                 "doppler_kind_undetermined")
HEADER_RULES = {"pw_or_cw_region": "spectral_doppler", "color_flow_region": "color_flow",
                "tissue_mmode_region": "mmode", "tissue_2d_with_allowed_regions": "bmode",
                "no_tissue_2d_region": "other", "region_outside_bmode_allowlist": "other"}

# --------------------------------------------------------------------------- #
#  Index schema (the "Acquisition index" data contract)
# --------------------------------------------------------------------------- #

INDEX_SCHEMA = pa.schema([
    ("dcm_path", pa.string()),
    ("source_size", pa.int64()),
    ("source_mtime_ns", pa.int64()),
    ("scanner_version", pa.string()),
    ("read_status", pa.string()),
    ("failure_reason", pa.string()),
    ("has_region_sequence", pa.bool_()),
    ("n_regions", pa.int64()),
    ("region_data_types", pa.string()),
    ("region_spatial_formats", pa.string()),
    ("has_tissue_2d", pa.bool_()),
    ("has_color_flow", pa.bool_()),
    ("has_pw_doppler", pa.bool_()),
    ("has_cw_doppler", pa.bool_()),
    ("has_mmode", pa.bool_()),
    ("acq_category", pa.string()),
    ("acq_confidence", pa.string()),
    ("acq_rule", pa.string()),
    ("n_frames_header", pa.int64()),
    ("manufacturer", pa.string()),
    ("model_name", pa.string()),
    ("software_versions", pa.string()),
    ("sop_class_uid", pa.string()),
    ("photometric_interpretation", pa.string()),
    # Diagnostic addition to the contract: each region's "type/format" pair ("?" when missing or
    # invalid), sorted unique. The two contract columns lose which type sat in which format.
    ("region_pairs", pa.string()),
])
INDEX_COLUMNS = INDEX_SCHEMA.names
_PANDAS = {pa.string(): "string", pa.int64(): "Int64", pa.bool_(): "boolean"}
DTYPES = {f.name: _PANDAS[f.type] for f in INDEX_SCHEMA}
# Arrow-backed strings when reading the index: millions of paths stay compact in memory.
ARROW_TO_PANDAS = {pa.string(): pd.StringDtype("pyarrow"), pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype()}

# Columns the scan needs from the cache to decide what to reuse.
CACHE_KEY_COLUMNS = ["dcm_path", "scanner_version", "source_size", "source_mtime_ns", "read_status"]

# --------------------------------------------------------------------------- #
#  Header -> row
# --------------------------------------------------------------------------- #

INVALID = "invalid"  # a region attribute that is present but not one integer

# errno values worth retrying (the NAS may hiccup); anything else fails at once.
TRANSIENT_ERRNOS = frozenset(getattr(errno, n) for n in (
    "EIO", "EAGAIN", "EINTR", "EBUSY", "ETIMEDOUT", "ESTALE", "ENOLCK", "ECONNRESET", "ECONNABORTED",
    "ENETDOWN", "ENETUNREACH", "EHOSTDOWN", "EHOSTUNREACH", "EREMOTEIO") if hasattr(errno, n))
MAX_BACKOFF_S = 8.0

# Seconds one header read may take in a worker process before the worker is killed and the file tried
# again. A read normally takes well under a second, plus at most ~4 s of transient-error backoff;
# 300 s also outlasts one NFS major timeout of the DICOM mount (timeo 60 s, 2 retransmissions: ~3 min),
# so a read that recovers from a server hiccup finishes instead of being killed.
SCAN_TIMEOUT = 300.0

# Further attempts within one run for a file whose worker timed out or died.
WORKER_RETRIES = 2

# `failure_reason` of a file still without a row after every attempt, when --accept-scan-failures admits it
# (by the last attempt: deadline overrun, worker death, or an exception). Read errors: --retry-errors retries them.
UNRESOLVED_REASONS = {"timeout": "scan_timeout", "crash": "scan_crash", "error": "scan_error"}


def configure_pydicom():
    """Silence pydicom's value warnings and logging: they are slow at archive scale and can
    quote header values (UIDs, names) into the work log.
    """
    import pydicom

    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    logging.getLogger("pydicom").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", module=r"pydicom(\..*)?$")


def region_code(item, keyword):
    """One region attribute as an int; None when absent or empty, INVALID when it is not one integer."""
    try:
        if keyword not in item:
            return None
        value = item[keyword].value
    except Exception:
        return INVALID
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return INVALID
    return int(value)


def header_regions(ds):
    """(data type, spatial format) of EVERY item of the Sequence of Ultrasound Regions, in order;
    None when the sequence is absent. Color flow is often not the first item.
    """
    if "SequenceOfUltrasoundRegions" not in ds:
        return None
    try:
        items = list(ds.SequenceOfUltrasoundRegions or [])
    except Exception:
        return [(INVALID, INVALID)]
    return [(region_code(it, "RegionDataType"), region_code(it, "RegionSpatialFormat")) for it in items]


def contradictory(t, f):
    """A region whose data type and spatial format cannot both be true (PS3.3 C.8.5.5.1.1/1.2):

    - PW/CW spectral Doppler outside the spectral format;
    - tissue or color flow outside the 2D / M-mode formats;
    - a spectral-format region holding anything but PW/CW Doppler or a Doppler trace;
    - a 2D / M-mode region whose data type is "none" (an image with no stated content).
    """
    return ((t in (T_PW, T_CW) and f != F_SPECTRAL)
            or (t in (T_TISSUE, T_COLOR) and f not in IMAGE_FORMATS)
            or (f == F_SPECTRAL and t not in SPECTRAL_CONTENT)
            or (f in IMAGE_FORMATS and t == T_NONE))


def bmode_companion(t, f):
    return (t in BMODE_COMPANION_TYPES and f in BMODE_COMPANION_FORMATS) or (t == T_NONE and f in BMODE_NONE_FORMATS)


def unknown_result(rule):
    return {**{flag: pd.NA for flag in FLAGS}, "acq_category": "unknown", "acq_confidence": "unknown",
            "acq_rule": rule}


def classify_regions(regions):
    """Flags and category from the regions of one DICOM (see `header_regions`).

    Unknown (every flag NA, confidence "unknown"), checked in this order: no sequence, an
    empty sequence, a region without a data type or spatial format, a value that is not one
    integer, a code outside the standard tables, a contradictory type/format pair, or spectral
    evidence (spectral format or a Doppler trace) without a PW or CW region, which leaves
    PW vs CW undecided.

    Otherwise each flag is observed true or false, and the category is the first that applies:
    spectral_doppler (any PW/CW) > color_flow (any color flow) > mmode (tissue in M-mode)
    > bmode (at least one tissue 2D region, every other region a tissue 2D region or an
    allow-listed companion, see `bmode_companion`) > other. RGB / photometric metadata never
    enters: colored overlays (ECG, text, markers) are common in plain B-mode.
    """
    if regions is None:
        return unknown_result("no_region_sequence")
    if not regions:
        return unknown_result("empty_region_sequence")
    values = [v for pair in regions for v in pair]
    if any(v is None for v in values):
        return unknown_result("region_missing_type_or_format")
    if any(v is INVALID for v in values):
        return unknown_result("invalid_region_value")
    if any(t not in REGION_DATA_TYPES or f not in REGION_SPATIAL_FORMATS for t, f in regions):
        return unknown_result("undefined_region_code")
    if any(contradictory(t, f) for t, f in regions):
        return unknown_result("contradictory_type_format")
    pairs, types = set(regions), {t for t, _ in regions}
    spectral = any(f == F_SPECTRAL for _, f in regions) or bool(types & DOPPLER_TRACES)
    if spectral and not types & {T_PW, T_CW}:
        return unknown_result("doppler_kind_undetermined")

    flags = {"has_tissue_2d": (T_TISSUE, F_2D) in pairs, "has_color_flow": T_COLOR in types,
             "has_pw_doppler": T_PW in types, "has_cw_doppler": T_CW in types,
             "has_mmode": any(f == F_MMODE for _, f in regions)}
    if flags["has_pw_doppler"] or flags["has_cw_doppler"]:
        rule = "pw_or_cw_region"
    elif flags["has_color_flow"]:
        rule = "color_flow_region"
    elif (T_TISSUE, F_MMODE) in pairs:
        rule = "tissue_mmode_region"
    elif not flags["has_tissue_2d"]:
        rule = "no_tissue_2d_region"
    elif all(p == (T_TISSUE, F_2D) or bmode_companion(*p) for p in pairs):
        rule = "tissue_2d_with_allowed_regions"
    else:
        rule = "region_outside_bmode_allowlist"
    return {**flags, "acq_category": HEADER_RULES[rule], "acq_confidence": "header", "acq_rule": rule}


def code_list(values):
    """Sorted unique integer codes joined by ",", e.g. "1,2"; missing / invalid values are left out."""
    return ",".join(str(v) for v in sorted({v for v in values if isinstance(v, int)}))


def pair_list(regions):
    def key(v):
        return (0, v) if isinstance(v, int) else (1, 0)

    def text(v):
        return str(v) if isinstance(v, int) else "?"

    unique = sorted(set(regions), key=lambda p: (key(p[0]), key(p[1])))
    return ",".join(f"{text(t)}/{text(f)}" for t, f in unique)


def header_text(ds, keyword):
    """A header attribute as text (multi-valued joined by a backslash, as DICOM stores it); None when absent."""
    try:
        value = ds.get(keyword)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        value = "\\".join(str(v) for v in value)
    value = str(value).strip()
    return value or None


def header_int(ds, keyword):
    try:
        value = ds.get(keyword)
        return None if value is None or str(value).strip() == "" else int(value)
    except Exception:
        return None


def describe_header(ds):
    """Row fields from one parsed header (everything but the file identity and read status)."""
    regions = header_regions(ds)
    return {
        "has_region_sequence": regions is not None,
        "n_regions": 0 if regions is None else len(regions),
        "region_data_types": code_list(t for t, _ in regions or []),
        "region_spatial_formats": code_list(f for _, f in regions or []),
        "region_pairs": pair_list(regions or []),
        **classify_regions(regions),
        "n_frames_header": header_int(ds, "NumberOfFrames"),
        "manufacturer": header_text(ds, "Manufacturer"),
        "model_name": header_text(ds, "ManufacturerModelName"),
        "software_versions": header_text(ds, "SoftwareVersions"),
        "sop_class_uid": header_text(ds, "SOPClassUID"),
        "photometric_interpretation": header_text(ds, "PhotometricInterpretation"),
    }


def empty_row(path, version=None):
    """A row of `path` with every field empty, stamped `version` (default SCANNER_VERSION)."""
    row = dict.fromkeys(INDEX_COLUMNS)
    row.update(dcm_path=path, scanner_version=version or SCANNER_VERSION)
    return row


def error_row(row, reason):
    """A file that could not be read: every flag unknown, never negative."""
    return {**row, "read_status": "error", "failure_reason": reason, **unknown_result("read_error")}


def os_error_reason(e):
    if e.errno in (errno.ENOENT, errno.ENOTDIR):
        return "missing"
    if e.errno in (errno.EACCES, errno.EPERM):
        return "permission_denied"
    if e.errno == errno.EISDIR:
        return "not_a_file"
    # pydicom raises a bare OSError (no errno) on some malformed headers.
    return "unparseable_header" if e.errno is None else "io_error"


def scan_file(path, max_retries=3, backoff=0.5, sleep=time.sleep, version=None):
    """One index row for one DICOM, from its header only.

    The size and mtime are taken from the opened file (fstat), so they describe the bytes
    that were parsed. A transient I/O error (see TRANSIENT_ERRNOS) is retried up to
    `max_retries` times with exponential backoff capped at MAX_BACKOFF_S; any other failure
    is recorded at once as `read_status == "error"` with a `failure_reason`.

    pydicom stops quietly at the end of a truncated header, so the parse must also have
    stopped AT the pixel data: a header that runs to the end of the file is an error
    (`ended_before_pixel_data`: truncated, or not an image), not a file without regions.
    The row is stamped `version` (default SCANNER_VERSION): a scan passes its own to every worker.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    row = empty_row(path, version)
    for attempt in range(max_retries + 1):
        try:
            with open(path, "rb") as f:
                st = os.fstat(f.fileno())
                row.update(source_size=st.st_size, source_mtime_ns=st.st_mtime_ns)
                if st.st_size == 0:
                    return error_row(row, "empty_file")
                ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=list(HEADER_TAGS))
                ended = f.tell() >= st.st_size
            break
        except InvalidDicomError:
            return error_row(row, "not_dicom")
        except OSError as e:
            if e.errno in TRANSIENT_ERRNOS and attempt < max_retries:
                sleep(min(backoff * 2 ** attempt, MAX_BACKOFF_S))
                continue
            return error_row(row, os_error_reason(e))
        except Exception:
            return error_row(row, "unparseable_header")
    if ended:
        return error_row(row, "ended_before_pixel_data")
    try:
        return {**row, "read_status": "ok", **describe_header(ds)}
    except Exception:
        return error_row(row, "unparseable_header")


def rows_frame(rows):
    """Index rows (dicts) as a DataFrame with the contract's nullable dtypes."""
    return pd.DataFrame(list(rows), columns=INDEX_COLUMNS).astype(DTYPES)


# --------------------------------------------------------------------------- #
#  Selection & stratified sampling
# --------------------------------------------------------------------------- #


def table_columns(source):
    if isinstance(source, pd.DataFrame):
        return list(source.columns)
    if str(source).endswith(".parquet"):
        return pq.ParquetFile(source).schema_arrow.names
    return list(pd.read_csv(source, nrows=0).columns)


def clean_text(series):
    """Strings with surrounding whitespace removed; empty strings become NA."""
    return series.astype("string").str.strip().replace("", pd.NA)


def parse_strata(text):
    return [s.strip() for s in (text or "").split(",") if s.strip()]


def read_paths_file(path):
    with open(path) as f:
        lines = [line.strip() for line in f]
    paths = [p for p in lines if p]
    unique = list(dict.fromkeys(paths))
    if len(unique) < len(paths):
        print(f"[select] --paths-file repeats {len(paths) - len(unique)} paths; each is scanned once.")
    return unique


def select_rows(metadata=None, select=None, paths=None, columns=(), batch_size=BATCH_SIZE):
    """The metadata rows to scan, one per DICOM: `dcm_path` plus the requested `columns`
    ("year" is derived from `date`). Also returns aggregate facts about the selection.

    - usable-tte:   TTE (`study_type`), `avi_status == "success"`, a `dcm_path`.
    - excluded-tte: TTE with a `dcm_path` whose conversion did not succeed.
    - `paths`:      these DICOMs (restricted further by `select` when both are given).

    The index is keyed by `dcm_path`, so a path on two selected rows raises (whatever the
    rest of the rows say), as does a path from `paths` the metadata does not know. The
    uniqueness of `file_id` is reported, not assumed.
    """
    if select is not None and select not in SELECTIONS:
        raise ValueError(f"--select must be one of {SELECTIONS}, got {select!r}.")
    if metadata is None:
        if select is not None or columns:
            raise ValueError("--select and --stratify need --metadata.")
        if paths is None:
            raise ValueError("Nothing selected: give --select or --paths-file.")
        return pd.DataFrame({"dcm_path": pd.array(list(paths), dtype="string")}), {
            "selection": "paths-file", "rows_selected": len(paths), "file_id": None}

    meta_cols = ["date" if c == "year" else c for c in columns]
    keep_cols = list(dict.fromkeys(meta_cols + (["file_id"] if "file_id" in table_columns(metadata) else [])))
    need = list(dict.fromkeys(["dcm_path"] + (["study_type", "avi_status"] if select else []) + keep_cols))
    wanted = set(paths) if paths is not None else None
    parts, tte_without_path = [], 0
    for batch in iter_table(metadata, need, batch_size):
        dcm = clean_text(batch["dcm_path"])
        keep = dcm.notna()
        if select:
            tte = clean_text(batch["study_type"]) == TTE
            ok = clean_text(batch["avi_status"]) == "success"
            tte_without_path += int((tte & dcm.isna()).sum())
            keep &= (tte & ok) if select == "usable-tte" else (tte & ~ok.fillna(False))
        if wanted is not None:
            keep &= dcm.isin(wanted)
        keep = keep.fillna(False).to_numpy(bool)
        parts.append(pd.DataFrame({"dcm_path": dcm[keep], **{c: clean_text(batch[c][keep]) for c in keep_cols}}))
    sel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"dcm_path": pd.array([], "string")})
    if "date" in meta_cols:
        sel["year"] = sel["date"].str[:4]
        if "date" not in columns:
            sel = sel.drop(columns="date")

    if wanted is not None:
        missing = len(wanted - set(sel.dcm_path))
        if missing:
            raise ValueError(f"{missing} paths from --paths-file are not in the metadata"
                             + (" selection" if select else "") + "; the scan joins on dcm_path.")
    dup = sel.dcm_path.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{sel.loc[dup, 'dcm_path'].nunique()} dcm_path values appear on more than one selected "
                         "metadata row; the acquisition index is keyed by dcm_path.")
    info = {"selection": select or "paths-file", "rows_selected": int(len(sel)), "file_id": None}
    if select:
        info["tte_rows_without_dcm_path"] = tte_without_path
    if "file_id" in sel:
        fid = sel.pop("file_id")
        info["file_id"] = {"missing": int(fid.isna().sum()),
                           "duplicated_values": int(fid.dropna()[fid.dropna().duplicated()].nunique()),
                           "unique": bool(not fid.dropna().duplicated().any())}
    return sel.reset_index(drop=True), info


def hash_rank(values, seed):
    """A seeded pseudo-random uint64 per value that depends on the value alone, so a row's draw
    does not depend on row order or on which other rows exist.
    """
    key = hashlib.sha256(f"scan_acquisition:{seed}".encode()).hexdigest()[:16]
    return pd.util.hash_array(np.asarray(values, dtype=object), hash_key=key, categorize=False)


def allocate(sizes, n):
    """Spread n draws over strata as equally as possible: small strata are taken whole and their
    unused share goes to the larger ones. `sizes` is a list; ties are broken by position.
    """
    order = sorted(range(len(sizes)), key=lambda i: (sizes[i], i))
    alloc, left = [0] * len(sizes), n
    for k, i in enumerate(order):
        alloc[i] = min(sizes[i], left // (len(order) - k))
        left -= alloc[i]
    return alloc


def stratified_sample(df, keys, seed, n=None, per_stratum=None):
    """A deterministic, row-order-independent stratified sample of `df` (keyed by `dcm_path`).

    Strata are the sorted unique values of `keys` (missing as "<missing>"); within a stratum,
    rows are ranked by a seeded hash of their dcm_path. `per_stratum` caps each stratum,
    then `n` (if given) is spread over the strata by `allocate` (equal-ish shares).
    """
    if df.empty:
        return df.copy()
    missing = [k for k in keys if k not in df]
    if missing:
        raise ValueError(f"cannot stratify by {missing}: not in the selection columns.")
    d = df.assign(**{k: df[k].astype("string").fillna(MISSING) for k in keys})
    d = d.assign(_h=hash_rank(d.dcm_path, seed))
    d = d.assign(_g=d.groupby(keys, sort=True).ngroup() if keys else 0)
    d = d.sort_values(["_g", "_h", "dcm_path"], kind="stable")
    d = d.assign(_r=d.groupby("_g").cumcount())
    sizes = np.bincount(d._g.to_numpy(), minlength=int(d._g.max()) + 1)
    quota = np.minimum(sizes, per_stratum) if per_stratum is not None else sizes
    if n is not None:
        quota = np.asarray(allocate(list(quota), n))
    keep = d._r.to_numpy() < quota[d._g.to_numpy()]
    return df.loc[d.index[keep]].sort_values("dcm_path").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Cache: atomic shards, progress, consolidation
# --------------------------------------------------------------------------- #


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def write_json_atomic(path, obj):
    tmp = os.path.join(os.path.dirname(path), f".tmp-{os.path.basename(path)}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_parquet_atomic(df, path):
    """Write a temporary file next to `path`, then rename: a reader (or a resumed scan) sees the
    whole file or none of it, even if the scan is killed mid-write.
    """
    table = pa.Table.from_pandas(df.astype(DTYPES)[INDEX_COLUMNS], schema=INDEX_SCHEMA, preserve_index=False)
    tmp = os.path.join(os.path.dirname(path), f".tmp-{os.path.basename(path)}")
    with open(tmp, "wb") as f:
        pq.write_table(table, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shard_dir(cache_dir):
    return os.path.join(cache_dir, "shards")


def shard_files(cache_dir):
    """Shards in write order: names are shard-<run id (UTC timestamp)>-<chunk>.parquet."""
    return sorted(glob.glob(os.path.join(shard_dir(cache_dir), "shard-*.parquet")))


def run_of(shard_path):
    return os.path.basename(shard_path).split("-")[1]


def read_frame(path, columns=None):
    return pq.read_table(path, columns=columns).to_pandas(types_mapper=ARROW_TO_PANDAS.get)


def read_shards(cache_dir, columns=None):
    """Every cached row, with its run (`_run`) and write order (`_seq`)."""
    parts = []
    for i, path in enumerate(shard_files(cache_dir)):
        parts.append(read_frame(path, columns).assign(_run=run_of(path), _seq=i))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=DTYPES[c]) for c in columns or INDEX_COLUMNS}).assign(
            _run=pd.Series(dtype=object), _seq=pd.Series(dtype=int))
    return pd.concat(parts, ignore_index=True)


def latest_rows(rows):
    """The latest row per dcm_path. One run scans each path once; a path on two rows of the same
    run with different content means the cache is corrupt, so it raises instead of choosing.
    """
    dup = rows.duplicated(["_run", "dcm_path"], keep=False)
    if dup.any():
        content = rows[dup].drop(columns="_seq").drop_duplicates()
        if content.duplicated(["_run", "dcm_path"]).any():
            n = content.loc[content.duplicated(["_run", "dcm_path"]), "dcm_path"].nunique()
            raise ValueError(f"{n} DICOMs have conflicting rows within one scan run; refusing to pick one.")
    return rows.sort_values("_seq", kind="stable").drop_duplicates("dcm_path", keep="last")


def load_index(path):
    """The acquisition index as a DataFrame with nullable dtypes: a consolidated parquet file, or
    a cache directory (consolidated in memory, latest row per dcm_path).
    """
    if os.path.isdir(path):
        if not shard_files(path):
            raise ValueError(f"{path} holds no scan shards.")
        return latest_rows(read_shards(path)).drop(columns=["_run", "_seq"]).reset_index(drop=True)
    return read_frame(path)


class ScanLock:
    """One scan per cache directory at a time: two concurrent scans would interleave shards
    and race on progress.json. A stale lock (crashed scan) must be removed by hand.
    """

    def __init__(self, cache_dir):
        self.path = os.path.join(cache_dir, "scan.lock")

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"{self.path} exists: another scan is using this cache directory. "
                             "If no scan is running (e.g. it crashed), delete the lock and retry.") from None
        with os.fdopen(fd, "w") as f:
            f.write(f"pid {os.getpid()} on {socket.gethostname()} since {utc_now().isoformat()}\n")
        return self

    def __exit__(self, *exc):
        os.remove(self.path)


def stat_identity(path):
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None, None


def unchanged(now, cached):
    """(size, mtime_ns) on disk vs in the cache. A file missing then and now is unchanged."""
    return all((a is None and pd.isna(b)) or (a is not None and not pd.isna(b) and int(b) == a)
               for a, b in zip(now, cached))


def plan_scan(paths, cached, check_changed=False, retry_errors=False, stat=stat_identity, workers=8,
              timeout=SCAN_TIMEOUT):
    """Split the selected paths into reused cache rows and paths to (re)scan, with counts.

    A cached row is reused when it was written by this SCANNER_VERSION; with `check_changed`,
    its file must also still have the cached size and mtime (a stat per file, on `workers` threads;
    PoolBroken when no stat answers for `timeout` s); with `retry_errors`, rows that failed to read
    (including accepted worker failures) are rescanned.
    """
    sel = pd.DataFrame({"dcm_path": pd.array(list(paths), dtype="string")})
    c = sel.merge(cached[CACHE_KEY_COLUMNS], on="dcm_path", how="left")
    new = c.scanner_version.isna().to_numpy(bool)
    stale = (~new & (c.scanner_version != SCANNER_VERSION).fillna(True)).to_numpy(bool)
    current = ~new & ~stale
    errors = np.zeros(len(c), bool)
    if retry_errors:
        errors = current & (c.read_status == "error").fillna(False).to_numpy(bool)
    changed = np.zeros(len(c), bool)
    if check_changed:
        idx = np.flatnonzero(current & ~errors)
        ids = thread_map(stat, c.dcm_path.iloc[idx].tolist(), max(1, workers), timeout)
        for j, now in zip(idx, ids):
            changed[j] = not unchanged(now, (c.source_size.iloc[j], c.source_mtime_ns.iloc[j]))
    todo = new | stale | errors | changed
    counts = {"selected": int(len(c)), "new": int(new.sum()), "stale_version": int(stale.sum()),
              "changed": int(changed.sum()), "retried_errors": int(errors.sum()), "reused": int((~todo).sum())}
    return sorted(c.dcm_path[todo].tolist()), counts


class ScanIncomplete(RuntimeError):
    """DICOMs left without a row after every attempt (deadline overrun, worker death): they are not
    cached, so the next run scans them again; every row read is cached."""


def run_scan(paths, cache_dir, workers=8, chunk_size=5000, max_retries=3, check_changed=False,
             retry_errors=False, limit_chunks=None, backoff=0.5, provenance=None, log=print,
             timeout=SCAN_TIMEOUT, worker_retries=WORKER_RETRIES, accept_failures=0):
    """Scan `paths` into `cache_dir`, resuming from what is cached. Returns the run summary.

    Files are read by `workers` processes (header reads are I/O bound), each read under a deadline
    of `timeout` s (`supervised_pool.SupervisedPool`): a worker that overruns it or dies is killed and
    replaced, and the file tried up to `worker_retries` more times. Every `chunk_size` rows are written
    as one atomic shard before progress.json is updated, so a killed scan loses at most the rows in
    flight. Nothing to do means nothing is read.

    A file still without a row after its attempts is unresolved: it gets no row, so a rerun scans it
    again, and ScanIncomplete is raised once everything else is cached. At most `accept_failures` of
    them are instead recorded as read errors (UNRESOLVED_REASONS), which `--retry-errors` retries.
    """
    if workers < 1 or chunk_size < 1 or max_retries < 0 or worker_retries < 0 or accept_failures < 0:
        raise ValueError("--workers and --chunk-size must be positive; --max-retries, --worker-retries and "
                         "--accept-scan-failures non-negative.")
    if not timeout > 0:
        raise ValueError(f"--timeout must be positive, got {timeout}.")
    configure_pydicom()
    os.makedirs(shard_dir(cache_dir), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "runs"), exist_ok=True)
    incomplete = None
    with ScanLock(cache_dir):
        run_id = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        previous = [run_of(p) for p in shard_files(cache_dir)]
        if previous and max(previous) >= run_id:
            raise RuntimeError("The clock is behind the newest shard; shard order would no longer be write order.")
        cached = latest_rows(read_shards(cache_dir, CACHE_KEY_COLUMNS))
        todo, plan = plan_scan(paths, cached, check_changed, retry_errors, workers=workers, timeout=timeout)
        n_chunks = -(-len(todo) // chunk_size)
        n_run = n_chunks if limit_chunks is None else min(limit_chunks, n_chunks)
        todo_run = todo[: n_run * chunk_size]
        log(f"[scan]   {plan['selected']} selected: {plan['reused']} reused from the cache, {plan['new']} new, "
            f"{plan['stale_version']} from another scanner version, {plan['changed']} changed on disk, "
            f"{plan['retried_errors']} earlier read errors retried -> {len(todo)} to scan in {n_chunks} "
            f"chunks ({n_run} this run).")

        summary = {"run_id": run_id, "scanner_version": SCANNER_VERSION, "status": "running",
                   "started_utc": utc_now().isoformat(timespec="seconds"), "workers": workers,
                   "chunk_size": chunk_size, "max_retries": max_retries, "check_changed": check_changed,
                   "retry_errors": retry_errors, "limit_chunks": limit_chunks, "timeout_s": timeout,
                   "worker_retries": worker_retries, "accept_scan_failures": accept_failures, "plan": plan,
                   "to_scan": len(todo), "chunks_total": n_chunks, "chunks_done": 0, "files_scanned": 0,
                   "failure_reason": {}, "acq_category": {}, "failed_attempts": {}, "unresolved": {},
                   "elapsed_s": 0.0, **(provenance or {})}
        progress = os.path.join(cache_dir, "progress.json")
        write_json_atomic(progress, summary)
        # The run's scanner version travels with each task, so every row of the run carries it.
        scan = functools.partial(scan_file, max_retries=max_retries, backoff=backoff, version=SCANNER_VERSION)
        start = last = time.monotonic()

        def write(done):
            nonlocal last
            done.sort(key=lambda r: r[0])
            rows = rows_frame(row for _, row in done)
            i = summary["chunks_done"]
            write_parquet_atomic(rows, os.path.join(shard_dir(cache_dir), f"shard-{run_id}-{i:06d}.parquet"))
            now = time.monotonic()
            dt, last = now - last, now
            summary["chunks_done"] = i + 1
            summary["files_scanned"] += len(rows)
            for key in ("failure_reason", "acq_category"):
                for k, n in rows[key].dropna().value_counts().items():
                    summary[key][k] = summary[key].get(k, 0) + int(n)
            summary["elapsed_s"] = round(now - start, 3)
            summary["rate_files_per_s"] = round(summary["files_scanned"] / max(summary["elapsed_s"], 1e-9), 2)
            summary["updated_utc"] = utc_now().isoformat(timespec="seconds")
            write_json_atomic(progress, summary)
            n_err = int((rows.read_status == "error").sum())
            log(f"[scan]   chunk {i + 1}/{n_chunks}: {len(rows)} files in {dt:.1f} s "
                f"({len(rows) / max(dt, 1e-9):.1f} files/s), {n_err} read errors; "
                f"{summary['files_scanned']}/{len(todo)} scanned, overall {summary['rate_files_per_s']} files/s.",
                flush=True)

        pool = SupervisedPool(scan, workers, timeout, worker_retries, init=configure_pydicom,
                              log=lambda line: log(f"[scan]   {line}", flush=True))
        buffer, unresolved = [], []
        try:
            with pool:
                try:
                    for i, row in pool.imap_unordered(todo_run):
                        if isinstance(row, Failure):
                            unresolved.append((i, row))
                            continue
                        buffer.append((i, row))
                        if len(buffer) >= chunk_size:
                            write(buffer)
                            buffer = []
                except PoolBroken as e:
                    summary["stopped_by"] = str(e)
                    summary["failed_attempts"] = {k: pool.counts[k] for k in KINDS}
                    if buffer:  # keep what is read
                        write(buffer)
                    raise
            summary["failed_attempts"] = {k: pool.counts[k] for k in KINDS}
            summary["recovered_on_retry"] = pool.counts["recovered"]
            if unresolved:
                unresolved.sort(key=lambda r: r[0])
                failures = [f for _, f in unresolved]
                summary["unresolved"] = {k: sum(f.kind == k for f in failures) for k in KINDS
                                         if any(f.kind == k for f in failures)}
                listing = os.path.join(cache_dir, "runs", f"unresolved-{run_id}.csv")
                pd.DataFrame({"dcm_path": [todo_run[i] for i, _ in unresolved],
                              "failure": [f.kind for f in failures], "attempts": [f.attempts for f in failures],
                              "detail": [f.detail for f in failures]}).to_csv(listing, index=False)
                if len(unresolved) > accept_failures:
                    incomplete = (
                        f"{len(unresolved)} DICOMs are unresolved after {worker_retries + 1} attempts each "
                        f"{summary['unresolved']} (deadline {timeout:g} s per attempt)"
                        + (f", more than --accept-scan-failures {accept_failures}" if accept_failures else "")
                        + f". They have no row, so rerunning the same command scans them again; every other row "
                        f"is cached. They are listed in {listing}. If the same files fail every time, inspect "
                        "them and rerun with --accept-scan-failures N (N at least their number) to record them "
                        "as read errors (" + " / ".join(UNRESOLVED_REASONS.values()) + ").")
                else:
                    summary["accepted_unresolved"] = {UNRESOLVED_REASONS[k]: n for k, n in summary["unresolved"].items()}
                    buffer += [(i, error_row(empty_row(todo_run[i]), UNRESOLVED_REASONS[f.kind]))
                               for i, f in unresolved]
                    log(f"[scan]   --accept-scan-failures {accept_failures}: {len(unresolved)} unresolved DICOMs "
                        f"recorded as read errors {summary['accepted_unresolved']}; --retry-errors scans them again.")
            if buffer:
                write(buffer)
            summary["status"] = ("incomplete" if incomplete else "complete" if n_run == n_chunks else "partial")
        except BaseException:
            summary["status"] = "interrupted"
            raise
        finally:
            summary["finished_utc"] = utc_now().isoformat(timespec="seconds")
            write_json_atomic(progress, summary)
            write_json_atomic(os.path.join(cache_dir, "runs", f"run-{run_id}.json"), summary)
    if incomplete:
        raise ScanIncomplete(incomplete)
    if summary["status"] == "partial":
        log(f"[scan]   stopped after {n_run} chunks (--limit-chunks); re-run the same command to continue.")
    return summary


def consolidate(cache_dir, out=None):
    """Write `acquisition_index.parquet` (latest row per dcm_path, sorted) and its provenance JSON."""
    out = out or os.path.join(cache_dir, "acquisition_index.parquet")
    files = shard_files(cache_dir)
    if not files:
        raise ValueError(f"{cache_dir} holds no scan shards.")
    rows = read_shards(cache_dir)
    index = latest_rows(rows).drop(columns=["_run", "_seq"]).sort_values("dcm_path").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_parquet_atomic(index, out)
    versions = {str(k): int(n) for k, n in index.scanner_version.value_counts().items()}
    info = {
        "created_utc": utc_now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "current_scanner_version": SCANNER_VERSION,
        "rows": int(len(index)),
        "rows_by_scanner_version": versions,
        "rows_from_other_scanner_versions": int((index.scanner_version != SCANNER_VERSION).sum()),
        "shards": len(files), "runs": sorted({run_of(f) for f in files}),
        "rows_in_shards": int(len(rows)),
        "acq_category": {str(k): int(n) for k, n in index.acq_category.value_counts().items()},
        "output": {"file": os.path.basename(out), "bytes": os.path.getsize(out), "sha256": sha256(out)},
    }
    write_json_atomic(os.path.splitext(out)[0] + ".json", info)
    return index, info


# --------------------------------------------------------------------------- #
#  Report (aggregates only)
# --------------------------------------------------------------------------- #


def value_counts(series):
    return {str(k): int(n) for k, n in series.astype("string").fillna(MISSING).value_counts().items()}


def flag_counts(series):
    s = series.astype("boolean")
    return {"true": int(s.fillna(False).sum()), "false": int((~s).fillna(False).sum()), "unknown": int(s.isna().sum())}


def crosstab(df, row, top=None):
    """Counts of `row` x acq_category, rows ordered by total (or by value for `year`), with a total column."""
    keys = df[row].astype("string").fillna(MISSING)
    t = pd.crosstab(keys, df.acq_category.astype("string").fillna(MISSING))
    t = t.reindex(columns=[c for c in CATEGORIES if c in t] + [c for c in t if c not in CATEGORIES], fill_value=0)
    t["total"] = t.sum(axis=1)
    t = t.sort_index() if row == "year" else t.sort_values("total", ascending=False, kind="stable")
    if top is not None and len(t) > top:
        rest = t.iloc[top:].sum().rename(f"<{len(t) - top} others>")
        t = pd.concat([t.iloc[:top], rest.to_frame().T])
    return t.rename_axis(row)


def table_json(t):
    return {str(k): {str(c): int(x) for c, x in row.items()} for k, row in t.iterrows()}


def build_report(index, meta=None, coverage=None):
    """Aggregate coverage / error report of an index (optionally joined to metadata columns
    year, brand, model, predicted_class). Counts only: no path or identifier.
    """
    df = index.copy()
    types, formats = (df[c].astype("string").fillna(MISSING).replace("", "-")
                      for c in ("region_data_types", "region_spatial_formats"))
    df["regions"] = "types " + types + " / formats " + formats
    report = {
        "rows": int(len(df)),
        "scanner_version": value_counts(df.scanner_version),
        "read_status": value_counts(df.read_status),
        "failure_reason": value_counts(df.failure_reason.dropna()),
        "acq_category": value_counts(df.acq_category),
        "acq_confidence": value_counts(df.acq_confidence),
        "acq_rule": value_counts(df.acq_rule),
        "flags": {f: flag_counts(df[f]) for f in FLAGS},
        "has_region_sequence": flag_counts(df.has_region_sequence),
        "region_types_by_formats": table_json(crosstab(df, "regions", top=40)),
        "tables": {},
    }
    if "region_pairs" in df:
        report["tables"]["region_pairs"] = crosstab(df, "region_pairs", top=40)
    report["tables"]["photometric_interpretation"] = crosstab(df, "photometric_interpretation", top=15)
    report["tables"]["manufacturer (DICOM header)"] = crosstab(df, "manufacturer", top=25)
    report["tables"]["model (DICOM header)"] = crosstab(df, "model_name", top=40)
    if meta is not None:
        m = df.merge(meta, on="dcm_path", how="inner")
        report["rows_joined_to_metadata"] = int(len(m))
        for col, top in (("year", None), ("brand", 25), ("model", 40), ("predicted_class", 40)):
            if col in m:
                report["tables"][f"{col} (metadata)"] = crosstab(m, col, top=top)
        if {"brand", "model"} <= set(m):
            brand, model = (m[c].astype("string").fillna(MISSING) for c in ("brand", "model"))
            m["brand / model"] = brand + " / " + model
            report["tables"]["brand / model (metadata)"] = crosstab(m, "brand / model", top=40)
        if "year" in m:
            report["failure_reason_by_year"] = table_json(
                pd.crosstab(m.year.fillna(MISSING), m.failure_reason.fillna("none")))
    if coverage is not None:
        report["coverage"] = coverage
    return report


def md_table(df):
    """`markdown_table` of a count table with "|" escaped in its labels, so a header value (e.g. a
    model name) cannot break the table. The cells are counts.
    """
    def esc(x):
        return x.replace("|", "\\|") if isinstance(x, str) else x

    return markdown_table(df.rename(index=esc, columns=esc).rename_axis(esc(df.index.name)))


def report_markdown(report):
    lines = ["# Acquisition-mode index report", "",
             "Aggregate counts only: no file paths, patient or study identifiers.", "",
             f"- Created (UTC): {utc_now().isoformat(timespec='seconds')}",
             f"- Current scanner version: {SCANNER_VERSION}; rows by version: {report['scanner_version']}",
             f"- Rows: {report['rows']}"]
    if "coverage" in report:
        lines += ["", "## Coverage of the selection", ""]
        lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in report["coverage"].items()]

    def counts(title, d, name):
        if d:
            nonlocal lines
            lines += ["", f"## {title}", "", md_table(pd.Series(d, name="rows").rename_axis(name).to_frame())]

    counts("Read status", report["read_status"], "read_status")
    counts("Failure reasons", report["failure_reason"], "failure_reason")
    counts("Acquisition category", report["acq_category"], "acq_category")
    counts("Confidence", report["acq_confidence"], "acq_confidence")
    counts("Rule that decided the category", report["acq_rule"], "acq_rule")
    flags = pd.DataFrame(report["flags"]).T.rename_axis("flag")
    lines += ["", "## Flags (unknown = header missing, unreadable or contradictory)", "", md_table(flags)]
    lines += ["", "## Region data types and spatial formats (sorted unique codes per DICOM)", "",
              md_table(pd.DataFrame(report["region_types_by_formats"]).T.rename_axis("region codes"))]
    for name, t in report["tables"].items():
        lines += ["", f"## By {name}", "", md_table(t)]
    if "failure_reason_by_year" in report:
        by_year = pd.DataFrame(report["failure_reason_by_year"]).T.fillna(0).astype(int)
        lines += ["", "## Failure reason by year", "", md_table(by_year.rename_axis("year"))]
    return "\n".join(lines) + "\n"


def metadata_for(metadata, paths, columns, batch_size=BATCH_SIZE):
    """Metadata columns (and `year`) of the rows whose dcm_path is in `paths`; unique dcm_path required."""
    names = table_columns(metadata)
    need = ["dcm_path"] + ["date" if c == "year" else c for c in columns if (c if c != "year" else "date") in names]
    parts = []
    for batch in iter_table(metadata, list(dict.fromkeys(need)), batch_size):
        dcm = clean_text(batch["dcm_path"])
        keep = dcm.isin(paths).fillna(False).to_numpy(bool)
        parts.append(pd.DataFrame({c: (dcm if c == "dcm_path" else clean_text(batch[c]))[keep] for c in need}))
    meta = pd.concat(parts, ignore_index=True)
    if "date" in meta:
        meta = meta.assign(year=meta.date.str[:4]).drop(columns="date")
    dup = meta.dcm_path.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{meta.loc[dup, 'dcm_path'].nunique()} dcm_path values appear on more than one "
                         "metadata row.")
    return meta


def write_report(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = {k: v for k, v in report.items() if k != "tables"}
    data["tables"] = {k: table_json(t) for k, t in report["tables"].items()}
    with open(os.path.join(out_dir, "acquisition_report.json"), "w") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(out_dir, "acquisition_report.md"), "w") as f:
        f.write(report_markdown(report))


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def input_provenance(path, checksum=True):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "sha256": sha256(path) if checksum else None}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Scan (or resume scanning) DICOM headers into a cache directory.")
    s.add_argument("--metadata", help="master_metadata.parquet (or .csv); required by --select and --stratify.")
    s.add_argument("--select", choices=SELECTIONS,
                   help="usable-tte, or excluded-tte (TTE rows without a usable video).")
    s.add_argument("--paths-file", help="Scan these DICOMs instead (one dcm_path per line).")
    s.add_argument("--sample", type=int, help="Scan a stratified sample of this many DICOMs.")
    s.add_argument("--seed", type=int, default=0, help="Sampling seed (default 0).")
    s.add_argument("--stratify", help="Comma-separated strata for --sample: `year` or metadata columns "
                                      "(default year,brand with --metadata, none without).")
    s.add_argument("--cache-dir", required=True,
                   help="Shards, progress and run records; must be outside the repository.")
    s.add_argument("--workers", type=int, default=8, help="Reader processes (default 8; benchmark before raising).")
    s.add_argument("--chunk-size", type=int, default=5000, help="Files per shard (default 5000).")
    s.add_argument("--max-retries", type=int, default=3, help="Retries of a transient I/O error (default 3).")
    s.add_argument("--timeout", type=float, default=SCAN_TIMEOUT, metavar="SECONDS",
                   help=f"Deadline for one header read in a worker (default {SCAN_TIMEOUT:g}); a worker that "
                        "overruns it or dies is killed and replaced, and the file tried again.")
    s.add_argument("--worker-retries", type=int, default=WORKER_RETRIES,
                   help=f"Further attempts for a file whose worker timed out or died (default {WORKER_RETRIES}).")
    s.add_argument("--accept-scan-failures", type=int, default=0, metavar="N",
                   help="Once inspected, record up to N files still unresolved after every attempt (listed in "
                        "runs/unresolved-<run>.csv) as read errors; without it, or with more, the scan exits "
                        "nonzero and a rerun tries them again.")
    s.add_argument("--check-changed", action="store_true", help="Stat cached files and rescan changed size/mtime.")
    s.add_argument("--retry-errors", action="store_true", help="Rescan cached rows whose read failed.")
    s.add_argument("--limit-chunks", type=int, help="Stop after this many chunks (bounded runs, benchmarks).")
    s.add_argument("--no-input-checksum", action="store_true",
                   help="Skip the SHA-256 of --metadata in the run record.")

    c = sub.add_parser("consolidate", help="Write one acquisition_index.parquet: the latest row per DICOM.")
    c.add_argument("--cache-dir", required=True)
    c.add_argument("--out", help="Default <cache-dir>/acquisition_index.parquet; must be outside the repository.")

    r = sub.add_parser("report", help="Aggregate coverage / error report (markdown + json).")
    r.add_argument("--index", required=True, help="acquisition_index.parquet or a cache directory.")
    r.add_argument("--metadata", help="Adds year, brand/model and view breakdowns (joined on dcm_path).")
    r.add_argument("--select", choices=SELECTIONS, help="Report this selection (with its coverage) only.")
    r.add_argument("--out-dir", required=True, help="Must be outside the repository.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "scan":
        ensure_outside_repo(args.cache_dir, "--cache-dir")
        if args.select is None and args.paths_file is None:
            raise SystemExit("Give --select and/or --paths-file.")
        strata = parse_strata(args.stratify) if args.stratify is not None else (
            ["year", "brand"] if args.metadata and args.sample else [])
        paths = read_paths_file(args.paths_file) if args.paths_file else None
        sel, info = select_rows(args.metadata, args.select, paths, strata if args.sample else [])
        print(f"[select] {info['rows_selected']} DICOMs selected ({info['selection']}); file_id: {info['file_id']}.")
        if args.sample is not None:
            sel = stratified_sample(sel, strata, args.seed, n=args.sample)
            info.update(sample=args.sample, seed=args.seed, stratify=strata, rows_sampled=int(len(sel)))
            print(f"[select] stratified sample of {len(sel)} by {strata or 'nothing'} (seed {args.seed}).")
        meta_info = input_provenance(args.metadata, not args.no_input_checksum) if args.metadata else None
        provenance = {"command": sys.argv if argv is None else argv, "git_commit": git_commit(),
                      "pydicom": __import__("pydicom").__version__, "selection": info, "metadata": meta_info}
        try:
            summary = run_scan(sel.dcm_path.tolist(), args.cache_dir, args.workers, args.chunk_size, args.max_retries,
                               args.check_changed, args.retry_errors, args.limit_chunks, provenance=provenance,
                               timeout=args.timeout, worker_retries=args.worker_retries,
                               accept_failures=args.accept_scan_failures)
        except (ScanIncomplete, PoolBroken) as e:
            raise SystemExit(f"[scan]   stopped: {e}") from None
        print(f"[scan]   {summary['status']}: {summary['files_scanned']} scanned, "
              f"categories {summary['acq_category']}, read failures {summary['failure_reason']}.")
    elif args.command == "consolidate":
        out = args.out or os.path.join(args.cache_dir, "acquisition_index.parquet")
        ensure_outside_repo(out, "--out")
        index, info = consolidate(args.cache_dir, out)
        print(f"[index]  {info['rows']} rows from {info['shards']} shards -> {info['output']['file']}; "
              f"{info['rows_from_other_scanner_versions']} rows from another scanner version.")
    else:
        ensure_outside_repo(args.out_dir, "--out-dir")
        if args.select and not args.metadata:
            raise SystemExit("--select needs --metadata.")
        index = load_index(args.index)
        meta, coverage = None, None
        if args.metadata:
            cols = [c for c in ("year", "brand", "model", "predicted_class")
                    if (c if c != "year" else "date") in table_columns(args.metadata)]
            if args.select:
                meta, _ = select_rows(args.metadata, args.select, None, cols)
                known = meta.dcm_path.isin(set(index.dcm_path))
                current = meta.dcm_path.isin(set(index.dcm_path[index.scanner_version == SCANNER_VERSION]))
                coverage = {"selection": args.select, "selected": int(len(meta)), "indexed": int(known.sum()),
                            "not_indexed": int((~known).sum()), "indexed_by_current_scanner": int(current.sum())}
                index = index[index.dcm_path.isin(set(meta.dcm_path))]
            else:
                meta = metadata_for(args.metadata, set(index.dcm_path), cols)
        report = build_report(index, meta, coverage)
        write_report(report, args.out_dir)
        print(f"[report] {report['rows']} rows -> acquisition_report.md / .json in --out-dir.")


if __name__ == "__main__":
    main()
