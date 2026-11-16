# data/build_still_manifest.py

"""The RADAR still table: one row per Doppler still that carries a sonographer measurement, from the DeepEcho
still-measurement dataset (`meas_<year>.parquet`, one row per measurement read from a still's burned-in box).

Each still gets:

- `still_id`: a hash of its SOPInstanceUID and PNG path (the token cache key; never a path or a UID in clear; the
  path is in the hash because the archive holds a few SOPInstanceUIDs shared by two DICOM files);
- `study_id`: the PACS StudyInstanceUID of RADAR's clip and caption tables. The still dataset's own
  `StudyInstanceUID` is the extractor's and never matches it, so the study is found through the DICOM study folder:
  `dirname(dcm_path)` is the folder of the same study's videos in the video metadata (`--metadata`, the
  `master_metadata.parquet` of the video pipeline), whose `study_uid` is RADAR's `study_id`. A folder with two
  `study_uid`s is ambiguous and left out;
- `still_tag`: the structure tag of the routing registry's `still_structs` (`still_tag`): the label's structure,
  with MV pulsed-wave stills measured for E, A, E/A or DT as `MV_INFLOW` and annular e'/a'/s' stills as `TDI`;
  `still_type` = `<mode>_<tag>` (`cw_AV`, `pw_LVOT`, `tdi_TDI`...);
- the crop and its velocity scale (`spec_x0..spec_y1`, `v_top_ms`, `v_bot_ms`) and the label boxes (`text_boxes`,
  JSON), which `src/datasets/still_preprocess.py` masks;
- wide targets `y_<STRUCT>_<QUANTITY>` in fixed units (m/s, cm, mmHg, ms), each the mean over the still's beats,
  after unit normalisation; a value outside `RANGES` is dropped (counted in `build_info.json`). A negative value is
  not a target unless its quantity is named in `--magnitude` (then its absolute value is); `signed_values` (JSON)
  keeps every value as read, sign included, and `build_info.json` counts the negatives by quantity and vendor
  (`negative_values`), which is what to read before choosing `--magnitude`;
- optional EchoNet columns (`--echonet`: `pred_cms`, `heat_max`, optionally `heatmap_path` already mapped to the crop):
  `echonet_vmax_ms`, `echonet_conf`, `echonet_present`, `echonet_heatmap`;
- `split`: the patient's split in the archive-wide patient split (`--patient-splits`: one patient, one split, in
  every project), which RADAR's full cohort split (`--splits`, as for the caption builder) must agree with; a
  patient who is in neither gets a split drawn from a salted hash of the patient key with `--fractions`. All studies of a patient
  share one split. A patient with two splits in `--splits`, or a study of `--splits` whose still split disagrees,
  stops the build.

`demographics.csv`: one row per study of the still table: `age_years` (study date minus birth date, to 0.1 year;
the birth date itself is never written), `sex` (M, F or empty) and `bsa_m2` (Mosteller, when weight and height are
known). `patient_hash` (salted hash of the patient key) replaces the patient id in both tables.

Example:
    python data/build_still_manifest.py \\
        --meas <still_measurements>/meas_2024.parquet <still_measurements>/meas_2023.parquet \\
        --metadata <preprocessing_results>/master_metadata.parquet \\
        --splits <manifests>/studies.csv --out-dir <stills_dir>

The tables hold image paths with patient folders: keep them outside the repository (enforced) and on the NAS.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from build_manifests import REPO_ROOT, ensure_outside_repo, normalize_id
from make_study_targets import patient_key

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from src.datasets.radar_study_dataset import file_sha256  # noqa: E402
from src.datasets.still_preprocess import velocity_scale  # noqa: E402

SPLITS = ("train", "val", "test")
TEMPORAL_SPLIT = "test_temporal"     # the stills of the held-out years (see `assign_splits`)
SALT_ENV = "RADAR_STILLS_SALT"
SALT_MIN_LENGTH = 32
MAPPING_VERSION = 1

# The label's structure (as read by the OCR) -> the registry's still tag. Labels not listed (AO, LV, RV, None...)
# give no tag: their stills feed no structure until someone decides where they belong.
STRUCT_TAGS = {"AV": "AV", "AS": "AV", "LVOT": "LVOT", "AVA": "AVA", "AVAI": "AVA", "AR": "AR", "MV": "MV",
               "MS": "MV", "MVA": "MVA", "MR": "MR", "TV": "TV", "TR": "TR", "PV": "PV", "PA": "PV", "MPA": "PV",
               "PR": "PR", "RVOT": "RVOT", "DAO": "DAO", "AAO": "DAO"}
DIASTOLIC = {"E", "A", "E/A", "DT"}
TISSUE = {"E'", "A'", "S'"}
MODES = {"CW": "cw", "PW_TDI": "pw", "MMODE": "mm"}

# Targets: (structure tag of the row, quantity) -> unit family. Velocities in m/s, VTI in cm, gradients in mmHg,
# times in ms.
QUANTITY_UNITS = {"VMAX": "velocity", "VMEAN": "velocity", "E": "velocity", "A": "velocity", "VTI": "length",
                  "MAXPG": "pressure", "MEANPG": "pressure", "PHT": "time", "DT": "time"}
TARGET_STRUCTS = ("AV", "LVOT", "MV", "TR", "TV", "PV", "PR", "AR")
RANGES = {"velocity": (0.1, 7.0), "length": (2.0, 200.0), "pressure": (0.0, 200.0), "time": (20.0, 2000.0)}
TARGET_COLUMNS = tuple(f"y_{s}_{q}" for s in TARGET_STRUCTS for q in QUANTITY_UNITS)
# Quantities that determine each other: never one as an input and another as a target (MaxPG = 4 Vmax^2).
LEAK_GROUPS = ({"VMAX", "MAXPG"}, {"VTI", "VMEAN", "MEANPG"}, {"E", "A", "DT", "PHT"})

STILL_COLUMNS = ("still_id", "study_id", "patient_hash", "split", "png_path", "dcm_path", "year", "manufacturer",
                 "rows", "cols", "mode", "struct_raw", "still_tag", "still_type", "n_rows", "spec_x0", "spec_y0",
                 "spec_x1", "spec_y1", "spec_dy", "spec_ref_y0", "v_top_ms", "v_bot_ms", "text_boxes", "signed_values",
                 "echonet_present", "echonet_vmax_ms", "echonet_conf", "echonet_heatmap", *TARGET_COLUMNS)
DEMOGRAPHIC_COLUMNS = ("study_id", "patient_hash", "age_years", "sex", "bsa_m2")


def blake(text, salt="", size=8):
    return hashlib.blake2b(f"{salt}\0{text}".encode("utf-8"), digest_size=size).hexdigest()


def study_folder(paths):
    """The DICOM study folder of each path (its directory)."""
    return pd.Series(paths, dtype="string").str.replace(r"/[^/]*$", "", regex=True)


def to_unit(value, unit, family):
    """`value` in the family's unit (m/s, cm, mmHg, ms), or NaN when the unit does not belong to the family."""
    unit = unit.strip().lower().replace("²", "2") if isinstance(unit, str) else ""     # a missing unit is NaN or NA
    factors = {"velocity": {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001},
               "length": {"cm": 1.0, "mm": 0.1, "": 1.0},
               "pressure": {"mmhg": 1.0, "": 1.0},
               "time": {"ms": 1.0, "s": 1000.0, "": 1.0}}[family]
    return value * factors[unit] if unit in factors else np.nan


def still_tag(structs, quantities, mode):
    """The still's registry tag from its rows' structures and quantities (see the module docstring), or None."""
    quantities = set(quantities)
    if quantities & TISSUE:
        return "TDI"
    tags = [STRUCT_TAGS[s] for s in structs if s in STRUCT_TAGS]
    if not tags:
        return None
    counts = pd.Series(tags).value_counts()
    tag = sorted(counts[counts == counts.max()].index)[0]
    if tag == "MV" and mode == "PW_TDI" and quantities & DIASTOLIC:
        return "MV_INFLOW"
    return tag


def still_type(mode, tag):
    if tag is None:
        return ""
    return "tdi_TDI" if tag == "TDI" else f"{MODES.get(mode, 'other')}_{tag}"


def targets_of(rows, magnitude=()):
    """({target column: mean value over the still's beats}, {target name: signed values as read}, counts).

    A negative value is kept as read in the signed record and is not a target, unless its quantity is in `magnitude`:
    for those quantities the sign is taken as the direction of flow (a trace below the baseline) and the target is
    the absolute value. `magnitude` is chosen after reading the builder's `negative_values` counts by quantity and
    vendor; it is never applied to every field. A target whose beats disagree in sign on one still is left out
    (`mixed_sign`): a direction convention gives one sign per trace. Counts: `out_of_range`, `negative_dropped`,
    `sign_flipped`, `mixed_sign`."""
    values, signed = {}, {}
    counts = {"out_of_range": 0, "negative_dropped": 0, "sign_flipped": 0, "mixed_sign": 0}
    for row in rows.itertuples():
        family = QUANTITY_UNITS.get(row.quantity)
        if family is None or row.struct not in TARGET_STRUCTS:
            continue
        value = to_unit(float(row.value), row.unit, family)
        if not np.isfinite(value):
            counts["out_of_range"] += 1
            continue
        name = f"{row.struct}_{row.quantity}"
        signed.setdefault(name, []).append(round(float(value), 6))
        values.setdefault(name, []).append((value, row.quantity, family))
    out = {}
    for name, beats in values.items():
        quantity, family = beats[0][1], beats[0][2]
        negative = [v < 0 for v, _, _ in beats]
        if any(negative) and not all(negative):
            counts["mixed_sign"] += 1
            continue
        if all(negative) and quantity not in magnitude:
            counts["negative_dropped"] += len(beats)
            continue
        counts["sign_flipped"] += len(beats) if all(negative) else 0
        low, high = RANGES[family]
        kept = [abs(v) for v, _, _ in beats if low <= abs(v) <= high]
        counts["out_of_range"] += len(beats) - len(kept)
        if kept:
            out[f"y_{name}"] = float(np.mean(kept))
    return out, signed, counts


def negative_values(meas):
    """{"<struct>_<quantity>|<manufacturer>": [negative values, all values]} of the target rows: what to read before
    choosing `--magnitude`. Counts from 1 to 4 are written as "<5"."""
    rows = meas[meas.struct.isin(TARGET_STRUCTS) & meas.quantity.isin(list(QUANTITY_UNITS))]
    value = pd.to_numeric(rows.value, errors="coerce")
    vendor = rows["Manufacturer"].fillna("unknown").astype(str) if "Manufacturer" in rows.columns else "unknown"
    key = rows.struct + "_" + rows.quantity + "|" + vendor
    table = pd.DataFrame({"key": key, "negative": value < 0}).groupby("key").negative.agg(["sum", "size"])
    return {k: ["<5" if 0 < int(n) < 5 else int(n), int(total)] for k, (n, total) in table.iterrows() if n > 0}


def normalize_meas(meas):
    return meas.assign(struct=meas.struct.fillna("").astype(str).str.upper(),
                       quantity=meas.quantity.fillna("").astype(str).str.upper())


def aggregate_stills(meas, magnitude=()):
    """One row per still (png_path) of the measurement table, with its tag, calibration, boxes, targets and the
    signed values the targets came from (`signed_values`, JSON); also the summed counts of `targets_of`."""
    rows, dropped = [], Counter()
    meas = normalize_meas(meas)
    for png, group in meas.groupby("png_path", sort=True):
        first = group.iloc[0]
        tag = still_tag(group.struct, group.quantity, first["mode"])
        targets, signed, lost = targets_of(group, magnitude)
        dropped.update(lost)
        top, bottom = velocity_scale(first.spec_y0, first.spec_y1, first.spec_ref_y0, first.spec_dy)
        boxes = sorted({(int(r.x0), int(r.y0), int(r.x1), int(r.y1)) for r in group.itertuples()})
        rows.append({"still_id": blake(f"{first.get('SOPInstanceUID')}\0{png}"), "png_path": png,
                     "dcm_path": first.dcm_path, "year": first.get("year"), "manufacturer": first.get("Manufacturer"),
                     "rows": first.get("rows"), "cols": first.get("cols"), "mode": first["mode"],
                     "struct_raw": ",".join(sorted(set(group.struct) - {""})), "still_tag": tag or "",
                     "still_type": still_type(first["mode"], tag), "n_rows": len(group),
                     **{c: first[c] for c in ("spec_x0", "spec_y0", "spec_x1", "spec_y1", "spec_dy", "spec_ref_y0")},
                     "v_top_ms": top, "v_bot_ms": bottom, "text_boxes": json.dumps(boxes),
                     "signed_values": json.dumps(signed, sort_keys=True), **targets})
    table = pd.DataFrame(rows)
    if table.still_id.duplicated().any():
        raise SystemExit("Two stills have the same still id.")
    return table, dict(dropped)


def age_years(birth, date):
    """Age at the study in years (0.1), NaN when either date is unreadable or the age is outside [0, 120]."""
    birth = pd.to_datetime(pd.Series(birth, dtype="string").str.strip("'\" "), format="%Y%m%d", errors="coerce")
    date = pd.to_datetime(pd.Series(date, dtype="string").str.strip("'\" "), format="%Y%m%d", errors="coerce")
    age = (date - birth).dt.days / 365.25
    return age.where((age >= 0) & (age <= 120)).round(1)


def link_studies(stills, metadata, salt):
    """`stills` with `study_id` and `patient_hash`, and the per-study demographics, through the study folder.

    A folder that holds two studies links to neither. Nor does a folder, or a study, whose metadata rows name two
    patients: which of them the stills belong to cannot be told, and taking the first row's would put the stills in
    an arbitrary patient's split. Their stills are counted with the unlinked ones."""
    meta = metadata.assign(folder=study_folder(metadata.dcm_path), study_id=normalize_id(metadata.study_uid),
                           patient=patient_key(metadata.mrn))
    per_folder = meta.groupby("folder").study_id.nunique()
    known = meta.dropna(subset=["patient"])
    two_in_folder = known.groupby("folder").patient.nunique()
    two_in_study = known.groupby("study_id").patient.nunique()
    meta = meta[meta.folder.isin(per_folder[per_folder == 1].index)
                & ~meta.folder.isin(two_in_folder[two_in_folder > 1].index)
                & ~meta.study_id.isin(two_in_study[two_in_study > 1].index)]
    meta = meta.drop(columns=["patient"]).drop_duplicates("folder")
    linked = stills.assign(folder=study_folder(stills.dcm_path)).merge(
        meta[["folder", "study_id", "mrn"]], on="folder", how="inner")
    linked["patient"] = patient_key(linked.pop("mrn"))
    linked = linked.dropna(subset=["patient", "study_id"])
    linked["patient_hash"] = [blake(p, salt) for p in linked.patient]
    studies = meta[meta.study_id.isin(set(linked.study_id))].drop_duplicates("study_id")
    height_cm = pd.to_numeric(studies.get("patient_height_m"), errors="coerce") * 100.0
    weight = pd.to_numeric(studies.get("patient_weight_kg"), errors="coerce")
    bsa = np.sqrt(height_cm * weight / 3600.0)
    demo = pd.DataFrame({"study_id": studies.study_id.to_numpy(),
                         "age_years": age_years(studies.birthdate, studies.date).to_numpy(),
                         "sex": studies.sex.where(studies.sex.isin(["M", "F"]), "").fillna("").to_numpy(),
                         "bsa_m2": bsa.where((bsa > 0.5) & (bsa < 3.5)).round(2).to_numpy()})
    patients = linked.drop_duplicates("study_id").set_index("study_id").patient_hash
    demo["patient_hash"] = demo.study_id.map(patients)
    return linked.drop(columns=["folder"]), demo[list(DEMOGRAPHIC_COLUMNS)], len(stills) - len(linked)


def read_salt(salt_file=None, environ=os.environ):
    """The private salt of the patient hash and of the drawn splits: the first line of `salt_file`, else the
    environment variable `RADAR_STILLS_SALT`. There is no default and it is never written to an output: with a salt
    anyone can read (a default in public code, a value in `build_info.json`), a patient hash can be recomputed from a
    list of patient numbers. The file must be outside the repository. Keep the salt: the same salt gives the same
    hashes and the same drawn splits in a later build."""
    if salt_file:
        root, target = os.path.realpath(REPO_ROOT), os.path.realpath(salt_file)
        if os.path.commonpath([root, target]) == root:
            raise SystemExit("--salt-file is inside the repository: a salt kept with public code is not private.")
        with open(salt_file, encoding="utf-8") as f:
            salt = f.readline().strip()
    else:
        salt = environ.get(SALT_ENV, "").strip()
    if len(salt) < SALT_MIN_LENGTH:
        raise SystemExit(f"No private salt: give --salt-file (or {SALT_ENV}) with at least {SALT_MIN_LENGTH} "
                         "characters, for example the output of `python -c \"import secrets; "
                         "print(secrets.token_hex(32))\"`, in a file only you can read.")
    return salt


def salt_check(salt):
    """A keyed digest that tells whether two builds used the same salt, without giving the salt."""
    return hashlib.blake2b(b"radar-stills-salt-check", key=salt.encode("utf-8")[:64], digest_size=8).hexdigest()


def hashed_split(patient, salt, fractions):
    """A split from a salted hash of the patient key: uniform in [0, 1) against the cumulative fractions."""
    u = int(blake(patient, salt), 16) / 16 ** 16
    for name, edge in zip(SPLITS, np.cumsum(fractions)):
        if u < edge:
            return name
    return SPLITS[-1]


def patient_split_table(patient_splits):
    """{patient key: split} of an archive-wide patient split (`patient_id`, `split`): the one table every project
    reads a patient's split from. A patient listed with two splits stops the build."""
    if patient_splits is None:
        return {}
    if not {"patient_id", "split"} <= set(patient_splits.columns):
        raise SystemExit("--patient-splits needs patient_id and split columns.")
    table = patient_splits.assign(patient=patient_key(patient_splits.patient_id),
                                  split=patient_splits.split.astype(str)).dropna(subset=["patient"])
    unknown = sorted(set(table.split) - set(SPLITS))
    if unknown:
        raise SystemExit(f"--patient-splits has splits {unknown}; expected {list(SPLITS)}.")
    if (table.groupby("patient").split.nunique() > 1).any():
        raise SystemExit("--patient-splits lists a patient in two splits.")
    return dict(zip(table.patient, table.split))


def assign_splits(stills, splits, metadata, salt, fractions, temporal_years=(), patient_splits=None):
    """`stills` with `split` (see the module docstring), and counts of inherited and drawn patients.

    `temporal_years`: the stills of these years are a separate temporal test and never train. Such a still gets the
    split `test_temporal` when its patient's split (inherited or drawn, as for any still) is `test`, and when the
    patient has stills in those years only; it is left out when the patient is in `train` or `val`, because a
    temporal test on patients the model trained on would not be held out. The random `test` split keeps the earlier
    years."""
    splits = splits.assign(study_id=normalize_id(splits.study_id), split=splits.split.astype(str))
    known = metadata.assign(study_id=normalize_id(metadata.study_uid), patient=patient_key(metadata.mrn)) \
        if len(metadata) else pd.DataFrame({"study_id": pd.Series(dtype=str), "patient": pd.Series(dtype=str)})
    known = known.dropna(subset=["patient", "study_id"]).drop_duplicates(["study_id", "patient"])
    if "patient_id" in splits.columns:
        # The two sources of a study's patient must agree: a split inherited through a patient number the metadata
        # does not give that study would let the same patient's other studies be drawn into another split.
        pairs = splits.assign(patient=patient_key(splits.patient_id))
        both = pairs.dropna(subset=["patient"]).merge(known[["study_id", "patient"]], on="study_id",
                                                      suffixes=("", "_metadata"))
        agrees = both.assign(same=both.patient == both.patient_metadata).groupby("study_id").same.any()
        if not agrees.all():
            raise SystemExit(f"{int((~agrees).sum())} studies of --splits have a patient_id that the metadata does "
                             "not give them: the patients cannot be told apart reliably, so no split is drawn.")
    else:
        meta = known.drop_duplicates("study_id")
        pairs = splits.merge(meta[["study_id", "patient"]], on="study_id", how="left")
    pairs = pairs.dropna(subset=["patient"])
    per_patient = pairs.groupby("patient").split.nunique()
    if (per_patient > 1).any():
        raise SystemExit(f"{int((per_patient > 1).sum())} patients have studies in two splits of --splits.")
    inherited = dict(zip(pairs.patient, pairs.split))
    cohort_patients = set(inherited)
    # The archive-wide patient split is the first authority: one patient, one split, in every project. The cohort
    # split must agree with it, and only a patient neither table holds is drawn from the hash.
    archive = patient_split_table(patient_splits)
    disagree = [p for p, split in inherited.items() if archive.get(p, split) != split]
    if disagree:
        raise SystemExit(f"{len(disagree)} patients have another split in --patient-splits than in --splits.")
    inherited = {**archive, **inherited}
    out = stills.assign(split=[inherited.get(p) or hashed_split(p, salt, fractions) for p in stills.patient])
    by_study = dict(zip(splits.study_id, splits.split))
    clash = out[out.study_id.map(by_study).notna() & (out.study_id.map(by_study) != out.split)]
    if len(clash):
        raise SystemExit(f"{clash.study_id.nunique()} studies of --splits would get another split for their stills.")
    counts = {"patients_inherited": int(out.loc[out.patient.isin(inherited), "patient"].nunique()),
              "patients_drawn": int(out.loc[~out.patient.isin(inherited), "patient"].nunique())}
    if patient_splits is not None:
        counts["patients_inherited_from_patient_splits"] = int(
            out.loc[out.patient.isin(archive) & ~out.patient.isin(cohort_patients), "patient"].nunique())
    if temporal_years:
        late = pd.to_numeric(out.year, errors="coerce").isin([int(y) for y in temporal_years])
        only_late = ~out.patient.isin(set(out.loc[~late, "patient"])) & ~out.patient.isin(inherited)
        held_out = late & ((out.split == "test") | only_late)
        seen = late & ~held_out
        out.loc[held_out, "split"] = TEMPORAL_SPLIT
        counts.update({"temporal_years": [int(y) for y in temporal_years],
                       "temporal_stills": int(held_out.sum()),
                       "temporal_patients": int(out.loc[held_out, "patient"].nunique()),
                       "temporal_stills_left_out_trained_patients": int(seen.sum())})
        out = out[~seen]
    return out, counts


def add_echonet(stills, echonet):
    """EchoNet's peak prediction (cm/s -> m/s), its heat-map maximum as confidence and the heat-map path, by png."""
    if echonet is None:
        return stills.assign(echonet_present=0, echonet_vmax_ms=np.nan, echonet_conf=np.nan, echonet_heatmap="")
    cols = {"pred_cms": "echonet_vmax_ms", "heat_max": "echonet_conf", "heatmap_path": "echonet_heatmap"}
    e = echonet[["png_path", *[c for c in cols if c in echonet.columns]]].drop_duplicates("png_path").rename(
        columns=cols)
    out = stills.merge(e, on="png_path", how="left")
    out["echonet_vmax_ms"] = out.echonet_vmax_ms / 100.0
    if "echonet_heatmap" not in out.columns:
        out["echonet_heatmap"] = ""
    out["echonet_heatmap"] = out.echonet_heatmap.fillna("")
    out["echonet_present"] = out.echonet_vmax_ms.notna().astype(int)
    return out


def read_meas(paths):
    tables = [pd.read_parquet(p) for p in paths]
    meas = pd.concat(tables, ignore_index=True)
    return meas[meas["mode"].isin(list(MODES))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meas", nargs="+", required=True, help="meas_<year>.parquet files of the still dataset.")
    p.add_argument("--metadata", required=True, help="Video metadata with study_uid, dcm_path, mrn, sex, birthdate, "
                                                     "date (patient_weight_kg, patient_height_m if known).")
    p.add_argument("--splits", required=True, help="RADAR's full cohort split (study_id, split, patient_id if any).")
    p.add_argument("--echonet", help="Optional EchoNet predictions (png_path, pred_cms, heat_max, heatmap_path).")
    p.add_argument("--fractions", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                   help="train/val/test fractions for patients outside --splits.")
    p.add_argument("--patient-splits", help="The archive-wide patient split (patient_id, split): a patient's split "
                                            "is read there first, so that every project places a patient alike; "
                                            "only a patient it does not hold is drawn from the salted hash.")
    p.add_argument("--salt-file", help=f"A private file outside the repository whose first line is the salt of the "
                                       f"patient hash and the drawn splits (else the variable {SALT_ENV}).")
    p.add_argument("--magnitude", nargs="*", default=[], choices=sorted(QUANTITY_UNITS),
                   help="Quantities whose negative values are a direction of flow: their target is the absolute "
                        "value. Choose after reading `negative_values` of a first build; default none.")
    p.add_argument("--temporal-test-years", nargs="*", type=int, default=[],
                   help=f"Years whose stills form the separate `{TEMPORAL_SPLIT}` split and never train.")
    p.add_argument("--out-dir", required=True, help="Output folder; must be outside this repository.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    if not np.isclose(sum(args.fractions), 1.0):
        raise SystemExit("--fractions must sum to 1.")
    salt = read_salt(args.salt_file)
    os.makedirs(args.out_dir, exist_ok=True)
    meas = read_meas(args.meas)
    columns = ["study_uid", "dcm_path", "mrn", "sex", "birthdate", "date", "patient_weight_kg", "patient_height_m"]
    metadata = pd.read_parquet(args.metadata, columns=columns)
    stills, dropped = aggregate_stills(meas, set(args.magnitude))
    stills, demo, unlinked = link_studies(stills, metadata, salt)
    splits = pd.read_csv(args.splits, dtype=str)
    if not {"study_id", "split"} <= set(splits.columns):
        raise SystemExit("--splits needs study_id and split columns.")
    patient_splits = pd.read_csv(args.patient_splits, dtype=str) if args.patient_splits else None
    stills, split_counts = assign_splits(stills, splits, metadata, salt, args.fractions, args.temporal_test_years,
                                         patient_splits)
    demo = demo[demo.study_id.isin(set(stills.study_id))]
    stills = add_echonet(stills, pd.read_parquet(args.echonet) if args.echonet else None)
    for column in STILL_COLUMNS:
        if column not in stills.columns:
            stills[column] = np.nan
    stills = stills[list(STILL_COLUMNS)].sort_values("still_id")
    path, demo_path = os.path.join(args.out_dir, "stills.csv"), os.path.join(args.out_dir, "demographics.csv")
    stills.to_csv(path, index=False)
    demo.sort_values("study_id").to_csv(demo_path, index=False)
    info = {"mapping_version": MAPPING_VERSION, "stills": len(stills), "studies": int(stills.study_id.nunique()),
            "unlinked_stills": int(unlinked), "values": dropped, "magnitude": sorted(args.magnitude),
            "negative_values": negative_values(normalize_meas(meas)), "salt_check": salt_check(salt),
            "fractions": list(args.fractions), **split_counts,
            "per_split": {k: int(v) for k, v in stills.split.value_counts().items()},
            "per_tag": {k or "<none>": int(v) for k, v in stills.still_tag.value_counts().items()},
            "targets": {c: int(stills[c].notna().sum()) for c in TARGET_COLUMNS if stills[c].notna().any()},
            "stills_sha256": file_sha256(path), "demographics_sha256": file_sha256(demo_path),
            "splits_sha256": file_sha256(args.splits), "meas_sha256": [file_sha256(p) for p in args.meas]}
    with open(os.path.join(args.out_dir, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: v for k, v in info.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
