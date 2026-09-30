"""Parse GE Vivid Comprehensive SR (DICOM structured report) echo measurements.

Returns one row per numeric measurement with its group-level and item-level modifiers
(Finding Site, Image Mode, Image View, Direction of Flow, Cardiac Cycle Point,
Measurement Method, Derivation, Selection Status) so that e.g. "Peak Velocity" can be
resolved to TR (Regurgitant Flow, CW), AV (Antegrade Flow, CW) or LVOT (Finding Site LVOT, PW).
"""
from __future__ import annotations
import glob, os
import pandas as pd
import pydicom

MOD_KEYS = ["Finding Site", "Image Mode", "Image View", "Direction of Flow", "Cardiac Cycle Point",
            "Measurement Method", "Derivation", "Selection Status"]


def _cm(item):
    s = getattr(item, "ConceptNameCodeSequence", None)
    return (s[0].CodeValue, s[0].CodeMeaning) if s else ("", "")


def _code_val(item):
    if item.ValueType == "CODE":
        return item.ConceptCodeSequence[0].CodeMeaning
    if item.ValueType == "TEXT":
        return item.TextValue
    return None


def parse_sr_dataset(ds: pydicom.Dataset) -> pd.DataFrame:
    rows = []

    def walk(seq, inherited, path):
        # group-level modifiers: CODE/TEXT children of a container apply to its NUM siblings
        local = dict(inherited)
        for it in seq:
            if it.ValueType in ("CODE", "TEXT"):
                _, n = _cm(it)
                if n in MOD_KEYS:
                    local[n] = _code_val(it)
        for it in seq:
            code, name = _cm(it)
            if it.ValueType == "NUM":
                mv = it.MeasuredValueSequence[0]
                mods = dict(local)
                for ch in getattr(it, "ContentSequence", []) or []:
                    _, n2 = _cm(ch)
                    if n2 in MOD_KEYS:
                        mods[n2] = _code_val(ch)
                rows.append(dict(code=code, name=name, value=float(mv.NumericValue),
                                 unit=mv.MeasurementUnitsCodeSequence[0].CodeValue,
                                 container=" > ".join(path[-2:]), **{k.lower().replace(" ", "_"): mods.get(k) for k in MOD_KEYS}))
            elif hasattr(it, "ContentSequence"):
                walk(it.ContentSequence, local, path + [name])

    walk(ds.ContentSequence, {}, [])
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates()
    return df


def find_sr_files(study_dir: str, max_size_kb: int = 2000):
    """SR objects are the small non-image files; read headers of files < max_size_kb."""
    out = []
    for f in glob.glob(os.path.join(study_dir, "*.dcm")):
        if os.path.getsize(f) > max_size_kb * 1024:
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["Modality", "SOPClassUID"])
            if str(ds.get("Modality")) == "SR" and "SR" in pydicom.uid.UID(ds.SOPClassUID).name:
                out.append(f)
        except Exception:
            pass
    return out


# Canonical measurement -> selector on the parsed rows. Values are converted with the unit of the
# selected row to the unit the paper reports (mm for linear, m/s for velocities, cm2 for areas, cm for VTI).
_TO = {
    "mm": {"mm": 1.0, "cm": 10.0, "m": 1000.0},
    "m/s": {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001},
    "cm": {"cm": 1.0, "mm": 0.1, "m": 100.0},
    "cm2": {"cm2": 1.0, "mm2": 0.01},
}


def _pick_row(df, name=None, code=None, **mods):
    q = df
    if code is not None:
        q = q[q["code"] == code]
    if name is not None:
        q = q[q["name"].str.contains(name, regex=False)]
    for k, v in mods.items():
        if v is None:
            continue
        q = q[q[k].astype(str).str.contains(v, regex=False, na=False)]
    if len(q) == 0:
        return None
    # prefer the sonographer-chosen value ('Mean value chosen', 'User chosen value', ...) over replicates
    pref = q[q["selection_status"].astype(str).str.contains("chosen", case=False, na=False)]
    return (pref if len(pref) else q).iloc[0]


def _pick(df, target_unit=None, **kw):
    row = _pick_row(df, **kw)
    if row is None:
        return None
    v, u = float(row["value"]), str(row["unit"])
    if target_unit is None or u == target_unit:
        return v
    f = _TO.get(target_unit, {}).get(u)
    return v * f if f is not None else None  # unknown unit: drop rather than mis-scale


def canonical_measurements(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {}
    return {
        "ivsd_mm": _pick(df, "mm", code="18154-5"),
        "lvidd_mm": _pick(df, "mm", code="29436-3"),
        "lvids_mm": _pick(df, "mm", code="29438-9"),
        "lvpwd_mm": _pick(df, "mm", code="18152-9"),
        "lvot_diam_mm": _pick(df, "mm", code="G-038F", finding_site="Left Ventricle Outflow Tract"),
        "la_diam_mm": _pick(df, "mm", code="M-02550"),                  # GE 'Diameter' in the LA group (PLAX LA AP dimension)
        "asc_ao_mm": _pick(df, "mm", code="18012-5"),
        "ivc_mm": _pick(df, "mm", code="18006-7"),
        "rv_base_mm": _pick(df, "mm", code="80080-5"),
        "la_area_a4c_cm2": _pick(df, "cm2", code="17977-0"),
        "la_area_a2c_cm2": _pick(df, "cm2", code="GEU-106-0104"),
        "lvef_pct": _pick(df, code="18043-0"),
        "lvot_vmax_ms": _pick(df, "m/s", code="11726-7", finding_site="Left Ventricle Outflow Tract", image_mode="Pulsed"),
        "lvot_vti_cm": _pick(df, "cm", code="20354-7", finding_site="Left Ventricle Outflow Tract"),
        "av_vti_cm": _pick(df, "cm", code="20354-7", finding_site="Aortic Valve"),
        "mv_vti_cm": _pick(df, "cm", code="20354-7", finding_site="Mitral Valve"),
        "tr_vmax_ms": _pick(df, "m/s", code="11726-7", direction_of_flow="Regurgitant", image_mode="Continuous"),
        "av_vmax_ms": _pick(df, "m/s", code="11726-7", direction_of_flow="Antegrade", image_mode="Continuous"),
        "tapse_or_tissue_vel_cms": _pick(df, code="59133-9"),
    }


if __name__ == "__main__":
    import sys, json
    d = sys.argv[1]
    for f in find_sr_files(d):
        df = parse_sr_dataset(pydicom.dcmread(f))
        print(f, len(df), "rows")
        print(json.dumps(canonical_measurements(df), indent=1))
