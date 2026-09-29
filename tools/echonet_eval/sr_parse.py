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


# Canonical measurement -> selector on the parsed rows (first match wins). Values in the
# unit the paper reports (mm for linear, m/s for velocities, cm2 for areas, cm for VTI).
def _pick(df, name=None, code=None, **mods):
    q = df
    if code is not None:
        q = q[q["code"] == code]
    if name is not None:
        q = q[q["name"].str.contains(name, regex=False)]
    for k, v in mods.items():
        if v is None:
            continue
        col = q[k].astype(str)
        q = q[col.str.contains(v, regex=False, na=False)]
    if len(q) == 0:
        return None
    # prefer the sonographer-chosen/mean value over individual replicates
    pref = q[q["selection_status"].astype(str).str.contains("chosen", na=False)]
    return float((pref if len(pref) else q)["value"].iloc[0])


def canonical_measurements(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {}
    to_mm = lambda v, u: None if v is None else (v * 10 if u == "cm" else v)
    def pick_mm(**kw):
        q = df[(df["name"].str.contains(kw["name"], regex=False))] if "name" in kw else df[df["code"] == kw["code"]]
        v = _pick(df, **kw)
        if v is None:
            return None
        u = q["unit"].iloc[0] if len(q) else "mm"
        return to_mm(v, u)
    out = {
        "ivsd_mm": pick_mm(code="18154-5"),
        "lvidd_mm": pick_mm(code="29436-3"),
        "lvids_mm": pick_mm(code="29438-9"),
        "lvpwd_mm": pick_mm(code="18152-9"),
        "lvot_diam_mm": pick_mm(code="G-038F", finding_site="Left Ventricle Outflow Tract"),
        "la_diam_mm": pick_mm(code="M-02550"),                       # GE 'Diameter' in the LA group (PLAX LA AP dimension)
        "asc_ao_mm": pick_mm(code="18012-5"),
        "ivc_mm": pick_mm(code="18006-7"),
        "rv_base_mm": pick_mm(code="80080-5"),
        "la_area_a4c_cm2": _pick(df, code="17977-0"),
        "la_area_a2c_cm2": _pick(df, code="GEU-106-0104"),
        "lvef_pct": _pick(df, code="18043-0"),
        "lvot_vmax_ms": _pick(df, code="11726-7", finding_site="Left Ventricle Outflow Tract", image_mode="Pulsed"),
        "lvot_vti_cm": _pick(df, code="20354-7", finding_site="Left Ventricle Outflow Tract"),
        "tr_vmax_ms": _pick(df, code="11726-7", direction_of_flow="Regurgitant", image_mode="Continuous"),
        "av_vmax_ms": _pick(df, code="11726-7", direction_of_flow="Antegrade", image_mode="Continuous"),
        "tapse_or_tissue_vel_cms": _pick(df, code="59133-9"),
    }
    return out


if __name__ == "__main__":
    import sys, json
    d = sys.argv[1]
    for f in find_sr_files(d):
        df = parse_sr_dataset(pydicom.dcmread(f))
        print(f, len(df), "rows")
        print(json.dumps(canonical_measurements(df), indent=1))
