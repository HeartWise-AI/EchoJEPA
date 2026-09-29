"""DICOM loading and pre-processing for running the EchoNet (Cedars/Stanford) models on
MHI GE Vivid exports.

Key pieces
- read_frames(): multi-frame or single-frame DICOM -> RGB uint8 frames (YBR_FULL_422 handled).
- regions(): SequenceOfUltrasoundRegions -> list of dicts (type, bbox, physical deltas, ref pixel).
  RegionDataType: 1 tissue (2D), 2 colour flow, 3 PW spectral Doppler, 4 CW spectral Doppler.
- to_canvas_3x4(): the fix for the 2D caliper models. They were trained on 480x640 (3:4)
  full-screen Cedars frames; GE exports are 708x1016 (1.43:1). We cut the 2D sector bounding
  box out of the frame (region tags), place it on a black 3:4 canvas with a small margin and
  resize the canvas to 640x480 with OpenCV. The uniform scale keeps the pixel calibration:
  distance_cm = distance_px_640 * (canvas_w / 640) * PhysicalDeltaX_cm.
- doppler_strip(): the Doppler image with the Cedars ECG-trace masking and the strip origin.
- LabelOCR: EasyOCR on the GE measurement label box (top-right) -> canonical trace type and
  the on-screen machine value (used both to route the still to the right Doppler model and
  as a per-image reference value).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

MODEL_W, MODEL_H = 640, 480  # Cedars 2D model input


@dataclass
class Region:
    dtype: int          # RegionDataType
    x0: int
    y0: int
    x1: int
    y1: int
    dx: Optional[float]  # PhysicalDeltaX (cm/px for 2D, s/px for Doppler)
    dy: Optional[float]  # PhysicalDeltaY (cm/px for 2D, cm/s per px for Doppler)
    ref_y0: Optional[int]  # ReferencePixelY0 (Doppler baseline offset inside the region)
    units_y: Optional[int]


def regions(ds: pydicom.Dataset) -> list[Region]:
    out = []
    for r in getattr(ds, "SequenceOfUltrasoundRegions", None) or []:
        try:
            out.append(Region(int(r.RegionDataType), int(r.RegionLocationMinX0), int(r.RegionLocationMinY0),
                              int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1),
                              float(r.PhysicalDeltaX) if "PhysicalDeltaX" in r else None,
                              float(r.PhysicalDeltaY) if "PhysicalDeltaY" in r else None,
                              int(r.ReferencePixelY0) if "ReferencePixelY0" in r else None,
                              int(r.PhysicalUnitsYDirection) if "PhysicalUnitsYDirection" in r else None))
        except Exception:
            continue
    return out


def read_frames(path: str, max_frames: Optional[int] = None) -> tuple[np.ndarray, pydicom.Dataset]:
    """Return frames as (T, H, W, 3) uint8 RGB and the dataset (pixel data already read)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    pi = str(ds.get("PhotometricInterpretation", "RGB"))
    if arr.ndim == 2:
        arr = np.repeat(arr[None, ..., None], 3, axis=-1)
    elif arr.ndim == 3:
        arr = arr[None] if arr.shape[-1] == 3 else np.repeat(arr[..., None], 3, axis=-1)
    if max_frames and arr.shape[0] > max_frames:
        idx = np.linspace(0, arr.shape[0] - 1, max_frames).round().astype(int)
        arr = arr[idx]
    if pi.startswith("YBR"):
        arr = np.stack([convert_color_space(f, pi, "RGB") for f in arr])
    return np.ascontiguousarray(arr.astype(np.uint8)), ds


def tissue_region(regs: list[Region]) -> Optional[Region]:
    t = [r for r in regs if r.dtype == 1]
    if not t:
        t = [r for r in regs if r.dtype == 2]  # colour-Doppler clip: use the colour box
    return max(t, key=lambda r: (r.x1 - r.x0) * (r.y1 - r.y0)) if t else None


@dataclass
class Canvas:
    frames: np.ndarray      # (T, 480, 640, 3) uint8 RGB
    scale: float            # original px per model px (canvas_w / 640)
    cm_per_model_px: Optional[float]  # scale * PhysicalDeltaX (cm)
    bbox: tuple             # sector bbox in the original frame
    canvas_wh: tuple


def to_canvas_3x4(frames: np.ndarray, reg: Optional[Region], margin: float = 0.04) -> Canvas:
    """Crop the sector bounding box and letterbox it onto a 3:4 canvas, then resize to 640x480."""
    T, H, W, _ = frames.shape
    if reg is None:
        x0, y0, x1, y1 = 0, 0, W - 1, H - 1
        dx = None
    else:
        x0, y0, x1, y1 = max(reg.x0, 0), max(reg.y0, 0), min(reg.x1, W - 1), min(reg.y1, H - 1)
        dx = abs(reg.dx) if reg.dx else None
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cw = int(round(bw * (1 + 2 * margin)))
    ch = int(round(cw * 3 / 4))
    if ch < bh * (1 + 2 * margin):
        ch = int(round(bh * (1 + 2 * margin)))
        cw = int(round(ch * 4 / 3))
    ox, oy = (cw - bw) // 2, (ch - bh) // 2
    canvas = np.zeros((T, ch, cw, 3), dtype=np.uint8)
    canvas[:, oy:oy + bh, ox:ox + bw] = frames[:, y0:y1 + 1, x0:x1 + 1]
    out = np.stack([cv2.resize(f, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA) for f in canvas])
    scale = cw / MODEL_W
    return Canvas(out, scale, (scale * dx) if dx else None, (x0, y0, x1, y1), (cw, ch))


def mask_ecg_trace(rgb: np.ndarray) -> np.ndarray:
    """Cedars pre-processing: black out the green ECG trace (G > 200 and R < 100)."""
    img = rgb.copy()
    m = np.logical_and(img[..., 1] > 200, img[..., 0] < 100)
    img[m] = 0
    return img


def spectral_region(regs: list[Region]) -> Optional[Region]:
    s = [r for r in regs if r.dtype in (3, 4)]
    return max(s, key=lambda r: (r.x1 - r.x0) * (r.y1 - r.y0)) if s else None


# ----------------------------------------------------------------------------- OCR
LABEL_PATTERNS = [
    # canonical, regex on the OCR text (upper-cased, spaces collapsed)
    ("TR_VMAX", r"\bTR\s*V\s*MAX"), ("AV_VMAX", r"\bAV\s*V\s*MAX"), ("MR_VMAX", r"\bMR\s*V\s*MAX"),
    ("LVOT_VMAX", r"\bLVOT\s*V\s*MAX"), ("PV_VMAX", r"\bPV\s*V\s*MAX"), ("RVOT_VMAX", r"\bRVOT\s*V\s*MAX"),
    ("MV_E", r"\bMV\s*E\s*VEL|\bMV\s*E\b|\bE\s*VEL"), ("MV_A", r"\bMV\s*A\s*VEL|\bA\s*VEL"),
    ("LAT_E_PRIME", r"LAT\S*\s*E'|E'\s*LAT"), ("SEPT_E_PRIME", r"SEPT\S*\s*E'|MED\S*\s*E'|E'\s*SEPT|E'\s*MED"),
    ("TV_S_PRIME", r"\bTV\s*S'|\bRV\s*S'"), ("TAPSE", r"TAPSE"),
    ("LVOT_VTI", r"LVOT\s*VTI"), ("AV_VTI", r"\bAV\s*VTI"), ("MV_VTI", r"\bMV\s*VTI"),
]
VALUE_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(M/S|MLS|M\s*/\s*S|CM/S|CMLS|CM\s*/\s*S|MMHG|CM)\b")


class LabelOCR:
    """EasyOCR on the GE label box. Lazy so worker processes only load it when needed."""

    def __init__(self, gpu: bool = True, box=(0, 110, 680, 1016)):
        self.gpu = gpu
        self.box = box  # y0, y1, x0, x1 of the label box in a 708x1016 GE frame
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
        return self._reader

    def read(self, rgb: np.ndarray) -> dict:
        H, W = rgb.shape[:2]
        y0, y1, x0, x1 = self.box
        y1, x1 = min(y1, H), min(x1, W)
        x0 = max(0, int(W * 0.67)) if W != 1016 else x0
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return {"label": None, "text": "", "value": None, "unit": None}
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        res = self.reader.readtext(big, detail=1, paragraph=False)
        text = " ".join(t for _, t, _ in res)
        up = re.sub(r"\s+", " ", text.upper()).replace("VMAX", "V MAX")
        label = None
        for canon, pat in LABEL_PATTERNS:
            if re.search(pat, up):
                label = canon
                break
        val, unit = None, None
        m = VALUE_RE.search(up.replace(",", "."))
        if m:
            try:
                val = float(m.group(1))
                unit = m.group(2).replace(" ", "").replace("MLS", "M/S").replace("CMLS", "CM/S")
            except ValueError:
                pass
        return {"label": label, "text": text, "value": val, "unit": unit}
