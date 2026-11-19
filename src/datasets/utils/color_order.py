"""Exact load-time fix for the legacy YBR colour swap in the MHI DeepEcho MP4s.

The legacy extractor that converted the MHI archive (DeepEcho pipeline box, pydicom 2.4) passed YBR_* pixel
data (Y, Cb, Cr) to ``cv2.COLOR_YCrCb2BGR``, which expects (Y, Cr, Cb). Cb and Cr were swapped: red and blue are
exchanged, and green and the colour gains shift (not a pure R/B swap). 10,216,967 of the 10,227,073 MP4s under
``.../DeepEcho/MP4/`` (99.9%) are affected; the other 10,106 came from RGB/MONOCHROME2/PALETTE DICOMs and are
correct, as are MIMIC-IV-ECHO and every other dataset. Decision (Robert, 2026-11-19): fix at load time, no
re-encode. Same function as HeartWise-AI/DicomVideoProcessing ``downloadAvi/color_order.py`` (PR #20), copied
because importing ``downloadAvi`` pulls in its own dependencies.

- ``needs_legacy_ybr_fix(path)``: True for a path under ``DeepEcho/MP4/`` that is not in the exception list
  (``$ECHOJEPA_YBR_EXCEPTIONS``, default ``DEFAULT_EXCEPTIONS``: one path per line, relative to ``DeepEcho/MP4/``).
  ``ECHOJEPA_LEGACY_YBR_FIX=0`` turns the fix off (to reproduce runs made before it).
- ``fix_legacy_colors(frames, path, channel_order)``: applies the exact inverse when the path needs it.
"""
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LEGACY_MARKER = "/DeepEcho/MP4/"
DEFAULT_EXCEPTIONS = "/media/data1/datasets/DeepEcho/preprocessing_results/color_order_exceptions.txt"
_EXCEPTIONS = None


def _enabled():
    return os.environ.get("ECHOJEPA_LEGACY_YBR_FIX", "1") != "0"


def _exceptions():
    """Paths (relative to DeepEcho/MP4/) of legacy clips whose DICOM was not YBR, loaded once per process."""
    global _EXCEPTIONS
    if _EXCEPTIONS is None:
        path = os.environ.get("ECHOJEPA_YBR_EXCEPTIONS", DEFAULT_EXCEPTIONS)
        try:
            with open(path) as f:
                _EXCEPTIONS = frozenset(line.strip() for line in f if line.strip())
        except OSError:
            logger.warning("Legacy YBR exception list not readable; fixing every DeepEcho/MP4 clip (99.9% are YBR).")
            _EXCEPTIONS = frozenset()
    return _EXCEPTIONS


def needs_legacy_ybr_fix(path):
    """True when ``path`` is a legacy-converted MHI clip from a YBR DICOM (works for /media and /mnt mounts)."""
    if not _enabled() or not isinstance(path, str):
        return False
    i = path.find(LEGACY_MARKER)
    return i >= 0 and path[i + len(LEGACY_MARKER):] not in _exceptions()


def undo_legacy_ybr_swap(frames, channel_order="RGB"):
    """Exact inverse of ``YCrCb2BGR(Y, Cb, Cr)`` for uint8 frames (..., H, W, 3); returns the same shape and order."""
    a = np.asarray(frames)
    if a.dtype != np.uint8 or a.ndim < 3 or a.shape[-1] != 3:
        raise ValueError("expected uint8 frames with 3 channels in the last axis")
    if channel_order not in ("BGR", "RGB"):
        raise ValueError("channel_order must be 'BGR' or 'RGB'")
    shape = a.shape
    tall = np.ascontiguousarray(a.reshape(-1, shape[-2], 3))       # all frames stacked vertically: one cvtColor
    to_ycc = cv2.COLOR_BGR2YCrCb if channel_order == "BGR" else cv2.COLOR_RGB2YCrCb
    from_ycc = cv2.COLOR_YCrCb2BGR if channel_order == "BGR" else cv2.COLOR_YCrCb2RGB
    ycc = cv2.cvtColor(tall, to_ycc)                                 # = (Y, Cb, Cr) of the DICOM
    fixed = cv2.cvtColor(np.ascontiguousarray(ycc[..., [0, 2, 1]]), from_ycc)
    return fixed.reshape(shape)


def fix_legacy_colors(frames, path, channel_order="RGB"):
    """``frames`` with the legacy swap undone when ``path`` needs it; unchanged (same object) otherwise."""
    if frames is None or len(frames) == 0 or not needs_legacy_ybr_fix(path):
        return frames
    return undo_legacy_ybr_swap(frames, channel_order)
