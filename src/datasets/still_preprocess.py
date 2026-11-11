# src/datasets/still_preprocess.py

"""A Doppler still as encoder input: crop to the spectral region, mask what shows the measured value, and a static
clip of one tubelet for the frozen video encoder. Pure functions on arrays; the token cache writer calls
`prepare_still` once per still and stores the encoder's tokens.

Order (`prepare_still`, recipe `PREPROCESS_VERSION`):

1. `mask_boxes`: every text box of the measurement block (label and value boxes, in screen pixels, from the
   manifest's `text_boxes`), padded by `pad` pixels, is filled with the median of a ring around it, so the box's
   place is not a hole that marks it.
2. `crop_spectral`: the DICOM spectral region (`spec_x0..spec_y1`). It leaves out the B-mode thumbnail, the scale
   labels and most of the measurement block; the velocity scale it removes comes back as numbers (`velocity_scale`).
3. `overlay_mask` + `inpaint`: the calipers and VTI traces the sonographer drew are coloured, the spectrum is grey
   (or one vendor tint, given as `tint`); chromatic pixels are inpainted from their neighbours.
4. grey, resized to `size` x `size`, normalised, and repeated over `frames` frames: with the encoder's tubelet of 2,
   two frames give one temporal token, a [1, H, W] token grid.

The same recipe runs on every still, measured or not, so masking itself does not tell a measured still from another.
That this holds is checked, not assumed (the "is-measured" probe in docs/radar/RADAR_stills_integration.md).

`heatmap_bias` turns a peak heat-map already mapped to the crop (EchoNet's DeepLabV3 Doppler models) into a per-token
attention bias for the structure pooler: log of the heat-map average-pooled to the token grid, 0 at the peak.
"""

import numpy as np
import torch
import torch.nn.functional as F

PREPROCESS_VERSION = "m1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def velocity_scale(spec_y0, spec_y1, ref_y0, delta_y):
    """(velocity at the crop's top row, at its bottom row), in m/s. `ref_y0` is the baseline row relative to the
    region's top (DICOM ReferencePixelY0), `delta_y` the region's PhysicalDeltaY in cm/s per pixel (unit code 7):
    v(row) = (row - ref_y0) * delta_y."""
    height = float(spec_y1) - float(spec_y0)
    return (0.0 - ref_y0) * delta_y / 100.0, (height - ref_y0) * delta_y / 100.0


def _ring_median(image, x0, y0, x1, y1, width=3):
    h, w = image.shape[:2]
    a, b = max(0, y0 - width), min(h, y1 + width)
    c, d = max(0, x0 - width), min(w, x1 + width)
    outer = image[a:b, c:d].reshape(-1, *image.shape[2:])
    inner = np.zeros((b - a, d - c), dtype=bool)
    inner[max(0, y0 - a):max(0, y1 - a), max(0, x0 - c):max(0, x1 - c)] = True
    ring = outer[~inner.reshape(-1)]
    return np.median(ring, axis=0) if len(ring) else np.zeros(image.shape[2:])


def mask_boxes(image, boxes, pad=4):
    """(a copy of `image` with each box (x0, y0, x1, y1) padded by `pad` and filled with its ring's median, the
    boolean mask of what was filled)."""
    out = image.copy()
    mask = np.zeros(image.shape[:2], dtype=bool)
    h, w = image.shape[:2]
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(w, int(x1) + pad + 1), min(h, int(y1) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] = _ring_median(image, x0, y0, x1, y1).astype(image.dtype)
        mask[y0:y1, x0:x1] = True
    return out, mask


def crop_spectral(image, x0, y0, x1, y1):
    """The spectral region [y0:y1, x0:x1], clipped to the image; refuses an empty region."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The spectral region is empty once clipped to the image.")
    return image[y0:y1, x0:x1]


def overlay_mask(image, chroma_threshold=40, tint=None, tint_tolerance=0.15):
    """Pixels of drawn overlays: chroma (max - min over RGB) above `chroma_threshold`. With `tint` (the vendor's
    spectrum colour, RGB), a pixel whose colour direction is within `tint_tolerance` of the tint's is spectrum, not
    overlay."""
    rgb = image.astype(np.float32)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    mask = chroma > chroma_threshold
    if tint is not None:
        tint = np.asarray(tint, dtype=np.float32)
        tint = (tint - tint.min()) / max(1e-6, float(np.linalg.norm(tint - tint.min())))
        direction = rgb - rgb.min(axis=-1, keepdims=True)
        direction = direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)
        mask &= np.linalg.norm(direction - tint, axis=-1) > tint_tolerance
    return mask


def inpaint(image, mask, radius=3):
    """`image` with the masked pixels filled from their neighbours (OpenCV Telea; deterministic)."""
    if not mask.any():
        return image.copy()
    import cv2
    return cv2.inpaint(np.ascontiguousarray(image), mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)


def static_clip(gray, size=224, frames=2, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """A [3, frames, size, size] float32 clip from a 2-D uint8 image: resized (area), scaled to [0, 1], repeated over
    three channels, normalised per channel and repeated over `frames` frames."""
    x = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)[None, None]
    x = F.interpolate(x, size=(size, size), mode="area")[0, 0]
    channels = torch.stack([(x - m) / s for m, s in zip(mean, std)])
    return channels[:, None].repeat(1, frames, 1, 1).contiguous()


def prepare_still(image, spectral_box, text_boxes=(), size=224, frames=2, tint=None, chroma_threshold=40, pad=4):
    """(clip [3, frames, size, size], report {"masked_text_px", "masked_overlay_px"}) of an RGB uint8 screen."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("prepare_still expects an RGB image [H, W, 3].")
    screen, text = mask_boxes(image, text_boxes, pad)
    crop = crop_spectral(screen, *spectral_box)
    overlay = overlay_mask(crop, chroma_threshold, tint)
    crop = inpaint(crop, overlay)
    gray = crop.astype(np.float32).mean(axis=-1).round().astype(np.uint8)
    report = {"masked_text_px": int(text.sum()), "masked_overlay_px": int(overlay.sum())}
    return static_clip(gray, size, frames), report


def heatmap_bias(heatmap, grid=14, eps=1e-3):
    """[grid * grid] float32 bias from a non-negative heat-map over the crop: average-pooled to the token grid,
    divided by its maximum, log(h + eps) - log(1 + eps) (0 at the peak, negative elsewhere). A heat-map that is all
    zero gives all zeros (no bias)."""
    h = torch.as_tensor(np.asarray(heatmap, dtype=np.float32)).clamp_min(0)
    if h.ndim != 2:
        raise ValueError("heatmap must be 2-D.")
    pooled = F.adaptive_avg_pool2d(h[None, None], grid)[0, 0]
    top = float(pooled.max())
    if top <= 0:
        return torch.zeros(grid * grid)
    pooled = pooled / top
    return (torch.log(pooled + eps) - np.log(1 + eps)).reshape(-1).float()
