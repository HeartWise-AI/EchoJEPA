# src/datasets/still_preprocess.py

"""A Doppler still as encoder input: a stretch of the spectral recording that the sonographer did not draw on, as a
static clip of one tubelet for the frozen video encoder. Pure functions on arrays; the token cache writer calls
`prepare_still` once per still and stores the encoder's tokens.

A measured still shows its answer three ways: the measurement panel (labels with their values), the traced envelope
and calipers on the measured beat, and whatever is painted over them. Filling the panel leaves a flat patch, and
inpainting a trace leaves a smoothed band along the envelope, so both still tell where the measurement was. The
recipe therefore removes nothing from the part it keeps: it keeps a part that had nothing on it.

Order (`prepare_still`, recipe `PREPROCESS_VERSION`):

1. `crop_spectral`: the DICOM spectral region (`spec_x0..spec_y1`). It leaves out the B-mode thumbnail and the scale
   labels; the velocity scale it removes comes back as numbers (`velocity_scale`).
2. `panel_columns`: the columns under the measurement panel, from the label boxes of the manifest's `text_boxes`
   grown to hold the value column and the unparsed lines.
3. `mark_columns`: the columns of drawn marks (traces, calipers, cursors, dashed time markers), found by their
   colour; running lines (baseline, ECG) are not marks.
4. `free_window`: a window of a fixed share of the width (`window_fraction`) free of both, the one with the most
   signal: a beat beside the measured one. A still without such a window is refused. A still with no panel and no
   mark (an unmeasured one) gets a window of the same share by the same rule.
5. `overlay_mask` + `inpaint`, inside the window: only the running lines are left to paint over, with their paler
   edge. A spectrum drawn in one colour keeps its pixels (`estimate_tint`).
6. grey, resized to `size` x `size`, normalised, and repeated over `frames` frames: with the encoder's tubelet of 2,
   two frames give one temporal token, a [1, H, W] token grid.

Limits. Marks are found by colour: a white or grey caliper or trace is not, so the stills of systems that draw them
must be left out upstream (`build_still_cache.py --vendors`). The model reads a neighbouring beat, not the measured
one: with a very irregular rhythm the two differ. That nothing of the measurement is left is checked on real stills,
not assumed (`data/still_leak_audit.py`, `data/still_overlay_audit.py`, and review by eye).

`mask_boxes` (filling a box with the median of a ring around it) is no longer part of the recipe.

`heatmap_bias` turns a peak heat-map already mapped to the crop (EchoNet's DeepLabV3 Doppler models) into a per-token
attention bias for the structure pooler: log of the heat-map average-pooled to the token grid, 0 at the peak. A
heat-map of the whole crop does not fit the window: it must be cut to the report's `window` first.
"""

import numpy as np
import torch
import torch.nn.functional as F

PREPROCESS_VERSION = "m4"    # m4: a window beside the measurement panel and the drawn marks; nothing is filled
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


def overlay_mask(image, chroma_threshold=40, tint=None, tint_tolerance=0.15, dilate=2, grow_threshold=12):
    """Pixels of drawn overlays: chroma (max - min over RGB) above `chroma_threshold`, with the connected pixels
    whose chroma is above `grow_threshold`, grown by `dilate` pixels. A thin coloured line is paler than the colour
    it was drawn in over most of its length (anti-aliasing, and the chroma subsampling of the stored image), so a
    single threshold finds only its strongest pixels and inpainting those leaves the rest of the line as a pale
    outline: the weaker pixels are taken when they touch a strong one (`grow_threshold`; None for the single
    threshold). With `tint` (the vendor's spectrum colour, RGB), a pixel whose colour direction is within
    `tint_tolerance` of the tint's is spectrum, not overlay. A white or grey overlay has no chroma and is not found
    here."""
    rgb = image.astype(np.float32)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    mask = chroma > chroma_threshold
    weak = chroma > grow_threshold if grow_threshold is not None else mask
    if tint is not None:
        tint = np.asarray(tint, dtype=np.float32)
        tint = (tint - tint.min()) / max(1e-6, float(np.linalg.norm(tint - tint.min())))
        direction = rgb - rgb.min(axis=-1, keepdims=True)
        direction = direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)
        other = np.linalg.norm(direction - tint, axis=-1) > tint_tolerance
        mask &= other
        weak &= other
    if grow_threshold is not None and mask.any():
        import cv2
        _, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
        mask = np.isin(labels, np.unique(labels[mask])) & weak
    if dilate and mask.any():
        size = 2 * int(dilate) + 1
        mask = F.max_pool2d(torch.from_numpy(mask)[None, None].float(), size, stride=1, padding=int(dilate))[0, 0] > 0
        mask = mask.numpy()
    return mask


def estimate_tint(image, chroma_threshold=40, tint_tolerance=0.15, min_fraction=0.2, min_agreement=0.8):
    """The colour (RGB direction) of a tinted spectrum, or None. A vendor may draw the spectrum in one colour (orange,
    sepia) instead of grey; then a large share of the crop is chromatic and nearly all of it has one colour
    direction. Returned when more than `min_fraction` of the pixels are chromatic and at least `min_agreement` of
    those are within `tint_tolerance` of their mean direction; overlays, which are thin lines of other colours, are
    then still found by `overlay_mask(tint=...)`. An overlay drawn in the spectrum's own colour is not."""
    rgb = image.astype(np.float32)
    chromatic = (rgb.max(axis=-1) - rgb.min(axis=-1)) > chroma_threshold
    if chromatic.mean() <= min_fraction:
        return None
    direction = rgb[chromatic] - rgb[chromatic].min(axis=-1, keepdims=True)
    direction = direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)
    mean = direction.mean(axis=0)
    mean = mean / max(1e-6, float(np.linalg.norm(mean)))
    if float((np.linalg.norm(direction - mean, axis=-1) <= tint_tolerance).mean()) < min_agreement:
        return None
    return tuple(float(v) for v in mean)


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


def panel_columns(spectral_box, text_boxes=(), value_extent=1.5, min_value_px=120, lines_above=1, lines_below=3,
                  pad=12):
    """Boolean [width]: the columns of the spectral crop that the measurement panel covers.

    The sonographer's panel lists labels with their values to the right, and lines the extraction did not parse
    (derived values) above and below the parsed ones. Its extent is taken from the label boxes (`text_boxes`, screen
    coordinates): their bounding rectangle, grown by `lines_above` and `lines_below` text lines, by `pad` pixels to
    the left, and to the right by the value column (`value_extent` times the labels' width, at least
    `min_value_px`). When that rectangle reaches into the spectral region, its columns are covered over the whole
    height."""
    sx0, sy0, sx1, sy1 = (int(v) for v in spectral_box)
    covered = np.zeros(max(0, sx1 - sx0), dtype=bool)
    boxes = [tuple(int(v) for v in box) for box in text_boxes]
    if not boxes:
        return covered
    line = float(np.median([y1 - y0 for _, y0, _, y1 in boxes])) * 1.6 + 2
    x0, x1 = min(b[0] for b in boxes), max(b[2] for b in boxes)
    y0, y1 = min(b[1] for b in boxes) - lines_above * line, max(b[3] for b in boxes) + lines_below * line
    x1 = x1 + max(value_extent * (x1 - x0), min_value_px)
    x0 = x0 - pad
    if y1 <= sy0 or y0 >= sy1 or x1 <= sx0 or x0 >= sx1:
        return covered                                          # the panel is outside the spectral region
    covered[max(0, int(x0 - sx0)):max(0, int(np.ceil(x1 - sx0)))] = True
    return covered


def mark_columns(crop, chroma_threshold=40, tint=None, running_fraction=0.5, hue_bins=12, min_pixels=4, margin=8,
                 min_piece_width=0.05, max_piece_height=0.3):
    """Boolean [width]: the columns of a crop that hold a drawn measurement mark: a traced envelope, a caliper, a
    cursor, a dashed time marker. The strongly coloured pixels are grouped by hue (`hue_bins` bins). A hue found in
    more than `running_fraction` of the columns holds a running line (the baseline, the ECG): its connected pieces
    that are wide (`min_piece_width` of the width) and low (`max_piece_height` of the height) are the line and no
    mark, with the pixels of a neighbouring hue inside their rows (the line's paler edge). Everything else is a
    mark, a small cross of the line's own hue included; a hue with fewer than `min_pixels` remaining pixels is
    noise. The columns of the marks, with `margin` on each side, are returned. Achromatic marks are not found
    (`overlay_mask`)."""
    import cv2
    mask = overlay_mask(crop, chroma_threshold, tint, dilate=0, grow_threshold=None)
    columns = np.zeros(crop.shape[1], dtype=bool)
    if not mask.any():
        return columns
    height, width = mask.shape
    rgb = crop.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.degrees(np.arctan2(np.sqrt(3.0) * (g - b), 2.0 * r - g - b)) % 360.0
    bins = (hue / (360.0 / hue_bins)).astype(int) % hue_bins
    marks = mask.copy()
    for k in range(hue_bins):
        own = mask & (bins == k)
        if not own.any() or own.any(axis=0).mean() <= running_fraction:
            continue
        joined = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8))      # bridge one-pixel gaps
        count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        line = np.zeros_like(own)
        for c in range(1, count):
            if (stats[c, cv2.CC_STAT_WIDTH] >= min_piece_width * width
                    and stats[c, cv2.CC_STAT_HEIGHT] <= max_piece_height * height):
                line |= (labels == c) & own
        if not line.any():
            continue
        marks &= ~line
        rows = np.nonzero(line.any(axis=1))[0]
        band = np.zeros(height, dtype=bool)
        band[max(0, rows.min() - 4):rows.max() + 5] = True
        edge = (bins == (k - 1) % hue_bins) | (bins == (k + 1) % hue_bins)
        near = cv2.dilate(line.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0   # only beside the line itself
        marks &= ~(edge & band[:, None] & near)
    for k in range(hue_bins):
        own = marks & (bins == k)
        if own.sum() < min_pixels:
            continue
        for x in np.nonzero(own.any(axis=0))[0]:
            columns[max(0, x - margin):x + margin + 1] = True
    return columns


def free_window(blocked, keep, weight=None):
    """(first, last) of `keep` consecutive columns none of which is `blocked`; refuses when there are none. With
    `weight` [width] (the signal of each column), the window with the most weight, the leftmost of equals: the
    stretch of free recording that shows the most flow, rather than a gap between two beats. Without it, the window
    centred in the longest free run."""
    blocked = np.asarray(blocked, dtype=bool)
    runs, start = [], None
    for i, b in enumerate([*blocked, True]):
        if not b and start is None:
            start = i
        elif b and start is not None:
            runs.append((start, i))
            start = None
    runs = [r for r in runs if r[1] - r[0] >= keep]
    if not runs:
        raise ValueError("No window of the wanted width is free of the measurement panel and of drawn marks.")
    if weight is None:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        first = a + (b - a - keep) // 2
        return first, first + keep
    total = np.concatenate([[0.0], np.cumsum(np.asarray(weight, dtype=np.float64))])
    best, first = -1.0, runs[0][0]
    for a, b in runs:
        sums = total[a + keep:b + 1] - total[a:b - keep + 1]
        k = int(np.argmax(sums))
        if sums[k] > best:
            best, first = float(sums[k]), a + k
    return first, first + keep


def prepare_crop(image, spectral_box, text_boxes=(), tint=None, chroma_threshold=40, pad=12, max_overlay_fraction=0.2,
                 window_fraction=0.25, dilate=3, max_signal_loss=0.5):
    """(grey uint8 crop [h, w] at the screen's resolution, report {"window", "blocked_fraction",
    "masked_overlay_px", "overlay_fraction", "tint_estimated"}) of an RGB uint8 screen.

    The spectral region is cut to a window of `window_fraction` of its width that holds neither the measurement
    panel (`panel_columns`) nor a drawn mark (`mark_columns`): the model reads a stretch of the recording the
    sonographer did not draw on, a beat beside the measured one: among the free windows, the one with the most
    signal (`free_window`). Nothing is filled where a panel or a trace was, so
    neither a flat patch nor a smoothed band marks the measured still, and every still, measured or not, keeps the
    same share of its width. Inside the window, only running lines (baseline, ECG) are inpainted, with their paler
    edge (`overlay_mask`, `dilate`). Without `tint`, the colour of a tinted spectrum is estimated from the crop
    (`estimate_tint`).

    Refused, rather than returned damaged: a still with no free window (the panel or the marks cover too much); a
    crop of which more than `max_overlay_fraction` is chromatic with no single tint; and one where the overlay mask
    would remove more than `max_signal_loss` of the bright pixels (a faint tinted spectrum not recognised as one
    would be painted away). A white or grey mark is not found: stills of systems that draw them are to be left out
    upstream."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("prepare_still expects an RGB image [H, W, 3].")
    crop = crop_spectral(image, *spectral_box)
    h, w = image.shape[:2]
    clipped = (max(0, int(spectral_box[0])), max(0, int(spectral_box[1])), min(w, int(spectral_box[2])),
               min(h, int(spectral_box[3])))
    estimated = tint is None
    if estimated:
        tint = estimate_tint(crop, chroma_threshold, min_fraction=max_overlay_fraction)
    whole = overlay_mask(crop, chroma_threshold, tint, dilate=0)
    if float(whole.mean()) > max_overlay_fraction:
        raise ValueError(f"{float(whole.mean()):.0%} of the spectral region is chromatic and no single tint explains "
                         "it: the overlays would take the spectrum with them.")
    bright = crop.astype(np.float32).mean(axis=-1) > 60
    if bright.sum() and float((whole & bright).sum()) / float(bright.sum()) > max_signal_loss:
        raise ValueError("The overlay mask covers most of the bright pixels: the spectrum itself would be painted "
                         "away (a faint tinted spectrum).")
    blocked = panel_columns(clipped, text_boxes, pad=pad) | mark_columns(crop, chroma_threshold, tint)
    keep = max(1, int(round(window_fraction * crop.shape[1])))
    signal = crop.astype(np.float32).mean(axis=-1)
    signal[whole] = 0.0                                         # a drawn line is no flow
    first, last = free_window(blocked, keep, weight=signal.sum(axis=0))
    crop = crop[:, first:last]
    overlay = overlay_mask(crop, chroma_threshold, tint, dilate=dilate)
    crop = inpaint(crop, overlay)
    gray = crop.astype(np.float32).mean(axis=-1).round().astype(np.uint8)
    report = {"window": [int(first), int(last)], "blocked_fraction": float(blocked.mean()),
              "masked_overlay_px": int(overlay.sum()), "overlay_fraction": float(overlay.mean()),
              "tint_estimated": bool(estimated and tint is not None)}
    return gray, report


def prepare_still(image, spectral_box, text_boxes=(), size=224, frames=2, tint=None, chroma_threshold=40, pad=12,
                  max_overlay_fraction=0.2, window_fraction=0.25):
    """(clip [3, frames, size, size], the report of `prepare_crop`) of an RGB uint8 screen: the prepared crop,
    resized and repeated over `frames` frames (`static_clip`)."""
    gray, report = prepare_crop(image, spectral_box, text_boxes, tint, chroma_threshold, pad, max_overlay_fraction,
                                window_fraction)
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
