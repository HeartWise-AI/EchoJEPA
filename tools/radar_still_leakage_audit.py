"""CPU-only synthetic counterexamples to still leakage safeguards.

No clinical images, model weights, model fitting, or network data inputs are used. The
reported rates describe these deliberately constructed fixtures only. They are
not estimates of leakage frequency or predictive performance on real screens.
Run from the checkout with ``python -m tools.radar_still_leakage_audit --out DIR``.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.datasets.still_preprocess import IMAGENET_MEAN, IMAGENET_STD, prepare_still

SIDE = 224
SPECTRAL_BOX = (0, 0, SIDE, SIDE)
ROWS = (35, 52, 69, 86, 103, 120, 137, 154)
LABEL_BOX = (20, 55, 77, 72)
VALUE_BOX = (100, 55, 169, 72)


def prepared_gray(image, **kwargs):
    """Invert channel-0 normalization to inspect actual encoder-input pixels."""
    clip, report = prepare_still(image, SPECTRAL_BOX, **kwargs)
    gray = (clip[0, 0].numpy() * IMAGENET_STD[0] + IMAGENET_MEAN[0]) * 255
    return np.rint(gray).clip(0, 255).astype(np.uint8), report


def caliper_fixture(row, intensity=255, tint_matched=False):
    """A programmed caliper position and numeric caption on a low-valued spectrum."""
    rng = np.random.default_rng(83)
    gray = rng.integers(24, 92, (SIDE, SIDE), dtype=np.uint8)
    clean = np.repeat(gray[..., None], 3, axis=-1)
    color = (intensity,) * 3
    if tint_matched:
        clean[..., 2] = gray + 60
        color = (180, 180, 255)  # high chroma, same blue direction as the vendor tint
    marked = clean.copy()
    cv2.line(marked, (20, row), (145, row), color, 1, cv2.LINE_8)
    cv2.line(marked, (90, row - 4), (90, row + 4), color, 1, cv2.LINE_8)
    cv2.putText(marked, str(row), (158, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_8)
    return clean, marked


def estimate_row(gray):
    """Prespecified threshold/row-count rule; no fitted parameters or model."""
    return int((gray[:, 20:146] > 170).sum(axis=1).argmax())


def has_flat_footprint(gray):
    """Image-only, unfitted rule: any 9x9 patch has variance below one.

    It receives no box coordinates or paired control at inference. This rule
    is deliberately specific to the synthetic noise backgrounds, not a proposed
    classifier for clinical screens (which can have naturally uniform regions).
    """
    pixels = gray.astype(np.float64)
    mean = cv2.boxFilter(pixels, -1, (9, 9))
    variance = cv2.boxFilter(pixels * pixels, -1, (9, 9)) - mean * mean
    return bool(variance.min() < 1)


def label_value_fixture():
    clean = np.full((SIDE, SIDE, 3), 40, dtype=np.uint8)
    marked = clean.copy()
    cv2.putText(marked, "Vmax", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,) * 3, 1, cv2.LINE_8)
    cv2.putText(marked, "4.20", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,) * 3, 1, cv2.LINE_8)
    return clean, marked


def textured_pair(index):
    """Identical unannotated spectra, differing only by application of a text box.

    Omitting text deliberately isolates the intervention's footprint from OCR
    and from differences in underlying physiology or image selection.
    """
    rng = np.random.default_rng(1000 + index)
    gray = rng.integers(25, 141, (SIDE, SIDE), dtype=np.uint8)
    image = np.repeat(gray[..., None], 3, axis=-1)
    x, y = 30 + index % 4 * 12, 40 + index % 3 * 17
    box = (x, y, x + 35 + index % 3 * 5, y + 16 + index % 2 * 5)
    before, _ = prepared_gray(image)
    after, report = prepared_gray(image, text_boxes=[box])
    # Public box coordinates plus the documented default 4-pixel padding.
    padded = (box[0] - 4, box[1] - 4, box[2] + 5, box[3] + 5)
    return before, after, padded, report


def collect_pixel_evidence():
    """JSON-safe metrics plus a small panel built from actual preprocessed outputs."""
    metrics = {"scope": "deterministic synthetic counterexamples only; no real-image rate estimates",
               "device": "cpu", "output_size": SIDE}
    panels = []
    for name, intensity, tinted in (("white", 255, False), ("grey", 200, False),
                                     ("tint_matched", 255, True)):
        errors, mask_counts, unchanged, max_clean_delta, clean_recovery = [], [], [], [], []
        for row in ROWS:
            clean, marked = caliper_fixture(row, intensity, tinted)
            kwargs = {"tint": (0, 0, 60)} if tinted else {}
            output, report = prepared_gray(marked, **kwargs)
            control, _ = prepared_gray(clean, **kwargs)
            expected = marked.astype(np.float32).mean(axis=-1).round().astype(np.uint8)
            errors.append(abs(estimate_row(output) - row))
            mask_counts.append(report["masked_overlay_px"])
            unchanged.append(bool(np.array_equal(output, expected)))
            max_clean_delta.append(int(np.abs(output.astype(int) - control.astype(int)).max()))
            clean_recovery.append(estimate_row(control) == row)
        metrics[name] = {"n": len(ROWS), "row_mae_px": float(np.mean(errors)),
                         "row_max_error_px": max(errors), "overlay_mask_px": mask_counts,
                         "row_exact_recovery_fraction": float(np.mean(np.asarray(errors) == 0)),
                         "clean_control_exact_recovery_fraction": float(np.mean(clean_recovery)),
                         "unchanged_after_grayscale": all(unchanged),
                         "max_delta_vs_clean_px": max(max_clean_delta)}
        panels.append((name, output))

    clean, marked = label_value_fixture()
    label_only, report = prepared_gray(marked, text_boxes=[LABEL_BOX])
    complete, _ = prepared_gray(marked, text_boxes=[LABEL_BOX, VALUE_BOX])
    # The value is separated from the label by substantially more than pad=4.
    value_ink = (marked[..., 0] != clean[..., 0]) & (np.indices(clean.shape[:2])[1] >= VALUE_BOX[0])
    metrics["label_value"] = {
        "value_ink_px": int(value_ink.sum()),
        "value_ink_surviving_label_only_px": int((label_only[value_ink] == marked[..., 0][value_ink]).sum()),
        "value_bright_ink_surviving_label_only_px": int((label_only[value_ink] > 170).sum()),
        "full_boxes_equal_clean": bool(np.array_equal(complete, clean[..., 0])),
        "label_only_masked_text_px": report["masked_text_px"],
    }
    panels.extend((("label box only", label_only), ("label + value", complete)))

    correct, count, bbox_matches = 0, 0, 0
    for index in range(16):
        before, after, (x0, y0, x1, y1), _ = textured_pair(index)
        # Fixed, unfitted rule applied to both members of each paired spectrum.
        for expected, image in ((False, before), (True, after)):
            predicted = has_flat_footprint(image)
            correct += int(predicted == expected)
            count += 1
        ys, xs = np.nonzero(before != after)
        bbox_matches += int((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                            == (x0, y0, x1, y1))
    metrics["box_footprint"] = {"paired_spectra": 16, "classified_images": count,
                                "prespecified_rule_accuracy": correct / count,
                                "paired_difference_box_bbox_exact": bbox_matches,
                                "interpretation": "box application and box geometry remain detectable on these fixtures"}
    panels.append(("textured box fill", after))
    return metrics, panels


def write_panel(path, panels):
    canvas = np.full((2 * (SIDE + 28), 3 * SIDE), 255, dtype=np.uint8)
    for index, (label, gray) in enumerate(panels):
        x, y = index % 3 * SIDE, index // 3 * (SIDE + 28)
        cv2.putText(canvas, label, (x + 5, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 0, 1, cv2.LINE_8)
        canvas[y + 28:y + 28 + SIDE, x:x + SIDE] = gray
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics, panels = collect_pixel_evidence()
    (args.out / "synthetic_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_panel(args.out / "synthetic_panel.png", panels)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
