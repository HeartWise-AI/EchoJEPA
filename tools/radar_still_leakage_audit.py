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


def collect_pixel_evidence():
    """Current-recipe observations, not historical m1/m2 pixel-equality claims."""
    from src.datasets.still_preprocess import PREPROCESS_VERSION

    metrics = {"scope": "deterministic synthetic counterexamples only; no real-image rate estimates",
               "device": "cpu", "output_size": SIDE, "preprocess": PREPROCESS_VERSION}
    panels = []
    for name, intensity, tinted in (("white", 255, False), ("grey", 200, False),
                                    ("tint_matched", 255, True)):
        recovered, clean_recovered, windows = [], [], []
        for row in ROWS:
            clean, marked = caliper_fixture(row, intensity, tinted)
            kwargs = {"tint": (0, 0, 60)} if tinted else {}
            output, report = prepared_gray(marked, **kwargs)
            control, _ = prepared_gray(clean, **kwargs)
            recovered.append(estimate_row(output) == row)
            clean_recovered.append(estimate_row(control) == row)
            windows.append(report["window"])
        metrics[name] = {"n": len(ROWS), "row_exact_recovery_fraction": float(np.mean(recovered)),
                         "clean_control_exact_recovery_fraction": float(np.mean(clean_recovered)),
                         "windows": windows}
        panels.append((name, output))

    _, marked = label_value_fixture()
    for name, boxes in (("label_only", [LABEL_BOX]), ("label_and_value", [LABEL_BOX, VALUE_BOX])):
        try:
            output, report = prepared_gray(marked, text_boxes=boxes)
        except ValueError as exc:
            if "No window" not in str(exc):
                raise
            metrics[name] = {"rejected_no_free_window": True}
        else:
            metrics[name] = {"rejected_no_free_window": False, "window": report["window"]}
            panels.append((name, output))

    # Current recipe crops away panel columns rather than median-filling rectangles.
    rng = np.random.default_rng(1000)
    gray = rng.integers(25, 141, (SIDE, SIDE), dtype=np.uint8)
    texture = np.repeat(gray[..., None], 3, axis=-1)
    output, report = prepared_gray(texture, text_boxes=[(100, 40, 135, 56)])
    metrics["textured_panel"] = {"flat_footprint": has_flat_footprint(output), "window": report["window"]}
    panels.append(("panel excluded", output))
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
