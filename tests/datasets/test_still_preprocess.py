# tests/datasets/test_still_preprocess.py

"""Still preprocessing on synthetic screens: the velocity scale of a crop, text boxes filled from their surroundings,
the crop, coloured overlays found (and a vendor tint spared) and inpainted so that a still with calipers and its
caption box comes out close to the same still without them, the static clip's shape, and the heat-map bias."""

import unittest

import numpy as np
import torch

from src.datasets.still_preprocess import (
    crop_spectral,
    heatmap_bias,
    mask_boxes,
    overlay_mask,
    prepare_still,
    static_clip,
    velocity_scale,
)


def screen(seed=0, calipers=False, box=False):
    """A 120 x 200 grey screen with a spectral band in rows 40-110; optionally yellow calipers and a white box."""
    rng = np.random.default_rng(seed)
    image = np.full((120, 200, 3), 10, dtype=np.uint8)
    envelope = (40 + 50 * np.abs(np.sin(np.linspace(0, 6, 200)))).astype(int)
    for x in range(200):
        image[40:envelope[x], x] = 120 + rng.integers(0, 20)
    if calipers:
        image[60:62, 20:180] = (255, 255, 0)      # a horizontal caliper line
        image[45:100, 100:102] = (0, 255, 255)
    if box:
        image[45:55, 150:190] = 255               # burned-in text inside the spectrum
    return image


class TestPreprocess(unittest.TestCase):

    def test_velocity_scale(self):
        top, bottom = velocity_scale(spec_y0=100, spec_y1=500, ref_y0=100, delta_y=-2.0)
        self.assertAlmostEqual(top, 2.0)          # 100 px above the baseline at -2 cm/s per px: +2 m/s
        self.assertAlmostEqual(bottom, -6.0)

    def test_mask_boxes_fills_from_ring(self):
        image = np.full((20, 20, 3), 50, dtype=np.uint8)
        image[8:12, 8:12] = 255
        out, mask = mask_boxes(image, [(8, 8, 11, 11)], pad=1)
        self.assertTrue((out == 50).all())
        self.assertEqual(int(mask.sum()), 36)
        self.assertTrue((image[8:12, 8:12] == 255).all())        # the input is not changed

    def test_crop(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        self.assertEqual(crop_spectral(image, 5, 2, 30, 8).shape, (6, 15, 3))
        with self.assertRaisesRegex(ValueError, "empty"):
            crop_spectral(image, 25, 2, 30, 8)

    def test_overlay_mask_and_tint(self):
        image = screen(calipers=True)
        mask = overlay_mask(image)
        self.assertTrue(mask[60, 50] and mask[70, 100])
        self.assertFalse(mask[50, 10])                          # grey spectrum
        tinted = screen().astype(np.int32)
        tinted[..., 2] = np.clip(tinted[..., 2] + 60, 0, 255)  # a blue-tinted spectrum
        tinted = tinted.astype(np.uint8)
        self.assertGreater(overlay_mask(tinted).mean(), 0.1)
        self.assertLess(overlay_mask(tinted, tint=(0, 0, 60)).mean(), 0.01)

    def test_overlay_edge_goes_with_the_overlay_and_a_tint_is_estimated(self):
        image = np.full((40, 60, 3), 60, dtype=np.uint8)
        image[20, 10:50] = (255, 255, 0)                        # a line
        image[19, 10:50] = image[21, 10:50] = (80, 80, 60)      # its anti-aliased edge: chroma 20, under the threshold
        self.assertFalse(overlay_mask(image, dilate=0)[19, 30])
        mask = overlay_mask(image)
        self.assertTrue(mask[19, 30] and mask[21, 30])
        self.assertFalse(mask[5, 30])
        tinted = screen().astype(np.int32)
        tinted[..., 2] = np.clip(tinted[..., 2] + 60, 0, 255)
        tinted = tinted.astype(np.uint8)
        _, report = prepare_still(tinted, (0, 40, 200, 110), size=32, tint=(0, 0, 60))
        self.assertLess(report["overlay_fraction"], 0.01)
        self.assertFalse(report["tint_estimated"])
        _, report = prepare_still(tinted, (0, 40, 200, 110), size=32)      # the tint is found in the crop
        self.assertTrue(report["tint_estimated"])
        self.assertLess(report["overlay_fraction"], 0.01)
        tinted[60:64, 20:180] = (255, 255, 0)                             # a caliper line of another colour
        _, report = prepare_still(tinted, (0, 40, 200, 110), size=32)
        self.assertTrue(report["tint_estimated"])
        self.assertGreater(report["masked_overlay_px"], 4 * 160 - 1)
        mixed = tinted.copy()
        mixed[40:110, :100] = (200, 30, 30)                               # two colours: no single tint
        mixed[40:110, 100:] = (30, 200, 30)
        with self.assertRaisesRegex(ValueError, "no single tint"):
            prepare_still(mixed, (0, 40, 200, 110), size=32)

    def test_masking_removes_overlays_and_text(self):
        clean, _ = prepare_still(screen(), (0, 40, 200, 110), size=32)
        marked, report = prepare_still(screen(calipers=True, box=True), (0, 40, 200, 110),
                                       text_boxes=[(150, 45, 189, 54)], size=32)
        unmasked, _ = prepare_still(screen(calipers=True, box=True), (0, 40, 200, 110), size=32,
                                    chroma_threshold=1000)
        self.assertGreater(report["masked_text_px"], 0)
        self.assertGreater(report["masked_overlay_px"], 0)
        self.assertLess(float((marked - clean).abs().mean()), 0.5 * float((unmasked - clean).abs().mean()))

    def test_static_clip(self):
        clip = static_clip(np.full((50, 80), 255, dtype=np.uint8), size=16, frames=2)
        self.assertEqual(tuple(clip.shape), (3, 2, 16, 16))
        self.assertTrue(torch.equal(clip[:, 0], clip[:, 1]))
        self.assertAlmostEqual(float(clip[0, 0, 0, 0]), (1 - 0.485) / 0.229, places=5)

    def test_heatmap_bias(self):
        heat = np.zeros((28, 28), dtype=np.float32)
        heat[2:4, 26:28] = 5.0
        bias = heatmap_bias(heat, grid=14)
        self.assertEqual(tuple(bias.shape), (196,))
        self.assertAlmostEqual(float(bias.max()), 0.0, places=6)
        self.assertEqual(int(bias.argmax()), 1 * 14 + 13)
        self.assertTrue(torch.equal(heatmap_bias(np.zeros((10, 10)), grid=14), torch.zeros(196)))


if __name__ == "__main__":
    unittest.main()
