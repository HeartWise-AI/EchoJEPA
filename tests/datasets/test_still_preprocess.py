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
    free_window,
    mark_columns,
    mask_boxes,
    overlay_mask,
    panel_columns,
    prepare_crop,
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
        self.assertFalse(overlay_mask(image, dilate=0, grow_threshold=None)[19, 30])
        self.assertTrue(overlay_mask(image, dilate=0)[19, 30])  # the paler edge touches the line: it goes with it
        lone = image.copy()
        lone[5, 10:50] = (80, 80, 60)                           # as pale, touching nothing strongly coloured
        self.assertFalse(overlay_mask(lone, dilate=0)[5, 30])
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
        tinted[60:64, :] = (255, 255, 0)                                  # a running line of another colour
        _, report = prepare_still(tinted, (0, 40, 200, 110), size=32)
        self.assertTrue(report["tint_estimated"])
        self.assertGreater(report["masked_overlay_px"], 4 * 50 - 1)       # inpainted over the window's 50 columns
        mixed = tinted.copy()
        mixed[40:110, :100] = (200, 30, 30)                               # two colours: no single tint
        mixed[40:110, 100:] = (30, 200, 30)
        with self.assertRaisesRegex(ValueError, "no single tint"):
            prepare_still(mixed, (0, 40, 200, 110), size=32)

    def test_panel_columns(self):
        covered = panel_columns((100, 50, 900, 400), [(600, 60, 700, 78), (600, 84, 690, 102)])
        self.assertEqual(covered.shape, (800,))
        self.assertFalse(covered[:488].any())                   # 12 px before the labels
        self.assertTrue(covered[488:750].all())                 # labels and the value column (1.5 x their width)
        self.assertFalse(covered[750:].any())
        self.assertFalse(panel_columns((100, 50, 900, 400), ()).any())
        below = panel_columns((100, 300, 900, 400), [(600, 60, 700, 78)])   # three text lines end above row 300
        self.assertFalse(below.any())
        near = panel_columns((100, 150, 900, 400), [(600, 60, 700, 78)])    # an unparsed line could reach the region
        self.assertTrue(near.any())

    def test_mark_columns_tell_a_trace_from_a_running_line(self):
        crop = np.full((100, 400, 3), 60, dtype=np.uint8)
        crop[90, :] = (220, 220, 40)                            # the ECG runs over the whole width: no mark
        crop[89, ::2] = (150, 200, 60)                          # its paler, greener edge
        crop[20:70, 200] = (230, 130, 110)                      # a traced beat: two sides and a dashed marker
        crop[20:70, 240] = (230, 130, 110)
        crop[70, 200:241] = (230, 130, 110)
        columns = mark_columns(crop)
        self.assertTrue(columns[192:249].all())
        self.assertFalse(columns[:190].any() or columns[251:].any())
        crop[40:47, 60] = crop[43, 57:64] = (220, 220, 40)      # a small caliper cross in the ECG's own colour
        columns = mark_columns(crop)
        self.assertTrue(columns[49:72].all())                   # a mark all the same
        self.assertFalse(columns[:48].any() or columns[73:190].any())
        self.assertFalse(mark_columns(np.full((50, 80, 3), 60, dtype=np.uint8)).any())

    def test_free_window(self):
        blocked = np.zeros(100, dtype=bool)
        blocked[40:60] = True
        self.assertEqual(free_window(blocked, 20), (10, 30))                 # centred in the first longest run
        weight = np.zeros(100)
        weight[70:80] = 1.0                                                  # a beat in the right-hand run
        first, last = free_window(blocked, 20, weight)
        self.assertTrue(60 <= first and first <= 70 and last >= 80 and last - first == 20)
        with self.assertRaisesRegex(ValueError, "No window"):
            free_window(blocked, 41)

    def test_the_window_holds_no_panel_no_trace_and_no_fill(self):
        rng = np.random.default_rng(0)
        image = rng.integers(30, 90, size=(200, 900, 3)).astype(np.uint8)
        image[...] = image[..., :1]                             # a grey, textured spectrum
        image[60:140, 100:160] = 200                            # three beats: bright columns
        image[60:140, 400:460] = 200
        image[60:140, 700:760] = 200
        image[10:60, 20:260] = 255                              # the panel over the first beat: labels and values
        image[60:142, 398] = image[60:142, 462] = (230, 130, 110)     # the second beat is traced
        image[142, 398:463] = (230, 130, 110)
        image[190, :] = (220, 220, 40)                          # the ECG
        boxes = [(24, 14, 100, 30), (24, 36, 100, 52)]
        gray, report = prepare_crop(image, (0, 0, 900, 200), boxes)
        first, last = report["window"]
        self.assertEqual(last - first, 225)                     # a quarter of the width, measured still or not
        self.assertTrue(first > 470 and first <= 700 and last >= 760)       # the free, untraced third beat
        self.assertEqual(gray.shape, (200, 225))
        self.assertLess(int((gray > 240).sum()), 1)             # nothing of the white panel
        original = image[:, first:last].astype(np.float32).mean(axis=-1)
        untouched = np.ones_like(original, dtype=bool)
        untouched[184:197] = False                              # only the ECG rows were inpainted
        self.assertTrue(np.array_equal(gray[untouched], original.round().astype(np.uint8)[untouched]))
        self.assertLess(float(gray[187:194].max()), 150)        # the ECG line is gone
        unmeasured, other = prepare_crop(image[:, 470:], (0, 0, 430, 200))  # no boxes: the same share of the width
        self.assertEqual(other["window"][1] - other["window"][0], 108)
        clip, _ = prepare_still(image, (0, 0, 900, 200), boxes, size=32)
        self.assertEqual(tuple(clip.shape), (3, 2, 32, 32))
        image[60:142, 698] = image[60:142, 762] = (230, 130, 110)           # every beat is drawn on
        image[60:142, 98] = image[60:142, 162] = (230, 130, 110)
        with self.assertRaisesRegex(ValueError, "No window"):
            prepare_crop(image, (0, 0, 900, 200), boxes)

    def test_a_faint_tinted_spectrum_is_refused_not_painted_away(self):
        image = np.full((100, 400, 3), 5, dtype=np.uint8)
        image[40:60, ::7] = (200, 120, 100)                     # sparse tinted signal: under the tint threshold
        with self.assertRaisesRegex(ValueError, "painted away|chromatic|No window"):
            prepare_crop(image, (0, 0, 400, 100))

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
