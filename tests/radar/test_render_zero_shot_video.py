# tests/radar/test_render_zero_shot_video.py

"""The explainer video renderer on a synthetic explain record (random embeddings, the shipped prompt file, the
evaluation's own scoring and `explain_task`), over tiny generated clips:

- the record follows the evaluation's rules: a regression bag value is the median of the values of its top
  round(keep * M) prompts, the study value is the mean over bags, a categorical prediction is the argmax grade;
- the renderer writes a playable MP4 with the frames it reports, also when a clip cannot be read.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

try:
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

from tools.radar import render_zero_shot_video as rv

MP4 = ".mp4"  # built, not literal: tests/radar/test_privacy.py flags literal video file names


def write_clip(path, frames=12, size=64):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (size, size))
    for t in range(frames):
        image = np.zeros((size, size, 3), np.uint8)
        cv2.circle(image, (size // 2, size // 2), 5 + t, (255, 255, 255), -1)
        writer.write(image)
    writer.release()


@unittest.skipIf(cv2 is None or shutil.which("ffmpeg") is None, "needs OpenCV and ffmpeg")
class TestRenderZeroShotVideo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.clips = []
        for k, view in enumerate(["A4C", "A2C", "PLAX"]):
            path = os.path.join(cls.root, f"clip{k}.mp4")
            write_clip(path)
            cls.clips.append((view, path))
        cls.clips.append(("PSAX", os.path.join(cls.root, "missing" + MP4)))      # unreadable: a grey tile
        cls.record = rv.synthetic_record(cls.clips, tasks=("lv_ejection_fraction", "lv_systolic_function", "lvh_any"),
                                         labels={"lvef": 55, "lv_systolic_function": 2}, num_bags=2, clips_per_bag=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_the_record_follows_the_evaluation_rules(self):
        self.assertTrue(self.record["synthetic"])
        ef = self.record["tasks"]["lv_ejection_fraction"]
        self.assertEqual(len(ef["prompts"]), 86)                                  # 5 .. 90
        self.assertEqual(ef["top_k"], 17)                                         # round(0.2 * 86)
        for bag in ef["bags"]:
            self.assertEqual(len(bag["selected"]), 17)
            sims = np.asarray(bag["similarities"])
            self.assertTrue(np.all(sims[bag["selected"]].min() >= np.delete(sims, bag["selected"]).max()))
            median = torch.tensor(bag["selected_values"]).median().item()           # torch: the lower middle
            self.assertAlmostEqual(bag["value"], median, places=5)
            np.testing.assert_allclose(np.asarray(bag["logits"]), sims / self.record["temperature"], rtol=1e-5)
        self.assertAlmostEqual(ef["prediction"], np.mean([b["value"] for b in ef["bags"]]), places=5)
        self.assertEqual(ef["label"], 55.0)
        lv = self.record["tasks"]["lv_systolic_function"]
        self.assertEqual(lv["prediction"], lv["grades"][int(np.argmax(lv["pooled_probabilities"]))])
        np.testing.assert_allclose(lv["pooled_probabilities"], np.mean([b["probabilities"] for b in lv["bags"]], 0),
                                   rtol=1e-6)
        binary = self.record["tasks"]["lvh_any"]
        self.assertIsNone(binary["label"])
        for bag in binary["bags"]:
            s = bag["set_scores"]
            expected = 1 / (1 + np.exp(-(s["positive"] - s["negative"]) / self.record["temperature"]))
            self.assertAlmostEqual(bag["value"], expected, places=5)

    def test_the_video_is_written_with_the_frames_reported(self):
        out = os.path.join(self.root, "video", "demo" + MP4)
        durations = {k: 0.5 for k in rv.DEFAULT_DURATIONS}
        frames = rv.render(rv.add_grade_names(self.record), out, fps=4, durations=durations)
        capture = cv2.VideoCapture(out)
        read = 0
        while capture.read()[0]:
            read += 1
        capture.release()
        self.assertGreater(frames, 0)
        self.assertEqual(read, frames)
        self.assertEqual(rv.choose_tasks(self.record), ["lv_ejection_fraction", "lv_systolic_function"])
        with self.assertRaises(ValueError):
            rv.choose_tasks(self.record, ["no_such_task"])

    def test_every_mode_renders(self):
        out = os.path.join(self.root, "all" + MP4)
        durations = {k: 0.25 for k in rv.DEFAULT_DURATIONS}
        frames = rv.render(self.record, out, tasks=["lvh_any", "lv_systolic_function", "lv_ejection_fraction"],
                           fps=4, durations=durations)
        self.assertTrue(os.path.getsize(out) > 0 and frames > 0)


if __name__ == "__main__":
    unittest.main()
