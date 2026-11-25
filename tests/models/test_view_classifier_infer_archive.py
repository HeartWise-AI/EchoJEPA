# tests/models/test_view_classifier_infer_archive.py

"""Archive inference (`infer_archive`) on CPU with synthetic videos and a tiny fake model: the plan, sharded streams,
resuming, the merge's one-row-per-clip check, unreadable clips, and the optional group rule."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from evals.view_classifier import classes as vc
from evals.view_classifier import infer_archive as ia
from tests.models.test_view_classifier_arm_b_input import write_video

AVI = ".avi"
CLASSES = {c: vc.PARENT[c] for c in vc.NEW_CLASSES}


class Fake(torch.nn.Module):
    """Logits from each clip's mean colour: deterministic, depends on the pixels."""

    def __init__(self, n):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(3, n, generator=g)

    def forward(self, x):
        return x.float().mean(dim=(2, 3, 4)) @ self.w * 10


def fake_loader(record, device):
    return Fake(len(record["classes"])).to(device), lambda x: x.float() / 255.0


class TestInferArchive(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.videos = []
        for i in range(5):
            path = os.path.join(root, "videos", f"c{i}{AVI}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_video(path, 12 + 6 * i, seed=i)
            self.videos.append(path)
        self.broken = os.path.join(root, "videos", "missing" + AVI)
        manifest = pd.DataFrame({"video_path": self.videos + [self.broken, self.videos[0]]})   # one duplicate
        self.manifest = os.path.join(root, "videos.parquet")
        manifest.to_parquet(self.manifest, index=False)
        runs = os.path.join(root, "runs", "s9")
        final = os.path.join(runs, "arm_b", "pseudo_off", "fold_final")
        os.makedirs(final)
        torch.save({"state_dict": {}, "hp": {"lr": 1e-5}, "classes": list(CLASSES)}, os.path.join(final, "model.pt"))
        self.cfg = {"paths": {"runs": runs, "view_classifier": "unused"}, "classes": CLASSES,
                    "arm_b": {"channel_order": "rgb",
                              "data": {"frames_per_clip": 16, "frame_step": 2, "resolution": 32,
                                       "preprocessing": "full_squash_aa", "frame_read": "echoprime"}}}
        self.config = os.path.join(root, "config.yaml")
        yaml.safe_dump(self.cfg, open(self.config, "w"), sort_keys=False)
        self.out = os.path.join(root, "archive")

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, *args):
        ia.main(list(args))

    def stream(self, j, n):
        args = ia.argparse.Namespace(out=self.out, stream=j, streams=n, device="cpu", workers=0, batch=2)
        ia.run(args, model_loader=fake_loader)

    def test_plan_streams_resume_merge(self):
        self.main("plan", "--config", self.config, "--manifest", self.manifest, "--out", self.out, "--shards", "3")
        record, clips = ia.read_plan(self.out)
        self.assertEqual(record["clips"], 6)                             # duplicates dropped
        self.assertEqual(sorted(clips.shard.unique()), [0, 1, 2])
        with self.assertRaises(FileExistsError):
            self.main("plan", "--config", self.config, "--manifest", self.manifest, "--out", self.out)
        with self.assertRaises(ValueError):
            self.main("merge", "--out", self.out)                        # nothing run yet
        self.stream(0, 2)
        self.assertEqual(sorted(os.listdir(os.path.join(self.out, "shards"))),
                         ["shard_00000.parquet", "shard_00002.parquet"])
        self.stream(1, 2)
        first = pd.read_parquet(ia.shard_path(self.out, 1))
        mtime = os.path.getmtime(ia.shard_path(self.out, 0))
        self.stream(0, 1)                                                 # everything finished: nothing rewritten
        self.assertEqual(os.path.getmtime(ia.shard_path(self.out, 0)), mtime)
        os.remove(ia.shard_path(self.out, 1))
        open(ia.shard_path(self.out, 1) + ".tmp123", "w").close()         # a stopped stream's half-written file
        self.stream(0, 1)
        pd.testing.assert_frame_equal(pd.read_parquet(ia.shard_path(self.out, 1)), first)
        self.assertFalse(os.path.exists(ia.shard_path(self.out, 1) + ".tmp123"))
        self.main("merge", "--out", self.out)
        pred = pd.read_parquet(os.path.join(self.out, "predictions.parquet"))
        self.assertEqual(sorted(pred.video_path), sorted(self.videos))
        self.assertEqual(list(pd.read_csv(os.path.join(self.out, "unreadable.csv")).video_path), [self.broken])
        p = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)
        self.assertTrue((pred.top_view == np.asarray(list(CLASSES))[p.argmax(axis=1)]).all())
        self.assertTrue((pred.view == pred.top_view).all())                # no group rule
        np.testing.assert_allclose(pred.top_prob, p.max(axis=1), rtol=1e-6)
        self.assertEqual(len(set(pred.top_prob.round(6))), 5)              # the pixels matter

    def test_merge_refuses_a_lost_clip(self):
        self.main("plan", "--config", self.config, "--manifest", self.manifest, "--out", self.out, "--shards", "2")
        self.stream(0, 1)
        t = pd.read_parquet(ia.shard_path(self.out, 0))
        t.iloc[1:].to_parquet(ia.shard_path(self.out, 0), index=False)
        with self.assertRaises(ValueError):
            self.main("merge", "--out", self.out)

    def test_plan_exclusions_limit_and_model_check(self):
        ex = os.path.join(self.tmp.name, "exclude.csv")
        pd.DataFrame({"avi_path": [self.broken]}).to_csv(ex, index=False)
        self.main("plan", "--config", self.config, "--manifest", self.manifest, "--out", self.out,
                  "--exclude", ex, "--limit", "3")
        record, clips = ia.read_plan(self.out)
        self.assertEqual((record["excluded"], record["clips"]), (1, 3))
        self.assertNotIn(self.broken, set(clips.video_path))
        other = dict(self.cfg, classes=dict(reversed(list(CLASSES.items()))))
        yaml.safe_dump(other, open(self.config, "w"), sort_keys=False)
        with self.assertRaises(ValueError):                              # the model's classes differ
            self.main("plan", "--config", self.config, "--manifest", self.manifest,
                      "--out", os.path.join(self.tmp.name, "b"))


class TestLoadModel(TestInferArchive):

    def test_loads_the_final_model_and_the_arm_input(self):
        from unittest import mock

        from evals.view_classifier import mvit
        original = mvit.build_mvit(len(vc.OLD_CLASSES))
        model = mvit.build_mvit(len(vc.OLD_CLASSES))
        model.head[1] = mvit.extend_head(model.head[1], vc.OLD_CLASSES, tuple(CLASSES), CLASSES)
        torch.nn.init.normal_(model.head[1].weight)
        final = os.path.join(self.cfg["paths"]["runs"], "arm_b", "pseudo_off", "fold_final", "model.pt")
        torch.save({"state_dict": model.state_dict(), "hp": {"lr": 1e-5}, "classes": list(CLASSES)}, final)
        self.main("plan", "--config", self.config, "--manifest", self.manifest, "--out", self.out)
        with mock.patch.object(mvit, "load_original", lambda *a, **k: original):
            got, prepare = ia.load_model(ia.read_plan(self.out)[0], torch.device("cpu"))
        torch.testing.assert_close(got.head[1].weight, model.head[1].weight)
        self.assertFalse(got.training)
        x = torch.zeros((1, 3, 16, 8, 8), dtype=torch.uint8)
        x[:, :, :10] = 100
        torch.testing.assert_close(prepare(x), mvit.model_input(x, "rgb", zero_padding=True))
        torch.save({"state_dict": {}, "hp": {}, "classes": list(CLASSES)}, final)       # the model changed since
        with self.assertRaises(ValueError):
            ia.load_model(ia.read_plan(self.out)[0], torch.device("cpu"))


class TestGroupRule(unittest.TestCase):

    def test_rule(self):
        classes = ["A", "L1", "L2", "L3"]
        cfg = {"inference": {"group_rule": {"enabled": True, "output": "L", "members": ["L1", "L2", "L3"],
                                            "together_min": 0.6, "single_max": 0.5}}}
        rule = ia.group_rule(cfg, classes)
        p = np.array([[0.1, 0.3, 0.3, 0.3],     # levels together likely, none alone: L
                      [0.1, 0.7, 0.1, 0.1],     # one level likely: kept
                      [0.5, 0.2, 0.2, 0.1]])    # levels together 0.5 < 0.6: kept
        view, prob = ia.apply_group_rule(p, classes, rule)
        self.assertEqual(list(view), ["L", "L1", "A"])
        np.testing.assert_allclose(prob, [0.9, 0.7, 0.5])
        self.assertIsNone(ia.group_rule({}, classes))
        self.assertIsNone(ia.group_rule({"inference": {"group_rule": dict(cfg["inference"]["group_rule"],
                                                                          enabled=False)}}, classes))
        with self.assertRaises(ValueError):
            ia.group_rule({"inference": {"group_rule": dict(cfg["inference"]["group_rule"], output="A")}}, classes)


if __name__ == "__main__":
    unittest.main()
