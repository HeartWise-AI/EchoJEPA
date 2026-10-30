# tests/utils/test_deepcoro_clip_encoder.py

"""`deepcoro_clip_echojepa_encoder`: the EchoJEPA ViT is read strictly out of a DeepCORO_CLIP
checkpoint (`video_encoder` -> `model.encoder.*`), the aggregator and projection are left out, and
a checkpoint without the encoder is refused."""

import logging
import os
import tempfile
import unittest

import torch

import src.models.vision_transformer as vit
from evals.video_classification_frozen.modelcustom import deepcoro_clip_echojepa_encoder as adapter

OPTIONS = dict(patch_size=16, tubelet_size=2, uniform_power=True, use_rope=True, img_temporal_dim_size=None)


class TestDeepcoroClipEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.INFO)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        torch.manual_seed(0)
        model = vit.vit_tiny(img_size=32, num_frames=4, **OPTIONS)
        cls.encoder_state = {f"model.encoder.{k}": v for k, v in model.state_dict().items()}
        video_encoder = {**cls.encoder_state, "aggregator.blocks.0.weight": torch.zeros(3), "proj.1.weight": torch.zeros(3)}
        cls.path = os.path.join(tmp.name, "best_model_epoch_0.pt")
        torch.save({"video_encoder": video_encoder, "text_encoder": {}, "epoch": 0}, cls.path)
        cls.bare_path = os.path.join(tmp.name, "bare.pt")
        torch.save({f"module.{k}": v for k, v in video_encoder.items()}, cls.bare_path)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def kwargs(self, **extra):
        return {"encoder": {"model_name": "vit_tiny", **OPTIONS, **extra}}

    def test_the_encoder_is_loaded_strictly_and_wrapped(self):
        module = adapter.init_module(32, 4, self.path, self.kwargs(), {"max_frames": 128})
        self.assertEqual(module.embed_dim, 192)
        loaded = module.model.state_dict()
        for k, v in self.encoder_state.items():
            self.assertTrue(torch.equal(loaded[k.removeprefix("model.encoder.")], v), k)
        clips = [[torch.randn(2, 3, 4, 32, 32)] for _ in range(3)]
        out = module(clips, None)
        self.assertEqual(tuple(out[0].shape), (2, 3 * 2 * 4, 192))  # 3 clips x (2 time x 4 patches) tokens.

    def test_a_bare_state_dict_with_module_prefixes_is_read(self):
        module = adapter.init_module(32, 4, self.bare_path, self.kwargs(state_dict_key=None), {})
        self.assertEqual(module.embed_dim, 192)

    def test_missing_entry_or_prefix_is_refused(self):
        with self.assertRaisesRegex(KeyError, "no `video_encoder` entry"):
            adapter.init_module(32, 4, self.bare_path, self.kwargs(), {})
        with self.assertRaisesRegex(KeyError, "No parameter starts with"):
            adapter.init_module(32, 4, self.path, self.kwargs(encoder_prefix="backbone."), {})
        with self.assertRaises(RuntimeError):  # another architecture: strict load fails.
            adapter.init_module(32, 4, self.path, {"encoder": {"model_name": "vit_small", **OPTIONS}}, {})
