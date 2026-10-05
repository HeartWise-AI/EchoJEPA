# tests/utils/test_vjepa_2_1_eval_encoder.py

"""The frozen-probe encoder for V-JEPA 2.1 checkpoints: built like the trainer's encoder,
loaded strictly from a trainer checkpoint, and returning the backbone's tokens per clip."""

import os
import tempfile
import unittest

import torch
import yaml

from app.vjepa_2_1.utils import init_video_model
from evals.video_classification_frozen.modelcustom import vjepa_2_1_encoder
from src.models.attentive_pooler import AttentiveRegressor
from tests.utils.test_vjepa_2_1_baseline import shipped
from tests.utils.test_vjepa_2_1_utils import QuietLogs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The encoder options of the shipped configs, on a vit_tiny with 4-frame 32 px clips.
OPTIONS = dict(patch_size=16, tubelet_size=2, uniform_power=True, use_rope=True, use_sdpa=True,
               img_temporal_dim_size=1, interpolate_rope=True, modality_embedding=True, n_output_distillation=1)


class TestVJEPA21EvalEncoder(QuietLogs):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        torch.manual_seed(0)
        self.encoder, _ = init_video_model(
            torch.device("cpu"), model_name="vit_tiny", crop_size=32, max_num_frames=4, pred_depth=4,
            pred_embed_dim=96, use_mask_tokens=True, return_all_tokens=True, **OPTIONS,
        )
        # As the trainer saves it: DDP around MultiSeqWrapper, so keys are `module.backbone.*`.
        self.path = os.path.join(tmp.name, "latest.pth.tar")
        state = {f"module.{k}": v for k, v in self.encoder.state_dict().items()}
        torch.save({"target_encoder": state, "encoder": {}, "epoch": 3}, self.path)

    def load(self, **changes):
        encoder = {"model_name": "vit_tiny", "checkpoint_key": "target_encoder", **OPTIONS, **changes}
        return vjepa_2_1_encoder.init_module(
            resolution=32, frames_per_clip=4, checkpoint=self.path, model_kwargs={"encoder": encoder},
            wrapper_kwargs={},
        )

    def test_returns_the_trained_backbone_tokens_for_each_clip(self):
        model = self.load().eval()
        clip = torch.randn(2, 3, 4, 32, 32)
        with torch.no_grad():
            outputs = model([[clip]])
            expected = self.encoder.backbone.eval()(clip)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(tuple(outputs[0].shape), (2, 2 * 2 * 2, 192))  # 2 tubelets x 2 x 2 patches
        self.assertTrue(torch.allclose(outputs[0], expected, atol=1e-6))
        self.assertEqual(model.embed_dim, 192)

    def test_a_videos_clips_become_one_token_sequence_for_the_probe(self):
        # The probe's 2 segments x 1 view: each clip is embedded on its own, and the tokens of both
        # are concatenated along time into one sequence, which the head pools into one prediction.
        model = self.load().eval()
        clips = [torch.randn(2, 3, 4, 32, 32), torch.randn(2, 3, 4, 32, 32)]
        with torch.no_grad():
            outputs = model([[c] for c in clips])
            each = [self.encoder.backbone.eval()(c) for c in clips]
        self.assertEqual(len(outputs), 1)  # one sequence per spatial view, not one per clip
        self.assertTrue(torch.allclose(outputs[0], torch.cat(each, dim=1), atol=1e-6))
        torch.manual_seed(0)
        head = AttentiveRegressor(embed_dim=192, num_heads=4, depth=1, num_targets=1).eval()
        with torch.no_grad():
            joint, mean_of_clips = head(outputs[0]), (head(each[0]) + head(each[1])) / 2
        self.assertEqual(tuple(joint.shape), (2, 1))
        self.assertFalse(torch.allclose(joint, mean_of_clips))  # attentive pooling, not a mean over clips

    def test_an_encoder_built_differently_fails_to_load(self):
        with self.assertRaises(RuntimeError):
            self.load(modality_embedding=False)

    def test_unknown_options_and_missing_keys_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown V-JEPA 2.1 encoder options"):
            self.load(pred_depth=12)
        with self.assertRaisesRegex(KeyError, "no `ema_encoder`"):
            self.load(checkpoint_key="ema_encoder")


class TestShippedProbeConfig(unittest.TestCase):

    def test_the_probe_builds_the_encoder_the_pretraining_configs_train(self):
        with open(os.path.join(ROOT, "configs/eval/vitb16/lvef-a4c-224px-16f.yaml")) as f:
            probe = yaml.safe_load(f)
        encoder = probe["model_kwargs"]["pretrain_kwargs"]["encoder"]
        self.assertEqual(probe["model_kwargs"]["module_name"], vjepa_2_1_encoder.__name__)
        self.assertLessEqual(set(encoder) - {"model_name", "checkpoint_key"}, set(vjepa_2_1_encoder.ENCODER_OPTIONS))
        for name in ("pretrain", "cooldown"):
            cfg = shipped(f"{name}-continued-224px-16f.yaml")
            trained = {**cfg["model"], **{k: cfg["data"][k] for k in ("patch_size", "tubelet_size")},
                       "use_sdpa": cfg["meta"]["use_sdpa"]}
            for key, value in encoder.items():
                if key != "checkpoint_key":
                    self.assertEqual(value, trained[key], f"{name}: {key}")
            self.assertEqual(probe["experiment"]["data"]["resolution"], cfg["data"]["crop_size"])
            self.assertEqual(probe["experiment"]["data"]["frames_per_clip"], cfg["data"]["dataset_fpcs"][0])


if __name__ == "__main__":
    unittest.main()
