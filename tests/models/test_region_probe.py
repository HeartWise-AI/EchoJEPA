# tests/models/test_region_probe.py

"""The two-pass region probe (`pooling_mode: region`): mask shares per token, tracked crop boxes,
the crop sampler, the mask decoder and its block taps, the validation of the `region` block, and
one study through both passes with exactly two encoder calls."""

import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

import src.models.vision_transformer as vit
from evals.video_classification_frozen.modelcustom.vit_encoder_multiclip import ClipAggregation
from src.datasets.video_study_dataset import VideoStudyDataset, read_split
from src.models.region_probe import (
    RegionMaskDecoder,
    RegionStudyProbe,
    StudyTokenPooler,
    crop_clip,
    encode_with_taps,
    region_options,
    token_fractions,
    tracked_boxes,
    validate_blocks,
)
from src.models.study_probe import head_specs, multitask_loss

FRAMES, RES, SOURCE, CROP, CLASSES = 4, 32, 64, 48, 4
TRACKING = {"min_pixels": 20, "min_frames": 2, "margin": 0.05, "min_side": 0.25}
CLASSIFIER = {
    "pooling_mode": "region",
    "query_mode": "group",
    "attention_hidden": 8,
    "target_label": ["a_size", "a_large", "b_size", "b_large", "a_bright", "b_bright", "any_bright", "flow",
                     "flow_low"],
    "head_task": {"a_size": "regression", "a_large": "binary", "b_size": "regression", "b_large": "binary",
                  "a_bright": "binary", "b_bright": "binary", "any_bright": "binary", "flow": "regression",
                  "flow_low": "binary"},
    "task_groups": {"a_size": "a", "a_large": "a", "b_size": "b", "b_large": "b", "a_bright": "a", "b_bright": "b"},
    "region": {
        "num_classes": CLASSES,
        "groups": {"a": [1], "b": [2, 3]},
        "decoder": {"blocks": [1, 3], "width": 16, "checkpoint": "decoder.pt"},
        "source_resolution": SOURCE,
        "crop_resolution": CROP,
        "crops_per_group": 2,
        "tracking": dict(TRACKING),
        "readouts": {
            "size": {"kind": "group_mlp", "hidden": 8, "heads": ["a_size", "a_large", "b_size", "b_large"]},
            "bright": {"kind": "group_linear", "bias": -4.0, "heads": ["a_bright", "b_bright"],
                       "study_head": "any_bright"},
            "flow": {"kind": "groups_mlp", "groups": ["b", "a"], "hidden": 8, "heads": ["flow", "flow_low"]},
        },
        "study_pooler": {"width": 16, "queries": 2, "num_heads": 2, "bias": -2.0},
        "output_range": {"flow": [0.0, 1.05]},
    },
}
DATA = {"frames_per_clip": FRAMES, "resolution": RES, "num_clips_per_video": 1}
PRETRAINED = {"pretrain_kwargs": {"encoder": {"patch_size": 16, "tubelet_size": 2}}}
NORMALIZATION = {"a_size": {"mean": 0.0, "std": 100.0}, "b_size": {"mean": 0.0, "std": 100.0},
                 "flow": {"mean": 0.5, "std": 0.05}}


def options(**changes):
    classifier = copy.deepcopy(CLASSIFIER)
    for path, value in changes.items():
        block = classifier
        *parents, key = path.split("__")
        for parent in parents:
            block = block[parent]
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    return region_options(classifier, DATA, PRETRAINED, head_specs(classifier))


def tiny_encoder():
    torch.manual_seed(0)
    return ClipAggregation(vit.vit_tiny(img_size=RES, num_frames=FRAMES, patch_size=16, tubelet_size=2,
                                        uniform_power=True, use_rope=True, img_temporal_dim_size=None)).eval()


def blob(labels, view, value, top, left, size):
    labels[view, :, top:top + size, left:left + size] = value


def paint(probe, labels):
    """Make the probe's region decoder label the pixels as `labels` [views, T, RES, RES]."""
    logits = 20.0 * F.one_hot(torch.as_tensor(labels), CLASSES).permute(0, 1, 4, 2, 3).float()
    probe.region_decoder.forward = lambda tokens, taps, frames, grid: logits


class TestTokenFractions(unittest.TestCase):

    def test_each_token_gets_the_share_of_its_pixels_and_frames(self):
        prob = torch.zeros(1, 4, 2, 4, 4)
        prob[:, :, 0] = 1.0
        prob[0, 0, :, :2, :2] = torch.tensor([0.0, 1.0]).view(2, 1, 1)   # frame 0, top-left patch: class 1.
        prob[0, 2:, :, 2:, 2:] = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)  # frames 2-3, bottom-right patch.
        shares = token_fractions(prob, (2, 2, 2))
        self.assertEqual(tuple(shares.shape), (1, 8, 2))
        torch.testing.assert_close(shares.sum(-1), torch.ones(1, 8))
        torch.testing.assert_close(shares[0, :, 1], torch.tensor([0.5, 0, 0, 0, 0, 0, 0, 1.0]))

    def test_frames_must_divide_into_the_token_frames(self):
        with self.assertRaisesRegex(ValueError, "do not divide"):
            token_fractions(torch.zeros(1, 3, 2, 4, 4), (2, 2, 2))


class TestTrackedBoxes(unittest.TestCase):

    def moving(self):
        mask = np.zeros((4, 32, 32), dtype=bool)
        for f in range(4):
            mask[f, 8:16, 4 + 2 * f:12 + 2 * f] = True
        return mask

    def test_the_box_follows_the_mask_in_source_pixels(self):
        boxes = tracked_boxes(self.moving(), (64, 64), **TRACKING)
        self.assertEqual((boxes.shape, boxes.dtype), ((4, 4), np.float32))
        side = 14 + 2 * 0.05 * 64                      # 90th percentile extent (7 px x 2) + the margins.
        centre_x = np.array([(15 + 15 + 19) / 3, 19, 23, (23 + 27 + 27) / 3])  # smoothed over three frames.
        np.testing.assert_allclose(boxes[:, 2:], side, rtol=1e-6)
        np.testing.assert_allclose(boxes[:, 0], 23 - side / 2, rtol=1e-6)
        np.testing.assert_allclose(boxes[:, 1], centre_x - side / 2, rtol=1e-6)

    def test_the_box_has_a_minimum_side_and_stays_in_the_frame(self):
        mask = np.zeros((4, 32, 32), dtype=bool)
        mask[:, 0:4, 28:32] = True
        boxes = tracked_boxes(mask, (64, 64), **{**TRACKING, "min_side": 0.5})
        np.testing.assert_allclose(boxes, np.tile([0.0, 32.0, 32.0, 32.0], (4, 1)))

    def test_frames_without_mask_are_interpolated(self):
        mask = self.moving()
        mask[1] = False
        boxes = tracked_boxes(mask, (64, 64), **TRACKING)
        centre_x = np.array([(15 + 15 + 19) / 3, 19, 23, (23 + 27 + 27) / 3])
        np.testing.assert_allclose(boxes[:, 1], centre_x - boxes[0, 2] / 2, rtol=1e-6)

    def test_too_little_mask_gives_no_box(self):
        mask = np.zeros((4, 32, 32), dtype=bool)
        mask[:, 4:6, 4:6] = True                       # 16 mask pixels x 4 = 64 source pixels.
        self.assertIsNone(tracked_boxes(mask, (64, 64), **{**TRACKING, "min_pixels": 65}))
        self.assertIsNotNone(tracked_boxes(mask, (64, 64), **{**TRACKING, "min_pixels": 64}))
        self.assertIsNone(tracked_boxes(np.zeros((4, 32, 32), dtype=bool), (64, 64), **{**TRACKING, "min_pixels": 0.5}))

    def test_a_component_seen_in_too_few_frames_is_dropped(self):
        mask = self.moving()
        mask[0, 26:31, 26:31] = True                   # a second component, in one frame only.
        np.testing.assert_allclose(tracked_boxes(mask, (64, 64), **TRACKING),
                                   tracked_boxes(self.moving(), (64, 64), **TRACKING))
        with_it = tracked_boxes(mask, (64, 64), **{**TRACKING, "min_frames": 1})
        self.assertFalse(np.allclose(with_it, tracked_boxes(self.moving(), (64, 64), **TRACKING)))

    def test_the_line_filter_drops_a_straight_line_only_when_set(self):
        mask = np.zeros((4, 32, 32), dtype=bool)
        mask[:, 10, 1:31] = True
        line_filter = {"linearity": 0.995, "length": 0.29, "min_pixels": 20}
        self.assertIsNotNone(tracked_boxes(mask, (64, 64), **TRACKING))
        self.assertIsNone(tracked_boxes(mask, (64, 64), **TRACKING, line_filter=line_filter))
        self.assertIsNotNone(tracked_boxes(self.moving(), (64, 64), **TRACKING, line_filter=line_filter))


class TestCropClip(unittest.TestCase):

    def test_the_whole_frame_at_the_same_size_is_unchanged(self):
        clip = torch.rand(3, 2, 8, 8)
        boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]]).repeat(3, 1)
        torch.testing.assert_close(crop_clip(clip, boxes, 8), clip, atol=1e-6, rtol=0)

    def test_each_frame_is_cut_at_its_own_box(self):
        ramp = torch.arange(8.0)
        clip = torch.stack([ramp[None, :].expand(8, 8), ramp[:, None].expand(8, 8)])[:, None]  # x ramp, y ramp.
        boxes = torch.tensor([[0.0, 2.0, 4.0, 4.0], [4.0, 0.0, 4.0, 4.0]])
        out = crop_clip(clip, boxes, 4)
        torch.testing.assert_close(out[0, 0], (ramp[2:6])[None, :].expand(4, 4))
        torch.testing.assert_close(out[1, 0], (ramp[4:8])[:, None].expand(4, 4))


class TestRegionMaskDecoder(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.decoder = RegionMaskDecoder(12, CLASSES, width=16, blocks=(1, 3))
        self.tokens, self.frames = torch.randn(2, 8, 12), torch.rand(2, 4, 32, 32)

    def test_logits_per_pixel_and_frame(self):
        out = self.decoder(self.tokens, [torch.randn(2, 8, 12)] * 2, self.frames, (2, 2, 2))
        self.assertEqual(tuple(out.shape), (2, 4, CLASSES, 32, 32))

    def test_the_tapped_blocks_enter_through_gates_that_start_closed(self):
        a = self.decoder(self.tokens, [torch.randn(2, 8, 12)] * 2, self.frames, (2, 2, 2))
        b = self.decoder(self.tokens, [torch.randn(2, 8, 12)] * 2, self.frames, (2, 2, 2))
        torch.testing.assert_close(a, b)
        with torch.no_grad():
            self.decoder.intermediate_gate.fill_(1.0)
        c = self.decoder(self.tokens, [torch.randn(2, 8, 12)] * 2, self.frames, (2, 2, 2))
        self.assertFalse(torch.allclose(a, c))

    def test_a_wrong_token_grid_is_refused(self):
        with self.assertRaisesRegex(ValueError, "expects 12 tokens"):
            self.decoder(self.tokens, [self.tokens] * 2, self.frames, (2, 2, 3))
        with self.assertRaisesRegex(ValueError, "2 tapped blocks"):
            self.decoder(self.tokens, [self.tokens], self.frames, (2, 2, 2))


class TestEncodeWithTaps(unittest.TestCase):

    def test_taps_are_the_block_outputs_of_one_call_and_hooks_are_removed(self):
        encoder = tiny_encoder()
        clips = torch.randn(2, 3, FRAMES, RES, RES)
        seen = {}
        handles = [encoder.model.blocks[i].register_forward_hook(lambda m, i_, o, i=i: seen.setdefault(i, o))
                   for i in (1, 3)]
        with torch.no_grad():
            tokens, taps = encode_with_taps(encoder, clips, None, (1, 3))
            expected = encoder([[clips]], None)[0]
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(tokens, expected)
        self.assertEqual(tuple(tokens.shape), (2, 8, 192))
        for tap, index in zip(taps, (1, 3)):
            self.assertTrue(torch.equal(tap, seen[index]))
        self.assertTrue(all(len(block._forward_hooks) == 0 for block in encoder.model.blocks))

    def test_hooks_are_removed_when_the_encoder_fails(self):
        encoder = tiny_encoder()
        with self.assertRaises(Exception):
            encode_with_taps(encoder, torch.randn(1, 3, FRAMES, RES), None, (1,))
        self.assertTrue(all(len(block._forward_hooks) == 0 for block in encoder.model.blocks))

    def test_blocks_outside_the_encoder_are_refused(self):
        validate_blocks([0, 11], 12)
        with self.assertRaisesRegex(ValueError, r"\[12\] are outside the encoder's 12 blocks"):
            validate_blocks([1, 12], 12)


class TestRegionOptions(unittest.TestCase):

    def test_a_valid_block_resolves_to_plain_values(self):
        resolved = options()
        self.assertEqual(resolved["groups"], {"a": [1], "b": [2, 3]})
        self.assertEqual(resolved["decoder"], {"blocks": [1, 3], "width": 16})
        self.assertEqual(resolved["task_groups"], CLASSIFIER["task_groups"])
        self.assertEqual((resolved["resolution"], resolved["source_resolution"], resolved["crop_resolution"]),
                         (RES, SOURCE, CROP))
        self.assertEqual(resolved["tracking"], {**TRACKING, "min_pixels": 20.0, "line_filter": None})
        self.assertEqual(resolved["readouts"]["bright"],
                         {"kind": "group_linear", "heads": ["a_bright", "b_bright"], "bias": -4.0,
                          "study_head": "any_bright"})
        self.assertEqual(resolved["output_range"], {"flow": [0.0, 1.05]})
        self.assertEqual((resolved["frames_per_clip"], resolved["patch_size"], resolved["tubelet_size"]), (4, 16, 2))

    def test_unknown_keys_are_refused_at_every_level(self):
        for path in ("region__crop_size", "region__decoder__depth", "region__tracking__smoothing",
                     "region__readouts__size__dropout", "region__study_pooler__depth"):
            with self.assertRaisesRegex(ValueError, "unknown keys", msg=path):
                options(**{path: 1})
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            options(region__tracking__line_filter={"linearity": 0.9, "length": 0.2, "min_pixels": 20, "angle": 1})

    def test_missing_and_malformed_values_are_refused(self):
        for changes, message in (
            ({"region": None}, "is a mapping"),
            ({"region__num_classes": None}, r"needs \['num_classes'\]"),
            ({"region__tracking__margin": None}, r"needs \['margin'\]"),
            ({"region__decoder__checkpoint": None}, r"needs \['checkpoint'\]"),
            ({"query_mode": "shared"}, "query_mode: group"),
            ({"use_view_embedding": True}, "no view embedding"),
            ({"region__crop_resolution": 40}, "divisible"),
            ({"region__crops_per_group": -1}, "non-negative integer"),
            ({"region__crops_per_group": 1.5}, "non-negative integer"),
            ({"region__source_resolution": 0}, "positive integer"),
            ({"region__groups__b": [2, 4]}, "distinct classes"),
            ({"region__groups__b": []}, "distinct classes"),
            ({"region__decoder__blocks": [1, 1]}, "distinct encoder block"),
            ({"region__decoder__width": 12}, "multiple of 8"),
            ({"region__tracking__min_side": 1.5}, "fractions"),
            ({"region__tracking__line_filter": {"linearity": 0.9}}, "needs"),
            ({"region__readouts__size__kind": "mlp"}, "is one of"),
            ({"region__readouts__size__hidden": None}, r"needs \['hidden'\]"),
            ({"region__readouts__size__bias": 1.0}, r"takes no \['bias'\]"),
            ({"region__readouts__bright__hidden": 8}, r"takes no \['hidden'\]"),
            ({"region__readouts__flow__groups": ["a", "c"]}, "distinct groups"),
            ({"region__readouts__flow__study_head": "any_bright"}, r"takes no \['study_head'\]"),
            ({"region__readouts__bright__study_head": "flow"}, "names a binary head"),
            ({"region__readouts__bright__study_head": None}, r"reads the heads \['any_bright'\]"),
            ({"region__study_pooler": None}, "exactly when"),
            ({"region__study_pooler__width": 15}, "multiple of `num_heads`"),
            ({"region__output_range": {"flow_low": [0, 1]}}, "regression head"),
            ({"region__output_range": {"flow": [1, 1]}}, "low < high"),
        ):
            with self.assertRaisesRegex(ValueError, message, msg=str(changes)):
                options(**changes)

    def test_every_head_is_read_exactly_once_from_a_known_group(self):
        with self.assertRaisesRegex(ValueError, r"No readout .* reads the heads \['flow_low'\]"):
            options(region__readouts__flow__heads=["flow"])
        with self.assertRaisesRegex(ValueError, r"reads \['a_size'\] more than once"):
            options(region__readouts__flow__heads=["flow", "flow_low", "a_size"])
        with self.assertRaisesRegex(ValueError, "does not list"):
            options(region__readouts__flow__heads=["flow", "flow_low", "ghost"])
        with self.assertRaisesRegex(ValueError, "task_groups.a_size"):
            options(task_groups__a_size="c")
        with self.assertRaisesRegex(ValueError, r"names heads \['flow'\] that no group readout reads"):
            options(task_groups__flow="a")

    def test_more_than_one_clip_per_video_is_refused(self):
        classifier = copy.deepcopy(CLASSIFIER)
        with self.assertRaisesRegex(ValueError, "one clip per video"):
            region_options(classifier, {**DATA, "num_clips_per_video": 2}, PRETRAINED, head_specs(classifier))


class TestStudyTokenPooler(unittest.TestCase):

    def test_offsets_start_at_zero_and_study_logits_near_the_bias(self):
        torch.manual_seed(0)
        pooler = StudyTokenPooler(12, num_groups=3, num_findings=2, width=16, queries=2, num_heads=2, bias=-2.0)
        offsets, logits = pooler(torch.randn(4, 8, 12))
        self.assertEqual((tuple(offsets.shape), tuple(logits.shape)), ((2, 3), (2,)))
        self.assertTrue(torch.equal(offsets, torch.zeros(2, 3)))
        torch.testing.assert_close(logits, torch.full((2,), -2.0), atol=0.1, rtol=0)


class TestRegionStudyProbe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.encoder = tiny_encoder()
        cls.specs = head_specs(CLASSIFIER)
        labels = np.zeros((3, FRAMES, RES, RES), dtype=np.int64)
        blob(labels, 0, 1, 4, 4, 8)      # group a: view 0 (64 px) and view 1 (144 px).
        blob(labels, 1, 1, 2, 2, 12)
        blob(labels, 0, 3, 18, 18, 10)   # group b: view 0 only.
        cls.labels = labels

    def setUp(self):
        self.calls = []
        handle = self.encoder.model.register_forward_hook(lambda m, i, o: self.calls.append(tuple(i[0].shape)))
        self.addCleanup(handle.remove)
        torch.manual_seed(2)
        self.clips = torch.rand(4, 3, FRAMES, SOURCE, SOURCE)
        self.mask = torch.tensor([True, True, True, False])

    def probe(self, labels="default", **changes):
        torch.manual_seed(1)
        probe = RegionStudyProbe(192, self.specs, attention_hidden=8, normalization=NORMALIZATION,
                                 **options(**changes)).eval()
        if labels is not None:
            paint(probe, self.labels if isinstance(labels, str) else labels)
        return probe

    def test_one_study_takes_exactly_two_encoder_calls(self):
        probe = self.probe()
        with torch.no_grad():
            outputs, regions = probe.forward_study(self.encoder, self.clips, self.mask)
        self.assertEqual(self.calls, [(3, 3, FRAMES, RES, RES), (3, 3, FRAMES, CROP, CROP)])
        self.assertEqual(regions["encoder_passes"], 2)
        self.assertEqual(list(outputs), list(self.specs))
        self.assertEqual({tuple(v.shape) for v in outputs.values()}, {(1, 1)})
        self.assertTrue(all(torch.isfinite(v).all() and v.dtype == torch.float32 for v in outputs.values()))
        # Group a: the larger mask (view 1) first, then view 0; group b: view 0.
        self.assertEqual([(view, group) for view, group, _ in regions["crops"]], [(1, "a"), (0, "a"), (0, "b")])
        self.assertEqual(tuple(regions["features"].shape), (2, 192))

    def test_a_study_without_regions_takes_one_encoder_call(self):
        probe = self.probe(labels=np.zeros_like(self.labels))
        with torch.no_grad():
            outputs, regions = probe.forward_study(self.encoder, self.clips, self.mask)
        self.assertEqual(self.calls, [(3, 3, FRAMES, RES, RES)])
        self.assertEqual((regions["encoder_passes"], regions["crops"]), (1, []))
        self.assertEqual(list(outputs), list(self.specs))

    def test_crops_per_group_limits_the_crops_and_zero_skips_pass_two(self):
        with torch.no_grad():
            one = self.probe(region__crops_per_group=1).forward_study(self.encoder, self.clips, self.mask)[1]
            none = self.probe(region__crops_per_group=0).forward_study(self.encoder, self.clips, self.mask)[1]
        self.assertEqual([(view, group) for view, group, _ in one["crops"]], [(1, "a"), (0, "b")])
        self.assertEqual((none["crops"], none["encoder_passes"]), ([], 1))

    def test_padded_slots_do_not_change_the_outputs(self):
        probe = self.probe()
        other = self.clips.clone()
        other[3] = 0.0
        with torch.no_grad():
            a, b = probe(self.encoder, self.clips, self.mask), probe(self.encoder, other, self.mask)
        for head in a:
            self.assertTrue(torch.equal(a[head], b[head]), head)

    def test_the_crop_is_cut_from_the_source_frames_at_the_tracked_box(self):
        probe = self.probe(region__crops_per_group=1)
        seen = []
        handle = self.encoder.model.register_forward_hook(lambda m, i, o: seen.append(i[0]))
        self.addCleanup(handle.remove)
        with torch.no_grad():
            _, regions = probe.forward_study(self.encoder, self.clips, self.mask)
        view, group, boxes = regions["crops"][1]
        self.assertEqual((view, group), (0, "b"))
        expected_boxes = tracked_boxes(np.isin(self.labels[0], [2, 3]), (SOURCE, SOURCE), **TRACKING)
        np.testing.assert_allclose(boxes, expected_boxes)
        expected = crop_clip(self.clips[0].transpose(0, 1), torch.from_numpy(boxes), CROP).transpose(0, 1)
        torch.testing.assert_close(seen[1][1], expected)   # the default pixel normalization is the identity.

    def test_pixel_normalization_is_applied_before_each_encoder_call(self):
        torch.manual_seed(1)
        probe = RegionStudyProbe(192, self.specs, attention_hidden=8, normalization=NORMALIZATION,
                                 pixel_normalization=((0.5, 0.4, 0.3), (0.2, 0.25, 0.5)), **options()).eval()
        paint(probe, self.labels)
        seen = []
        handle = self.encoder.model.register_forward_hook(lambda m, i, o: seen.append(i[0]))
        self.addCleanup(handle.remove)
        with torch.no_grad():
            probe(self.encoder, self.clips, self.mask)
        full = F.interpolate(self.clips[:3].transpose(1, 2).flatten(0, 1), (RES, RES), mode="bilinear",
                             align_corners=False).reshape(3, FRAMES, 3, RES, RES).transpose(1, 2)
        mean = torch.tensor([0.5, 0.4, 0.3]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.2, 0.25, 0.5]).view(1, 3, 1, 1, 1)
        torch.testing.assert_close(seen[0], (full - mean) / std)

    def test_initial_outputs_follow_the_configured_priors_and_ranges(self):
        probe = self.probe()
        with torch.no_grad():
            outputs = probe(self.encoder, self.clips, self.mask)
        for head in ("a_bright", "b_bright"):   # zero weights and zero pooler offsets: the bias.
            torch.testing.assert_close(outputs[head], torch.full((1, 1), -4.0))
        torch.testing.assert_close(outputs["any_bright"], torch.full((1, 1), -2.0), atol=0.1, rtol=0)
        flow = outputs["flow"] * 0.05 + 0.5       # back from the target normalization.
        self.assertTrue(0.0 < flow.item() < 1.05)

    def test_the_study_pooler_offset_is_added_to_its_readout_only(self):
        probe = self.probe()
        with torch.no_grad():
            before = probe(self.encoder, self.clips, self.mask)
            probe.study_pooler.segment[0, 1] = 1.0
            after = probe(self.encoder, self.clips, self.mask)
        self.assertFalse(torch.equal(before["b_bright"], after["b_bright"]))
        for head in self.specs:
            if head != "b_bright":
                self.assertTrue(torch.equal(before[head], after[head]), head)

    def test_the_loss_trains_the_encoder_and_the_probe_but_not_the_decoder(self):
        probe = self.probe(labels=None).train()
        encoder = copy.deepcopy(self.encoder)
        outputs = probe(encoder, self.clips, self.mask)
        targets = torch.tensor([[0.3, 1.0, 0.1, 0.0, 1.0, 0.0, 1.0, 0.2, 1.0]])
        loss, parts = multitask_loss(self.specs, outputs, targets)
        loss.backward()
        self.assertEqual(set(parts), set(self.specs))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in probe.region_decoder.parameters()))
        self.assertGreater(sum(p.grad.abs().sum() for p in encoder.model.blocks[-1].parameters()).item(), 0)
        trained = {name for name, p in probe.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        for name in ("view_proj.1.weight", "attention.V.weight", "group_embedding", "heads.a_size.1.weight",
                     "trunks.size.a.1.weight", "trunks.flow.1.weight", "heads.b_bright.1.weight",
                     "study_pooler.any", "study_pooler.query"):
            self.assertIn(name, trained)

    def test_the_state_round_trips_through_a_weights_only_checkpoint(self):
        probe = self.probe()
        with torch.no_grad():
            for p in probe.parameters():
                p.add_(0.05 * torch.randn_like(p))
        buffer = io.BytesIO()
        torch.save({"probes": [probe.state_dict()]}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)["probes"][0]
        self.assertTrue(any(k.startswith("region_decoder.") for k in state))
        self.assertFalse(any(k.startswith("pixel_") for k in state))
        other = self.probe()
        other.load_state_dict(state)
        with torch.no_grad():
            a, b = probe(self.encoder, self.clips, self.mask), other(self.encoder, self.clips, self.mask)
        for head in a:
            self.assertTrue(torch.equal(a[head], b[head]), head)

    def test_probe_weights_must_fit_except_for_a_missing_study_pooler(self):
        trained = self.probe()
        with torch.no_grad():
            for p in trained.parameters():
                p.add_(0.05 * torch.randn_like(p))
        decoder = trained.region_decoder.state_dict()
        state = {k: v for k, v in trained.state_dict().items() if not k.startswith("region_decoder.")}
        other = self.probe(labels=None)
        self.assertFalse(other.load_weights(decoder, state))
        for name, value in trained.state_dict().items():
            self.assertTrue(torch.equal(other.state_dict()[name], value), name)

        # Without the whole study pooler: the rest is loaded and the pooler keeps its initial values.
        other = self.probe(labels=None)
        initial = {k: v.clone() for k, v in other.state_dict().items() if k.startswith("study_pooler.")}
        without = {k: v for k, v in state.items() if not k.startswith("study_pooler.")}
        self.assertTrue(other.load_weights(decoder, without))
        for name, value in other.state_dict().items():
            self.assertTrue(torch.equal(value, initial[name] if name in initial else trained.state_dict()[name]), name)

        # Anything else that is missing, and anything unexpected, is refused.
        for wrong in ({k: v for k, v in state.items() if k != "study_pooler.query"},
                      {k: v for k, v in state.items() if k != "crop_query"},
                      {k: v for k, v in without.items() if not k.startswith("heads.b_bright.")},
                      {**state, "extra": torch.zeros(1)},
                      {**state, **{f"region_decoder.{k}": v for k, v in decoder.items()}}):
            with self.assertRaisesRegex(ValueError, "do not fit this probe"):
                self.probe(labels=None).load_weights(decoder, wrong)

    def test_inputs_of_another_size_are_refused(self):
        probe = self.probe()
        with self.assertRaisesRegex(ValueError, "source_resolution"):
            probe(self.encoder, self.clips[..., :48, :48], self.mask)
        with self.assertRaisesRegex(ValueError, "at least one video"):
            probe(self.encoder, self.clips, torch.zeros(4, dtype=torch.bool))

    def test_an_encoder_with_another_token_grid_is_refused(self):
        probe = self.probe(labels=None)
        probe.patch_size = 8
        with self.assertRaisesRegex(ValueError, "assumes 32 tokens"):
            probe(self.encoder, self.clips, self.mask)

    def test_output_range_needs_the_head_normalization(self):
        with self.assertRaisesRegex(ValueError, "needs the target normalization"):
            RegionStudyProbe(192, self.specs, attention_hidden=8, normalization={}, **options())


class TestDeepcoroJepaConfig(unittest.TestCase):
    """configs/eval/vitl/study_multitask_deepcoro_jepa.yaml is valid and builds the probe it describes."""

    def test_the_shipped_config_builds_a_probe_with_every_head(self):
        root = Path(__file__).resolve().parents[2]
        with open(root / "configs/eval/vitl/study_multitask_deepcoro_jepa.yaml") as f:
            config = yaml.safe_load(f)
        experiment = config["experiment"]
        classifier = experiment["classifier"]
        self.assertEqual((config["eval_name"], classifier["pooling_mode"]), ("study_multitask_frozen", "region"))
        self.assertEqual((config["tasks_per_node"], len(experiment["optimization"]["multihead_kwargs"])), (1, 1))
        specs = head_specs(classifier)
        resolved = region_options(classifier, experiment["data"], config["model_kwargs"], specs)
        groups = list(resolved["groups"])
        self.assertEqual((len(specs), len(groups), resolved["decoder"]["blocks"]), (97, 18, [5, 11, 17]))
        # The label definitions are in the head names: >= 70 % per segment, >= 50 % for the left main.
        findings = [h for h in resolved["readouts"]["stenosis"]["heads"] if specs[h]["task"] == "binary"]
        self.assertEqual(findings, ["left_main_ge50" if g == "left_main" else f"{g}_ge70" for g in groups])
        self.assertEqual(resolved["readouts"]["pressure"]["heads"], ["ffr", "ifr", "ffr_le80", "ifr_le89"])
        normalization = {h: {"mean": classifier["target_mean"][h], "std": classifier["target_std"][h]}
                         for h in classifier["target_mean"]}
        self.assertEqual(set(normalization), {h for h, spec in specs.items() if spec["task"] == "regression"})
        probe = RegionStudyProbe(embed_dim=16, heads=specs, attention_hidden=classifier["attention_hidden"],
                                 normalization=normalization, **resolved).eval()
        with torch.no_grad():
            outputs = probe.read(torch.zeros(len(groups), 16), torch.zeros(2, 8, 16))
        self.assertEqual(list(outputs), list(specs))
        self.assertTrue(all(v.shape == (1, 1) and bool(torch.isfinite(v).all()) for v in outputs.values()))


class TestDeepcoroJepaFinetuneExample(unittest.TestCase):
    """configs/eval/vitl/study_multitask_deepcoro_jepa_finetune_example.yaml: the shipped config with
    a warm start, and a header that describes a manifest the study dataset accepts."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.path = root / "configs/eval/vitl/study_multitask_deepcoro_jepa_finetune_example.yaml"
        with open(cls.path) as f:
            cls.example = yaml.safe_load(f)
        with open(root / "configs/eval/vitl/study_multitask_deepcoro_jepa.yaml") as f:
            cls.shipped = yaml.safe_load(f)
        cls.heads = list(head_specs(cls.example["experiment"]["classifier"]))

    def manifest(self, filled):
        """Two train studies, one validation and one test study of two cines each, one patient per
        study; the `filled` target columns hold 0 and the others are empty."""
        rows = [{"video_path": f"/your_videos/study_{s:04d}/run_{v:02d}.avi", "study_id": f"study_{s:04d}",
                 "patient_id": f"patient_{s:04d}", "split": split,
                 **{h: (0 if h in filled else None) for h in self.heads}}
                for s, split in enumerate(["train", "train", "val", "test"]) for v in range(2)]
        return pd.DataFrame(rows)

    def test_it_is_the_shipped_config_with_a_warm_start(self):
        example, shipped = copy.deepcopy(self.example), copy.deepcopy(self.shipped)
        region = example["experiment"]["classifier"]["region"]
        self.assertEqual(region["init_checkpoint"], region["decoder"]["checkpoint"])
        self.assertEqual(region["init_checkpoint"], example["model_kwargs"]["checkpoint"])
        self.assertIsNone(shipped["experiment"]["classifier"]["region"]["init_checkpoint"])
        for config in (example, shipped):
            del config["folder"], config["tag"], config["experiment"]["data"]["data_filename"]
            del config["experiment"]["classifier"]["region"]["init_checkpoint"]
        self.assertEqual(example, shipped)

    def test_the_manifest_of_the_header_is_read_with_every_label_or_stenosis_alone(self):
        data = self.example["experiment"]["data"]
        stenosis = [h for h in self.heads if h.endswith(("_stenosis", "_ge70", "_ge50"))]
        self.assertEqual((len(self.heads), len(stenosis)), (97, 36))
        with tempfile.TemporaryDirectory() as tmp:
            for filled in (self.heads, stenosis):
                path = Path(tmp) / "manifest.csv"
                self.manifest(set(filled)).to_csv(path, index=False)
                splits = {}
                for split, studies in (("train", 2), ("val", 1), ("test", 1)):
                    rows = read_split({**data, "data_filename": str(path)}, split)
                    splits[split] = VideoStudyDataset(
                        rows, path_column=data["datapoint_loc_label"], group_column=data["groupby_column"],
                        target_columns=self.heads, num_videos=data["num_videos"],
                        shuffle_videos=data["shuffle_videos"], min_videos=data["min_videos"])
                    self.assertEqual(len(splits[split]), studies)
                    labelled = np.isfinite(splits[split].targets).all(axis=0)
                    self.assertEqual([h for h, known in zip(self.heads, labelled) if known], list(filled))
                self.assertFalse(splits["train"].patient_ids & splits["val"].patient_ids)

    def test_a_missing_column_and_a_missing_patient_id_are_refused(self):
        table = self.manifest(set(self.heads))
        with self.assertRaisesRegex(ValueError, "has no `any_cto` column"):
            VideoStudyDataset(table.drop(columns="any_cto"), target_columns=self.heads)
        with self.assertRaisesRegex(ValueError, "needs a patient id column"):
            VideoStudyDataset(table.drop(columns="patient_id"), target_columns=self.heads)

    def test_the_header_names_real_columns(self):
        header = [line[1:].strip() for line in self.path.read_text().splitlines() if line.startswith("#")]
        sample = next(line for line in header if line.startswith("video_path,study_id")).split(",")
        self.assertEqual(sample[:4], ["video_path", "study_id", "patient_id", "split"])
        self.assertTrue(set(sample[4:]) <= set(self.heads))
        words = set(" ".join(header).replace(",", " ").split())
        self.assertTrue(set(self.example["experiment"]["classifier"]["region"]["groups"]) <= words)


if __name__ == "__main__":
    unittest.main()
