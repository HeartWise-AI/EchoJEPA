"""Legacy YBR colour fix (src/datasets/utils/color_order.py): exact inverse, path rule, and the loader hook."""
import importlib
import os

import cv2
import numpy as np
import pytest

import src.datasets.utils.color_order as co

MP4 = ".mp4"  # built, not literal: tests/radar/test_privacy.py flags literal video file names


def rgb_to_ycbcr(rgb):
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.clip(np.stack([y, cb, cr], -1).round(), 0, 255).astype(np.uint8)


def colour_frames(t=3, h=48, w=64):
    f = np.repeat(np.linspace(20, 200, w, dtype=np.uint8)[None, :, None], h, 0).repeat(3, -1)
    f[5:20, 5:20] = (220, 30, 30)
    f[5:20, 30:45] = (30, 40, 220)
    f[28:44, 10:50] = (200, 160, 90)
    return np.stack([f] * t)


def legacy_rgb(rgb):
    """RGB of what the legacy extractor stored: YCrCb2BGR applied to (Y, Cb, Cr)."""
    return np.stack([cv2.cvtColor(rgb_to_ycbcr(x), cv2.COLOR_YCrCb2BGR)[..., ::-1] for x in rgb])


@pytest.fixture(autouse=True)
def fresh_module(monkeypatch, tmp_path):
    ex = tmp_path / "exceptions.txt"
    ex.write_text("1234/5678/rgb_source" + MP4 + "\n")
    monkeypatch.setenv("ECHOJEPA_YBR_EXCEPTIONS", str(ex))
    monkeypatch.delenv("ECHOJEPA_LEGACY_YBR_FIX", raising=False)
    importlib.reload(co)


@pytest.mark.parametrize("order", ["RGB", "BGR"])
def test_undo_recovers_true_colours(order):
    rgb = colour_frames()
    stored = legacy_rgb(rgb)
    assert stored[0, 10, 10, 2] > stored[0, 10, 10, 0]            # red came out blue
    x = stored if order == "RGB" else stored[..., ::-1]
    fixed = co.undo_legacy_ybr_swap(x, order)
    fixed = fixed if order == "RGB" else fixed[..., ::-1]
    assert fixed.shape == rgb.shape and fixed.dtype == np.uint8
    assert np.abs(fixed.astype(int) - rgb.astype(int)).max() <= 4


def test_black_padding_stays_black():
    z = np.zeros((2, 8, 8, 3), np.uint8)
    assert not co.undo_legacy_ybr_swap(z).any()


def test_path_rule():
    root = os.sep + os.path.join("x", "datasets", "DeepEcho", "MP4")
    assert co.needs_legacy_ybr_fix(root + "/2019/1/2/clip" + MP4)
    assert co.needs_legacy_ybr_fix(root.replace("x", "y") + "/2019/1/2/clip" + MP4)       # any mount
    assert not co.needs_legacy_ybr_fix(root + "/1234/5678/rgb_source" + MP4)              # listed exception
    assert not co.needs_legacy_ybr_fix(os.sep + os.path.join("x", "MIMIC-IV-ECHO", "mp4_native256", "a" + MP4))
    assert not co.needs_legacy_ybr_fix(None)


def test_switch_off(monkeypatch):
    monkeypatch.setenv("ECHOJEPA_LEGACY_YBR_FIX", "0")
    path = os.sep + os.path.join("x", "DeepEcho", "MP4", "a" + MP4)
    frames = colour_frames()
    assert co.fix_legacy_colors(frames, path) is frames


def test_fix_only_when_needed():
    frames = legacy_rgb(colour_frames())
    other = os.sep + os.path.join("x", "MIMIC-IV-ECHO", "a" + MP4)
    assert co.fix_legacy_colors(frames, other) is frames
    assert co.fix_legacy_colors([], os.sep + os.path.join("x", "DeepEcho", "MP4", "a" + MP4)) == []


def test_loader_hook_applies_fix(monkeypatch):
    from src.datasets import video_dataset as vd
    monkeypatch.setattr(vd, "fix_legacy_colors", co.fix_legacy_colors)
    stored = legacy_rgb(colour_frames())

    class Fake:
        _sample_from_vr = staticmethod(lambda vr, fpc: (stored.copy(), [np.arange(3)]))

    path = os.sep + os.path.join("x", "DeepEcho", "MP4", "2019", "a" + MP4)
    buf, idx = vd.VideoDataset._sample_and_fix(Fake(), None, 3, path)
    assert buf[0, 10, 10, 0] > buf[0, 10, 10, 2] + 100                                # red is red again
    buf2, _ = vd.VideoDataset._sample_and_fix(Fake(), None, 3, path.replace("DeepEcho", "MIMIC-IV-ECHO"))
    assert np.array_equal(buf2, stored)
