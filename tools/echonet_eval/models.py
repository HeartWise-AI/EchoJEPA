"""Model loaders for the EchoNet evaluation toolkit.

All weights live on the NAS under /media/data1/models/EchoNet (see README there); the code
for the Segmentation models is imported from the echonet/measurements checkout, with the
`cvair` stand-in on sys.path.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50

NAS = os.environ.get("ECHONET_NAS", "/media/data1/models/EchoNet")
REPO = os.environ.get("ECHONET_MEAS_REPO", "/volume/echonet-measurements")
CVAIR_PARENT = os.environ.get("CVAIR_PARENT", os.path.dirname(os.path.abspath(__file__)))  # holds cvair/ shim

if CVAIR_PARENT not in sys.path:
    sys.path.append(CVAIR_PARENT)  # provides the cvair shim for Segmentation/utils.py


def _load_module(name: str, path: str):
    """Import a repo file under a unique module name (both repos have a `utils.py`)."""
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.append(d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def meas_utils():
    return _load_module("echonet_meas_utils", os.path.join(REPO, "Measurement", "utils.py"))


def seg_utils():
    return _load_module("echonet_seg_utils", os.path.join(REPO, "Segmentation", "utils.py"))

TWO_D_MODELS = ["ivs", "lvid", "lvpw", "aorta", "aortic_root", "la", "rv_base", "pa", "ivc"]
DOPPLER_MODELS = ["avvmax", "trvmax", "mrvmax", "lvotvmax", "latevel", "medevel"]
# which OCR label routes to which Doppler model
LABEL_TO_DOPPLER = {"TR_VMAX": "trvmax", "AV_VMAX": "avvmax", "MR_VMAX": "mrvmax", "LVOT_VMAX": "lvotvmax",
                    "LAT_E_PRIME": "latevel", "SEPT_E_PRIME": "medevel"}
# VTI targets evaluated here (PV / RVOT weights exist on the NAS but are not loaded or evaluated)
LABEL_TO_VTI = {"LVOT_VMAX": "LVOT", "LVOT_VTI": "LVOT", "AV_VMAX": "AV", "AV_VTI": "AV", "MV_E": "MV", "MV_VTI": "MV"}


def _load_ckpt_into(backbone, path, device):
    w = torch.load(path, map_location="cpu", weights_only=False)
    w = w.get("state_dict", w)
    w = {k.replace("m.", "", 1) if k.startswith("m.") else k: v for k, v in w.items()}
    backbone.load_state_dict(w)
    return backbone.to(device).eval()


class Calipers2D:
    """The 9 Cedars 2D linear-measurement models: DeepLabV3 (2 output maps = 2 landmarks)."""

    def __init__(self, device, names=TWO_D_MODELS):
        self.seg2coord = meas_utils().segmentation_to_coordinates
        self.device = device
        self.models = {n: _load_ckpt_into(deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2), f"{NAS}/measurements/weights/2D_models/{n}_weights.ckpt", device) for n in names}

    @torch.no_grad()
    def predict(self, name: str, frames_640x480: np.ndarray, batch: int = 32):
        """frames (T,480,640,3) uint8 -> (T,2,2) landmark coords in (y,x) model pixels."""
        m = self.models[name]
        out = []
        for i in range(0, len(frames_640x480), batch):
            x = torch.from_numpy(frames_640x480[i:i + batch]).permute(0, 3, 1, 2).float().div(255).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                logits = m(x)["out"]
            probs = torch.sigmoid(logits.float())
            coords = self.seg2coord(probs, normalize=False, order="YX")  # (B,2,2)
            out.append(coords.cpu().numpy())
        return np.concatenate(out)


class DopplerVmax:
    """Cedars Doppler peak-velocity models: DeepLabV3 (1 map) heat-map, argmax = peak."""

    def __init__(self, device, names=DOPPLER_MODELS):
        self.device = device
        self.models = {n: _load_ckpt_into(deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1), f"{NAS}/measurements/weights/Doppler_models/{n}_weights.ckpt", device) for n in names}

    @torch.no_grad()
    def predict(self, name: str, strip_rgb: np.ndarray):
        """strip (h,w,3) uint8 from the Doppler-strip origin downward -> (x, y_in_strip, peak_prob)."""
        x = torch.from_numpy(strip_rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        logit = torch.sigmoid(self.models[name](x)["out"].float())[0, 0]
        idx = int(torch.argmax(logit))
        y, xx = divmod(idx, logit.shape[1])
        return int(xx), int(y), float(logit.max())


class AreaSeg:
    def __init__(self, device, targets=("LA_AREA", "RA_AREA")):
        segutils = seg_utils()
        self.u = segutils
        self.device = device
        import glob
        self.models = {t: segutils.load_seg_model(glob.glob(f"{NAS}/segmentation/weights/Area/{t}/*.ckpt")[0], device) for t in targets}

    @torch.no_grad()
    def predict(self, target: str, frames_640x480: np.ndarray, batch: int = 16):
        """Returns (T,480,640) binary masks and per-frame mean confidence, at model resolution.
        Segmentation/utils.load_seg_model registers a forward hook that returns output['out'], so m.m(x) is a tensor."""
        m = self.models[target]
        masks, confs = [], []
        for i in range(0, len(frames_640x480), batch):
            x = torch.from_numpy(frames_640x480[i:i + batch]).permute(0, 3, 1, 2).float().div(255).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                logits = m.m(x)
            prob = torch.sigmoid(logits.float())[:, 0]
            mk = prob >= 0.5
            masks.append(mk.cpu().numpy().astype(np.uint8))
            c = torch.where(mk, prob, torch.nan).nanmean(dim=(1, 2))
            confs.append(torch.nan_to_num(c).cpu().numpy())
        return np.concatenate(masks), np.concatenate(confs)


class VTISeg:
    def __init__(self, device, targets=("LVOT", "MV", "AV")):
        import glob
        segutils = seg_utils()
        self.u = segutils
        self.device = device
        self.gpu = device.index if device.type == "cuda" else 0
        self.seg = {t: segutils.load_seg_model(glob.glob(f"{NAS}/segmentation/weights/{t}/*Segm*.ckpt")[0], device) for t in targets}
        self.yolo = {t: segutils.load_yolo_model(glob.glob(f"{NAS}/segmentation/weights/{t}/*YOLO*.pt")[0]) for t in targets}

    def predict(self, target: str, strip_bgr: np.ndarray, baseline_ratio: float = 0.5) -> dict:
        r = self.u.process_doppler_still(doppler_bgr=strip_bgr, seg_model=self.seg[target], yolo_model=self.yolo[target],
                                          device=self.device, model_key=target.lower(), baseline_ratio=baseline_ratio, gpu=self.gpu)
        return r


class EchoNetDynamic:
    MEAN, STD = 32.7, 50.0  # EchoNet-Dynamic training statistics in 0-255 pixel units (echonet.utils.get_mean_and_std on raw uint8 videos); not in the checkpoint

    def __init__(self, device):
        self.device = device
        m = torchvision.models.video.r2plus1d_18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 1)
        ck = torch.load(f"{NAS}/dynamic/weights/r2plus1d_18_32_2_pretrained.pt", map_location="cpu", weights_only=False)
        m.load_state_dict({k.replace("module.", ""): v for k, v in ck["state_dict"].items()})
        self.ef, self.frames_n, self.period = m.to(device).eval(), ck["frames"], ck["period"]
        s = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False)
        s.classifier[-1] = torch.nn.Conv2d(s.classifier[-1].in_channels, 1, 1)
        ck2 = torch.load(f"{NAS}/dynamic/weights/deeplabv3_resnet50_random.pt", map_location="cpu", weights_only=False)
        s.load_state_dict({k.replace("module.", ""): v for k, v in ck2["state_dict"].items()}, strict=False)
        self.seg = s.to(device).eval()

    @staticmethod
    def to_112(frames_rgb: np.ndarray, reg) -> np.ndarray:
        import cv2
        a = frames_rgb
        if reg is not None:
            a = a[:, reg.y0:reg.y1 + 1, reg.x0:reg.x1 + 1]
        T, H, W, _ = a.shape
        s = min(H, W); y = (H - s) // 2; x = (W - s) // 2
        a = a[:, y:y + s, x:x + s]
        return np.stack([cv2.resize(f, (112, 112), interpolation=cv2.INTER_CUBIC) for f in a])

    @torch.no_grad()
    def predict(self, frames112: np.ndarray, max_clips: int = 8, period: Optional[int] = None) -> dict:
        """period: temporal stride between the 32 sampled frames. The released model was trained on
        50 fps videos with period 2 (25 fps effective); pass round(fps / 25) for other frame rates."""
        period = int(period or self.period)
        x = torch.from_numpy((frames112.astype(np.float32) - self.MEAN) / self.STD).permute(3, 0, 1, 2)
        T = x.shape[1]; need = self.frames_n * period
        if T < need:
            x = torch.cat([x] * int(np.ceil(need / T)), 1)[:, :need]; T = need
        starts = list(range(0, T - need + 1, period))
        if len(starts) > max_clips:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_clips).round().astype(int)]
        clips = torch.stack([x[:, s:s + need:period] for s in starts]).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            preds = self.ef(clips).float().squeeze(1).cpu().numpy()
        xs = torch.from_numpy((frames112.astype(np.float32) - self.MEAN) / self.STD).permute(0, 3, 1, 2)
        areas = []
        for i in range(0, len(xs), 64):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                mk = self.seg(xs[i:i + 64].to(self.device))["out"][:, 0] > 0
            areas.append(mk.sum(dim=(1, 2)).cpu().numpy())
        area = np.concatenate(areas).astype(float)
        return {"ef_clips": preds.tolist(), "ef_mean": float(preds.mean()), "lv_area_max": float(area.max()), "lv_area_min": float(area.min()),
                "lv_frac_change": float(1 - area.min() / max(area.max(), 1)), "lv_frac_frames": float((area > 50).mean())}
