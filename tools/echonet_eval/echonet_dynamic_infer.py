"""EchoNet-Dynamic inference on MHI DICOMs: EF (r2plus1d_18, 32 frames, period 2) and LV
segmentation (deeplabv3_resnet50) from the v1.0.0 release weights.

Preprocessing mirrors scripts/ConvertDICOMToAVI.ipynb as far as the DICOM allows: crop to the
ultrasound sector (SequenceOfUltrasoundRegions, RegionDataType 1) so that text/ECG trace is
excluded, resize to 112x112 (cubic), RGB. Normalisation uses the EchoNet-Dynamic training-set
statistics (mean ~32.7, std ~50.0 per channel, as published with the dataset); the released
checkpoint does not store them, so this is an approximation.

Usage:
    python echonet_dynamic_infer.py --csv dicoms.csv --out out_dir
    # dicoms.csv needs columns dcm_path and view (A4C clips only make sense for EF)
"""
import argparse, os, sys, json
import numpy as np, pandas as pd, pydicom, torch, torchvision, cv2
from pydicom.pixel_data_handlers.util import convert_color_space

MEAN, STD = 32.7, 50.0
NAS = os.environ.get("ECHONET_NAS", "/media/data1/models/EchoNet")

def dicom_frames(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 3:  # single frame HxWx3 or multi-frame gray
        arr = arr[None] if arr.shape[-1] == 3 else arr[..., None].repeat(3, -1)
    pi = str(ds.get("PhotometricInterpretation", "RGB"))
    # pydicom >= 3 already returns RGB for YBR data (as_rgb=True); converting again gives green/pink frames
    if pi.startswith("YBR") and int(pydicom.__version__.split(".")[0]) < 3:
        arr = np.stack([convert_color_space(f, pi, "RGB") for f in arr])
    # crop to the 2D tissue region if the tags exist
    reg = None
    for r in getattr(ds, "SequenceOfUltrasoundRegions", []) or []:
        if int(getattr(r, "RegionDataType", 0)) == 1:
            reg = (int(r.RegionLocationMinX0), int(r.RegionLocationMinY0), int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1)); break
    if reg:
        x0, y0, x1, y1 = reg; arr = arr[:, y0:y1 + 1, x0:x1 + 1]
    # square crop (centre) then resize to 112x112
    T, H, W, _ = arr.shape
    s = min(H, W); y = (H - s) // 2; x = (W - s) // 2
    arr = arr[:, y:y + s, x:x + s]
    frames = np.stack([cv2.resize(f, (112, 112), interpolation=cv2.INTER_CUBIC) for f in arr])
    fps = 1000.0 / float(ds.FrameTime) if "FrameTime" in ds and float(ds.FrameTime) > 0 else float(ds.get("CineRate", ds.get("RecommendedDisplayFrameRate", 30)) or 30)
    return frames, fps, reg

def ef_model(path, device):
    m = torchvision.models.video.r2plus1d_18(weights=None); m.fc = torch.nn.Linear(m.fc.in_features, 1); m.fc.bias.data[0] = 55.6
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}; m.load_state_dict(sd); return m.to(device).eval(), ck["frames"], ck["period"]

def seg_model(path, device):
    m = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=False); m.classifier[-1] = torch.nn.Conv2d(m.classifier[-1].in_channels, 1, 1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}; print("seg load:", m.load_state_dict(sd, strict=False)); return m.to(device).eval()

@torch.no_grad()
def run(dcm, tag, ef, seg, frames_n, period, device, out_dir, match_fps=True):
    frames, fps, reg = dicom_frames(dcm)               # T,112,112,3 uint8
    if match_fps:  # the model was trained at 50 fps with period 2 (25 fps effective); GE clips are 25-30 fps
        period = max(1, int(round(fps / 25.0)))
    x = (frames.astype(np.float32) - MEAN) / STD
    x = torch.from_numpy(x).permute(3, 0, 1, 2)         # 3,T,H,W
    T = x.shape[1]; need = frames_n * period
    if T < need:  # loop the clip like echonet.datasets.Echo does when too short
        x = torch.cat([x] * int(np.ceil(need / T)), 1)[:, :need]; T = need
    clips = [x[:, s:s + need:period] for s in range(0, T - need + 1, period)] or [x[:, :need:period]]
    preds = [float(ef(c[None].to(device)).item()) for c in clips]
    # LV segmentation on every original frame
    xs = torch.from_numpy((frames.astype(np.float32) - MEAN) / STD).permute(0, 3, 1, 2)
    masks = []
    for i in range(0, len(xs), 32):
        masks.append((seg(xs[i:i + 32].to(device))["out"][:, 0] > 0).cpu().numpy())
    mask = np.concatenate(masks); area = mask.reshape(len(mask), -1).sum(1)
    # overlay of the largest- and smallest-LV frames
    i_max, i_min = int(area.argmax()), int(area.argmin())
    panel = []
    for i in (i_max, i_min):
        f = frames[i].copy(); f[mask[i]] = (0.5 * f[mask[i]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8); panel.append(cv2.resize(f, (224, 224), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(os.path.join(out_dir, f"{tag}_lvseg_ED_ES.png"), cv2.cvtColor(np.concatenate(panel, 1), cv2.COLOR_RGB2BGR))
    res = dict(file=dcm, view=tag, n_frames=int(len(frames)), fps=fps, period=period, sector_crop=reg, ef_mean=float(np.mean(preds)), ef_min=float(min(preds)), ef_max=float(max(preds)),
               n_clips=len(clips), lv_area_px_max=int(area.max()), lv_area_px_min=int(area.min()), lv_area_fraction_change=float(1 - area.min() / max(area.max(), 1)), frac_frames_with_lv=float((area > 50).mean()))
    print(json.dumps(res)); return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True, help="CSV with columns dcm_path, view"); ap.add_argument("--rows", default="", help="comma-separated row indices (default: all)")
    ap.add_argument("--weights", default=f"{NAS}/dynamic/weights"); ap.add_argument("--out", default="echonet_dynamic_out"); ap.add_argument("--fixed_period", action="store_true", help="keep the released period 2 instead of matching the clip frame rate"); a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(a.out, exist_ok=True)
    ef, n, p = ef_model(os.path.join(a.weights, "r2plus1d_18_32_2_pretrained.pt"), device); seg = seg_model(os.path.join(a.weights, "deeplabv3_resnet50_random.pt"), device)
    df = pd.read_csv(a.csv); rows = []
    idx = [int(r) for r in a.rows.split(",")] if a.rows else list(range(len(df)))
    for i in idx:
        r = df.iloc[i]
        try: rows.append(run(r["dcm_path"], f"{r['view']}_row{i}", ef, seg, n, p, device, a.out, match_fps=not a.fixed_period))
        except Exception as e: print(json.dumps(dict(file=r["dcm_path"], view=r["view"], error=str(e)[:200])))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "echonet_dynamic_results.csv"), index=False)
