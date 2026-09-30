"""Render inference examples (contact sheets) of every EchoNet model on MHI DICOMs, with the corrected colour
handling. Output PNGs go to --out (default: the NAS inference_examples folder, suffix _v2)."""
import argparse, glob, os, sys, json
import numpy as np, pandas as pd, cv2, torch, pydicom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dicom_io as dio
from models import Calipers2D, DopplerVmax, AreaSeg, VTISeg, EchoNetDynamic, LABEL_TO_DOPPLER, LABEL_TO_VTI


def put(img, txt, y=22, color=(255, 255, 0)):
    cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3); cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def main(a):
    dev = torch.device("cuda:0"); os.makedirs(a.out, exist_ok=True)
    rows = pd.read_parquet(a.rows); coh = pd.read_parquet(a.cohort).set_index("StudyInstanceUID")
    rng = np.random.default_rng(a.seed)
    cal = Calipers2D(dev, names=["ivs", "lvid", "lvpw", "la", "aorta", "rv_base", "ivc"]); dop = DopplerVmax(dev); area = AreaSeg(dev, targets=("LA_AREA",))
    vti = VTISeg(dev, targets=("LVOT", "AV", "MV")); dyn = EchoNetDynamic(dev); ocr = dio.LabelOCR(gpu=True)
    # ---- 2D calipers on PLAX (ivs/lvid/lvpw/la/aorta), A4C (rv_base), SUBCOSTAL (ivc)
    panels = []
    for view, models, n in [("PLAX", ["ivs", "lvid", "lvpw", "la", "aorta"], 3), ("A4C", ["rv_base"], 1), ("SUBCOSTAL", ["ivc"], 1)]:
        sub = rows[(rows.predicted_class == view) & (rows.avi_status == "success")].sample(n, random_state=a.seed)
        for _, r in sub.iterrows():
            fr, ds = dio.read_frames(r.dcm_path, max_frames=16); cv = dio.to_canvas_3x4(fr, dio.tissue_region(dio.regions(ds)))
            img = cv.frames[0].copy(); y = 22
            for m, col in zip(models, [(255, 60, 60), (60, 255, 60), (60, 160, 255), (255, 200, 0), (255, 0, 255)]):
                c = cal.predict(m, cv.frames); d = np.linalg.norm(c[:, 0] - c[:, 1], axis=1) * cv.cm_per_model_px * 10; c0 = c[0]
                cv2.line(img, (int(c0[0, 1]), int(c0[0, 0])), (int(c0[1, 1]), int(c0[1, 0])), col, 2)
                for (yy, xx) in c0: cv2.circle(img, (int(xx), int(yy)), 5, col, 2)
                put(img, f"{m}: {d.min():.0f}-{d.max():.0f} mm", y, col); y += 20
            rep = coh.loc[r.StudyInstanceUID] if r.StudyInstanceUID in coh.index else None
            if rep is not None and view == "PLAX":
                put(img, f"report LVIDd {rep.lvidd*10:.0f} IVSd {rep.ivsd*10:.0f} LVPWd {rep.lvpwd*10:.0f} LA {rep.la_diam*10:.0f}", 470, (255, 255, 255))
            put(img, view, 470 - 20 if view == "PLAX" else 470, (200, 200, 200))
            panels.append(img)
    cv2.imwrite(f"{a.out}/2d_calipers_contact_sheet_v2.png", cv2.cvtColor(np.concatenate(panels, 1), cv2.COLOR_RGB2BGR)); print("2D done")
    # ---- Doppler stills: Vmax models + VTI segmentation
    dpan, vpan, seen = [], [], set()
    for uid, sd in coh.study_dir.sample(80, random_state=a.seed).items():
        for f in sorted(glob.glob(sd + "/*.dcm")):
            if os.path.getsize(f) > 6_000_000: continue
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if int(ds.get("NumberOfFrames", 1) or 1) > 1: continue
            sreg = dio.spectral_region(dio.regions(ds))
            if sreg is None: continue
            fr, ds = dio.read_frames(f); img = fr[0]; o = ocr.read(img); lab = o["label"]
            dm = LABEL_TO_DOPPLER.get(lab); vt = LABEL_TO_VTI.get(lab)
            if dm and dm not in seen and len(dpan) < 6:
                strip = dio.mask_ecg_trace(img)[sreg.y0:, :, :]; x, y, p = dop.predict(dm, strip); v = abs(sreg.dy) * (y - (sreg.ref_y0 or 0)) / 100
                ov = img.copy(); cv2.circle(ov, (x, y + sreg.y0), 10, (255, 0, 0), 2); cv2.line(ov, (sreg.x0, sreg.y0 + (sreg.ref_y0 or 0)), (sreg.x1, sreg.y0 + (sreg.ref_y0 or 0)), (0, 255, 0), 1)
                unit = "cm/s" if dm in ("latevel", "medevel") else "m/s"; vv = v * 100 if unit == "cm/s" else v
                put(ov, f"{dm}: EchoNet {vv:.2f} {unit}   GE label: {o['value']} {o['unit']}", 700, (255, 255, 0)); dpan.append(cv2.resize(ov, (508, 354))); seen.add(dm)
            if vt in vti.seg and len(vpan) < 3 and vt not in seen:
                strip_bgr = cv2.cvtColor(img[sreg.y0:, :, :], cv2.COLOR_RGB2BGR); H = img.shape[0]
                br = (sreg.ref_y0 / (H - sreg.y0)) if sreg.ref_y0 and sreg.ref_y0 > 0 else 0.5
                try:
                    r = vti.predict(vt, strip_bgr, baseline_ratio=br); mk = r["best_mask"].astype(bool)
                    if mk.shape != strip_bgr.shape[:2]: print("VTI mask shape", mk.shape, "strip", strip_bgr.shape)
                    col_h = mk.sum(0); vti_cm = col_h.sum() * abs(sreg.dy) * abs(sreg.dx)
                    ov = img.copy(); sub = ov[sreg.y0:sreg.y0 + mk.shape[0], :mk.shape[1]]; sub[mk] = (0.5 * sub[mk] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                    put(ov, f"VTI {vt}: {r['n_mountains']} beats found, {r['n_passed']} passed, VTI {vti_cm:.1f} cm  (label {o['text'][:40]})", 700); vpan.append(cv2.resize(ov, (508, 354))); seen.add(vt)
                except Exception as e:
                    print("vti err", e)
        if len(dpan) >= 6 and len(vpan) >= 3: break
    if dpan: cv2.imwrite(f"{a.out}/doppler_vmax_contact_sheet_v2.png", cv2.cvtColor(np.concatenate([np.concatenate(dpan[:3], 1), np.concatenate((dpan[3:6] + dpan[:3])[:3], 1)], 0), cv2.COLOR_RGB2BGR))
    if vpan: cv2.imwrite(f"{a.out}/vti_contact_sheet_v2.png", cv2.cvtColor(np.concatenate(vpan, 1), cv2.COLOR_RGB2BGR))
    print("doppler done", len(dpan), len(vpan))
    # ---- LA area + EchoNet-Dynamic on A4C
    apan, epan = [], []
    for _, r in rows[(rows.predicted_class == "A4C") & (rows.avi_status == "success")].sample(3, random_state=a.seed).iterrows():
        fr, ds = dio.read_frames(r.dcm_path); treg = dio.tissue_region(dio.regions(ds)); cv = dio.to_canvas_3x4(fr[:: max(1, len(fr) // 32)], treg)
        masks, confs = area.predict("LA_AREA", cv.frames); px = masks.reshape(len(masks), -1).sum(1); i = int(px.argmax())
        ov = cv.frames[i].copy(); m = masks[i].astype(bool); ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
        put(ov, f"LA area max {px.max() * cv.cm_per_model_px ** 2:.1f} cm2", 470); apan.append(ov)
        f112 = dyn.to_112(fr, treg); fps = 1000.0 / float(ds.get("FrameTime", 33.3)); per = max(1, int(round(fps / 25)))
        res = dyn.predict(f112, period=per)
        xs = torch.from_numpy((f112.astype(np.float32) - dyn.MEAN) / dyn.STD).permute(0, 3, 1, 2)
        with torch.no_grad(): mk = (dyn.seg(xs[:64].to(dev))["out"][:, 0] > 0).cpu().numpy()
        ar = mk.reshape(len(mk), -1).sum(1); pan = []
        for j in (int(ar.argmax()), int(ar.argmin())):
            g = f112[j].copy(); g[mk[j]] = (0.5 * g[mk[j]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8); pan.append(cv2.resize(g, (224, 224), interpolation=cv2.INTER_NEAREST))
        e = np.concatenate(pan, 1); rep = coh.loc[r.StudyInstanceUID].ef_visual if r.StudyInstanceUID in coh.index else np.nan
        put(e, f"EF {res['ef_mean']:.0f}% (visual EF {rep:.0f}%) fps {fps:.0f} period {per}", 215, (255, 255, 0)); epan.append(e)
    cv2.imwrite(f"{a.out}/la_area_contact_sheet_v2.png", cv2.cvtColor(np.concatenate(apan, 1), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{a.out}/echonet_dynamic_lvseg_ED_ES_v2.png", cv2.cvtColor(np.concatenate(epan, 0), cv2.COLOR_RGB2BGR)); print("A4C done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default="/volume/echonet_eval/cohort_rows.parquet"); ap.add_argument("--cohort", default="/volume/echonet_eval/cohort_10k.parquet")
    ap.add_argument("--out", default="/media/data1/models/EchoNet/inference_examples"); ap.add_argument("--seed", type=int, default=7); main(ap.parse_args())
