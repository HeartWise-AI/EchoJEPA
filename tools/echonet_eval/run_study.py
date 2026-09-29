"""Per-study EchoNet inference on MHI DICOMs + a multi-GPU batch runner.

For each study (StudyInstanceUID folder under /media/data1/datasets/DeepEcho/DICOM/<year>/<pid>/<uid>/):
  * 2D calipers (Cedars Measurement): PLAX/PLAX_ZOOM clips -> ivs, lvid, lvpw, aorta, aortic_root, la;
    A4C clips -> rv_base; SUBCOSTAL -> ivc. Frames go through the 3:4 sector canvas (dicom_io.to_canvas_3x4).
    Per clip we keep the smoothed diameter series and its min/median/max (diastole = max for LVID/LA/RV,
    min for wall thicknesses; the analysis picks).
  * LA area (EchoNet Segmentation LA_AREA): A4C and A2C clips -> per-frame area (cm2), max over frames.
  * EchoNet-Dynamic: A4C clips -> EF per clip (+ LV mask stats).
  * Spectral Doppler stills (RegionDataType 3/4): OCR the GE label -> route to the matching Cedars Doppler
    Vmax model (peak velocity in m/s from the heat-map argmax and the region calibration) and, for
    LVOT/AV/MV stills, the VTI segmentation (VTI in cm = sum over columns of mask height x |dy| x dx).
  * Structured report (GE Comprehensive SR) -> sonographer measurements (sr_parse.py).

Output: one parquet of "rows" per shard (long format: study, kind, model, clip, metric, value, ...).
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time, traceback
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dicom_io as dio  # noqa: E402
import sr_parse  # noqa: E402

VIEW_PLAN = {  # predicted_class -> (max clips, list of 2D models), plus flags
    "PLAX": (3, ["ivs", "lvid", "lvpw", "aorta", "aortic_root", "la"]),
    "PLAX_ZOOM": (2, ["ivs", "lvid", "lvpw", "aorta", "aortic_root", "la"]),
    "A4C": (4, ["rv_base"]),
    "A2C": (2, []),
    "SUBCOSTAL": (1, ["ivc"]),
}
MAX_FRAMES = 64


def smooth(sig: np.ndarray, fps: float = 30.0, bpm: float = 140.0) -> np.ndarray:
    """Cedars low-pass filter: keep frequencies up to the max heart-rate."""
    n = len(sig)
    if n < 8:
        return sig
    cutoff = int(np.ceil(bpm / 60 / fps * n))
    fft = np.fft.fft(sig)
    fft[cutoff + 1:-cutoff] = 0
    return np.real(np.fft.ifft(fft))


class StudyRunner:
    def __init__(self, device: torch.device, ocr: bool = True):
        from models import Calipers2D, DopplerVmax, AreaSeg, VTISeg, EchoNetDynamic
        self.device = device
        self.cal = Calipers2D(device)
        self.dop = DopplerVmax(device)
        self.area = AreaSeg(device, targets=("LA_AREA",))
        self.vti = VTISeg(device, targets=("LVOT", "MV", "AV"))
        self.dyn = EchoNetDynamic(device)
        self.ocr = dio.LabelOCR(gpu=device.type == "cuda") if ocr else None

    # ------------------------------------------------------------------ helpers
    def _row(self, base, **kw):
        r = dict(base); r.update(kw); return r

    def run_clip(self, base: dict, path: str, view: str, models_2d: list[str], out: list):
        frames, ds = dio.read_frames(path, max_frames=MAX_FRAMES)
        regs = dio.regions(ds)
        treg = dio.tissue_region(regs)
        has_color = any(r.dtype == 2 for r in regs)
        fps = float(ds.get("CineRate", 0) or ds.get("RecommendedDisplayFrameRate", 0) or 30)
        cv = dio.to_canvas_3x4(frames, treg)
        b = dict(base, file=os.path.basename(path), view=view, n_frames=int(len(frames)), color_doppler=bool(has_color),
                 cm_per_px=cv.cm_per_model_px, fps=fps)
        # 2D calipers
        if models_2d and cv.cm_per_model_px:
            for m in models_2d:
                coords = self.cal.predict(m, cv.frames)  # (T,2,2) (y,x)
                d_px = np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1)
                d_cm = d_px * cv.cm_per_model_px
                sm = smooth(d_cm, fps=fps)
                out.append(self._row(b, kind="caliper", model=m, d_min_mm=float(sm.min() * 10), d_med_mm=float(np.median(sm) * 10),
                                     d_max_mm=float(sm.max() * 10), d_raw_med_mm=float(np.median(d_cm) * 10),
                                     p1_y=float(coords[:, 0, 0].mean()), p1_x=float(coords[:, 0, 1].mean()),
                                     p2_y=float(coords[:, 1, 0].mean()), p2_x=float(coords[:, 1, 1].mean())))
        # LA area on apical clips
        if view in ("A4C", "A2C") and cv.cm_per_model_px:
            masks, confs = self.area.predict("LA_AREA", cv.frames)
            px = masks.reshape(len(masks), -1).sum(1).astype(float)
            area_cm2 = px * (cv.cm_per_model_px ** 2)
            sm = smooth(area_cm2, fps=fps) if len(area_cm2) >= 8 else area_cm2
            out.append(self._row(b, kind="la_area", model="LA_AREA", area_max_cm2=float(sm.max()), area_min_cm2=float(sm.min()),
                                 area_med_cm2=float(np.median(sm)), conf_mean=float(confs.mean()), frac_frames_with_mask=float((px > 200).mean())))
        # EchoNet-Dynamic on A4C
        if view == "A4C":
            r = self.dyn.predict(self.dyn.to_112(frames, treg))
            out.append(self._row(b, kind="dynamic", model="echonet_dynamic", ef_mean=r["ef_mean"], ef_clips=json.dumps(r["ef_clips"]),
                                 lv_area_max=r["lv_area_max"], lv_area_min=r["lv_area_min"], lv_frac_change=r["lv_frac_change"], lv_frac_frames=r["lv_frac_frames"]))

    def run_still(self, base: dict, path: str, out: list):
        ds = pydicom_read_header(path)
        if int(ds.get("NumberOfFrames", 1) or 1) > 1:
            return  # a clip whose AVI conversion failed, not a still
        regs = dio.regions(ds)
        sreg = dio.spectral_region(regs)
        if sreg is None:
            return
        frames, ds = dio.read_frames(path)
        img = frames[0]
        H, W = img.shape[:2]
        ocr = self.ocr.read(img) if self.ocr else {"label": None, "text": "", "value": None, "unit": None}
        b = dict(base, file=os.path.basename(path), view="DOPPLER", doppler_type=("CW" if sreg.dtype == 4 else "PW"),
                 ocr_label=ocr["label"], ocr_text=ocr["text"][:120], ocr_value=ocr["value"], ocr_unit=ocr["unit"],
                 strip_y0=sreg.y0, strip_y1=sreg.y1, dy_cms_per_px=abs(sreg.dy) if sreg.dy else None, dx_s_per_px=abs(sreg.dx) if sreg.dx else None,
                 ref_y0=sreg.ref_y0)
        from models import LABEL_TO_DOPPLER, LABEL_TO_VTI
        masked = dio.mask_ecg_trace(img)
        strip = masked[sreg.y0:, :, :]
        dmodel = LABEL_TO_DOPPLER.get(ocr["label"])
        if dmodel and sreg.dy:
            x, y, p = self.dop.predict(dmodel, strip)
            baseline = (sreg.ref_y0 or 0)
            v_cms = abs(sreg.dy) * (y - baseline)
            out.append(self._row(b, kind="doppler_vmax", model=dmodel, pred_x=x, pred_y_strip=y, peak_prob=p, vmax_ms=float(abs(v_cms) / 100.0),
                                 edge_hit=bool(x <= 2 or x >= strip.shape[1] - 3)))
        vt = LABEL_TO_VTI.get(ocr["label"])
        if vt in self.vti.seg and sreg.dy and sreg.dx:
            strip_bgr = cv2.cvtColor(img[sreg.y0:, :, :], cv2.COLOR_RGB2BGR)
            base_ratio = (sreg.ref_y0 / (H - sreg.y0)) if sreg.ref_y0 is not None and sreg.ref_y0 > 0 else 0.5
            try:
                r = self.vti.predict(vt, strip_bgr, baseline_ratio=base_ratio)
                mk = r["best_mask"].astype(bool)
                col_h = mk.sum(0)  # px of envelope per time column
                vti_cm = float(col_h.sum() * abs(sreg.dy) * abs(sreg.dx))  # (cm/s per px) * (s per px) * px^2 = cm
                vmax_from_mask = float(col_h.max() * abs(sreg.dy) / 100.0) if col_h.max() > 0 else None
                out.append(self._row(b, kind="vti", model=vt, n_mountains=r["n_mountains"], n_passed=r["n_passed"], vti_cm=vti_cm if r["n_passed"] else None,
                                     vmax_from_mask_ms=vmax_from_mask, mask_px=int(mk.sum())))
            except Exception as e:
                out.append(self._row(b, kind="vti", model=vt, error=str(e)[:120]))

    # ------------------------------------------------------------------ study
    def run_study(self, study: dict, rows_df: pd.DataFrame) -> list:
        base = {"StudyInstanceUID": study["StudyInstanceUID"], "AccessionNumber": study.get("AccessionNumber")}
        out = []
        sdir = study["study_dir"]
        # structured report
        try:
            srf = sr_parse.find_sr_files(sdir)
            if srf:
                import pydicom
                meas = sr_parse.canonical_measurements(sr_parse.parse_sr_dataset(pydicom.dcmread(srf[0])))
                out.append(self._row(base, kind="sr", model="ge_sr", file=os.path.basename(srf[0]), **{f"sr_{k}": v for k, v in meas.items()}))
        except Exception as e:
            out.append(self._row(base, kind="error", model="sr", error=str(e)[:200]))
        # video clips by view
        vids = rows_df[(rows_df.avi_status == "success") & rows_df.predicted_class.isin(VIEW_PLAN.keys())]
        for view, (nmax, models_2d) in VIEW_PLAN.items():
            sub = vids[vids.predicted_class == view]
            if len(sub) == 0:
                continue
            sub = sub.assign(nf=pd.to_numeric(sub.NumberOfFrames, errors="coerce")).sort_values("nf", ascending=False).head(nmax)
            for p in sub.dcm_path:
                try:
                    if os.path.exists(p):
                        self.run_clip(base, p, view, models_2d, out)
                except Exception as e:
                    out.append(self._row(base, kind="error", model="clip", view=view, file=os.path.basename(p), error=str(e)[:200]))
        # stills (non-video DICOMs of the study, from the folder listing)
        stills = [p for p in glob.glob(os.path.join(sdir, "*.dcm")) if os.path.getsize(p) < 6_000_000]
        video_paths = set(rows_df.loc[rows_df.avi_status == "success", "dcm_path"].dropna())
        for p in stills:
            if p in video_paths:
                continue
            try:
                self.run_still(base, p, out)
            except Exception as e:
                out.append(self._row(base, kind="error", model="still", file=os.path.basename(p), error=str(e)[:200]))
        return out


def pydicom_read_header(path):
    import pydicom
    return pydicom.dcmread(path, stop_before_pixels=True)


# ---------------------------------------------------------------------- batch runner
def worker(args):
    shard_id, n_shards, gpu, cohort_path, rows_path, out_dir, limit = args
    # Pin the worker to one physical GPU by visibility: EasyOCR wraps its nets in DataParallel with
    # device_ids=[0], so the process must see its GPU as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cohort = pd.read_parquet(cohort_path)
    cohort = cohort.iloc[shard_id::n_shards]
    if limit:
        cohort = cohort.head(limit)
    rows = pd.read_parquet(rows_path)
    rows = rows[rows.StudyInstanceUID.isin(set(cohort.StudyInstanceUID))]
    by_study = {k: v for k, v in rows.groupby("StudyInstanceUID")}
    out_path = os.path.join(out_dir, f"rows_shard{shard_id:02d}.parquet")
    done = set()
    if os.path.exists(out_path):
        done = set(pd.read_parquet(out_path, columns=["StudyInstanceUID"]).StudyInstanceUID)
    runner = StudyRunner(device)
    buf, t0, n = [], time.time(), 0
    log = open(os.path.join(out_dir, f"shard{shard_id:02d}.log"), "a")
    for _, st in cohort.iterrows():
        if st.StudyInstanceUID in done:
            continue
        try:
            rws = runner.run_study(st.to_dict(), by_study.get(st.StudyInstanceUID, rows.iloc[0:0]))
            if not rws:
                rws = [{"StudyInstanceUID": st.StudyInstanceUID, "kind": "empty", "model": None}]
            buf.extend(rws)
        except Exception as e:
            buf.append({"StudyInstanceUID": st.StudyInstanceUID, "kind": "error", "model": "study", "error": traceback.format_exc()[-300:]})
        n += 1
        if n % 10 == 0:
            df = pd.DataFrame(buf)
            if os.path.exists(out_path):
                df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
            df.to_parquet(out_path, index=False); buf = []
            log.write(f"{time.strftime('%H:%M:%S')} shard {shard_id} gpu {gpu}: {n} studies, {(time.time()-t0)/n:.1f} s/study\n"); log.flush()
    if buf:
        df = pd.DataFrame(buf)
        if os.path.exists(out_path):
            df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
        df.to_parquet(out_path, index=False)
    log.write(f"DONE shard {shard_id}: {n} studies in {(time.time()-t0)/60:.1f} min\n"); log.close()
    return shard_id, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="/volume/echonet_eval/cohort_10k.parquet")
    ap.add_argument("--rows", default="/volume/echonet_eval/cohort_rows.parquet")
    ap.add_argument("--out", default="/volume/echonet_eval/results")
    ap.add_argument("--gpus", default="0,1,2,3", help="comma list; repeat a GPU id to put more workers on it")
    ap.add_argument("--workers_per_gpu", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="studies per shard (pilot)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gpus = [int(g) for g in a.gpus.split(",")]
    n_shards = len(gpus) * a.workers_per_gpu
    jobs = [(i, n_shards, gpus[i % len(gpus)], a.cohort, a.rows, a.out, a.limit) for i in range(n_shards)]
    if n_shards == 1:
        print(worker(jobs[0]))
    else:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_shards) as pool:
            for r in pool.imap_unordered(worker, jobs):
                print("finished", r, flush=True)
