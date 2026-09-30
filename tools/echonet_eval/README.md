# EchoNet evaluation toolkit (MHI)

Runs the public EchoNet models from Cedars-Sinai / Stanford on MHI GE Vivid DICOMs and compares them
with the sonographer measurements, at scale (10,000 exams) and with paper-style metrics.

| Model family | Repo | Weights on NAS |
|---|---|---|
| EchoNet-Measurement 2D calipers (IVS, LVID, LVPW, aorta, aortic root, LA, RV base, PA, IVC) and Doppler peak velocities (AV/TR/MR/LVOT Vmax, lateral/septal e') | https://github.com/echonet/measurements `Measurement/` | `/media/data1/models/EchoNet/measurements/weights/` |
| EchoNet-Segmentation LA/RA area, VTI (MV, AV, LVOT, RVOT, PV) | same repo, `Segmentation/` | `/media/data1/models/EchoNet/segmentation/weights/` |
| EchoNet-Dynamic EF regression + LV segmentation | https://github.com/echonet/dynamic | `/media/data1/models/EchoNet/dynamic/weights/` |

## What is in here

| File | Purpose |
|---|---|
| `dicom_io.py` | DICOM -> RGB frames; ultrasound-region tags; **`to_canvas_3x4()`**: the pre-processing that makes GE 708x1016 exports usable by the 2D caliper models (cut the 2D sector bounding box, letterbox it on a black 3:4 canvas with OpenCV, resize to 640x480, keep a single scale so `distance_cm = px * canvas_w/640 * PhysicalDeltaX`); Doppler strip extraction with the Cedars ECG-trace masking; `LabelOCR` (EasyOCR on the GE label box -> `TR_VMAX`, `AV_VMAX`, `LVOT_VMAX`, `LAT_E_PRIME`, ... + the on-screen value). |
| `models.py` | Loaders / thin wrappers: `Calipers2D`, `DopplerVmax`, `AreaSeg`, `VTISeg`, `EchoNetDynamic`. Imports the two upstream `utils.py` under unique module names; needs the `cvair` stand-in (`cvair/`) because `Segmentation/utils.py` imports a private Cedars package. |
| `sr_parse.py` | GE Comprehensive SR (DICOM structured report) parser -> sonographer measurements with their modifiers (Finding Site, Image Mode, Direction of Flow, ...) and a canonical dict (IVSd, LVIDd/s, LVPWd, LVOT diameter, LA diameter, ascending aorta, IVC, RV base, LA area A4C/A2C, LVEF, LVOT Vmax/VTI, TR Vmax, AV Vmax). |
| `run_study.py` | Per-study pipeline + multi-GPU batch runner (spawn pool, N workers per GPU, resumable per-shard parquet). Clips are chosen from the view predictions in `master_metadata.parquet` (PLAX/PLAX_ZOOM -> calipers, A4C -> RV base + LA area + EchoNet-Dynamic, A2C -> LA area, SUBCOSTAL -> IVC); spectral-Doppler stills (RegionDataType 3/4) are routed by the OCR label to the matching Vmax model and to the VTI segmentation. |
| `analyze.py` | Joins predictions with the report export, the SR and the on-screen values; Pearson r (bootstrap CI), r², coefficient of determination, MAE, bias, limits of agreement; scatter + Bland-Altman PNGs per measurement; EchoNet-Dynamic AUROC/AUPRC for EF < 50 % under 10 clip-quality gates. |
| `echonet_dynamic_infer.py` | Stand-alone EchoNet-Dynamic EF + LV segmentation on a CSV of DICOMs (sector crop, 112x112, ED/ES overlay). |
| `cvair/` | 20-line `cvair.training.model_wrappers.SegmentationModelWrapper` stand-in. |
| `report_md.py` | `analyze.py` outputs -> paper-style Markdown tables (Notion / PR). |
| `make_examples.py` | Contact sheets of every model on random cohort DICOMs (the `_v2` images on the NAS / Notion). |

## Setup (torch 2.14 NGC image)

```bash
pip install pytorch-lightning gdown python-bidi pyclipper shapely ninja
pip install --no-deps ultralytics easyocr          # plain installs would replace torch
git clone https://github.com/echonet/measurements /volume/echonet-measurements   # code only; weights are on the NAS
export ECHONET_NAS=/media/data1/models/EchoNet ECHONET_MEAS_REPO=/volume/echonet-measurements CVAIR_PARENT=$(pwd)   # this dir holds cvair/
```

## Run

```bash
# cohort: 10,000 2024 TTE studies with a report (built from the PACS index + df_report_2006_2024.csv), and their
# video rows from master_metadata.parquet (view predictions)
python run_study.py --cohort cohort_10k.parquet --rows cohort_rows.parquet --out results --gpus 0,1,2,3 --workers_per_gpu 3
python analyze.py --results results --out analysis
python report_md.py --analysis analysis --out analysis/report.md
```

`results/rows_shard*.parquet` is long format: one row per (study, clip or still, model) with the predicted
value(s), the OCR label/value for stills, and one `kind == "sr"` row per study with the SR measurements.
About 25 s per study per worker; 12 workers on 4 x 48 GB GPUs process 10,000 studies in ~6 h.

## Ground truth used

- `/media/data1/ravram/DeepECHO/TTE_reports_concat/df_report_2006_2024.csv` (per AccessionNumber; joined to StudyInstanceUID through `/media/data1/datasets/echo/database_tte_2009_2024.csv`).
- GE Comprehensive SR object inside ~75 % of study folders (`sr_parse.py`).
- The on-screen GE measurement label on each Doppler still (OCR), which gives both the trace type and the machine value.

## Pre-processing decisions that matter (all validated on MHI GE Vivid 708x1016 exports)

1. **Colour.** pydicom >= 3 already returns RGB from `pixel_array` for `YBR_FULL_422` and leaves the
   `PhotometricInterpretation` tag unchanged. Converting again (as the upstream `ybr_to_rgb` does) gives
   green/pink frames and silently breaks every model. `read_frames` converts only on pydicom < 3, and
   handles GE clips tagged 4:2:2 but stored 4:4:4.
2. **2D calipers: 3:4 sector canvas.** Crop the 2D sector bounding box (RegionDataType 1), letterbox it on a
   black 3:4 canvas (4 % margin), resize to 640x480; `cm = px * canvas_w/640 * PhysicalDeltaX`.
3. **Doppler baseline orientation.** The Cedars Vmax models expect the jet below the baseline. When GE puts
   the baseline in the lower 40 % of the strip (inverted display, common on AV CW) the strip is flipped
   vertically before inference and the peak row mapped back (AV errors 2.5 -> 0.3 m/s on those stills).
4. **EchoNet-Dynamic frame rate.** GE clips are 25-30 fps; the released model expects 50 fps with period 2.
   `period = round(fps / 25)` (= 1 here); the period-2 result is stored alongside (`ef_mean_p2`).
5. **Views.** Clips are routed by `predicted_class` (PLAX/PLAX_ZOOM/A4C/A2C/SUBCOSTAL); `SUBCOSTAL` is the
   subcostal 4-chamber view, not an IVC view, so the IVC caliper has no proper input yet.
6. **Workers.** EasyOCR wraps its nets in `DataParallel(device_ids=[0])`, so each worker sets
   `CUDA_VISIBLE_DEVICES` to its own GPU.

## Known limits

- No image-quality filter (the EchoNet-Measurements paper excludes low-quality images with a QC model).
- Report e' columns and the SR TR Vmax item did not agree with the on-screen values (r ~ 0 / 0.38) and are not used as references; report TR/AV Vmax contain entry errors and are filtered to 0-8 m/s.
- No IVC-view label in the view classifier (see above); RA_AREA did not fire on our A4C clips.
- Doppler stills without a GE label cannot be typed and are skipped.
- EchoNet-Dynamic is A4C-only; the study EF is aggregated over A4C clips gated by LV-mask quality.
