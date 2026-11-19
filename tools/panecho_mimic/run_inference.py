# tools/panecho_mimic/run_inference.py

"""PanEcho (Holste et al., JAMA 2025) inference on MIMIC-IV-ECHO, one row per video, sharded and resumable.

PanEcho's published test protocol (`modelcustom/PanEcho/src/{dataset,utils}.py`): every video of a study, any view
(Doppler included), 4 random 16-frame clips of consecutive native-rate frames (last frame repeated when the video is
shorter), frames read with OpenCV (BGR channel order, as in training), centre crop 224, scaled to [0, 1] and ImageNet-
normalized; the head outputs (probabilities after PanEcho's own activations, raw values for regression) are averaged
over the 4 clips and then over the study's videos (`score.py`). Videos are the `mp4_native256` files (short side 256,
aspect preserved), so the centre crop is the same 224 x 224 field of view the EchoJEPA probes get.

Each worker process takes the shards `shard % world == rank` and skips shards whose parquet already exists, so a
killed run resumes by relaunching the same command. Clip start frames come from a per-video seed (reproducible).

Example (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python tools/panecho_mimic/run_inference.py --rank 0 --world 3 --out-dir <dir>
"""

import argparse
import os  # noqa: I001 — the environment is set before cv2 is imported.
import sys
import time
import zlib

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")   # one decode thread per worker: FFmpeg's
import cv2                                                            # auto threads x 48 workers thrash the CPUs.
import numpy as np
import pandas as pd
import torch

MIMIC = "/media/data1/datasets/external_echo/MIMIC-IV-ECHO"
WEIGHTS = "/media/data1/models/PanEcho/content"
PANECHO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evals", "video_classification_frozen",
                           "modelcustom", "PanEcho")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_LEN, NUM_CLIPS, SIZE = 16, 4, 224


def load_panecho(device):
    """PanEcho as built by its hubconf (frame transformer, ConvNeXt-T, 4 layers, 8 heads, mean pooling, all 39 tasks,
    activations on), from the local weights instead of the GitHub URLs."""
    sys.path.insert(0, os.path.abspath(PANECHO_SRC))
    from src.models import FrameTransformer, MultiTaskModel

    class Task:
        def __init__(self, name, spec):
            self.task_name, self.task_type = name, spec["task_type"]
            self.class_names = np.asarray(spec["class_names"])
            self.class_indices = np.arange(self.class_names.size)
            self.mean = spec["mean"]

    task_dict = pd.read_pickle(os.path.join(WEIGHTS, "tasks.pkl"))
    tasks = [Task(name, spec) for name, spec in task_dict.items()]
    encoder = FrameTransformer("convnext_tiny", 8, 4, 0.0, "mean", CLIP_LEN)
    model = MultiTaskModel(encoder, encoder.encoder.n_features, tasks, 0.25, True)
    weights = torch.load(os.path.join(WEIGHTS, "panecho.pt"), map_location="cpu", weights_only=False)["weights"]
    del weights["encoder.time_encoder.pe"]
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if k != "encoder.time_encoder.pe"]
    if missing or unexpected:
        raise RuntimeError(f"PanEcho weights do not match: missing={missing} unexpected={unexpected}")
    return model.eval().to(device), tasks


def output_columns(tasks):
    """One column per regression / binary task (binary = probability of PanEcho's positive class) and one per class
    of a multi-class task (`<task>__<class>`)."""
    cols = []
    for t in tasks:
        if t.task_type == "multi-class_classification":
            cols += [f"{t.task_name}__{c}" for c in t.class_names]
        else:
            cols.append(t.task_name)
    return cols


class Videos(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            cap = cv2.VideoCapture(os.path.join(MIMIC, path))
            frames = []
            while True:
                ok, frame = cap.read()        # BGR, as PanEcho was trained.
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            if not frames:
                raise ValueError("no frames decoded")
            v = np.stack(frames)              # f, h, w, 3
            h, w = v.shape[1:3]
            if min(h, w) != 256:              # mp4_native256 is short side 256; resize anything else the same way.
                s = 256 / min(h, w)
                v = np.stack([cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) for f in v])
                h, w = v.shape[1:3]
            y0, x0 = (h - SIZE) // 2, (w - SIZE) // 2
            v = v[:, y0:y0 + SIZE, x0:x0 + SIZE]
            n = v.shape[0]
            rng = np.random.default_rng(zlib.crc32(path.encode()))
            clips = []
            for _ in range(NUM_CLIPS):
                start = 0 if n < CLIP_LEN else int(rng.integers(0, n - CLIP_LEN + 1))
                idx = np.minimum(np.arange(start, start + CLIP_LEN), n - 1)   # last frame carried forward.
                clips.append(v[idx])
            x = torch.from_numpy(np.ascontiguousarray(np.stack(clips)))       # uint8 4, 16, 224, 224, 3 (BGR)
            return x, i, n, ""
        except Exception as e:  # noqa: BLE001 — a bad file is recorded, not fatal.
            return torch.zeros(NUM_CLIPS, CLIP_LEN, SIZE, SIZE, 3, dtype=torch.uint8), i, 0, f"{type(e).__name__}: {e}"


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def collate(batch):
    return (torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch])


@torch.no_grad()
def run_shard(model, tasks, cols, rows, device, batch_size, workers, amp):
    loader = torch.utils.data.DataLoader(Videos(rows["mp4_native256"]), batch_size=batch_size, num_workers=workers,
                                         collate_fn=collate, pin_memory=True, worker_init_fn=worker_init, prefetch_factor=4 if workers else None)
    out = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
    n_frames = np.zeros(len(rows), dtype=np.int32)
    errors = [""] * len(rows)
    mean, std = torch.tensor(MEAN, device=device), torch.tensor(STD, device=device)
    for x, idx, n, err in loader:
        b = x.shape[0]
        x = x.to(device, non_blocking=True).flatten(0, 1).float().div_(255.0)   # b*4, 16, 224, 224, 3
        x = ((x - mean) / std).permute(0, 4, 1, 2, 3).contiguous()              # b*4, 3, 16, 224, 224, ImageNet-normalized
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = model(x)
        parts = []
        for t in tasks:
            p = pred[t.task_name].float().reshape(b, NUM_CLIPS, -1).mean(1)  # average the 4 clips.
            parts.append(p)
        res = torch.cat(parts, 1).cpu().numpy()
        for j, i in enumerate(idx):
            n_frames[i], errors[i] = n[j], err[j]
            if not err[j]:
                out[i] = res[j]
    df = pd.concat([rows.reset_index(drop=True), pd.DataFrame(out, columns=cols)], axis=1)
    df["decoded_frames"], df["error"] = n_frames, errors
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--studies-per-shard", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8, help="videos per batch (x4 clips)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit-videos", type=int, default=0, help="throughput test: first N videos only, no shards")
    ap.add_argument("--study-subsample", type=int, default=0, help="random subset of N studies (seed 0)")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model, tasks = load_panecho(device)
    cols = output_columns(tasks)
    keep = ["subject_id", "study_id", "mp4_native256", "view_hw20", "n_frames", "fps", "status"]
    videos = pd.read_parquet(os.path.join(MIMIC, "metadata", "videos.parquet"), columns=keep)
    videos = videos.sort_values(["study_id", "mp4_native256"]).reset_index(drop=True)
    shard_dir = os.path.join(args.out_dir, "video_shards")
    os.makedirs(shard_dir, exist_ok=True)

    if args.limit_videos:
        rows = videos.iloc[:args.limit_videos]
        t0 = time.time()
        df = run_shard(model, tasks, cols, rows, device, args.batch_size, args.workers, not args.no_amp)
        dt = time.time() - t0
        tag = "fp32" if args.no_amp else "amp"
        df.to_parquet(os.path.join(args.out_dir, f"throughput_test_{tag}.parquet"), index=False)
        print(f"THROUGHPUT {len(rows)} videos in {dt:.1f}s = {len(rows) / dt:.2f} videos/s/GPU "
              f"(errors: {(df.error != '').sum()})", flush=True)
        return

    studies = np.sort(videos.study_id.unique())
    if args.study_subsample:
        studies = np.sort(np.random.default_rng(0).choice(studies, args.study_subsample, replace=False))
        videos = videos[videos.study_id.isin(studies)]
    shards = [studies[i:i + args.studies_per_shard] for i in range(0, len(studies), args.studies_per_shard)]
    mine = [s for s in range(len(shards)) if s % args.world == args.rank]
    print(f"rank {args.rank}/{args.world}: {len(mine)} of {len(shards)} shards, {len(videos)} videos total", flush=True)
    t_start, done = time.time(), 0
    for k, s in enumerate(mine):
        path = os.path.join(shard_dir, f"shard_{s:05d}.parquet")
        if os.path.exists(path):
            continue
        rows = videos[videos.study_id.isin(shards[s])]
        t0 = time.time()
        df = run_shard(model, tasks, cols, rows, device, args.batch_size, args.workers, not args.no_amp)
        df.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        done += len(rows)
        rate = done / (time.time() - t_start)
        print(f"[{time.strftime('%H:%M:%S')}] rank {args.rank} shard {s} ({k + 1}/{len(mine)}): {len(rows)} videos "
              f"in {time.time() - t0:.0f}s, {rate:.1f} videos/s, errors {(df.error != '').sum()}", flush=True)
    print(f"rank {args.rank} DONE", flush=True)


if __name__ == "__main__":
    main()
