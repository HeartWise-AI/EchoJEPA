# app/vjepa_2_1/benchmark_loader.py

"""Measure how fast the pretraining data pipeline delivers clips, without a model.

Builds the same loader as app/vjepa_2_1/train.py (VideoDataset sampling, the training transforms and
the mask collator) for one simulated rank and times batches for a fixed wall-clock budget or batch
count, once per worker count. It reports clips/s, batch-time percentiles, failed loads (the
dataset's retry warnings, which replace a failed video with a random one and therefore change
the sampled distribution), and process CPU/RSS. Paths are never printed.

Run it once per rank count you plan to use: four trainer ranks each run their own loader, so the
host must sustain four times one loader's rate.

Example:
    python -m app.vjepa_2_1.benchmark_loader --config <pretrain.yaml> --manifest <sample.csv> \\
        --workers 8 16 --batches 60 --out <results.json>
"""

import argparse
import json
import multiprocessing
import os
import time

import numpy as np
import psutil
import torch
import yaml

from app.vjepa_2_1.transforms import make_transforms
from src.datasets.data_manager import init_data
from src.datasets.video_dataset import VideoDataset
from src.masks.multiseq_multiblock3d import MaskCollator

NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Failed loads, counted in the loader workers. `VideoDataset.__getitem__` replaces a failed video
# with a random one and only warns inside the worker, so the parent cannot see it otherwise.
# Workers are forked (this script never switches to spawn), so they share this counter.
FAILED = multiprocessing.Value("i", 0)
_get_item_video = VideoDataset.get_item_video


def _counted_get_item_video(self, index):
    loaded = _get_item_video(self, index)
    if not loaded:
        with FAILED.get_lock():
            FAILED.value += 1
    return loaded


VideoDataset.get_item_video = _counted_get_item_video


def build_loader(cfg, manifest, num_workers, temporal_padding=None, seed=0):
    data, aug = cfg["data"], cfg["data_aug"]
    transform = make_transforms(
        random_horizontal_flip=aug.get("random_horizontal_flip", False),
        random_resize_aspect_ratio=aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=aug.get("random_resize_scale", [0.3, 1.0]),
        reprob=aug.get("reprob", 0.0),
        auto_augment=aug.get("auto_augment", False),
        motion_shift=aug.get("motion_shift", False),
        crop_size=data.get("crop_size", 224),
        normalize=NORMALIZE,
    )
    collator = MaskCollator(cfgs_mask=cfg["mask"], dataset_fpcs=data["dataset_fpcs"],
                            crop_size=data.get("crop_size", 224), patch_size=data["patch_size"],
                            tubelet_size=data["tubelet_size"])
    kwargs = {}
    policy = temporal_padding or data.get("temporal_padding")
    if policy is not None:
        kwargs["temporal_padding"] = policy
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader, sampler = init_data(
        data=data.get("dataset_type", "videodataset"), root_path=[manifest], batch_size=data["batch_size"],
        training=True, dataset_fpcs=data["dataset_fpcs"], fps=data.get("fps"), transform=transform,
        rank=0, world_size=1, collator=collator, num_workers=num_workers, pin_mem=False,
        persistent_workers=False, in_order=data.get("in_order", True), log_dir=None, **kwargs,
    )
    sampler.set_epoch(seed)
    return loader


def process_tree_usage(proc):
    """CPU seconds and resident memory of this process and its loader workers."""
    procs = [proc] + proc.children(recursive=True)
    cpu = rss = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            cpu += t.user + t.system
            rss += p.memory_info().rss
        except psutil.Error:
            pass
    return cpu, rss


def run(cfg, manifest, num_workers, batches, max_seconds, warmup, temporal_padding=None):
    proc = psutil.Process()
    with FAILED.get_lock():
        FAILED.value = 0
    start = time.time()
    loader = iter(build_loader(cfg, manifest, num_workers, temporal_padding))
    times, clips, padded, peak_rss = [], 0, [], 0.0
    cpu0, _ = process_tree_usage(proc)
    last = time.time()
    first_batch_s = None
    for i in range(batches + warmup):
        try:
            sample = next(loader)
        except StopIteration:
            break
        now = time.time()
        if first_batch_s is None:
            first_batch_s = now - start
        if i >= warmup:
            times.append(now - last)
            for fpc_sample in sample:
                clip_indices = fpc_sample[0][-1][0]
                clips += clip_indices.shape[0]
                padded.append(float((clip_indices < 0).float().mean()))
        last = now
        _, rss = process_tree_usage(proc)
        peak_rss = max(peak_rss, rss)
        if now - start > max_seconds:
            break
    cpu1, _ = process_tree_usage(proc)
    elapsed = sum(times)
    del loader
    t = np.array(times) if times else np.array([np.nan])
    return {
        "num_workers": num_workers,
        "batches_timed": len(times),
        "first_batch_s": first_batch_s,
        "clips_per_s": clips / elapsed if elapsed else None,
        "batch_s_median": float(np.median(t)), "batch_s_p90": float(np.percentile(t, 90)),
        "batch_s_p99": float(np.percentile(t, 99)), "batch_s_max": float(np.max(t)),
        "black_padding_frame_frac": float(np.mean(padded)) if padded else None,
        "failed_loads": FAILED.value,
        "cpu_cores_busy": (cpu1 - cpu0) / max(time.time() - start, 1e-9),
        "peak_rss_gb": peak_rss / 1e9,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="A pretraining config (its data/data_aug/mask blocks are used).")
    p.add_argument("--manifest", required=True, help="Loader manifest to read (`<path> 0` rows).")
    p.add_argument("--workers", type=int, nargs="+", default=[16], help="Worker counts to try, one run each.")
    p.add_argument("--batches", type=int, default=50, help="Timed batches per run (after warm-up).")
    p.add_argument("--warmup", type=int, default=5, help="Untimed batches first (worker start-up, first reads).")
    p.add_argument("--max-seconds", type=float, default=600, help="Stop a run after this long.")
    p.add_argument("--batch-size", type=int, help="Override data.batch_size (per rank).")
    p.add_argument("--temporal-padding", choices=["black", "loop"], help="Override data.temporal_padding.")
    p.add_argument("--out", help="Write the results as JSON here.")
    args = p.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    results = []
    for w in args.workers:
        r = run(cfg, args.manifest, w, args.batches, args.max_seconds, args.warmup, args.temporal_padding)
        r["batch_size"] = cfg["data"]["batch_size"]
        r["temporal_padding"] = args.temporal_padding or cfg["data"].get("temporal_padding", "black")
        results.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": os.path.basename(args.config), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
