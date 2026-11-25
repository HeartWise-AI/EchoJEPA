# evals/view_classifier/bench.py

"""Inference throughput of both arms on synthetic input (random tensors: no clip is read).

- Arm A: the frozen encoder (as the token cache builds it: bf16 autocast) followed by the attentive probe.
- Arm B: the MViT-v2-S classifier, in fp32 (as the zero-shot scoring runs it) and with bf16 autocast.

Each is timed over `--iters` batches of `--batch` clips after `--warmup` batches, with CUDA synchronised around the
timed region; the result is clips per second of model compute only. Decoding is measured separately (the zero-shot
step's `decode_clips_per_s`, the same decoder for both arms); the report combines the two into the cost of the 6.7M
archive videos.

    python -m evals.view_classifier.bench --config CONFIG --device cuda:0 --out throughput.json
"""

import argparse
import json
import time

import torch
import yaml

from evals.view_classifier import classes as vc


@torch.no_grad()
def clips_per_second(forward, make_input, batch, iters=20, warmup=3, device=torch.device("cpu")):
    x = make_input(batch).to(device)
    for _ in range(warmup):
        forward(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(iters):
        forward(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return batch * iters / (time.time() - start)


def clip_input(frames=16, size=224):
    return lambda b: torch.rand(b, 3, frames, size, size)


def bench_arm_a(cfg, device, batch, iters, warmup):
    from evals.radar_frozen.build_token_cache import build_encoder, encode_batch
    from evals.view_classifier.probe import build_probe
    with open(cfg["arm_a"]["token_cache_config"]) as f:
        cache_cfg = yaml.safe_load(f)
    cache_cfg["encoder"]["checkpoint"] = cfg["paths"]["encoder"]
    encoder = build_encoder(cache_cfg, device)
    probe_cfg = cfg["arm_a"].get("probe", {})
    probe = build_probe(int(cfg["arm_a"].get("embed_dim", 768)), int(probe_cfg.get("num_heads", 16)),
                        int(probe_cfg.get("depth", 4)), len(vc.from_config(cfg)[0])).to(device).eval()
    frames = int(cache_cfg["data"].get("frames_per_clip", 16))
    indices = torch.arange(0, 2 * frames, 2).repeat(batch, 1)

    def encoder_only(x):
        return encode_batch(encoder, x, indices, device, True)

    def full(x):
        tokens = encode_batch(encoder, x, indices, device, True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return probe(tokens)

    make = clip_input(frames)
    return {"encoder_clips_per_s": clips_per_second(encoder_only, make, batch, iters, warmup, device),
            "encoder_plus_probe_clips_per_s": clips_per_second(full, make, batch, iters, warmup, device)}


def bench_arm_b(cfg, device, batch, iters, warmup):
    from evals.view_classifier.mvit import build_mvit
    model = build_mvit(len(vc.from_config(cfg)[0])).to(device).eval()

    def bf16(x):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return model(x)

    make = clip_input(16)
    return {"fp32_clips_per_s": clips_per_second(model, make, batch, iters, warmup, device),
            "bf16_clips_per_s": clips_per_second(bf16, make, batch, iters, warmup, device)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)
    record = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "batch": args.batch, "iters": args.iters,
              "arm_b": bench_arm_b(cfg, device, args.batch, args.iters, args.warmup)}
    if device.type == "cuda":
        torch.cuda.empty_cache()
    record["arm_a"] = bench_arm_a(cfg, device, args.batch, args.iters, args.warmup)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
