# src/utils/wandb_logging.py

"""Weights & Biases logging helpers for EchoJEPA pretraining.

Help check the input pipeline bugs that can silently hurt training.

Note: failures from logging are warned but shouldn't crash the training.
"""

import os
from logging import getLogger
import torch

# This file can print warning/info.
logger = getLogger()

# Skip logging rather than crashing if wandb not installed.
try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(cfgs_meta, args, rank, folder):
    """Initialize a wandb experiment run on rank 0 only.

    Returns the run, or None when wandb is not configured/available.
    """
    # Log only when rank=0 to avoid duplicated wand runs.
    if rank != 0:
        return None

    project = cfgs_meta.get("wandb_project", None)
    if not project:
        return None
    if wandb is None:
        logger.warning("Skipping wandb Logging: meta.wandb_project is set but wandb is not installed.")
        return None

    # Reuse the run id when resuming the same wandb run.
    id_path = os.path.join(folder, "wandb_run_id.txt")
    run_id = None
    if os.path.exists(id_path):
        try:
            with open(id_path) as f:
                run_id = f.read().strip() or None
        # If the file cannot be read, just ignore.
        except OSError:
            run_id = None

    try:
        # Start or reconnect to a wandb experiment run.
        run = wandb.init(
            project=project,
            entity=cfgs_meta.get("wandb_entity", None),
            name=cfgs_meta.get("wandb_run_name", None),
            id=run_id,
            resume="allow",
            dir=folder,
            config=args,
        )
    except Exception as e:
        logger.warning(f"wandb.init failed ({e}): continuing without wandb logging.")
        return None

    # Write the current wandb id to disk, so if the job resumes later, it knows which wandb run to reconnect to.
    try:
        os.makedirs(folder, exist_ok=True)
        with open(id_path, "w") as f:
            f.write(run.id)
    except OSError:
        pass

    logger.info(f"wandb logging enabled: {run.url}")
    return run


def denormalize_clips(clips, normalize):
    """Undo the mean/std normalization applied by `VideoTransform`.

    Args:
        clips: (B, C, T, H, W) float tensor, exactly as fed to the encoder (normalized).
        normalize: ((mean_r, mean_g, mean_b), (std_r, std_g, std_b)) in [0, 1],
            matching what was passed to `make_transforms`.

    Returns:
        (B, T, C, H, W) uint8 tensor, the per-video layout `wandb.Video` wants.
    """
    # Mean and std are per-channel RGB statistics.
    mean, std = normalize
    mean = torch.as_tensor(
        mean,
        dtype=torch.float32,
        device=clips.device
    ).view(1, -1, 1, 1, 1) * 255.0
    std = torch.as_tensor(
        std,
        dtype=torch.float32,
        device=clips.device
    ).view(1, -1, 1, 1, 1) * 255.0

    # Denormalization: must never mutate the tensor being trained on.
    out = clips.detach().float() * std + mean
    out = out.clamp_(0.0, 255.0).to(torch.uint8)
    return out.permute(0, 2, 1, 3, 4)


def _select_clips(clips):
    """Select the first clip batch if passing in a list of per-frames-per-clip batches."""
    if isinstance(clips, (list, tuple)):
        if not clips:
            return None
        clips = clips[0]
    if not torch.is_tensor(clips) or clips.ndim != 5:
        return None
    return clips


def unique_frame_fraction(clips):
    """A diagnostic function that checks the average fraction of unique frames
    within each clip across the batch, which helps detect repeated-frame padding.

    Args:
        clips: (B, C, T, H, W) float tensor, exactly as fed to the encoder.

    Returns a float number between 0.0 and 1.0(=all frames are unique).
    """
    b, _c, t, _h, _w = clips.shape
    # Shape: (B, C, T, H, W) -> (B, T, C, H, W) -> (B, T, N) for N = C x H x W.
    flat = clips.detach().float().permute(0, 2, 1, 3, 4).reshape(b, t, -1)
    # Loop over each video in the batch to compute fraction of unique frames within each clip.
    total = 0.0
    for i in range(b):
        total += len(torch.unique(flat[i], dim=0)) / float(t)
    # Get the average fraction across the batch.
    return total / float(b)


def log_input_clips(run, clips, step, normalize, num_videos=2, fps=4, tag="train/input_clips"):
    """Actual logging function to wandb."""
    if run is None or wandb is None:
        return
    try:
        clips = _select_clips(clips)
        if clips is None:
            return
        # n: number of videos to log.
        n = min(int(num_videos), clips.shape[0])
        if n <= 0:
            return
        vids = denormalize_clips(clips[:n], normalize).cpu().numpy()
        payload = {f"{tag}/{i}": wandb.Video(vids[i], fps=fps, format="gif") for i in range(n)}

        # Checks all clips in the batch for repeated frames.
        frac = unique_frame_fraction(clips)
        payload["data/unique_frame_frac"] = frac
        # Warning if fewer than 95% of frames are unique.
        if frac < 0.95:
            logger.warning(
                f"Only {frac:.0%} of sampled frames are unique. The clip sampler is padding by "
                f"repeating frames. Source videos are likely shorter than dataset_fpcs/fps implies."
            )
        run.log(payload, step=step)
    except Exception as e:
        logger.warning(f"Failed to log input clips to wandb: {e}")


def log_scalars(run, metrics, step):
    """Helper function for logging a dictionary of scalars."""
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"Failed to log scalars to wandb: {e}")


def finish_wandb(run):
    """Close the wandb run cleanly."""
    if run is None:
        return
    try:
        run.finish()
    except Exception as e:
        logger.warning(f"Failed to finish wandb run: {e}")


def save_clips_as_gifs(clips, normalize, out_dir, prefix="clip", num_videos=2, fps=4):
    """Write clips to local GIFs instead of wandb. Almost same visualization work as wandb.

    Used by `python -m app.vjepa.inspect_inputs` so the input pipeline can be reviewed locally.
    """
    import imageio

    clips = _select_clips(clips)
    if clips is None:
        return []
    os.makedirs(out_dir, exist_ok=True)
    # n: number of videos to do the visualization.
    n = min(int(num_videos), clips.shape[0])
    vids = denormalize_clips(clips[:n], normalize).cpu().numpy()  # (n, T, C, H, W)
    paths = []
    for i in range(n):
        # Shape: (T, C, H, W) -> (T, H, W, C).
        frames = [f for f in vids[i].transpose(0, 2, 3, 1)]
        p = os.path.join(out_dir, f"{prefix}_{i}.gif")
        # loop=0: keep replaying the clip forever.
        imageio.mimsave(p, frames, fps=fps, loop=0)
        paths.append(p)
    return paths
