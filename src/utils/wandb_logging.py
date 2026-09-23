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

# Log this step metric instead of wandb's implicit step.
# wandb expects the explicit steps to keep increasing, while we only support
# epoch-boundary resume.
STEP_METRIC = "train/global_step"

def init_wandb(
    cfgs_meta,
    args,
    rank,
    folder,
    resuming_training=False,
    checkpoint_run_id=None,
    checkpoint_has_wandb_run_id=False,
):
    """Initialize a wandb experiment run on rank 0 only.

    Args:
        cfgs_meta: the `meta` block of the training config.
        args: the full config, recorded as the run's config.
        rank: distributed rank; only rank 0 opens a run.
        folder: the run's checkpoint folder, also used as the wandb dir.
        resuming_training: True only when this job is continuing an existing training run.
        checkpoint_run_id: run id recovered from the resumed checkpoint, so the
            tracking state resumes together with the model state. Falls back to
            the `wandb_run_id.txt` for checkpoints written before that field existed.
        checkpoint_has_wandb_run_id: whether the resumed checkpoint contains the
            run-id field. An explicit None must not fall back to a stale text file.

    Returns (run, effective_run_id).
    `run`: actual live wandb object. None when wandb is not configured/available, 
        or when a resume could not reattach to its previous run.
    `effective_run_id`: the id of the wandb run that this training state logically belongs to.
    """
    # Validate on every rank so a configuration error stops the whole DDP job
    # consistently rather than leaving nonzero ranks waiting for rank 0.
    project = cfgs_meta.get("wandb_project", None)
    if not project:
        return None, None
    entity = cfgs_meta.get("wandb_entity", None)
    if not entity:
        raise ValueError("meta.wandb_entity is required whenever meta.wandb_project is set.")

    # Log only when rank=0 to avoid duplicated wand runs.
    if rank != 0:
        return None, None

    id_path = os.path.join(folder, "wandb_run_id.txt")
    run_id = None
    # Only resume if the model actually loaded a checkpoint.
    if resuming_training:
        # Only legacy checkpoints without the metadata field may fall back to the
        # sidecar text file.
        run_id = checkpoint_run_id or None
        if not checkpoint_has_wandb_run_id and run_id is None and os.path.exists(id_path):
            try:
                with open(id_path) as f:
                    run_id = f.read().strip() or None
            # If the file cannot be read, just ignore.
            except OSError:
                run_id = None
        if run_id is None:
            logger.warning("Resuming training but no previous wandb run id was found: starting a new run.")
    elif os.path.exists(id_path):
        logger.info(
            f"Ignoring the wandb run id in {id_path}: this job is not resuming a checkpoint, "
            f"so it starts a fresh wandb run."
        )

    if wandb is None:
        message = "meta.wandb_project is set but wandb is not installed."
        if run_id is not None:
            logger.error(f"Cannot resume wandb run {run_id}: {message} Continuing without wandb logging.")
        else:
            logger.warning(f"Skipping wandb logging: {message}.")
        # Hand `run_id` back even though nothing opened: on a legacy resume it was
        # recovered from the sidecar and exists nowhere else.
        return None, run_id

    try:
        # Start or reconnect to a wandb experiment run.
        run = wandb.init(
            project=project,
            entity=entity,
            name=cfgs_meta.get("wandb_run_name", None),
            id=run_id,
            # "must": a resume that cannot reattach fails here instead of quietly
            # forking a new run. The failure is then downgraded to disabled logging.
            # "never": keeps fresh jobs fresh.
            resume="must" if run_id else "never",
            dir=folder,
            config=args,
        )
    except Exception as e:
        if run_id is not None:
            # `resume="must"` has already ruled out silently forking a new run, so
            # the only thing lost here is logging. Training is worth more than its
            # metrics, so report it loudly and carry on unlogged.
            logger.error(
                f"Failed to resume wandb run {run_id} ({e}): continuing without wandb logging. "
                f"The run id is returned so the caller keeps it in the checkpoint and a "
                f"later resume can reattach to it."
            )
        else:
            logger.warning(f"wandb.init failed ({e}): continuing without wandb logging.")
        return None, run_id

    # Check if wandb actually put the run in the correct location.
    if (run.entity, run.project) != (entity, project):
        logger.warning(
            f"wandb run landed in {run.entity}/{run.project} but the config asked for "
            f"{entity}/{project}."
        )

    # Chart everything against an explicit step metric instead of wandb's implicit one.
    try:
        run.define_metric(STEP_METRIC)
        run.define_metric("*", step_metric=STEP_METRIC)
    except Exception as e:
        logger.warning(f"Failed to set the wandb step metric ({e}): charts will use wandb's implicit step.")

    # Write the current wandb id to disk, so if the job resumes later, it knows which wandb run to reconnect to.
    try:
        os.makedirs(folder, exist_ok=True)
        with open(id_path, "w") as f:
            f.write(run.id)
    except OSError:
        pass

    logger.info(f"wandb logging enabled: run {run.id} in {run.entity}/{run.project} ({run.url}).")
    return run, run.id


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
    # Round before the uint8 cast.
    out = out.round_().clamp_(0.0, 255.0).to(torch.uint8)
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
    within each clip across uploaded samples, which helps detect repeated-frame padding.

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
    # Get the average fraction across uploaded samples.
    return total / float(b)


def log_input_clips(run, clips, step, normalize, num_videos=2, fps=4, tag="train/input_clips"):
    """Actual logging function to wandb. `step` is recorded as the `STEP_METRIC` value 
    rather than passed to `run.log(step=...)`, so a resumed run never replays a step 
    wandb would drop.
    """
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
        sample = clips[:n]
        vids = denormalize_clips(sample, normalize).cpu().numpy()
        payload = {f"{tag}/{i}": wandb.Video(vids[i], fps=fps, format="gif") for i in range(n)}

        # Check the same clips that get uploaded in the batch for repeated frames.
        frac = unique_frame_fraction(sample)
        payload["data/unique_frame_frac"] = frac
        # Warning if fewer than 95% of frames are unique.
        if frac < 0.95:
            logger.warning(
                f"Only {frac:.0%} of frames in the {n} logged clip(s) are unique. "
                "The sampler may be padding clips with repeated frames because source videos "
                "are shorter than dataset_fpcs/fps implies."
            )
        payload[STEP_METRIC] = step
        run.log(payload)
    except Exception as e:
        logger.warning(f"Failed to log input clips to wandb: {e}")


def log_scalars(run, metrics, step):
    """Helper function for logging a dictionary of scalars. `step` is recorded as the 
    `STEP_METRIC` value rather than passed to `run.log(step=...)`, so a resumed run 
    never replays a step wandb would drop.
    """
    if run is None:
        return
    try:
        run.log({**metrics, STEP_METRIC: step})
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