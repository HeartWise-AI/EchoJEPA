# src/models/radar/schedule.py

"""Optimizer-step arithmetic for RADAR training, from the study sampler (not from any one structure's captions).

Every optimizer step reads one global batch of studies; each structure contributes only the rows that have an
eligible clip and a caption, so the step count comes from the number of training studies.
"""

import math


def step_schedule(n_train_studies, global_batch, epochs, warmup_fraction=0.05, accumulation=1, drop_last=True):
    """Steps per epoch, total steps and warmup steps.

    `n_train_studies`: studies the sampler draws from in one epoch; `global_batch`: studies per optimizer step
    summed over ranks and accumulation micro-batches; with `drop_last` (DDP default) a partial last batch is dropped.
    Warmup ends before the last step, so the decay reaches its floor at the last step.
    """
    if n_train_studies < 1 or global_batch < 1 or epochs < 1 or accumulation < 1:
        raise ValueError("n_train_studies, global_batch, epochs and accumulation must be positive.")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1).")
    if global_batch % accumulation:
        raise ValueError("global_batch must be a multiple of accumulation.")
    steps = n_train_studies // global_batch if drop_last else math.ceil(n_train_studies / global_batch)
    if steps < 1:
        raise ValueError(f"{n_train_studies} studies do not fill one global batch of {global_batch}.")
    total = epochs * steps
    return {"steps_per_epoch": steps, "total_steps": total,
            "warmup_steps": min(round(warmup_fraction * total), total - 1)}
