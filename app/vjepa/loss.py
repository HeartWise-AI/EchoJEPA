# app/vjepa/loss.py

import torch

from src.masks.utils import apply_masks


def jepa_loss(z, h, masks_pred, loss_exp):
    """Compute the prediction loss and its detached per-sample aggregates. This
    separates the loss used to train the model from the loss used to report metrics.

    Args:
        z: predictor outputs, one list of (B_bucket, K, D) tensors per fpc bucket.
        h: target-encoder outputs, one (B_bucket, N, D) tensor per fpc bucket.
        masks_pred: prediction masks, one list of (B_bucket, K) tensors per bucket.
        loss_exp: exponent of the elementwise error.

    Returns:
        (loss, sample_loss_sum, num_samples)

        `loss`: unchanged value to backpropagate, the mean over every (bucket, mask) term. 
        `sample_loss_sum`: detached 0-dim tensor, the sum of the per-sample losses.
        `num_samples`: the number of clips it covers.
        `sample_loss_sum / num_samples`: the mean loss per sample.
    """
    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

    loss, n = 0, 0
    # Detached, metrics-only.
    bucket_sums = []
    num_samples = 0

    for zi, hi in zip(z, h):
        bucket_loss, bucket_n, bucket_size = None, 0, 0
        for zij, hij in zip(zi, hi):
            term = torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            loss += term
            n += 1
            detached = term.detach().float()
            bucket_loss = detached if bucket_loss is None else bucket_loss + detached
            bucket_n += 1
            bucket_size = hij.shape[0]
        if bucket_n > 0:
            bucket_sums.append((bucket_loss / bucket_n) * bucket_size)
            num_samples += bucket_size

    if n == 0:
        raise ValueError("jepa_loss received no prediction targets: nothing to compute a loss over.")

    loss /= n
    sample_loss_sum = torch.stack(bucket_sums).sum()
    return loss, sample_loss_sum, num_samples
