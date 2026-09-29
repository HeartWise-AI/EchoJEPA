"""Inference-only shim for cvair.training.model_wrappers.SegmentationModelWrapper.

The EchoNet Segmentation checkpoints were trained with a Lightning wrapper that stores the
DeepLabV3 backbone as `self.m` (state-dict keys `m.backbone.*`, `m.classifier.*`).
Segmentation/utils.py only needs: construction with (backbone, num_classes, lr,
tracked_metric), `load_state_dict`, `.eval().to(device)` and `seg_model.m(tensor)`.
"""
import torch.nn as nn


class SegmentationModelWrapper(nn.Module):
    def __init__(self, m: nn.Module, num_classes: int = 1, lr: float = 1e-3, tracked_metric: str = "loss", **kwargs):
        super().__init__()
        self.m = m
        self.num_classes = num_classes
        self.lr = lr
        self.tracked_metric = tracked_metric

    def forward(self, x):
        return self.m(x)
