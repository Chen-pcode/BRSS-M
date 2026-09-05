from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def sample_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p, t = prediction.astype(bool), target.astype(bool)
    tp, tn = np.logical_and(p, t).sum(), np.logical_and(~p, ~t).sum()
    fp, fn = np.logical_and(p, ~t).sum(), np.logical_and(~p, t).sum()
    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float((tp + tn + eps) / (tp + tn + fp + fn + eps)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "hd95": hd95(p, t),
    }


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(max(prediction.shape))
    p_border = np.logical_xor(prediction, binary_erosion(prediction))
    t_border = np.logical_xor(target, binary_erosion(target))
    distances = np.concatenate([distance_transform_edt(~t_border)[p_border], distance_transform_edt(~p_border)[t_border]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0
