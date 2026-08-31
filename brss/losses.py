from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, boundary: torch.Tensor, boundary_weight: float = 2.0) -> torch.Tensor:
    weights = 1 + boundary_weight * boundary
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
    return bce + dice_loss(logits, target)


def total_loss(outputs: dict[str, torch.Tensor | list[torch.Tensor]], target: torch.Tensor, boundary: torch.Tensor, use_boundary_loss: bool = True, use_multiscale_boundary_loss: bool = True) -> torch.Tensor:
    loss = segmentation_loss(outputs["logits"], target, boundary)  # type: ignore[arg-type]
    for weight, auxiliary in zip((0.25, 0.15), outputs.get("aux", [])):  # type: ignore[arg-type]
        loss = loss + weight * segmentation_loss(auxiliary, target, boundary)  # type: ignore[arg-type]
    if use_boundary_loss and "boundary" in outputs:
        loss = loss + 0.4 * F.binary_cross_entropy_with_logits(outputs["boundary"], boundary)  # type: ignore[arg-type]
    if use_boundary_loss and use_multiscale_boundary_loss:
        for boundary_scale in outputs.get("boundary_scales", []):  # type: ignore[arg-type]
            target_scale = F.interpolate(boundary, size=boundary_scale.shape[-2:], mode="nearest")
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(boundary_scale, target_scale)
    return loss
