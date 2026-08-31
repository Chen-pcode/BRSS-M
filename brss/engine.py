from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .losses import total_loss
from .metrics import sample_metrics


def train_epoch(model, loader, optimizer, scaler, device: torch.device, amp: bool, use_boundary_loss: bool, use_multiscale_boundary_loss: bool) -> float:
    model.train()
    total, samples = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        image, mask, boundary = (batch[key].to(device, non_blocking=True) for key in ("image", "mask", "boundary"))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            loss = total_loss(model(image), mask, boundary, use_boundary_loss, use_multiscale_boundary_loss)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * image.shape[0]
        samples += image.shape[0]
    return total / max(samples, 1)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, threshold: float, prediction_dir: Path | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        outputs = model(batch["image"].to(device, non_blocking=True))
        logits = outputs["logits"]
        predicted = (logits.sigmoid() >= threshold).cpu().numpy().astype(np.uint8)
        target = batch["mask"].numpy().astype(np.uint8)
        boundary_target = batch["boundary"]
        boundary_predictions = [outputs["boundary"], *outputs.get("boundary_scales", [])]
        boundary_dice = []
        for boundary_prediction in boundary_predictions:
            target_scale = torch.nn.functional.interpolate(boundary_target, size=boundary_prediction.shape[-2:], mode="nearest")
            pred_scale = (boundary_prediction.sigmoid().cpu() >= threshold).float()
            intersection = (pred_scale * target_scale).sum((1, 2, 3))
            denominator = pred_scale.sum((1, 2, 3)) + target_scale.sum((1, 2, 3))
            boundary_dice.append(((2 * intersection + 1e-7) / (denominator + 1e-7)).numpy())
        for index, sample_id in enumerate(batch["id"]):
            row = {"id": sample_id, **sample_metrics(predicted[index, 0], target[index, 0])}
            row.update({f"boundary_scale_{scale}_dice": float(values[index]) for scale, values in enumerate(boundary_dice)})
            rows.append(row)
            if prediction_dir:
                Image.fromarray(predicted[index, 0] * 255).save(prediction_dir / f"{sample_id}.png")
    frame = pd.DataFrame(rows)
    return {column: float(frame[column].mean()) for column in frame.columns if column != "id"}, frame
