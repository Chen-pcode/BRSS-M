from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def model_stats(model: torch.nn.Module) -> dict[str, float]:
    params = sum(parameter.numel() for parameter in model.parameters())
    model_size_mb = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024**2)
    return {"params": params, "params_m": params / 1e6, "size_mb": model_size_mb, "model_size_mb": model_size_mb}


def estimate_flops(model: torch.nn.Module, image_size: int, device: torch.device) -> float:
    """Estimate one-image inference FLOPs, including fused Mamba selective scans.

    Generic FLOP profilers omit Mamba's fused CUDA operator. This hook-based
    estimator counts Conv2d/Linear operations from runtime tensor shapes and
    uses the standard selective-scan operation count for official Mamba blocks.
    """

    total = 0
    handles = []

    def conv2d_flops(module: torch.nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        batch, channels, height, width = output.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total += 2 * batch * channels * height * width * kernel_ops

    def linear_flops(module: torch.nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        total += 2 * output.numel() * module.in_features

    def mamba_flops(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        batch, length, d_model = inputs[0].shape
        d_inner = getattr(module, "d_inner", d_model * getattr(module, "expand", 2))
        d_state = getattr(module, "d_state", 16)
        d_conv = getattr(module, "d_conv", 4)
        dt_rank = getattr(module, "dt_rank", max(1, (d_model + 15) // 16))
        # Input/output projections, depthwise causal convolution, parameter
        # projection, dt projection, and selective state update respectively.
        total += 2 * batch * length * d_model * (2 * d_inner)
        total += 2 * batch * length * d_inner * d_conv
        total += 2 * batch * length * d_inner * (dt_rank + 2 * d_state)
        total += 2 * batch * length * dt_rank * d_inner
        total += 9 * batch * length * d_inner * d_state + 2 * batch * length * d_inner
        total += 2 * batch * length * d_inner * d_model

    def attach(module: torch.nn.Module) -> None:
        if module.__class__.__name__ == "Mamba":
            handles.append(module.register_forward_hook(mamba_flops))
            return
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv2d_flops))
            return
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_flops))
            return
        for child in module.children():
            attach(child)

    attach(model)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total / 1e9
