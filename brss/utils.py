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
    return {"params": params, "size_mb": params * 4 / (1024**2)}
