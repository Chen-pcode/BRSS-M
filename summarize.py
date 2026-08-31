from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    frames = [pd.read_csv(path) for path in root.rglob("summary.csv")]
    if not frames:
        raise FileNotFoundError(f"No summary.csv files under {root}")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(root / "all_runs.csv", index=False)
    metric_columns = ["dice", "iou", "accuracy", "hd95", "params", "size_mb", "runtime_min"]
    summary = raw.groupby(["model", "train_dataset", "eval_dataset", "split"], as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = ["_".join(column).strip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary.to_csv(root / "ablation_mean_std.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
