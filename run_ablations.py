from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = [
    "brss_mamba",
    "brss_4stage_matched",
    "brss_5stage",
    "brss_no_ssm",
    "brss_plain_scan",
    "brss_no_boundary_router",
    "brss_ssm_router_only",
    "brss_decoder_router_only",
    "brss_no_local_path",
    "brss_no_cross_scale",
    "brss_no_boundary_loss",
    "brss_final_boundary_only",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled BRSS-MambaSeg ablation suite.")
    parser.add_argument("--data-root", default="/kaggle/input/skin-lesion-data")
    parser.add_argument("--output-root", default="./outputs/ablations")
    parser.add_argument("--train-dataset", default="isic2018")
    parser.add_argument("--val-dataset", default="isic2018")
    parser.add_argument("--test-datasets", nargs="*", default=["isic2017", "PH2"])
    parser.add_argument("--models", nargs="*", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 1234, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-completed", action="store_true", help="Skip a run when its summary.csv already exists.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = Path(args.output_root)
    for seed in args.seeds:
        for model in args.models:
            actual_model = "brss_mamba" if model in {"brss_no_boundary_loss", "brss_final_boundary_only"} else model
            output_dir = root / f"{model}_seed{seed}"
            if args.skip_completed and (output_dir / "summary.csv").exists():
                print(f"Skipping completed run: {output_dir}")
                continue
            command = [sys.executable, "train.py", "--data-root", args.data_root, "--train-dataset", args.train_dataset, "--val-dataset", args.val_dataset, "--test-datasets", *args.test_datasets, "--model", actual_model, "--experiment-name", model, "--output-dir", str(output_dir), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--image-size", str(args.image_size), "--workers", str(args.workers), "--seed", str(seed)]
            if model == "brss_no_boundary_loss":
                command.append("--no-boundary-loss")
            if model == "brss_final_boundary_only":
                command.append("--no-multiscale-boundary-loss")
            if args.amp:
                command.append("--amp")
            if args.deterministic:
                command.append("--deterministic")
            subprocess.run(command, check=True)
    subprocess.run([sys.executable, "summarize.py", "--root", str(root)], check=True)


if __name__ == "__main__":
    main()
