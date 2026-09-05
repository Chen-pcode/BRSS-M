from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brss.data import SkinDataset
from brss.engine import evaluate, train_epoch
from brss.models import ABLATIONS, get_model
from brss.utils import git_revision, model_stats, seed_everything, write_json


LOSS_ABLATIONS = {
    "brss_no_boundary_loss": {"no_boundary_loss": True},
    "brss_final_boundary_only": {"no_multiscale_boundary_loss": True},
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Boundary-Routed Selective State Space MambaSeg.")
    parser.add_argument("--data-root", default="./data", help="Fallback root for the original unified local data layout.")
    parser.add_argument("--isic2017-root", default="/kaggle/input/datasets/zichengdoctor/isic2017")
    parser.add_argument("--isic2018-root", default="/kaggle/input/datasets/zichengdoctor/isic2018")
    parser.add_argument("--ph2-root", default="/kaggle/input/datasets/zichengdoctor/ph2dataset")
    parser.add_argument("--train-dataset", default="isic2018")
    parser.add_argument("--val-dataset", default="isic2018")
    parser.add_argument("--test-datasets", nargs="*", default=["isic2017", "PH2"])
    parser.add_argument("--model", choices=sorted({*ABLATIONS, *LOSS_ABLATIONS}), default="brss_mamba")
    parser.add_argument("--experiment-name", default=None, help="Result label; defaults to --model.")
    parser.add_argument("--output-dir", default="./outputs/brss_mamba")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Compile fixed-shape SSM recurrences with torch.compile on Kaggle.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-boundary-loss", action="store_true")
    parser.add_argument("--no-multiscale-boundary-loss", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from output-dir/latest.pt after a stopped session.")
    return parser.parse_args()


def loader(dataset: SkinDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)


def log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    args = arguments()
    if args.model in LOSS_ABLATIONS:
        loss_ablation = args.model
        if args.experiment_name is None:
            args.experiment_name = loss_ablation
        for option, value in LOSS_ABLATIONS[loss_ablation].items():
            setattr(args, option, value)
        args.model = "brss_mamba"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_roots = {"isic2017": args.isic2017_root, "isic2018": args.isic2018_root, "ph2": args.ph2_root}
    train_set = SkinDataset(args.data_root, args.train_dataset, "train", args.image_size, augment=True, dataset_roots=dataset_roots)
    validation_set = SkinDataset(args.data_root, args.val_dataset, "val", args.image_size, dataset_roots=dataset_roots)
    train_loader, validation_loader = loader(train_set, args, True), loader(validation_set, args, False)
    model = get_model(args.model).to(device)
    if args.compile:
        if device.type != "cuda" or not hasattr(torch, "compile"):
            log("torch.compile is unavailable; using eager execution.", log_path)
        else:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                log("Enabled torch.compile(mode=reduce-overhead) for the SSM recurrence.", log_path)
            except Exception as error:
                log(f"torch.compile failed ({error}); using eager execution.", log_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    experiment_name = args.experiment_name or args.model
    metadata = {**vars(args), "experiment_name": experiment_name, "device": str(device), "git_revision": git_revision(Path(__file__).parent), "dataset_counts": {"train": len(train_set), "validation": len(validation_set)}, **model_stats(model._orig_mod if hasattr(model, "_orig_mod") else model)}
    write_json(out / "config.json", metadata)
    log(
        f"Starting {experiment_name} on {device} | train={args.train_dataset} ({len(train_set)}) | "
        f"validation={args.val_dataset} ({len(validation_set)}) | params={metadata['params']:,}",
        log_path,
    )
    history_path = out / "history.csv"
    history, best_dice, best_epoch, first_epoch = [], -1.0, -1, 1
    if args.resume:
        latest_path = out / "latest.pt"
        if not latest_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {latest_path}")
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_model.load_state_dict(latest["model"])
        optimizer.load_state_dict(latest["optimizer"])
        scheduler.load_state_dict(latest["scheduler"])
        if scaler.is_enabled() and latest.get("scaler") is not None:
            scaler.load_state_dict(latest["scaler"])
        best_dice, best_epoch = latest["best_dice"], latest["best_epoch"]
        first_epoch = latest["epoch"] + 1
        history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
        log(f"Resuming from epoch {first_epoch}; best epoch={best_epoch}, best_dice={best_dice:.4f}", log_path)
    started = time.time()
    for epoch in range(first_epoch, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, args.amp, not args.no_boundary_loss, not args.no_multiscale_boundary_loss)
        scheduler.step()
        metrics, _ = evaluate(model, validation_loader, device, args.threshold)
        history.append({"epoch": epoch, "loss": loss, **{f"val_{key}": value for key, value in metrics.items()}})
        pd.DataFrame(history).to_csv(history_path, index=False)
        if metrics["dice"] > best_dice:
            best_dice, best_epoch = metrics["dice"], epoch
            state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"model": state_model.state_dict(), "epoch": epoch, "dice": best_dice, "config": metadata}, out / "best.pt")
            marker = " [best checkpoint]"
        else:
            marker = ""
        torch.save(
            {
                "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None, "epoch": epoch,
                "best_dice": best_dice, "best_epoch": best_epoch, "config": metadata,
            },
            out / "latest.pt",
        )
        log(
            f"epoch={epoch:03d}/{args.epochs} loss={loss:.4f} val_dice={metrics['dice']:.4f} "
            f"val_iou={metrics['iou']:.4f} val_hd95={metrics['hd95']:.3f} best_epoch={best_epoch}{marker}",
            log_path,
        )
        if epoch - best_epoch >= args.patience:
            log(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best_dice={best_dice:.4f}", log_path)
            break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(checkpoint["model"])
    rows = []
    for name, split in [(args.val_dataset, "val")] + [(name, "test" if name.lower() in {"ph2", "ph2dataset"} else "val") for name in args.test_datasets]:
        dataset = SkinDataset(args.data_root, name, split, args.image_size, dataset_roots=dataset_roots)
        metrics, samples = evaluate(model, loader(dataset, args, False), device, args.threshold, out / "predictions" / f"{name}_{split}" if args.save_predictions else None)
        samples.to_csv(out / f"samples_{name}_{split}.csv", index=False)
        rows.append({"model": experiment_name, "architecture": args.model, "seed": args.seed, "train_dataset": args.train_dataset, "eval_dataset": name, "split": split, "best_epoch": best_epoch, "runtime_min": (time.time() - started) / 60, **model_stats(model._orig_mod if hasattr(model, "_orig_mod") else model), **metrics})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    log("Final evaluation:\n" + pd.DataFrame(rows).to_string(index=False), log_path)


if __name__ == "__main__":
    main()
