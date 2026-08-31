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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Boundary-Routed Selective State Space MambaSeg.")
    parser.add_argument("--data-root", default="./data", help="Fallback root for the original unified local data layout.")
    parser.add_argument("--isic2017-root", default="/kaggle/input/datasets/zichengdoctor/isic2017")
    parser.add_argument("--isic2018-root", default="/kaggle/input/datasets/zichengdoctor/isic2018")
    parser.add_argument("--ph2-root", default="/kaggle/input/datasets/zichengdoctor/ph2dataset")
    parser.add_argument("--train-dataset", default="isic2018")
    parser.add_argument("--val-dataset", default="isic2018")
    parser.add_argument("--test-datasets", nargs="*", default=["isic2017", "PH2"])
    parser.add_argument("--model", choices=sorted(ABLATIONS), default="brss_mamba")
    parser.add_argument("--experiment-name", default=None, help="Result label; defaults to --model.")
    parser.add_argument("--output-dir", default="./outputs/brss_mamba")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-boundary-loss", action="store_true")
    parser.add_argument("--no-multiscale-boundary-loss", action="store_true")
    return parser.parse_args()


def loader(dataset: SkinDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)


def main() -> None:
    args = arguments()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_roots = {"isic2017": args.isic2017_root, "isic2018": args.isic2018_root, "ph2": args.ph2_root}
    train_set = SkinDataset(args.data_root, args.train_dataset, "train", args.image_size, augment=True, dataset_roots=dataset_roots)
    validation_set = SkinDataset(args.data_root, args.val_dataset, "val", args.image_size, dataset_roots=dataset_roots)
    train_loader, validation_loader = loader(train_set, args, True), loader(validation_set, args, False)
    model = get_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    experiment_name = args.experiment_name or args.model
    metadata = {**vars(args), "experiment_name": experiment_name, "device": str(device), "git_revision": git_revision(Path(__file__).parent), "dataset_counts": {"train": len(train_set), "validation": len(validation_set)}, **model_stats(model)}
    write_json(out / "config.json", metadata)
    history, best_dice, best_epoch, started = [], -1.0, -1, time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, args.amp, not args.no_boundary_loss, not args.no_multiscale_boundary_loss)
        scheduler.step()
        metrics, _ = evaluate(model, validation_loader, device, args.threshold)
        history.append({"epoch": epoch, "loss": loss, **{f"val_{key}": value for key, value in metrics.items()}})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if metrics["dice"] > best_dice:
            best_dice, best_epoch = metrics["dice"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "dice": best_dice, "config": metadata}, out / "best.pt")
        if epoch - best_epoch >= args.patience:
            break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    rows = []
    for name, split in [(args.val_dataset, "val")] + [(name, "test" if name.lower() in {"ph2", "ph2dataset"} else "val") for name in args.test_datasets]:
        dataset = SkinDataset(args.data_root, name, split, args.image_size, dataset_roots=dataset_roots)
        metrics, samples = evaluate(model, loader(dataset, args, False), device, args.threshold, out / "predictions" / f"{name}_{split}" if args.save_predictions else None)
        samples.to_csv(out / f"samples_{name}_{split}.csv", index=False)
        rows.append({"model": experiment_name, "architecture": args.model, "seed": args.seed, "train_dataset": args.train_dataset, "eval_dataset": name, "split": split, "best_epoch": best_epoch, "runtime_min": (time.time() - started) / 60, **model_stats(model), **metrics})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
