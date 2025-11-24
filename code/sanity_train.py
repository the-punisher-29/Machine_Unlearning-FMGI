from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__" and __package__ is None:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    __package__ = "code"

from .config import DatasetConfig, TrainingConfig, resolve_device
from .data import CIFAR10WithIndices, get_transforms
from .models import build_model
from .trainer import SliceTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone CIFAR-10 trainer (no SISA)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--data-root", default=str(Path("data")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cfg = DatasetConfig(root=Path(args.data_root))
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        epochs_per_slice=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        device=args.device,
        seed=args.seed,
    )
    device = resolve_device(train_cfg)
    torch.manual_seed(train_cfg.seed)

    full_train = CIFAR10WithIndices(
        root=dataset_cfg.root,
        train=True,
        transform=get_transforms(train=True),
        download=True,
    )
    val_size = int(len(full_train) * args.val_split)
    train_size = len(full_train) - val_size
    train_subset, val_subset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )
    test_set = CIFAR10WithIndices(
        root=dataset_cfg.root,
        train=False,
        transform=get_transforms(train=False),
        download=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = build_model(dataset_cfg)
    trainer = SliceTrainer(train_cfg, log_interval=25)
    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.lr,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            weight_decay=train_cfg.weight_decay,
        )
        if train_cfg.optimizer == "adam"
        else torch.optim.SGD(
            model.parameters(),
            lr=train_cfg.lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs_per_slice
    )

    trainer.train_slice(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs_per_slice,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        slice_tag="full_dataset",
        log_path=Path("results") / "sanity_logs.csv",
    )

    metrics = trainer.evaluate(model, test_loader)
    print("Test metrics:", metrics)
    torch.save(model.state_dict(), Path("results") / "sanity_model.pt")


if __name__ == "__main__":
    main()
