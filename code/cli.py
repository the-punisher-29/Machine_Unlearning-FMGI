from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ExperimentConfig, SISAConfig, TrainingConfig
from .experiments import (
    BaselineResult,
    prepare_data,
    run_baseline,
    run_mia_suite,
    run_naive_retraining,
    run_sisa_deletion,
)
from .models import build_model
from .sisa import SISAOrchestrator, load_ensemble_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SISA Training CLI")
    parser.add_argument("command", choices=["baseline", "deletion", "naive-delete", "mia"])
    parser.add_argument("--output-dir", default="results", dest="output_dir")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints", dest="checkpoint_dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4, dest="weight_decay")
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="adam",  # switched default from SGD to Adam to mirror the stable sanity-training recipe
    )
    parser.add_argument("--num-shards", type=int, default=20)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--percent", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    training = TrainingConfig(
        batch_size=args.batch_size,
        epochs_per_slice=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        device=args.device,
        seed=args.seed,
    )
    sisa = SISAConfig(num_shards=args.num_shards, num_slices=args.num_slices)
    cfg = ExperimentConfig(training=training, sisa=sisa)
    cfg.output_dir = Path(args.output_dir)
    cfg.checkpoint_dir = Path(args.checkpoint_dir)
    cfg.log_interval = args.log_interval
    return cfg


def ensure_baseline_artifacts(cfg: ExperimentConfig) -> BaselineResult:
    summary_path = cfg.output_dir / "baseline_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run baseline experiment first.")
    deletion_idx_path = cfg.output_dir / "deletion_indices.json"
    summary = json.loads(summary_path.read_text())
    deletion_indices = json.loads(deletion_idx_path.read_text())
    return BaselineResult(
        metrics=summary,
        shard_paths=[],
        aggregated_model_path=cfg.output_dir / "aggregated_baseline.pt",
        deletion_indices=deletion_indices,
    )


def command_baseline(cfg: ExperimentConfig) -> None:
    run_baseline(cfg)


def command_deletion(cfg: ExperimentConfig, percent: float) -> None:
    baseline = ensure_baseline_artifacts(cfg)
    data = prepare_data(cfg)
    orchestrator = SISAOrchestrator(cfg, data.train_subset, data.val_loader)
    key = f"{percent:.4f}"
    # Fallback if key is missing (e.g. if baseline was run with different config)
    if key not in baseline.deletion_indices:
        print(f"Warning: Deletion indices for {key} not found in baseline. Generating new indices.")
        from .experiments import sample_deletions
        delete_indices = sample_deletions(len(data.train_subset), percent, cfg.training.seed)
    else:
        delete_indices = baseline.deletion_indices[key]
    run_sisa_deletion(cfg, orchestrator, data, baseline, percent, delete_indices)


def command_naive_delete(cfg: ExperimentConfig, percent: float) -> None:
    baseline = ensure_baseline_artifacts(cfg)
    data = prepare_data(cfg)
    key = f"{percent:.4f}"
    # Fallback if key is missing
    if key not in baseline.deletion_indices:
        print(f"Warning: Deletion indices for {key} not found in baseline. Generating new indices.")
        from .experiments import sample_deletions
        delete_indices = sample_deletions(len(data.train_subset), percent, cfg.training.seed)
    else:
        delete_indices = baseline.deletion_indices[key]
    run_naive_retraining(cfg, data, percent, delete_indices)


def command_mia(cfg: ExperimentConfig, percent: float) -> None:
    data = prepare_data(cfg)
    models = {}
    model_paths = {
        "baseline": cfg.output_dir / "aggregated_baseline.pt",
        "naive_delete": cfg.output_dir / f"aggregated_naive_delete_{int(percent*100)}.pt",
        "sisa_unlearn": cfg.output_dir / f"aggregated_sisa_unlearn_{int(percent*100)}.pt",
    }
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model checkpoint {path}")
        # Aggregated checkpoints now hold per-shard states; reuse the helper so older
        # single-state artifacts (if any) still load transparently.
        models[name] = load_ensemble_checkpoint(path, cfg.dataset)
    run_mia_suite(cfg, models, data)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    if args.command == "baseline":
        command_baseline(cfg)
    elif args.command == "deletion":
        command_deletion(cfg, args.percent)
    elif args.command == "naive-delete":
        command_naive_delete(cfg, args.percent)
    elif args.command == "mia":
        command_mia(cfg, args.percent)


if __name__ == "__main__":
    main()
