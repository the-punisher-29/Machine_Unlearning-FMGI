from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .attacks import run_confidence_mia
from .config import ExperimentConfig, ensure_dirs
from .data import load_datasets, make_eval_loader, split_train_val
from .metrics import CSVLogger
from .sisa import SISAOrchestrator
from .trainer import SliceTrainer


@dataclass
class DataBundle:
    train_subset: torch.utils.data.Dataset
    val_subset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    member_loader: torch.utils.data.DataLoader
    nonmember_loader: torch.utils.data.DataLoader


@dataclass
class BaselineResult:
    metrics: Dict[str, float]
    shard_paths: List[Path]
    aggregated_model_path: Path
    deletion_indices: Dict[str, List[int]]


def prepare_data(cfg: ExperimentConfig) -> DataBundle:
    train_set, test_set = load_datasets(cfg.dataset)
    train_subset, val_subset = split_train_val(
        train_set, cfg.dataset.val_split, cfg.training.seed
    )
    val_loader = make_eval_loader(val_subset, cfg.training)
    test_loader = make_eval_loader(test_set, cfg.training)
    member_loader = make_eval_loader(train_subset, cfg.training)
    nonmember_loader = make_eval_loader(val_subset, cfg.training)
    return DataBundle(
        train_subset=train_subset,
        val_subset=val_subset,
        test_dataset=test_set,
        val_loader=val_loader,
        test_loader=test_loader,
        member_loader=member_loader,
        nonmember_loader=nonmember_loader,
    )


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    trainer: SliceTrainer,
) -> Dict[str, float]:
    return trainer.evaluate(model, data_loader)


def sample_deletions(num_samples: int, percent: float, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    delete_count = max(1, int(num_samples * percent))
    return sorted(rng.choice(num_samples, size=delete_count, replace=False).tolist())


def run_baseline(cfg: ExperimentConfig) -> Tuple[BaselineResult, SISAOrchestrator, DataBundle]:
    ensure_dirs(cfg)
    data = prepare_data(cfg)
    orchestrator = SISAOrchestrator(cfg, data.train_subset, data.val_loader)
    shard_paths = orchestrator.train_baseline()
    agg_model = orchestrator.aggregate(shard_paths)
    trainer = SliceTrainer(cfg.training, cfg.log_interval)
    test_metrics = evaluate_model(agg_model, data.test_loader, trainer)
    baseline_time = sum(
        artifact.metadata.get("wall_clock", 0.0)
        for artifact in orchestrator.slice_artifacts.values()
    )
    agg_path = cfg.output_dir / "aggregated_baseline.pt"
    orchestrator.serialize_ensemble_checkpoint(shard_paths, agg_path)

    deletion_indices: Dict[str, List[int]] = {}
    for scenario in cfg.deletion_percents:
        key = f"{scenario.percent:.4f}"
        deletion_indices[key] = sample_deletions(
            len(data.train_subset), scenario.percent, cfg.training.seed
        )

    summary = {
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1"],
        "test_loss": test_metrics["loss"],
        "baseline_time": baseline_time,
        "num_params": sum(p.numel() for p in agg_model.parameters()),
        "num_train_samples": len(data.train_subset),
        "num_shards": cfg.sisa.num_shards,
        "num_slices": cfg.sisa.num_slices,
    }
    (cfg.output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (cfg.output_dir / "deletion_indices.json").write_text(
        json.dumps(deletion_indices, indent=2)
    )
    return (
        BaselineResult(
            metrics=summary,
            shard_paths=shard_paths,
            aggregated_model_path=agg_path,
            deletion_indices=deletion_indices,
        ),
        orchestrator,
        data,
    )


def run_sisa_deletion(
    cfg: ExperimentConfig,
    orchestrator: SISAOrchestrator,
    data: DataBundle,
    baseline: BaselineResult,
    percent: float,
    delete_indices: Sequence[int],
) -> Dict[str, float]:
    trainer = SliceTrainer(cfg.training, cfg.log_interval)
    shard_paths, retrain_stats = orchestrator.retrain_for_deletions(delete_indices)
    agg_model = orchestrator.aggregate(shard_paths)
    metrics = evaluate_model(agg_model, data.test_loader, trainer)
    retrain_time = sum(
        v for k, v in retrain_stats.items() if k not in {"fraction_retrained", "num_retrained", "total_slices"}
    )
    result = {
        "percent_deleted": percent,
        "test_acc": metrics["acc"],
        "test_f1": metrics["f1"],
        "delta_acc": metrics["acc"] - baseline.metrics["test_acc"],
        "delta_f1": metrics["f1"] - baseline.metrics["test_f1"],
        "time_saved": baseline.metrics["baseline_time"] - retrain_time,
        "fraction_retrained": retrain_stats["fraction_retrained"],
        "num_retrained": retrain_stats["num_retrained"],
        "retrain_time": retrain_time,
    }
    log_path = cfg.output_dir / "deletion_results.csv"
    logger = CSVLogger(
        log_path,
        headers=[
            "percent_deleted",
            "test_acc",
            "test_f1",
            "delta_acc",
            "delta_f1",
            "time_saved",
            "fraction_retrained",
            "num_retrained",
            "retrain_time",
        ],
    )
    logger.log(result)
    orchestrator.serialize_ensemble_checkpoint(
        shard_paths,
        cfg.output_dir / f"aggregated_sisa_unlearn_{int(percent*100)}.pt",
    )
    return result


def run_naive_retraining(
    cfg: ExperimentConfig,
    data: DataBundle,
    percent: float,
    delete_indices: Sequence[int],
) -> Dict[str, float]:
    # Naive retraining: Train a fresh SISA model on the dataset excluding deleted points.
    # This serves as the "Gold Standard" for accuracy (but is slow).
    delete_set = set(delete_indices)
    # We need to reconstruct the dataset without the deleted indices.
    # Since SISAOrchestrator handles sharding internally based on the dataset length,
    # we just pass it the subset of remaining data.
    keep_indices = [idx for idx in range(len(data.train_subset)) if idx not in delete_set]
    pruned_train = torch.utils.data.Subset(data.train_subset, keep_indices)
    
    # Initialize a new orchestrator for this run
    orchestrator = SISAOrchestrator(cfg, pruned_train, data.val_loader)
    
    # Run full training (baseline mode for this pruned dataset)
    shard_paths = orchestrator.train_baseline()
    
    # Aggregate and evaluate
    agg_model = orchestrator.aggregate(shard_paths)
    trainer = SliceTrainer(cfg.training, cfg.log_interval)
    metrics = evaluate_model(agg_model, data.test_loader, trainer)
    
    # Save the aggregated model for MIA
    orchestrator.serialize_ensemble_checkpoint(
        shard_paths,
        cfg.output_dir / f"aggregated_naive_delete_{int(percent*100)}.pt",
    )
    
    return {
        "percent_deleted": percent,
        "test_acc": metrics["acc"],
        "test_f1": metrics["f1"],
    }


def run_mia_suite(
    cfg: ExperimentConfig,
    models: Dict[str, torch.nn.Module],
    data: DataBundle,
) -> Dict[str, Dict[str, float]]:
    trainer = SliceTrainer(cfg.training, cfg.log_interval)
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        metrics = run_confidence_mia(
            model,
            member_loader=data.member_loader,
            nonmember_loader=data.nonmember_loader,
            device=cfg.training.device,
        )
        results[name] = metrics
    (cfg.output_dir / "mia_results.json").write_text(json.dumps(results, indent=2))
    return results
