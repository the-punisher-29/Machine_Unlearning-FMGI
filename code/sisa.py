from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import ExperimentConfig, TrainingConfig
from .data import (
    SliceSpec,
    build_index_mapping,
    build_sisa_partitions,
    slice_dataloader,
)
from .metrics import CSVLogger
from .models import build_model
from .trainer import SliceTrainer


@dataclass
class SliceArtifact:
    slice_spec: SliceSpec
    checkpoint_path: Path
    metadata: Dict[str, float]


class ShardLogitEnsemble(nn.Module):
    """Average shard logits instead of averaging raw weights."""

    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = None
        for model in self.models:
            out = model(x)
            logits = out if logits is None else logits + out
        return logits / len(self.models)


class SISAOrchestrator:
    def __init__(
        self,
        config: ExperimentConfig,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.slice_trainer = SliceTrainer(config.training, config.log_interval)
        self.slice_specs = build_sisa_partitions(
            num_samples=len(train_dataset),
            num_shards=config.sisa.num_shards,
            num_slices=config.sisa.num_slices,
            seed=config.sisa.shuffle_seed,
        )
        self.index_map = build_index_mapping(self.slice_specs)
        self.slice_groups = self._group_by_shard(self.slice_specs)
        self.slice_artifacts: Dict[Tuple[int, int], SliceArtifact] = {}
        self.training_log = CSVLogger(
            config.output_dir / "sisa_training_log.csv",
            headers=[
                "phase",
                "shard",
                "slice",
                "num_samples",
                "acc",
                "f1",
                "train_loss",
                "wall_clock",
                "gpu_mem_gb",
            ],
        )
        # Previously every shard bootstrapped from its own random init; caching a shared
        # base_state keeps shards weight-aligned so aggregation has a meaningful reference.
        base_model = build_model(config.dataset)
        self.base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    @staticmethod
    def _group_by_shard(slices: Sequence[SliceSpec]) -> Dict[int, List[SliceSpec]]:
        grouped: Dict[int, List[SliceSpec]] = defaultdict(list)
        for spec in slices:
            grouped[spec.shard_id].append(spec)
        for specs in grouped.values():
            specs.sort(key=lambda s: s.slice_id)
        return dict(grouped)

    def _checkpoint_path(self, shard_id: int, slice_id: int) -> Path:
        tag = f"shard{shard_id:02d}_slice{slice_id:02d}"
        return self.config.checkpoint_dir / f"{tag}.pt"

    def _save_checkpoint(self, model: nn.Module, shard_id: int, slice_id: int) -> Path:
        path = self._checkpoint_path(shard_id, slice_id)
        torch.save(model.state_dict(), path)
        return path

    def _load_checkpoint(self, model: nn.Module, shard_id: int, slice_id: int) -> None:
        state = torch.load(self._checkpoint_path(shard_id, slice_id), map_location="cpu")
        model.load_state_dict(state)

    def _init_model(self) -> nn.Module:
        model = build_model(self.config.dataset)
        model.load_state_dict(self.base_state)
        device = torch.device(
            self.config.training.device if torch.cuda.is_available() else "cpu"
        )
        model.to(device)
        return model

    @staticmethod
    def _optimizer(model: nn.Module, train_cfg: TrainingConfig) -> torch.optim.Optimizer:
        if train_cfg.optimizer.lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=train_cfg.lr,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                weight_decay=train_cfg.weight_decay,
            )
        return torch.optim.SGD(
            model.parameters(),
            lr=train_cfg.lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
        )

    def _scheduler(
        self, optimizer: torch.optim.Optimizer, train_cfg: TrainingConfig
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.epochs_per_slice
        )

    def _criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _log_slice_metrics(
        self,
        phase: str,
        shard_id: int,
        slice_id: int,
        metrics: Dict[str, float],
        num_samples: int,
    ) -> None:
        payload = {
            "phase": phase,
            "shard": shard_id,
            "slice": slice_id,
            "num_samples": num_samples,
            "acc": metrics.get("acc", 0.0),
            "f1": metrics.get("f1", 0.0),
            "train_loss": metrics.get("train_loss", 0.0),
            "wall_clock": metrics.get("wall_clock", 0.0),
            "gpu_mem_gb": metrics.get("gpu_mem_gb", 0.0),
        }
        self.training_log.log(payload)

    def _record_artifact(
        self,
        spec: SliceSpec,
        checkpoint: Path,
        metrics: Dict[str, float],
    ) -> None:
        self.slice_artifacts[(spec.shard_id, spec.slice_id)] = SliceArtifact(
            slice_spec=spec,
            checkpoint_path=checkpoint,
            metadata=metrics,
        )

    def _effective_indices(
        self, spec: SliceSpec, delete_set: Optional[set[int]] = None
    ) -> List[int]:
        if delete_set is None:
            return spec.indices.tolist()
        return [idx for idx in spec.indices.tolist() if idx not in delete_set]

    def _train_slice(
        self,
        model: nn.Module,
        spec: SliceSpec,
        phase: str,
        delete_set: Optional[set[int]] = None,
    ) -> Dict[str, float]:
        indices = self._effective_indices(spec, delete_set)
        if not indices:
            return {"train_loss": 0.0, "acc": 0.0, "f1": 0.0, "wall_clock": 0.0}
        loader = slice_dataloader(
            dataset=self.train_dataset,
            slice_spec=SliceSpec(spec.shard_id, spec.slice_id, np.array(indices)),
            train_cfg=self.config.training,
        )
        optimizer = self._optimizer(model, self.config.training)
        scheduler = self._scheduler(optimizer, self.config.training)
        metrics = self.slice_trainer.train_slice(
            model=model,
            train_loader=loader,
            val_loader=self.val_loader,
            epochs=self.config.training.epochs_per_slice,
            criterion=self._criterion(),
            optimizer=optimizer,
            scheduler=scheduler,
            slice_tag=spec.tag,
            log_path=self.config.output_dir / f"{phase}_slice_logs.csv",
        )
        self._log_slice_metrics(
            phase=phase,
            shard_id=spec.shard_id,
            slice_id=spec.slice_id,
            metrics=metrics,
            num_samples=len(indices),
        )
        return metrics

    def train_baseline(self) -> List[Path]:
        shard_final_paths: List[Path] = []
        criterion = self._criterion()
        shard_items = sorted(self.slice_groups.items(), key=lambda kv: kv[0])
        shard_progress = tqdm(shard_items, desc="Baseline shards", leave=True)
        for shard_id, shard_slices in shard_progress:
            model = build_model(self.config.dataset)
            model.load_state_dict(self.base_state)
            init_path = self._checkpoint_path(shard_id, -1)
            if not init_path.exists():
                # Earlier we stored whatever random weights torch gave us; now we persist the
                # shared base so deletion retrains can always rewind to the common reference.
                torch.save(self.base_state, init_path)
            slice_progress = tqdm(
                shard_slices,
                desc=f"Shard {shard_id} slices",
                leave=False,
                dynamic_ncols=True,
            )
            for spec in slice_progress:
                loader = slice_dataloader(self.train_dataset, spec, self.config.training)
                optimizer = self._optimizer(model, self.config.training)
                scheduler = self._scheduler(optimizer, self.config.training)
                metrics = self.slice_trainer.train_slice(
                    model=model,
                    train_loader=loader,
                    val_loader=self.val_loader,
                    epochs=self.config.training.epochs_per_slice,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    slice_tag=spec.tag,
                    log_path=self.config.output_dir / "baseline_slice_logs.csv",
                )
                ckpt = self._save_checkpoint(model, spec.shard_id, spec.slice_id)
                self._record_artifact(spec, ckpt, metrics)
                self._log_slice_metrics(
                    phase="baseline",
                    shard_id=spec.shard_id,
                    slice_id=spec.slice_id,
                    metrics=metrics,
                    num_samples=len(spec.indices),
                )
            shard_final_paths.append(self._checkpoint_path(shard_id, shard_slices[-1].slice_id))
        return shard_final_paths

    def aggregate(self, shard_paths: Iterable[Path]) -> nn.Module:
        # Weight-averaging gave garbage accuracy because shards had diverged; logit averaging
        # preserves each shard's learned decision function without forcing parameter alignment.
        models = []
        for path in shard_paths:
            model = build_model(self.config.dataset)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            models.append(model)
        ensemble = ShardLogitEnsemble(models)
        ensemble.eval()
        return ensemble

    def serialize_ensemble_checkpoint(self, shard_paths: Iterable[Path], out_path: Path) -> None:
        payload = {
            "aggregation": "logit_ensemble",
            "states": [torch.load(path, map_location="cpu") for path in shard_paths],
        }
        # The legacy code saved a single averaged state dict; we now tag the payload so loaders
        # can reconstruct the per-shard ensemble for evaluation/MIA.
        torch.save(payload, out_path)

    def impacted_slices(self, delete_indices: Sequence[int]) -> Dict[int, int]:
        delete_set = set(delete_indices)
        impacted: Dict[int, int] = {}
        for idx in delete_set:
            shard_id, slice_id = self.index_map[idx]
            if shard_id not in impacted:
                impacted[shard_id] = slice_id
            else:
                impacted[shard_id] = min(impacted[shard_id], slice_id)
        return impacted

    def retrain_for_deletions(
        self, delete_indices: Sequence[int]
    ) -> Tuple[List[Path], Dict[str, float]]:
        delete_set = set(delete_indices)
        slice_impacts = self.impacted_slices(delete_indices)
        retrained_count = 0
        total_slices = len(self.slice_specs)
        retrain_stats: Dict[str, float] = {}
        shard_paths: List[Path] = []
        shard_items = sorted(self.slice_groups.items(), key=lambda kv: kv[0])
        shard_progress = tqdm(shard_items, desc="Deletion shards", leave=True)
        for shard_id, shard_slices in shard_progress:
            model = build_model(self.config.dataset)
            if shard_id in slice_impacts:
                start_slice = slice_impacts[shard_id]
                # Load checkpoint from previous slice (or init).
                prev_slice = start_slice - 1
                model_state = torch.load(
                    self._checkpoint_path(shard_id, prev_slice)
                    if prev_slice >= 0
                    else self._checkpoint_path(shard_id, -1),
                    map_location="cpu",
                )
                model.load_state_dict(model_state)
                slice_progress = tqdm(
                    shard_slices[start_slice:],
                    desc=f"Shard {shard_id} retrain",
                    leave=False,
                    dynamic_ncols=True,
                )
                for spec in slice_progress:
                    metrics = self._train_slice(
                        model=model,
                        spec=spec,
                        phase="deletion",
                        delete_set=delete_set,
                    )
                    ckpt = self._save_checkpoint(model, spec.shard_id, spec.slice_id)
                    retrained_count += 1
                    retrain_stats[spec.tag] = metrics.get("wall_clock", 0.0)
            shard_paths.append(self._checkpoint_path(shard_id, shard_slices[-1].slice_id))
        retrain_stats["fraction_retrained"] = retrained_count / total_slices
        retrain_stats["num_retrained"] = retrained_count
        retrain_stats["total_slices"] = total_slices
        return shard_paths, retrain_stats

    def serialize_metadata(self, path: Path) -> None:
        payload = {
            "slices": [
                {
                    "shard": spec.shard_id,
                    "slice": spec.slice_id,
                    "num_samples": len(spec.indices),
                }
                for spec in self.slice_specs
            ],
        }
        path.write_text(json.dumps(payload, indent=2))


def load_ensemble_checkpoint(path: Path, dataset_cfg) -> nn.Module:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("aggregation") == "logit_ensemble":
        models: List[nn.Module] = []
        for state in payload.get("states", []):
            model = build_model(dataset_cfg)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        ensemble = ShardLogitEnsemble(models)
        ensemble.eval()
        return ensemble
    # Fallback: load legacy single-checkpoint files so older artifacts remain usable.
    model = build_model(dataset_cfg)
    model.load_state_dict(payload)
    model.eval()
    return model
